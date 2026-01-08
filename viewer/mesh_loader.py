# Mesh Loader for the viewer
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import trimesh
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.cm as cm
from scipy.spatial import cKDTree

from sklearn.decomposition import PCA
from itertools import product
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import defaultdict
import dartpy as dart

# Import mixins
from viewer.contour_mesh import ContourMeshMixin
from viewer.tetrahedron_mesh import TetrahedronMeshMixin
from viewer.fiber_architecture import FiberArchitectureMixin
from viewer.muscle_mesh import MuscleMeshMixin
from viewer.skeleton_mesh import SkeletonMeshMixin

scale = 0.01
cmap = cm.get_cmap("turbo")

def cotangent_weight_matrix(vertices, faces):
    """
    Compute the cotangent weight matrix using opposite angles from adjacent triangles.
    """
    n = len(vertices)
    L = scipy.sparse.lil_matrix((n, n))
    edge_to_faces = {}  # To store adjacent faces for each edge

    # Step 1: Find adjacent faces for each edge
    for face in faces:
        for i in range(3):
            v0, v1, v2 = face[i][0], face[(i + 1) % 3][0], face[(i + 2) % 3][0]

            edge = tuple(sorted([v0, v1]))  # Order (smallest, largest) to avoid duplicates
            
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append((v2, face))  # Store opposite vertex and face

    # Step 2: Compute cotangent weights
    for (v1, v2), adjacent_faces in edge_to_faces.items():
        if len(adjacent_faces) < 2:
            continue  # Skip boundary edges (only one adjacent face)

        # Get the two opposite vertices
        v_opposite_1, face1 = adjacent_faces[0]
        v_opposite_2, face2 = adjacent_faces[1]

        # Compute the two cotangent values
        def cotangent_angle(v_a, v_b, v_opposite):
            edge1 = vertices[v_a] - vertices[v_opposite]
            edge2 = vertices[v_b] - vertices[v_opposite]
            dot_product = np.dot(edge1, edge2)
            norm_product = np.linalg.norm(edge1) * np.linalg.norm(edge2)
            cosine_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
            angle = np.arccos(cosine_angle)
            return 1.0 / np.tan(angle) if angle > 1e-5 else 0  # Avoid division by zero

        cot_alpha = cotangent_angle(v1, v2, v_opposite_1)
        cot_beta = cotangent_angle(v1, v2, v_opposite_2)

        # Compute the cotangent weight sum
        r_ij = 0.5 * (cot_alpha + cot_beta)

        # Assign weights symmetrically
        L[v1, v2] += r_ij
        L[v2, v1] += r_ij

    return L.tocsr()  # Convert to efficient sparse matrix format

def solve_scalar_field(vertices, faces, origin_indices, insertion_indices, kmin=1.0, kmax=10.0):
    """
    Solve for the scalar field satisfying the Laplace equation with given constraints.
    """
    n = len(vertices)

    # Compute weight matrix
    W = cotangent_weight_matrix(vertices, faces)

    # Construct the Laplace matrix
    L = -scipy.sparse.diags(W.sum(axis=1).A1) + W

    # Boundary conditions
    b = np.zeros(n)
    boundary_mask = np.zeros(n, dtype=bool)

    # Set known values
    b[origin_indices] = kmin
    b[insertion_indices] = kmax
    boundary_mask[origin_indices] = True
    boundary_mask[insertion_indices] = True

    # Construct A matrix and b vector
    free_vertices = ~boundary_mask
    A = L[free_vertices][:, free_vertices]
    b_free = -L[free_vertices][:, boundary_mask] @ b[boundary_mask]

    # Solve the system
    u_free = scipy.sparse.linalg.spsolve(A, b_free)

    # Assign values back
    u = np.zeros(n)
    u[free_vertices] = u_free
    u[boundary_mask] = b[boundary_mask]

    return u

def compute_newell_normal(vertices):
    """Computes the normal vector of a closed loop using Newell's method."""
    normal = np.zeros(3)
    n = len(vertices)

    for i in range(n):
        v_curr = np.array(vertices[i])
        v_next = np.array(vertices[(i + 1) % n])  # Looping back

        normal[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
        normal[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
        normal[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])

    return normal / np.linalg.norm(normal)  if np.linalg.norm(normal) > 0.0 else np.array([0, 0, 0]) # Normalize

def compute_best_fitting_plane(vertices):
    """Finds an optimal coordinate system using Newell's normal as the fixed normal."""
    mean = np.mean(vertices, axis=0)  # Compute centroid
    centered = vertices - mean        # Center the points

    # Perform PCA using Singular Value Decomposition (SVD)
    _, _, eigenvectors = np.linalg.svd(centered)

    # # Extract first principal direction (largest spread)
    # basis_x = eigenvectors[0]  # Correctly extracts a (3,) shape vector
    # # Ensure basis_x is orthogonal to the Newell normal
    # basis_x -= np.dot(basis_x, newell_normal) * newell_normal
    # basis_x /= np.linalg.norm(basis_x)  # Normalize

    # # Second basis vector: Cross product to ensure a right-handed coordinate system
    # basis_y = np.cross(newell_normal, basis_x)

    basis_x = eigenvectors[0]
    basis_y = eigenvectors[1]

    basis_x /= np.linalg.norm(basis_x)
    basis_y /= np.linalg.norm(basis_y)

    return basis_x, basis_y, mean  # Newell normal is already the Z-axis


def project_vertices_to_global(vertices, basis_x, basis_y, mean):
    """Projects 3D vertices onto a plane and returns their new global positions."""
    projected = []
    for v in vertices:
        relative_v = np.array(v) - mean
        x_proj = np.dot(relative_v, basis_x)
        y_proj = np.dot(relative_v, basis_y)
        projected_3d = mean + x_proj * basis_x + y_proj * basis_y  # Convert back to 3D
        projected.append(projected_3d.tolist())

    return np.array(projected)

def compute_polygon_area(vertices_2d):
    """Computes the area of a 2D projected polygon using the Shoelace Theorem."""
    x = vertices_2d[:, 0]
    y = vertices_2d[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def compute_bounding_plane(vertices_2d):
    """Finds the bounding box of 2D projected vertices."""
    min_x, max_x = np.min(vertices_2d[:, 0]), np.max(vertices_2d[:, 0])
    min_y, max_y = np.min(vertices_2d[:, 1]), np.max(vertices_2d[:, 1])
    
    return np.array([
        [min_x, min_y], [max_x, min_y], 
        [max_x, max_y], [min_x, max_y]
    ])

class MeshLoader(ContourMeshMixin, TetrahedronMeshMixin, FiberArchitectureMixin, MuscleMeshMixin, SkeletonMeshMixin):
    def __init__(self):
        self.obj = None
        self.vertices = []
        self.normals = []
        self.faces = []
        self.texcoords = []

        self.faces_3 = []
        self.faces_4 = []
        self.faces_other = []  # New list to handle faces with more than 4 vertices

        self.new_vertices = []

        # self.joint_offset = None
        # self.axis = None

        self.edges = {}  # Dictionary to count edge occurrences
        self.open_edges = []  # New variable to store open edges
        self.edge_groups = []
        self.edge_classes = []

        self.is_draw = True
        self.is_draw_open_edges = False
        self.is_draw_scalar_field = True
        self.is_draw_contours = False
        self.is_draw_edges = False
        self.is_draw_centroid = False
        self.is_draw_bounding_box = False
        self.is_draw_discarded = False
        self.draw_contour_stream = None

        self.color = np.array([0.5, 0.5, 0.5, 1.0])
        self.transparency = 1.0
        
        # For Muscle Meshes
        self.frames = []
        self.bboxes = []
        self.contour_matches = []
        self.bounding_planes = []

        self.trimesh = None
        self.scalar_field = None
        self.vertex_colors = None
        self.contours = None
        self.contour_mesh_vertices = None
        self.contour_mesh_faces = None
        self.contour_mesh_normals = None
        self.is_draw_contour_mesh = False
        self.contour_mesh_color = np.array([0.8, 0.5, 0.5])
        self.contour_mesh_transparency = 0.8
        self.specific_contour = None
        self.specific_contour_value = 1.0
        self.contour_value_min = 1.1
        self.contour_value_max = 9.9
        self.normalized_contours = []

        self.structure_vectors = []

        self.contours_discarded = None
        self.bounding_planes_discarded = None

        self.fiber_architecture = [self.sobol_sampling_barycentric(16)]
        self.is_draw_fiber_architecture = False
        self.is_one_fiber = False
        self.sampling_method = 'sobol_unit_square'  # 'sobol_unit_square' or 'sobol_min_contour'
        self.cutting_method = 'area_based'  # 'area_based', 'voronoi', 'angular', or 'gradient'

        self.waypoints = []

        self.link_mode = 'mean' # or 'vertex'

        self.attach_skeletons = []
        self.attach_skeletons_sub = []

        # For Skeleton Meshes
        self.corners = None
        self.num_boxes = 1
        self.weld_joints = []
        self.is_draw_corners = True
        self.sizes = None

        self.is_root = False
        self.is_contact = False
        self.cand_parent_index = 0
        self.parent_name = None
        self.parent_mesh = None
        self.children_names = []
        self.joint_to_parent = None
        self.is_weld = False
        self.main_box_num = 0

        # Initialize mixin properties
        self._init_contour_properties()
        self._init_tetrahedron_properties()
        self._init_fiber_properties()
        self._init_muscle_properties()
        self._init_skeleton_properties()

    def find_contour_match(self, muscle_contour_orig, template_contour, prev_P0=None, preserve_order=False):
        # for each poiint from template_contours, find closest point from muscle_contour
        muscle_contour = muscle_contour_orig.copy()

        for v in template_contour:
            min_distance = float("inf")
            min_index = 0
            min_p = None
            for i in range(len(muscle_contour)):
                # make a line segment using muscle_contour[i - 1] and muscle_contour[i]
                # find the closest point p on the line segment from v
                # for all line segments, find the closest point to the v and add it to the new muscle_contour
                dir = muscle_contour[i] - muscle_contour[i - 1]
                # if t is 0, it is muscle_contour[i - 1], if t is 1, it is muscle_contour[i]
                if np.dot(dir, dir) > 0:
                    t = np.dot(v - muscle_contour[i - 1], dir) / np.dot(dir, dir)
                else:
                    t = 0

                if t > 0 and t < 1:
                    p = muscle_contour[i - 1] + t * dir
                    distance = np.linalg.norm(v - p)
                    if distance < min_distance:
                        min_distance = distance
                        min_index = i
                        min_p = p

            if min_p is not None:
                muscle_contour = np.insert(muscle_contour, min_index, min_p, axis=0)

        closest_points = []
        closest_muscle_index = []
        for template_point in template_contour:
            distances = np.linalg.norm(muscle_contour - template_point, axis=1)
            closest_points.append(muscle_contour[np.argmin(distances)])
            closest_muscle_index.append(np.argmin(distances))

        closest_distances = np.linalg.norm(template_contour - closest_points, axis=1)
        Q0_index = np.argmin(closest_distances)
        P0_index = closest_muscle_index[Q0_index]
        Q0 = template_contour[Q0_index]
        P0 = muscle_contour[P0_index]

        template_contour = np.roll(template_contour, -Q0_index, axis=0)
        muscle_contour = np.roll(muscle_contour, -P0_index, axis=0)

        # template_dir1 = template_contour[1] - template_contour[0]
        # template_dir2 = template_contour[-1] - template_contour[0]

        # i = 1
        # while True:
        #     muscle_dir1 = muscle_contour[i] - muscle_contour[0]
        #     if np.linalg.norm(muscle_dir1) > 0:
        #         muscle_dir1 = muscle_dir1 / np.linalg.norm(muscle_dir1)
        #         break
        #     i += 1

        # i = -1
        # while True:
        #     muscle_dir2 = muscle_contour[i] - muscle_contour[0]
        #     if np.linalg.norm(muscle_dir2) > 0:
        #         muscle_dir2 = muscle_dir2 / np.linalg.norm(muscle_dir2)
        #         break
        #     i -= 1

        # template_dir1 = template_dir1 / np.linalg.norm(template_dir1)
        # template_dir2 = template_dir2 / np.linalg.norm(template_dir2)

        # if np.dot(template_dir1, muscle_dir2) > np.dot(template_dir1, muscle_dir1) and np.dot(template_dir2, muscle_dir1) > np.dot(template_dir2, muscle_dir2):
        #     # first element should be the same, others should be reversed
        #     muscle_contour = np.roll(muscle_contour[::-1], 1, axis=0)

        closest_points = []
        closest_muscle_index = []
        for template_point in template_contour:
            distances = np.linalg.norm(muscle_contour - template_point, axis=1)
            closest_muscle_index.append(np.argmin(distances))
        
        if closest_muscle_index[2] < closest_muscle_index[1]:
            muscle_contour = np.roll(muscle_contour[::-1], 1, axis=0)
            closest_muscle_index = []
            for template_point in template_contour:
                distances = np.linalg.norm(muscle_contour - template_point, axis=1)
                closest_muscle_index.append(np.argmin(distances))

        result_index = []
        result = []
        for i in range(len(template_contour)):
            result_index.append((closest_muscle_index[i], i))

        # tQ = [0]
        # tP = [0]
        # tQ_sum = 0
        # tP_sum = 0

        # for i in range(len(template_contour)):
        #     tQ_sum += np.linalg.norm(template_contour[i] - template_contour[(i + 1) % len(template_contour)])
        #     tQ.append(tQ_sum)
        # for i in range(len(muscle_contour)):
        #     tP_sum += np.linalg.norm(muscle_contour[i] - muscle_contour[(i + 1) % len(muscle_contour)])
        #     tP.append(tP_sum)

        # # tQ_sum += np.linalg.norm(template_contour[0] - template_contour[-1])
        # # tP_sum += np.linalg.norm(muscle_contour[0] - muscle_contour[-1])

        # tQ = [t / tQ_sum for t in tQ]
        # tP = [t / tP_sum for t in tP]

        # # print(tQ, tP)

        # i0 = 0
        # result = [(P0, Q0)]
        # result_index = [(0, 0)]
        # for j in range(1, len(tQ)):
        #     for i in range(i0, len(tP)):
        #         if i > i0 and tP[i] > tQ[j]:
        #             if i > i0 + 1:
        #                 if tQ[j] - tP[i - 1] < tP[i] - tQ[j]:
        #                     result.append((muscle_contour[i - 1], template_contour[j]))
        #                     result_index.append((i - 1, j))
        #                     i0 = i - 1
        #                 else:
        #                     if i == len(muscle_contour):
        #                         result.append((muscle_contour[0], template_contour[j]))
        #                         result_index.append((0, j))
        #                     else:
        #                         result.append((muscle_contour[i], template_contour[j]))
        #                         result_index.append((i, j))
                            
        #                     i0 = i
        #             else:
        #                 if i == len(muscle_contour):
        #                     result.append((muscle_contour[0], template_contour[j]))
        #                     result_index.append((0, j))
        #                 else:
        #                     result.append((muscle_contour[i], template_contour[j]))
        #                     result_index.append((i, j))
        #                 i0 = i

        #             break

        for i in range(len(result_index)):
            muscle_start = result_index[i][0]
            muscle_end = result_index[(i + 1) % len(result_index)][0]

            template_start_pos = template_contour[result_index[i][1]]
            template_end_pos = template_contour[result_index[(i + 1) % len(result_index)][1]]

            if muscle_end != 0:
                muscle_list = list(range(muscle_start, muscle_end + 1))
            else:
                muscle_list = list(range(muscle_start, len(muscle_contour))) + [0]

            segment_t = []
            t_sum = 0
            for j in range(1, len(muscle_list)):
                t = np.linalg.norm(muscle_contour[muscle_list[j]] - muscle_contour[muscle_list[j - 1]])
                t_sum += t
                segment_t.append(t_sum)

            # t_sum += np.linalg.norm(muscle_contour[muscle_list[-1]] - muscle_contour[muscle_list[-2]])

            if t_sum > 0:
                segment_t = [t / t_sum for t in segment_t]

            # print(muscle_start, segment_t, muscle_end)

            for j in range(len(segment_t)):
                t = segment_t[j]
                template_pos = (1 - t) * template_start_pos + t * template_end_pos
                result.append((muscle_contour[muscle_list[j + 1]], template_pos))

        return muscle_contour, result
        

    def load(self, filename):
        # print('Loading mesh from', filename)
        self.obj = filename
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    self.vertices.append(list(map(float, line[2:].split())))
                elif line.startswith('vn '):
                    self.normals.append(list(map(float, line[3:].split())))
                elif line.startswith('vt '):
                    self.texcoords.append(list(map(float, line[3:].split())))
                elif line.startswith('f '):
                    self.faces.append([list(map(int, face.split('/'))) for face in line[2:].split()])
        
        # Convert lists to numpy arrays and scale the vertices
        self.vertices = np.array(self.vertices).astype(np.float32) * scale
        self.normals = np.array(self.normals)
        self.texcoords = np.array(self.texcoords)
        
        self.centroids = [np.mean(self.vertices, axis=0)]

        # Process faces based on the number of vertices
        for f in self.faces:
            if len(f) == 3:
                self.faces_3.append(f)
            elif len(f) == 4:
                self.faces_4.append(f)
            else:
                self.faces_other.append(f)  # Store faces with more than 4 vertices

            # Extract edges for the current face
            for i in range(len(f)):
                v1 = f[i][0] - 1  # 0-indexed vertex
                v2 = f[(i + 1) % len(f)][0] - 1  # 0-indexed vertex (next vertex)
                edge = tuple(sorted([v1, v2]))  # Sort to ensure consistent order

                # Count the edge occurrences
                if edge not in self.edges:
                    self.edges[edge] = 0
                self.edges[edge] += 1

        # Convert face arrays to numpy and adjust for 0-indexing
        self.faces_3 = np.array(self.faces_3) - 1
        self.faces_4 = np.array(self.faces_4) - 1
        self.faces_other = [np.array(f) - 1 for f in self.faces_other]  # Handle inhomogeneous faces with more than 4 vertices

        # Now we detect and store open edges
        self.open_edges, self.edge_groups, self.edge_classes = self.detect_open_edges()

        for edge_group in self.edge_groups:
            self.centroids.append(np.mean(self.vertices[edge_group], axis=0))
            
        '''
        # Find one large bounding box for the entire mesh; Not useful for curved muscles
        if len(self.centroids) == 3:
            cand_frames = []

            # principal_axis_simple = self.centroids[1] - self.centroids[2]

            # e1 = principal_axis_simple / np.linalg.norm(principal_axis_simple)
            # arbitrary_vec = np.array([1, 0, 0]) if abs(e1[0]) < 0.9 else np.array([0, 1, 0]) 
            # e2 = np.cross(e1, arbitrary_vec)
            # e2 /= np.linalg.norm(e2)
            # e3 = np.cross(e1, e2)

            # cand_frames.append([principal_axis_simple, e1, e2, e3])

            centered_vertices = self.vertices - self.centroids[0]
            cov_mat = np.dot(centered_vertices.T, centered_vertices) / len(self.vertices)
            w, v = np.linalg.eig(cov_mat)
            principal_axis_cov = v[:, np.argmax(w)]
            if principal_axis_cov[1] < 0:
                principal_axis_cov *= -1
            # length is same as principal_axis_simple
            principal_axis_cov *= np.linalg.norm(principal_axis_cov) / np.linalg.norm(principal_axis_cov)

            e1 = principal_axis_cov / np.linalg.norm(principal_axis_cov)
            arbitrary_vec = np.array([1, 0, 0]) if abs(e1[0]) < 0.9 else np.array([0, 1, 0]) 
            e2 = np.cross(e1, arbitrary_vec)
            e2 /= np.linalg.norm(e2)
            e3 = np.cross(e1, e2)

            cand_frames.append([principal_axis_cov, e1, e2, e3])

            for i, frame in enumerate(cand_frames):
                e1, e2, e3 = np.array(frame[1:])
                best_frame = None
                best_bbox = None
                best_angle = 0
                min_distance_sum = float("inf")

                origin = self.centroids[0]

                for angle in np.linspace(0, 360, 360+1):
                    # Rodrigues' Rotation Formula
                    def rodrigues_rotation(v, k, angle):
                        angle_radians = np.radians(angle)
                        cos_theta = np.cos(angle_radians)
                        sin_theta = np.sin(angle_radians)

                        return v * cos_theta + np.cross(k, v) * sin_theta + k * np.dot(k, v) * (1 - cos_theta)

                    # Rotate e2 and e3 around e1
                    new_e2 = rodrigues_rotation(e2, e1, angle)
                    new_e3 = rodrigues_rotation(e3, e1, angle)

                    rotation_matrix = np.vstack([e1, new_e2, new_e3]).T # Construct rotation matrix
                    local_vertices = (self.vertices - origin) @ rotation_matrix
                    
                    bbox_min = np.min(local_vertices, axis=0)
                    bbox_max = np.max(local_vertices, axis=0)

                    local_corners = np.array([
                        [bbox_max[0], bbox_max[1], bbox_max[2]],
                        [bbox_max[0], bbox_min[1], bbox_max[2]],
                        [bbox_max[0], bbox_min[1], bbox_min[2]],
                        [bbox_max[0], bbox_max[1], bbox_min[2]],
                        [bbox_min[0], bbox_max[1], bbox_max[2]],
                        [bbox_min[0], bbox_min[1], bbox_max[2]],
                        [bbox_min[0], bbox_min[1], bbox_min[2]],
                        [bbox_min[0], bbox_max[1], bbox_min[2]],
                    ])

                    # Transform back to global coordinates
                    bbox_corners = (local_corners @ rotation_matrix.T) + origin
                    
                    upper_face = bbox_corners[:4]  # Top 4 vertices
                    lower_face = bbox_corners[4:]  # Bottom 4 vertices

                    # Compute minimum distances
                    distance_upper = np.linalg.norm(upper_face[:, None, :] - self.vertices[self.edge_groups[0]][None, :, :], axis=2)
                    distance_upper = np.sum(np.min(distance_upper, axis=1))
                    distance_lower = np.linalg.norm(lower_face[:, None, :] - self.vertices[self.edge_groups[1]][None, :, :], axis=2)
                    distance_lower = np.sum(np.min(distance_lower, axis=1))

                    total_distance = distance_upper + distance_lower

                    if total_distance < min_distance_sum:
                        min_distance_sum = total_distance
                        best_frame = (e1, new_e2, new_e3)
                        best_bbox = bbox_corners
                        best_angle = angle

                self.frames.append([cand_frames[i][0], best_frame[0] * scale, best_frame[1] * scale, best_frame[2] * scale])
                # print(f'{filename}: best angle was {best_angle} degrees')
                contour_match = []
                # for self.vertices[self.edge_groups[0]] and upper_face, find closest vertex pair
                upper_face = best_bbox[:4]
                lower_face = best_bbox[4:]

                # for i, face in enumerate([upper_face, lower_face]):
                #     # face_info = []
                #     # for vertex in face:
                #     #     distance = np.linalg.norm(self.vertices[self.edge_groups[i]] - vertex, axis=1)
                #     #     face_info.append((vertex, self.vertices[self.edge_groups[i]][np.argmin(distance)]))

                #     # contour_match.append(face_info)

                #     contour_match.append(self.find_contour_match(self.vertices[self.edge_groups[i]], face))
                        
                self.contour_matches.append(contour_match)
                print(f'{filename}: best angle was {best_angle} degrees')
                self.bboxes.append(best_bbox)
        '''

        # Prepare vertex and normal arrays for all face types
        if len(self.faces_3) > 0:
            self.vertices_3 = self.vertices[self.faces_3[:,:,0].flatten()]
            self.normals_3 = self.normals[self.faces_3[:,:,2].flatten()]
        else:
            self.vertices_3 = np.array([])
            self.normals_3 = np.array([])
        
        if len(self.faces_4) > 0:
            self.vertices_4 = self.vertices[self.faces_4[:,:,0].flatten()]
            self.normals_4 = self.normals[self.faces_4[:,:,2].flatten()]
        else:
            self.vertices_4 = np.array([])
            self.normals_4 = np.array([])

        # Handle faces with more than 4 vertices
        self.vertices_other = [self.vertices[face[:, 0]] for face in self.faces_other]
        self.normals_other = [self.normals[face[:, 2]] for face in self.faces_other]

        self.new_vertices_3 = self.vertices_3.copy()
        self.new_vertices_4 = self.vertices_4.copy()
        self.new_vertices_other = self.vertices_other.copy()

    def detect_open_edges(self):
        open_edges = []

        # Iterate through all edges and find those that are not shared by two faces
        for edge, count in self.edges.items():
            if count == 1:  # Edge is only part of one face (open edge)
                open_edges.append(edge)

        # group the edges by checking if they share a vertex
        edge_groups = []
        for edge in open_edges:
            group_contained = []
            for i, group in enumerate(edge_groups):
                if edge[0] in group or edge[1] in group:
                    group_contained.append(i)

            if len(group_contained) == 0:
                edge_groups.append([edge[0], edge[1]])
            elif len(group_contained) == 1:
                group = edge_groups[group_contained[0]]
                if edge[0] not in group:
                    group.append(edge[0])
                if edge[1] not in group:
                    group.append(edge[1])
            else:
                # append merged group and remove original group_contained
                merged_group = []
                for i in group_contained:
                    merged_group += edge_groups[i]
                
                for i in sorted(group_contained, reverse=True):
                    del edge_groups[i]

                edge_groups.append(merged_group)

        # order edge_groups based on mean_vertex_positions y value in descending order
        edge_groups = sorted(edge_groups, key=lambda x: np.mean(self.vertices[x], axis=0)[1], reverse=True)

        grouped_open_edges = []
        for group in edge_groups:
            grouped_open_edges.append([edge for edge in open_edges if edge[0] in group and edge[1] in group])

        ordered_edge_groups = []
        for i in range(len(grouped_open_edges)):
            ordered_edge_group = []
            first_edge = grouped_open_edges[i][0]
            ordered_edge_group.append(first_edge[0])
            ordered_edge_group.append(first_edge[1])
            
            while len(ordered_edge_group) < len(edge_groups[i]):
                connection_found = False
                for edge in grouped_open_edges[i][1:]:
                    if edge[0] == ordered_edge_group[-1] and not edge[1] in ordered_edge_group:
                        ordered_edge_group.append(edge[1])
                        connection_found = True
                    elif edge[1] == ordered_edge_group[-1] and not edge[0] in ordered_edge_group:
                        ordered_edge_group.append(edge[0])
                        connection_found = True

                    if connection_found:
                        break
            
            ordered_edge_groups.append(ordered_edge_group)

        classes = []
        if len(ordered_edge_groups) > 0:
            # print('Open edge groups detected:', len(ordered_edge_groups))

            meanpoints = [np.mean(self.vertices[group], axis=0) for group in ordered_edge_groups]
            mean_meanpoints = np.mean(meanpoints, axis=0)
            for group in ordered_edge_groups:
                mean = np.mean(self.vertices[group], axis=0)
                if mean[1] < mean_meanpoints[1]:
                    classes.append('insertion')
                else:
                    classes.append('origin')
            # print(classes)
        
        return grouped_open_edges, ordered_edge_groups, classes
    
    def draw(self, color=np.array([0.5, 0.5, 0.5, 1.0])):
        glPushMatrix()

        # Determine if color array should be enabled for scalar field rendering
        use_color_array = self.vertex_colors is not None and self.is_draw_scalar_field

        if use_color_array:
            glEnableClientState(GL_COLOR_ARRAY)
        else:
            glColor4f(color[0], color[1], color[2], color[3])  # Default solid color

        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Draw filled faces (triangles)
        if len(self.new_vertices_3) > 0:
            glVertexPointer(3, GL_FLOAT, 0, self.new_vertices_3)
            if use_color_array:
                glColorPointer(4, GL_FLOAT, 0, self.vertex_colors)
            if len(self.normals_3) > 0:
                glEnableClientState(GL_NORMAL_ARRAY)
                glNormalPointer(GL_FLOAT, 0, self.normals_3)
            glDrawArrays(GL_TRIANGLES, 0, len(self.new_vertices_3))

        # Draw filled faces (quads)
        if len(self.new_vertices_4) > 0:
            glVertexPointer(3, GL_FLOAT, 0, self.new_vertices_4)
            if use_color_array:
                glColorPointer(4, GL_FLOAT, 0, self.vertex_colors)
            if len(self.normals_4) > 0:
                glEnableClientState(GL_NORMAL_ARRAY)
                glNormalPointer(GL_FLOAT, 0, self.normals_4)
            glDrawArrays(GL_QUADS, 0, len(self.new_vertices_4))

        # Draw filled faces (polygons with more than 4 vertices)
        for face_vertices, face_normals in zip(self.new_vertices_other, self.normals_other):
            glVertexPointer(3, GL_FLOAT, 0, face_vertices)
            if use_color_array:
                glColorPointer(4, GL_FLOAT, 0, self.vertex_colors)
            if len(face_normals) > 0:
                glEnableClientState(GL_NORMAL_ARRAY)
                glNormalPointer(GL_FLOAT, 0, face_normals)
            glDrawArrays(GL_POLYGON, 0, len(face_vertices))

        # Disable color array if it was enabled
        if use_color_array:
            glDisableClientState(GL_COLOR_ARRAY)
        
        glDisableClientState(GL_VERTEX_ARRAY)
        glPopMatrix()

    def draw_corners(self):
        if self.corners is not None:
            edges = [
                (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
                (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
                (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
            ]

            for corners in self.corners:
                glColor3f(0, 0, 0)
                glLineWidth(2.0)
                glBegin(GL_LINES)
                for start, end in edges:
                    glVertex3f(*corners[start])
                    glVertex3f(*corners[end])
                glEnd()

            # faces = [
            #     [0, 1, 3, 2],  # Bottom
            #     [4, 5, 7, 6],  # Top
            #     [0, 1, 5, 4],  # Front
            #     [2, 3, 7, 6],  # Back
            #     [0, 2, 6, 4],  # Left
            #     [1, 3, 7, 5],  # Right
            # ]

            # # Draw faces
            # glColor4f(0.6, 0.8, 1.0, 0.3)  # Transparent light blue
            # glEnable(GL_BLEND)
            # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            # glBegin(GL_QUADS)
            # for face in faces:
            #     for idx in face:
            #         glVertex3f(*self.corners[idx])
            # glEnd()
            # glDisable(GL_BLEND)

        glDisable(GL_LIGHTING)
        if len(self.weld_joints) > 0:
            glColor3f(1, 0, 0)
            glPointSize(5)
            glBegin(GL_POINTS)
            for joint in self.weld_joints:
                glVertex3f(*joint)
            glEnd()

        if self.joint_to_parent is not None:
            glColor3f(0, 0, 1)
            glPointSize(10)
            glBegin(GL_POINTS)
            glVertex3f(*self.joint_to_parent)
            glEnd()
        glEnable(GL_LIGHTING)
    
    def draw_edges(self):
        glPushMatrix()
        glColor4f(0, 0, 0, 1.0)
        glLineWidth(0.5)
        glBegin(GL_LINES)
        for edge in self.edges:
            v1, v2 = edge
            glVertex3fv(self.vertices[v1])
            glVertex3fv(self.vertices[v2])
        glEnd()
        glPopMatrix()

    def draw_open_edges(self, color=np.array([0.0, 0.0, 1.0, 1.0])):
        glDisable(GL_LIGHTING)
        glPushMatrix()
        for i, group in enumerate(self.edge_groups):
            # color = cmap(1 - i / max((len(self.edge_groups)- 1), 1))[:3]
            if self.edge_classes[i] == 'origin':
                color = np.array([1, 0, 0])
            else:
                color = np.array([0, 0, 1])
            glColor3fv(color)

            glPointSize(2)
            glBegin(GL_POINTS)
            for element in group:    
                glVertex3f(self.vertices[element][0], self.vertices[element][1], self.vertices[element][2])
            glEnd()
        
        for i, edges in enumerate(self.open_edges):
            # color = cmap(1 - i / max((len(self.edge_groups)- 1), 1))[:3]
            if self.edge_classes[i] == 'origin':
                color = np.array([1, 0, 0])
            else:
                color = np.array([0, 0, 1])
            glColor3fv(color)
            glLineWidth(1)
            glBegin(GL_LINES)
            for edge in edges:
                v1, v2 = edge
                glVertex3fv(self.vertices[v1])
                glVertex3fv(self.vertices[v2])
            glEnd()
        glPopMatrix()
        glEnable(GL_LIGHTING)

    def draw_contours(self):
        if self.specific_contour is not None:
            glDisable(GL_LIGHTING)
            glColor3f(1, 0, 0)

            glLineWidth(0.5)
            for contour in self.specific_contour:
                glBegin(GL_LINE_LOOP)
                for v in contour:
                    glVertex3fv(v)
                glEnd()

            glBegin(GL_POINTS)
            for contour in self.specific_contour:
                for v in contour:
                    glVertex3fv(v)
            glEnd()

            glEnable(GL_LIGHTING)

        if self.contours is None:
            return
        
        glDisable(GL_LIGHTING)

        glLineWidth(0.5)
        for structure_vector in self.structure_vectors:
            glBegin(GL_LINES)
            glColor4f(0, 0, 0, 0.5)
            glVertex3fv(structure_vector[0])
            glVertex3fv(structure_vector[1])
            glEnd()

        values = np.linspace(0, 1, len(self.contours) + 2)[1:-1]
        
        for i, contour_set in enumerate(self.contours):
            if self.draw_contour_stream is None:
                break

            if not self.draw_contour_stream[i]:
                continue

            # glPointSize(1)
            # glBegin(GL_POINTS)
            # for j, v in enumerate(contour_set):
            #     glColor3f(1 - j / len(contour_set), 0, 0)
            #     glVertex3fv(v)
            # glEnd()

            # color = cmap(1 - values[i])[:3]
            # color = np.array([i / len(self.contours), 1 - i / len(self.contours), 0.5 + 0.5 * i / len(self.contours)])

            if len(self.draw_contour_stream) > 8:
                color = cmap(1 - values[i])[:3]
            else:
                if i == 0:
                    color = np.array([1, 0, 0])
                elif i == 1:
                    color = np.array([0, 1, 0])
                elif i == 2:
                    color = np.array([0, 0, 1])
                elif i == 3:
                    color = np.array([1, 1, 0])
                elif i == 4:
                    color = np.array([0, 1, 1])
                elif i == 5:
                    color = np.array([1, 0, 1])
                elif i == 6:
                    color = np.array([1, 0.5, 0])
                elif i == 7:
                    color = np.array([0.5, 1, 0])
                elif i == 8:
                    color = np.array([0, 0, 0])

            
            glColor3fv(color)

            glLineWidth(0.5)
            for i, contour in enumerate(contour_set):
                glBegin(GL_LINE_LOOP)
                for v in contour:
                    glVertex3fv(v)
                glEnd()

            glPointSize(4)
            for contour in contour_set:
                glBegin(GL_POINTS)
                for i, v in enumerate(contour):
                    glColor3f(1 - i / len(contour), 0, 0)
                    glVertex3fv(v)
                glEnd()

        if self.is_draw_discarded:
            t = 0.1
            color = np.array([0, 0, 0, t])
            glColor4fv(color)
            for contour_set in self.contours_discarded:
                for contour in contour_set:
                    glBegin(GL_LINE_LOOP)
                    for v in contour:
                        glVertex3fv(v)
                    glEnd()

        glEnable(GL_LIGHTING)

    def draw_centroid(self):
        glDisable(GL_LIGHTING)
        glPushMatrix()
        
        glPointSize(5)
        glColor3f(1, 0, 0)
        glBegin(GL_POINTS)
        for centroid in self.centroids:
            glVertex3fv(centroid)
        glEnd()

        if len(self.frames) == 2:
            glLineWidth(2)

            # glColor3f(0, 1, 0)
            # glBegin(GL_LINES)
            # glVertex3fv(self.centroids[2])
            # glVertex3fv(self.centroids[2] + self.frames[0][0])
            # glEnd()

            # glColor3f(0, 0, 1)
            # offset = -self.frames[1][0] / 2
            # glPushMatrix()
            # glTranslatef(offset[0], offset[1], offset[2])
            # glBegin(GL_LINES)
            # # Draw self.frames[1] with its center at self.centroids[0]
            # glVertex3fv(self.centroids[0])
            # glVertex3fv(self.centroids[0] + self.frames[1][0])
            # glEnd()
            # glPopMatrix()

            for i, axes in enumerate(self.frames):
                if i == 0:
                    glColor3f(0, 1, 0)
                else:
                    glColor3f(0, 0, 1)

                glBegin(GL_LINES)
                glVertex3fv(self.centroids[0])
                glVertex3fv(self.centroids[0] + axes[1])
                glVertex3fv(self.centroids[0])
                glVertex3fv(self.centroids[0] + axes[2])
                glVertex3fv(self.centroids[0])
                glVertex3fv(self.centroids[0] + axes[3])
                glEnd()

        glPopMatrix()
        glEnable(GL_LIGHTING)

    def draw_bounding_box(self):
        # # Large Bounding box; no longer necessary
        # edges = [
        #         (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        #         (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        #         (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        #     ]
        
        # t = 0.5
        # for i, bbox in enumerate(self.bboxes):
        #     if i == 0:
        #         glColor4f(0, 1, 0, t)
        #     else:
        #         glColor4f(0, 0, 1, t)

        #     glPushMatrix()
        #     glDisable(GL_LIGHTING)  # Disable lighting for wireframe
        #     glLineWidth(1.0)
            
        #     glBegin(GL_LINES)
        #     for edge in edges:
        #         glVertex3fv(bbox[edge[0]])
        #         glVertex3fv(bbox[edge[1]])
        #     glEnd()

        #     glColor4f(0, 0, 0, t)
        #     glPointSize(10)
        #     glBegin(GL_POINTS)
        #     glVertex3fv(bbox[0])
        #     glEnd()
        #     glEnable(GL_LIGHTING)  # Restore lighting
        #     glPopMatrix()


        # for contour_info in self.bounding_planes:
        #     for j, face_info in enumerate(contour_info['contour_match']):
        #         if j == 0:
        #             glColor4f(1, 0, 0, t)
        #         else:
        #             glColor4f(1, 1, 0, t)
        #         glBegin(GL_LINES)
        #         glVertex3fv(face_info[0])
        #         glVertex3fv(face_info[1])
        #         glEnd()

        glLineWidth(0.5)
        glDisable(GL_LIGHTING)
        for i, bounding_planes in enumerate(self.bounding_planes):
            if self.draw_contour_stream is not None and not self.draw_contour_stream[i]:
                continue

            for plane_info in bounding_planes:
                glPushMatrix()

                glPushMatrix()
                glPointSize(5)
                glColor3f(0, 0, 0)
                glBegin(GL_POINTS)
                glVertex3fv(plane_info['mean'])
                glEnd()
                glPopMatrix()

                glColor3f(1, 0, 0)
                glBegin(GL_LINES)
                glVertex3fv(plane_info['mean'])
                glVertex3fv(plane_info['mean'] + plane_info['basis_x'] * scale * 0.1)
                glEnd()

                glColor3f(0, 1, 0)
                glBegin(GL_LINES)
                glVertex3fv(plane_info['mean'])
                glVertex3fv(plane_info['mean'] + plane_info['basis_y'] * scale * 0.1)
                glEnd()

                glColor3f(0, 0, 1)
                glBegin(GL_LINES)
                glVertex3fv(plane_info['mean'])
                glVertex3fv(plane_info['mean'] + plane_info['basis_z'] * scale * 0.1)
                glEnd()

                # glColor3f(0, 1, 1)
                # glBegin(GL_LINES)
                # glVertex3fv(plane_info['mean'])
                # glVertex3fv(plane_info['mean'] + plane_info['newell_normal'] * scale * 0.1)
                # glEnd()

                if plane_info['square_like']:
                    glColor3f(1, 0, 0)
                else:
                    glColor3f(0, 0, 0)
                glBegin(GL_LINE_LOOP)
                for point in plane_info['bounding_plane']:
                    glVertex3fv(point)
                glEnd()

                # glBegin(GL_LINE_LOOP)
                # for point_2d in plane_info['projected_2d']:
                #     glVertex3fv(point_2d)
                # glEnd()
                glPopMatrix()
        glEnable(GL_LIGHTING)

        glColor3f(0, 0, 0)
        glDisable(GL_LIGHTING)
        for i, bounding_plane_stream in enumerate(self.bounding_planes):
            if self.draw_contour_stream is not None and not self.draw_contour_stream[i]:
                continue
            for i in range(len(bounding_plane_stream) - 1):
                glBegin(GL_LINES)
                for p1, p2 in zip(bounding_plane_stream[i]['bounding_plane'], bounding_plane_stream[i + 1]['bounding_plane']):
                    glVertex3fv(p1)
                    glVertex3fv(p2)
                glVertex3fv(bounding_plane_stream[i]['mean'])
                glVertex3fv(bounding_plane_stream[i + 1]['mean'])
                glEnd()
        glEnable(GL_LIGHTING)
                
        if self.is_draw_discarded:
            t = 0.1
            for bounding_planes in self.bounding_planes_discarded:
                for plane_info in bounding_planes:
                    glPushMatrix()
                    glDisable(GL_LIGHTING)

                    glPushMatrix()
                    glColor4f(0, 0, 0, t)
                    glPointSize(5)
                    glBegin(GL_POINTS)
                    glVertex3fv(plane_info['mean'])
                    glEnd()
                    glPopMatrix()

                    glColor4f(1, 0, 0, t)
                    glBegin(GL_LINES)
                    glVertex3fv(plane_info['mean'])
                    glVertex3fv(plane_info['mean'] + plane_info['basis_x'] * scale)
                    glEnd()

                    glColor4f(0, 1, 0, t)
                    glBegin(GL_LINES)
                    glVertex3fv(plane_info['mean'])
                    glVertex3fv(plane_info['mean'] + plane_info['basis_y'] * scale)
                    glEnd()

                    glColor4f(0, 0, 1, t)
                    glBegin(GL_LINES)
                    glVertex3fv(plane_info['mean'])
                    glVertex3fv(plane_info['mean'] + plane_info['basis_z'] * scale)
                    glEnd()

                    glColor4f(0, 0, 0, t)
                    glBegin(GL_LINE_LOOP)
                    for point in plane_info['bounding_plane']:
                        glVertex3fv(point)
                    glEnd()

                    glEnable(GL_LIGHTING)
                    glPopMatrix()

    def draw_fiber_architecture(self):
        if self.fiber_architecture is None:
            return

        glDisable(GL_LIGHTING)
        glPointSize(3)
        glColor4f(1, 0, 0, 1)
        glBegin(GL_POINTS)
        # for waypoint_group in self.waypoints:
        #     for p in waypoint_group:
        #         glVertex3fv(p)
        for i, waypoint_group in enumerate(self.waypoints):
            if self.draw_contour_stream is not None and self.draw_contour_stream[i]:
                for waypoints in waypoint_group:
                    for p in waypoints:
                        glVertex3fv(p)
        glEnd()

        glColor4f(0.75, 0, 0, 1)
        glLineWidth(2)
        glBegin(GL_LINES)
        # for i in range(len(self.waypoints) - 1):
        #     for p1, p2 in zip(self.waypoints[i], self.waypoints[i + 1]):
        #         glVertex3fv(p1)
        #         glVertex3fv(p2)
        for i, waypoint_group in enumerate(self.waypoints):
            if self.draw_contour_stream is not None and self.draw_contour_stream[i]:
                for i in range(len(waypoint_group) - 1):
                    for p1, p2 in zip(waypoint_group[i], waypoint_group[i + 1]):
                        glVertex3fv(p1)
                        glVertex3fv(p2)
        glEnd()

        glEnable(GL_LIGHTING)

        return
        glDisable(GL_LIGHTING)
        glLineWidth(1)
        glPushMatrix()
        glScalef(scale, scale, scale)

        # for fiber in self.fiber_architecture:
        #     glBegin(GL_LINE_STRIP)
        #     for point in fiber:
        #         glVertex3fv(point)
        #     glEnd()

        for i, fibers in enumerate(self.fiber_architecture):
            glPushMatrix()
            glTranslatef(i, 0, 0)

            glColor3f(0, 0, 1)
            glBegin(GL_LINE_LOOP)
            glVertex3f(0, 0, 1)
            glVertex3f(1, 0, 1)
            glVertex3f(1, 1, 1)
            glVertex3f(0, 1, 1)
            glEnd()

            glColor3f(1, 0, 0)
            glBegin(GL_LINE_LOOP)
            glVertex3f(0, 0, 0)
            glVertex3f(1, 0, 0)
            glVertex3f(1, 1, 0)
            glVertex3f(0, 1, 0)
            glEnd()

            glColor3f(0, 0, 0)
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, 1)
            glVertex3f(1, 0, 0)
            glVertex3f(1, 0, 1)
            glVertex3f(1, 1, 0)
            glVertex3f(1, 1, 1)
            glVertex3f(0, 1, 0)
            glVertex3f(0, 1, 1)
            glEnd()

            glColor3f(0, 1, 0)
            glBegin(GL_LINES)
            for fiber in fibers:
                glVertex3f(fiber[0], fiber[1], 0)
                glVertex3f(fiber[0], fiber[1], 1)
            glEnd()

            # glPointSize(3)
            # glColor3f(0, 0, 0)
            # for contour in self.normalized_contours:
            #     glBegin(GL_POINTS)
            #     for point in contour:
            #         glVertex3fv(point)
            #     glEnd()

            # glColor4f(0, 0, 0, 0.1)
            # for contour in self.normalized_contours:
            #     glBegin(GL_LINE_LOOP)
            #     for point in contour:
            #         glVertex3fv(point)
            #     glEnd() 
            glPopMatrix()
        glPopMatrix()

        glEnable(GL_LIGHTING)

    # compute_scalar_field and save_bounding_planes are inherited from MuscleMeshMixin

    def find_contour(self, contour_value):
        # Find contour vertices for contour value
        contour_vertices = []
        contour_edges = []
        vertex_tree = None
        for face in self.faces_3:
            face_cand_vertices = []
            v0, v1, v2 = face[:, 0]
            verts = self.vertices[[v0, v1, v2]]
            values = self.scalar_field[[v0, v1, v2]]

            edges = [(0, 1), (1, 2), (2, 0)]

            for i, j in edges:
                v_start, v_end = values[i], values[j]

                if (v_start < contour_value and v_end > contour_value) or (v_start > contour_value and v_end < contour_value):
                    t = (contour_value - v_start) / (v_end - v_start)
                    p_contour = (1 - t) * verts[i] + t * verts[j]
                    face_cand_vertices.append(p_contour)
            
            if len(face_cand_vertices) == 2 and np.linalg.norm(face_cand_vertices[0] - face_cand_vertices[1]) >= 1e-7:
            # if len(face_cand_vertices) > 0:
                edge = []
                # check if two vertices are already in contour_vertices
                for vertex in face_cand_vertices:
                    vertex_found = False
                    if vertex_tree is not None and len(contour_vertices) > 0:
                        dist, index = vertex_tree.query(vertex, k=1)
                        if dist < 1e-7:
                            edge.append(index)
                            vertex_found = True

                    if not vertex_found:
                        index = len(contour_vertices)
                        contour_vertices.append(vertex)
                        vertex_tree = cKDTree(np.array(contour_vertices))
                        edge.append(index)

                contour_edges.append(tuple(sorted(edge)))

        # print(f'Number of contour edges: {len(contour_edges)}')

        # Classify contour edges into groups if there are multiple streams
        contour_edge_groups = []
        for edge in contour_edges:
            group_contained = []
            for i, group in enumerate(contour_edge_groups):
                if edge[0] in group or edge[1] in group:
                    group_contained.append(i)

            if len(group_contained) == 0:
                contour_edge_groups.append([edge[0], edge[1]])
            elif len(group_contained) == 1:
                group = contour_edge_groups[group_contained[0]]
                if edge[0] not in group:
                    group.append(edge[0])
                if edge[1] not in group:
                    group.append(edge[1])
            else:   # this edge is connecting two groups -> merge them into one
                # append merged group and remove original group_contained
                merged_group = []
                for i in group_contained:
                    merged_group += contour_edge_groups[i]
                
                for i in sorted(group_contained, reverse=True):
                    del contour_edge_groups[i]

                contour_edge_groups.append(merged_group)

        grouped_contour_edges = []
        for group in contour_edge_groups:
            grouped_contour_edges.append([edge for edge in contour_edges if edge[0] in group and edge[1] in group])
        
        # print(f'Number of contour edge groups: {len(grouped_contour_edges)}')

        # Order the vertices in edge group so that they form a closed loop
        ordered_contour_edge_groups = []
        for i in range(len(grouped_contour_edges)):
            ordered_edge_group = []
            first_edge = grouped_contour_edges[i][0]
            ordered_edge_group.append(first_edge[0])
            ordered_edge_group.append(first_edge[1])
            
            # trial_number = 0
            while len(ordered_edge_group) < len(contour_edge_groups[i]):
                connection_found = False
                for edge in grouped_contour_edges[i][1:]:
                    if edge[0] == ordered_edge_group[-1] and not edge[1] in ordered_edge_group:
                        ordered_edge_group.append(edge[1])
                        connection_found = True
                    elif edge[1] == ordered_edge_group[-1] and not edge[0] in ordered_edge_group:
                        ordered_edge_group.append(edge[0])
                        connection_found = True

                    if connection_found:
                        break

                # trial_number += 1
                # if trial_number == 100:
                #     break

            ordered_contour_edge_groups.append(ordered_edge_group)

        ordered_contour_vertices = []
        ordered_contour_vertices_orig = []
        bounding_planes = []
        for edge_group in ordered_contour_edge_groups:
            ordered_contour_vertices.append(np.array(contour_vertices)[edge_group])
            ordered_contour_vertices_orig.append(np.array(contour_vertices)[edge_group])
            ordered_contour_vertices[-1], bounding_plane = self.save_bounding_planes(ordered_contour_vertices[-1], contour_value)
            bounding_planes.append(bounding_plane)

        return bounding_planes, ordered_contour_vertices, ordered_contour_vertices_orig

    def find_contour_with_value(self, contour_value):
        if self.scalar_field is None:
            print("Please compute scalar field first")
            return
        
        self.specific_contour = None

        _, ordered_contour_vertices, _ = self.find_contour(contour_value)
        self.specific_contour = ordered_contour_vertices

        self.is_draw_contours = True

    def find_contours(self, scalar_step=0.1, skeleton_meshes=None, use_geodesic_edges=False):
        if self.scalar_field is None:
            print("Please compute scalar field first")
            return
        
        # # uniform sampling num_contours points between 1 and 10
        # contour_values = np.linspace(1, 10, num_contours + 2)[1:-1]

        # Reset contours and bounding planes
        self.contours = []
        self.bounding_planes = []

        self.contours_discarded = []
        self.bounding_planes_discarded = []

        contours = []
        contours_orig = []

        origin_bounding_planes = []
        origin_contours = []
        origin_contours_orig = []
        for i, edge_group in enumerate(self.edge_groups):
            if self.edge_classes[i] == 'origin':
                new_origin_contour, bounding_plane = self.save_bounding_planes(self.vertices[edge_group], 1)
                origin_bounding_planes.append(bounding_plane)
                origin_contours.append(new_origin_contour)
                origin_contours_orig.append(self.vertices[edge_group])
        self.bounding_planes.append(origin_bounding_planes)
        contours.append(origin_contours)
        contours_orig.append(origin_contours_orig)

        if False:
            contour_values = np.arange(1 + scalar_step, 10, scalar_step)
            for contour_value in contour_values:
                bounding_planes, ordered_contour_vertices, ordered_contour_vertices_orig = self.find_contour(contour_value)
                self.bounding_planes.append(bounding_planes)
                contours.append(ordered_contour_vertices)
                contours_orig.append(ordered_contour_vertices_orig)
  
        else:
            length_min_threshold = scale * 0.5
            length_max_threshold = scale * 1.0
            prev_contour_value = 1.0
            contour_value = 1.0 + scalar_step
            trial = 0
            while True:
                bounding_planes, ordered_contour_vertices, ordered_contour_vertices_orig = self.find_contour(contour_value)
                min_lengths = []
                for plane_i, plane in enumerate(bounding_planes):
                    min_length = np.inf
                    for prev_plane in self.bounding_planes[-1]:
                        # distance = np.linalg.norm(prev_plane['mean'] - plane['mean'])
                        distance = np.abs(np.dot(plane['mean'] - prev_plane['mean'], prev_plane['basis_z']))
                        
                        # measure distance by plane['mean'] from prev_plane made by prev_plane['basis_z'] and prev_plane['mean']
                        # if len(self.bounding_planes[-1]) != len(bounding_planes):
                        #     distance = np.linalg.norm(plane['mean'] - prev_plane['mean'])
                        # else:
                        #     distance = np.dot(plane['mean'] - prev_plane['mean'], prev_plane['basis_z'])

                        if distance < min_length:
                            min_length = distance
                        # if distance > max_length:
                        #     max_length = distance
                    min_lengths.append(min_length)

                print(f"num contours: {len(bounding_planes)}")
                print(f"min: {min_lengths} threshold: {length_min_threshold} {length_max_threshold}")
                # print(f"max: {max_length} threshold: {length_max_threshold}")
                print(contour_value)
                print()
                # if False:
                # if max_length > length_max_threshold:
                #     contour_value = (prev_contour_value + contour_value) / 2.0
                if np.any(np.array(min_lengths) > length_max_threshold):
                    # print(f"max: {max_length} threshold: {length_max_threshold}")
                    contour_value = (prev_contour_value + contour_value) / 2.0
                    trial += 1
                else:
                    if np.any(np.array(min_lengths) >= length_min_threshold) or trial >= 3:
                        print("ADDED")
                        print()
                        trial = 0
                        self.bounding_planes.append(bounding_planes)
                        contours.append(ordered_contour_vertices)
                        contours_orig.append(ordered_contour_vertices_orig)
                    else:
                        trial += 1

                    prev_contour_value = contour_value
                    if contour_value + scalar_step >= 10.0:
                        contour_value = (contour_value + 10.0) / 2
                        if 10.0 - contour_value < 1e-2:
                            break
                    else:
                        contour_value += scalar_step
            
        insertion_bounding_planes = []
        insertion_contours = []
        insertion_contours_orig = []
        for i, edge_group in enumerate(self.edge_groups):
            if self.edge_classes[i] == 'insertion':
                new_insertion_contour, bounding_plane = self.save_bounding_planes(self.vertices[edge_group], 10)
                insertion_bounding_planes.append(bounding_plane)
                insertion_contours.append(new_insertion_contour)
                insertion_contours_orig.append(self.vertices[edge_group])
        
        self.bounding_planes.append(insertion_bounding_planes)
        contours.append(insertion_contours)
        contours_orig.append(insertion_contours_orig)
        
        self.draw_contour_stream = [True] * len(contours)
        self.is_draw_contours = True
        self.contours = contours

        # for contour_group in self.contours:
        #     for contour in contour_group:
        #         print(len(contour), end=' ')
        #     print()

        # self.trim_contours(contours_orig)

    def trim_contours(self):
        if len(self.bounding_planes) == 0:
            print("No Contours found")
            return

        # Select best contours and discard others
        origin_num = len(self.bounding_planes[0])
        insertion_num = len(self.bounding_planes[-1])
        print(f"Number of origins/insertions: {origin_num}/{insertion_num}")

        contours_orig = [contour.copy() for contour in self.contours]

        # Current Formulation
        # stream_i = 0              -------------------
        # stream_i = 1             ---------------------
        # ...
        # stream_i = n - 1           -------    -------


        # It's hard to determine Square-like contours orientation
        # If they are not properly determined, contour directions twist at the middle
        # Therefore, re-align square-like contours based on the previous and next contours
        for stream_i, bounding_plane_infos in enumerate(self.bounding_planes):
            for i, bounding_plane_info in enumerate(bounding_plane_infos):
            # bounding_plane_info = bounding_plane_infos[0]
                if bounding_plane_info['square_like']:
                    basis_x = bounding_plane_info['basis_x']
                    basis_y = bounding_plane_info['basis_y']
                    basis_z = bounding_plane_info['basis_z']
                    mean = bounding_plane_info['mean']

                    # If there are multiple stream in prev or next bounding planes,
                    # It's better to align to at least one of them than doing nothing!
                    prev_plane_info = None
                    for stream_j in range(stream_i - 1, -1, -1):
                        if not self.bounding_planes[stream_j][0]['square_like']:
                            prev_plane_info = self.bounding_planes[stream_j][0]
                            break

                    next_plane_info = None
                    for stream_j in range(stream_i + 1, len(self.bounding_planes)):
                        if not self.bounding_planes[stream_j][0]['square_like']:
                            next_plane_info = self.bounding_planes[stream_j][0]
                            break

                    if prev_plane_info is not None and next_plane_info is not None:
                        prev_basis_x = prev_plane_info['basis_x'] if prev_plane_info is not None else basis_x
                        prev_mean = prev_plane_info['mean'] if prev_plane_info is not None else mean

                        next_basis_x = next_plane_info['basis_x'] if next_plane_info is not None else basis_x
                        next_mean = next_plane_info['mean'] if next_plane_info is not None else mean

                        prev_distance = np.linalg.norm(mean - prev_mean)
                        next_distance = np.linalg.norm(mean - next_mean)
                        sum_distance = prev_distance + next_distance

                        new_basis_x_prev = np.dot(prev_basis_x, basis_x) * basis_x + np.dot(prev_basis_x, basis_y) * basis_y
                        new_basis_x_prev /= np.linalg.norm(new_basis_x_prev)

                        new_basis_x_next = np.dot(next_basis_x, basis_x) * basis_x + np.dot(next_basis_x, basis_y) * basis_y
                        new_basis_x_next /= np.linalg.norm(new_basis_x_next)

                        bounding_plane_info['basis_x'] = next_distance / sum_distance * new_basis_x_prev + prev_distance / sum_distance * new_basis_x_next
                        bounding_plane_info['basis_x'] /= np.linalg.norm(bounding_plane_info['basis_x'])
                        bounding_plane_info['basis_y'] = np.cross(basis_z, bounding_plane_info['basis_x'])
                        bounding_plane_info['basis_y'] = bounding_plane_info['basis_y'] / (np.linalg.norm(bounding_plane_info['basis_y']) + 1e-10)

                        basis_x = bounding_plane_info['basis_x']
                        basis_y = bounding_plane_info['basis_y']

                        projected_2d = np.array([[np.dot(v - mean, basis_x), np.dot(v - mean, basis_y)] for v in contours_orig[stream_i][i]])
                        area = compute_polygon_area(projected_2d)

                        min_x, max_x = np.min(projected_2d[:, 0]), np.max(projected_2d[:, 0])
                        min_y, max_y = np.min(projected_2d[:, 1]), np.max(projected_2d[:, 1])

                        bounding_plane_2d = np.array([
                            [min_x, min_y], [max_x, min_y],
                            [max_x, max_y], [min_x, max_y]
                        ])
                        bounding_plane = np.array([mean + x * basis_x + y * basis_y for x, y in bounding_plane_2d])
                        projected_2d = np.array([mean + x * basis_x + y * basis_y for x, y in projected_2d])

                        bounding_plane_info['bounding_plane'] = bounding_plane
                        bounding_plane_info['projected_2d'] = projected_2d
                        bounding_plane_info['area'] = area
                        self.contours[stream_i][i], contour_match = self.find_contour_match(contours_orig[stream_i][i], bounding_plane)
                        bounding_plane_info['contour_match'] = contour_match
        
    def smoothen_contours(self):
        if len(self.bounding_planes) == 0:
            print("No Contours found")
            return
        
        contours_orig = [contour.copy() for contour in self.contours]

        # Contour extraction may result in jerky directions
        # Overall smoothing procedure
        for _ in range(10):
            smooth_bounding_planes = []
            for i, bounding_plane_infos in enumerate(self.bounding_planes):
                if i == 0 or i == len(self.bounding_planes) - 1:
                    smooth_bounding_planes.append(bounding_plane_infos)
                    continue

                if len(bounding_plane_infos) > 1:
                    smooth_bounding_planes.append(bounding_plane_infos)
                    continue
                
                bounding_plane_info = bounding_plane_infos[0]

                # Skip non-square bounding planes - don't smooth them
                if not bounding_plane_info.get('square_like', True):
                    smooth_bounding_planes.append(bounding_plane_infos)
                    continue

                prev_plane_info = self.bounding_planes[i - 1][0] if (i > 0 and len(self.bounding_planes[i - 1]) == 1) else bounding_plane_info
                next_plane_info = self.bounding_planes[i + 1][0] if (i < len(self.bounding_planes) - 1 and len(self.bounding_planes[i + 1]) == 1) else bounding_plane_info

                basis_x = bounding_plane_info['basis_x']
                basis_y = bounding_plane_info['basis_y']
                basis_z = bounding_plane_info['basis_z']
                mean = bounding_plane_info['mean']
                
                prev_basis_x = prev_plane_info['basis_x']
                prev_basis_y = prev_plane_info['basis_y']
                prev_basis_z = prev_plane_info['basis_z']
                prev_mean = prev_plane_info['mean']

                next_basis_x = next_plane_info['basis_x']
                next_basis_y = next_plane_info['basis_y']
                next_basis_z = next_plane_info['basis_z']
                next_mean = next_plane_info['mean']

                prev_x_proj = np.dot(prev_basis_x, basis_x) * basis_x + np.dot(prev_basis_x, basis_y) * basis_y
                next_x_proj = np.dot(next_basis_x, basis_x) * basis_x + np.dot(next_basis_x, basis_y) * basis_y

                prev_x_proj /= np.linalg.norm(prev_x_proj)
                next_x_proj /= np.linalg.norm(next_x_proj)

                new_basis_x = ((prev_x_proj + next_x_proj) / 2 + basis_x) / 2
                # new_basis_x = (prev_x_proj + next_x_proj) / 2
                new_basis_x /= np.linalg.norm(new_basis_x)
                new_basis_y = np.cross(basis_z, new_basis_x)

                new_basis_x = ((prev_basis_x + next_basis_x) / 2 + basis_x) / 2
                new_basis_z = ((prev_basis_z + next_basis_z) / 2 + basis_z) / 2
                new_basis_x /= np.linalg.norm(new_basis_x)
                new_basis_z /= np.linalg.norm(new_basis_z)
                # Re-orthogonalize: compute basis_y from cross product to ensure 90-degree corners
                new_basis_y = np.cross(new_basis_z, new_basis_x)
                new_basis_y /= (np.linalg.norm(new_basis_y) + 1e-10)
                new_newell = new_basis_z

                # new_mean is intersection between line made by (prev_mean, next_mean) and plane made by (mean, basis_z)
                new_mean = prev_mean + np.dot(mean - prev_mean, basis_z) / np.dot(next_mean - prev_mean, basis_z) * (next_mean - prev_mean)
                new_mean = (new_mean + mean) / 2

                projected_2d = np.array([[np.dot(v - new_mean, new_basis_x), np.dot(v - new_mean, new_basis_y)] for v in contours_orig[i][0]])
                area = compute_polygon_area(projected_2d)

                min_x, max_x = np.min(projected_2d[:, 0]), np.max(projected_2d[:, 0])
                min_y, max_y = np.min(projected_2d[:, 1]), np.max(projected_2d[:, 1])

                bounding_plane_2d = np.array([
                    [min_x, min_y], [max_x, min_y],
                    [max_x, max_y], [min_x, max_y]
                ])
                bounding_plane = np.array([new_mean + x * new_basis_x + y * new_basis_y for x, y in bounding_plane_2d])
                projected_2d = np.array([new_mean + x * new_basis_x + y * new_basis_y for x, y in projected_2d])

                self.contours[i][0], contour_match = self.find_contour_match(contours_orig[i][0], bounding_plane)

                new_bounding_plane_info = {
                    'basis_x': new_basis_x,
                    'basis_y': new_basis_y,
                    'basis_z': basis_z,
                    'mean': new_mean,
                    'bounding_plane': bounding_plane,
                    'projected_2d': projected_2d,
                    'area': area,
                    'contour_match': contour_match,
                    'scalar_value': bounding_plane_info['scalar_value'],
                    'square_like': bounding_plane_info['square_like'],

                    # 'newell_normal': bounding_plane_info['newell_normal'],
                    'newell_normal': new_newell,
                }

                smooth_bounding_planes.append([new_bounding_plane_info])

            self.bounding_planes = smooth_bounding_planes

    def resample_contours(self, num_samples=32):
        """
        Resample all contours to have the same number of vertices with consistent topology.
        Uses arc-length parameterization and uniform sampling.
        Also updates bounding planes via find_contour_match to maintain linkage.

        Args:
            num_samples: int - desired number of vertices per contour
        """
        if self.contours is None or len(self.contours) == 0:
            print("No contours found. Please run find_contours first.")
            return

        if self.bounding_planes is None or len(self.bounding_planes) == 0:
            print("No bounding planes found. Please run find_contours first.")
            return

        # Compute reference direction from origin to insertion centroids
        origin_centroids = [np.mean(c, axis=0) for c in self.contours[0]]
        insertion_centroids = [np.mean(c, axis=0) for c in self.contours[-1]]
        origin_center = np.mean(origin_centroids, axis=0)
        insertion_center = np.mean(insertion_centroids, axis=0)
        reference_direction = insertion_center - origin_center
        ref_norm = np.linalg.norm(reference_direction)
        if ref_norm > 1e-10:
            reference_direction = reference_direction / ref_norm
        else:
            reference_direction = np.array([0, 1, 0])

        resampled_contours = []
        new_bounding_planes = []

        for level_idx, (contour_group, bounding_plane_group) in enumerate(zip(self.contours, self.bounding_planes)):
            resampled_group = []
            new_bp_group = []

            for contour_idx, (contour, bounding_plane_info) in enumerate(zip(contour_group, bounding_plane_group)):
                # First resample the contour uniformly
                resampled = self._resample_single_contour(
                    np.array(contour), num_samples, reference_direction
                )

                # Get the bounding plane rectangle from existing bounding_plane_info
                bounding_plane = bounding_plane_info['bounding_plane']

                # Re-run find_contour_match to align resampled contour to bounding plane
                aligned_contour, contour_match = self.find_contour_match(resampled, bounding_plane)

                # Update bounding plane info with new contour match
                new_bp_info = bounding_plane_info.copy()
                new_bp_info['contour_match'] = contour_match

                resampled_group.append(aligned_contour)
                new_bp_group.append(new_bp_info)

            resampled_contours.append(resampled_group)
            new_bounding_planes.append(new_bp_group)

        self.contours = resampled_contours
        self.bounding_planes = new_bounding_planes
        print(f"Resampled {len(self.contours)} contour levels to {num_samples} vertices each")

    def _resample_single_contour(self, contour_points, num_samples, reference_direction):
        """
        Resample a single closed contour to have exactly num_samples vertices.
        Uses arc-length parameterization with consistent starting point.

        Args:
            contour_points: numpy array of shape (N, 3)
            num_samples: int - desired number of output vertices
            reference_direction: numpy array of shape (3,) - for consistent starting point

        Returns:
            numpy array of shape (num_samples, 3)
        """
        contour = np.array(contour_points)
        n = len(contour)

        if n < 2:
            return np.tile(contour[0], (num_samples, 1))

        # Step 1: Compute cumulative arc lengths (closed loop)
        segments = np.zeros((n, 3))
        for i in range(n):
            segments[i] = contour[(i + 1) % n] - contour[i]
        segment_lengths = np.linalg.norm(segments, axis=1)
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_length[-1]

        if total_length < 1e-10:
            return np.tile(contour[0], (num_samples, 1))

        # Normalize to [0, 1]
        t_original = cumulative_length / total_length

        # Step 2: Find consistent starting point using reference direction
        centroid = np.mean(contour, axis=0)
        directions = contour - centroid
        dots = directions @ reference_direction
        start_idx = np.argmax(dots)

        # Roll arrays to start from this index
        contour = np.roll(contour, -start_idx, axis=0)
        t_rolled = np.roll(t_original[:-1], -start_idx)
        t_rolled = t_rolled - t_rolled[0]
        t_rolled[t_rolled < 0] += 1.0
        t_original = np.concatenate([np.sort(t_rolled), [1.0]])

        # Step 3: Sample at uniform parameter values
        t_new = np.linspace(0, 1, num_samples, endpoint=False)

        # Step 4: Interpolate positions
        resampled = np.zeros((num_samples, 3))
        for i, t in enumerate(t_new):
            # Find segment containing t
            idx = np.searchsorted(t_original, t, side='right') - 1
            idx = max(0, min(idx, n - 1))

            # Local interpolation parameter
            t_start = t_original[idx]
            t_end = t_original[idx + 1] if idx + 1 < len(t_original) else 1.0

            if abs(t_end - t_start) < 1e-10:
                local_t = 0
            else:
                local_t = (t - t_start) / (t_end - t_start)

            # Linear interpolation
            p0 = contour[idx]
            p1 = contour[(idx + 1) % n]
            resampled[i] = (1 - local_t) * p0 + local_t * p1

        return resampled

    def build_contour_mesh(self):
        """
        Build a triangle mesh from contours after find_contour_stream has been called.
        Handles merged streams properly by sharing vertices and stitching at merge points.

        After find_contour_stream, self.contours is organized as:
        - self.contours[stream_idx][level_idx] = contour vertices

        For muscles like biceps (2 origins -> 1 insertion):
        - At merged levels, adjacent streams share boundary vertices
        - Stitching faces connect the streams at shared boundaries
        """
        if self.contours is None or len(self.contours) == 0:
            print("No contours found. Run find_contours, resample_contours, and find_contour_stream first.")
            return

        num_streams = len(self.contours)
        if num_streams == 0:
            print("No streams found.")
            return

        # First, align all streams to minimize twist
        aligned_streams = []
        for stream_idx, stream_contours in enumerate(self.contours):
            if len(stream_contours) < 2:
                print(f"Stream {stream_idx} has less than 2 contours, skipping.")
                aligned_streams.append([])
                continue
            aligned_streams.append(self._align_stream_contours(stream_contours))

        # Find number of levels (assuming all streams have same number of levels)
        num_levels = max(len(s) for s in aligned_streams) if aligned_streams else 0
        if num_levels < 2:
            print("Not enough levels to build mesh.")
            return

        # Build vertices and track indices per stream per level
        all_vertices = []
        # stream_level_indices[stream_idx][level_idx] = list of vertex indices for that contour
        stream_level_indices = [[[] for _ in range(num_levels)] for _ in range(num_streams)]

        # Epsilon for detecting shared vertices
        eps = 1e-6

        for level_idx in range(num_levels):
            level_vertices = []  # All vertices at this level (before dedup)
            level_stream_ranges = []  # (start, end) indices into level_vertices for each stream

            # Collect all vertices at this level from all streams
            for stream_idx, aligned_stream in enumerate(aligned_streams):
                if level_idx >= len(aligned_stream) or len(aligned_stream[level_idx]) == 0:
                    level_stream_ranges.append((len(level_vertices), len(level_vertices)))
                    continue
                start = len(level_vertices)
                for v in aligned_stream[level_idx]:
                    level_vertices.append(np.array(v))
                end = len(level_vertices)
                level_stream_ranges.append((start, end))

            if len(level_vertices) == 0:
                continue

            # Find duplicate vertices at this level (shared boundaries between streams)
            level_vertices = np.array(level_vertices)
            n = len(level_vertices)

            # Map from original index to deduplicated index
            dedup_map = list(range(n))

            # Find duplicates across different streams
            for stream_i in range(num_streams):
                start_i, end_i = level_stream_ranges[stream_i]
                for stream_j in range(stream_i + 1, num_streams):
                    start_j, end_j = level_stream_ranges[stream_j]
                    for i in range(start_i, end_i):
                        for j in range(start_j, end_j):
                            if np.linalg.norm(level_vertices[i] - level_vertices[j]) < eps:
                                # j is a duplicate of i, map j to i's final index
                                dedup_map[j] = dedup_map[i]

            # Build deduplicated vertex list and create final index mapping
            final_indices = {}  # original index -> final vertex index (global)
            for orig_idx in range(n):
                canonical = dedup_map[orig_idx]
                if canonical not in final_indices:
                    final_indices[canonical] = len(all_vertices)
                    all_vertices.append(level_vertices[canonical])
                final_indices[orig_idx] = final_indices[canonical]

            # Store vertex indices for each stream at this level
            for stream_idx in range(num_streams):
                start, end = level_stream_ranges[stream_idx]
                indices = [final_indices[i] for i in range(start, end)]
                stream_level_indices[stream_idx][level_idx] = indices

        if len(all_vertices) == 0:
            print("No vertices generated.")
            return

        # Build faces
        all_faces = []

        # Create faces between consecutive levels for each stream
        for stream_idx in range(num_streams):
            for level_idx in range(num_levels - 1):
                curr_indices = stream_level_indices[stream_idx][level_idx]
                next_indices = stream_level_indices[stream_idx][level_idx + 1]

                if len(curr_indices) < 2 or len(next_indices) < 2:
                    continue

                n_curr = len(curr_indices)
                n_next = len(next_indices)

                if n_curr == n_next:
                    # Same size - create band with direct indices
                    for i in range(n_curr):
                        i_next = (i + 1) % n_curr
                        v0 = curr_indices[i]
                        v1 = curr_indices[i_next]
                        v2 = next_indices[i_next]
                        v3 = next_indices[i]
                        # Two triangles for the quad
                        all_faces.append([v0, v1, v2])
                        all_faces.append([v0, v2, v3])
                else:
                    # Different sizes - variable band
                    faces = self._create_contour_band_variable_indices(
                        curr_indices, next_indices
                    )
                    all_faces.extend(faces)

        # Create stitching faces between adjacent streams at merged levels
        for level_idx in range(num_levels):
            for stream_i in range(num_streams):
                for stream_j in range(stream_i + 1, num_streams):
                    indices_i = stream_level_indices[stream_i][level_idx]
                    indices_j = stream_level_indices[stream_j][level_idx]

                    if len(indices_i) == 0 or len(indices_j) == 0:
                        continue

                    # Find shared vertex indices (boundary points)
                    shared_i = []  # positions in indices_i that are shared
                    shared_j = []  # corresponding positions in indices_j

                    for pi, vi in enumerate(indices_i):
                        for pj, vj in enumerate(indices_j):
                            if vi == vj:  # Same vertex index = shared
                                shared_i.append(pi)
                                shared_j.append(pj)

                    # If there are exactly 2 shared points, we can create stitching
                    if len(shared_i) >= 2:
                        # Streams share boundary - this is a merged level
                        # The shared vertices are the boundaries where streams meet
                        # No additional stitching needed since vertices are already shared
                        pass

        if len(all_faces) == 0:
            print("No faces generated.")
            return

        # Convert to numpy arrays
        self.contour_mesh_vertices = np.array(all_vertices, dtype=np.float32)
        self.contour_mesh_faces = np.array(all_faces, dtype=np.int32)

        # Compute normals
        self._compute_contour_mesh_normals()

        print(f"Built contour mesh: {len(self.contour_mesh_vertices)} vertices, "
              f"{len(self.contour_mesh_faces)} faces from {num_streams} streams")

    def _create_contour_band_variable_indices(self, curr_indices, next_indices):
        """
        Create triangular faces between two contours with different vertex counts.
        Uses direct vertex indices instead of offsets.
        """
        n_curr = len(curr_indices)
        n_next = len(next_indices)
        faces = []

        # Use ratio-based vertex pairing
        i_curr = 0
        i_next = 0

        while i_curr < n_curr or i_next < n_next:
            # Current progress ratios
            ratio_curr = i_curr / n_curr if n_curr > 0 else 1
            ratio_next = i_next / n_next if n_next > 0 else 1

            v0 = curr_indices[i_curr % n_curr]
            v1 = curr_indices[(i_curr + 1) % n_curr]
            v2 = next_indices[(i_next + 1) % n_next]
            v3 = next_indices[i_next % n_next]

            if ratio_curr <= ratio_next and i_curr < n_curr:
                # Advance on curr contour
                faces.append([v0, v1, v3])
                i_curr += 1
            elif i_next < n_next:
                # Advance on next contour
                faces.append([v0, v2, v3])
                i_next += 1
            else:
                break

        return faces

    def _align_stream_contours(self, stream_contours):
        """
        Align consecutive contours in a stream to minimize twist.
        Uses optimal rotation to minimize total distance between corresponding vertices.
        """
        if len(stream_contours) < 2:
            return stream_contours

        aligned = [np.array(stream_contours[0])]  # First contour stays fixed

        for i in range(1, len(stream_contours)):
            prev_contour = aligned[-1]
            curr_contour = np.array(stream_contours[i])

            # Find optimal rotation offset
            best_offset = self._find_best_rotation_offset(prev_contour, curr_contour)

            # Apply rotation
            aligned_contour = np.roll(curr_contour, -best_offset, axis=0)
            aligned.append(aligned_contour)

        return aligned

    def _find_best_rotation_offset(self, contour_a, contour_b):
        """
        Find the rotation offset for contour_b that minimizes total distance to contour_a.
        Returns the offset to apply with np.roll.
        """
        n_a = len(contour_a)
        n_b = len(contour_b)

        if n_a != n_b:
            # Different sizes - use centroid-based alignment
            return self._find_best_rotation_offset_variable(contour_a, contour_b)

        n = n_a
        min_distance = float('inf')
        best_offset = 0

        # Try all rotations and find the one with minimum total distance
        for offset in range(n):
            rotated_b = np.roll(contour_b, -offset, axis=0)
            total_distance = np.sum(np.linalg.norm(contour_a - rotated_b, axis=1))

            if total_distance < min_distance:
                min_distance = total_distance
                best_offset = offset

        return best_offset

    def _find_best_rotation_offset_variable(self, contour_a, contour_b):
        """
        Find rotation offset for contours with different sizes.
        Uses direction from centroid to first vertex as reference.
        """
        center_a = np.mean(contour_a, axis=0)
        center_b = np.mean(contour_b, axis=0)

        # Direction from centroid to first vertex of A
        ref_dir = contour_a[0] - center_a
        ref_dir = ref_dir / (np.linalg.norm(ref_dir) + 1e-10)

        # Find vertex in B closest to this reference direction
        dirs_b = contour_b - center_b
        norms_b = np.linalg.norm(dirs_b, axis=1, keepdims=True)
        norms_b[norms_b < 1e-10] = 1
        dirs_b = dirs_b / norms_b

        dots = dirs_b @ ref_dir
        best_offset = np.argmax(dots)

        return best_offset

    def _create_contour_band_variable(self, start_a, start_b, n_a, n_b):
        """Create triangular band between two contours with different vertex counts."""
        faces = []

        # Use ratio-based mapping
        for i in range(n_a):
            i_next = (i + 1) % n_a

            # Map to contour B indices
            j = int(i * n_b / n_a) % n_b
            j_next = int(i_next * n_b / n_a) % n_b

            # Triangle 1
            faces.append([start_a + i, start_a + i_next, start_b + j])

            # Triangle 2 (if j changes)
            if j != j_next:
                faces.append([start_a + i_next, start_b + j_next, start_b + j])

        return faces

    def _create_contour_band(self, start_a, start_b, num_samples):
        """Create triangular band between two contours with same vertex count."""
        faces = []
        for i in range(num_samples):
            i_next = (i + 1) % num_samples
            # Two triangles per quad
            # Triangle 1: (A[i], A[i+1], B[i])
            faces.append([start_a + i, start_a + i_next, start_b + i])
            # Triangle 2: (A[i+1], B[i+1], B[i])
            faces.append([start_a + i_next, start_b + i_next, start_b + i])
        return faces

    def _compute_contour_mesh_normals(self):
        """Compute vertex normals for the contour mesh."""
        if self.contour_mesh_vertices is None or self.contour_mesh_faces is None:
            return

        num_vertices = len(self.contour_mesh_vertices)
        normals = np.zeros((num_vertices, 3), dtype=np.float32)

        for face in self.contour_mesh_faces:
            v0, v1, v2 = self.contour_mesh_vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(face_normal)
            if norm > 1e-10:
                face_normal = face_normal / norm

            for vi in face:
                normals[vi] += face_normal

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1
        self.contour_mesh_normals = normals / norms

    def draw_contour_mesh(self):
        """Draw the contour mesh using OpenGL."""
        if self.contour_mesh_vertices is None or self.contour_mesh_faces is None:
            return

        glPushMatrix()
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        color = self.contour_mesh_color
        alpha = self.contour_mesh_transparency
        glColor4f(color[0], color[1], color[2], alpha)

        glBegin(GL_TRIANGLES)
        for face in self.contour_mesh_faces:
            for vi in face:
                if self.contour_mesh_normals is not None:
                    glNormal3fv(self.contour_mesh_normals[vi])
                glVertex3fv(self.contour_mesh_vertices[vi])
        glEnd()

        glPopMatrix()

    # _cut_contour_voronoi, _cut_contour_angular, _cut_contour_gradient,
    # _smooth_assignments_circular are inherited from MuscleMeshMixin

    def find_contour_stream(self, skeleton_meshes=None):
        origin_num = len(self.bounding_planes[0])
        insertion_num = len(self.bounding_planes[-1])

        # Stream Search
        contour_stream_match = []
        check_count = 0

        # Procedure will start from origin or insertion where stream number is larger
        if origin_num >= insertion_num:
            self.draw_contour_stream = [True] * origin_num
            ordered_contours_trim = [[self.vertices[edge_group]] for i, edge_group in enumerate(self.edge_groups) if self.edge_classes[i] == 'origin']  
            ordered_contours_trim_orig = [[self.vertices[edge_group].copy()] for i, edge_group in enumerate(self.edge_groups) if self.edge_classes[i] == 'origin']  
            bounding_planes_trim = [[bounding_plane] for bounding_plane in self.bounding_planes[0]]
            contour_stream_match.append([[i] for i in range(origin_num)])

            contour_bounding_zip = zip (self.contours[1:], self.bounding_planes[1:])

        else:
            self.draw_contour_stream = [True] * insertion_num
            ordered_contours_trim = [[self.vertices[edge_group]] for i, edge_group in enumerate(self.edge_groups) if self.edge_classes[i] == 'insertion']  
            ordered_contours_trim_orig = [[self.vertices[edge_group].copy()] for i, edge_group in enumerate(self.edge_groups) if self.edge_classes[i] == 'insertion']
            bounding_planes_trim = [[bounding_plane] for bounding_plane in self.bounding_planes[-1]]
            contour_stream_match.append([[i] for i in range(insertion_num)])

            contour_bounding_zip = zip(self.contours[-2::-1], self.bounding_planes[-2::-1])

        division_start = False
        division_start_index = 0
        zip_num = len(self.bounding_planes)
        zip_index = 0
        # contours from reversed self.contours
        for next_contours, next_bounding_planes in contour_bounding_zip:
            # Two cases: 

            # 1. next contours num is same as current num
            # In this case, each group must link to one next contour

            # 2. next contours num is smaller than current num
            # In this case, some next contours must link to more than one current contour
            
            # For both cases, if there's a group that is not linked to any next contour,
            # it means scalar value is not small enough to slice the mesh
            
            linked = [[] for _ in range(len(ordered_contours_trim))]
            match = [[] for _ in range(len(next_contours))]
            counts = [0] * len(next_contours)

            if self.link_mode == 'vertex':
                for i, bounding_plane_stream in enumerate(bounding_planes_trim):
                    mean = bounding_plane_stream[-1]['mean']
                    basis_z = bounding_plane_stream[-1]['basis_z']

                    min_distance = np.inf
                    closest = None
                    for contour_i, contour in enumerate(next_contours):
                        projected_contour = [point - np.dot(point - mean, basis_z) * basis_z for point in contour]
                        distances = np.linalg.norm(mean - projected_contour, axis=1)

                        cand_distance = np.min(distances)
                        if cand_distance < min_distance:
                            min_distance = cand_distance
                            closest = contour_i
                    
                    linked[i].append(closest)
                    match[closest].append(i)
                    counts[closest] += 1
            else:   # link_mode = 'mean'
                next_means = np.array([np.mean(contour, axis=0) for contour in next_contours])
                next_basis_z = [bounding_plane['basis_z'] for bounding_plane in next_bounding_planes]
                # print(len(ordered_contours_trim), linked, counts)
                for i, bounding_planes in enumerate(bounding_planes_trim):
                    # mean = np.mean(contours[-1], axis=0)
                    mean = bounding_planes[-1]['mean']
                    # basis_z = bounding_planes[-1]['basis_z']

                    projected_means = [mean - np.dot(mean - next_mean, basis_z) * basis_z for next_mean, basis_z in zip(next_means, next_basis_z)]
                    distances = np.linalg.norm([mean - projected_mean for mean, projected_mean in zip(next_means, projected_means)], axis=1)

                    # distances = np.linalg.norm(mean - next_means, axis=1)

                    closest = np.argmin(distances)
                    linked[i].append(closest)
                    match[closest].append(i)
                    counts[closest] += 1

            if not all(count > 0 for count in counts):
                print("Something's wrong for next contours linking")
                return

            # print(linked, match, counts)
            contour_stream_match.append(match)

            if len(next_contours) == len(ordered_contours_trim):
                for i in range(len(ordered_contours_trim)):
                    closest = linked[i][0]
                    ordered_contours_trim[i].append(next_contours[closest])
                    ordered_contours_trim_orig[i].append(next_contours[closest].copy())
                    # bounding_planes_trim[i].append(next_bounding_planes[closest])
                    _, bounding_plane = self.save_bounding_planes(next_contours[closest], next_bounding_planes[closest]['scalar_value'], 
                                                                    prev_bounding_plane=bounding_planes_trim[i][-1])
                    bounding_planes_trim[i].append(bounding_plane)
            else:
                for i, count in enumerate(counts):
                    # Multiple contours are linked to the same next_contour
                    if count > 1:
                        target_next_contour = next_contours[i]
                        target_next_bounding_plane_info = next_bounding_planes[i]
                        target_next_contour_match = target_next_bounding_plane_info['contour_match']
                        target_next_bounding_vertices = target_next_bounding_plane_info['bounding_plane']
                        v0, v1, v2, v3 = target_next_bounding_vertices
                        
                        #  v3  -------------- v2
                        #  |                   |
                        #  |                   |
                        #  |                   |
                        #  v0  -------------- v1

                        w0 = np.array([0, 0, 0])
                        w1 = np.array([1, 0, 0])
                        w2 = np.array([1, 1, 0])
                        w3 = np.array([0, 1, 0])

                        # make 3x4 matrix from v0, v1, v2, v3
                        A = np.array([v0, v1, v2, v3]).T
                        B = np.array([w0, w1, w2, w3]).T
                        A_pinv = np.linalg.pinv(A)
                        H = B @ A_pinv

                        Ps = np.array([pair[0] for pair in target_next_contour_match])
                        Qs = np.array([pair[1] for pair in target_next_contour_match])
                        # normalized_Ps = np.dot(H, np.array(Ps).T).T
                        normalized_Qs = np.dot(H, np.array(Qs).T).T
                        normalized_Qs = np.clip(normalized_Qs, 0, 1)

                        target_next_mean = target_next_bounding_plane_info['mean']
                        target_next_basis_z = target_next_bounding_plane_info['basis_z']

                        target_current_contours_indices = [j for j in range(len(linked)) if linked[j][0] == i]
                        # target_current_contours = [ordered_contours_trim[j][-1] for j in target_current_contours_indices]
                        target_current_bounding_plane_info = [bounding_planes_trim[j][-1] for j in target_current_contours_indices]
                        target_current_contours_areas = [plane_info['area'] for plane_info in target_current_bounding_plane_info]
                        # target_current_contours_areas = [np.abs(np.linalg.norm(v1[0] - v0[0]) * 
                        #                                  np.linalg.norm(v3[1] - v0[1])) for v0, v1, v2, v3 
                        #                                  in [plane_info['bounding_plane'] for plane_info in target_current_bounding_plane_info]]
                        print(target_current_contours_areas)

                        target_current_contours_means = [plane_info['mean'] for plane_info in target_current_bounding_plane_info]
                        projected_target_current_contours_means = [mean - np.dot(mean - target_next_mean, target_next_basis_z) * target_next_basis_z for mean in target_current_contours_means]

                        # Choose cutting method
                        if self.cutting_method == 'voronoi':
                            # Voronoi-based cutting with smoothing
                            new_target_next_contours = self._cut_contour_voronoi(
                                target_next_contour,
                                target_next_contour_match,
                                projected_target_current_contours_means,
                                target_next_basis_z,
                                smooth_window=5
                            )
                        elif self.cutting_method == 'angular':
                            # Angular/radial cutting with pie-slice sectors
                            new_target_next_contours = self._cut_contour_angular(
                                target_next_contour,
                                target_next_contour_match,
                                projected_target_current_contours_means,
                                target_next_basis_z
                            )
                        elif self.cutting_method == 'gradient':
                            # Gradient-based cutting at natural transition points
                            new_target_next_contours = self._cut_contour_gradient(
                                target_next_contour,
                                target_next_contour_match,
                                projected_target_current_contours_means,
                                target_next_basis_z
                            )
                        else:
                            # Area-based cutting (original method)
                            structure_vector = np.array([0.0, 0.0, 0.0])
                            # longest_distance = 0
                            start_point = None
                            end_point = None

                            for mean_i in range(len(projected_target_current_contours_means)):
                                for mean_j in range(mean_i + 1, len(projected_target_current_contours_means)):
                                    cand_vector = projected_target_current_contours_means[mean_i] - projected_target_current_contours_means[mean_j]

                                    if np.dot(cand_vector, np.array([1, 0, 0])) < 0:
                                        cand_vector *= -1
                                    structure_vector += np.array(cand_vector)
                            structure_vector /= len(projected_target_current_contours_means)

                            # projected_means_mean = np.array(projected_target_current_contours_means)
                            # projected_means_mean = np.mean(projected_means_mean, axis=0)
                            structure_vector /= np.linalg.norm(structure_vector)

                            horizontal_vector = v1 - v0
                            vertical_vector = v3 - v0

                            horizontal_vector /= np.linalg.norm(horizontal_vector)
                            vertical_vector /= np.linalg.norm(vertical_vector)

                            horizontal_projection = np.abs(np.dot(structure_vector, horizontal_vector))
                            vertical_projection = np.abs(np.dot(structure_vector, vertical_vector))

                            if horizontal_projection > vertical_projection:
                                # print()
                                # print(f"Horizontal: {horizontal_projection} > {vertical_projection}")
                                # project target_current_contours_means onto line made by horizontal vector and target_next_mean
                                target_values = [np.dot(mean - target_next_mean, horizontal_vector) for mean in projected_target_current_contours_means]
                                target_axis = 0
                            else:
                                target_values = [np.dot(mean - target_next_mean, vertical_vector) for mean in projected_target_current_contours_means]
                                target_axis = 1
                            target_order = np.argsort(target_values)


                            # if accumul_areas is None:
                            #     sorted_areas = [target_current_contours_areas[index] for index in target_order]
                            #     accumul_areas = np.cumsum(sorted_areas, dtype=float)
                            #     accumul_areas /= accumul_areas[-1]

                            # sorted_areas = [target_current_contours_areas[index] for index in target_order]
                            # # sorted_areas = [area + np.mean(sorted_areas) for area in sorted_areas]

                            # # sorted_areas = [1 / len(target_order)] * len(target_order)

                            # sorted_areas = []
                            # mul_num = 1.5
                            # for area_i in range(len(target_order)):
                            #     if area_i == 0 or area_i == len(target_order) - 1:
                            #         sorted_areas.append(1)
                            #     else:
                            #         sorted_areas.append(mul_num)

                            # accumul_areas = np.cumsum(sorted_areas, dtype=float)
                            # accumul_areas /= accumul_areas[-1]

                            if not division_start:
                                division_start = True
                                division_start_index = zip_index

                            sorted_areas1 = [target_current_contours_areas[index] for index in target_order]
                            accumul_areas1 = np.cumsum(sorted_areas1, dtype=float)
                            accumul_areas1 /= accumul_areas1[-1]

                            sorted_areas2 = []
                            mul_num = 1.5
                            for area_i in range(len(target_order)):
                                if area_i == 0 or area_i == len(target_order) - 1:
                                    sorted_areas2.append(1)
                                else:
                                    sorted_areas2.append(mul_num)
                            accumul_areas2 = np.cumsum(sorted_areas2, dtype=float)
                            accumul_areas2 /= accumul_areas2[-1]

                            # ratio2 = (zip_index - division_start_index) / (zip_num - division_start_index)
                            ratio2 = (zip_index) / (zip_num)
                            ratio1 = 1 - ratio2
                            accumul_areas = accumul_areas1 * ratio1 + accumul_areas2 * ratio2

                            # target_basis_x = target_next_bounding_plane_info['basis_x']
                            # target_basis_y = target_next_bounding_plane_info['basis_y']
                            # target_basis_z = target_next_bounding_plane_info['basis_z']
                            # proj_basis_xs = [np.dot(basis_x, target_basis_x) * target_basis_x + np.dot(basis_x, target_basis_y) * target_basis_y for basis_x in [basis_x, prev_basis_x, next_basis_x]]
                            # new_basis_x = np.mean(proj_basis_xs, axis=0)
                            # new_basis_x = new_basis_x / np.linalg.norm(new_basis_x)
                            # # new_basis_x = (new_basis_x + target_basis_x) / 2
                            # new_basis_y = np.cross(target_basis_z, new_basis_x)
                            # new_plane_info = {
                            #     'basis_x': new_basis_x,
                            #     'basis_y': new_basis_y,
                            #     'basis_z': target_basis_z,
                            #     'newell_normal': target_basis_z,
                            # }
                            # _, new_bounding_plane = self.save_bounding_planes(target_next_contour, target_next_bounding_plane_info['scalar_value'],
                            #                                                   prev_bounding_plane=target_next_bounding_plane_info,
                            #                                                   bounding_plane_info_orig = new_plane_info)
                            # target_next_bounding_plane_info = new_bounding_plane

                            # print(accumul_areas)

                            prev_inserted_index = None
                            prev_target_value = None
                            for index, accumul_area in enumerate(accumul_areas):
                                if normalized_Qs[-1][target_axis] <= accumul_area:
                                    prev_inserted_index = target_order[index]
                                    prev_target_value = normalized_Qs[-1][target_axis]
                                    break
                            new_target_next_contours = [[] for _ in range(len(target_current_contours_indices))]

                            for Q_index, Q in enumerate(normalized_Qs):
                                target_value = Q[target_axis]

                                for index, accumul_area in enumerate(accumul_areas):
                                    if target_value <= accumul_area:
                                        # if index == 0:
                                        #     print(f'{target_value} <= {accumul_area}')
                                        # else:
                                        #     print(f'{accumul_areas[index - 1]} < {target_value} <= {accumul_area}')

                                        inserted_index = target_order[index]
                                        if prev_inserted_index is not None and prev_inserted_index != inserted_index:
                                            # print('prev index and current index are different')
                                            # print(f'prev index: {prev_inserted_index}, current index: {inserted_index}')
                                            # print(f'prev value: {prev_target_value}, current value: {target_value}')
                                            prev_P = Ps[Q_index - 1]
                                            P = Ps[Q_index]

                                            # find index of inserted_index from target_order
                                            idx1 = np.where(target_order == prev_inserted_index)[0][0]
                                            idx2 = np.where(target_order == inserted_index)[0][0]

                                            if idx1 <= idx2:
                                                # forward
                                                path1 = list(range(idx1, idx2 + 1))
                                                # reverse
                                                path2 = np.concatenate([range(idx1, -1, -1), range(len(target_order) - 1, idx2 - 1, -1)])
                                            else:
                                                # reverse
                                                path1 = list(range(idx1, idx2 - 1, -1))
                                                # forward
                                                path2 = np.concatenate([range(idx1, len(target_order)), range(0, idx2 + 1)])

                                            # Ascending or Descending Violation must be checked
                                            wanted_order = path1
                                            temp_areas = accumul_areas[wanted_order]
                                            if temp_areas[0] < temp_areas[-1]:
                                                # Ascending order
                                                for i in range(len(temp_areas) - 1):
                                                    if temp_areas[i] > temp_areas[i + 1]:
                                                        wanted_order = path2
                                                        break
                                            else:
                                                # Descending order
                                                for i in range(len(temp_areas) - 1):
                                                    if temp_areas[i] < temp_areas[i + 1]:
                                                        wanted_order = path2
                                                        break

                                            # print(f'path1: {path1}, path2: {path2}')
                                            # print(f'target order: {target_order[wanted_order]}')
                                            # print(f'accumul areas: {accumul_areas[wanted_order]}')

                                            if prev_target_value < target_value:
                                                dividing_values = np.concatenate([[prev_target_value], accumul_areas[wanted_order[:-1]], [target_value]])
                                            else:
                                                dividing_values = np.concatenate([[prev_target_value], accumul_areas[wanted_order[1:]], [target_value]])
                                            # print(f'dividing values: {dividing_values}')

                                            dividing_order = target_order[wanted_order]

                                            for dividing_index in range(len(dividing_order) - 1):
                                                weight_P = (dividing_values[0] - dividing_values[dividing_index + 1]) / (dividing_values[0] - dividing_values[-1])
                                                weight_prev_P = 1 - weight_P

                                                mid_P = weight_P * P + weight_prev_P * prev_P
                                                mid_Q = weight_P * Q + weight_prev_P * normalized_Qs[Q_index - 1]

                                                new_target_next_contours[dividing_order[dividing_index]].append(mid_P)
                                                new_target_next_contours[dividing_order[dividing_index + 1]].append(mid_P)
                                                # print(f'mid_P corresponding {mid_Q} inserted into {dividing_order[dividing_index]} and {dividing_order[dividing_index + 1]}')

                                        # print(f'P corresponding {Q} inserted into {inserted_index}')
                                        new_target_next_contours[inserted_index].append(Ps[Q_index])
                                        prev_inserted_index = inserted_index
                                        prev_target_value = target_value
                                        break

                        # [print(contour) for contour in new_target_next_contours]

                        for index, j in enumerate(target_current_contours_indices):
                            if len(new_target_next_contours[index]) > 0:
                                ordered_contours_trim[j].append(new_target_next_contours[index])
                                ordered_contours_trim_orig[j].append(new_target_next_contours[index].copy())
                                # bounding_planes_trim[j].append(next_bounding_planes[i])
                                # _, bounding_plane = self.save_bounding_planes(new_target_next_contours[index], next_bounding_planes[0]['scalar_value'],
                                #                                                          prev_bounding_plane=bounding_planes_trim[j][-1])
                                _, bounding_plane = self.save_bounding_planes(new_target_next_contours[index], next_bounding_planes[0]['scalar_value'],
                                                                                prev_bounding_plane=bounding_planes_trim[j][-1],
                                                                                bounding_plane_info_orig=target_next_bounding_plane_info)
                                bounding_planes_trim[j].append(bounding_plane)

                        check_count += 1
                        # print(f"Check count: {check_count}")

                        '''
                        # Tried to divide the contour simply by vertices or mean point, but not pretty
                        
                        target_next_contour_mean = target_next_bounding_plane['mean']
                        target_next_contour_basis_z = target_next_bounding_plane['basis_z']

                        target_current_contours_means = [np.mean(contour, axis=0) for contour in target_current_contours]
                        # target_current_contours_basis = [bounding_planes_trim[j][-1]['basis_z'] for j in target_current_contours_indices]

                        # project target_current_contours_means onto target_next_bounding_plane
                        projected_means = [mean - np.dot(mean - target_next_contour_mean, target_next_contour_basis_z) * target_next_contour_basis_z for mean in target_current_contours_means]
                    
                        new_target_next_contours = [[] for _ in range(len(target_current_contours))]
                        vertex_matches = []

                        for p in target_next_contour:
                            # distances = np.linalg.norm(p - target_current_contours_means, axis=1)
                            # distances = np.linalg.norm(np.cross(p - target_current_contours_means, target_current_contours_basis), axis=1)
                            distances= np.linalg.norm(np.cross(p - projected_means, target_next_contour_basis_z), axis=1)
                            closest = np.argmin(distances)
                            # new_target_next_contours[closest].append(p)
                            vertex_matches.append(closest)

                        for index in range(len(vertex_matches)):
                            if vertex_matches[index] != vertex_matches[index - 1]:
                                mean = np.mean([target_next_contour[index], target_next_contour[index - 1]], axis=0)
                                new_target_next_contours[vertex_matches[index - 1]].append(mean)
                                new_target_next_contours[vertex_matches[index - 1]].append(target_next_contour_mean)
                                new_target_next_contours[vertex_matches[index]].append(target_next_contour_mean)
                                new_target_next_contours[vertex_matches[index]].append(mean)
                                
                            new_target_next_contours[vertex_matches[index]].append(target_next_contour[index])

                        # [print(len(contour)) for contour in new_target_next_contours]

                        for index, j in enumerate(target_current_contours_indices):
                            if len(new_target_next_contours[index]) > 0:
                                ordered_contours_trim[j].append(new_target_next_contours[index])
                                # bounding_planes_trim[j].append(next_bounding_planes[i])
                                bounding_planes_trim[j].append(self.save_bounding_planes(new_target_next_contours[index], contour_values[-1]))
                        '''
                    else:
                        target_current_contours_index = [j for j in range(len(linked)) if linked[j][0] == i][0]
                        ordered_contours_trim[target_current_contours_index].append(next_contours[i])
                        ordered_contours_trim_orig[target_current_contours_index].append(next_contours[i].copy())
                        # bounding_planes_trim[target_current_contours_index].append(next_bounding_planes[i])
                        _, bounding_plane = self.save_bounding_planes(next_contours[i], next_bounding_planes[i]['scalar_value'], 
                                                                        prev_bounding_plane=bounding_planes_trim[target_current_contours_index][-1])
                        bounding_planes_trim[target_current_contours_index].append(bounding_plane)

            self.contours = ordered_contours_trim
            self.bounding_planes = bounding_planes_trim

            zip_index += 1


        # After stream search
        #
        #         Stream 1      Stream 2
        #       ------------/-----------
        #      ------------/-------------
        #                ...
        #      ----------     ---------


        for stream_i, bounding_plane_stream in enumerate(self.bounding_planes):
            for i, bounding_plane_info in enumerate(bounding_plane_stream):
                if bounding_plane_info['square_like']:
                    basis_x = bounding_plane_info['basis_x']
                    basis_y = bounding_plane_info['basis_y']
                    basis_z = bounding_plane_info['basis_z']
                    mean = bounding_plane_info['mean']

                    prev_plane_info = None
                    if origin_num >= insertion_num:
                        prev_range = range(i - 1, -1, -1)
                    else:
                        prev_range = range(i + 1, len(bounding_plane_stream))
                    for j in prev_range:
                        # if not self.bounding_planes[j]['square_like']:
                        if not bounding_plane_stream[j]['square_like']:
                            prev_plane_info = bounding_plane_stream[j]
                            break

                    next_plane_info = None
                    if origin_num >= insertion_num:
                        next_range = range(i + 1, len(bounding_plane_stream))
                    else:
                        next_range = range(i - 1, -1, -1)
                    for j in next_range:
                        # if not self.bounding_planes[j]['square_like']:
                        if not bounding_plane_stream[j]['square_like']:
                            next_plane_info = bounding_plane_stream[j]
                            break

                    if prev_plane_info is not None and next_plane_info is not None:
                        prev_basis_x = prev_plane_info['basis_x']
                        prev_mean = prev_plane_info['mean']

                        next_basis_x = next_plane_info['basis_x']
                        next_mean = next_plane_info['mean']

                        prev_distance = np.linalg.norm(mean - prev_mean)
                        next_distance = np.linalg.norm(mean - next_mean)
                        sum_distance = prev_distance + next_distance

                        new_basis_x_prev = np.dot(prev_basis_x, basis_x) * basis_x + np.dot(prev_basis_x, basis_y) * basis_y
                        new_basis_x_prev /= np.linalg.norm(new_basis_x_prev)

                        new_basis_x_next = np.dot(next_basis_x, basis_x) * basis_x + np.dot(next_basis_x, basis_y) * basis_y
                        new_basis_x_next /= np.linalg.norm(new_basis_x_next)

                        bounding_plane_info['basis_x'] = next_distance / sum_distance * new_basis_x_prev + prev_distance / sum_distance * new_basis_x_next
                        bounding_plane_info['basis_x'] /= np.linalg.norm(bounding_plane_info['basis_x'])
                        bounding_plane_info['basis_y'] = np.cross(basis_z, bounding_plane_info['basis_x'])
                        bounding_plane_info['basis_y'] = bounding_plane_info['basis_y'] / (np.linalg.norm(bounding_plane_info['basis_y']) + 1e-10)

                        basis_x = bounding_plane_info['basis_x']
                        basis_y = bounding_plane_info['basis_y']

                        # print(stream_i, len(ordered_contours_trim_orig))
                        # print(i, len(ordered_contours_trim_orig[stream_i]))
                        projected_2d = np.array([[np.dot(v - mean, basis_x), np.dot(v - mean, basis_y)] for v in ordered_contours_trim_orig[stream_i][i]])
                        area = compute_polygon_area(projected_2d)

                        min_x, max_x = np.min(projected_2d[:, 0]), np.max(projected_2d[:, 0])
                        min_y, max_y = np.min(projected_2d[:, 1]), np.max(projected_2d[:, 1])

                        bounding_plane_2d = np.array([
                            [min_x, min_y], [max_x, min_y],
                            [max_x, max_y], [min_x, max_y]
                        ])
                        bounding_plane = np.array([mean + x * basis_x + y * basis_y for x, y in bounding_plane_2d])
                        projected_2d = np.array([mean + x * basis_x + y * basis_y for x, y in projected_2d])

                        self.contours[stream_i][i], contour_match = self.find_contour_match(ordered_contours_trim_orig[stream_i][i], bounding_plane)

                        bounding_plane_info['bounding_plane'] = bounding_plane
                        bounding_plane_info['projected_2d'] = projected_2d
                        bounding_plane_info['area'] = area
                        bounding_plane_info['contour_match'] = contour_match

        for _ in range(10):
            smooth_bounding_planes = [[] for _ in range(len(self.bounding_planes))]
            for stream_i, bounding_plane_stream in enumerate(self.bounding_planes):
                for i, bounding_plane_info in enumerate(bounding_plane_stream):
                    if i == 0 or i == len(bounding_plane_stream) - 1:
                        smooth_bounding_planes[stream_i].append(bounding_plane_info)
                        continue

                    prev_plane_info = bounding_plane_stream[i - 1] if i > 0 else bounding_plane_info
                    next_plane_info = bounding_plane_stream[i + 1] if i < len(bounding_plane_stream) - 1 else bounding_plane_info

                    basis_x = bounding_plane_info['basis_x']
                    basis_y = bounding_plane_info['basis_y']
                    basis_z = bounding_plane_info['basis_z']
                    mean = bounding_plane_info['mean']
                    
                    prev_basis_x = prev_plane_info['basis_x']
                    prev_basis_y = prev_plane_info['basis_y']
                    prev_basis_z = prev_plane_info['basis_z']
                    prev_mean = prev_plane_info['mean']
                    prev_distance = np.linalg.norm(mean - prev_mean)

                    next_basis_x = next_plane_info['basis_x']
                    next_basis_y = next_plane_info['basis_y']
                    next_basis_z = next_plane_info['basis_z']
                    next_mean = next_plane_info['mean']
                    next_distance = np.linalg.norm(mean - next_mean)

                    distance_sum = prev_distance + next_distance

                    prev_x_proj = np.dot(prev_basis_x, basis_x) * basis_x + np.dot(prev_basis_x, basis_y) * basis_y
                    next_x_proj = np.dot(next_basis_x, basis_x) * basis_x + np.dot(next_basis_x, basis_y) * basis_y

                    prev_x_proj /= np.linalg.norm(prev_x_proj)
                    next_x_proj /= np.linalg.norm(next_x_proj)

                    # new_basis_x = ((prev_x_proj + next_x_proj) / 2 + basis_x) / 2
                    # # new_basis_x = (prev_x_proj + next_x_proj) / 2
                    # new_basis_x /= np.linalg.norm(new_basis_x)
                    # new_basis_y = np.cross(basis_z, new_basis_x)

                    # if i == 0 or i == len(bounding_plane_stream) - 1:
                    #     new_basis_x = (next_basis_x + basis_x * 3) / 4
                    #     new_basis_z = basis_z
                    #     new_basis_y = np.cross(new_basis_x, new_basis_z)
                    # else:
                    #     new_basis_x = ((prev_basis_x + next_basis_x) / 2 + basis_x) / 2
                    #     new_basis_y = ((prev_basis_y + next_basis_y) / 2 + basis_y) / 2
                    #     new_basis_z = np.cross(new_basis_x, new_basis_y)

                    # new_basis_x = ((prev_basis_x + next_basis_x) / 2 + basis_x) / 2
                    # new_basis_y = ((prev_basis_y + next_basis_y) / 2 + basis_y) / 2
                    new_basis_x = ((prev_basis_x * next_distance + next_basis_x * prev_distance) / distance_sum + basis_x) / 2
                    new_basis_z = ((prev_basis_z * next_distance + next_basis_z * prev_distance) / distance_sum + basis_z) / 2

                    new_basis_x /= np.linalg.norm(new_basis_x)
                    new_basis_z /= np.linalg.norm(new_basis_z)
                    # Re-orthogonalize: compute basis_y from cross product to ensure 90-degree corners
                    new_basis_y = np.cross(new_basis_z, new_basis_x)
                    new_basis_y /= (np.linalg.norm(new_basis_y) + 1e-10)
                    new_newell = new_basis_z

                    new_mean = prev_mean + np.dot(mean - prev_mean, basis_z) / np.dot(next_mean - prev_mean, basis_z) * (next_mean - prev_mean)
                    new_mean = (new_mean + mean) / 2
                    # new_mean = mean
                    # new_mean = (prev_mean + next_mean) / 2

                    projected_2d = np.array([[np.dot(v - new_mean, new_basis_x), np.dot(v - new_mean, new_basis_y)] for v in ordered_contours_trim_orig[stream_i][i]])
                    area = compute_polygon_area(projected_2d)

                    min_x, max_x = np.min(projected_2d[:, 0]), np.max(projected_2d[:, 0])
                    min_y, max_y = np.min(projected_2d[:, 1]), np.max(projected_2d[:, 1])

                    bounding_plane_2d = np.array([
                        [min_x, min_y], [max_x, min_y],
                        [max_x, max_y], [min_x, max_y]
                    ])
                    bounding_plane = np.array([new_mean + x * new_basis_x + y * new_basis_y for x, y in bounding_plane_2d])
                    projected_2d = np.array([new_mean + x * new_basis_x + y * new_basis_y for x, y in projected_2d])

                    # if i < 2:
                    #     print(prev_basis_x)
                    #     print(basis_x, basis_y, mean)
                    #     print(next_basis_x)
                    #     print(new_basis_x, new_basis_y, new_mean)

                    self.contours[stream_i][i], contour_match = self.find_contour_match(ordered_contours_trim_orig[stream_i][i], bounding_plane)

                    new_bounding_plane_info = {
                        'basis_x': new_basis_x,
                        'basis_y': new_basis_y,
                        'basis_z': basis_z,
                        'mean': new_mean,
                        'bounding_plane': bounding_plane,
                        'projected_2d': projected_2d,
                        'area': area,
                        'contour_match': contour_match,
                        'scalar_value': bounding_plane_info['scalar_value'],
                        'square_like': bounding_plane_info['square_like'],

                        # 'newell_normal': bounding_plane_info['newell_normal'],
                        'newell_normal': new_newell,
                    }

                    smooth_bounding_planes[stream_i].append(new_bounding_plane_info)

            self.bounding_planes = smooth_bounding_planes

        area_threshold = 1.75
        angle_threshold = 10
        length_min_threshold = scale * 0.2
        length_max_threshold = scale * 1.5

        stream_num = len(self.contours)
        contours_trim = [[] for _ in range(stream_num)]
        bounding_planes_trim = [[] for _ in range(stream_num)]
        level_num = len(self.bounding_planes[0])

        prev_bounding_planes = [self.bounding_planes[stream_i][0] for stream_i in range(stream_num)]
        end_means = [self.bounding_planes[stream_i][-1]['mean'] for stream_i in range(stream_num)]
        for i in range(level_num):
            if i == 0 or i == level_num - 1:
                for stream_i in range(stream_num):
                    contours_trim[stream_i].append(self.contours[stream_i][i])
                    bounding_planes_trim[stream_i].append(self.bounding_planes[stream_i][i])
                continue

            add_necessary = True
            for stream_i in range(stream_num):
                curr_area = self.bounding_planes[stream_i][i]['area']
                prev_area = prev_bounding_planes[stream_i]['area']

                area_check = curr_area / prev_area if curr_area > prev_area else prev_area / curr_area
                
                prev_basis_x = prev_bounding_planes[stream_i]['basis_x']
                prev_basis_y = prev_bounding_planes[stream_i]['basis_y']
                prev_basis_z = prev_bounding_planes[stream_i]['basis_z']

                basis_x = self.bounding_planes[stream_i][i]['basis_x']
                basis_y = self.bounding_planes[stream_i][i]['basis_y']
                basis_z = self.bounding_planes[stream_i][i]['basis_z']

                prev_rot = np.vstack([prev_basis_x, prev_basis_y, prev_basis_z]).T
                new_rot = np.vstack([basis_x, basis_y, basis_z]).T

                R_rel = new_rot @ prev_rot.T

                trace = np.trace(R_rel)
                angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0)) * 180 / np.pi

                mean = self.bounding_planes[stream_i][i]['mean']
                prev_mean = prev_bounding_planes[stream_i]['mean']
                distance = np.linalg.norm(mean - prev_mean)
                distance_end = np.linalg.norm(mean - end_means[stream_i])

                if (area_check > area_threshold or angle > angle_threshold) and distance > length_min_threshold and distance_end > length_min_threshold:
                    continue
                elif distance > length_max_threshold and distance_end > length_max_threshold:
                    continue
                else:
                    add_necessary = False
                    break

            if add_necessary:
                for stream_i in range(stream_num):
                    contours_trim[stream_i].append(self.contours[stream_i][i])
                    bounding_planes_trim[stream_i].append(self.bounding_planes[stream_i][i])

                    prev_bounding_planes[stream_i] = self.bounding_planes[stream_i][i]

        self.contours = contours_trim
        self.bounding_planes = bounding_planes_trim

        # self.structure_vectors = []
        # for stream_i, bounding_plane_stream in enumerate(self.bounding_planes):
        #     for i in range(len(bounding_plane_stream) - 1):
        #         self.structure_vectors.append((bounding_plane_stream[i]['mean'], bounding_plane_stream[i + 1]['mean']))

        '''
        if origin_num < insertion_num:
            contour_stream_match = contour_stream_match[::-1]
            self.bounding_planes = [stream[::-1] for stream in self.bounding_planes]

        length_thr = 0.01 * scale
        for contour_set in contours_orig:
            for contour in contour_set:
                cut_finished = False
                while cut_finished:
                    for i in range(len(contour)):
                        start = contour[i - 1]
                        end = contour[i]

                        if np.linalg.norm(start - end) < length_thr:
                            # find minimum number that divides the line segment into equal length segments shorter than length_thr
                            num_div = int(np.ceil(np.linalg.norm(start - end) / length_thr))
                            for j in range(1, num_div):
                                new_point = start + (end - start) * j / num_div
                                contour.insert(i, new_point)
                                i += 1
                            cut_finished = False
                            break

                    if i == len(contour) - 1:
                        cut_finished = True

        new_contours = [[] for _ in range(max(origin_num, insertion_num))]
        for i, (matches, contour) in enumerate(zip(contour_stream_match, contours_orig)):   # 1, 1 + scalar_step, ..., 10 - scalar_step, 10
            for stream in new_contours:
                stream.append([])

            for match_i, match in enumerate(matches):
                matched_contour = contour[match_i]
                means = [self.bounding_planes[index][i]['mean'] for index in match]

                prev_v = matched_contour[-1]
                distances = np.linalg.norm(prev_v - means, axis=1)
                prev_closest = np.argmin(distances)
                for v in matched_contour:
                    distances = np.linalg.norm(v - means, axis=1)
                    closest = np.argmin(distances)
                    if closest != prev_closest:
                        mean_v = (v + prev_v) / 2.0
                        new_contours[match[prev_closest]][-1].append(mean_v)
                        new_contours[match[closest]][-1].append(mean_v)
                        self.structure_vectors.append((means[prev_closest], mean_v))
                        self.structure_vectors.append((means[closest], mean_v))

                    new_contours[match[closest]][-1].append(v)
                    self.structure_vectors.append((v, means[closest]))
                    prev_closest = closest
                    prev_v = v

            # break

        self.contours = new_contours
        '''

        if self.is_one_fiber:
            self.fiber_architecture = [np.array([0.5, 0.5])]
            triplet_num = len(self.contours[0]) // 3
            self.contours = [[self.contours[0][0], self.contours[0][triplet_num], self.contours[0][triplet_num * 2], self.contours[0][-1]]]
            self.bounding_planes = [[self.bounding_planes[0][0], self.bounding_planes[0][triplet_num], self.bounding_planes[0][triplet_num * 2], self.bounding_planes[0][-1]]]
        else:
            fiber_nums = []
            for bounding_plane_stream in self.bounding_planes:
                max_area = 0
                min_area = np.inf
                for bounding_plane in bounding_plane_stream:
                    if bounding_plane['area'] > max_area:
                        max_area = bounding_plane['area']
                    if bounding_plane['area'] < min_area:
                        min_area = bounding_plane['area']

                # Choose area based on sampling method
                if self.sampling_method == 'sobol_min_contour':
                    # Use min_area to ensure samples fit in smallest contour
                    fiber_num = int(np.sqrt(min_area * 10 / (scale * scale)))
                else:
                    # Use max_area (original behavior)
                    fiber_num = int(np.sqrt(max_area * 10 / (scale * scale)))
                fiber_num = max(int(fiber_num), 1)
                fiber_nums.append(fiber_num)

            # Generate fiber architecture based on sampling method
            if self.sampling_method == 'sobol_min_contour':
                # Sample inside smallest contour of each stream
                self.fiber_architecture = [
                    self.sobol_sampling_min_contour(bounding_plane_stream, fiber_num)
                    for bounding_plane_stream, fiber_num in zip(self.bounding_planes, fiber_nums)
                ]
            else:
                # Original: Sobol sampling on unit square
                self.fiber_architecture = [self.sobol_sampling_barycentric(fiber_num) for fiber_num in fiber_nums]

        # for i, bounding_plane_info in enumerate(self.bounding_planes):
        #     if i == 0:
        #         is_origin = True
        #     else:
        #         is_origin = False
        #     self.find_waypoints(bounding_plane_info, is_origin=is_origin)

        self.normalized_Qs = [[] for _ in range(len(self.bounding_planes))]
        self.waypoints = [[] for _ in range(len(self.bounding_planes))]
        self.mvc_weights = [[] for _ in range(len(self.bounding_planes))]
        self.attach_skeletons = [[0, 0] for _ in range(len(self.waypoints))]
        self.attach_skeletons_sub = [[0, 0] for _ in range(len(self.waypoints))]

        for i, bounding_plane_stream in enumerate(self.bounding_planes):
            for bounding_plane in bounding_plane_stream:
                normalized_Qs, waypoints, mvc_weights = self.find_waypoints(bounding_plane, self.fiber_architecture[i])
                self.normalized_Qs[i].append(normalized_Qs)
                self.waypoints[i].append(waypoints)
                self.mvc_weights[i].append(mvc_weights)

        # for waypoint_group in self.waypoints:
        #     num_fibers = waypoint_group[0].shape[0]
        #     num_waypoints = len(waypoint_group)
        #     for i in range(num_fibers):
        #         for _ in range(1):
        #             for line_i in range(num_waypoints):
        #                 if line_i == 0 or line_i == num_waypoints - 1:
        #                     continue
                        
        #                 for fiber in range(num_fibers):
        #                     waypoint_group[line_i][fiber] = (((waypoint_group[line_i - 1][fiber] + waypoint_group[line_i + 1][fiber]) / 2) + waypoint_group[line_i][fiber] * 3) / 4

        self.is_draw = False
        self.is_draw_fiber_architecture = True
        
        # self.contours_discarded = ordered_contours_discarded
        # self.bounding_planes_discarded = bounding_planes_discarded

    def sobol_sampling_barycentric(self, num_samples):
        from scipy.stats.qmc import Sobol
        # Generate Sobol samples in unit square
        sobol = Sobol(d=2, scramble=True)
        samples = sobol.random(num_samples)

        return samples

    def sobol_sampling_min_contour(self, bounding_plane_stream, num_samples):
        """
        Sample 2D positions inside the smallest contour of a stream using Sobol + rejection.
        These positions are valid for all contours in the stream.

        Args:
            bounding_plane_stream: List of bounding plane dicts for one stream
            num_samples: Number of samples to generate

        Returns:
            np.array of shape (num_samples, 2) with positions in [0,1]^2
        """
        from scipy.stats.qmc import Sobol
        from shapely.geometry import Point, Polygon

        # Find the smallest contour (by area)
        min_area = np.inf
        smallest_plane = None
        for bp in bounding_plane_stream:
            if bp['area'] < min_area:
                min_area = bp['area']
                smallest_plane = bp

        # Get 2D projected contour
        projected_2d = smallest_plane['projected_2d']

        # Normalize to unit square [0,1]^2 for Sobol compatibility
        min_pt = np.min(projected_2d, axis=0)
        max_pt = np.max(projected_2d, axis=0)
        extent = max_pt - min_pt

        # Handle degenerate case
        if extent[0] < 1e-10 or extent[1] < 1e-10:
            return self.sobol_sampling_barycentric(num_samples)

        normalized_contour = (projected_2d - min_pt) / extent

        poly = Polygon(normalized_contour)
        if not poly.is_valid:
            poly = poly.buffer(0)  # Fix invalid polygons

        # Rejection sampling with Sobol
        sobol = Sobol(d=2, scramble=True)
        samples = []
        max_iterations = 100
        iteration = 0

        while len(samples) < num_samples and iteration < max_iterations:
            batch_size = (num_samples - len(samples)) * 4
            candidates = sobol.random(batch_size)
            for c in candidates:
                if poly.contains(Point(c[0], c[1])):
                    samples.append(c)
                    if len(samples) >= num_samples:
                        break
            iteration += 1

        # If we couldn't get enough samples, fill with centroid or fallback
        if len(samples) < num_samples:
            centroid = np.array(poly.centroid.coords[0])
            while len(samples) < num_samples:
                samples.append(centroid)

        return np.array(samples[:num_samples])

    def find_waypoints(self, bounding_plane_info, fiber_architecture, is_origin=False):
        bounding_vertices = bounding_plane_info['bounding_plane']
        v0, v1, v2, v3 = bounding_vertices
        y = bounding_plane_info['scalar_value']
        y = (y - 1) / (10 - 1)
        w0 = np.array([0, 0, y])
        w1 = np.array([1, 0, y])
        w2 = np.array([1, 1, y])
        w3 = np.array([0, 1, y])

        # make 3x4 matrix from v0, v1, v2, v3
        A = np.array([v0, v1, v2, v3]).T
        B = np.array([w0, w1, w2, w3]).T
        A_pinv = np.linalg.pinv(A)
        H = B @ A_pinv

        # convert all contour points and save as normalized_contour_points
        contour_match = bounding_plane_info['contour_match']
        Ps = np.array([pair[0] for pair in contour_match])
        Qs = np.array([pair[1] for pair in contour_match])
        # normalized_Ps = np.dot(H, np.array(Ps).T).T
        normalized_Qs = np.dot(H, np.array(Qs).T).T

        fs = []
        for v in fiber_architecture:
            f_found = False
            s_v = []
            for Q in normalized_Qs:
                s_v.append(Q[:2] - v)

            for i in range(len(normalized_Qs)):
                i_plus = ((i + 1) % len(normalized_Qs))
                r_i = np.linalg.norm(s_v[i])
                A_i = np.linalg.det(np.array([s_v[i], s_v[i_plus]])) / 2
                D_i = np.dot(s_v[i], s_v[i_plus])
                if r_i == 0:
                    f = np.zeros(len(normalized_Qs))
                    f[i] = 1
                    fs.append(f)
                    f_found = True
                    break
                if A_i == 0 and D_i < 0:
                    r_i_plus = np.linalg.norm(s_v[i_plus])
                    f_i = np.zeros(len(normalized_Qs))
                    f_i[i] = 1

                    f_i_plus = np.zeros(len(normalized_Qs))
                    f_i_plus[i_plus] = 1

                    fs.append((r_i_plus * f_i + r_i * f_i_plus) / (r_i + r_i_plus))
                    f_found = True
                    break
            
            if f_found:
                continue

            f = np.zeros(len(normalized_Qs))
            W = 0
            for i in range(len(normalized_Qs)):
                i_plus = ((i + 1) % len(normalized_Qs))
                i_minus = ((i - 1) % len(normalized_Qs))
                r_i = np.linalg.norm(s_v[i])
                w = 0

                A_i_minus = np.linalg.det(np.array([s_v[i_minus], s_v[i]])) / 2
                if A_i_minus != 0:
                    r_i_minus = np.linalg.norm(s_v[i_minus])
                    D_i_minus = np.dot(s_v[i_minus], s_v[i])
                    w += (r_i_minus - D_i_minus / r_i) / A_i_minus

                A_i = np.linalg.det(np.array([s_v[i], s_v[i_plus]])) / 2
                if A_i != 0:
                    r_i_plus = np.linalg.norm(s_v[i_plus])
                    D_i = np.dot(s_v[i], s_v[i_plus])
                    w += (r_i_plus - D_i / r_i) / A_i

                f_i = np.zeros(len(normalized_Qs))
                f_i[i] = w
                f += f_i
                W += w
            fs.append(f / W)
        
        fs = np.array(fs)
        # define 'waypoint' as sum product of fs and Ps
        # waypoints = np.dot(fs, normalized_Qs) # for validity check
        waypoints = np.dot(fs, Ps)
        # self.normalized_contours.append(np.concatenate([normalized_Ps, normalized_Qs]))

        # self.normalized_contours.append(normalized_Qs)
        # self.waypoints.append(waypoints)
        return normalized_Qs, waypoints, fs
    
    def find_overlap_point(self, corners1, corners2, grid_points=20):
        """
        Deterministically find a representative point that is inside both bounding boxes.
        
        Args:
            corners1: (8, 3) numpy array of the 8 corner points of the first bounding box.
            corners2: (8, 3) numpy array of the 8 corner points of the second bounding box.
            grid_points: Number of grid samples per dimension.
            
        Returns:
            A numpy array with shape (3,) representing the representative point inside both boxes,
            or None if no overlap is detected.
        """
        # Build Delaunay triangulations for point-in-polyhedron tests.
        from scipy.spatial import Delaunay
        hull1 = Delaunay(corners1)
        hull2 = Delaunay(corners2)
        
        # Determine axis-aligned bounds for each box.
        min1, max1 = corners1.min(axis=0), corners1.max(axis=0)
        min2, max2 = corners2.min(axis=0), corners2.max(axis=0)
        
        # The overlapping region of the axis-aligned bounding boxes:
        global_min = np.maximum(min1, min2)
        global_max = np.minimum(max1, max2)
        
        # If there's no overlap in at least one dimension, return None.
        if np.any(global_min > global_max):
            return None
        
        # Create a deterministic grid of candidate points inside the overlapping region.
        xs = np.linspace(global_min[0], global_max[0], grid_points)
        ys = np.linspace(global_min[1], global_max[1], grid_points)
        zs = np.linspace(global_min[2], global_max[2], grid_points)
        grid = np.meshgrid(xs, ys, zs, indexing='ij')
        candidate_points = np.vstack([grid[0].ravel(), grid[1].ravel(), grid[2].ravel()]).T
        
        # Check deterministically which candidate points are inside both bounding boxes.
        inside1 = hull1.find_simplex(candidate_points) >= 0
        inside2 = hull2.find_simplex(candidate_points) >= 0
        inside_points = candidate_points[inside1 & inside2]
        
        if inside_points.shape[0] == 0:
            return None  # No overlapping region detected.
        
        # Return the centroid of all candidate points inside both boxes.
        return inside_points.mean(axis=0)
    
    def find_bounding_box(self, axis=None, method='pca-cluster'):
        def compute_bbox(vertices_subset):
            mean = np.mean(vertices_subset, axis=0)
            centered = vertices_subset - mean

            try:
                if axis == 'xyz':
                    axes = np.eye(3)
                elif axis == 'x':
                    first = np.array([1, 0, 0])
                    pca = PCA(n_components=2).fit(centered[:, [1, 2]])
                    yz = pca.components_
                    second = np.array([0, yz[0, 0], yz[0, 1]])
                    third = np.array([0, yz[1, 0], yz[1, 1]])
                    axes = np.vstack([first, second, third])
                elif axis == 'y':
                    second = np.array([0, 1, 0])
                    pca = PCA(n_components=2).fit(centered[:, [0, 2]])
                    xz = pca.components_
                    first = np.array([xz[0, 0], 0, xz[0, 1]])
                    third = np.array([xz[1, 0], 0, xz[1, 1]])
                    axes = np.vstack([first, second, third])
                elif axis == 'z':
                    third = np.array([0, 0, 1])
                    pca = PCA(n_components=2).fit(centered[:, [0, 1]])
                    xy = pca.components_
                    first = np.array([xy[0, 0], xy[0, 1], 0])
                    second = np.array([xy[1, 0], xy[1, 1], 0])
                    axes = np.vstack([first, second, third])
                else:
                    pca = PCA(n_components=3).fit(centered)
                    axes = pca.components_
            except:
                axes = np.eye(3)

            # Normalize axes.
            axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
            axes[2] = np.cross(axes[0], axes[1])

            # --- New: If computed y and z axes are nearly in the yz plane, fix x axis.
            # Check if both axes[1] and axes[2] have almost zero x-component.
            threshold = 1e-3
            if np.abs(axes[1, 0]) < threshold and np.abs(axes[2, 0]) < threshold:
                # Fix first axis to [1, 0, 0]
                xaxis = np.array([1, 0, 0])
                yaxis = np.array([0, axes[1, 1], axes[1, 2]])
                yaxis = yaxis / np.linalg.norm(yaxis)
                zaxis = np.cross(xaxis, yaxis)

                axes = np.vstack([xaxis, yaxis, zaxis])
            # --- End of new section.

            proj = centered @ axes.T
            min_proj = proj.min(axis=0)
            max_proj = proj.max(axis=0)
            sizes = max_proj - min_proj

            corners = np.array(list(product(
                [min_proj[0], max_proj[0]],
                [min_proj[1], max_proj[1]],
                [min_proj[2], max_proj[2]],
            )))
            return corners @ axes + mean, axes, sizes

        bounding_box_complete = False
        self.corners_list = []
        self.bbox_axes_list = []

        target_vertices = self.vertices
        while not bounding_box_complete:
            if method == "agglo":
                clustering = AgglomerativeClustering(n_clusters=self.num_boxes).fit(target_vertices)
                labels = clustering.labels_

            elif method == "thickness-aware":
                centered = target_vertices - np.mean(target_vertices, axis=0)
                pca = PCA(n_components=3).fit(centered)
                main_axis = pca.components_[0]
                ortho_axes = pca.components_[1:]

                main_proj = centered @ main_axis
                ortho_proj = centered @ ortho_axes.T
                thickness = np.linalg.norm(ortho_proj, axis=1)
                features = np.stack([main_proj, thickness], axis=1)

                clustering = KMeans(n_clusters=self.num_boxes, n_init=10).fit(features)
                labels = clustering.labels_

            elif method == "pca-cluster":
                pca = PCA(n_components=3).fit(target_vertices)
                proj = target_vertices @ pca.components_.T
                clustering = KMeans(n_clusters=self.num_boxes, n_init=10).fit(proj)
                labels = clustering.labels_

            else:
                raise ValueError(f"Unknown method: {method}")


            self.sizes =[]
            # For clustering-based methods, expand clusters to face-connected vertices.
            if method in ("original", "thickness-aware", "pca-cluster"):
                vertex_to_faces = defaultdict(set)
                for face_idx, face in enumerate(self.faces_3):
                    for v_idx in face[:, 0]:
                        vertex_to_faces[v_idx].add(face_idx)

                visited_faces = set()
                vertex_indices_list = []
                for label in range(self.num_boxes):
                    cluster_vertex_indices = np.where(labels == label)[0]
                    face_indices = set()
                    for v in cluster_vertex_indices:
                        face_indices.update(vertex_to_faces[v])
                    face_indices -= visited_faces
                    visited_faces.update(face_indices)

                    if not face_indices:
                        continue

                    face_subset = self.faces_3[list(face_indices)]
                    vertex_indices = list(np.unique(face_subset[:, :, 0].flatten()))
                    vertex_indices_list.append(vertex_indices)
                    cluster_vertices = self.vertices[vertex_indices]

                    corners, axes, sizes = compute_bbox(cluster_vertices)
                    self.corners_list.append(corners)
                    self.bbox_axes_list.append(axes)
                    self.sizes.append(sizes)

                bounding_box_complete = True

        self.corners = self.corners_list
        if len(self.corners) > 1:
            means = np.array([np.mean(corners, axis=0) for corners in self.corners])
            # order self.corners with means[:, 1] descending
            order = np.argsort(means[:, 1])[::-1]
            self.corners = [self.corners[i] for i in order]
            self.bbox_axes_list = [self.bbox_axes_list[i] for i in order]
            self.sizes = [self.sizes[i] for i in order]

        self.body_rs = []
        self.body_ts = []
        for i in range(len(self.corners)):
            self.body_rs.append(self.bbox_axes_list[i].T)
            self.body_ts.append(np.mean(self.corners[i], axis=0))

        self.weld_joints = []
        for i in range(len(self.corners_list)):
            for j in range(i + 1, len(self.corners_list)):
                weld_joint = self.find_overlap_point(self.corners_list[i], self.corners_list[j])
                if weld_joint is not None:
                    self.weld_joints.append(weld_joint)

        # Use only as many colors as there are bounding boxes.
        num_boxes = len(self.corners_list)
        if num_boxes == 1:
            bbox_colors = np.array([[1.0, 1.0, 1.0]])
        elif num_boxes == 2:
            bbox_colors = np.array([[1.0, 0.5, 0.0],    # Orange
                                    [0.0, 0.5, 1.0]])   # Light Blue
        elif num_boxes == 3:
            bbox_colors = np.array([[1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 0.0, 1.0]])
        elif num_boxes == 4:
            bbox_colors = np.array([[0.5, 0.0, 0.0],
                                    [0.0, 0.5, 0.0],
                                    [0.0, 0.0, 0.5],
                                    [0.5, 0.5, 0.5]])
        elif num_boxes == 5:
            bbox_colors = np.array([[0.4, 0.0, 0.0],
                                    [0.0, 0.4, 0.0],
                                    [0.0, 0.0, 0.4],
                                    [0.4, 0.3, 0.2],
                                    [0.2, 0.3, 0.4]])
        elif num_boxes == 6:
            bbox_colors = np.array([[0.4, 0.3, 0.0],
                                    [0.0, 0.4, 0.3],
                                    [0.3, 0.0, 0.4],
                                    [0.3, 0.3, 0.0],
                                    [0.0, 0.3, 0.3],
                                    [0.3, 0.0, 0.3]])
        else:
            bbox_colors = np.random.rand(num_boxes, 4)
            bbox_colors[:, 3] = 1.0

        # Initialize vertex colors with 4 channels (RGB + transparency).
        vertex_colors = np.zeros((len(self.vertices), 4), dtype=float)
        for idx, (corners, axes) in enumerate(zip(self.corners_list, self.bbox_axes_list)):
            center = np.mean(corners, axis=0)
            local_box_coords = (corners - center) @ axes.T
            lower_bounds = local_box_coords.min(axis=0)
            upper_bounds = local_box_coords.max(axis=0)
            local_vertices = (self.vertices - center) @ axes.T
            inside = np.all((local_vertices >= lower_bounds) & (local_vertices <= upper_bounds), axis=1)
            # Add the bounding box's fixed color to the first three channels.
            vertex_colors[inside, :3] += bbox_colors[idx]

        # Set the transparency (4th channel) for all vertices.
        vertex_colors[:, 3] = self.transparency
        self.vertex_colors = vertex_colors[self.faces_3[:, :, 0].flatten()]