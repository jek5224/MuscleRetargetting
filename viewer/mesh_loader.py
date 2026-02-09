# Mesh Loader for the viewer
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import trimesh

# Import mixins
from viewer.contour_mesh import ContourMeshMixin
from viewer.tetrahedron_mesh import TetrahedronMeshMixin
from viewer.fiber_architecture import FiberArchitectureMixin
from viewer.muscle_mesh import MuscleMeshMixin
from viewer.skeleton_mesh import SkeletonMeshMixin

scale = 0.01


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
        self.is_draw_contour_vertices = False
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
        self.cutting_method = 'bp'  # 'bp', 'area_based', 'voronoi', 'angular', or 'gradient'

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
    
    def _draw_mesh_arrays(self, use_color_array):
        """Internal method to draw mesh geometry arrays."""
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

        # Check if transparent (use vertex color alpha or color parameter)
        is_transparent = self.transparency < 1.0 if hasattr(self, 'transparency') else color[3] < 1.0
        # Check if two-pass culling should be used (some meshes have bad winding)
        use_two_pass = getattr(self, 'use_two_pass_culling', True)

        if is_transparent and use_two_pass:
            # Two-pass rendering for correct transparency
            # Draw back faces first, then front faces on top
            glEnable(GL_CULL_FACE)
            glCullFace(GL_FRONT)  # Cull front, draw back
            self._draw_mesh_arrays(use_color_array)
            glCullFace(GL_BACK)   # Cull back, draw front
            self._draw_mesh_arrays(use_color_array)
            glDisable(GL_CULL_FACE)
        else:
            # Single pass for opaque meshes or meshes with bad winding
            self._draw_mesh_arrays(use_color_array)

        # Disable color array if it was enabled
        if use_color_array:
            glDisableClientState(GL_COLOR_ARRAY)

        glDisableClientState(GL_VERTEX_ARRAY)
        glPopMatrix()

    def draw_simple(self, color=np.array([0.5, 0.5, 0.5, 1.0])):
        """Original draw method without transparency logic - for OBJ meshes."""
        glPushMatrix()

        # Determine if color array should be enabled for scalar field rendering
        use_color_array = self.vertex_colors is not None and self.is_draw_scalar_field

        if use_color_array:
            glEnableClientState(GL_COLOR_ARRAY)
        else:
            glColor4f(color[0], color[1], color[2], color[3])

        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self._draw_mesh_arrays(use_color_array)

        if use_color_array:
            glDisableClientState(GL_COLOR_ARRAY)

        glDisableClientState(GL_VERTEX_ARRAY)
        glPopMatrix()

    def draw_edges(self):
        if len(self.edges) == 0:
            return

        # Build edge vertex array if not cached or vertices changed
        if not hasattr(self, '_edge_verts_cache') or self._edge_verts_cache is None:
            edge_verts = []
            for edge in self.edges:
                v1, v2 = edge
                edge_verts.append(self.vertices[v1])
                edge_verts.append(self.vertices[v2])
            self._edge_verts_cache = np.array(edge_verts, dtype=np.float32)

        glPushMatrix()
        glDisable(GL_LIGHTING)
        glColor4f(0, 0, 0, 1.0)
        glLineWidth(0.5)

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self._edge_verts_cache)
        glDrawArrays(GL_LINES, 0, len(self._edge_verts_cache))
        glDisableClientState(GL_VERTEX_ARRAY)

        glEnable(GL_LIGHTING)
        glPopMatrix()

    def draw_open_edges(self, color=np.array([0.0, 0.0, 1.0, 1.0])):
        glDisable(GL_LIGHTING)
        glPushMatrix()

        glEnableClientState(GL_VERTEX_ARRAY)

        # Draw points for each edge group
        glPointSize(2)
        for i, group in enumerate(self.edge_groups):
            if len(group) == 0:
                continue
            if self.edge_classes[i] == 'origin':
                glColor3f(1, 0, 0)
            else:
                glColor3f(0, 0, 1)

            # Build point array
            points = np.array([self.vertices[idx] for idx in group], dtype=np.float32)
            glVertexPointer(3, GL_FLOAT, 0, points)
            glDrawArrays(GL_POINTS, 0, len(points))

        # Draw lines for each open edge group
        glLineWidth(1)
        for i, edges in enumerate(self.open_edges):
            if len(edges) == 0:
                continue
            if self.edge_classes[i] == 'origin':
                glColor3f(1, 0, 0)
            else:
                glColor3f(0, 0, 1)

            # Build line vertex array
            line_verts = []
            for edge in edges:
                v1, v2 = edge
                line_verts.append(self.vertices[v1])
                line_verts.append(self.vertices[v2])
            line_verts = np.array(line_verts, dtype=np.float32)
            glVertexPointer(3, GL_FLOAT, 0, line_verts)
            glDrawArrays(GL_LINES, 0, len(line_verts))

        glDisableClientState(GL_VERTEX_ARRAY)

        glPopMatrix()
        glEnable(GL_LIGHTING)

    def draw_centroid(self):
        if len(self.centroids) == 0:
            return

        glDisable(GL_LIGHTING)
        glPushMatrix()

        glEnableClientState(GL_VERTEX_ARRAY)

        # Draw centroid points
        glPointSize(5)
        glColor3f(1, 0, 0)
        centroids_arr = np.array(self.centroids, dtype=np.float32)
        glVertexPointer(3, GL_FLOAT, 0, centroids_arr)
        glDrawArrays(GL_POINTS, 0, len(centroids_arr))

        # Draw frame axes
        if len(self.frames) == 2:
            glLineWidth(2)
            for i, axes in enumerate(self.frames):
                if i == 0:
                    glColor3f(0, 1, 0)
                else:
                    glColor3f(0, 0, 1)

                # Build axis lines
                axis_verts = np.array([
                    self.centroids[0], self.centroids[0] + axes[1],
                    self.centroids[0], self.centroids[0] + axes[2],
                    self.centroids[0], self.centroids[0] + axes[3],
                ], dtype=np.float32)
                glVertexPointer(3, GL_FLOAT, 0, axis_verts)
                glDrawArrays(GL_LINES, 0, len(axis_verts))

        glDisableClientState(GL_VERTEX_ARRAY)

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

        # BP scale animation dict (used during contour reveal animation)
        bp_scale_dict = getattr(self, '_contour_anim_bp_scale', None)

        glLineWidth(0.5)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        for i, bounding_planes in enumerate(self.bounding_planes):
            for j, plane_info in enumerate(bounding_planes):
                # Check visibility based on draw_contour_stream structure
                if self.draw_contour_stream is not None:
                    if isinstance(self.draw_contour_stream[0], (list, tuple)):
                        # 2D structure: draw_contour_stream[stream_idx][level_idx]
                        # After build_fibers: bounding_planes[stream_i][level_i]
                        if i < len(self.draw_contour_stream) and j < len(self.draw_contour_stream[i]):
                            if not self.draw_contour_stream[i][j]:
                                continue
                    else:
                        # 1D: draw_contour_stream[stream_idx]
                        if not self.draw_contour_stream[i]:
                            continue
                glPushMatrix()

                # BP scale animation: _contour_anim_bp_scale keys are level indices
                # After find_contours (pre-cut): i=level, j=contour within level
                # After cut_streams (post-cut): i=stream, j=level (draw_contour_stream is 2D)
                bp_level = j if (self.draw_contour_stream is not None and
                                 len(self.draw_contour_stream) > 0 and
                                 isinstance(self.draw_contour_stream[0], list)) else i
                bp_s = bp_scale_dict.get(bp_level, 1.0) if bp_scale_dict else 1.0
                bp_alpha = bp_s  # Fade bounding plane with scale
                mean = plane_info['mean']

                glPushMatrix()
                glPointSize(max(5 * bp_s, 0.1))
                glColor4f(0, 0, 0, bp_alpha)
                glBegin(GL_POINTS)
                glVertex3fv(mean)
                glEnd()
                glPopMatrix()

                glColor3f(1, 0, 0)
                glBegin(GL_LINES)
                glVertex3fv(mean)
                glVertex3fv(mean + plane_info['basis_x'] * scale * 0.1 * bp_s)
                glEnd()

                glColor3f(0, 1, 0)
                glBegin(GL_LINES)
                glVertex3fv(mean)
                glVertex3fv(mean + plane_info['basis_y'] * scale * 0.1 * bp_s)
                glEnd()

                glColor3f(0, 0, 1)
                glBegin(GL_LINES)
                glVertex3fv(mean)
                glVertex3fv(mean + plane_info['basis_z'] * scale * 0.1 * bp_s)
                glEnd()

                if plane_info.get('bounding_plane') is not None:
                    if plane_info['square_like']:
                        glColor4f(1, 0, 0, bp_alpha)
                    else:
                        glColor4f(0, 0, 0, bp_alpha)
                    glBegin(GL_LINE_LOOP)
                    for point in plane_info['bounding_plane']:
                        glVertex3fv(mean + (point - mean) * bp_s)
                    glEnd()

                # glBegin(GL_LINE_LOOP)
                # for point_2d in plane_info['projected_2d']:
                #     glVertex3fv(point_2d)
                # glEnd()
                glPopMatrix()
        glEnable(GL_LIGHTING)

        # Only draw connecting lines after find_contour_stream (streams have been built)
        # Use _stream_endpoints as indicator that streams are built, not draw_contour_stream
        streams_built = (hasattr(self, '_stream_endpoints') and
                         self._stream_endpoints is not None and
                         len(self._stream_endpoints) > 0)
        if streams_built:
            is_2d = (self.draw_contour_stream is not None and
                     len(self.draw_contour_stream) > 0 and
                     isinstance(self.draw_contour_stream[0], list))
            glColor3f(0, 0, 0)
            glDisable(GL_LIGHTING)
            for i, bounding_plane_stream in enumerate(self.bounding_planes):
                if self.draw_contour_stream is not None:
                    if is_2d:
                        # 2D: skip stream if all levels are hidden
                        if i < len(self.draw_contour_stream) and not any(self.draw_contour_stream[i]):
                            continue
                    else:
                        if not self.draw_contour_stream[i]:
                            continue
                for j in range(len(bounding_plane_stream) - 1):
                    bp1 = bounding_plane_stream[j]
                    bp2 = bounding_plane_stream[j + 1]
                    # Skip if either level is hidden in 2D mode
                    if is_2d and i < len(self.draw_contour_stream):
                        dcs = self.draw_contour_stream[i]
                        if (j < len(dcs) and not dcs[j]) or (j + 1 < len(dcs) and not dcs[j + 1]):
                            continue
                    if bp1.get('bounding_plane') is None or bp2.get('bounding_plane') is None:
                        continue
                    # Apply BP scale animation if active
                    s1 = bp_scale_dict.get(j, 1.0) if bp_scale_dict else 1.0
                    s2 = bp_scale_dict.get(j + 1, 1.0) if bp_scale_dict else 1.0
                    line_alpha = min(s1, s2)
                    glColor4f(0, 0, 0, line_alpha)
                    glBegin(GL_LINES)
                    for p1, p2 in zip(bp1['bounding_plane'], bp2['bounding_plane']):
                        glVertex3fv(bp1['mean'] + (p1 - bp1['mean']) * s1)
                        glVertex3fv(bp2['mean'] + (p2 - bp2['mean']) * s2)
                    glVertex3fv(bp1['mean'])
                    glVertex3fv(bp2['mean'])
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

