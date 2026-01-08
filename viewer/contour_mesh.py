# Contour Mesh operations for muscle mesh processing
# Geometric utilities and contour visualization
# Scalar field and bounding plane methods moved to muscle_mesh.py

import numpy as np
from OpenGL.GL import *
import matplotlib.cm as cm
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from collections import defaultdict

# Import triangulation functions for fiber mapping
from viewer.fiber_architecture import (
    create_angular_unit_circle_vertices,
    find_geodesic_vertex_indices,
    create_unit_circle_triangulation,
    embed_fibers_in_triangulation,
    find_waypoints_triangulated,
    create_direct_fiber_triangulation,
    find_waypoints_harmonic_direct,
    find_waypoints_geodesic,
    GeodesicTriangulation,
    create_shared_geodesic_triangulation,
    find_waypoints_geodesic_shared
)

# Constants for contour finding
MESH_SCALE = 0.01
scale = MESH_SCALE
CONTOUR_VALUE_MIN_DEFAULT = 1.1
CONTOUR_VALUE_MAX_DEFAULT = 9.9
COLOR_MAP = cm.get_cmap("turbo")


def align_basis_to_reference_continuous(basis_x, basis_y, ref_basis_x, basis_z):
    """
    Align basis vectors to match reference orientation using continuous rotation.

    Instead of discrete 90-degree choices, computes the optimal rotation angle
    to align basis_x with ref_basis_x while keeping basis_z fixed.

    Args:
        basis_x, basis_y: current in-plane basis vectors
        ref_basis_x: reference x-axis to align to
        basis_z: plane normal (kept fixed)

    Returns:
        (new_basis_x, new_basis_y) - aligned basis vectors
    """
    # Project reference basis_x onto the current plane (perpendicular to basis_z)
    proj_ref_x = ref_basis_x - np.dot(ref_basis_x, basis_z) * basis_z
    proj_norm = np.linalg.norm(proj_ref_x)

    if proj_norm < 1e-6:
        # Reference is parallel to plane normal, keep current basis
        return basis_x.copy(), basis_y.copy()

    proj_ref_x = proj_ref_x / proj_norm

    # Compute the optimal rotation angle using atan2 for robustness
    cos_angle = np.clip(np.dot(basis_x, proj_ref_x), -1.0, 1.0)
    sin_angle = np.dot(np.cross(basis_x, proj_ref_x), basis_z)
    angle = np.arctan2(sin_angle, cos_angle)

    # Rotate basis vectors by this angle around basis_z
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    new_basis_x = cos_a * basis_x + sin_a * basis_y
    new_basis_y = -sin_a * basis_x + cos_a * basis_y

    # Normalize to ensure unit vectors
    new_basis_x = new_basis_x / np.linalg.norm(new_basis_x)
    new_basis_y = new_basis_y / np.linalg.norm(new_basis_y)

    return new_basis_x, new_basis_y


# ============================================================================
# GEOMETRIC UTILITY FUNCTIONS
# ============================================================================

def compute_newell_normal(vertices):
    """Computes the normal vector of a closed loop using Newell's method."""
    normal = np.zeros(3)
    n = len(vertices)

    for i in range(n):
        v_curr = np.array(vertices[i])
        v_next = np.array(vertices[(i + 1) % n])

        normal[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
        normal[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
        normal[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])

    return normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0.0 else np.array([0, 0, 0])


def compute_best_fitting_plane(vertices):
    """Finds an optimal coordinate system using PCA."""
    mean = np.mean(vertices, axis=0)
    centered = vertices - mean
    _, _, eigenvectors = np.linalg.svd(centered)

    basis_x = eigenvectors[0]
    basis_y = eigenvectors[1]

    basis_x /= np.linalg.norm(basis_x)
    basis_y /= np.linalg.norm(basis_y)

    return basis_x, basis_y, mean


def project_vertices_to_global(vertices, basis_x, basis_y, mean):
    """Projects 3D vertices onto a plane and returns their new global positions."""
    projected = []
    for v in vertices:
        relative_v = np.array(v) - mean
        x_proj = np.dot(relative_v, basis_x)
        y_proj = np.dot(relative_v, basis_y)
        projected_3d = mean + x_proj * basis_x + y_proj * basis_y
        projected.append(projected_3d.tolist())

    return np.array(projected)


def compute_polygon_area(vertices_2d):
    """Computes the area of a 2D projected polygon using the Shoelace Theorem."""
    x = vertices_2d[:, 0]
    y = vertices_2d[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def compute_minimum_area_bbox(points_2d):
    """
    Compute minimum area oriented bounding box using rotating calipers algorithm.

    The optimal rectangle has one edge aligned with an edge of the convex hull.
    This guarantees 4 right angles (90 degrees) in the bounding box.

    Args:
        points_2d: (N, 2) array of 2D points

    Returns:
        dict with:
            'corners': (4, 2) array of corner points [v0, v1, v2, v3]
            'basis_x': unit vector along first edge (width direction)
            'basis_y': unit vector perpendicular to first edge (height direction)
            'center': center of the bounding box
            'width': width of the box (along basis_x)
            'height': height of the box (along basis_y)
            'area': area of the bounding box
    """
    from scipy.spatial import ConvexHull

    points_2d = np.array(points_2d)
    if len(points_2d) < 3:
        # Fallback for degenerate cases
        min_pt = np.min(points_2d, axis=0)
        max_pt = np.max(points_2d, axis=0)
        center = (min_pt + max_pt) / 2
        width = max_pt[0] - min_pt[0]
        height = max_pt[1] - min_pt[1]
        return {
            'corners': np.array([
                [min_pt[0], min_pt[1]],
                [max_pt[0], min_pt[1]],
                [max_pt[0], max_pt[1]],
                [min_pt[0], max_pt[1]]
            ]),
            'basis_x': np.array([1.0, 0.0]),
            'basis_y': np.array([0.0, 1.0]),
            'center': center,
            'width': max(width, 1e-10),
            'height': max(height, 1e-10),
            'area': width * height
        }

    try:
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]
    except:
        # Fallback if convex hull fails (collinear points)
        min_pt = np.min(points_2d, axis=0)
        max_pt = np.max(points_2d, axis=0)
        center = (min_pt + max_pt) / 2
        width = max_pt[0] - min_pt[0]
        height = max_pt[1] - min_pt[1]
        return {
            'corners': np.array([
                [min_pt[0], min_pt[1]],
                [max_pt[0], min_pt[1]],
                [max_pt[0], max_pt[1]],
                [min_pt[0], max_pt[1]]
            ]),
            'basis_x': np.array([1.0, 0.0]),
            'basis_y': np.array([0.0, 1.0]),
            'center': center,
            'width': max(width, 1e-10),
            'height': max(height, 1e-10),
            'area': width * height
        }

    n_hull = len(hull_points)
    min_area = np.inf
    best_result = None

    for i in range(n_hull):
        # Edge vector from hull point i to i+1
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % n_hull]
        edge = p2 - p1
        edge_len = np.linalg.norm(edge)

        if edge_len < 1e-10:
            continue

        # Unit vectors: basis_x along edge, basis_y perpendicular
        basis_x = edge / edge_len
        basis_y = np.array([-basis_x[1], basis_x[0]])  # 90 degree rotation

        # Project all points onto these axes
        proj_x = points_2d @ basis_x
        proj_y = points_2d @ basis_y

        # Bounding box extents in this orientation
        min_x, max_x = proj_x.min(), proj_x.max()
        min_y, max_y = proj_y.min(), proj_y.max()

        width = max_x - min_x
        height = max_y - min_y
        area = width * height

        if area < min_area:
            min_area = area

            # Compute corner points in original 2D space
            # v0 = bottom-left, v1 = bottom-right, v2 = top-right, v3 = top-left
            corners = np.array([
                min_x * basis_x + min_y * basis_y,
                max_x * basis_x + min_y * basis_y,
                max_x * basis_x + max_y * basis_y,
                min_x * basis_x + max_y * basis_y
            ])

            center = (corners[0] + corners[2]) / 2

            best_result = {
                'corners': corners,
                'basis_x': basis_x,
                'basis_y': basis_y,
                'center': center,
                'width': width,
                'height': height,
                'area': area
            }

    if best_result is None:
        # Fallback
        min_pt = np.min(points_2d, axis=0)
        max_pt = np.max(points_2d, axis=0)
        center = (min_pt + max_pt) / 2
        width = max_pt[0] - min_pt[0]
        height = max_pt[1] - min_pt[1]
        best_result = {
            'corners': np.array([
                [min_pt[0], min_pt[1]],
                [max_pt[0], min_pt[1]],
                [max_pt[0], max_pt[1]],
                [min_pt[0], max_pt[1]]
            ]),
            'basis_x': np.array([1.0, 0.0]),
            'basis_y': np.array([0.0, 1.0]),
            'center': center,
            'width': max(width, 1e-10),
            'height': max(height, 1e-10),
            'area': width * height
        }

    return best_result


# ============================================================================
# CONTOUR MESH MIXIN
# ============================================================================

class ContourMeshMixin:
    """
    Mixin class providing contour-related methods for MeshLoader.
    Handles contour visualization and property initialization.

    Note: compute_scalar_field and save_bounding_planes moved to MuscleMeshMixin.
    """

    def _init_contour_properties(self):
        """Initialize contour-related properties. Call from MeshLoader.__init__."""
        self.scalar_field = None
        self.is_draw_scalar_field = False
        self.contours = None
        self.bounding_planes = None
        self.specific_contour = None
        self.specific_contour_value = 1.0
        self.contour_value_min = CONTOUR_VALUE_MIN_DEFAULT
        self.contour_value_max = CONTOUR_VALUE_MAX_DEFAULT
        self.normalized_contours = []
        self.structure_vectors = []
        self.contours_discarded = None
        self.bounding_planes_discarded = None
        self.is_draw_contours = False
        self.is_draw_discarded = False
        self.draw_contour_stream = None

        # Inspector highlight (set by viewer when 2D inspector is open)
        self.inspector_highlight_stream = None  # Stream index to highlight
        self.inspector_highlight_level = None   # Level index to highlight

        # Contour mesh properties
        self.contour_mesh_vertices = None
        self.contour_mesh_faces = None
        self.contour_mesh_normals = None
        self.is_draw_contour_mesh = False

    def draw_contours(self):
        """Draw contour lines, optionally using deformed positions from tet mesh."""
        # Determine which contours to draw (original or deformed)
        use_deformed = getattr(self, 'draw_deformed_contours', False)

        if use_deformed and hasattr(self, 'get_deformed_contours'):
            contours_to_draw = self.get_deformed_contours()
        else:
            contours_to_draw = self.contours

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

        if contours_to_draw is None:
            return

        glDisable(GL_LIGHTING)

        glLineWidth(0.5)
        for structure_vector in self.structure_vectors:
            glBegin(GL_LINES)
            glColor4f(0, 0, 0, 0.5)
            glVertex3fv(structure_vector[0])
            glVertex3fv(structure_vector[1])
            glEnd()

        values = np.linspace(0, 1, len(contours_to_draw) + 2)[1:-1]

        for i, contour_set in enumerate(contours_to_draw):
            if self.draw_contour_stream is None:
                break

            if i >= len(self.draw_contour_stream):
                continue

            if not self.draw_contour_stream[i]:
                continue

            if len(self.draw_contour_stream) > 8:
                color = COLOR_MAP(1 - values[i])[:3]
            else:
                colors = [
                    np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                    np.array([1, 1, 0]), np.array([0, 1, 1]), np.array([1, 0, 1]),
                    np.array([1, 0.5, 0]), np.array([0.5, 1, 0]), np.array([0, 0, 0])
                ]
                color = colors[i] if i < len(colors) else COLOR_MAP(1 - values[i])[:3]

            for j, contour in enumerate(contour_set):
                # Check if this contour should be highlighted (inspector is viewing it)
                highlight_stream = getattr(self, 'inspector_highlight_stream', None)
                highlight_level = getattr(self, 'inspector_highlight_level', None)
                is_highlighted = (highlight_stream == i and highlight_level == j)

                # Draw transparent fill for highlighted contour
                if is_highlighted and len(contour) >= 3:
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                    glColor4f(0.3, 0.6, 0.9, 0.25)  # Subtle blue with transparency
                    # Use triangle fan from centroid for non-convex support
                    centroid = np.mean(contour, axis=0)
                    glBegin(GL_TRIANGLE_FAN)
                    glVertex3fv(centroid)
                    for v in contour:
                        glVertex3fv(v)
                    glVertex3fv(contour[0])  # Close the fan
                    glEnd()
                    glDisable(GL_BLEND)

                if is_highlighted:
                    glLineWidth(2.0)
                    glColor3f(0.3, 0.6, 0.9)  # Subtle blue for highlighted
                else:
                    glLineWidth(1.0)
                    glColor3fv(color)

                glBegin(GL_LINE_LOOP)
                for v in contour:
                    glVertex3fv(v)
                glEnd()

                # Draw vertices with color gradient: red (first) -> black (last)
                glPointSize(6 if is_highlighted else 5)
                glBegin(GL_POINTS)
                n_verts = len(contour)
                for k, v in enumerate(contour):
                    if is_highlighted:
                        # Blue tint for highlighted, green for first vertex
                        if k == 0:
                            glColor3f(0.2, 0.8, 0.4)  # Muted green
                        else:
                            glColor3f(0.3, 0.6, 0.9)  # Subtle blue
                    else:
                        # Interpolate from red (1,0,0) to black (0,0,0)
                        t = k / max(n_verts - 1, 1)
                        vert_color = (1 - t, 0, 0)  # red -> black
                        glColor3f(*vert_color)
                    glVertex3fv(v)
                glEnd()

        if self.is_draw_discarded and self.contours_discarded is not None:
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

    def find_reference_intersection(self, contour_value):
        """
        Find where iso-contour at contour_value crosses the geodesic reference line.

        Args:
            contour_value: The scalar value of the iso-contour

        Returns:
            3D position where the contour crosses the reference line, or None if not found.
        """
        if not hasattr(self, '_geodesic_reference_path') or self._geodesic_reference_path is None:
            return None

        path = self._geodesic_reference_path
        scalar = self.scalar_field

        # Check each edge in the path for crossing
        for i in range(len(path) - 1):
            v0, v1 = path[i], path[i + 1]
            s0, s1 = scalar[v0], scalar[v1]

            # Check if contour_value is between s0 and s1
            if (s0 <= contour_value <= s1) or (s1 <= contour_value <= s0):
                # Avoid division by zero
                if abs(s1 - s0) < 1e-10:
                    t = 0.5
                else:
                    t = (contour_value - s0) / (s1 - s0)
                # Clamp t to [0, 1]
                t = max(0.0, min(1.0, t))
                intersection = (1 - t) * self.vertices[v0] + t * self.vertices[v1]
                return intersection

        return None

    def find_contour(self, contour_value, prev_bounding_plane=None, use_geodesic_edges=False):
        # Find contour vertices for contour value
        contour_vertices = []
        contour_edges = []
        vertex_tree = None

        # Build geodesic edge set if option enabled
        geodesic_edge_set = None
        if use_geodesic_edges and hasattr(self, '_geodesic_reference_paths') and self._geodesic_reference_paths:
            geodesic_edge_set = set()
            for path_info in self._geodesic_reference_paths:
                chain = path_info['chain']
                for i in range(len(chain) - 1):
                    edge = tuple(sorted([chain[i], chain[i + 1]]))
                    geodesic_edge_set.add(edge)

        for face in self.faces_3:
            face_cand_vertices = []
            v0, v1, v2 = int(face[0, 0]), int(face[1, 0]), int(face[2, 0])
            face_vert_indices = [v0, v1, v2]
            verts = self.vertices[[v0, v1, v2]]
            values = self.scalar_field[[v0, v1, v2]]

            edges = [(0, 1), (1, 2), (2, 0)]

            for i, j in edges:
                # If using geodesic edges only, skip edges not in geodesic lines
                if geodesic_edge_set is not None:
                    mesh_edge = tuple(sorted([face_vert_indices[i], face_vert_indices[j]]))
                    if mesh_edge not in geodesic_edge_set:
                        continue

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
            
            max_iterations = len(contour_edge_groups[i]) * 2  # Safety limit
            iteration = 0
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

                iteration += 1
                if not connection_found or iteration >= max_iterations:
                    # No connection found or max iterations reached - break to avoid infinite loop
                    break

            ordered_contour_edge_groups.append(ordered_edge_group)

        ordered_contour_vertices = []
        ordered_contour_vertices_orig = []
        bounding_planes = []
        for edge_group in ordered_contour_edge_groups:
            ordered_contour_vertices.append(np.array(contour_vertices)[edge_group])
            ordered_contour_vertices_orig.append(np.array(contour_vertices)[edge_group])
            ordered_contour_vertices[-1], bounding_plane = self.save_bounding_planes(ordered_contour_vertices[-1], contour_value, prev_bounding_plane=prev_bounding_plane)
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

        # Spacing thresholds (defined here so they're available for refinement)
        length_min_threshold = scale * 0.5
        length_max_threshold = scale * 1.0

        if False:
            contour_values = np.arange(1 + scalar_step, 10, scalar_step)
            for contour_value in contour_values:
                bounding_planes, ordered_contour_vertices, ordered_contour_vertices_orig = self.find_contour(contour_value, use_geodesic_edges=use_geodesic_edges)
                self.bounding_planes.append(bounding_planes)
                contours.append(ordered_contour_vertices)
                contours_orig.append(ordered_contour_vertices_orig)

        else:
            target_distance = (length_min_threshold + length_max_threshold) / 2  # Target distance for adaptive stepping
            prev_contour_value = 1.0
            contour_value = 1.0 + scalar_step

            # Safeguards to prevent infinite loops
            min_scalar_step = scalar_step * 0.001  # Minimum step to avoid infinite convergence
            max_scalar_step = scalar_step * 10.0   # Maximum step to avoid skipping too much
            max_iterations = 500
            max_binary_depth = 5

            # Adaptive stepping: track distance/scalar ratio to predict better steps
            current_adaptive_step = scalar_step
            distance_scalar_ratio = None  # Will be estimated from observations

            trial = 0
            binary_search_depth = 0
            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                bounding_planes, ordered_contour_vertices, ordered_contour_vertices_orig = self.find_contour(contour_value, use_geodesic_edges=use_geodesic_edges)
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

                # Calculate current scalar step and average distance
                current_scalar_step = contour_value - prev_contour_value
                avg_distance = np.mean(min_lengths) if len(min_lengths) > 0 else 0

                print(f"[{iteration}] num contours: {len(bounding_planes)}")
                print(f"min: {min_lengths} threshold: {length_min_threshold} {length_max_threshold}")
                print(f"contour_value: {contour_value:.4f}, scalar_step: {current_scalar_step:.6f}, binary_depth: {binary_search_depth}")
                print(f"adaptive_step: {current_adaptive_step:.6f}, ratio: {distance_scalar_ratio}")
                print()

                if np.any(np.array(min_lengths) > length_max_threshold):
                    # Distance too large - need smaller scalar step
                    # Update ratio estimate: distance / scalar_step
                    if current_scalar_step > 0 and avg_distance > 0:
                        observed_ratio = avg_distance / current_scalar_step
                        if distance_scalar_ratio is None:
                            distance_scalar_ratio = observed_ratio
                        else:
                            # Exponential moving average
                            distance_scalar_ratio = 0.5 * distance_scalar_ratio + 0.5 * observed_ratio

                        # Predict scalar step for target distance
                        predicted_step = target_distance / distance_scalar_ratio
                        predicted_step = np.clip(predicted_step, min_scalar_step, max_scalar_step)
                        current_adaptive_step = predicted_step

                    # Binary search backward, but with limits
                    step_size = contour_value - prev_contour_value
                    if binary_search_depth < max_binary_depth and step_size > min_scalar_step:
                        # Use adaptive prediction if available, otherwise binary search
                        if distance_scalar_ratio is not None and binary_search_depth == 0:
                            # First attempt: use predicted step
                            contour_value = prev_contour_value + current_adaptive_step
                        else:
                            contour_value = (prev_contour_value + contour_value) / 2.0
                        binary_search_depth += 1
                        trial += 1
                    else:
                        # Force accept despite exceeding threshold (to avoid infinite loop)
                        print(f"FORCE ADDED (binary depth {binary_search_depth}, step {step_size:.6f})")
                        print()
                        binary_search_depth = 0
                        trial = 0
                        self.bounding_planes.append(bounding_planes)
                        contours.append(ordered_contour_vertices)
                        contours_orig.append(ordered_contour_vertices_orig)
                        prev_contour_value = contour_value
                        contour_value += current_adaptive_step
                else:
                    binary_search_depth = 0  # Reset binary search depth on success

                    # Update ratio estimate from successful contour
                    if current_scalar_step > 0 and avg_distance > 0:
                        observed_ratio = avg_distance / current_scalar_step
                        if distance_scalar_ratio is None:
                            distance_scalar_ratio = observed_ratio
                        else:
                            distance_scalar_ratio = 0.7 * distance_scalar_ratio + 0.3 * observed_ratio

                        # Update adaptive step for next iteration
                        predicted_step = target_distance / distance_scalar_ratio
                        current_adaptive_step = np.clip(predicted_step, min_scalar_step, max_scalar_step)

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
                    if contour_value + current_adaptive_step >= 10.0:
                        contour_value = (contour_value + 10.0) / 2
                        if 10.0 - contour_value < 1e-2:
                            break
                    else:
                        contour_value += current_adaptive_step

            if iteration >= max_iterations:
                print(f"Warning: find_contours reached max iterations ({max_iterations})")
            
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

        # # Auto-detect skeleton attachments if skeleton_meshes provided
        # if skeleton_meshes is not None and len(skeleton_meshes) > 0:
        #     self.auto_detect_attachments(skeleton_meshes)

    def refine_contours(self, max_spacing_threshold, max_refinement_depth=3):
        """
        Refine contours by inserting new contours where physical spacing exceeds threshold.

        After initial contour finding, check physical distance between consecutive contours.
        If spacing is too large, binary search on scalar values to find a contour that's
        physically centered between them.

        Args:
            max_spacing_threshold: Maximum allowed physical spacing between contours
            max_refinement_depth: Maximum recursion depth to prevent infinite refinement
        """
        if len(self.bounding_planes) < 2:
            return

        print(f"\n=== Contour Refinement Pass ===")
        print(f"Max spacing threshold: {max_spacing_threshold}")
        print(f"Number of contour levels: {len(self.bounding_planes)}")

        # Debug: show all current spacings (using centroid Euclidean distance)
        print("Current spacings:")
        for level_idx in range(len(self.bounding_planes) - 1):
            current_planes = self.bounding_planes[level_idx]
            next_planes = self.bounding_planes[level_idx + 1]
            centroid_current = np.mean([p['mean'] for p in current_planes], axis=0)
            centroid_next = np.mean([p['mean'] for p in next_planes], axis=0)
            spacing = np.linalg.norm(centroid_next - centroid_current)
            status = "GAP" if spacing > max_spacing_threshold else "ok"
            print(f"  Level {level_idx}->{level_idx+1}: {spacing:.6f} [{status}]")

        refinement_iteration = 0
        max_refinement_iterations = 50  # Safety limit
        failed_scalar_ranges = set()  # Track ranges where we couldn't insert contours

        while refinement_iteration < max_refinement_iterations:
            refinement_iteration += 1
            gaps_found = []

            # Check spacing between consecutive contour levels
            for level_idx in range(len(self.bounding_planes) - 1):
                current_planes = self.bounding_planes[level_idx]
                next_planes = self.bounding_planes[level_idx + 1]

                # Get scalar values
                current_scalar = current_planes[0]['scalar_value'] if len(current_planes) > 0 else None
                next_scalar = next_planes[0]['scalar_value'] if len(next_planes) > 0 else None

                if current_scalar is None or next_scalar is None:
                    continue

                curr_means = np.array([p['mean'] for p in current_planes])
                next_means = np.array([p['mean'] for p in next_planes])

                # Different spacing calculation based on topology
                if len(current_planes) == len(next_planes):
                    # Same count - check each contour's min distance to any in other level
                    max_min_dist = 0
                    has_gap = False
                    for next_mean in next_means:
                        dists = np.linalg.norm(curr_means - next_mean, axis=1)
                        min_dist = np.min(dists)
                        if min_dist > max_min_dist:
                            max_min_dist = min_dist
                        if min_dist > max_spacing_threshold:
                            has_gap = True
                    spacing = max_min_dist
                else:
                    # Diverging/merging point - use centroid-to-centroid distance
                    # This is more lenient to avoid over-refinement at topology changes
                    centroid_curr = np.mean(curr_means, axis=0)
                    centroid_next = np.mean(next_means, axis=0)
                    spacing = np.linalg.norm(centroid_next - centroid_curr)
                    has_gap = spacing > max_spacing_threshold

                # Skip if this range already failed
                range_key = (round(current_scalar, 6), round(next_scalar, 6))
                if range_key in failed_scalar_ranges:
                    if has_gap:
                        print(f"  [Skip] Level {level_idx}: already failed, spacing={spacing:.6f}")
                    continue

                # Skip if scalar range is too small to subdivide
                scalar_range = abs(next_scalar - current_scalar)
                if scalar_range < 1e-6:
                    if has_gap:
                        print(f"  [Skip] Level {level_idx}: scalar range too small ({scalar_range:.10f}), spacing={spacing:.6f}")
                    continue

                if has_gap:
                    gaps_found.append({
                        'level_idx': level_idx,
                        'current_planes': current_planes,
                        'next_planes': next_planes,
                        'current_scalar': current_scalar,
                        'next_scalar': next_scalar,
                        'spacing': spacing
                    })

            if len(gaps_found) == 0:
                print(f"Refinement complete after {refinement_iteration} iterations. No more gaps.")
                break

            print(f"\n[Refinement {refinement_iteration}] Found {len(gaps_found)} gaps exceeding threshold:")
            for gap in gaps_found:
                print(f"  Level {gap['level_idx']}: scalar {gap['current_scalar']:.8f} -> {gap['next_scalar']:.8f}, spacing={gap['spacing']:.6f}")

            # Process gaps in reverse order to maintain correct indices
            gaps_found.sort(key=lambda x: x['level_idx'], reverse=True)

            contours_inserted = 0
            for gap in gaps_found:
                insert_idx = gap['level_idx'] + 1

                # Calculate centroids for physical midpoint targeting
                centroid_low = np.mean([p['mean'] for p in gap['current_planes']], axis=0)
                centroid_high = np.mean([p['mean'] for p in gap['next_planes']], axis=0)

                # Use current plane as reference for alignment (not last plane in list)
                prev_plane = gap['current_planes'][0] if len(gap['current_planes']) > 0 else None

                # Find a contour physically centered between the two
                best_scalar, best_planes, best_contours = self._find_contour_between(
                    gap['current_scalar'], gap['next_scalar'],
                    centroid_low, centroid_high,
                    prev_bounding_plane=prev_plane
                )

                if best_planes is not None and len(best_planes) > 0:
                    print(f"  Inserting contour at scalar {best_scalar:.4f} (index {insert_idx})")
                    self.bounding_planes.insert(insert_idx, best_planes)
                    self.contours.insert(insert_idx, best_contours)
                    contours_inserted += 1
                else:
                    # Mark this range as failed to avoid retrying
                    range_key = (round(gap['current_scalar'], 6), round(gap['next_scalar'], 6))
                    failed_scalar_ranges.add(range_key)
                    print(f"  Warning: Could not find valid contour between {gap['current_scalar']:.4f} and {gap['next_scalar']:.4f} (skipping)")

            # Update draw_contour_stream
            self.draw_contour_stream = [True] * len(self.contours)

            # Safety: if no contours were inserted this iteration, we can't make progress
            if contours_inserted == 0:
                print(f"No contours inserted this iteration. Stopping refinement.")
                break

        if refinement_iteration >= max_refinement_iterations:
            print(f"Warning: Refinement reached max iterations ({max_refinement_iterations})")

        print(f"=== Refinement Done: {len(self.bounding_planes)} contour levels ===\n")

    # _trim_independent_section moved to MuscleMeshMixin in muscle_mesh.py

    def _count_shared_contours_from_end(self, merge_threshold=0.005):
        """
        Count how many contours are shared across all streams, comparing from the end.
        Shared contours have similar centroids across all streams at the same offset from end.
        """
        if len(self.contours) <= 1:
            return 0

        min_len = min(len(stream) for stream in self.contours)
        shared_count = 0

        for offset in range(min_len):
            # Get centroids from the end of each stream
            centroids = [np.mean(stream[-(offset + 1)], axis=0) for stream in self.contours]

            # Check if all streams have similar centroids
            all_similar = True
            for i in range(1, len(centroids)):
                if np.linalg.norm(centroids[i] - centroids[0]) > merge_threshold:
                    all_similar = False
                    break

            if all_similar:
                shared_count = offset + 1
            else:
                break

        return shared_count

    def _check_shared_contour_spacing(self, max_spacing_threshold):
        """
        Check if shared contours (after the merge point) satisfy the spacing threshold.
        Shared contours cannot be removed, but we warn if spacing is too large.
        """
        if len(self.contours) <= 1:
            return

        # Find where streams have the same contours (shared section)
        # Compare from the end backwards
        min_len = min(len(stream) for stream in self.contours)
        shared_count = 0

        for offset in range(min_len):
            centroids = [np.mean(stream[-(offset + 1)], axis=0) for stream in self.contours]
            all_similar = True
            for i in range(1, len(centroids)):
                if np.linalg.norm(centroids[i] - centroids[0]) > 0.005:
                    all_similar = False
                    break
            if all_similar:
                shared_count = offset + 1
            else:
                break

        if shared_count <= 1:
            return

        # Check spacing within the shared section
        stream = self.contours[0]  # All streams have same shared section
        shared_start = len(stream) - shared_count

        gaps_exceeding = []
        for i in range(shared_start, len(stream) - 1):
            centroid_curr = np.mean(stream[i], axis=0)
            centroid_next = np.mean(stream[i + 1], axis=0)
            spacing = np.linalg.norm(centroid_next - centroid_curr)

            if spacing > max_spacing_threshold:
                gaps_exceeding.append((i - shared_start, spacing))

        if gaps_exceeding:
            print(f"  Warning: {len(gaps_exceeding)} shared contour gaps exceed threshold:")
            for idx, spacing in gaps_exceeding[:5]:  # Show first 5
                print(f"    Shared level {idx}: spacing {spacing:.4f} > {max_spacing_threshold}")
            if len(gaps_exceeding) > 5:
                print(f"    ... and {len(gaps_exceeding) - 5} more")

    def _find_contour_between(self, scalar_low, scalar_high, centroid_low, centroid_high, num_samples=10, prev_bounding_plane=None):
        """
        Find a contour that is physically centered between two existing contours.

        Samples multiple scalar values, finds valid contours, and picks the one
        closest to the physical midpoint.

        Args:
            scalar_low: Lower scalar value
            scalar_high: Higher scalar value
            centroid_low: Centroid of lower contour
            centroid_high: Centroid of higher contour
            num_samples: Number of scalar values to sample
            prev_bounding_plane: Previous bounding plane for alignment reference

        Returns:
            (scalar, planes, contours) or (None, None, None) if failed
        """
        # Skip if scalar range is too small
        if abs(scalar_high - scalar_low) < 1e-6:
            return None, None, None

        target_centroid = (centroid_low + centroid_high) / 2

        best_scalar = None
        best_planes = None
        best_contours = None
        best_dist_to_target = np.inf

        # Sample scalar values evenly
        for i in range(1, num_samples):
            ratio = i / num_samples
            try_scalar = scalar_low + (scalar_high - scalar_low) * ratio

            planes, contours, _ = self.find_contour(try_scalar, prev_bounding_plane=prev_bounding_plane)

            if len(planes) > 0 and len(contours) > 0:
                # Calculate centroid of this contour
                centroid = np.mean([p['mean'] for p in planes], axis=0)
                dist_to_target = np.linalg.norm(centroid - target_centroid)

                if dist_to_target < best_dist_to_target:
                    best_dist_to_target = dist_to_target
                    best_scalar = try_scalar
                    best_planes = planes
                    best_contours = contours

        return best_scalar, best_planes, best_contours

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
                        preserve = getattr(self, '_contours_normalized', False)
                        self.contours[stream_i][i], contour_match = self.find_contour_match(contours_orig[stream_i][i], bounding_plane, preserve_order=preserve)
                        bounding_plane_info['contour_match'] = contour_match
        
    def _align_bounding_planes(self):
        """
        Align rectangle-like bounding planes to have consistent orientation.
        Square-like planes are SKIPPED (they will be interpolated during smoothing).
        Only rectangle-like planes are used as reference for alignment.
        """
        if len(self.bounding_planes) == 0:
            return

        # Process each stream separately
        for stream_idx in range(len(self.bounding_planes[0])):
            # Find first non-square-like plane as reference
            if len(self.bounding_planes[0]) <= stream_idx:
                continue

            prev_basis_x = None
            prev_basis_y = None

            # Find first rectangle-like plane to use as reference
            for level_idx in range(len(self.bounding_planes)):
                if len(self.bounding_planes[level_idx]) <= stream_idx:
                    continue
                plane_info = self.bounding_planes[level_idx][stream_idx]
                if not plane_info.get('square_like', False):
                    prev_basis_x = plane_info['basis_x'].copy()
                    prev_basis_y = plane_info['basis_y'].copy()
                    break

            if prev_basis_x is None:
                # All planes are square-like, use first one
                prev_basis_x = self.bounding_planes[0][stream_idx]['basis_x'].copy()
                prev_basis_y = self.bounding_planes[0][stream_idx]['basis_y'].copy()

            # Align each level (only rectangle-like planes)
            for level_idx in range(len(self.bounding_planes)):
                if len(self.bounding_planes[level_idx]) <= stream_idx:
                    continue

                plane_info = self.bounding_planes[level_idx][stream_idx]

                # Skip square-like planes - they will be interpolated during smoothing
                if plane_info.get('square_like', False):
                    continue

                basis_x = plane_info['basis_x']
                basis_y = plane_info['basis_y']
                basis_z = plane_info['basis_z']
                mean = plane_info['mean']

                # Find best rotation (0°, 90°, 180°, 270°) to align with previous
                candidates = [
                    (basis_x, basis_y),
                    (basis_y, -basis_x),
                    (-basis_x, -basis_y),
                    (-basis_y, basis_x)
                ]

                best_candidate = None
                best_dot = -np.inf
                for cand_x, cand_y in candidates:
                    # Measure alignment with previous plane
                    dot_x = np.dot(cand_x, prev_basis_x)
                    dot_y = np.dot(cand_y, prev_basis_y)
                    total_dot = dot_x + dot_y
                    if total_dot > best_dot:
                        best_dot = total_dot
                        best_candidate = (cand_x, cand_y)

                new_basis_x, new_basis_y = best_candidate

                # Re-orthogonalize to ensure exactly 90 degrees
                new_basis_x = new_basis_x / (np.linalg.norm(new_basis_x) + 1e-10)
                new_basis_y = np.cross(basis_z, new_basis_x)
                new_basis_y = new_basis_y / (np.linalg.norm(new_basis_y) + 1e-10)

                # Update plane info with aligned basis
                plane_info['basis_x'] = new_basis_x
                plane_info['basis_y'] = new_basis_y

                # Recompute bounding plane corners with new basis (don't modify contour vertices)
                contour_points = self.contours[level_idx][stream_idx]
                projected_2d = np.array([[np.dot(v - mean, new_basis_x), np.dot(v - mean, new_basis_y)] for v in contour_points])
                min_x, max_x = np.min(projected_2d[:, 0]), np.max(projected_2d[:, 0])
                min_y, max_y = np.min(projected_2d[:, 1]), np.max(projected_2d[:, 1])

                bounding_plane_2d = np.array([
                    [min_x, min_y], [max_x, min_y],
                    [max_x, max_y], [min_x, max_y]
                ])
                plane_info['bounding_plane'] = np.array([mean + x * new_basis_x + y * new_basis_y for x, y in bounding_plane_2d])
                plane_info['projected_2d'] = np.array([mean + x * new_basis_x + y * new_basis_y for x, y in projected_2d])

                # Update contour match WITHOUT modifying the original contour vertices
                # Only update the P-Q correspondence for the new bounding plane orientation
                contour_match = []
                for v in contour_points:
                    # Project vertex to bounding plane and find Q
                    rel_v = v - mean
                    x_proj = np.dot(rel_v, new_basis_x)
                    y_proj = np.dot(rel_v, new_basis_y)
                    # Normalize to [0,1] range based on bounding box
                    width = max_x - min_x
                    height = max_y - min_y
                    if width > 1e-10 and height > 1e-10:
                        q_x = (x_proj - min_x) / width
                        q_y = (y_proj - min_y) / height
                        q = mean + q_x * (max_x - min_x) * new_basis_x + q_y * (max_y - min_y) * new_basis_y
                        q = plane_info['bounding_plane'][0] + q_x * (plane_info['bounding_plane'][1] - plane_info['bounding_plane'][0]) + q_y * (plane_info['bounding_plane'][3] - plane_info['bounding_plane'][0])
                    else:
                        q = mean
                    contour_match.append([v, q])
                plane_info['contour_match'] = contour_match

                # Use this rectangle-like plane as reference for next
                prev_basis_x = new_basis_x.copy()
                prev_basis_y = new_basis_y.copy()

    def smoothen_contours(self):
        """
        Smoothen bounding plane orientations to avoid twisted bounding boxes.

        For square-like contours (circles), bounding box orientation can be arbitrary
        and may cause twisting. This function adjusts orientations by interpolating
        between non-square-like neighbors.

        Handles multiple contours at diverging/merging points by finding closest
        matching contours in neighboring levels.

        Should be run before resampling - uses original contour points.

        NOTE: Currently disabled as it interferes with contour matching.
        """
        print("smoothen_contours: skipped (disabled)")
        return

        if len(self.bounding_planes) == 0:
            print("No Contours found")
            return

        # First align all bounding planes to have consistent orientation
        self._align_bounding_planes()

        # Keep original contours for consistent projection across iterations
        contours_orig = [[c.copy() for c in contour_group] for contour_group in self.contours]

        # Smooth square-like planes by propagating orientation from nearest non-square-like planes
        for iteration in range(10):
            smooth_bounding_planes = []
            for i, bounding_plane_infos in enumerate(self.bounding_planes):
                # Process each contour at this level (including first/last)
                new_level_planes = []
                for contour_idx, bounding_plane_info in enumerate(bounding_plane_infos):
                    # Only smooth square-like bounding planes, keep rectangle-like ones unchanged
                    if not bounding_plane_info.get('square_like', False):
                        new_level_planes.append(bounding_plane_info)
                        continue

                    curr_mean = bounding_plane_info['mean']
                    basis_x = bounding_plane_info['basis_x']
                    basis_y = bounding_plane_info['basis_y']
                    basis_z = bounding_plane_info['basis_z']
                    mean = bounding_plane_info['mean']

                    # Find closest non-square-like prev plane (and its level index)
                    prev_basis_x = None
                    prev_level = None
                    for search_i in range(i - 1, -1, -1):
                        for bp in self.bounding_planes[search_i]:
                            if not bp.get('square_like', False):
                                dist = np.linalg.norm(curr_mean - bp['mean'])
                                if prev_basis_x is None or dist < prev_dist:
                                    prev_basis_x = bp['basis_x']
                                    prev_dist = dist
                                    prev_level = search_i
                        if prev_basis_x is not None:
                            break

                    # Find closest non-square-like next plane (and its level index)
                    next_basis_x = None
                    next_level = None
                    for search_i in range(i + 1, len(self.bounding_planes)):
                        for bp in self.bounding_planes[search_i]:
                            if not bp.get('square_like', False):
                                dist = np.linalg.norm(curr_mean - bp['mean'])
                                if next_basis_x is None or dist < next_dist:
                                    next_basis_x = bp['basis_x']
                                    next_dist = dist
                                    next_level = search_i
                        if next_basis_x is not None:
                            break

                    # If no non-square neighbors found, skip this plane
                    if prev_basis_x is None and next_basis_x is None:
                        new_level_planes.append(bounding_plane_info)
                        continue

                    # Interpolate based on position between prev and next
                    if prev_basis_x is None:
                        ref_basis_x = next_basis_x
                    elif next_basis_x is None:
                        ref_basis_x = prev_basis_x
                    else:
                        # Align next_basis_x to prev_basis_x to avoid twisting
                        # Find rotation (0°, 90°, 180°, 270°) that minimizes angle
                        next_bp = self.bounding_planes[next_level][0]
                        for bp in self.bounding_planes[next_level]:
                            if not bp.get('square_like', False):
                                next_bp = bp
                                break
                        next_bx = next_bp['basis_x']
                        next_by = next_bp['basis_y']

                        candidates = [
                            next_bx,
                            next_by,
                            -next_bx,
                            -next_by
                        ]
                        best_aligned = next_bx
                        best_dot = -np.inf
                        for cand in candidates:
                            dot = np.dot(cand, prev_basis_x)
                            if dot > best_dot:
                                best_dot = dot
                                best_aligned = cand

                        aligned_next_basis_x = best_aligned

                        # Compute interpolation weight based on level index
                        total_span = next_level - prev_level
                        if total_span > 0:
                            t = (i - prev_level) / total_span  # 0 = at prev, 1 = at next
                        else:
                            t = 0.5
                        # Interpolate basis_x (will be re-orthogonalized later)
                        ref_basis_x = (1 - t) * prev_basis_x + t * aligned_next_basis_x
                        ref_basis_x = ref_basis_x / (np.linalg.norm(ref_basis_x) + 1e-10)

                    # Project reference onto this plane (perpendicular to basis_z)
                    new_basis_x = ref_basis_x - np.dot(ref_basis_x, basis_z) * basis_z
                    norm = np.linalg.norm(new_basis_x)
                    if norm < 1e-10:
                        new_level_planes.append(bounding_plane_info)
                        continue
                    new_basis_x = new_basis_x / norm

                    # Compute basis_y as cross product to ensure 90 degrees
                    new_basis_y = np.cross(basis_z, new_basis_x)
                    new_basis_y = new_basis_y / (np.linalg.norm(new_basis_y) + 1e-10)

                    # Keep original mean and basis_z
                    new_mean = mean
                    new_newell = basis_z

                    # Use original contour points for projection
                    original_contour = contours_orig[i][contour_idx]
                    projected_2d = np.array([[np.dot(v - new_mean, new_basis_x), np.dot(v - new_mean, new_basis_y)] for v in original_contour])
                    area = compute_polygon_area(projected_2d)

                    min_x, max_x = np.min(projected_2d[:, 0]), np.max(projected_2d[:, 0])
                    min_y, max_y = np.min(projected_2d[:, 1]), np.max(projected_2d[:, 1])

                    bounding_plane_2d = np.array([
                        [min_x, min_y], [max_x, min_y],
                        [max_x, max_y], [min_x, max_y]
                    ])
                    bounding_plane = np.array([new_mean + x * new_basis_x + y * new_basis_y for x, y in bounding_plane_2d])
                    projected_2d_3d = np.array([new_mean + x * new_basis_x + y * new_basis_y for x, y in projected_2d])

                    # Keep original contour order - only update bounding plane orientation
                    # Don't call find_contour_match here as it messes up well-found contour matching
                    contour_match = bounding_plane_info.get('contour_match', bounding_plane)

                    new_bounding_plane_info = {
                        'basis_x': new_basis_x,
                        'basis_y': new_basis_y,
                        'basis_z': basis_z,
                        'mean': new_mean,
                        'bounding_plane': bounding_plane,
                        'projected_2d': projected_2d_3d,
                        'area': area,
                        'contour_match': contour_match,
                        'scalar_value': bounding_plane_info['scalar_value'],
                        'square_like': bounding_plane_info['square_like'],
                        'newell_normal': new_newell,
                    }

                    new_level_planes.append(new_bounding_plane_info)

                smooth_bounding_planes.append(new_level_planes)

            self.bounding_planes = smooth_bounding_planes

    def fix_90_degree_corners(self):
        """
        Fix all bounding planes to have exactly 90-degree corners.

        Re-orthogonalizes basis vectors and recomputes bounding plane corners
        to ensure perfect rectangles.
        """
        if len(self.bounding_planes) == 0:
            print("No bounding planes found")
            return

        for stream_idx, bp_stream in enumerate(self.bounding_planes):
            for contour_idx, bp_info in enumerate(bp_stream):
                basis_x = bp_info.get('basis_x')
                basis_y = bp_info.get('basis_y')
                basis_z = bp_info.get('basis_z')
                mean = bp_info.get('mean')
                old_corners = bp_info.get('bounding_plane')

                if basis_x is None or basis_z is None or mean is None or old_corners is None:
                    continue

                # Re-orthogonalize: basis_y = cross(basis_z, basis_x), normalized
                basis_x = basis_x / (np.linalg.norm(basis_x) + 1e-10)
                new_basis_y = np.cross(basis_z, basis_x)
                new_basis_y = new_basis_y / (np.linalg.norm(new_basis_y) + 1e-10)

                # Compute the center and extents from old corners
                center = np.mean(old_corners, axis=0)

                # Project old corners onto the orthonormal basis to get extents
                x_coords = [np.dot(c - center, basis_x) for c in old_corners]
                y_coords = [np.dot(c - center, new_basis_y) for c in old_corners]

                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)

                # Recompute corners with orthonormal basis (guaranteed 90 degrees)
                bounding_plane_2d = np.array([
                    [min_x, min_y], [max_x, min_y],
                    [max_x, max_y], [min_x, max_y]
                ])
                new_corners = np.array([center + x * basis_x + y * new_basis_y for x, y in bounding_plane_2d])

                # Update the bounding plane info
                bp_info['basis_y'] = new_basis_y
                bp_info['bounding_plane'] = new_corners

                # Also update contour_match if it exists
                contour_match = bp_info.get('contour_match')
                if contour_match is not None:
                    # Recompute Q points (projections onto the new bounding plane)
                    new_contour_match = []
                    for p, q in contour_match:
                        # Project P onto the new plane
                        p = np.array(p)
                        p_2d = np.array([np.dot(p - center, basis_x), np.dot(p - center, new_basis_y)])
                        # Clip to bounding box
                        p_2d[0] = np.clip(p_2d[0], min_x, max_x)
                        p_2d[1] = np.clip(p_2d[1], min_y, max_y)
                        new_q = center + p_2d[0] * basis_x + p_2d[1] * new_basis_y
                        new_contour_match.append([p, new_q])
                    bp_info['contour_match'] = new_contour_match

        print(f"Fixed 90-degree corners for all bounding planes")

    def normalize_contours_to_max(self):
        """
        Normalize all contours before stream finding:
        1. Find max vertex count across all contours
        2. Resample all contours to that max count
        3. Ensure consistent winding direction (CCW when viewed from normal)
        4. Ensure consistent first vertex (using reference direction)

        This should be called BEFORE find_contour_stream.
        """
        if self.contours is None or len(self.contours) == 0:
            print("No contours found. Please run find_contours first.")
            return

        if self.bounding_planes is None or len(self.bounding_planes) == 0:
            print("No bounding planes found. Please run find_contours first.")
            return

        print("Normalizing contours (aligning first vertex and winding direction)...")

        # Compute reference direction from origin to insertion
        origin_centroids = [np.mean(c, axis=0) for c in self.contours[0]]
        insertion_centroids = [np.mean(c, axis=0) for c in self.contours[-1]]
        origin_center = np.mean(origin_centroids, axis=0)
        insertion_center = np.mean(insertion_centroids, axis=0)
        muscle_axis = insertion_center - origin_center
        muscle_axis_norm = np.linalg.norm(muscle_axis)
        if muscle_axis_norm > 1e-10:
            muscle_axis = muscle_axis / muscle_axis_norm
        else:
            muscle_axis = np.array([0, 1, 0])

        # Reference direction perpendicular to muscle axis (for first vertex)
        world_up = np.array([0, 1, 0])
        if abs(np.dot(muscle_axis, world_up)) > 0.9:
            world_up = np.array([1, 0, 0])
        reference_direction = np.cross(muscle_axis, world_up)
        reference_direction = reference_direction / (np.linalg.norm(reference_direction) + 1e-10)

        # Store as instance variables for use by find_contour_stream
        self._muscle_axis = muscle_axis
        self._reference_direction = reference_direction

        normalized_contours = []
        new_bounding_planes = []

        # Track previous level's first vertices for each stream (to chain alignment)
        prev_first_vertices = {}
        prev_winding_signs = {}

        for level_idx, (contour_group, bounding_plane_group) in enumerate(zip(self.contours, self.bounding_planes)):
            normalized_group = []
            new_bp_group = []

            for contour_idx, (contour, bp_info) in enumerate(zip(contour_group, bounding_plane_group)):
                contour = np.array(contour)
                n = len(contour)

                if n < 2:
                    # Keep as-is for degenerate contours
                    new_bp_info = bp_info.copy()
                    new_bp_info['contour_vertices'] = contour.copy()
                    normalized_group.append(contour)
                    new_bp_group.append(new_bp_info)
                    continue

                centroid = np.mean(contour, axis=0)

                # Project contour to best-fit plane
                basis_x = bp_info.get('basis_x')
                basis_y = bp_info.get('basis_y')
                basis_z = bp_info.get('basis_z')
                plane_center = bp_info.get('mean', centroid)

                if basis_x is not None and basis_y is not None:
                    contour_planar = []
                    for p in contour:
                        rel = p - plane_center
                        x_coord = np.dot(rel, basis_x)
                        y_coord = np.dot(rel, basis_y)
                        p_proj = plane_center + x_coord * basis_x + y_coord * basis_y
                        contour_planar.append(p_proj)
                    contour = np.array(contour_planar)
                    centroid = np.mean(contour, axis=0)

                # Find consistent first vertex
                scalar_value = bp_info.get('scalar_value')
                ref_point = self.find_reference_intersection(scalar_value) if scalar_value is not None else None

                if ref_point is not None:
                    distances = np.linalg.norm(contour - ref_point, axis=1)
                    start_idx = np.argmin(distances)
                elif contour_idx in prev_first_vertices:
                    prev_first = prev_first_vertices[contour_idx]
                    distances = np.linalg.norm(contour - prev_first, axis=1)
                    start_idx = np.argmin(distances)
                else:
                    basis_x = bp_info.get('basis_x')
                    if basis_x is not None:
                        directions = contour - centroid
                        dots = directions @ basis_x
                        start_idx = np.argmax(dots)
                    else:
                        start_idx = 0

                contour = np.roll(contour, -start_idx, axis=0)

                # Sync winding direction
                basis_x = bp_info.get('basis_x')
                basis_y = bp_info.get('basis_y')
                basis_z = bp_info.get('basis_z')

                if basis_x is not None and basis_y is not None:
                    signed_area = 0.0
                    for i in range(len(contour)):
                        p1 = contour[i] - centroid
                        p2 = contour[(i + 1) % len(contour)] - centroid
                        cross = np.cross(p1, p2)
                        if basis_z is not None:
                            signed_area += np.dot(cross, basis_z)
                        else:
                            signed_area += np.linalg.norm(cross)

                    if contour_idx in prev_winding_signs:
                        if (signed_area > 0) != prev_winding_signs[contour_idx]:
                            contour = np.roll(contour[::-1], 1, axis=0)
                            signed_area = -signed_area
                    else:
                        if signed_area < 0:
                            contour = np.roll(contour[::-1], 1, axis=0)
                            signed_area = -signed_area

                    prev_winding_signs[contour_idx] = (signed_area > 0)

                prev_first_vertices[contour_idx] = contour[0].copy()

                # Keep original vertex count (no resampling)
                resampled = contour

                new_bp_info = bp_info.copy()
                new_bp_info['contour_vertices'] = resampled.copy()

                normalized_group.append(resampled)
                new_bp_group.append(new_bp_info)

            normalized_contours.append(normalized_group)
            new_bounding_planes.append(new_bp_group)

        self.contours = normalized_contours
        self.bounding_planes = new_bounding_planes
        self._contours_normalized = True

        print("Contours normalized (first vertex and winding aligned)")

    def _align_bounding_plane_corners(self, muscle_axis, reference_direction, stream_first=False):
        """
        Align bounding planes by adjusting basis vectors, then regenerating corners.
        Delegates to _align_bounding_planes for the actual work.
        """
        if len(self.bounding_planes) == 0:
            return

        if not stream_first:
            # Data is [level][stream] - use original _align_bounding_planes directly
            self._align_bounding_planes()
        else:
            # Data is [stream][level] - transpose, align, transpose back
            num_streams = len(self.bounding_planes)
            if num_streams == 0:
                return
            num_levels = len(self.bounding_planes[0])

            # Transpose bounding_planes: [stream][level] -> [level][stream]
            transposed_bp = []
            for level_idx in range(num_levels):
                level_planes = []
                for stream_idx in range(num_streams):
                    if level_idx < len(self.bounding_planes[stream_idx]):
                        level_planes.append(self.bounding_planes[stream_idx][level_idx])
                transposed_bp.append(level_planes)

            # Transpose contours: [stream][level] -> [level][stream]
            transposed_contours = []
            for level_idx in range(num_levels):
                level_contours = []
                for stream_idx in range(num_streams):
                    if level_idx < len(self.contours[stream_idx]):
                        level_contours.append(self.contours[stream_idx][level_idx])
                transposed_contours.append(level_contours)

            # Temporarily swap
            orig_bp = self.bounding_planes
            orig_contours = self.contours
            self.bounding_planes = transposed_bp
            self.contours = transposed_contours

            # Call original alignment function
            self._align_bounding_planes()

            # Transpose back: [level][stream] -> [stream][level]
            result_bp = []
            for stream_idx in range(num_streams):
                stream_planes = []
                for level_idx in range(num_levels):
                    if stream_idx < len(self.bounding_planes[level_idx]):
                        stream_planes.append(self.bounding_planes[level_idx][stream_idx])
                result_bp.append(stream_planes)

            result_contours = []
            for stream_idx in range(num_streams):
                stream_contours = []
                for level_idx in range(num_levels):
                    if stream_idx < len(self.contours[level_idx]):
                        stream_contours.append(self.contours[level_idx][stream_idx])
                result_contours.append(stream_contours)

            self.bounding_planes = result_bp
            self.contours = result_contours

    def _fix_intersecting_planes(self, stream_first=False):
        """
        Detect and fix cases where consecutive bounding planes intersect.
        When planes would intersect, adjust the orientation to prevent it.
        """
        if len(self.bounding_planes) == 0:
            return

        if stream_first:
            num_streams = len(self.bounding_planes)
            num_levels = len(self.bounding_planes[0]) if num_streams > 0 else 0
        else:
            num_levels = len(self.bounding_planes)
            num_streams = len(self.bounding_planes[0]) if num_levels > 0 else 0

        for stream_idx in range(num_streams):
            for level_idx in range(num_levels - 1):
                # Get current and next plane
                if stream_first:
                    if stream_idx >= len(self.bounding_planes):
                        continue
                    if level_idx >= len(self.bounding_planes[stream_idx]) or level_idx + 1 >= len(self.bounding_planes[stream_idx]):
                        continue
                    bp_curr = self.bounding_planes[stream_idx][level_idx]
                    bp_next = self.bounding_planes[stream_idx][level_idx + 1]
                else:
                    if level_idx >= len(self.bounding_planes) or level_idx + 1 >= len(self.bounding_planes):
                        continue
                    if stream_idx >= len(self.bounding_planes[level_idx]) or stream_idx >= len(self.bounding_planes[level_idx + 1]):
                        continue
                    bp_curr = self.bounding_planes[level_idx][stream_idx]
                    bp_next = self.bounding_planes[level_idx + 1][stream_idx]

                corners_curr = np.array(bp_curr.get('bounding_plane', []))
                corners_next = np.array(bp_next.get('bounding_plane', []))

                if len(corners_curr) != 4 or len(corners_next) != 4:
                    continue

                center_curr = bp_curr.get('mean', np.mean(corners_curr, axis=0))
                center_next = bp_next.get('mean', np.mean(corners_next, axis=0))

                basis_z_curr = bp_curr.get('basis_z')
                basis_z_next = bp_next.get('basis_z')

                if basis_z_curr is None or basis_z_next is None:
                    continue

                # Check if planes might intersect:
                # 1. Distance between centers along average normal
                # 2. Maximum extent of planes perpendicular to this direction
                avg_normal = (basis_z_curr + basis_z_next) / 2
                avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-10)

                center_dist = abs(np.dot(center_next - center_curr, avg_normal))

                # Compute max extent of each plane perpendicular to avg_normal
                def plane_extent_perp(corners, center, normal):
                    max_dist = 0
                    for c in corners:
                        rel = c - center
                        perp = rel - np.dot(rel, normal) * normal
                        max_dist = max(max_dist, np.linalg.norm(perp))
                    return max_dist

                extent_curr = plane_extent_perp(corners_curr, center_curr, avg_normal)
                extent_next = plane_extent_perp(corners_next, center_next, avg_normal)

                # If center distance is less than extents overlap, planes might intersect
                # Only trigger when planes are very close - use 0.1 factor to be conservative
                # Also require that planes are significantly non-parallel (angle > 15 degrees)
                normal_dot = abs(np.dot(basis_z_curr, basis_z_next))
                angle_deg = np.degrees(np.arccos(np.clip(normal_dot, 0, 1)))

                if center_dist < 0.1 * (extent_curr + extent_next) and angle_deg > 15:
                    # Planes are very close AND significantly tilted - might intersect
                    # Fix by making the smaller plane parallel to the larger one
                    area_curr = bp_curr.get('area', 0)
                    area_next = bp_next.get('area', 0)

                    if area_curr >= area_next:
                        # Adjust next plane to be parallel to current
                        self._make_plane_parallel(bp_next, bp_curr, center_next)
                    else:
                        # Adjust current plane to be parallel to next
                        self._make_plane_parallel(bp_curr, bp_next, center_curr)

    def _make_plane_parallel(self, bp_to_adjust, bp_reference, new_center):
        """
        Adjust bp_to_adjust to be parallel to bp_reference while keeping its center.
        Preserves the size/shape of the plane being adjusted.
        """
        corners_adjust = np.array(bp_to_adjust.get('bounding_plane', []))
        corners_ref = np.array(bp_reference.get('bounding_plane', []))

        if len(corners_adjust) != 4 or len(corners_ref) != 4:
            return

        # Get reference plane's basis
        basis_x_ref = bp_reference.get('basis_x')
        basis_y_ref = bp_reference.get('basis_y')

        if basis_x_ref is None or basis_y_ref is None:
            return

        # Compute size of plane to adjust (in its current basis)
        center_adjust = np.mean(corners_adjust, axis=0)

        # Get the half-widths along each axis
        basis_x_adj = bp_to_adjust.get('basis_x')
        basis_y_adj = bp_to_adjust.get('basis_y')

        if basis_x_adj is None or basis_y_adj is None:
            return

        # Compute extents in original basis
        x_coords = [np.dot(c - center_adjust, basis_x_adj) for c in corners_adjust]
        y_coords = [np.dot(c - center_adjust, basis_y_adj) for c in corners_adjust]
        half_w = (max(x_coords) - min(x_coords)) / 2
        half_h = (max(y_coords) - min(y_coords)) / 2

        # Safety check: don't create degenerate planes
        if half_w < 1e-6 or half_h < 1e-6:
            print(f"Warning: _make_plane_parallel would create degenerate plane (half_w={half_w}, half_h={half_h}), skipping")
            return

        # Rebuild corners using reference basis but original size
        # Create corners in standard order: (-x,-y), (+x,-y), (+x,+y), (-x,+y)
        raw_corners = np.array([
            new_center - half_w * basis_x_ref - half_h * basis_y_ref,
            new_center + half_w * basis_x_ref - half_h * basis_y_ref,
            new_center + half_w * basis_x_ref + half_h * basis_y_ref,
            new_center - half_w * basis_x_ref + half_h * basis_y_ref,
        ])

        # Match to reference plane's corner order using distance-based rotation
        best_rotation = 0
        min_total_dist = float('inf')
        for rotation in range(4):
            total_dist = 0
            for i in range(4):
                raw_corner = raw_corners[(i + rotation) % 4]
                ref_corner = corners_ref[i]
                total_dist += np.linalg.norm(raw_corner - ref_corner)
            if total_dist < min_total_dist:
                min_total_dist = total_dist
                best_rotation = rotation

        new_corners = np.roll(raw_corners, -best_rotation, axis=0)

        bp_to_adjust['bounding_plane'] = new_corners
        bp_to_adjust['basis_x'] = basis_x_ref.copy()
        bp_to_adjust['basis_y'] = basis_y_ref.copy()
        bp_to_adjust['basis_z'] = bp_reference.get('basis_z').copy()

    def _resample_single_contour_preserve_start(self, contour_points, num_samples):
        """
        Resample a contour to num_samples vertices, preserving the first vertex position.
        Uses arc-length parameterization. First vertex stays at index 0.

        Args:
            contour_points: numpy array of shape (N, 3) - already has correct start/winding
            num_samples: int - desired number of output vertices

        Returns:
            numpy array of shape (num_samples, 3)
        """
        contour = np.array(contour_points)
        n = len(contour)

        if n < 2:
            return np.tile(contour[0], (num_samples, 1))

        # Always resample for uniform arc-length spacing, even if n == num_samples

        # Compute cumulative arc lengths (closed loop)
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

        # Sample at uniform parameter values (endpoint=False for closed loop)
        t_new = np.linspace(0, 1, num_samples, endpoint=False)

        # Interpolate positions
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

        # Store original contours before resampling (deep copy)
        self.contours_orig = [[c.copy() for c in contour_group] for contour_group in self.contours]
        self.bounding_planes_orig = [[bp.copy() for bp in bp_group] for bp_group in self.bounding_planes]

        resampled_contours = []
        new_bounding_planes = []

        # Track previous level's first vertices for chained alignment
        # This ensures consistency between resampled levels
        prev_first_vertices = {}  # contour_idx -> first vertex position

        print(f"Resampling {len(self.contours)} levels, {len(self.bounding_planes)} bounding plane groups")
        for level_idx, (contour_group, bounding_plane_group) in enumerate(zip(self.contours, self.bounding_planes)):
            resampled_group = []
            new_bp_group = []

            level_type = "origin" if level_idx == 0 else ("insertion" if level_idx == len(self.contours) - 1 else "mid")
            print(f"  Level {level_idx} ({level_type}): {len(contour_group)} contours, {len(bounding_plane_group)} planes")

            for contour_idx, (contour, bounding_plane_info) in enumerate(zip(contour_group, bounding_plane_group)):
                original_size = len(contour)

                # Use bounding plane corner 0 as reference for consistent winding
                bounding_plane_corners = bounding_plane_info.get('bounding_plane')
                if bounding_plane_corners is not None and len(bounding_plane_corners) >= 1:
                    # Use corner 0 as the reference point for this contour
                    corner_ref = np.array(bounding_plane_corners[0])
                else:
                    # Fallback to chaining if no bounding plane available
                    corner_ref = prev_first_vertices.get(contour_idx, None)

                # Resample the contour uniformly, starting from vertex closest to corner 0
                resampled = self._resample_single_contour(np.array(contour), num_samples, corner_ref)

                # Store first vertex for next level (fallback for contours without bounding planes)
                prev_first_vertices[contour_idx] = resampled[0].copy()

                print(f"    Contour {contour_idx}: {original_size} -> {len(resampled)} points")

                # Get the bounding plane rectangle from existing bounding_plane_info
                bounding_plane = bounding_plane_info['bounding_plane']

                # Re-run find_contour_match to align resampled contour to bounding plane
                # Use preserve_order=True to keep normalized vertex order
                aligned_contour, contour_match = self.find_contour_match(resampled, bounding_plane, preserve_order=True)

                # Update bounding plane info with new contour match
                new_bp_info = bounding_plane_info.copy()
                # Store original contour_match for waypoint finding (before resampling)
                if 'contour_match' in bounding_plane_info and 'contour_match_orig' not in bounding_plane_info:
                    new_bp_info['contour_match_orig'] = bounding_plane_info['contour_match']
                # Store resampled contour_match for mesh building
                new_bp_info['contour_match'] = contour_match

                resampled_group.append(aligned_contour)
                new_bp_group.append(new_bp_info)

            resampled_contours.append(resampled_group)
            new_bounding_planes.append(new_bp_group)

        self.contours = resampled_contours
        self.bounding_planes = new_bounding_planes
        print(f"Resampled {len(self.contours)} contour levels to {num_samples} vertices each")

    def _resample_single_contour(self, contour_points, num_samples, reference_point=None):
        """
        Resample a single closed contour to have exactly num_samples vertices.
        Uses arc-length parameterization with consistent starting point.

        Args:
            contour_points: numpy array of shape (N, 3)
            num_samples: int - desired number of output vertices
            reference_point: numpy array of shape (3,) - reference point for consistent winding
                            (typically bounding plane corner 0). The contour is rolled to start
                            from the vertex closest to this point before resampling.
                            If None, uses contour[0] directly.

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

        # Step 2: Find consistent starting point
        if reference_point is not None:
            # Find vertex closest to reference point (bounding plane corner 0)
            distances = np.linalg.norm(contour - reference_point, axis=1)
            start_idx = np.argmin(distances)
        else:
            # First level: use normalized contour[0] directly (already correct)
            start_idx = 0

        # Roll arrays to start from this index (preserves winding direction)
        if start_idx != 0:
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

    def _align_stream_contours_for_mesh(self, stream_contours):
        """
        Align contours within a stream to minimize twist when building mesh.
        For each contour after the first, find the rotation that minimizes
        total vertex-to-vertex distance with the previous contour.

        Args:
            stream_contours: list of contour arrays for a single stream

        Returns:
            list of aligned contour arrays
        """
        if len(stream_contours) == 0:
            return []

        aligned = [np.array(stream_contours[0])]

        for i in range(1, len(stream_contours)):
            prev_contour = aligned[-1]
            curr_contour = np.array(stream_contours[i])

            n_prev = len(prev_contour)
            n_curr = len(curr_contour)

            if n_curr == 0:
                aligned.append(curr_contour)
                continue

            # Find the rotation that minimizes total distance
            best_rotation = 0
            min_total_dist = float('inf')

            # For efficiency, sample a subset of vertices for comparison
            # when contours have many vertices
            if n_prev == n_curr:
                # Same size - compare all corresponding vertices
                for rotation in range(n_curr):
                    total_dist = 0
                    for j in range(n_curr):
                        prev_v = prev_contour[j]
                        curr_v = curr_contour[(j + rotation) % n_curr]
                        total_dist += np.linalg.norm(prev_v - curr_v)
                    if total_dist < min_total_dist:
                        min_total_dist = total_dist
                        best_rotation = rotation
            else:
                # Different sizes - sample at regular intervals
                num_samples = min(n_prev, n_curr, 16)
                prev_samples = [prev_contour[j * n_prev // num_samples] for j in range(num_samples)]

                for rotation in range(n_curr):
                    total_dist = 0
                    for j in range(num_samples):
                        prev_v = prev_samples[j]
                        curr_idx = (j * n_curr // num_samples + rotation) % n_curr
                        curr_v = curr_contour[curr_idx]
                        total_dist += np.linalg.norm(prev_v - curr_v)
                    if total_dist < min_total_dist:
                        min_total_dist = total_dist
                        best_rotation = rotation

            # Apply rotation
            if best_rotation != 0:
                curr_contour = np.roll(curr_contour, -best_rotation, axis=0)

            aligned.append(curr_contour)

        return aligned

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

        # Clear existing mesh to force rebuild
        self.contour_mesh_vertices = None
        self.contour_mesh_faces = None

        print(f"Building contour mesh from {len(self.contours)} streams...")
        num_streams = len(self.contours)
        if num_streams == 0:
            print("No streams found.")
            return

        # Contours are already normalized - use directly without re-alignment
        aligned_streams = []
        for stream_idx, stream_contours in enumerate(self.contours):
            if len(stream_contours) < 2:
                print(f"Stream {stream_idx} has less than 2 contours, skipping.")
                aligned_streams.append([])
                continue
            # First vertex connects to first vertex since contours are normalized
            aligned_streams.append([np.array(c) for c in stream_contours])

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
                    if level_idx == 0:
                        print(f"  Level 0->1 (origin->next): {n_curr} == {n_next} vertices (equal path)")
                    for i in range(n_curr):
                        i_next = (i + 1) % n_curr
                        v0 = curr_indices[i]
                        v1 = curr_indices[i_next]
                        v2 = next_indices[i_next]
                        v3 = next_indices[i]

                        # Choose shorter diagonal to minimize dents
                        p0, p1 = all_vertices[v0], all_vertices[v1]
                        p2, p3 = all_vertices[v2], all_vertices[v3]
                        diag_02 = np.linalg.norm(p0 - p2)
                        diag_13 = np.linalg.norm(p1 - p3)

                        if diag_02 <= diag_13:
                            # Split along v0-v2
                            all_faces.append([v0, v1, v2])
                            all_faces.append([v0, v2, v3])
                        else:
                            # Split along v1-v3
                            all_faces.append([v0, v1, v3])
                            all_faces.append([v1, v2, v3])
                else:
                    # Different sizes - variable band
                    faces = self._create_contour_band_variable_indices(
                        curr_indices, next_indices, all_vertices
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

        # Store original vertices for reference
        self.contour_mesh_vertices_original = self.contour_mesh_vertices.copy()

        # Store vertex-to-level mapping for edge classification
        # vertex_contour_level[vertex_idx] = level_idx
        self.vertex_contour_level = np.full(len(all_vertices), -1, dtype=np.int32)
        for stream_idx in range(num_streams):
            for level_idx in range(num_levels):
                for vertex_idx in stream_level_indices[stream_idx][level_idx]:
                    self.vertex_contour_level[vertex_idx] = level_idx

        # Compute normals
        self._compute_contour_mesh_normals()

        print(f"Built contour mesh: {len(self.contour_mesh_vertices)} vertices, "
              f"{len(self.contour_mesh_faces)} faces from {num_streams} streams")

    # build_contour_to_tet_mapping, compute_tet_edge_contour_types, compute_outward_directions,
    # update_contour_mesh_from_tet methods moved to MuscleMeshMixin in muscle_mesh.py

    # _ensure_waypoints_inside_mesh method moved to FiberArchitectureMixin in fiber_architecture.py
    # _update_contours_from_smoothed_mesh method moved to FiberArchitectureMixin in fiber_architecture.py

    def _create_contour_band_variable_indices(self, curr_indices, next_indices, all_vertices):
        """
        Create triangular faces between two contours with different vertex counts.
        Uses direct vertex indices instead of offsets.
        """
        n_curr = len(curr_indices)
        n_next = len(next_indices)
        faces = []

        print(f"  Variable band: {n_curr} -> {n_next} vertices")

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

    # _align_stream_contours, _find_best_alignment, _compute_alignment_cost_variable
    # methods moved to FiberArchitectureMixin in fiber_architecture.py

    def _find_best_rotation_offset_variable(self, contour_a, contour_b):
        """
        Find rotation offset for contours with different sizes.
        Tests all rotations and returns the best offset.
        """
        n_b = len(contour_b)
        best_offset = 0
        best_cost = float('inf')

        for offset in range(n_b):
            rotated = np.roll(contour_b, -offset, axis=0)
            cost = self._compute_alignment_cost_variable(contour_a, rotated)
            if cost < best_cost:
                best_cost = cost
                best_offset = offset

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
        """Compute vertex normals for the contour mesh using area-weighted averaging."""
        if self.contour_mesh_vertices is None or self.contour_mesh_faces is None:
            return

        num_vertices = len(self.contour_mesh_vertices)
        normals = np.zeros((num_vertices, 3), dtype=np.float32)

        for face in self.contour_mesh_faces:
            v0, v1, v2 = self.contour_mesh_vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            # Cross product magnitude = 2 * triangle area (area weighting)
            face_normal = np.cross(edge1, edge2)
            # Don't normalize - keep area weighting for smoother results

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

    def find_contour_stream_simplified(self, skeleton_meshes=None):
        """
        Create simplified streams with 5 waypoints: origin, 25%, 50%, 75%, insertion.

        For multi-stream muscles, selects contours based on cumulative area to best
        represent the muscle shape at each percentage position.
        """
        if self.contours is None or len(self.contours) < 2:
            print("Need at least 2 contour levels (origin and insertion)")
            return

        num_levels = len(self.contours)
        num_streams = max(len(self.contours[0]), len(self.contours[-1]))

        # Target positions: 0%, 25%, 50%, 75%, 100%
        target_percentages = [0.0, 0.25, 0.5, 0.75, 1.0]

        # Select levels based on contour count (index-based)
        # Level index / (num_levels - 1) gives normalized position
        selected_levels = []
        for target in target_percentages:
            # Find level index closest to target percentage
            target_index = target * (num_levels - 1)
            best_level = round(target_index)
            best_level = max(0, min(best_level, num_levels - 1))
            selected_levels.append(best_level)

        # Ensure first is 0 and last is num_levels-1
        selected_levels[0] = 0
        selected_levels[-1] = num_levels - 1

        # Remove duplicates while preserving order
        unique_levels = []
        for level in selected_levels:
            if level not in unique_levels:
                unique_levels.append(level)
        selected_levels = unique_levels

        print(f"Simplified waypoints: selected levels {selected_levels} from {num_levels} total")

        # Build simplified contours and bounding planes
        self.draw_contour_stream = [True] * num_streams

        # For each stream, collect contours at selected levels
        simplified_contours = []
        simplified_bounding_planes = []

        for level_idx in selected_levels:
            level_contours = self.contours[level_idx]
            level_planes = self.bounding_planes[level_idx]

            if len(level_contours) == 1:
                # Single contour at this level - assign to all streams (merging point)
                simplified_contours.append([level_contours[0]] * num_streams)
                simplified_bounding_planes.append([level_planes[0]] * num_streams)
            else:
                # ALWAYS match by centroid proximity (order might differ between levels!)
                matched_contours = []
                matched_planes = []

                # Use previous level's centroids as reference
                if len(simplified_contours) > 0:
                    ref_centroids = [np.mean(c, axis=0) for c in simplified_contours[-1]]
                else:
                    # First level: use contours in their original order
                    ref_centroids = [np.mean(c, axis=0) for c in level_contours]
                    while len(ref_centroids) < num_streams:
                        ref_centroids.append(ref_centroids[-1])

                for stream_idx in range(num_streams):
                    ref = ref_centroids[stream_idx] if stream_idx < len(ref_centroids) else ref_centroids[-1]

                    # Find closest contour
                    best_idx = 0
                    best_dist = np.inf
                    for c_idx, contour in enumerate(level_contours):
                        centroid = np.mean(contour, axis=0)
                        dist = np.linalg.norm(centroid - ref)
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = c_idx

                    matched_contours.append(level_contours[best_idx])
                    matched_planes.append(level_planes[best_idx])

                simplified_contours.append(matched_contours)
                simplified_bounding_planes.append(matched_planes)

        # Reorganize to per-stream format [stream][level] - use original bounding planes
        # Also filter out contours that are too close together
        min_dist = getattr(self, 'min_contour_distance', 0.005)

        self.contours = []
        self.bounding_planes = []

        for stream_idx in range(num_streams):
            stream_contours = []
            stream_planes = []

            for level_idx in range(len(selected_levels)):
                contour = simplified_contours[level_idx][stream_idx]
                orig_plane = simplified_bounding_planes[level_idx][stream_idx]

                # Check distance from previous contour in this stream
                if len(stream_contours) > 0:
                    prev_mean = stream_planes[-1]['mean']
                    next_mean = orig_plane['mean']
                    dist = np.linalg.norm(next_mean - prev_mean)

                    # Skip if too close to previous contour (but always keep first and last)
                    if dist < min_dist and level_idx != len(selected_levels) - 1:
                        continue

                # Use the original bounding plane directly (don't regenerate)
                stream_contours.append(contour)
                stream_planes.append(orig_plane)

            self.contours.append(stream_contours)
            self.bounding_planes.append(stream_planes)

        print(f"Created {num_streams} streams with {len(self.contours[0]) if self.contours else 0} levels each (after distance filtering)")

    def optimize_contour_stream(self, shape_threshold=0.15, min_contours=3, max_contours=None):
        """
        Optimize contour stream by selecting minimal contours that best represent the mesh shape.

        Uses shape metrics (area, perimeter, aspect ratio) to identify significant shape changes
        and removes redundant contours where neighboring contours are similar.

        Args:
            shape_threshold: Minimum shape change score to keep a contour (0-1, default 0.15)
                            Higher values = fewer contours, lower = more contours
            min_contours: Minimum number of contours to keep per stream (default 3: origin, middle, insertion)
            max_contours: Maximum number of contours per stream (None = no limit)

        Returns:
            Number of contours removed
        """
        if self.contours is None or len(self.contours) == 0:
            print("No contours to optimize. Run find_contour_stream first.")
            return 0

        if self.bounding_planes is None or len(self.bounding_planes) == 0:
            print("No bounding planes found.")
            return 0

        total_removed = 0

        for stream_idx in range(len(self.contours)):
            stream_contours = self.contours[stream_idx]
            stream_planes = self.bounding_planes[stream_idx]

            if len(stream_contours) <= min_contours:
                continue  # Already at minimum

            # Calculate shape metrics for each contour
            metrics = []
            for i, plane in enumerate(stream_planes):
                area = plane.get('area', 0)

                # Calculate perimeter from contour vertices
                if i < len(stream_contours):
                    contour = np.array(stream_contours[i])
                    if len(contour) > 1:
                        perimeter = np.sum(np.linalg.norm(np.diff(contour, axis=0), axis=1))
                        perimeter += np.linalg.norm(contour[-1] - contour[0])  # Close loop
                    else:
                        perimeter = 0
                else:
                    perimeter = 0

                # Aspect ratio from bounding plane
                bp = plane.get('bounding_plane', None)
                if bp is not None and len(bp) >= 4:
                    width = np.linalg.norm(np.array(bp[1]) - np.array(bp[0]))
                    height = np.linalg.norm(np.array(bp[3]) - np.array(bp[0]))
                    aspect = width / (height + 1e-10)
                else:
                    aspect = 1.0

                metrics.append({
                    'area': area,
                    'perimeter': perimeter,
                    'aspect': aspect,
                    'index': i
                })

            # Calculate shape change scores between consecutive contours
            n = len(metrics)
            shape_scores = [1.0]  # First contour always kept (score = 1)

            for i in range(1, n - 1):
                prev_m = metrics[i - 1]
                curr_m = metrics[i]
                next_m = metrics[i + 1]

                # Area change (relative to neighbors)
                avg_neighbor_area = (prev_m['area'] + next_m['area']) / 2
                if avg_neighbor_area > 0:
                    area_change = abs(curr_m['area'] - avg_neighbor_area) / avg_neighbor_area
                else:
                    area_change = 0

                # Perimeter change
                avg_neighbor_perim = (prev_m['perimeter'] + next_m['perimeter']) / 2
                if avg_neighbor_perim > 0:
                    perim_change = abs(curr_m['perimeter'] - avg_neighbor_perim) / avg_neighbor_perim
                else:
                    perim_change = 0

                # Aspect ratio change
                avg_neighbor_aspect = (prev_m['aspect'] + next_m['aspect']) / 2
                aspect_change = abs(curr_m['aspect'] - avg_neighbor_aspect) / (avg_neighbor_aspect + 1e-10)

                # Combined score (weighted)
                # Inflection points where shape changes direction are most important
                score = 0.4 * area_change + 0.3 * perim_change + 0.3 * aspect_change

                # Bonus for inflection points (where shape change reverses direction)
                if i > 1:
                    prev_area_trend = metrics[i-1]['area'] - metrics[i-2]['area']
                    curr_area_trend = curr_m['area'] - prev_m['area']
                    if prev_area_trend * curr_area_trend < 0:  # Sign change = inflection
                        score *= 1.5

                shape_scores.append(score)

            shape_scores.append(1.0)  # Last contour always kept

            # Select contours to keep based on shape scores
            keep_indices = [0, n - 1]  # Always keep first and last

            # Sort middle contours by score (descending)
            middle_scores = [(i, shape_scores[i]) for i in range(1, n - 1)]
            middle_scores.sort(key=lambda x: x[1], reverse=True)

            # Add contours with scores above threshold
            for idx, score in middle_scores:
                if score >= shape_threshold:
                    keep_indices.append(idx)

            # Ensure minimum contours
            while len(keep_indices) < min_contours and len(middle_scores) > 0:
                for idx, score in middle_scores:
                    if idx not in keep_indices:
                        keep_indices.append(idx)
                        break

            # Apply maximum contours limit
            if max_contours is not None and len(keep_indices) > max_contours:
                # Keep first, last, and top (max_contours-2) scoring middle contours
                keep_indices = [0, n - 1]
                added = 0
                for idx, score in middle_scores:
                    if added >= max_contours - 2:
                        break
                    keep_indices.append(idx)
                    added += 1

            keep_indices = sorted(set(keep_indices))

            # Rebuild stream with selected contours
            new_contours = [stream_contours[i] for i in keep_indices]
            new_planes = [stream_planes[i] for i in keep_indices]

            removed = len(stream_contours) - len(new_contours)
            total_removed += removed

            self.contours[stream_idx] = new_contours
            self.bounding_planes[stream_idx] = new_planes

            print(f"  Stream {stream_idx}: {len(stream_contours)} -> {len(new_contours)} contours (removed {removed})")

        print(f"Optimization complete: removed {total_removed} contours total")

        # Update draw_contour_stream if it exists
        if hasattr(self, 'draw_contour_stream') and self.draw_contour_stream is not None:
            self.draw_contour_stream = [True] * len(self.contours)

        return total_removed

    def optimize_contour_stream_fit(self, fit_threshold=0.005, min_contours=3, num_samples=1000):
        """
        Optimize contour stream by measuring actual fit to original mesh surface.
        Greedily removes contours while maintaining fit quality.

        Args:
            fit_threshold: Maximum allowed mean distance from original mesh to contour surface (meters)
            min_contours: Minimum number of contours to keep per stream
            num_samples: Number of points to sample on original mesh surface

        Returns:
            (total_removed, final_fit_error)
        """
        if self.contours is None or len(self.contours) == 0:
            print("No contours to optimize. Run find_contour_stream first.")
            return 0, 0

        if self.vertices is None or self.faces_3 is None:
            print("No original mesh data available.")
            return 0, 0

        # Sample points on original mesh surface
        sample_points = self._sample_mesh_surface(num_samples)
        if len(sample_points) == 0:
            print("Failed to sample mesh surface.")
            return 0, 0

        print(f"Sampled {len(sample_points)} points on original mesh surface")

        total_removed = 0

        for stream_idx in range(len(self.contours)):
            stream_contours = self.contours[stream_idx]
            stream_planes = self.bounding_planes[stream_idx]

            if len(stream_contours) <= min_contours:
                print(f"  Stream {stream_idx}: already at minimum ({len(stream_contours)} contours)")
                continue

            original_count = len(stream_contours)

            # Compute initial fit
            current_fit = self._compute_contour_fit(stream_contours, sample_points)
            print(f"  Stream {stream_idx}: initial fit = {current_fit*1000:.2f}mm ({len(stream_contours)} contours)")

            # Greedy removal
            while len(stream_contours) > min_contours:
                best_removal_idx = None
                best_fit_after = float('inf')

                # Try removing each middle contour (keep first and last)
                for i in range(1, len(stream_contours) - 1):
                    # Create test configuration without contour i
                    test_contours = stream_contours[:i] + stream_contours[i+1:]
                    test_fit = self._compute_contour_fit(test_contours, sample_points)

                    if test_fit < best_fit_after:
                        best_fit_after = test_fit
                        best_removal_idx = i

                # Check if best removal is acceptable
                if best_removal_idx is not None and best_fit_after <= fit_threshold:
                    # Remove the contour
                    stream_contours = stream_contours[:best_removal_idx] + stream_contours[best_removal_idx+1:]
                    stream_planes = stream_planes[:best_removal_idx] + stream_planes[best_removal_idx+1:]
                    current_fit = best_fit_after
                else:
                    # Can't remove any more without exceeding threshold
                    break

            removed = original_count - len(stream_contours)
            total_removed += removed

            self.contours[stream_idx] = stream_contours
            self.bounding_planes[stream_idx] = stream_planes

            print(f"  Stream {stream_idx}: {original_count} -> {len(stream_contours)} contours "
                  f"(removed {removed}, final fit = {current_fit*1000:.2f}mm)")

        print(f"Fit optimization complete: removed {total_removed} contours total")

        # Update draw_contour_stream
        if hasattr(self, 'draw_contour_stream') and self.draw_contour_stream is not None:
            self.draw_contour_stream = [True] * len(self.contours)

        # Compute final overall fit
        final_fit = self._compute_overall_fit(sample_points)
        return total_removed, final_fit

    def _sample_mesh_surface(self, num_samples):
        """Sample random points on the original mesh surface."""
        if self.vertices is None or self.faces_3 is None or len(self.faces_3) == 0:
            return np.array([])

        # Compute face areas for weighted sampling
        faces = np.array(self.faces_3)
        v0 = self.vertices[faces[:, 0]]
        v1 = self.vertices[faces[:, 1]]
        v2 = self.vertices[faces[:, 2]]

        # Cross product gives area * 2
        cross = np.cross(v1 - v0, v2 - v0)
        areas = np.linalg.norm(cross, axis=1) / 2

        # Normalize to probabilities
        total_area = np.sum(areas)
        if total_area < 1e-10:
            return np.array([])
        probs = areas / total_area

        # Sample faces weighted by area
        face_indices = np.random.choice(len(faces), size=num_samples, p=probs)

        # Generate random barycentric coordinates
        r1 = np.random.random(num_samples)
        r2 = np.random.random(num_samples)
        sqrt_r1 = np.sqrt(r1)

        # Barycentric coords that give uniform distribution on triangle
        u = 1 - sqrt_r1
        v = sqrt_r1 * (1 - r2)
        w = sqrt_r1 * r2

        # Compute sample points
        sampled_v0 = self.vertices[faces[face_indices, 0]]
        sampled_v1 = self.vertices[faces[face_indices, 1]]
        sampled_v2 = self.vertices[faces[face_indices, 2]]

        samples = u[:, np.newaxis] * sampled_v0 + v[:, np.newaxis] * sampled_v1 + w[:, np.newaxis] * sampled_v2
        return samples

    def _compute_contour_fit(self, stream_contours, sample_points):
        """
        Compute fit metric: mean distance from sample points to contour surface.
        Uses interpolation between adjacent contours.
        """
        if len(stream_contours) < 2:
            return float('inf')

        # For each sample point, find distance to nearest point on contour surface
        distances = []

        for pt in sample_points:
            min_dist = float('inf')

            # Check distance to each contour level
            for i, contour in enumerate(stream_contours):
                contour = np.array(contour)
                if len(contour) < 3:
                    continue

                # Distance to contour polygon (approximate as distance to nearest edge)
                for j in range(len(contour)):
                    p1 = contour[j]
                    p2 = contour[(j + 1) % len(contour)]

                    # Point to line segment distance
                    d = self._point_to_segment_distance(pt, p1, p2)
                    min_dist = min(min_dist, d)

            # Also check interpolated surfaces between contours
            for i in range(len(stream_contours) - 1):
                c1 = np.array(stream_contours[i])
                c2 = np.array(stream_contours[i + 1])

                if len(c1) != len(c2) or len(c1) < 3:
                    continue

                # Check distance to quad faces connecting contours
                for j in range(len(c1)):
                    j_next = (j + 1) % len(c1)
                    # Quad: c1[j], c1[j_next], c2[j_next], c2[j]
                    # Split into two triangles
                    d1 = self._point_to_triangle_distance(pt, c1[j], c1[j_next], c2[j])
                    d2 = self._point_to_triangle_distance(pt, c1[j_next], c2[j_next], c2[j])
                    min_dist = min(min_dist, d1, d2)

            if min_dist < float('inf'):
                distances.append(min_dist)

        if len(distances) == 0:
            return float('inf')

        return np.mean(distances)

    def _point_to_segment_distance(self, p, a, b):
        """Compute distance from point p to line segment ab."""
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-10)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    def _point_to_triangle_distance(self, p, a, b, c):
        """Compute distance from point p to triangle abc."""
        # Project point onto triangle plane
        ab = b - a
        ac = c - a
        normal = np.cross(ab, ac)
        normal_len = np.linalg.norm(normal)
        if normal_len < 1e-10:
            return float('inf')
        normal = normal / normal_len

        # Distance to plane
        ap = p - a
        plane_dist = np.dot(ap, normal)
        projected = p - plane_dist * normal

        # Check if projected point is inside triangle (barycentric)
        v0 = c - a
        v1 = b - a
        v2 = projected - a

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-10:
            return float('inf')

        u = (dot11 * dot02 - dot01 * dot12) / denom
        v = (dot00 * dot12 - dot01 * dot02) / denom

        if u >= 0 and v >= 0 and u + v <= 1:
            # Inside triangle
            return abs(plane_dist)
        else:
            # Outside - find closest edge
            d1 = self._point_to_segment_distance(p, a, b)
            d2 = self._point_to_segment_distance(p, b, c)
            d3 = self._point_to_segment_distance(p, c, a)
            return min(d1, d2, d3)

    def _compute_overall_fit(self, sample_points):
        """Compute overall fit across all streams."""
        if self.contours is None or len(self.contours) == 0:
            return float('inf')

        all_contours = []
        for stream in self.contours:
            all_contours.extend(stream)

        if len(all_contours) < 2:
            return float('inf')

        # Flatten: compute mean distance from samples to any contour surface
        distances = []
        for pt in sample_points:
            min_dist = float('inf')
            for contour in all_contours:
                contour = np.array(contour)
                if len(contour) < 3:
                    continue
                for j in range(len(contour)):
                    p1 = contour[j]
                    p2 = contour[(j + 1) % len(contour)]
                    d = self._point_to_segment_distance(pt, p1, p2)
                    min_dist = min(min_dist, d)
            if min_dist < float('inf'):
                distances.append(min_dist)

        return np.mean(distances) if distances else float('inf')

    def _find_contour_stream_post_process(self, skeleton_meshes=None):
        """Post-processing for find_contour_stream: fiber architecture, waypoints, etc."""
        # Contours are already normalized - don't reverse or re-align
        # First vertex of each contour connects to first vertex of next contour

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

        self.normalized_Qs = [[] for _ in range(len(self.bounding_planes))]
        self.waypoints = [[] for _ in range(len(self.bounding_planes))]
        self.mvc_weights = [[] for _ in range(len(self.bounding_planes))]  # MVC weights per fiber sample
        self.attach_skeletons = [[0, 0] for _ in range(len(self.waypoints))]
        self.attach_skeletons_sub = [[0, 0] for _ in range(len(self.waypoints))]
        # Store stream endpoint positions for auto-detection
        self._stream_endpoints = []  # [(origin_pos, insertion_pos), ...]

        # Get fiber positioning method
        positioning_method = getattr(self, 'fiber_positioning_method', 'mvc')
        print(f"[find_contour_stream_impl] positioning_method = {positioning_method}")

        # Initialize triangulation storage for triangulated/harmonic/geodesic methods
        if positioning_method in ('triangulated', 'harmonic', 'geodesic'):
            # Store triangulations and embeddings per stream, per level
            self.unit_circle_triangulations = [[] for _ in range(len(self.bounding_planes))]
            self.fiber_embeddings = [[] for _ in range(len(self.bounding_planes))]
            self.triangulated_deformed_2d = [[] for _ in range(len(self.bounding_planes))]

        # Get geodesic info (shared across all contours)
        num_geo = 0
        geodesic_paths = None
        mesh_verts = None
        if hasattr(self, '_geodesic_reference_paths') and self._geodesic_reference_paths:
            geodesic_paths = self._geodesic_reference_paths
            num_geo = len(geodesic_paths)
        if hasattr(self, 'vertices'):
            mesh_verts = self.vertices

        # For geodesic mode: create shared triangulation ONCE (same boundary count for all contours)
        shared_geo_triangulation = None
        shared_geo_embedding = None
        if positioning_method == 'geodesic' and num_geo >= 3:
            # Use first stream's fiber samples (all streams have same fiber architecture)
            first_fiber_samples = self.fiber_architecture[0] if len(self.fiber_architecture) > 0 else None
            if first_fiber_samples is not None:
                n_rings = getattr(self, 'n_interior_rings', 3)
                shared_geo_triangulation, shared_geo_embedding = create_shared_geodesic_triangulation(
                    num_geo, first_fiber_samples, n_rings
                )

        for i, bounding_plane_stream in enumerate(self.bounding_planes):
            fiber_samples = self.fiber_architecture[i]

            for j, bounding_plane in enumerate(bounding_plane_stream):
                # Ensure contour_vertices is set from actual contour
                if 'contour_vertices' not in bounding_plane and i < len(self.contours) and j < len(self.contours[i]):
                    bounding_plane['contour_vertices'] = np.array(self.contours[i][j])

                if positioning_method == 'radial':
                    # Radial interpolation: guaranteed interior for star-shaped contours
                    normalized_Qs, waypoints = self.find_waypoints_radial(bounding_plane, fiber_samples)
                    # Also compute MVC weights for visualization
                    _, _, mvc_weights = self.find_waypoints(bounding_plane, fiber_samples)

                elif positioning_method == 'triangulated':
                    # Triangulation + ARAP: preserves topology, prevents triangle flips
                    # Create triangulation for THIS contour (each contour has different vertex count)
                    deformed_2d = None
                    triangulation = None
                    embedding = None

                    contour_verts = bounding_plane.get('contour_vertices')
                    if contour_verts is not None and len(contour_verts) >= 3:
                        n_boundary = len(contour_verts)

                        # Compute geodesic indices and unit circle angles for this contour
                        geodesic_indices = find_geodesic_vertex_indices(contour_verts, num_geo, geodesic_paths, mesh_verts) if num_geo > 0 else []
                        uc_angles = create_angular_unit_circle_vertices(contour_verts, geodesic_indices)

                        # Create triangulation for this contour
                        n_rings = getattr(self, 'n_interior_rings', 3)
                        triangulation = create_unit_circle_triangulation(n_boundary, uc_angles, n_rings)

                        # Embed fibers in this triangulation
                        embedding = embed_fibers_in_triangulation(fiber_samples, triangulation)

                        # Store for visualization
                        self.unit_circle_triangulations[i].append(triangulation)
                        self.fiber_embeddings[i].append(embedding)

                        # Compute waypoints using ARAP
                        deformed_2d, waypoints = find_waypoints_triangulated(
                            bounding_plane, fiber_samples, triangulation, embedding
                        )
                        if waypoints is None or len(waypoints) == 0:
                            # Fallback to MVC if triangulated method fails
                            normalized_Qs, waypoints, mvc_weights = self.find_waypoints(bounding_plane, fiber_samples)
                            deformed_2d = None
                        else:
                            # Compute normalized_Qs and MVC weights for compatibility/visualization
                            normalized_Qs, _, mvc_weights = self.find_waypoints(bounding_plane, fiber_samples)
                    else:
                        # Fallback to MVC if contour not available
                        normalized_Qs, waypoints, mvc_weights = self.find_waypoints(bounding_plane, fiber_samples)
                        self.unit_circle_triangulations[i].append(None)
                        self.fiber_embeddings[i].append(None)

                    # Store deformed 2D for visualization
                    self.triangulated_deformed_2d[i].append(deformed_2d)

                elif positioning_method == 'harmonic':
                    # Direct harmonic interpolation: fibers ARE interior vertices
                    # Simpler than 'triangulated' - no extra grid points
                    deformed_2d = None
                    triangulation = None

                    contour_verts = bounding_plane.get('contour_vertices')
                    if contour_verts is not None and len(contour_verts) >= 3:
                        n_boundary = len(contour_verts)

                        # Compute geodesic indices and unit circle angles
                        geodesic_indices = find_geodesic_vertex_indices(contour_verts, num_geo, geodesic_paths, mesh_verts) if num_geo > 0 else []
                        uc_angles = create_angular_unit_circle_vertices(contour_verts, geodesic_indices)

                        # Create direct triangulation (boundary + fibers only)
                        triangulation = create_direct_fiber_triangulation(n_boundary, uc_angles, fiber_samples)

                        # Store for visualization
                        self.unit_circle_triangulations[i].append(triangulation)
                        self.fiber_embeddings[i].append(None)  # No separate embedding needed

                        # Compute waypoints directly via harmonic interpolation
                        deformed_2d, waypoints = find_waypoints_harmonic_direct(
                            bounding_plane, fiber_samples, triangulation
                        )
                        if waypoints is None or len(waypoints) == 0:
                            normalized_Qs, waypoints, mvc_weights = self.find_waypoints(bounding_plane, fiber_samples)
                            deformed_2d = None
                        else:
                            normalized_Qs, _, mvc_weights = self.find_waypoints(bounding_plane, fiber_samples)
                    else:
                        normalized_Qs, waypoints, mvc_weights = self.find_waypoints(bounding_plane, fiber_samples)
                        self.unit_circle_triangulations[i].append(None)
                        self.fiber_embeddings[i].append(None)

                    self.triangulated_deformed_2d[i].append(deformed_2d)

                elif positioning_method == 'geodesic':
                    # Geodesic mode: uses SHARED triangulation with geodesic crossing vertices as boundary
                    # Same triangulation structure for ALL contours (same num_geo boundary vertices)
                    deformed_2d = None

                    contour_verts = bounding_plane.get('contour_vertices')
                    if (contour_verts is not None and len(contour_verts) >= 3 and
                        shared_geo_triangulation is not None and shared_geo_embedding is not None):
                        # Compute waypoints using shared geodesic triangulation
                        deformed_2d, waypoints = find_waypoints_geodesic_shared(
                            bounding_plane, fiber_samples, shared_geo_triangulation, shared_geo_embedding,
                            geodesic_paths, mesh_verts
                        )
                        if waypoints is None or len(waypoints) == 0:
                            # Fallback to MVC if geodesic method fails
                            normalized_Qs, waypoints, mvc_weights = self.find_waypoints(bounding_plane, fiber_samples)
                            deformed_2d = None
                        else:
                            # Compute normalized_Qs and MVC weights for compatibility/visualization
                            normalized_Qs, _, mvc_weights = self.find_waypoints(bounding_plane, fiber_samples)

                        # Store shared triangulation for visualization (same reference for all)
                        self.unit_circle_triangulations[i].append(shared_geo_triangulation)
                        self.fiber_embeddings[i].append(shared_geo_embedding)
                    else:
                        normalized_Qs, waypoints, mvc_weights = self.find_waypoints(bounding_plane, fiber_samples)
                        self.unit_circle_triangulations[i].append(None)
                        self.fiber_embeddings[i].append(None)

                    self.triangulated_deformed_2d[i].append(deformed_2d)

                else:  # 'mvc' (default)
                    normalized_Qs, waypoints, mvc_weights = self.find_waypoints(bounding_plane, fiber_samples)

                self.normalized_Qs[i].append(normalized_Qs)
                self.waypoints[i].append(waypoints)
                self.mvc_weights[i].append(mvc_weights)

        self.is_draw = False
        self.is_draw_fiber_architecture = True

        # Auto-detect skeleton attachments if skeleton_meshes provided
        if skeleton_meshes is not None and len(skeleton_meshes) > 0:
            print(f"[_find_contour_stream_post_process] Calling auto_detect_attachments with {len(skeleton_meshes)} skeletons")
            print(f"  contours: {len(self.contours) if self.contours else 'None'} streams")
            if self.contours and len(self.contours) > 0:
                print(f"  contours[0]: {len(self.contours[0])} levels")
            result = self.auto_detect_attachments(skeleton_meshes)
            print(f"  auto_detect_attachments returned: {result}")
        else:
            print(f"[_find_contour_stream_post_process] Skipping auto_detect_attachments: skeleton_meshes={skeleton_meshes is not None}, len={len(skeleton_meshes) if skeleton_meshes else 0}")

    def find_contour_stream(self, skeleton_meshes=None, optimize_fit=False, fit_threshold=0.005, min_contours=3):
        # Check for simplified mode - prepare data and skip to post-processing
        if getattr(self, 'simplified_waypoints', False):
            self.find_contour_stream_simplified(skeleton_meshes)
            # Optimize fit if enabled
            if optimize_fit:
                self.optimize_contour_stream_fit(fit_threshold=fit_threshold, min_contours=min_contours)
            # Continue with post-processing below (fiber architecture, waypoints, etc.)
            return self._find_contour_stream_post_process(skeleton_meshes)

        # Validate input data
        if self.bounding_planes is None or len(self.bounding_planes) == 0:
            print("Error: No bounding planes found. Run find_contours first.")
            return
        if self.contours is None or len(self.contours) == 0:
            print("Error: No contours found. Run find_contours first.")
            return

        print(f"Starting find_contour_stream: {len(self.bounding_planes)} levels, {len(self.contours)} contour groups")

        try:
            origin_num = len(self.bounding_planes[0])
            insertion_num = len(self.bounding_planes[-1])
            print(f"  Origins: {origin_num}, Insertions: {insertion_num}")
        except Exception as e:
            print(f"Error accessing bounding_planes: {e}")
            return

        # Minimum distance threshold for contours in a stream
        min_dist = getattr(self, 'min_contour_distance', 0.005)

        # Stream Search
        contour_stream_match = []
        check_count = 0

        # Procedure will start from origin or insertion where stream number is larger
        if origin_num >= insertion_num:
            self.draw_contour_stream = [True] * origin_num
            # Use resampled contours from self.contours[0], not raw edge_group vertices
            ordered_contours_trim = [[contour] for contour in self.contours[0]]
            ordered_contours_trim_orig = [[contour.copy()] for contour in self.contours[0]]
            bounding_planes_trim = [[bounding_plane] for bounding_plane in self.bounding_planes[0]]
            contour_stream_match.append([[i] for i in range(origin_num)])

            contour_bounding_zip = zip (self.contours[1:], self.bounding_planes[1:])

        else:
            self.draw_contour_stream = [True] * insertion_num
            # Use resampled contours from self.contours[-1], not raw edge_group vertices
            ordered_contours_trim = [[contour] for contour in self.contours[-1]]
            ordered_contours_trim_orig = [[contour.copy()] for contour in self.contours[-1]]
            bounding_planes_trim = [[bounding_plane] for bounding_plane in self.bounding_planes[-1]]
            contour_stream_match.append([[i] for i in range(insertion_num)])

            contour_bounding_zip = zip(self.contours[-2::-1], self.bounding_planes[-2::-1])

        # Initialize cumulative area contributions for each stream
        # Each stream tracks a dict: {original_stream_index: area_contribution}
        # This allows proper splitting when streams merge and then split again
        cumulative_area_contributions = []
        initial_areas = [bp[0]['area'] for bp in bounding_planes_trim]
        total_initial_area = sum(initial_areas)
        for stream_i in range(len(bounding_planes_trim)):
            # Normalize so all contributions sum to 1.0
            cumulative_area_contributions.append({stream_i: initial_areas[stream_i] / total_initial_area})

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

                    # Skip if no next means to compare
                    if len(next_means) == 0:
                        continue

                    projected_means = [mean - np.dot(mean - next_mean, basis_z) * basis_z for next_mean, basis_z in zip(next_means, next_basis_z)]
                    diff_list = [mean - projected_mean for mean, projected_mean in zip(next_means, projected_means)]
                    if len(diff_list) == 0:
                        continue
                    distances = np.linalg.norm(diff_list, axis=1)

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

                    # Check distance from previous contour in this stream
                    prev_mean = bounding_planes_trim[i][-1]['mean']
                    next_mean = next_bounding_planes[closest]['mean']
                    dist = np.linalg.norm(next_mean - prev_mean)

                    # Skip if too close to previous contour
                    if dist < min_dist:
                        continue

                    ordered_contours_trim[i].append(next_contours[closest])
                    ordered_contours_trim_orig[i].append(next_contours[closest].copy())
                    # Use original bounding plane directly (don't regenerate)
                    bounding_planes_trim[i].append(next_bounding_planes[closest])
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
                        elif self.cutting_method == 'ratio':
                            # Ratio-based cutting with interpolated area ratios
                            # Calculate structure vector to determine spatial order
                            structure_vector = np.array([0.0, 0.0, 0.0])
                            for mean_i in range(len(projected_target_current_contours_means)):
                                for mean_j in range(mean_i + 1, len(projected_target_current_contours_means)):
                                    cand_vector = projected_target_current_contours_means[mean_i] - projected_target_current_contours_means[mean_j]
                                    if np.dot(cand_vector, np.array([1, 0, 0])) < 0:
                                        cand_vector *= -1
                                    structure_vector += np.array(cand_vector)
                            structure_vector /= len(projected_target_current_contours_means)
                            structure_vector /= np.linalg.norm(structure_vector) + 1e-10

                            # Get spatial order of streams
                            target_values = [np.dot(mean - target_next_mean, structure_vector) for mean in projected_target_current_contours_means]
                            target_order = list(np.argsort(target_values))

                            # Calculate interpolated ratios
                            if not division_start:
                                division_start = True
                                division_start_index = zip_index

                            # Area-based ratios
                            sorted_areas1 = [target_current_contours_areas[index] for index in target_order]
                            total_area1 = sum(sorted_areas1)
                            ratios1 = [a / total_area1 for a in sorted_areas1]

                            # Equal-ish ratios (edge streams = 1, middle = 1.5)
                            num_s = len(target_order)
                            weights2 = [1.0 if (i == 0 or i == num_s - 1) else 1.5 for i in range(num_s)]
                            total_w2 = sum(weights2)
                            ratios2 = [w / total_w2 for w in weights2]

                            # Interpolate based on progress
                            progress = zip_index / zip_num
                            target_ratios = [r1 * (1 - progress) + r2 * progress for r1, r2 in zip(ratios1, ratios2)]

                            new_target_next_contours = self._cut_contour_ratio(
                                target_next_contour,
                                target_next_contour_match,
                                projected_target_current_contours_means,
                                target_next_basis_z,
                                target_ratios,
                                target_order
                            )
                        elif self.cutting_method == 'cumulative_area':
                            # Cumulative area-based cutting with endpoint interpolation
                            # Uses tracked cumulative area contributions from previous merges
                            # Based on area_based vertex assignment logic
                            structure_vector = np.array([0.0, 0.0, 0.0])
                            for mean_i in range(len(projected_target_current_contours_means)):
                                for mean_j in range(mean_i + 1, len(projected_target_current_contours_means)):
                                    cand_vector = projected_target_current_contours_means[mean_i] - projected_target_current_contours_means[mean_j]
                                    if np.dot(cand_vector, np.array([1, 0, 0])) < 0:
                                        cand_vector *= -1
                                    structure_vector += np.array(cand_vector)
                            structure_vector /= len(projected_target_current_contours_means)
                            structure_vector /= np.linalg.norm(structure_vector) + 1e-10

                            horizontal_vector = v1 - v0
                            vertical_vector = v3 - v0
                            horizontal_vector /= np.linalg.norm(horizontal_vector)
                            vertical_vector /= np.linalg.norm(vertical_vector)

                            horizontal_projection = np.abs(np.dot(structure_vector, horizontal_vector))
                            vertical_projection = np.abs(np.dot(structure_vector, vertical_vector))

                            if horizontal_projection > vertical_projection:
                                target_values = [np.dot(mean - target_next_mean, horizontal_vector) for mean in projected_target_current_contours_means]
                                target_axis = 0
                            else:
                                target_values = [np.dot(mean - target_next_mean, vertical_vector) for mean in projected_target_current_contours_means]
                                target_axis = 1
                            target_order = np.argsort(target_values)

                            if not division_start:
                                division_start = True
                                division_start_index = zip_index

                            # Get cumulative area contributions for merging streams
                            merged_contributions = {}
                            for j in target_current_contours_indices:
                                for orig_stream, contrib in cumulative_area_contributions[j].items():
                                    if orig_stream in merged_contributions:
                                        merged_contributions[orig_stream] += contrib
                                    else:
                                        merged_contributions[orig_stream] = contrib

                            # Calculate accumul_areas using cumulative contributions
                            sorted_contribs = [sum(cumulative_area_contributions[target_current_contours_indices[idx]].values())
                                               for idx in target_order]
                            accumul_areas1 = np.cumsum(sorted_contribs, dtype=float)
                            accumul_areas1 /= accumul_areas1[-1]

                            # Even distribution for endpoint
                            sorted_areas2 = []
                            mul_num = 1.5
                            for area_i in range(len(target_order)):
                                if area_i == 0 or area_i == len(target_order) - 1:
                                    sorted_areas2.append(1)
                                else:
                                    sorted_areas2.append(mul_num)
                            accumul_areas2 = np.cumsum(sorted_areas2, dtype=float)
                            accumul_areas2 /= accumul_areas2[-1]

                            # Interpolate based on progress
                            ratio2 = zip_index / zip_num
                            ratio1 = 1 - ratio2
                            accumul_areas = accumul_areas1 * ratio1 + accumul_areas2 * ratio2

                            # Vertex assignment using normalized_Qs (same as area_based)
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
                                        inserted_index = target_order[index]
                                        if prev_inserted_index is not None and prev_inserted_index != inserted_index:
                                            prev_P = Ps[Q_index - 1]
                                            P = Ps[Q_index]

                                            idx1 = np.where(target_order == prev_inserted_index)[0][0]
                                            idx2 = np.where(target_order == inserted_index)[0][0]

                                            if idx1 <= idx2:
                                                path1 = list(range(idx1, idx2 + 1))
                                                path2 = np.concatenate([range(idx1, -1, -1), range(len(target_order) - 1, idx2 - 1, -1)])
                                            else:
                                                path1 = list(range(idx1, idx2 - 1, -1))
                                                path2 = np.concatenate([range(idx1, len(target_order)), range(0, idx2 + 1)])

                                            wanted_order = path1
                                            temp_areas = accumul_areas[wanted_order]
                                            if temp_areas[0] < temp_areas[-1]:
                                                for i in range(len(temp_areas) - 1):
                                                    if temp_areas[i] > temp_areas[i + 1]:
                                                        wanted_order = path2
                                                        break
                                            else:
                                                for i in range(len(temp_areas) - 1):
                                                    if temp_areas[i] < temp_areas[i + 1]:
                                                        wanted_order = path2
                                                        break

                                            if prev_target_value < target_value:
                                                dividing_values = np.concatenate([[prev_target_value], accumul_areas[wanted_order[:-1]], [target_value]])
                                            else:
                                                dividing_values = np.concatenate([[prev_target_value], accumul_areas[wanted_order[1:]], [target_value]])

                                            dividing_order = target_order[wanted_order]

                                            for dividing_index in range(len(dividing_order) - 1):
                                                weight_P = (dividing_values[0] - dividing_values[dividing_index + 1]) / (dividing_values[0] - dividing_values[-1])
                                                weight_prev_P = 1 - weight_P

                                                mid_P = weight_P * P + weight_prev_P * prev_P
                                                new_target_next_contours[dividing_order[dividing_index]].append(mid_P)
                                                new_target_next_contours[dividing_order[dividing_index + 1]].append(mid_P)

                                        new_target_next_contours[inserted_index].append(Ps[Q_index])
                                        prev_inserted_index = inserted_index
                                        prev_target_value = target_value
                                        break

                            # Update cumulative area contributions for merged streams
                            for j in target_current_contours_indices:
                                cumulative_area_contributions[j] = merged_contributions.copy()
                        elif self.cutting_method == 'projected_area':
                            # Projected area-based cutting
                            # Project each current contour onto the next bounding plane and use projected areas
                            # Based on area_based vertex assignment logic
                            target_next_basis_x = target_next_bounding_plane_info['basis_x']
                            target_next_basis_y = target_next_bounding_plane_info['basis_y']

                            # Calculate projected areas for each merging stream
                            projected_areas = []
                            for j in target_current_contours_indices:
                                current_contour = ordered_contours_trim[j][-1]
                                # Project contour points onto the next bounding plane
                                projected_points = []
                                for point in current_contour:
                                    projected_point = point - np.dot(point - target_next_mean, target_next_basis_z) * target_next_basis_z
                                    projected_points.append(projected_point)
                                projected_points = np.array(projected_points)

                                # Calculate area in 2D plane coordinates
                                projected_2d = np.array([
                                    [np.dot(p - target_next_mean, target_next_basis_x),
                                     np.dot(p - target_next_mean, target_next_basis_y)]
                                    for p in projected_points
                                ])
                                projected_area = compute_polygon_area(projected_2d)
                                projected_areas.append(abs(projected_area))

                            # Structure vector for spatial ordering
                            structure_vector = np.array([0.0, 0.0, 0.0])
                            for mean_i in range(len(projected_target_current_contours_means)):
                                for mean_j in range(mean_i + 1, len(projected_target_current_contours_means)):
                                    cand_vector = projected_target_current_contours_means[mean_i] - projected_target_current_contours_means[mean_j]
                                    if np.dot(cand_vector, np.array([1, 0, 0])) < 0:
                                        cand_vector *= -1
                                    structure_vector += np.array(cand_vector)
                            structure_vector /= len(projected_target_current_contours_means)
                            structure_vector /= np.linalg.norm(structure_vector) + 1e-10

                            horizontal_vector = v1 - v0
                            vertical_vector = v3 - v0
                            horizontal_vector /= np.linalg.norm(horizontal_vector)
                            vertical_vector /= np.linalg.norm(vertical_vector)

                            horizontal_projection = np.abs(np.dot(structure_vector, horizontal_vector))
                            vertical_projection = np.abs(np.dot(structure_vector, vertical_vector))

                            if horizontal_projection > vertical_projection:
                                target_values = [np.dot(mean - target_next_mean, horizontal_vector) for mean in projected_target_current_contours_means]
                                target_axis = 0
                            else:
                                target_values = [np.dot(mean - target_next_mean, vertical_vector) for mean in projected_target_current_contours_means]
                                target_axis = 1
                            target_order = np.argsort(target_values)

                            if not division_start:
                                division_start = True
                                division_start_index = zip_index

                            # Combine contributions from merging streams
                            merged_contributions = {}
                            for j in target_current_contours_indices:
                                for orig_stream, contrib in cumulative_area_contributions[j].items():
                                    if orig_stream in merged_contributions:
                                        merged_contributions[orig_stream] += contrib
                                    else:
                                        merged_contributions[orig_stream] = contrib

                            # Calculate accumul_areas using projected areas
                            sorted_proj_areas = [projected_areas[idx] for idx in target_order]
                            accumul_areas1 = np.cumsum(sorted_proj_areas, dtype=float)
                            accumul_areas1 /= accumul_areas1[-1]

                            # Even distribution for endpoint
                            sorted_areas2 = []
                            mul_num = 1.5
                            for area_i in range(len(target_order)):
                                if area_i == 0 or area_i == len(target_order) - 1:
                                    sorted_areas2.append(1)
                                else:
                                    sorted_areas2.append(mul_num)
                            accumul_areas2 = np.cumsum(sorted_areas2, dtype=float)
                            accumul_areas2 /= accumul_areas2[-1]

                            # Interpolate based on progress
                            ratio2 = zip_index / zip_num
                            ratio1 = 1 - ratio2
                            accumul_areas = accumul_areas1 * ratio1 + accumul_areas2 * ratio2

                            # Vertex assignment using normalized_Qs (same as area_based)
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
                                        inserted_index = target_order[index]
                                        if prev_inserted_index is not None and prev_inserted_index != inserted_index:
                                            prev_P = Ps[Q_index - 1]
                                            P = Ps[Q_index]

                                            idx1 = np.where(target_order == prev_inserted_index)[0][0]
                                            idx2 = np.where(target_order == inserted_index)[0][0]

                                            if idx1 <= idx2:
                                                path1 = list(range(idx1, idx2 + 1))
                                                path2 = np.concatenate([range(idx1, -1, -1), range(len(target_order) - 1, idx2 - 1, -1)])
                                            else:
                                                path1 = list(range(idx1, idx2 - 1, -1))
                                                path2 = np.concatenate([range(idx1, len(target_order)), range(0, idx2 + 1)])

                                            wanted_order = path1
                                            temp_areas = accumul_areas[wanted_order]
                                            if temp_areas[0] < temp_areas[-1]:
                                                for i in range(len(temp_areas) - 1):
                                                    if temp_areas[i] > temp_areas[i + 1]:
                                                        wanted_order = path2
                                                        break
                                            else:
                                                for i in range(len(temp_areas) - 1):
                                                    if temp_areas[i] < temp_areas[i + 1]:
                                                        wanted_order = path2
                                                        break

                                            if prev_target_value < target_value:
                                                dividing_values = np.concatenate([[prev_target_value], accumul_areas[wanted_order[:-1]], [target_value]])
                                            else:
                                                dividing_values = np.concatenate([[prev_target_value], accumul_areas[wanted_order[1:]], [target_value]])

                                            dividing_order = target_order[wanted_order]

                                            for dividing_index in range(len(dividing_order) - 1):
                                                weight_P = (dividing_values[0] - dividing_values[dividing_index + 1]) / (dividing_values[0] - dividing_values[-1])
                                                weight_prev_P = 1 - weight_P

                                                mid_P = weight_P * P + weight_prev_P * prev_P
                                                new_target_next_contours[dividing_order[dividing_index]].append(mid_P)
                                                new_target_next_contours[dividing_order[dividing_index + 1]].append(mid_P)

                                        new_target_next_contours[inserted_index].append(Ps[Q_index])
                                        prev_inserted_index = inserted_index
                                        prev_target_value = target_value
                                        break

                            # Update cumulative contributions for merged streams
                            for j in target_current_contours_indices:
                                cumulative_area_contributions[j] = merged_contributions.copy()
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
                                # Check distance from previous contour in this stream
                                prev_mean = bounding_planes_trim[j][-1]['mean']
                                next_mean = np.mean(new_target_next_contours[index], axis=0)
                                dist = np.linalg.norm(next_mean - prev_mean)

                                # Skip if too close to previous contour
                                if dist < min_dist:
                                    continue

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

                        # Check distance from previous contour in this stream
                        prev_mean = bounding_planes_trim[target_current_contours_index][-1]['mean']
                        next_mean = next_bounding_planes[i]['mean']
                        dist = np.linalg.norm(next_mean - prev_mean)

                        # Skip if too close to previous contour
                        if dist < min_dist:
                            continue

                        ordered_contours_trim[target_current_contours_index].append(next_contours[i])
                        ordered_contours_trim_orig[target_current_contours_index].append(next_contours[i].copy())
                        # Use original bounding plane directly (don't regenerate)
                        bounding_planes_trim[target_current_contours_index].append(next_bounding_planes[i])

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

                        self.contours[stream_i][i], contour_match = self.find_contour_match(ordered_contours_trim_orig[stream_i][i], bounding_plane, preserve_order=True)

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

                    self.contours[stream_i][i], contour_match = self.find_contour_match(ordered_contours_trim_orig[stream_i][i], bounding_plane, preserve_order=True)

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

        # Error-based contour selection: keep contours that are critical for shape reconstruction
        stream_num = len(self.contours)
        level_num = len(self.bounding_planes[0]) if self.bounding_planes else 0

        print(f"After stream search: {stream_num} streams, {level_num} levels per stream")

        if level_num < 3:
            # Not enough contours to simplify, proceed to post-processing
            print(f"Only {level_num} contours, skipping simplification")
        else:
            # Error threshold: maximum allowed reconstruction error (as fraction of muscle length)
            muscle_length = np.linalg.norm(
                self.bounding_planes[0][-1]['mean'] - self.bounding_planes[0][0]['mean']
            )
            # Default 1% of muscle length - keep more contours for accurate shape
            error_threshold = getattr(self, 'contour_error_threshold', 0.01) * muscle_length

            def compute_reconstruction_error(contour_actual, bp_actual, bp_prev, bp_next, t):
                """
                Compute reconstruction error combining:
                1. Centroid deviation from linear interpolation
                2. Area deviation from linear interpolation
                """
                # Centroid interpolation error
                centroid_actual = np.mean(contour_actual, axis=0)
                centroid_prev = bp_prev['mean']
                centroid_next = bp_next['mean']
                centroid_interpolated = (1 - t) * centroid_prev + t * centroid_next
                centroid_error = np.linalg.norm(centroid_actual - centroid_interpolated)

                # Area deviation error (normalized by expected area)
                area_actual = bp_actual['area']
                area_prev = bp_prev['area']
                area_next = bp_next['area']
                area_interpolated = (1 - t) * area_prev + t * area_next

                # Convert area error to distance scale (sqrt of area ~ length)
                area_error = abs(np.sqrt(area_actual) - np.sqrt(area_interpolated))

                # Combine errors (use max to ensure both are satisfied)
                return max(centroid_error, area_error)

            # Greedy selection: start with origin and insertion, add contours with highest error
            selected_indices = [0, level_num - 1]  # Always include first and last

            while True:
                max_error = 0
                max_error_idx = -1

                # For each gap between selected contours, find the contour with max error
                selected_sorted = sorted(selected_indices)
                for gap_idx in range(len(selected_sorted) - 1):
                    prev_idx = selected_sorted[gap_idx]
                    next_idx = selected_sorted[gap_idx + 1]

                    if next_idx - prev_idx <= 1:
                        continue  # No intermediate contours

                    # Check each intermediate contour
                    for mid_idx in range(prev_idx + 1, next_idx):
                        # Compute interpolation parameter
                        t = (mid_idx - prev_idx) / (next_idx - prev_idx)

                        # Compute error across all streams
                        total_error = 0
                        for stream_i in range(stream_num):
                            contour_actual = np.array(self.contours[stream_i][mid_idx])
                            bp_prev = self.bounding_planes[stream_i][prev_idx]
                            bp_next = self.bounding_planes[stream_i][next_idx]
                            bp_actual = self.bounding_planes[stream_i][mid_idx]

                            error = compute_reconstruction_error(
                                contour_actual, bp_actual, bp_prev, bp_next, t
                            )
                            total_error = max(total_error, error)  # Use max across streams

                        if total_error > max_error:
                            max_error = total_error
                            max_error_idx = mid_idx

                # If max error is below threshold, we're done
                if max_error <= error_threshold or max_error_idx < 0:
                    break

                # Add the contour with maximum error
                selected_indices.append(max_error_idx)

            # Sort selected indices and extract contours
            selected_indices = sorted(selected_indices)
            print(f"Selected {len(selected_indices)} contours from {level_num} (error threshold: {error_threshold:.4f})")

            contours_trim = [[] for _ in range(stream_num)]
            bounding_planes_trim = [[] for _ in range(stream_num)]

            for idx in selected_indices:
                for stream_i in range(stream_num):
                    contours_trim[stream_i].append(self.contours[stream_i][idx])
                    bounding_planes_trim[stream_i].append(self.bounding_planes[stream_i][idx])

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

        # Optimize fit if enabled - remove redundant contours while maintaining mesh shape fit
        if optimize_fit:
            self.optimize_contour_stream_fit(fit_threshold=fit_threshold, min_contours=min_contours)

        # Post-process: fiber architecture, waypoints, etc.
        self._find_contour_stream_post_process(skeleton_meshes)

