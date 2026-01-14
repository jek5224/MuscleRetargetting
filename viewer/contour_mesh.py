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
    # Validate vertices to ensure proper shape
    vertices = np.asarray(vertices)
    if vertices.ndim == 0 or len(vertices) == 0:
        return np.array([0.0, 0.0, 1.0])  # Default normal
    if vertices.ndim == 1:
        # Flat array or single vertex
        if len(vertices) >= 9:
            vertices = vertices[:len(vertices)//3*3].reshape(-1, 3)
        else:
            return np.array([0.0, 0.0, 1.0])  # Default normal for single vertex
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        return np.array([0.0, 0.0, 1.0])  # Default normal for invalid shape

    normal = np.zeros(3)
    n = len(vertices)

    for i in range(n):
        v_curr = np.asarray(vertices[i]).flatten()[:3]
        v_next = np.asarray(vertices[(i + 1) % n]).flatten()[:3]

        normal[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
        normal[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
        normal[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])

    return normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0.0 else np.array([0.0, 0.0, 1.0])


def compute_best_fitting_plane(vertices):
    """Finds an optimal coordinate system using PCA."""
    # Validate vertices
    vertices = np.asarray(vertices)
    if vertices.ndim == 0 or len(vertices) == 0:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0])
    if vertices.ndim == 1:
        if len(vertices) >= 9:
            vertices = vertices[:len(vertices)//3*3].reshape(-1, 3)
        elif len(vertices) >= 3:
            return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), vertices[:3]
        else:
            return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0])
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0])

    # Ensure we have at least 3 vertices for SVD
    if len(vertices) < 3:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.mean(vertices, axis=0)

    mean = np.mean(vertices, axis=0)
    centered = vertices - mean
    try:
        _, _, eigenvectors = np.linalg.svd(centered)
        basis_x = eigenvectors[0]
        basis_y = eigenvectors[1]
        basis_x /= np.linalg.norm(basis_x) + 1e-10
        basis_y /= np.linalg.norm(basis_y) + 1e-10
    except:
        basis_x = np.array([1.0, 0.0, 0.0])
        basis_y = np.array([0.0, 1.0, 0.0])

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

            # Ensure basis_x is always the long direction (width >= height)
            if height > width:
                # Swap so that basis_x is the long side
                basis_x, basis_y = basis_y, -basis_x
                width, height = height, width
                # Recompute corners with swapped basis
                proj_x = points_2d @ basis_x
                proj_y = points_2d @ basis_y
                min_x, max_x = proj_x.min(), proj_x.max()
                min_y, max_y = proj_y.min(), proj_y.max()
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

        # Manual cutting state
        self._manual_cut_pending = False  # True when waiting for user to draw cutting line
        self._manual_cut_data = None      # Dict with all data needed for manual cutting
        self._manual_cut_line = None      # ((x1, y1), (x2, y2)) - user-drawn cutting line in 2D

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
                    v_arr = np.asarray(v).flatten()
                    if len(v_arr) >= 3:
                        glVertex3fv(v_arr[:3])
                glEnd()

            glBegin(GL_POINTS)
            for contour in self.specific_contour:
                for v in contour:
                    v_arr = np.asarray(v).flatten()
                    if len(v_arr) >= 3:
                        glVertex3fv(v_arr[:3])
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
                    v_arr = np.asarray(v).flatten()
                    if len(v_arr) >= 3:
                        glVertex3fv(v_arr[:3])
                glEnd()

                # Draw vertices with color gradient: red (first) -> black (last)
                if getattr(self, 'is_draw_contour_vertices', False):
                    glPointSize(6 if is_highlighted else 5)
                    glBegin(GL_POINTS)
                    n_verts = len(contour)
                    for k, v in enumerate(contour):
                        v_arr = np.asarray(v).flatten()
                        if len(v_arr) < 3:
                            continue
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
                        glVertex3fv(v_arr[:3])
                    glEnd()

                # Draw small x, y, z axes from bounding plane info
                if (self.bounding_planes is not None and
                    i < len(self.bounding_planes) and
                    j < len(self.bounding_planes[i])):
                    bp_info = self.bounding_planes[i][j]
                    if 'mean' in bp_info and 'basis_x' in bp_info and 'basis_y' in bp_info and 'basis_z' in bp_info:
                        origin = bp_info['mean']
                        basis_x = bp_info['basis_x']
                        basis_y = bp_info['basis_y']
                        basis_z = bp_info['basis_z']
                        axis_length = 0.01  # Small axes length

                        glLineWidth(2.0)
                        # X axis - red
                        glColor3f(1.0, 0.0, 0.0)
                        glBegin(GL_LINES)
                        glVertex3fv(origin)
                        glVertex3fv(origin + axis_length * basis_x)
                        glEnd()
                        # Y axis - green
                        glColor3f(0.0, 1.0, 0.0)
                        glBegin(GL_LINES)
                        glVertex3fv(origin)
                        glVertex3fv(origin + axis_length * basis_y)
                        glEnd()
                        # Z axis - blue
                        glColor3f(0.0, 0.0, 1.0)
                        glBegin(GL_LINES)
                        glVertex3fv(origin)
                        glVertex3fv(origin + axis_length * basis_z)
                        glEnd()
                        glLineWidth(1.0)

        if self.is_draw_discarded and self.contours_discarded is not None:
            t = 0.1
            color = np.array([0, 0, 0, t])
            glColor4fv(color)
            for contour_set in self.contours_discarded:
                for contour in contour_set:
                    glBegin(GL_LINE_LOOP)
                    for v in contour:
                        v_arr = np.asarray(v).flatten()
                        if len(v_arr) >= 3:
                            glVertex3fv(v_arr[:3])
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

    def _precompute_face_scalar_ranges(self):
        """Precompute min/max scalar values per face for fast contour finding."""
        if self.scalar_field is None or self.faces_3 is None:
            return

        n_faces = len(self.faces_3)
        self._face_scalar_min = np.zeros(n_faces)
        self._face_scalar_max = np.zeros(n_faces)

        for i, face in enumerate(self.faces_3):
            v0, v1, v2 = int(face[0, 0]), int(face[1, 0]), int(face[2, 0])
            values = self.scalar_field[[v0, v1, v2]]
            self._face_scalar_min[i] = values.min()
            self._face_scalar_max[i] = values.max()

    def find_contour(self, contour_value, prev_bounding_plane=None, use_geodesic_edges=False):
        # Precompute face scalar ranges if not done
        if not hasattr(self, '_face_scalar_min') or self._face_scalar_min is None:
            self._precompute_face_scalar_ranges()

        # Build geodesic edge set if option enabled
        geodesic_edge_set = None
        if use_geodesic_edges and hasattr(self, '_geodesic_reference_paths') and self._geodesic_reference_paths:
            geodesic_edge_set = set()
            for path_info in self._geodesic_reference_paths:
                chain = path_info['chain']
                for i in range(len(chain) - 1):
                    edge = tuple(sorted([chain[i], chain[i + 1]]))
                    geodesic_edge_set.add(edge)

        # OPTIMIZATION 1: Collect all candidate edge vertices first, then deduplicate with single KDTree
        raw_edges = []  # List of (vertex1, vertex2) pairs

        for face_idx, face in enumerate(self.faces_3):
            # Skip faces that don't contain contour value (acceleration)
            if self._face_scalar_min is not None:
                if contour_value < self._face_scalar_min[face_idx] or contour_value > self._face_scalar_max[face_idx]:
                    continue

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
                raw_edges.append((face_cand_vertices[0], face_cand_vertices[1]))

        if len(raw_edges) == 0:
            return [], [], []

        # Collect all vertices
        all_raw_vertices = []
        for v1, v2 in raw_edges:
            all_raw_vertices.append(v1)
            all_raw_vertices.append(v2)
        all_raw_vertices = np.array(all_raw_vertices)

        # OPTIMIZED: Build single KDTree and deduplicate in batch
        # Build KDTree once from all raw vertices
        raw_tree = cKDTree(all_raw_vertices)

        # Find clusters of near-identical vertices
        n_raw = len(all_raw_vertices)
        vertex_indices = [-1] * n_raw  # Maps raw vertex index to deduplicated index
        contour_vertices = []

        for i in range(n_raw):
            if vertex_indices[i] != -1:
                continue  # Already assigned to a cluster

            # Find all vertices within tolerance of this one
            nearby = raw_tree.query_ball_point(all_raw_vertices[i], r=1e-7)

            # Assign all nearby vertices to the same deduplicated index
            new_idx = len(contour_vertices)
            contour_vertices.append(all_raw_vertices[i])
            for j in nearby:
                if vertex_indices[j] == -1:
                    vertex_indices[j] = new_idx

        # Build edges with deduplicated indices
        contour_edges = []
        for i, (v1, v2) in enumerate(raw_edges):
            idx1 = vertex_indices[i * 2]
            idx2 = vertex_indices[i * 2 + 1]
            if idx1 != idx2:  # Skip degenerate edges
                contour_edges.append(tuple(sorted([idx1, idx2])))

        # OPTIMIZATION 2: Union-Find for edge grouping - O(n * alpha(n)) instead of O(n²)
        # Initialize parent array for Union-Find
        n_vertices = len(contour_vertices)
        parent = list(range(n_vertices))
        rank = [0] * n_vertices

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            # Union by rank
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Union all connected vertices
        for e0, e1 in contour_edges:
            union(e0, e1)

        # Group vertices by their root
        root_to_vertices = defaultdict(set)
        for v in range(n_vertices):
            # Only include vertices that are part of edges
            root_to_vertices[find(v)].add(v)

        # Filter out singleton groups (vertices not in any edge)
        edge_vertex_set = set()
        for e0, e1 in contour_edges:
            edge_vertex_set.add(e0)
            edge_vertex_set.add(e1)

        contour_edge_groups = []
        for root, vertices in root_to_vertices.items():
            group_verts = vertices & edge_vertex_set
            if len(group_verts) > 0:
                contour_edge_groups.append(list(group_verts))

        # Assign edges to groups
        vertex_to_group = {}
        for i, group in enumerate(contour_edge_groups):
            for v in group:
                vertex_to_group[v] = i

        grouped_contour_edges = [[] for _ in contour_edge_groups]
        for edge in contour_edges:
            group_idx = vertex_to_group.get(edge[0])
            if group_idx is not None:
                grouped_contour_edges[group_idx].append(edge)

        # OPTIMIZATION 3: Adjacency dict for edge ordering - O(n) instead of O(n²)
        ordered_contour_edge_groups = []
        for i in range(len(grouped_contour_edges)):
            if len(grouped_contour_edges[i]) == 0:
                ordered_contour_edge_groups.append([])
                continue

            # Build adjacency dict: vertex -> list of connected vertices
            adjacency = defaultdict(list)
            for e0, e1 in grouped_contour_edges[i]:
                adjacency[e0].append(e1)
                adjacency[e1].append(e0)

            # Start from first edge
            first_edge = grouped_contour_edges[i][0]
            ordered_edge_group = [first_edge[0], first_edge[1]]
            visited = {first_edge[0], first_edge[1]}

            # Walk forward using adjacency dict - O(1) lookup per step
            max_iterations = len(contour_edge_groups[i]) * 2
            iteration = 0
            while len(ordered_edge_group) < len(contour_edge_groups[i]) and iteration < max_iterations:
                current = ordered_edge_group[-1]
                neighbors = adjacency.get(current, [])
                found = False
                for neighbor in neighbors:
                    if neighbor not in visited:
                        ordered_edge_group.append(neighbor)
                        visited.add(neighbor)
                        found = True
                        break
                if not found:
                    break
                iteration += 1

            ordered_contour_edge_groups.append(ordered_edge_group)

        ordered_contour_vertices = []
        ordered_contour_vertices_orig = []
        bounding_planes = []
        for edge_group in ordered_contour_edge_groups:
            verts = np.array(contour_vertices)[edge_group]

            # Skip contours with too few vertices
            if len(verts) < 3:
                continue

            # Check for self-intersection using Shapely
            try:
                from shapely.geometry import Polygon
                # Project to 2D for self-intersection check (use first two principal axes)
                centroid = verts.mean(axis=0)
                centered = verts - centroid
                # Simple 2D projection using XY or best-fit plane
                if centered.shape[1] == 3:
                    # Use SVD to find principal plane
                    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                    verts_2d = centered @ Vt[:2].T
                else:
                    verts_2d = centered[:, :2]

                poly = Polygon(verts_2d)
                if not poly.is_valid:
                    # Skip self-intersecting contours
                    print(f"  [find_contour] WARNING: Skipping self-intersecting contour at value {contour_value}")
                    continue
            except Exception as e:
                # If check fails, still include the contour
                pass

            ordered_contour_vertices.append(verts)
            ordered_contour_vertices_orig.append(verts.copy())
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

    def find_contours(self, scalar_step=0.1, skeleton_meshes=None, use_geodesic_edges=False, spacing_scale=1.0):
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

        # Clear level selection backup
        self._contours_backup = None
        self._bounding_planes_backup = None
        self.selected_stream_levels = None

        contours = []
        contours_orig = []

        origin_bounding_planes = []
        origin_contours = []
        origin_contours_orig = []
        prev_origin_plane = None
        for i, edge_group in enumerate(self.edge_groups):
            if self.edge_classes[i] == 'origin':
                # Pass previous origin's bounding plane for axis consistency
                new_origin_contour, bounding_plane = self.save_bounding_planes(
                    self.vertices[edge_group], 1, prev_bounding_plane=prev_origin_plane)
                origin_bounding_planes.append(bounding_plane)
                origin_contours.append(new_origin_contour)
                origin_contours_orig.append(self.vertices[edge_group])
                prev_origin_plane = bounding_plane  # Use this as reference for next origin
        self.bounding_planes.append(origin_bounding_planes)
        contours.append(origin_contours)
        contours_orig.append(origin_contours_orig)

        # Spacing thresholds (scale is global MESH_SCALE = 0.01)
        length_min_threshold = scale * 0.5 * spacing_scale
        length_max_threshold = scale * 1.0 * spacing_scale

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
                # Pass previous bounding plane to maintain z-axis consistency
                prev_plane = self.bounding_planes[-1][0] if len(self.bounding_planes) > 0 and len(self.bounding_planes[-1]) > 0 else None
                bounding_planes, ordered_contour_vertices, ordered_contour_vertices_orig = self.find_contour(contour_value, prev_bounding_plane=prev_plane, use_geodesic_edges=use_geodesic_edges)
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

                # Skip this contour value if no valid contours found (all skipped due to self-intersection)
                if len(bounding_planes) == 0:
                    print("SKIPPED (no valid contours at this level)")
                    print()
                    contour_value += current_adaptive_step
                    continue

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
        # Use last contour level as reference for insertion axes
        prev_insertion_plane = self.bounding_planes[-1][0] if len(self.bounding_planes) > 0 and len(self.bounding_planes[-1]) > 0 else None
        for i, edge_group in enumerate(self.edge_groups):
            if self.edge_classes[i] == 'insertion':
                new_insertion_contour, bounding_plane = self.save_bounding_planes(
                    self.vertices[edge_group], 10, prev_bounding_plane=prev_insertion_plane)
                insertion_bounding_planes.append(bounding_plane)
                insertion_contours.append(new_insertion_contour)
                insertion_contours_orig.append(self.vertices[edge_group])
                prev_insertion_plane = bounding_plane  # Use for next insertion
        
        self.bounding_planes.append(insertion_bounding_planes)
        contours.append(insertion_contours)
        contours_orig.append(insertion_contours_orig)
        
        self.draw_contour_stream = [True] * len(contours)
        self.is_draw_contours = True
        self.contours = contours

        # # Auto-detect skeleton attachments if skeleton_meshes provided
        # if skeleton_meshes is not None and len(skeleton_meshes) > 0:
        #     self.auto_detect_attachments(skeleton_meshes)

    def _refine_transition_points(self):
        """
        Refine contours at transition points where contour count changes.

        Always process from LARGER count side to SMALLER count side.
        For transitions like 3→2, 2→1, find the exact scalar value where
        the merge happens.
        """
        if len(self.bounding_planes) < 2:
            return

        print(f"\n=== Transition Point Refinement ===")

        # Determine processing direction: from larger count side to smaller
        origin_count = len(self.bounding_planes[0])
        insertion_count = len(self.bounding_planes[-1])
        process_forward = origin_count >= insertion_count  # origin → insertion if origin has more

        print(f"Origin count: {origin_count}, Insertion count: {insertion_count}")
        print(f"Processing direction: {'origin → insertion' if process_forward else 'insertion → origin'}")

        # Find all transition points (always expressed as large_count → small_count)
        transitions = []
        num_levels = len(self.bounding_planes)

        if process_forward:
            # Process origin → insertion
            for level_idx in range(num_levels - 1):
                count_curr = len(self.bounding_planes[level_idx])
                count_next = len(self.bounding_planes[level_idx + 1])
                if count_curr != count_next:
                    scalar_curr = self.bounding_planes[level_idx][0]['scalar_value']
                    scalar_next = self.bounding_planes[level_idx + 1][0]['scalar_value']
                    # Always store as large → small
                    if count_curr > count_next:
                        transitions.append({
                            'level_idx': level_idx,
                            'large_level': level_idx,
                            'small_level': level_idx + 1,
                            'large_count': count_curr,
                            'small_count': count_next,
                            'scalar_large': scalar_curr,
                            'scalar_small': scalar_next,
                        })
                    else:
                        transitions.append({
                            'level_idx': level_idx,
                            'large_level': level_idx + 1,
                            'small_level': level_idx,
                            'large_count': count_next,
                            'small_count': count_curr,
                            'scalar_large': scalar_next,
                            'scalar_small': scalar_curr,
                        })
        else:
            # Process insertion → origin (reverse order)
            for level_idx in range(num_levels - 1, 0, -1):
                count_curr = len(self.bounding_planes[level_idx])
                count_prev = len(self.bounding_planes[level_idx - 1])
                if count_curr != count_prev:
                    scalar_curr = self.bounding_planes[level_idx][0]['scalar_value']
                    scalar_prev = self.bounding_planes[level_idx - 1][0]['scalar_value']
                    # Always store as large → small
                    if count_curr > count_prev:
                        transitions.append({
                            'level_idx': level_idx - 1,
                            'large_level': level_idx,
                            'small_level': level_idx - 1,
                            'large_count': count_curr,
                            'small_count': count_prev,
                            'scalar_large': scalar_curr,
                            'scalar_small': scalar_prev,
                        })
                    else:
                        transitions.append({
                            'level_idx': level_idx - 1,
                            'large_level': level_idx - 1,
                            'small_level': level_idx,
                            'large_count': count_prev,
                            'small_count': count_curr,
                            'scalar_large': scalar_prev,
                            'scalar_small': scalar_curr,
                        })

        if not transitions:
            print("No transition points found.")
            return

        print(f"Found {len(transitions)} transition points:")
        for t in transitions:
            print(f"  Level {t['level_idx']}: {t['large_count']}→{t['small_count']} "
                  f"(scalar {t['scalar_large']:.6f}→{t['scalar_small']:.6f})")

        # Process transitions in reverse order to maintain indices
        transitions.sort(key=lambda x: x['level_idx'], reverse=True)

        for t in transitions:
            level_idx = t['level_idx']
            large_count = t['large_count']
            small_count = t['small_count']
            scalar_large = t['scalar_large']
            scalar_small = t['scalar_small']

            max_iterations = 20
            tolerance = 1e-6

            # Use bounding plane from large count side as reference
            prev_plane = self.bounding_planes[t['large_level']][0] if len(self.bounding_planes[t['large_level']]) > 0 else None

            # Find both transition points:
            # - best_large_scalar: last scalar with large_count (just before transition)
            # - best_small_scalar: first scalar with small_count (just after transition)
            best_large_scalar = None
            best_large_contours = None
            best_large_planes = None

            best_small_scalar = None
            best_small_contours = None
            best_small_planes = None

            # Binary search from large side towards small side
            search_start = scalar_large
            search_end = scalar_small

            print(f"  Searching transition {large_count}→{small_count} from scalar {scalar_large:.6f} to {scalar_small:.6f}")

            # Phase 1: Binary search to find transition boundary
            for iteration in range(max_iterations):
                mid_scalar = (search_start + search_end) / 2

                if abs(search_end - search_start) < tolerance:
                    print(f"    [Phase1 iter {iteration}] Converged at {mid_scalar:.6f}")
                    break

                planes_at_mid, contours_at_mid, _ = self.find_contour(mid_scalar, prev_bounding_plane=prev_plane)

                if len(contours_at_mid) == 0:
                    print(f"    [Phase1 iter {iteration}] scalar={mid_scalar:.6f}: 0 contours (invalid)")
                    # Move towards large side (which should have valid contours)
                    search_end = mid_scalar
                    continue

                count_at_mid = len(contours_at_mid)

                if count_at_mid == large_count:
                    best_large_scalar = mid_scalar
                    best_large_contours = contours_at_mid
                    best_large_planes = planes_at_mid
                    # Search towards small side
                    search_start = mid_scalar
                elif count_at_mid == small_count:
                    best_small_scalar = mid_scalar
                    best_small_contours = contours_at_mid
                    best_small_planes = planes_at_mid
                    # Search towards large side
                    search_end = mid_scalar
                elif count_at_mid > large_count:
                    print(f"    [Phase1 iter {iteration}] count {count_at_mid} > large {large_count}, unexpected")
                    search_start = mid_scalar
                else:
                    print(f"    [Phase1 iter {iteration}] count {count_at_mid} < small {small_count}, unexpected")
                    search_end = mid_scalar

            # Phase 2: If we only found one side, search explicitly for the other
            if best_small_scalar is not None and best_large_scalar is None:
                print(f"    [Phase2] Found small_count={small_count}, searching for large_count={large_count}")
                # Search towards large side
                for step in range(10):
                    test_scalar = best_small_scalar + (scalar_large - best_small_scalar) * (step + 1) / 10
                    planes_test, contours_test, _ = self.find_contour(test_scalar, prev_bounding_plane=prev_plane)
                    if len(contours_test) == large_count:
                        best_large_scalar = test_scalar
                        best_large_contours = contours_test
                        best_large_planes = planes_test
                        print(f"    [Phase2] Found large_count={large_count} at scalar={test_scalar:.6f}")
                        break

            if best_large_scalar is not None and best_small_scalar is None:
                print(f"    [Phase2] Found large_count={large_count}, searching for small_count={small_count}")
                # Search towards small side
                for step in range(10):
                    test_scalar = best_large_scalar + (scalar_small - best_large_scalar) * (step + 1) / 10
                    planes_test, contours_test, _ = self.find_contour(test_scalar, prev_bounding_plane=prev_plane)
                    if len(contours_test) == small_count:
                        best_small_scalar = test_scalar
                        best_small_contours = contours_test
                        best_small_planes = planes_test
                        print(f"    [Phase2] Found small_count={small_count} at scalar={test_scalar:.6f}")
                        break

            # Fallback to original endpoints if still not found
            if best_large_scalar is None:
                print(f"    Using original level {t['large_level']} as large_count={large_count}")
                best_large_scalar = scalar_large
                best_large_planes = list(self.bounding_planes[t['large_level']])
                best_large_contours = list(self.contours[t['large_level']])
            if best_small_scalar is None:
                print(f"    Using original level {t['small_level']} as small_count={small_count}")
                best_small_scalar = scalar_small
                best_small_planes = list(self.bounding_planes[t['small_level']])
                best_small_contours = list(self.contours[t['small_level']])

            print(f"    Result: best_large_scalar={best_large_scalar:.6f} (count={large_count}), best_small_scalar={best_small_scalar:.6f} (count={small_count})")

            # Collect results to insert (in order by scalar value)
            # Only insert the SMALL count contour - manual cut will handle the large side
            to_insert = []
            # Skip large count contour - manual cut handles the first SEPARATE division
            if best_large_scalar is not None and best_large_contours is not None:
                print(f"    Skipping large_scalar={best_large_scalar:.6f} (manual cut will handle)")
            if best_small_scalar is not None and best_small_contours is not None:
                # Don't insert if it's essentially the same as original small level
                if abs(best_small_scalar - scalar_small) > tolerance:
                    to_insert.append((best_small_scalar, best_small_planes, best_small_contours, small_count))
                else:
                    print(f"    Skipping small_scalar={best_small_scalar:.6f} (same as original level)")

            # Sort by scalar value
            to_insert.sort(key=lambda x: x[0])

            # Insert in reverse order to maintain indices
            for scalar, planes, contours, count in reversed(to_insert):
                insert_idx = level_idx + 1
                print(f"  Inserting transition contour at scalar {scalar:.6f} "
                      f"(count={count}) at index {insert_idx}")
                self.bounding_planes.insert(insert_idx, planes)
                self.contours.insert(insert_idx, contours)

            if len(to_insert) == 0:
                print(f"  Note: Transition {large_count}→{small_count} already has sharp boundary (no refinement needed)")

        # Update draw_contour_stream
        self.draw_contour_stream = [True] * len(self.contours)
        print(f"=== Transition Refinement Done: {len(self.bounding_planes)} contour levels ===\n")

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

        # First, refine transition points where contour count changes
        self._refine_transition_points()

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

            # Traverse levels: DON'T MODIFY non-square planes, just use them as reference
            for level_idx in range(len(self.bounding_planes)):
                if len(self.bounding_planes[level_idx]) <= stream_idx:
                    continue

                plane_info = self.bounding_planes[level_idx][stream_idx]

                # Skip square-like planes - they will be interpolated during smoothing
                if plane_info.get('square_like', False):
                    continue

                # NON-SQUARE (rectangle) planes: DON'T MODIFY them at all
                # save_bounding_planes already computed good axes using farthest point pair
                # Just use them as reference for tracking orientation
                prev_basis_x = plane_info['basis_x'].copy()
                prev_basis_y = plane_info['basis_y'].copy()

    def smoothen_contours(self):
        """
        Smoothen bounding plane orientations to avoid twisted bounding boxes.

        For square-like contours (circles), bounding box orientation can be arbitrary
        and may cause twisting. This function adjusts orientations by interpolating
        between non-square-like neighbors.

        Steps:
        1. For square-like planes, interpolate basis_x from nearest non-square neighbors
        2. If only one side has non-square, propagate from that side
        3. Ensure orthonormal basis (basis_y = cross(basis_z, basis_x))
        4. Compute new 90-degree bounding plane that fits the projected contour
        5. Re-run find_contour_match to get proper P,Q pairs
        """
        if len(self.bounding_planes) == 0:
            print("No Contours found")
            return

        # First align all bounding planes to have consistent orientation
        self._align_bounding_planes()

        print("Smoothening square-like contours...")

        # Smooth square-like planes by propagating orientation from nearest non-square-like planes
        for iteration in range(10):
            changed = False
            smooth_bounding_planes = []

            for i, bounding_plane_infos in enumerate(self.bounding_planes):
                new_level_planes = []

                for contour_idx, bounding_plane_info in enumerate(bounding_plane_infos):
                    # Only smooth square-like bounding planes, keep rectangle-like ones unchanged
                    if not bounding_plane_info.get('square_like', False):
                        new_level_planes.append(bounding_plane_info)
                        continue

                    curr_mean = bounding_plane_info['mean']
                    basis_z = bounding_plane_info['basis_z']
                    mean = bounding_plane_info['mean']

                    # Find closest non-square-like prev plane
                    prev_basis_x = None
                    prev_level = None
                    prev_dist = np.inf
                    for search_i in range(i - 1, -1, -1):
                        for bp in self.bounding_planes[search_i]:
                            if not bp.get('square_like', False):
                                dist = np.linalg.norm(curr_mean - bp['mean'])
                                if dist < prev_dist:
                                    prev_basis_x = bp['basis_x'].copy()
                                    prev_dist = dist
                                    prev_level = search_i
                        if prev_basis_x is not None:
                            break

                    # Find closest non-square-like next plane
                    next_basis_x = None
                    next_level = None
                    next_dist = np.inf
                    for search_i in range(i + 1, len(self.bounding_planes)):
                        for bp in self.bounding_planes[search_i]:
                            if not bp.get('square_like', False):
                                dist = np.linalg.norm(curr_mean - bp['mean'])
                                if dist < next_dist:
                                    next_basis_x = bp['basis_x'].copy()
                                    next_dist = dist
                                    next_level = search_i
                        if next_basis_x is not None:
                            break

                    # If no non-square neighbors found, skip this plane
                    if prev_basis_x is None and next_basis_x is None:
                        new_level_planes.append(bounding_plane_info)
                        continue

                    # Determine reference basis_x
                    if prev_basis_x is None:
                        # Only next side has non-square - propagate from next
                        ref_basis_x = next_basis_x
                    elif next_basis_x is None:
                        # Only prev side has non-square - propagate from prev
                        ref_basis_x = prev_basis_x
                    else:
                        # Both sides have non-square - interpolate
                        # Align next_basis_x to prev_basis_x (find best 90° rotation)
                        next_bp = None
                        for bp in self.bounding_planes[next_level]:
                            if not bp.get('square_like', False):
                                next_bp = bp
                                break
                        if next_bp is None:
                            next_bp = self.bounding_planes[next_level][0]

                        next_bx = next_bp['basis_x']
                        next_by = next_bp['basis_y']

                        candidates = [next_bx, next_by, -next_bx, -next_by]
                        best_aligned = next_bx
                        best_dot = -np.inf
                        for cand in candidates:
                            dot = np.dot(cand, prev_basis_x)
                            if dot > best_dot:
                                best_dot = dot
                                best_aligned = cand

                        # Interpolate based on level position
                        total_span = next_level - prev_level
                        if total_span > 0:
                            t = (i - prev_level) / total_span
                        else:
                            t = 0.5

                        ref_basis_x = (1 - t) * prev_basis_x + t * best_aligned
                        ref_basis_x = ref_basis_x / (np.linalg.norm(ref_basis_x) + 1e-10)

                    # Project reference onto this plane (perpendicular to basis_z)
                    new_basis_x = ref_basis_x - np.dot(ref_basis_x, basis_z) * basis_z
                    norm = np.linalg.norm(new_basis_x)
                    if norm < 1e-10:
                        new_level_planes.append(bounding_plane_info)
                        continue
                    new_basis_x = new_basis_x / norm

                    # Compute orthonormal basis_y (90 degrees guaranteed)
                    new_basis_y = np.cross(basis_z, new_basis_x)
                    new_basis_y = new_basis_y / (np.linalg.norm(new_basis_y) + 1e-10)

                    # Get original contour vertices for re-matching
                    # Try multiple sources: contour_vertices, self.contours, projected_2d
                    contour_vertices = bounding_plane_info.get('contour_vertices')
                    if contour_vertices is None and i < len(self.contours) and contour_idx < len(self.contours[i]):
                        contour_vertices = self.contours[i][contour_idx]
                    if contour_vertices is None:
                        contour_vertices = bounding_plane_info.get('projected_2d')
                    if contour_vertices is None:
                        new_level_planes.append(bounding_plane_info)
                        continue

                    contour_vertices = np.array(contour_vertices)

                    # Project contour onto new basis to get 2D coordinates
                    projected_2d = np.array([
                        [np.dot(v - mean, new_basis_x), np.dot(v - mean, new_basis_y)]
                        for v in contour_vertices
                    ])

                    # Compute bounding box that fits the projected contour (90-degree corners)
                    min_x, max_x = np.min(projected_2d[:, 0]), np.max(projected_2d[:, 0])
                    min_y, max_y = np.min(projected_2d[:, 1]), np.max(projected_2d[:, 1])

                    # Create new bounding plane corners (always 90 degrees)
                    bounding_plane_2d = np.array([
                        [min_x, min_y], [max_x, min_y],
                        [max_x, max_y], [min_x, max_y]
                    ])
                    new_bounding_plane = np.array([
                        mean + x * new_basis_x + y * new_basis_y
                        for x, y in bounding_plane_2d
                    ])

                    # Compute contour_match: for each contour vertex P, find closest point Q on bounding plane EDGES
                    # Q should lie on the boundary of the unit square, not inside
                    contour_match = []
                    bp = new_bounding_plane
                    for i, p in enumerate(contour_vertices):
                        # Find closest point on each of the 4 edges
                        best_q = None
                        best_dist = np.inf

                        # 4 edges: (0,1), (1,2), (2,3), (3,0)
                        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
                        for e0, e1 in edges:
                            edge_start = bp[e0]
                            edge_end = bp[e1]
                            edge_vec = edge_end - edge_start
                            edge_len_sq = np.dot(edge_vec, edge_vec)

                            if edge_len_sq > 1e-10:
                                t = np.dot(p - edge_start, edge_vec) / edge_len_sq
                                t = np.clip(t, 0, 1)
                                q_candidate = edge_start + t * edge_vec
                            else:
                                q_candidate = edge_start

                            dist = np.linalg.norm(p - q_candidate)
                            if dist < best_dist:
                                best_dist = dist
                                best_q = q_candidate

                        contour_match.append([p, best_q])

                    # Compute projected 3D points
                    projected_2d_3d = np.array([
                        mean + x * new_basis_x + y * new_basis_y
                        for x, y in projected_2d
                    ])

                    area = compute_polygon_area(projected_2d)

                    new_bounding_plane_info = {
                        'basis_x': new_basis_x,
                        'basis_y': new_basis_y,
                        'basis_z': basis_z,
                        'mean': mean,
                        'bounding_plane': new_bounding_plane,
                        'projected_2d': projected_2d_3d,
                        'area': area,
                        'contour_match': contour_match,
                        'scalar_value': bounding_plane_info['scalar_value'],
                        'square_like': bounding_plane_info['square_like'],
                        'newell_normal': basis_z,
                        'contour_vertices': contour_vertices,
                    }

                    # Check if basis changed significantly
                    old_basis_x = bounding_plane_info.get('basis_x')
                    if old_basis_x is not None:
                        dot = abs(np.dot(old_basis_x, new_basis_x))
                        if dot < 0.999:
                            changed = True

                    new_level_planes.append(new_bounding_plane_info)

                smooth_bounding_planes.append(new_level_planes)

            self.bounding_planes = smooth_bounding_planes

            if not changed:
                print(f"  Converged after {iteration + 1} iterations")
                break

        # Align axes consistently from origin to insertion
        # First align z-axes to point consistently, then align x/y within each plane
        print("  Aligning axes from origin to insertion...")

        num_levels = len(self.bounding_planes)
        if num_levels > 1:
            # First pass: align z-axes to point consistently by propagating from origin
            for stream_idx in range(len(self.bounding_planes[0])):
                if stream_idx >= len(self.bounding_planes[0]):
                    continue

                # Use origin's z-axis as reference and propagate forward
                ref_basis_z = self.bounding_planes[0][stream_idx]['basis_z'].copy()

                # Propagate z-axis alignment from origin to insertion
                for level_idx in range(1, num_levels):
                    if stream_idx >= len(self.bounding_planes[level_idx]):
                        continue
                    bp_info = self.bounding_planes[level_idx][stream_idx]
                    basis_z = bp_info['basis_z']

                    # If z-axis points opposite to reference, flip it
                    if np.dot(basis_z, ref_basis_z) < 0:
                        # Flip z-axis and also flip y to maintain right-handed system
                        bp_info['basis_z'] = -basis_z
                        bp_info['basis_y'] = -bp_info['basis_y']
                        if 'newell_normal' in bp_info:
                            bp_info['newell_normal'] = -bp_info['newell_normal']
                        ref_basis_z = -basis_z  # Update reference
                    else:
                        ref_basis_z = basis_z  # Update reference to current

            # Second pass: align x/y axes consistently
            for stream_idx in range(len(self.bounding_planes[0])):
                # Get reference basis_x from origin (first level)
                if stream_idx < len(self.bounding_planes[0]):
                    ref_basis_x = self.bounding_planes[0][stream_idx]['basis_x'].copy()
                else:
                    continue

                # Propagate from origin to insertion
                for level_idx in range(1, num_levels):
                    if stream_idx >= len(self.bounding_planes[level_idx]):
                        continue

                    bp_info = self.bounding_planes[level_idx][stream_idx]
                    basis_z = bp_info['basis_z']
                    mean = bp_info['mean']

                    # Project reference basis_x onto current plane
                    proj_ref_x = ref_basis_x - np.dot(ref_basis_x, basis_z) * basis_z
                    proj_norm = np.linalg.norm(proj_ref_x)

                    if not bp_info.get('square_like', False):
                        # NON-SQUARE: Only allow 90-degree rotations
                        # Bounding plane shape stays same, but corner indices rotate
                        basis_x = bp_info['basis_x']
                        basis_y = bp_info['basis_y']

                        # Find which 90-degree rotation best aligns with projected reference
                        # Track index: 0=0°, 1=90°, 2=180°, 3=270°
                        candidates = [
                            (basis_x, basis_y),      # 0 deg
                            (basis_y, -basis_x),     # 90 deg
                            (-basis_x, -basis_y),    # 180 deg
                            (-basis_y, basis_x)      # 270 deg
                        ]

                        best_idx = 0
                        if proj_norm > 1e-6:
                            proj_ref_x = proj_ref_x / proj_norm

                            best_dot = -np.inf
                            for idx, (cand_x, cand_y) in enumerate(candidates):
                                dot = np.dot(cand_x, proj_ref_x)
                                if dot > best_dot:
                                    best_dot = dot
                                    best_idx = idx

                            bp_info['basis_x'] = candidates[best_idx][0]
                            bp_info['basis_y'] = candidates[best_idx][1]
                            ref_basis_x = candidates[best_idx][0].copy()

                            # If rotation != 0, rotate bounding_plane corner indices
                            # contour_match will be recomputed in final step with fresh vertices
                            if best_idx != 0:
                                old_bp = bp_info['bounding_plane']
                                bp_info['bounding_plane'] = np.roll(old_bp, -best_idx, axis=0)
                        else:
                            ref_basis_x = basis_x.copy()

                    else:
                        # SQUARE-LIKE: Use arbitrary projection and recompute bounding plane
                        if proj_norm > 1e-6:
                            new_basis_x = proj_ref_x / proj_norm
                        else:
                            new_basis_x = bp_info['basis_x']

                        new_basis_y = np.cross(basis_z, new_basis_x)
                        new_basis_y = new_basis_y / (np.linalg.norm(new_basis_y) + 1e-10)

                        bp_info['basis_x'] = new_basis_x
                        bp_info['basis_y'] = new_basis_y

                        # Recompute bounding plane corners with new basis
                        contour_vertices = bp_info.get('contour_vertices')
                        if contour_vertices is None and level_idx < len(self.contours) and stream_idx < len(self.contours[level_idx]):
                            contour_vertices = self.contours[level_idx][stream_idx]

                        if contour_vertices is not None:
                            contour_vertices = np.array(contour_vertices)
                            projected_2d = np.array([
                                [np.dot(v - mean, new_basis_x), np.dot(v - mean, new_basis_y)]
                                for v in contour_vertices
                            ])

                            min_x, max_x = np.min(projected_2d[:, 0]), np.max(projected_2d[:, 0])
                            min_y, max_y = np.min(projected_2d[:, 1]), np.max(projected_2d[:, 1])

                            bounding_plane_2d = np.array([
                                [min_x, min_y], [max_x, min_y],
                                [max_x, max_y], [min_x, max_y]
                            ])
                            new_bounding_plane = np.array([
                                mean + x * new_basis_x + y * new_basis_y
                                for x, y in bounding_plane_2d
                            ])

                            bp_info['bounding_plane'] = new_bounding_plane

                            # Recompute contour_match with new bounding plane
                            bp = new_bounding_plane
                            n_verts = len(contour_vertices)

                            corner_indices = []
                            for corner in bp:
                                dists = np.linalg.norm(contour_vertices - corner, axis=1)
                                corner_indices.append(np.argmin(dists))

                            new_contour_match = [None] * n_verts
                            for edge_idx in range(4):
                                next_edge_idx = (edge_idx + 1) % 4
                                start_corner_idx = corner_indices[edge_idx]
                                end_corner_idx = corner_indices[next_edge_idx]
                                edge_start = bp[edge_idx]
                                edge_end = bp[next_edge_idx]

                                if end_corner_idx >= start_corner_idx:
                                    segment_indices = list(range(start_corner_idx, end_corner_idx + 1))
                                else:
                                    segment_indices = list(range(start_corner_idx, n_verts)) + list(range(0, end_corner_idx + 1))

                                if len(segment_indices) < 2:
                                    for idx in segment_indices:
                                        new_contour_match[idx] = [contour_vertices[idx], edge_start.copy()]
                                    continue

                                arc_lengths = [0.0]
                                for j in range(1, len(segment_indices)):
                                    prev_idx = segment_indices[j - 1]
                                    curr_idx = segment_indices[j]
                                    arc_lengths.append(arc_lengths[-1] + np.linalg.norm(contour_vertices[curr_idx] - contour_vertices[prev_idx]))

                                total_arc = arc_lengths[-1] if arc_lengths[-1] > 1e-10 else 1.0

                                for j, idx in enumerate(segment_indices):
                                    t = arc_lengths[j] / total_arc
                                    q = edge_start + t * (edge_end - edge_start)
                                    new_contour_match[idx] = [contour_vertices[idx], q]

                            # Fill None entries
                            for idx in range(n_verts):
                                if new_contour_match[idx] is None:
                                    p = contour_vertices[idx]
                                    best_q = bp[0]
                                    best_dist = np.inf
                                    for e0, e1 in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                                        edge_vec = bp[e1] - bp[e0]
                                        edge_len_sq = np.dot(edge_vec, edge_vec)
                                        if edge_len_sq > 1e-10:
                                            t = np.clip(np.dot(p - bp[e0], edge_vec) / edge_len_sq, 0, 1)
                                            q_cand = bp[e0] + t * edge_vec
                                        else:
                                            q_cand = bp[e0]
                                        d = np.linalg.norm(p - q_cand)
                                        if d < best_dist:
                                            best_dist = d
                                            best_q = q_cand
                                    new_contour_match[idx] = [p, best_q]

                            bp_info['contour_match'] = new_contour_match

                        ref_basis_x = new_basis_x.copy()

        # Final step: recompute contour_match for ALL planes using current bounding_plane
        # Use fresh contour_vertices - don't reuse old contour_match P values
        print("  Recomputing contour_match for all planes...")
        for level_idx, bounding_plane_infos in enumerate(self.bounding_planes):
            for stream_idx, bp_info in enumerate(bounding_plane_infos):
                bp = bp_info.get('bounding_plane')
                if bp is None or len(bp) < 4:
                    continue

                # Get fresh contour_vertices - don't reuse old contour_match
                contour_vertices = bp_info.get('contour_vertices')
                if contour_vertices is None and level_idx < len(self.contours) and stream_idx < len(self.contours[level_idx]):
                    contour_vertices = self.contours[level_idx][stream_idx]
                if contour_vertices is None:
                    continue
                Ps = np.array(contour_vertices)

                n_verts = len(Ps)
                if n_verts < 4:
                    continue

                # Find closest P vertex to each bounding plane corner
                corner_indices = []
                for corner in bp:
                    dists = np.linalg.norm(Ps - corner, axis=1)
                    corner_indices.append(np.argmin(dists))

                # Build contour_match by mapping segments to edges
                new_contour_match = [None] * n_verts
                for edge_idx in range(4):
                    next_edge_idx = (edge_idx + 1) % 4
                    start_corner_idx = corner_indices[edge_idx]
                    end_corner_idx = corner_indices[next_edge_idx]
                    edge_start = bp[edge_idx]
                    edge_end = bp[next_edge_idx]

                    if end_corner_idx >= start_corner_idx:
                        segment_indices = list(range(start_corner_idx, end_corner_idx + 1))
                    else:
                        segment_indices = list(range(start_corner_idx, n_verts)) + list(range(0, end_corner_idx + 1))

                    if len(segment_indices) < 2:
                        for seg_idx in segment_indices:
                            new_contour_match[seg_idx] = [Ps[seg_idx].copy(), edge_start.copy()]
                        continue

                    arc_lengths = [0.0]
                    for j in range(1, len(segment_indices)):
                        prev_idx = segment_indices[j - 1]
                        curr_idx = segment_indices[j]
                        arc_lengths.append(arc_lengths[-1] + np.linalg.norm(Ps[curr_idx] - Ps[prev_idx]))

                    total_arc = arc_lengths[-1] if arc_lengths[-1] > 1e-10 else 1.0

                    for j, seg_idx in enumerate(segment_indices):
                        t = arc_lengths[j] / total_arc
                        q = edge_start + t * (edge_end - edge_start)
                        new_contour_match[seg_idx] = [Ps[seg_idx].copy(), q]

                # Fill None entries
                for v_idx in range(n_verts):
                    if new_contour_match[v_idx] is None:
                        p = Ps[v_idx]
                        best_q = bp[0]
                        best_dist = np.inf
                        for e0, e1 in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                            edge_vec = bp[e1] - bp[e0]
                            edge_len_sq = np.dot(edge_vec, edge_vec)
                            if edge_len_sq > 1e-10:
                                t = np.clip(np.dot(p - bp[e0], edge_vec) / edge_len_sq, 0, 1)
                                q_cand = bp[e0] + t * edge_vec
                            else:
                                q_cand = bp[e0]
                            d = np.linalg.norm(p - q_cand)
                            if d < best_dist:
                                best_dist = d
                                best_q = q_cand
                        new_contour_match[v_idx] = [p.copy(), best_q]

                bp_info['contour_match'] = new_contour_match

        print("Smoothening complete")

    def smoothen_contours_z(self):
        """
        Align z-axes consistently across all contour levels.

        Algorithm:
        1. Find starting level: level with minimum contour count, closest to origin
        2. Forward pass (starting → insertion): compare z with previous level's z
        3. Backward pass (starting → origin): compare z with next level's z

        When contour counts match: one-to-one correspondence by index
        When counts differ: compare with closest contour by mean position
        """
        if len(self.bounding_planes) < 2:
            print("Need at least 2 contour levels")
            return

        # Check if we're in stream mode (after cutting)
        # Stream mode: bounding_planes[stream][level] - each stream has multiple levels
        # Level mode: bounding_planes[level][contour_idx] - each level has multiple contours
        if hasattr(self, 'stream_bounding_planes') and self.stream_bounding_planes is not None:
            print("Smoothening z-axes (stream mode)...")
            self._smoothen_contours_z_stream_mode()
            return

        print("Smoothening z-axes...")

        num_levels = len(self.bounding_planes)

        # Find contour counts per level
        contour_counts = [len(self.bounding_planes[i]) for i in range(num_levels)]
        print(f"  Contour counts: {contour_counts}")

        # Find starting level: minimum contour count, closest to origin (level 0)
        min_count = min(contour_counts)
        start_level = None
        for i in range(num_levels):
            if contour_counts[i] == min_count:
                start_level = i
                break

        print(f"  Starting level: {start_level} (count={min_count})")

        # Helper: find closest contour in other_level to target contour
        def find_closest_contour(target_bp, other_level):
            target_mean = target_bp['mean']
            best_idx = 0
            best_dist = np.inf
            for i, bp in enumerate(self.bounding_planes[other_level]):
                dist = np.linalg.norm(bp['mean'] - target_mean)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            return best_idx

        # ========== FORWARD PASS: start_level → insertion ==========
        print("  Forward pass...")

        # Handle starting level specially - flip z to point toward next contours
        if start_level < num_levels - 1:
            curr_count = contour_counts[start_level]

            for contour_idx in range(curr_count):
                curr_bp = self.bounding_planes[start_level][contour_idx]
                curr_mean = curr_bp['mean']
                curr_z = curr_bp['basis_z']

                # Find closest next contour
                next_idx = find_closest_contour(curr_bp, start_level + 1)
                next_mean = self.bounding_planes[start_level + 1][next_idx]['mean']

                direction = next_mean - curr_mean
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-10:
                    direction = direction / direction_norm

                    if np.dot(curr_z, direction) < 0:
                        print(f"    Level {start_level}, contour {contour_idx}: flipping z and x (toward next)")
                        curr_bp['basis_z'] = -curr_bp['basis_z']
                        curr_bp['basis_x'] = -curr_bp['basis_x']

        # Continue forward from start_level+1 to insertion
        for level_idx in range(start_level + 1, num_levels):
            prev_level = level_idx - 1
            curr_count = contour_counts[level_idx]
            prev_count = contour_counts[prev_level]

            for contour_idx in range(curr_count):
                curr_bp = self.bounding_planes[level_idx][contour_idx]
                curr_z = curr_bp['basis_z']

                # Find corresponding previous contour
                if curr_count == prev_count:
                    # One-to-one correspondence by index
                    prev_idx = contour_idx
                else:
                    # Different counts: find closest
                    prev_idx = find_closest_contour(curr_bp, prev_level)

                prev_z = self.bounding_planes[prev_level][prev_idx]['basis_z']

                if np.dot(curr_z, prev_z) < 0:
                    print(f"    Level {level_idx}, contour {contour_idx}: flipping z and x")
                    curr_bp['basis_z'] = -curr_bp['basis_z']
                    curr_bp['basis_x'] = -curr_bp['basis_x']

        # ========== BACKWARD PASS: start_level → origin ==========
        print("  Backward pass...")

        # Continue backward from start_level-1 to origin
        for level_idx in range(start_level - 1, -1, -1):
            next_level = level_idx + 1  # "next" in backward = toward insertion
            curr_count = contour_counts[level_idx]
            next_count = contour_counts[next_level]

            for contour_idx in range(curr_count):
                curr_bp = self.bounding_planes[level_idx][contour_idx]
                curr_z = curr_bp['basis_z']

                # Find corresponding next contour (toward insertion)
                if curr_count == next_count:
                    # One-to-one correspondence by index
                    next_idx = contour_idx
                else:
                    # Different counts: find closest
                    next_idx = find_closest_contour(curr_bp, next_level)

                next_z = self.bounding_planes[next_level][next_idx]['basis_z']

                if np.dot(curr_z, next_z) < 0:
                    print(f"    Level {level_idx}, contour {contour_idx}: flipping z and x (align with next)")
                    curr_bp['basis_z'] = -curr_bp['basis_z']
                    curr_bp['basis_x'] = -curr_bp['basis_x']

        # ========== Final check: ensure origin contours point toward insertion ==========
        print("  Final origin check...")
        if num_levels > 1:
            origin_count = contour_counts[0]
            for contour_idx in range(origin_count):
                curr_bp = self.bounding_planes[0][contour_idx]
                curr_mean = curr_bp['mean']
                curr_z = curr_bp['basis_z']

                # Find closest contour at level 1
                next_idx = find_closest_contour(curr_bp, 1)
                next_mean = self.bounding_planes[1][next_idx]['mean']

                # Check if z points toward next level
                direction = next_mean - curr_mean
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-10:
                    direction = direction / direction_norm

                    if np.dot(curr_z, direction) < 0:
                        print(f"    Origin contour {contour_idx}: flipping z and x (toward insertion)")
                        curr_bp['basis_z'] = -curr_bp['basis_z']
                        curr_bp['basis_x'] = -curr_bp['basis_x']

        print("  Z-axis smoothening complete")

    def _smoothen_contours_z_stream_mode(self):
        """
        Align z-axes for stream mode (after cutting).

        In stream mode: bounding_planes[stream][level]
        For each stream, align z from origin (level 0) toward insertion (last level).
        """
        num_streams = len(self.stream_bounding_planes)

        for stream_i in range(num_streams):
            bp_stream = self.stream_bounding_planes[stream_i]
            stream_len = len(bp_stream)

            if stream_len < 2:
                continue

            # First contour (origin): z should point toward second contour (toward insertion)
            first_mean = bp_stream[0]['mean']
            second_mean = bp_stream[1]['mean']
            forward_dir = second_mean - first_mean
            forward_norm = np.linalg.norm(forward_dir)

            if forward_norm > 1e-10:
                forward_dir = forward_dir / forward_norm
                if np.dot(bp_stream[0]['basis_z'], forward_dir) < 0:
                    bp_stream[0]['basis_z'] = -bp_stream[0]['basis_z']
                    bp_stream[0]['basis_x'] = -bp_stream[0]['basis_x']
                    print(f"  Stream {stream_i}, level 0: flipped z toward insertion")

            # Forward pass: align z with previous level
            for level in range(1, stream_len):
                prev_z = bp_stream[level - 1]['basis_z']
                curr_z = bp_stream[level]['basis_z']

                if np.dot(curr_z, prev_z) < 0:
                    bp_stream[level]['basis_z'] = -bp_stream[level]['basis_z']
                    bp_stream[level]['basis_x'] = -bp_stream[level]['basis_x']
                    print(f"  Stream {stream_i}, level {level}: flipped z to align with prev")

        # Update self.bounding_planes to reflect changes
        self.bounding_planes = self.stream_bounding_planes

        print("  Z-axis smoothening (stream mode) complete")

    def smoothen_contours_x(self):
        """
        Align x-axes consistently across all contour levels.

        Algorithm:
        1. Find starting level: level with minimum contour count closest to origin
        2. For starting contour: compare x with (1,0,0), flip x and y if dot < 0
        3. Forward pass (starting → insertion): compare x with previous level's x
        4. Backward pass (starting → origin): compare x with next level's x

        When contour counts match: one-to-one correspondence by index
        When counts differ: compare with closest contour by mean position
        """
        if len(self.bounding_planes) < 2:
            print("Need at least 2 contour levels")
            return

        # Check if we're in stream mode (after cutting)
        if hasattr(self, 'stream_bounding_planes') and self.stream_bounding_planes is not None:
            print("Smoothening x-axes (stream mode)...")
            self._smoothen_contours_x_stream_mode()
            return

        print("Smoothening x-axes...")

        num_levels = len(self.bounding_planes)

        # Find contour counts per level
        contour_counts = [len(self.bounding_planes[i]) for i in range(num_levels)]
        print(f"  Contour counts: {contour_counts}")

        # Find starting level: minimum contour count, closest to origin (level 0)
        min_count = min(contour_counts)
        start_level = None
        for i in range(num_levels):
            if contour_counts[i] == min_count:
                start_level = i
                break

        print(f"  Starting level: {start_level} (count={min_count})")

        # Helper: find closest contour in other_level to target contour
        def find_closest_contour(target_bp, other_level):
            target_mean = target_bp['mean']
            best_idx = 0
            best_dist = np.inf
            for i, bp in enumerate(self.bounding_planes[other_level]):
                dist = np.linalg.norm(bp['mean'] - target_mean)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            return best_idx

        # Reference vector for initial alignment
        ref_x = np.array([1.0, 0.0, 0.0])

        # ========== STARTING LEVEL: compare with (1,0,0) ==========
        for contour_idx in range(contour_counts[start_level]):
            curr_bp = self.bounding_planes[start_level][contour_idx]
            curr_x = curr_bp['basis_x']

            if np.dot(curr_x, ref_x) < 0:
                print(f"    Level {start_level}, contour {contour_idx}: flipping x and y (initial)")
                curr_bp['basis_x'] = -curr_bp['basis_x']
                curr_bp['basis_y'] = -curr_bp['basis_y']

        # ========== FORWARD PASS: start_level+1 → insertion ==========
        print("  Forward pass...")

        for level_idx in range(start_level + 1, num_levels):
            prev_level = level_idx - 1
            curr_count = contour_counts[level_idx]
            prev_count = contour_counts[prev_level]

            for contour_idx in range(curr_count):
                curr_bp = self.bounding_planes[level_idx][contour_idx]
                curr_x = curr_bp['basis_x']

                # Find corresponding previous contour
                if curr_count == prev_count:
                    # One-to-one correspondence by index
                    prev_idx = contour_idx
                else:
                    # Different counts: find closest
                    prev_idx = find_closest_contour(curr_bp, prev_level)

                prev_x = self.bounding_planes[prev_level][prev_idx]['basis_x']

                if np.dot(curr_x, prev_x) < 0:
                    print(f"    Level {level_idx}, contour {contour_idx}: flipping x and y")
                    curr_bp['basis_x'] = -curr_bp['basis_x']
                    curr_bp['basis_y'] = -curr_bp['basis_y']

        # ========== BACKWARD PASS: start_level-1 → origin ==========
        print("  Backward pass...")

        for level_idx in range(start_level - 1, -1, -1):
            next_level = level_idx + 1  # "next" in backward = toward insertion
            curr_count = contour_counts[level_idx]
            next_count = contour_counts[next_level]

            for contour_idx in range(curr_count):
                curr_bp = self.bounding_planes[level_idx][contour_idx]
                curr_x = curr_bp['basis_x']

                # Find corresponding next contour (toward insertion)
                if curr_count == next_count:
                    # One-to-one correspondence by index
                    next_idx = contour_idx
                else:
                    # Different counts: find closest
                    next_idx = find_closest_contour(curr_bp, next_level)

                next_x = self.bounding_planes[next_level][next_idx]['basis_x']

                if np.dot(curr_x, next_x) < 0:
                    print(f"    Level {level_idx}, contour {contour_idx}: flipping x and y")
                    curr_bp['basis_x'] = -curr_bp['basis_x']
                    curr_bp['basis_y'] = -curr_bp['basis_y']

        print("  X-axis smoothening complete")

    def _smoothen_contours_x_stream_mode(self):
        """
        Align x-axes for stream mode (after cutting).

        In stream mode: bounding_planes[stream][level]
        For each stream:
        1. First contour: align x with (1,0,0) projected onto plane
        2. Forward pass: align x with previous level's x
        """
        num_streams = len(self.stream_bounding_planes)

        for stream_i in range(num_streams):
            bp_stream = self.stream_bounding_planes[stream_i]
            stream_len = len(bp_stream)

            if stream_len < 1:
                continue

            # First contour: align x with (1,0,0) projected onto its plane
            first_bp = bp_stream[0]
            first_z = first_bp['basis_z']
            first_x = first_bp['basis_x']

            ref_x = np.array([1.0, 0.0, 0.0])
            ref_x_proj = ref_x - np.dot(ref_x, first_z) * first_z
            ref_x_norm = np.linalg.norm(ref_x_proj)

            if ref_x_norm > 0.1:
                ref_x_proj = ref_x_proj / ref_x_norm
                if np.dot(first_x, ref_x_proj) < 0:
                    first_bp['basis_x'] = -first_bp['basis_x']
                    first_bp['basis_y'] = -first_bp['basis_y']
                    print(f"  Stream {stream_i}, level 0: flipped x to align with (1,0,0)")

            # Forward pass: align x with previous level's x
            for level in range(1, stream_len):
                curr_bp = bp_stream[level]
                prev_bp = bp_stream[level - 1]

                curr_x = curr_bp['basis_x']
                curr_z = curr_bp['basis_z']
                prev_x = prev_bp['basis_x']

                # Project previous x onto current plane
                prev_x_proj = prev_x - np.dot(prev_x, curr_z) * curr_z
                prev_x_norm = np.linalg.norm(prev_x_proj)

                if prev_x_norm > 1e-10:
                    prev_x_proj = prev_x_proj / prev_x_norm
                    if np.dot(curr_x, prev_x_proj) < 0:
                        curr_bp['basis_x'] = -curr_bp['basis_x']
                        curr_bp['basis_y'] = -curr_bp['basis_y']
                        print(f"  Stream {stream_i}, level {level}: flipped x to align with prev")

        # Update self.bounding_planes to reflect changes
        self.bounding_planes = self.stream_bounding_planes

        print("  X-axis smoothening (stream mode) complete")

    def smoothen_contours_bp(self):
        """
        Smooth bounding plane orientations for square-like contours.

        Algorithm:
        1. Group levels by contour count
        2. For each group, establish streams (one-to-one correspondence by distance)
        3. For each stream:
           - Find square-like contours
           - Interpolate basis_x from nearest non-square-like neighbors (rotate around z)
           - If all contours are square-like, pick most non-square-like as reference
        """
        if len(self.bounding_planes) < 2:
            print("Need at least 2 contour levels")
            return

        # Check if we're in stream mode (after cutting)
        if hasattr(self, 'stream_bounding_planes') and self.stream_bounding_planes is not None:
            print("Smoothening bounding planes (stream mode)...")
            self._smoothen_contours_bp_stream_mode()
            return

        print("Smoothening bounding planes...")

        num_levels = len(self.bounding_planes)
        contour_counts = [len(self.bounding_planes[i]) for i in range(num_levels)]
        print(f"  Contour counts: {contour_counts}")

        # ========== Step 1: Group levels by contour count ==========
        groups = []  # List of (start_level, end_level, count)
        i = 0
        while i < num_levels:
            count = contour_counts[i]
            start = i
            while i < num_levels and contour_counts[i] == count:
                i += 1
            groups.append((start, i - 1, count))

        print(f"  Groups: {groups}")

        # ========== Step 2: Process each group ==========
        for group_start, group_end, group_count in groups:
            group_levels = list(range(group_start, group_end + 1))
            print(f"  Processing group levels {group_start}-{group_end} (count={group_count})")

            if group_count == 1:
                # Single contour per level - treat as one stream
                streams = [[self.bounding_planes[lvl][0] for lvl in group_levels]]
                stream_level_map = [[lvl for lvl in group_levels]]
                stream_contour_map = [[0 for _ in group_levels]]  # contour index is always 0
            else:
                # Multiple contours - establish streams by distance
                # Build streams using greedy matching from first level
                streams = [[] for _ in range(group_count)]
                stream_level_map = [[] for _ in range(group_count)]
                stream_contour_map = [[] for _ in range(group_count)]

                # Initialize streams with first level's contours
                for c_idx in range(group_count):
                    streams[c_idx].append(self.bounding_planes[group_start][c_idx])
                    stream_level_map[c_idx].append(group_start)
                    stream_contour_map[c_idx].append(c_idx)

                # Match subsequent levels
                for lvl in group_levels[1:]:
                    # For each contour in this level, find closest stream (by last contour mean)
                    used = set()
                    assignments = []

                    for c_idx in range(group_count):
                        bp = self.bounding_planes[lvl][c_idx]
                        bp_mean = bp['mean']

                        best_stream = -1
                        best_dist = np.inf
                        for s_idx in range(group_count):
                            if s_idx in used:
                                continue
                            last_mean = streams[s_idx][-1]['mean']
                            dist = np.linalg.norm(bp_mean - last_mean)
                            if dist < best_dist:
                                best_dist = dist
                                best_stream = s_idx

                        assignments.append((c_idx, best_stream))
                        used.add(best_stream)

                    for c_idx, s_idx in assignments:
                        streams[s_idx].append(self.bounding_planes[lvl][c_idx])
                        stream_level_map[s_idx].append(lvl)
                        stream_contour_map[s_idx].append(c_idx)

            # ========== Step 3: Process each stream ==========
            for stream_idx, stream in enumerate(streams):
                levels_in_stream = stream_level_map[stream_idx]
                contours_in_stream = stream_contour_map[stream_idx]
                print(f"    Stream {stream_idx}: {len(stream)} contours")

                # Check if any non-square-like exists
                # Cut contours are also smoothened if square-like (same principle as other contours)
                # Reference indices: non-square-like
                reference_indices = [i for i, bp in enumerate(stream)
                                     if not bp.get('square_like', False)]
                # Indices to smooth: square-like (including cut contours)
                smooth_indices = [i for i, bp in enumerate(stream)
                                  if bp.get('square_like', False)]
                cut_count = sum(1 for bp in stream if bp.get('is_cut', False))
                print(f"      Reference indices: {len(reference_indices)}, Cut contours: {cut_count}")
                print(f"      Smooth indices: {smooth_indices[:5]}...{smooth_indices[-5:] if len(smooth_indices) > 5 else ''}")

                if len(reference_indices) == 0 and len(smooth_indices) > 0:
                    # All square-like - find most non-square-like by aspect ratio
                    print(f"      All square-like, finding reference...")
                    best_idx = 0
                    best_ratio_diff = 0

                    for i, bp in enumerate(stream):
                        corners = bp.get('bounding_plane')
                        if corners is None or len(corners) < 4:
                            continue
                        width = np.linalg.norm(corners[1] - corners[0])
                        height = np.linalg.norm(corners[3] - corners[0])
                        if min(width, height) > 1e-10:
                            ratio = max(width, height) / min(width, height)
                            ratio_diff = abs(ratio - 1.0)
                            if ratio_diff > best_ratio_diff:
                                best_ratio_diff = ratio_diff
                                best_idx = i

                    # Mark this one as non-square-like (reference)
                    stream[best_idx]['square_like'] = False
                    reference_indices = [best_idx]
                    smooth_indices = [i for i in smooth_indices if i != best_idx]
                    print(f"      Selected contour {best_idx} as reference (ratio_diff={best_ratio_diff:.3f})")

                # ========== Interpolate square-like contours (including cut) (basis_x only) ==========
                for i in smooth_indices:
                    bp = stream[i]
                    curr_mean = bp['mean']
                    basis_z = bp['basis_z']

                    # Find prev reference (non-square-like)
                    prev_idx = None
                    prev_dist = np.inf
                    for j in range(i - 1, -1, -1):
                        if j in reference_indices:
                            dist = np.linalg.norm(curr_mean - stream[j]['mean'])
                            if dist < prev_dist:
                                prev_dist = dist
                                prev_idx = j
                            break  # Take closest one going backward

                    # Find next reference (non-square-like)
                    next_idx = None
                    next_dist = np.inf
                    for j in range(i + 1, len(stream)):
                        if j in reference_indices:
                            dist = np.linalg.norm(curr_mean - stream[j]['mean'])
                            if dist < next_dist:
                                next_dist = dist
                                next_idx = j
                            break  # Take closest one going forward

                    if prev_idx is None and next_idx is None:
                        print(f"        [{i}] SKIP: no prev or next found")
                        continue  # Should not happen after selecting reference

                    # Print for first few square contours processed
                    if len([x for x in smooth_indices if x < i]) < 3:
                        print(f"        [{i}] processing: prev={prev_idx}, next={next_idx}")

                    # Determine target basis_x
                    # Project prev/next onto current's plane, align next to prev, then interpolate
                    t = -1

                    if prev_idx is not None and next_idx is not None:
                        # Project prev's x onto current's plane
                        prev_x = stream[prev_idx]['basis_x']
                        prev_x_proj = prev_x - np.dot(prev_x, basis_z) * basis_z
                        prev_x_proj = prev_x_proj / (np.linalg.norm(prev_x_proj) + 1e-10)

                        # Project next's x onto current's plane
                        next_x = stream[next_idx]['basis_x']
                        next_x_proj = next_x - np.dot(next_x, basis_z) * basis_z
                        next_x_proj = next_x_proj / (np.linalg.norm(next_x_proj) + 1e-10)

                        # Interpolation ratio based on distance
                        total_dist = prev_dist + next_dist
                        if total_dist > 1e-10:
                            t = prev_dist / total_dist
                        else:
                            t = 0.5

                        # Compute signed angle from prev to next (rotation around basis_z)
                        cos_angle = np.clip(np.dot(prev_x_proj, next_x_proj), -1, 1)
                        cross = np.cross(prev_x_proj, next_x_proj)
                        sin_angle = np.dot(cross, basis_z)
                        angle = np.arctan2(sin_angle, cos_angle)

                        # Interpolate the rotation angle
                        interp_angle = angle * t

                        # Rotate prev_x_proj by interp_angle around basis_z (Rodrigues formula)
                        cos_t = np.cos(interp_angle)
                        sin_t = np.sin(interp_angle)
                        new_basis_x = prev_x_proj * cos_t + np.cross(basis_z, prev_x_proj) * sin_t

                        # Debug
                        if len([x for x in smooth_indices if x < i]) < 3:
                            print(f"          angle={np.degrees(angle):.1f}, interp={np.degrees(interp_angle):.1f}, t={t:.2f}")

                    elif prev_idx is not None:
                        # Propagate from prev - project onto current's plane
                        prev_x = stream[prev_idx]['basis_x']
                        new_basis_x = prev_x - np.dot(prev_x, basis_z) * basis_z
                        new_basis_x = new_basis_x / (np.linalg.norm(new_basis_x) + 1e-10)
                    else:
                        # Propagate from next - project onto current's plane
                        next_x = stream[next_idx]['basis_x']
                        new_basis_x = next_x - np.dot(next_x, basis_z) * basis_z
                        new_basis_x = new_basis_x / (np.linalg.norm(new_basis_x) + 1e-10)

                    new_basis_y = np.cross(basis_z, new_basis_x)
                    new_basis_y = new_basis_y / (np.linalg.norm(new_basis_y) + 1e-10)

                    # Update basis vectors for square-like contours
                    bp['basis_x'] = new_basis_x
                    bp['basis_y'] = new_basis_y

                # ========== Recompute bounding planes for ALL contours ==========
                for i, bp in enumerate(stream):
                    level_idx = levels_in_stream[i]
                    contour_idx = contours_in_stream[i]

                    basis_z = bp['basis_z']
                    basis_x = bp['basis_x']
                    basis_y = bp['basis_y']

                    # Get contour vertices from self.contours
                    contour_points = self.contours[level_idx][contour_idx]
                    if contour_points is None or len(contour_points) == 0:
                        continue

                    contour_points = np.array(contour_points)
                    mean = bp['mean']

                    # Final re-orthogonalization to ensure exactly 90-degree corners
                    basis_x = basis_x / (np.linalg.norm(basis_x) + 1e-10)
                    basis_y = np.cross(basis_z, basis_x)
                    basis_y = basis_y / (np.linalg.norm(basis_y) + 1e-10)

                    # Reproject contour points with orthonormal basis
                    projected_2d = np.array([[np.dot(v - mean, basis_x), np.dot(v - mean, basis_y)] for v in contour_points])
                    area = compute_polygon_area(projected_2d)

                    min_x, max_x = np.min(projected_2d[:, 0]), np.max(projected_2d[:, 0])
                    min_y, max_y = np.min(projected_2d[:, 1]), np.max(projected_2d[:, 1])
                    x_len = max_x - min_x
                    y_len = max_y - min_y

                    ratio_threshold = 2.0
                    square_like = max(x_len, y_len) / min(x_len, y_len) < ratio_threshold if min(x_len, y_len) > 1e-10 else False

                    bounding_plane_2d = np.array([
                        [min_x, min_y], [max_x, min_y],
                        [max_x, max_y], [min_x, max_y]
                    ])

                    # Optimize plane position along z-axis to minimize total distance from vertices
                    z_coords = np.array([np.dot(v - mean, basis_z) for v in contour_points])
                    optimal_z_offset = np.median(z_coords)
                    optimal_mean = mean + optimal_z_offset * basis_z

                    bounding_plane = np.array([optimal_mean + x * basis_x + y * basis_y for x, y in bounding_plane_2d])
                    projected_2d_3d = np.array([optimal_mean + x * basis_x + y * basis_y for x, y in projected_2d])

                    # Update bp_info
                    bp['basis_x'] = basis_x
                    bp['basis_y'] = basis_y
                    bp['mean'] = optimal_mean
                    bp['bounding_plane'] = bounding_plane
                    bp['projected_2d'] = projected_2d_3d
                    bp['area'] = area
                    bp['square_like'] = square_like

                    # Update contour_match and self.contours
                    preserve = getattr(self, '_contours_normalized', False)
                    new_contour, contour_match = self.find_contour_match(contour_points, bounding_plane, preserve_order=preserve)
                    bp['contour_match'] = contour_match
                    self.contours[level_idx][contour_idx] = new_contour
                    self.bounding_planes[level_idx][contour_idx] = bp

        print("  Bounding plane smoothening complete")

    def _smoothen_contours_bp_stream_mode(self):
        """
        Smooth bounding plane orientations for stream mode (after cutting).

        In stream mode: bounding_planes[stream][level]
        For each stream, interpolate basis_x for square-like contours from
        non-square-like neighbors.
        """
        num_streams = len(self.stream_bounding_planes)

        for stream_i in range(num_streams):
            bp_stream = self.stream_bounding_planes[stream_i]
            stream_len = len(bp_stream)

            if stream_len < 2:
                continue

            print(f"  Stream {stream_i}: {stream_len} levels")

            # Find reference indices (non-square-like) and smooth indices (square-like)
            reference_indices = [i for i, bp in enumerate(bp_stream)
                                 if not bp.get('square_like', False)]
            smooth_indices = [i for i, bp in enumerate(bp_stream)
                              if bp.get('square_like', False)]

            print(f"    Reference: {len(reference_indices)}, Smooth: {len(smooth_indices)}")

            if len(reference_indices) == 0 and len(smooth_indices) > 0:
                # All square-like - find most non-square-like as reference
                best_idx = 0
                best_ratio_diff = 0
                for i, bp in enumerate(bp_stream):
                    corners = bp.get('bounding_plane')
                    if corners is None or len(corners) < 4:
                        continue
                    width = np.linalg.norm(corners[1] - corners[0])
                    height = np.linalg.norm(corners[3] - corners[0])
                    if min(width, height) > 1e-10:
                        ratio = max(width, height) / min(width, height)
                        ratio_diff = abs(ratio - 1.0)
                        if ratio_diff > best_ratio_diff:
                            best_ratio_diff = ratio_diff
                            best_idx = i
                bp_stream[best_idx]['square_like'] = False
                reference_indices = [best_idx]
                smooth_indices = [i for i in smooth_indices if i != best_idx]
                print(f"    All square-like, using level {best_idx} as reference")

            # Interpolate square-like contours
            for i in smooth_indices:
                bp = bp_stream[i]
                curr_mean = bp['mean']
                basis_z = bp['basis_z']

                # Find prev reference (non-square-like)
                prev_idx = None
                prev_dist = np.inf
                for j in range(i - 1, -1, -1):
                    if j in reference_indices:
                        prev_dist = np.linalg.norm(curr_mean - bp_stream[j]['mean'])
                        prev_idx = j
                        break

                # Find next reference (non-square-like)
                next_idx = None
                next_dist = np.inf
                for j in range(i + 1, stream_len):
                    if j in reference_indices:
                        next_dist = np.linalg.norm(curr_mean - bp_stream[j]['mean'])
                        next_idx = j
                        break

                if prev_idx is None and next_idx is None:
                    continue

                # Determine target basis_x by interpolation
                if prev_idx is not None and next_idx is not None:
                    # Interpolate between prev and next
                    prev_x = bp_stream[prev_idx]['basis_x']
                    prev_x_proj = prev_x - np.dot(prev_x, basis_z) * basis_z
                    prev_x_proj = prev_x_proj / (np.linalg.norm(prev_x_proj) + 1e-10)

                    next_x = bp_stream[next_idx]['basis_x']
                    next_x_proj = next_x - np.dot(next_x, basis_z) * basis_z
                    next_x_proj = next_x_proj / (np.linalg.norm(next_x_proj) + 1e-10)

                    total_dist = prev_dist + next_dist
                    t = prev_dist / total_dist if total_dist > 1e-10 else 0.5

                    # Compute angle and interpolate
                    cos_angle = np.clip(np.dot(prev_x_proj, next_x_proj), -1, 1)
                    cross = np.cross(prev_x_proj, next_x_proj)
                    sin_angle = np.dot(cross, basis_z)
                    angle = np.arctan2(sin_angle, cos_angle)
                    interp_angle = angle * t

                    cos_t = np.cos(interp_angle)
                    sin_t = np.sin(interp_angle)
                    new_basis_x = prev_x_proj * cos_t + np.cross(basis_z, prev_x_proj) * sin_t
                elif prev_idx is not None:
                    prev_x = bp_stream[prev_idx]['basis_x']
                    new_basis_x = prev_x - np.dot(prev_x, basis_z) * basis_z
                    new_basis_x = new_basis_x / (np.linalg.norm(new_basis_x) + 1e-10)
                else:
                    next_x = bp_stream[next_idx]['basis_x']
                    new_basis_x = next_x - np.dot(next_x, basis_z) * basis_z
                    new_basis_x = new_basis_x / (np.linalg.norm(new_basis_x) + 1e-10)

                new_basis_y = np.cross(basis_z, new_basis_x)
                new_basis_y = new_basis_y / (np.linalg.norm(new_basis_y) + 1e-10)

                bp['basis_x'] = new_basis_x
                bp['basis_y'] = new_basis_y

                # Recalculate bounding plane corners with new basis
                contour_points = self.stream_contours[stream_i][i]
                mean = bp['mean']

                # Project contour to 2D with new basis
                projected_2d = np.array([
                    [np.dot(v - mean, new_basis_x), np.dot(v - mean, new_basis_y)]
                    for v in contour_points
                ])

                min_x, max_x = np.min(projected_2d[:, 0]), np.max(projected_2d[:, 0])
                min_y, max_y = np.min(projected_2d[:, 1]), np.max(projected_2d[:, 1])

                bounding_plane_2d = np.array([
                    [min_x, min_y], [max_x, min_y],
                    [max_x, max_y], [min_x, max_y]
                ])

                bounding_plane = np.array([mean + x * new_basis_x + y * new_basis_y for x, y in bounding_plane_2d])
                projected_2d_3d = np.array([mean + x * new_basis_x + y * new_basis_y for x, y in projected_2d])

                bp['bounding_plane'] = bounding_plane
                bp['projected_2d'] = projected_2d_3d

                # Update contour_match
                preserve = getattr(self, '_contours_normalized', False)
                new_contour, contour_match = self.find_contour_match(contour_points, bounding_plane, preserve_order=preserve)
                bp['contour_match'] = contour_match
                self.stream_contours[stream_i][i] = new_contour

            if len(smooth_indices) > 0:
                print(f"    Smoothed {len(smooth_indices)} square-like contours")

        # Update self.contours and self.bounding_planes to reflect changes
        self.contours = self.stream_contours
        self.bounding_planes = self.stream_bounding_planes

        print("  Bounding plane smoothening (stream mode) complete")

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

                # Snap vertices near shared cut positions to exact positions
                if hasattr(self, 'shared_cut_vertices') and len(self.shared_cut_vertices) > 0:
                    snap_threshold = 0.01  # Distance threshold for snapping
                    snapped_count = 0
                    for v_idx in range(len(aligned_contour)):
                        for shared_pos in self.shared_cut_vertices:
                            dist = np.linalg.norm(aligned_contour[v_idx] - shared_pos)
                            if dist < snap_threshold:
                                aligned_contour[v_idx] = shared_pos.copy()
                                snapped_count += 1
                                break
                    if snapped_count > 0:
                        print(f"      Snapped {snapped_count} vertices to shared cut positions")

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

    def select_stream_levels(self, error_threshold=None):
        """
        Error-based level selection BEFORE stream search.

        Groups levels by contour count, then uses greedy error-based selection
        to minimize number of levels while representing the mesh well.
        """
        # Restore from backup if available (allows re-running with different threshold)
        if hasattr(self, '_contours_backup') and self._contours_backup is not None:
            self.contours = [list(level) for level in self._contours_backup]
            self.bounding_planes = [list(level) for level in self._bounding_planes_backup]
            print("Restored contours from backup for re-selection")

        if self.contours is None or len(self.contours) < 2:
            print("Need at least 2 contour levels")
            return

        if self.bounding_planes is None or len(self.bounding_planes) < 2:
            print("Need bounding planes - run find_contours first")
            return

        # Backup original contours (before selection)
        if not hasattr(self, '_contours_backup') or self._contours_backup is None:
            self._contours_backup = [list(level) for level in self.contours]
            self._bounding_planes_backup = [list(level) for level in self.bounding_planes]

        num_levels = len(self.contours)
        print(f"\n=== Select Stream Levels ===")
        print(f"Total levels: {num_levels}")

        # Compute muscle scale for error threshold
        all_means = []
        for level_bps in self.bounding_planes:
            for bp in level_bps:
                all_means.append(bp['mean'])
        if len(all_means) > 1:
            all_means = np.array(all_means)
            muscle_length = np.linalg.norm(all_means.max(axis=0) - all_means.min(axis=0))
        else:
            muscle_length = 1.0

        # Default error threshold: 2% of muscle length
        if error_threshold is None:
            error_threshold = getattr(self, 'level_select_error_threshold', 0.02) * muscle_length

        print(f"Error threshold: {error_threshold:.6f} ({error_threshold/muscle_length*100:.1f}% of muscle length)")

        # Group levels by contour count
        groups = []
        current_group = {'k': len(self.contours[0]), 'levels': [0]}

        for level_idx in range(1, num_levels):
            k = len(self.contours[level_idx])
            if k == current_group['k']:
                current_group['levels'].append(level_idx)
            else:
                groups.append(current_group)
                current_group = {'k': k, 'levels': [level_idx]}
        groups.append(current_group)

        print(f"Groups by contour count:")
        for i, g in enumerate(groups):
            print(f"  Group {i}: k={g['k']}, {len(g['levels'])} levels")

        # Helper function to compute centroid interpolation error
        def compute_interpolation_error(level_idx, prev_idx, next_idx):
            from scipy.optimize import linear_sum_assignment

            max_error = 0
            curr_means = [bp['mean'] for bp in self.bounding_planes[level_idx]]
            prev_means = [bp['mean'] for bp in self.bounding_planes[prev_idx]]
            next_means = [bp['mean'] for bp in self.bounding_planes[next_idx]]

            curr_count = len(curr_means)
            prev_count = len(prev_means)
            next_count = len(next_means)

            # Use optimal assignment when counts match, otherwise use closest
            if curr_count == prev_count and prev_count > 1:
                # Optimal assignment between curr and prev
                cost_matrix = np.zeros((curr_count, prev_count))
                for i in range(curr_count):
                    for j in range(prev_count):
                        cost_matrix[i, j] = np.linalg.norm(np.array(curr_means[i]) - np.array(prev_means[j]))
                _, prev_assignment = linear_sum_assignment(cost_matrix)
            else:
                prev_assignment = None

            if curr_count == next_count and next_count > 1:
                # Optimal assignment between curr and next
                cost_matrix = np.zeros((curr_count, next_count))
                for i in range(curr_count):
                    for j in range(next_count):
                        cost_matrix[i, j] = np.linalg.norm(np.array(curr_means[i]) - np.array(next_means[j]))
                _, next_assignment = linear_sum_assignment(cost_matrix)
            else:
                next_assignment = None

            for contour_idx in range(curr_count):
                actual_mean = np.array(curr_means[contour_idx])

                # Get matched prev mean
                if prev_assignment is not None:
                    prev_mean = np.array(prev_means[prev_assignment[contour_idx]])
                elif len(prev_means) > 0:
                    prev_dists = [np.linalg.norm(actual_mean - np.array(m)) for m in prev_means]
                    prev_mean = np.array(prev_means[np.argmin(prev_dists)])
                else:
                    prev_mean = actual_mean

                # Get matched next mean
                if next_assignment is not None:
                    next_mean = np.array(next_means[next_assignment[contour_idx]])
                elif len(next_means) > 0:
                    next_dists = [np.linalg.norm(actual_mean - np.array(m)) for m in next_means]
                    next_mean = np.array(next_means[np.argmin(next_dists)])
                else:
                    next_mean = actual_mean

                prev_scalar = self.bounding_planes[prev_idx][0].get('scalar_value', prev_idx)
                next_scalar = self.bounding_planes[next_idx][0].get('scalar_value', next_idx)
                actual_scalar = self.bounding_planes[level_idx][contour_idx].get('scalar_value', level_idx)

                if abs(next_scalar - prev_scalar) > 1e-10:
                    t = (actual_scalar - prev_scalar) / (next_scalar - prev_scalar)
                else:
                    t = 0.5
                t = np.clip(t, 0, 1)

                interpolated = (1 - t) * prev_mean + t * next_mean
                error = np.linalg.norm(actual_mean - interpolated)
                max_error = max(max_error, error)
            return max_error

        # ========== Step 1: Identify MUST-USE levels ==========
        # At contour count transitions, use the level with FEWER contours
        # Example: [2,2,2,1,1,1] -> index 3 (first with 1)
        # Example: [1,1,1,2,2,2] -> index 2 (last with 1)
        # Example: [2,2,1,1,2,2] -> index 2 (first with 1) and index 3 (last with 1)

        must_use_levels = set()
        must_use_levels.add(0)  # Origin
        must_use_levels.add(num_levels - 1)  # Insertion

        contour_counts = [len(self.contours[i]) for i in range(num_levels)]
        print(f"Contour counts: {contour_counts}")

        for i in range(len(groups) - 1):
            curr_group = groups[i]
            next_group = groups[i + 1]
            curr_k = curr_group['k']
            next_k = next_group['k']

            if curr_k < next_k:
                # Merge point: curr has fewer contours, use last of curr group
                must_use_levels.add(curr_group['levels'][-1])
                print(f"  Must-use level {curr_group['levels'][-1]} (last of group with k={curr_k}, before split to k={next_k})")
            else:
                # Split point: next has fewer contours, use first of next group
                must_use_levels.add(next_group['levels'][0])
                print(f"  Must-use level {next_group['levels'][0]} (first of group with k={next_k}, after merge from k={curr_k})")

        print(f"Must-use levels: {sorted(must_use_levels)}")

        # ========== Step 2: Error-based selection with must-use as anchors ==========
        all_selected_levels = set(must_use_levels)

        # Greedy: add levels with max interpolation error until all errors < threshold
        while True:
            max_error = 0
            max_error_level = None
            selected_sorted = sorted(all_selected_levels)

            for gap_idx in range(len(selected_sorted) - 1):
                prev_idx = selected_sorted[gap_idx]
                next_idx = selected_sorted[gap_idx + 1]

                # Check all levels between prev and next
                for level_idx in range(prev_idx + 1, next_idx):
                    if level_idx in all_selected_levels:
                        continue

                    error = compute_interpolation_error(level_idx, prev_idx, next_idx)
                    if error > max_error:
                        max_error = error
                        max_error_level = level_idx

            if max_error <= error_threshold or max_error_level is None:
                break

            all_selected_levels.add(max_error_level)
            print(f"  Added level {max_error_level} (error={max_error:.6f})")

        selected_levels_list = sorted(all_selected_levels)
        print(f"\nSelected {len(selected_levels_list)} levels from {num_levels} total:")
        print(f"  {selected_levels_list}")

        self.selected_stream_levels = selected_levels_list
        self.stream_level_groups = groups

        # Actually remove non-selected levels
        new_contours = []
        new_bounding_planes = []

        for level_idx in selected_levels_list:
            new_contours.append(self.contours[level_idx])
            new_bounding_planes.append(self.bounding_planes[level_idx])

        self.contours = new_contours
        self.bounding_planes = new_bounding_planes

        self.draw_contour_stream = [[True] * len(level_contours) for level_contours in self.contours]

        print(f"Contours updated: now {len(self.contours)} levels")

    def _prepare_manual_cut_data(self, muscle_name=None):
        """
        Prepare data for manual cutting window.

        Called when contour count changes along levels and manual cutting is needed.
        Sets up _manual_cut_data with target/source contours projected to 2D.
        """
        num_levels = len(self.contours)
        origin_count = len(self.contours[0])
        insertion_count = len(self.contours[-1])
        max_stream_count = max(origin_count, insertion_count)

        # Determine processing direction (from larger count side)
        if origin_count >= insertion_count:
            # Process origin → insertion
            level_order = list(range(num_levels))
            process_forward = True
        else:
            # Process insertion → origin
            level_order = list(range(num_levels - 1, -1, -1))
            process_forward = False

        # Find the first level where contour count changes (decreases)
        # Source = level before change (more contours)
        # Target = level where change happens (fewer contours, merged)
        source_level = None
        target_level = None

        prev_level = level_order[0]
        prev_count = len(self.contours[prev_level])

        for i in range(1, len(level_order)):
            curr_level = level_order[i]
            curr_count = len(self.contours[curr_level])

            if curr_count < prev_count:
                # Found a transition where count decreases
                source_level = prev_level
                target_level = curr_level
                break

            prev_level = curr_level
            prev_count = curr_count

        if source_level is None or target_level is None:
            print(f"No transition found where contour count decreases")
            return

        source_count = len(self.contours[source_level])
        target_count = len(self.contours[target_level])

        print(f"\n=== Preparing Manual Cut ===")
        print(f"Source level: {source_level} ({source_count} contours)")
        print(f"Target level: {target_level} ({target_count} contours)")

        # For now, handle the case where source has 2 and target has 1
        if target_count != 1 or source_count != 2:
            print(f"Manual cutting currently only supports 2 sources → 1 target")
            print(f"Falling back to automatic cutting")
            return

        # Get target contour and its bounding plane (the merged contour to cut)
        target_contour = np.array(self.contours[target_level][0])
        target_bp = self.bounding_planes[target_level][0]

        # Get source contours and their bounding planes (the separate contours as reference)
        source_contours = [np.array(self.contours[source_level][i]) for i in range(source_count)]
        source_bps = [self.bounding_planes[source_level][i] for i in range(source_count)]

        # Project target contour to 2D (using its own bounding plane)
        target_mean = target_bp['mean']
        target_x = target_bp['basis_x']
        target_y = target_bp['basis_y']

        target_2d = np.array([
            [np.dot(v - target_mean, target_x), np.dot(v - target_mean, target_y)]
            for v in target_contour
        ])

        # Project source contours to target's 2D plane
        source_2d_list = []
        for i, (src_contour, src_bp) in enumerate(zip(source_contours, source_bps)):
            src_mean = src_bp['mean']
            src_x = src_bp['basis_x']
            src_y = src_bp['basis_y']

            # Project source to its own 2D plane first
            src_2d_own = np.array([
                [np.dot(v - src_mean, src_x), np.dot(v - src_mean, src_y)]
                for v in src_contour
            ])

            # Compute initial translation (project source mean onto target plane)
            src_to_target = src_mean - target_mean
            dist_along_z = np.dot(src_to_target, target_bp['basis_z'])
            src_mean_projected = src_mean - dist_along_z * target_bp['basis_z']
            init_tx = np.dot(src_mean_projected - target_mean, target_x)
            init_ty = np.dot(src_mean_projected - target_mean, target_y)

            # Compute initial rotation (align source basis_x with target basis_x)
            src_x_on_target = src_x - np.dot(src_x, target_bp['basis_z']) * target_bp['basis_z']
            src_x_norm = np.linalg.norm(src_x_on_target)
            if src_x_norm > 1e-10:
                src_x_on_target = src_x_on_target / src_x_norm
                src_x_2d = np.array([np.dot(src_x_on_target, target_x), np.dot(src_x_on_target, target_y)])
                init_theta = np.arctan2(src_x_2d[1], src_x_2d[0])
            else:
                init_theta = 0.0

            # Apply rotation to source 2D shape
            cos_t = np.cos(init_theta)
            sin_t = np.sin(init_theta)
            rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            src_2d_rotated = src_2d_own @ rot.T

            # Translate to initial position
            src_2d_final = src_2d_rotated + np.array([init_tx, init_ty])
            source_2d_list.append(src_2d_final)

        # Store all data for the manual cutting window
        self._manual_cut_data = {
            'muscle_name': muscle_name,
            'target_level': target_level,
            'source_level': source_level,
            'target_contour': target_contour,
            'target_bp': target_bp,
            'target_2d': target_2d,
            'source_contours': source_contours,
            'source_bps': source_bps,
            'source_2d_list': source_2d_list,
            'stream_indices': list(range(source_count)),
            'process_forward': process_forward,
        }

        # Set pending flag to open the window
        self._manual_cut_pending = True
        self._manual_cut_line = None

        print(f"Manual cutting window ready - draw a cutting line")

    def _apply_manual_cut(self):
        """
        Apply the manually drawn cutting line to split the target contour.

        Called when user clicks OK in the manual cutting window.
        Computes intersection points, adds new vertices, and splits the contour.
        """
        if self._manual_cut_data is None or self._manual_cut_line is None:
            print("No manual cut data or line available")
            return None

        line_start, line_end = self._manual_cut_line
        target_2d = self._manual_cut_data['target_2d']
        target_contour = self._manual_cut_data['target_contour']
        target_bp = self._manual_cut_data['target_bp']

        n_verts = len(target_2d)

        # Find intersections between the cutting line and contour edges
        intersections = []  # List of (edge_index, t_param, intersection_point_2d)

        for i in range(n_verts):
            p1 = target_2d[i]
            p2 = target_2d[(i + 1) % n_verts]

            # Line segment intersection
            # Line: line_start + s * (line_end - line_start)
            # Edge: p1 + t * (p2 - p1)
            line_dir = np.array(line_end) - np.array(line_start)
            edge_dir = p2 - p1

            # Cross product for 2D (scalar)
            cross = line_dir[0] * edge_dir[1] - line_dir[1] * edge_dir[0]

            if abs(cross) < 1e-10:
                continue  # Parallel lines

            diff = p1 - np.array(line_start)
            # t = edge parameter (we want t in [0,1] for intersection on edge)
            # Using standard line intersection formula
            t = (diff[0] * line_dir[1] - diff[1] * line_dir[0]) / cross

            # Check if intersection is within edge (t in [0, 1])
            if 0 < t < 1:  # Strictly inside edge (not at vertices)
                intersection_2d = p1 + t * edge_dir
                intersections.append((i, t, intersection_2d))

        if len(intersections) != 2:
            print(f"Expected 2 intersections, got {len(intersections)}")
            return None

        # Sort intersections by edge index
        intersections.sort(key=lambda x: x[0])

        (edge1_idx, t1, int1_2d), (edge2_idx, t2, int2_2d) = intersections

        # Convert intersection points from 2D back to 3D
        target_mean = target_bp['mean']
        target_x = target_bp['basis_x']
        target_y = target_bp['basis_y']

        int1_3d = target_mean + int1_2d[0] * target_x + int1_2d[1] * target_y
        int2_3d = target_mean + int2_2d[0] * target_x + int2_2d[1] * target_y

        # Build two contour pieces
        # Piece 0: from int1 → edge1+1 → ... → edge2 → int2 → int1 (closed)
        # Piece 1: from int2 → edge2+1 → ... → edge1 → int1 → int2 (closed)

        piece0_verts = [int1_3d]
        for i in range(edge1_idx + 1, edge2_idx + 1):
            piece0_verts.append(target_contour[i])
        piece0_verts.append(int2_3d)

        piece1_verts = [int2_3d]
        for i in range(edge2_idx + 1, n_verts):
            piece1_verts.append(target_contour[i])
        for i in range(0, edge1_idx + 1):
            piece1_verts.append(target_contour[i])
        piece1_verts.append(int1_3d)

        piece0 = np.array(piece0_verts)
        piece1 = np.array(piece1_verts)

        print(f"Manual cut: piece0 has {len(piece0)} vertices, piece1 has {len(piece1)} vertices")

        # Determine which piece corresponds to which source contour
        # Simple 3D centroid distance matching
        source_contours = self._manual_cut_data['source_contours']

        # Compute 3D centroids
        piece0_centroid = piece0.mean(axis=0)
        piece1_centroid = piece1.mean(axis=0)
        src0_centroid = source_contours[0].mean(axis=0)
        src1_centroid = source_contours[1].mean(axis=0)

        # Match by distance
        dist_00 = np.linalg.norm(piece0_centroid - src0_centroid)
        dist_01 = np.linalg.norm(piece0_centroid - src1_centroid)
        dist_10 = np.linalg.norm(piece1_centroid - src0_centroid)
        dist_11 = np.linalg.norm(piece1_centroid - src1_centroid)

        print(f"  Piece centroids: {piece0_centroid}, {piece1_centroid}")
        print(f"  Source centroids: {src0_centroid}, {src1_centroid}")
        print(f"  Distances: d00={dist_00:.4f}, d01={dist_01:.4f}, d10={dist_10:.4f}, d11={dist_11:.4f}")

        if dist_00 + dist_11 < dist_01 + dist_10:
            cut_contours = [piece0, piece1]
            print(f"  Matching: piece0 → src0, piece1 → src1")
        else:
            cut_contours = [piece1, piece0]
            print(f"  Matching: piece0 → src1, piece1 → src0 (swapped)")

        # Store the result
        self._manual_cut_data['cut_result'] = cut_contours
        self._manual_cut_data['intersection_3d'] = (int1_3d, int2_3d)

        # Register shared cut edge vertices globally
        # These vertices are shared between the two pieces and must be preserved exactly
        if not hasattr(self, 'shared_cut_vertices'):
            self.shared_cut_vertices = []
        self.shared_cut_vertices.append(int1_3d.copy())
        self.shared_cut_vertices.append(int2_3d.copy())
        print(f"  Registered {len(self.shared_cut_vertices)} shared cut edge vertices")

        # Store cut edge indices for each piece
        # piece0: cut edges at indices 0 (int1) and last-1 (int2, before closing)
        # piece1: cut edges at indices 0 (int2) and last-1 (int1, before closing)
        self._manual_cut_data['cut_edge_indices'] = {
            0: [0, len(piece0) - 1],  # indices of shared vertices in piece0
            1: [0, len(piece1) - 1],  # indices of shared vertices in piece1
        }

        return cut_contours

    def _cancel_manual_cut(self):
        """Cancel manual cutting and reset state."""
        self._manual_cut_pending = False
        self._manual_cut_data = None
        self._manual_cut_line = None
        print("Manual cutting cancelled")

    def _compute_cut_preview(self):
        """
        Compute preview of how the contour would be cut by the current line.
        Returns two 2D contour pieces for visualization.
        """
        if self._manual_cut_data is None or self._manual_cut_line is None:
            return None, None

        line_start, line_end = self._manual_cut_line
        target_2d = self._manual_cut_data['target_2d']

        n_verts = len(target_2d)

        # Find intersections
        intersections = []

        for i in range(n_verts):
            p1 = target_2d[i]
            p2 = target_2d[(i + 1) % n_verts]

            line_dir = np.array(line_end) - np.array(line_start)
            edge_dir = p2 - p1

            cross = line_dir[0] * edge_dir[1] - line_dir[1] * edge_dir[0]

            if abs(cross) < 1e-10:
                continue

            diff = p1 - np.array(line_start)
            # t = edge parameter (we want t in [0,1] for intersection on edge)
            t = (diff[0] * line_dir[1] - diff[1] * line_dir[0]) / cross

            if 0 < t < 1:
                intersection_2d = p1 + t * edge_dir
                intersections.append((i, t, intersection_2d))

        if len(intersections) != 2:
            return None, None

        intersections.sort(key=lambda x: x[0])
        (edge1_idx, t1, int1_2d), (edge2_idx, t2, int2_2d) = intersections

        # Build 2D pieces for preview
        piece0_2d = [int1_2d]
        for i in range(edge1_idx + 1, edge2_idx + 1):
            piece0_2d.append(target_2d[i])
        piece0_2d.append(int2_2d)

        piece1_2d = [int2_2d]
        for i in range(edge2_idx + 1, n_verts):
            piece1_2d.append(target_2d[i])
        for i in range(0, edge1_idx + 1):
            piece1_2d.append(target_2d[i])
        piece1_2d.append(int1_2d)

        return np.array(piece0_2d), np.array(piece1_2d)

    def cut_streams(self, cut_method='bp', muscle_name=None):
        """
        Step 1: Pre-cut all contours to max stream count.

        - Determines max_stream_count = max(origin_count, insertion_count)
        - Cuts all contours to have max_stream_count per level
        - Tracks which streams came from same original contour (stream_groups)
        - Smooths z, x, bp for each stream

        After this, self.stream_contours[stream_i][level_i] and
        self.stream_bounding_planes[stream_i][level_i] are available.

        Args:
            cut_method: 'bp' (default) for bounding plane transform optimization,
                       'mesh' for topology-aware cutting at contour corners,
                       'area' for position-based equal-area cutting
            muscle_name: Name of the muscle (for visualization file naming)
        """
        # Store muscle name for visualization
        if muscle_name is not None:
            self._muscle_name = muscle_name

        if self.contours is None or len(self.contours) < 2:
            print("Need at least 2 contour levels")
            return

        if self.bounding_planes is None or len(self.bounding_planes) < 2:
            print("Need bounding planes - run find_contours first")
            return

        num_levels = len(self.contours)
        origin_count = len(self.contours[0])
        insertion_count = len(self.contours[-1])
        max_stream_count = max(origin_count, insertion_count)

        print(f"\n=== Cut Streams ===")
        print(f"Levels: {num_levels}")
        print(f"Origin count: {origin_count}, Insertion count: {insertion_count}")
        print(f"Max stream count: {max_stream_count}")
        print(f"Cut method: {cut_method}")

        # Check if manual cutting is needed (contour count changes along levels)
        # Only for 'bp' method with max_stream_count >= 2
        if cut_method == 'bp' and max_stream_count >= 2 and origin_count != insertion_count:
            # Check if we need to open manual cutting window
            if not self._manual_cut_pending and self._manual_cut_data is None:
                # Prepare data for manual cutting and open window
                self._prepare_manual_cut_data(muscle_name)
                return  # Wait for user to draw cutting line
            elif self._manual_cut_pending:
                # Still waiting for user to finish drawing
                print("Manual cutting in progress - waiting for user to confirm")
                return

        # Get contour counts per level
        contour_counts = [len(self.contours[i]) for i in range(num_levels)]
        print(f"Contour counts: {contour_counts}")

        # Initialize stream structure: [stream_i][level_i]
        stream_contours = [[] for _ in range(max_stream_count)]
        stream_bounding_planes = [[] for _ in range(max_stream_count)]

        # Track which streams came from same original contour per level
        # stream_groups[level_i] = [[stream indices from same original], ...]
        stream_groups = []

        # Determine processing direction (from larger count end)
        if origin_count >= insertion_count:
            level_order = list(range(num_levels))
            print("Processing: origin → insertion")
        else:
            level_order = list(range(num_levels - 1, -1, -1))
            print("Processing: insertion → origin")

        # Process first level (no cutting needed if it has max_stream_count)
        first_level = level_order[0]
        first_count = contour_counts[first_level]

        if first_count == max_stream_count:
            # No cutting needed - assign each contour to its own stream
            for stream_i in range(max_stream_count):
                stream_contours[stream_i].append(self.contours[first_level][stream_i])
                stream_bounding_planes[stream_i].append(self.bounding_planes[first_level][stream_i])
            stream_groups.append([[i] for i in range(max_stream_count)])
            print(f"  Level {first_level}: {first_count} contours (no cut needed)")
        else:
            # Need to cut first level contours
            print(f"  Level {first_level}: {first_count} contours → cutting to {max_stream_count}")
            # This shouldn't happen if we start from the end with max_stream_count
            # But handle it anyway by distributing streams among contours
            groups = []
            streams_per_contour = max_stream_count // first_count
            remainder = max_stream_count % first_count
            stream_idx = 0
            for contour_i in range(first_count):
                group = []
                n_streams = streams_per_contour + (1 if contour_i < remainder else 0)
                for _ in range(n_streams):
                    stream_contours[stream_idx].append(self.contours[first_level][contour_i])
                    stream_bounding_planes[stream_idx].append(self.bounding_planes[first_level][contour_i].copy())
                    group.append(stream_idx)
                    stream_idx += 1
                groups.append(group)
            stream_groups.append(groups)

        # Track which stream combinations have been cut (for first vs propagated division)
        cut_stream_combos = set()

        # Process remaining levels
        for level_i_idx in range(1, len(level_order)):
            level_i = level_order[level_i_idx]
            prev_level = level_order[level_i_idx - 1]
            curr_count = contour_counts[level_i]

            if curr_count == max_stream_count:
                # No cutting needed - match by distance using optimal assignment
                from scipy.optimize import linear_sum_assignment

                prev_means = [stream_bounding_planes[s][-1]['mean'] for s in range(max_stream_count)]
                curr_means = [self.bounding_planes[level_i][c]['mean'] for c in range(curr_count)]

                # Build cost matrix (distance from each stream to each contour)
                cost_matrix = np.zeros((max_stream_count, curr_count))
                for stream_i in range(max_stream_count):
                    for contour_i in range(curr_count):
                        cost_matrix[stream_i, contour_i] = np.linalg.norm(prev_means[stream_i] - curr_means[contour_i])

                # Optimal assignment using Hungarian algorithm
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for stream_i, contour_i in zip(row_ind, col_ind):
                    stream_contours[stream_i].append(self.contours[level_i][contour_i])
                    stream_bounding_planes[stream_i].append(self.bounding_planes[level_i][contour_i])

                stream_groups.append([[i] for i in range(max_stream_count)])
                print(f"  Level {level_i}: {curr_count} contours (no cut needed)")

            else:
                # Need to cut contours
                print(f"  Level {level_i}: {curr_count} contours → cutting to {max_stream_count}")

                # IMPORTANT: Save previous level's contours BEFORE any cuts
                # Otherwise stream_contours[s][-1] changes as we process contours
                prev_level_contours = [stream_contours[s][-1].copy() if hasattr(stream_contours[s][-1], 'copy') else np.array(stream_contours[s][-1]) for s in range(max_stream_count)]
                prev_level_bps = [stream_bounding_planes[s][-1].copy() for s in range(max_stream_count)]

                # Debug: show prev level contour sizes
                print(f"  [DEBUG] prev_level_contours sizes: {[len(c) for c in prev_level_contours]}")

                # Find which streams map to which contours (by distance)
                prev_means = [prev_level_bps[s]['mean'] for s in range(max_stream_count)]
                curr_means = [self.bounding_planes[level_i][c]['mean'] for c in range(curr_count)]

                # Assign each stream to closest contour
                stream_to_contour = []
                for stream_i in range(max_stream_count):
                    dists = [np.linalg.norm(prev_means[stream_i] - cm) for cm in curr_means]
                    closest_contour = np.argmin(dists)
                    stream_to_contour.append(closest_contour)

                # Group streams by which contour they map to
                contour_to_streams = [[] for _ in range(curr_count)]
                for stream_i, contour_i in enumerate(stream_to_contour):
                    contour_to_streams[contour_i].append(stream_i)

                # Build stream_groups for this level
                groups = []
                for contour_i in range(curr_count):
                    if contour_to_streams[contour_i]:
                        groups.append(contour_to_streams[contour_i])
                stream_groups.append(groups)

                # Cut each contour that has multiple streams
                for contour_i in range(curr_count):
                    streams_for_contour = contour_to_streams[contour_i]
                    if len(streams_for_contour) == 0:
                        continue
                    elif len(streams_for_contour) == 1:
                        # No cut needed
                        stream_i = streams_for_contour[0]
                        stream_contours[stream_i].append(self.contours[level_i][contour_i])
                        stream_bounding_planes[stream_i].append(self.bounding_planes[level_i][contour_i])
                    else:
                        # Cut this contour into len(streams_for_contour) pieces
                        target_contour = self.contours[level_i][contour_i]
                        target_bp = self.bounding_planes[level_i][contour_i]

                        # Get reference means from previous level streams
                        ref_means = [prev_means[s] for s in streams_for_contour]

                        # Project reference means onto target plane
                        target_mean = target_bp['mean']
                        target_z = target_bp['basis_z']
                        projected_refs = [m - np.dot(m - target_mean, target_z) * target_z for m in ref_means]

                        # Cut contour using selected method
                        if cut_method == 'bp':
                            # Get source contours and bounding planes from previous level
                            # Use saved prev_level data to avoid getting cut results from earlier contours at this level
                            source_contours = [prev_level_contours[s] for s in streams_for_contour]
                            source_bps = [prev_level_bps[s] for s in streams_for_contour]

                            # Check if this is first division for this stream combination
                            stream_combo = tuple(sorted(streams_for_contour))
                            is_first_division = stream_combo not in cut_stream_combos
                            cut_stream_combos.add(stream_combo)

                            # Check if we have a saved manual cut result for first division
                            if is_first_division and self._manual_cut_data is not None and 'cut_result' in self._manual_cut_data:
                                # Use the manual cut result
                                cut_contours = self._manual_cut_data['cut_result']
                                cutting_info = None
                                print(f"  [BP Transform] Using manual cut result for first division")
                                print(f"  [BP Transform] streams_for_contour = {streams_for_contour}")
                                print(f"  [BP Transform] cut_contours[0] will go to stream {streams_for_contour[0]}")
                                print(f"  [BP Transform] cut_contours[1] will go to stream {streams_for_contour[1]}")
                                # Clear manual cut data after using it
                                self._manual_cut_data = None
                            else:
                                cut_contours, cutting_info = self._cut_contour_bp_transform(
                                    target_contour, target_bp, source_contours, source_bps, streams_for_contour,
                                    is_first_division=is_first_division
                                )
                        elif cut_method == 'mesh':
                            cut_contours = self._cut_contour_mesh_aware(
                                target_contour, target_bp, projected_refs, streams_for_contour
                            )
                            cutting_info = None
                        else:
                            cut_contours = self._cut_contour_for_streams(
                                target_contour, target_bp, projected_refs, streams_for_contour
                            )
                            cutting_info = None

                        # Assign cut pieces to streams by distance matching
                        # For each stream, find the cut piece closest to its previous contour
                        cut_centroids = [np.mean(c, axis=0) for c in cut_contours]
                        prev_centroids = [np.mean(stream_contours[s][-1], axis=0) for s in streams_for_contour]

                        # Greedy matching: each stream gets the closest unassigned piece
                        used_pieces = set()
                        for stream_i in streams_for_contour:
                            prev_centroid = np.mean(stream_contours[stream_i][-1], axis=0)

                            # Find closest unused piece
                            best_idx = None
                            best_dist = float('inf')
                            for idx, cut_centroid in enumerate(cut_centroids):
                                if idx in used_pieces:
                                    continue
                                dist = np.linalg.norm(cut_centroid - prev_centroid)
                                if dist < best_dist:
                                    best_dist = dist
                                    best_idx = idx

                            used_pieces.add(best_idx)
                            cut_contour = cut_contours[best_idx]

                            # Debug: warn about small cut contours
                            if len(cut_contour) <= 5:
                                print(f"  [WARNING] Level {level_i}: stream {stream_i} got cut contour with only {len(cut_contour)} vertices!")

                            # Create new bounding plane for cut piece
                            _, new_bp = self.save_bounding_planes(
                                cut_contour,
                                target_bp['scalar_value'],
                                prev_bounding_plane=stream_bounding_planes[stream_i][-1]
                            )
                            new_bp['is_cut'] = True

                            stream_contours[stream_i].append(cut_contour)
                            stream_bounding_planes[stream_i].append(new_bp)

        # If we processed in reverse order, reverse the results
        if origin_count < insertion_count:
            for stream_i in range(max_stream_count):
                stream_contours[stream_i] = stream_contours[stream_i][::-1]
                stream_bounding_planes[stream_i] = stream_bounding_planes[stream_i][::-1]
            stream_groups = stream_groups[::-1]

        # Store results
        self.stream_contours = stream_contours
        self.stream_bounding_planes = stream_bounding_planes
        self.stream_groups = stream_groups
        self.max_stream_count = max_stream_count

        print(f"Created {max_stream_count} streams, each with {len(stream_contours[0])} levels")

        # Bounding planes are computed naturally from cut contour vertices
        # User can apply z, x, bp smoothening manually using the buttons

        # Update visualization: convert to [stream_i][level_i] format
        self.contours = self.stream_contours
        self.bounding_planes = self.stream_bounding_planes
        self.draw_contour_stream = [[True] * len(self.stream_contours[0]) for _ in range(max_stream_count)]

        print("Cut streams complete")

    def select_levels(self, error_threshold=None):
        """
        Step 2: Error-based level selection for cut streams.

        - Each stream can select different levels (from originally separate contours)
        - Streams from same original contour must select same levels
        - Final level count must be same across all streams
        - Must-use at merge points still applies
        """
        if not hasattr(self, 'stream_contours') or self.stream_contours is None:
            print("Run cut_streams first")
            return

        max_stream_count = self.max_stream_count
        num_levels = len(self.stream_contours[0])

        print(f"\n=== Select Levels ===")
        print(f"Streams: {max_stream_count}, Levels: {num_levels}")

        # Compute muscle length for error threshold
        first_mean = self.stream_bounding_planes[0][0]['mean']
        last_mean = self.stream_bounding_planes[0][-1]['mean']
        muscle_length = np.linalg.norm(last_mean - first_mean)

        if error_threshold is None:
            error_threshold = getattr(self, 'level_select_error_threshold', 0.005) * muscle_length
        print(f"Error threshold: {error_threshold:.6f} ({error_threshold/muscle_length*100:.1f}% of muscle length)")

        # Identify original contour counts per level
        original_counts = []
        for level_i in range(num_levels):
            # Count unique groups at this level
            groups = self.stream_groups[level_i]
            original_counts.append(len(groups))
        print(f"Original counts: {original_counts}")

        # ========== Step 1: Identify MUST-USE levels ==========
        # At contour count transitions, use the level with FEWER contours
        must_use_levels = set()
        must_use_levels.add(0)  # Origin
        must_use_levels.add(num_levels - 1)  # Insertion

        # Find first dividing position (first merge where count decreases)
        first_dividing_level = None
        for i in range(num_levels - 1):
            if original_counts[i] > original_counts[i + 1]:
                first_dividing_level = i + 1  # First merged contour after division
                break

        for i in range(num_levels - 1):
            if original_counts[i] != original_counts[i + 1]:
                if original_counts[i] < original_counts[i + 1]:
                    must_use_levels.add(i)  # i has fewer
                    print(f"  Must-use level {i} (fewer contours before split)")
                else:
                    must_use_levels.add(i + 1)  # i+1 has fewer
                    print(f"  Must-use level {i+1} (fewer contours after merge)")

        # Enforce first dividing position merged contour
        if first_dividing_level is not None:
            must_use_levels.add(first_dividing_level)
            print(f"  Must-use level {first_dividing_level} (first dividing position - enforced)")

        print(f"Must-use levels: {sorted(must_use_levels)}")

        # ========== Step 2: Per-stream error-based selection ==========
        # For each stream, select levels independently where allowed

        # Helper: compute interpolation error for a stream at a level
        def compute_stream_error(stream_i, level_i, prev_level, next_level):
            bp_actual = self.stream_bounding_planes[stream_i][level_i]
            bp_prev = self.stream_bounding_planes[stream_i][prev_level]
            bp_next = self.stream_bounding_planes[stream_i][next_level]

            actual_mean = bp_actual['mean']
            prev_mean = bp_prev['mean']
            next_mean = bp_next['mean']

            # Interpolation parameter
            prev_scalar = bp_prev.get('scalar_value', prev_level)
            next_scalar = bp_next.get('scalar_value', next_level)
            actual_scalar = bp_actual.get('scalar_value', level_i)

            if abs(next_scalar - prev_scalar) > 1e-10:
                t = (actual_scalar - prev_scalar) / (next_scalar - prev_scalar)
            else:
                t = 0.5
            t = np.clip(t, 0, 1)

            interpolated = (1 - t) * prev_mean + t * next_mean
            return np.linalg.norm(actual_mean - interpolated)

        # Group levels by original count
        # Levels with same original count in consecutive range form a region
        regions = []
        region_start = 0
        for i in range(1, num_levels):
            if original_counts[i] != original_counts[region_start]:
                regions.append({
                    'start': region_start,
                    'end': i - 1,
                    'count': original_counts[region_start]
                })
                region_start = i
        regions.append({
            'start': region_start,
            'end': num_levels - 1,
            'count': original_counts[region_start]
        })

        print(f"Regions: {regions}")

        # For each stream, maintain selected levels
        stream_selected = [set(must_use_levels) for _ in range(max_stream_count)]

        # Process each region
        for region in regions:
            start, end = region['start'], region['end']
            orig_count = region['count']

            if end - start < 1:
                continue  # Region has only 1-2 levels, skip

            # Check if this region has merged contours (orig_count < max_stream_count)
            is_merged = orig_count < max_stream_count

            if is_merged:
                # All streams must select same levels in this region
                # Use combined error across all streams
                region_selected = set()
                for level_i in range(start, end + 1):
                    if level_i in must_use_levels:
                        region_selected.add(level_i)

                # Greedy selection using max error across streams
                while True:
                    max_error = 0
                    max_error_level = None
                    selected_sorted = sorted(region_selected)

                    for gap_idx in range(len(selected_sorted) - 1):
                        prev_idx = selected_sorted[gap_idx]
                        next_idx = selected_sorted[gap_idx + 1]

                        for level_i in range(prev_idx + 1, next_idx):
                            if level_i in region_selected:
                                continue
                            if level_i < start or level_i > end:
                                continue

                            # Max error across all streams
                            error = max(compute_stream_error(s, level_i, prev_idx, next_idx)
                                       for s in range(max_stream_count))
                            if error > max_error:
                                max_error = error
                                max_error_level = level_i

                    if max_error <= error_threshold or max_error_level is None:
                        break

                    region_selected.add(max_error_level)
                    print(f"  Region [{start}-{end}] (merged): added level {max_error_level} (error={max_error:.6f})")

                # Apply to all streams
                for s in range(max_stream_count):
                    stream_selected[s].update(region_selected)

            else:
                # Non-merged region: originally separate contours
                # Each stream selects independently based on its own error
                for stream_i in range(max_stream_count):
                    region_selected = set()
                    for level_i in range(start, end + 1):
                        if level_i in must_use_levels:
                            region_selected.add(level_i)

                    # Add start boundary as anchor
                    region_selected.add(start)
                    # Use first dividing level as end anchor if it's right after this region
                    if first_dividing_level is not None and first_dividing_level == end + 1:
                        region_selected.add(first_dividing_level)
                    else:
                        region_selected.add(end)

                    # Greedy selection for this stream
                    while True:
                        max_error = 0
                        max_error_level = None
                        selected_sorted = sorted(region_selected)

                        for gap_idx in range(len(selected_sorted) - 1):
                            prev_idx = selected_sorted[gap_idx]
                            next_idx = selected_sorted[gap_idx + 1]

                            for level_i in range(prev_idx + 1, next_idx):
                                if level_i in region_selected:
                                    continue

                                error = compute_stream_error(stream_i, level_i, prev_idx, next_idx)
                                if error > max_error:
                                    max_error = error
                                    max_error_level = level_i

                        if max_error <= error_threshold or max_error_level is None:
                            break

                        region_selected.add(max_error_level)

                    stream_selected[stream_i].update(region_selected)
                    print(f"  Region [{start}-{end}] stream {stream_i} (non-merged): {len(region_selected)} levels selected")

        # Final enforcement: ensure first dividing level is always included
        if first_dividing_level is not None:
            for s in range(max_stream_count):
                stream_selected[s].add(first_dividing_level)
            print(f"  Enforced first dividing level {first_dividing_level} in all streams")

        # No equalization - each stream keeps its own selected levels
        # Originally-separate regions: independent selection per stream
        # Originally-merged regions: same levels for all streams
        counts = [len(s) for s in stream_selected]
        print(f"Level counts per stream: {counts} (no equalization)")

        # Store results
        self.stream_selected_levels = [sorted(s) for s in stream_selected]
        print(f"\nFinal selected levels per stream:")
        for s in range(max_stream_count):
            print(f"  Stream {s}: {self.stream_selected_levels[s]}")

        # Apply selection to stream_contours and stream_bounding_planes
        new_stream_contours = [[] for _ in range(max_stream_count)]
        new_stream_bounding_planes = [[] for _ in range(max_stream_count)]
        new_stream_groups = []

        # Build new stream_groups (only for selected levels)
        all_selected = set()
        for s in range(max_stream_count):
            all_selected.update(self.stream_selected_levels[s])
        all_selected_sorted = sorted(all_selected)

        for level_i in all_selected_sorted:
            new_stream_groups.append(self.stream_groups[level_i])

        for stream_i in range(max_stream_count):
            for level_i in self.stream_selected_levels[stream_i]:
                new_stream_contours[stream_i].append(self.stream_contours[stream_i][level_i])
                new_stream_bounding_planes[stream_i].append(self.stream_bounding_planes[stream_i][level_i])

        self.stream_contours = new_stream_contours
        self.stream_bounding_planes = new_stream_bounding_planes
        self.stream_groups = new_stream_groups

        # Update visualization
        self.contours = self.stream_contours
        self.bounding_planes = self.stream_bounding_planes
        # Each stream may have different number of levels now
        self.draw_contour_stream = [[True] * len(self.stream_contours[s]) for s in range(max_stream_count)]

        level_counts = [len(self.stream_contours[s]) for s in range(max_stream_count)]
        print(f"Levels updated: {level_counts} per stream")

    def build_fibers(self, skeleton_meshes=None):
        """
        Step 3: Build final fiber structure from selected streams.

        Converts stream_contours/stream_bounding_planes to the format
        used by downstream operations (mesh building, visualization).
        Each stream may have different number of levels.
        """
        if not hasattr(self, 'stream_contours') or self.stream_contours is None:
            print("Run cut_streams and select_levels first")
            return

        max_stream_count = self.max_stream_count
        level_counts = [len(self.stream_contours[s]) for s in range(max_stream_count)]

        print(f"\n=== Build Fibers ===")
        print(f"Streams: {max_stream_count}, Levels per stream: {level_counts}")

        # Convert to format expected by downstream: [stream_i][level_i]
        self.contours = self.stream_contours
        self.bounding_planes = self.stream_bounding_planes

        # Set up draw flags - each stream may have different number of levels
        self.draw_contour_stream = [[True] * level_counts[s] for s in range(max_stream_count)]

        # Call post-processing to compute fiber architecture
        self._find_contour_stream_post_process(skeleton_meshes)

        print(f"Build fibers complete: {max_stream_count} streams, levels: {level_counts}")

    def _cut_contour_mesh_aware(self, contour, bp, projected_refs, stream_indices):
        """
        Cut a contour into pieces using mesh topology-aware method.

        Finds natural cut points by detecting corners (high curvature vertices)
        in the contour, which correspond to mesh topology features.

        Args:
            contour: The contour vertices to cut
            bp: Bounding plane info for this contour
            projected_refs: Reference mean positions projected onto this plane
            stream_indices: Which stream indices these pieces are for

        Returns:
            List of cut contour pieces (one per stream)
        """
        n_pieces = len(stream_indices)
        if n_pieces == 1:
            return [contour]

        contour = np.array(contour)
        n_verts = len(contour)

        if n_verts < 3:
            # Not enough vertices for curvature analysis
            return self._cut_contour_for_streams(contour, bp, projected_refs, stream_indices)

        target_mean = bp['mean']
        target_z = bp['basis_z']
        v0, v1, v2, v3 = bp['bounding_plane']

        # ========== Step 1: Compute curvature at each vertex ==========
        # Curvature = angle between incoming and outgoing edges
        curvatures = np.zeros(n_verts)
        edge_lengths = np.zeros(n_verts)

        for i in range(n_verts):
            prev_idx = (i - 1) % n_verts
            next_idx = (i + 1) % n_verts

            edge_prev = contour[i] - contour[prev_idx]
            edge_next = contour[next_idx] - contour[i]

            len_prev = np.linalg.norm(edge_prev)
            len_next = np.linalg.norm(edge_next)
            edge_lengths[i] = len_prev + len_next

            if len_prev > 1e-10 and len_next > 1e-10:
                edge_prev = edge_prev / len_prev
                edge_next = edge_next / len_next

                # Angle between edges (0 = straight, pi = sharp corner)
                dot = np.clip(np.dot(edge_prev, edge_next), -1, 1)
                angle = np.arccos(dot)
                curvatures[i] = np.pi - angle  # Higher = sharper corner

        # ========== Step 2: Find corner candidates ==========
        # Corners are local maxima of curvature with significant value
        curvature_threshold = np.percentile(curvatures, 70)  # Top 30% curvatures

        corner_candidates = []
        for i in range(n_verts):
            if curvatures[i] < curvature_threshold:
                continue

            # Check if local maximum (higher than neighbors)
            prev_idx = (i - 1) % n_verts
            next_idx = (i + 1) % n_verts
            if curvatures[i] >= curvatures[prev_idx] and curvatures[i] >= curvatures[next_idx]:
                corner_candidates.append((i, curvatures[i]))

        # Sort by curvature (highest first)
        corner_candidates.sort(key=lambda x: -x[1])

        # ========== Step 3: Determine cut axis and target positions ==========
        horizontal_vector = v1 - v0
        vertical_vector = v3 - v0
        horizontal_vector = horizontal_vector / (np.linalg.norm(horizontal_vector) + 1e-10)
        vertical_vector = vertical_vector / (np.linalg.norm(vertical_vector) + 1e-10)

        # Structure vector from reference points
        structure_vector = np.array([0.0, 0.0, 0.0])
        for i in range(len(projected_refs)):
            for j in range(i + 1, len(projected_refs)):
                cand_vector = projected_refs[i] - projected_refs[j]
                if np.dot(cand_vector, np.array([1, 0, 0])) < 0:
                    cand_vector *= -1
                structure_vector += cand_vector
        structure_vector = structure_vector / (np.linalg.norm(structure_vector) + 1e-10)

        h_proj = np.abs(np.dot(structure_vector, horizontal_vector))
        v_proj = np.abs(np.dot(structure_vector, vertical_vector))
        cut_axis = horizontal_vector if h_proj > v_proj else vertical_vector

        # Project contour vertices onto cut axis
        contour_axis_values = np.array([np.dot(p - target_mean, cut_axis) for p in contour])
        min_val, max_val = contour_axis_values.min(), contour_axis_values.max()
        axis_range = max_val - min_val

        if axis_range < 1e-10:
            # Degenerate case - fall back to area-based
            return self._cut_contour_for_streams(contour, bp, projected_refs, stream_indices)

        normalized_axis = (contour_axis_values - min_val) / axis_range

        # Target cut positions (equal spacing)
        target_cuts = [(i + 1) / n_pieces for i in range(n_pieces - 1)]

        # ========== Step 4: Select best corners near target positions ==========
        n_cuts_needed = n_pieces - 1
        selected_cut_indices = []

        # For each target cut position, find best corner nearby
        search_radius = 0.15  # Search within 15% of contour range

        for target_pos in target_cuts:
            best_corner = None
            best_score = -np.inf

            for corner_idx, curv in corner_candidates:
                if corner_idx in selected_cut_indices:
                    continue

                corner_pos = normalized_axis[corner_idx]
                distance = abs(corner_pos - target_pos)

                if distance > search_radius:
                    continue

                # Score = curvature bonus - distance penalty
                score = curv * 2.0 - distance * 5.0

                if score > best_score:
                    best_score = score
                    best_corner = corner_idx

            if best_corner is not None:
                selected_cut_indices.append(best_corner)
            else:
                # No good corner found - use position-based cut point
                # Find vertex closest to target position
                distances = np.abs(normalized_axis - target_pos)
                best_vertex = np.argmin(distances)
                # Avoid duplicates
                while best_vertex in selected_cut_indices:
                    distances[best_vertex] = np.inf
                    best_vertex = np.argmin(distances)
                selected_cut_indices.append(best_vertex)

        # ========== Step 5: Sort cut indices and create pieces ==========
        # Sort cuts by their axis position
        selected_cut_indices.sort(key=lambda idx: normalized_axis[idx])

        # Determine which vertices go to which piece
        # Order streams by their reference position along cut axis
        ref_values = [np.dot(r - target_mean, cut_axis) for r in projected_refs]
        stream_order = np.argsort(ref_values)

        # Create pieces by splitting at cut points
        new_contours = [[] for _ in range(n_pieces)]

        for v_idx, v in enumerate(contour):
            v_pos = normalized_axis[v_idx]

            # Determine which piece this vertex belongs to
            piece_idx = 0
            for cut_idx in selected_cut_indices:
                cut_pos = normalized_axis[cut_idx]
                if v_pos > cut_pos:
                    piece_idx += 1
                else:
                    break

            # Map to actual stream using ordering
            assigned_stream = stream_order[piece_idx]
            new_contours[assigned_stream].append(v)

        # Ensure each piece has at least some vertices and convert to proper numpy arrays
        for i in range(n_pieces):
            if len(new_contours[i]) == 0:
                new_contours[i] = [np.array(target_mean).flatten()[:3]]

            # Convert list of vertices to numpy array with shape (N, 3)
            valid_vertices = []
            for v in new_contours[i]:
                v_arr = np.asarray(v).flatten()
                if len(v_arr) >= 3:
                    valid_vertices.append(v_arr[:3])
                elif len(v_arr) > 0:
                    padded = np.zeros(3)
                    padded[:len(v_arr)] = v_arr
                    valid_vertices.append(padded)

            if len(valid_vertices) == 0:
                valid_vertices = [np.array(target_mean).flatten()[:3]]

            new_contours[i] = np.array(valid_vertices)

            # Ensure at least 3 vertices for valid contour
            while len(new_contours[i]) < 3:
                new_contours[i] = np.vstack([new_contours[i], new_contours[i][-1]])

        return new_contours

    def _cut_contour_bp_transform(self, target_contour, target_bp, source_contours, source_bps, stream_indices, is_first_division=True):
        """
        Cut a contour using bounding plane transformation and optimization.

        Places source contour 2D shapes on target plane, optimizes their positions
        to best cover the target contour, then assigns vertices by distance.

        Args:
            target_contour: The contour vertices to cut (merged contour)
            target_bp: Bounding plane info for target contour
            source_contours: List of source contours (from previous level)
            source_bps: List of bounding planes for source contours
            stream_indices: Which stream indices these pieces are for
            is_first_division: True if this is first division (SEPARATE mode),
                              False if propagated (COMMON mode).
                              SEPARATE: each source optimized independently, cut at target waist
                              COMMON: sources move together, cut at shared boundary vertices

        Returns:
            List of cut contour pieces (one per stream)
        """
        from scipy.optimize import minimize
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        n_pieces = len(stream_indices)
        print(f"  [BP Transform] n_pieces={n_pieces}, stream_indices={stream_indices}")
        if n_pieces == 1:
            return [target_contour]

        target_contour = np.array(target_contour)
        print(f"  [BP Transform] target_contour has {len(target_contour)} vertices")

        # Check for degenerate source contours
        has_degenerate = False
        for i, src_contour in enumerate(source_contours):
            src_arr = np.array(src_contour) if src_contour is not None else np.array([])
            n_src_verts = len(src_arr)
            print(f"  [BP Transform] source_contour[{i}] has {n_src_verts} vertices")
            if n_src_verts < 3:
                print(f"  [BP Transform] WARNING: source {i} has only {n_src_verts} vertices (need >= 3)")
                has_degenerate = True
            elif n_src_verts >= 2:
                # Check if vertices are nearly identical (degenerate triangle/contour)
                max_dist = np.max([np.linalg.norm(src_arr[j] - src_arr[0]) for j in range(1, n_src_verts)])
                if n_src_verts <= 5:
                    print(f"  [BP Transform] source {i} 3D vertices: {src_arr.tolist()}")
                print(f"  [BP Transform] source {i} max vertex spread: {max_dist:.6f}")
                if max_dist < 1e-6:
                    print(f"  [BP Transform] WARNING: source {i} has nearly identical vertices (degenerate)")
                    has_degenerate = True

        # If any source is degenerate, fall back to simple area-based cutting
        if has_degenerate:
            print(f"  [BP Transform] Falling back to area-based cutting due to degenerate sources")
            projected_refs = [src_bp['mean'] for src_bp in source_bps]
            return self._cut_contour_for_streams(target_contour, target_bp, projected_refs, stream_indices)

        # ========== Step 1: Project target contour to 2D ==========
        target_mean = target_bp['mean']
        target_x = target_bp['basis_x']
        target_y = target_bp['basis_y']

        target_2d = np.array([
            [np.dot(v - target_mean, target_x), np.dot(v - target_mean, target_y)]
            for v in target_contour
        ])

        # ========== Step 2: Project source contours to 2D ==========
        source_2d_shapes = []
        initial_translations = []
        initial_rotations = []

        for i, (src_contour, src_bp) in enumerate(zip(source_contours, source_bps)):
            src_contour = np.array(src_contour)
            src_mean = src_bp['mean']

            if is_first_division:
                # SEPARATE mode: project each source to its own 2D plane
                # Then compute rotation to align with target basis
                src_x = src_bp['basis_x']
                src_y = src_bp['basis_y']

                # Project source contour to its own 2D plane (centered at origin)
                src_2d = np.array([
                    [np.dot(v - src_mean, src_x), np.dot(v - src_mean, src_y)]
                    for v in src_contour
                ])
                source_2d_shapes.append(src_2d)
                print(f"  [BP Transform] source {i} has {len(src_2d)} vertices (own basis)")

                # Initial translation: project source mean onto target plane
                src_to_target = src_mean - target_mean
                dist_along_z = np.dot(src_to_target, target_bp['basis_z'])
                src_mean_projected = src_mean - dist_along_z * target_bp['basis_z']
                src_mean_on_target_x = np.dot(src_mean_projected - target_mean, target_x)
                src_mean_on_target_y = np.dot(src_mean_projected - target_mean, target_y)
                initial_translations.append([src_mean_on_target_x, src_mean_on_target_y])
                print(f"  [BP Transform] source {i} initial pos: ({src_mean_on_target_x:.4f}, {src_mean_on_target_y:.4f})")

                # Initial rotation: align source basis_x with target basis_x
                src_x_on_target = src_x - np.dot(src_x, target_bp['basis_z']) * target_bp['basis_z']
                src_x_norm = np.linalg.norm(src_x_on_target)
                if src_x_norm > 1e-10:
                    src_x_on_target = src_x_on_target / src_x_norm
                    src_x_2d = np.array([np.dot(src_x_on_target, target_x), np.dot(src_x_on_target, target_y)])
                    init_theta = np.arctan2(src_x_2d[1], src_x_2d[0])
                else:
                    init_theta = 0.0
                initial_rotations.append(init_theta)
                print(f"  [BP Transform] source {i} initial rotation: {np.degrees(init_theta):.1f}°")
            else:
                # COMMON mode: project source directly to TARGET's 2D plane
                # This maintains relative positions between sources
                src_2d = np.array([
                    [np.dot(v - target_mean, target_x), np.dot(v - target_mean, target_y)]
                    for v in src_contour
                ])
                # Center at source mean position (for proper transform_shape usage)
                src_2d_mean = src_2d.mean(axis=0)
                src_2d_centered = src_2d - src_2d_mean
                source_2d_shapes.append(src_2d_centered)
                print(f"  [BP Transform] source {i} has {len(src_2d)} vertices (target basis)")

                # Initial translation is the source's mean position on target plane
                initial_translations.append([src_2d_mean[0], src_2d_mean[1]])
                print(f"  [BP Transform] source {i} initial pos: ({src_2d_mean[0]:.4f}, {src_2d_mean[1]:.4f})")

                # No rotation needed - already in target basis
                initial_rotations.append(0.0)
                print(f"  [BP Transform] source {i} initial rotation: 0.0° (common basis)")

        # Validate source polygons - check for self-intersection, holes, tiny area
        source_areas = []
        has_invalid = False
        for i, src_2d in enumerate(source_2d_shapes):
            # Debug: show the actual 2D vertices
            if len(src_2d) <= 5:
                print(f"  [BP Transform] source {i} 2D vertices: {src_2d.tolist()}")
            try:
                src_poly = Polygon(src_2d)

                # Check for self-intersection
                if not src_poly.is_valid:
                    print(f"  [BP Transform] WARNING: source {i} is self-intersecting")
                    # Try to fix with buffer(0)
                    src_poly = src_poly.buffer(0)
                    if not src_poly.is_valid:
                        print(f"  [BP Transform] WARNING: source {i} could not be fixed")
                        has_invalid = True

                # Check for holes (interior rings)
                if hasattr(src_poly, 'interiors') and len(list(src_poly.interiors)) > 0:
                    print(f"  [BP Transform] WARNING: source {i} has holes")
                    has_invalid = True

                # Check if it became a MultiPolygon after buffer(0)
                if src_poly.geom_type == 'MultiPolygon':
                    print(f"  [BP Transform] WARNING: source {i} split into multiple polygons")
                    has_invalid = True

                source_areas.append(src_poly.area if src_poly.is_valid else 0.0)
            except Exception as e:
                print(f"  [BP Transform] WARNING: source {i} polygon error: {e}")
                source_areas.append(0.0)
                has_invalid = True

        print(f"  [BP Transform] source areas: {[f'{a:.6f}' for a in source_areas]}")

        # Fall back if any source is invalid or has tiny area
        min_area_threshold = 1e-6
        if has_invalid or any(a < min_area_threshold for a in source_areas):
            reason = "invalid geometry" if has_invalid else "tiny source areas"
            print(f"  [BP Transform] Falling back to area-based cutting due to {reason}")
            projected_refs = [src_bp['mean'] for src_bp in source_bps]
            return self._cut_contour_for_streams(target_contour, target_bp, projected_refs, stream_indices)

        total_source_area = sum(source_areas)

        # ========== Step 3: Define optimization objective ==========
        def transform_shape(shape_2d, scale_x, scale_y, tx, ty, theta):
            """Apply 2D affine transformation: scale (separate x/y), rotate, then translate."""
            # Scale (separate x and y)
            scaled = shape_2d * np.array([scale_x, scale_y])
            # Rotate
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            rotated = scaled @ rotation.T
            # Translate
            return rotated + np.array([tx, ty])

        # Pre-compute target polygon
        try:
            target_poly = Polygon(target_2d)
            if not target_poly.is_valid:
                target_poly = target_poly.buffer(0)
            target_area = target_poly.area
        except:
            print("  [BP Transform] WARNING: Could not create target polygon, falling back")
            projected_refs = [src_bp['mean'] for src_bp in source_bps]
            return self._cut_contour_for_streams(target_contour, target_bp, projected_refs, stream_indices)

        # First division: each source has separate transform (scale, tx, ty, theta)
        # Propagated division: all sources share the same transform (move as one rigid body)
        use_separate_transforms = is_first_division
        print(f"  [BP Transform] is_first_division={is_first_division}, use_separate_transforms={use_separate_transforms}")

        # For common mode, compute combined center of all sources
        # All sources will rotate/scale around this center
        if not use_separate_transforms:
            combined_center = np.mean(initial_translations, axis=0)
            print(f"  [BP Transform] combined center: ({combined_center[0]:.4f}, {combined_center[1]:.4f})")

        def add_polygon(transformed, polygon_list):
            """Helper to create and add a polygon from transformed points."""
            if len(transformed) >= 3:
                try:
                    poly = Polygon(transformed)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    if poly.is_valid and poly.area > 0:
                        polygon_list.append(poly)
                except:
                    pass

        def objective(params):
            """Maximize overlap with target, minimize source overlap and rotation."""
            # If separate transforms: params = [scale_x0, scale_y0, tx0, ty0, theta0, scale_x1, scale_y1, tx1, ty1, theta1, ...]
            # If common transform: params = [scale_x, scale_y, delta_tx, delta_ty, delta_theta]
            #   - Each source uses initial_pos + delta, maintaining relative positions
            transformed_polygons = []
            thetas = []
            scales = []

            min_scale = 0.5  # Minimum allowed scale

            if use_separate_transforms:
                # Each source has its own transform (first division)
                for i in range(n_pieces):
                    scale_x = params[i * 5]
                    scale_y = params[i * 5 + 1]
                    tx = params[i * 5 + 2]
                    ty = params[i * 5 + 3]
                    theta = params[i * 5 + 4]

                    if scale_x <= 0 or scale_y <= 0:
                        return 1e10
                    if scale_x < min_scale or scale_y < min_scale:
                        return 1e10  # Reject scales too small

                    scales.append((scale_x, scale_y))
                    thetas.append(theta)
                    transformed = transform_shape(source_2d_shapes[i], scale_x, scale_y, tx, ty, theta)
                    add_polygon(transformed, transformed_polygons)
            else:
                # Common transform - sources move as ONE rigid body
                # params = [scale_x, scale_y, tx, ty, theta] - transform applied to combined center
                # All sources rotate/scale around combined_center, then translate
                scale_x = params[0]
                scale_y = params[1]
                tx = params[2]  # final x position of combined center
                ty = params[3]  # final y position of combined center
                theta = params[4]  # rotation around combined center

                if scale_x <= 0 or scale_y <= 0:
                    return 1e10
                if scale_x < min_scale or scale_y < min_scale:
                    return 1e10  # Reject scales too small

                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

                for i in range(n_pieces):
                    # Get absolute vertices (centered shape + its mean position)
                    abs_vertices = source_2d_shapes[i] + initial_translations[i]

                    # Transform all vertices around combined_center
                    # 1. Translate to origin (relative to combined_center)
                    rel_vertices = abs_vertices - combined_center
                    # 2. Scale around origin (separate x/y)
                    scaled = rel_vertices * np.array([scale_x, scale_y])
                    # 3. Rotate around origin
                    rotated = scaled @ rot.T
                    # 4. Translate to final position
                    transformed = rotated + np.array([tx, ty])

                    scales.append((scale_x, scale_y))
                    thetas.append(theta)
                    add_polygon(transformed, transformed_polygons)

            if len(transformed_polygons) == 0:
                return 1e10

            # Compute union of all source polygons
            try:
                union_poly = unary_union(transformed_polygons)
            except:
                return 1e10

            # Compute intersection with target (measures actual coverage)
            try:
                intersection = union_poly.intersection(target_poly)
                intersection_area = intersection.area
            except:
                intersection_area = 0

            # Main objective: minimize symmetric difference
            union_area = union_poly.area
            coverage_cost = union_area + target_area - 2 * intersection_area

            # Strong penalty if sources don't cover the target well
            # This prevents the optimizer from shrinking sources or moving them away
            coverage_ratio = intersection_area / (target_area + 1e-10)
            if coverage_ratio < 0.5:
                # Low coverage - strongly penalize to prevent shrinkage
                coverage_cost += 100.0 * target_area * (0.5 - coverage_ratio)

            # Area matching cost - penalize when source area differs from target area
            # This encourages the optimizer to adjust scales to match areas
            area_ratio = union_area / (target_area + 1e-10)
            area_cost = abs(area_ratio - 1.0) * target_area * 10.0  # Weight = 10

            # Gap cost - penalize distance between polygons when they don't overlap
            # This provides gradient to pull sources towards target even with zero intersection
            if intersection_area < 1e-10:
                # No overlap - use minimum boundary distance
                try:
                    gap_dist = union_poly.distance(target_poly)
                    # Strong penalty to close the gap
                    gap_cost = gap_dist * gap_dist * 1000.0  # Quadratic penalty
                except:
                    gap_cost = 0
            else:
                gap_cost = 0

            return coverage_cost + area_cost + gap_cost

        # ========== Step 4: Build initial configuration ==========
        # Compute initial scale based on area ratio (sources should roughly cover target)
        total_source_area = sum(source_areas)
        if total_source_area > 0 and target_area > 0:
            # Scale factor to make total source area match target area
            initial_scale = np.sqrt(target_area / total_source_area)
            # Clamp to reasonable range
            initial_scale = np.clip(initial_scale, 0.1, 10.0)
        else:
            initial_scale = 1.0

        # Compute separate initial scale_x and scale_y based on bounding box aspect ratios
        # This helps the optimizer start from a better point if non-uniform scaling is needed
        target_2d_arr = np.array(target_2d)
        target_bbox_size = target_2d_arr.max(axis=0) - target_2d_arr.min(axis=0)

        # Combine all source shapes for bounding box
        all_source_pts = np.vstack([np.array(s) + np.array(t) for s, t in zip(source_2d_shapes, initial_translations)])
        source_bbox_size = all_source_pts.max(axis=0) - all_source_pts.min(axis=0)

        # Compute aspect ratio difference
        if source_bbox_size[0] > 1e-10 and source_bbox_size[1] > 1e-10:
            target_aspect = target_bbox_size[0] / target_bbox_size[1] if target_bbox_size[1] > 1e-10 else 1.0
            source_aspect = source_bbox_size[0] / source_bbox_size[1]

            # Adjust initial scales to match aspect ratios
            aspect_ratio = target_aspect / source_aspect if source_aspect > 1e-10 else 1.0
            # scale_x * aspect_correction and scale_y should give target aspect ratio
            # sqrt to distribute the correction between both axes
            aspect_sqrt = np.sqrt(aspect_ratio)
            initial_scale_x = initial_scale * aspect_sqrt
            initial_scale_y = initial_scale / aspect_sqrt
            # Clamp to bounds
            initial_scale_x = np.clip(initial_scale_x, 0.5, 2.0)
            initial_scale_y = np.clip(initial_scale_y, 0.5, 2.0)
        else:
            initial_scale_x = initial_scale
            initial_scale_y = initial_scale

        print(f"  [BP Transform] initial scale=({initial_scale_x:.4f},{initial_scale_y:.4f}) (target_area={target_area:.6f}, total_source_area={total_source_area:.6f})")

        # Use computed translations and rotations that align source basis with target basis
        if use_separate_transforms:
            # params: [scale_x0, scale_y0, tx0, ty0, theta0, scale_x1, scale_y1, tx1, ty1, theta1, ...]
            # Each source has its own transform
            x0 = []
            for i, (tx, ty) in enumerate(initial_translations):
                x0.extend([initial_scale_x, initial_scale_y, tx, ty, initial_rotations[i]])
        else:
            # params: [scale_x, scale_y, tx, ty, theta]
            # (tx, ty) = position of combined center, theta = rotation around it
            # Start at combined_center with no rotation
            x0 = [initial_scale_x, initial_scale_y, combined_center[0], combined_center[1], 0.0]

        # Debug: show initial cost breakdown
        def objective_debug(params, verbose=False):
            """Same as objective but with optional verbose output."""
            transformed_polygons = []
            thetas = []
            scales_dbg = []

            min_scale = 0.5

            if use_separate_transforms:
                for i in range(n_pieces):
                    scale_x = params[i * 5]
                    scale_y = params[i * 5 + 1]
                    tx = params[i * 5 + 2]
                    ty = params[i * 5 + 3]
                    theta = params[i * 5 + 4]
                    if scale_x <= 0 or scale_y <= 0 or scale_x < min_scale or scale_y < min_scale:
                        return 1e10
                    scales_dbg.append((scale_x, scale_y))
                    thetas.append(theta)
                    transformed = transform_shape(source_2d_shapes[i], scale_x, scale_y, tx, ty, theta)
                    add_polygon(transformed, transformed_polygons)
            else:
                scale_x = params[0]
                scale_y = params[1]
                tx, ty, theta = params[2], params[3], params[4]
                if scale_x <= 0 or scale_y <= 0 or scale_x < min_scale or scale_y < min_scale:
                    return 1e10
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                for i in range(n_pieces):
                    abs_vertices = source_2d_shapes[i] + initial_translations[i]
                    rel_vertices = abs_vertices - combined_center
                    scaled = rel_vertices * np.array([scale_x, scale_y])
                    rotated = scaled @ rot.T
                    transformed = rotated + np.array([tx, ty])
                    scales_dbg.append((scale_x, scale_y))
                    thetas.append(theta)
                    add_polygon(transformed, transformed_polygons)

            if len(transformed_polygons) == 0:
                return 1e10

            try:
                union_poly = unary_union(transformed_polygons)
                intersection = union_poly.intersection(target_poly)
                intersection_area = intersection.area
                union_area = union_poly.area
            except:
                return 1e10

            coverage_cost = union_area + target_area - 2 * intersection_area

            # Area matching cost
            area_ratio = union_area / (target_area + 1e-10)
            area_cost = abs(area_ratio - 1.0) * target_area * 10.0

            # Gap cost when no overlap
            if intersection_area < 1e-10:
                try:
                    gap_dist = union_poly.distance(target_poly)
                    gap_cost = gap_dist * gap_dist * 1000.0
                except:
                    gap_cost = 0
            else:
                gap_cost = 0

            if verbose:
                print(f"    scales={[f'({sx:.3f},{sy:.3f})' for sx,sy in scales_dbg]}, coverage={coverage_cost:.6f}, area={area_cost:.6f}, gap={gap_cost:.6f}")

            return coverage_cost + area_cost + gap_cost

        init_cost = objective(x0)
        objective_debug(x0, verbose=True)
        print(f"  [BP Transform] initial cost={init_cost:.4f}")

        # ========== Step 5: Optimize from initial configuration ==========
        if use_separate_transforms:
            # SEPARATE mode: skip optimization, just use initial config
            # The goal is only to find the cutting line at the target waist
            optimal_params = x0
            print(f"  [BP Transform] SEPARATE mode: skipping optimization, using initial config")
        else:
            # COMMON mode: run optimization to fit sources to target
            # Compute bounds to prevent optimizer from going crazy
            target_size = np.sqrt(target_area)  # Approximate size
            max_translation = target_size * 3  # Don't go more than 3x size away

            # Bounds for [scale_x, scale_y, tx, ty, theta]
            bounds = [
                (0.5, 2.0),  # scale_x: 0.5 to 2x (prevent excessive shrink/grow)
                (0.5, 2.0),  # scale_y: 0.5 to 2x (prevent excessive shrink/grow)
                (-max_translation, max_translation),  # tx
                (-max_translation, max_translation),  # ty
                (-np.pi, np.pi)  # theta
            ]

            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-9, 'gtol': 1e-7}
            )
            print(f"  [BP Transform] optimizer iterations: {result.nit}")

            optimal_params = result.x
            print(f"  [BP Transform] optimization: success={result.success}, final_cost={result.fun:.4f}")
            objective_debug(optimal_params, verbose=True)

        # ========== Step 6: Get final transformed source shapes ==========
        min_scale = 0.5  # Minimum allowed scale
        final_transformed = []
        optimal_scales = []
        if use_separate_transforms:
            for i in range(n_pieces):
                scale_x = max(optimal_params[i * 5], min_scale)  # Clamp scale_x
                scale_y = max(optimal_params[i * 5 + 1], min_scale)  # Clamp scale_y
                tx = optimal_params[i * 5 + 2]
                ty = optimal_params[i * 5 + 3]
                theta = optimal_params[i * 5 + 4]
                optimal_scales.append((scale_x, scale_y))
                transformed = transform_shape(source_2d_shapes[i], scale_x, scale_y, tx, ty, theta)
                final_transformed.append(transformed)
                print(f"  [BP Transform] piece {i}: scale=({scale_x:.4f},{scale_y:.4f}), tx={tx:.4f}, ty={ty:.4f}, theta={np.degrees(theta):.1f}°")
        else:
            # Common transform - sources move as one rigid body around combined center
            common_scale_x = max(optimal_params[0], min_scale)  # Clamp scale_x
            common_scale_y = max(optimal_params[1], min_scale)  # Clamp scale_y
            center_tx = optimal_params[2]  # final x position of combined center
            center_ty = optimal_params[3]  # final y position of combined center
            theta = optimal_params[4]      # rotation around combined center
            print(f"  [BP Transform] common: scale=({common_scale_x:.4f},{common_scale_y:.4f}), center=({center_tx:.4f}, {center_ty:.4f}), theta={np.degrees(theta):.1f}°")

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

            for i in range(n_pieces):
                # Get absolute vertices (centered shape + its mean position)
                abs_vertices = source_2d_shapes[i] + initial_translations[i]

                # Transform all vertices around combined_center
                rel_vertices = abs_vertices - combined_center
                scaled = rel_vertices * np.array([common_scale_x, common_scale_y])
                rotated = scaled @ rot.T
                transformed = rotated + np.array([center_tx, center_ty])

                optimal_scales.append((common_scale_x, common_scale_y))
                final_transformed.append(transformed)
                print(f"  [BP Transform] piece {i}: transformed around combined center")

        # Compute centroids of transformed sources (needed for cutting line and assignment)
        centroids = []
        for piece_idx, transformed in enumerate(final_transformed):
            if len(transformed) > 0:
                centroid = np.mean(transformed, axis=0)
                centroids.append(centroid)
                print(f"  [BP Transform] piece {piece_idx} centroid: ({centroid[0]:.4f}, {centroid[1]:.4f})")
            else:
                centroids.append(np.array(initial_translations[piece_idx]))
                print(f"  [BP Transform] WARNING: piece {piece_idx} has no vertices, using initial position")

        # ========== Step 6.5: Find cutting line ==========
        # SEPARATE mode: Use nearest non-adjacent vertices on target (waist/concave points)
        # COMMON mode: Transform the original cutting line from first division
        cutting_line_2d = None  # (point, direction) in 2D target plane
        cutting_line_3d = None  # direction vector in 3D for bounding plane creation

        if n_pieces == 2:
            try:
                src_0 = np.array(final_transformed[0])
                src_1 = np.array(final_transformed[1])

                if use_separate_transforms:
                    # SEPARATE mode (first division): Find nearest non-adjacent vertex-edge pair on target
                    # These form a natural "waist" at concave regions
                    n_target = len(target_2d)
                    min_dist = np.inf
                    best_cut = None  # (cut_point, cut_dir, description)

                    # Minimum separation to avoid adjacent vertices/edges
                    min_separation = max(3, n_target // 4)

                    # Helper function to compute point-to-segment distance and closest point
                    def point_to_segment_dist(point, seg_start, seg_end):
                        """Return (distance, closest_point_on_segment)"""
                        seg_vec = seg_end - seg_start
                        seg_len_sq = np.dot(seg_vec, seg_vec)
                        if seg_len_sq < 1e-10:
                            return np.linalg.norm(point - seg_start), seg_start
                        t = np.clip(np.dot(point - seg_start, seg_vec) / seg_len_sq, 0, 1)
                        closest = seg_start + t * seg_vec
                        return np.linalg.norm(point - closest), closest

                    # Check vertex-edge pairs (vertex i to edge j->j+1)
                    for i in range(n_target):
                        for j in range(n_target):
                            # Edge is from j to (j+1) % n_target
                            j_next = (j + 1) % n_target

                            # Check if vertex i is adjacent to edge j->j_next
                            # Adjacent means i == j or i == j_next
                            if i == j or i == j_next:
                                continue

                            # Check separation in both directions along the contour
                            # Distance from i to j (forward)
                            dist_i_to_j = (j - i) % n_target
                            # Distance from i to j_next (forward)
                            dist_i_to_jnext = (j_next - i) % n_target
                            # Distance in the other direction
                            dist_j_to_i = (i - j) % n_target
                            dist_jnext_to_i = (i - j_next) % n_target

                            # Both endpoints of edge must be sufficiently separated from vertex i
                            if min(dist_i_to_j, dist_j_to_i) < min_separation:
                                continue
                            if min(dist_i_to_jnext, dist_jnext_to_i) < min_separation:
                                continue

                            # Compute distance from vertex i to edge j->j_next
                            dist, closest_pt = point_to_segment_dist(
                                target_2d[i], target_2d[j], target_2d[j_next]
                            )

                            if dist < min_dist:
                                min_dist = dist
                                # Cut line goes from vertex to closest point on edge
                                p1 = target_2d[i]
                                p2 = closest_pt
                                cut_point = (p1 + p2) / 2
                                cut_dir = p2 - p1
                                cut_dir = cut_dir / (np.linalg.norm(cut_dir) + 1e-10)
                                best_cut = (cut_point, cut_dir, f"vertex {i} to edge {j}-{j_next}")

                    if best_cut is not None:
                        cutting_line_2d = (best_cut[0], best_cut[1])
                        print(f"  [BP Transform] SEPARATE: cut at {best_cut[2]} (dist={min_dist:.4f})")

                elif len(src_0) >= 3 and len(src_1) >= 3:
                    # COMMON mode: Find shared boundary vertices between OPTIMIZED sources
                    # The cutting line moves together with the sources during optimization
                    boundary_vertices = []

                    # Look for vertices in src_0 that are very close to vertices in src_1
                    size_0 = np.max(np.linalg.norm(src_0 - src_0.mean(axis=0), axis=1))
                    size_1 = np.max(np.linalg.norm(src_1 - src_1.mean(axis=0), axis=1))

                    # Use adaptive threshold - start small, increase if needed
                    for threshold_mult in [0.01, 0.05, 0.1, 0.2]:
                        threshold = threshold_mult * min(size_0, size_1)
                        boundary_vertices = []
                        for v0 in src_0:
                            dists = np.linalg.norm(src_1 - v0, axis=1)
                            min_dist = np.min(dists)
                            if min_dist < threshold:
                                closest_idx = np.argmin(dists)
                                # Use midpoint of the two close vertices
                                boundary_vertices.append((v0 + src_1[closest_idx]) / 2)
                        if len(boundary_vertices) >= 2:
                            break

                    print(f"  [BP Transform] COMMON: found {len(boundary_vertices)} shared boundary vertices (threshold={threshold:.4f})")

                    if len(boundary_vertices) >= 2:
                        boundary_arr = np.array(boundary_vertices)
                        boundary_mean = boundary_arr.mean(axis=0)
                        centered = boundary_arr - boundary_mean
                        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                        boundary_dir = Vt[0]
                        boundary_dir = boundary_dir / (np.linalg.norm(boundary_dir) + 1e-10)

                        cutting_line_2d = (boundary_mean, boundary_dir)
                        print(f"  [BP Transform] COMMON: boundary line from shared vertices")

                if cutting_line_2d is None:
                    # Fallback: Use perpendicular bisector between optimized centroids
                    centroid_vec = centroids[1] - centroids[0]
                    centroid_dist = np.linalg.norm(centroid_vec)
                    if centroid_dist > 1e-10:
                        centroid_dir = centroid_vec / centroid_dist
                        # Cutting line is perpendicular to centroid direction
                        cutting_dir = np.array([-centroid_dir[1], centroid_dir[0]])
                        centroid_mid = (centroids[0] + centroids[1]) / 2
                        cutting_line_2d = (centroid_mid, cutting_dir)
                        print(f"  [BP Transform] cutting line: perpendicular bisector at ({centroid_mid[0]:.4f}, {centroid_mid[1]:.4f})")

                # Convert cutting line direction to 3D
                if cutting_line_2d is not None:
                    cut_dir_2d = cutting_line_2d[1]
                    # The cutting line direction in 3D: expressed in target basis
                    cutting_line_3d = cut_dir_2d[0] * target_x + cut_dir_2d[1] * target_y
                    cutting_line_3d = cutting_line_3d / (np.linalg.norm(cutting_line_3d) + 1e-10)
                    print(f"  [BP Transform] cutting line 3D direction: {cutting_line_3d}")

            except Exception as e:
                print(f"  [BP Transform] WARNING: Could not find cutting line: {e}")

        # Check adjacency between transformed sources
        if len(final_transformed) >= 2:
            for i in range(len(final_transformed)):
                for j in range(i + 1, len(final_transformed)):
                    try:
                        if len(final_transformed[i]) < 3 or len(final_transformed[j]) < 3:
                            continue
                        poly_i = Polygon(final_transformed[i])
                        poly_j = Polygon(final_transformed[j])
                        if not poly_i.is_valid:
                            poly_i = poly_i.buffer(0)
                        if not poly_j.is_valid:
                            poly_j = poly_j.buffer(0)
                        if poly_i.is_empty or poly_j.is_empty:
                            continue
                        dist = poly_i.distance(poly_j)
                        if np.isnan(dist):
                            continue
                        if dist > 1e-6:
                            print(f"  [BP Transform] WARNING: pieces {i} and {j} have gap of {dist:.6f}")
                        else:
                            print(f"  [BP Transform] pieces {i} and {j} are adjacent (dist={dist:.6f})")
                    except Exception as e:
                        pass  # Silently skip invalid polygon checks

        # ========== Step 7: Assign target vertices by cutting line ==========
        # Strategy: Use cutting line to assign vertices based on which side they're on
        # This is cleaner than the hybrid contour proximity + centroid approach

        # First compute distances for boundary interpolation
        all_distances = []  # distances to each source for interpolation
        for v_idx, v_2d in enumerate(target_2d):
            contour_dists = []
            for piece_idx, transformed in enumerate(final_transformed):
                if len(transformed) > 0:
                    distances = np.linalg.norm(transformed - v_2d, axis=1)
                    min_dist = np.min(distances)
                    contour_dists.append(min_dist)
                else:
                    contour_dists.append(np.inf)
            all_distances.append(contour_dists)

        # Assign vertices based on cutting line (moves with optimized sources)
        assignments = []

        if cutting_line_2d is not None and n_pieces == 2:
            # Use the cutting line from optimized sources
            line_point, line_dir = cutting_line_2d
            # Normal to the cutting line (perpendicular)
            line_normal = np.array([-line_dir[1], line_dir[0]])

            # Determine which side each optimized centroid is on
            centroid_sides = []
            for c in centroids:
                side = np.dot(c - line_point, line_normal)
                centroid_sides.append(side)

            # Ensure centroid 0 is on the negative side (for consistent assignment)
            if centroid_sides[0] > 0:
                line_normal = -line_normal

            # Assign each vertex based on which side of the boundary line it's on
            for v_idx, v_2d in enumerate(target_2d):
                side = np.dot(v_2d - line_point, line_normal)
                if side < 0:
                    assignments.append(0)
                else:
                    assignments.append(1)

            print(f"  [BP Transform] assignment by boundary line: {assignments.count(0)} to piece 0, {assignments.count(1)} to piece 1")

        else:
            # Fallback: assign by nearest centroid
            for v_idx, v_2d in enumerate(target_2d):
                centroid_dists = [np.linalg.norm(v_2d - c) for c in centroids]
                assigned_piece = centroid_dists.index(min(centroid_dists))
                assignments.append(assigned_piece)

            print(f"  [BP Transform] assignment by centroid: {[assignments.count(i) for i in range(n_pieces)]}")

        # Remove islands: for 2 pieces, find the 2 best split points
        # This guarantees exactly 2 contiguous regions on a closed contour
        def remove_islands_2pieces(arr):
            """For 2 pieces, keep only 2 boundaries to ensure no islands."""
            n = len(arr)
            if n < 3:
                return arr

            # Find all boundary positions (where value changes)
            boundaries = []
            for i in range(n):
                if arr[i] != arr[(i + 1) % n]:
                    boundaries.append(i)  # boundary after index i

            print(f"  [BP Transform] found {len(boundaries)} boundaries")

            if len(boundaries) <= 2:
                return arr  # already clean

            # Keep only 2 boundaries - pick the pair with best separation
            # (maximize the smaller segment for balanced split)
            best_pair = None
            best_score = -1

            for i in range(len(boundaries)):
                for j in range(i + 1, len(boundaries)):
                    b1, b2 = boundaries[i], boundaries[j]
                    # Segment sizes
                    seg1 = (b2 - b1) % n
                    seg2 = n - seg1
                    score = min(seg1, seg2)  # balance score
                    if score > best_score:
                        best_score = score
                        best_pair = (b1, b2)

            if best_pair is None:
                return arr

            b1, b2 = best_pair
            # Ensure b1 < b2
            if b1 > b2:
                b1, b2 = b2, b1

            # Determine which value goes to which segment
            # Count original values in each segment
            seg1_indices = list(range(b1 + 1, b2 + 1))
            seg2_indices = [i for i in range(n) if i not in seg1_indices]

            seg1_votes = [arr[i] for i in seg1_indices]
            seg2_votes = [arr[i] for i in seg2_indices]

            # Majority vote for each segment
            seg1_val = 1 if seg1_votes.count(1) > seg1_votes.count(0) else 0
            seg2_val = 1 - seg1_val  # opposite value

            # Reassign
            result = list(arr)
            for i in seg1_indices:
                result[i] = seg1_val
            for i in seg2_indices:
                result[i] = seg2_val

            return result

        if n_pieces == 2:
            # Remove islands to ensure exactly 2 contiguous pieces
            assignments = remove_islands_2pieces(assignments)
        # For n_pieces > 2, would need different logic

        # Debug: count assignments per piece
        assignment_counts = [assignments.count(i) for i in range(n_pieces)]
        print(f"  [BP Transform] vertex assignments (after smoothing): {assignment_counts}")

        # Helper function to find intersection of cutting line with edge
        def find_edge_cut_point(v_idx_a, v_idx_b):
            """Find where the cutting line intersects the edge from vertex a to vertex b.
            Returns interpolation parameter t in [0,1] and the 3D intersection point."""
            # Default to midpoint
            t = 0.5

            if cutting_line_2d is not None:
                line_point, line_dir = cutting_line_2d
                line_normal = np.array([-line_dir[1], line_dir[0]])  # Perpendicular to line

                # Edge endpoints in 2D
                p_a = target_2d[v_idx_a]
                p_b = target_2d[v_idx_b]
                edge_vec = p_b - p_a

                # Find t where the edge crosses the line
                # Line equation: dot(p - line_point, line_normal) = 0
                # Edge point: p_a + t * edge_vec
                # Solve: dot(p_a + t*edge_vec - line_point, line_normal) = 0
                denom = np.dot(edge_vec, line_normal)
                if abs(denom) > 1e-10:
                    t = np.dot(line_point - p_a, line_normal) / denom
                    t = np.clip(t, 0.0, 1.0)

            # Compute 3D intersection point (always returns valid point)
            boundary_pt = target_contour[v_idx_a] + t * (target_contour[v_idx_b] - target_contour[v_idx_a])
            return t, boundary_pt

        # Second pass: assign vertices and interpolate at boundaries
        new_contours = [[] for _ in range(n_pieces)]
        shared_boundary_points = []  # Collect shared cut edge vertices
        n_verts = len(target_2d)

        prev_piece = assignments[-1]  # Start with last vertex's assignment
        for v_idx in range(n_verts):
            curr_piece = assignments[v_idx]

            # Check if we crossed a boundary from prev vertex to this one
            if prev_piece != curr_piece and v_idx > 0:
                prev_v_idx = v_idx - 1
                # Find where cutting line intersects this edge
                t, boundary_pt = find_edge_cut_point(prev_v_idx, v_idx)

                # Add boundary point to BOTH pieces (shared vertex)
                new_contours[prev_piece].append(boundary_pt)
                new_contours[curr_piece].append(boundary_pt)
                shared_boundary_points.append(boundary_pt.copy())

            # Add current vertex to its piece
            new_contours[curr_piece].append(target_contour[v_idx])
            prev_piece = curr_piece

        # Handle wrap-around: check boundary between last and first vertex
        if assignments[-1] != assignments[0]:
            prev_piece = assignments[-1]
            curr_piece = assignments[0]

            # Find where cutting line intersects wrap-around edge
            t, boundary_pt = find_edge_cut_point(n_verts - 1, 0)

            new_contours[prev_piece].append(boundary_pt)
            new_contours[curr_piece].append(boundary_pt)
            shared_boundary_points.append(boundary_pt.copy())

        # Register shared cut edge vertices globally
        if len(shared_boundary_points) > 0:
            if not hasattr(self, 'shared_cut_vertices'):
                self.shared_cut_vertices = []
            for pt in shared_boundary_points:
                self.shared_cut_vertices.append(pt)
            print(f"  [BP Transform] Registered {len(shared_boundary_points)} shared cut edge vertices (total: {len(self.shared_cut_vertices)})")

        # Ensure each piece has at least some vertices and convert to proper numpy arrays
        for i in range(n_pieces):
            if len(new_contours[i]) == 0:
                new_contours[i] = [np.array(target_mean).flatten()[:3]]

            # Convert list of vertices to numpy array with shape (N, 3)
            valid_vertices = []
            for v in new_contours[i]:
                v_arr = np.asarray(v).flatten()
                if len(v_arr) >= 3:
                    valid_vertices.append(v_arr[:3])
                elif len(v_arr) > 0:
                    # Pad with zeros if needed
                    padded = np.zeros(3)
                    padded[:len(v_arr)] = v_arr
                    valid_vertices.append(padded)
                # Skip empty/invalid vertices

            if len(valid_vertices) == 0:
                # Fallback to target mean
                valid_vertices = [np.array(target_mean).flatten()[:3]]

            new_contours[i] = np.array(valid_vertices)

            # Ensure at least 3 vertices for valid contour (duplicate if needed)
            while len(new_contours[i]) < 3:
                new_contours[i] = np.vstack([new_contours[i], new_contours[i][-1]])

        # ========== Step 8: Save visualization with assignments ==========
        self._save_bp_transform_visualization(
            target_2d, target_poly, source_2d_shapes, final_transformed,
            stream_indices, optimal_scales, initial_translations, initial_rotations,
            use_separate_transforms, assignments, centroids, cutting_line_2d
        )

        print(f"  [BP Transform] result: {[len(c) for c in new_contours]} vertices per piece")

        # Return cut contours along with cutting info for bounding plane creation
        # cutting_info contains:
        #   - cutting_line_3d: the cutting line direction in 3D (for x-axis perpendicular)
        #   - original_z: the original target bounding plane z-axis (to preserve)
        #   - is_cut: True to mark these contours as cut (don't smooth them)
        cutting_info = {
            'cutting_line_3d': cutting_line_3d,
            'original_z': target_bp['basis_z'].copy(),
            'is_cut': True
        }

        return new_contours, cutting_info

    def _save_bp_transform_visualization(self, target_2d, target_poly, source_2d_shapes,
                                         final_transformed, stream_indices, scales,
                                         initial_translations, initial_rotations,
                                         use_separate_transforms=True, assignments=None, centroids=None,
                                         cutting_line_2d=None):
        """
        Store visualization data for BP Viz imgui window and save to file.
        """
        # Store for imgui display
        if not hasattr(self, '_bp_viz_data'):
            self._bp_viz_data = []
        self._bp_viz_data.append({
            'target_2d': np.array(target_2d),
            'source_2d_shapes': [np.array(s) for s in source_2d_shapes],
            'final_transformed': [np.array(f) for f in final_transformed],
            'stream_indices': list(stream_indices),
            'scales': list(scales),
            'cutting_line_2d': cutting_line_2d,
            'initial_translations': [np.array(t) for t in initial_translations],
            'initial_rotations': list(initial_rotations),
            'use_separate_transforms': use_separate_transforms,
            'centroids': [np.array(c) for c in centroids] if centroids else [],
            'assignments': list(assignments) if assignments else [],
        })

        # Save to file using matplotlib
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import os

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, len(stream_indices)))

        # Helper to transform shape
        def transform_shape(shape_2d, scale, tx, ty, theta):
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            scaled = shape_2d * scale
            rotated = scaled @ rot.T
            return rotated + np.array([tx, ty])

        # Compute initial transformed shapes
        initial_transformed = []
        for i, src_2d in enumerate(source_2d_shapes):
            if use_separate_transforms:
                # Separate mode: apply individual rotation
                tx, ty = initial_translations[i]
                theta = initial_rotations[i]
                transformed = transform_shape(src_2d, 1.0, tx, ty, theta)
            else:
                # Common mode: just absolute position (no rotation yet)
                transformed = src_2d + initial_translations[i]
            initial_transformed.append(transformed)

        target_arr = np.array(target_2d)
        mode_str = "SEPARATE" if use_separate_transforms else "COMMON"

        # Left plot: Initial configuration (no dashed original)
        ax1 = axes[0]
        ax1.set_title(f'Initial Config [{mode_str}]')
        ax1.plot(target_arr[:, 0], target_arr[:, 1], 'k-', linewidth=2, label='Target')
        ax1.fill(target_arr[:, 0], target_arr[:, 1], alpha=0.1, color='gray')

        for i, init_trans in enumerate(initial_transformed):
            if len(init_trans) >= 3:
                init_arr = np.array(init_trans)
                init_closed = np.vstack([init_arr, init_arr[0]])
                ax1.plot(init_closed[:, 0], init_closed[:, 1], '-', color=colors[i],
                        linewidth=1.5, label=f'Src {stream_indices[i]}')
                ax1.fill(init_arr[:, 0], init_arr[:, 1], alpha=0.2, color=colors[i])

        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # Right plot: Final result (or initial for SEPARATE mode)
        ax2 = axes[1]

        if use_separate_transforms:
            # SEPARATE mode: show initial config with cutting line
            ax2.set_title(f'Cutting Line [{mode_str}]')

            # Draw target as gray
            ax2.plot(target_arr[:, 0], target_arr[:, 1], 'k-', linewidth=2, label='Target')
            ax2.fill(target_arr[:, 0], target_arr[:, 1], alpha=0.1, color='gray')

            # Draw initial source contours (same as left plot)
            for i, init_trans in enumerate(initial_transformed):
                if len(init_trans) >= 3:
                    init_arr = np.array(init_trans)
                    init_closed = np.vstack([init_arr, init_arr[0]])
                    ax2.fill(init_arr[:, 0], init_arr[:, 1], alpha=0.2, color=colors[i])
                    ax2.plot(init_closed[:, 0], init_closed[:, 1], '-', color=colors[i],
                            linewidth=1.5, label=f'Src {stream_indices[i]}')
        else:
            # COMMON mode: show optimized result
            # scales is now a list of (scale_x, scale_y) tuples
            if scales and isinstance(scales[0], (tuple, list)):
                scales_str = ', '.join([f'({sx:.2f},{sy:.2f})' for sx, sy in scales])
            else:
                scales_str = ', '.join([f'{s:.2f}' for s in scales])
            ax2.set_title(f'Final (scales=[{scales_str}])')

            # Draw target contour colored by assignments
            if assignments and len(assignments) == len(target_arr):
                # Draw edges colored by assignment
                for i in range(len(target_arr)):
                    p1 = target_arr[i]
                    p2 = target_arr[(i + 1) % len(target_arr)]
                    piece_idx = assignments[i]
                    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=colors[piece_idx],
                            linewidth=2.5, zorder=5)
                # Draw vertices as colored dots
                for v_idx, (v_2d, piece_idx) in enumerate(zip(target_arr, assignments)):
                    ax2.scatter(v_2d[0], v_2d[1], c=[colors[piece_idx]], s=25, zorder=10)
            else:
                # No assignments - draw target as gray
                ax2.plot(target_arr[:, 0], target_arr[:, 1], 'k-', linewidth=2, label='Target')
                ax2.fill(target_arr[:, 0], target_arr[:, 1], alpha=0.1, color='gray')

            # Draw optimized source contours (filled like initial config)
            for i, transformed in enumerate(final_transformed):
                if len(transformed) >= 3:
                    trans_arr = np.array(transformed)
                    trans_closed = np.vstack([trans_arr, trans_arr[0]])
                    ax2.fill(trans_arr[:, 0], trans_arr[:, 1], alpha=0.3, color=colors[i])
                    ax2.plot(trans_closed[:, 0], trans_closed[:, 1], '-', color=colors[i],
                            linewidth=2.0, zorder=15, label=f'Opt {stream_indices[i]}')

        # Draw centroids as large X markers (only for COMMON mode)
        if not use_separate_transforms and centroids:
            for i, c in enumerate(centroids):
                ax2.scatter(c[0], c[1], marker='X', c=[colors[i]], s=100, zorder=20)

        # Draw cutting/boundary line (magenta for visibility)
        # Clip line to actual contour intersections
        if cutting_line_2d is not None:
            line_point, line_dir = cutting_line_2d
            # Extend line far enough to intersect contour
            target_center = target_arr.mean(axis=0)
            target_radius = np.max(np.linalg.norm(target_arr - target_center, axis=1))
            extent = target_radius * 2.0
            line_start = line_point - line_dir * extent
            line_end = line_point + line_dir * extent

            # Find intersections with target contour
            intersections = []
            n_verts = len(target_arr)
            d = line_end - line_start
            for i in range(n_verts):
                q1 = target_arr[i]
                q2 = target_arr[(i + 1) % n_verts]
                e = q2 - q1
                denom = d[0] * e[1] - d[1] * e[0]
                if abs(denom) < 1e-10:
                    continue
                t = ((q1[0] - line_start[0]) * e[1] - (q1[1] - line_start[1]) * e[0]) / denom
                s = ((q1[0] - line_start[0]) * d[1] - (q1[1] - line_start[1]) * d[0]) / denom
                if 0 <= s <= 1:  # Intersection on contour edge
                    pt = line_start + t * d
                    intersections.append((t, pt))
            intersections.sort(key=lambda x: x[0])

            if len(intersections) >= 2:
                # Draw line between first two intersection points (clipped to contour)
                p1 = intersections[0][1]
                p2 = intersections[1][1]
                ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color='magenta',
                        linewidth=2.5, zorder=25, label='Cut line')
            else:
                # Fallback to extent-based drawing
                p1 = line_point - line_dir * target_radius * 0.9
                p2 = line_point + line_dir * target_radius * 0.9
                ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', color='magenta',
                        linewidth=2.5, zorder=25, label='Cut line')

        ax2.legend(loc='upper right', fontsize=8)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save to bp_viz/
        os.makedirs('bp_viz', exist_ok=True)
        if not hasattr(self, '_bp_viz_counter'):
            self._bp_viz_counter = 0
        self._bp_viz_counter += 1

        obj_name = getattr(self, '_muscle_name', None)
        if obj_name is None:
            obj_name = getattr(self, 'name', getattr(self, 'mesh_name', 'unknown'))

        filepath = f'bp_viz/bp_transform_{obj_name}_{self._bp_viz_counter:03d}.png'
        plt.savefig(filepath, dpi=100)
        plt.close(fig)

    def _cut_contour_for_streams(self, contour, bp, projected_refs, stream_indices):
        """
        Cut a contour into pieces for multiple streams using area-based method.

        Args:
            contour: The contour vertices to cut
            bp: Bounding plane info for this contour
            projected_refs: Reference mean positions projected onto this plane
            stream_indices: Which stream indices these pieces are for

        Returns:
            List of cut contour pieces (one per stream)
        """
        n_pieces = len(stream_indices)
        if n_pieces == 1:
            return [contour]

        contour = np.array(contour)
        target_mean = bp['mean']
        target_z = bp['basis_z']
        v0, v1, v2, v3 = bp['bounding_plane']

        # Compute structure vector (direction separating the streams)
        structure_vector = np.array([0.0, 0.0, 0.0])
        for i in range(len(projected_refs)):
            for j in range(i + 1, len(projected_refs)):
                cand_vector = projected_refs[i] - projected_refs[j]
                if np.dot(cand_vector, np.array([1, 0, 0])) < 0:
                    cand_vector *= -1
                structure_vector += cand_vector
        structure_vector = structure_vector / (np.linalg.norm(structure_vector) + 1e-10)

        # Determine axis to cut along
        horizontal_vector = v1 - v0
        vertical_vector = v3 - v0
        horizontal_vector = horizontal_vector / (np.linalg.norm(horizontal_vector) + 1e-10)
        vertical_vector = vertical_vector / (np.linalg.norm(vertical_vector) + 1e-10)

        h_proj = np.abs(np.dot(structure_vector, horizontal_vector))
        v_proj = np.abs(np.dot(structure_vector, vertical_vector))

        if h_proj > v_proj:
            cut_axis = horizontal_vector
        else:
            cut_axis = vertical_vector

        # Order streams along cut axis
        ref_values = [np.dot(r - target_mean, cut_axis) for r in projected_refs]
        order = np.argsort(ref_values)

        # Equal area distribution
        areas = [1.0] * n_pieces
        cumul_areas = np.cumsum(areas)
        cumul_areas = cumul_areas / cumul_areas[-1]

        # Project contour points onto cut axis (normalized 0-1)
        contour_values = [np.dot(p - target_mean, cut_axis) for p in contour]
        min_val, max_val = min(contour_values), max(contour_values)
        if max_val - min_val > 1e-10:
            normalized_values = [(v - min_val) / (max_val - min_val) for v in contour_values]
        else:
            normalized_values = [0.5] * len(contour)

        # Assign vertices to pieces
        new_contours = [[] for _ in range(n_pieces)]
        for v_idx, v in enumerate(contour):
            nv = normalized_values[v_idx]
            for piece_idx, thresh in enumerate(cumul_areas):
                if nv <= thresh:
                    assigned_stream = order[piece_idx]
                    new_contours[assigned_stream].append(v)
                    break

        # Ensure each piece has at least some vertices and convert to proper numpy arrays
        for i in range(n_pieces):
            if len(new_contours[i]) == 0:
                new_contours[i] = [np.array(target_mean).flatten()[:3]]

            # Convert list of vertices to numpy array with shape (N, 3)
            valid_vertices = []
            for v in new_contours[i]:
                v_arr = np.asarray(v).flatten()
                if len(v_arr) >= 3:
                    valid_vertices.append(v_arr[:3])
                elif len(v_arr) > 0:
                    # Pad with zeros if needed
                    padded = np.zeros(3)
                    padded[:len(v_arr)] = v_arr
                    valid_vertices.append(padded)

            if len(valid_vertices) == 0:
                valid_vertices = [np.array(target_mean).flatten()[:3]]

            new_contours[i] = np.array(valid_vertices)

            # Ensure at least 3 vertices for valid contour
            while len(new_contours[i]) < 3:
                new_contours[i] = np.vstack([new_contours[i], new_contours[i][-1]])

        return new_contours

    def build_streams(self, skeleton_meshes=None):
        """
        Build streams from selected levels.
        """
        if self.contours is None or self.bounding_planes is None:
            print("No contours/bounding planes available")
            return

        if len(self.contours) < 2:
            print("Need at least 2 contour levels - run 'Select Levels' first")
            return

        print(f"\n=== Build Streams ===")
        print(f"Building streams from {len(self.contours)} levels")

        self.find_contour_stream(skeleton_meshes)

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
                # Skip distance check if levels were pre-selected
                skip_level = False
                if not getattr(self, 'selected_stream_levels', None):
                    # Check distance for ALL streams first to ensure consistent behavior
                    # Only skip this level if ALL streams have dist < min_dist
                    skip_level = True
                    for i in range(len(ordered_contours_trim)):
                        closest = linked[i][0]
                        prev_mean = bounding_planes_trim[i][-1]['mean']
                        next_mean = next_bounding_planes[closest]['mean']
                        dist = np.linalg.norm(next_mean - prev_mean)
                        if dist >= min_dist:
                            skip_level = False
                            break

                if not skip_level:
                    # Add contours for ALL streams at this level
                    for i in range(len(ordered_contours_trim)):
                        closest = linked[i][0]
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
                                # Skip distance check if levels were pre-selected
                                if not getattr(self, 'selected_stream_levels', None):
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

                        # Skip distance check if levels were pre-selected
                        if not getattr(self, 'selected_stream_levels', None):
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
            continue  # SKIP SQUARE-LIKE HANDLING FOR DEBUGGING
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

                        self.contours[stream_i][i], contour_match = self.find_contour_match(ordered_contours_trim_orig[stream_i][i], bounding_plane, preserve_order=False)

                        bounding_plane_info['bounding_plane'] = bounding_plane
                        bounding_plane_info['projected_2d'] = projected_2d
                        bounding_plane_info['area'] = area
                        bounding_plane_info['contour_match'] = contour_match

        for _ in range(10):
            break  # SKIP 10-ITERATION SMOOTHING FOR DEBUGGING
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

                    self.contours[stream_i][i], contour_match = self.find_contour_match(ordered_contours_trim_orig[stream_i][i], bounding_plane, preserve_order=False)

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

        # Debug: verify all streams have the same length
        stream_lengths = [len(bp_stream) for bp_stream in self.bounding_planes]
        contour_lengths = [len(c_stream) for c_stream in self.contours]
        if len(set(stream_lengths)) > 1 or len(set(contour_lengths)) > 1:
            print(f"WARNING: Inconsistent stream lengths!")
            print(f"  bounding_planes lengths: {stream_lengths}")
            print(f"  contours lengths: {contour_lengths}")

        print(f"After stream search: {stream_num} streams, {level_num} levels per stream")

        # ========== Smoothen z and x axes for each stream ==========
        print("Smoothening stream axes...")
        for stream_i in range(stream_num):
            bp_stream = self.bounding_planes[stream_i]
            stream_len = len(bp_stream)
            if stream_len < 2:
                continue

            # Determine direction: origin to insertion
            # First contour should point toward second contour
            first_mean = bp_stream[0]['mean']
            second_mean = bp_stream[1]['mean']
            forward_dir = second_mean - first_mean
            forward_norm = np.linalg.norm(forward_dir)
            if forward_norm > 1e-10:
                forward_dir = forward_dir / forward_norm

                # Flip first contour's z if it doesn't point forward
                if np.dot(bp_stream[0]['basis_z'], forward_dir) < 0:
                    bp_stream[0]['basis_z'] = -bp_stream[0]['basis_z']
                    bp_stream[0]['basis_x'] = -bp_stream[0]['basis_x']
                    print(f"  Stream {stream_i}, contour 0: flipped z toward insertion")

            # Forward pass: align each contour's z with previous
            for i in range(1, stream_len):
                curr_z = bp_stream[i]['basis_z']
                prev_z = bp_stream[i - 1]['basis_z']

                if np.dot(curr_z, prev_z) < 0:
                    bp_stream[i]['basis_z'] = -bp_stream[i]['basis_z']
                    bp_stream[i]['basis_x'] = -bp_stream[i]['basis_x']
                    print(f"  Stream {stream_i}, contour {i}: flipped z to align")

            # X-axis smoothing: align with (1,0,0) projected onto plane, then propagate
            # Start from first contour
            first_z = bp_stream[0]['basis_z']
            first_x = bp_stream[0]['basis_x']

            # Project (1,0,0) onto plane perpendicular to z
            ref_x = np.array([1.0, 0.0, 0.0])
            ref_x_proj = ref_x - np.dot(ref_x, first_z) * first_z
            ref_x_norm = np.linalg.norm(ref_x_proj)
            if ref_x_norm > 0.1:  # Only use if not too parallel to z
                ref_x_proj = ref_x_proj / ref_x_norm
                if np.dot(first_x, ref_x_proj) < 0:
                    bp_stream[0]['basis_x'] = -bp_stream[0]['basis_x']
                    bp_stream[0]['basis_y'] = -bp_stream[0]['basis_y']
                    print(f"  Stream {stream_i}, contour 0: flipped x toward (1,0,0)")

            # Propagate x alignment forward
            for i in range(1, stream_len):
                curr_x = bp_stream[i]['basis_x']
                curr_z = bp_stream[i]['basis_z']
                prev_x = bp_stream[i - 1]['basis_x']

                # Project prev_x onto current plane
                prev_x_proj = prev_x - np.dot(prev_x, curr_z) * curr_z
                prev_x_norm = np.linalg.norm(prev_x_proj)
                if prev_x_norm > 1e-10:
                    prev_x_proj = prev_x_proj / prev_x_norm

                    if np.dot(curr_x, prev_x_proj) < 0:
                        bp_stream[i]['basis_x'] = -bp_stream[i]['basis_x']
                        bp_stream[i]['basis_y'] = -bp_stream[i]['basis_y']
                        print(f"  Stream {stream_i}, contour {i}: flipped x to align")

        print("Stream axes smoothening complete")

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

    def save_contours(self, filepath):
        """
        Save contours and bounding planes to a file.

        Args:
            filepath: Path to save the contour data (JSON format)
        """
        import json

        def convert_to_serializable(obj):
            """Recursively convert numpy arrays and other non-serializable types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        if self.contours is None or len(self.contours) == 0:
            print("No contours to save")
            return

        if self.bounding_planes is None or len(self.bounding_planes) == 0:
            print("No bounding planes to save")
            return

        # Convert all data to serializable format
        contours_data = convert_to_serializable(self.contours)
        bp_data = convert_to_serializable(self.bounding_planes)

        # Save additional state
        save_data = {
            'contours': contours_data,
            'bounding_planes': bp_data,
            'draw_contour_stream': self.draw_contour_stream if hasattr(self, 'draw_contour_stream') else None,
            '_contours_normalized': getattr(self, '_contours_normalized', False),
        }

        # Save stream endpoints if available
        if hasattr(self, '_stream_endpoints') and self._stream_endpoints is not None:
            save_data['_stream_endpoints'] = convert_to_serializable(self._stream_endpoints)

        # Save stream data if available (after cutting)
        if hasattr(self, 'stream_contours') and self.stream_contours is not None:
            save_data['stream_contours'] = convert_to_serializable(self.stream_contours)
        if hasattr(self, 'stream_bounding_planes') and self.stream_bounding_planes is not None:
            save_data['stream_bounding_planes'] = convert_to_serializable(self.stream_bounding_planes)
        if hasattr(self, 'stream_groups') and self.stream_groups is not None:
            save_data['stream_groups'] = convert_to_serializable(self.stream_groups)
        if hasattr(self, 'max_stream_count') and self.max_stream_count is not None:
            save_data['max_stream_count'] = self.max_stream_count

        with open(filepath, 'w') as f:
            json.dump(save_data, f)

        print(f"Contours saved to {filepath}")
        print(f"  {len(self.contours)} contour levels, {len(self.bounding_planes)} bounding plane levels")
        if hasattr(self, 'stream_contours') and self.stream_contours is not None:
            print(f"  {len(self.stream_contours)} streams saved")

    def load_contours(self, filepath):
        """
        Load contours and bounding planes from a file.

        Args:
            filepath: Path to load the contour data from (.npz format)
        """
        import json

        with open(filepath, 'r') as f:
            save_data = json.load(f)

        # Convert contours back to numpy arrays
        self.contours = []
        for level in save_data['contours']:
            level_contours = []
            for contour in level:
                level_contours.append(np.array(contour))
            self.contours.append(level_contours)

        # Convert bounding planes back to numpy arrays
        self.bounding_planes = []
        numpy_keys = ['mean', 'basis_x', 'basis_y', 'basis_z', 'bounding_plane', 'projected_2d',
                      'contour_vertices', 'newell_normal']
        for level in save_data['bounding_planes']:
            level_bp = []
            for bp in level:
                bp_dict = {}
                for key, value in bp.items():
                    if key in numpy_keys and value is not None:
                        bp_dict[key] = np.array(value)
                    elif key == 'contour_match' and value is not None:
                        # contour_match is a list of [P, Q] pairs where P and Q are 3D points
                        bp_dict[key] = []
                        for match in value:
                            if isinstance(match, list) and len(match) == 2:
                                # Each match is [P, Q] pair of 3D points
                                bp_dict[key].append([np.array(match[0]), np.array(match[1])])
                            elif match is None:
                                bp_dict[key].append(None)
                            else:
                                bp_dict[key].append(match)
                    else:
                        bp_dict[key] = value
                level_bp.append(bp_dict)
            self.bounding_planes.append(level_bp)

        # Restore additional state
        if save_data.get('draw_contour_stream') is not None:
            self.draw_contour_stream = save_data['draw_contour_stream']
        else:
            # Create draw_contour_stream matching the structure of contours
            # If contours is nested (streams), create nested structure
            if len(self.contours) > 0 and isinstance(self.contours[0], list):
                # Nested structure: [stream][level] -> contour
                self.draw_contour_stream = [[True] * len(stream) for stream in self.contours]
            else:
                # Flat structure
                self.draw_contour_stream = [True] * len(self.contours)

        if '_contours_normalized' in save_data:
            self._contours_normalized = save_data['_contours_normalized']

        if '_stream_endpoints' in save_data:
            self._stream_endpoints = save_data['_stream_endpoints']

        # Load stream data if available
        if 'stream_contours' in save_data:
            self.stream_contours = []
            for stream in save_data['stream_contours']:
                stream_list = []
                for contour in stream:
                    stream_list.append(np.array(contour))
                self.stream_contours.append(stream_list)

        if 'stream_bounding_planes' in save_data:
            self.stream_bounding_planes = []
            for stream in save_data['stream_bounding_planes']:
                stream_bp = []
                for bp in stream:
                    bp_dict = {}
                    for key, value in bp.items():
                        if key in numpy_keys and value is not None:
                            bp_dict[key] = np.array(value)
                        elif key == 'contour_match' and value is not None:
                            bp_dict[key] = []
                            for match in value:
                                if isinstance(match, list) and len(match) == 2:
                                    bp_dict[key].append([np.array(match[0]), np.array(match[1])])
                                elif match is None:
                                    bp_dict[key].append(None)
                                else:
                                    bp_dict[key].append(match)
                        else:
                            bp_dict[key] = value
                    stream_bp.append(bp_dict)
                self.stream_bounding_planes.append(stream_bp)

        if 'stream_groups' in save_data:
            self.stream_groups = save_data['stream_groups']

        if 'max_stream_count' in save_data:
            self.max_stream_count = save_data['max_stream_count']

        # Enable drawing
        self.is_draw_contours = True
        self.is_draw_bounding_box = True

        print(f"Contours loaded from {filepath}")
        print(f"  {len(self.contours)} contour levels, {len(self.bounding_planes)} bounding plane levels")
        if hasattr(self, 'stream_contours') and self.stream_contours is not None:
            print(f"  {len(self.stream_contours)} streams loaded")

