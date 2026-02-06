# Skeleton Mesh functionality for the viewer
# Contains SkeletonMeshMixin for MeshLoader with bounding box methods
# and skeleton binding methods for muscle-skeleton attachment

import numpy as np
from OpenGL.GL import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from itertools import product
from collections import defaultdict
from scipy.spatial import cKDTree


class SkeletonMeshMixin:
    """
    Mixin class containing skeleton mesh specific functionality for MeshLoader.
    This includes bounding box computation, corner drawing, and hierarchy management.
    """

    def _init_skeleton_properties(self):
        """Initialize skeleton-specific properties. Called from MeshLoader.__init__"""
        # For Skeleton Meshes
        self.corners = None
        self.corners_list = []
        self.bbox_axes_list = []
        self.num_boxes = 1
        self.auto_num_boxes = False  # Auto-find optimal number of boxes
        self.weld_joints = []
        self.is_draw_corners = True
        self.sizes = None

        # Bounding box alignment options (checkboxes)
        self.bb_align_x = False
        self.bb_align_y = False
        self.bb_align_z = False
        self.bb_enforce_symmetry = False  # Mirror across YZ plane (x=0)

        # Skeleton hierarchy
        self.is_root = False
        self.is_contact = False
        self.cand_parent_index = 0
        self.parent_name = None
        self.parent_mesh = None
        self.children_names = []
        self.joint_to_parent = None
        self.is_weld = False
        self.is_revolute = False
        self.revolute_axis = np.array([1.0, 0.0, 0.0])  # Default x-axis
        self.revolute_lower = 0.0  # Lower limit in radians
        self.revolute_upper = 2.5  # Upper limit in radians (~143 degrees)
        self.bends_backward = False  # True for knees, False for elbows/fingers
        self.main_box_num = 0

        # Body transformation data
        self.body_rs = []
        self.body_ts = []

    def _init_skeleton_binding_properties(self):
        """Initialize skeleton binding properties. Called from MeshLoader.__init__"""
        # Skeleton binding data for muscle attachment
        self.tet_skeleton_bindings = None  # List of (origin_body, insertion_body, weight, orig_pos)
        self.tet_initial_bone_transforms = {}  # body_name -> (rotation, translation)
        self.tet_original_positions = None  # Original tet vertex positions

        # Skinning weights for LBS
        self.skinning_bones = []
        self.skinning_weights = None
        self.skinning_local_positions = {}
        self.skinning_initial_transforms = {}

        # Vertex-to-bone assignments
        self.vertex_bone_assignments = {}

        # Skeleton rest transforms for delta computation
        self.skeleton_rest_transforms = {}

        # Bone proximal vertices for proximity constraints
        self.bone_proximal_vertices = {}

    def draw_corners(self):
        """Draw bounding box corners and weld joints."""
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

    def find_optimal_num_boxes(self, axis=None, method='pca-cluster', symmetry=False, max_boxes=9):
        """
        Find the optimal number of bounding boxes that minimizes empty volume.

        Args:
            axis: Axis alignment parameter for find_bounding_box
            method: Clustering method
            symmetry: If True, only try symmetric configurations (1, 3, 5, 7...)
            max_boxes: Maximum number of boxes to try

        Returns:
            Optimal number of boxes
        """
        if self.vertices is None or len(self.vertices) == 0:
            return 1

        # Compute approximate mesh volume using convex hull
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(self.vertices)
            mesh_volume = hull.volume
        except Exception:
            # Fallback: use bounding box volume as reference
            mesh_volume = np.prod(self.vertices.max(axis=0) - self.vertices.min(axis=0))

        best_num = 1
        best_efficiency = 0.0  # mesh_volume / total_box_volume

        # Try different numbers of boxes
        if symmetry:
            candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        else:
            candidates = list(range(1, max_boxes + 1))

        candidates = [c for c in candidates if c <= max_boxes]

        original_num = self.num_boxes

        for n in candidates:
            self.num_boxes = n

            # Temporarily compute bounding boxes (with _from_auto=True to prevent recursion)
            try:
                self.find_bounding_box(axis=axis, method=method, symmetry=symmetry, _from_auto=True)
            except Exception:
                continue

            if self.corners_list is None or len(self.corners_list) == 0:
                continue

            # Compute total bounding box volume
            total_volume = 0.0
            for i, corners in enumerate(self.corners_list):
                if self.sizes is not None and i < len(self.sizes):
                    box_vol = np.prod(self.sizes[i])
                    total_volume += box_vol

            if total_volume <= 0:
                continue

            efficiency = mesh_volume / total_volume

            # Prefer configurations with higher efficiency (less empty space)
            # But also penalize too many boxes (complexity)
            if symmetry:
                penalty = 0.015 * n  # Lower penalty for symmetric
            else:
                penalty = 0.03 * n
            score = efficiency * (1.0 - penalty)

            if score > best_efficiency:
                best_efficiency = score
                best_num = n

        # Restore and recompute with optimal number
        self.num_boxes = best_num

        return best_num

    def find_bounding_box(self, axis=None, method='pca-cluster', symmetry=False, _from_auto=False):
        """
        Find bounding boxes for the mesh.

        Args:
            axis: String specifying which axes to align to world axes.
                  Can be None (full PCA), 'x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz'
            method: Clustering method ('pca-cluster', 'thickness-aware', 'agglo')
            symmetry: If True, enforce symmetry across YZ plane (x=0)
            _from_auto: Internal flag to prevent recursion
        """
        # If auto mode is enabled, find optimal number first
        if self.auto_num_boxes and not _from_auto:
            self.num_boxes = self.find_optimal_num_boxes(axis=axis, method=method, symmetry=symmetry)

        # Parse axis alignment
        align_x = axis is not None and 'x' in axis.lower() if axis else False
        align_y = axis is not None and 'y' in axis.lower() if axis else False
        align_z = axis is not None and 'z' in axis.lower() if axis else False

        def compute_bbox(vertices_subset):
            mean = np.mean(vertices_subset, axis=0)
            centered = vertices_subset - mean

            try:
                n_aligned = sum([align_x, align_y, align_z])

                if n_aligned == 3:
                    # All axes aligned to world
                    axes = np.eye(3)
                elif n_aligned == 2:
                    # Two axes aligned, one from PCA
                    if align_x and align_y:
                        first = np.array([1.0, 0.0, 0.0])
                        second = np.array([0.0, 1.0, 0.0])
                        third = np.array([0.0, 0.0, 1.0])
                    elif align_x and align_z:
                        first = np.array([1.0, 0.0, 0.0])
                        third = np.array([0.0, 0.0, 1.0])
                        second = np.array([0.0, 1.0, 0.0])
                    else:  # align_y and align_z
                        second = np.array([0.0, 1.0, 0.0])
                        third = np.array([0.0, 0.0, 1.0])
                        first = np.array([1.0, 0.0, 0.0])
                    axes = np.vstack([first, second, third])
                elif n_aligned == 1:
                    # One axis aligned, two from PCA
                    if align_x:
                        first = np.array([1.0, 0.0, 0.0])
                        pca = PCA(n_components=2).fit(centered[:, [1, 2]])
                        yz = pca.components_
                        second = np.array([0.0, yz[0, 0], yz[0, 1]])
                        third = np.array([0.0, yz[1, 0], yz[1, 1]])
                        axes = np.vstack([first, second, third])
                    elif align_y:
                        second = np.array([0.0, 1.0, 0.0])
                        pca = PCA(n_components=2).fit(centered[:, [0, 2]])
                        xz = pca.components_
                        first = np.array([xz[0, 0], 0.0, xz[0, 1]])
                        third = np.array([xz[1, 0], 0.0, xz[1, 1]])
                        axes = np.vstack([first, second, third])
                    else:  # align_z
                        third = np.array([0.0, 0.0, 1.0])
                        pca = PCA(n_components=2).fit(centered[:, [0, 1]])
                        xy = pca.components_
                        first = np.array([xy[0, 0], xy[0, 1], 0.0])
                        second = np.array([xy[1, 0], xy[1, 1], 0.0])
                        axes = np.vstack([first, second, third])
                else:
                    # No alignment, full PCA
                    pca = PCA(n_components=3).fit(centered)
                    axes = pca.components_
            except Exception:
                axes = np.eye(3)

            # Normalize axes
            axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
            axes[2] = np.cross(axes[0], axes[1])

            # If computed y and z axes are nearly in the yz plane, fix x axis
            threshold = 1e-3
            if np.abs(axes[1, 0]) < threshold and np.abs(axes[2, 0]) < threshold:
                xaxis = np.array([1.0, 0.0, 0.0])
                yaxis = np.array([0.0, axes[1, 1], axes[1, 2]])
                if np.linalg.norm(yaxis) > 1e-6:
                    yaxis = yaxis / np.linalg.norm(yaxis)
                else:
                    yaxis = np.array([0.0, 1.0, 0.0])
                zaxis = np.cross(xaxis, yaxis)
                axes = np.vstack([xaxis, yaxis, zaxis])

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

        # For symmetric clustering, use |x| so symmetric vertices cluster together
        if symmetry:
            sym_vertices = target_vertices.copy()
            sym_vertices[:, 0] = np.abs(sym_vertices[:, 0])
        else:
            sym_vertices = target_vertices

        while not bounding_box_complete:
            if method == "agglo":
                clustering = AgglomerativeClustering(n_clusters=self.num_boxes).fit(sym_vertices)
                labels = clustering.labels_

            elif method == "thickness-aware":
                centered = sym_vertices - np.mean(sym_vertices, axis=0)
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
                pca = PCA(n_components=3).fit(sym_vertices)
                proj = sym_vertices @ pca.components_.T
                clustering = KMeans(n_clusters=self.num_boxes, n_init=10).fit(proj)
                labels = clustering.labels_

            else:
                raise ValueError(f"Unknown method: {method}")

            self.sizes = []
            vertex_indices_list = []
            # For clustering-based methods, expand clusters to face-connected vertices.
            if method in ("agglo", "thickness-aware", "pca-cluster"):
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

        # Ensure boxes overlap at connection points while staying tight
        if len(self.corners_list) > 1:
            self._ensure_minimal_overlap(vertex_indices_list)

        self.corners = self.corners_list
        if len(self.corners) > 1:
            means = np.array([np.mean(corners, axis=0) for corners in self.corners])
            # order self.corners with means[:, 1] descending
            order = np.argsort(means[:, 1])[::-1]
            self.corners = [self.corners[i] for i in order]
            self.corners_list = [self.corners_list[i] for i in order]
            self.bbox_axes_list = [self.bbox_axes_list[i] for i in order]
            self.sizes = [self.sizes[i] for i in order]

        # Enforce symmetry across YZ plane (x=0) if requested
        if symmetry and len(self.corners) > 0:
            self._enforce_bbox_symmetry()

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
            bbox_colors = np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]])
        elif num_boxes == 3:
            bbox_colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        elif num_boxes == 4:
            bbox_colors = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.5, 0.5, 0.5]])
        elif num_boxes == 5:
            bbox_colors = np.array([[0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.4], [0.4, 0.3, 0.2], [0.2, 0.3, 0.4]])
        elif num_boxes == 6:
            bbox_colors = np.array([[0.4, 0.3, 0.0], [0.0, 0.4, 0.3], [0.3, 0.0, 0.4], [0.3, 0.3, 0.0], [0.0, 0.3, 0.3], [0.3, 0.0, 0.3]])
        else:
            bbox_colors = np.random.rand(num_boxes, 3)

        # Initialize vertex colors with 4 channels (RGB + transparency).
        vertex_colors = np.zeros((len(self.vertices), 4), dtype=float)
        for idx, (corners, axes) in enumerate(zip(self.corners_list, self.bbox_axes_list)):
            center = np.mean(corners, axis=0)
            local_box_coords = (corners - center) @ axes.T
            lower_bounds = local_box_coords.min(axis=0)
            upper_bounds = local_box_coords.max(axis=0)
            local_vertices = (self.vertices - center) @ axes.T
            inside = np.all((local_vertices >= lower_bounds) & (local_vertices <= upper_bounds), axis=1)
            vertex_colors[inside, :3] += bbox_colors[idx]

        # Set the transparency (4th channel) for all vertices.
        vertex_colors[:, 3] = self.transparency
        self.vertex_colors = vertex_colors[self.faces_3[:, :, 0].flatten()]

    def _ensure_minimal_overlap(self, vertex_indices_list):
        """
        Ensure adjacent boxes have minimal overlap based on mesh connectivity.

        1. Find which box pairs share mesh edges (are connected)
        2. For connected pairs that don't overlap, extend toward connection point
        3. Keep extension minimal - just enough to overlap
        """
        n_boxes = len(self.corners_list)
        if n_boxes <= 1 or vertex_indices_list is None:
            return

        # Build edge connectivity: which vertex pairs are connected by edges
        edge_set = set()
        for face in self.faces_3:
            v_indices = face[:, 0]
            for i in range(3):
                v1, v2 = int(v_indices[i]), int(v_indices[(i + 1) % 3])
                edge_set.add((min(v1, v2), max(v1, v2)))

        # Create vertex-to-box mapping
        vertex_to_box = {}
        for box_idx, v_indices in enumerate(vertex_indices_list):
            for v in v_indices:
                vertex_to_box[v] = box_idx

        # Find connected box pairs and their connection points
        box_connections = {}  # (box_i, box_j) -> list of connection vertices
        for v1, v2 in edge_set:
            if v1 in vertex_to_box and v2 in vertex_to_box:
                box1, box2 = vertex_to_box[v1], vertex_to_box[v2]
                if box1 != box2:
                    key = (min(box1, box2), max(box1, box2))
                    if key not in box_connections:
                        box_connections[key] = []
                    box_connections[key].append(self.vertices[v1])
                    box_connections[key].append(self.vertices[v2])

        # For each connected pair, ensure minimal overlap
        for (box_i, box_j), connection_verts in box_connections.items():
            if len(connection_verts) == 0:
                continue

            connection_center = np.mean(connection_verts, axis=0)

            # Check if boxes currently overlap
            corners_i = self.corners_list[box_i]
            corners_j = self.corners_list[box_j]
            axes_i = self.bbox_axes_list[box_i]
            axes_j = self.bbox_axes_list[box_j]

            center_i = np.mean(corners_i, axis=0)
            center_j = np.mean(corners_j, axis=0)

            # Check overlap using axis-aligned bounds in world space
            min_i, max_i = corners_i.min(axis=0), corners_i.max(axis=0)
            min_j, max_j = corners_j.min(axis=0), corners_j.max(axis=0)

            overlap = np.all(min_i <= max_j) and np.all(min_j <= max_i)

            if not overlap:
                # Need to extend boxes toward connection point

                # For box i: extend toward connection_center
                local_conn_i = (connection_center - center_i) @ axes_i.T
                local_corners_i = (corners_i - center_i) @ axes_i.T
                half_ext_i = np.abs(local_corners_i).max(axis=0)

                # Check if connection point is outside box i
                outside_i = np.abs(local_conn_i) > half_ext_i
                if np.any(outside_i):
                    # Extend box i to include connection point + small margin
                    new_half_ext_i = np.maximum(half_ext_i, np.abs(local_conn_i) * 1.05)

                    # Rebuild corners for box i
                    new_corners_i = np.array(list(product(
                        [-new_half_ext_i[0], new_half_ext_i[0]],
                        [-new_half_ext_i[1], new_half_ext_i[1]],
                        [-new_half_ext_i[2], new_half_ext_i[2]],
                    )))
                    self.corners_list[box_i] = new_corners_i @ axes_i + center_i
                    self.sizes[box_i] = 2 * new_half_ext_i

                # For box j: extend toward connection_center
                local_conn_j = (connection_center - center_j) @ axes_j.T
                local_corners_j = (corners_j - center_j) @ axes_j.T
                half_ext_j = np.abs(local_corners_j).max(axis=0)

                outside_j = np.abs(local_conn_j) > half_ext_j
                if np.any(outside_j):
                    new_half_ext_j = np.maximum(half_ext_j, np.abs(local_conn_j) * 1.05)

                    new_corners_j = np.array(list(product(
                        [-new_half_ext_j[0], new_half_ext_j[0]],
                        [-new_half_ext_j[1], new_half_ext_j[1]],
                        [-new_half_ext_j[2], new_half_ext_j[2]],
                    )))
                    self.corners_list[box_j] = new_corners_j @ axes_j + center_j
                    self.sizes[box_j] = 2 * new_half_ext_j

    def _refine_boxes_iterative(self, compute_bbox_func, max_iters=5):
        """
        Iteratively refine box assignments to minimize total volume.

        For each vertex, check if moving it to a different box would reduce
        total volume. Repeat until convergence or max iterations.
        """
        n_boxes = len(self.corners_list)
        if n_boxes <= 1:
            return

        # Build vertex-to-box assignment from current boxes
        vertex_assignments = np.zeros(len(self.vertices), dtype=int)

        for i, (corners, axes) in enumerate(zip(self.corners_list, self.bbox_axes_list)):
            center = np.mean(corners, axis=0)
            local_corners = (corners - center) @ axes.T
            half_extents = np.abs(local_corners).max(axis=0)

            verts_local = (self.vertices - center) @ axes.T
            inside = np.all(np.abs(verts_local) <= half_extents + 0.01, axis=1)
            vertex_assignments[inside] = i

        # Compute initial total volume
        def compute_total_volume(assignments):
            total = 0.0
            box_vertices = [[] for _ in range(n_boxes)]
            for v_idx, box_idx in enumerate(assignments):
                box_vertices[box_idx].append(self.vertices[v_idx])

            for i in range(n_boxes):
                if len(box_vertices[i]) < 4:
                    continue
                verts = np.array(box_vertices[i])
                _, _, sizes = compute_bbox_func(verts)
                total += np.prod(sizes)
            return total

        best_volume = compute_total_volume(vertex_assignments)

        for iteration in range(max_iters):
            improved = False

            # Try reassigning boundary vertices
            for v_idx in range(len(self.vertices)):
                current_box = vertex_assignments[v_idx]
                vertex = self.vertices[v_idx]

                # Check if this vertex is near a box boundary
                corners = self.corners_list[current_box]
                center = np.mean(corners, axis=0)
                axes = self.bbox_axes_list[current_box]
                local = (vertex - center) @ axes.T
                local_corners = (corners - center) @ axes.T
                half_extents = np.abs(local_corners).max(axis=0)

                # Only consider vertices near boundaries (within 20% of extent)
                boundary_dist = half_extents - np.abs(local)
                if np.min(boundary_dist / (half_extents + 1e-6)) > 0.2:
                    continue

                # Try moving to each other box
                best_new_box = current_box
                for new_box in range(n_boxes):
                    if new_box == current_box:
                        continue

                    # Check if vertex is close to this box
                    other_corners = self.corners_list[new_box]
                    other_center = np.mean(other_corners, axis=0)
                    dist_to_other = np.linalg.norm(vertex - other_center)
                    dist_to_current = np.linalg.norm(vertex - center)

                    # Only consider if reasonably close
                    if dist_to_other > dist_to_current * 2:
                        continue

                    # Temporarily reassign and compute new volume
                    test_assignments = vertex_assignments.copy()
                    test_assignments[v_idx] = new_box
                    new_volume = compute_total_volume(test_assignments)

                    if new_volume < best_volume - 1e-9:
                        best_volume = new_volume
                        best_new_box = new_box
                        improved = True

                vertex_assignments[v_idx] = best_new_box

            if not improved:
                break

        # Rebuild boxes from final assignments
        self.corners_list = []
        self.bbox_axes_list = []
        self.sizes = []

        for i in range(n_boxes):
            box_mask = vertex_assignments == i
            if np.sum(box_mask) < 4:
                continue

            box_verts = self.vertices[box_mask]
            corners, axes, sizes = compute_bbox_func(box_verts)
            self.corners_list.append(corners)
            self.bbox_axes_list.append(axes)
            self.sizes.append(sizes)

    def _compute_tight_box(self, vertices, axes):
        """
        Compute the tightest bounding box for vertices with given axes.
        Returns optimal center and minimal half_extents.
        """
        proj = vertices @ axes.T
        min_proj = proj.min(axis=0)
        max_proj = proj.max(axis=0)

        # Optimal local center is at midpoint of min/max
        local_center = (min_proj + max_proj) / 2
        half_extents = (max_proj - min_proj) / 2

        # Transform local center to world coords
        world_center = local_center @ axes

        return world_center, half_extents

    def _enforce_bbox_symmetry(self):
        """
        Enforce symmetry of bounding boxes across the YZ plane (x=0).
        Also computes tightest possible boxes while maintaining symmetry.

        For odd number of boxes:
        - Middle box (closest to x=0) gets X-axis aligned and centered at x=0
        - Paired boxes get mirrored rotations and symmetric centers/sizes

        For even number of boxes:
        - Paired boxes get mirrored rotations and symmetric centers/sizes
        """
        if not hasattr(self, 'corners') or len(self.corners) == 0:
            return

        if self.vertices is None or len(self.vertices) == 0:
            return

        n_boxes = len(self.corners)

        # Collect box data with centers
        box_data = []
        for i in range(n_boxes):
            corners = self.corners[i]
            axes = self.bbox_axes_list[i].copy()
            center = np.mean(corners, axis=0)

            # Get current extents in local coordinates
            local_corners = (corners - center) @ axes.T
            half_extents = np.abs(local_corners).max(axis=0)

            # Find vertices belonging to this box
            verts_local = (self.vertices - center) @ axes.T
            margin = 0.01
            inside = np.all(np.abs(verts_local) <= half_extents + margin, axis=1)

            if np.sum(inside) > 0:
                box_vertices = self.vertices[inside]
            else:
                box_vertices = self.vertices

            box_data.append({
                'original_idx': i,
                'center': center,
                'axes': axes,
                'half_extents': half_extents,
                'vertices': box_vertices
            })

        # Sort boxes by X-coordinate of center to properly pair left/right
        box_data.sort(key=lambda b: b['center'][0])

        # Find pairs: leftmost with rightmost, etc.
        pairs = []
        middle_idx = None

        if n_boxes % 2 == 1:
            # Find box closest to x=0
            x_coords = [b['center'][0] for b in box_data]
            middle_idx = np.argmin(np.abs(x_coords))

            # Pair others: left side with right side
            left_indices = [i for i in range(middle_idx)]
            right_indices = [i for i in range(middle_idx + 1, n_boxes)][::-1]

            for li, ri in zip(left_indices, right_indices):
                pairs.append((li, ri))
        else:
            # Even: pair left half with right half
            half = n_boxes // 2
            for i in range(half):
                pairs.append((i, n_boxes - 1 - i))

        # Process middle box: X-axis align it
        if middle_idx is not None:
            mid_data = box_data[middle_idx]
            axes = mid_data['axes'].copy()
            verts = mid_data['vertices']

            # Find which axis is most aligned with X
            x_alignment = np.abs(axes[:, 0])
            primary_idx = np.argmax(x_alignment)

            # Build new axes with that axis becoming pure X
            new_axes = np.zeros((3, 3))
            x_sign = np.sign(axes[primary_idx, 0]) if axes[primary_idx, 0] != 0 else 1.0
            new_axes[primary_idx] = np.array([x_sign, 0.0, 0.0])

            # Get the other two axis indices
            other_indices = [j for j in range(3) if j != primary_idx]

            # Project them to YZ plane and orthogonalize
            for j in other_indices:
                yz = np.array([0.0, axes[j, 1], axes[j, 2]])
                norm = np.linalg.norm(yz)
                if norm > 1e-6:
                    new_axes[j] = yz / norm
                else:
                    if j == other_indices[0]:
                        new_axes[j] = np.array([0.0, 1.0, 0.0])
                    else:
                        new_axes[j] = np.array([0.0, 0.0, 1.0])

            # Orthogonalize using Gram-Schmidt on the two YZ-plane axes
            idx0, idx1 = other_indices
            dot = np.dot(new_axes[idx0], new_axes[idx1])
            new_axes[idx1] = new_axes[idx1] - dot * new_axes[idx0]
            norm1 = np.linalg.norm(new_axes[idx1])
            if norm1 > 1e-6:
                new_axes[idx1] = new_axes[idx1] / norm1
            else:
                new_axes[idx1] = np.array([0.0, -new_axes[idx0, 2], new_axes[idx0, 1]])
                new_axes[idx1] = new_axes[idx1] / np.linalg.norm(new_axes[idx1])

            # Compute tight box with new axes
            optimal_center, half_extents = self._compute_tight_box(verts, new_axes)

            # Force x=0 for symmetry, keep optimal y and z
            new_center = np.array([0.0, optimal_center[1], optimal_center[2]])

            # Recompute extents with x=0 constraint
            proj = (verts - new_center) @ new_axes.T
            min_proj = proj.min(axis=0)
            max_proj = proj.max(axis=0)

            # For axes not aligned with X, use tight extents
            # For X-aligned axis, need to cover from x=0
            for k in range(3):
                if np.abs(new_axes[k, 0]) > 0.9:  # This is the X-aligned axis
                    half_extents[k] = max(np.abs(min_proj[k]), np.abs(max_proj[k]))
                else:
                    half_extents[k] = (max_proj[k] - min_proj[k]) / 2

            box_data[middle_idx]['axes'] = new_axes
            box_data[middle_idx]['center'] = new_center
            box_data[middle_idx]['half_extents'] = half_extents

        # Process pairs: find optimal symmetric configuration
        for li, ri in pairs:
            left_data = box_data[li]
            right_data = box_data[ri]

            verts_left = left_data['vertices']
            verts_right = right_data['vertices']

            # Use left box axes as reference
            axes_left = left_data['axes'].copy()
            axes_right = axes_left.copy()
            axes_right[:, 0] = -axes_right[:, 0]  # Mirror X component

            # Compute tight boxes for each with their axes
            center_left_opt, _ = self._compute_tight_box(verts_left, axes_left)
            center_right_opt, _ = self._compute_tight_box(verts_right, axes_right)

            # Find optimal symmetric center that minimizes max box volume
            best_volume = float('inf')
            best_center = None

            # Search ranges based on optimal centers
            all_verts = np.vstack([verts_left, verts_right])
            spread = all_verts.max(axis=0) - all_verts.min(axis=0)
            margin = spread * 0.1

            x_range = np.linspace(
                min(np.abs(center_left_opt[0]), np.abs(center_right_opt[0])) * 0.8,
                max(np.abs(center_left_opt[0]), np.abs(center_right_opt[0])) * 1.2,
                5
            )
            y_range = np.linspace(
                min(center_left_opt[1], center_right_opt[1]) - margin[1],
                max(center_left_opt[1], center_right_opt[1]) + margin[1],
                5
            )
            z_range = np.linspace(
                min(center_left_opt[2], center_right_opt[2]) - margin[2],
                max(center_left_opt[2], center_right_opt[2]) + margin[2],
                5
            )

            for x in x_range:
                for y in y_range:
                    for z in z_range:
                        test_center_left = np.array([-x, y, z])
                        test_center_right = np.array([x, y, z])

                        # Compute extents for both boxes
                        proj_l = (verts_left - test_center_left) @ axes_left.T
                        proj_r = (verts_right - test_center_right) @ axes_right.T

                        ext_l = np.maximum(np.abs(proj_l.min(axis=0)), np.abs(proj_l.max(axis=0)))
                        ext_r = np.maximum(np.abs(proj_r.min(axis=0)), np.abs(proj_r.max(axis=0)))

                        max_ext = np.maximum(ext_l, ext_r)
                        volume = np.prod(max_ext)

                        if volume < best_volume:
                            best_volume = volume
                            best_center = (x, y, z)

            # Use best center found
            x, y, z = best_center
            new_center_left = np.array([-x, y, z])
            new_center_right = np.array([x, y, z])

            # Recompute extents with optimal symmetric centers
            proj_left = (verts_left - new_center_left) @ axes_left.T
            proj_right = (verts_right - new_center_right) @ axes_right.T

            extents_left = np.maximum(np.abs(proj_left.min(axis=0)), np.abs(proj_left.max(axis=0)))
            extents_right = np.maximum(np.abs(proj_right.min(axis=0)), np.abs(proj_right.max(axis=0)))

            max_extents = np.maximum(extents_left, extents_right)

            left_data['center'] = new_center_left
            right_data['center'] = new_center_right
            left_data['axes'] = axes_left
            right_data['axes'] = axes_right
            left_data['half_extents'] = max_extents.copy()
            right_data['half_extents'] = max_extents.copy()

        # Rebuild corners for all boxes
        for data in box_data:
            half_extents = data['half_extents']
            axes = data['axes']
            center = data['center']

            new_corners = np.array(list(product(
                [-half_extents[0], half_extents[0]],
                [-half_extents[1], half_extents[1]],
                [-half_extents[2], half_extents[2]],
            )))
            new_corners = new_corners @ axes + center

            orig_idx = data['original_idx']
            self.corners[orig_idx] = new_corners
            self.corners_list[orig_idx] = new_corners
            self.bbox_axes_list[orig_idx] = axes
            self.sizes[orig_idx] = 2 * half_extents

    # ========================================================================
    # Skeleton Binding Methods - For Muscle-Skeleton Attachment
    # ========================================================================

    def _assign_fixed_vertices_to_nearest_bones(self, skeleton, skeleton_meshes, mesh_to_body):
        """
        Assign each fixed vertex to its nearest bone.
        This supports muscles like psoas major that attach to multiple vertebrae.

        For each fixed vertex (origin/insertion), finds the closest bone mesh
        and attaches the vertex to that specific bone.

        Args:
            skeleton: DART skeleton object
            skeleton_meshes: Dict of mesh_name -> MeshLoader
            mesh_to_body: Dict mapping mesh_name -> body_name
        """
        if skeleton is None or skeleton_meshes is None:
            return

        if not hasattr(self, 'soft_body_fixed_vertices') or len(self.soft_body_fixed_vertices) == 0:
            return

        # Build KD-trees for each skeleton mesh for fast nearest-point queries
        bone_trees = {}  # mesh_name -> (KDTree, body_name)
        for mesh_name, mesh_loader in skeleton_meshes.items():
            if mesh_loader.vertices is None or len(mesh_loader.vertices) == 0:
                continue

            # Get body name for this mesh
            body_name = mesh_to_body.get(mesh_name)
            if body_name is None:
                # Try to find body node directly
                try:
                    body_node = skeleton.getBodyNode(mesh_name)
                    if body_node is not None:
                        body_name = mesh_name
                except:
                    pass

            if body_name is None:
                continue

            # Verify body exists in skeleton
            try:
                body_node = skeleton.getBodyNode(body_name)
                if body_node is None:
                    continue
            except:
                continue

            # Build KD-tree from mesh vertices
            bone_trees[mesh_name] = (cKDTree(mesh_loader.vertices), body_name)

        if len(bone_trees) == 0:
            print(f"  No valid bone meshes found for nearest-bone assignment")
            return

        # For each fixed vertex, find the nearest bone
        assigned_count = 0
        bones_used = set()

        for fixed_vi in self.soft_body_fixed_vertices:
            if fixed_vi in self.soft_body_local_anchors:
                # Already assigned (e.g., cap centroid)
                bones_used.add(self.soft_body_local_anchors[fixed_vi][0])
                continue

            vertex_pos = self.soft_body.rest_positions[fixed_vi]

            # Find nearest bone mesh
            min_dist = float('inf')
            nearest_body = None
            nearest_mesh = None

            for mesh_name, (tree, body_name) in bone_trees.items():
                dist, _ = tree.query(vertex_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_body = body_name
                    nearest_mesh = mesh_name

            if nearest_body is not None:
                try:
                    body_node = skeleton.getBodyNode(nearest_body)
                    if body_node is not None:
                        world_transform = body_node.getWorldTransform()
                        rotation = world_transform.rotation()
                        translation = world_transform.translation()
                        local_pos = rotation.T @ (vertex_pos - translation)
                        self.soft_body_local_anchors[fixed_vi] = (nearest_body, local_pos.copy())
                        assigned_count += 1
                        bones_used.add(nearest_body)
                except Exception as e:
                    continue

        print(f"  Nearest-bone assignment: {assigned_count} vertices -> {len(bones_used)} bones")
        if len(bones_used) > 1:
            print(f"    Bones: {sorted(bones_used)}")

    def _compute_skinning_weights(self, skeleton, mesh_to_body, skeleton_names):
        """
        Compute LBS skinning weights using distance to anchors with better falloff.
        """
        self.skinning_bones = []
        self.skinning_weights = None
        self.skinning_local_positions = {}
        self.skinning_initial_transforms = {}

        if skeleton is None or not hasattr(self, 'tet_cap_attachments') or len(self.tet_cap_attachments) == 0:
            print("  Warning: No cap attachments for skinning")
            return

        # Collect unique body names from cap attachments
        body_names_set = set()
        anchor_to_body = {}

        for attachment in self.tet_cap_attachments:
            anchor_idx, stream_idx, end_type, skel_mesh_idx, subpart_idx = attachment
            if anchor_idx in self.soft_body_local_anchors:
                body_name = self.soft_body_local_anchors[anchor_idx][0]
                body_names_set.add(body_name)
                anchor_to_body[anchor_idx] = body_name

        if len(body_names_set) == 0:
            print("  Warning: No bones found for skinning")
            return

        self.skinning_bones = list(body_names_set)
        num_bones = len(self.skinning_bones)
        num_verts = len(self.soft_body.rest_positions)

        print(f"  Computing skinning weights for {num_bones} bones: {self.skinning_bones}")

        # Compute weights based on distance to anchor points
        bone_anchor_positions = {bone: [] for bone in self.skinning_bones}
        for anchor_idx, body_name in anchor_to_body.items():
            if body_name in bone_anchor_positions:
                anchor_pos = self.soft_body.rest_positions[anchor_idx]
                bone_anchor_positions[body_name].append(anchor_pos)
                print(f"    Bone '{body_name}': anchor at {anchor_pos}")

        # For each vertex, compute weight to each bone
        self.skinning_weights = np.zeros((num_verts, num_bones))
        vertices = self.soft_body.rest_positions

        # Compute characteristic length scale for falloff
        bbox_size = np.ptp(vertices, axis=0)
        char_length = np.max(bbox_size)
        falloff_distance = char_length * 0.5  # 50% of bounding box

        for bone_idx, bone_name in enumerate(self.skinning_bones):
            anchors = bone_anchor_positions.get(bone_name, [])
            if len(anchors) == 0:
                print(f"  Warning: Bone '{bone_name}' has no anchors")
                continue

            anchors = np.array(anchors)

            # For each vertex, find distance to closest anchor of this bone
            diff = vertices[:, np.newaxis, :] - anchors[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            min_dists = np.min(dists, axis=1)

            # === Improved weight function: smooth falloff ===
            # Use normalized distance with smooth falloff
            normalized_dist = min_dists / falloff_distance

            # Smooth kernel: (1 - d²)² for d < 1, else 0
            # This gives smooth gradients and compact support
            weights = np.where(
                normalized_dist < 1.0,
                (1.0 - normalized_dist**2)**2,
                0.0
            )

            self.skinning_weights[:, bone_idx] = weights

            print(f"    Bone '{bone_name}': {np.sum(weights > 0.01)} vertices affected")

        # Normalize weights per vertex
        weight_sums = np.sum(self.skinning_weights, axis=1, keepdims=True)
        valid_sums = weight_sums > 1e-10

        if not np.all(valid_sums):
            num_zero = np.sum(~valid_sums)
            print(f"  Warning: {num_zero} vertices have zero total weight")
            # For vertices with zero weight, use equal weights for all bones
            self.skinning_weights[~valid_sums.flatten()] = 1.0 / num_bones
            weight_sums = np.sum(self.skinning_weights, axis=1, keepdims=True)
            valid_sums = weight_sums > 1e-10

        self.skinning_weights = self.skinning_weights / weight_sums

        # Override: Fixed vertices get 100% weight to their attached bone
        for anchor_idx, body_name in anchor_to_body.items():
            if body_name in self.skinning_bones:
                bone_idx = self.skinning_bones.index(body_name)
                if anchor_idx < num_verts:
                    self.skinning_weights[anchor_idx, :] = 0.0
                    self.skinning_weights[anchor_idx, bone_idx] = 1.0

        print(f"  Skinning weights computed: {num_verts} vertices, {num_bones} bones")

        # Debug: print weight statistics
        for bone_idx, bone_name in enumerate(self.skinning_bones):
            bone_weights = self.skinning_weights[:, bone_idx]
            dominant = np.sum(bone_weights > 0.5)
            print(f"    Bone '{bone_name}': {dominant} vertices with weight > 0.5")

    def _apply_simple_initial_transform(self, skeleton):
        """
        Simple initial positioning: interpolate between origin and insertion transforms.
        Much simpler than LBS, often more robust for muscles.
        """
        if skeleton is None or self.soft_body is None:
            return

        if not hasattr(self, 'soft_body_local_anchors') or len(self.soft_body_local_anchors) < 2:
            print("  Warning: Need at least 2 anchors (origin and insertion)")
            return

        # Use tet_cap_attachments to identify actual cap centroids (not all fixed vertices)
        # Each entry: (anchor_idx, stream_idx, end_type, skel_mesh_idx, subpart_idx)
        # end_type: 0=origin, 1=insertion
        origin_anchor = None
        insertion_anchor = None

        if hasattr(self, 'tet_cap_attachments') and len(self.tet_cap_attachments) > 0:
            for attachment in self.tet_cap_attachments:
                anchor_idx, stream_idx, end_type, skel_mesh_idx, subpart_idx = attachment
                if anchor_idx not in self.soft_body_local_anchors:
                    continue

                body_name, local_pos = self.soft_body_local_anchors[anchor_idx]
                body_node = skeleton.getBodyNode(body_name)
                if body_node is None:
                    continue

                # Current world transform
                world_transform = body_node.getWorldTransform()
                R_curr = world_transform.rotation()
                T_curr = world_transform.translation()
                world_pos = R_curr @ local_pos + T_curr

                # Get rest-pose rotation from stored transforms
                R_rest = np.eye(3)
                if hasattr(self, 'skeleton_rest_transforms') and body_name in self.skeleton_rest_transforms:
                    R_rest, _ = self.skeleton_rest_transforms[body_name]

                # Compute delta rotation: how much the bone has rotated from rest
                R_delta = R_curr @ R_rest.T

                # Rest position
                rest_pos = self.soft_body.rest_positions[anchor_idx]

                anchor_info = {
                    'anchor_idx': anchor_idx,
                    'rest_pos': rest_pos,
                    'current_pos': world_pos,
                    'body_name': body_name,
                    'R_delta': R_delta,
                }

                if end_type == 0 and origin_anchor is None:
                    origin_anchor = anchor_info
                elif end_type == 1 and insertion_anchor is None:
                    insertion_anchor = anchor_info

                if origin_anchor is not None and insertion_anchor is not None:
                    break

        if origin_anchor is None or insertion_anchor is None:
            print("  Warning: Could not find origin and insertion cap centroids")
            return

        origin = origin_anchor
        insertion = insertion_anchor

        # Compute axis-aligned distance for interpolation
        rest_vertices = self.soft_body.rest_positions

        # Use primary axis (typically Y or longest axis)
        rest_axis = insertion['rest_pos'] - origin['rest_pos']
        rest_length = np.linalg.norm(rest_axis)
        if rest_length < 1e-6:
            print("  Warning: Origin and insertion are too close")
            return
        rest_axis_norm = rest_axis / rest_length

        # Project each vertex onto muscle axis to get interpolation parameter
        vertex_projections = np.dot(rest_vertices - origin['rest_pos'], rest_axis_norm)
        t = vertex_projections / rest_length  # 0 at origin, 1 at insertion
        t = np.clip(t, 0.0, 1.0)[:, np.newaxis]  # Clamp and add dimension for broadcasting

        # Get transforms
        origin_rest = origin['rest_pos']
        origin_curr = origin['current_pos']
        origin_R_delta = origin['R_delta']

        insertion_rest = insertion['rest_pos']
        insertion_curr = insertion['current_pos']
        insertion_R_delta = insertion['R_delta']

        # Check if rotation should be applied (only if bones have actually rotated)
        # If R_delta is close to identity, skip rotation to avoid numerical issues
        origin_has_rotation = np.linalg.norm(origin_R_delta - np.eye(3)) > 0.01
        insertion_has_rotation = np.linalg.norm(insertion_R_delta - np.eye(3)) > 0.01
        use_rotation = origin_has_rotation or insertion_has_rotation

        # Transform vertices
        new_positions = np.zeros_like(rest_vertices)

        for i in range(len(rest_vertices)):
            v_rest = rest_vertices[i]
            ti = t[i, 0]

            # Origin transform
            v_from_origin = v_rest - origin_rest
            if use_rotation:
                v_origin_transformed = origin_curr + origin_R_delta @ v_from_origin
            else:
                v_origin_transformed = origin_curr + v_from_origin

            # Insertion transform
            v_from_insertion = v_rest - insertion_rest
            if use_rotation:
                v_insertion_transformed = insertion_curr + insertion_R_delta @ v_from_insertion
            else:
                v_insertion_transformed = insertion_curr + v_from_insertion

            # Blend between origin and insertion transforms
            new_positions[i] = (1 - ti) * v_origin_transformed + ti * v_insertion_transformed

        # Apply
        self.soft_body.positions = new_positions.copy()

        # Debug
        displacements = np.linalg.norm(new_positions - rest_vertices, axis=1)
        rot_str = "with rotation" if use_rotation else "translation only"
        print(f"  Transform ({rot_str}): mean displacement={np.mean(displacements):.4f}, "
              f"max={np.max(displacements):.4f}")

    def _apply_lbs_to_all_vertices(self, skeleton):
        """
        Apply Linear Blend Skinning to ALL vertices for initial positioning.
        Uses simple bone-relative transforms without complex delta calculations.
        """
        if skeleton is None or self.soft_body is None:
            return

        if not hasattr(self, 'skinning_bones') or len(self.skinning_bones) == 0:
            print("  Warning: No skinning weights computed, skipping LBS")
            return

        if not hasattr(self, 'skinning_weights') or self.skinning_weights is None:
            print("  Warning: No skinning weights, skipping LBS")
            return

        num_verts = len(self.soft_body.rest_positions)
        rest_positions = self.soft_body.rest_positions

        # Compute weighted average position for each vertex
        new_positions = np.zeros((num_verts, 3))

        for bone_idx, bone_name in enumerate(self.skinning_bones):
            body_node = skeleton.getBodyNode(bone_name)
            if body_node is None:
                continue

            # Current world transform
            world_transform = body_node.getWorldTransform()
            R_curr = world_transform.rotation()
            T_curr = world_transform.translation()

            # Get rest/initial transform
            if bone_name in self.skeleton_rest_transforms:
                R_rest, T_rest = self.skeleton_rest_transforms[bone_name]
            else:
                print(f"  Warning: No rest transform for {bone_name}, using identity")
                R_rest, T_rest = np.eye(3), np.zeros(3)

            # === Key fix: Proper bone-space transformation ===
            # 1. Transform rest vertices to bone's local space
            #    local = R_rest^T @ (v_rest - T_rest)
            # 2. Transform back using current pose
            #    v_new = R_curr @ local + T_curr

            # Combined: v_new = R_curr @ R_rest^T @ (v_rest - T_rest) + T_curr
            local_vertices = (R_rest.T @ (rest_positions - T_rest).T).T
            transformed_vertices = (R_curr @ local_vertices.T).T + T_curr

            # Add weighted contribution
            weights = self.skinning_weights[:, bone_idx:bone_idx+1]
            new_positions += weights * transformed_vertices

        # Sanity check: if all weights are zero for some vertices, keep rest position
        total_weights = np.sum(self.skinning_weights, axis=1)
        zero_weight_mask = total_weights < 1e-6
        if np.any(zero_weight_mask):
            print(f"  Warning: {np.sum(zero_weight_mask)} vertices have zero weights, keeping rest positions")
            new_positions[zero_weight_mask] = rest_positions[zero_weight_mask]

        # Apply to soft body
        self.soft_body.positions = new_positions.copy()

        # Debug: print displacement statistics
        displacements = np.linalg.norm(new_positions - rest_positions, axis=1)
        print(f"  LBS applied: mean displacement={np.mean(displacements):.4f}, "
              f"max={np.max(displacements):.4f}, min={np.min(displacements):.4f}")

    def _compute_vertex_bone_assignments(self, skeleton):
        """
        Assign each vertex to its closest bone based on rest pose positions.
        Uses skeleton mesh bounding boxes for efficient nearest-bone lookup.
        """
        if self.soft_body is None:
            return

        self.vertex_bone_assignments = {}
        rest_pos = self.soft_body.rest_positions
        n_verts = len(rest_pos)

        # Get list of bones that have anchors (these are the relevant bones for this muscle)
        anchor_bones = set()
        if hasattr(self, 'soft_body_local_anchors'):
            for anchor_idx, (bone_name, local_pos) in self.soft_body_local_anchors.items():
                anchor_bones.add(bone_name)

        if len(anchor_bones) == 0:
            return

        # For each bone, get its initial transform and compute distances
        bone_data = {}  # bone_name -> (init_rot, init_trans, centroid)
        for bone_name in anchor_bones:
            body_node = skeleton.getBodyNode(bone_name)
            if body_node is None:
                continue

            if bone_name in self.skinning_initial_transforms:
                init_rot, init_trans = self.skinning_initial_transforms[bone_name]
            else:
                wt = body_node.getWorldTransform()
                init_rot, init_trans = wt.rotation(), wt.translation()
                self.skinning_initial_transforms[bone_name] = (init_rot.copy(), init_trans.copy())

            bone_data[bone_name] = (init_rot, init_trans)

        if len(bone_data) == 0:
            return

        # Assign each vertex to the nearest bone (by distance to bone's initial translation)
        bone_names = list(bone_data.keys())
        bone_positions = np.array([bone_data[bn][1] for bn in bone_names])

        for i in range(n_verts):
            p = rest_pos[i]
            distances = np.linalg.norm(bone_positions - p, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_bone = bone_names[nearest_idx]

            # Store local position relative to bone's initial transform
            init_rot, init_trans = bone_data[nearest_bone]
            local_pos = init_rot.T @ (p - init_trans)

            self.vertex_bone_assignments[i] = (nearest_bone, local_pos)

        print(f"  Assigned {n_verts} vertices to {len(bone_names)} bones")

    def _build_transformed_collision_meshes(self, skeleton_meshes, skeleton, verbose=False):
        """
        Build collision meshes transformed to current DART skeleton pose.

        Args:
            skeleton_meshes: Dict of mesh_name -> MeshLoader (with .trimesh attribute)
            skeleton: DART skeleton object
            verbose: Whether to print debug info

        Returns:
            List of trimesh objects in world coordinates
        """
        import trimesh

        collision_trimeshes = []
        skipped_no_trimesh = 0
        skipped_no_body = 0

        if skeleton_meshes is None or skeleton is None:
            return collision_trimeshes

        for mesh_name, mesh_loader in skeleton_meshes.items():
            # Check if mesh has trimesh
            if not hasattr(mesh_loader, 'trimesh') or mesh_loader.trimesh is None:
                skipped_no_trimesh += 1
                continue

            try:
                # Find corresponding DART body node
                body_node = None
                try:
                    body_node = skeleton.getBodyNode(mesh_name)
                except:
                    pass

                # Try common name variations if not found
                if body_node is None:
                    # Try with _L/_R suffix handling
                    name_variants = [
                        mesh_name,
                        mesh_name.replace('_L', '_l').replace('_R', '_r'),
                        mesh_name.replace('_l', '_L').replace('_r', '_R'),
                    ]
                    for variant in name_variants:
                        try:
                            body_node = skeleton.getBodyNode(variant)
                            if body_node is not None:
                                break
                        except:
                            continue

                if body_node is None:
                    skipped_no_body += 1
                    continue

                # Get world transform from DART body node
                world_transform = body_node.getWorldTransform()
                rotation = world_transform.rotation()  # 3x3
                translation = world_transform.translation()  # 3x1

                # Transform mesh vertices to world coordinates
                original_verts = mesh_loader.trimesh.vertices.copy()
                transformed_verts = (rotation @ original_verts.T).T + translation

                # Create transformed trimesh
                transformed_mesh = trimesh.Trimesh(
                    vertices=transformed_verts,
                    faces=mesh_loader.trimesh.faces.copy(),
                    process=False
                )
                collision_trimeshes.append(transformed_mesh)

            except Exception as e:
                if verbose:
                    print(f"  Collision: error transforming {mesh_name}: {e}")
                continue

        if verbose:
            print(f"  Collision: built {len(collision_trimeshes)} DART-transformed meshes "
                  f"(skipped: {skipped_no_trimesh} no trimesh, {skipped_no_body} no body node)")

        return collision_trimeshes

    def _update_fixed_targets_from_skeleton(self, skeleton_meshes, skeleton):
        """
        Update fixed vertex positions based on skeleton body transforms.
        Uses stored local anchor positions to transform with body movement.

        Args:
            skeleton_meshes: Dict of skeleton name -> MeshLoader
            skeleton: DART skeleton object
        """
        if self.soft_body is None:
            print("  _update: soft_body is None")
            return

        if skeleton is None:
            print("  _update: skeleton is None")
            return

        if not hasattr(self, 'soft_body_local_anchors') or len(self.soft_body_local_anchors) == 0:
            print("  _update: no local anchors stored")
            return

        # Compute new target positions for fixed vertices
        new_targets = self.soft_body.fixed_targets.copy() if self.soft_body.fixed_targets is not None else None
        if new_targets is None:
            print("  _update: fixed_targets is None")
            return

        # Map from anchor vertex index to its position in fixed_indices array
        fixed_indices = self.soft_body.fixed_indices
        anchor_to_fixed_idx = {int(vi): idx for idx, vi in enumerate(fixed_indices)}

        updated_count = 0
        for anchor_idx, (body_name, local_pos) in self.soft_body_local_anchors.items():
            anchor_idx = int(anchor_idx)
            if anchor_idx not in anchor_to_fixed_idx:
                print(f"  _update: anchor {anchor_idx} not in fixed_indices")
                continue

            fixed_idx = anchor_to_fixed_idx[anchor_idx]

            try:
                body_node = skeleton.getBodyNode(body_name)

                # Fallback: try robust matching if direct lookup fails
                if body_node is None:
                    normalized = body_name.lower().replace('_', '').replace('-', '')
                    for i in range(skeleton.getNumBodyNodes()):
                        bn = skeleton.getBodyNode(i)
                        bn_name = bn.getName()
                        bn_normalized = bn_name.lower().replace('_', '').replace('-', '')

                        if normalized == bn_normalized:
                            body_node = bn
                            # Update stored body name for future lookups
                            self.soft_body_local_anchors[anchor_idx] = (bn_name, local_pos)
                            print(f"  _update: remapped '{body_name}' -> '{bn_name}'")
                            break

                        if normalized in bn_normalized or bn_normalized in normalized:
                            body_node = bn
                            self.soft_body_local_anchors[anchor_idx] = (bn_name, local_pos)
                            print(f"  _update: remapped '{body_name}' -> '{bn_name}'")
                            break

                if body_node is None:
                    print(f"  _update: body node '{body_name}' not found in skeleton")
                    all_bodies = [skeleton.getBodyNode(i).getName() for i in range(skeleton.getNumBodyNodes())]
                    print(f"  Available bodies: {all_bodies}")
                    continue

                # Get current world transform of the body
                world_transform = body_node.getWorldTransform()
                rotation = world_transform.rotation()
                translation = world_transform.translation()

                # Transform local position to world position
                # world_pos = R @ local_pos + t
                new_world_pos = rotation @ local_pos + translation

                old_pos = new_targets[fixed_idx].copy()
                new_targets[fixed_idx] = new_world_pos
                updated_count += 1

            except Exception as e:
                print(f"  _update: exception for anchor {anchor_idx}: {e}")
                continue

        if not getattr(self, '_baking_mode', False):
            print(f"  _update: updated {updated_count} anchor targets")
        self.soft_body.set_fixed_targets(new_targets)

    def _update_proximity_targets_from_skeleton(self, skeleton):
        """
        Update proximity soft constraint targets based on current skeleton pose.
        Transforms local positions to world positions using current bone transforms.
        """
        if skeleton is None:
            return
        if not hasattr(self, 'bone_proximal_vertices') or len(self.bone_proximal_vertices) == 0:
            return
        if self.soft_body.proximity_indices is None:
            return

        # Build new targets array
        new_targets = []
        for vi in self.soft_body.proximity_indices:
            if vi in self.bone_proximal_vertices:
                body_name, local_pos, dist = self.bone_proximal_vertices[vi]

                body_node = skeleton.getBodyNode(body_name)
                if body_node is not None:
                    world_transform = body_node.getWorldTransform()
                    rotation = world_transform.rotation()
                    translation = world_transform.translation()

                    # Transform local position to world
                    world_pos = rotation @ local_pos + translation
                    new_targets.append(world_pos)
                else:
                    # Fallback: keep current position
                    new_targets.append(self.soft_body.positions[vi])
            else:
                new_targets.append(self.soft_body.positions[vi])

        if len(new_targets) > 0:
            self.soft_body.update_proximity_targets(np.array(new_targets))

    def _compute_tet_skeleton_bindings(self, skeleton_meshes, skeleton):
        """
        For each tet vertex, compute blend weights to origin and insertion bones.
        Uses distance along muscle axis to smoothly interpolate between bones.

        Stores:
            self.tet_skeleton_bindings[vertex_idx] = (origin_body, insertion_body, weight, orig_pos)
            self.tet_initial_bone_transforms[body_name] = (rotation, translation)
        """
        if skeleton_meshes is None or skeleton is None:
            print("  Skeleton bindings: no skeleton provided")
            return
        if not hasattr(self, 'tet_vertices') or self.tet_vertices is None:
            print("  Skeleton bindings: no tet vertices")
            return
        if not hasattr(self, 'attach_skeletons') or len(self.attach_skeletons) == 0:
            print("  Skeleton bindings: no attach_skeletons info")
            return

        tet_verts = np.array(self.tet_vertices)
        n_verts = len(tet_verts)

        # Get origin and insertion bone names from attach_skeletons
        skeleton_names = list(skeleton_meshes.keys())
        # Use first stream's attachments (they should all share same bones)
        origin_idx = self.attach_skeletons[0][0]
        insertion_idx = self.attach_skeletons[0][1]

        if origin_idx >= len(skeleton_names) or insertion_idx >= len(skeleton_names):
            print("  Skeleton bindings: invalid attach_skeletons indices")
            return

        origin_mesh_name = skeleton_names[origin_idx]
        insertion_mesh_name = skeleton_names[insertion_idx]

        # Find DART body nodes using robust matching
        def find_body_node(mesh_name):
            """Find DART body node using multiple matching strategies."""
            # Strategy 1: Direct lookup
            body_node = skeleton.getBodyNode(mesh_name)
            if body_node is not None:
                return body_node

            # Normalize for fuzzy matching
            normalized = mesh_name.lower().replace('_', '').replace('-', '')

            for i in range(skeleton.getNumBodyNodes()):
                bn = skeleton.getBodyNode(i)
                bn_name = bn.getName()
                bn_normalized = bn_name.lower().replace('_', '').replace('-', '')

                # Strategy 2: Exact normalized match
                if normalized == bn_normalized:
                    return bn

                # Strategy 3: Substring match
                if normalized in bn_normalized or bn_normalized in normalized:
                    return bn

                # Strategy 4: Core name match (ignore L/R prefix/suffix)
                mesh_core = normalized.replace('l', '').replace('r', '')
                bn_core = bn_normalized.replace('l', '').replace('r', '')
                if len(mesh_core) > 2 and len(bn_core) > 2 and mesh_core == bn_core:
                    return bn

            return None

        origin_body = find_body_node(origin_mesh_name)
        insertion_body = find_body_node(insertion_mesh_name)

        if origin_body is None or insertion_body is None:
            print(f"  Skeleton bindings: could not find body nodes for {origin_mesh_name} or {insertion_mesh_name}")
            return

        origin_body_name = origin_body.getName()
        insertion_body_name = insertion_body.getName()

        print(f"  Computing skeleton bindings: {origin_body_name} -> {insertion_body_name}")

        # Store initial bone transforms
        self.tet_initial_bone_transforms = {}
        for body_node in [origin_body, insertion_body]:
            wt = body_node.getWorldTransform()
            self.tet_initial_bone_transforms[body_node.getName()] = (
                wt.rotation().copy(),
                wt.translation().copy()
            )

        # Compute muscle axis from fixed vertices (caps)
        if hasattr(self, 'soft_body_fixed_vertices') and len(self.soft_body_fixed_vertices) > 0:
            fixed_verts = tet_verts[list(self.soft_body_fixed_vertices)]
            # Cluster into origin and insertion groups by distance to bone centers
            origin_center = origin_body.getWorldTransform().translation()
            insertion_center = insertion_body.getWorldTransform().translation()

            origin_pts = []
            insertion_pts = []
            for fv in fixed_verts:
                if np.linalg.norm(fv - origin_center) < np.linalg.norm(fv - insertion_center):
                    origin_pts.append(fv)
                else:
                    insertion_pts.append(fv)

            if len(origin_pts) > 0 and len(insertion_pts) > 0:
                origin_mean = np.mean(origin_pts, axis=0)
                insertion_mean = np.mean(insertion_pts, axis=0)
            else:
                origin_mean = origin_center
                insertion_mean = insertion_center
        else:
            origin_mean = origin_body.getWorldTransform().translation()
            insertion_mean = insertion_body.getWorldTransform().translation()

        muscle_axis = insertion_mean - origin_mean
        muscle_length = np.linalg.norm(muscle_axis)
        if muscle_length < 1e-6:
            print("  Skeleton bindings: muscle axis too short")
            return
        muscle_axis_normalized = muscle_axis / muscle_length

        # For each vertex, compute weight based on position along muscle axis
        self.tet_skeleton_bindings = [None] * n_verts
        self.tet_original_positions = tet_verts.copy()

        for v_idx in range(n_verts):
            point = tet_verts[v_idx]

            # Project onto muscle axis to get interpolation weight
            t = np.dot(point - origin_mean, muscle_axis_normalized) / muscle_length
            t = np.clip(t, 0.0, 1.0)  # weight: 0 = origin, 1 = insertion

            self.tet_skeleton_bindings[v_idx] = (
                origin_body_name,
                insertion_body_name,
                t,  # blend weight
                point.copy()
            )

        print(f"  Skeleton bindings: {n_verts} vertices with blended weights")

    def _update_tet_positions_from_skeleton(self, skeleton):
        """
        Update tet vertex positions by blending transforms from origin and insertion bones.
        Uses linear blend skinning with weights based on position along muscle axis.
        """
        if not hasattr(self, 'tet_skeleton_bindings') or self.tet_skeleton_bindings is None:
            return 0
        if not hasattr(self, 'tet_initial_bone_transforms'):
            return 0
        if not hasattr(self, 'soft_body') or self.soft_body is None:
            return 0

        # Cache current bone transforms and check if any bone actually moved
        current_transforms = {}
        any_bone_moved = False
        for body_name in self.tet_initial_bone_transforms:
            body_node = skeleton.getBodyNode(body_name)
            if body_node is not None:
                wt = body_node.getWorldTransform()
                R1 = wt.rotation()
                t1 = wt.translation()
                current_transforms[body_name] = (R1, t1)

                # Check if this bone moved from initial
                R0, t0 = self.tet_initial_bone_transforms[body_name]
                rot_diff = np.linalg.norm(R1 - R0)
                trans_diff = np.linalg.norm(t1 - t0)
                if rot_diff > 1e-6 or trans_diff > 1e-6:
                    any_bone_moved = True

        # If no bones moved, skip transform entirely (avoid numerical drift)
        if not any_bone_moved:
            return 0

        positions = self.soft_body.get_positions().copy()
        updated = 0

        # Safety check: ensure bindings cover all vertices
        n_verts = len(positions)
        n_bindings = len(self.tet_skeleton_bindings)
        if n_bindings < n_verts:
            # Extend bindings for new vertices by interpolating based on position
            print(f"  WARNING: binding count ({n_bindings}) < vertex count ({n_verts}), extending bindings")
            # Get origin/insertion info from existing bindings
            origin_body = insertion_body = None
            origin_mean = insertion_mean = None
            for b in self.tet_skeleton_bindings:
                if b is not None:
                    origin_body, insertion_body, _, _ = b
                    break
            if origin_body is not None and origin_body in self.tet_initial_bone_transforms:
                _, origin_mean = self.tet_initial_bone_transforms[origin_body]
            if insertion_body is not None and insertion_body in self.tet_initial_bone_transforms:
                _, insertion_mean = self.tet_initial_bone_transforms[insertion_body]

            if origin_mean is not None and insertion_mean is not None:
                muscle_axis = insertion_mean - origin_mean
                muscle_length = np.linalg.norm(muscle_axis)
                if muscle_length > 1e-6:
                    muscle_axis_normalized = muscle_axis / muscle_length
                    # Extend bindings list
                    for v_idx in range(n_bindings, n_verts):
                        point = positions[v_idx]
                        t = np.dot(point - origin_mean, muscle_axis_normalized) / muscle_length
                        t = np.clip(t, 0.0, 1.0)
                        self.tet_skeleton_bindings.append((origin_body, insertion_body, t, point.copy()))
                    print(f"    Extended bindings to {len(self.tet_skeleton_bindings)} vertices")

        for v_idx, binding in enumerate(self.tet_skeleton_bindings):
            if binding is None:
                continue

            origin_body, insertion_body, weight, orig_pos = binding

            if origin_body not in self.tet_initial_bone_transforms:
                continue
            if insertion_body not in self.tet_initial_bone_transforms:
                continue
            if origin_body not in current_transforms:
                continue
            if insertion_body not in current_transforms:
                continue

            # Get initial and current transforms for both bones
            R0_o, t0_o = self.tet_initial_bone_transforms[origin_body]
            R1_o, t1_o = current_transforms[origin_body]

            R0_i, t0_i = self.tet_initial_bone_transforms[insertion_body]
            R1_i, t1_i = current_transforms[insertion_body]

            # Compute position from origin bone transform
            rel_o = orig_pos - t0_o
            pos_from_origin = R1_o @ (R0_o.T @ rel_o) + t1_o

            # Compute position from insertion bone transform
            rel_i = orig_pos - t0_i
            pos_from_insertion = R1_i @ (R0_i.T @ rel_i) + t1_i

            # Blend based on weight (0 = origin, 1 = insertion)
            new_pos = (1.0 - weight) * pos_from_origin + weight * pos_from_insertion

            positions[v_idx] = new_pos
            updated += 1

        # Update soft body positions
        self.soft_body.set_positions(positions)
        return updated

    def auto_detect_attachments(self, skeleton_meshes):
        """
        Automatically detect which skeleton each stream endpoint should attach to.
        Uses proximity-based matching with multiple sample points for accuracy.

        Args:
            skeleton_meshes: Dict of skeleton name -> MeshLoader

        Returns:
            True if successful, False otherwise
        """
        if not skeleton_meshes or len(skeleton_meshes) == 0:
            print("No skeleton meshes available for auto-detection")
            return False

        skeleton_names = list(skeleton_meshes.keys())

        # Debug: check how many skeletons have vertices
        valid_count = 0
        for name in skeleton_names:
            mesh = skeleton_meshes[name]
            if hasattr(mesh, 'vertices') and mesh.vertices is not None and len(mesh.vertices) > 0:
                valid_count += 1
        print(f"  {valid_count}/{len(skeleton_names)} skeletons have valid vertices")

        # Try to use contours first (most accurate), then waypoints, then edge_groups
        if hasattr(self, 'contours') and self.contours is not None and len(self.contours) > 0:
            num_streams = len(self.contours)
            use_contours = True
            print(f"  Using contours: {num_streams} streams, first stream has {len(self.contours[0])} levels")
        elif hasattr(self, 'waypoints') and self.waypoints is not None and len(self.waypoints) > 0:
            num_streams = len(self.waypoints)
            use_contours = False
            print(f"  Using waypoints: {num_streams} streams")
        else:
            print("No contours or waypoints available. Run contour extraction first.")
            print(f"  hasattr contours: {hasattr(self, 'contours')}, contours is None: {getattr(self, 'contours', None) is None}")
            print(f"  hasattr waypoints: {hasattr(self, 'waypoints')}, waypoints is None: {getattr(self, 'waypoints', None) is None}")
            return False

        # Ensure attach_skeletons arrays exist
        while len(self.attach_skeletons) < num_streams:
            self.attach_skeletons.append([0, 0])
        while len(self.attach_skeletons_sub) < num_streams:
            self.attach_skeletons_sub.append([0, 0])
        while len(self.attach_skeleton_names) < num_streams:
            self.attach_skeleton_names.append(['', ''])

        print(f"Auto-detecting attachments for {num_streams} streams...")

        for stream_idx in range(num_streams):
            if use_contours:
                contour_list = self.contours[stream_idx]
                if len(contour_list) < 2:
                    print(f"  Stream {stream_idx}: skipping - only {len(contour_list)} levels")
                    continue
                # Get origin (first contour) and insertion (last contour) points
                origin_points = np.array(contour_list[0])
                insertion_points = np.array(contour_list[-1])
                print(f"  Stream {stream_idx}: origin shape={origin_points.shape}, insertion shape={insertion_points.shape}")
            else:
                waypoint_group = self.waypoints[stream_idx]
                if len(waypoint_group) < 2:
                    continue
                origin_points = np.array(waypoint_group[0])
                insertion_points = np.array(waypoint_group[-1])

            # Find closest skeleton using ALL points (not just mean)
            origin_skel_idx = self._find_closest_skeleton_multi(origin_points, skeleton_meshes, skeleton_names)
            insertion_skel_idx = self._find_closest_skeleton_multi(insertion_points, skeleton_meshes, skeleton_names)

            self.attach_skeletons[stream_idx][0] = origin_skel_idx
            self.attach_skeletons[stream_idx][1] = insertion_skel_idx
            # Also store skeleton names for stable save/load
            self.attach_skeleton_names[stream_idx][0] = skeleton_names[origin_skel_idx]
            self.attach_skeleton_names[stream_idx][1] = skeleton_names[insertion_skel_idx]

            print(f"  Stream {stream_idx}: origin -> {skeleton_names[origin_skel_idx]}, "
                  f"insertion -> {skeleton_names[insertion_skel_idx]}")

        return True

    def _find_closest_skeleton_multi(self, points, skeleton_meshes, skeleton_names, debug=True):
        """
        Find the index of the closest skeleton mesh to a set of points.
        Uses mesh vertices for accurate distance calculation.
        """
        if len(points) == 0:
            return 0

        points = np.atleast_2d(points)
        query_center = np.mean(points, axis=0)

        best_idx = 0
        best_score = float('inf')
        all_scores = []

        for idx, name in enumerate(skeleton_names):
            mesh = skeleton_meshes[name]
            score = float('inf')

            try:
                # Primary method: Use mesh vertices directly (most accurate)
                if hasattr(mesh, 'vertices') and mesh.vertices is not None and len(mesh.vertices) > 0:
                    verts = np.array(mesh.vertices)

                    # Distance from query center to closest vertex
                    dists = np.linalg.norm(verts - query_center, axis=1)
                    min_dist = np.min(dists)

                    # Also check distance from each query point to mesh
                    for p in points:
                        d = np.min(np.linalg.norm(verts - p, axis=1))
                        min_dist = min(min_dist, d)

                    score = min_dist

                # Fallback: Use trimesh for surface distance
                elif hasattr(mesh, 'trimesh') and mesh.trimesh is not None:
                    closest, distances, _ = mesh.trimesh.nearest.on_surface(points)
                    score = np.min(distances)

                all_scores.append((name, score))

                if score < best_score:
                    best_score = score
                    best_idx = idx

            except Exception as e:
                all_scores.append((name, f"error: {e}"))
                continue

        # Debug: print top 5 closest bones
        if debug:
            valid_scores = [(n, s) for n, s in all_scores if isinstance(s, (int, float))]
            valid_scores.sort(key=lambda x: x[1])
            print(f"    Query center: {query_center}")
            print(f"    Top 5 closest: {valid_scores[:5]}")

        return best_idx

    def resolve_skeleton_attachments(self, skeleton_names):
        """
        Resolve attach_skeleton_names to attach_skeletons indices based on current skeleton order.
        Call this after loading when skeleton data is available.
        """
        if not hasattr(self, 'attach_skeleton_names') or not self.attach_skeleton_names:
            return False

        # Build name-to-index mapping
        name_to_idx = {name: idx for idx, name in enumerate(skeleton_names)}

        resolved_count = 0
        for stream_idx, names in enumerate(self.attach_skeleton_names):
            if stream_idx >= len(self.attach_skeletons):
                continue
            if len(names) >= 2:
                origin_name = names[0]
                insertion_name = names[1]

                if origin_name in name_to_idx:
                    old_idx = self.attach_skeletons[stream_idx][0]
                    new_idx = name_to_idx[origin_name]
                    if old_idx != new_idx:
                        print(f"  Stream {stream_idx} origin: idx {old_idx} -> {new_idx} ({origin_name})")
                    self.attach_skeletons[stream_idx][0] = new_idx
                    resolved_count += 1

                if insertion_name in name_to_idx:
                    old_idx = self.attach_skeletons[stream_idx][1]
                    new_idx = name_to_idx[insertion_name]
                    if old_idx != new_idx:
                        print(f"  Stream {stream_idx} insertion: idx {old_idx} -> {new_idx} ({insertion_name})")
                    self.attach_skeletons[stream_idx][1] = new_idx
                    resolved_count += 1

        if resolved_count > 0:
            print(f"Resolved {resolved_count} skeleton attachments from names")
        return True
