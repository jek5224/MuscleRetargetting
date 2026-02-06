# Tetrahedron Mesh operations for muscle mesh processing
# Extracted from mesh_loader.py for better organization

import numpy as np
from OpenGL.GL import *
import os
import pickle
import trimesh
from scipy.spatial import Delaunay
from collections import defaultdict


def laplacian_smooth_mesh(vertices, faces, iterations=3, lambda_factor=0.5, preserve_boundary=True):
    """
    Apply Taubin smoothing to a mesh to reduce surface dents without shrinking.

    Taubin smoothing alternates between shrinking (lambda) and inflating (mu) steps
    to smooth the mesh while preserving volume.

    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices
        iterations: Number of smoothing iterations (each iteration = shrink + inflate)
        lambda_factor: Smoothing strength (0-1, higher = more smoothing)
        preserve_boundary: If True, don't move boundary vertices

    Returns:
        Smoothed vertices array
    """
    vertices = np.array(vertices, dtype=np.float64)
    n_verts = len(vertices)

    # Taubin parameters: mu should be slightly larger than -lambda to prevent shrinkage
    lambda_val = lambda_factor
    mu_val = -lambda_factor * 1.02  # Slightly stronger expansion to counteract shrinkage

    # Build vertex adjacency from faces
    adjacency = defaultdict(set)
    for face in faces:
        for i in range(3):
            v0, v1 = face[i], face[(i + 1) % 3]
            adjacency[v0].add(v1)
            adjacency[v1].add(v0)

    # Find boundary vertices (vertices on edges with only one adjacent face)
    boundary_verts = set()
    if preserve_boundary:
        edge_count = defaultdict(int)
        for face in faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edge_count[edge] += 1
        for edge, count in edge_count.items():
            if count == 1:  # Boundary edge
                boundary_verts.add(edge[0])
                boundary_verts.add(edge[1])

    def apply_laplacian_step(verts, factor):
        """Apply one Laplacian smoothing step with given factor."""
        new_verts = verts.copy()
        for vi in range(n_verts):
            if vi in boundary_verts:
                continue
            neighbors = list(adjacency[vi])
            if len(neighbors) == 0:
                continue
            centroid = np.mean(verts[neighbors], axis=0)
            new_verts[vi] = verts[vi] + factor * (centroid - verts[vi])
        return new_verts

    # Taubin smoothing: alternate shrink and inflate
    for _ in range(iterations):
        # Shrink step (positive lambda)
        vertices = apply_laplacian_step(vertices, lambda_val)
        # Inflate step (negative mu)
        vertices = apply_laplacian_step(vertices, mu_val)

    return vertices.astype(np.float32)


class TetrahedronMeshMixin:
    """
    Mixin class providing tetrahedron-related methods for MeshLoader.
    Handles tetrahedralization, tet mesh I/O, and rendering.
    """

    def _init_tetrahedron_properties(self):
        """Initialize tetrahedron-related properties. Call from MeshLoader.__init__."""
        # Tetrahedron mesh data
        self.tet_vertices = None  # (N, 3) array of tet vertex positions
        self.tet_tetrahedra = None  # (M, 4) array of tet indices
        self.tet_faces = None  # (F, 3) array of surface face indices (for backwards compat)
        self.tet_render_faces = None  # Original contour faces for rendering/collision
        self.tet_sim_faces = None  # Tet boundary faces for simulation
        self.tet_face_normals = None  # (F, 3) array of face normals
        self.tet_cap_face_indices = []  # Indices of cap faces (for skeleton attachment)
        self.tet_anchor_vertices = []  # Vertices at origin/insertion caps
        self.tet_surface_face_count = 0  # Number of original surface faces (before caps)

        # Tet drawing settings
        self.is_draw_tet_mesh = False
        self.is_draw_tet_edges = False

        # Pre-computed draw arrays for efficient rendering
        self._tet_surface_verts = None
        self._tet_surface_normals = None
        self._tet_cap_verts = None
        self._tet_cap_normals = None
        self._tet_edge_verts = None

        # Cap attachment info
        self.tet_cap_attachments = []  # List of (anchor_idx, stream_idx, end_type, skeleton_idx, subpart_idx)

        # Contour-to-tet mapping
        self.contour_to_tet_vertex_map = None  # Maps contour vertex indices to tet vertex indices
        self.cross_contour_edge_mask = None  # Boolean mask for cross-contour edges
        self.intra_contour_edge_mask = None  # Boolean mask for intra-contour edges

    def _extract_tet_boundary_faces(self, tetrahedra):
        """
        Extract boundary faces from tetrahedra (faces shared by exactly 1 tet).
        Used for simulation boundary conditions.
        """
        from collections import Counter
        face_counts = Counter()
        face_to_original = {}

        for tet_idx, tet in enumerate(tetrahedra):
            # 4 faces per tet with outward-facing winding
            tet_faces_local = [
                (tet[1], tet[2], tet[3]),
                (tet[0], tet[3], tet[2]),
                (tet[0], tet[1], tet[3]),
                (tet[0], tet[2], tet[1]),
            ]
            for face in tet_faces_local:
                sorted_face = tuple(sorted(face))
                face_counts[sorted_face] += 1
                face_to_original[sorted_face] = face

        boundary_faces = []
        for sorted_face, count in face_counts.items():
            if count == 1:
                boundary_faces.append(face_to_original[sorted_face])

        return np.array(boundary_faces, dtype=np.int32) if boundary_faces else np.array([], dtype=np.int32).reshape(0, 3)

    def tetrahedralize_contour_mesh(self):
        """
        Tetrahedralize the contour mesh for soft body simulation.
        - Caps open edges at origins/insertions with triangular faces using mean point as anchor
        - Tetrahedralizes the closed volume
        - Marks cap faces as fixed (for attachment to skeleton)

        Shared boundary handling (cut contours):
        - The contour mesh may have duplicate vertices at shared cut boundaries
        - Step 0.5 merges vertices within merge_epsilon=1e-6
        - This ensures tetrahedra on either side of a cut boundary share vertices
        - Result: one connected tet mesh even for cut/split contour streams
        """
        if self.contour_mesh_vertices is None or self.contour_mesh_faces is None:
            print("No contour mesh to tetrahedralize. Build contour mesh first.")
            return False

        vertices_original = self.contour_mesh_vertices.copy()
        faces = self.contour_mesh_faces.copy()

        # Step 0: Smooth the mesh to reduce dents while preserving boundary (optional)
        if getattr(self, 'smooth_contour_mesh', False):
            print("Applying light Laplacian smoothing to contour mesh...")
            vertices = laplacian_smooth_mesh(
                vertices_original,
                faces,
                iterations=3,       # Reduced from 10
                lambda_factor=0.2,  # Reduced from 0.5
                preserve_boundary=True
            )
            print(f"  Smoothed {len(vertices)} vertices")
        else:
            print("Laplacian smoothing disabled, using original contour mesh vertices")
            vertices = vertices_original.copy()

        # Step 0.5: Merge duplicate vertices (shared cut edge vertices)
        # After contour cutting, adjacent pieces have vertices at identical positions
        # on shared boundaries. build_contour_mesh deduplicates per-level, but this
        # step catches any remaining duplicates to ensure the tet mesh forms one
        # connected body with shared vertices at cut boundaries.
        merge_epsilon = 1e-6
        n_verts_before = len(vertices)

        # Build spatial hash for finding duplicates
        vertex_map = {}  # Maps old index to new index
        unique_vertices = []
        merged_count = 0

        for old_idx, v in enumerate(vertices):
            # Check if this vertex is close to any existing unique vertex
            found_match = False
            for new_idx, uv in enumerate(unique_vertices):
                if np.linalg.norm(v - uv) < merge_epsilon:
                    vertex_map[old_idx] = new_idx
                    found_match = True
                    merged_count += 1
                    break

            if not found_match:
                vertex_map[old_idx] = len(unique_vertices)
                unique_vertices.append(v)

        if merged_count > 0:
            # Update faces to use new vertex indices
            new_faces = []
            for face in faces:
                new_face = [vertex_map[v] for v in face]
                # Skip degenerate faces (where vertices merged to same point)
                if len(set(new_face)) == 3:
                    new_faces.append(new_face)

            vertices = np.array(unique_vertices)
            faces = np.array(new_faces)
            print(f"Merged {merged_count} duplicate vertices ({n_verts_before} -> {len(vertices)})")

        # Step 1: Find open boundary edges (edges that belong to only one face)
        edge_count = defaultdict(list)
        for face_idx, face in enumerate(faces):
            for i in range(3):
                v0, v1 = face[i], face[(i + 1) % 3]
                edge = tuple(sorted([v0, v1]))
                edge_count[edge].append(face_idx)

        # Open edges are those with only one adjacent face
        open_edges = [edge for edge, face_list in edge_count.items() if len(face_list) == 1]
        print(f"Found {len(open_edges)} open edges")

        if len(open_edges) == 0:
            print("Mesh is already closed.")
            closed_vertices = vertices
            closed_faces = faces
            cap_face_indices = []
        else:
            # Step 2: Group open edges into boundary loops
            # Build adjacency for open edges
            edge_adjacency = defaultdict(list)
            for edge in open_edges:
                edge_adjacency[edge[0]].append(edge[1])
                edge_adjacency[edge[1]].append(edge[0])

            # Find connected boundary loops
            visited_vertices = set()
            boundary_loops = []

            for start_vertex in edge_adjacency:
                if start_vertex in visited_vertices:
                    continue

                loop = []
                current = start_vertex
                prev = None

                while True:
                    loop.append(current)
                    visited_vertices.add(current)

                    neighbors = edge_adjacency[current]
                    next_vertex = None
                    for n in neighbors:
                        if n != prev and n not in visited_vertices:
                            next_vertex = n
                            break

                    if next_vertex is None:
                        # Check if we can close the loop
                        if start_vertex in neighbors and len(loop) > 2:
                            break  # Loop closed
                        else:
                            break  # Dead end

                    prev = current
                    current = next_vertex

                if len(loop) >= 3:
                    boundary_loops.append(loop)

            print(f"Found {len(boundary_loops)} boundary loops")

            # Step 3: Create cap faces for each boundary loop
            closed_vertices = list(vertices)
            closed_faces = list(faces)
            cap_face_indices = []
            self.tet_anchor_vertices = []  # Store anchor vertex indices for each cap

            for loop_idx, loop in enumerate(boundary_loops):
                # Calculate mean point of the boundary loop
                loop_vertices = np.array([vertices[vi] for vi in loop])
                mean_point = np.mean(loop_vertices, axis=0)

                # Add mean point as new vertex
                anchor_idx = len(closed_vertices)
                closed_vertices.append(mean_point)
                self.tet_anchor_vertices.append(anchor_idx)

                # Create triangular faces connecting mean point to boundary edges
                # Determine winding order by checking normal direction
                loop_center = mean_point
                v0 = vertices[loop[0]]
                v1 = vertices[loop[1]]
                edge_vec = v1 - v0
                to_center = loop_center - v0

                # Use cross product to determine orientation
                # Check against first face's normal for consistency
                first_edge = [loop[0], loop[1]]
                first_edge_sorted = tuple(sorted(first_edge))
                if first_edge_sorted in edge_count:
                    adj_face_idx = edge_count[first_edge_sorted][0]
                    adj_face = faces[adj_face_idx]
                    fv0, fv1, fv2 = vertices[adj_face[0]], vertices[adj_face[1]], vertices[adj_face[2]]
                    face_normal = np.cross(fv1 - fv0, fv2 - fv0)
                    face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-10)

                    # Cap normal should point outward (opposite to face normal at boundary)
                    cap_normal = np.cross(v1 - v0, mean_point - v0)
                    if np.dot(cap_normal, face_normal) > 0:
                        # Reverse winding
                        loop = loop[::-1]

                for i in range(len(loop)):
                    vi = loop[i]
                    vj = loop[(i + 1) % len(loop)]
                    cap_face_idx = len(closed_faces)
                    closed_faces.append([vi, vj, anchor_idx])
                    cap_face_indices.append(cap_face_idx)

                print(f"  Loop {loop_idx}: {len(loop)} vertices, anchor at index {anchor_idx}")

            closed_vertices = np.array(closed_vertices, dtype=np.float32)
            closed_faces = np.array(closed_faces, dtype=np.int32)

            # Step 3.5: Map each boundary loop to stream origin/insertion
            # Match anchors to stream endpoints based on distance
            self.tet_cap_attachments = []
            if hasattr(self, 'contours') and self.contours is not None and len(self.contours) > 0:
                for anchor_idx in self.tet_anchor_vertices:
                    anchor_pos = closed_vertices[anchor_idx]
                    best_match = None
                    best_dist = float('inf')

                    for stream_idx, stream_contours in enumerate(self.contours):
                        if len(stream_contours) == 0:
                            continue

                        # Origin: first contour of stream
                        origin_contour = stream_contours[0]
                        if len(origin_contour) > 0:
                            origin_mean = np.mean(origin_contour, axis=0)
                            dist = np.linalg.norm(anchor_pos - origin_mean)
                            if dist < best_dist:
                                best_dist = dist
                                # end_type: 0=origin, 1=insertion
                                best_match = (anchor_idx, stream_idx, 0)

                        # Insertion: last contour of stream
                        insertion_contour = stream_contours[-1]
                        if len(insertion_contour) > 0:
                            insertion_mean = np.mean(insertion_contour, axis=0)
                            dist = np.linalg.norm(anchor_pos - insertion_mean)
                            if dist < best_dist:
                                best_dist = dist
                                best_match = (anchor_idx, stream_idx, 1)

                    if best_match is not None:
                        anchor_idx, stream_idx, end_type = best_match
                        # Get skeleton attachment from attach_skeletons
                        if stream_idx < len(self.attach_skeletons):
                            skel_idx = self.attach_skeletons[stream_idx][end_type]
                            subpart_idx = self.attach_skeletons_sub[stream_idx][end_type] if stream_idx < len(self.attach_skeletons_sub) else 0
                        else:
                            skel_idx = 0
                            subpart_idx = 0
                        self.tet_cap_attachments.append((anchor_idx, stream_idx, end_type, skel_idx, subpart_idx))
                        end_name = "origin" if end_type == 0 else "insertion"
                        print(f"  Anchor {anchor_idx} -> stream {stream_idx} {end_name}, skeleton {skel_idx}")

        # Step 4: Tetrahedralize using Delaunay (robust, preserves all input vertices)
        # Delaunay is more robust than TetGen for open/non-manifold meshes
        # Original surface vertices are preserved; we keep original faces for rendering
        print("Performing Delaunay tetrahedralization...")
        try:
            delaunay = Delaunay(closed_vertices)
            tetrahedra = delaunay.simplices

            # Filter to keep only tetrahedra whose centroids are inside the surface
            tet_centroids = np.mean(closed_vertices[tetrahedra], axis=1)
            mesh = trimesh.Trimesh(vertices=closed_vertices, faces=closed_faces)

            # Fix mesh normals and attempt to make watertight for contains check
            mesh.fix_normals()
            if not mesh.is_watertight:
                print(f"  Warning: mesh is not watertight, contains() may be inaccurate")

            inside_mask = mesh.contains(tet_centroids)
            interior_tetrahedra = tetrahedra[inside_mask]

            if len(interior_tetrahedra) == 0:
                print("  Warning: No interior tetrahedra found, using all tetrahedra")
                interior_tetrahedra = tetrahedra

            print(f"  Delaunay: {len(tetrahedra)} total tets, {len(interior_tetrahedra)} interior")

            # Diagnostic: Check for unused vertices after filtering
            n_verts = len(closed_vertices)
            used_vertices = set(interior_tetrahedra.flatten())
            unused_count = n_verts - len(used_vertices)
            if unused_count > 0:
                unused_indices = [i for i in range(n_verts) if i not in used_vertices]
                print(f"  WARNING: {unused_count} vertices unused after interior filtering")
                for vi in unused_indices[:5]:  # Show first 5
                    pos = closed_vertices[vi]
                    print(f"    Unused vertex {vi}: pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
                # Check if unused vertices are cap anchor vertices
                if hasattr(self, 'tet_anchor_vertices'):
                    unused_anchors = [vi for vi in unused_indices if vi in self.tet_anchor_vertices]
                    if unused_anchors:
                        print(f"    {len(unused_anchors)} of these are cap anchor vertices!")

        except Exception as e:
            print(f"Tetrahedralization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Step 5: Extract tet boundary faces (faces shared by exactly 1 tetrahedron)
        # These are used for simulation boundary conditions
        from collections import Counter
        face_counts = Counter()
        face_to_tet = {}  # Track which tet each face belongs to

        for tet_idx, tet in enumerate(interior_tetrahedra):
            # 4 faces per tet: opposite to each vertex
            tet_faces_local = [
                (tet[1], tet[2], tet[3]),  # opposite to vertex 0
                (tet[0], tet[3], tet[2]),  # opposite to vertex 1 (flipped for outward normal)
                (tet[0], tet[1], tet[3]),  # opposite to vertex 2
                (tet[0], tet[2], tet[1]),  # opposite to vertex 3 (flipped for outward normal)
            ]
            for face in tet_faces_local:
                sorted_face = tuple(sorted(face))
                face_counts[sorted_face] += 1
                face_to_tet[sorted_face] = (tet_idx, face)  # Store original winding

        # Boundary faces appear exactly once
        sim_faces = []
        for sorted_face, count in face_counts.items():
            if count == 1:
                # Use the original winding from the tet
                _, original_face = face_to_tet[sorted_face]
                sim_faces.append(original_face)

        sim_faces = np.array(sim_faces, dtype=np.int32)
        print(f"  Extracted {len(sim_faces)} tet boundary faces for simulation")

        # Step 6: Store results with dual face system
        self.tet_vertices = closed_vertices
        self.tet_tetrahedra = interior_tetrahedra

        # Dual face system:
        # - tet_render_faces: Original contour faces + caps (for rendering/collision)
        # - tet_sim_faces: Tet boundary faces (for simulation boundary conditions)
        self.tet_render_faces = closed_faces  # Original surface faces + cap faces
        self.tet_sim_faces = sim_faces  # Tet boundary faces
        self.tet_faces = closed_faces  # Backwards compatibility (alias to render faces)

        self.tet_cap_face_indices = cap_face_indices  # Indices of cap faces (fixed during simulation)
        self.tet_surface_face_count = len(faces)  # Number of original surface faces

        print(f"Tetrahedralization complete:")
        print(f"  Vertices: {len(self.tet_vertices)}")
        print(f"  Render faces: {len(self.tet_render_faces)} ({self.tet_surface_face_count} original + {len(cap_face_indices)} caps)")
        print(f"  Sim faces: {len(self.tet_sim_faces)} (tet boundary)")
        print(f"  Tetrahedra: {len(self.tet_tetrahedra)}")
        print(f"  Fixed cap faces: {len(self.tet_cap_face_indices)}")

        return True

    def save_tetrahedron_mesh(self, name, filepath=None):
        """
        Save tetrahedron mesh to tet/.tet.npz file.
        """
        # Validate required attributes
        if not hasattr(self, 'tet_vertices') or self.tet_vertices is None:
            print(f"[{name}] No tetrahedron mesh to save. Run tetrahedralization first.")
            return False

        if not hasattr(self, 'tet_faces') or self.tet_faces is None:
            print(f"[{name}] Missing tet_faces. Cannot save.")
            return False

        if not hasattr(self, 'tet_tetrahedra') or self.tet_tetrahedra is None:
            print(f"[{name}] Missing tet_tetrahedra. Cannot save.")
            return False

        # Create tet directory if it doesn't exist
        tet_dir = "tet"
        try:
            if not os.path.exists(tet_dir):
                os.makedirs(tet_dir)
        except OSError as e:
            print(f"[{name}] Failed to create tet directory: {e}")
            return False

        if filepath is None:
            filepath = os.path.join(tet_dir, f"{name}_tet.npz")

        # Prepare waypoint data for saving
        waypoints_data = None
        if hasattr(self, 'waypoints') and len(self.waypoints) > 0:
            # Convert waypoints to saveable format: list of arrays
            waypoints_data = []
            for stream in self.waypoints:
                stream_data = [np.array(wp) for wp in stream]
                waypoints_data.append(stream_data)

        # Prepare barycentric coords for saving
        bary_data = None
        if hasattr(self, 'waypoint_bary_coords') and len(self.waypoint_bary_coords) > 0:
            bary_data = self.waypoint_bary_coords

        # Prepare contours data for saving (needed for soft body fixed vertices)
        contours_data = None
        if hasattr(self, 'contours') and self.contours is not None and len(self.contours) > 0:
            # Convert contours to saveable format: list of list of arrays
            contours_data = []
            for stream_contours in self.contours:
                stream_data = [np.array(c) for c in stream_contours]
                contours_data.append(stream_data)

        # Prepare fiber architecture for saving
        fiber_data = None
        if hasattr(self, 'fiber_architecture') and self.fiber_architecture is not None:
            # Convert to list of numpy arrays for proper serialization
            fiber_data = [np.array(f) for f in self.fiber_architecture]

        # Prepare bounding planes for saving (needed for fiber operations)
        bounding_planes_data = None
        if hasattr(self, 'bounding_planes') and self.bounding_planes is not None and len(self.bounding_planes) > 0:
            bounding_planes_data = self.bounding_planes

        # Prepare contour-to-tet mapping for saving (for deformed contour visualization)
        contour_mapping_data = None
        if hasattr(self, 'contour_to_tet_mapping') and self.contour_to_tet_mapping is not None:
            contour_mapping_data = self.contour_to_tet_mapping

        # Save with pickle for complex nested structures
        try:
            # Get render and sim faces (dual face system)
            render_faces = self.tet_render_faces if hasattr(self, 'tet_render_faces') and self.tet_render_faces is not None else self.tet_faces
            sim_faces = self.tet_sim_faces if hasattr(self, 'tet_sim_faces') and self.tet_sim_faces is not None else None

            save_dict = {
                'vertices': self.tet_vertices,
                'faces': render_faces,  # Backwards compatible: 'faces' = render faces
                'render_faces': render_faces,  # Explicit render faces
                'sim_faces': sim_faces,  # Tet boundary faces for simulation
                'tetrahedra': self.tet_tetrahedra,
                'cap_face_indices': np.array(self.tet_cap_face_indices) if hasattr(self, 'tet_cap_face_indices') else np.array([]),
                'anchor_vertices': np.array(self.tet_anchor_vertices) if hasattr(self, 'tet_anchor_vertices') else np.array([]),
                'surface_face_count': self.tet_surface_face_count if hasattr(self, 'tet_surface_face_count') else 0,
                'cap_attachments': np.array(self.tet_cap_attachments) if hasattr(self, 'tet_cap_attachments') and self.tet_cap_attachments and len(self.tet_cap_attachments) > 0 else np.array([]),
                'waypoints': waypoints_data,
                'waypoint_bary_coords': bary_data,
                'attach_skeletons': self.attach_skeletons if hasattr(self, 'attach_skeletons') else None,
                'attach_skeletons_sub': self.attach_skeletons_sub if hasattr(self, 'attach_skeletons_sub') else None,
                'attach_skeleton_names': self.attach_skeleton_names if hasattr(self, 'attach_skeleton_names') else None,
                'contours': contours_data,
                'fiber_architecture': fiber_data,
                'bounding_planes': bounding_planes_data,
                'draw_contour_stream': self.draw_contour_stream if hasattr(self, 'draw_contour_stream') else None,
                'contour_to_tet_mapping': contour_mapping_data,
                'fiber_sampling_seed': getattr(self, 'fiber_sampling_seed', 42),
            }

            with open(filepath, 'wb') as f:
                pickle.dump(save_dict, f)

            print(f"[{name}] Saved tetrahedron mesh to {filepath}")
            return True
        except Exception as e:
            print(f"[{name}] Failed to save tetrahedron mesh: {e}")
            return False

    def load_tetrahedron_mesh(self, name, filepath=None):
        """
        Load tetrahedron mesh from tet/.tet.npz file (pickle format with waypoints).
        """
        if filepath is None:
            filepath = os.path.join("tet", f"{name}_tet.npz")

        # Check if file exists before trying to load
        if not os.path.exists(filepath):
            print(f"[{name}] Tet file not found: {filepath}")
            return False

        try:
            # Try loading as pickle first (new format)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.tet_vertices = data['vertices']
            self.tet_tetrahedra = data['tetrahedra']
            self.tet_cap_face_indices = list(data['cap_face_indices'])
            self.tet_anchor_vertices = list(data['anchor_vertices']) if 'anchor_vertices' in data and len(data['anchor_vertices']) > 0 else []
            self.tet_surface_face_count = int(data['surface_face_count'])

            # Load dual face system (with backwards compatibility)
            if 'render_faces' in data and data['render_faces'] is not None:
                self.tet_render_faces = data['render_faces']
            else:
                self.tet_render_faces = data['faces']  # Old format: faces = render faces

            if 'sim_faces' in data and data['sim_faces'] is not None:
                self.tet_sim_faces = data['sim_faces']
            else:
                # Old format: regenerate sim faces from tetrahedra
                self.tet_sim_faces = self._extract_tet_boundary_faces(self.tet_tetrahedra)

            self.tet_faces = self.tet_render_faces  # Backwards compatibility alias

            if 'cap_attachments' in data and data['cap_attachments'] is not None and len(data['cap_attachments']) > 0:
                self.tet_cap_attachments = [tuple(row) for row in data['cap_attachments']]
            else:
                self.tet_cap_attachments = []

            # Load waypoints
            if 'waypoints' in data and data['waypoints'] is not None:
                self.waypoints = data['waypoints']
                self.waypoints_from_tet_sim = True
                print(f"  Loaded {len(self.waypoints)} waypoint streams")

            # Load barycentric coords
            if 'waypoint_bary_coords' in data and data['waypoint_bary_coords'] is not None:
                self.waypoint_bary_coords = data['waypoint_bary_coords']
                print(f"  Loaded waypoint barycentric coordinates")

            # Load attach_skeletons
            if 'attach_skeletons' in data and data['attach_skeletons'] is not None:
                self.attach_skeletons = data['attach_skeletons']

            # Load attach_skeletons_sub
            if 'attach_skeletons_sub' in data and data['attach_skeletons_sub'] is not None:
                self.attach_skeletons_sub = data['attach_skeletons_sub']

            # Load attach_skeleton_names (for stable name-based resolution)
            if 'attach_skeleton_names' in data and data['attach_skeleton_names'] is not None:
                self.attach_skeleton_names = data['attach_skeleton_names']
                print(f"  Loaded attach_skeleton_names: {self.attach_skeleton_names}")

            # Load contours (needed for soft body fixed vertices)
            if 'contours' in data and data['contours'] is not None:
                self.contours = data['contours']
                print(f"  Loaded {len(self.contours)} contour streams")

            # Load fiber architecture
            if 'fiber_architecture' in data and data['fiber_architecture'] is not None:
                self.fiber_architecture = data['fiber_architecture']
                print(f"  Loaded fiber architecture with {len(self.fiber_architecture)} fiber sets")

            # Load bounding planes
            if 'bounding_planes' in data and data['bounding_planes'] is not None:
                self.bounding_planes = data['bounding_planes']
                print(f"  Loaded {len(self.bounding_planes)} bounding plane levels")

            # Load draw_contour_stream
            if 'draw_contour_stream' in data and data['draw_contour_stream'] is not None:
                self.draw_contour_stream = data['draw_contour_stream']

            # Load contour-to-tet mapping (for deformed contour visualization)
            if 'contour_to_tet_mapping' in data and data['contour_to_tet_mapping'] is not None:
                self.contour_to_tet_mapping = data['contour_to_tet_mapping']
                print(f"  Loaded contour-to-tet mapping")

            # Load fiber sampling seed
            if 'fiber_sampling_seed' in data:
                self.fiber_sampling_seed = data['fiber_sampling_seed']

            print(f"[{name}] Loaded tetrahedron mesh from {filepath}")
            print(f"  Vertices: {len(self.tet_vertices)}, Render faces: {len(self.tet_render_faces)}, Sim faces: {len(self.tet_sim_faces)}, Tets: {len(self.tet_tetrahedra)}")
            return True

        except Exception as e:
            # Try old npz format as fallback
            try:
                data = np.load(filepath)
                self.tet_vertices = data['vertices']
                self.tet_tetrahedra = data['tetrahedra']
                self.tet_cap_face_indices = list(data['cap_face_indices'])
                self.tet_anchor_vertices = list(data['anchor_vertices']) if 'anchor_vertices' in data else []
                self.tet_surface_face_count = int(data['surface_face_count'])
                if 'cap_attachments' in data and len(data['cap_attachments']) > 0:
                    self.tet_cap_attachments = [tuple(row) for row in data['cap_attachments']]
                else:
                    self.tet_cap_attachments = []

                # Old format: faces = render faces, regenerate sim faces
                self.tet_render_faces = data['faces']
                self.tet_sim_faces = self._extract_tet_boundary_faces(self.tet_tetrahedra)
                self.tet_faces = self.tet_render_faces  # Backwards compatibility alias

                print(f"[{name}] Loaded tetrahedron mesh (old format) from {filepath}")
                return True
            except Exception as e2:
                print(f"[{name}] Failed to load tetrahedron mesh: {e}")
                return False

    def _prepare_tet_draw_arrays(self):
        """Prepare vertex arrays for efficient tetrahedron mesh drawing."""
        if not hasattr(self, 'tet_vertices') or self.tet_vertices is None:
            return

        # Use render faces for drawing (original contour mesh surface)
        render_faces = self.tet_render_faces if hasattr(self, 'tet_render_faces') and self.tet_render_faces is not None else self.tet_faces

        # Prepare surface face arrays
        cap_set = set(self.tet_cap_face_indices)

        surface_verts = []
        surface_normals = []
        surface_colors = []
        cap_verts = []
        cap_normals = []

        color = self.contour_mesh_color
        cap_color = np.array([0.2, 0.6, 0.2])

        # Build index arrays for fast position updates
        surface_face_indices = []
        cap_face_indices = []

        for face_idx, face in enumerate(render_faces):
            v0, v1, v2 = self.tet_vertices[face]
            normal = np.cross(v1 - v0, v2 - v0)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-10:
                normal = normal / norm_len

            if face_idx in cap_set:
                for vi in face:
                    cap_verts.append(self.tet_vertices[vi])
                    cap_normals.append(normal)
                cap_face_indices.append(face)
            else:
                for vi in face:
                    surface_verts.append(self.tet_vertices[vi])
                    surface_normals.append(normal)
                surface_face_indices.append(face)

        self._tet_surface_verts = np.array(surface_verts, dtype=np.float32) if surface_verts else None
        self._tet_surface_normals = np.array(surface_normals, dtype=np.float32) if surface_normals else None
        self._tet_cap_verts = np.array(cap_verts, dtype=np.float32) if cap_verts else None
        self._tet_cap_normals = np.array(cap_normals, dtype=np.float32) if cap_normals else None

        # Store flattened vertex index arrays for fast update path
        self._tet_surface_vidx = np.array(surface_face_indices, dtype=np.int32).reshape(-1) if surface_face_indices else None
        self._tet_cap_vidx = np.array(cap_face_indices, dtype=np.int32).reshape(-1) if cap_face_indices else None

        # Prepare surface edge arrays (from render faces for display)
        edge_set = set()
        for face in render_faces:
            for i in range(3):
                v0, v1 = face[i], face[(i + 1) % 3]
                edge = (min(v0, v1), max(v0, v1))
                edge_set.add(edge)

        edge_verts = []
        edge_vidx = []
        for v0, v1 in edge_set:
            edge_verts.append(self.tet_vertices[v0])
            edge_verts.append(self.tet_vertices[v1])
            edge_vidx.extend([v0, v1])
        self._tet_edge_verts = np.array(edge_verts, dtype=np.float32) if edge_verts else None
        self._tet_edge_vidx = np.array(edge_vidx, dtype=np.int32) if edge_vidx else None

    def _update_tet_draw_positions(self):
        """Fast path: update draw arrays from tet_vertices using precomputed index arrays.
        Call this instead of _prepare_tet_draw_arrays when only positions changed (not topology)."""
        if not hasattr(self, '_tet_surface_vidx') or self._tet_surface_vidx is None:
            self._prepare_tet_draw_arrays()
            return
        verts = self.tet_vertices
        # Update surface verts + normals
        if self._tet_surface_vidx is not None and self._tet_surface_verts is not None:
            self._tet_surface_verts[:] = verts[self._tet_surface_vidx]
            # Recompute normals vectorized: every 3 verts is a triangle
            v = self._tet_surface_verts.reshape(-1, 3, 3)
            normals = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            normals /= norms
            # Broadcast per-face normal to 3 vertices
            self._tet_surface_normals[:] = np.repeat(normals, 3, axis=0)
        # Update cap verts + normals
        if self._tet_cap_vidx is not None and self._tet_cap_verts is not None:
            self._tet_cap_verts[:] = verts[self._tet_cap_vidx]
            v = self._tet_cap_verts.reshape(-1, 3, 3)
            normals = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            normals /= norms
            self._tet_cap_normals[:] = np.repeat(normals, 3, axis=0)
        # Update edge verts
        if self._tet_edge_vidx is not None and self._tet_edge_verts is not None:
            self._tet_edge_verts[:] = verts[self._tet_edge_vidx]

    def draw_tetrahedron_mesh(self, draw_tets=False, draw_caps=True):
        """
        Draw the tetrahedron mesh surface using vertex arrays for efficiency.
        """
        if not hasattr(self, 'tet_vertices') or self.tet_vertices is None:
            return

        # Prepare arrays if not done yet
        if not hasattr(self, '_tet_surface_verts') or self._tet_surface_verts is None:
            self._prepare_tet_draw_arrays()

        glPushMatrix()
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)

        alpha = self.contour_mesh_transparency
        color = self.contour_mesh_color

        # Draw surface faces
        if self._tet_surface_verts is not None and len(self._tet_surface_verts) > 0:
            glColor4f(color[0], color[1], color[2], alpha)
            glVertexPointer(3, GL_FLOAT, 0, self._tet_surface_verts)
            glNormalPointer(GL_FLOAT, 0, self._tet_surface_normals)
            glDrawArrays(GL_TRIANGLES, 0, len(self._tet_surface_verts))

        # Draw cap faces in green
        if draw_caps and self._tet_cap_verts is not None and len(self._tet_cap_verts) > 0:
            glColor4f(0.2, 0.6, 0.2, alpha)
            glVertexPointer(3, GL_FLOAT, 0, self._tet_cap_verts)
            glNormalPointer(GL_FLOAT, 0, self._tet_cap_normals)
            glDrawArrays(GL_TRIANGLES, 0, len(self._tet_cap_verts))

        glDisableClientState(GL_NORMAL_ARRAY)

        # Draw tetrahedra edges (lines don't need two-pass)
        if draw_tets and self._tet_edge_verts is not None and len(self._tet_edge_verts) > 0:
            glDisable(GL_LIGHTING)
            glColor4f(0.5, 0.5, 0.5, 0.3)
            glLineWidth(1.0)
            glVertexPointer(3, GL_FLOAT, 0, self._tet_edge_verts)
            glDrawArrays(GL_LINES, 0, len(self._tet_edge_verts))
            glEnable(GL_LIGHTING)

        glDisableClientState(GL_VERTEX_ARRAY)
        glPopMatrix()

    def invalidate_constraints_cache(self):
        """Call this when constraint data changes."""
        self._constraints_cache = None

    def draw_constraints(self):
        """
        Draw soft body constraints visualization:
        - Fixed vertices (red spheres)
        - Skeleton attachments (lines to attachment points)
        - Edge constraints (optional)
        """
        if not hasattr(self, 'tet_vertices') or self.tet_vertices is None:
            return
        if not hasattr(self, 'soft_body') or self.soft_body is None:
            return

        # Build cache if needed
        if not hasattr(self, '_constraints_cache') or self._constraints_cache is None:
            verts = self.tet_vertices
            cache = {}

            # Fixed vertices
            if hasattr(self, 'soft_body_fixed_vertices') and len(self.soft_body_fixed_vertices) > 0:
                fixed_pts = np.array([verts[vi] for vi in self.soft_body_fixed_vertices if vi < len(verts)], dtype=np.float32)
                if len(fixed_pts) > 0:
                    cache['fixed_pts'] = fixed_pts

            # Anchor lines
            if hasattr(self, 'soft_body_local_anchors') and len(self.soft_body_local_anchors) > 0:
                anchor_lines = []
                for vi, (body_name, local_pos) in self.soft_body_local_anchors.items():
                    if vi < len(verts):
                        v = verts[vi]
                        anchor_lines.append(v)
                        anchor_lines.append([v[0], v[1] + 0.01, v[2]])
                if len(anchor_lines) > 0:
                    cache['anchor_lines'] = np.array(anchor_lines, dtype=np.float32)

            # Contour level vertices
            if hasattr(self, 'contour_level_vertices') and len(self.contour_level_vertices) > 0:
                origin_pts = np.array([verts[vi] for vi, (stream_idx, end_type) in self.contour_level_vertices.items()
                                       if end_type == 0 and vi < len(verts)], dtype=np.float32)
                if len(origin_pts) > 0:
                    cache['origin_pts'] = origin_pts

                insert_pts = np.array([verts[vi] for vi, (stream_idx, end_type) in self.contour_level_vertices.items()
                                       if end_type == 1 and vi < len(verts)], dtype=np.float32)
                if len(insert_pts) > 0:
                    cache['insert_pts'] = insert_pts

            # Edge constraints
            if hasattr(self, 'soft_body') and hasattr(self.soft_body, 'edges'):
                edges = self.soft_body.edges
                if len(edges) > 0 and len(edges) < 5000:
                    edge_verts = []
                    for i, j in edges:
                        if i < len(verts) and j < len(verts):
                            edge_verts.append(verts[i])
                            edge_verts.append(verts[j])
                    if len(edge_verts) > 0:
                        cache['edge_verts'] = np.array(edge_verts, dtype=np.float32)

            self._constraints_cache = cache

        # Draw from cache
        glPushMatrix()
        glDisable(GL_LIGHTING)
        glEnableClientState(GL_VERTEX_ARRAY)

        cache = self._constraints_cache

        if 'fixed_pts' in cache:
            glColor4f(1.0, 0.2, 0.2, 1.0)
            glPointSize(8.0)
            glVertexPointer(3, GL_FLOAT, 0, cache['fixed_pts'])
            glDrawArrays(GL_POINTS, 0, len(cache['fixed_pts']))

        if 'anchor_lines' in cache:
            glColor4f(1.0, 1.0, 0.0, 1.0)
            glLineWidth(2.0)
            glVertexPointer(3, GL_FLOAT, 0, cache['anchor_lines'])
            glDrawArrays(GL_LINES, 0, len(cache['anchor_lines']))

        if 'origin_pts' in cache:
            glPointSize(6.0)
            glColor4f(0.2, 0.4, 1.0, 1.0)
            glVertexPointer(3, GL_FLOAT, 0, cache['origin_pts'])
            glDrawArrays(GL_POINTS, 0, len(cache['origin_pts']))

        if 'insert_pts' in cache:
            glPointSize(6.0)
            glColor4f(0.2, 1.0, 0.4, 1.0)
            glVertexPointer(3, GL_FLOAT, 0, cache['insert_pts'])
            glDrawArrays(GL_POINTS, 0, len(cache['insert_pts']))

        if 'edge_verts' in cache:
            glColor4f(0.5, 0.5, 0.5, 0.2)
            glLineWidth(1.0)
            glVertexPointer(3, GL_FLOAT, 0, cache['edge_verts'])
            glDrawArrays(GL_LINES, 0, len(cache['edge_verts']))

        glDisableClientState(GL_VERTEX_ARRAY)
        glEnable(GL_LIGHTING)
        glPopMatrix()
