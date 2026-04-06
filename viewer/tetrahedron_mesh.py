# Tetrahedron Mesh operations for muscle mesh processing
# Extracted from mesh_loader.py for better organization

import numpy as np
try:
    from OpenGL.GL import *
except ImportError:
    pass
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
            for li, loop in enumerate(boundary_loops):
                if len(loop) < 10:
                    loop_pos = np.array([vertices[vi] for vi in loop])
                    loop_mean = loop_pos.mean(axis=0)
                    loop_span = np.linalg.norm(loop_pos.max(axis=0) - loop_pos.min(axis=0))
                    print(f"  [Small loop {li}] {len(loop)} verts, span={loop_span:.6f}, "
                          f"mean=[{loop_mean[0]:.4f},{loop_mean[1]:.4f},{loop_mean[2]:.4f}]")

            # Step 3: Create cap faces for each boundary loop
            # Skip tiny loops (merge artifacts, not real openings)
            min_cap_size = 10
            real_loops = []
            for loop in boundary_loops:
                if len(loop) >= min_cap_size:
                    real_loops.append(loop)
                else:
                    print(f"  Skipping tiny boundary loop ({len(loop)} verts) — merge artifact")
            boundary_loops = real_loops

            closed_vertices = list(vertices)
            closed_faces = list(faces)
            cap_face_indices = []
            self.tet_anchor_vertices = []  # Store anchor vertex indices for each cap

            from shapely.geometry import Polygon as ShapelyPolygon
            from shapely.algorithms.polylabel import polylabel

            for loop_idx, loop in enumerate(boundary_loops):
                # Find a guaranteed-interior anchor using pole of inaccessibility
                loop_vertices = np.array([vertices[vi] for vi in loop])

                # Project loop to 2D using PCA to find the plane
                centroid = loop_vertices.mean(axis=0)
                centered = loop_vertices - centroid
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                basis_x = Vt[0]
                basis_y = Vt[1]

                pts_2d = np.array([[np.dot(v - centroid, basis_x),
                                    np.dot(v - centroid, basis_y)] for v in loop_vertices])

                # Use pole of inaccessibility for interior point
                try:
                    poly = ShapelyPolygon(pts_2d)
                    if poly.is_valid and not poly.is_empty:
                        pole = polylabel(poly, tolerance=1e-4)
                        mean_point = centroid + pole.x * basis_x + pole.y * basis_y
                    else:
                        mean_point = centroid
                except Exception:
                    mean_point = centroid

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

        # Step 4: Tetrahedralization
        # Try TetGen first (quality constrained Delaunay — well-shaped interior tets,
        # preserves surface shape, adds Steiner points on edges/faces/interior).
        # Falls back to contour-guided approach if TetGen fails.
        use_tetgen = getattr(self, 'use_tetgen', True)
        tetgen_success = False

        if use_tetgen:
            try:
                print("Performing TetGen tetrahedralization (subprocess)...")
                n_original = len(closed_vertices)
                # Save cap vertex positions before TetGen modifies vertices
                _anchor_positions = {}
                if hasattr(self, 'tet_anchor_vertices'):
                    for ai in self.tet_anchor_vertices:
                        if ai < len(closed_vertices):
                            _anchor_positions[ai] = closed_vertices[ai].copy()
                # Track cap vertices through subdivision
                _cap_verts = set()
                for fi in cap_face_indices:
                    if fi < len(closed_faces):
                        for vi in closed_faces[fi]:
                            _cap_verts.add(int(vi))

                # Run TetGen in a subprocess to isolate crashes
                import tempfile, subprocess, json, sys
                with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_in:
                    tmp_in_path = tmp_in.name
                    np.savez(tmp_in, vertices=closed_vertices.astype(np.float64),
                             faces=closed_faces.astype(np.int32),
                             cap_verts=np.array(sorted(_cap_verts), dtype=np.int32))
                tmp_out_path = tmp_in_path.replace('.npz', '_tet.npz')

                script = f'''
import numpy as np, sys
try:
    import tetgen, trimesh, pymeshfix
    from scipy.spatial import cKDTree
    data = np.load("{tmp_in_path}")
    verts = data["vertices"].astype(np.float64)
    faces = data["faces"].astype(np.int32)
    cap_vert_indices = set(data["cap_verts"].tolist())
    n_orig = len(verts)
    # Scale up for numerical precision
    rv = verts * 1000.0
    rf = faces.copy()
    # Track cap status: set of vertex indices that are cap vertices
    is_cap = set(cap_vert_indices)
    # Fix non-manifold edges by duplicating shared vertices.
    # Multi-stream muscles share edges at cut boundaries (4+ faces per edge).
    # TetGen needs manifold input (max 2 faces per edge).
    from collections import Counter as _Ctr, defaultdict as _ddict
    edge_faces = _ddict(list)
    for fi, f in enumerate(rf):
        for i in range(3):
            e = tuple(sorted([int(f[i]), int(f[(i+1)%3])]))
            edge_faces[e].append(fi)
    non_manifold_edges = {{e: flist for e, flist in edge_faces.items() if len(flist) > 2}}
    boundary_edges = {{e: flist for e, flist in edge_faces.items() if len(flist) == 1}}
    print(f"FIX_MANIFOLD: {{len(non_manifold_edges)}} non-manifold, {{len(boundary_edges)}} boundary edges")
    if len(non_manifold_edges) > 0:
        print(f"FIX_MANIFOLD: {{len(non_manifold_edges)}} non-manifold edges")
        # Collect vertices that need duplication
        nm_verts = set()
        for e in non_manifold_edges:
            nm_verts.add(e[0])
            nm_verts.add(e[1])
        # Group faces into connected components (streams)
        face_adj = _ddict(set)
        for e, flist in edge_faces.items():
            if len(flist) == 2:
                face_adj[flist[0]].add(flist[1])
                face_adj[flist[1]].add(flist[0])
        visited = set()
        components = []
        for fi in range(len(rf)):
            if fi in visited:
                continue
            comp = []
            stack = [fi]
            while stack:
                f = stack.pop()
                if f in visited:
                    continue
                visited.add(f)
                comp.append(f)
                for nb in face_adj[f]:
                    if nb not in visited:
                        stack.append(nb)
            components.append(comp)
        print(f"FIX_MANIFOLD: {{len(components)}} face components")
        # For each component beyond the first, duplicate shared vertices
        rv_list = list(rv)
        rf_new = rf.copy()
        dup_map = {{}}  # (component_idx, orig_vi) -> new_vi
        for ci in range(1, len(components)):
            for fi in components[ci]:
                for vi_pos in range(3):
                    vi = int(rf_new[fi][vi_pos])
                    if vi in nm_verts:
                        key = (ci, vi)
                        if key not in dup_map:
                            dup_map[key] = len(rv_list)
                            rv_list.append(rv_list[vi].copy() if hasattr(rv_list[vi], 'copy') else np.array(rv_list[vi]))
                            # Propagate cap status
                            if vi in is_cap:
                                is_cap.add(dup_map[key])
                        rf_new[fi][vi_pos] = dup_map[key]
        rv_fixed = np.array(rv_list, dtype=np.float64)
        rf_fixed = rf_new
        print(f"FIX_MANIFOLD: duplicated {{len(dup_map)}} vertices ({{len(rv)}} -> {{len(rv_fixed)}})")
    else:
        rv_fixed = rv.copy()
        rf_fixed = rf.copy()
    # Subdivide long surface edges so TetGen produces fine surface tets
    edge_lens = []
    for f_face in rf_fixed:
        for i in range(3):
            edge_lens.append(np.linalg.norm(rv_fixed[f_face[(i+1)%3]] - rv_fixed[f_face[i]]))
    median_edge = np.median(edge_lens)
    max_edge_len = median_edge * 0.7  # subdivide most edges for uniform surface
    print(f"EDGE_STATS: median={{median_edge:.2f}} threshold={{max_edge_len:.2f}} exceed={{sum(1 for e in edge_lens if e > max_edge_len)}}")
    for _subdiv_iter in range(3):
        new_verts = list(rv_fixed)
        new_faces = []
        edge_midpoints = dict()
        n_split = 0
        for f in rf_fixed:
            splits = []
            for i in range(3):
                v0i, v1i = int(f[i]), int(f[(i+1)%3])
                elen = np.linalg.norm(rv_fixed[v0i] - rv_fixed[v1i]) if v0i < len(rv_fixed) and v1i < len(rv_fixed) else 0
                if elen > max_edge_len:
                    ekey = (min(v0i,v1i), max(v0i,v1i))
                    if ekey not in edge_midpoints:
                        mid = (np.array(new_verts[v0i]) + np.array(new_verts[v1i])) / 2
                        mid_idx = len(new_verts)
                        edge_midpoints[ekey] = mid_idx
                        new_verts.append(mid)
                        # Propagate cap: if both endpoints are cap, midpoint is cap
                        if v0i in is_cap and v1i in is_cap:
                            is_cap.add(mid_idx)
                    splits.append((i, edge_midpoints[ekey]))
                else:
                    splits.append((i, None))
            split_edges = [(i, mi) for i, mi in splits if mi is not None]
            if len(split_edges) == 0:
                new_faces.append(list(f))
            elif len(split_edges) == 1:
                ei, mi = split_edges[0]
                v0, v1, v2 = int(f[ei]), int(f[(ei+1)%3]), int(f[(ei+2)%3])
                new_faces.append([v0, mi, v2])
                new_faces.append([mi, v1, v2])
                n_split += 1
            elif len(split_edges) == 2:
                ei0, mi0 = split_edges[0]
                ei1, mi1 = split_edges[1]
                vs = [int(f[0]), int(f[1]), int(f[2])]
                mids = {{}}
                for ei, mi in split_edges:
                    mids[ei] = mi
                if 0 in mids and 1 in mids:
                    new_faces.append([vs[0], mids[0], vs[2]])
                    new_faces.append([mids[0], vs[1], mids[1]])
                    new_faces.append([mids[0], mids[1], vs[2]])
                elif 0 in mids and 2 in mids:
                    new_faces.append([vs[0], mids[0], mids[2]])
                    new_faces.append([mids[0], vs[1], vs[2]])
                    new_faces.append([mids[0], vs[2], mids[2]])
                elif 1 in mids and 2 in mids:
                    new_faces.append([vs[0], vs[1], mids[1]])
                    new_faces.append([vs[0], mids[1], mids[2]])
                    new_faces.append([mids[1], vs[2], mids[2]])
                n_split += 1
            else:  # all 3 edges split
                vs = [int(f[0]), int(f[1]), int(f[2])]
                m01 = splits[0][1]
                m12 = splits[1][1]
                m20 = splits[2][1]
                new_faces.append([vs[0], m01, m20])
                new_faces.append([m01, vs[1], m12])
                new_faces.append([m20, m12, vs[2]])
                new_faces.append([m01, m12, m20])
                n_split += 1
        rv_fixed = np.array(new_verts, dtype=np.float64)
        rf_fixed = np.array(new_faces, dtype=np.int32)
        if n_split == 0:
            break
        print(f"SUBDIVIDE iter {{_subdiv_iter}}: split {{n_split}} faces, now {{len(rv_fixed)}}v {{len(rf_fixed)}}f")
    # Fix normals
    mesh = trimesh.Trimesh(vertices=rv_fixed, faces=rf_fixed, process=False)
    mesh.fix_normals()
    rv_fixed, rf_fixed = mesh.vertices.astype(np.float64), mesh.faces.astype(np.int32)
    # Compute max tet volume targeting ~2000 tets
    _mesh_vol = abs(mesh.volume)
    target_tets = 2000
    max_vol = max(_mesh_vol / target_tets, 1e-6)
    print(f"MESH_VOL={{_mesh_vol:.1f}} MAX_VOL={{max_vol:.1f}} TARGET={{target_tets}}")
    # TetGen — quality + maxvolume, with steinerleft to cap count
    # For thin muscles, quality constraints force many tets.
    # Use steinerleft to limit total vertex count while allowing
    # surface subdivision (nobisect=False).
    target_verts = 3000
    n_surface = len(rv_fixed)
    steiner_budget = max(target_verts - n_surface, 500)
    try:
        tet = tetgen.TetGen(rv_fixed.copy(), rf_fixed.copy())
        tet.tetrahedralize(order=1, mindihedral=5, minratio=2.0,
                           maxvolume=max_vol, nobisect=True,
                           steinerleft=steiner_budget)
        n_tets = len(tet.elem)
        n_verts = len(tet.node)
        print(f"QUALITY: steiner_budget={{steiner_budget}} -> {{n_tets}} tets, {{n_verts}} verts")
    except Exception:
        tet = tetgen.TetGen(rv_fixed.copy(), rf_fixed.copy())
        tet.tetrahedralize(quality=False, nobisect=False)
        print(f"NOQUALITY -> {{len(tet.elem)}} tets")
    # With nobisect=True, TetGen only adds interior Steiner points.
    # Surface vertices are unchanged — cap status is already correct.
    tet_node = tet.node
    tet_elem = tet.elem

    cap_verts_out = np.array(sorted(is_cap), dtype=np.int32)
    print(f"CAP_VERTS: {{len(cap_verts_out)}}")
    np.savez("{tmp_out_path}", node=tet_node / 1000.0, elem=tet_elem,
             n_orig=np.array([n_orig]), cap_verts=cap_verts_out)
    print(f"OK {{len(tet_elem)}} {{len(tet_node)}}")
except Exception as e:
    print(f"FAIL: {{e}}")
    sys.exit(1)
'''
                result = subprocess.run(
                    [sys.executable, '-c', script],
                    capture_output=True, text=True, timeout=120)

                import os
                stdout_lines = result.stdout.strip().split('\n') if result.stdout else []
                for line in stdout_lines:
                    if line.startswith(('REPAIRED', 'QUALITY', 'NOQUALITY', 'EDGE_STATS', 'SUBDIVIDE', 'MESH_VOL', 'CAP_VERTS', 'SKIP_REPAIR', 'FIX_MANIFOLD')):
                        print(f"  {line}")

                if result.returncode == 0 and os.path.exists(tmp_out_path):
                    tet_data = np.load(tmp_out_path)
                    tet_verts = tet_data['node']
                    tet_elems = tet_data['elem'].astype(np.int32)

                    # pymeshfix may have changed vertex count — check what
                    # fraction of original vertices are preserved
                    from scipy.spatial import cKDTree as _cKDTree
                    orig_tree = _cKDTree(closed_vertices.astype(np.float64))
                    dists, indices = orig_tree.query(tet_verts[:len(tet_verts)])
                    # Count how many original vertices have a near match in tet output
                    tet_tree = _cKDTree(tet_verts)
                    orig_dists, _ = tet_tree.query(closed_vertices.astype(np.float64))
                    n_preserved = int(np.sum(orig_dists < 1e-6))

                    # Quality stats — signed volume for inversion check
                    tv0 = tet_verts[tet_elems[:, 0]]
                    tcr = np.cross(tet_verts[tet_elems[:, 1]] - tv0,
                                   tet_verts[tet_elems[:, 2]] - tv0)
                    tvol_signed = np.einsum('ij,ij->i', tcr,
                                  tet_verts[tet_elems[:, 3]] - tv0) / 6.0
                    tvol = np.abs(tvol_signed)
                    n_good = int(np.sum(tvol >= 1e-12))
                    n_inverted = int(np.sum(tvol_signed < 0))
                    print(f"  TetGen: {len(tet_elems)} tets, {len(tet_verts)} verts, "
                          f"{n_preserved}/{n_original} original preserved, {n_good} good, {n_inverted} inverted")
                    print(f"    Volume range: [{tvol.min():.2e}, {tvol.max():.2e}]")

                    # Build old→new vertex index mapping via nearest vertex
                    old_verts = closed_vertices.astype(np.float64)
                    tet_tree2 = _cKDTree(tet_verts)
                    old_to_new = np.zeros(len(old_verts), dtype=np.int32)
                    for oi in range(len(old_verts)):
                        _, ni = tet_tree2.query(old_verts[oi])
                        old_to_new[oi] = ni

                    # Remap render faces to new vertex indices
                    new_render_faces = []
                    for f in closed_faces:
                        nf = [int(old_to_new[int(v)]) if int(v) < len(old_to_new) else 0 for v in f]
                        if len(set(nf)) == 3:  # skip degenerate
                            new_render_faces.append(nf)
                    closed_faces = np.array(new_render_faces, dtype=np.int32)

                    # Re-identify cap faces: faces touching anchor vertices
                    anchor_set = set()
                    if hasattr(self, 'tet_anchor_vertices'):
                        for ai in self.tet_anchor_vertices:
                            if ai < len(old_to_new):
                                anchor_set.add(int(old_to_new[ai]))
                    cap_face_indices = []
                    for fi, f in enumerate(closed_faces):
                        if any(int(v) in anchor_set for v in f):
                            cap_face_indices.append(fi)

                    closed_vertices = tet_verts.astype(np.float32)
                    interior_tetrahedra = tet_elems
                    tetgen_success = True

                    # Rebuild contour_to_tet mapping
                    if hasattr(self, 'contour_to_tet_indices') and self.contour_to_tet_indices is not None:
                        old_c2t = self.contour_to_tet_indices
                        new_c2t = []
                        for ci, ti in enumerate(old_c2t):
                            if ti >= 0 and ti < len(old_to_new):
                                new_c2t.append(int(old_to_new[ti]))
                            else:
                                new_c2t.append(-1)
                        self.contour_to_tet_indices = new_c2t
                else:
                    stderr = result.stderr.strip()[-200:] if result.stderr else ''
                    stdout_msg = stdout_lines[-1] if stdout_lines else ''
                    print(f"  TetGen subprocess failed: {stdout_msg} {stderr}")

                # Cleanup temp files
                for p in [tmp_in_path, tmp_out_path]:
                    try:
                        os.remove(p)
                    except OSError:
                        pass

            except Exception as e:
                print(f"  TetGen failed: {e}, falling back to contour-guided")

        try:
         if not tetgen_success:
            # Build vertex-to-level mapping
            vertex_level = getattr(self, 'vertex_contour_level', None)
            if vertex_level is not None:
                n_cv = len(closed_vertices)
                if len(vertex_level) < n_cv:
                    # Extend to cover cap anchor vertices
                    extended = np.full(n_cv, -1, dtype=np.int32)
                    extended[:len(vertex_level)] = vertex_level
                    vertex_level = extended
                elif len(vertex_level) > n_cv:
                    # Trim if gap-closing added vertices not in closed_vertices
                    vertex_level = vertex_level[:n_cv]
            if vertex_level is None:
                print("  Warning: no vertex_contour_level, falling back to Delaunay")

            if vertex_level is not None:
                n_original = len(closed_vertices)
                num_levels = int(vertex_level.max()) + 1
                new_verts = list(closed_vertices)
                interior_tetrahedra = []

                if hasattr(self, 'contours') and self.contours is not None:
                    n_streams = len(self.contours)
                else:
                    n_streams = 1

                # Assign each vertex to a stream via connected components per level
                vertex_stream = np.full(len(closed_vertices), -1, dtype=np.int32)

                from collections import defaultdict as _dd
                from shapely.geometry import Polygon as ShapelyPolygon
                from shapely.algorithms.polylabel import polylabel

                # ── Step 4a: Compute pole-of-inaccessibility centers per stream per level ──
                level_centers = {}  # (stream_idx, level) -> vertex index in new_verts
                mid_centers = {}    # (stream_idx, level) -> vertex index (between level and level+1)

                for lev in range(num_levels):
                    lev_verts = np.where(vertex_level == lev)[0]
                    if len(lev_verts) == 0:
                        continue

                    # Split into per-stream clusters via face connectivity
                    if n_streams > 1 and len(lev_verts) > 6:
                        adj = _dd(set)
                        lev_set = set(lev_verts)
                        for face in closed_faces:
                            fv = [int(face[0]), int(face[1]), int(face[2])]
                            fv_on_lev = [v for v in fv if v in lev_set]
                            for a in fv_on_lev:
                                for b in fv_on_lev:
                                    if a != b:
                                        adj[a].add(b)

                        visited = set()
                        clusters = []
                        for start in lev_verts:
                            if start in visited:
                                continue
                            cluster = []
                            queue = [start]
                            visited.add(start)
                            while queue:
                                v = queue.pop()
                                cluster.append(v)
                                for nb in adj[v]:
                                    if nb not in visited:
                                        visited.add(nb)
                                        queue.append(nb)
                            clusters.append(cluster)
                    else:
                        clusters = [list(lev_verts)]

                    for ci_idx, cluster in enumerate(clusters):
                        # Tag vertices with stream
                        for vi in cluster:
                            vertex_stream[vi] = ci_idx

                        # Get bounding plane data for this stream/level to project to 2D
                        bp = None
                        if (hasattr(self, 'bounding_planes') and self.bounding_planes is not None
                                and ci_idx < len(self.bounding_planes)
                                and lev < len(self.bounding_planes[ci_idx])):
                            bp = self.bounding_planes[ci_idx][lev]

                        if bp is not None and 'basis_x' in bp and 'basis_y' in bp and 'mean' in bp:
                            # Project cluster vertices to 2D using contour's basis
                            basis_x = bp['basis_x']
                            basis_y = bp['basis_y']
                            mean = bp['mean']
                            pts_3d = closed_vertices[cluster]
                            pts_2d = np.array([[np.dot(v - mean, basis_x),
                                                np.dot(v - mean, basis_y)] for v in pts_3d])

                            from scipy.spatial import ConvexHull
                            try:
                                if len(pts_2d) >= 3:
                                    hull = ConvexHull(pts_2d)
                                    hull_pts = pts_2d[hull.vertices]
                                    poly = ShapelyPolygon(hull_pts)
                                    if not poly.is_valid or poly.is_empty:
                                        raise ValueError("invalid polygon")

                                    # Check aspect ratio — for thin cross-sections,
                                    # add multiple centers along the major axis
                                    min_rect = poly.minimum_rotated_rectangle
                                    rect_coords = np.array(min_rect.exterior.coords[:-1])
                                    edge_lens = [np.linalg.norm(rect_coords[(i+1)%4] - rect_coords[i])
                                                 for i in range(4)]
                                    major = max(edge_lens)
                                    minor = min(edge_lens)
                                    aspect = major / max(minor, 1e-8)

                                    if aspect > 3.0 and len(pts_2d) >= 6:
                                        # Thin cross-section: slice polygon along major
                                        # axis into strips, polylabel each strip.
                                        # Each polylabel is guaranteed inside the polygon.
                                        n_centers = min(int(aspect / 2), 5)
                                        n_centers = max(n_centers, 2)
                                        print(f"    [THIN] stream {ci_idx} level {lev}: aspect={aspect:.1f}, major={major:.4f}, minor={minor:.4f}, {n_centers} centers")
                                        from shapely.geometry import box as shapely_box
                                        from shapely import affinity as shapely_affinity
                                        # Major axis direction and angle
                                        edge0 = rect_coords[1] - rect_coords[0]
                                        edge1 = rect_coords[2] - rect_coords[1]
                                        if np.linalg.norm(edge0) > np.linalg.norm(edge1):
                                            major_dir = edge0 / np.linalg.norm(edge0)
                                        else:
                                            major_dir = edge1 / np.linalg.norm(edge1)
                                        angle = np.degrees(np.arctan2(major_dir[1], major_dir[0]))
                                        # Rotate polygon to align major axis with X
                                        cx, cy = poly.centroid.x, poly.centroid.y
                                        rotated = shapely_affinity.rotate(poly, -angle, origin=(cx, cy))
                                        rx0, ry0, rx1, ry1 = rotated.bounds
                                        centers_3d = []
                                        for i in range(n_centers):
                                            # Slice bounds along X
                                            t0 = i / n_centers
                                            t1 = (i + 1) / n_centers
                                            sx0 = rx0 + t0 * (rx1 - rx0)
                                            sx1 = rx0 + t1 * (rx1 - rx0)
                                            clip = shapely_box(sx0, ry0 - 1, sx1, ry1 + 1)
                                            strip = rotated.intersection(clip)
                                            if strip.is_empty or strip.area < 1e-12:
                                                continue
                                            try:
                                                pole_r = polylabel(strip, tolerance=1e-4)
                                                # Rotate back
                                                pole_back = shapely_affinity.rotate(
                                                    pole_r, angle, origin=(cx, cy))
                                                pt_2d = np.array([pole_back.x, pole_back.y])
                                            except Exception:
                                                continue
                                            c3d = mean + pt_2d[0] * basis_x + pt_2d[1] * basis_y
                                            centers_3d.append(c3d)
                                        if len(centers_3d) == 0:
                                            # Fallback to single polylabel
                                            pole = polylabel(poly, tolerance=1e-4)
                                            centers_3d = [mean + pole.x * basis_x + pole.y * basis_y]
                                            print(f"      fallback to single polylabel (all strips empty)")
                                        else:
                                            print(f"      placed {len(centers_3d)}/{n_centers} strip centers")
                                    else:
                                        # Normal aspect — single pole center
                                        pole = polylabel(poly, tolerance=1e-4)
                                        centers_3d = [mean + pole.x * basis_x + pole.y * basis_y]
                                else:
                                    centers_3d = [pts_3d.mean(axis=0)]
                            except Exception:
                                centers_3d = [closed_vertices[cluster].mean(axis=0)]
                        else:
                            centers_3d = [closed_vertices[cluster].mean(axis=0)]

                        # Store all centers for this stream/level
                        center_indices = []
                        for c3d in centers_3d:
                            ci = len(new_verts)
                            new_verts.append(c3d)
                            center_indices.append(ci)
                        level_centers[(ci_idx, lev)] = center_indices

                # ── Step 4b: Compute mid-centers between adjacent levels ──
                for (si, lev), cis in list(level_centers.items()):
                    next_key = (si, lev + 1)
                    if next_key in level_centers:
                        next_cis = level_centers[next_key]
                        # Average all centers from both levels
                        all_pts = [new_verts[c] for c in cis] + [new_verts[c] for c in next_cis]
                        mid_3d = np.mean(all_pts, axis=0)
                        mid_idx = len(new_verts)
                        new_verts.append(mid_3d)
                        mid_centers[(si, lev)] = mid_idx  # between lev and lev+1

                n_level = sum(len(v) for v in level_centers.values())
                n_mid = len(mid_centers)
                print(f"  {n_level} level centers + {n_mid} mid-centers added "
                      f"({n_streams} streams, {num_levels} levels)")

                # ── Step 4c: Connect each face to the best center ──
                def compute_tet_volume(p0, p1, p2, p3):
                    """Absolute volume of tetrahedron (p0,p1,p2,p3)."""
                    return abs(np.dot(np.cross(p1 - p0, p2 - p0), p3 - p0)) / 6.0

                def find_candidates(stream, levels_in_face):
                    """Collect all candidate center indices for a face."""
                    candidates = []
                    for lev in levels_in_face:
                        for dl in [-1, 0, 1]:
                            target_lev = lev + dl
                            if target_lev < 0:
                                continue
                            key = (stream, target_lev)
                            if key in level_centers:
                                candidates.extend(level_centers[key])  # list of indices
                            if key in mid_centers:
                                candidates.append(mid_centers[key])
                            prev_key = (stream, target_lev - 1)
                            if prev_key in mid_centers:
                                candidates.append(mid_centers[prev_key])
                    if not candidates:
                        for lev in levels_in_face:
                            for s in range(n_streams):
                                for dl in [-1, 0, 1]:
                                    target_lev = lev + dl
                                    key = (s, target_lev)
                                    if key in level_centers:
                                        candidates.extend(level_centers[key])
                                    if key in mid_centers:
                                        candidates.append(mid_centers[key])
                    return list(set(candidates))

                n_skipped = 0
                for face in closed_faces:
                    v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
                    l0 = int(vertex_level[v0]) if v0 < len(vertex_level) else -1
                    l1 = int(vertex_level[v1]) if v1 < len(vertex_level) else -1
                    l2 = int(vertex_level[v2]) if v2 < len(vertex_level) else -1

                    levels_in_face = set([l0, l1, l2]) - {-1}
                    if len(levels_in_face) == 0:
                        continue

                    face_streams = set([vertex_stream[v0], vertex_stream[v1], vertex_stream[v2]]) - {-1}
                    face_stream = min(face_streams) if face_streams else 0
                    p0, p1, p2 = new_verts[v0], new_verts[v1], new_verts[v2]

                    candidates = find_candidates(face_stream, levels_in_face)

                    best_ci = None
                    best_vol = 0
                    for ci in candidates:
                        vol = compute_tet_volume(p0, p1, p2, new_verts[ci])
                        if vol > best_vol:
                            best_vol = vol
                            best_ci = ci

                    if best_ci is not None and best_vol > 0:
                        interior_tetrahedra.append([v0, v1, v2, best_ci])
                    else:
                        n_skipped += 1

                if n_skipped > 0:
                    print(f"  Skipped {n_skipped} faces (no valid center found)")

                closed_vertices = np.array(new_verts, dtype=np.float32)
                interior_tetrahedra = np.array(interior_tetrahedra, dtype=np.int32)

                # (No post-hoc degenerate tet removal — centers are guaranteed
                # interior and offset from face planes)

                # Check vertex coverage
                used = set(interior_tetrahedra.ravel()) if len(interior_tetrahedra) > 0 else set()
                unused = [i for i in range(n_original) if i not in used]
                if unused:
                    print(f"  {len(unused)} original vertices unused, adding via Delaunay...")
                    # For unused vertices, use local Delaunay to connect them
                    delaunay = Delaunay(closed_vertices[:n_original])
                    all_tets = delaunay.simplices
                    for tet in all_tets:
                        if any(v in unused for v in tet) and any(v in used for v in tet):
                            interior_tetrahedra = np.vstack([interior_tetrahedra, tet])
                            for v in tet:
                                if v in unused:
                                    unused.remove(v)
                    if unused:
                        print(f"  Warning: {len(unused)} vertices still unused")

                print(f"  Contour-guided: {len(interior_tetrahedra)} tets, "
                      f"{len(closed_vertices)} verts "
                      f"(+{len(closed_vertices) - n_original} centers)")

                # Quality check: detailed volume distribution
                v0 = closed_vertices[interior_tetrahedra[:, 0]].astype(np.float64)
                cr = np.cross(
                    closed_vertices[interior_tetrahedra[:, 1]].astype(np.float64) - v0,
                    closed_vertices[interior_tetrahedra[:, 2]].astype(np.float64) - v0)
                vol_signed = np.einsum('ij,ij->i', cr,
                             closed_vertices[interior_tetrahedra[:, 3]].astype(np.float64) - v0) / 6.0
                vol = np.abs(vol_signed)
                n_total = len(vol)
                n_inverted = int(np.sum(vol_signed < 0))
                print(f"  Tet quality ({n_total} tets, {n_inverted} inverted):")
                print(f"    Volume range: [{vol.min():.2e}, {vol.max():.2e}]")
                for threshold in [1e-15, 1e-12, 1e-10, 1e-8]:
                    n = int(np.sum(vol < threshold))
                    if n > 0:
                        print(f"    vol < {threshold:.0e}: {n} ({100*n/n_total:.1f}%)")
                n_good = int(np.sum(vol >= 1e-12))
                print(f"    Good tets (vol >= 1e-12): {n_good} ({100*n_good/n_total:.1f}%)")

                # Detail on thin tets
                thin_mask = vol < 1e-10
                if np.any(thin_mask):
                    thin_indices = np.where(thin_mask)[0]
                    for ti in thin_indices:
                        tet = interior_tetrahedra[ti]
                        tv0, tv1, tv2, tv3 = int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])
                        tl = [int(vertex_level[v]) if v < len(vertex_level) else -1 for v in tet]
                        ts = [int(vertex_stream[v]) if v < len(vertex_stream) else -1 for v in tet]
                        is_center = [v >= n_original for v in tet]
                        print(f"    [THIN tet {ti}] vol={vol[ti]:.2e}")
                        print(f"      verts=({tv0},{tv1},{tv2},{tv3}) levels=({tl[0]},{tl[1]},{tl[2]},{tl[3]}) streams=({ts[0]},{ts[1]},{ts[2]},{ts[3]})")
                        print(f"      is_center=({is_center[0]},{is_center[1]},{is_center[2]},{is_center[3]})")
                        print(f"      positions:")
                        for vi, v in enumerate(tet):
                            pos = closed_vertices[int(v)]
                            print(f"        v{vi}={int(v)}: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}] L{tl[vi]} S{ts[vi]}{'  [CENTER]' if is_center[vi] else ''}")

                # --- Diagnostic checks ---

                # [1] Unused vertices
                all_tet_verts = set(interior_tetrahedra.ravel())
                n_unused_orig = sum(1 for i in range(n_original) if i not in all_tet_verts)
                if n_unused_orig > 0:
                    print(f"  [!] UNUSED VERTICES: {n_unused_orig}/{n_original} original verts not in any tet")
                else:
                    print(f"  [OK] All {n_original} original vertices used in tets")

                # [2] Waypoint info
                if hasattr(self, 'waypoints') and self.waypoints is not None and len(self.waypoints) > 0:
                    n_wp = sum(len(wp) for stream in self.waypoints
                               for wp in stream if hasattr(wp, '__len__'))
                    print(f"  [INFO] Waypoints: {n_wp} (embedding checked at runtime, "
                          f"watch for 'clamped/outside' warnings)")
                else:
                    print(f"  [INFO] No waypoints stored")

                # [3] Tet count vs surface faces
                n_faces = len(closed_faces)
                print(f"  [INFO] Tet density: {n_total} tets from {n_faces} faces "
                      f"(ratio {n_total/max(n_faces,1):.2f}x)")

                # [4] Steiner center vertices
                n_centers = len(closed_vertices) - n_original
                print(f"  [INFO] Steiner centers: {n_centers} level centroids added")

                # [5] Internal connectivity
                from collections import Counter as _Counter
                _fc = _Counter()
                for tet in interior_tetrahedra:
                    for f in [(tet[0],tet[1],tet[2]), (tet[0],tet[1],tet[3]),
                              (tet[0],tet[2],tet[3]), (tet[1],tet[2],tet[3])]:
                        _fc[tuple(sorted(f))] += 1
                n_shared = sum(1 for c in _fc.values() if c >= 2)
                n_boundary = sum(1 for c in _fc.values() if c == 1)
                pct = 100 * n_shared / max(len(_fc), 1)
                print(f"  [INFO] Connectivity: {n_shared} shared / {n_boundary} boundary faces "
                      f"({pct:.0f}% internal)")

                # [6] Stream info
                if hasattr(self, 'contours') and self.contours is not None:
                    n_streams = len(self.contours)
                    stream_lvls = [len(s) for s in self.contours]
                    print(f"  [INFO] Streams: {n_streams}, levels: {stream_lvls}")

            else:
                # Fallback: Delaunay (no vertex_contour_level available)
                print("  Falling back to Delaunay (no vertex_contour_level)")
                delaunay = Delaunay(closed_vertices)
                tetrahedra = delaunay.simplices
                mesh = trimesh.Trimesh(vertices=closed_vertices, faces=closed_faces)
                mesh.fix_normals()
                inside_mask = mesh.contains(np.mean(closed_vertices[tetrahedra], axis=1))
                interior_tetrahedra = tetrahedra[inside_mask]
                if len(interior_tetrahedra) == 0:
                    interior_tetrahedra = tetrahedra
                # Quality check for Delaunay fallback
                v0 = closed_vertices[interior_tetrahedra[:, 0]].astype(np.float64)
                cr = np.cross(
                    closed_vertices[interior_tetrahedra[:, 1]].astype(np.float64) - v0,
                    closed_vertices[interior_tetrahedra[:, 2]].astype(np.float64) - v0)
                vol = np.abs(np.einsum('ij,ij->i', cr,
                             closed_vertices[interior_tetrahedra[:, 3]].astype(np.float64) - v0)) / 6.0
                n_total = len(vol)
                print(f"  Delaunay fallback: {n_total} tets")
                print(f"  Tet quality ({n_total} tets):")
                print(f"    Volume range: [{vol.min():.2e}, {vol.max():.2e}]")
                for threshold in [1e-15, 1e-12, 1e-10, 1e-8]:
                    n = int(np.sum(vol < threshold))
                    if n > 0:
                        print(f"    vol < {threshold:.0e}: {n} ({100*n/n_total:.1f}%)")
                n_good = int(np.sum(vol >= 1e-12))
                print(f"    Good tets (vol >= 1e-12): {n_good} ({100*n_good/n_total:.1f}%)")
                print(f"  Delaunay fallback: {len(interior_tetrahedra)} tets")

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
        # - tet_render_faces: Surface faces for rendering
        # - tet_sim_faces: Tet boundary faces (for simulation boundary conditions)
        if tetgen_success:
            # TetGen subdivides the surface — use tet boundary as render faces
            self.tet_render_faces = sim_faces
            # Cap faces = faces where ALL 3 vertices are tracked cap vertices
            tet_cap_set = set()
            if 'cap_verts' in tet_data.files:
                tet_cap_set = set(tet_data['cap_verts'].tolist())
            cap_face_indices = []
            for fi, f in enumerate(sim_faces):
                if all(int(v) in tet_cap_set for v in f):
                    cap_face_indices.append(fi)
        else:
            self.tet_render_faces = closed_faces  # Original surface + caps
        self.tet_sim_faces = sim_faces  # Tet boundary faces
        self.tet_faces = self.tet_render_faces  # Backwards compatibility

        self.tet_cap_face_indices = cap_face_indices  # Indices of cap faces (fixed during simulation)
        self.tet_surface_face_count = len(self.tet_render_faces) - len(cap_face_indices)

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
                'mvc_weights': getattr(self, 'mvc_weights', None),
                '_stream_endpoints': getattr(self, '_stream_endpoints', None),
                'stream_contours': getattr(self, 'stream_contours', None),
                'stream_bounding_planes': getattr(self, 'stream_bounding_planes', None),
                'stream_groups': getattr(self, 'stream_groups', None),
                'vertex_contour_level': getattr(self, 'vertex_contour_level', None),
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
                self._fiber_draw_dirty = True
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

            # Load vertex contour level (for contour-guided tetrahedralization)
            if 'vertex_contour_level' in data and data['vertex_contour_level'] is not None:
                self.vertex_contour_level = data['vertex_contour_level']

            # Load MVC weights (for deforming waypoints with tet sim)
            if 'mvc_weights' in data and data['mvc_weights'] is not None:
                self.mvc_weights = data['mvc_weights']
                print(f"  Loaded MVC weights")

            # Load stream endpoints (for fiber architecture)
            if '_stream_endpoints' in data and data['_stream_endpoints'] is not None:
                self._stream_endpoints = data['_stream_endpoints']
                print(f"  Loaded stream endpoints")

            # Load stream data (post-cut contours/BPs/groups)
            if 'stream_contours' in data and data['stream_contours'] is not None:
                self.stream_contours = data['stream_contours']
                print(f"  Loaded {len(self.stream_contours)} stream contours")
            if 'stream_bounding_planes' in data and data['stream_bounding_planes'] is not None:
                self.stream_bounding_planes = data['stream_bounding_planes']
            if 'stream_groups' in data and data['stream_groups'] is not None:
                self.stream_groups = data['stream_groups']

            # Ensure contour_to_tet_indices exists (may be missing in stripped files)
            if not hasattr(self, 'contour_to_tet_indices'):
                self.contour_to_tet_indices = []

            # Store name for bary coords caching
            self._tet_name = name

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

        # Prepare edge arrays from tetrahedra (shows all edges including interior)
        edge_set = set()
        if self.tet_tetrahedra is not None and len(self.tet_tetrahedra) > 0:
            for tet in self.tet_tetrahedra:
                for i in range(4):
                    for j in range(i + 1, 4):
                        edge = (min(tet[i], tet[j]), max(tet[i], tet[j]))
                        edge_set.add(edge)
        else:
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

    def _update_tet_draw_positions(self, skip_normals=False):
        """Fast path: update draw arrays from tet_vertices using precomputed index arrays.
        Call this instead of _prepare_tet_draw_arrays when only positions changed (not topology).
        If skip_normals=True, only update vertex positions (normals stay from previous frame)."""
        if not hasattr(self, '_tet_surface_vidx') or self._tet_surface_vidx is None:
            self._prepare_tet_draw_arrays()
            return
        verts = self.tet_vertices
        # Update surface verts + normals
        if self._tet_surface_vidx is not None and self._tet_surface_verts is not None:
            self._tet_surface_verts[:] = verts[self._tet_surface_vidx]
            if not skip_normals:
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
            if not skip_normals:
                v = self._tet_cap_verts.reshape(-1, 3, 3)
                normals = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                normals /= norms
                self._tet_cap_normals[:] = np.repeat(normals, 3, axis=0)
        # Update edge verts
        if self._tet_edge_vidx is not None and self._tet_edge_verts is not None:
            self._tet_edge_verts[:] = verts[self._tet_edge_vidx]

    def _draw_tet_mesh_animated(self):
        """Tet edge animation phases 1-2 (phase 3+ uses normal draw_tetrahedron_mesh).
        Phase 1: edges grow from origin→insertion (like contour mesh edge anim)
        Phase 2: edges fade out alpha 1.0→0.0"""
        if self.tet_vertices is None:
            return

        verts = self.tet_vertices
        color = self.contour_mesh_color
        num_bands = getattr(self, '_tet_anim_num_bands', 0)
        band_edges = getattr(self, '_tet_anim_band_edges', None)
        vcl = getattr(self, '_tet_vertex_level', None)
        phase = self._tet_anim_phase
        progress = self._tet_anim_progress

        if band_edges is None or vcl is None or num_bands == 0:
            return

        # Timing must match update_tet_animation
        fade_dur = 1.5
        grow_dur = 2.0
        edge_fade_dur = 0.8

        def smoothstep(x):
            x = max(0.0, min(1.0, x))
            return x * x * (3.0 - 2.0 * x)

        glPushMatrix()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if phase == 1:
            # Edge grow: lines grow from origin→insertion
            t = (progress - fade_dur) / grow_dur if grow_dur > 0 else 1.0
            level_prog = smoothstep(t) * num_bands

            glDisable(GL_LIGHTING)
            glLineWidth(1.5)
            glColor4f(color[0], color[1], color[2], 1.0)
            glBegin(GL_LINES)
            for band_idx in range(num_bands):
                if band_idx > level_prog + 1:
                    break
                if band_idx >= len(band_edges):
                    continue
                for vi0, vi1 in band_edges[band_idx]:
                    lv0 = max(int(vcl[vi0]), 0)
                    lv1 = max(int(vcl[vi1]), 0)
                    min_lv = min(lv0, lv1)
                    max_lv = max(lv0, lv1)
                    if min_lv > level_prog:
                        continue
                    p0 = verts[vi0]
                    p1 = verts[vi1]
                    if lv0 == lv1:
                        glVertex3fv(p0)
                        glVertex3fv(p1)
                    elif max_lv <= level_prog:
                        glVertex3fv(p0)
                        glVertex3fv(p1)
                    else:
                        frac = max(0.0, min(1.0, level_prog - min_lv))
                        if lv0 <= lv1:
                            glVertex3fv(p0)
                            glVertex3fv(p0 + frac * (p1 - p0))
                        else:
                            glVertex3fv(p1)
                            glVertex3fv(p1 + frac * (p0 - p1))
            glEnd()
            glEnable(GL_LIGHTING)

        elif phase == 2:
            # Edge fade: all edges visible, alpha 1.0→0.0
            t = (progress - fade_dur - grow_dur) / edge_fade_dur if edge_fade_dur > 0 else 1.0
            wire_alpha = 1.0 - smoothstep(t)

            if wire_alpha > 0.005:
                glDisable(GL_LIGHTING)
                glLineWidth(1.5)
                glColor4f(color[0], color[1], color[2], wire_alpha)
                glBegin(GL_LINES)
                for band_idx in range(num_bands):
                    if band_idx >= len(band_edges):
                        continue
                    for v0, v1 in band_edges[band_idx]:
                        glVertex3fv(verts[v0])
                        glVertex3fv(verts[v1])
                glEnd()
                glEnable(GL_LIGHTING)

        glPopMatrix()

    def draw_tetrahedron_mesh(self, draw_tets=False, draw_caps=True):
        """
        Draw the tetrahedron mesh surface using vertex arrays for efficiency.
        """
        if not hasattr(self, 'tet_vertices') or self.tet_vertices is None:
            return

        # During tet animation phase 0: tet not visible yet, skip draw
        if getattr(self, '_tet_anim_active', False) and self._tet_anim_phase == 0:
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

        # During tet animation cross-fade (phase 1), use override alpha
        if getattr(self, '_tet_anim_active', False) and self._tet_anim_phase == 1:
            alpha = getattr(self, '_tet_anim_tet_alpha', alpha)

        # Draw surface faces
        if self._tet_surface_verts is not None and len(self._tet_surface_verts) > 0:
            heatmap_colors = getattr(self, '_tet_surface_colors', None)
            if heatmap_colors is not None and len(heatmap_colors) == len(self._tet_surface_verts):
                glEnableClientState(GL_COLOR_ARRAY)
                glColorPointer(4, GL_FLOAT, 0, heatmap_colors)
            else:
                glColor4f(color[0], color[1], color[2], alpha)
            glVertexPointer(3, GL_FLOAT, 0, self._tet_surface_verts)
            glNormalPointer(GL_FLOAT, 0, self._tet_surface_normals)
            glDrawArrays(GL_TRIANGLES, 0, len(self._tet_surface_verts))
            if heatmap_colors is not None and len(heatmap_colors) == len(self._tet_surface_verts):
                glDisableClientState(GL_COLOR_ARRAY)

        # Draw cap faces in green
        if draw_caps and self._tet_cap_verts is not None and len(self._tet_cap_verts) > 0:
            glColor4f(0.2, 0.6, 0.2, alpha)
            glVertexPointer(3, GL_FLOAT, 0, self._tet_cap_verts)
            glNormalPointer(GL_FLOAT, 0, self._tet_cap_normals)
            glDrawArrays(GL_TRIANGLES, 0, len(self._tet_cap_verts))

        glDisableClientState(GL_NORMAL_ARRAY)

        # Draw tetrahedra edges
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
