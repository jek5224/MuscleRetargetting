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

        # Step 0.5: Skip vertex merging — per-stream separation in the subprocess
        # handles shared boundary vertices via deterministic merge after TetGen.

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
            # Use face-consistent traversal: at each vertex, follow the open edge
            # that shares a face with the current edge (handles T-junctions).

            # Build per-vertex open edge adjacency

            # For each open edge, find which face(s) it belongs to
            open_edge_sorted_set = set(open_edges)  # open_edges already (min,max) sorted
            edge_to_face = defaultdict(list)
            for fi, f in enumerate(faces):
                for ei in range(3):
                    a, b = int(f[ei]), int(f[(ei+1)%3])
                    e_sorted = (min(a,b), max(a,b))
                    if e_sorted in open_edge_sorted_set:
                        edge_to_face[(a,b)].append(fi)
                        edge_to_face[(b,a)].append(fi)

            # For each vertex, group its open edges by face connectivity
            # At a vertex V with incoming edge (U,V), the next edge (V,W) should
            # share a face with (U,V) — i.e., face contains both edges U-V and V-W
            vertex_open_neighbors = defaultdict(list)
            for edge in open_edges:
                vertex_open_neighbors[edge[0]].append(edge[1])
                vertex_open_neighbors[edge[1]].append(edge[0])

            # Report T-junctions
            for v, nbrs in vertex_open_neighbors.items():
                if len(nbrs) > 2:
                    print(f"  T-junction at vertex {v}: {len(nbrs)} open-edge neighbors {nbrs}")

            # Trace boundary loops using edge traversal (not vertex traversal)
            visited_edges = set()
            boundary_loops = []

            for start_edge in open_edges:
                if start_edge in visited_edges:
                    continue

                loop = []
                # Start: follow the edge start_edge[0] -> start_edge[1]
                prev_v = start_edge[0]
                curr_v = start_edge[1]
                loop.append(prev_v)
                visited_edges.add((min(prev_v,curr_v), max(prev_v,curr_v)))

                loop_closed = False
                max_steps = len(open_edges) + 1
                for _ in range(max_steps):
                    loop.append(curr_v)

                    # Find next open edge from curr_v (not going back to prev_v)
                    # Prefer the edge that shares a face with (prev_v, curr_v)
                    incoming_faces = set(edge_to_face.get((prev_v, curr_v), []))
                    candidates = []
                    for nbr in vertex_open_neighbors[curr_v]:
                        if nbr == prev_v:
                            continue
                        e_key = (min(curr_v,nbr), max(curr_v,nbr))
                        if e_key in visited_edges:
                            continue
                        # Check if this edge shares a face with the incoming edge
                        nbr_faces = set(edge_to_face.get((curr_v, nbr), []))
                        shared = incoming_faces & nbr_faces
                        candidates.append((nbr, e_key, len(shared) > 0))

                    # Prefer face-consistent neighbor, then any unvisited
                    next_v = None
                    next_key = None
                    for nbr, e_key, face_shared in candidates:
                        if face_shared:
                            next_v, next_key = nbr, e_key
                            break
                    if next_v is None and candidates:
                        next_v, next_key = candidates[0][0], candidates[0][1]

                    if next_v is None:
                        # Check if we can close back to start
                        start_v = loop[0]
                        e_close = (min(curr_v,start_v), max(curr_v,start_v))
                        if e_close in open_edge_sorted_set and e_close not in visited_edges:
                            visited_edges.add(e_close)
                            loop_closed = True
                        break

                    visited_edges.add(next_key)
                    prev_v = curr_v
                    curr_v = next_v

                    # Check if we closed the loop
                    if curr_v == loop[0]:
                        loop_closed = True
                        break

                if len(loop) >= 3:
                    loop_pos = np.array([vertices[vi] for vi in loop])
                    gap = np.linalg.norm(loop_pos[0] - loop_pos[-1])
                    loop_span = np.linalg.norm(loop_pos.max(axis=0) - loop_pos.min(axis=0))
                    centered = loop_pos - loop_pos.mean(axis=0)
                    _, S, _ = np.linalg.svd(centered, full_matrices=False)
                    planarity = S[2] / S[0] if S[0] > 0 else 0
                    status = "CLOSED" if loop_closed else f"OPEN (gap={gap:.4f})"
                    print(f"  Loop {len(boundary_loops)}: {len(loop)} verts, span={loop_span:.4f}, "
                          f"planarity={planarity:.4f}, {status}")
                    # Remove duplicate last vertex if loop closed (start == end)
                    if loop_closed and len(loop) > 1 and loop[-1] == loop[0]:
                        loop = loop[:-1]
                    boundary_loops.append(loop)

            print(f"Found {len(boundary_loops)} boundary loops")

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
            self.tet_anchor_vertices = []

            import triangle as tr

            def _segments_cross(pts_2d, seg_a, seg_b):
                """Check if two 2D segments cross (strictly, not at shared endpoints)."""
                a, b = pts_2d[seg_a[0]], pts_2d[seg_a[1]]
                c, d = pts_2d[seg_b[0]], pts_2d[seg_b[1]]
                def cross2(o, p, q):
                    return (p[0]-o[0])*(q[1]-o[1]) - (p[1]-o[1])*(q[0]-o[0])
                d1, d2 = cross2(c,d,a), cross2(c,d,b)
                d3, d4 = cross2(a,b,c), cross2(a,b,d)
                if ((d1>0 and d2<0) or (d1<0 and d2>0)) and \
                   ((d3>0 and d4<0) or (d3<0 and d4>0)):
                    return True
                return False

            def _polygon_self_intersects(pts_2d, n):
                """Check if closed 2D polygon has any crossing edges."""
                for i in range(n):
                    for j in range(i+2, n):
                        if i == 0 and j == n-1: continue  # adjacent
                        if _segments_cross(pts_2d, (i, (i+1)%n), (j, (j+1)%n)):
                            return True
                return False

            def _cap_faces(loop_indices, all_vertices):
                """Cap a boundary loop: CDT if 2D projection is clean, else centroid fan."""
                n = len(loop_indices)
                if n < 3: return [], None
                if n == 3: return [[loop_indices[0], loop_indices[1], loop_indices[2]]], None
                pts_3d = np.array([all_vertices[vi] for vi in loop_indices])
                centroid = pts_3d.mean(axis=0)
                centered = pts_3d - centroid
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)

                # Try projections: best-fit plane, then XY, XZ, YZ
                projections = [
                    ('best-fit', centered @ Vt[:2].T),
                    ('XY', pts_3d[:, :2]),
                    ('XZ', pts_3d[:, [0,2]]),
                    ('YZ', pts_3d[:, 1:3]),
                ]
                for proj_name, pts_2d in projections:
                    if _polygon_self_intersects(pts_2d, n):
                        continue
                    # CDT with boundary constraints
                    segments = np.array([[i, (i+1)%n] for i in range(n)], dtype=np.int32)
                    try:
                        result = tr.triangulate({'vertices': pts_2d, 'segments': segments}, 'p')
                        faces = []
                        for tri_idx in result['triangles']:
                            faces.append([loop_indices[tri_idx[0]], loop_indices[tri_idx[1]], loop_indices[tri_idx[2]]])
                        if len(faces) >= n - 2:
                            print(f"    CDT ({proj_name}): {len(faces)} faces")
                            return faces, None
                    except Exception as e:
                        print(f"    CDT ({proj_name}) failed: {e}")
                        continue

                # All projections self-intersect → centroid fan
                center_idx = len(all_vertices)
                all_vertices.append(centroid.tolist())
                faces = []
                for i in range(n):
                    faces.append([loop_indices[i], loop_indices[(i+1)%n], center_idx])
                print(f"    Fan fallback: {len(faces)} faces (center vertex {center_idx})")
                return faces, center_idx

            for loop_idx, loop in enumerate(boundary_loops):
                cap_faces, center_vi = _cap_faces(list(loop), closed_vertices)
                for cf in cap_faces:
                    cap_face_idx = len(closed_faces)
                    closed_faces.append(cf)
                    cap_face_indices.append(cap_face_idx)
                for vi in loop:
                    if vi not in self.tet_anchor_vertices:
                        self.tet_anchor_vertices.append(vi)
                if center_vi is not None:
                    self.tet_anchor_vertices.append(center_vi)
                expected = len(loop) - 2
                print(f"  Loop {loop_idx}: {len(loop)} vertices, {len(cap_faces)} cap faces")

            closed_vertices = np.array(closed_vertices, dtype=np.float32)
            closed_faces = np.array(closed_faces, dtype=np.int32)

            # Step 3.5: Map each boundary loop to stream origin/insertion
            # Use loop centroid position for matching (no pole anchor vertex)
            self.tet_cap_attachments = []
            if hasattr(self, 'contours') and self.contours is not None and len(self.contours) > 0:
                for loop_idx, loop in enumerate(boundary_loops):
                    loop_pts = np.array([closed_vertices[vi] for vi in loop])
                    anchor_pos = loop_pts.mean(axis=0)
                    anchor_idx = loop[0]  # use first loop vertex as representative
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
        use_tetgen = getattr(self, 'use_tetgen', False)  # Delaunay is primary now
        tetgen_success = False

        if use_tetgen:
            try:
                print("Performing TetGen tetrahedralization (subprocess)...")
                n_original = len(closed_vertices)
                # Save cap info before TetGen modifies vertices
                _anchor_positions = {}
                _anchor_radii = {}  # max distance from anchor to its loop vertices
                if hasattr(self, 'tet_anchor_vertices'):
                    for li, ai in enumerate(self.tet_anchor_vertices):
                        if ai < len(closed_vertices):
                            _anchor_positions[ai] = closed_vertices[ai].copy()
                            # Compute cap radius from boundary loop
                            if li < len(boundary_loops):
                                loop_pts = closed_vertices[boundary_loops[li]]
                                _anchor_radii[ai] = float(np.max(np.linalg.norm(
                                    loop_pts - closed_vertices[ai], axis=1)))
                # Track cap vertices through subdivision
                _cap_verts = set()
                for fi in cap_face_indices:
                    if fi < len(closed_faces):
                        for vi in closed_faces[fi]:
                            _cap_verts.add(int(vi))

                # Save script to file for debugging, then run as subprocess
                import tempfile, subprocess, json, sys
                # Get per-stream face mapping
                face_stream = getattr(self, '_face_stream_map', None)
                # Map cap faces to streams: surface faces have stream assignment,
                # cap faces are after surface faces
                n_surface = len(faces)  # original surface face count (before caps)

                with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_in:
                    tmp_in_path = tmp_in.name
                    save_dict = dict(
                        vertices=closed_vertices.astype(np.float64),
                        faces=closed_faces.astype(np.int32),
                        cap_verts=np.array(sorted(_cap_verts), dtype=np.int32),
                        n_surface=np.array([n_surface], dtype=np.int32),
                    )
                    if face_stream is not None and len(face_stream) == n_surface:
                        save_dict['face_stream'] = face_stream
                    shared_pairs = getattr(self, '_shared_vertex_pairs', [])
                    if shared_pairs:
                        save_dict['shared_pairs'] = np.array(shared_pairs, dtype=np.int32)
                    np.savez(tmp_in, **save_dict)
                tmp_out_path = tmp_in_path.replace('.npz', '_tet.npz')
                tmp_script_path = tmp_in_path.replace('.npz', '_script.py')

                script = f'''
import numpy as np, sys
try:
    import tetgen, trimesh, pymeshfix
    from scipy.spatial import cKDTree
    from collections import defaultdict as _ddict
    data = np.load("{tmp_in_path}")
    verts = data["vertices"].astype(np.float64)
    faces = data["faces"].astype(np.int32)
    cap_vert_indices = set(data["cap_verts"].tolist())
    n_orig = len(verts)
    n_surface = int(data["n_surface"][0])
    rv = verts * 1000.0
    rf = faces.copy()
    is_cap = set(cap_vert_indices)
    print(f"MESH_INPUT: {{len(rv)}}v {{len(rf)}}f, {{len(is_cap)}} cap verts", flush=True)

    # Per-stream face grouping
    if "face_stream" in data.files:
        face_stream = data["face_stream"]
        stream_ids = sorted(set(face_stream.tolist()))
        # Build per-stream face lists (surface faces + their cap faces)
        # Cap faces (after n_surface) need to be assigned to streams via their vertices
        stream_faces = {{si: [] for si in stream_ids}}
        for fi in range(n_surface):
            stream_faces[int(face_stream[fi])].append(fi)
        # Assign cap faces to nearest stream
        for fi in range(n_surface, len(rf)):
            # Find which stream this cap face's vertices belong to
            cap_f_verts = set(int(v) for v in rf[fi])
            best_stream = stream_ids[0]
            best_overlap = 0
            for si in stream_ids:
                si_verts = set()
                for sfi in stream_faces[si]:
                    for v in rf[sfi]:
                        si_verts.add(int(v))
                overlap = len(cap_f_verts & si_verts)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_stream = si
            stream_faces[best_stream].append(fi)
        big_comps = [stream_faces[si] for si in stream_ids if len(stream_faces[si]) >= 10]
    else:
        # Single stream — all faces in one component
        big_comps = [list(range(len(rf)))]
    print(f"COMPONENTS: {{len(big_comps)}} streams")

    all_nodes = []
    all_elems = []
    all_cap_verts = set()
    node_offset = 0
    # Load shared vertex pairs from build_contour_mesh
    shared_pairs_data = data["shared_pairs"] if "shared_pairs" in data.files else np.zeros((0,2), dtype=np.int32)
    print(f"SHARED: {{len(shared_pairs_data)}} deterministic vertex pairs")
    # Pre-compute per-component vertex sets
    comp_used_list = []
    comp_remap_list = []
    for ci, comp in enumerate(big_comps):
        comp_faces_tmp = rf[comp]
        used_tmp = np.unique(comp_faces_tmp.ravel())
        remap_tmp = np.full(len(rv), -1, dtype=np.int32)
        remap_tmp[used_tmp] = np.arange(len(used_tmp), dtype=np.int32)
        comp_used_list.append(used_tmp)
        comp_remap_list.append(remap_tmp)

    for ci, comp in enumerate(big_comps):
        comp_faces = rf[comp]
        used = comp_used_list[ci]
        local_remap = comp_remap_list[ci]
        local_verts = rv[used].copy()
        local_faces = local_remap[comp_faces]
        local_cap = set()
        for vi_global in used:
            if int(vi_global) in is_cap:
                local_cap.add(int(local_remap[vi_global]))
        print(f"COMP_DEBUG {{ci}}: initial local_cap={{len(local_cap)}}")

        # Close holes with manual capping first, pymeshfix as fallback
        mesh = trimesh.Trimesh(vertices=local_verts, faces=local_faces, process=False)
        if not mesh.is_watertight:
            # Find boundary edges and cap with fan triangles
            local_edge_count = _ddict(int)
            for f in local_faces:
                for i in range(3):
                    e = tuple(sorted([int(f[i]), int(f[(i+1)%3])]))
                    local_edge_count[e] += 1
            boundary_edges_local = [e for e, c in local_edge_count.items() if c == 1]
            if len(boundary_edges_local) > 0:
                b_adj = _ddict(list)
                for e in boundary_edges_local:
                    b_adj[e[0]].append(e[1])
                    b_adj[e[1]].append(e[0])
                b_visited = set()
                b_loops = []
                for start in b_adj:
                    if start in b_visited: continue
                    loop = []
                    current = start
                    prev = None
                    while True:
                        loop.append(current)
                        b_visited.add(current)
                        nbs = [n for n in b_adj[current] if n != prev and n not in b_visited]
                        if not nbs: break
                        prev = current
                        current = nbs[0]
                    if len(loop) >= 3:
                        b_loops.append(loop)
                new_faces_list = list(local_faces)
                for loop in b_loops:
                    centroid = np.mean([local_verts[vi] for vi in loop], axis=0)
                    anchor_vi = len(local_verts)
                    local_verts = np.vstack([local_verts, centroid.reshape(1,-1)])
                    local_cap.add(anchor_vi)
                    for i in range(len(loop)):
                        new_faces_list.append([loop[i], loop[(i+1)%len(loop)], anchor_vi])
                        local_cap.add(loop[i])
                local_faces = np.array(new_faces_list, dtype=np.int32)
                print(f"COMP_DEBUG {{ci}}: capped {{len(b_loops)}} boundary loops")
            # Post-cap check
            mesh = trimesh.Trimesh(vertices=local_verts, faces=local_faces, process=False)
            post_edge_count = _ddict(int)
            for f in local_faces:
                for i in range(3):
                    e = tuple(sorted([int(f[i]), int(f[(i+1)%3])]))
                    post_edge_count[e] += 1
            post_nm = sum(1 for c in post_edge_count.values() if c > 2)
            post_bd = sum(1 for c in post_edge_count.values() if c == 1)
            print(f"COMP_DEBUG {{ci}}: after cap: {{len(local_verts)}}v {{len(local_faces)}}f, watertight={{mesh.is_watertight}}, nm={{post_nm}} bd={{post_bd}}")
            # If still not watertight, try pymeshfix on the uncapped component
            if not mesh.is_watertight and post_bd > 0:
                print(f"COMP_DEBUG {{ci}}: still {{post_bd}} boundary edges, using pymeshfix to close")
                verts_before_fix = local_verts.copy()
                cap_before_fix = set(local_cap)
                fixer = pymeshfix.MeshFix(local_verts.copy(), local_faces.copy())
                fixer.repair(verbose=False)
                local_verts = fixer.v.astype(np.float64)
                local_faces = fixer.f.astype(np.int32)
                # Remap cap vertices: find each old cap vertex position in new mesh
                local_cap = set()
                fix_tree = cKDTree(local_verts)
                for vi in cap_before_fix:
                    if vi < len(verts_before_fix):
                        d, ni = fix_tree.query(verts_before_fix[vi])
                        if d < 1.0:
                            local_cap.add(ni)
                print(f"COMP_DEBUG {{ci}}: pymeshfix {{len(verts_before_fix)}}->{{len(local_verts)}}v, cap {{len(cap_before_fix)}}->{{len(local_cap)}}")
                mesh = trimesh.Trimesh(vertices=local_verts, faces=local_faces, process=False)
        # Save pre-subdivision state
        pre_subdiv_verts = local_verts.copy()
        pre_subdiv_faces = local_faces.copy()
        pre_subdiv_cap = set(local_cap)
        use_pymeshfix = False

        mesh.fix_normals()
        local_verts, local_faces = mesh.vertices.astype(np.float64), mesh.faces.astype(np.int32)
        # TetGen — try direct first, pymeshfix fallback
        _mesh_vol = abs(mesh.volume)
        max_vol = max(_mesh_vol / 1500, 1e-6)
        # Scale budget by component size — ~2x surface vertices
        steiner_budget = max(len(local_verts), 300)
        tet = None
        try:
            t = tetgen.TetGen(local_verts.copy(), local_faces.copy())
            t.tetrahedralize(order=1, mindihedral=5, minratio=2.0,
                             maxvolume=max_vol, nobisect=True, steinerleft=steiner_budget)
            tet = t
        except Exception:
            try:
                t = tetgen.TetGen(local_verts.copy(), local_faces.copy())
                t.tetrahedralize(quality=False, nobisect=True)
                tet = t
            except Exception:
                # TetGen failed — use Delaunay + inside filter (preserves all vertices & caps)
                # Use post-subdivision mesh (same level of detail as TetGen path)
                print(f"COMP_DEBUG {{ci}}: TetGen failed, using Delaunay+inside filter", flush=True)
                from scipy.spatial import Delaunay as _Delaunay
                dl_verts = local_verts.copy()
                dl_faces = local_faces.copy()
                dl = _Delaunay(dl_verts)
                all_dl_tets = dl.simplices
                dl_mesh = trimesh.Trimesh(vertices=dl_verts, faces=dl_faces, process=False)
                dl_mesh.fix_normals()
                tet_centers = np.mean(dl_verts[all_dl_tets], axis=1)
                inside = dl_mesh.contains(tet_centers)
                interior_tets = all_dl_tets[inside]
                # Fix orientation
                v0t = dl_verts[interior_tets[:,0]]
                cr = np.cross(dl_verts[interior_tets[:,1]]-v0t, dl_verts[interior_tets[:,2]]-v0t)
                vol = np.einsum('ij,ij->i', cr, dl_verts[interior_tets[:,3]]-v0t) / 6.0
                neg = vol < 0
                if np.any(neg):
                    interior_tets[neg,1], interior_tets[neg,2] = interior_tets[neg,2].copy(), interior_tets[neg,1].copy()
                class FakeTet:
                    pass
                tet = FakeTet()
                tet.node = dl_verts
                tet.elem = interior_tets.astype(np.int32)
                local_verts = dl_verts
                local_faces = dl_faces
                print(f"COMP_DEBUG {{ci}}: Delaunay: {{len(interior_tets)}} interior tets from {{len(all_dl_tets)}}", flush=True)
        if tet is None:
            print(f"COMP {{ci}}: FAILED completely")
            continue
        print(f"COMP {{ci}}: {{len(tet.elem)}} tets, {{len(tet.node)}} verts, {{len(local_cap)}} cap verts")
        # Map cap verts to global output indices
        for vi in local_cap:
            if vi < len(tet.node):
                all_cap_verts.add(vi + node_offset)
        all_nodes.append(tet.node / 1000.0)
        all_elems.append(tet.elem + node_offset)
        node_offset += len(tet.node)

    if len(all_nodes) == 0:
        raise RuntimeError("No components tetrahedralized")
    merged_nodes = np.vstack(all_nodes)
    merged_elems = np.vstack(all_elems).astype(np.int32)
    # Deduplicate shared boundary vertices using known correspondences
    n_before_dedup = len(merged_nodes)
    dedup_map = np.arange(n_before_dedup, dtype=np.int32)
    # Deterministic merge of shared boundary vertices using known pairs
    # Each pair (gvi_a, gvi_b) represents the same contour point in different streams.
    # After per-stream tetrahedralization, find each global vertex's merged index
    # and unify them.
    node_offsets = [0]
    for nodes in all_nodes:
        node_offsets.append(node_offsets[-1] + len(nodes))
    n_merged = 0
    for pair in shared_pairs_data:
        gvi_a, gvi_b = int(pair[0]), int(pair[1])
        # Find which component each global vertex is in
        merged_a = merged_b = -1
        for ci_s in range(len(big_comps)):
            remap_s = comp_remap_list[ci_s]
            if gvi_a < len(remap_s) and remap_s[gvi_a] >= 0:
                merged_a = node_offsets[ci_s] + int(remap_s[gvi_a])
            if gvi_b < len(remap_s) and remap_s[gvi_b] >= 0:
                merged_b = node_offsets[ci_s] + int(remap_s[gvi_b])
        if merged_a >= 0 and merged_b >= 0 and merged_a != merged_b:
            lo, hi = min(merged_a, merged_b), max(merged_a, merged_b)
            if lo < n_before_dedup and hi < n_before_dedup:
                dedup_map[hi] = lo
                if hi in all_cap_verts:
                    all_cap_verts.add(lo)
                n_merged += 1
    # Compact
    for i in range(n_before_dedup):
        v = i
        while dedup_map[v] != v:
            v = dedup_map[v]
        dedup_map[i] = v
    unique_mask = np.array([dedup_map[i] == i for i in range(n_before_dedup)])
    new_idx = np.full(n_before_dedup, -1, dtype=np.int32)
    new_idx[unique_mask] = np.arange(int(unique_mask.sum()), dtype=np.int32)
    for i in range(n_before_dedup):
        if new_idx[i] < 0:
            new_idx[i] = new_idx[dedup_map[i]]
    merged_nodes = merged_nodes[unique_mask]
    merged_elems = new_idx[merged_elems]
    all_cap_verts = set(int(new_idx[v]) for v in all_cap_verts if v < len(new_idx) and new_idx[v] >= 0)
    n_deduped = n_before_dedup - len(merged_nodes)
    print(f"DEDUP: {{n_merged}} pairs merged ({{n_before_dedup}}->{{len(merged_nodes)}} verts)")
    cap_verts_out = np.array(sorted(all_cap_verts), dtype=np.int32)
    print(f"MERGED: {{len(merged_elems)}} tets, {{len(merged_nodes)}} verts")
    print(f"CAP_VERTS: {{len(cap_verts_out)}}")
    np.savez("{tmp_out_path}", node=merged_nodes, elem=merged_elems,
             n_orig=np.array([n_orig]), cap_verts=cap_verts_out)
    print(f"OK {{len(merged_elems)}} {{len(merged_nodes)}}")
except Exception as e:
    print(f"FAIL: {{e}}")
    import traceback; traceback.print_exc()
    sys.exit(1)
'''
                with open(tmp_script_path, 'w') as sf:
                    sf.write(script)
                result = subprocess.run(
                    [sys.executable, tmp_script_path],
                    capture_output=True, text=True, timeout=120)

                import os
                stdout_lines = result.stdout.strip().split('\n') if result.stdout else []
                for line in stdout_lines:
                    if line.startswith(('REPAIRED', 'QUALITY', 'NOQUALITY', 'EDGE_STATS', 'SUBDIVIDE', 'MESH_VOL', 'CAP_VERTS', 'SKIP_REPAIR', 'FIX_MANIFOLD', 'MESH_INPUT', 'FAIL', 'COMP', 'MERGED', 'COMPONENTS', 'COMP_DEBUG', 'DEDUP')):
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
                    stderr = result.stderr.strip()[-500:] if result.stderr else ''
                    stdout_msg = result.stdout.strip()[-500:] if result.stdout else ''
                    print(f"  TetGen subprocess failed (rc={result.returncode}):")
                    if stdout_msg:
                        print(f"    stdout: {stdout_msg}")
                    if stderr:
                        print(f"    stderr: {stderr}")
                    print(f"    Script saved: {tmp_script_path}")

                # Cleanup temp files (keep script for debugging)
                for p in [tmp_in_path, tmp_out_path]:
                    try:
                        os.remove(p)
                    except OSError:
                        pass

            except Exception as e:
                print(f"  TetGen failed: {e}, falling back to contour-guided")

        delaunay_success = False
        if not tetgen_success:
            # Delaunay tetrahedralization with inside filtering — 100% coverage
            try:
                from scipy.spatial import Delaunay as _Delaunay
                print("Performing Delaunay tetrahedralization...")
                dl = _Delaunay(closed_vertices.astype(np.float64))
                all_dl_tets = dl.simplices

                # Filter: keep tets whose centroid is inside the surface mesh
                mesh_dl = trimesh.Trimesh(vertices=closed_vertices.astype(np.float64),
                                          faces=closed_faces.astype(np.int32), process=False)
                mesh_dl.fix_normals()
                dl_centroids = np.mean(closed_vertices[all_dl_tets].astype(np.float64), axis=1)
                interior_mask = mesh_dl.contains(dl_centroids)

                # Add tets needed for uncovered interior points
                n_dense = min(50000, max(10000, len(closed_vertices) * 50))
                bbox_min = closed_vertices.min(axis=0)
                bbox_max = closed_vertices.max(axis=0)
                dense_samples = np.random.uniform(bbox_min, bbox_max, (n_dense, 3)).astype(np.float64)
                interior_pts = dense_samples[mesh_dl.contains(dense_samples)]
                if len(interior_pts) > 0:
                    simplex_ids = dl.find_simplex(interior_pts)
                    interior_set = set(np.where(interior_mask)[0])
                    n_added = 0
                    for si in simplex_ids:
                        if si >= 0 and si not in interior_set:
                            interior_set.add(si)
                            interior_mask[si] = True
                            n_added += 1
                    if n_added > 0:
                        print(f"  Added {n_added} boundary tets for full coverage")

                interior_tetrahedra = all_dl_tets[interior_mask].astype(np.int32)

                # Fix orientation
                v0_dl = closed_vertices[interior_tetrahedra[:, 0]].astype(np.float64)
                cr_dl = np.cross(
                    closed_vertices[interior_tetrahedra[:, 1]].astype(np.float64) - v0_dl,
                    closed_vertices[interior_tetrahedra[:, 2]].astype(np.float64) - v0_dl)
                vol_dl = np.einsum('ij,ij->i', cr_dl,
                    closed_vertices[interior_tetrahedra[:, 3]].astype(np.float64) - v0_dl) / 6.0
                neg_dl = vol_dl < 0
                if np.any(neg_dl):
                    interior_tetrahedra[neg_dl, 1], interior_tetrahedra[neg_dl, 2] = \
                        interior_tetrahedra[neg_dl, 2].copy(), interior_tetrahedra[neg_dl, 1].copy()

                n_inverted = int(np.sum(neg_dl))
                n_total = len(interior_tetrahedra)
                used_verts = set(interior_tetrahedra.ravel())
                vol_abs = np.abs(vol_dl)
                mesh_vol = abs(mesh_dl.volume)

                print(f"  Delaunay: {n_total} tets from {len(all_dl_tets)} total")
                print(f"    {n_inverted} orientation-fixed, 0 inverted")
                print(f"    Volume: {vol_abs.sum():.6f} vs mesh {mesh_vol:.6f} ({vol_abs.sum()/mesh_vol*100:.1f}%)")
                print(f"    Vertices used: {len(used_verts)}/{len(closed_vertices)}")
                print(f"    Volume range: [{vol_abs.min():.2e}, {vol_abs.max():.2e}]")

                delaunay_success = True

            except Exception as e:
                print(f"  Delaunay failed: {e}, falling back to contour-guided")
                import traceback
                traceback.print_exc()

        try:
         if not tetgen_success and not delaunay_success:
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

                # Step 4d: Add bridge tets to fill gaps between adjacent tets
                # Adjacent contour-guided tets share a surface edge but NOT a face
                # (different center vertices). Bridge tets connect the shared edge
                # to both centers, making the mesh properly connected.
                surface_verts_set = set(int(v) for f in closed_faces for v in f)
                center_verts_set = set(int(v) for t in interior_tetrahedra for v in t) - surface_verts_set

                if len(center_verts_set) > 0:
                    from collections import defaultdict as _dd_bridge
                    surface_edge_tets = _dd_bridge(list)
                    for ti, t in enumerate(interior_tetrahedra):
                        centers = [int(v) for v in t if int(v) in center_verts_set]
                        surfaces = [int(v) for v in t if int(v) in surface_verts_set]
                        if len(centers) != 1 or len(surfaces) != 3:
                            continue
                        center = centers[0]
                        for i in range(3):
                            for j in range(i + 1, 3):
                                ekey = (min(surfaces[i], surfaces[j]), max(surfaces[i], surfaces[j]))
                                surface_edge_tets[ekey].append((ti, center))

                    bridge_set = set()
                    bridge_tets_raw = []
                    for ekey, tet_list in surface_edge_tets.items():
                        centers_seen = sorted(set(c for _, c in tet_list))
                        if len(centers_seen) < 2:
                            continue
                        for i in range(len(centers_seen)):
                            for j in range(i + 1, len(centers_seen)):
                                bridge = tuple(sorted([ekey[0], ekey[1], centers_seen[i], centers_seen[j]]))
                                if bridge in bridge_set:
                                    continue
                                bridge_set.add(bridge)
                                b = [ekey[0], ekey[1], centers_seen[i], centers_seen[j]]
                                p0 = closed_vertices[b[0]].astype(np.float64)
                                cross_b = np.cross(
                                    closed_vertices[b[1]].astype(np.float64) - p0,
                                    closed_vertices[b[2]].astype(np.float64) - p0)
                                vol_b = np.dot(cross_b, closed_vertices[b[3]].astype(np.float64) - p0) / 6.0
                                if abs(vol_b) < 1e-12:
                                    continue
                                if vol_b < 0:
                                    b[2], b[3] = b[3], b[2]
                                    vol_b = -vol_b
                                bridge_tets_raw.append((bridge, b, vol_b))

                    # Remove bridges that would create over-shared faces
                    orig_face_count = _dd_bridge(int)
                    for t in interior_tetrahedra:
                        for face in [(t[0],t[1],t[2]),(t[0],t[1],t[3]),(t[0],t[2],t[3]),(t[1],t[2],t[3])]:
                            orig_face_count[tuple(sorted(int(v) for v in face))] += 1
                    face_to_bridges = _dd_bridge(list)
                    for idx, (key, b, vol_b) in enumerate(bridge_tets_raw):
                        for face in [(b[0],b[1],b[2]),(b[0],b[1],b[3]),(b[0],b[2],b[3]),(b[1],b[2],b[3])]:
                            face_to_bridges[tuple(sorted(face))].append((idx, vol_b))
                    remove_set = set()
                    for fkey, blist in face_to_bridges.items():
                        total = orig_face_count.get(fkey, 0) + len(blist)
                        if total > 2:
                            blist.sort(key=lambda x: -x[1])
                            allowed = max(0, 2 - orig_face_count.get(fkey, 0))
                            for idx, _ in blist[allowed:]:
                                remove_set.add(idx)

                    bridge_tets_final = [b for idx, (_, b, _) in enumerate(bridge_tets_raw) if idx not in remove_set]

                    if bridge_tets_final:
                        bridge_arr = np.array(bridge_tets_final, dtype=interior_tetrahedra.dtype)
                        interior_tetrahedra = np.vstack([interior_tetrahedra, bridge_arr])
                        print(f"  Bridge tets: {len(bridge_tets_final)} added "
                              f"({len(remove_set)} removed for over-sharing)")
                    else:
                        print(f"  Bridge tets: 0 (no gaps found)")
                else:
                    print(f"  Bridge tets: skipped (no center vertices found)")

                # Fix inverted tets (both original and bridge)
                v0_fix = closed_vertices[interior_tetrahedra[:, 0]].astype(np.float64)
                cr_fix = np.cross(
                    closed_vertices[interior_tetrahedra[:, 1]].astype(np.float64) - v0_fix,
                    closed_vertices[interior_tetrahedra[:, 2]].astype(np.float64) - v0_fix)
                vol_fix = np.einsum('ij,ij->i', cr_fix,
                    closed_vertices[interior_tetrahedra[:, 3]].astype(np.float64) - v0_fix) / 6.0
                neg_fix = vol_fix < 0
                if np.any(neg_fix):
                    interior_tetrahedra[neg_fix, 1], interior_tetrahedra[neg_fix, 2] = \
                        interior_tetrahedra[neg_fix, 2].copy(), interior_tetrahedra[neg_fix, 1].copy()
                    print(f"  Fixed {int(np.sum(neg_fix))} inverted tets")

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
            # Use original contour mesh faces (remapped) for rendering
            # sim_faces are for simulation boundary conditions
            self.tet_render_faces = closed_faces
            # Cap faces = faces where ALL 3 vertices are tracked cap vertices
            tet_cap_set = set()
            if 'cap_verts' in tet_data.files:
                tet_cap_set = set(tet_data['cap_verts'].tolist())
            cap_face_indices = []
            for fi, f in enumerate(closed_faces):
                if all(int(v) in tet_cap_set for v in f):
                    cap_face_indices.append(fi)
            # Fallback: for anchors with too few nearby cap faces
            if hasattr(self, 'tet_anchor_vertices') and _anchor_positions:
                from scipy.spatial import cKDTree as _cKDTree_fb
                tet_tree_fb = _cKDTree_fb(closed_vertices.astype(np.float64))
                cap_fi_set = set(cap_face_indices)
                sim_fc = np.mean(closed_vertices[closed_faces].astype(np.float64), axis=1)
                for loop_idx, (ai, apos) in enumerate(_anchor_positions.items()):
                    _, ani = tet_tree_fb.query(apos.astype(np.float64))
                    dists = np.linalg.norm(sim_fc - closed_vertices[ani].astype(np.float64), axis=1)
                    near = np.where(dists < 0.02)[0]
                    n_cap = sum(1 for fi in near if fi in cap_fi_set)
                    if n_cap < 5:
                        # pymeshfix replaced the pole fan with zig-zag triangulation.
                        # Find the cap plane from closest cap verts, then mark
                        # ALL faces on that plane within the cap radius.
                        anchor_pos = closed_vertices[ani].astype(np.float64)
                        cap_radius = _anchor_radii.get(ai, 0.01)
                        # Find cap verts where THIS anchor is their NEAREST anchor
                        all_anchor_pos = {a: closed_vertices[tet_tree_fb.query(p.astype(np.float64))[1]].astype(np.float64)
                                          for a, p in _anchor_positions.items()}
                        nearby_cap_verts = []
                        for cvi in tet_cap_set:
                            if cvi >= len(closed_vertices): continue
                            pt = closed_vertices[cvi].astype(np.float64)
                            # Is this anchor the nearest?
                            d_this = np.linalg.norm(pt - anchor_pos)
                            is_nearest = True
                            for other_ai, other_pos in all_anchor_pos.items():
                                if other_ai == ai: continue
                                if np.linalg.norm(pt - other_pos) < d_this:
                                    is_nearest = False
                                    break
                            if is_nearest and d_this < cap_radius * 1.5:
                                nearby_cap_verts.append(cvi)
                        if len(nearby_cap_verts) >= 3:
                            cap_pts = closed_vertices[nearby_cap_verts].astype(np.float64)
                            centroid = cap_pts.mean(axis=0)
                            _, _, Vt = np.linalg.svd(cap_pts - centroid, full_matrices=False)
                            cap_normal = Vt[2]
                            # Search ALL sim_faces within cap_radius (not just 0.02 "near")
                            added = 0
                            for fi in range(len(sim_faces)):
                                if fi in cap_fi_set:
                                    continue
                                fv = closed_vertices[closed_faces[fi]].astype(np.float64)
                                fc_pos = fv.mean(axis=0)
                                # Must be within cap radius
                                if np.linalg.norm(fc_pos - anchor_pos) > cap_radius * 1.2:
                                    continue
                                # All verts must be near the cap plane
                                plane_dists = np.abs(np.dot(fv - centroid, cap_normal))
                                if np.max(plane_dists) < 0.003:
                                    cap_face_indices.append(int(fi))
                                    cap_fi_set.add(int(fi))
                                    added += 1
                            print(f"  Anchor {ai}: recovered {added} cap faces (plane+radius, {len(nearby_cap_verts)} verts)")
            # Debug: check pole vertex existence
            if hasattr(self, 'tet_anchor_vertices') and _anchor_positions:
                from scipy.spatial import cKDTree as _cKDTree4
                tet_tree4 = _cKDTree4(closed_vertices.astype(np.float64))
                for ai, apos in _anchor_positions.items():
                    d, ni = tet_tree4.query(apos.astype(np.float64))
                    # Count faces containing this vertex
                    n_touching = sum(1 for f in sim_faces if ni in f)
                    print(f"  Pole check {ai}: nearest vi={ni}, dist={d:.6f}, {n_touching} faces touch it, in_cap_set={ni in tet_cap_set}")

            # Debug: per-anchor cap face counts
            if hasattr(self, 'tet_anchor_vertices') and _anchor_positions:
                from scipy.spatial import cKDTree as _cKDTree3
                tet_tree3 = _cKDTree3(closed_vertices.astype(np.float64))
                cap_fi_set = set(cap_face_indices)
                for ai, apos in _anchor_positions.items():
                    _, ani = tet_tree3.query(apos.astype(np.float64))
                    # Count sim_faces near this anchor (within 0.02)
                    fc = np.mean(closed_vertices[sim_faces].astype(np.float64), axis=1)
                    dists = np.linalg.norm(fc - closed_vertices[ani].astype(np.float64), axis=1)
                    near = np.where(dists < 0.02)[0]
                    n_cap = sum(1 for fi in near if fi in cap_fi_set)
                    n_not = len(near) - n_cap
                    print(f"  Anchor {ai} (vi={ani}): {len(near)} faces nearby, {n_cap} cap, {n_not} not cap")
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

    def _update_tet_draw_positions(self, skip_normals=False):
        """Fast path: update draw arrays from tet_vertices using precomputed index arrays.
        Call this instead of _prepare_tet_draw_arrays when only positions changed (not topology).
        If skip_normals=True, only update vertex positions (normals stay from previous frame)."""
        if not hasattr(self, '_tet_surface_vidx') or self._tet_surface_vidx is None:
            self._prepare_tet_draw_arrays()
            return
        verts = self.tet_vertices
        if verts is None:
            return
        n_verts = len(verts)
        # Update surface verts + normals
        if self._tet_surface_vidx is not None and self._tet_surface_verts is not None:
            if self._tet_surface_vidx.max() >= n_verts:
                self._prepare_tet_draw_arrays()
                return
            self._tet_surface_verts[:] = verts[self._tet_surface_vidx]
            if not skip_normals:
                v = self._tet_surface_verts.reshape(-1, 3, 3)
                normals = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                normals /= norms
                self._tet_surface_normals[:] = np.repeat(normals, 3, axis=0)
        # Update cap verts + normals
        if self._tet_cap_vidx is not None and self._tet_cap_verts is not None:
            if self._tet_cap_vidx.max() >= n_verts:
                self._prepare_tet_draw_arrays()
                return
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
            if self._tet_edge_vidx.max() >= n_verts:
                self._prepare_tet_draw_arrays()
                return
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
