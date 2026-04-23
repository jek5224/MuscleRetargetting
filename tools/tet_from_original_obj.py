#!/usr/bin/env python3
"""Create tet files from original Zygote OBJ muscle meshes, using the same
tetrahedralization pipeline as contour meshes.

Pipeline (matches viewer/tetrahedron_mesh.py:tetrahedralize_contour_mesh):
  1. Load OBJ (NO vertex welding, NO pymeshfix — preserve surface exactly).
  2. Find open boundary loops via face-consistent edge traversal.
  3. Cap each loop with Constrained Delaunay Triangulation (CDT). Cap
     triangles use the original loop vertices (+ centroid if CDT fails on
     every projection).
  4. Tetrahedralize with TetGen — nobisect=True so boundary faces are
     preserved; mindihedral=5 + minratio=2.0 + maxvolume for quality.
  5. Mark cap faces in render_faces so they render green via the existing
     tet_cap_face_indices → _prepare_tet_draw_arrays green-cap convention.

Outputs format-compatible tet npz for bake_layered.py --tet-dir.

Usage:
    python tools/tet_from_original_obj.py --output-dir tet_orig_open
"""
import argparse
import os
import pickle
import sys
from collections import defaultdict, Counter

import numpy as np
import trimesh
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.bake_original_mesh import parse_muscle_xml

SKEL_XML = "data/zygote_skel.xml"
# Match the OBJ paths the viewer loads from .last_loaded_muscles.json. These
# are the hi-res versions that are open at origin/insertion.
OBJ_ROOT = "Zygote_Meshes_251229/Muscle/"
OBJ_SUBDIRS = ['UpLeg', 'LowLeg', 'Foot', 'Hand', 'Torso']
MESH_SCALE = 0.01


# ---------------------------------------------------------------------------
# Boundary loop tracing — face-consistent traversal (handles T-junctions)
# ---------------------------------------------------------------------------
def _find_boundary_loops(vertices, faces):
    edge_count = defaultdict(list)
    for fi, face in enumerate(faces):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge_count[(min(v0, v1), max(v0, v1))].append(fi)
    open_edges = [e for e, fl in edge_count.items() if len(fl) == 1]
    if not open_edges:
        return []
    open_edge_set = set(open_edges)
    edge_to_face = defaultdict(list)
    for fi, f in enumerate(faces):
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            e = (min(a, b), max(a, b))
            if e in open_edge_set:
                edge_to_face[(a, b)].append(fi)
                edge_to_face[(b, a)].append(fi)
    vertex_nbrs = defaultdict(list)
    for a, b in open_edges:
        vertex_nbrs[a].append(b)
        vertex_nbrs[b].append(a)
    visited = set()
    loops = []
    for start in open_edges:
        if start in visited:
            continue
        prev_v = start[0]
        curr_v = start[1]
        loop = [prev_v]
        visited.add(start)
        for _ in range(len(open_edges) + 1):
            loop.append(curr_v)
            incoming = set(edge_to_face.get((prev_v, curr_v), []))
            candidates = []
            for nbr in vertex_nbrs[curr_v]:
                if nbr == prev_v:
                    continue
                key = (min(curr_v, nbr), max(curr_v, nbr))
                if key in visited:
                    continue
                shared = incoming & set(edge_to_face.get((curr_v, nbr), []))
                candidates.append((nbr, key, len(shared) > 0))
            if not candidates:
                break
            candidates.sort(key=lambda x: (not x[2],))
            next_v, next_key, _ = candidates[0]
            visited.add(next_key)
            prev_v = curr_v
            curr_v = next_v
            if curr_v == loop[0]:
                break
        if len(loop) > 1 and loop[-1] == loop[0]:
            loop = loop[:-1]
        if len(loop) >= 3:
            loops.append(loop)
    return loops


# ---------------------------------------------------------------------------
# CDT cap — matches viewer/tetrahedron_mesh.py:_cap_faces
# ---------------------------------------------------------------------------
def _cap_loop(loop, all_vertices):
    import triangle as tr

    def cross2(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    def seg_cross(pts, a, b):
        p0, p1 = pts[a[0]], pts[a[1]]
        p2, p3 = pts[b[0]], pts[b[1]]
        d1 = cross2(p2, p3, p0); d2 = cross2(p2, p3, p1)
        d3 = cross2(p0, p1, p2); d4 = cross2(p0, p1, p3)
        return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))

    def self_intersects(pts, n):
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                if seg_cross(pts, (i, (i + 1) % n), (j, (j + 1) % n)):
                    return True
        return False

    n = len(loop)
    if n < 3:
        return [], None
    if n == 3:
        return [[loop[0], loop[1], loop[2]]], None

    pts_3d = np.array([all_vertices[vi] for vi in loop])
    centroid = pts_3d.mean(axis=0)
    centered = pts_3d - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    projections = [
        ('best-fit', centered @ Vt[:2].T),
        ('XY', pts_3d[:, :2]),
        ('XZ', pts_3d[:, [0, 2]]),
        ('YZ', pts_3d[:, 1:3]),
    ]
    for name, pts_2d in projections:
        if self_intersects(pts_2d, n):
            continue
        segments = np.array([[i, (i + 1) % n] for i in range(n)], dtype=np.int32)
        try:
            result = tr.triangulate({'vertices': pts_2d, 'segments': segments}, 'p')
            faces = [[loop[t[0]], loop[t[1]], loop[t[2]]] for t in result['triangles']]
            if len(faces) >= n - 2:
                return faces, None
        except Exception:
            continue

    # Fan fallback — add centroid vertex
    center_idx = len(all_vertices)
    all_vertices.append(centroid.tolist())
    return [[loop[i], loop[(i + 1) % n], center_idx] for i in range(n)], center_idx


# ---------------------------------------------------------------------------
# TetGen with contour-mesh-compatible parameters
# ---------------------------------------------------------------------------
def _tetrahedralize(vertices, faces):
    """Contour-mesh compatible tetgen. Use subprocess so any tetgen segfault
    surfaces as a RuntimeError instead of killing the parent.
    Delaunay fallback if subprocess fails."""
    import subprocess, sys as _sys, tempfile, json as _json
    rv = (vertices * 1000.0).astype(np.float64)
    rf = faces.astype(np.int32)
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as inp:
        np.savez(inp, v=rv, f=rf)
        inp_path = inp.name
    out_path = inp_path.replace('.npz', '_out.npz')
    script = f'''
import numpy as np, tetgen, trimesh, sys
d = np.load("{inp_path}")
rv = d["v"].astype(np.float64); rf = d["f"].astype(np.int32)
mesh = trimesh.Trimesh(vertices=rv, faces=rf, process=False)
mv = abs(mesh.volume); max_vol = max(mv / 1500.0, 1e-6)
steiner = max(len(rv), 300)
try:
    t = tetgen.TetGen(rv.copy(), rf.copy())
    t.tetrahedralize(order=1, mindihedral=5, minratio=2.0,
                     maxvolume=max_vol, nobisect=True, steinerleft=steiner)
    np.savez("{out_path}", node=np.asarray(t.node), elem=np.asarray(t.elem).astype(np.int32), mode=np.array([1]))
    sys.exit(0)
except Exception: pass
try:
    t = tetgen.TetGen(rv.copy(), rf.copy())
    t.tetrahedralize(quality=False, nobisect=True)
    np.savez("{out_path}", node=np.asarray(t.node), elem=np.asarray(t.elem).astype(np.int32), mode=np.array([2]))
    sys.exit(0)
except Exception: pass
sys.exit(1)
'''
    proc = subprocess.run([_sys.executable, '-c', script], capture_output=True, timeout=90)
    if proc.returncode == 0 and os.path.exists(out_path):
        o = np.load(out_path)
        os.unlink(inp_path); os.unlink(out_path)
        return np.asarray(o['node']) / 1000.0, np.asarray(o['elem'], dtype=np.int32)
    os.unlink(inp_path)
    if os.path.exists(out_path):
        os.unlink(out_path)
    # Delaunay + inside filter — preserves all vertices, no Steiner points.
    from scipy.spatial import Delaunay as _Delaunay
    dl = _Delaunay(rv)
    all_tets = dl.simplices
    mesh_check = trimesh.Trimesh(vertices=rv, faces=rf, process=False)
    mesh_check.fix_normals()
    centers = rv[all_tets].mean(axis=1)
    try:
        inside = mesh_check.contains(centers)
    except Exception:
        inside = np.ones(len(centers), dtype=bool)
    interior = all_tets[inside].astype(np.int32)
    v0 = rv[interior[:, 0]]
    cr = np.cross(rv[interior[:, 1]] - v0, rv[interior[:, 2]] - v0)
    vol = np.einsum('ij,ij->i', cr, rv[interior[:, 3]] - v0)
    neg = vol < 0
    if neg.any():
        interior[neg, 1], interior[neg, 2] = interior[neg, 2].copy(), interior[neg, 1].copy()
    return rv / 1000.0, interior


# ---------------------------------------------------------------------------
# Main per-muscle pipeline
# ---------------------------------------------------------------------------
def process_muscle(muscle_name, obj_path, contour_tet_path, output_path, skel, mesh_info):
    print(f'  [{muscle_name}] start', flush=True)
    all_xml = parse_muscle_xml()
    short = muscle_name.replace('L_', '').replace('R_', '')
    xml_data = all_xml.get(muscle_name) or all_xml.get(short)
    if xml_data is None:
        print(f'    No XML data for {muscle_name}')
        return 0

    # Load OBJ
    print(f'  [{muscle_name}] load OBJ', flush=True)
    mesh = trimesh.load(obj_path, process=False)
    obj_v = np.array(mesh.vertices, dtype=np.float64) * MESH_SCALE
    obj_f = np.array(mesh.faces, dtype=np.int32)
    print(f'  [{muscle_name}] OBJ {len(obj_v)}v {len(obj_f)}f', flush=True)

    # Decimate very dense OBJs — ARAP bake cost scales with total verts across
    # all muscles (~25 × target). Keep quality tetgen feasible and layered ARAP
    # converging within reasonable time.
    target_verts = 1500
    if len(obj_v) > target_verts:
        ratio = target_verts / len(obj_v)
        try:
            simp = trimesh.Trimesh(vertices=obj_v, faces=obj_f, process=False)
            simp = simp.simplify_quadric_decimation(face_count=int(len(obj_f) * ratio))
            obj_v = np.array(simp.vertices, dtype=np.float64)
            obj_f = np.array(simp.faces, dtype=np.int32)
        except Exception as e:
            print(f'    decimate failed: {e} — using raw OBJ')

    # Merge exact-position duplicate verts — OBJ cap seams duplicate boundary
    # verts at the same 3D position. Keeping them separate poisons CDT with
    # degenerate segments; merging keeps surface geometry intact.
    rounded = np.round(obj_v, 6)
    _, uniq_idx, inv = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
    obj_v = obj_v[uniq_idx]
    obj_f = inv[obj_f].astype(np.int32)
    keep_f = np.array([len({int(x) for x in f}) == 3 for f in obj_f])
    obj_f = obj_f[keep_f]

    # Boundary loops
    loops = _find_boundary_loops(obj_v, obj_f)
    if not loops:
        print(f'    No boundary loops (already closed mesh)')
    # Keep only the 2 largest loops — origin and insertion caps. Smaller
    # loops are decimation artifacts that confuse CDT + TetGen.
    loops = sorted([L for L in loops if len(L) >= 5], key=len, reverse=True)[:2]

    # CDT-cap each loop
    closed_v = obj_v.tolist()
    closed_f = obj_f.tolist()
    n_surface = len(obj_f)
    cap_face_indices_pre = []  # indices into closed_f of cap triangles
    anchor_verts = set()
    for loop in loops:
        cap_tris, center_vi = _cap_loop(list(loop), closed_v)
        for tri in cap_tris:
            cap_face_indices_pre.append(len(closed_f))
            closed_f.append(tri)
        for vi in loop:
            anchor_verts.add(int(vi))
        if center_vi is not None:
            anchor_verts.add(int(center_vi))

    closed_v = np.array(closed_v, dtype=np.float64)
    closed_f = np.array(closed_f, dtype=np.int32)
    # CDT cap winding is independent of OBJ winding; align via trimesh.
    _tm_close = trimesh.Trimesh(vertices=closed_v, faces=closed_f, process=False)
    _tm_close.fix_normals()
    closed_f = _tm_close.faces.astype(np.int32)
    # fix_normals may reorder faces — rebuild cap_face_indices via face identity.
    cap_face_set = set(tuple(sorted(int(x) for x in closed_f[fi])) for fi in cap_face_indices_pre
                       if fi < len(closed_f))
    cap_face_indices_pre = [fi for fi, f in enumerate(closed_f)
                            if tuple(sorted(int(x) for x in f)) in cap_face_set]
    n_closed_v = len(closed_v)

    # Tetrahedralize — TetGen preserves boundary vertices (nobisect)
    print(f'  [{muscle_name}] tetgen on {len(closed_v)}v {len(closed_f)}f', flush=True)
    tet_v, tet_e = _tetrahedralize(closed_v, closed_f)
    print(f'  [{muscle_name}] tetgen done {len(tet_v)}v {len(tet_e)}e', flush=True)

    # Remap closed_v indices → tet_v indices via nearest-neighbor.
    # nobisect keeps boundary verts exact, so distances should be ~0.
    kd = cKDTree(tet_v)
    _, c2t = kd.query(closed_v)

    # Orientation: ensure positive tet volume
    v4 = tet_v[tet_e]
    vol = np.einsum('ij,ij->i', v4[:, 1] - v4[:, 0],
                    np.cross(v4[:, 2] - v4[:, 0], v4[:, 3] - v4[:, 0]))
    neg = vol < 0
    if neg.any():
        tet_e[neg, 1], tet_e[neg, 2] = tet_e[neg, 2].copy(), tet_e[neg, 1].copy()

    # Render faces = all closed_f (surface + caps), remapped to tet indices.
    # Drop degenerate tris where remap collapses.
    render_faces_list = []
    cap_face_indices_new = []
    cap_set_pre = set(cap_face_indices_pre)
    for old_fi, face in enumerate(closed_f):
        nf = [int(c2t[int(v)]) for v in face]
        if len(set(nf)) != 3:
            continue
        new_fi = len(render_faces_list)
        render_faces_list.append(nf)
        if old_fi in cap_set_pre:
            cap_face_indices_new.append(new_fi)
    render_faces = np.array(render_faces_list, dtype=np.int32)
    cap_face_indices = np.array(cap_face_indices_new, dtype=np.int32)

    # Sim faces: tet boundary faces (for ARAP/collision).
    fc = Counter()
    face_orient = {}
    for t in tet_e:
        for tri in [(t[0], t[2], t[1]), (t[0], t[1], t[3]),
                    (t[1], t[2], t[3]), (t[0], t[3], t[2])]:
            key = tuple(sorted(int(x) for x in tri))
            fc[key] += 1
            if key not in face_orient:
                face_orient[key] = tri
    sim_faces = np.array([face_orient[k] for k, c in fc.items() if c == 1],
                         dtype=np.int32)

    # Anchor verts remapped to tet space
    tet_anchor_verts = sorted(set(int(c2t[int(vi)]) for vi in anchor_verts))

    # Bone assignment per loop using XML waypoints
    origin_pts = np.vstack([s[2] for s in xml_data]) if xml_data else np.zeros((1, 3))
    insertion_pts = np.vstack([s[3] for s in xml_data]) if xml_data else np.zeros((1, 3))
    origin_bone = xml_data[0][0] if xml_data else None
    insertion_bone = xml_data[0][1] if xml_data else None
    origin_tree = cKDTree(origin_pts)
    insertion_tree = cKDTree(insertion_pts)

    # Fallback: OBJ has no open boundary (e.g. L_Gluteus_Maximus) → no anchors
    # from cap loops. Find tet verts near XML waypoints directly.
    if not tet_anchor_verts:
        tet_kd = cKDTree(tet_v)
        fallback = set()
        for wp in origin_pts:
            _, ids = tet_kd.query(wp, k=min(5, len(tet_v)))
            for ni in np.atleast_1d(ids):
                if np.linalg.norm(tet_v[int(ni)] - wp) < 0.02:
                    fallback.add(int(ni))
        for wp in insertion_pts:
            _, ids = tet_kd.query(wp, k=min(5, len(tet_v)))
            for ni in np.atleast_1d(ids):
                if np.linalg.norm(tet_v[int(ni)] - wp) < 0.02:
                    fallback.add(int(ni))
        tet_anchor_verts = sorted(fallback)
        print(f'    (closed OBJ) XML-waypoint anchors: {len(tet_anchor_verts)}')

    fixed_verts = {}  # tet_vi -> (bone_name, 'origin'/'insertion')
    for tet_vi in tet_anchor_verts:
        pos = tet_v[tet_vi]
        d_o, _ = origin_tree.query(pos)
        d_i, _ = insertion_tree.query(pos)
        end_type = 'origin' if d_o < d_i else 'insertion'
        bone = origin_bone if end_type == 'origin' else insertion_bone
        if bone:
            fixed_verts[tet_vi] = (bone, end_type)

    # Cap attachments (representative per cap)
    cap_attachments = []
    origin_verts = [v for v, (b, t) in fixed_verts.items() if t == 'origin']
    insertion_verts = [v for v, (b, t) in fixed_verts.items() if t == 'insertion']
    if origin_verts:
        cap_attachments.append([origin_verts[0], 0, 0, 0, 0])
    if insertion_verts:
        cap_attachments.append([insertion_verts[0], 0, 1, 0, 0])

    # Copy metadata from contour tet (waypoints, contours, attach_skeleton_names …)
    contour_data = {}
    if contour_tet_path and os.path.exists(contour_tet_path):
        with open(contour_tet_path, 'rb') as f:
            contour_data = pickle.load(f)

    data = {
        'vertices': tet_v.astype(np.float32),
        'tetrahedra': tet_e.astype(np.int32),
        'faces': render_faces,
        'render_faces': render_faces,
        'sim_faces': sim_faces,
        'cap_face_indices': cap_face_indices,
        'anchor_vertices': np.array(tet_anchor_verts, dtype=np.int32),
        'cap_attachments': np.array(cap_attachments, dtype=np.int32)
                           if cap_attachments else np.zeros((0, 5), dtype=np.int32),
        'surface_face_count': len(render_faces),
        'vertex_contour_level': np.full(len(tet_v), -1, dtype=np.int32),
        'contour_to_tet_mapping': [None] * len(tet_v),
        'mvc_weights': None,
        'cap_vertex_types': {v: t for v, (b, t) in fixed_verts.items()},
        'fixed_verts_with_bones': fixed_verts,
        'orig_n_verts': len(tet_v),
    }

    if origin_bone and insertion_bone:
        data['attach_skeleton_names'] = [[origin_bone, insertion_bone]]

    for key in ['attach_skeletons', 'attach_skeletons_sub',
                'waypoints', 'waypoint_bary_coords', 'contours', 'fiber_architecture',
                'bounding_planes', 'draw_contour_stream', 'fiber_sampling_seed',
                'stream_contours', 'stream_bounding_planes', 'stream_groups',
                '_stream_endpoints']:
        if key in contour_data:
            data[key] = contour_data[key]

    if contour_tet_path and os.path.exists(contour_tet_path):
        from tools.remesh_tet_for_collision import build_mapping
        contour_verts = contour_data['vertices']
        data['contour_mapping'] = build_mapping(contour_verts, tet_v, tet_e)
        data['contour_n_verts'] = len(contour_verts)

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f'  {muscle_name}: {len(tet_v)} verts, {len(tet_e)} tets, '
          f'{len(render_faces)} render_faces ({len(cap_face_indices)} cap), '
          f'{len(tet_anchor_verts)} anchors, {len(loops)} cap loops')
    return len(tet_v)


def main():
    ap = argparse.ArgumentParser(description='OBJ → tet via contour-compatible pipeline')
    ap.add_argument('--output-dir', default='tet_orig_open')
    ap.add_argument('--contour-dir', default='tet')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from core.dartHelper import saveSkeletonInfo, buildFromInfo
    skel_info, root_name, _bvh_info, _pd, mesh_info, _smpl = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)
    skel.setPositions(np.zeros(skel.getNumDofs()))

    # Build name → OBJ path map over all subdirs. File name is already
    # <muscle_name>.obj (no UpLegA_/Hip_ prefix), so muscle_name matches
    # contour tet names directly.
    obj_map = {}
    for sub in OBJ_SUBDIRS:
        sub_path = os.path.join(OBJ_ROOT, sub)
        if not os.path.isdir(sub_path):
            continue
        for fname in sorted(os.listdir(sub_path)):
            if not fname.endswith('.obj'):
                continue
            muscle_name = fname[:-len('.obj')]
            obj_map[muscle_name] = os.path.join(sub_path, fname)

    for muscle_name, obj_path in sorted(obj_map.items()):
        contour_path = os.path.join(args.contour_dir, f'{muscle_name}_tet.npz')
        output_path = os.path.join(args.output_dir, f'{muscle_name}_tet.npz')
        if not os.path.exists(contour_path):
            continue
        try:
            process_muscle(muscle_name, obj_path, contour_path, output_path, skel, mesh_info)
        except Exception as e:
            print(f'  {muscle_name}: ERROR {e}')
            import traceback; traceback.print_exc()

    # Fallback: copy contour tet for muscles without OBJ (so bake_layered.py
    # has a tet file for every muscle in the list).
    import shutil
    for fname in sorted(os.listdir(args.contour_dir)):
        if not fname.endswith('_tet.npz'):
            continue
        dst = os.path.join(args.output_dir, fname)
        if not os.path.exists(dst):
            shutil.copy(os.path.join(args.contour_dir, fname), dst)
            print(f'  {fname}: copied from contour (no original mesh)')

    print(f'\nOutput: {args.output_dir}/')


if __name__ == '__main__':
    main()
