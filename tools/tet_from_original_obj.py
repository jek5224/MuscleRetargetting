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
SKEL_OBJ_DIR = "Zygote_Meshes_251229/Skeleton"
MESH_SCALE = 0.01


_bone_kd = None
_bone_names = None

def _load_bone_kd():
    """KD-tree over all skeleton OBJ vertices. Returns (tree, bone_name_per_point)."""
    global _bone_kd, _bone_names
    if _bone_kd is not None:
        return _bone_kd, _bone_names
    import os
    pts = []
    names = []
    for fname in sorted(os.listdir(SKEL_OBJ_DIR)):
        if not fname.endswith('.obj'):
            continue
        # Zygote file naming: "L_Femur.obj" → DART body node "L_Femur0"
        bone_name = fname[:-len('.obj')] + '0'
        m = trimesh.load(os.path.join(SKEL_OBJ_DIR, fname), process=False)
        bv = np.array(m.vertices, dtype=np.float64) * MESH_SCALE
        pts.append(bv)
        names.extend([bone_name] * len(bv))
    pts = np.vstack(pts)
    _bone_kd = cKDTree(pts)
    _bone_names = names
    return _bone_kd, _bone_names


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
    allow_boundary_steiner = (
        os.environ.get('ORIG_TET_BOUNDARY_STEINER', '0') == '1')
    nobisect_literal = 'False' if allow_boundary_steiner else 'True'
    script = f'''
import numpy as np, tetgen, trimesh, sys
d = np.load("{inp_path}")
rv = d["v"].astype(np.float64); rf = d["f"].astype(np.int32)
mesh = trimesh.Trimesh(vertices=rv, faces=rf, process=False)
mv = abs(mesh.volume); max_vol = max(mv / 1500.0, 1e-6)
steiner = max(len(rv), 300)
try:
    t = tetgen.TetGen(rv.copy(), rf.copy())
    # Tighter quality: minratio=1.2 + mindihedral=10 = fewer sliver tets.
    # Sliver tets (low dihedral angle, small volume vs edge length) caused
    # ARAP to produce outlier verts even after boundary preservation.
    t.tetrahedralize(order=1, mindihedral=10, minratio=1.2,
                     maxvolume=max_vol, nobisect={nobisect_literal},
                     steinerleft=steiner)
    np.savez("{out_path}", node=np.asarray(t.node), elem=np.asarray(t.elem).astype(np.int32), mode=np.array([1]))
    sys.exit(0)
except Exception: pass
try:
    t = tetgen.TetGen(rv.copy(), rf.copy())
    t.tetrahedralize(quality=False, nobisect={nobisect_literal})
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

    print(f'  [{muscle_name}] decimate (preserving boundary verts)', flush=True)
    target_verts = 1200
    if len(obj_v) > target_verts:
        # Detect boundary verts BEFORE decimation — these are attachment
        # sites and must be preserved anatomically.
        _ec = Counter()
        for face in obj_f:
            for i in range(3):
                a, b = int(face[i]), int(face[(i+1) % 3])
                _ec[(min(a, b), max(a, b))] += 1
        bnd_verts_raw = set()
        for (a, b), c in _ec.items():
            if c == 1:
                bnd_verts_raw.add(a); bnd_verts_raw.add(b)
        bnd_pos_raw = np.array([obj_v[i] for i in sorted(bnd_verts_raw)])
        print(f'    raw boundary verts: {len(bnd_verts_raw)}', flush=True)

        ratio = target_verts / len(obj_v)
        try:
            simp = trimesh.Trimesh(vertices=obj_v, faces=obj_f, process=False)
            # Lower aggressiveness → quadric decimator tries harder to keep
            # feature edges. Available via fast_simplification backend.
            try:
                simp = simp.simplify_quadric_decimation(
                    face_count=int(len(obj_f) * ratio), aggression=5.0)
            except TypeError:
                simp = simp.simplify_quadric_decimation(
                    face_count=int(len(obj_f) * ratio))
            obj_v = np.array(simp.vertices, dtype=np.float64)
            obj_f = np.array(simp.faces, dtype=np.int32)
        except Exception as e:
            print(f'    decimate failed: {e} — using raw OBJ')

        # Snap decimated boundary verts to nearest raw boundary verts so
        # attachment geometry is preserved exactly at origin/insertion.
        if len(bnd_pos_raw) > 0:
            _ec2 = Counter()
            for face in obj_f:
                for i in range(3):
                    a, b = int(face[i]), int(face[(i+1) % 3])
                    _ec2[(min(a, b), max(a, b))] += 1
            bnd_verts_dec = set()
            for (a, b), c in _ec2.items():
                if c == 1:
                    bnd_verts_dec.add(a); bnd_verts_dec.add(b)
            bnd_verts_dec = sorted(bnd_verts_dec)
            if bnd_verts_dec:
                raw_bnd_tree = cKDTree(bnd_pos_raw)
                dec_pos = obj_v[bnd_verts_dec]
                _, nn_idx = raw_bnd_tree.query(dec_pos)
                snapped = bnd_pos_raw[nn_idx]
                max_snap = np.linalg.norm(snapped - dec_pos, axis=1).max()
                obj_v[bnd_verts_dec] = snapped
                print(f'    snapped {len(bnd_verts_dec)} decimated boundary '
                      f'verts; max_snap={max_snap*1000:.2f}mm', flush=True)

    print(f'  [{muscle_name}] dedup', flush=True)
    rounded = np.round(obj_v, 6)
    _, uniq_idx, inv = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
    obj_v = obj_v[uniq_idx]
    obj_f = inv[obj_f].astype(np.int32)
    keep_f = np.array([len({int(x) for x in f}) == 3 for f in obj_f])
    obj_f = obj_f[keep_f]
    print(f'  [{muscle_name}] loops', flush=True)
    all_loops = _find_boundary_loops(obj_v, obj_f)
    all_loops = [L for L in all_loops if len(L) >= 3]
    print(f'  [{muscle_name}] loops={[len(L) for L in all_loops]}', flush=True)
    if not all_loops:
        print(f'    No boundary loops (already closed mesh)')
    # Split every boundary loop into per-end SUB-CONTOURS. OBJ boundaries
    # often wrap from origin around to insertion as ONE long loop; a single
    # centroid-based label would force the whole loop to one bone. Instead,
    # classify each loop vert by nearest XML waypoint cluster ('origin' or
    # 'insertion') and find contiguous runs of same label → each run is a
    # cap sub-contour. All verts in a sub-contour share the same bone.
    all_loops = [L for L in all_loops if len(L) >= 3]
    # Per-vert mesh-based bone assignment: each boundary vert labels to its
    # nearest skeleton OBJ vertex's bone. Contiguous runs of same bone on a
    # loop → one sub-contour (cap). No XML waypoints used.
    _bone_tree, _bone_names_list = _load_bone_kd()
    _allowed_bones = set()
    if xml_data:
        for (o_bone, i_bone, _, _) in xml_data:
            _allowed_bones.add(o_bone); _allowed_bones.add(i_bone)
    _CAP_BONE_MAX = 0.015

    def _classify_vi(vi):
        if vi >= len(obj_v):
            return None
        p = obj_v[vi]
        d, idx = _bone_tree.query(p)
        if d > _CAP_BONE_MAX:
            return None
        bone = _bone_names_list[idx]
        if _allowed_bones and bone not in _allowed_bones:
            return None
        end_type = 'origin' if bone == (xml_data[0][0] if xml_data else '') else 'insertion'
        return (end_type, bone)

    named_cap_loops = []
    named_cap_end_types = []
    other_loops = []
    for loop in all_loops:
        labels = [_classify_vi(vi) for vi in loop]
        n = len(loop)
        if any(lab is not None for lab in labels):
            start = 0
            for i in range(n):
                if labels[i] != labels[i - 1]:
                    start = i
                    break
            rot_loop = loop[start:] + loop[:start]
            rot_labels = labels[start:] + labels[:start]
        else:
            rot_loop, rot_labels = loop, labels
        i = 0
        made_any = False
        while i < len(rot_loop):
            lab = rot_labels[i]
            j = i
            while j < len(rot_loop) and rot_labels[j] == lab:
                j += 1
            run = rot_loop[i:j]
            if lab is not None and len(run) >= 3:
                named_cap_loops.append(run)
                named_cap_end_types.append(lab)
                made_any = True
            i = j
        if not made_any:
            other_loops.append(loop)

    # Close every full boundary loop with a single centroid fan. Each fan
    # centroid also gets anchored to the dominant bone of its loop's
    # sub-contours — so when AM's loop runs around from Os_Coxae (top) to
    # Femur (bottom), the centroid follows whichever bone has more anchors
    # in that loop, preventing arbitrary drift of the fan star vertex.
    closed_v = obj_v.tolist()
    closed_f = obj_f.tolist()
    cap_face_indices_pre = []
    anchor_verts = set()
    extra_anchor_centers = {}  # center_idx -> (bone, end_type)
    for arc in named_cap_loops:
        for vi in arc:
            anchor_verts.add(int(vi))
    # Build per-loop dominant bone from sub-contour (arc) assignments.
    loop_dominant = {}  # id(loop) -> (bone, end_type)
    from collections import Counter as _Counter
    for li, loop in enumerate(all_loops):
        loop_set = set(loop)
        votes = _Counter()
        for arc, (et, bone) in zip(named_cap_loops, named_cap_end_types):
            if arc and arc[0] in loop_set:
                votes[(bone, et)] += len(arc)
        if votes:
            loop_dominant[li] = votes.most_common(1)[0][0]
    for li, loop in enumerate(all_loops):
        n = len(loop)
        if n < 3:
            continue
        centroid = np.mean([closed_v[vi] for vi in loop], axis=0)
        center_idx = len(closed_v)
        closed_v.append(centroid.tolist())
        dom = loop_dominant.get(li)
        if dom:
            extra_anchor_centers[center_idx] = dom
            anchor_verts.add(center_idx)
        for i in range(n):
            fi = len(closed_f)
            closed_f.append([loop[i], loop[(i + 1) % n], center_idx])
            cap_face_indices_pre.append(fi)

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
    # Attachment loops and fan centers below index this pre-remesh surface.
    # Keep it explicitly: an isotropic boundary remesh changes both vertex
    # count and indexing, so using its c2t table for these indices is invalid.
    attachment_source_v = closed_v.copy()

    isotropic_boundary = (
        os.environ.get('ORIG_TET_ISOTROPIC_BOUNDARY', '0') == '1')
    if isotropic_boundary:
        from viewer.zygote_mesh_ui import _isotropic_voxel_surface_for_tet
        target_tets = int(os.environ.get(
            'ORIG_TET_TARGET_TETS', '12000'))
        closed_v, closed_f, _ = _isotropic_voxel_surface_for_tet(
            muscle_name, closed_v, closed_f, target_tets)
        closed_v = np.asarray(closed_v, dtype=np.float64)
        closed_f = np.asarray(closed_f, dtype=np.int32)

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

    # Prune tet_v to only verts actually used by tet elements AND to only
    # tet elements whose verts each belong to ≥2 tets (so no vert is a
    # singleton corner of exactly 1 tet — those produce ARAP outliers).
    # Orphan (0-tet) + singleton (1-tet) removal via iterative pass.
    for _pass in range(3):
        used_mask = np.zeros(len(tet_v), dtype=bool)
        used_mask[tet_e.flatten()] = True
        # Count tet membership per vert
        tet_count = np.zeros(len(tet_v), dtype=np.int32)
        for tet in tet_e:
            for v in tet:
                tet_count[int(v)] += 1
        singleton = (tet_count == 1)
        if not singleton.any() and used_mask.all():
            break
        # Drop tets containing any singleton vert
        keep_tet = np.ones(len(tet_e), dtype=bool)
        if singleton.any():
            for ti, tet in enumerate(tet_e):
                if any(singleton[int(v)] for v in tet):
                    keep_tet[ti] = False
            tet_e = tet_e[keep_tet]
        used_mask = np.zeros(len(tet_v), dtype=bool)
        used_mask[tet_e.flatten()] = True
        n_drop = int((~used_mask).sum())
        if n_drop:
            old_to_new = np.full(len(tet_v), -1, dtype=np.int32)
            old_to_new[used_mask] = np.arange(int(used_mask.sum()), dtype=np.int32)
            tet_v = tet_v[used_mask]
            tet_e = old_to_new[tet_e]
    # Rebuild c2t with pruned tet_v.
    kd = cKDTree(tet_v)
    _, c2t = kd.query(closed_v)
    _, attachment_c2t = kd.query(attachment_source_v)
    print(f'    pruned to {len(tet_v)}v {len(tet_e)}e (no orphans, no singleton-tet verts)', flush=True)

    # Render faces = all closed_f (surface + caps), remapped to pruned tet
    # indices. Every render-face vert now belongs to a tet (pruning above
    # guarantees it), so ARAP moves the whole surface.
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
    tet_anchor_verts = sorted(
        set(int(attachment_c2t[int(vi)]) for vi in anchor_verts))

    # Bone assignment per loop using XML waypoints
    origin_pts = np.vstack([s[2] for s in xml_data]) if xml_data else np.zeros((1, 3))
    insertion_pts = np.vstack([s[3] for s in xml_data]) if xml_data else np.zeros((1, 3))
    origin_bone = xml_data[0][0] if xml_data else None
    insertion_bone = xml_data[0][1] if xml_data else None
    origin_tree = cKDTree(origin_pts)
    insertion_tree = cKDTree(insertion_pts)

    # Apply per-sub-contour classification. named_cap_end_types holds
    # (end_type, bone_name) from the per-waypoint lookup — supports
    # multi-head muscles with different bones per origin stream.
    fixed_verts = {}
    anchor_tet_set = set()
    if isotropic_boundary and len(sim_faces):
        # Transfer each source attachment as a connected surface patch. A
        # pointwise nearest-vertex remap leaves isolated pins on a different
        # triangulation, which produces the visible spikes at pes insertions.
        surface = np.unique(sim_faces)
        surface_points = tet_v[surface]
        surface_lookup = {int(v): i for i, v in enumerate(surface)}
        adjacency = [set() for _ in range(len(surface))]
        surface_edges = set()
        for face in sim_faces:
            for a, b in ((face[0], face[1]), (face[1], face[2]),
                         (face[2], face[0])):
                ia, ib = surface_lookup[int(a)], surface_lookup[int(b)]
                adjacency[ia].add(ib)
                adjacency[ib].add(ia)
                surface_edges.add(tuple(sorted((int(a), int(b)))))
        edge_length = np.asarray([
            np.linalg.norm(tet_v[a] - tet_v[b])
            for a, b in surface_edges])
        patch_radius = max(0.004, 2.25 * float(np.median(edge_length)))

        # XML waypoints are the authoritative anatomical endpoints. Boundary
        # loops may also include cut seams along the belly; using those loops
        # as patch seeds made Sartorius' insertion extend 13 cm proximally.
        grouped_sources = {}
        for xml_origin_bone, xml_insertion_bone, origin_point, insertion_point in xml_data:
            if xml_origin_bone:
                grouped_sources.setdefault(
                    (xml_origin_bone, 'origin'), []).append(
                        np.asarray(origin_point, dtype=np.float64).reshape(-1, 3))
            if xml_insertion_bone:
                grouped_sources.setdefault(
                    (xml_insertion_bone, 'insertion'), []).append(
                        np.asarray(insertion_point, dtype=np.float64).reshape(-1, 3))
        group_items = [
            (key, np.vstack(parts))
            for key, parts in grouped_sources.items() if parts]
        group_distance = []
        for _, source_points in group_items:
            distance, _ = cKDTree(source_points).query(surface_points)
            group_distance.append(distance)
        if group_distance:
            group_distance = np.stack(group_distance, axis=1)
            nearest_group = np.argmin(group_distance, axis=1)
            for group_i, ((bone, end_type), source_points) in enumerate(
                    group_items):
                # XML often contains a dense fiber grid. Using every fiber
                # endpoint as an independent seed over-constrains broad
                # muscles (notably adductor magnus), fixing much of the
                # belly. Collapse endpoints into 4 mm spatial cells first;
                # the resulting patch represents attachment *area*, not
                # fiber count.
                cell = np.floor(source_points / 0.004).astype(np.int64)
                _, representative = np.unique(
                    cell, axis=0, return_index=True)
                seed_points = source_points[np.sort(representative)]
                _, seed_local = cKDTree(surface_points).query(seed_points)
                selected = set(int(i) for i in np.atleast_1d(seed_local))
                frontier = set(selected)
                # One surface ring makes a connected patch while avoiding
                # the former two-ring expansion far into the muscle body.
                for _ in range(1):
                    grown = set()
                    for local_i in frontier:
                        grown.update(adjacency[local_i])
                    grown = {
                        i for i in grown
                        if group_distance[i, group_i] <= patch_radius
                        and nearest_group[i] == group_i
                    }
                    grown -= selected
                    selected.update(grown)
                    frontier = grown
                for local_i in selected:
                    tet_vi = int(surface[local_i])
                    fixed_verts[tet_vi] = (bone, end_type)
                    anchor_tet_set.add(tet_vi)
        print(f'    connected attachment patches: {len(anchor_tet_set)} '
              f'verts, radius={patch_radius*1000:.2f}mm')
    else:
        for arc, (end_type, bone) in zip(
                named_cap_loops, named_cap_end_types):
            tet_arc_verts = sorted(
                {int(attachment_c2t[int(vi)]) for vi in arc})
            if not bone:
                continue
            for vi in tet_arc_verts:
                fixed_verts[vi] = (bone, end_type)
                anchor_tet_set.add(vi)
        # Also anchor fan-centroid verts to their loop's dominant bone.
        for center_idx_obj, (bone, end_type) in extra_anchor_centers.items():
            tet_vi = int(attachment_c2t[int(center_idx_obj)])
            if bone:
                fixed_verts[tet_vi] = (bone, end_type)
                anchor_tet_set.add(tet_vi)

    # If no cap loops classified (all seams, or closed-mesh OBJ with no
    # boundaries): fall back to XML-waypoint-nearest tet verts. Each tet
    # vert gets the bone of the waypoint cluster it's closest to.
    if not anchor_tet_set:
        tet_kd = cKDTree(tet_v)
        if origin_mean is not None and origin_bone:
            for wp in origin_pts:
                _, ids = tet_kd.query(wp, k=min(5, len(tet_v)))
                for ni in np.atleast_1d(ids):
                    if np.linalg.norm(tet_v[int(ni)] - wp) < 0.02:
                        fixed_verts[int(ni)] = (origin_bone, 'origin')
                        anchor_tet_set.add(int(ni))
        if insertion_mean is not None and insertion_bone:
            for wp in insertion_pts:
                _, ids = tet_kd.query(wp, k=min(5, len(tet_v)))
                for ni in np.atleast_1d(ids):
                    if np.linalg.norm(tet_v[int(ni)] - wp) < 0.02:
                        fixed_verts[int(ni)] = (insertion_bone, 'insertion')
                        anchor_tet_set.add(int(ni))
        print(f'    (no cap loops classified) XML fallback anchors: {len(anchor_tet_set)}')

    tet_anchor_verts = sorted(anchor_tet_set)

    cap_attachments = []
    origin_verts_list = [v for v, (b, t) in fixed_verts.items() if t == 'origin']
    insertion_verts_list = [v for v, (b, t) in fixed_verts.items() if t == 'insertion']
    if origin_verts_list:
        cap_attachments.append([origin_verts_list[0], 0, 0, 0, 0])
    if insertion_verts_list:
        cap_attachments.append([insertion_verts_list[0], 0, 1, 0, 0])

    anchor_bone_map = {int(vi): bone for vi, (bone, _) in fixed_verts.items()}

    # cap_face_indices = faces where all 3 verts are anchors with same bone
    # AND max edge ≤30mm. Excludes body-spanning CDT fan tris from green
    # rendering but KEEPS them in render_faces so mesh stays watertight.
    anchor_set = set(anchor_bone_map.keys())
    cap_list = []
    for fi, f in enumerate(render_faces):
        fa, fb, fc = int(f[0]), int(f[1]), int(f[2])
        if fa not in anchor_set or fb not in anchor_set or fc not in anchor_set:
            continue
        if len({anchor_bone_map[fa], anchor_bone_map[fb], anchor_bone_map[fc]}) != 1:
            continue
        p = tet_v[[fa, fb, fc]]
        max_edge = max(np.linalg.norm(p[0] - p[1]),
                       np.linalg.norm(p[1] - p[2]),
                       np.linalg.norm(p[2] - p[0]))
        if max_edge > 0.030:
            continue
        cap_list.append(fi)
    cap_face_indices = np.array(cap_list, dtype=np.int32)

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
        'anchor_bone_map': anchor_bone_map,
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
          f'{len(tet_anchor_verts)} anchors, {len(named_cap_loops)} named + {len(other_loops)} filler loops')
    return len(tet_v)


def main():
    ap = argparse.ArgumentParser(description='OBJ → tet via contour-compatible pipeline')
    ap.add_argument('--output-dir', default='tet_orig_open')
    ap.add_argument('--contour-dir', default='tet')
    ap.add_argument('--muscles-json', default=None,
                    help='Optional loaded-muscle JSON used to restrict output.')
    ap.add_argument('--allow-boundary-steiner', action='store_true',
                    help='Allow TetGen to subdivide poor boundary triangles.')
    ap.add_argument('--isotropic-boundary', action='store_true',
                    help='Voxel-remesh the simulation boundary before TetGen.')
    ap.add_argument('--target-tets', type=int, default=12000,
                    help='Approximate tet count for isotropic-boundary mode.')
    args = ap.parse_args()
    if args.allow_boundary_steiner:
        os.environ['ORIG_TET_BOUNDARY_STEINER'] = '1'
    if args.isotropic_boundary:
        os.environ['ORIG_TET_ISOTROPIC_BOUNDARY'] = '1'
        os.environ['ORIG_TET_TARGET_TETS'] = str(args.target_tets)

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

    selected = None
    if args.muscles_json:
        import json
        with open(args.muscles_json) as f:
            selected = {entry['name'] for entry in json.load(f)}

    for muscle_name, obj_path in sorted(obj_map.items()):
        if selected is not None and muscle_name not in selected:
            continue
        # Prefer contour_backup (true contour tet) over tet/ (may be swapped
        # to original mesh from a previous viewer session).
        contour_backup = os.path.join(args.contour_dir, f'{muscle_name}_tet.npz.contour_backup')
        contour_main = os.path.join(args.contour_dir, f'{muscle_name}_tet.npz')
        contour_path = contour_backup if os.path.exists(contour_backup) else contour_main
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
        if selected is not None and fname[:-len('_tet.npz')] not in selected:
            continue
        dst = os.path.join(args.output_dir, fname)
        if not os.path.exists(dst):
            shutil.copy(os.path.join(args.contour_dir, fname), dst)
            print(f'  {fname}: copied from contour (no original mesh)')

    print(f'\nOutput: {args.output_dir}/')


if __name__ == '__main__':
    main()
