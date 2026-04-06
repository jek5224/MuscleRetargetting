#!/usr/bin/env python3
"""Bake muscle deformation with pyuipc IPC contact.

Loads skeleton, applies BVH motion frame by frame, drives muscle
vertices via multi-bone LBS, runs pyuipc with IPC contact.
Saves per-frame deformed positions to npz chunks.

Usage (on A6000 server):
    python tools/bake_pyuipc.py --bvh data/motion/walk.bvh --frames 0-131 --sides LR
"""
import argparse
import gc
import os
import pickle
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH

from uipc import view, Animation
import uipc.builtin as builtin
from uipc.core import Engine, World, Scene
from uipc.geometry import tetmesh, label_surface, label_triangle_orient
from uipc.constitution import StableNeoHookean, SoftPositionConstraint, ElasticModuli
from uipc.unit import kPa, GPa, MPa

SKEL_XML = "data/zygote_skel.xml"
SCALE = 1000.0  # m → mm

UPLEG_MUSCLES = [
    "Adductor_Brevis", "Adductor_Longus", "Adductor_Magnus",
    "Biceps_Femoris", "Gluteus_Maximus", "Gluteus_Medius",
    "Gluteus_Minimus", "Gracilis", "Iliacus",
    "Inferior_Gemellus", "Obturator_Externus", "Obturator_Internus",
    "Pectineus", "Piriformis", "Popliteus",
    "Quadratus_Femoris", "Rectus_Femoris", "Sartorius",
    "Semimembranosus", "Semitendinosus", "Superior_Gemellus",
    "Tensor_Fascia_Lata", "Vastus_Intermedius", "Vastus_Lateralis",
    "Vastus_Medialis",
]


def _find_body(skel, name):
    """Find DART body node by name, trying exact match then fuzzy."""
    bn = skel.getBodyNode(name)
    if bn is not None:
        return bn, name
    bn = skel.getBodyNode(name + "0")
    if bn is not None:
        return bn, name + "0"
    norm = name.lower().replace('_', '')
    for bi in range(skel.getNumBodyNodes()):
        b = skel.getBodyNode(bi)
        bn_norm = b.getName().lower().replace('_', '')
        if norm in bn_norm or bn_norm in norm:
            return b, b.getName()
    return None, name


def load_skeleton():
    skel_info, root_name, bvh_info, _pd, mesh_info, _smpl = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)
    return skel, bvh_info, mesh_info


def load_muscle(tet_dir, name):
    path = os.path.join(tet_dir, f"{name}_tet.npz")
    if not os.path.exists(path):
        return None

    with open(path, 'rb') as f:
        data = pickle.load(f)

    verts = data['vertices'].astype(np.float64) * SCALE
    tets = data['tetrahedra'].astype(np.int32)

    # Fix tet orientation
    v0 = verts[tets[:, 0]]
    cross = np.cross(verts[tets[:, 1]] - v0, verts[tets[:, 2]] - v0)
    vol = np.einsum('ij,ij->i', cross, verts[tets[:, 3]] - v0) / 6.0
    neg = vol < 0
    if np.any(neg):
        tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy()

    # Remove degenerate tets
    v0 = verts[tets[:, 0]]
    cross = np.cross(verts[tets[:, 1]] - v0, verts[tets[:, 2]] - v0)
    vol = np.einsum('ij,ij->i', cross, verts[tets[:, 3]] - v0) / 6.0
    good = vol > 0.01
    if not np.all(good):
        tets = tets[good]

    # Collect fixed vertices from caps
    cap_faces = data.get('cap_face_indices', [])
    fixed_verts = set()
    sim_faces = data.get('sim_faces', data.get('faces'))
    if sim_faces is not None and len(cap_faces) > 0:
        for fi in cap_faces:
            if fi < len(sim_faces):
                for vi in sim_faces[fi]:
                    fixed_verts.add(int(vi))
    anchors = data.get('anchor_vertices', np.array([], dtype=np.int32))
    for vi in anchors:
        fixed_verts.add(int(vi))

    # Remove unused vertices
    used = np.unique(tets.ravel())
    remap = None
    if len(used) < len(verts):
        remap = np.full(len(verts), -1, dtype=np.int32)
        remap[used] = np.arange(len(used), dtype=np.int32)
        verts = verts[used]
        tets = remap[tets]
        fixed_verts = {int(remap[vi]) for vi in fixed_verts if vi < len(remap) and remap[vi] >= 0}

    # Get skeleton attachment info
    cap_attachments = data.get('cap_attachments', [])
    attach_skel_names = data.get('attach_skeleton_names', [])

    return {
        'name': name,
        'vertices': verts,
        'rest_vertices': verts.copy(),
        'tetrahedra': tets,
        'fixed_vertices': sorted(fixed_verts),
        'cap_attachments': cap_attachments,
        'attach_skeleton_names': attach_skel_names,
        'cap_face_indices': cap_faces,
        'sim_faces': sim_faces,
        'remap': remap,
    }


def compute_multibone_lbs(muscle, skel):
    """Compute multi-bone LBS bindings using cap_attachments.

    Cap anchor vertices: rigidly follow their assigned bone.
    Cap face vertices: follow nearest anchor's bone.
    Interior vertices: distance-weighted blend of all muscle bones.

    Returns per-vertex list of [(bone_name, weight, R0, t0, rest_pos), ...].
    """
    rest_verts = muscle['rest_vertices']
    n_verts = len(rest_verts)
    attach_names = muscle.get('attach_skeleton_names', [])
    cap_att = muscle.get('cap_attachments', [])
    cap_faces_idx = muscle.get('cap_face_indices', [])
    sim_faces = muscle.get('sim_faces')
    remap = muscle.get('remap')

    if len(attach_names) == 0:
        return None

    # Collect all unique bones from all streams
    skel.setPositions(np.zeros(skel.getNumDofs()))
    bone_info = {}  # bone_name -> {node, R0, t0}
    for stream_names in attach_names:
        for raw_name in stream_names:
            if raw_name in bone_info:
                continue
            node, resolved = _find_body(skel, raw_name)
            if node is None:
                continue
            R0 = node.getWorldTransform().rotation()
            t0 = node.getWorldTransform().translation() * SCALE
            bone_info[raw_name] = {'node_name': resolved, 'R0': R0, 't0': t0}

    if len(bone_info) == 0:
        return None

    bone_names = list(bone_info.keys())

    # Step 1: Assign cap anchor vertices to their bone
    per_vertex = {}  # vi -> [(bone_name, weight)]
    anchor_bone = {}  # vi -> bone_name (for cap face assignment)

    if len(cap_att) > 0:
        for row in cap_att:
            orig_vi = int(row[0])
            stream_idx = int(row[1])
            end_idx = int(row[2])  # 0=origin, 1=insertion
            if stream_idx < len(attach_names) and end_idx < len(attach_names[stream_idx]):
                bone_name = attach_names[stream_idx][end_idx]
                # Remap vertex index
                vi = orig_vi
                if remap is not None:
                    vi = int(remap[orig_vi]) if orig_vi < len(remap) else -1
                if vi >= 0 and vi < n_verts and bone_name in bone_info:
                    per_vertex[vi] = [(bone_name, 1.0)]
                    anchor_bone[vi] = bone_name

    # Step 2: Cap face vertices -> same bone as nearest anchor
    cap_face_verts = set()
    if sim_faces is not None:
        for fi in cap_faces_idx:
            if fi < len(sim_faces):
                for vi_orig in sim_faces[fi]:
                    vi = int(vi_orig)
                    if remap is not None:
                        vi = int(remap[vi]) if vi < len(remap) else -1
                    if vi >= 0 and vi < n_verts:
                        cap_face_verts.add(vi)

    if len(anchor_bone) > 0:
        anchor_vis = list(anchor_bone.keys())
        anchor_pos = rest_verts[anchor_vis]
        anchor_tree = cKDTree(anchor_pos)
        for vi in cap_face_verts:
            if vi in per_vertex:
                continue
            _, idx = anchor_tree.query(rest_verts[vi])
            nearest_anchor = anchor_vis[idx]
            per_vertex[vi] = [(anchor_bone[nearest_anchor], 1.0)]

    # Step 3: Interior vertices — heat-diffused weights on tet mesh
    # Build adjacency from tetrahedra
    tets = muscle['tetrahedra']
    adj = [set() for _ in range(n_verts)]
    for t in tets:
        for i in range(4):
            for j in range(i + 1, 4):
                adj[t[i]].add(t[j])
                adj[t[j]].add(t[i])

    # Build per-bone seed sets (cap vertices assigned to each bone)
    bone_cap_verts = {}
    for vi, bname in anchor_bone.items():
        bone_cap_verts.setdefault(bname, []).append(vi)
    for vi in cap_face_verts:
        if vi in per_vertex and len(per_vertex[vi]) == 1:
            bname = per_vertex[vi][0][0]
            bone_cap_verts.setdefault(bname, []).append(vi)

    # Fixed vertices (caps) — don't change during diffusion
    fixed = set(per_vertex.keys())

    # Heat diffusion: for each bone, init heat=1 at its caps, 0 elsewhere
    # then iteratively average with neighbors (Jacobi iteration)
    n_iters = 50
    bone_heat = {}
    for bname in bone_names:
        if bname not in bone_cap_verts:
            continue
        heat = np.zeros(n_verts)
        for vi in bone_cap_verts.get(bname, []):
            heat[vi] = 1.0
        # Also set cap face vertices for this bone
        for vi in cap_face_verts:
            if vi in per_vertex and per_vertex[vi][0][0] == bname:
                heat[vi] = 1.0
        bone_heat[bname] = heat

    for _ in range(n_iters):
        for bname, heat in bone_heat.items():
            new_heat = heat.copy()
            for vi in range(n_verts):
                if vi in fixed or len(adj[vi]) == 0:
                    continue
                new_heat[vi] = np.mean([heat[ni] for ni in adj[vi]])
            bone_heat[bname] = new_heat

    # Assign normalized weights to interior vertices
    for vi in range(n_verts):
        if vi in per_vertex:
            continue
        weights = []
        for bname, heat in bone_heat.items():
            if heat[vi] > 1e-8:
                weights.append((bname, heat[vi]))
        if len(weights) == 0:
            # Fallback: nearest bone by joint center
            best_dist = float('inf')
            best_bone = bone_names[0]
            for bname in bone_names:
                d = np.linalg.norm(rest_verts[vi] - bone_info[bname]['t0'])
                if d < best_dist:
                    best_dist = d
                    best_bone = bname
            weights = [(best_bone, 1.0)]
        total = sum(w for _, w in weights)
        per_vertex[vi] = [(b, w / total) for b, w in weights]

    # Build final bindings
    bindings = []
    for vi in range(n_verts):
        if vi not in per_vertex:
            per_vertex[vi] = [(bone_names[0], 1.0)]
        bone_weights = []
        for bname, w in per_vertex[vi]:
            if bname not in bone_info:
                continue
            bi = bone_info[bname]
            bone_weights.append((bi['node_name'], w, bi['R0'], bi['t0']))
        bindings.append((rest_verts[vi].copy(), bone_weights))

    return bindings


def compute_lbs_positions(bindings, skel, n_verts):
    """Compute multi-bone LBS-deformed positions at current skeleton pose."""
    if bindings is None:
        return None

    positions = np.zeros((n_verts, 3))
    for vi, (rest_pos, bone_weights) in enumerate(bindings):
        if len(bone_weights) == 0:
            positions[vi] = rest_pos
            continue
        pos = np.zeros(3)
        for node_name, w, R0, t0 in bone_weights:
            node = skel.getBodyNode(node_name)
            if node is None:
                pos += w * rest_pos
                continue
            R1 = node.getWorldTransform().rotation()
            t1 = node.getWorldTransform().translation() * SCALE
            pos += w * (R1 @ (R0.T @ (rest_pos - t0)) + t1)
        positions[vi] = pos

    return positions


def extract_surface_verts(muscle):
    """Get set of surface vertex indices from tetrahedra surface faces."""
    tets = muscle['tetrahedra']
    # Extract surface by finding faces that appear only once
    from collections import Counter
    face_count = Counter()
    tet_faces = []
    for ti, t in enumerate(tets):
        faces = [(t[0],t[1],t[2]), (t[0],t[1],t[3]),
                 (t[0],t[2],t[3]), (t[1],t[2],t[3])]
        for f in faces:
            key = tuple(sorted(f))
            face_count[key] += 1
            tet_faces.append(key)
    surface_verts = set()
    for face, count in face_count.items():
        if count == 1:
            surface_verts.update(face)
    return sorted(surface_verts)


def resolve_collisions(muscles, all_pos_frame, d_min=0.5, n_iters=3):
    """Push apart overlapping surface vertices between different muscles.

    For each pair of close surface vertices from different muscles,
    push them apart along the line connecting them.
    d_min: minimum allowed distance in mm.
    """
    # Build global surface vertex array
    global_pts = []
    global_muscle_id = []
    global_local_idx = []  # (muscle_index, vertex_index)
    for mi, m in enumerate(muscles):
        pos = all_pos_frame[m['name']]
        surf = m.get('_surface_verts')
        if surf is None:
            surf = extract_surface_verts(m)
            m['_surface_verts'] = surf
        for vi in surf:
            if vi < len(pos):
                global_pts.append(pos[vi])
                global_muscle_id.append(mi)
                global_local_idx.append((mi, vi))

    if len(global_pts) == 0:
        return

    global_pts = np.array(global_pts, dtype=np.float64)
    global_muscle_id = np.array(global_muscle_id)

    for _ in range(n_iters):
        tree = cKDTree(global_pts)
        pairs = tree.query_pairs(r=d_min)
        n_pushed = 0
        for i, j in pairs:
            if global_muscle_id[i] == global_muscle_id[j]:
                continue  # same muscle
            diff = global_pts[j] - global_pts[i]
            dist = np.linalg.norm(diff)
            if dist < 1e-8:
                diff = np.array([0.0, 1.0, 0.0])
                dist = 1.0
            push = (d_min - dist) * 0.5 * diff / dist
            global_pts[i] -= push
            global_pts[j] += push
            n_pushed += 1

        if n_pushed == 0:
            break

    # Write back
    for gi, (mi, vi) in enumerate(global_local_idx):
        m = muscles[mi]
        all_pos_frame[m['name']][vi] = global_pts[gi].astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bvh', required=True, help='BVH motion file')
    parser.add_argument('--frames', default='0-30', help='Frame range (e.g. 0-100)')
    parser.add_argument('--tet-dir', default='tet_sim')
    parser.add_argument('--output-dir', default='bake_pyuipc')
    parser.add_argument('--sides', default='L', help='L, R, or LR')
    parser.add_argument('--no-contact', action='store_true', help='Disable IPC contact')
    args = parser.parse_args()

    # Parse frame range
    parts = args.frames.split('-')
    start_frame = int(parts[0])
    end_frame = int(parts[1]) if len(parts) > 1 else start_frame

    # Load skeleton and BVH
    print("[1] Loading skeleton...")
    skel, bvh_info, mesh_info = load_skeleton()
    print(f"    DOFs: {skel.getNumDofs()}, Bodies: {skel.getNumBodyNodes()}")

    print(f"[2] Loading BVH: {args.bvh}")
    motion_bvh = MyBVH(args.bvh, bvh_info, skel=skel)
    n_frames = len(motion_bvh.mocap_refs)
    end_frame = min(end_frame, n_frames - 1)
    print(f"    Frames: {n_frames}, using {start_frame}-{end_frame}")

    # Load muscles
    print("[3] Loading muscles...")
    muscles = []
    for side in args.sides:
        for muscle_name in UPLEG_MUSCLES:
            name = f"{side}_{muscle_name}"
            data = load_muscle(args.tet_dir, name)
            if data is not None:
                muscles.append(data)

    total_tets = sum(len(m['tetrahedra']) for m in muscles)
    total_verts = sum(len(m['vertices']) for m in muscles)
    print(f"    {len(muscles)} muscles: {total_verts} verts, {total_tets} tets")

    # Compute multi-bone LBS bindings at rest pose
    print("[4] Computing multi-bone LBS bindings...")
    for m in muscles:
        m['lbs_bindings'] = compute_multibone_lbs(m, skel)
        if m['lbs_bindings'] is not None:
            # Count unique bones
            bones_used = set()
            for _, bw in m['lbs_bindings']:
                for name, *_ in bw:
                    bones_used.add(name)
            print(f"    {m['name']}: {len(m['vertices'])}v, bones={bones_used}")
        else:
            print(f"    {m['name']}: WARNING no LBS bindings!")
        m['aim_targets'] = m['rest_vertices'].copy()

    use_contact = not args.no_contact

    if use_contact:
        # Pre-position at frame-0 LBS so contact doesn't fight the huge
        # rest→walk displacement.  Re-orient tets for the deformed mesh.
        print("[5] Pre-positioning muscles at frame-0 LBS (contact mode)...")
        pose0 = motion_bvh.mocap_refs[start_frame].copy()
        skel.setPositions(pose0)
        for m in muscles:
            if m['lbs_bindings'] is not None:
                lbs0 = compute_lbs_positions(m['lbs_bindings'], skel, len(m['vertices']))
                m['vertices'] = lbs0
                m['aim_targets'][:] = lbs0
                tets = m['tetrahedra']
                v0 = lbs0[tets[:, 0]]
                cross = np.cross(lbs0[tets[:, 1]] - v0, lbs0[tets[:, 2]] - v0)
                vol = np.einsum('ij,ij->i', cross, lbs0[tets[:, 3]] - v0) / 6.0
                neg = vol < 0
                if np.any(neg):
                    tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy()
                v0 = lbs0[tets[:, 0]]
                cross = np.cross(lbs0[tets[:, 1]] - v0, lbs0[tets[:, 2]] - v0)
                vol = np.einsum('ij,ij->i', cross, lbs0[tets[:, 3]] - v0) / 6.0
                good = vol > 0.001
                if not np.all(good):
                    n_removed = int(np.sum(~good))
                    tets = tets[good]
                    m['tetrahedra'] = tets
                    print(f"    {m['name']}: removed {n_removed} degenerate tets")
        total_tets = sum(len(m['tetrahedra']) for m in muscles)
        print(f"    Total tets after re-orient: {total_tets}")

    # Setup pyuipc
    print(f"[6] Setting up pyuipc (contact={'ON' if use_contact else 'OFF'})...")
    engine = Engine('cuda')
    world = World(engine)

    config = Scene.default_config()
    config['dt'] = 0.01
    config['gravity'] = [[0.0], [0.0], [0.0]]
    if use_contact:
        config['contact'] = {
            'enable': True,
            'friction': {'enable': False},
            'd_hat': 0.1,  # 0.1mm — only resolve near-penetrations
        }
    else:
        config['contact'] = {'enable': False}
    config['sanity_check'] = {'enable': False}
    config['newton'] = {'max_iter': 256}
    scene = Scene(config)
    if use_contact:
        scene.contact_tabular().default_model(0.0, 10.0 * kPa)

    snh = StableNeoHookean()
    spc = SoftPositionConstraint()
    moduli = ElasticModuli.youngs_poisson(0.5 * kPa, 0.40)
    spc_stiffness = 1e4  # strong guidance to LBS targets

    geo_slots = []
    for m in muscles:
        mesh = tetmesh(m['vertices'], m['tetrahedra'])
        label_surface(mesh)
        label_triangle_orient(mesh)
        snh.apply_to(mesh, moduli, mass_density=1060.0)
        spc.apply_to(mesh, spc_stiffness)

        obj = scene.objects().create(m['name'])
        geo_slot, _ = obj.geometries().create(mesh)
        geo_slots.append(geo_slot)

        # Animator: drive ALL vertices to LBS targets
        aim_targets = m['aim_targets']
        n_v = len(m['vertices'])
        def make_animate(targets_ref, nv):
            _first_error = [True]
            def animate(info: Animation.UpdateInfo):
                try:
                    geo = info.geo_slots()[0].geometry()
                    cv = view(geo.vertices().find(builtin.is_constrained))
                    av = view(geo.vertices().find(builtin.aim_position))
                    for idx in range(min(nv, len(cv))):
                        cv[idx] = 1
                        av[idx][0] = float(targets_ref[idx][0])
                        av[idx][1] = float(targets_ref[idx][1])
                        av[idx][2] = float(targets_ref[idx][2])
                except Exception as e:
                    if _first_error[0]:
                        print(f"    Animator error: {e}", flush=True)
                        _first_error[0] = False
            return animate
        scene.animator().insert(obj, make_animate(aim_targets, n_v))

        print(f"    {m['name']}: {len(m['vertices'])}v, {len(m['tetrahedra'])}t, {len(m['fixed_vertices'])} fixed")

    # Init
    print("[7] Initializing pyuipc...")
    t0 = time.time()
    world.init(scene)
    print(f"    Init: {time.time()-t0:.1f}s")

    # Bake
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n[8] Baking frames {start_frame}-{end_frame}...")

    all_positions = {m['name']: {} for m in muscles}
    stall_count = 0

    for frame in range(start_frame, end_frame + 1):
        t0 = time.time()

        # Set skeleton pose and compute LBS positions for all muscles
        pose = motion_bvh.mocap_refs[frame].copy()
        skel.setPositions(pose)

        for m in muscles:
            if m['lbs_bindings'] is not None:
                lbs_pos = compute_lbs_positions(m['lbs_bindings'], skel, len(m['vertices']))
                m['aim_targets'][:] = lbs_pos
            # else: aim_targets stays at rest_vertices (shouldn't happen)

        # Advance simulation
        world.advance()
        world.retrieve()
        dt = time.time() - t0

        # Capture positions
        frame_positions = {}
        total_inv = 0
        for i, m in enumerate(muscles):
            geo = geo_slots[i].geometry()
            pos = np.array(geo.positions().view()).reshape(-1, 3)
            frame_positions[m['name']] = pos.astype(np.float32)

            # Check inversions
            tets = m['tetrahedra']
            if len(tets) > 0:
                v0 = pos[tets[:, 0]]
                cr = np.cross(pos[tets[:, 1]] - v0, pos[tets[:, 2]] - v0)
                vol = np.einsum('ij,ij->i', cr, pos[tets[:, 3]] - v0) / 6.0
                total_inv += int(np.sum(vol <= 0))

        # Post-process: resolve muscle-muscle collisions
        resolve_collisions(muscles, frame_positions, d_min=0.5, n_iters=3)

        for m in muscles:
            all_positions[m['name']][frame] = frame_positions[m['name']]

        if dt > 5.0:
            stall_count += 1
        print(f"  Frame {frame}: {dt:.2f}s, inv={total_inv}/{total_tets}", flush=True)

        # Abort if solver consistently stalling
        if stall_count > 5:
            print(f"\nWARNING: Solver stalling ({stall_count} slow frames). Saving partial results.")
            break

    # Save
    out_path = os.path.join(args.output_dir, f"pyuipc_{os.path.basename(args.bvh).replace('.bvh','')}_f{start_frame}-{end_frame}.npz")
    np.savez_compressed(out_path, **{
        f"{name}_f{f}": pos for name, frames in all_positions.items() for f, pos in frames.items()
    })
    print(f"\nSaved to {out_path}")
    print(f"Done. {len(muscles)} muscles, {len(all_positions[muscles[0]['name']])} frames.")


if __name__ == '__main__':
    main()
