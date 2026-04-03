#!/usr/bin/env python3
"""Bake muscle deformation with pyuipc IPC contact.

Loads skeleton, applies BVH motion frame by frame, drives anchor
vertices via skeleton transforms, runs pyuipc with IPC contact.
Saves per-frame deformed positions to npz chunks.

Usage (on A6000 server):
    python tools/bake_pyuipc.py --bvh data/motion/walk1_subject1.bvh --frames 0-100
    python tools/bake_pyuipc.py --bvh data/motion/dance.bvh --frames 0-200 --sides LR
"""
import argparse
import gc
import os
import pickle
import sys
import time

import numpy as np

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
from uipc.unit import kPa, GPa

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

SKELETON_MESHES_ORDER = None  # populated at runtime


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

    # Collect fixed vertices
    anchors = data.get('anchor_vertices', np.array([], dtype=np.int32))
    cap_faces = data.get('cap_face_indices', [])
    fixed_verts = set()
    sim_faces = data.get('sim_faces', data.get('faces'))
    if sim_faces is not None and len(cap_faces) > 0:
        for fi in cap_faces:
            if fi < len(sim_faces):
                for vi in sim_faces[fi]:
                    fixed_verts.add(int(vi))
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

    # Get skeleton attachment info for driving fixed vertices
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
        'remap': remap,
    }


def compute_local_anchors(muscle, skel):
    """Compute local coordinates of each fixed vertex in its nearest bone frame.

    Uses rest pose skeleton transforms. Each fixed vertex is assigned to the
    nearest bone body node. Returns dict: {vertex_idx: (body_name, local_pos)}.
    """
    rest_verts = muscle['rest_vertices']  # in mm
    fixed = muscle['fixed_vertices']
    if len(fixed) == 0:
        return {}

    # Get all body transforms at rest pose
    skel.setPositions(np.zeros(skel.getNumDofs()))
    bodies = []
    for i in range(skel.getNumBodyNodes()):
        bn = skel.getBodyNode(i)
        wt = bn.getWorldTransform()
        bodies.append({
            'name': bn.getName(),
            'rotation': wt.rotation(),
            'translation': wt.translation() * SCALE,  # convert to mm
        })

    # For each fixed vertex, find nearest bone
    local_anchors = {}
    for vi in fixed:
        if vi >= len(rest_verts):
            continue
        world_pos = rest_verts[vi]  # mm

        # Find nearest bone by translation distance
        best_dist = float('inf')
        best_body = None
        for body in bodies:
            dist = np.linalg.norm(world_pos - body['translation'])
            if dist < best_dist:
                best_dist = dist
                best_body = body

        if best_body is not None:
            R = best_body['rotation']
            t = best_body['translation']
            local_pos = R.T @ (world_pos - t)
            local_anchors[vi] = (best_body['name'], local_pos)

    return local_anchors


def compute_driven_positions(local_anchors, skel, rest_verts):
    """Compute target positions for fixed vertices based on current skeleton pose.

    Returns array of positions (mm) for ALL vertices. Fixed vertices get
    skeleton-driven positions, others keep rest positions.
    """
    target = rest_verts.copy()

    for vi, (body_name, local_pos) in local_anchors.items():
        bn = skel.getBodyNode(body_name)
        if bn is None:
            continue
        wt = bn.getWorldTransform()
        R = wt.rotation()
        t = wt.translation() * SCALE  # mm
        target[vi] = R @ local_pos + t

    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bvh', required=True, help='BVH motion file')
    parser.add_argument('--frames', default='0-30', help='Frame range (e.g. 0-100)')
    parser.add_argument('--tet-dir', default='tet_sim')
    parser.add_argument('--output-dir', default='bake_pyuipc')
    parser.add_argument('--sides', default='L', help='L, R, or LR')
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

    # Compute local anchors at rest pose
    print("[4] Computing skeleton anchors...")
    for m in muscles:
        m['local_anchors'] = compute_local_anchors(m, skel)
        print(f"    {m['name']}: {len(m['local_anchors'])} anchored to skeleton")
        # Shared target array (updated before each frame advance)
        m['aim_targets'] = m['rest_vertices'].copy()

    # Setup pyuipc
    print("[5] Setting up pyuipc...")
    engine = Engine('cuda')
    world = World(engine)

    config = Scene.default_config()
    config['dt'] = 0.01
    config['gravity'] = [[0.0], [0.0], [0.0]]
    config['contact']['friction']['enable'] = False
    config['contact']['d_hat'] = 0.1
    config['sanity_check'] = {'enable': False}
    scene = Scene(config)
    scene.contact_tabular().default_model(0.0, 1.0 * GPa)

    snh = StableNeoHookean()
    spc = SoftPositionConstraint()
    moduli = ElasticModuli.youngs_poisson(0.5 * kPa, 0.40)

    geo_slots = []
    for m in muscles:
        mesh = tetmesh(m['vertices'], m['tetrahedra'])
        label_surface(mesh)
        label_triangle_orient(mesh)
        snh.apply_to(mesh, moduli, mass_density=1060.0)
        if len(m['fixed_vertices']) > 0:
            spc.apply_to(mesh, 1e4)

        obj = scene.objects().create(m['name'])
        geo_slot, _ = obj.geometries().create(mesh)
        geo_slots.append(geo_slot)

        # Animator: reads from m['aim_targets'] which is updated before each advance
        fixed = m['fixed_vertices']
        aim_targets = m['aim_targets']  # shared mutable array
        if len(fixed) > 0:
            def make_animate(fixed_verts, targets_ref):
                _first_error = [True]
                def animate(info: Animation.UpdateInfo):
                    try:
                        geo = info.geo_slots()[0].geometry()
                        rest_geo = info.rest_geo_slots()[0].geometry()
                        cv = view(geo.vertices().find(builtin.is_constrained))
                        av = view(geo.vertices().find(builtin.aim_position))
                        rv = rest_geo.positions().view()
                        for idx in fixed_verts:
                            if idx < len(cv):
                                cv[idx] = 1
                                # Copy target values into rv-style format
                                t = targets_ref[idx]
                                av[idx] = rv[idx]  # init from rest
                                av[idx][0] = float(t[0])
                                av[idx][1] = float(t[1])
                                av[idx][2] = float(t[2])
                    except Exception as e:
                        if _first_error[0]:
                            print(f"    Animator error: {e}", flush=True)
                            _first_error[0] = False
                return animate
            scene.animator().insert(obj, make_animate(fixed, aim_targets))

        print(f"    {m['name']}: {len(m['vertices'])} v, {len(m['tetrahedra'])} t, {len(fixed)} fixed")

    # Init
    print("[5] Initializing pyuipc...")
    t0 = time.time()
    world.init(scene)
    print(f"    Init: {time.time()-t0:.1f}s")

    # Bake
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n[6] Baking frames {start_frame}-{end_frame}...")

    all_positions = {m['name']: {} for m in muscles}

    for frame in range(start_frame, end_frame + 1):
        t0 = time.time()

        # Set skeleton pose and update aim targets for all muscles
        pose = motion_bvh.mocap_refs[frame].copy()
        skel.setPositions(pose)

        for m in muscles:
            driven = compute_driven_positions(m['local_anchors'], skel, m['rest_vertices'])
            # Update shared aim_targets in-place (animator reads this)
            m['aim_targets'][:] = driven

        # Advance simulation
        world.advance()
        world.retrieve()
        dt = time.time() - t0

        # Capture positions
        total_inv = 0
        for i, m in enumerate(muscles):
            geo = geo_slots[i].geometry()
            pos = np.array(geo.positions().view()).reshape(-1, 3)
            all_positions[m['name']][frame] = pos.astype(np.float32)

            # Check inversions
            tets = m['tetrahedra']
            if len(tets) > 0:
                v0 = pos[tets[:, 0]]
                cr = np.cross(pos[tets[:, 1]] - v0, pos[tets[:, 2]] - v0)
                vol = np.einsum('ij,ij->i', cr, pos[tets[:, 3]] - v0) / 6.0
                total_inv += int(np.sum(vol <= 0))

        print(f"  Frame {frame}: {dt:.2f}s, inv={total_inv}/{total_tets}")

    # Save
    out_path = os.path.join(args.output_dir, f"pyuipc_{os.path.basename(args.bvh).replace('.bvh','')}_f{start_frame}-{end_frame}.npz")
    np.savez_compressed(out_path, **{
        f"{name}_f{f}": pos for name, frames in all_positions.items() for f, pos in frames.items()
    })
    print(f"\nSaved to {out_path}")
    print(f"Done. {len(muscles)} muscles, {end_frame-start_frame+1} frames.")


if __name__ == '__main__':
    main()
