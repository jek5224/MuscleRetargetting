#!/usr/bin/env python3
"""Phase 2 IPC validation: take ARAP bake as rest shape, resolve contacts.

Tests on 3 muscles (L_Sartorius, L_Gracilis, L_Semitendinosus) to validate
that IPC contact resolves overlaps without shrinking or detachment.

Usage (on A6000 server):
    python tools/test_ipc_phase2.py --bvh data/motion/walk.bvh --frame 60
"""
import argparse
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
SCALE = 1000.0

TEST_MUSCLES = ["L_Sartorius", "L_Gracilis", "L_Semitendinosus"]


def load_skeleton():
    skel_info, root_name, bvh_info, _pd, mesh_info, _smpl = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)
    return skel, bvh_info, mesh_info


def load_arap_cache(cache_dir, muscle_name, frame):
    """Load ARAP-baked positions for a specific muscle and frame."""
    import glob
    for pattern in [f'{muscle_name}_chunk_*.npz', f'{muscle_name}.npz']:
        for path in glob.glob(os.path.join(cache_dir, pattern)):
            data = np.load(path)
            frames = data['frames']
            positions = data['positions']
            for i, f in enumerate(frames):
                if int(f) == frame:
                    return positions[i]
    return None


def load_muscle(name):
    """Load tet mesh and identify cap vertices."""
    path = f'tet/{name}_tet.npz'
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

    # Remove degenerate
    v0 = verts[tets[:, 0]]
    cross = np.cross(verts[tets[:, 1]] - v0, verts[tets[:, 2]] - v0)
    vol = np.einsum('ij,ij->i', cross, verts[tets[:, 3]] - v0) / 6.0
    good = vol > 0.01
    if not np.all(good):
        tets = tets[good]

    # Collect cap/fixed vertices
    cap_faces = data.get('cap_face_indices', [])
    fixed_verts = set()
    sim_faces = data.get('sim_faces', data.get('faces'))
    if sim_faces is not None:
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

    return {
        'name': name,
        'vertices': verts,
        'tetrahedra': tets,
        'fixed_vertices': sorted(fixed_verts),
        'remap': remap,
        'n_orig': len(data['vertices']),
    }


def compute_volume(verts, tets):
    """Total tet mesh volume in mm³."""
    v0 = verts[tets[:, 0]]
    cross = np.cross(verts[tets[:, 1]] - v0, verts[tets[:, 2]] - v0)
    vol = np.einsum('ij,ij->i', cross, verts[tets[:, 3]] - v0) / 6.0
    return np.sum(np.abs(vol))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bvh', default='data/motion/walk.bvh')
    parser.add_argument('--frame', type=int, default=60)
    parser.add_argument('--cache-dir', default='data/motion_cache/walk/L_UpLeg')
    parser.add_argument('--d-hat', type=float, default=0.1, help='IPC distance threshold (mm)')
    parser.add_argument('--elastic', type=float, default=5.0, help='Elastic modulus (kPa)')
    parser.add_argument('--spc-stiffness', type=float, default=1e5, help='SPC stiffness for cap vertices')
    parser.add_argument('--n-steps', type=int, default=10, help='Number of IPC steps')
    args = parser.parse_args()

    print(f"=== Phase 2 IPC Validation ===")
    print(f"Muscles: {TEST_MUSCLES}")
    print(f"Frame: {args.frame}, d_hat: {args.d_hat}mm, elastic: {args.elastic}kPa")
    print(f"SPC (caps only): {args.spc_stiffness}")

    # Load muscles
    muscles = []
    for name in TEST_MUSCLES:
        m = load_muscle(name)
        muscles.append(m)
        print(f"  {name}: {len(m['vertices'])}v, {len(m['tetrahedra'])}t, {len(m['fixed_vertices'])} caps")

    # Load ARAP positions for this frame
    print(f"\nLoading ARAP cache from {args.cache_dir}...")
    for m in muscles:
        arap_pos = load_arap_cache(args.cache_dir, m['name'], args.frame)
        if arap_pos is None:
            print(f"  ERROR: No ARAP cache for {m['name']} frame {args.frame}")
            return

        # Convert to mm and remap
        arap_mm = arap_pos.astype(np.float64) * SCALE
        if m['remap'] is not None:
            used = np.where(m['remap'] >= 0)[0]
            arap_mm = arap_mm[used]

        if len(arap_mm) != len(m['vertices']):
            print(f"  ERROR: {m['name']} vertex count mismatch: ARAP={len(arap_mm)}, tet={len(m['vertices'])}")
            return

        m['arap_positions'] = arap_mm
        m['rest_volume'] = compute_volume(m['vertices'], m['tetrahedra'])
        m['arap_volume'] = compute_volume(arap_mm, m['tetrahedra'])
        print(f"  {m['name']}: loaded, vol_rest={m['rest_volume']:.0f} vol_arap={m['arap_volume']:.0f}mm³")

    # Check inter-muscle distance before IPC
    print(f"\nPre-IPC inter-muscle distances:")
    for i in range(len(muscles)):
        for j in range(i + 1, len(muscles)):
            tree = cKDTree(muscles[i]['arap_positions'])
            dists, _ = tree.query(muscles[j]['arap_positions'])
            print(f"  {muscles[i]['name']} <-> {muscles[j]['name']}: min={dists.min():.3f}mm")

    # Re-orient tets for ARAP positions (LBS/ARAP deformation may invert)
    for m in muscles:
        tets = m['tetrahedra']
        pos = m['arap_positions']
        v0 = pos[tets[:, 0]]
        cross = np.cross(pos[tets[:, 1]] - v0, pos[tets[:, 2]] - v0)
        vol = np.einsum('ij,ij->i', cross, pos[tets[:, 3]] - v0) / 6.0
        neg = vol < 0
        if np.any(neg):
            tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy()
            print(f"  {m['name']}: flipped {np.sum(neg)} tets")
        # Remove degenerate
        v0 = pos[tets[:, 0]]
        cross = np.cross(pos[tets[:, 1]] - v0, pos[tets[:, 2]] - v0)
        vol = np.einsum('ij,ij->i', cross, pos[tets[:, 3]] - v0) / 6.0
        good = vol > 0.001
        if not np.all(good):
            n_bad = int(np.sum(~good))
            tets = tets[good]
            m['tetrahedra'] = tets
            print(f"  {m['name']}: removed {n_bad} degenerate tets")

    # Setup pyuipc — ARAP positions as rest shape, contact ON, SPC on caps only
    print(f"\nSetting up pyuipc...")
    engine = Engine('cuda')
    world = World(engine)

    config = Scene.default_config()
    config['dt'] = 0.01
    config['gravity'] = [[0.0], [0.0], [0.0]]
    config['contact'] = {
        'enable': True,
        'friction': {'enable': False},
        'd_hat': args.d_hat,
    }
    config['sanity_check'] = {'enable': False}
    config['newton'] = {'max_iter': 512}
    scene = Scene(config)
    scene.contact_tabular().default_model(0.0, 1.0 * MPa)

    snh = StableNeoHookean()
    spc = SoftPositionConstraint()
    moduli = ElasticModuli.youngs_poisson(args.elastic * kPa, 0.45)

    geo_slots = []
    for m in muscles:
        # Use ARAP positions as both initial and rest shape
        mesh = tetmesh(m['arap_positions'], m['tetrahedra'])
        label_surface(mesh)
        label_triangle_orient(mesh)
        snh.apply_to(mesh, moduli, mass_density=1060.0)

        # SPC ONLY on cap vertices — keeps them attached to skeleton
        spc.apply_to(mesh, args.spc_stiffness)

        obj = scene.objects().create(m['name'])
        geo_slot, _ = obj.geometries().create(mesh)
        geo_slots.append(geo_slot)

        # Animator: constrain only cap vertices, leave interior free
        cap_set = set(m['fixed_vertices'])
        arap_pos_ref = m['arap_positions']
        n_v = len(m['arap_positions'])

        def make_animate(caps, pos_ref, nv):
            def animate(info: Animation.UpdateInfo):
                geo = info.geo_slots()[0].geometry()
                cv = view(geo.vertices().find(builtin.is_constrained))
                av = view(geo.vertices().find(builtin.aim_position))
                for idx in range(min(nv, len(cv))):
                    if idx in caps:
                        cv[idx] = 1
                        av[idx][0] = float(pos_ref[idx][0])
                        av[idx][1] = float(pos_ref[idx][1])
                        av[idx][2] = float(pos_ref[idx][2])
                    else:
                        cv[idx] = 0  # unconstrained — elastic + contact only
            return animate
        scene.animator().insert(obj, make_animate(cap_set, arap_pos_ref, n_v))

        print(f"  {m['name']}: {n_v}v, {len(m['tetrahedra'])}t, {len(cap_set)} caps constrained")

    # Init and run
    print(f"\nInitializing...")
    t0 = time.time()
    world.init(scene)
    print(f"  Init: {time.time() - t0:.1f}s")

    print(f"\nRunning {args.n_steps} IPC steps...")
    for step in range(args.n_steps):
        t0 = time.time()
        world.advance()
        world.retrieve()
        dt = time.time() - t0

        # Check results
        for i, m in enumerate(muscles):
            geo = geo_slots[i].geometry()
            pos = np.array(geo.positions().view()).reshape(-1, 3)
            m['ipc_positions'] = pos

            vol = compute_volume(pos, m['tetrahedra'])
            vol_ratio = vol / m['arap_volume']

            # Max displacement from ARAP
            disp = np.linalg.norm(pos - m['arap_positions'], axis=1)

            # Cap displacement (should be near zero)
            cap_disp = np.array([disp[vi] for vi in m['fixed_vertices'] if vi < len(disp)])

            print(f"  Step {step}: {m['name']}: {dt:.2f}s, vol_ratio={vol_ratio:.4f}, "
                  f"max_disp={disp.max():.3f}mm, cap_max={cap_disp.max():.4f}mm")

    # Final inter-muscle distances
    print(f"\nPost-IPC inter-muscle distances:")
    for i in range(len(muscles)):
        for j in range(i + 1, len(muscles)):
            tree = cKDTree(muscles[i]['ipc_positions'])
            dists, _ = tree.query(muscles[j]['ipc_positions'])
            print(f"  {muscles[i]['name']} <-> {muscles[j]['name']}: min={dists.min():.3f}mm")

    # Summary
    print(f"\n=== Summary ===")
    for m in muscles:
        vol_before = m['arap_volume']
        vol_after = compute_volume(m['ipc_positions'], m['tetrahedra'])
        disp = np.linalg.norm(m['ipc_positions'] - m['arap_positions'], axis=1)
        print(f"  {m['name']}:")
        print(f"    Volume: {vol_before:.0f} -> {vol_after:.0f} mm³ (ratio={vol_after/vol_before:.4f})")
        print(f"    Max displacement from ARAP: {disp.max():.3f}mm, mean: {disp.mean():.3f}mm")
        print(f"    Inversions: {np.sum(np.einsum('ij,ij->i', np.cross(m['ipc_positions'][m['tetrahedra'][:,1]]-m['ipc_positions'][m['tetrahedra'][:,0]], m['ipc_positions'][m['tetrahedra'][:,2]]-m['ipc_positions'][m['tetrahedra'][:,0]]), m['ipc_positions'][m['tetrahedra'][:,3]]-m['ipc_positions'][m['tetrahedra'][:,0]]) <= 0)}/{len(m['tetrahedra'])}")


if __name__ == '__main__':
    main()
