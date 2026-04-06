#!/usr/bin/env python3
"""Phase 2 IPC bake: take ARAP bake as rest shape, resolve contacts per frame.

For each frame:
  1. Load ARAP-baked positions as rest shape
  2. Run pyuipc with IPC contact + SPC on cap vertices only
  3. Save contact-resolved positions

Usage (on A6000 server):
    python tools/bake_ipc_phase2.py --bvh data/motion/walk.bvh --sides LR
"""
import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from uipc import view, Animation
import uipc.builtin as builtin
from uipc.core import Engine, World, Scene
from uipc.geometry import tetmesh, label_surface, label_triangle_orient
from uipc.constitution import StableNeoHookean, SoftPositionConstraint, ElasticModuli
from uipc.unit import kPa, MPa

SCALE = 1000.0

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


def load_muscle(name):
    """Load tet mesh and identify cap vertices."""
    path = f'tet/{name}_tet.npz'
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
    # Remove degenerate
    v0 = verts[tets[:, 0]]
    cross = np.cross(verts[tets[:, 1]] - v0, verts[tets[:, 2]] - v0)
    vol = np.einsum('ij,ij->i', cross, verts[tets[:, 3]] - v0) / 6.0
    good = vol > 0.01
    if not np.all(good):
        tets = tets[good]

    # Cap vertices
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

    # Precompute barycentric coords for unmapped vertices (in render faces but not tets)
    inv_remap = used if remap is not None else None
    unmapped_bary = {}  # orig_vi -> (tet_idx_in_remapped, bary_coords)
    if remap is not None:
        unmapped_vis = [vi for vi in range(len(data['vertices'])) if remap[vi] < 0]
        if len(unmapped_vis) > 0:
            all_orig_verts = data['vertices'].astype(np.float64) * SCALE
            # Find nearest tet for each unmapped vertex
            tet_centers = np.mean(verts[tets], axis=1)
            tet_tree = cKDTree(tet_centers)
            for orig_vi in unmapped_vis:
                pt = all_orig_verts[orig_vi]
                # Search several nearest tets
                dists, idxs = tet_tree.query(pt, k=min(10, len(tets)))
                if not hasattr(idxs, '__len__'):
                    idxs = [idxs]
                for ti in idxs:
                    t = tets[ti]
                    # Compute barycentric coordinates
                    v0, v1, v2, v3 = verts[t[0]], verts[t[1]], verts[t[2]], verts[t[3]]
                    mat = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
                    det = np.linalg.det(mat)
                    if abs(det) < 1e-12:
                        continue
                    bary123 = np.linalg.solve(mat, pt - v0)
                    bary0 = 1.0 - bary123.sum()
                    bary = np.array([bary0, bary123[0], bary123[1], bary123[2]])
                    # Accept if roughly inside (allow tolerance for surface verts)
                    if np.all(bary > -2.0):
                        unmapped_bary[orig_vi] = (ti, bary)
                        break

    return {
        'name': name,
        'rest_vertices': verts,
        'tetrahedra': tets,
        'fixed_vertices': sorted(fixed_verts),
        'remap': remap,
        'inv_remap': inv_remap,
        'n_orig': len(data['vertices']),
        'unmapped_bary': unmapped_bary,
        'all_orig_rest': data['vertices'].astype(np.float64) * SCALE,
    }


def load_arap_cache_all_frames(cache_dir, muscle_name):
    """Load all ARAP-baked frames for a muscle. Returns {frame: positions_mm}."""
    cache = {}
    for pattern in [f'{muscle_name}_chunk_*.npz', f'{muscle_name}.npz']:
        for path in sorted(glob.glob(os.path.join(cache_dir, pattern))):
            data = np.load(path)
            frames = data['frames']
            positions = data['positions']
            for i, f in enumerate(frames):
                cache[int(f)] = positions[i].astype(np.float64) * SCALE
    return cache


def expand_to_full(muscle, remapped_pos):
    """Expand remapped positions to full original vertex array.

    Mapped vertices get their solved positions.
    Unmapped vertices get barycentric-interpolated positions from nearest tet.
    """
    remap = muscle['remap']
    if remap is None:
        return remapped_pos.copy()

    n_orig = muscle['n_orig']
    full = muscle['all_orig_rest'].copy()  # fallback: rest positions
    # Fill mapped vertices
    inv_remap = muscle['inv_remap']
    full[inv_remap] = remapped_pos[:len(inv_remap)]
    # Interpolate unmapped vertices
    tets = muscle['tetrahedra']
    for orig_vi, (ti, bary) in muscle['unmapped_bary'].items():
        t = tets[ti]
        full[orig_vi] = (bary[0] * remapped_pos[t[0]] + bary[1] * remapped_pos[t[1]] +
                         bary[2] * remapped_pos[t[2]] + bary[3] * remapped_pos[t[3]])
    return full


def reorient_tets(positions, tets):
    """Fix tet orientation for deformed positions. Returns cleaned tets."""
    tets = tets.copy()
    v0 = positions[tets[:, 0]]
    cross = np.cross(positions[tets[:, 1]] - v0, positions[tets[:, 2]] - v0)
    vol = np.einsum('ij,ij->i', cross, positions[tets[:, 3]] - v0) / 6.0
    neg = vol < 0
    if np.any(neg):
        tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy()
    v0 = positions[tets[:, 0]]
    cross = np.cross(positions[tets[:, 1]] - v0, positions[tets[:, 2]] - v0)
    vol = np.einsum('ij,ij->i', cross, positions[tets[:, 3]] - v0) / 6.0
    good = vol > 0.001
    return tets[good] if not np.all(good) else tets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bvh', default='data/motion/walk.bvh')
    parser.add_argument('--sides', default='LR')
    parser.add_argument('--cache-base', default='data/motion_cache/walk',
                        help='Base cache dir (ARAP subdirs: L_UpLeg/, R_UpLeg/)')
    parser.add_argument('--output-dir', default='bake_ipc_phase2')
    parser.add_argument('--d-hat', type=float, default=0.1)
    parser.add_argument('--elastic', type=float, default=5.0)
    parser.add_argument('--spc-stiffness', type=float, default=1e5)
    parser.add_argument('--n-steps', type=int, default=5,
                        help='IPC steps per frame')
    args = parser.parse_args()

    # Determine frame range from ARAP cache
    sample_side = args.sides[0]
    sample_dir = os.path.join(args.cache_base, f'{sample_side}_UpLeg')
    sample_files = glob.glob(os.path.join(sample_dir, '*_chunk_*.npz'))
    if not sample_files:
        print(f"ERROR: No ARAP cache in {sample_dir}")
        return
    sample_data = np.load(sample_files[0])
    all_frames = sorted(sample_data['frames'].tolist())
    start_frame, end_frame = all_frames[0], all_frames[-1]
    print(f"Frames: {start_frame}-{end_frame} ({len(all_frames)} frames)")

    # Load muscles
    print("[1] Loading muscles...")
    muscles = []
    for side in args.sides:
        for mname in UPLEG_MUSCLES:
            name = f"{side}_{mname}"
            m = load_muscle(name)
            if m is not None:
                muscles.append(m)
    print(f"    {len(muscles)} muscles loaded")

    # Load all ARAP caches
    print("[2] Loading ARAP caches...")
    for m in muscles:
        side = m['name'][0]  # L or R
        cache_dir = os.path.join(args.cache_base, f'{side}_UpLeg')
        m['arap_cache'] = load_arap_cache_all_frames(cache_dir, m['name'])

        # Remap: ARAP cache has n_orig vertices, we need remapped
        if m['remap'] is not None:
            used = np.where(m['remap'] >= 0)[0]
            for f in m['arap_cache']:
                m['arap_cache'][f] = m['arap_cache'][f][used]

        n_frames = len(m['arap_cache'])
        if n_frames == 0:
            print(f"    WARNING: {m['name']} has no ARAP cache!")
        else:
            sample_f = list(m['arap_cache'].keys())[0]
            nv_cache = len(m['arap_cache'][sample_f])
            nv_tet = len(m['rest_vertices'])
            if nv_cache != nv_tet:
                print(f"    WARNING: {m['name']} vertex mismatch: cache={nv_cache}, tet={nv_tet}")
            else:
                print(f"    {m['name']}: {n_frames} frames, {nv_tet}v")

    # Process frame by frame
    os.makedirs(args.output_dir, exist_ok=True)
    all_positions = {m['name']: {} for m in muscles}

    prev_engine = None
    for fi, frame in enumerate(all_frames):
        t_frame = time.time()

        # Get ARAP positions for this frame
        frame_positions = {}
        skip_frame = False
        for m in muscles:
            if frame not in m['arap_cache']:
                skip_frame = True
                break
            frame_positions[m['name']] = m['arap_cache'][frame]
        if skip_frame:
            print(f"  Frame {frame}: skipped (missing ARAP data)")
            continue

        # Build pyuipc scene fresh per frame (rest shape = ARAP positions)
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
        valid_muscles = []
        for m in muscles:
            arap_pos = frame_positions[m['name']]
            tets = reorient_tets(arap_pos, m['tetrahedra'])

            try:
                mesh = tetmesh(arap_pos, tets)
                label_surface(mesh)
                label_triangle_orient(mesh)
                snh.apply_to(mesh, moduli, mass_density=1060.0)
                spc.apply_to(mesh, args.spc_stiffness)
            except Exception as e:
                print(f"  Frame {frame}: {m['name']} setup failed: {e}")
                all_positions[m['name']][frame] = expand_to_full(m, arap_pos).astype(np.float32)
                continue

            obj = scene.objects().create(m['name'])
            geo_slot, _ = obj.geometries().create(mesh)
            geo_slots.append(geo_slot)
            valid_muscles.append(m)

            # SPC on cap vertices only
            cap_set = set(m['fixed_vertices'])
            n_v = len(arap_pos)

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
                            cv[idx] = 0
                return animate
            scene.animator().insert(obj, make_animate(cap_set, arap_pos, n_v))

        # Init and run
        try:
            world.init(scene)
            for _ in range(args.n_steps):
                world.advance()
                world.retrieve()
        except Exception as e:
            print(f"  Frame {frame}: IPC failed: {e}")
            for m in valid_muscles:
                all_positions[m['name']][frame] = expand_to_full(m, frame_positions[m['name']]).astype(np.float32)
            dt = time.time() - t_frame
            print(f"  Frame {frame}: {dt:.2f}s (FALLBACK to ARAP)")
            del world, engine
            continue

        # Capture resolved positions — expand to full vertex array
        for i, m in enumerate(valid_muscles):
            geo = geo_slots[i].geometry()
            pos = np.array(geo.positions().view()).reshape(-1, 3)
            all_positions[m['name']][frame] = expand_to_full(m, pos).astype(np.float32)

        # For muscles that failed setup, use ARAP directly
        for m in muscles:
            if frame not in all_positions[m['name']]:
                all_positions[m['name']][frame] = expand_to_full(m, frame_positions[m['name']]).astype(np.float32)

        dt = time.time() - t_frame
        if fi % 10 == 0 or fi == len(all_frames) - 1:
            print(f"  Frame {frame}: {dt:.2f}s, {len(valid_muscles)} muscles", flush=True)

        del world, engine

    # Save directly as viewer cache chunks (positions in viewer scale = mm / SCALE)
    bvh_stem = os.path.basename(args.bvh).replace('.bvh', '')
    cache_out = os.path.join(args.output_dir, bvh_stem)
    os.makedirs(cache_out, exist_ok=True)

    for m in muscles:
        mframes = all_positions[m['name']]
        if not mframes:
            continue
        sorted_f = sorted(mframes.keys())
        positions = np.array([mframes[f] / SCALE for f in sorted_f], dtype=np.float32)
        out_path = os.path.join(cache_out, f"{m['name']}_chunk_0000.npz")
        np.savez(out_path,
                 frames=np.array(sorted_f, dtype=np.int32),
                 positions=positions)

    print(f"\nSaved viewer cache to {cache_out}/")
    print(f"Done. {len(muscles)} muscles, {len(all_frames)} frames.")
    print(f"Copy to data/motion_cache/{bvh_stem}/ to use in viewer.")


if __name__ == '__main__':
    main()
