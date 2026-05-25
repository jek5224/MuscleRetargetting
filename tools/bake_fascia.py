"""Per-region fascia bake (paper §4.2; fat/skin omitted).

Pipeline:
  1. Reuses tools/bake_headless.py machinery to bake the region's muscles
     **without inter-muscle pins** (ctx.inter_muscle_constraints = []).
  2. After each frame's muscle solve, computes fascia vertex positions
     by barycentric pull from the bound muscle anatomical triangle at
     the current deformed state.
  3. Caches per-muscle tet positions and the fascia mesh positions per
     frame in separate dirs:
         data/motion_cache/{stem}/{REGION}_fascia/{muscle}_chunk_*.npz
         data/motion_cache/{stem}/fascia_{REGION}/fascia_chunk_*.npz

Cage feedback (muscle-stay-inside-fascia) and explicit fascia ARAP are
intentionally out-of-scope for this first cut.  Muscles ride free; fascia
rides muscles passively.  Validate visual + drift, then iterate.
"""
import argparse
import gc
import json
import os
import pickle
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.bake_headless import (
    load_skeleton, load_skeleton_meshes, load_muscle_meshes,
    load_tet_meshes, init_soft_bodies, build_context, flush_bake_data,
    FLUSH_INTERVAL, _detect_bvh_tframe,
)
from core.bvhparser import MyBVH
from viewer.zygote_mesh_ui import run_all_tet_sim_with_constraints

REGION_FASCIA_FILES = {
    'L_UpLeg':  'data/zygote_fascia_rest_L_UpLeg.npz',
    'R_UpLeg':  'data/zygote_fascia_rest_R_UpLeg.npz',
    'L_LowLeg': 'data/zygote_fascia_rest_L_LowLeg.npz',
    'R_LowLeg': 'data/zygote_fascia_rest_R_LowLeg.npz',
}
REGION_MUSCLE_JSON = {
    'L_UpLeg':  '.muscles_L_UpLeg.json',
    'R_UpLeg':  '.muscles_R_UpLeg.json',
    'L_LowLeg': '.muscles_L_LowLeg.json',
    'R_LowLeg': '.muscles_R_LowLeg.json',
}
TET_DIR = 'tet'


def build_fascia_binding(fascia_npz_path, active_muscles):
    """Map fascia verts to (muscle_name, [3 tet-vert indices], bary)."""
    d = np.load(fascia_npz_path)
    Vf = d['vertices']
    Ff = d['faces']
    edges = d['edges']
    src_muscle = d['src_muscle']
    src_tri = d['src_tri']
    bary = d['bary']
    muscle_names_in_npz = [str(n) for n in d['muscle_names']]

    # For each muscle in this region, load its anatomical tri list
    # (render_faces minus cap_face_indices).
    anatomical_tris = {}
    for name in muscle_names_in_npz:
        p = os.path.join(TET_DIR, f'{name}_tet.npz')
        if not os.path.exists(p):
            anatomical_tris[name] = None
            continue
        with open(p, 'rb') as f:
            td = pickle.load(f)
        F_render = np.asarray(td['render_faces'], dtype=np.int32)
        cap = np.asarray(td['cap_face_indices'], dtype=np.int64)
        mask = np.ones(len(F_render), dtype=bool)
        mask[cap] = False
        anatomical_tris[name] = F_render[mask]

    # Build per-fascia-vert tet-vert index triple.
    V_f = len(Vf)
    binding = {
        'fascia_rest': Vf,
        'faces': Ff,
        'edges': edges,
        'src_muscle_name': [muscle_names_in_npz[m] for m in src_muscle],
        'src_tri_verts': np.zeros((V_f, 3), dtype=np.int32),
        'bary': bary,
    }
    skipped = 0
    for i in range(V_f):
        m_name = binding['src_muscle_name'][i]
        anat = anatomical_tris.get(m_name)
        if anat is None or m_name not in active_muscles:
            binding['src_tri_verts'][i] = -1
            skipped += 1
            continue
        tri = anat[src_tri[i]]
        binding['src_tri_verts'][i] = tri
    if skipped:
        print(f'[fascia] WARN: {skipped}/{V_f} fascia verts have no active muscle binding')
    return binding


def compute_fascia_positions(binding, active_muscles):
    """Vectorised barycentric pull from current muscle positions."""
    Vf = binding['fascia_rest']
    bary = binding['bary']                       # (V_f, 3)
    src_verts = binding['src_tri_verts']         # (V_f, 3) int
    names = binding['src_muscle_name']
    n = Vf.shape[0]
    out = np.zeros_like(Vf)
    # Group by muscle so we hit each soft_body once.
    name_to_indices = {}
    for i in range(n):
        if src_verts[i, 0] < 0:
            out[i] = Vf[i]  # fallback to rest
            continue
        name_to_indices.setdefault(names[i], []).append(i)
    for mname, idxs in name_to_indices.items():
        mobj = active_muscles[mname]
        pos = mobj.soft_body.get_positions()     # (Nv, 3)
        idxs = np.array(idxs, dtype=np.int32)
        tri_verts = src_verts[idxs]              # (k, 3)
        w = bary[idxs][:, :, None]               # (k, 3, 1)
        tri_pos = pos[tri_verts]                 # (k, 3, 3)
        out[idxs] = (w * tri_pos).sum(axis=1)
    return out.astype(np.float32)


def flush_fascia_data(fascia_frames, cache_dir, flush_count):
    """Write fascia per-frame positions to a chunk NPZ."""
    if not fascia_frames:
        return flush_count
    sorted_frames = sorted(fascia_frames.keys())
    filepath = os.path.join(cache_dir, f'fascia_chunk_{flush_count:04d}.npz')
    np.savez(filepath,
             frames=np.array(sorted_frames, dtype=np.int32),
             positions=np.array([fascia_frames[f] for f in sorted_frames],
                                dtype=np.float32))
    fascia_frames.clear()
    gc.collect()
    return flush_count + 1


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bvh', required=True)
    ap.add_argument('--region', required=True, choices=list(REGION_FASCIA_FILES))
    ap.add_argument('--cache-tag', default='fascia',
                    help='Cache dir suffix (region tag = "{REGION}_{cache-tag}").')
    ap.add_argument('--start-frame', type=int, default=0)
    ap.add_argument('--end-frame', type=int, default=None)
    ap.add_argument('--backend', default='taichi',
                    choices=['auto', 'taichi', 'gpu', 'cpu'])
    ap.add_argument('--settle-iters', type=int, default=50)
    ap.add_argument('--no-plateau-exit', action='store_true', default=True)
    ap.add_argument('--no-self-collision', action='store_true', default=False)
    ap.add_argument('--tet-dir', default=TET_DIR)
    # Inherited bake-headless args (defaults match its defaults)
    ap.add_argument('--constraint-threshold', type=float, default=0.015)
    ap.add_argument('--inter-k', type=int, default=2)
    ap.add_argument('--inter-muscle-weight', type=float, default=1.0)
    ap.add_argument('--anisotropic', action='store_true', default=False)
    ap.add_argument('--arap-cross-w', type=float, default=1.0)
    ap.add_argument('--arap-intra-w', type=float, default=1.0)
    ap.add_argument('--arap-neutral-w', type=float, default=1.0)
    ap.add_argument('--use-muscle-aware-arap', action='store_true', default=False)
    ap.add_argument('--skin-prior', action='store_true', default=False)
    ap.add_argument('--skin-prior-include', default='')
    ap.add_argument('--skin-prior-exclude', default='')
    ap.add_argument('--skin-prior-sigma', type=float, default=0.02)
    ap.add_argument('--skin-prior-max-dist', type=float, default=0.05)
    ap.add_argument('--skin-prior-strength', type=float, default=0.35)
    ap.add_argument('--unified-bone-contact', dest='unified_bone_contact',
                    action='store_true', default=True)
    ap.add_argument('--no-unified-bone-contact', dest='unified_bone_contact',
                    action='store_false')
    ap.add_argument('--anisotropic-contact', action='store_true', default=True,
                    help='Paper §4.1.2 dynamic muscle-muscle + muscle-bone contact '
                         '(sliding-allowed repulsion).')
    ap.add_argument('--no-anisotropic-contact', dest='anisotropic_contact',
                    action='store_false')
    ap.add_argument('--inter-muscle-contact-margin', type=float, default=0.002)
    ap.add_argument('--fascia-constraints', action='store_true', default=True,
                    help='Paper §4.1.2 barycentric fascia constraints: each '
                         'muscle boundary vert bound to nearest OTHER-muscle '
                         'triangle at A-pose; soft pull toward barycentric '
                         'point on that triangle at current frame.')
    ap.add_argument('--no-fascia-constraints', dest='fascia_constraints',
                    action='store_false')
    ap.add_argument('--fascia-constraint-threshold', type=float, default=0.01,
                    help='Max A-pose distance (m) for forming a fascia '
                         'constraint. Default 1 cm.')
    ap.add_argument('--fascia-constraint-weight', type=float, default=1.0,
                    help='Soft penalty weight for fascia constraints.')
    ap.add_argument('--unified-bone-contact-margin', type=float, default=0.005)
    ap.add_argument('--unified-bone-contact-weight', type=float, default=1.5)
    ap.add_argument('--axial-min-ratio', type=float, default=0.65)
    ap.add_argument('--axial-max-bulge', type=float, default=2.0)
    ap.add_argument('--axial-curve', choices=['linear', 'smooth'], default='smooth')
    ap.add_argument('--fiber-spring', action='store_true', default=False)
    ap.add_argument('--fiber-spring-weight', type=float, default=10.0)
    ap.add_argument('--fiber-spring-rest-scale', type=float, default=0.5)
    ap.add_argument('--pes-bundle', action='store_true', default=False)
    ap.add_argument('--pes-bundle-weight', type=float, default=1.5)
    ap.add_argument('--pes-bundle-offset-scale', type=float, default=1.0)
    return ap.parse_args()


def main():
    args = parse_args()

    region = args.region
    muscles_json = REGION_MUSCLE_JSON[region]
    fascia_rest = REGION_FASCIA_FILES[region]
    for path in (args.bvh, muscles_json, fascia_rest):
        if not os.path.exists(path):
            print(f'ERROR: missing {path}')
            sys.exit(1)

    print(f'[1/9] Loading skeleton + BVH info...')
    skel, bvh_info, mesh_info = load_skeleton()
    skeleton_meshes = load_skeleton_meshes()

    print(f'[2/9] Loading muscles for region {region}...')
    args.muscles = muscles_json  # bake_headless helpers expect args.muscles
    muscle_meshes = load_muscle_meshes(args.muscles)
    load_tet_meshes(muscle_meshes, tet_dir=args.tet_dir)

    print(f'[3/9] Resetting skeleton + init soft bodies...')
    skel.setPositions(np.zeros(skel.getNumDofs()))
    init_soft_bodies(muscle_meshes, skeleton_meshes, skel, mesh_info,
                     smooth_skinning=None)

    print(f'[4/9] Building context (paper §4.1.2: barycentric fascia '
          f'constraints + anisotropic contact)...')
    ctx = build_context(skel, muscle_meshes, skeleton_meshes, mesh_info, args)
    ctx.inter_muscle_constraints = []   # legacy pins OFF

    active_muscles = {n: m for n, m in muscle_meshes.items()
                      if m.soft_body is not None}
    print(f'      active muscles: {len(active_muscles)}')

    print(f'[5/9] Loading fascia rest mesh: {fascia_rest}')
    binding = build_fascia_binding(fascia_rest, active_muscles)
    print(f'      fascia: {len(binding["fascia_rest"])} verts, '
          f'{len(binding["faces"])} faces, {len(binding["edges"])} edges')

    print(f'[6/9] Loading BVH: {args.bvh}')
    t_frame = _detect_bvh_tframe(args.bvh)
    motion_bvh = MyBVH(args.bvh, bvh_info, skel, T_frame=t_frame)
    n_frames = motion_bvh.mocap_refs.shape[0]
    start = args.start_frame
    end = args.end_frame if args.end_frame is not None else n_frames - 1
    end = min(end, n_frames - 1)
    total = end - start + 1
    print(f'      frames: {start}..{end} ({total} of {n_frames})')

    stem = os.path.splitext(os.path.basename(args.bvh))[0]
    muscle_cache_dir = os.path.join('data', 'motion_cache', stem,
                                    f'{region}_{args.cache_tag}')
    fascia_cache_dir = os.path.join('data', 'motion_cache', stem,
                                    f'fascia_{region}')
    for d in (muscle_cache_dir, fascia_cache_dir):
        os.makedirs(d, exist_ok=True)
        import glob as _g
        for old in _g.glob(os.path.join(d, '*_chunk_*.npz')):
            os.remove(old)

    print(f'[7/9] Output dirs:')
    print(f'      muscles -> {muscle_cache_dir}')
    print(f'      fascia  -> {fascia_cache_dir}')

    # Reset soft bodies to rest before bake.
    for mobj in active_muscles.values():
        mobj.soft_body.positions = mobj.soft_body.rest_positions.copy()
        mobj.tet_vertices = mobj.soft_body.rest_positions.astype(np.float32).copy()
        mobj._baking_phase1_iters = 50
        if args.no_self_collision:
            mobj.soft_body_collision = False
    ctx._unified_arap_backend = None
    ctx._unified_arap_backend_ext = None
    ctx._unified_arap_backend_flex = None
    ctx._unified_sim_cache = None

    print(f'[8/9] Frame loop...')
    bake_data = {name: {} for name in active_muscles}
    fascia_frames = {}
    flush_count = 0
    fascia_flush_count = 0
    bake_start = time.time()

    for frame in range(start, end + 1):
        f_start = time.time()
        skel.setPositions(motion_bvh.mocap_refs[frame].copy())

        saved_flags = {}
        for mname, mobj in active_muscles.items():
            saved_flags[mname] = getattr(mobj, 'waypoints_from_tet_sim', True)
            mobj.waypoints_from_tet_sim = False
            mobj._baking_mode = True

        run_all_tet_sim_with_constraints(
            ctx, max_iterations=args.settle_iters,
            tolerance=1e-4, outer_iterations=20,
            snapshot_callback=None,
        )

        # Capture muscle positions
        for mname, mobj in active_muscles.items():
            bake_data[mname][frame] = mobj.soft_body.get_positions().astype(np.float32)

        # Compute fascia positions from current muscle surfaces
        fascia_frames[frame] = compute_fascia_positions(binding, active_muscles)

        for mname, mobj in active_muscles.items():
            mobj.waypoints_from_tet_sim = saved_flags[mname]
            mobj._baking_mode = False

        # Flush periodically
        n_accum = sum(len(fd) for fd in bake_data.values())
        if n_accum >= FLUSH_INTERVAL * len(active_muscles):
            print(f'  Flushing chunk {flush_count}...')
            flush_count = flush_bake_data(bake_data, muscle_cache_dir,
                                          flush_count, None)
            fascia_flush_count = flush_fascia_data(fascia_frames,
                                                   fascia_cache_dir,
                                                   fascia_flush_count)
            gc.collect()

        dt = time.time() - f_start
        done = frame - start + 1
        elapsed = time.time() - bake_start
        avg = elapsed / done
        eta = avg * (total - done)
        print(f'  Frame {frame}/{end}  ({done}/{total})  {dt:.2f}s  '
              f'avg {avg:.2f}s/frame  ETA {eta:.0f}s', flush=True)

    if any(len(fd) > 0 for fd in bake_data.values()):
        flush_count = flush_bake_data(bake_data, muscle_cache_dir,
                                      flush_count, None)
    if fascia_frames:
        fascia_flush_count = flush_fascia_data(fascia_frames,
                                               fascia_cache_dir,
                                               fascia_flush_count)

    print(f'\n[9/9] Done. Muscle chunks: {flush_count}, '
          f'fascia chunks: {fascia_flush_count}.')


if __name__ == '__main__':
    main()
