"""Multi-BVH variant of reverse_lbs_solve.

Concatenates waypoint trajectories + body transforms across one or more
(BVH, cache_template) sources before solving per-waypoint LBS.  Use for
single-BVH (just one source) or combined (multiple sources) reverse-LBS.

Usage:
  python tools/reverse_lbs_solve_multi.py \
      --src walk.bvh data/motion_cache/walk/_iter50_nobone \
      --out reverse_lbs_results_walk

  python tools/reverse_lbs_solve_multi.py \
      --src walk.bvh data/motion_cache/walk/_iter50_nobone \
      --src run.bvh  data/motion_cache/run/_quads_tfl_skin \
      --out reverse_lbs_results_combined

`{cache_template}` is the per-region prefix; full path becomes
`{cache_template}/{REGION}` (note the trailing region subdir).  E.g. if
your dirs are `data/motion_cache/walk/L_UpLeg_iter50_nobone` etc., pass
`data/motion_cache/walk/_iter50_nobone` and we'll match by region name +
suffix.

Actually simpler — pass the FULL region path WITH placeholder `{REGION}`,
e.g. `data/motion_cache/walk/{REGION}_iter50_nobone`.  We substitute.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Reuse helpers from the single-BVH solver
from tools.reverse_lbs_solve import (
    resolve_body_name, collect_body_names, cache_body_transforms,
    load_muscle_cache, solve_endpoint_single_bone, solve_two_bone_lbs,
    solve_three_bone_lbs, chain_between, pick_intermediate_bone, aggregate,
)
from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH
from viewer.zygote_mesh_ui import _detect_bvh_tframe
import pickle


SKEL_XML = 'data/zygote_skel.xml'
TET_DIR = 'tet'
REGIONS = ['L_UpLeg', 'L_LowLeg', 'R_UpLeg', 'R_LowLeg']


def load_muscle_cache_multi(cache_dirs_per_source, muscle_name):
    """Concat per-source cache for one muscle along the frame axis."""
    all_wp = []
    shape = None
    total_frames = 0
    for cdir in cache_dirs_per_source:
        loaded = load_muscle_cache(cdir, muscle_name)
        if loaded is None:
            return None
        wp, sh, _ = loaded
        all_wp.append(wp)
        total_frames += wp.shape[0]
        if shape is None:
            shape = sh
    return np.concatenate(all_wp, axis=0), shape, total_frames


def process_muscle_multi(muscle_name, cache_dirs_per_source, tet_path,
                        body_T_cache, skel, out_dir):
    with open(tet_path, 'rb') as f:
        td = pickle.load(f)
    rest_wps = td.get('waypoints')
    asn = td.get('attach_skeleton_names')
    if rest_wps is None or asn is None:
        return None

    skel.resetPositions()
    resolved = []
    for s_idx, names in enumerate(asn):
        if len(names) < 2:
            resolved.append(None); continue
        o = resolve_body_name(skel, names[0])
        i = resolve_body_name(skel, names[1])
        if o is None or i is None or o not in body_T_cache or i not in body_T_cache:
            resolved.append(None); continue
        rest_levels = [np.asarray(rest_wps[s_idx][L], dtype=np.float64)
                       for L in range(len(rest_wps[s_idx]))]
        if not rest_levels or rest_levels[0].size == 0:
            resolved.append((o, i, None)); continue
        n_f = min(arr.shape[0] for arr in rest_levels)
        mid_L = len(rest_levels) // 2
        mid_world = rest_levels[mid_L][:n_f].mean(axis=0)
        m = pick_intermediate_bone(skel, o, i, mid_world)
        if m is not None and m not in body_T_cache:
            m = None
        resolved.append((o, i, m))

    cached = load_muscle_cache_multi(cache_dirs_per_source, muscle_name)
    if cached is None:
        return None
    all_wp, shape, T_n = cached

    records = []
    offset = 0
    for s_idx, level_counts in enumerate(shape):
        n_levels = len(level_counts)
        if resolved[s_idx] is None or n_levels < 2:
            offset += sum(level_counts); continue
        o_body, i_body, m_body = resolved[s_idx]
        has_mid = m_body is not None

        rest_levels = [np.asarray(rest_wps[s_idx][L], dtype=np.float64)
                       for L in range(n_levels)]
        n_f_common = min(arr.shape[0] for arr in rest_levels)
        rest_stack = np.stack([arr[:n_f_common] for arr in rest_levels], axis=0)
        segs = np.linalg.norm(rest_stack[1:] - rest_stack[:-1], axis=2)
        cum_O = np.zeros((n_levels, n_f_common), dtype=np.float64)
        cum_O[1:] = np.cumsum(segs, axis=0)
        total_len = cum_O[-1]

        R_o = body_T_cache[o_body][:, :3, :3]
        t_o = body_T_cache[o_body][:, :3, 3]
        R_i = body_T_cache[i_body][:, :3, :3]
        t_i = body_T_cache[i_body][:, :3, 3]
        if has_mid:
            R_m = body_T_cache[m_body][:, :3, :3]
            t_m = body_T_cache[m_body][:, :3, 3]
            skel.resetPositions()
            m_rest = np.asarray(skel.getBodyNode(m_body).getWorldTransform().translation())
            dist_per_level = np.linalg.norm(rest_stack - m_rest, axis=-1)
            mid_level_per_fiber = np.argmin(dist_per_level, axis=0)
            u_m_per_fiber = np.zeros(n_f_common, dtype=np.float64)
            for f in range(n_f_common):
                if total_len[f] > 1e-9:
                    u_m_per_fiber[f] = cum_O[mid_level_per_fiber[f], f] / total_len[f]
                else:
                    u_m_per_fiber[f] = 0.5
                u_m_per_fiber[f] = float(np.clip(u_m_per_fiber[f], 0.1, 0.9))

        for L_idx in range(n_levels):
            n_f_this = level_counts[L_idx]
            for f in range(n_f_this):
                wp_idx = offset + sum(level_counts[:L_idx]) + f
                P_world = all_wp[:, wp_idx, :]
                if f < n_f_common and total_len[f] > 1e-9:
                    u = float(cum_O[L_idx, f] / total_len[f])
                else:
                    u = float(L_idx) / max(n_levels - 1, 1)
                u = float(np.clip(u, 0.0, 1.0))

                rec = {
                    'stream': s_idx, 'level': L_idx, 'fiber': f,
                    'origin_body': o_body, 'insertion_body': i_body,
                    'mid_body': m_body if has_mid else None,
                    'local_o': np.zeros(3, dtype=np.float32),
                    'local_m': np.zeros(3, dtype=np.float32),
                    'local_i': np.zeros(3, dtype=np.float32),
                    'w_o': 0.0, 'w_m': 0.0, 'w_i': 0.0,
                }
                if L_idx == 0:
                    local_o, resid = solve_endpoint_single_bone(P_world, R_o, t_o)
                    rec['local_o'] = local_o.astype(np.float32); rec['w_o'] = 1.0
                elif L_idx == n_levels - 1:
                    local_i, resid = solve_endpoint_single_bone(P_world, R_i, t_i)
                    rec['local_i'] = local_i.astype(np.float32); rec['w_i'] = 1.0
                else:
                    w_a = u; w_b = 1.0 - u
                    l2o, l2i, r2 = solve_two_bone_lbs(P_world, R_o, t_o, R_i, t_i, w_a, w_b)
                    r2_p95 = float(np.percentile(r2, 95))
                    r3_p95 = np.inf
                    if has_mid and f < n_f_common:
                        u_m = u_m_per_fiber[f]
                        if u <= u_m:
                            w_o_ = max(0.0, 1.0 - u / u_m)
                            w_m_ = u / u_m
                            w_i_ = 0.0
                        else:
                            w_o_ = 0.0
                            w_m_ = (1.0 - u) / (1.0 - u_m)
                            w_i_ = (u - u_m) / (1.0 - u_m)
                        l3o, l3m, l3i, r3 = solve_three_bone_lbs(
                            P_world, R_o, t_o, R_m, t_m, R_i, t_i, w_o_, w_m_, w_i_)
                        r3_p95 = float(np.percentile(r3, 95))
                    if r3_p95 < r2_p95:
                        rec['local_o'] = l3o.astype(np.float32)
                        rec['local_m'] = l3m.astype(np.float32)
                        rec['local_i'] = l3i.astype(np.float32)
                        rec['w_o'] = w_o_; rec['w_m'] = w_m_; rec['w_i'] = w_i_
                        resid = r3
                    else:
                        rec['local_o'] = l2o.astype(np.float32)
                        rec['local_i'] = l2i.astype(np.float32)
                        rec['w_o'] = w_a; rec['w_i'] = w_b
                        resid = r2
                # diagnostics
                rec['resid_max'] = float(resid.max())
                rec['resid_rms'] = float(np.sqrt((resid**2).mean()))
                rec['resid_p95'] = float(np.percentile(resid, 95))
                records.append(rec)
        offset += sum(level_counts)

    if not records:
        return None
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f'{muscle_name}.npz'),
             records=np.array(records, dtype=object))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', nargs=2, action='append', required=True,
                    metavar=('BVH', 'CACHE_TEMPLATE'),
                    help='BVH path + cache template with {REGION} placeholder. '
                         'Specify multiple --src for combined solve.')
    ap.add_argument('--out', required=True)
    ap.add_argument('--muscles', nargs='*')
    args = ap.parse_args()

    skel_info, root_name, bvh_info, *_ = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)

    # Load each source's mocap + per-region cache dirs
    sources = []
    for bvh_path, cache_tmpl in args.src:
        if not os.path.exists(bvh_path):
            print(f'BVH not found: {bvh_path}')
            sys.exit(1)
        motion = MyBVH(bvh_path, bvh_info, skel, T_frame=_detect_bvh_tframe(bvh_path))
        cache_dirs = {R: cache_tmpl.replace('{REGION}', R) for R in REGIONS}
        sources.append({'bvh': bvh_path, 'mocap': motion.mocap_refs,
                        'cache_dirs': cache_dirs})
        print(f'  {bvh_path}: {motion.mocap_refs.shape[0]} frames')

    # Enumerate muscles from all sources, must exist in EVERY source
    muscle_first_dir = {}
    for src in sources:
        for R, cdir in src['cache_dirs'].items():
            for chunk in glob.glob(os.path.join(cdir, '*_chunk_0000.npz')):
                base = os.path.basename(chunk).replace('_chunk_0000.npz', '')
                muscle_first_dir.setdefault(base, {})[id(src)] = cdir
    muscle_to_cache_dirs = {}
    for muscle, src_dirs in muscle_first_dir.items():
        if len(src_dirs) == len(sources):
            muscle_to_cache_dirs[muscle] = [src_dirs[id(s)] for s in sources]
    print(f'Discovered {len(muscle_to_cache_dirs)} muscles (intersection across sources)')

    if args.muscles:
        sel = set(args.muscles)
        muscle_to_cache_dirs = {k: v for k, v in muscle_to_cache_dirs.items() if k in sel}

    # Build combined body_T_cache by concat across sources
    tet_paths = [os.path.join(TET_DIR, f'{m}_tet.npz')
                 for m in muscle_to_cache_dirs.keys()
                 if os.path.exists(os.path.join(TET_DIR, f'{m}_tet.npz'))]
    body_names = collect_body_names(tet_paths, skel)
    print(f'Caching body transforms for {len(body_names)} bodies across all sources...')
    body_T_parts = []
    for src in sources:
        body_T_parts.append(cache_body_transforms(skel, src['mocap'], body_names))
    body_T_combined = {n: np.concatenate([p[n] for p in body_T_parts], axis=0)
                       for n in body_names}

    per_muscle = {}
    for muscle_name, cdirs in sorted(muscle_to_cache_dirs.items()):
        tet_path = os.path.join(TET_DIR, f'{muscle_name}_tet.npz')
        if not os.path.exists(tet_path):
            continue
        recs = process_muscle_multi(muscle_name, cdirs, tet_path,
                                    body_T_combined, skel, args.out)
        if recs:
            max_max = max(r['resid_max'] for r in recs)
            p95 = float(np.percentile([r['resid_max'] for r in recs], 95))
            print(f'  {muscle_name}: {len(recs)} wp, max={max_max*1000:.1f}mm, p95={p95*1000:.1f}mm')
            per_muscle[muscle_name] = recs

    summary = aggregate(per_muscle)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved {len(per_muscle)} muscle .npz + summary to {args.out}/')


if __name__ == '__main__':
    main()
