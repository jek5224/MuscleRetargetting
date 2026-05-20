"""Reverse-LBS feasibility study from walk.bvh cache.

For each waypoint in every R/L UpLeg/LowLeg muscle cache, solve for the rest
local positions (local_O, local_I) of the underlying 2-bone LBS such that the
DART formula

    P_world(t) = w_O * (R_O(t) @ local_O + t_O(t))
               + w_I * (R_I(t) @ local_I + t_I(t))

reproduces the cached `P_world(t)` over all 132 walk frames in least-squares
sense.  Endpoints (first/last level of each stream) use single-body inverse
transform.

Per-waypoint residual statistics saved to `reverse_lbs_results/{muscle}.npz`.
Aggregate summary saved to `reverse_lbs_results/summary.json`.
"""
import argparse
import glob
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH
from viewer.zygote_mesh_ui import _detect_bvh_tframe


CACHE_REGIONS = [
    ('data/motion_cache/walk/L_UpLeg', 'L'),
    ('data/motion_cache/walk/L_LowLeg', 'L'),
    ('data/motion_cache/walk/R_UpLeg', 'R'),
    ('data/motion_cache/walk/R_LowLeg', 'R'),
]
TET_DIR = 'tet'
BVH_PATH = 'data/motion/walk.bvh'
SKEL_XML = 'data/zygote_skel.xml'
OUT_DIR = 'reverse_lbs_results'


def resolve_body_name(skel, name):
    """attach_skeleton_names entries may lack the trailing '0' DART suffix."""
    if not name:
        return None
    if skel.getBodyNode(name) is not None:
        return name
    cand = name + '0'
    if skel.getBodyNode(cand) is not None:
        return cand
    return None


def collect_body_names(muscle_paths, skel):
    """Walk every R/L muscle's tet, collect every body name referenced in
    attach_skeleton_names, plus all chain intermediates for 3-bone LBS.
    Returns sorted set of resolved DART body names."""
    bodies = set()
    for path in muscle_paths:
        with open(path, 'rb') as f:
            td = pickle.load(f)
        asn = td.get('attach_skeleton_names')
        if asn is None:
            continue
        for stream_names in asn:
            if len(stream_names) < 2:
                continue
            o = resolve_body_name(skel, stream_names[0])
            i = resolve_body_name(skel, stream_names[1])
            if o is None or i is None:
                continue
            bodies.add(o); bodies.add(i)
            for mid in chain_between(skel, o, i):
                bodies.add(mid)
    return sorted(bodies)


def cache_body_transforms(skel, mocap_refs, body_names):
    """Return dict[body_name] -> (T, 4, 4) world transforms across all frames."""
    n_frames = mocap_refs.shape[0]
    out = {n: np.zeros((n_frames, 4, 4), dtype=np.float64) for n in body_names}
    for t in range(n_frames):
        skel.setPositions(mocap_refs[t])
        for bn in body_names:
            b = skel.getBodyNode(bn)
            if b is None:
                continue
            out[bn][t] = np.asarray(b.getWorldTransform().matrix())
    return out


def load_muscle_cache(cache_dir, muscle_name):
    """Concatenate all chunk files for one muscle, sort by frame index.

    Returns (waypoints_xyz (T, n_wp, 3), shape_per_stream, frames (T,))."""
    files = sorted(glob.glob(os.path.join(cache_dir, f'{muscle_name}_chunk_*.npz')))
    if not files:
        return None
    frames = []
    wps = []
    shape = None
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        wf = d['waypoints_flat']
        n_wp = wf.shape[1] // 3
        wps.append(wf.reshape(wf.shape[0], n_wp, 3))
        frames.append(np.asarray(d['frames']))
        if shape is None:
            raw = d['waypoints_shape']
            sj = raw.item().decode() if hasattr(raw, 'item') else raw[()]
            shape = json.loads(sj)
    frames = np.concatenate(frames)
    wps = np.concatenate(wps, axis=0)
    order = np.argsort(frames)
    return wps[order], shape, frames[order]


def solve_endpoint_single_bone(P_world, R_b, t_b):
    """Endpoint waypoint single-bone inverse: P(t) = R_b(t) @ local + t_b(t).
    R^T orthonormal so closed form: local = mean_t(R_b(t)^T @ (P(t) - t_b(t)))."""
    diff = P_world - t_b  # (T, 3)
    local = np.einsum('tij,tj->i', np.transpose(R_b, (0, 2, 1)), diff) / R_b.shape[0]
    recon = np.einsum('tij,j->ti', R_b, local) + t_b
    resid = np.linalg.norm(P_world - recon, axis=1)
    return local, resid


def solve_two_bone_lbs(P_world, R_o, t_o, R_i, t_i, w_a, w_b):
    """Two-bone LBS least-squares: solve [local_o; local_i] (6 unknowns)
    minimising sum_t ||P(t) - w_a*(R_o(t)@local_o + t_o(t)) - w_b*(R_i(t)@local_i + t_i(t))||^2.

    Stacks 3T equations vs 6 unknowns and calls np.linalg.lstsq."""
    T_n = P_world.shape[0]
    b = P_world - (w_a * t_o + w_b * t_i)  # (T, 3)
    A = np.zeros((T_n, 3, 6), dtype=np.float64)
    A[:, :, :3] = w_a * R_o
    A[:, :, 3:] = w_b * R_i
    x, *_ = np.linalg.lstsq(A.reshape(-1, 6), b.reshape(-1), rcond=None)
    local_o, local_i = x[:3], x[3:]
    recon = (w_a * (np.einsum('tij,j->ti', R_o, local_o) + t_o)
             + w_b * (np.einsum('tij,j->ti', R_i, local_i) + t_i))
    resid = np.linalg.norm(P_world - recon, axis=1)
    return local_o, local_i, resid


def solve_three_bone_lbs(P_world, R_o, t_o, R_m, t_m, R_i, t_i, w_o, w_m, w_i):
    """Three-bone LBS least-squares: solve [local_o; local_m; local_i] (9 unknowns)."""
    T_n = P_world.shape[0]
    b = P_world - (w_o * t_o + w_m * t_m + w_i * t_i)
    A = np.zeros((T_n, 3, 9), dtype=np.float64)
    A[:, :, :3] = w_o * R_o
    A[:, :, 3:6] = w_m * R_m
    A[:, :, 6:] = w_i * R_i
    x, *_ = np.linalg.lstsq(A.reshape(-1, 9), b.reshape(-1), rcond=None)
    local_o, local_m, local_i = x[:3], x[3:6], x[6:]
    recon = (w_o * (np.einsum('tij,j->ti', R_o, local_o) + t_o)
             + w_m * (np.einsum('tij,j->ti', R_m, local_m) + t_m)
             + w_i * (np.einsum('tij,j->ti', R_i, local_i) + t_i))
    resid = np.linalg.norm(P_world - recon, axis=1)
    return local_o, local_m, local_i, resid


def chain_between(skel, origin_name, insertion_name):
    """Return list of body names on the kinematic path from origin → LCA → insertion,
    EXCLUSIVE of origin and insertion themselves.  Used to pick an intermediate
    bone for 3-bone LBS in biarticular/multi-joint muscles."""
    def ancestors(name):
        chain = []
        b = skel.getBodyNode(name)
        while b is not None:
            chain.append(b.getName())
            b = b.getParentBodyNode()
        return chain
    O_chain = ancestors(origin_name)
    I_chain = ancestors(insertion_name)
    O_set = set(O_chain)
    lca = next((n for n in I_chain if n in O_set), None)
    if lca is None:
        return []
    o_part = []
    for n in O_chain:
        if n == lca: break
        o_part.append(n)
    i_part = []
    for n in I_chain:
        if n == lca: break
        i_part.append(n)
    full = o_part + [lca] + list(reversed(i_part))
    return [n for n in full if n not in (origin_name, insertion_name)]


def pick_intermediate_bone(skel, origin_name, insertion_name, fiber_mid_world):
    """Pick an intermediate body for 3-bone LBS by proximity to fiber rest
    midpoint.  Returns None if no intermediate body exists in the kinematic
    chain between origin and insertion.  Whether to actually USE the result
    (vs. 2-bone) is decided per-waypoint by residual comparison."""
    chain = chain_between(skel, origin_name, insertion_name)
    if not chain:
        return None
    best = None
    best_d = np.inf
    for bn in chain:
        b = skel.getBodyNode(bn)
        if b is None:
            continue
        t = np.asarray(b.getWorldTransform().translation())
        d = np.linalg.norm(t - fiber_mid_world)
        if d < best_d:
            best_d = d
            best = bn
    return best


def process_muscle(muscle_name, cache_dir, tet_path, body_T_cache, skel, out_dir):
    """Solve reverse-LBS for one muscle. Returns list of per-waypoint records,
    or None if missing data."""
    with open(tet_path, 'rb') as f:
        td = pickle.load(f)
    rest_wps = td.get('waypoints')
    asn = td.get('attach_skeleton_names')
    if rest_wps is None or asn is None:
        return None

    # Resolve body names per stream. Reset skel to T-pose for intermediate-bone selection.
    skel.resetPositions()
    resolved = []
    for s_idx, stream_names in enumerate(asn):
        if len(stream_names) < 2:
            resolved.append(None); continue
        o = resolve_body_name(skel, stream_names[0])
        i = resolve_body_name(skel, stream_names[1])
        if o is None or i is None or o not in body_T_cache or i not in body_T_cache:
            resolved.append(None); continue
        # Find intermediate bone (if any) by chain. Use fiber mid-rest centroid
        # to pick the most relevant intermediate.
        rest_levels = [np.asarray(rest_wps[s_idx][L], dtype=np.float64) for L in range(len(rest_wps[s_idx]))]
        if not rest_levels or rest_levels[0].size == 0:
            resolved.append((o, i, None)); continue
        n_f_common = min(arr.shape[0] for arr in rest_levels)
        mid_L = len(rest_levels) // 2
        mid_world = rest_levels[mid_L][:n_f_common].mean(axis=0)
        m = pick_intermediate_bone(skel, o, i, mid_world)
        if m is not None and m not in body_T_cache:
            m = None
        resolved.append((o, i, m))

    cached = load_muscle_cache(cache_dir, muscle_name)
    if cached is None:
        return None
    all_wp, shape, _ = cached
    T_n = all_wp.shape[0]

    records = []
    offset = 0
    for s_idx, level_counts in enumerate(shape):
        n_levels = len(level_counts)
        if resolved[s_idx] is None or n_levels < 2:
            offset += sum(level_counts); continue
        o_body, i_body, m_body = resolved[s_idx]
        has_mid = m_body is not None

        # Build cum arc length per fiber from rest waypoints
        rest_levels = [np.asarray(rest_wps[s_idx][L], dtype=np.float64) for L in range(n_levels)]
        n_f_common = min(arr.shape[0] for arr in rest_levels)
        rest_stack = np.stack([arr[:n_f_common] for arr in rest_levels], axis=0)  # (L, F, 3)
        segs = np.linalg.norm(rest_stack[1:] - rest_stack[:-1], axis=2)  # (L-1, F)
        cum_O = np.zeros((n_levels, n_f_common), dtype=np.float64)
        cum_O[1:] = np.cumsum(segs, axis=0)
        total_len = cum_O[-1]  # (F,)

        R_o = body_T_cache[o_body][:, :3, :3]
        t_o = body_T_cache[o_body][:, :3, 3]
        R_i = body_T_cache[i_body][:, :3, :3]
        t_i = body_T_cache[i_body][:, :3, 3]
        if has_mid:
            R_m = body_T_cache[m_body][:, :3, :3]
            t_m = body_T_cache[m_body][:, :3, 3]
            # Per-fiber u_m: rest waypoint nearest to intermediate bone's REST translation.
            # Use first-frame body transform from cache as REST proxy (frame 0 is BVH frame 0,
            # but bone position changes little vs rest for proximal bones; using REST is fine).
            skel.resetPositions()
            m_rest = np.asarray(skel.getBodyNode(m_body).getWorldTransform().translation())
            # For each fiber, find which level is closest to m_rest at rest
            dist_per_level = np.linalg.norm(rest_stack - m_rest, axis=-1)  # (L, F)
            mid_level_per_fiber = np.argmin(dist_per_level, axis=0)  # (F,)
            # u_m fiber = arc length from origin to that level / total
            u_m_per_fiber = np.zeros(n_f_common, dtype=np.float64)
            for f in range(n_f_common):
                if total_len[f] > 1e-9:
                    u_m_per_fiber[f] = cum_O[mid_level_per_fiber[f], f] / total_len[f]
                else:
                    u_m_per_fiber[f] = 0.5
                # Clamp away from 0/1 to avoid degenerate tent at endpoints
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
                    rec['method'] = 'endpoint_o'
                elif L_idx == n_levels - 1:
                    local_i, resid = solve_endpoint_single_bone(P_world, R_i, t_i)
                    rec['local_i'] = local_i.astype(np.float32); rec['w_i'] = 1.0
                    rec['method'] = 'endpoint_i'
                else:
                    # Try 2-bone with arc-length weights as baseline
                    w_a = u; w_b = 1.0 - u
                    l2o, l2i, r2 = solve_two_bone_lbs(P_world, R_o, t_o, R_i, t_i, w_a, w_b)
                    r2_p95 = float(np.percentile(r2, 95))
                    # Try 3-bone if intermediate available
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
                    # Pick lower-residual per waypoint
                    if r3_p95 < r2_p95:
                        rec['local_o'] = l3o.astype(np.float32)
                        rec['local_m'] = l3m.astype(np.float32)
                        rec['local_i'] = l3i.astype(np.float32)
                        rec['w_o'] = w_o_; rec['w_m'] = w_m_; rec['w_i'] = w_i_
                        rec['method'] = '3bone'
                        resid = r3
                    else:
                        rec['local_o'] = l2o.astype(np.float32)
                        rec['local_i'] = l2i.astype(np.float32)
                        rec['w_o'] = w_a; rec['w_i'] = w_b
                        rec['method'] = '2bone'
                        resid = r2

                rec['resid_max'] = float(resid.max())
                rec['resid_rms'] = float(np.sqrt((resid**2).mean()))
                rec['resid_p95'] = float(np.percentile(resid, 95))
                records.append(rec)
        offset += sum(level_counts)

    if not records:
        return None
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{muscle_name}.npz')
    np.savez(out_path, records=np.array(records, dtype=object))
    return records


def aggregate(per_muscle):
    """Aggregate per-muscle residual statistics."""
    summary = {}
    for muscle, recs in per_muscle.items():
        if not recs:
            continue
        max_resids = np.array([r['resid_max'] for r in recs])
        rms_resids = np.array([r['resid_rms'] for r in recs])
        endpoint = [r for r in recs if r['w_o'] in (1.0, 0.0) and r['w_i'] in (1.0, 0.0)]
        intermediate = [r for r in recs if r not in endpoint]
        summary[muscle] = {
            'n_waypoints': len(recs),
            'n_endpoint': len(endpoint),
            'n_intermediate': len(intermediate),
            'max_resid_max_m': float(max_resids.max()),
            'p99_resid_max_m': float(np.percentile(max_resids, 99)),
            'p95_resid_max_m': float(np.percentile(max_resids, 95)),
            'p50_resid_max_m': float(np.percentile(max_resids, 50)),
            'mean_resid_rms_m': float(rms_resids.mean()),
            'max_resid_rms_m': float(rms_resids.max()),
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--muscles', nargs='*', help='Restrict to these muscle names (without .npz)')
    args = ap.parse_args()

    print('Loading skeleton + walk.bvh ...')
    skel_info, root_name, bvh_info, *_ = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)
    motion = MyBVH(BVH_PATH, bvh_info, skel, T_frame=_detect_bvh_tframe(BVH_PATH))
    mocap = motion.mocap_refs
    print(f'  {mocap.shape[0]} frames')

    # Enumerate muscles from cache dirs
    muscle_to_region = {}
    for cdir, side in CACHE_REGIONS:
        for chunk in glob.glob(os.path.join(cdir, '*_chunk_0000.npz')):
            base = os.path.basename(chunk).replace('_chunk_0000.npz', '')
            muscle_to_region[base] = cdir
    print(f'Discovered {len(muscle_to_region)} muscles')

    if args.muscles:
        sel = set(args.muscles)
        muscle_to_region = {k: v for k, v in muscle_to_region.items() if k in sel}
        print(f'  restricted to {len(muscle_to_region)} muscles')

    # Collect all body names we'll need
    tet_paths = [os.path.join(TET_DIR, f'{m}_tet.npz')
                 for m in muscle_to_region.keys()
                 if os.path.exists(os.path.join(TET_DIR, f'{m}_tet.npz'))]
    body_names = collect_body_names(tet_paths, skel)
    print(f'Caching body transforms for {len(body_names)} bodies across {mocap.shape[0]} frames ...')
    body_T_cache = cache_body_transforms(skel, mocap, body_names)

    per_muscle = {}
    for muscle_name, cache_dir in sorted(muscle_to_region.items()):
        tet_path = os.path.join(TET_DIR, f'{muscle_name}_tet.npz')
        if not os.path.exists(tet_path):
            print(f'  {muscle_name}: missing tet, skip')
            continue
        recs = process_muscle(muscle_name, cache_dir, tet_path, body_T_cache, skel, args.out)
        if recs:
            max_max = max(r['resid_max'] for r in recs)
            p95 = float(np.percentile([r['resid_max'] for r in recs], 95))
            print(f'  {muscle_name}: {len(recs)} wp, max={max_max*1000:.1f}mm, p95={p95*1000:.1f}mm')
            per_muscle[muscle_name] = recs

    summary = aggregate(per_muscle)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved summary to {args.out}/summary.json')

    # Print sorted by p95
    print('\n=== Per-muscle p95 residual (sorted) ===')
    for m, s in sorted(summary.items(), key=lambda kv: kv[1]['p95_resid_max_m']):
        print(f'  {m:35s}  n={s["n_waypoints"]:5d}  '
              f'p95={s["p95_resid_max_m"]*1000:7.2f}mm  '
              f'max={s["max_resid_max_m"]*1000:7.2f}mm')


if __name__ == '__main__':
    main()
