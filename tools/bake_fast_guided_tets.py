#!/usr/bin/env python3
"""Fast collision-free tet bake guided by the viewer's ordinary LBS.

This is intentionally not a contact simulator.  Each muscle is processed
independently:

1. ordinary per-vertex LBS supplies a plausible pose and anatomical placement;
2. saved attachment vertices are overwritten by their exact bone targets;
3. a signed-Jacobian barrier repairs only invalid or severely distorted tets;
4. the closest barrier-valid result to LBS is written to the normal cache.

The method accepts intersecting source anatomy because it never attempts to
infer an ordering between overlapping muscle or bone surfaces.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.bvhparser import MyBVH
from tools.bake_headless import (
    FLUSH_INTERVAL,
    flush_bake_data,
    init_soft_bodies,
    load_muscle_meshes,
    load_skeleton,
    load_skeleton_meshes,
    load_tet_meshes,
    patch_waypoints,
)
from viewer.fem_sim import _untangle_lbs_positions
from viewer.zygote_mesh_ui import _detect_bvh_tframe


def build_rest_contact_graph(active, threshold=0.006, min_patch=12):
    """Sparse reciprocal surface contacts from the supplied rest anatomy."""
    surfaces = {}
    for name, mobj in active.items():
        faces = np.asarray(mobj.tet_faces, dtype=np.int32)
        ids = np.unique(faces)
        surfaces[name] = (ids, np.asarray(
            mobj.soft_body.rest_positions, dtype=np.float64)[ids])
    directed = {}
    names = sorted(surfaces)
    for ai, a in enumerate(names):
        ids_a, pa = surfaces[a]
        for b in names[ai + 1:]:
            ids_b, pb = surfaces[b]
            da, ja = cKDTree(pb).query(pa)
            db, ib = cKDTree(pa).query(pb)
            ka = np.where(da < threshold)[0]
            kb = np.where(db < threshold)[0]
            if len(ka) >= min_patch and len(kb) >= min_patch:
                directed[(a, b)] = np.column_stack(
                    (ids_a[ka], ids_b[ja[ka]]))
                directed[(b, a)] = np.column_stack(
                    (ids_b[kb], ids_a[ib[kb]]))
    # Deduplicate reciprocal observations into undirected material pairs.
    records = {}
    for (a, b), pairs in directed.items():
        for ia, ib in pairs:
            key = (a, int(ia), b, int(ib))
            reverse = (b, int(ib), a, int(ia))
            records[min(key, reverse)] = None
    graph = []
    for a, ia, b, ib in records:
        delta = (active[a].soft_body.rest_positions[ia]
                 - active[b].soft_body.rest_positions[ib])
        dist = float(np.linalg.norm(delta))
        if dist < 1e-8:
            continue
        graph.append((a, ia, b, ib, delta / dist, dist))
    return graph


def build_shared_displacement_field(active, radius=0.015, sigma=0.008):
    """Rest-space cross-muscle neighborhood for a common bundle warp.

    Each vertex samples at most one nearest material point from every other
    muscle. Thus dense muscles cannot dominate merely through tessellation.
    """
    names = sorted(active)
    offsets = {}
    rest_parts = []
    offset = 0
    for name in names:
        rest = np.asarray(active[name].soft_body.rest_positions,
                          dtype=np.float64)
        offsets[name] = (offset, offset + len(rest))
        rest_parts.append(rest)
        offset += len(rest)
    rest_all = np.concatenate(rest_parts)
    neighbor_ids = np.full((len(rest_all), len(names)), -1, dtype=np.int32)
    weights = np.zeros((len(rest_all), len(names)), dtype=np.float64)
    for column, name in enumerate(names):
        start, end = offsets[name]
        source = rest_all[start:end]
        distance, local = cKDTree(source).query(rest_all)
        keep = distance <= radius
        neighbor_ids[keep, column] = start + local[keep]
        weights[keep, column] = np.exp(
            -0.5 * (distance[keep] / sigma) ** 2)
    return names, offsets, rest_all, neighbor_ids, weights


def apply_shared_displacement_field(positions, field, iterations=2,
                                    blend=0.75):
    """Apply one continuous displacement field to the complete muscle bundle."""
    names, offsets, rest_all, neighbor_ids, weights = field
    current = np.concatenate([positions[name] for name in names])
    displacement = current - rest_all
    valid = neighbor_ids >= 0
    safe_ids = np.maximum(neighbor_ids, 0)
    denominator = np.maximum(weights.sum(axis=1), 1e-12)
    for _ in range(iterations):
        averaged = np.sum(
            displacement[safe_ids] * weights[:, :, None], axis=1
        ) / denominator[:, None]
        displacement = ((1.0 - blend) * displacement
                        + blend * averaged)
    warped = rest_all + displacement
    return {name: warped[start:end].copy()
            for name, (start, end) in offsets.items()}


def apply_rest_order_compaction(positions, graph, fixed_masks, passes=4,
                                max_extra_gap=0.0005, min_gap=0.0002,
                                relaxation=0.35):
    """Sparse rest-neighbor distance tethers that compact fascial interfaces."""
    out = {name: value.copy() for name, value in positions.items()}
    for _ in range(passes):
        for a, ia, b, ib, normal, rest_gap in graph:
            pair_delta = out[a][ia] - out[b][ib]
            current_gap = float(np.linalg.norm(pair_delta))
            if current_gap < 1e-10:
                direction = normal
            else:
                direction = pair_delta / current_gap
            # A rest pair can be several millimetres apart because the source
            # anatomy was authored as separate overlapping meshes. Treat its
            # direction as ordering information, but compress its magnitude
            # to a thin fascial layer.
            desired = np.clip(rest_gap, min_gap, max_extra_gap)
            correction = relaxation * (desired - current_gap) * direction
            free_a = not fixed_masks[a][ia]
            free_b = not fixed_masks[b][ib]
            if free_a and free_b:
                out[a][ia] += 0.5 * correction
                out[b][ib] -= 0.5 * correction
            elif free_a:
                out[a][ia] += correction
            elif free_b:
                out[b][ib] -= correction
    return out


def tet_jacobians(rest, tets, positions):
    rest = np.asarray(rest, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int32)
    positions = np.asarray(positions, dtype=np.float64)
    r = rest[tets]
    x = positions[tets]
    Dr = np.stack((r[:, 0] - r[:, 3], r[:, 1] - r[:, 3],
                   r[:, 2] - r[:, 3]), axis=-1)
    Ds = np.stack((x[:, 0] - x[:, 3], x[:, 1] - x[:, 3],
                   x[:, 2] - x[:, 3]), axis=-1)
    det_r = np.linalg.det(Dr)
    edge_lengths = np.stack([
        np.linalg.norm(r[:, i] - r[:, j], axis=1)
        for i, j in ((0, 1), (0, 2), (0, 3),
                     (1, 2), (1, 3), (2, 3))], axis=1)
    quality = (np.abs(det_r) / 6.0) / np.maximum(
        np.max(edge_lengths, axis=1) ** 3, 1e-30)
    valid = quality > 1e-5
    # Match the simulation's degenerate-tet filter. Near-zero source elements
    # have undefined relative Jacobians and are removed by every FEM backend.
    return np.linalg.det(Ds[valid]) / det_r[valid]


def guided_valid_pose(mobj, skel, j_floor, j_ceiling, max_iterations,
                      projection_sweeps):
    """Return exact-attachment LBS, repaired only when its tets need it."""
    sb = mobj.soft_body
    # Use the same authoritative LBS path as the interactive viewer. The older
    # standalone helper reconstructs transform conventions independently and
    # can produce a different, non-repairable embedding.
    mobj._update_tet_positions_from_skeleton(skel)
    target = np.asarray(sb.positions, dtype=np.float64).copy()
    fixed = np.asarray(sb.fixed_indices, dtype=np.int32)
    if len(fixed):
        target[fixed] = np.asarray(sb.fixed_targets, dtype=np.float64)

    rest = np.asarray(sb.rest_positions, dtype=np.float64)
    tets = np.asarray(sb.tetrahedra, dtype=np.int32)
    J = tet_jacobians(rest, tets, target)
    if np.all((J >= j_floor) & (J <= j_ceiling)):
        return target, J, False

    repaired, _ = _untangle_lbs_positions(
        rest, tets, target, fixed,
        max_iterations=max_iterations,
        j_floor=j_floor, j_ceiling=j_ceiling,
        report_residual=False,
        max_projection_sweeps=projection_sweeps)
    # Reassert hard constraints after numerical optimization.
    if len(fixed):
        repaired[fixed] = np.asarray(sb.fixed_targets, dtype=np.float64)
    final_j = tet_jacobians(rest, tets, repaired)
    if np.any(final_j <= 0.02):
        # A deliberately bounded fast pass may not resolve the large frame-0
        # attachment jump. Retry this muscle robustly; never write an invalid
        # tet cache merely to report a shorter runtime.
        repaired, _ = _untangle_lbs_positions(
            rest, tets, target, fixed,
            max_iterations=max(800, max_iterations),
            j_floor=j_floor, j_ceiling=j_ceiling,
            report_residual=False,
            max_projection_sweeps=max(4000, projection_sweeps))
        if len(fixed):
            repaired[fixed] = np.asarray(sb.fixed_targets, dtype=np.float64)
        final_j = tet_jacobians(rest, tets, repaired)
        retry = 0
        while np.any(final_j <= 0.0) and retry < 3:
            repaired, _ = _untangle_lbs_positions(
                rest, tets, repaired, fixed,
                max_iterations=2000,
                j_floor=0.12 + 0.04 * retry, j_ceiling=j_ceiling,
                report_residual=False,
                max_projection_sweeps=max(4000, projection_sweeps))
            if len(fixed):
                repaired[fixed] = np.asarray(
                    sb.fixed_targets, dtype=np.float64)
            final_j = tet_jacobians(rest, tets, repaired)
            retry += 1
    if np.any(final_j <= 0.0):
        raise RuntimeError(
            f"{getattr(mobj, 'name', 'muscle')}: validity repair failed, "
            f"minJ={final_j.min():.6g}")
    return repaired, final_j, True


def main():
    p = argparse.ArgumentParser(
        description="Fast independent-tet bake with no collision handling")
    p.add_argument('--bvh', default='data/motion/run.bvh')
    p.add_argument('--muscles', default='.muscles_L_UpLeg.json')
    p.add_argument('--region-tag', default='L_UpLeg_fast_guided_tets')
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--end-frame', type=int, default=4)
    p.add_argument('--tet-dir', default='tet')
    p.add_argument('--output-root', default='.bake_outputs/motion_cache')
    p.add_argument('--j-floor', type=float, default=0.08)
    p.add_argument('--j-ceiling', type=float, default=5.0)
    p.add_argument('--repair-iters', type=int, default=30)
    p.add_argument('--projection-sweeps', type=int, default=150)
    p.add_argument('--contact-threshold', type=float, default=0.006)
    p.add_argument('--compact-passes', type=int, default=8)
    p.add_argument('--compact-strength', type=float, default=0.65)
    p.add_argument('--max-contact-gap', type=float, default=0.0005)
    p.add_argument('--max-compact-move', type=float, default=0.005)
    p.add_argument('--no-compact', action='store_true')
    p.add_argument('--bundle-radius', type=float, default=0.015)
    p.add_argument('--bundle-sigma', type=float, default=0.008)
    p.add_argument('--bundle-iterations', type=int, default=2)
    p.add_argument('--bundle-blend', type=float, default=0.75)
    p.add_argument('--skip-waypoints', action='store_true')
    args = p.parse_args()

    print("[1/6] Loading skeleton and saved tet muscles...")
    skel, bvh_info, mesh_info = load_skeleton()
    skeleton_meshes = load_skeleton_meshes()
    muscle_meshes = load_muscle_meshes(args.muscles)
    load_tet_meshes(muscle_meshes, tet_dir=args.tet_dir)
    skel.setPositions(np.zeros(skel.getNumDofs()))
    init_soft_bodies(muscle_meshes, skeleton_meshes, skel, mesh_info)
    active = {name: muscle for name, muscle in muscle_meshes.items()
              if muscle.soft_body is not None}
    print(f"      active muscles: {len(active)}")
    bundle_field = None if args.no_compact else \
        build_shared_displacement_field(
            active, radius=args.bundle_radius, sigma=args.bundle_sigma)
    if bundle_field is not None:
        valid = bundle_field[3] >= 0
        print(f"      shared bundle field: {valid.sum()} cross-muscle samples")

    print("[2/6] Loading motion...")
    motion = MyBVH(args.bvh, bvh_info, skel,
                   T_frame=_detect_bvh_tframe(args.bvh))
    n_frames = len(motion.mocap_refs)
    start = max(0, args.start_frame)
    end = min(args.end_frame, n_frames - 1)
    if end < start:
        raise ValueError(f"empty frame range {start}..{end}")

    stem = os.path.splitext(os.path.basename(args.bvh))[0]
    cache_dir = os.path.join(args.output_root, stem, args.region_tag)
    os.makedirs(cache_dir, exist_ok=True)
    for old in glob.glob(os.path.join(cache_dir, '*_chunk_*.npz')):
        os.remove(old)

    print(f"[3/6] Baking {start}..{end} without muscle/bone contact...")
    bake_data = {name: {} for name in active}
    flush_count = 0
    repaired_total = 0
    worst_j = np.inf
    t0 = time.time()
    for frame in range(start, end + 1):
        skel.setPositions(motion.mocap_refs[frame].copy())
        frame_repaired = 0
        frame_min_j = np.inf
        frame_positions = {}
        for name, mobj in active.items():
            mobj._update_fixed_targets_from_skeleton(skeleton_meshes, skel)
            pos, J, repaired = guided_valid_pose(
                mobj, skel, args.j_floor, args.j_ceiling, args.repair_iters,
                args.projection_sweeps)
            frame_positions[name] = pos
            frame_repaired += int(repaired)
            frame_min_j = min(frame_min_j, float(J.min()))
        if bundle_field is not None:
            precompact = {
                name: value.copy() for name, value in frame_positions.items()}
            frame_positions = apply_shared_displacement_field(
                frame_positions, bundle_field,
                iterations=args.bundle_iterations,
                blend=args.bundle_blend)
            for name in frame_positions:
                displacement = frame_positions[name] - precompact[name]
                length = np.linalg.norm(displacement, axis=1)
                scale = np.minimum(
                    1.0, args.max_compact_move
                    / np.maximum(length, 1e-12))
                frame_positions[name] = (
                    precompact[name] + displacement * scale[:, None])
            # Accept the strongest compactness fraction that preserves every
            # valid tet. This is a cheap per-muscle feasibility line search.
            frame_min_j = np.inf
            for name, mobj in active.items():
                sb = mobj.soft_body
                base = precompact[name]
                proposed = frame_positions[name]
                fixed = np.asarray(sb.fixed_indices, dtype=np.int32)
                pos = proposed.copy()
                if len(fixed):
                    pos[fixed] = np.asarray(
                        sb.fixed_targets, dtype=np.float64)
                J = tet_jacobians(sb.rest_positions, sb.tetrahedra, pos)
                if np.any(J <= 0.01):
                    pos, _ = _untangle_lbs_positions(
                        sb.rest_positions, sb.tetrahedra, pos, fixed,
                        max_iterations=800, j_floor=0.02,
                        max_projection_sweeps=4000,
                        report_residual=False)
                    if len(fixed):
                        pos[fixed] = np.asarray(
                            sb.fixed_targets, dtype=np.float64)
                    J = tet_jacobians(
                        sb.rest_positions, sb.tetrahedra, pos)
                if np.any(J <= 0.0):
                    # Last-resort feasible blend; normally the local repair
                    # above keeps the full attached surface proposal.
                    pos = base
                    J = tet_jacobians(
                        sb.rest_positions, sb.tetrahedra, base)
                    for scale in (0.5, 0.25, 0.125, 0.0625):
                        candidate = base + scale * (proposed - base)
                        if len(fixed):
                            candidate[fixed] = np.asarray(
                                sb.fixed_targets, dtype=np.float64)
                        candidate_j = tet_jacobians(
                            sb.rest_positions, sb.tetrahedra, candidate)
                        if np.all(candidate_j > 0.0):
                            pos, J = candidate, candidate_j
                            break
                if len(fixed):
                    pos[fixed] = np.asarray(
                        sb.fixed_targets, dtype=np.float64)
                    J = tet_jacobians(
                        sb.rest_positions, sb.tetrahedra, pos)
                if np.any(J <= 0.0):
                    raise RuntimeError(
                        f"{name}: compaction inverted tets, minJ={J.min()}")
                frame_positions[name] = pos
                frame_min_j = min(frame_min_j, float(J.min()))
        for name, pos in frame_positions.items():
            bake_data[name][frame] = pos.astype(np.float32)
        repaired_total += frame_repaired
        worst_j = min(worst_j, frame_min_j)
        done = frame - start + 1
        elapsed = time.time() - t0
        eta = elapsed / done * (end - frame)
        print(f"      frame {frame}: repaired {frame_repaired}/{len(active)}, "
              f"minJ={frame_min_j:.4f}, {elapsed/done:.2f}s/frame, "
              f"ETA {eta:.0f}s", flush=True)
        if sum(map(len, bake_data.values())) >= FLUSH_INTERVAL * len(active):
            flush_count = flush_bake_data(
                bake_data, cache_dir, flush_count)

    if any(bake_data[name] for name in bake_data):
        flush_count = flush_bake_data(bake_data, cache_dir, flush_count)

    print("[4/6] Patching fiber waypoints...")
    if not args.skip_waypoints:
        try:
            patch_waypoints(cache_dir, active, motion, skel)
        except Exception as exc:
            print(f"      waypoint patch skipped: {exc}")

    print("[5/6] Writing completion marker...")
    with open(os.path.join(cache_dir, '.done'), 'w') as marker:
        marker.write('done\n')
    print("[6/6] Done.")
    print(f"      output: {cache_dir}")
    print(f"      repaired muscle-frames: {repaired_total}")
    print(f"      global minJ: {worst_j:.6f}")
    print(f"      elapsed: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
