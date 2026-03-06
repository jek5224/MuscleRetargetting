#!/usr/bin/env python3
"""Headless FEM baking over a systematic DOF grid for lower-body muscles.

Instead of extracting training data from BVH motions, this script samples
joint angle combinations and runs the FEM simulation for each pose.
Lower body has 7 DOFs per leg (hip 3, knee 1, ankle 3), making systematic
sampling feasible and more efficient than motion capture.

All 7 DOFs are sampled together in a single grid so that the same sample
index maps to a consistent skeleton pose across both L_UpLeg and L_LowLeg.

Samples are ordered by nearest-neighbor traversal starting from rest pose
(all zeros) so that consecutive poses are close in DOF space. This enables
warm-starting the FEM solver from the previous frame's deformation, avoiding
large vertex jumps and improving convergence.

Usage:
    python tools/bake_dof_grid.py --region L_UpLeg                 # one region
    python tools/bake_dof_grid.py --region L_LowLeg --start 5000   # resume
    python tools/bake_dof_grid.py --lhs-samples 20000              # more samples
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.bake_headless import (
    load_skeleton,
    load_skeleton_meshes,
    load_muscle_meshes,
    load_tet_meshes,
    init_soft_bodies,
    build_context,
    flush_bake_data,
)
from viewer.zygote_mesh_ui import find_inter_muscle_constraints, run_all_tet_sim_with_constraints

# ---------------------------------------------------------------------------
# DOF grid definitions — ranges based on biomechanics literature + BVH data
# ---------------------------------------------------------------------------
# DART BallJoint uses exponential map (rotation vector): 3 DOFs = axis * angle
# RevoluteJoint: 1 DOF = angle around the joint axis
#
# ROM sources:
#   CDC Normal Joint ROM Study (2003-2006)
#   Roaas & Andersson 1982 (Acta Orthop Scand)
#   Observed ranges from 82 LaFAN1 BVH files
#
# All values in radians.

# All 7 left-leg DOFs — always sampled together
ALL_DOF_NAMES = [
    "L_Femur_x", "L_Femur_y", "L_Femur_z",  # hip ball joint
    "L_Knee",                                  # knee revolute
    "L_Ankle_x", "L_Ankle_y", "L_Ankle_z",   # ankle ball joint
]

# Skeleton DOF indices
L_DOF_MAP = {
    "L_Femur_x":  6,
    "L_Femur_y":  7,
    "L_Femur_z":  8,
    "L_Knee":     9,
    "L_Ankle_x": 10,
    "L_Ankle_y": 11,
    "L_Ankle_z": 12,
}

# DOF ranges in radians — derived from observed BVH data + 10% margin.
# BallJoints use exponential map (rotation vector) which can span wide ranges.
DOF_RANGES = {
    # Hip (BallJoint exponential map components)
    "L_Femur_x":  (np.radians(-195), np.radians(195)),
    "L_Femur_y":  (np.radians(-216), np.radians(216)),
    "L_Femur_z":  (np.radians(-130), np.radians(140)),
    # Knee (RevoluteJoint)
    "L_Knee":     (np.radians(-20),  np.radians(190)),
    # Ankle (BallJoint exponential map components)
    "L_Ankle_x":  (np.radians(-60),  np.radians(65)),
    "L_Ankle_y":  (np.radians(-105), np.radians(80)),
    "L_Ankle_z":  (np.radians(-70),  np.radians(155)),
}

REGION_MUSCLES = {
    "L_UpLeg":  "tools/muscles_L_UpLeg.json",
    "L_LowLeg": "tools/muscles_L_LowLeg.json",
}

FLUSH_INTERVAL = 1000


def generate_latin_hypercube(n_samples, seed=42):
    """Generate Latin Hypercube samples over all 7 DOFs.

    Deterministic for a given (n_samples, seed) — same grid regardless
    of which region is being baked.
    """
    rng = np.random.default_rng(seed)
    n_dofs = len(ALL_DOF_NAMES)

    samples = np.zeros((n_samples, n_dofs))
    for d, name in enumerate(ALL_DOF_NAMES):
        lo, hi = DOF_RANGES[name]
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            lo_s = lo + (hi - lo) * perm[i] / n_samples
            hi_s = lo + (hi - lo) * (perm[i] + 1) / n_samples
            samples[i, d] = rng.uniform(lo_s, hi_s)
    return samples


def order_nearest_neighbor(samples):
    """Reorder samples by greedy nearest-neighbor starting from rest pose (origin).

    This ensures temporally smooth traversal so the FEM solver can warm-start
    from the previous frame's deformation.

    Uses normalized DOF space (each dimension scaled to [0,1]) for fair distance
    computation across DOFs with different ranges.
    """
    n = len(samples)
    if n <= 1:
        return samples, np.arange(n)

    # Normalize to [0,1] per DOF for fair distance
    scales = np.array([DOF_RANGES[name][1] - DOF_RANGES[name][0] for name in ALL_DOF_NAMES])
    offsets = np.array([DOF_RANGES[name][0] for name in ALL_DOF_NAMES])
    normed = (samples - offsets) / scales

    # Start from rest pose (origin in normalized space = -offsets/scales)
    origin_normed = -offsets / scales

    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.int64)

    # Find nearest to origin first
    dists_to_origin = np.sum((normed - origin_normed) ** 2, axis=1)
    current = np.argmin(dists_to_origin)
    order[0] = current
    visited[current] = True

    # Greedy nearest-neighbor
    for step in range(1, n):
        current_pt = normed[current]
        dists = np.sum((normed - current_pt) ** 2, axis=1)
        dists[visited] = np.inf
        current = np.argmin(dists)
        order[step] = current
        visited[current] = True

        if step % 1000 == 0:
            print(f"  Ordering: {step}/{n}...")

    ordered_samples = samples[order]

    # Report path smoothness
    steps = np.linalg.norm(np.diff(ordered_samples, axis=0), axis=1)
    print(f"  Path: mean step={np.degrees(steps.mean()):.1f}°, "
          f"max step={np.degrees(steps.max()):.1f}°, "
          f"total path={np.degrees(steps.sum()):.0f}°")

    return ordered_samples, order


def bake_region(region, samples, original_indices, skel, skeleton_meshes,
                mesh_info, args, start_idx, output_base):
    """Bake one region over the given DOF samples."""
    muscles_path = REGION_MUSCLES[region]
    muscle_meshes = load_muscle_meshes(muscles_path)
    load_tet_meshes(muscle_meshes)

    num_dofs = skel.getNumDofs()
    skel.setPositions(np.zeros(num_dofs))
    init_soft_bodies(muscle_meshes, skeleton_meshes, skel, mesh_info)

    ctx = build_context(skel, muscle_meshes, skeleton_meshes, mesh_info, args)
    print(f"[{region}] Finding inter-muscle constraints...")
    n_constraints = find_inter_muscle_constraints(ctx)
    print(f"       Found {n_constraints} constraints")

    active_muscles = {
        name: mobj for name, mobj in muscle_meshes.items()
        if mobj.soft_body is not None
    }
    print(f"[{region}] Active muscles: {len(active_muscles)}")

    cache_dir = os.path.join(output_base, region)
    os.makedirs(cache_dir, exist_ok=True)

    # Save metadata
    meta = {
        "region": region,
        "total_samples": len(samples) + start_idx,
        "dof_names": ALL_DOF_NAMES,
        "dof_indices": [L_DOF_MAP[n] for n in ALL_DOF_NAMES],
        "dof_ranges": {n: [float(lo), float(hi)] for n, (lo, hi) in DOF_RANGES.items()},
        "muscles": list(active_muscles.keys()),
        "settle_iters": args.settle_iters,
        "lhs_samples": args.lhs_samples,
        "seed": 42,
        "ordering": "nearest_neighbor",
    }
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Remove old chunks if starting fresh
    if start_idx == 0:
        import glob as glob_mod
        for old in glob_mod.glob(os.path.join(cache_dir, "*_chunk_*.npz")):
            os.remove(old)

    # Reset soft bodies to rest
    for mobj in active_muscles.values():
        mobj.soft_body.positions = mobj.soft_body.rest_positions.copy()
        mobj.tet_vertices = mobj.soft_body.rest_positions.astype(np.float32).copy()

    ctx._unified_arap_backend = None
    ctx._unified_sim_cache = None

    bake_data = {name: {} for name in active_muscles}
    dof_frames = []
    orig_idx_frames = []
    flush_count = start_idx // FLUSH_INTERVAL
    bake_start = time.time()
    total = len(samples)
    dof_indices = [L_DOF_MAP[n] for n in ALL_DOF_NAMES]

    print(f"[{region}] Baking {total} poses (warm-started, nearest-neighbor order)...")

    for i, dof_vals in enumerate(samples):
        sample_idx = i + start_idx

        # Build full pose
        pose = np.zeros(num_dofs)
        for idx, val in zip(dof_indices, dof_vals):
            pose[idx] = val
        skel.setPositions(pose)

        # NOTE: soft body positions carry over from previous sample (warm-start)

        for mobj in active_muscles.values():
            mobj.waypoints_from_tet_sim = False
            mobj._baking_mode = True

        run_all_tet_sim_with_constraints(
            ctx, max_iterations=args.settle_iters, tolerance=1e-4
        )

        for mname, mobj in active_muscles.items():
            bake_data[mname][sample_idx] = mobj.soft_body.get_positions().astype(np.float32)
        dof_frames.append(dof_vals.astype(np.float32))
        orig_idx_frames.append(original_indices[i + start_idx])

        for mobj in active_muscles.values():
            mobj.waypoints_from_tet_sim = True
            mobj._baking_mode = False

        # Periodic flush
        n_accumulated = sum(len(fd) for fd in bake_data.values())
        if n_accumulated >= FLUSH_INTERVAL * len(active_muscles):
            print(f"  Flushing chunk {flush_count} to disk...")
            dof_path = os.path.join(cache_dir, f"dofs_chunk_{flush_count:04d}.npz")
            frame_indices = sorted(bake_data[next(iter(bake_data))].keys())
            np.savez_compressed(
                dof_path,
                sample_indices=np.array(frame_indices, dtype=np.int32),
                original_indices=np.array(orig_idx_frames, dtype=np.int32),
                dof_values=np.array(dof_frames, dtype=np.float32),
            )
            dof_frames.clear()
            orig_idx_frames.clear()
            flush_count = flush_bake_data(bake_data, cache_dir, flush_count)

        done = i + 1
        if done % 10 == 0 or done == total:
            elapsed = time.time() - bake_start
            avg = elapsed / done
            remaining = avg * (total - done)
            print(
                f"  [{region}] Sample {sample_idx} ({done}/{total})  "
                f"{elapsed:.1f}s elapsed  {avg:.2f}s/sample  "
                f"ETA {remaining:.0f}s",
                flush=True,
            )

    # Final flush
    if any(len(fd) > 0 for fd in bake_data.values()):
        dof_path = os.path.join(cache_dir, f"dofs_chunk_{flush_count:04d}.npz")
        frame_indices = sorted(bake_data[next(iter(bake_data))].keys())
        np.savez_compressed(
            dof_path,
            sample_indices=np.array(frame_indices, dtype=np.int32),
            original_indices=np.array(orig_idx_frames, dtype=np.int32),
            dof_values=np.array(dof_frames, dtype=np.float32),
        )
        dof_frames.clear()
        orig_idx_frames.clear()
        flush_count = flush_bake_data(bake_data, cache_dir, flush_count)

    done_marker = os.path.join(cache_dir, ".done")
    with open(done_marker, "w") as f:
        f.write(f"{total} samples\n")

    elapsed = time.time() - bake_start
    print(f"[{region}] Done: {total} samples in {elapsed:.1f}s "
          f"({elapsed/max(total,1):.2f}s/sample)")

    return len(active_muscles)


def main():
    parser = argparse.ArgumentParser(description="Headless FEM baking over DOF grid")
    parser.add_argument(
        "--region", required=True, choices=["L_UpLeg", "L_LowLeg"],
        help="Region to bake (run one at a time for GPU ARAP)",
    )
    parser.add_argument(
        "--lhs-samples", type=int, default=10000,
        help="Number of LHS samples over all 7 DOFs (default: 10000)",
    )
    parser.add_argument(
        "--settle-iters", type=int, default=50,
        help="Simulation iterations per pose (default: 50)",
    )
    parser.add_argument(
        "--constraint-threshold", type=float, default=0.015,
        help="Inter-muscle constraint distance in meters (default: 0.015)",
    )
    parser.add_argument(
        "--backend", choices=["auto", "taichi", "gpu", "cpu"], default="auto",
        help="ARAP solver backend (default: auto)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output base directory (default: data/motion_cache/dof_grid)",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start sample index in traversal order (for resuming, default: 0)",
    )
    args = parser.parse_args()

    output_base = args.output_dir or os.path.join("data", "motion_cache", "dof_grid")

    # Generate the same LHS grid (deterministic seed)
    print(f"Generating {args.lhs_samples} LHS samples over {len(ALL_DOF_NAMES)} DOFs...")
    raw_samples = generate_latin_hypercube(args.lhs_samples)
    for name in ALL_DOF_NAMES:
        lo, hi = DOF_RANGES[name]
        print(f"  {name:15s}: [{np.degrees(lo):6.1f}°, {np.degrees(hi):6.1f}°]")

    # Order by nearest-neighbor from rest pose for smooth warm-starting
    print(f"\nOrdering samples by nearest-neighbor from rest pose...")
    ordered_samples, order = order_nearest_neighbor(raw_samples)

    # Save shared grid (unordered + order mapping) for viewer
    os.makedirs(output_base, exist_ok=True)
    grid_path = os.path.join(output_base, "dof_grid.npz")
    np.savez_compressed(
        grid_path,
        dof_names=np.array(ALL_DOF_NAMES),
        dof_indices=np.array([L_DOF_MAP[n] for n in ALL_DOF_NAMES]),
        raw_samples=raw_samples.astype(np.float32),
        ordered_samples=ordered_samples.astype(np.float32),
        traversal_order=order.astype(np.int32),
    )
    print(f"Grid saved to {grid_path}")

    # Slice for resuming
    samples_to_bake = ordered_samples[args.start:]
    if args.start > 0:
        print(f"Resuming from traversal step {args.start}, {len(samples_to_bake)} remaining")

    if len(samples_to_bake) == 0:
        print("No samples to bake.")
        return

    # Load shared infrastructure
    skel, bvh_info, mesh_info = load_skeleton()
    skeleton_meshes = load_skeleton_meshes()

    print(f"\n{'='*60}")
    print(f"  Baking region: {args.region}")
    print(f"{'='*60}\n")
    bake_region(args.region, samples_to_bake, order, skel, skeleton_meshes,
                mesh_info, args, args.start, output_base)

    print("\nDone!")


if __name__ == "__main__":
    main()
