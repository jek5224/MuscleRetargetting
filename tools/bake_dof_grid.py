#!/usr/bin/env python3
"""Headless FEM baking over a systematic DOF grid for lower-body muscles.

Instead of extracting training data from BVH motions, this script samples
joint angle combinations on a grid and runs the FEM simulation for each.
Lower body has few DOFs (hip 3, knee 1, ankle 3 = 7 per leg), making
systematic sampling feasible and more efficient than motion capture.

Usage:
    python tools/bake_dof_grid.py --region L_UpLeg
    python tools/bake_dof_grid.py --region L_LowLeg --samples-per-dof 12
    python tools/bake_dof_grid.py --region L_UpLeg --backend taichi
"""
import argparse
import gc
import itertools
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

# Left leg DOF indices in the skeleton
L_DOF_MAP = {
    "L_Femur_x":  6,   # BallJoint component 0
    "L_Femur_y":  7,   # BallJoint component 1
    "L_Femur_z":  8,   # BallJoint component 2
    "L_Knee":     9,   # RevoluteJoint (axis=[1,0,0])
    "L_Ankle_x": 10,   # BallJoint component 0
    "L_Ankle_y": 11,   # BallJoint component 1
    "L_Ankle_z": 12,   # BallJoint component 2
}

# Anatomical ROM in radians (conservative ranges that cover functional movement)
# Hip:   flex ~120°, ext ~30°, abd ~45°, add ~30°, int rot ~45°, ext rot ~45°
# Knee:  flex ~140°, ext ~0° (revolute, positive = flexion)
# Ankle: dorsi ~20°, plantar ~50°, inv ~35°, ev ~20°
#
# Note: Because BallJoint uses exponential map, the x/y/z components are NOT
# simple Euler angles — they form a rotation vector. For moderate angles (<90°)
# they approximate Euler angles. For training data diversity we sample
# component-wise which gives good coverage of the rotation space.

DOF_RANGES = {
    # Hip (BallJoint exponential map components)
    "L_Femur_x":  (np.radians(-30),  np.radians(120)),   # extension(-) / flexion(+)
    "L_Femur_y":  (np.radians(-45),  np.radians(45)),    # int/ext rotation
    "L_Femur_z":  (np.radians(-45),  np.radians(30)),    # abduction(-) / adduction(+)
    # Knee (RevoluteJoint)
    "L_Knee":     (np.radians(0),    np.radians(140)),   # 0=extended, positive=flexion
    # Ankle (BallJoint exponential map components)
    "L_Ankle_x":  (np.radians(-20),  np.radians(20)),    # inversion/eversion component
    "L_Ankle_y":  (np.radians(-30),  np.radians(30)),    # inversion/eversion component
    "L_Ankle_z":  (np.radians(-20),  np.radians(50)),    # dorsiflexion(-) / plantarflexion(+)
}

# Which DOFs affect which regions
REGION_DOFS = {
    "L_UpLeg":  ["L_Femur_x", "L_Femur_y", "L_Femur_z", "L_Knee"],
    "L_LowLeg": ["L_Femur_x", "L_Femur_y", "L_Femur_z", "L_Knee",
                  "L_Ankle_x", "L_Ankle_y", "L_Ankle_z"],
}

REGION_MUSCLES = {
    "L_UpLeg":  "tools/muscles_L_UpLeg.json",
    "L_LowLeg": "tools/muscles_L_LowLeg.json",
}

FLUSH_INTERVAL = 1000


def generate_grid(dof_names, samples_per_dof):
    """Generate a full grid of DOF combinations.

    Returns:
        poses: np.ndarray of shape (N, len(dof_names)) with DOF values in radians
    """
    axes = []
    for name in dof_names:
        lo, hi = DOF_RANGES[name]
        axes.append(np.linspace(lo, hi, samples_per_dof))

    # Full grid via meshgrid
    grids = np.meshgrid(*axes, indexing='ij')
    # Flatten to (N, ndofs)
    poses = np.stack([g.ravel() for g in grids], axis=-1)
    return poses


def generate_latin_hypercube(dof_names, n_samples, seed=42):
    """Generate Latin Hypercube samples for DOF combinations.

    More efficient than full grid for high-dimensional spaces.
    """
    rng = np.random.default_rng(seed)
    n_dofs = len(dof_names)

    # Latin Hypercube: divide each dimension into n_samples equal strata
    samples = np.zeros((n_samples, n_dofs))
    for d, name in enumerate(dof_names):
        lo, hi = DOF_RANGES[name]
        # Random permutation of strata
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            # Random point within stratum perm[i]
            lo_s = lo + (hi - lo) * perm[i] / n_samples
            hi_s = lo + (hi - lo) * (perm[i] + 1) / n_samples
            samples[i, d] = rng.uniform(lo_s, hi_s)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Headless FEM baking over DOF grid")
    parser.add_argument(
        "--region", required=True, choices=list(REGION_DOFS.keys()),
        help="Muscle region to bake",
    )
    parser.add_argument(
        "--samples-per-dof", type=int, default=10,
        help="Samples per DOF axis for grid mode (default: 10)",
    )
    parser.add_argument(
        "--mode", choices=["auto", "grid", "lhs"], default="auto",
        help="Sampling mode: 'auto' picks grid for ≤4 DOFs else LHS (default: auto)",
    )
    parser.add_argument(
        "--lhs-samples", type=int, default=5000,
        help="Number of samples for LHS mode (default: 5000)",
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
        help="Output directory (default: data/motion_cache/dof_grid/<region>)",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start sample index (for resuming, default: 0)",
    )
    args = parser.parse_args()

    region = args.region
    dof_names = REGION_DOFS[region]
    muscles_path = REGION_MUSCLES[region]

    # Generate DOF samples
    # Auto-select mode: grid for ≤4 DOFs, LHS for more (grid explodes combinatorially)
    mode = args.mode
    if mode == "auto":
        mode = "grid" if len(dof_names) <= 4 else "lhs"
        print(f"Auto-selected mode: {mode} ({len(dof_names)} DOFs)")

    if mode == "grid":
        samples = generate_grid(dof_names, args.samples_per_dof)
        print(f"Grid sampling: {args.samples_per_dof} per DOF × {len(dof_names)} DOFs "
              f"= {len(samples)} poses")
    else:
        samples = generate_latin_hypercube(dof_names, args.lhs_samples)
        print(f"LHS sampling: {args.lhs_samples} poses × {len(dof_names)} DOFs")

    # Apply start offset for resuming
    if args.start > 0:
        samples = samples[args.start:]
        print(f"Resuming from sample {args.start}, {len(samples)} remaining")

    if len(samples) == 0:
        print("No samples to bake.")
        return

    # --- Load infrastructure (reuse from bake_headless) ---
    skel, bvh_info, mesh_info = load_skeleton()
    skeleton_meshes = load_skeleton_meshes()
    muscle_meshes = load_muscle_meshes(muscles_path)
    load_tet_meshes(muscle_meshes)

    # Reset to rest pose
    num_dofs = skel.getNumDofs()
    print(f"[6/8] Resetting skeleton to rest pose ({num_dofs} DOFs)...")
    skel.setPositions(np.zeros(num_dofs))
    init_soft_bodies(muscle_meshes, skeleton_meshes, skel, mesh_info)

    # Build context and find constraints
    ctx = build_context(skel, muscle_meshes, skeleton_meshes, mesh_info, args)
    print("[7/8] Finding inter-muscle constraints...")
    n_constraints = find_inter_muscle_constraints(ctx)
    print(f"       Found {n_constraints} constraints")

    # Active muscles
    active_muscles = {
        name: mobj for name, mobj in muscle_meshes.items()
        if mobj.soft_body is not None
    }
    print(f"\nActive muscles: {len(active_muscles)}")

    # Output directory
    cache_dir = args.output_dir or os.path.join("data", "motion_cache", "dof_grid", region)
    os.makedirs(cache_dir, exist_ok=True)

    # Save DOF metadata
    meta = {
        "region": region,
        "mode": args.mode,
        "samples_per_dof": args.samples_per_dof if args.mode == "grid" else None,
        "lhs_samples": args.lhs_samples if args.mode == "lhs" else None,
        "total_samples": len(samples) + args.start,
        "dof_names": dof_names,
        "dof_indices": [L_DOF_MAP[n] for n in dof_names],
        "dof_ranges": {n: [float(lo), float(hi)] for n, (lo, hi) in DOF_RANGES.items() if n in dof_names},
        "muscles": list(active_muscles.keys()),
        "settle_iters": args.settle_iters,
    }
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Remove old chunks if starting fresh
    if args.start == 0:
        import glob as glob_mod
        for old in glob_mod.glob(os.path.join(cache_dir, "*_chunk_*.npz")):
            os.remove(old)

    # Reset soft bodies
    for mobj in active_muscles.values():
        mobj.soft_body.positions = mobj.soft_body.rest_positions.copy()
        mobj.tet_vertices = mobj.soft_body.rest_positions.astype(np.float32).copy()

    ctx._unified_arap_backend = None
    ctx._unified_sim_cache = None

    # Bake accumulators
    bake_data = {name: {} for name in active_muscles}
    # Also accumulate the DOF values for each sample
    dof_frames = []
    flush_count = args.start // FLUSH_INTERVAL
    bake_start = time.time()
    total = len(samples)

    # Map DOF names to skeleton indices
    dof_indices = [L_DOF_MAP[n] for n in dof_names]

    print(f"\nBaking {total} poses...")
    print(f"Output: {cache_dir}/")
    print()

    for i, dof_vals in enumerate(samples):
        sample_idx = i + args.start

        # Build full pose (all zeros + the sampled DOFs)
        pose = np.zeros(num_dofs)
        for idx, val in zip(dof_indices, dof_vals):
            pose[idx] = val
        skel.setPositions(pose)

        # Disable waypoint updates during baking
        for mobj in active_muscles.values():
            mobj.waypoints_from_tet_sim = False
            mobj._baking_mode = True

        # Run simulation
        run_all_tet_sim_with_constraints(
            ctx, max_iterations=args.settle_iters, tolerance=1e-4
        )

        # Capture positions
        for mname, mobj in active_muscles.items():
            bake_data[mname][sample_idx] = mobj.soft_body.get_positions().astype(np.float32)

        # Store DOF values
        dof_frames.append(dof_vals.astype(np.float32))

        # Restore flags
        for mobj in active_muscles.values():
            mobj.waypoints_from_tet_sim = True
            mobj._baking_mode = False

        # Periodic flush
        n_accumulated = sum(len(fd) for fd in bake_data.values())
        if n_accumulated >= FLUSH_INTERVAL * len(active_muscles):
            print(f"  Flushing chunk {flush_count} to disk...")
            # Save DOF values alongside positions
            dof_path = os.path.join(cache_dir, f"dofs_chunk_{flush_count:04d}.npz")
            frame_indices = sorted(bake_data[next(iter(bake_data))].keys())
            np.savez_compressed(
                dof_path,
                sample_indices=np.array(frame_indices, dtype=np.int32),
                dof_values=np.array(dof_frames, dtype=np.float32),
            )
            dof_frames.clear()
            flush_count = flush_bake_data(bake_data, cache_dir, flush_count)

        # Progress
        done = i + 1
        if done % 10 == 0 or done == total:
            elapsed = time.time() - bake_start
            avg = elapsed / done
            remaining = avg * (total - done)
            print(
                f"  Sample {sample_idx} ({done}/{total})  "
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
            dof_values=np.array(dof_frames, dtype=np.float32),
        )
        dof_frames.clear()
        flush_count = flush_bake_data(bake_data, cache_dir, flush_count)

    # Completion marker
    done_marker = os.path.join(cache_dir, ".done")
    with open(done_marker, "w") as f:
        f.write(f"{total} samples\n")

    elapsed = time.time() - bake_start
    print(f"\nDone: {total} samples in {elapsed:.1f}s ({elapsed/max(total,1):.2f}s/sample)")
    print(f"Output: {cache_dir}/")


if __name__ == "__main__":
    main()
