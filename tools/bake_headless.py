#!/usr/bin/env python3
"""Headless FEM muscle baking — replicates the viewer's bake pipeline without GUI/OpenGL.

Usage:
    python tools/bake_headless.py --bvh data/motion/dance.bvh
    python tools/bake_headless.py --bvh data/motion/walk1_subject1.bvh --settle-iters 80
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import trimesh

# Ensure project root is on sys.path so `from viewer.*` / `from core.*` work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from types import SimpleNamespace

from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH
from viewer.mesh_loader import MeshLoader
from viewer.zygote_mesh_ui import (
    find_inter_muscle_constraints,
    run_all_tet_sim_with_constraints,
    _detect_bvh_tframe,
    _flatten_waypoints,
)
from viewer.arap_backends import check_taichi_available, check_gpu_available

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SKEL_XML = "data/zygote_skel.xml"
ZYGOTE_DIR = "Zygote_Meshes_251229/"
MESH_SCALE = 0.01
FLUSH_INTERVAL = 20  # frames between disk flushes


def load_skeleton():
    """Build DART skeleton from XML and return (skel, bvh_info, mesh_info, skeleton_meshes)."""
    print("[1/8] Loading skeleton...")
    skel_info, root_name, bvh_info, _pd, mesh_info, _smpl = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)
    print(f"       DOFs: {skel.getNumDofs()}, Bodies: {skel.getNumBodyNodes()}")
    return skel, bvh_info, mesh_info


def load_skeleton_meshes():
    """Load all skeleton OBJ meshes from the Zygote directory."""
    print("[2/8] Loading skeleton meshes...")
    skeleton_meshes = {}
    skel_dir = os.path.join(ZYGOTE_DIR, "Skeleton")
    for fname in sorted(os.listdir(skel_dir)):
        if not fname.endswith(".obj"):
            continue
        name = fname.split(".")[0]
        path = os.path.join(skel_dir, fname)
        skeleton_meshes[name] = MeshLoader()
        skeleton_meshes[name].load(path)
        skeleton_meshes[name].color = np.array([0.9, 0.9, 0.9])
        skel_tri = trimesh.load_mesh(path)
        skel_tri.vertices *= MESH_SCALE
        skeleton_meshes[name].trimesh = skel_tri
    print(f"       Loaded {len(skeleton_meshes)} skeleton meshes")
    return skeleton_meshes


def load_muscle_meshes(muscles_path):
    """Load muscle OBJ meshes listed in the JSON file."""
    print(f"[3/8] Loading muscle meshes from {muscles_path}...")
    with open(muscles_path, "r") as f:
        muscle_list = json.load(f)

    muscle_meshes = {}
    for entry in muscle_list:
        name = entry["name"]
        path = entry["path"]
        if not os.path.exists(path):
            print(f"       WARNING: {path} not found, skipping {name}")
            continue
        muscle_meshes[name] = MeshLoader()
        muscle_meshes[name].load(path)
        muscle_meshes[name].color = np.array([0.8, 0.2, 0.2])
        mtri = trimesh.load_mesh(path)
        mtri.vertices *= MESH_SCALE
        muscle_meshes[name].trimesh = mtri

    muscle_meshes = dict(sorted(muscle_meshes.items()))
    print(f"       Loaded {len(muscle_meshes)} muscles")
    return muscle_meshes


def load_tet_meshes(muscle_meshes):
    """Load tetrahedron meshes for each muscle."""
    print("[4/8] Loading tet meshes...")
    loaded = 0
    for name, mobj in muscle_meshes.items():
        mobj.load_tetrahedron_mesh(name)
        if mobj.tet_vertices is not None:
            loaded += 1
        else:
            print(f"       WARNING: No tet mesh for {name}")
    print(f"       Loaded {loaded}/{len(muscle_meshes)} tet meshes")


def init_soft_bodies(muscle_meshes, skeleton_meshes, skel, mesh_info):
    """Initialize soft body simulation for each muscle."""
    print("[5/8] Initializing soft bodies...")
    count = 0
    for name, mobj in muscle_meshes.items():
        if mobj.tet_vertices is None:
            continue
        mobj.init_soft_body(
            skeleton_meshes=skeleton_meshes,
            skeleton=skel,
            mesh_info=mesh_info,
        )
        if mobj.soft_body is not None:
            count += 1
    print(f"       Initialized {count} soft bodies")
    return count


def build_context(skel, muscle_meshes, skeleton_meshes, mesh_info, args):
    """Build a lightweight SimpleNamespace that mimics the viewer for baking functions."""
    # Resolve backend: auto picks taichi > gpu > cpu (same priority as viewer)
    backend = args.backend
    if backend == "auto":
        if check_taichi_available():
            backend = "taichi"
        elif check_gpu_available():
            backend = "gpu"
        else:
            backend = "cpu"
    use_taichi = backend == "taichi"
    use_gpu = backend == "gpu"
    print(f"ARAP backend: {backend.upper()}")

    sim_method = getattr(args, 'sim_method', 'arap')
    use_fem = sim_method in ('fem', 'vbd', 'pn')
    use_vbd = sim_method == 'vbd'
    use_pn = sim_method == 'pn'
    if use_pn:
        print("Simulation method: Projected Newton (CG + inversion-safe line search)")
    elif use_vbd:
        print("Simulation method: VBD (Vertex Block Descent)")
    elif use_fem:
        print("Simulation method: FEM (Neo-Hookean XPBD)")

    ctx = SimpleNamespace(
        env=SimpleNamespace(skel=skel, mesh_info=mesh_info),
        zygote_muscle_meshes=muscle_meshes,
        zygote_skeleton_meshes=skeleton_meshes,
        inter_muscle_constraints=[],
        inter_muscle_constraint_threshold=args.constraint_threshold,
        coupled_as_unified_volume=True,
        use_gpu_arap=use_gpu,
        use_taichi_arap=use_taichi,
        use_muscle_aware_arap=True,
        use_fem_sim=use_fem,
        use_vbd_sim=use_vbd,
        use_pn_sim=use_pn,
        fem_youngs_modulus=500.0,
        fem_poisson_ratio=0.40,
        fem_collision_kappa=1e4,
        fem_volume_penalty=5000.0,
        fem_contact_threshold=args.constraint_threshold,
        fem_outer_iterations=3,
        fem_load_steps=getattr(args, 'load_steps', 10),
        motion_settle_iters=args.settle_iters,
        _unified_arap_backend=None,
        _unified_sim_cache=None,
    )
    return ctx


def patch_waypoints(cache_dir, active_muscles, motion_bvh, skel):
    """Compute waypoints from cached tet positions via barycentric interpolation.

    Mirrors viewer's _motion_patch_waypoints: iterates frames, sets skeleton
    pose, interpolates waypoints in deformed tetrahedra, and saves them into
    the existing chunk NPZ files.
    """
    import glob as glob_mod

    # Collect muscles that have waypoint bary coords
    to_patch = {}  # mname -> [(filepath, frames, positions), ...]
    for mname, mobj in active_muscles.items():
        if not (hasattr(mobj, 'waypoints') and len(mobj.waypoints) > 0):
            continue
        if not (hasattr(mobj, 'waypoint_bary_coords') and len(mobj.waypoint_bary_coords) > 0):
            continue
        file_list = sorted(glob_mod.glob(os.path.join(cache_dir, f'{mname}_chunk_*.npz')))
        if not file_list:
            continue
        entries = []
        for fp in file_list:
            data = np.load(fp, allow_pickle=True)
            entries.append((fp, data['frames'], data['positions']))
        to_patch[mname] = entries

    if not to_patch:
        print("No muscles with waypoints to patch")
        return

    # Sorted unique frames across all muscles
    all_frames = sorted(set(
        int(f)
        for entries in to_patch.values()
        for _, frames, _ in entries
        for f in frames
    ))

    # Per-muscle result accumulators
    muscle_wp = {mname: {} for mname in to_patch}
    muscle_wp_shape = {}

    # Fast lookup: mname -> {frame: (entry_idx, pos_idx)}
    muscle_frame_map = {}
    for mname, entries in to_patch.items():
        fmap = {}
        for entry_idx, (fp, frames, positions) in enumerate(entries):
            for pos_idx, f in enumerate(frames):
                fmap[int(f)] = (entry_idx, pos_idx)
        muscle_frame_map[mname] = fmap

    num_frames = len(all_frames)
    t0 = time.time()
    for i, frame_idx in enumerate(all_frames):
        # Set skeleton pose so origin/insertion endpoints update correctly
        if frame_idx < motion_bvh.mocap_refs.shape[0]:
            skel.setPositions(motion_bvh.mocap_refs[frame_idx])

        for mname, entries in to_patch.items():
            fmap = muscle_frame_map[mname]
            if frame_idx not in fmap:
                continue
            entry_idx, pos_idx = fmap[frame_idx]
            positions = entries[entry_idx][2]
            mobj = active_muscles[mname]
            mobj.tet_vertices = positions[pos_idx].astype(np.float32).copy()
            mobj._update_waypoints_from_tet(skel, verbose=False)
            wp_flat, wp_shape_str = _flatten_waypoints(mobj.waypoints)
            muscle_wp[mname][frame_idx] = wp_flat
            muscle_wp_shape[mname] = wp_shape_str

        if (i + 1) % 100 == 0 or i + 1 == num_frames:
            elapsed = time.time() - t0
            print(f"  Waypoints: {i+1}/{num_frames} frames  ({elapsed:.1f}s)")

    # Write patched files
    patched = 0
    for mname, entries in to_patch.items():
        for filepath, frames, positions in entries:
            frame_list = [int(f) for f in frames]
            wp_flats = [muscle_wp[mname][f] for f in frame_list]
            save_dict = dict(
                frames=frames,
                positions=positions,
                waypoints_flat=np.stack(wp_flats).astype(np.float32),
                waypoints_shape=np.array([muscle_wp_shape[mname].encode('utf-8')]),
            )
            np.savez_compressed(filepath, **save_dict)
        patched += 1
        n_frames = sum(len(frames) for _, frames, _ in entries)
        print(f"  Patched {mname}: {n_frames} frames across {len(entries)} file(s)")

    print(f"Waypoint patch complete: {patched} muscles updated")


def flush_bake_data(bake_data, cache_dir, flush_count):
    """Write accumulated frame data to chunk files, then clear memory."""
    for mname, frame_data in bake_data.items():
        if len(frame_data) == 0:
            continue
        sorted_frames = sorted(frame_data.keys())
        filepath = os.path.join(cache_dir, f"{mname}_chunk_{flush_count:04d}.npz")
        np.savez(
            filepath,
            frames=np.array(sorted_frames, dtype=np.int32),
            positions=np.array(
                [frame_data[f] for f in sorted_frames], dtype=np.float32
            ),
        )
        frame_data.clear()
    gc.collect()
    return flush_count + 1


def main():
    parser = argparse.ArgumentParser(description="Headless FEM muscle baking")
    parser.add_argument("--bvh", required=True, help="Path to BVH file")
    parser.add_argument(
        "--muscles",
        default=".last_loaded_muscles.json",
        help="Path to muscles JSON (default: .last_loaded_muscles.json)",
    )
    parser.add_argument(
        "--settle-iters",
        type=int,
        default=150,
        help="Simulation iterations per frame (default: 150)",
    )
    parser.add_argument(
        "--constraint-threshold",
        type=float,
        default=0.015,
        help="Inter-muscle constraint distance in meters (default: 0.015)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "taichi", "gpu", "cpu"],
        default="auto",
        help="ARAP solver backend (default: auto — picks taichi > gpu > cpu)",
    )
    parser.add_argument(
        "--sim-method",
        choices=["arap", "fem", "vbd", "pn"],
        default="arap",
        help="Simulation method: arap, fem (XPBD), vbd (VBD), or pn (Projected Newton)",
    )
    parser.add_argument(
        "--start-frame", type=int, default=0, help="First frame to bake (default: 0)"
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last frame to bake (default: all frames)",
    )
    parser.add_argument(
        "--load-steps",
        type=int,
        default=10,
        help="Incremental pose loading steps for first frame (default: 10). "
             "Prevents muscles crossing through bones during initialization.",
    )
    parser.add_argument(
        "--region-tag",
        default=None,
        help="Region tag for per-region baking (e.g. L_UpLeg). "
             "Output goes to motion_cache/<bvh>/<tag>/ instead of motion_cache/<bvh>/",
    )
    args = parser.parse_args()

    if not os.path.exists(args.bvh):
        print(f"ERROR: BVH file not found: {args.bvh}")
        sys.exit(1)
    if not os.path.exists(args.muscles):
        print(f"ERROR: Muscles file not found: {args.muscles}")
        sys.exit(1)

    # --- Pipeline ---
    skel, bvh_info, mesh_info = load_skeleton()
    skeleton_meshes = load_skeleton_meshes()
    muscle_meshes = load_muscle_meshes(args.muscles)
    load_tet_meshes(muscle_meshes)

    # Reset skeleton to rest pose before init
    print("[6/8] Resetting skeleton to rest pose...")
    skel.setPositions(np.zeros(skel.getNumDofs()))
    init_soft_bodies(muscle_meshes, skeleton_meshes, skel, mesh_info)

    # Build context and find constraints
    ctx = build_context(skel, muscle_meshes, skeleton_meshes, mesh_info, args)
    print("[7/8] Finding inter-muscle constraints...")
    n_constraints = find_inter_muscle_constraints(ctx)
    print(f"       Found {n_constraints} constraints")

    # Load BVH
    print(f"[8/8] Loading BVH: {args.bvh}")
    t_frame = _detect_bvh_tframe(args.bvh)
    motion_bvh = MyBVH(args.bvh, bvh_info, skel, T_frame=t_frame)
    num_frames = motion_bvh.mocap_refs.shape[0]
    print(f"       Frames: {num_frames}, T_frame: {t_frame}")

    # Frame range
    start_frame = args.start_frame
    end_frame = args.end_frame if args.end_frame is not None else num_frames - 1
    end_frame = min(end_frame, num_frames - 1)
    total_frames = end_frame - start_frame + 1
    print(f"\nBaking frames {start_frame}..{end_frame} ({total_frames} frames)")

    # Prepare output directory
    bvh_stem = os.path.splitext(os.path.basename(args.bvh))[0]
    if args.region_tag:
        cache_dir = os.path.join("data", "motion_cache", bvh_stem, args.region_tag)
    else:
        cache_dir = os.path.join("data", "motion_cache", bvh_stem)
    os.makedirs(cache_dir, exist_ok=True)

    # Remove old chunk files
    import glob as glob_mod
    for old_chunk in glob_mod.glob(os.path.join(cache_dir, "*_chunk_*.npz")):
        os.remove(old_chunk)

    # Identify active muscles (those with soft bodies)
    active_muscles = {
        name: mobj
        for name, mobj in muscle_meshes.items()
        if mobj.soft_body is not None
    }
    print(f"Active muscles: {len(active_muscles)}")

    # Reset soft bodies to rest state
    for mobj in active_muscles.values():
        mobj.soft_body.positions = mobj.soft_body.rest_positions.copy()
        mobj.tet_vertices = mobj.soft_body.rest_positions.astype(np.float32).copy()

    # Clear cached backend
    ctx._unified_arap_backend = None
    ctx._unified_sim_cache = None

    # Init bake accumulators
    bake_data = {name: {} for name in active_muscles}
    flush_count = 0
    bake_start = time.time()

    # --- Frame loop ---
    for frame in range(start_frame, end_frame + 1):
        frame_start = time.time()

        # Apply pose
        pose = motion_bvh.mocap_refs[frame].copy()
        skel.setPositions(pose)

        # Disable waypoint updates and draw array rebuilds during baking
        saved_flags = {}
        for mname, mobj in active_muscles.items():
            saved_flags[mname] = getattr(mobj, "waypoints_from_tet_sim", True)
            mobj.waypoints_from_tet_sim = False
            mobj._baking_mode = True

        # Run simulation
        if ctx.use_fem_sim:
            from viewer.fem_sim import run_all_fem_sim
            run_all_fem_sim(ctx, max_iterations=args.settle_iters, tolerance=1e-4,
                            verbose=(frame == start_frame))
        else:
            run_all_tet_sim_with_constraints(
                ctx, max_iterations=args.settle_iters, tolerance=1e-4
            )

        # Capture positions
        for mname, mobj in active_muscles.items():
            bake_data[mname][frame] = mobj.soft_body.get_positions().astype(np.float32)

        # Restore flags
        for mname, mobj in active_muscles.items():
            mobj.waypoints_from_tet_sim = saved_flags[mname]
            mobj._baking_mode = False

        # Periodic flush
        n_accumulated = sum(len(fd) for fd in bake_data.values())
        if n_accumulated >= FLUSH_INTERVAL * len(active_muscles):
            print(f"  Flushing chunk {flush_count} to disk...")
            flush_count = flush_bake_data(bake_data, cache_dir, flush_count)
            gc.collect()

        # Progress reporting — every frame for monitoring
        frame_dt = time.time() - frame_start
        frames_done = frame - start_frame + 1
        elapsed = time.time() - bake_start
        avg = elapsed / frames_done
        remaining = avg * (total_frames - frames_done)
        print(
            f"  Frame {frame}/{end_frame}  "
            f"({frames_done}/{total_frames})  "
            f"{frame_dt:.2f}s  "
            f"avg {avg:.2f}s/frame  "
            f"ETA {remaining:.0f}s",
            flush=True,
        )

    # Final flush
    if any(len(fd) > 0 for fd in bake_data.values()):
        flush_count = flush_bake_data(bake_data, cache_dir, flush_count)

    # Remove legacy single-file caches
    for mname in active_muscles:
        legacy = os.path.join(cache_dir, f"{mname}.npz")
        if os.path.exists(legacy):
            os.remove(legacy)

    bake_elapsed = time.time() - bake_start
    cache = getattr(ctx, "_unified_sim_cache", None)
    n_muscles = len(cache["muscle_names"]) if cache else len(active_muscles)
    total_verts = cache["total_verts"] if cache else "?"
    avg_frame = bake_elapsed / max(total_frames, 1)
    print(
        f"\nBake complete: {total_frames} frames in {bake_elapsed:.1f}s — "
        f"{n_muscles} muscles, {total_verts} verts, "
        f"{avg_frame:.2f}s/frame"
    )

    # Patch waypoints into chunk files
    print("\nComputing waypoints...")
    patch_waypoints(cache_dir, active_muscles, motion_bvh, skel)

    # Write completion marker so batch runner can detect fully-baked caches
    done_marker = os.path.join(cache_dir, ".done")
    with open(done_marker, "w") as f:
        f.write(f"{total_frames} frames\n")

    total_elapsed = time.time() - bake_start
    print(f"\nDone in {total_elapsed:.1f}s. Output: {cache_dir}/")


if __name__ == "__main__":
    main()
