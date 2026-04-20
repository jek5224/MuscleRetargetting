#!/usr/bin/env python3
"""Layered ARAP bake — same as bake_headless.py but processes muscles in depth order.

Each layer uses the exact same _run_unified_volume_sim as the standard bake,
including inter-muscle vertex-distance constraints and warm-start.

Usage:
    python tools/bake_layered.py --bvh data/motion/walk.bvh --sides L --start-frame 60 --end-frame 70
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import trimesh

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

SKEL_XML = "data/zygote_skel.xml"
ZYGOTE_DIR = "Zygote_Meshes_251229/"
MESH_SCALE = 0.01
FLUSH_INTERVAL = 20

# Muscle depth layers
LAYERS = {
    0: [  # Deep
        "Iliacus", "Obturator_Internus", "Obturator_Externus",
        "Inferior_Gemellus", "Superior_Gemellus", "Quadratus_Femoris",
        "Piriformis", "Popliteus", "Vastus_Intermedius", "Gluteus_Minimus",
    ],
    1: [  # Mid
        "Adductor_Brevis", "Adductor_Longus", "Adductor_Magnus",
        "Pectineus", "Vastus_Medialis", "Vastus_Lateralis",
    ],
    2: [  # Superficial
        "Gluteus_Maximus", "Gluteus_Medius", "Rectus_Femoris",
        "Sartorius", "Gracilis", "Biceps_Femoris",
        "Semimembranosus", "Semitendinosus", "Tensor_Fascia_Lata",
    ],
}


def main():
    parser = argparse.ArgumentParser(description="Layered ARAP bake")
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--muscles", default=".last_loaded_muscles.json")
    parser.add_argument("--settle-iters", type=int, default=150)
    parser.add_argument("--constraint-threshold", type=float, default=0.015)
    parser.add_argument("--backend", choices=["auto","taichi","gpu","cpu"], default="auto")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--sides", default="L")
    parser.add_argument("--region-tag", default="layered")
    args = parser.parse_args()

    # ── Load everything (same as bake_headless.py) ────────────────────
    print("[1] Loading skeleton...")
    skel_info, root_name, bvh_info, _pd, mesh_info, _smpl = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)

    print("[2] Loading skeleton meshes...")
    skeleton_meshes = {}
    skel_dir = os.path.join(ZYGOTE_DIR, "Skeleton")
    for fname in sorted(os.listdir(skel_dir)):
        if not fname.endswith(".obj"): continue
        name = fname.split(".")[0]
        path = os.path.join(skel_dir, fname)
        skeleton_meshes[name] = MeshLoader()
        skeleton_meshes[name].load(path)
        skeleton_meshes[name].color = np.array([0.9, 0.9, 0.9])
        skel_tri = trimesh.load_mesh(path)
        skel_tri.vertices *= MESH_SCALE
        skeleton_meshes[name].trimesh = skel_tri

    print("[3] Loading muscle meshes...")
    with open(args.muscles, "r") as f:
        muscle_list = json.load(f)
    all_muscle_meshes = {}
    for entry in muscle_list:
        name, path = entry["name"], entry["path"]
        if not os.path.exists(path): continue
        all_muscle_meshes[name] = MeshLoader()
        all_muscle_meshes[name].load(path)
        all_muscle_meshes[name].color = np.array([0.8, 0.2, 0.2])
        mtri = trimesh.load_mesh(path); mtri.vertices *= MESH_SCALE
        all_muscle_meshes[name].trimesh = mtri
    all_muscle_meshes = dict(sorted(all_muscle_meshes.items()))

    print("[4] Loading tet meshes...")
    for name, mobj in all_muscle_meshes.items():
        mobj.load_tetrahedron_mesh(name)

    print("[5] Initializing soft bodies...")
    skel.setPositions(np.zeros(skel.getNumDofs()))
    for name, mobj in all_muscle_meshes.items():
        if mobj.tet_vertices is None: continue
        mobj.init_soft_body(skeleton_meshes=skeleton_meshes, skeleton=skel, mesh_info=mesh_info)

    # Filter to active muscles on requested side
    side = args.sides[0]
    active_all = {n: m for n, m in all_muscle_meshes.items()
                  if m.soft_body is not None and n.startswith(f"{side}_")}
    print(f"    {len(active_all)} active {side}-side muscles")

    # ── Classify into layers ──────────────────────────────────────────
    print("[6] Classifying into layers...")
    layer_muscles = [[], [], []]
    for name in active_all:
        short = name.replace(f"{side}_", "")
        assigned = False
        for li, lm in LAYERS.items():
            if short in lm:
                layer_muscles[li].append(name)
                assigned = True
                break
        if not assigned:
            layer_muscles[2].append(name)

    for li in range(3):
        print(f"    Layer {li}: {len(layer_muscles[li])} — {layer_muscles[li]}")

    # ── Load BVH ──────────────────────────────────────────────────────
    print("[7] Loading BVH...")
    t_frame = _detect_bvh_tframe(args.bvh)
    motion_bvh = MyBVH(args.bvh, bvh_info, skel, T_frame=t_frame)
    n_frames = motion_bvh.mocap_refs.shape[0]
    end_frame = args.end_frame if args.end_frame is not None else n_frames - 1
    end_frame = min(end_frame, n_frames - 1)
    total_frames = end_frame - args.start_frame + 1

    # ── Output ────────────────────────────────────────────────────────
    bvh_stem = os.path.splitext(os.path.basename(args.bvh))[0]
    cache_dir = os.path.join("data", "motion_cache", bvh_stem, args.region_tag)
    os.makedirs(cache_dir, exist_ok=True)

    # Remove old chunks
    import glob as glob_mod
    for old in glob_mod.glob(os.path.join(cache_dir, "*_chunk_*.npz")):
        os.remove(old)

    bake_data = {n: {} for n in active_all}
    flush_count = 0
    bake_start = time.time()

    # ── Build per-layer contexts ──────────────────────────────────────
    # Each layer gets its own ctx with only its muscles, so
    # run_all_tet_sim_with_constraints operates on the layer subset.
    # The ctx mimics the viewer object that _run_unified_volume_sim expects.

    def make_ctx(layer_muscle_names):
        """Build a viewer-like context for a subset of muscles."""
        subset = {n: active_all[n] for n in layer_muscle_names}
        backend_name = args.backend
        if backend_name == "auto":
            if check_taichi_available(): backend_name = "taichi"
            elif check_gpu_available(): backend_name = "gpu"
            else: backend_name = "cpu"

        ctx = SimpleNamespace(
            env=SimpleNamespace(skel=skel, mesh_info=mesh_info),
            zygote_muscle_meshes=subset,
            zygote_skeleton_meshes=skeleton_meshes,
            inter_muscle_constraints=[],
            inter_muscle_constraint_threshold=args.constraint_threshold,
            coupled_as_unified_volume=True,
            use_gpu_arap=(backend_name == 'gpu'),
            use_taichi_arap=(backend_name == 'taichi'),
            use_muscle_aware_arap=True,
            use_fem_sim=False,
            use_vbd_sim=False,
            use_pn_sim=False,
            fem_youngs_modulus=500.0,
            fem_poisson_ratio=0.40,
            fem_collision_kappa=1e4,
            fem_volume_penalty=5000.0,
            fem_contact_threshold=args.constraint_threshold,
            fem_outer_iterations=3,
            fem_load_steps=10,
            motion_settle_iters=args.settle_iters,
            _unified_arap_backend=None,
            _unified_sim_cache=None,
        )
        return ctx

    # Find ALL inter-muscle constraints globally (cross-layer + within-layer)
    print("    Finding global inter-muscle constraints...")
    global_ctx = make_ctx(list(active_all.keys()))
    n_global = find_inter_muscle_constraints(global_ctx)
    all_constraints = global_ctx.inter_muscle_constraints
    print(f"    {n_global} global constraints")

    # Build per-layer contexts, each including constraints where BOTH
    # vertices belong to muscles in that layer OR earlier settled layers.
    # For layer N: include constraints between any muscles in layers 0..N.
    layer_ctxs = []
    cumulative_muscles = set()
    for li in range(3):
        if not layer_muscles[li]:
            layer_ctxs.append(None)
            continue

        # This layer's muscles + all previously settled muscles
        cumulative_muscles.update(layer_muscles[li])
        ctx = make_ctx(list(cumulative_muscles))

        # Filter global constraints to only those involving muscles in cumulative set
        layer_constraints = []
        for c in all_constraints:
            name1, v1, f1, name2, v2, f2, dist = c
            if name1 in cumulative_muscles and name2 in cumulative_muscles:
                layer_constraints.append(c)
        ctx.inter_muscle_constraints = layer_constraints
        print(f"    Layer {li}: {len(layer_muscles[li])} muscles, "
              f"{len(layer_constraints)} constraints (cumulative {len(cumulative_muscles)} muscles)")
        layer_ctxs.append(ctx)

    # ── Frame loop ────────────────────────────────────────────────────
    print(f"\n[8] Baking frames {args.start_frame}-{end_frame}...")

    for frame in range(args.start_frame, end_frame + 1):
        frame_start = time.time()
        skel.setPositions(motion_bvh.mocap_refs[frame])

        # Process each layer in depth order
        for li in range(3):
            ctx = layer_ctxs[li]
            if ctx is None:
                continue

            # Disable waypoint updates during baking
            for mname, mobj in ctx.zygote_muscle_meshes.items():
                mobj.waypoints_from_tet_sim = False
                mobj._baking_mode = True

            # Run the exact same ARAP as bake_headless.py
            run_all_tet_sim_with_constraints(
                ctx, max_iterations=args.settle_iters, tolerance=1e-4)

            # Restore flags and capture positions
            for mname, mobj in ctx.zygote_muscle_meshes.items():
                mobj.waypoints_from_tet_sim = True
                mobj._baking_mode = False
                bake_data[mname][frame] = mobj.soft_body.get_positions().astype(np.float32)

        frame_dt = time.time() - frame_start
        print(f"  Frame {frame}: {frame_dt:.2f}s", flush=True)

        # Flush
        n_acc = sum(len(fd) for fd in bake_data.values())
        if n_acc >= FLUSH_INTERVAL * len(active_all):
            for mname, fd in bake_data.items():
                if not fd: continue
                sf = sorted(fd.keys())
                fp = os.path.join(cache_dir, f"{mname}_chunk_{flush_count:04d}.npz")
                np.savez(fp, frames=np.array(sf, dtype=np.int32),
                         positions=np.array([fd[f] for f in sf], dtype=np.float32))
                fd.clear()
            flush_count += 1; gc.collect()

    # Final flush
    for mname, fd in bake_data.items():
        if not fd: continue
        sf = sorted(fd.keys())
        fp = os.path.join(cache_dir, f"{mname}_chunk_{flush_count:04d}.npz")
        np.savez(fp, frames=np.array(sf, dtype=np.int32),
                 positions=np.array([fd[f] for f in sf], dtype=np.float32))
        fd.clear()

    elapsed = time.time() - bake_start
    print(f"\nDone. {total_frames} frames in {elapsed:.1f}s ({elapsed/max(total_frames,1):.2f}s/frame)")
    print(f"Output: {cache_dir}")


if __name__ == "__main__":
    main()
