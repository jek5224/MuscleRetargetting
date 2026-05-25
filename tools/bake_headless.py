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


def load_tet_meshes(muscle_meshes, tet_dir="tet"):
    """Load tetrahedron meshes for each muscle from `tet_dir`."""
    print(f"[4/8] Loading tet meshes from {tet_dir}/...")
    loaded = 0
    for name, mobj in muscle_meshes.items():
        path = os.path.join(tet_dir, f"{name}_tet.npz")
        if os.path.exists(path):
            mobj.load_tetrahedron_mesh(name, filepath=path)
        else:
            mobj.load_tetrahedron_mesh(name)
        if mobj.tet_vertices is not None:
            loaded += 1
        else:
            print(f"       WARNING: No tet mesh for {name}")
    print(f"       Loaded {loaded}/{len(muscle_meshes)} tet meshes")


def override_anatomical_skinning(mobj, origin_bone, mid_bone, insertion_bone,
                                  origin_end=0.2, insertion_start=0.8):
    """Override skinning with anatomical 3-zone scheme along u-axis:
    u ∈ [0, origin_end]:  smoothstep blend origin_bone → mid_bone.
    u ∈ [origin_end, insertion_start]:  100% mid_bone (wraps around it).
    u ∈ [insertion_start, 1]:  smoothstep blend mid_bone → insertion_bone.
    For biarticular pes-anserinus muscles where pelvis–femur–tibia
    transitions cause Z-shape kinks with naive multi-bone LBS.
    """
    sw = getattr(mobj, 'skinning_weights', None)
    bones = getattr(mobj, 'skinning_bones', None)
    vcl = getattr(mobj, 'vertex_contour_level', None)
    if sw is None or bones is None or vcl is None:
        return
    vcl = np.asarray(vcl, dtype=np.int32)
    if vcl.size != sw.shape[0]:
        return
    max_level = max(int(vcl.max()), 1)
    u = np.clip(vcl.astype(np.float64) / max_level, 0.0, 1.0)
    def _find(name_substr):
        for i, b in enumerate(bones):
            if name_substr in b:
                return i
        return -1
    oi = _find(origin_bone)
    mi = _find(mid_bone)
    ii = _find(insertion_bone)
    if oi < 0 or mi < 0 or ii < 0:
        return
    # smoothstep helper
    def smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)
    # Origin blend factor: at u=0 → 1 (full origin), at u=origin_end → 0
    origin_w = 1.0 - smoothstep(u / max(origin_end, 1e-6))
    # Insertion blend factor: at u=insertion_start → 0, at u=1 → 1
    insertion_w = smoothstep((u - insertion_start) /
                              max(1.0 - insertion_start, 1e-6))
    # Mid (femur) gets the remainder
    mid_w = np.clip(1.0 - origin_w - insertion_w, 0.0, 1.0)
    new_sw = np.zeros_like(sw)
    new_sw[:, oi] = origin_w
    new_sw[:, mi] = mid_w
    new_sw[:, ii] = insertion_w
    s = new_sw.sum(axis=1, keepdims=True)
    s = np.where(s > 1e-9, s, 1.0)
    mobj.skinning_weights = new_sw / s


def override_uaxis_skinning(mobj, origin_bone, insertion_bone, femur_share=0.2):
    """Override skinning weights with a u-coord linear blend:
       w_origin = (1-u) * (1-femur_share)
       w_insertion = u * (1-femur_share)
       w_femur = femur_share  (preserves wrap-around-condyle routing)
    where u = vertex_contour_level / max_level, 0=origin, 1=insertion.
    """
    sw = getattr(mobj, 'skinning_weights', None)
    bones = getattr(mobj, 'skinning_bones', None)
    vcl = getattr(mobj, 'vertex_contour_level', None)
    if sw is None or bones is None or vcl is None:
        return
    vcl = np.asarray(vcl, dtype=np.int32)
    if vcl.size != sw.shape[0]:
        return
    max_level = max(int(vcl.max()), 1)
    u = np.clip(vcl.astype(np.float64) / max_level, 0.0, 1.0)
    # Find bone indices
    def _find(name_substr):
        for i, b in enumerate(bones):
            if name_substr in b:
                return i
        return -1
    oi = _find(origin_bone)
    ii = _find(insertion_bone)
    fi = _find('Femur')
    if oi < 0 or ii < 0:
        return
    new_sw = np.zeros_like(sw)
    if fi >= 0 and femur_share > 0:
        new_sw[:, fi] = femur_share
        rem = 1.0 - femur_share
    else:
        rem = 1.0
    new_sw[:, oi] += (1.0 - u) * rem
    new_sw[:, ii] += u * rem
    # Normalize per-vert
    s = new_sw.sum(axis=1, keepdims=True)
    s = np.where(s > 1e-9, s, 1.0)
    mobj.skinning_weights = new_sw / s


def smooth_skinning_along_axis(mobj, n_passes=3):
    """Smooth `mobj.skinning_weights` along `vertex_contour_level` axis.

    Each contour level k receives the weight-average of levels {k-1, k, k+1}.
    Multiple passes broaden the smoothing kernel.  Eliminates sharp bone-
    weight transitions that cause Z-kinks at joint boundaries for long
    biarticular muscles (sartorius, gracilis, semitendinosus).
    Re-normalizes per-vert row sums to 1 after smoothing.
    """
    sw = getattr(mobj, 'skinning_weights', None)
    vcl = getattr(mobj, 'vertex_contour_level', None)
    if sw is None or vcl is None:
        return
    sw = np.asarray(sw, dtype=np.float64)
    vcl = np.asarray(vcl, dtype=np.int32)
    if sw.shape[0] != vcl.shape[0]:
        return
    levels = sorted(set(int(v) for v in vcl if v >= 0))
    if len(levels) < 3:
        return
    cur = sw.copy()
    for _ in range(n_passes):
        new = cur.copy()
        for level in levels:
            cur_idx = np.where(vcl == level)[0]
            if len(cur_idx) == 0:
                continue
            blend = [cur_idx]
            for al in (level - 1, level + 1):
                ai = np.where(vcl == al)[0]
                if len(ai) > 0:
                    blend.append(ai)
            avg = cur[np.concatenate(blend)].mean(axis=0)
            new[cur_idx] = avg
        s = new.sum(axis=1, keepdims=True)
        s = np.where(s > 1e-9, s, 1.0)
        cur = new / s
    mobj.skinning_weights = cur


def init_soft_bodies(muscle_meshes, skeleton_meshes, skel, mesh_info,
                     smooth_skinning=None):
    """Initialize soft body simulation for each muscle.

    smooth_skinning: optional iterable of muscle names whose skinning_weights
    should be smoothed along the contour-level axis after init.  Used for
    biarticular muscles where weight discontinuity causes Z-kinks.
    """
    print("[5/8] Initializing soft bodies...")
    count = 0
    smooth_set = set(smooth_skinning or ())
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
            if name in smooth_set:
                smooth_skinning_along_axis(mobj, n_passes=3)
                print(f"       Smoothed skinning weights along axis: {name}")
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

    ctx = SimpleNamespace(
        env=SimpleNamespace(skel=skel, mesh_info=mesh_info),
        zygote_muscle_meshes=muscle_meshes,
        zygote_skeleton_meshes=skeleton_meshes,
        inter_muscle_constraints=[],
        inter_muscle_constraint_threshold=args.constraint_threshold,
        inter_muscle_k_cap=args.inter_k,
        inter_muscle_weight=args.inter_muscle_weight,
        arap_anisotropic=args.anisotropic,
        arap_cross_w=args.arap_cross_w,
        arap_intra_w=args.arap_intra_w,
        arap_neutral_w=args.arap_neutral_w,
        coupled_as_unified_volume=True,
        use_gpu_arap=use_gpu,
        use_taichi_arap=use_taichi,
        use_muscle_aware_arap=args.use_muscle_aware_arap,
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
        skin_prior_enabled=getattr(args, 'skin_prior', False),
        skin_prior_sigma=getattr(args, 'skin_prior_sigma', 0.02),
        skin_prior_max_dist=getattr(args, 'skin_prior_max_dist', 0.05),
        skin_prior_strength=getattr(args, 'skin_prior_strength', 0.35),
        disable_plateau_exit=getattr(args, 'no_plateau_exit', False),
        use_anisotropic_contact=getattr(args, 'anisotropic_contact', False),
        inter_muscle_contact_margin=getattr(args, 'inter_muscle_contact_margin', 0.002),
        fascia_constraints_on=getattr(args, 'fascia_constraints', False),
        fascia_constraint_threshold=getattr(args, 'fascia_constraint_threshold', 0.01),
        fascia_constraint_weight=getattr(args, 'fascia_constraint_weight', 1.0),
        unified_bone_contact=getattr(args, 'unified_bone_contact', True),
        unified_bone_contact_margin=getattr(args, 'unified_bone_contact_margin', 0.005),
        unified_bone_collision_weight=getattr(args, 'unified_bone_contact_weight', 1.5),
        axial_min_ratio=getattr(args, 'axial_min_ratio', 0.65),
        axial_max_bulge=getattr(args, 'axial_max_bulge', 2.0),
        axial_curve=getattr(args, 'axial_curve', 'smooth'),
        fiber_spring=getattr(args, 'fiber_spring', False),
        fiber_spring_weight=getattr(args, 'fiber_spring_weight', 10.0),
        fiber_spring_rest_scale=getattr(args, 'fiber_spring_rest_scale', 0.5),
        pes_bundle=getattr(args, 'pes_bundle', False),
        pes_bundle_weight=getattr(args, 'pes_bundle_weight', 1.5),
        pes_bundle_offset_scale=getattr(args, 'pes_bundle_offset_scale', 1.0),
        pes_bundle_obj=None,
        skin_prior_binder=None,
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
            np.savez(filepath, **save_dict)
        patched += 1
        n_frames = sum(len(frames) for _, frames, _ in entries)
        print(f"  Patched {mname}: {n_frames} frames across {len(entries)} file(s)")

    print(f"Waypoint patch complete: {patched} muscles updated")


def flush_bake_data(bake_data, cache_dir, flush_count, bake_anim=None):
    """Write accumulated frame data to chunk files, then clear memory.

    If bake_anim is provided, also writes a `positions_anim` array of shape
    (F, K, V, 3) where K is the number of convergence snapshots per frame
    (padded to the per-chunk max with the final position).
    """
    for mname, frame_data in bake_data.items():
        if len(frame_data) == 0:
            continue
        sorted_frames = sorted(frame_data.keys())
        filepath = os.path.join(cache_dir, f"{mname}_chunk_{flush_count:04d}.npz")
        save_dict = {
            'frames': np.array(sorted_frames, dtype=np.int32),
            'positions': np.array(
                [frame_data[f] for f in sorted_frames], dtype=np.float32
            ),
        }
        if bake_anim is not None and mname in bake_anim:
            anim = bake_anim[mname]
            present = [f for f in sorted_frames if f in anim and len(anim[f]) > 0]
            if present:
                k_max = max(len(anim[f]) for f in present)
                v = save_dict['positions'].shape[1]
                anim_arr = np.empty((len(sorted_frames), k_max, v, 3), dtype=np.float32)
                for i, f in enumerate(sorted_frames):
                    snaps = anim.get(f, [])
                    if not snaps:
                        snaps = [frame_data[f]]
                    # pad short sequences with the final converged frame so
                    # k stays uniform per chunk.
                    last = snaps[-1]
                    for k in range(k_max):
                        anim_arr[i, k] = snaps[k] if k < len(snaps) else last
                save_dict['positions_anim'] = anim_arr
            anim.clear()
        np.savez(filepath, **save_dict)
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
        default=0.03,
        help="Inter-muscle constraint distance in meters (default: 0.03). "
             "Contour tets are coarse (~160-512 verts/muscle vs ~1k+ for "
             "original meshes) so the search radius needs to span the "
             "wider vertex spacing.",
    )
    parser.add_argument(
        "--no-muscle-aware",
        dest="use_muscle_aware_arap",
        action="store_false",
        default=True,
        help="Disable muscle-aware ARAP per-frame target-edge scaling "
             "(rest edges stay at base length, no contraction-tracking).  "
             "Use to test whether the per-frame ratio tracking is causing "
             "frame-to-frame tremble.",
    )
    parser.add_argument(
        "--no-plateau-exit",
        action="store_true",
        default=False,
        help="Disable ARAP solver plateau early-exit so every frame "
             "runs the full --settle-iters.  Trades compute for uniform "
             "per-frame convergence depth — reduces per-frame ticking "
             "at fast-pose frames where the early-exit accepts higher "
             "residuals.",
    )
    parser.add_argument(
        "--inter-muscle-weight",
        type=float,
        default=1.0,
        help="ARAP edge weight for inter-muscle constraint springs "
             "(default 1.0 = same as internal neutral edges).  Lower "
             "values weaken muscle-to-muscle pull; useful when adjacent "
             "muscles pinch each other to near-zero volume.",
    )
    parser.add_argument(
        "--inter-k",
        type=int,
        default=3,
        help="Per-vertex cap on cross-muscle neighbors (default: 3). "
             "Bounds inter-muscle edge count; prevents quadratic blowup "
             "in dense regions like LowLeg.",
    )
    parser.add_argument(
        "--isotropic",
        dest="anisotropic",
        action="store_false",
        default=True,
        help="Disable anisotropic ARAP weights (cross-fiber softer than "
             "perpendicular).  Defaults on; pass --isotropic to revert.",
    )
    parser.add_argument(
        "--arap-cross-w",
        type=float,
        default=0.1,
        help="Cross-contour (along-fiber) edge weight when anisotropic is "
             "on (default: 0.1 — fibers contract more freely).",
    )
    parser.add_argument(
        "--arap-intra-w",
        type=float,
        default=3.0,
        help="Intra-contour (perpendicular) edge weight when anisotropic "
             "is on (default: 3.0 — strongly preserves cross-section).",
    )
    parser.add_argument(
        "--arap-neutral-w",
        type=float,
        default=1.0,
        help="Weight for edges that are neither cross- nor intra-contour "
             "(default: 1.0).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "taichi", "gpu", "cpu"],
        default="auto",
        help="ARAP solver backend (default: auto — picks taichi > gpu > cpu)",
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
        "--region-tag",
        default=None,
        help="Region tag for per-region baking (e.g. L_UpLeg). "
             "Output goes to motion_cache/<bvh>/<tag>/ instead of motion_cache/<bvh>/",
    )
    parser.add_argument(
        "--tet-dir",
        default="tet",
        help="Directory holding <muscle>_tet.npz tet meshes (default: tet).",
    )
    parser.add_argument(
        "--self-collision",
        dest="no_self_collision",
        action="store_false",
        default=True,
        help="Enable per-muscle bone collision push.  Defaults off; pass "
             "--self-collision to engage.",
    )
    parser.add_argument(
        "--skin-prior",
        action="store_true",
        default=False,
        help="Enable nearest-bone skinning prior in ARAP.  At rest each "
             "tet vert is bound to the nearest triangle on the nearest "
             "bone; ARAP energy gains a soft term keeping the deformed "
             "vert at the bone-local rest position transformed by the "
             "current bone pose.  Cooperates with ARAP's min-deformation "
             "objective rather than fighting it post-solve.",
    )
    parser.add_argument(
        "--skin-prior-sigma",
        type=float,
        default=0.02,
        help="Decay length scale (m) for skinning-prior weights "
             "(default 0.02 = 2cm).  w_i = exp(-rest_dist/sigma).",
    )
    parser.add_argument(
        "--skin-prior-max-dist",
        type=float,
        default=0.05,
        help="Bind only verts whose nearest bone is within this distance "
             "at rest (default 0.05 = 5cm).",
    )
    parser.add_argument(
        "--skin-prior-strength",
        type=float,
        default=0.35,
        help="Global multiplier on skin-prior weight (w_i = strength * "
             "exp(-rest_dist/sigma)). Default 0.35 — weakened from the "
             "original 0.7/1.0 because penetration prevention now belongs "
             "to the bone-contact term; skin prior is a weak visual "
             "stabilizer only.",
    )
    parser.add_argument(
        "--skin-prior-exclude",
        type=str,
        default="",
        help="Comma-separated muscle names to exclude from skin-prior "
             "precompute (e.g. 'L_Popliteus,R_Popliteus').  Listed muscles "
             "bake without the skin-prior term.",
    )
    parser.add_argument(
        "--skin-prior-include",
        type=str,
        default="",
        help="Comma-separated muscle names that get skin-prior; all others "
             "are excluded.  Whitelist mode — overrides --skin-prior-exclude.",
    )
    parser.add_argument(
        "--pes-bundle",
        action="store_true",
        default=False,
        help="Enable shared 4-anchor anatomical bundle prior for pes "
             "anserinus muscles (Sartorius, Gracilis, Semitendinosus).  "
             "All 3 muscle mid-shaft verts target one curve threaded "
             "pelvis-origin → mid-femur → medial-condyle → tibia-insertion. "
             "Eliminates Z-kink and ties them as fascia.",
    )
    parser.add_argument(
        "--pes-bundle-weight",
        type=float,
        default=1.5,
        help="Per-vert pull weight scale for the pes bundle prior. "
             "Higher = stronger curve adherence.  Default 1.5.",
    )
    parser.add_argument(
        "--pes-bundle-offset-scale",
        type=float,
        default=1.0,
        help="Scale on rest-frame transverse offset in the per-vert target "
             "(0 = collapse muscles onto curve, 1 = preserve thickness).",
    )
    parser.add_argument(
        "--fiber-spring",
        action="store_true",
        default=False,
        help="Enable scalar Hookean fiber spring on cross-contour edges, "
             "applied per ARAP iter as a corrective displacement target.  "
             "Dominates ARAP when --fiber-spring-weight is high.  Pure 1D "
             "spring (length-based, rotation-invariant) — closer to a real "
             "muscle elastic band than ARAP's edge-vector term.",
    )
    parser.add_argument(
        "--fiber-spring-weight",
        type=float,
        default=10.0,
        help="Diagonal weight per vert per spring neighbor.  Higher = "
             "spring dominates ARAP.  Default 10.0.",
    )
    parser.add_argument(
        "--fiber-spring-rest-scale",
        type=float,
        default=0.5,
        help="Multiplier on rest edge length used as the spring target "
             "length.  <1.0 = active contraction (band wants to shorten).  "
             "Default 0.5 (50%% of rest).",
    )
    parser.add_argument(
        "--axial-min-ratio",
        type=float,
        default=0.65,
        help="Axial pose prior: minimum cross-contour rest scale at full "
             "knee flex.  Default 0.65 (softer than previous 0.3).  1.0 "
             "disables contraction.",
    )
    parser.add_argument(
        "--axial-max-bulge",
        type=float,
        default=2.0,
        help="Axial pose prior: cap on intra-contour rest scaling for "
             "volume-preserving bulge.  Default 2.0 (was 3.0).",
    )
    parser.add_argument(
        "--axial-curve",
        choices=["linear", "smooth"],
        default="smooth",
        help="Axial pose prior: mapping from knee angle to contraction "
             "strength.  smooth uses a smoothstep curve, linear is "
             "direct proportional.",
    )
    parser.add_argument(
        "--unified-bone-contact",
        dest="unified_bone_contact",
        action="store_true",
        default=False,
        help="Enable explicit one-sided bone collision penalty in the "
             "unified-volume ARAP path.  Adds a per-iter target push on "
             "verts that penetrate any bone (target = surface + margin * "
             "normal).  Outside-margin verts pay no force.  Replaces the "
             "skin-prior-as-collision workaround.",
    )
    parser.add_argument(
        "--no-unified-bone-contact",
        dest="unified_bone_contact",
        action="store_false",
        help="Disable unified-path bone contact penalty.",
    )
    parser.add_argument(
        "--unified-bone-contact-margin",
        type=float,
        default=0.005,
        help="Safety margin (m) outside the bone surface when pushing "
             "penetrating verts (default 0.005 = 5mm).",
    )
    parser.add_argument(
        "--unified-bone-contact-weight",
        type=float,
        default=1.5,
        help="Diagonal weight added to penetrating verts for the bone "
             "contact spring.  Should exceed skin-prior diag (~0.35) but "
             "stay below combined ARAP intra-contour pull.  Default 1.5.",
    )
    parser.add_argument(
        "--save-anim",
        action="store_true",
        help="Save per-outer-iter convergence snapshots inside each chunk so "
             "the viewer can scrub the iron-man-style settling sequence.",
    )
    parser.add_argument(
        "--skip-waypoints",
        action="store_true",
        help="Skip post-bake waypoint patching. Use when only tet positions "
             "are needed (e.g. distillation training).",
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
    load_tet_meshes(muscle_meshes, tet_dir=args.tet_dir)

    # Reset skeleton to rest pose before init
    print("[6/8] Resetting skeleton to rest pose...")
    skel.setPositions(np.zeros(skel.getNumDofs()))
    from viewer.skin_prior import DQS_MUSCLES
    PES_ANSERINUS = {'L_Sartorius', 'L_Gracilis', 'L_Semitendinosus',
                     'R_Sartorius', 'R_Gracilis', 'R_Semitendinosus'}
    init_soft_bodies(muscle_meshes, skeleton_meshes, skel, mesh_info,
                     smooth_skinning=None)
    # 3-zone skinning override DISABLED — reverted to default wrap-bone LBS.
    # (Override was making mid-shaft follow femur rigidly which conflicted
    # with insertion cap pinned to tibia → wrong shape, not Z fix.)

    # Build context and find constraints
    ctx = build_context(skel, muscle_meshes, skeleton_meshes, mesh_info, args)
    if ctx.skin_prior_enabled:
        include = {n.strip() for n in getattr(args, 'skin_prior_include', '').split(',') if n.strip()}
        exclude = {n.strip() for n in getattr(args, 'skin_prior_exclude', '').split(',') if n.strip()}
        if include:
            sp_targets = {n: m for n, m in muscle_meshes.items() if n in include}
            print(f"[6.5/8] Skin-prior includes (whitelist): {sorted(include)}")
        else:
            sp_targets = {n: m for n, m in muscle_meshes.items() if n not in exclude}
            if exclude:
                print(f"[6.5/8] Skin-prior excludes: {sorted(exclude)}")
        print(f"[6.5/8] Precomputing skinning prior bindings (sigma={ctx.skin_prior_sigma}m, max_dist={ctx.skin_prior_max_dist}m, strength={ctx.skin_prior_strength}) on {len(sp_targets)}/{len(muscle_meshes)} muscles...")
        from viewer.skin_prior import SkinPriorBinder
        binder = SkinPriorBinder(
            mesh_scale=MESH_SCALE,
            sigma=ctx.skin_prior_sigma,
            max_bind_dist=ctx.skin_prior_max_dist,
            strength=ctx.skin_prior_strength,
        )
        binder.precompute(sp_targets, skeleton_meshes, skel, "Zygote_Meshes_251229/Skeleton")
        ctx.skin_prior_binder = binder
    # Pes anserinus bundle precompute (T-pose) — optional.
    if getattr(ctx, 'pes_bundle', False):
        from viewer.skin_prior import PesAnserinusBundle, PES_ANSERINUS_BUNDLE
        bundle_muscles = {n: muscle_meshes[n]
                          for n in PES_ANSERINUS_BUNDLE
                          if n in muscle_meshes and muscle_meshes[n].soft_body is not None}
        if len(bundle_muscles) >= 1:
            print(f"[6.6/8] Pes anserinus bundle precompute on "
                  f"{list(bundle_muscles.keys())}")
            bundle = PesAnserinusBundle()
            ok = bundle.precompute(bundle_muscles, skel,
                                   weight=ctx.pes_bundle_weight, sigma=0.04)
            if ok:
                ctx.pes_bundle_obj = bundle
                total = sum(len(d['vi']) for d in bundle.per_muscle.values())
                print(f"       {total} bundle bindings across "
                      f"{len(bundle.per_muscle)} muscles")
            else:
                print("       Pes bundle precompute failed (missing caps?)")
                ctx.pes_bundle_obj = None
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

    # Reset soft bodies to rest state + override Phase-1/2 iter caps for the
    # baking driver. With dense original tets, Phase-1 (mass-spring) becomes
    # the bottleneck — phase1-iters lets the driver lower the floor.
    for mobj in active_muscles.values():
        mobj.soft_body.positions = mobj.soft_body.rest_positions.copy()
        mobj.tet_vertices = mobj.soft_body.rest_positions.astype(np.float32).copy()
        mobj._baking_phase1_iters = 50  # contour-tet default
        if args.no_self_collision:
            mobj.soft_body_collision = False

    # Clear cached backend
    ctx._unified_arap_backend = None
    ctx._unified_arap_backend_ext = None
    ctx._unified_arap_backend_flex = None
    ctx._unified_sim_cache = None

    # Capture rest-pose knee R_rel (in T-pose) so per-frame knee-angle
    # measurement can subtract this baseline.  Bones aren't exactly
    # aligned at rest (XML axis conventions), so raw R_rel != I in T-pose.
    skel.setPositions(np.zeros(skel.getNumDofs()))
    rest_knee_R_rel = {}
    for side, body_pair in (('L', ('L_Femur0', 'L_Tibia_Fibula0')),
                            ('R', ('R_Femur0', 'R_Tibia_Fibula0'))):
        fem = skel.getBodyNode(body_pair[0])
        tib = skel.getBodyNode(body_pair[1])
        if fem is None or tib is None:
            rest_knee_R_rel[side] = np.eye(3)
            continue
        fR = np.array(fem.getWorldTransform().rotation())
        tR = np.array(tib.getWorldTransform().rotation())
        rest_knee_R_rel[side] = tR @ fR.T
    ctx._rest_knee_R_rel = rest_knee_R_rel

    # Init bake accumulators
    bake_data = {name: {} for name in active_muscles}
    # bake_anim[mname][frame] = list of (n_verts, 3) per snapshot (init + each outer iter)
    bake_anim = {name: {} for name in active_muscles} if args.save_anim else None
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
        anim_cb = None
        if bake_anim is not None:
            # Snapshot each outer iter so the viewer can scrub the
            # iron-man-style settling sequence per frame.
            def anim_cb(stage, ams, _f=frame, _ba=bake_anim):
                for nm, mo in ams.items():
                    if mo.soft_body is None:
                        continue
                    snap = mo.soft_body.get_positions().astype(np.float32).copy()
                    _ba[nm].setdefault(_f, []).append(snap)
        run_all_tet_sim_with_constraints(
            ctx, max_iterations=args.settle_iters,
            tolerance=1e-4,
            outer_iterations=20,
            snapshot_callback=anim_cb,
        )

        # Capture positions — verify fixed vertices match bone positions
        for mname, mobj in active_muscles.items():
            positions = mobj.soft_body.get_positions()
            if frame == start_frame and hasattr(mobj, 'soft_body_local_anchors'):
                # Check every fixed vertex
                for vi, (body_name, local_pos) in mobj.soft_body_local_anchors.items():
                    body_node = skel.getBodyNode(body_name)
                    if body_node is None:
                        print(f"  WARN {mname} vi={vi}: bone '{body_name}' not found!", flush=True)
                        continue
                    wt = body_node.getWorldTransform()
                    expected = wt.rotation() @ local_pos + wt.translation()
                    actual = positions[int(vi)]
                    err = np.linalg.norm(actual - expected)
                    if err > 0.001:
                        print(f"  WARN {mname} vi={vi}: bone={body_name}, err={err:.4f}m, "
                              f"actual={actual}, expected={expected}", flush=True)
            bake_data[mname][frame] = positions.astype(np.float32)

        # Restore flags
        for mname, mobj in active_muscles.items():
            mobj.waypoints_from_tet_sim = saved_flags[mname]
            mobj._baking_mode = False

        # Periodic flush
        n_accumulated = sum(len(fd) for fd in bake_data.values())
        if n_accumulated >= FLUSH_INTERVAL * len(active_muscles):
            print(f"  Flushing chunk {flush_count} to disk...")
            flush_count = flush_bake_data(bake_data, cache_dir, flush_count, bake_anim)
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
        flush_count = flush_bake_data(bake_data, cache_dir, flush_count, bake_anim)

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
    if args.skip_waypoints:
        print("\nSkipping waypoint patching (--skip-waypoints)")
    else:
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
