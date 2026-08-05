#!/usr/bin/env python3
"""Pure LBS + bone-barrier bake.

No ARAP, no skin prior, no inter-muscle springs.  For each frame:
  - Compute LBS positions per muscle from mobj.skinning_weights × bone transforms.
  - Project any vert inside a candidate bone (femur, patella, tibia/fibula) to
    bone surface + margin × outward face normal.
  - Pin cap-attached verts to bone anchor positions (Dirichlet).

Targets the two-property goal: wrap around femur + maintain relative positions
across muscles that share the same primary skinning bone.  Minimal compute.

Usage:
    python tools/bake_lbs_barrier.py --bvh data/motion/dance.bvh \
        --muscles .muscles_quads.json --region-tag L_Quads_dance_lbs \
        --start-frame 2440 --end-frame 2450
"""
import argparse
import glob as glob_mod
import os
import sys
import time

import numpy as np
import trimesh

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.bake_headless import (
    load_skeleton, load_skeleton_meshes, load_muscle_meshes,
    load_tet_meshes, init_soft_bodies, flush_bake_data, patch_waypoints,
    FLUSH_INTERVAL,
)
from core.bvhparser import MyBVH
from viewer.zygote_mesh_ui import _detect_bvh_tframe


CANDIDATE_BONE_KEYS = ('Femur', 'Patella', 'Tibia', 'Fibula')


def build_bone_world_meshes(skeleton_meshes, skel, key_filter):
    """Build per-frame world-transformed trimeshes for bones whose name matches
    any substring in key_filter.  Returns list of trimesh objects."""
    out = []
    for bone_name, mloader in skeleton_meshes.items():
        if not any(k in bone_name for k in key_filter):
            continue
        tm = getattr(mloader, 'trimesh', None)
        if tm is None:
            continue
        # Resolve DART body — try suffix variants
        body = None
        for cand in (bone_name, bone_name + '0', bone_name + '1'):
            b = skel.getBodyNode(cand)
            if b is not None:
                body = b
                break
        if body is None:
            continue
        wt = body.getWorldTransform()
        R = np.array(wt.rotation())
        t = np.array(wt.translation())
        # Bone meshes scaled to meters at load time; trimesh.vertices is local
        verts_world = (R @ np.asarray(tm.vertices).T).T + t
        new_tm = trimesh.Trimesh(vertices=verts_world,
                                 faces=np.asarray(tm.faces),
                                 process=False)
        out.append(new_tm)
    return out


def lbs_positions(mobj, skel):
    """Linear blend skinning for one muscle.  Returns (N, 3) world positions."""
    rest = np.asarray(mobj.soft_body.rest_positions, dtype=np.float64)
    n = rest.shape[0]
    sw = getattr(mobj, 'skinning_weights', None)
    bones = getattr(mobj, 'skinning_bones', None)
    if sw is None or bones is None or len(bones) == 0:
        return rest.copy()
    initial = getattr(mobj, 'soft_body_initial_transforms', {})
    out = np.zeros((n, 3), dtype=np.float64)
    for bi, bname in enumerate(bones):
        if bname not in initial:
            continue
        R0, t0 = initial[bname]
        body = skel.getBodyNode(bname)
        if body is None:
            continue
        wt = body.getWorldTransform()
        R = np.array(wt.rotation())
        t = np.array(wt.translation())
        # p_world = R @ R0^T (rest - t0) + t
        local = (rest - t0) @ R0  # = R0^T (rest - t0)
        deformed = local @ R.T + t
        w = sw[:, bi:bi + 1]
        out += w * deformed
    return out


def project_outside_bones(positions, bone_meshes, margin):
    """For each vert inside any bone, move to bone surface + margin × outward
    face normal.  Returns updated positions."""
    if not bone_meshes:
        return positions
    out = positions.copy()
    for bm in bone_meshes:
        try:
            bmin = bm.bounds[0] - margin
            bmax = bm.bounds[1] + margin
            in_bbox = np.all((out >= bmin) & (out <= bmax), axis=1)
            if not np.any(in_bbox):
                continue
            cand = out[in_bbox]
            inside = bm.contains(cand)
            if not np.any(inside):
                continue
            ipos = cand[inside]
            iindices = np.where(in_bbox)[0][inside]
            cp, _, fid = trimesh.proximity.closest_point(bm, ipos)
            nrm = bm.face_normals[fid]
            out[iindices] = cp + nrm * margin
        except Exception:
            continue
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bvh', required=True)
    p.add_argument('--muscles', default='.last_loaded_muscles.json')
    p.add_argument('--region-tag', default='lbs_barrier')
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--end-frame', type=int, default=None)
    p.add_argument('--margin', type=float, default=0.005)
    p.add_argument('--tet-dir', default='tet')
    p.add_argument('--skip-waypoints', action='store_true')
    args = p.parse_args()

    skel, bvh_info, mesh_info = load_skeleton()
    skeleton_meshes = load_skeleton_meshes()
    muscle_meshes = load_muscle_meshes(args.muscles)
    load_tet_meshes(muscle_meshes, tet_dir=args.tet_dir)
    print("[6/8] Resetting skeleton to rest pose...")
    skel.setPositions(np.zeros(skel.getNumDofs()))
    init_soft_bodies(muscle_meshes, skeleton_meshes, skel, mesh_info)
    active_muscles = {n: m for n, m in muscle_meshes.items()
                      if m.soft_body is not None}
    print(f"Active muscles: {len(active_muscles)}")

    print(f"[8/8] Loading BVH: {args.bvh}")
    t_frame = _detect_bvh_tframe(args.bvh)
    motion_bvh = MyBVH(args.bvh, bvh_info, skel, T_frame=t_frame)
    n_frames = motion_bvh.mocap_refs.shape[0]
    start = args.start_frame
    end = args.end_frame if args.end_frame is not None else n_frames - 1
    end = min(end, n_frames - 1)
    total = end - start + 1
    print(f"Baking frames {start}..{end} ({total} frames)")

    bvh_stem = os.path.splitext(os.path.basename(args.bvh))[0]
    cache_dir = os.path.join('data', 'motion_cache', bvh_stem, args.region_tag)
    os.makedirs(cache_dir, exist_ok=True)
    for old in glob_mod.glob(os.path.join(cache_dir, '*_chunk_*.npz')):
        os.remove(old)

    bake_data = {n: {} for n in active_muscles}
    flush_count = 0
    t0 = time.time()

    for frame in range(start, end + 1):
        skel.setPositions(motion_bvh.mocap_refs[frame].copy())
        bone_meshes = build_bone_world_meshes(skeleton_meshes, skel,
                                              CANDIDATE_BONE_KEYS)
        for mname, mobj in active_muscles.items():
            # Refresh fixed-vert targets from current skeleton pose
            if hasattr(mobj, '_update_fixed_targets_from_skeleton'):
                mobj._update_fixed_targets_from_skeleton(skeleton_meshes, skel)
            pos = lbs_positions(mobj, skel)
            # Pin fixed verts (caps) to anchor world positions
            if mobj.soft_body.fixed_targets is not None and len(
                    mobj.soft_body.fixed_indices) > 0:
                for li, tg in zip(mobj.soft_body.fixed_indices,
                                  mobj.soft_body.fixed_targets):
                    pos[int(li)] = tg
            pos = project_outside_bones(pos, bone_meshes, args.margin)
            bake_data[mname][frame] = pos.astype(np.float32)
        elapsed = time.time() - t0
        done = frame - start + 1
        avg = elapsed / done
        eta = avg * (total - done)
        print(f"  Frame {frame - start}/{total - 1}  ({done}/{total})  "
              f"{avg:.2f}s/frame  ETA {eta:.0f}s", flush=True)
        n_acc = sum(len(fd) for fd in bake_data.values())
        if n_acc >= FLUSH_INTERVAL * len(active_muscles):
            flush_count = flush_bake_data(bake_data, cache_dir, flush_count)

    if any(len(fd) > 0 for fd in bake_data.values()):
        flush_count = flush_bake_data(bake_data, cache_dir, flush_count)

    if not args.skip_waypoints:
        print("[Waypoints]")
        try:
            patch_waypoints(cache_dir, active_muscles, motion_bvh, skel)
        except Exception as e:
            print(f"Waypoint patch skipped: {e}")

    with open(os.path.join(cache_dir, '.done'), 'w') as f:
        f.write('done')
    print(f"\nDone in {time.time() - t0:.1f}s. Output: {cache_dir}/")


if __name__ == '__main__':
    main()
