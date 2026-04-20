#!/usr/bin/env python3
"""Layered ARAP bake with depth-ordered collision handling.

Simulates muscles in depth order: deep muscles (close to bone) first,
then mid, then superficial. Each layer's ARAP solve uses collision_target_fn
to resolve penetrations with bones + settled deeper layers.

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
from scipy.spatial import cKDTree
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from types import SimpleNamespace
from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH
from viewer.mesh_loader import MeshLoader
from viewer.zygote_mesh_ui import (
    find_inter_muscle_constraints,
    _detect_bvh_tframe,
    _flatten_waypoints,
)
from viewer.arap_backends import check_taichi_available, check_gpu_available, get_backend

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SKEL_XML = "data/zygote_skel.xml"
ZYGOTE_DIR = "Zygote_Meshes_251229/"
MESH_SCALE = 0.01
FLUSH_INTERVAL = 20

BONE_NAME_TO_BODY = {
    'L_Os_Coxae': 'L_Os_Coxae0', 'L_Femur': 'L_Femur0',
    'L_Tibia_Fibula': 'L_Tibia_Fibula0', 'L_Patella': 'L_Patella0',
    'R_Os_Coxae': 'R_Os_Coxae0', 'R_Femur': 'R_Femur0',
    'R_Tibia_Fibula': 'R_Tibia_Fibula0', 'R_Patella': 'R_Patella0',
    'Saccrum_Coccyx': 'Saccrum_Coccyx0',
}

# Muscle depth layers (L side)
LAYERS = {
    0: [  # Deep — directly on bone
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


def extract_surface_triangles(tet_elements):
    face_count = Counter()
    face_orient = {}
    for t in tet_elements:
        v0, v1, v2, v3 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        for f in [(v0,v2,v1),(v0,v1,v3),(v1,v2,v3),(v0,v3,v2)]:
            key = tuple(sorted(f)); face_count[key] += 1
            if key not in face_orient: face_orient[key] = f
    return np.array([face_orient[k] for k,c in face_count.items() if c==1], dtype=np.int32)


# ---------------------------------------------------------------------------
# Collision target function factory
# ---------------------------------------------------------------------------
def make_collision_target_fn(obstacle_meshes, collision_verts, fixed_set, margin=0.002):
    """Create collision_target_fn for ARAP Taichi backend.

    Called every ARAP iteration with current positions.
    Returns dict {vertex_idx: target_position}.
    - Inside obstacle: target = surface + margin (pulls out)
    - Outside: target = current_pos (zero net force)
    """
    if not obstacle_meshes:
        return None

    # Pre-build KDTree from all obstacle vertices
    all_obs_verts = np.vstack([m.vertices for m in obstacle_meshes])
    obs_kdtree = cKDTree(all_obs_verts)

    _cache = {'targets': None, 'iter': 0, 'interval': 5}

    def collision_target_fn(positions):
        _cache['iter'] += 1

        # Recompute collision targets every N iterations (expensive trimesh.contains)
        if _cache['targets'] is not None and _cache['iter'] % _cache['interval'] != 0:
            # Reuse cached targets but update "outside" verts to current pos
            targets = {}
            for vi in collision_verts:
                if vi in _cache['targets'] and np.linalg.norm(_cache['targets'][vi] - _cache['_last_pos'][vi]) > 1e-6:
                    targets[vi] = _cache['targets'][vi]  # still colliding → keep target
                else:
                    targets[vi] = positions[vi].copy()  # no collision → current pos
            return targets

        targets = {}
        for vi in collision_verts:
            targets[vi] = positions[vi].copy()

        coll_arr = np.array(sorted(collision_verts))
        coll_pos = positions[coll_arr]
        dists, _ = obs_kdtree.query(coll_pos)
        near_mask = dists < 0.03

        if not np.any(near_mask):
            _cache['targets'] = targets
            _cache['_last_pos'] = positions.copy()
            return targets

        near_cv = coll_arr[near_mask]
        near_pos = positions[near_cv]

        for obs_mesh in obstacle_meshes:
            try:
                bmin = obs_mesh.bounds[0] - 0.03
                bmax = obs_mesh.bounds[1] + 0.03
                in_bbox = np.all((near_pos >= bmin) & (near_pos <= bmax), axis=1)
                if not np.any(in_bbox):
                    continue

                bbox_cv = near_cv[in_bbox]
                bbox_pos = near_pos[in_bbox]

                # Use contains() for watertight, dot-product for non-watertight
                if obs_mesh.is_watertight:
                    inside = obs_mesh.contains(bbox_pos)
                else:
                    closest, _, face_ids = trimesh.proximity.closest_point(obs_mesh, bbox_pos)
                    normals = obs_mesh.face_normals[face_ids]
                    to_vert = bbox_pos - closest
                    signed_dist = np.sum(to_vert * normals, axis=1)
                    dist_to_surf = np.linalg.norm(to_vert, axis=1)
                    inside = (signed_dist < 0) & (dist_to_surf < margin * 10)

                if not np.any(inside):
                    continue

                inside_cv = bbox_cv[inside]
                inside_pos = bbox_pos[inside]
                closest_pts, _, face_ids = trimesh.proximity.closest_point(obs_mesh, inside_pos)
                normals = obs_mesh.face_normals[face_ids]

                for k in range(len(inside_cv)):
                    vi = int(inside_cv[k])
                    if vi in fixed_set:
                        continue
                    targets[vi] = closest_pts[k] + normals[k] * margin
            except Exception:
                continue

        _cache['targets'] = targets
        _cache['_last_pos'] = positions.copy()
        return targets

    return collision_target_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Layered ARAP bake with collision")
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--muscles", default=".last_loaded_muscles.json")
    parser.add_argument("--settle-iters", type=int, default=80)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--sides", default="L")
    parser.add_argument("--collision-weight", type=float, default=10.0)
    parser.add_argument("--margin", type=float, default=0.002)
    parser.add_argument("--region-tag", default=None)
    args = parser.parse_args()

    # ── Load skeleton ────────────────────────────────────────────────
    print("[1] Loading skeleton...")
    skel_info, root_name, bvh_info, _pd, mesh_info, _smpl = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)

    # ── Load skeleton meshes ─────────────────────────────────────────
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

    # ── Load muscle meshes ───────────────────────────────────────────
    print("[3] Loading muscle meshes...")
    with open(args.muscles, "r") as f:
        muscle_list = json.load(f)
    muscle_meshes = {}
    for entry in muscle_list:
        name = entry["name"]; path = entry["path"]
        if not os.path.exists(path): continue
        muscle_meshes[name] = MeshLoader()
        muscle_meshes[name].load(path)
        muscle_meshes[name].color = np.array([0.8, 0.2, 0.2])
        mtri = trimesh.load_mesh(path); mtri.vertices *= MESH_SCALE
        muscle_meshes[name].trimesh = mtri
    muscle_meshes = dict(sorted(muscle_meshes.items()))

    # ── Load tet meshes + init soft bodies ────────────────────────────
    print("[4] Loading tet meshes...")
    for name, mobj in muscle_meshes.items():
        mobj.load_tetrahedron_mesh(name)

    print("[5] Initializing soft bodies...")
    skel.setPositions(np.zeros(skel.getNumDofs()))
    for name, mobj in muscle_meshes.items():
        if mobj.tet_vertices is None: continue
        mobj.init_soft_body(skeleton_meshes=skeleton_meshes, skeleton=skel, mesh_info=mesh_info)

    # Active muscles
    active_muscles = {n: m for n, m in muscle_meshes.items() if m.soft_body is not None}
    print(f"    {len(active_muscles)} active muscles")

    # ── Load BVH ─────────────────────────────────────────────────────
    print("[6] Loading BVH...")
    t_frame = _detect_bvh_tframe(args.bvh)
    motion_bvh = MyBVH(args.bvh, bvh_info, skel, T_frame=t_frame)
    n_frames = motion_bvh.mocap_refs.shape[0]
    end_frame = args.end_frame if args.end_frame is not None else n_frames - 1
    end_frame = min(end_frame, n_frames - 1)
    total_frames = end_frame - args.start_frame + 1

    # ── Classify muscles into layers ──────────────────────────────────
    print("[7] Classifying muscles into depth layers...")
    side = args.sides[0]
    layers = [[], [], []]  # [deep, mid, superficial]
    layer_names = ["deep", "mid", "superficial"]

    for name in active_muscles:
        if not name.startswith(f"{side}_"):
            continue  # skip other side
        short = name.replace(f"{side}_", "")
        assigned = False
        for li, layer_muscles in LAYERS.items():
            if short in layer_muscles:
                layers[li].append(name)
                assigned = True
                break
        if not assigned:
            layers[2].append(name)

    for li in range(3):
        print(f"    Layer {li} ({layer_names[li]}): {len(layers[li])} muscles — {layers[li]}")

    # ── Build bone collision meshes infrastructure ────────────────────
    print("[8] Setting up bone collision...")
    # Cache rest transforms
    skel.setPositions(np.zeros(skel.getNumDofs()))
    bone_rest_transforms = {}
    for i in range(skel.getNumBodyNodes()):
        bn = skel.getBodyNode(i)
        wt = bn.getWorldTransform()
        bone_rest_transforms[bn.getName()] = (wt.rotation().copy(), wt.translation().copy())

    def build_bone_trimeshes():
        """Build posed bone trimeshes at current skeleton pose."""
        meshes = []
        for mesh_name, mesh_obj in skeleton_meshes.items():
            if not (hasattr(mesh_obj, 'trimesh') and mesh_obj.trimesh is not None): continue
            body_node = None; body_name = None
            for candidate in [mesh_name, mesh_name + '0']:
                body_node = skel.getBodyNode(candidate)
                if body_node is not None:
                    body_name = candidate; break
            if body_node is None: continue
            R_rest, t_rest = bone_rest_transforms.get(body_name, (np.eye(3), np.zeros(3)))
            R_posed = body_node.getWorldTransform().rotation()
            t_posed = body_node.getWorldTransform().translation()
            verts = mesh_obj.trimesh.vertices.copy()
            local_verts = (R_rest.T @ (verts - t_rest).T).T
            posed_verts = (R_posed @ local_verts.T).T + t_posed
            tm = trimesh.Trimesh(vertices=posed_verts, faces=mesh_obj.trimesh.faces.copy(), process=True)
            meshes.append(tm)
        return meshes

    # ── Precompute per-layer ARAP systems ─────────────────────────────
    print("[9] Building per-layer ARAP systems...")
    layer_systems = []

    for li in range(3):
        layer_muscle_names = layers[li]
        if not layer_muscle_names:
            layer_systems.append(None)
            continue

        layer_active = {n: active_muscles[n] for n in layer_muscle_names if n in active_muscles}
        if not layer_active:
            layer_systems.append(None)
            continue

        # Build unified system for this layer
        muscle_names = list(layer_active.keys())
        global_offset = {}
        offset = 0
        for name in muscle_names:
            global_offset[name] = offset
            offset += layer_active[name].soft_body.num_vertices

        total_verts = offset
        global_rest = np.zeros((total_verts, 3))
        global_fixed_mask = np.zeros(total_verts, dtype=bool)

        for name, mobj in layer_active.items():
            off = global_offset[name]
            n = mobj.soft_body.num_vertices
            global_rest[off:off+n] = mobj.soft_body.rest_positions
            global_fixed_mask[off:off+n] = mobj.soft_body.fixed_mask

        # Build edges
        neighbors = [[] for _ in range(total_verts)]
        edge_weights = {}
        rest_edge_vectors = [{} for _ in range(total_verts)]

        for name, mobj in layer_active.items():
            off = global_offset[name]
            sb = mobj.soft_body
            for edge_idx, (i, j) in enumerate(zip(sb.edge_i, sb.edge_j)):
                gi, gj = off + i, off + j
                neighbors[gi].append(gj); neighbors[gj].append(gi)
                edge_weights[(gi, gj)] = 1.0; edge_weights[(gj, gi)] = 1.0
                rest_edge_vectors[gi][gj] = global_rest[gj] - global_rest[gi]
                rest_edge_vectors[gj][gi] = global_rest[gi] - global_rest[gj]

        # Surface verts for collision candidates
        collision_verts = set()
        surf_faces_per_muscle = {}
        for name, mobj in layer_active.items():
            off = global_offset[name]
            tets = mobj.tet_tetrahedra if hasattr(mobj, 'tet_tetrahedra') else getattr(mobj.soft_body, 'tetrahedra', None)
            if tets is not None:
                sf = extract_surface_triangles(tets)
                sv = set(np.unique(sf).tolist())
                surf_faces_per_muscle[name] = sf
                for v in sv:
                    gv = off + v
                    if not global_fixed_mask[gv]:
                        collision_verts.add(gv)

        # Build backend
        backend = get_backend('taichi')
        backend.build_system(total_verts, neighbors, edge_weights, global_fixed_mask,
                             regularization=1e-6,
                             collision_vertices=collision_verts,
                             collision_weight=args.collision_weight)

        layer_systems.append({
            'backend': backend,
            'muscle_names': muscle_names,
            'layer_active': layer_active,
            'global_offset': global_offset,
            'global_rest': global_rest,
            'global_fixed_mask': global_fixed_mask,
            'neighbors': neighbors,
            'edge_weights': edge_weights,
            'rest_edge_vectors': rest_edge_vectors,
            'collision_verts': collision_verts,
            'surf_faces_per_muscle': surf_faces_per_muscle,
            'total_verts': total_verts,
        })

        print(f"    Layer {li}: {total_verts} verts, {len(collision_verts)} collision candidates")

    # ── Output ───────────────────────────────────────────────────────
    bvh_stem = os.path.splitext(os.path.basename(args.bvh))[0]
    tag = args.region_tag or "layered"
    cache_dir = os.path.join("data", "motion_cache", bvh_stem, tag)
    os.makedirs(cache_dir, exist_ok=True)

    bake_data = {n: {} for n in active_muscles}
    flush_count = 0

    # ── Frame loop ───────────────────────────────────────────────────
    print(f"\n[10] Baking frames {args.start_frame}-{end_frame}...")
    bake_start = time.time()

    for frame in range(args.start_frame, end_frame + 1):
        frame_start = time.time()
        skel.setPositions(motion_bvh.mocap_refs[frame])

        # Build bone collision meshes at current pose
        bone_tms = build_bone_trimeshes()
        obstacle_meshes = list(bone_tms)  # start with bones only

        total_coll = 0

        # Process each layer in depth order
        for li in range(3):
            sys_data = layer_systems[li]
            if sys_data is None:
                continue

            backend = sys_data['backend']
            layer_active = sys_data['layer_active']
            global_offset = sys_data['global_offset']
            global_rest = sys_data['global_rest']
            global_fixed_mask = sys_data['global_fixed_mask']
            collision_verts = sys_data['collision_verts']
            total_verts = sys_data['total_verts']

            # Update positions and fixed targets from skeleton
            # (same as _run_unified_volume_sim — uses muscle-axis blending, not LBS)
            for name, mobj in layer_active.items():
                if hasattr(mobj, '_update_tet_positions_from_skeleton'):
                    mobj._update_tet_positions_from_skeleton(skel)
                if hasattr(mobj, '_update_fixed_targets_from_skeleton'):
                    mobj._update_fixed_targets_from_skeleton(skeleton_meshes, skel)

            # Collect positions from soft body (already updated by skeleton bindings)
            global_positions = np.zeros((total_verts, 3))
            for name, mobj in layer_active.items():
                off = global_offset[name]
                n = mobj.soft_body.num_vertices
                global_positions[off:off+n] = mobj.soft_body.positions

            # Fixed targets
            fixed_indices = np.where(global_fixed_mask)[0]
            fixed_targets = np.zeros((len(fixed_indices), 3))
            for name, mobj in layer_active.items():
                off = global_offset[name]
                if mobj.soft_body.fixed_targets is not None:
                    for local_idx, target in zip(mobj.soft_body.fixed_indices, mobj.soft_body.fixed_targets):
                        gi = off + local_idx
                        idx_in_fixed = np.searchsorted(fixed_indices, gi)
                        if idx_in_fixed < len(fixed_indices) and fixed_indices[idx_in_fixed] == gi:
                            fixed_targets[idx_in_fixed] = target

            # Build collision_target_fn with current obstacles
            fixed_set = set(fixed_indices.tolist())
            coll_fn = make_collision_target_fn(
                obstacle_meshes, collision_verts, fixed_set, margin=args.margin)

            # ARAP solve with collision
            solve_iters = args.settle_iters
            result, iterations, max_disp = backend.solve(
                global_positions, global_rest,
                sys_data['neighbors'], sys_data['edge_weights'], sys_data['rest_edge_vectors'],
                global_fixed_mask, fixed_targets,
                max_iterations=solve_iters, tolerance=2e-3,
                collision_target_fn=coll_fn,
                verbose=(frame == args.start_frame and li == 0))

            # Store results and build obstacle meshes for next layer
            for name, mobj in layer_active.items():
                off = global_offset[name]
                n = mobj.soft_body.num_vertices
                positions = result[off:off+n]
                mobj.soft_body.positions = positions.copy()
                mobj.tet_vertices = positions.astype(np.float32)
                bake_data[name][frame] = positions.astype(np.float32)

                # Build trimesh obstacle for next layer
                sf = sys_data['surf_faces_per_muscle'].get(name)
                if sf is not None:
                    try:
                        tm = trimesh.Trimesh(vertices=positions, faces=sf, process=False)
                        obstacle_meshes.append(tm)
                    except Exception:
                        pass

            # Count collisions for this layer
            if coll_fn:
                targets = coll_fn(result)
                layer_coll = sum(1 for vi, t in targets.items()
                                 if np.linalg.norm(t - result[vi]) > 1e-6)
                total_coll += layer_coll

        frame_dt = time.time() - frame_start
        print(f"  Frame {frame}: {frame_dt:.2f}s, coll={total_coll}", flush=True)

        # Flush
        n_acc = sum(len(fd) for fd in bake_data.values())
        if n_acc >= FLUSH_INTERVAL * len(active_muscles):
            for mname, fd in bake_data.items():
                if not fd: continue
                sf = sorted(fd.keys())
                fp = os.path.join(cache_dir, f"{mname}_chunk_{flush_count:04d}.npz")
                np.savez(fp, frames=np.array(sf, dtype=np.int32),
                         positions=np.array([fd[f] for f in sf], dtype=np.float32))
                fd.clear()
            flush_count += 1
            gc.collect()

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
