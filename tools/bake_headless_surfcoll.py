#!/usr/bin/env python3
"""Headless ARAP muscle baking with surface collision inside the ARAP iteration.

Collision is integrated as penalty-in-diagonal in the ARAP global solve:
  - build_system: collision_weight added to diagonal for surface vertices near bones
  - Every ARAP iteration: collision_target_fn detects bone penetration (vertex + edge)
    and sets targets; the linear solve converges smoothly, no oscillation.

Usage:
    python tools/bake_headless_surfcoll.py --bvh data/motion/walk.bvh
    python tools/bake_headless_surfcoll.py --bvh data/motion/dance.bvh --settle-iters 80
"""
import argparse
import gc
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import trimesh
from scipy.spatial import cKDTree

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


# ---------------------------------------------------------------------------
# Surface collision helpers
# ---------------------------------------------------------------------------
def extract_surface_triangles(tet_elements):
    """Extract boundary faces from tet mesh (faces belonging to exactly 1 tet)."""
    face_count = Counter()
    face_orient = {}
    for t in tet_elements:
        v0, v1, v2, v3 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        faces = [(v0, v2, v1), (v0, v1, v3), (v1, v2, v3), (v0, v3, v2)]
        for f in faces:
            key = tuple(sorted(f))
            face_count[key] += 1
            if key not in face_orient:
                face_orient[key] = f
    return np.array([face_orient[k] for k, c in face_count.items() if c == 1],
                    dtype=np.int64)


def extract_edges_from_faces(faces):
    """Extract unique edges from surface faces."""
    edges = set()
    for f in faces:
        for i in range(3):
            e = (min(int(f[i]), int(f[(i + 1) % 3])),
                 max(int(f[i]), int(f[(i + 1) % 3])))
            edges.add(e)
    return np.array(sorted(edges), dtype=np.int64)


def precompute_surface_data(active_muscles):
    """Precompute surface triangles and edges for each muscle."""
    for mname, mobj in active_muscles.items():
        tets = getattr(mobj, 'tet_tetrahedra', None)
        if tets is None and hasattr(mobj, 'soft_body'):
            tets = getattr(mobj.soft_body, 'tetrahedra', None)
        if tets is None:
            continue
        sf = extract_surface_triangles(tets)
        se = extract_edges_from_faces(sf)
        sv = sorted(set(np.unique(sf).tolist()))
        mobj._surf_faces = sf
        mobj._surf_edges = se
        mobj._surf_verts = sv
        fixed_set = set()
        if hasattr(mobj, 'soft_body') and mobj.soft_body is not None:
            fi = mobj.soft_body.fixed_indices
            if fi is not None:
                fixed_set = set(int(i) for i in fi)
        mobj._surf_fixed = fixed_set
    total_sf = sum(len(getattr(m, '_surf_faces', [])) for m in active_muscles.values())
    total_se = sum(len(getattr(m, '_surf_edges', [])) for m in active_muscles.values())
    print(f"  Surface data: {total_sf} faces, {total_se} edges")


def build_bone_collision_meshes(skeleton_meshes, skel, rest_transforms):
    """Build bone trimeshes at current skeleton pose."""
    bone_meshes = []
    for mesh_name, mesh_obj in skeleton_meshes.items():
        if not (hasattr(mesh_obj, 'trimesh') and mesh_obj.trimesh is not None):
            continue
        body_node = None
        body_name = None
        for candidate in [mesh_name, mesh_name + '0']:
            body_node = skel.getBodyNode(candidate)
            if body_node is not None:
                body_name = candidate
                break
        if body_node is None:
            continue
        R_rest, t_rest = rest_transforms.get(body_name, (np.eye(3), np.zeros(3)))
        R_posed = body_node.getWorldTransform().rotation()
        t_posed = body_node.getWorldTransform().translation()
        verts = mesh_obj.trimesh.vertices.copy()
        local_verts = (R_rest.T @ (verts - t_rest).T).T
        posed_verts = (R_posed @ local_verts.T).T + t_posed
        tm = trimesh.Trimesh(vertices=posed_verts,
                             faces=mesh_obj.trimesh.faces.copy(), process=True)
        bone_meshes.append(tm)
    return bone_meshes


def cache_bone_rest_transforms(skel):
    """Cache rest-pose transforms for all body nodes."""
    saved_pos = skel.getPositions().copy()
    skel.setPositions(np.zeros(skel.getNumDofs()))
    rest_transforms = {}
    for i in range(skel.getNumBodyNodes()):
        bn = skel.getBodyNode(i)
        wt = bn.getWorldTransform()
        rest_transforms[bn.getName()] = (wt.rotation().copy(), wt.translation().copy())
    skel.setPositions(saved_pos)
    return rest_transforms


def make_collision_target_fn(bone_trimeshes, global_surf_edges, global_surf_verts,
                              global_fixed_mask, collision_vertex_set, margin=0.002):
    """Create a collision_target_fn for the ARAP solver's penalty-in-diagonal approach.

    Called every ARAP iteration with current positions.
    Returns dict {global_vertex_idx: target_position}:
      - Inside bone: target = bone surface + margin (pulls vertex out)
      - Outside but in collision_vertex_set: target = current_pos (zero net force)
    """
    bone_kdtree = cKDTree(np.vstack([bm.vertices for bm in bone_trimeshes]))

    def collision_target_fn(positions):
        targets = {}
        # For all collision candidates, default to current pos (zero force)
        for vi in collision_vertex_set:
            targets[vi] = positions[vi].copy()

        # Phase 1: Vertex-bone via contains()
        sv_arr = np.array(sorted(collision_vertex_set), dtype=np.int64)
        sv_pos = positions[sv_arr]
        dists_kd, _ = bone_kdtree.query(sv_pos)
        near_mask = dists_kd < 0.03
        if np.any(near_mask):
            near_sv = sv_arr[near_mask]
            near_pos = positions[near_sv]
            for bone_mesh in bone_trimeshes:
                try:
                    bmin = bone_mesh.bounds[0] - 0.03
                    bmax = bone_mesh.bounds[1] + 0.03
                    in_bbox = np.all((near_pos >= bmin) & (near_pos <= bmax), axis=1)
                    if not np.any(in_bbox):
                        continue
                    bbox_pos = near_pos[in_bbox]
                    bbox_sv = near_sv[in_bbox]
                    inside = bone_mesh.contains(bbox_pos)
                    if not np.any(inside):
                        continue
                    inside_pos = bbox_pos[inside]
                    inside_sv = bbox_sv[inside]
                    closest, _, face_ids = trimesh.proximity.closest_point(
                        bone_mesh, inside_pos)
                    normals = bone_mesh.face_normals[face_ids]
                    for k in range(len(inside_sv)):
                        vi = int(inside_sv[k])
                        if global_fixed_mask[vi]:
                            continue
                        targets[vi] = closest[k] + normals[k] * margin
                except Exception:
                    continue

        # Phase 2: Edge-bone via ray-cast
        if len(global_surf_edges) > 0:
            edge_v0 = positions[global_surf_edges[:, 0]]
            edge_v1 = positions[global_surf_edges[:, 1]]
            d0, _ = bone_kdtree.query(edge_v0)
            d1, _ = bone_kdtree.query(edge_v1)
            edge_near = (d0 < 0.02) | (d1 < 0.02)
            if np.any(edge_near):
                candidate_edges = global_surf_edges[edge_near]
                origins = positions[candidate_edges[:, 0]]
                endpoints = positions[candidate_edges[:, 1]]
                dirs = endpoints - origins
                lengths = np.linalg.norm(dirs, axis=1)
                valid = lengths > 1e-10
                if np.any(valid):
                    dirs_norm = dirs.copy()
                    dirs_norm[valid] /= lengths[valid, None]
                    for bone_mesh in bone_trimeshes:
                        try:
                            locations, ray_idx, tri_idx = bone_mesh.ray.intersects_location(
                                origins[valid], dirs_norm[valid], multiple_hits=True)
                        except Exception:
                            continue
                        if len(locations) == 0:
                            continue
                        valid_edges = candidate_edges[valid]
                        for loc, ri, ti in zip(locations, ray_idx, tri_idx):
                            t_param = np.dot(loc - origins[valid][ri], dirs_norm[valid][ri])
                            edge_len = lengths[valid][ri]
                            if 0 < t_param < edge_len:
                                v0i, v1i = int(valid_edges[ri, 0]), int(valid_edges[ri, 1])
                                face_normal = bone_mesh.face_normals[ti]
                                face_center = bone_mesh.triangles[ti].mean(axis=0)
                                for vi in [v0i, v1i]:
                                    if global_fixed_mask[vi]:
                                        continue
                                    di = np.dot(positions[vi] - face_center, face_normal)
                                    if di < 0:
                                        cp, _, fid = trimesh.proximity.closest_point(
                                            bone_mesh, positions[vi:vi + 1])
                                        fn = bone_mesh.face_normals[fid[0]]
                                        targets[vi] = cp[0] + fn * margin
        return targets

    return collision_target_fn


# ---------------------------------------------------------------------------
# Unified ARAP sim with collision (inlined from _run_unified_volume_sim)
# ---------------------------------------------------------------------------
def run_unified_sim_with_collision(ctx, active_muscles, skeleton_meshes, skel,
                                    bone_rest_transforms, bone_trimeshes,
                                    max_iterations=100, tolerance=1e-4,
                                    collision_weight=10.0, collision_margin=0.002,
                                    verbose=True):
    """Run unified ARAP sim with surface collision integrated in the iteration loop."""
    import time as _time

    if ctx.use_taichi_arap:
        backend_name = 'taichi'
    elif ctx.use_gpu_arap:
        backend_name = 'gpu'
    else:
        backend_name = 'cpu'

    # Step 1: Update positions and fixed targets from skeleton
    for name, mobj in active_muscles.items():
        if hasattr(mobj, '_update_tet_positions_from_skeleton'):
            mobj._update_tet_positions_from_skeleton(skel)
        if hasattr(mobj, '_update_fixed_targets_from_skeleton'):
            mobj._update_fixed_targets_from_skeleton(skeleton_meshes, skel)

    # Build or reuse cached topology
    muscle_names = list(active_muscles.keys())
    total_verts = sum(active_muscles[n].soft_body.num_vertices for n in muscle_names)

    cache = getattr(ctx, '_unified_sim_cache', None)
    cache_valid = (cache is not None
                   and cache['muscle_names'] == muscle_names
                   and cache['total_verts'] == total_verts)

    if not cache_valid:
        global_offset = {}
        offset_accum = 0
        for name in muscle_names:
            global_offset[name] = offset_accum
            offset_accum += active_muscles[name].soft_body.num_vertices

        global_rest_positions = np.zeros((total_verts, 3))
        global_fixed_mask = np.zeros(total_verts, dtype=bool)
        for name, mobj in active_muscles.items():
            offset = global_offset[name]
            n = mobj.soft_body.num_vertices
            global_rest_positions[offset:offset+n] = mobj.soft_body.rest_positions
            global_fixed_mask[offset:offset+n] = mobj.soft_body.fixed_mask

        # Build edges
        all_edges = []
        for name, mobj in active_muscles.items():
            offset = global_offset[name]
            sb = mobj.soft_body
            for edge_idx, (i, j) in enumerate(zip(sb.edge_i, sb.edge_j)):
                if hasattr(sb, 'rest_lengths') and sb.rest_lengths is not None and edge_idx < len(sb.rest_lengths):
                    rest_len = sb.rest_lengths[edge_idx]
                else:
                    rest_len = np.linalg.norm(sb.rest_positions[j] - sb.rest_positions[i])
                all_edges.append((offset + i, offset + j, rest_len, 1.0))

        # Inter-muscle constraints as edges
        for constraint in ctx.inter_muscle_constraints:
            name1, v1_idx, v1_fixed, name2, v2_idx, v2_fixed, rest_dist = constraint
            if name1 in active_muscles and name2 in active_muscles:
                gi = global_offset[name1] + v1_idx
                gj = global_offset[name2] + v2_idx
                all_edges.append((gi, gj, rest_dist, 1.0))

        neighbors = [[] for _ in range(total_verts)]
        edge_weights = {}
        rest_edge_vectors = [{} for _ in range(total_verts)]
        for gi, gj, rest_len, weight in all_edges:
            neighbors[gi].append(gj)
            neighbors[gj].append(gi)
            edge_weights[(gi, gj)] = weight
            edge_weights[(gj, gi)] = weight
            rest_edge_vectors[gi][gj] = global_rest_positions[gj] - global_rest_positions[gi]
            rest_edge_vectors[gj][gi] = global_rest_positions[gi] - global_rest_positions[gj]

        # Build collision vertex set: free surface verts (global indices)
        collision_vertex_set = set()
        global_surf_edges_list = []
        for name, mobj in active_muscles.items():
            offset = global_offset[name]
            if hasattr(mobj, '_surf_verts'):
                for lv in mobj._surf_verts:
                    gv = offset + lv
                    if not global_fixed_mask[gv]:
                        collision_vertex_set.add(gv)
            if hasattr(mobj, '_surf_edges'):
                for e in mobj._surf_edges:
                    global_surf_edges_list.append([offset + int(e[0]), offset + int(e[1])])
        global_surf_edges = np.array(global_surf_edges_list, dtype=np.int64) if global_surf_edges_list else np.zeros((0, 2), dtype=np.int64)

        ctx._unified_sim_cache = {
            'global_offset': global_offset,
            'total_verts': total_verts,
            'global_rest_positions': global_rest_positions,
            'global_fixed_mask': global_fixed_mask,
            'neighbors': neighbors,
            'edge_weights': edge_weights,
            'rest_edge_vectors': rest_edge_vectors,
            'muscle_names': muscle_names,
            'collision_vertex_set': collision_vertex_set,
            'global_surf_edges': global_surf_edges,
        }
        print(f"  Built topology: {total_verts} verts, {len(all_edges)} edges, "
              f"{len(collision_vertex_set)} collision candidates")
    else:
        print(f"  Reusing cached topology ({total_verts} verts)")

    cache = ctx._unified_sim_cache
    global_offset = cache['global_offset']
    global_rest_positions = cache['global_rest_positions']
    global_fixed_mask = cache['global_fixed_mask']
    neighbors = cache['neighbors']
    edge_weights = cache['edge_weights']
    rest_edge_vectors = cache['rest_edge_vectors']
    collision_vertex_set = cache['collision_vertex_set']
    global_surf_edges = cache['global_surf_edges']

    # Compute LBS positions
    global_lbs = np.zeros((total_verts, 3))
    for name, mobj in active_muscles.items():
        offset = global_offset[name]
        n = mobj.soft_body.num_vertices
        rest = mobj.soft_body.rest_positions
        if hasattr(mobj, 'skinning_weights') and mobj.skinning_weights is not None and len(mobj.skinning_bones) > 0:
            lbs = np.zeros((n, 3))
            for bone_idx, bone_name in enumerate(mobj.skinning_bones):
                body_node = skel.getBodyNode(bone_name)
                if body_node is None:
                    continue
                R = body_node.getWorldTransform().rotation()
                t = body_node.getWorldTransform().translation()
                if bone_name in mobj.soft_body_initial_transforms:
                    R0, t0 = mobj.soft_body_initial_transforms[bone_name]
                else:
                    continue
                w = mobj.skinning_weights[:, bone_idx:bone_idx+1]
                local = (R0.T @ (rest - t0).T).T
                deformed = (R @ local.T).T + t
                lbs += w * deformed
            global_lbs[offset:offset+n] = lbs
        else:
            global_lbs[offset:offset+n] = mobj.soft_body.positions

    # Warm-start blend
    prev_solution = cache.get('prev_solution', None) if cache_valid else None
    if prev_solution is not None and prev_solution.shape[0] == total_verts:
        global_positions = 0.7 * global_lbs + 0.3 * prev_solution
        fixed_idx = np.where(global_fixed_mask)[0]
        global_positions[fixed_idx] = global_lbs[fixed_idx]
    else:
        global_positions = global_lbs.copy()

    # Fixed targets
    fixed_indices = np.where(global_fixed_mask)[0]
    global_fixed_targets = {}
    for name, mobj in active_muscles.items():
        offset = global_offset[name]
        if mobj.soft_body.fixed_targets is not None and len(mobj.soft_body.fixed_indices) > 0:
            for local_idx, target in zip(mobj.soft_body.fixed_indices, mobj.soft_body.fixed_targets):
                global_fixed_targets[offset + local_idx] = target
    fixed_targets_array = np.array([global_fixed_targets.get(i, global_rest_positions[i])
                                     for i in fixed_indices])

    # Get or create backend
    cached_backend = getattr(ctx, '_unified_arap_backend', None)
    if cached_backend is not None and getattr(cached_backend, '_backend_name', None) == backend_name:
        backend = cached_backend
    else:
        backend = get_backend(backend_name)
        backend._backend_name = backend_name
        ctx._unified_arap_backend = backend

    # Build system with collision weight on diagonal
    need_build = (not cache_valid
                  or (getattr(backend, '_splu', None) is None
                      and getattr(backend, '_scipy_solver', None) is None))
    if need_build:
        t0 = _time.time()
        backend.build_system(total_verts, neighbors, edge_weights, global_fixed_mask,
                             regularization=1e-6,
                             collision_vertices=collision_vertex_set,
                             collision_weight=collision_weight)
        print(f"  System built in {_time.time() - t0:.3f}s (collision_weight={collision_weight})")

    # Create collision target function
    coll_fn = make_collision_target_fn(
        bone_trimeshes, global_surf_edges, np.array(sorted(collision_vertex_set), dtype=np.int64),
        global_fixed_mask, collision_vertex_set, margin=collision_margin)

    # First frame gets more iterations
    is_first_frame = prev_solution is None
    solve_iters = max_iterations * 10 if is_first_frame else max_iterations

    # Solve with collision inside the iteration loop
    t0 = _time.time()
    global_positions, iterations, max_disp = backend.solve(
        global_positions, global_rest_positions, neighbors, edge_weights, rest_edge_vectors,
        global_fixed_mask, fixed_targets_array,
        max_iterations=solve_iters, tolerance=tolerance,
        verbose=verbose,
        collision_target_fn=coll_fn,
    )
    dt = _time.time() - t0
    print(f"  ARAP+collision solved in {dt:.3f}s ({iterations} iters, max_disp={max_disp:.2e})")

    # Stash for warm-start
    ctx._unified_sim_cache['prev_solution'] = global_positions.copy()

    # Write back to muscles
    for name, mobj in active_muscles.items():
        offset = global_offset[name]
        n = mobj.soft_body.num_vertices
        mobj.soft_body.positions = global_positions[offset:offset+n].copy()
        mobj.tet_vertices = mobj.soft_body.get_positions().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Headless ARAP muscle baking with integrated surface collision")
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

    # Precompute surface data for collision
    print("Precomputing surface data for collision...")
    precompute_surface_data(active_muscles)
    bone_rest_transforms = cache_bone_rest_transforms(skel)

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

        # Build bone collision meshes at current pose
        bone_tms = build_bone_collision_meshes(skeleton_meshes, skel, bone_rest_transforms)

        # Run ARAP with surface collision inside the iteration loop
        run_unified_sim_with_collision(
            ctx, active_muscles, skeleton_meshes, skel,
            bone_rest_transforms, bone_tms,
            max_iterations=args.settle_iters, tolerance=1e-4,
            collision_weight=10.0, collision_margin=0.002,
            verbose=(frame == start_frame))

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
            f""
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
