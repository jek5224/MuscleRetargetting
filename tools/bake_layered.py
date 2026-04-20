#!/usr/bin/env python3
"""Layered ARAP bake with surface collision projection (Iron Man approach).

Muscles start OUTSIDE bones and are pulled in by ARAP toward attachment points.
Collision projection (every 5 ARAP iterations) prevents vertices from crossing
bone/obstacle surfaces. Direction is always unambiguous — vertices approach from
outside, never start inside.

Deep muscles settle first → become rigid obstacles for mid/superficial layers.

Usage:
    python tools/bake_layered.py --bvh data/motion/walk.bvh --sides L --start-frame 60 --end-frame 70
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


# ---------------------------------------------------------------------------
# Surface / collision helpers
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


def make_collision_target_fn(obstacle_meshes, collision_vertex_set, fixed_mask,
                              margin=0.002, check_interval=10):
    """Create collision_target_fn with cached contains() checks.

    Uses contains() for reliable inside/outside detection (handles concave bones).
    Full check runs every `check_interval` iterations; other iterations reuse
    cached targets. Penalty-in-diagonal gives smooth convergence.

    Returns dict {vertex_idx: target_position}:
      - Inside bone: target = surface + margin (pulls vertex out)
      - Outside: target = current_pos (zero net force)
    """
    sv_arr = np.array(sorted(collision_vertex_set), dtype=np.int64)
    cached_inside = {}  # vi -> target_position (bone surface + margin)
    call_count = [0]

    def collision_target_fn(positions):
        call_count[0] += 1

        # Default: current position for all candidates (zero force)
        targets = {}
        for vi in collision_vertex_set:
            targets[vi] = positions[vi].copy()

        # Only do expensive contains() check every N iterations
        do_full_check = (call_count[0] % check_interval == 1) or not cached_inside

        if do_full_check:
            cached_inside.clear()
            sv_pos = positions[sv_arr]

            for obs_mesh in obstacle_meshes:
                # AABB pre-filter
                bmin = obs_mesh.bounds[0] - 0.005
                bmax = obs_mesh.bounds[1] + 0.005
                in_bbox = np.all((sv_pos >= bmin) & (sv_pos <= bmax), axis=1)
                if not np.any(in_bbox):
                    continue

                bbox_sv = sv_arr[in_bbox]
                bbox_pos = sv_pos[in_bbox]

                # Reliable inside/outside via contains()
                try:
                    inside = obs_mesh.contains(bbox_pos)
                except Exception:
                    continue
                if not np.any(inside):
                    continue

                # Compute surface targets for inside vertices
                inside_sv = bbox_sv[inside]
                inside_pos = bbox_pos[inside]
                closest, _, face_ids = trimesh.proximity.closest_point(obs_mesh, inside_pos)
                normals = obs_mesh.face_normals[face_ids]

                for k in range(len(inside_sv)):
                    vi = int(inside_sv[k])
                    if fixed_mask[vi]:
                        continue
                    target = closest[k] + normals[k] * margin
                    cached_inside[vi] = target
                    targets[vi] = target
        else:
            # Reuse cached targets — re-check positions against cached surface points
            for vi, cached_target in cached_inside.items():
                targets[vi] = cached_target

        return targets

    return collision_target_fn



# ---------------------------------------------------------------------------
# Per-layer unified ARAP solve with collision projection
# ---------------------------------------------------------------------------
def run_layer_sim_with_collision(layer_muscles, frozen_muscles, skeleton_meshes, skel,
                                  obstacle_meshes, inter_muscle_constraints,
                                  layer_cache, backend_name,
                                  max_iterations=100, tolerance=1e-4,
                                  collision_margin=0.002, collision_weight=10.0,
                                  check_interval=10, verbose=False):
    """Run unified ARAP for one layer with collision projection.

    Muscles start outside obstacles. ARAP pulls toward attachments.
    Collision projection (every 5 iters) prevents surface crossing.

    frozen_muscles: dict of earlier-layer muscles (already settled). Included in
    system as fixed vertices so cross-layer distance constraints work.
    """
    muscle_names = list(layer_muscles.keys())
    frozen_names = list(frozen_muscles.keys())
    all_names = muscle_names + frozen_names
    total_verts = (sum(layer_muscles[n].soft_body.num_vertices for n in muscle_names)
                   + sum(frozen_muscles[n].soft_body.num_vertices for n in frozen_names))

    # Step 1: Update positions and fixed targets from skeleton
    for name, mobj in layer_muscles.items():
        if hasattr(mobj, '_update_tet_positions_from_skeleton'):
            mobj._update_tet_positions_from_skeleton(skel)
        if hasattr(mobj, '_update_fixed_targets_from_skeleton'):
            mobj._update_fixed_targets_from_skeleton(skeleton_meshes, skel)

    # Check if cached topology is still valid
    cache_valid = (layer_cache.get('muscle_names') == muscle_names
                   and layer_cache.get('frozen_names') == frozen_names
                   and layer_cache.get('total_verts') == total_verts)

    if not cache_valid:
        global_offset = {}
        offset_accum = 0
        # Current layer muscles first
        for name in muscle_names:
            global_offset[name] = offset_accum
            offset_accum += layer_muscles[name].soft_body.num_vertices
        # Frozen (earlier layer) muscles after
        for name in frozen_names:
            global_offset[name] = offset_accum
            offset_accum += frozen_muscles[name].soft_body.num_vertices

        global_rest_positions = np.zeros((total_verts, 3))
        global_fixed_mask = np.zeros(total_verts, dtype=bool)

        # Current layer: use their own fixed_mask
        for name, mobj in layer_muscles.items():
            offset = global_offset[name]
            n = mobj.soft_body.num_vertices
            global_rest_positions[offset:offset+n] = mobj.soft_body.rest_positions
            global_fixed_mask[offset:offset+n] = mobj.soft_body.fixed_mask

        # Frozen muscles: ALL vertices are fixed
        for name, mobj in frozen_muscles.items():
            offset = global_offset[name]
            n = mobj.soft_body.num_vertices
            global_rest_positions[offset:offset+n] = mobj.soft_body.rest_positions
            global_fixed_mask[offset:offset+n] = True  # All frozen

        # Build edges from tet connectivity (current layer only)
        all_edges = []
        for name, mobj in layer_muscles.items():
            offset = global_offset[name]
            sb = mobj.soft_body
            for edge_idx, (i, j) in enumerate(zip(sb.edge_i, sb.edge_j)):
                if hasattr(sb, 'rest_lengths') and sb.rest_lengths is not None and edge_idx < len(sb.rest_lengths):
                    rest_len = sb.rest_lengths[edge_idx]
                else:
                    rest_len = np.linalg.norm(sb.rest_positions[j] - sb.rest_positions[i])
                all_edges.append((offset + i, offset + j, rest_len, 1.0))

        # Inter-muscle constraints as edges (within-layer AND cross-layer)
        all_muscle_set = set(all_names)
        n_within = 0
        n_cross = 0
        for constraint in inter_muscle_constraints:
            name1, v1_idx, v1_fixed, name2, v2_idx, v2_fixed, rest_dist = constraint
            if name1 in all_muscle_set and name2 in all_muscle_set:
                gi = global_offset[name1] + v1_idx
                gj = global_offset[name2] + v2_idx
                all_edges.append((gi, gj, rest_dist, 1.0))
                if name1 in set(muscle_names) and name2 in set(muscle_names):
                    n_within += 1
                else:
                    n_cross += 1

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

        # Build collision vertex set: free surface verts (current layer only)
        collision_vertex_set = set()
        for name, mobj in layer_muscles.items():
            offset = global_offset[name]
            if hasattr(mobj, '_surf_verts'):
                for lv in mobj._surf_verts:
                    gv = offset + lv
                    if not global_fixed_mask[gv]:
                        collision_vertex_set.add(gv)

        layer_cache.update({
            'global_offset': global_offset,
            'total_verts': total_verts,
            'global_rest_positions': global_rest_positions,
            'global_fixed_mask': global_fixed_mask,
            'neighbors': neighbors,
            'edge_weights': edge_weights,
            'rest_edge_vectors': rest_edge_vectors,
            'muscle_names': muscle_names,
            'frozen_names': frozen_names,
            'collision_vertex_set': collision_vertex_set,
        })
        if verbose:
            print(f"    Built topology: {total_verts} verts, {len(all_edges)} edges, "
                  f"{len(collision_vertex_set)} collision candidates, "
                  f"{n_within} within-layer + {n_cross} cross-layer constraints")

    # Unpack cache
    global_offset = layer_cache['global_offset']
    global_rest_positions = layer_cache['global_rest_positions']
    global_fixed_mask = layer_cache['global_fixed_mask']
    neighbors = layer_cache['neighbors']
    edge_weights = layer_cache['edge_weights']
    rest_edge_vectors = layer_cache['rest_edge_vectors']
    collision_vertex_set = layer_cache['collision_vertex_set']

    # Compute LBS positions (skeleton-following) for current layer
    global_lbs = np.zeros((total_verts, 3))
    for name, mobj in layer_muscles.items():
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

    # Frozen muscles: use their current settled positions
    for name, mobj in frozen_muscles.items():
        offset = global_offset[name]
        n = mobj.soft_body.num_vertices
        global_lbs[offset:offset+n] = mobj.soft_body.get_positions()

    # Warm-start: 70% LBS + 30% previous solution (current layer only)
    prev_solution = layer_cache.get('prev_solution', None)
    if prev_solution is not None and prev_solution.shape[0] == total_verts:
        global_positions = 0.7 * global_lbs + 0.3 * prev_solution
        fixed_idx = np.where(global_fixed_mask)[0]
        global_positions[fixed_idx] = global_lbs[fixed_idx]
    else:
        global_positions = global_lbs.copy()

    # Fixed targets: bone attachments for current layer + all positions for frozen
    fixed_indices = np.where(global_fixed_mask)[0]
    global_fixed_targets = {}
    # Current layer: origin/insertion targets from skeleton
    for name, mobj in layer_muscles.items():
        offset = global_offset[name]
        if mobj.soft_body.fixed_targets is not None and len(mobj.soft_body.fixed_indices) > 0:
            for local_idx, target in zip(mobj.soft_body.fixed_indices, mobj.soft_body.fixed_targets):
                global_fixed_targets[offset + local_idx] = target
    # Frozen muscles: all vertices are fixed at their settled positions
    for name, mobj in frozen_muscles.items():
        offset = global_offset[name]
        settled_pos = mobj.soft_body.get_positions()
        for local_idx in range(mobj.soft_body.num_vertices):
            global_fixed_targets[offset + local_idx] = settled_pos[local_idx]
    fixed_targets_array = np.array([global_fixed_targets.get(i, global_rest_positions[i])
                                     for i in fixed_indices])

    # First frame: offset each muscle outward from its bone axis (Iron Man start)
    is_first_frame = prev_solution is None
    if is_first_frame and len(obstacle_meshes) > 0:
        n_offset = 0
        offset_amount = 0.02  # 20mm detachment
        for name, mobj in layer_muscles.items():
            off = global_offset[name]
            n = mobj.soft_body.num_vertices

            # Compute direction: muscle centroid relative to bone midpoint at rest
            if hasattr(mobj, 'skinning_bones') and len(mobj.skinning_bones) >= 2:
                bone_positions = []
                for bname in mobj.skinning_bones:
                    bn = skel.getBodyNode(bname)
                    if bn is not None:
                        bone_positions.append(bn.getWorldTransform().translation())
                if len(bone_positions) >= 2:
                    bone_mid = np.mean(bone_positions, axis=0)
                    muscle_centroid = global_positions[off:off+n].mean(axis=0)
                    direction = muscle_centroid - bone_mid
                    dist = np.linalg.norm(direction)
                    if dist > 1e-6:
                        direction /= dist
                        # Offset all FREE vertices of this muscle outward
                        for vi in range(off, off + n):
                            if not global_fixed_mask[vi]:
                                global_positions[vi] += direction * offset_amount
                                n_offset += 1
        if verbose and n_offset > 0:
            print(f"    First frame: offset {n_offset} vertices outward (Iron Man start)")

    # Get or create backend
    backend = layer_cache.get('backend', None)
    if backend is None or getattr(backend, '_backend_name', None) != backend_name:
        backend = get_backend(backend_name)
        backend._backend_name = backend_name
        layer_cache['backend'] = backend

    # Build system with collision_weight on diagonal for penalty approach
    need_build = not cache_valid or getattr(backend, '_splu', None) is None
    if need_build:
        backend.build_system(total_verts, neighbors, edge_weights, global_fixed_mask,
                             regularization=1e-6,
                             collision_vertices=collision_vertex_set,
                             collision_weight=collision_weight)

    # Create collision target function (cached contains() every N iterations)
    coll_fn = make_collision_target_fn(
        obstacle_meshes, collision_vertex_set, global_fixed_mask,
        margin=collision_margin, check_interval=check_interval)

    # First frame: extra iterations; convergence tolerance accounts for
    # penalty oscillation (~3e-4 between collision and ARAP shape energy)
    solve_iters = max_iterations * 2 if is_first_frame else max_iterations

    # Solve ARAP with penalty-in-diagonal collision
    # Use slightly relaxed tolerance to avoid oscillation between penalty and shape
    solve_tol = max(tolerance, 5e-4)
    global_positions, iterations, max_disp = backend.solve(
        global_positions, global_rest_positions, neighbors, edge_weights, rest_edge_vectors,
        global_fixed_mask, fixed_targets_array,
        max_iterations=solve_iters, tolerance=solve_tol,
        verbose=verbose,
        collision_target_fn=coll_fn,
    )

    # Stash for warm-start
    layer_cache['prev_solution'] = global_positions.copy()

    # Write back to current layer muscles only (frozen are unchanged)
    for name, mobj in layer_muscles.items():
        offset = global_offset[name]
        n = mobj.soft_body.num_vertices
        mobj.soft_body.positions = global_positions[offset:offset+n].copy()
        mobj.tet_vertices = mobj.soft_body.get_positions().astype(np.float32)

    return iterations, max_disp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Layered ARAP bake with collision projection")
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--muscles", default=".last_loaded_muscles.json")
    parser.add_argument("--settle-iters", type=int, default=150)
    parser.add_argument("--constraint-threshold", type=float, default=0.015)
    parser.add_argument("--collision-margin", type=float, default=0.002)
    parser.add_argument("--collision-weight", type=float, default=10.0)
    parser.add_argument("--check-interval", type=int, default=20,
                        help="Run contains() check every N ARAP iterations (default: 20)")
    parser.add_argument("--backend", choices=["auto","taichi","gpu","cpu"], default="auto")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--sides", default="L")
    parser.add_argument("--region-tag", default="layered")
    args = parser.parse_args()

    # Resolve backend
    backend_name = args.backend
    if backend_name == "auto":
        if check_taichi_available(): backend_name = "taichi"
        elif check_gpu_available(): backend_name = "gpu"
        else: backend_name = "cpu"
    print(f"ARAP backend: {backend_name.upper()}")

    # ── Load everything ──────────────────────────────────────────────────
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

    # ── Precompute surface data ───────────────────────────────────────────
    print("[6] Precomputing surface data...")
    precompute_surface_data(active_all)
    bone_rest_transforms = cache_bone_rest_transforms(skel)
    total_sv = sum(len(getattr(m, '_surf_verts', [])) for m in active_all.values())
    print(f"    {total_sv} surface vertices for collision")

    # ── Classify into layers ──────────────────────────────────────────────
    print("[7] Classifying into layers...")
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

    # ── Find inter-muscle constraints ─────────────────────────────────────
    print("[8] Finding inter-muscle constraints...")
    global_ctx = SimpleNamespace(
        env=SimpleNamespace(skel=skel, mesh_info=mesh_info),
        zygote_muscle_meshes=active_all,
        zygote_skeleton_meshes=skeleton_meshes,
        inter_muscle_constraints=[],
        inter_muscle_constraint_threshold=args.constraint_threshold,
        coupled_as_unified_volume=True,
        use_gpu_arap=False,
        use_taichi_arap=False,
        use_muscle_aware_arap=True,
        use_fem_sim=False,
        use_vbd_sim=False,
        use_pn_sim=False,
        _unified_arap_backend=None,
        _unified_sim_cache=None,
    )
    n_global = find_inter_muscle_constraints(global_ctx)
    all_constraints = global_ctx.inter_muscle_constraints
    print(f"    {n_global} global constraints")

    # Filter constraints per-layer
    cumulative_muscles = set()
    layer_constraints = [[], [], []]
    for li in range(3):
        cumulative_muscles.update(layer_muscles[li])
        layer_set = set(layer_muscles[li])
        for c in all_constraints:
            name1, v1, f1, name2, v2, f2, dist = c
            if name1 in cumulative_muscles and name2 in cumulative_muscles:
                if name1 in layer_set or name2 in layer_set:
                    layer_constraints[li].append(c)
        print(f"    Layer {li}: {len(layer_constraints[li])} constraints")

    # ── Load BVH ──────────────────────────────────────────────────────────
    print("[9] Loading BVH...")
    t_frame = _detect_bvh_tframe(args.bvh)
    motion_bvh = MyBVH(args.bvh, bvh_info, skel, T_frame=t_frame)
    n_frames = motion_bvh.mocap_refs.shape[0]
    end_frame = args.end_frame if args.end_frame is not None else n_frames - 1
    end_frame = min(end_frame, n_frames - 1)
    total_frames = end_frame - args.start_frame + 1

    # ── Output ────────────────────────────────────────────────────────────
    bvh_stem = os.path.splitext(os.path.basename(args.bvh))[0]
    cache_dir = os.path.join("data", "motion_cache", bvh_stem, args.region_tag)
    os.makedirs(cache_dir, exist_ok=True)

    import glob as glob_mod
    for old in glob_mod.glob(os.path.join(cache_dir, "*_chunk_*.npz")):
        os.remove(old)

    bake_data = {n: {} for n in active_all}
    flush_count = 0
    bake_start = time.time()

    # Per-layer caches
    layer_caches = [{}, {}, {}]

    # ── Frame loop ────────────────────────────────────────────────────────
    print(f"\n[10] Baking frames {args.start_frame}-{end_frame}...")

    for frame in range(args.start_frame, end_frame + 1):
        frame_start = time.time()
        skel.setPositions(motion_bvh.mocap_refs[frame])

        # Build bone collision meshes at current pose
        bone_tms = build_bone_collision_meshes(skeleton_meshes, skel, bone_rest_transforms)
        obstacle_meshes = list(bone_tms)  # Bones always in obstacle set
        settled_muscles = {}  # Accumulate settled earlier-layer muscles

        # Process each layer in depth order
        for li in range(3):
            if not layer_muscles[li]:
                continue

            layer_active = {n: active_all[n] for n in layer_muscles[li]}

            # Disable waypoint updates during baking
            for mname, mobj in layer_active.items():
                mobj.waypoints_from_tet_sim = False
                mobj._baking_mode = True

            # Run ARAP with penalty-in-diagonal collision
            # frozen_muscles = all muscles from earlier layers (already settled)
            iters, max_disp = run_layer_sim_with_collision(
                layer_active, settled_muscles, skeleton_meshes, skel,
                obstacle_meshes, layer_constraints[li],
                layer_caches[li], backend_name,
                max_iterations=args.settle_iters, tolerance=1e-4,
                collision_margin=args.collision_margin,
                collision_weight=args.collision_weight,
                check_interval=args.check_interval,
                verbose=(frame == args.start_frame and li == 0))

            # Restore flags and capture positions
            for mname, mobj in layer_active.items():
                mobj.waypoints_from_tet_sim = True
                mobj._baking_mode = False
                bake_data[mname][frame] = mobj.soft_body.get_positions().astype(np.float32)

            # Settled muscles become obstacles AND frozen constraints for next layer
            for mname, mobj in layer_active.items():
                settled_muscles[mname] = mobj
                if hasattr(mobj, '_surf_faces'):
                    verts = mobj.soft_body.get_positions()
                    tm = trimesh.Trimesh(vertices=verts,
                                         faces=mobj._surf_faces.copy(), process=True)
                    obstacle_meshes.append(tm)

        frame_dt = time.time() - frame_start
        frames_done = frame - args.start_frame + 1
        elapsed = time.time() - bake_start
        avg = elapsed / frames_done
        remaining = avg * (total_frames - frames_done)
        print(f"  Frame {frame}: {frame_dt:.2f}s  avg {avg:.2f}s/frame  ETA {remaining:.0f}s",
              flush=True)

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
