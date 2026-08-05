#!/usr/bin/env python3
"""Blend attached and collision-safe VL states across a smooth tet-graph zone."""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from core.bvhparser import MyBVH
from tools import bake_emu
from tools.bake_surface_fast import load_tet, surface_faces
from tools.bake_stiff_tet_arap import (
    BoneSDF, prepare_full_cap_group, project_collision_constraints)
from tools.bake_stiff_tet_pbd import (
    build_surface_samples, project_volumes, signed_volumes, unique_edges)
import test_emu


def graph_distance(vertex_count, edges, seeds, maximum):
    adjacency = [[] for _ in range(vertex_count)]
    for a, b in edges:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    distance = np.full(vertex_count, maximum + 1, dtype=np.int32)
    queue = deque()
    for seed in seeds:
        distance[int(seed)] = 0
        queue.append(int(seed))
    while queue:
        vertex = queue.popleft()
        if distance[vertex] >= maximum:
            continue
        for neighbor in adjacency[vertex]:
            if distance[neighbor] > distance[vertex] + 1:
                distance[neighbor] = distance[vertex] + 1
                queue.append(neighbor)
    return distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attached-cache", type=Path, required=True)
    parser.add_argument("--safe-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bvh", type=Path, required=True)
    parser.add_argument("--tet", type=Path, required=True)
    parser.add_argument("--name", default="L_Vastus_Lateralis")
    parser.add_argument("--frame", type=int, default=5)
    parser.add_argument("--rings", type=int, default=8)
    parser.add_argument(
        "--contact-exclusion-rings", type=int, default=0)
    parser.add_argument(
        "--sdf", type=Path,
        default=Path(".bake_outputs/collision_sdf/L_Femur0_sdf.npz"))
    args = parser.parse_args()

    attached = np.load(args.attached_cache)
    safe = np.load(args.safe_cache)
    positions = safe["positions"].astype(np.float64).copy()
    data = load_tet(args.tet)
    skeleton, bvh_info, _ = bake_emu.load_skeleton()
    group = prepare_full_cap_group(
        data, args.name, args.tet, skeleton,
        test_emu._load_bone_trees())
    rest = np.asarray(group["vertices"], dtype=np.float64)
    tets = np.asarray(group["tetrahedra"], dtype=np.int32)
    fixed = np.asarray(group["fixed_vertices"], dtype=np.int32)
    insertion = np.asarray(group["insertion_fixed"], dtype=np.int32)
    distance = graph_distance(
        len(rest), unique_edges(tets), insertion, args.rings)
    u = np.clip(distance / float(args.rings), 0.0, 1.0)
    attached_weight = 0.5 * (1.0 + np.cos(np.pi * u))
    attached_weight[distance > args.rings] = 0.0
    x = (
        attached_weight[:, None]
        * attached["positions"][args.frame].astype(np.float64)
        + (1.0 - attached_weight[:, None])
        * safe["positions"][args.frame].astype(np.float64))
    x[fixed] = attached["positions"][args.frame, fixed]

    fixed_mask = np.zeros(len(rest), dtype=bool)
    fixed_mask[fixed] = True
    contact_exclusion_mask = fixed_mask.copy()
    if args.contact_exclusion_rings > 0:
        contact_exclusion_mask |= distance <= args.contact_exclusion_rings
    inverse_mass = (~fixed_mask).astype(np.float64)
    samples_i, samples_w = build_surface_samples(surface_faces(tets))
    sdf = BoneSDF(args.sdf)
    motion = MyBVH(
        str(args.bvh), bvh_info, skeleton,
        T_frame=bake_emu._detect_bvh_tframe(str(args.bvh)))
    skeleton.setPositions(motion.mocap_refs[args.frame].copy())
    rest_volumes = signed_volumes(rest, tets)
    projections = 0
    for _ in range(4):
        for _ in range(8):
            raw = sdf.constraints(
                x, samples_i, samples_w, contact_exclusion_mask, skeleton,
                margin=0.0015, max_step=0.0005)
            _, count, _ = project_collision_constraints(
                x, raw, fixed_mask, 0.0005)
            projections += count
            if count == 0:
                break
        for _ in range(5):
            before = x.copy()
            project_volumes(
                x, tets, rest_volumes, inverse_mass, stiffness=0.12)
            delta = x - before
            length = np.linalg.norm(delta, axis=1)
            limited = length > 0.00015
            delta[limited] *= (0.00015 / length[limited])[:, None]
            x[:] = before + delta
            x[fixed] = attached["positions"][args.frame, fixed]
    for _ in range(20):
        raw = sdf.constraints(
            x, samples_i, samples_w, contact_exclusion_mask, skeleton,
            margin=0.0015, max_step=0.0005)
        _, count, _ = project_collision_constraints(
            x, raw, fixed_mask, 0.0005)
        projections += count
        if count == 0:
            break

    oriented = signed_volumes(x, tets) * np.sign(rest_volumes)
    positions[args.frame] = x
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / f"{args.name}_chunk_0000.npz",
        frames=safe["frames"], positions=positions.astype(np.float32))
    (args.output / ".done").touch()
    print(
        f"frame={args.frame} inverted={int(np.sum(oriented <= 0))}/"
        f"{len(tets)} volume_ratio="
        f"{np.abs(oriented).sum()/np.abs(rest_volumes).sum():.6f} "
        f"projections={projections}")


if __name__ == "__main__":
    main()
