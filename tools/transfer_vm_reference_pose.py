#!/usr/bin/env python3
"""Transfer a good VM pose to other frames using two-end rigid blending."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np

from tools.bake_surface_fast import load_tet
from tools.bake_stiff_tet_arap import prepare_full_cap_group
from tools.bake_stiff_tet_pbd import unique_edges
import test_emu
from tools import bake_emu


NAME = "L_Vastus_Medialis"
NAMES = ("L_Vastus_Intermedius", "L_Vastus_Lateralis", NAME)


def rigid_fit(source: np.ndarray, target: np.ndarray):
    cs, ct = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - cs).T @ (target - ct))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return rotation, ct - cs @ rotation.T


def graph_distances(count, edges, seeds):
    adjacency = [[] for _ in range(count)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    distance = np.full(count, np.inf)
    distance[seeds] = 0
    frontier = list(map(int, seeds))
    for vertex in frontier:
        for neighbor in adjacency[vertex]:
            if not np.isfinite(distance[neighbor]):
                distance[neighbor] = distance[vertex] + 1
                frontier.append(neighbor)
    return distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-frame", type=int, default=14)
    parser.add_argument("--frames", default="12,13,15")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        shutil.copy2(args.source / f"{name}_chunk_0000.npz",
                     args.output / f"{name}_chunk_0000.npz")

    tet_path = Path("tet/L_Vastus_Medialis_tet.npz")
    data = load_tet(tet_path)
    skeleton, _, _ = bake_emu.load_skeleton()
    group = prepare_full_cap_group(
        data, NAME, tet_path, skeleton, test_emu._load_bone_trees())
    origin = np.asarray(group["origin_fixed"], dtype=np.int32)
    insertion = np.asarray(group["insertion_fixed"], dtype=np.int32)
    rest = np.asarray(group["vertices"], dtype=np.float64)
    edges = unique_edges(np.asarray(group["tetrahedra"], dtype=np.int32))
    do = graph_distances(len(rest), edges, origin)
    di = graph_distances(len(rest), edges, insertion)
    insertion_weight = do / np.maximum(do + di, 1.0)

    path = args.source / f"{NAME}_chunk_0000.npz"
    cache = np.load(path)
    positions = cache["positions"].astype(np.float64).copy()
    reference = positions[args.reference_frame].copy()
    for frame in map(int, args.frames.split(",")):
        target = positions[frame].copy()
        ro, to = rigid_fit(reference[origin], target[origin])
        ri, ti = rigid_fit(reference[insertion], target[insertion])
        from_origin = reference @ ro.T + to
        from_insertion = reference @ ri.T + ti
        w = insertion_weight[:, None]
        positions[frame] = (1.0 - w) * from_origin + w * from_insertion
        positions[frame, origin] = target[origin]
        positions[frame, insertion] = target[insertion]
    np.savez_compressed(
        args.output / f"{NAME}_chunk_0000.npz",
        frames=cache["frames"], positions=positions.astype(np.float32))
    (args.output / ".done").touch()


if __name__ == "__main__":
    main()
