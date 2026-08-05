#!/usr/bin/env python3
"""Hard-clamp free muscle vertices to their rest femur-distance envelope."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial import cKDTree
import trimesh

from core.bvhparser import MyBVH
from tools import bake_emu
from tools.bake_surface_fast import load_tet
from tools.bake_stiff_tet_arap import prepare_full_cap_group
import test_emu

NAMES = ("L_Vastus_Intermedius", "L_Vastus_Lateralis",
         "L_Vastus_Medialis")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bvh", type=Path, required=True)
    parser.add_argument("--frames", default="26,27,28,29")
    parser.add_argument("--slack", type=float, default=0.006)
    parser.add_argument("--equi", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.002)
    args = parser.parse_args()
    frames = [int(value) for value in args.frames.split(",")]
    args.output.mkdir(parents=True, exist_ok=True)
    skeleton, info, _ = bake_emu.load_skeleton()
    trees = test_emu._load_bone_trees()
    groups = {}
    for name in NAMES:
        tet_path = Path("tet") / f"{name}_tet.npz"
        groups[name] = prepare_full_cap_group(
            load_tet(tet_path), name, tet_path, skeleton, trees)
    femur = trimesh.load(
        Path(bake_emu.SKEL_MESH_DIR) / "L_Femur.obj", process=False)
    femur_vertices = (
        np.asarray(femur.vertices, dtype=np.float64) * bake_emu.MESH_SCALE)
    femur_tree = cKDTree(femur_vertices)
    rest_tf = skeleton.getBodyNode("L_Femur0").getWorldTransform()
    motion = MyBVH(
        str(args.bvh), info, skeleton,
        T_frame=bake_emu._detect_bvh_tframe(str(args.bvh)))
    for name in NAMES:
        group = groups[name]
        rest = np.asarray(group["vertices"], dtype=np.float64)
        rest_distance, _ = femur_tree.query(rest)
        maximum = rest_distance + args.slack
        fixed = np.asarray(group["fixed_vertices"], dtype=np.int32)
        fixed_mask = np.zeros(len(rest), dtype=bool)
        fixed_mask[fixed] = True
        cache = np.load(args.source / f"{name}_chunk_0000.npz")
        positions = cache["positions"].astype(np.float64).copy()
        for frame in frames:
            skeleton.setPositions(motion.mocap_refs[frame].copy())
            tf = skeleton.getBodyNode("L_Femur0").getWorldTransform()
            body_local = (
                tf.rotation().T @
                (positions[frame] - tf.translation()).T).T
            rest_space = (
                body_local @ rest_tf.rotation().T
                + rest_tf.translation())
            distance, nearest_id = femur_tree.query(rest_space)
            if args.equi:
                selected = (
                    np.abs(distance - rest_distance) > args.tolerance
                ) & ~fixed_mask
                target_distance = rest_distance
            else:
                selected = (distance > maximum) & ~fixed_mask
                target_distance = maximum
            direction = (
                rest_space[selected] - femur_vertices[nearest_id[selected]])
            length = np.linalg.norm(direction, axis=1)
            valid = length > 1e-12
            direction[valid] /= length[valid, None]
            rest_space[selected] = (
                femur_vertices[nearest_id[selected]]
                + direction * target_distance[selected, None])
            body_local = (
                (rest_space - rest_tf.translation())
                @ rest_tf.rotation())
            positions[frame] = (
                body_local @ tf.rotation().T + tf.translation())
            print(name, frame, "clamped", int(np.sum(selected)),
                  "max_m", float(np.max(np.minimum(distance, maximum))))
        np.savez_compressed(
            args.output / f"{name}_chunk_0000.npz",
            frames=cache["frames"], positions=positions.astype(np.float32))
    (args.output / ".done").touch()


if __name__ == "__main__":
    main()
