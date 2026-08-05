#!/usr/bin/env python3
"""Resample accepted VI key poses in femur-local coordinates."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from core.bvhparser import MyBVH
from tools import bake_emu
from tools.bake_surface_fast import load_tet
import test_emu


def smoothstep(x):
    return x * x * (3.0 - 2.0 * x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bvh", type=Path, required=True)
    parser.add_argument("--target-bvh", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--between", type=int, default=4)
    parser.add_argument(
        "--no-attachment-overwrite", action="store_true",
        help="Preserve interpolated key-pose attachments exactly.")
    parser.add_argument(
        "--legacy-vi-v4-attachments", action="store_true",
        help="Reproduce the historical v4 eight-vertex guide overwrite.")
    args = parser.parse_args()

    name = "L_Vastus_Intermedius"
    skeleton, info, _ = bake_emu.load_skeleton()
    group = test_emu.prepare_group_data(
        load_tet(f"tet/{name}_tet.npz"), name, skeleton,
        test_emu._load_bone_trees(),
        source_path=Path(f"tet/{name}_tet.npz"), attachment_rings=0)
    source_motion = MyBVH(
        str(args.source_bvh), info, skeleton,
        T_frame=bake_emu._detect_bvh_tframe(str(args.source_bvh)))
    target_motion = MyBVH(
        str(args.target_bvh), info, skeleton,
        T_frame=bake_emu._detect_bvh_tframe(str(args.target_bvh)))
    source = np.load(
        args.source_cache / f"{name}_chunk_0000.npz")["positions"].astype(
            np.float64)

    local_key = []
    for pose, x in zip(source_motion.mocap_refs, source):
        skeleton.setPositions(pose.copy())
        tf = skeleton.getBodyNode("L_Femur0").getWorldTransform()
        local_key.append(
            (tf.rotation().T @ (x - tf.translation()).T).T)
    local_key = np.asarray(local_key)

    output = []
    stride = args.between + 1
    fixed = np.asarray(group["fixed_vertices"], dtype=np.int32)
    if args.legacy_vi_v4_attachments:
        fixed = np.asarray(
            [224, 249, 250, 251, 252, 253, 254, 255],
            dtype=np.int32)
    for frame, pose in enumerate(target_motion.mocap_refs):
        key = min(frame // stride, len(source) - 1)
        if key == len(source) - 1:
            local = local_key[key]
        else:
            alpha = smoothstep((frame % stride) / stride)
            local = (
                (1.0 - alpha) * local_key[key]
                + alpha * local_key[key + 1])
        skeleton.setPositions(pose.copy())
        tf = skeleton.getBodyNode("L_Femur0").getWorldTransform()
        x = (tf.rotation() @ local.T).T + tf.translation()
        guide = bake_emu.compute_rigid_blend_positions(
            group["lbs_bindings"], skeleton, group["axis_coordinate"])
        if not args.no_attachment_overwrite:
            x[fixed] = guide[fixed]
        output.append(x.astype(np.float32))

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / f"{name}_chunk_0000.npz",
        frames=np.arange(len(output), dtype=np.int32),
        positions=np.asarray(output))
    (args.output / ".done").touch()


if __name__ == "__main__":
    main()
