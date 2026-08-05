#!/usr/bin/env python3
"""Initialize a frame range by transporting the prior frame with the femur."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

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
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    skeleton, info, _ = bake_emu.load_skeleton()
    trees = test_emu._load_bone_trees()
    groups = {}
    for name in NAMES:
        tet_path = Path("tet") / f"{name}_tet.npz"
        groups[name] = prepare_full_cap_group(
            load_tet(tet_path), name, tet_path, skeleton, trees)
    motion = MyBVH(
        str(args.bvh), info, skeleton,
        T_frame=bake_emu._detect_bvh_tframe(str(args.bvh)))
    transforms = []
    guides = {name: [] for name in NAMES}
    for pose in motion.mocap_refs:
        skeleton.setPositions(pose.copy())
        tf = skeleton.getBodyNode("L_Femur0").getWorldTransform()
        transforms.append((tf.rotation().copy(), tf.translation().copy()))
        for name, group in groups.items():
            guides[name].append(bake_emu.compute_rigid_blend_positions(
                group["lbs_bindings"], skeleton,
                group["axis_coordinate"]))
    for name, group in groups.items():
        cache = np.load(args.source / f"{name}_chunk_0000.npz")
        positions = cache["positions"].astype(np.float64).copy()
        fixed = np.asarray(group["fixed_vertices"], dtype=np.int32)
        for frame in range(args.first, args.last + 1):
            previous_r, previous_t = transforms[frame - 1]
            current_r, current_t = transforms[frame]
            local = (
                positions[frame - 1] - previous_t) @ previous_r
            positions[frame] = local @ current_r.T + current_t
            positions[frame, fixed] = guides[name][frame][fixed]
        np.savez_compressed(
            args.output / f"{name}_chunk_0000.npz",
            frames=cache["frames"], positions=positions.astype(np.float32))
    (args.output / ".done").touch()


if __name__ == "__main__":
    main()
