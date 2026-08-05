#!/usr/bin/env python3
"""Create exact bone-following initial caches for the joint solver."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    skeleton, info, _ = bake_emu.load_skeleton()
    motion = MyBVH(
        str(args.bvh), info, skeleton,
        T_frame=bake_emu._detect_bvh_tframe(str(args.bvh)))
    trees = test_emu._load_bone_trees()
    groups = {}
    for name in NAMES:
        tet_path = Path("tet") / f"{name}_tet.npz"
        groups[name] = prepare_full_cap_group(
            load_tet(tet_path), name, tet_path, skeleton, trees)
    for name in NAMES:
        group = groups[name]
        positions = []
        for pose in motion.mocap_refs:
            skeleton.setPositions(pose.copy())
            positions.append(bake_emu.compute_rigid_blend_positions(
                group["lbs_bindings"], skeleton, group["axis_coordinate"]))
        old = np.load(args.source / f"{name}_chunk_0000.npz")
        np.savez_compressed(
            args.output / f"{name}_chunk_0000.npz",
            frames=np.arange(len(positions), dtype=np.int32),
            positions=np.asarray(positions, dtype=np.float32))
    (args.output / ".done").touch()


if __name__ == "__main__":
    main()
