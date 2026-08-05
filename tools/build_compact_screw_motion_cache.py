#!/usr/bin/env python3
"""Bake compact muscles by screw-blending a good reference-frame shape."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from core.bvhparser import MyBVH
from tools import bake_emu
from tools.bake_surface_fast import load_tet
from tools.bake_stiff_tet_arap import prepare_full_cap_group
from tools.transfer_vm_reference_pose import rigid_fit
import test_emu

NAMES = ("L_Vastus_Intermedius", "L_Vastus_Lateralis",
         "L_Vastus_Medialis")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bvh", type=Path, required=True)
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--insertion-blend-start", type=float, default=0.78)
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
    for name, group in groups.items():
        cache = np.load(args.source / f"{name}_chunk_0000.npz")
        reference = cache["positions"][args.reference_frame].astype(np.float64)
        origin = np.asarray(group["origin_fixed"], dtype=np.int32)
        insertion = np.asarray(group["insertion_fixed"], dtype=np.int32)
        fixed = np.asarray(group["fixed_vertices"], dtype=np.int32)
        u = np.asarray(group["axis_coordinate"], dtype=np.float64)
        blend_u = np.clip(
            (u - args.insertion_blend_start)
            / max(1.0 - args.insertion_blend_start, 1e-6), 0.0, 1.0)
        blend_u = blend_u * blend_u * (3.0 - 2.0 * blend_u)
        positions = []
        for pose in motion.mocap_refs:
            skeleton.setPositions(pose.copy())
            guide = bake_emu.compute_rigid_blend_positions(
                group["lbs_bindings"], skeleton, u)
            ro, to = rigid_fit(reference[origin], guide[origin])
            ri, ti = rigid_fit(reference[insertion], guide[insertion])
            rotations = Slerp(
                [0.0, 1.0], Rotation.from_matrix([ro, ri]))(
                    blend_u).as_matrix()
            translations = (
                (1.0 - blend_u[:, None]) * to
                + blend_u[:, None] * ti)
            posed = np.einsum("nij,nj->ni", rotations, reference) + translations
            posed[fixed] = guide[fixed]
            positions.append(posed)
        np.savez_compressed(
            args.output / f"{name}_chunk_0000.npz",
            frames=np.arange(len(positions), dtype=np.int32),
            positions=np.asarray(positions, dtype=np.float32))
    (args.output / ".done").touch()


if __name__ == "__main__":
    main()
