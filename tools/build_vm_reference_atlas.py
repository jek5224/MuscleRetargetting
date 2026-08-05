#!/usr/bin/env python3
"""Build every VM frame from the best collision-free reference transfer."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
from scipy.ndimage import map_coordinates

from core.bvhparser import MyBVH
from tools import bake_emu
from tools.bake_surface_fast import load_tet, surface_faces
from tools.bake_stiff_tet_arap import BoneSDF, prepare_full_cap_group
from tools.bake_stiff_tet_pbd import build_surface_samples, signed_volumes
from tools.transfer_vm_reference_pose import graph_distances, rigid_fit
from tools.bake_stiff_tet_pbd import unique_edges
import test_emu


NAME = "L_Vastus_Medialis"
NAMES = ("L_Vastus_Intermedius", "L_Vastus_Lateralis", NAME)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bvh", type=Path, required=True)
    parser.add_argument(
        "--sdf", type=Path,
        default=Path(".bake_outputs/collision_sdf/L_Femur0_sdf.npz"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        shutil.copy2(args.source / f"{name}_chunk_0000.npz",
                     args.output / f"{name}_chunk_0000.npz")

    tet_path = Path("tet/L_Vastus_Medialis_tet.npz")
    data = load_tet(tet_path)
    skeleton, info, _ = bake_emu.load_skeleton()
    group = prepare_full_cap_group(
        data, NAME, tet_path, skeleton, test_emu._load_bone_trees())
    rest = np.asarray(group["vertices"], dtype=np.float64)
    tets = np.asarray(group["tetrahedra"], dtype=np.int32)
    origin = np.asarray(group["origin_fixed"], dtype=np.int32)
    insertion = np.asarray(group["insertion_fixed"], dtype=np.int32)
    fixed = np.asarray(group["fixed_vertices"], dtype=np.int32)
    fixed_mask = np.zeros(len(rest), dtype=bool)
    fixed_mask[fixed] = True
    do = graph_distances(len(rest), unique_edges(tets), origin)
    di = graph_distances(len(rest), unique_edges(tets), insertion)
    insertion_weight = (do / np.maximum(do + di, 1.0))[:, None]
    samples_i, samples_w = build_surface_samples(surface_faces(tets))
    active = ~np.any(
        fixed_mask[samples_i] & (samples_w > 0), axis=1)
    samples_i, samples_w = samples_i[active], samples_w[active]
    rest_volume = signed_volumes(rest, tets)
    rest_sign = np.sign(rest_volume)

    cache = np.load(args.source / f"{NAME}_chunk_0000.npz")
    source = cache["positions"].astype(np.float64)
    result = source.copy()
    motion = MyBVH(
        str(args.bvh), info, skeleton,
        T_frame=bake_emu._detect_bvh_tframe(str(args.bvh)))
    field = BoneSDF(args.sdf)
    for frame in range(len(source)):
        target = source[frame]
        skeleton.setPositions(motion.mocap_refs[frame].copy())
        transform = skeleton.getBodyNode(field.body).getWorldTransform()
        rotation = transform.rotation()
        translation = transform.translation()
        best = None
        for reference_frame in range(len(source)):
            reference = source[reference_frame]
            ro, to = rigid_fit(reference[origin], target[origin])
            ri, ti = rigid_fit(reference[insertion], target[insertion])
            candidate = (
                (1.0 - insertion_weight) * (reference @ ro.T + to)
                + insertion_weight * (reference @ ri.T + ti))
            candidate[origin] = target[origin]
            candidate[insertion] = target[insertion]
            points = np.einsum(
                "nij,ni->nj", candidate[samples_i], samples_w)
            local = (
                rotation.T @ (points - translation).T).T
            distance = map_coordinates(
                field.sdf, ((local - field.origin) / field.pitch).T,
                order=1, mode="constant", cval=1.0)
            oriented = signed_volumes(candidate, tets) * rest_sign
            ratio = oriented / np.maximum(np.abs(rest_volume), 1e-12)
            score = (
                int(np.sum(distance < 0.0)),
                int(np.sum(oriented <= 0.0)),
                float(np.mean(np.clip(np.abs(ratio - 1.0), 0, 20))),
                reference_frame)
            if best is None or score < best[0]:
                best = (score, candidate)
        result[frame] = best[1]
        print(f"frame={frame} reference={best[0][3]} "
              f"inside={best[0][0]} inverted={best[0][1]} "
              f"distortion={best[0][2]:.4f}")
    np.savez_compressed(
        args.output / f"{NAME}_chunk_0000.npz",
        frames=cache["frames"], positions=result.astype(np.float32))
    (args.output / ".done").touch()


if __name__ == "__main__":
    main()
