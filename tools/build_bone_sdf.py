#!/usr/bin/env python3
"""Build a padded, bone-local voxel SDF from a skeleton OBJ."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import bake_emu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bone", default="L_Femur")
    ap.add_argument("--body", default="L_Femur0")
    ap.add_argument("--pitch", type=float, default=0.002)
    ap.add_argument("--padding", type=int, default=6)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    skel, _, _ = bake_emu.load_skeleton()
    skel.setPositions(np.zeros(skel.getNumDofs()))
    body = skel.getBodyNode(args.body)
    if body is None:
        raise ValueError(f"body not found: {args.body}")
    wt = body.getWorldTransform()
    R0, t0 = wt.rotation().copy(), wt.translation().copy()

    path = ROOT / bake_emu.SKEL_MESH_DIR / f"{args.bone}.obj"
    raw = trimesh.load(path, force="mesh", process=False)
    world = np.asarray(raw.vertices, dtype=np.float64) * bake_emu.MESH_SCALE
    local = (R0.T @ (world - t0).T).T
    mesh = trimesh.Trimesh(local, raw.faces, process=True)

    # Voxelization followed by morphological filling turns the imperfect source
    # surface into a closed solid before computing the signed distance.
    vox = mesh.voxelized(args.pitch).fill()
    occupied = np.asarray(vox.matrix, dtype=bool)
    pad = args.padding
    solid = np.pad(occupied, pad, mode="constant", constant_values=False)
    origin = np.asarray(vox.transform[:3, 3], dtype=np.float64) - pad * args.pitch

    outside = distance_transform_edt(~solid) * args.pitch
    inside = distance_transform_edt(solid) * args.pitch
    sdf = outside.astype(np.float32)
    sdf[solid] = -inside[solid].astype(np.float32)
    gradients = np.stack(np.gradient(sdf, args.pitch), axis=-1).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, sdf=sdf, gradients=gradients, origin=origin,
        pitch=np.float64(args.pitch), body=np.asarray(args.body),
        source=np.asarray(str(path)), rest_rotation=R0, rest_translation=t0)
    print(
        f"saved {args.output}: shape={sdf.shape}, pitch={args.pitch:g} m, "
        f"solid_voxels={solid.sum()}, range=[{sdf.min():.4g}, {sdf.max():.4g}]")


if __name__ == "__main__":
    main()
