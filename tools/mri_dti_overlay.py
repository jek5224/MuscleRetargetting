#!/usr/bin/env python3
"""Create quick T1/DTI-b0 overlay PNGs from ConvertedNPZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


def robust01(image: np.ndarray) -> np.ndarray:
    vals = image[np.isfinite(image)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return np.zeros_like(image, dtype=float)
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        return np.zeros_like(image, dtype=float)
    return np.clip((image - lo) / (hi - lo), 0, 1)


def resample_slice(moving: np.ndarray, moving_affine: np.ndarray, fixed_shape: tuple[int, int], fixed_affine: np.ndarray, z: int) -> np.ndarray:
    yy, xx = np.mgrid[0:fixed_shape[0], 0:fixed_shape[1]]
    zz = np.full_like(yy, z)
    ones = np.ones_like(yy)
    fixed_zyx = np.stack([zz, yy, xx, ones], axis=0).reshape(4, -1)
    world = fixed_affine @ fixed_zyx
    moving_zyx = np.linalg.inv(moving_affine) @ world
    coords = moving_zyx[:3].reshape(3, *fixed_shape)
    return map_coordinates(moving, coords, order=1, mode="constant", cval=0.0)


def make_overlay(t1_slice: np.ndarray, b0_slice: np.ndarray) -> Image.Image:
    t1 = robust01(t1_slice)
    b0 = robust01(b0_slice)
    rgb = np.zeros((*t1.shape, 3), dtype=np.uint8)
    gray = (t1 * 190).astype(np.uint8)
    rgb[..., 0] = np.maximum(gray, (b0 * 255).astype(np.uint8))
    rgb[..., 1] = gray
    rgb[..., 2] = np.maximum(gray, (b0 * 255).astype(np.uint8))
    return Image.fromarray(rgb)


def series_pairs(manifest: list[dict]) -> list[tuple[dict, dict]]:
    t1 = sorted([x for x in manifest if x["kind"] == "t1"], key=lambda x: x["series_number"])
    dti = sorted([x for x in manifest if x["kind"] == "dti"], key=lambda x: x["series_number"])
    return list(zip(t1, dti))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted-dir", default="DTI/Subject 01/ConvertedNPZ")
    parser.add_argument("--out-dir", default="DTI/Subject 01/AlignmentOverlays")
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    converted_dir = Path(args.converted_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((converted_dir / "manifest.json").read_text())

    summary = []
    for t1_info, dti_info in series_pairs(manifest):
        t1_npz = np.load(t1_info["path"])
        dti_npz = np.load(dti_info["path"])
        t1 = t1_npz["volume"].astype(float)
        dti = dti_npz["volume"].astype(float)
        bvals = dti_npz["bvals"]
        b0 = dti[bvals == 0].mean(axis=0)
        t1_affine = t1_npz["affine_zyx"]
        dti_affine = dti_npz["affine_zyx"]

        slices = sorted(set([t1.shape[0] // 4, t1.shape[0] // 2, 3 * t1.shape[0] // 4]))
        written = []
        for z in slices:
            moving = resample_slice(b0, dti_affine, t1.shape[1:], t1_affine, z)
            image = make_overlay(t1[z], moving)
            if args.stride > 1:
                image = image.resize((image.width // args.stride, image.height // args.stride), Image.Resampling.BILINEAR)
            path = out_dir / f"t1_{t1_info['series_number']:02d}_dti_{dti_info['series_number']:02d}_z{z:02d}.png"
            image.save(path)
            written.append(str(path))
        summary.append({
            "t1_series": t1_info["series_number"],
            "dti_series": dti_info["series_number"],
            "slices": slices,
            "files": written,
        })
        print(f"T1 {t1_info['series_number']} + DTI {dti_info['series_number']}: wrote {len(written)} overlays")

    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
