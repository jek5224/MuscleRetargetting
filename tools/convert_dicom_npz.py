#!/usr/bin/env python3
"""Convert the local Siemens T1/DTI DICOMs into simple NumPy volumes.

Outputs are not NIfTI. They are explicit intermediate arrays for inspection:
- T1: volume[z, y, x]
- DTI mosaic: volume[t, z, y, x], bvals[t], bvecs[t, 3]
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

from dicom_inventory import dot, normal_from_orientation, parse_dicom


def read_pixel_array(path: Path, rows: int, columns: int) -> np.ndarray:
    data = path.read_bytes()
    tag = struct.pack("<HH", 0x7FE0, 0x0010)
    n = rows * columns
    candidates = []
    start = 0
    while True:
        pos = data.find(tag, start)
        if pos < 0:
            break
        start = pos + 1
        payload_pos = pos + 4
        vr = data[payload_pos:payload_pos + 2]
        if vr in {b"OB", b"OD", b"OF", b"OL", b"OW", b"SQ", b"UC", b"UR", b"UT", b"UN"}:
            payload_pos += 4
            length = struct.unpack_from("<I", data, payload_pos)[0]
            payload_pos += 4
        else:
            length = struct.unpack_from("<I", data, payload_pos)[0]
            payload_pos += 4
        if length >= n * 2 and payload_pos + n * 2 <= len(data):
            candidates.append(payload_pos)
    if not candidates:
        raise ValueError(f"matching pixel data tag missing: {path}")
    arr = np.frombuffer(data, dtype="<u2", count=n, offset=candidates[-1])
    return arr.reshape(rows, columns).copy()


def slice_coord(meta: dict) -> float:
    normal = normal_from_orientation(meta.get("image_orientation_patient"))
    pos = meta.get("image_position_patient")
    if normal is None or not isinstance(pos, list) or len(pos) < 3:
        return float(meta.get("instance_number", 0))
    return dot(pos[:3], normal)


def is_diffusion(meta: dict) -> bool:
    text = " ".join(str(meta.get(k, "")) for k in ("image_type", "protocol_name", "series_description")).upper()
    return any(key in text for key in ("DIFFUSION", "DIFF", "EP2D"))


def affine_zyx(meta: dict, slice_step: np.ndarray, origin_override: np.ndarray | None = None) -> np.ndarray:
    orient = np.array(meta.get("image_orientation_patient", []), dtype=float)
    if orient.shape[0] < 6:
        raise ValueError(f"missing image orientation for {meta.get('path')}")
    row_dir = orient[:3]
    col_dir = orient[3:6]
    pixel_spacing = np.array(meta["pixel_spacing"], dtype=float)
    origin = np.array(meta["image_position_patient"], dtype=float) if origin_override is None else origin_override

    affine = np.eye(4)
    affine[:3, 0] = slice_step
    affine[:3, 1] = col_dir * pixel_spacing[0]
    affine[:3, 2] = row_dir * pixel_spacing[1]
    affine[:3, 3] = origin
    return affine


def convert_t1(items: list[dict], out_path: Path) -> dict:
    sorted_items = sorted(items, key=slice_coord)
    rows = int(sorted_items[0]["rows"])
    cols = int(sorted_items[0]["columns"])
    volume = np.stack([read_pixel_array(Path(x["path"]), rows, cols) for x in sorted_items], axis=0)
    positions = np.array([x.get("image_position_patient", [0, 0, 0]) for x in sorted_items], dtype=float)
    if len(positions) > 1:
        slice_step = np.median(np.diff(positions, axis=0), axis=0)
    else:
        normal = np.array(normal_from_orientation(sorted_items[0].get("image_orientation_patient")), dtype=float)
        slice_step = normal * float(sorted_items[0]["slice_thickness"])
    affine = affine_zyx(sorted_items[0], slice_step)
    spacing = np.array([
        float(np.linalg.norm(slice_step)),
        float(sorted_items[0]["pixel_spacing"][0]),
        float(sorted_items[0]["pixel_spacing"][1]),
    ])
    np.savez_compressed(
        out_path,
        volume=volume,
        spacing_zyx=spacing,
        affine_zyx=affine,
        orientation=np.array(sorted_items[0].get("image_orientation_patient", []), dtype=float),
        positions=positions,
        series_number=sorted_items[0].get("series_number"),
        protocol=str(sorted_items[0].get("protocol_name", "")),
    )
    return {"path": str(out_path), "shape": list(volume.shape), "spacing_zyx": spacing.tolist(), "affine_zyx": affine.tolist()}


def unmosaic(image: np.ndarray, n_slices: int) -> np.ndarray:
    grid = int(math.ceil(math.sqrt(n_slices)))
    tile_rows = image.shape[0] // grid
    tile_cols = image.shape[1] // grid
    slices = []
    for i in range(n_slices):
        r = i // grid
        c = i % grid
        slices.append(image[r * tile_rows:(r + 1) * tile_rows, c * tile_cols:(c + 1) * tile_cols])
    return np.stack(slices, axis=0)


def convert_dti(items: list[dict], out_path: Path) -> dict:
    sorted_items = sorted(items, key=lambda x: float(x.get("instance_number", 0)))
    rows = int(sorted_items[0]["rows"])
    cols = int(sorted_items[0]["columns"])
    n_slices = int(sorted_items[0].get("number_of_images_in_mosaic", 0))
    if n_slices <= 0:
        raise ValueError(f"missing mosaic slice count for series {sorted_items[0].get('series_number')}")

    volumes = []
    bvals = []
    bvecs = []
    for meta in sorted_items:
        mosaic = read_pixel_array(Path(meta["path"]), rows, cols)
        volumes.append(unmosaic(mosaic, n_slices))
        bval = float(meta.get("b_value", 0))
        bvals.append(bval)
        bvecs.append(meta.get("diffusion_gradient_direction", [0.0, 0.0, 0.0]) if bval > 0 else [0.0, 0.0, 0.0])

    volume = np.stack(volumes, axis=0)
    bvals_arr = np.array(bvals, dtype=float)
    bvecs_arr = np.array(bvecs, dtype=float)
    grid = int(math.ceil(math.sqrt(n_slices)))
    spacing = np.array([
        float(sorted_items[0]["slice_thickness"]),
        float(sorted_items[0]["pixel_spacing"][0]),
        float(sorted_items[0]["pixel_spacing"][1]),
    ])
    normal = np.array(normal_from_orientation(sorted_items[0].get("image_orientation_patient")), dtype=float)
    slice_step = -normal * spacing[0]
    orient = np.array(sorted_items[0].get("image_orientation_patient", []), dtype=float)
    row_dir = orient[:3]
    col_dir = orient[3:6]
    mosaic_origin = np.array(sorted_items[0]["image_position_patient"], dtype=float)
    tile_rows = rows // grid
    tile_cols = cols // grid
    tile_origin = (
        mosaic_origin
        + row_dir * ((cols - tile_cols) * 0.5 * spacing[2])
        + col_dir * ((rows - tile_rows) * 0.5 * spacing[1])
    )
    affine = affine_zyx(sorted_items[0], slice_step, origin_override=tile_origin)
    np.savez_compressed(
        out_path,
        volume=volume,
        bvals=bvals_arr,
        bvecs=bvecs_arr,
        spacing_zyx=spacing,
        affine_zyx=affine,
        orientation=np.array(sorted_items[0].get("image_orientation_patient", []), dtype=float),
        tile_shape_yx=np.array([tile_rows, tile_cols], dtype=int),
        series_number=sorted_items[0].get("series_number"),
        protocol=str(sorted_items[0].get("protocol_name", "")),
    )
    return {
        "path": str(out_path),
        "shape": list(volume.shape),
        "spacing_zyx": spacing.tolist(),
        "affine_zyx": affine.tolist(),
        "bvals": {str(int(v)): int((bvals_arr == v).sum()) for v in sorted(set(bvals_arr))},
        "unique_bvecs": int(len({tuple(np.round(v, 6)) for v in bvecs_arr if np.linalg.norm(v) > 0})),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out-dir", default="DTI/Subject 01/ConvertedNPZ")
    parser.add_argument("--series", type=float, nargs="*", help="optional series numbers to convert")
    args = parser.parse_args()

    files = []
    for arg in args.paths:
        p = Path(arg)
        files.extend(sorted(p.rglob("*.dcm")) if p.is_dir() else [p])

    groups = defaultdict(list)
    for path in files:
        meta = parse_dicom(path)
        if args.series and float(meta.get("series_number", -1)) not in args.series:
            continue
        groups[meta.get("series_instance_uid", f"UNKNOWN:{path.parent}")].append(meta)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for _, items in sorted(groups.items(), key=lambda kv: float(kv[1][0].get("series_number", 0))):
        first = items[0]
        series_no = int(float(first.get("series_number", 0)))
        kind = "dti" if is_diffusion(first) else "t1"
        out_path = out_dir / f"series_{series_no:02d}_{kind}.npz"
        info = convert_dti(items, out_path) if kind == "dti" else convert_t1(items, out_path)
        info.update({"series_number": series_no, "kind": kind, "n_files": len(items)})
        manifest.append(info)
        print(f"{kind.upper()} series {series_no}: shape={info['shape']} -> {out_path}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
