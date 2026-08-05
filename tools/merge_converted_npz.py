#!/usr/bin/env python3
"""Merge slab-wise ConvertedNPZ T1/DTI series into whole-subject volumes.

The source files already carry z/y/x -> patient-space affines.  This merger
orders acquired slices by patient-space slice position.  It does not insert
blank slices for unscanned inter-slab gaps; those gaps are kept as metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_manifest(converted_dir: Path) -> list[dict]:
    return json.loads((converted_dir / "manifest.json").read_text())


def slice_coords(affine: np.ndarray, n_slices: int) -> np.ndarray:
    zyx = np.ones((4, n_slices), dtype=np.float64)
    zyx[0] = np.arange(n_slices, dtype=np.float64)
    zyx[1] = 0.0
    zyx[2] = 0.0
    pts = affine @ zyx
    return pts[2]


def acquired_z_positions(entries: list[dict]) -> tuple[np.ndarray, float, np.ndarray]:
    rows = []
    steps = []
    for entry_i, entry in enumerate(entries):
        data = np.load(entry["path"])
        vol = data["volume"]
        n_slices = vol.shape[1] if entry["kind"] == "dti" else vol.shape[0]
        coords = slice_coords(data["affine_zyx"], n_slices)
        for src_i, coord in enumerate(coords):
            rows.append((float(coord), entry_i, src_i))
        if len(coords) > 1:
            steps.extend(np.abs(np.diff(coords)).tolist())
    step = float(np.median(steps))
    rows.sort(key=lambda r: r[0])

    acquired = []
    for row in rows:
        if acquired and abs(row[0] - acquired[-1][0]) < step * 0.25:
            prev = acquired[-1]
            acquired[-1] = ((prev[0] + row[0]) * 0.5, prev[1], prev[2])
        else:
            acquired.append(row)
    coords = np.array([r[0] for r in acquired], dtype=np.float64)
    gaps = np.where(np.diff(coords) > step * 1.5)[0]
    return coords, step, gaps


def target_affine_like(entries: list[dict], z0: float, step: float) -> np.ndarray:
    first = np.load(entries[0]["path"])
    aff = np.asarray(first["affine_zyx"], dtype=np.float64)
    out = aff.copy()
    out[:3, 0] = np.array([0.0, 0.0, step])
    out[2, 3] = z0

    origins = []
    for entry in entries:
        data = np.load(entry["path"])
        coords = slice_coords(data["affine_zyx"], data["volume"].shape[1] if entry["kind"] == "dti" else data["volume"].shape[0])
        idx = int(np.argmin(coords))
        p = data["affine_zyx"] @ np.array([idx, 0, 0, 1.0])
        origins.append(p[:3])
    origin = np.median(np.asarray(origins), axis=0)
    out[0, 3] = origin[0]
    out[1, 3] = origin[1]
    out[2, 3] = z0
    return out.astype(np.float32)


def merge_t1(entries: list[dict], out_path: Path) -> dict:
    z_positions, step, gap_after = acquired_z_positions(entries)
    first = np.load(entries[0]["path"])
    y, x = first["volume"].shape[1:]
    slice_records = []
    for entry in entries:
        data = np.load(entry["path"])
        coords = slice_coords(data["affine_zyx"], data["volume"].shape[0])
        for src_i, coord in enumerate(coords):
            slice_records.append((float(coord), data["volume"][src_i]))

    slice_records.sort(key=lambda r: r[0])
    merged = np.stack([r[1] for r in slice_records], axis=0).astype(np.uint16)
    affine = target_affine_like(entries, float(z_positions[0]), step)
    spacing = np.array([step, first["spacing_zyx"][1], first["spacing_zyx"][2]], dtype=np.float32)
    np.savez_compressed(
        out_path,
        volume=merged,
        spacing_zyx=spacing,
        affine_zyx=affine,
        source_series=np.array([e["series_number"] for e in entries], dtype=np.int32),
        slice_z_positions=z_positions.astype(np.float32),
        gap_after_slices=gap_after.astype(np.int32),
        protocol="merged_t1",
    )
    return {"path": str(out_path), "kind": "t1", "shape": list(merged.shape), "spacing_zyx": spacing.tolist(), "gap_after_slices": int(len(gap_after))}


def merge_dti(entries: list[dict], out_path: Path) -> dict:
    z_positions, step, gap_after = acquired_z_positions(entries)
    first = np.load(entries[0]["path"])
    t, _, y, x = first["volume"].shape
    slice_records = []
    bvals = first["bvals"].astype(np.float32)
    bvecs = first["bvecs"].astype(np.float32)
    for entry in entries:
        data = np.load(entry["path"])
        if data["volume"].shape[0] != t:
            raise ValueError(f"DTI time dimension mismatch: {entry['path']}")
        coords = slice_coords(data["affine_zyx"], data["volume"].shape[1])
        for src_i, coord in enumerate(coords):
            slice_records.append((float(coord), data["volume"][:, src_i]))

    slice_records.sort(key=lambda r: r[0])
    merged = np.stack([r[1] for r in slice_records], axis=1).astype(np.uint16)
    affine = target_affine_like(entries, float(z_positions[0]), step)
    spacing = np.array([step, first["spacing_zyx"][1], first["spacing_zyx"][2]], dtype=np.float32)
    np.savez_compressed(
        out_path,
        volume=merged,
        bvals=bvals,
        bvecs=bvecs,
        spacing_zyx=spacing,
        affine_zyx=affine,
        source_series=np.array([e["series_number"] for e in entries], dtype=np.int32),
        slice_z_positions=z_positions.astype(np.float32),
        gap_after_slices=gap_after.astype(np.int32),
        protocol="merged_dti",
    )
    return {
        "path": str(out_path),
        "kind": "dti",
        "shape": list(merged.shape),
        "spacing_zyx": spacing.tolist(),
        "bvals": {str(int(v)): int((bvals == v).sum()) for v in sorted(set(bvals))},
        "unique_bvecs": int(len({tuple(np.round(v, 6)) for v in bvecs if np.linalg.norm(v) > 0})),
        "gap_after_slices": int(len(gap_after)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted-dir", default="DTI/Subject 01/ConvertedNPZ")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    converted_dir = Path(args.converted_dir)
    out_dir = Path(args.out_dir) if args.out_dir else converted_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(converted_dir)
    base_manifest = [e for e in manifest if not str(Path(e["path"]).name).startswith("merged_")]
    t1_entries = sorted([e for e in base_manifest if e["kind"] == "t1"], key=lambda e: e["series_number"])
    dti_entries = sorted([e for e in base_manifest if e["kind"] == "dti"], key=lambda e: e["series_number"])

    merged_entries = []
    if t1_entries:
        info = merge_t1(t1_entries, out_dir / "merged_t1.npz")
        info.update({"series_number": 1002, "n_files": len(t1_entries), "merged": True})
        merged_entries.append(info)
        print(f"Merged T1: shape={info['shape']} gaps={info['gap_after_slices']} -> {info['path']}")
    if dti_entries:
        info = merge_dti(dti_entries, out_dir / "merged_dti.npz")
        info.update({"series_number": 1007, "n_files": len(dti_entries), "merged": True})
        merged_entries.append(info)
        print(f"Merged DTI: shape={info['shape']} gaps={info['gap_after_slices']} -> {info['path']}")

    new_manifest = base_manifest + merged_entries
    (out_dir / "manifest.json").write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
