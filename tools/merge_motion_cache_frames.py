#!/usr/bin/env python3
"""Replace selected frames in a complete motion cache."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def chunk(cache: Path, name: str, must_exist: bool = True) -> Path:
    path = cache / f"{name}_chunk_0000.npz"
    if must_exist and not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", required=True, type=Path)
    parser.add_argument("--patch-cache", required=True, type=Path)
    parser.add_argument("--output-cache", required=True, type=Path)
    parser.add_argument("--name", default="L_Vastus_Intermedius")
    parser.add_argument(
        "--frames",
        help="Optional comma-separated subset of patch frames to merge.")
    args = parser.parse_args()

    with np.load(chunk(args.base_cache, args.name)) as data:
        base_frames = np.asarray(data["frames"], dtype=np.int32)
        positions = np.asarray(data["positions"], dtype=np.float32).copy()
    with np.load(chunk(args.patch_cache, args.name)) as data:
        patch_frames = np.asarray(data["frames"], dtype=np.int32)
        patch_positions = np.asarray(data["positions"], dtype=np.float32)
    if args.frames:
        selected = {
            int(value.strip()) for value in args.frames.split(",")
            if value.strip()}
        keep = np.asarray(
            [int(frame) in selected for frame in patch_frames], dtype=bool)
        patch_frames = patch_frames[keep]
        patch_positions = patch_positions[keep]

    if len(patch_frames) != len(patch_positions):
        raise ValueError("patch frame/position count mismatch")
    row_by_frame = {int(frame): row for row, frame in enumerate(base_frames)}
    for frame, pose in zip(patch_frames, patch_positions):
        if int(frame) not in row_by_frame:
            raise ValueError(f"patch frame {int(frame)} is absent from base")
        positions[row_by_frame[int(frame)]] = pose

    args.output_cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        chunk(args.output_cache, args.name, must_exist=False),
        frames=base_frames, positions=positions)
    (args.output_cache / ".done").touch()
    print(
        f"Merged {len(patch_frames)} selective frames into "
        f"{len(base_frames)}-frame cache: {args.output_cache}")


if __name__ == "__main__":
    main()
