#!/usr/bin/env python3
"""Smoothly interpolate a pose BVH and its three-muscle cache."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import numpy as np


NAMES = ("L_Vastus_Intermedius", "L_Vastus_Lateralis",
         "L_Vastus_Medialis")


def smoothstep(value):
    return value * value * (3.0 - 2.0 * value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh", type=Path, required=True)
    parser.add_argument("--output-bvh", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output-cache", type=Path)
    parser.add_argument("--between", type=int, default=4)
    parser.add_argument("--names", default=",".join(NAMES))
    args = parser.parse_args()
    lines = args.bvh.read_text().splitlines()
    marker = lines.index("MOTION")
    rows = np.asarray([
        [float(value) for value in line.split()]
        for line in lines[marker + 3:]], dtype=np.float64)
    alphas = [0.0] + [
        smoothstep(step / (args.between + 1))
        for step in range(1, args.between + 1)]
    motion = []
    source_intervals = []
    for frame in range(len(rows) - 1):
        for alpha in alphas:
            motion.append((1.0 - alpha) * rows[frame] + alpha * rows[frame + 1])
            source_intervals.append((frame, alpha))
    motion.append(rows[-1])
    source_intervals.append((len(rows) - 1, 0.0))
    args.output_bvh.parent.mkdir(parents=True, exist_ok=True)
    header = lines[:marker + 1] + [
        f"Frames:\t{len(motion)}", lines[marker + 2]]
    args.output_bvh.write_text("\n".join(
        header + [" ".join(f"{v:.6f}" for v in row) for row in motion]) + "\n")

    if args.cache is None and args.output_cache is None:
        print(f"frames={len(motion)} keyframes={len(rows)} between={args.between}")
        return
    if args.cache is None or args.output_cache is None:
        parser.error("--cache and --output-cache must be provided together")
    args.output_cache.mkdir(parents=True, exist_ok=True)
    for name in args.names.split(","):
        cache = np.load(args.cache / f"{name}_chunk_0000.npz")
        source = cache["positions"].astype(np.float64)
        output = []
        for frame, alpha in source_intervals:
            if frame == len(source) - 1:
                output.append(source[frame])
            else:
                output.append(
                    (1.0 - alpha) * source[frame] + alpha * source[frame + 1])
        output = np.asarray(output, dtype=np.float32)
        np.savez_compressed(
            args.output_cache / f"{name}_chunk_0000.npz",
            frames=np.arange(len(output), dtype=np.int32), positions=output)
    (args.output_cache / ".done").touch()
    print(f"frames={len(motion)} keyframes={len(rows)} between={args.between}")


if __name__ == "__main__":
    main()
