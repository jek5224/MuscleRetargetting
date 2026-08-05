#!/usr/bin/env python3
"""Accumulate joint VI/VL/VM refinement over every BVH frame."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np


MUSCLES = (
    "L_Vastus_Intermedius",
    "L_Vastus_Lateralis",
    "L_Vastus_Medialis",
)


def copy_cache(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for muscle in MUSCLES:
        shutil.copy2(
            source / f"{muscle}_chunk_0000.npz",
            target / f"{muscle}_chunk_0000.npz")
    (target / ".done").touch()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bvh", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=900)
    parser.add_argument("--skip-frame", type=int, default=5)
    parser.add_argument("--guide-weight", type=float, default=0.05)
    parser.add_argument("--max-displacement", type=float, default=0.0)
    parser.add_argument("--volume-weight", type=float, default=4.0)
    parser.add_argument("--inversion-weight", type=float, default=80.0)
    parser.add_argument("--manifold-weight", type=float, default=0.0)
    parser.add_argument("--manifold-tolerance", type=float, default=0.002)
    args = parser.parse_args()

    work_a = args.output.with_name(args.output.name + "_work_a")
    work_b = args.output.with_name(args.output.name + "_work_b")
    copy_cache(args.source, work_a)
    current = work_a
    sample = np.load(
        args.source / "L_Vastus_Intermedius_chunk_0000.npz")
    frame_count = len(sample["positions"])
    for frame in range(frame_count):
        if args.skip_frame >= 0 and frame == args.skip_frame:
            continue
        target = work_b if current == work_a else work_a
        command = [
            sys.executable, "tools/refine_vi_vl_joint_frame.py",
            "--cache-dir", str(current),
            "--output", str(target),
            "--bvh", str(args.bvh),
            "--frame", str(frame),
            "--iterations", str(args.iterations),
            "--lr", "0.000003",
            "--bone-weight", "10000",
            "--include-vm",
            "--cohesion-weight", "50",
            "--cohesion-radius", "0.008",
            "--guide-weight", str(args.guide_weight),
            "--max-displacement", str(args.max_displacement),
            "--volume-weight", str(args.volume_weight),
            "--inversion-weight", str(args.inversion_weight),
            "--manifold-weight", str(args.manifold_weight),
            "--manifold-tolerance", str(args.manifold_tolerance),
        ]
        print(f"=== joint frame {frame} ===", flush=True)
        subprocess.run(command, check=True)
        current = target
    copy_cache(current, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
