#!/usr/bin/env python3
"""Fast VM-only GPU stabilization while VI/VL remain contact obstacles."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


NAMES = ("L_Vastus_Intermedius", "L_Vastus_Lateralis", "L_Vastus_Medialis")


def copy_cache(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        shutil.copy2(source / f"{name}_chunk_0000.npz",
                     target / f"{name}_chunk_0000.npz")
    (target / ".done").touch()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bvh", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--bone-weight", type=float, default=10000)
    parser.add_argument("--max-displacement", type=float, default=0.008)
    parser.add_argument("--frames", default="0-15")
    args = parser.parse_args()
    if "-" in args.frames:
        first, last = map(int, args.frames.split("-", 1))
        frames = range(first, last + 1)
    else:
        frames = [int(value) for value in args.frames.split(",")]
    work_a = args.output.with_name(args.output.name + "_work_a")
    work_b = args.output.with_name(args.output.name + "_work_b")
    copy_cache(args.source, work_a)
    current = work_a
    for frame in frames:
        target = work_b if current == work_a else work_a
        subprocess.run([
            sys.executable, "tools/refine_vi_vl_joint_frame.py",
            "--cache-dir", str(current), "--output", str(target),
            "--bvh", str(args.bvh), "--frame", str(frame),
            "--iterations", str(args.iterations), "--lr", "0.000004",
            "--bone-weight", str(args.bone_weight), "--include-vm", "--vm-only",
            "--cohesion-weight", "50", "--cohesion-radius", "0.008",
            "--volume-weight", "100", "--inversion-weight", "50000",
            "--max-displacement", str(args.max_displacement),
        ], check=True)
        current = target
    copy_cache(current, args.output)


if __name__ == "__main__":
    main()
