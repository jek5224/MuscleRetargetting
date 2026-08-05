#!/usr/bin/env python3
"""Apply final bounded SDF cleanup to selected muscle/frame cache entries."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


MUSCLES = (
    "L_Vastus_Intermedius",
    "L_Vastus_Lateralis",
    "L_Vastus_Medialis",
)
TETS = {
    name: Path("tet") / f"{name}_tet.npz" for name in MUSCLES
}
TARGETS = {
    "L_Vastus_Medialis": (13,),
}


def copy_cache(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for name in MUSCLES:
        shutil.copy2(
            source / f"{name}_chunk_0000.npz",
            target / f"{name}_chunk_0000.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bvh", type=Path, required=True)
    args = parser.parse_args()
    work_a = args.output.with_name(args.output.name + "_cleanup_a")
    work_b = args.output.with_name(args.output.name + "_cleanup_b")
    copy_cache(args.source, work_a)
    current = work_a
    for name, frames in TARGETS.items():
        for frame in frames:
            target = work_b if current == work_a else work_a
            copy_cache(current, target)
            command = [
                sys.executable, "tools/blend_vl_frame5_attachment.py",
                "--attached-cache",
                str(current / f"{name}_chunk_0000.npz"),
                "--safe-cache", str(current / f"{name}_chunk_0000.npz"),
                "--output", str(target),
                "--bvh", str(args.bvh),
                "--tet", str(TETS[name]),
                "--name", name,
                "--frame", str(frame),
                "--rings", "8",
                "--contact-exclusion-rings", "0",
            ]
            print(f"cleanup {name} frame {frame}", flush=True)
            subprocess.run(command, check=True)
            current = target
    copy_cache(current, args.output)
    (args.output / ".done").touch()


if __name__ == "__main__":
    main()
