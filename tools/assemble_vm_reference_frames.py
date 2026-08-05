#!/usr/bin/env python3
"""Assemble individually validated VM reference transfers."""
from pathlib import Path
import argparse
import shutil
import numpy as np


NAME = "L_Vastus_Medialis"
NAMES = ("L_Vastus_Intermedius", "L_Vastus_Lateralis", NAME)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-source", action="append", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        shutil.copy2(args.base / f"{name}_chunk_0000.npz",
                     args.output / f"{name}_chunk_0000.npz")
    path = args.output / f"{NAME}_chunk_0000.npz"
    cache = np.load(path)
    positions = cache["positions"].copy()
    for value in args.frame_source:
        frame_text, source_text = value.split("=", 1)
        frame = int(frame_text)
        source = np.load(Path(source_text) / f"{NAME}_chunk_0000.npz")
        positions[frame] = source["positions"][frame]
    np.savez_compressed(
        path, frames=cache["frames"], positions=positions)
    (args.output / ".done").touch()


if __name__ == "__main__":
    main()
