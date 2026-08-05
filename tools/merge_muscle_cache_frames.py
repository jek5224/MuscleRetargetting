#!/usr/bin/env python3
"""Replace selected frames for all three quadriceps caches."""
import argparse
from pathlib import Path
import numpy as np

NAMES = ("L_Vastus_Intermedius", "L_Vastus_Lateralis",
         "L_Vastus_Medialis")

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--overlay", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--frames", required=True)
parser.add_argument("--overlay-frames")
parser.add_argument("--names", default=",".join(NAMES))
args = parser.parse_args()
frames = [int(value) for value in args.frames.split(",")]
overlay_frames = (
    [int(value) for value in args.overlay_frames.split(",")]
    if args.overlay_frames else frames)
if len(overlay_frames) != len(frames):
    parser.error("--overlay-frames must match --frames length")
selected_names = set(args.names.split(","))
args.output.mkdir(parents=True, exist_ok=True)
for name in NAMES:
    base = np.load(args.base / f"{name}_chunk_0000.npz")
    positions = base["positions"].copy()
    if name in selected_names:
        overlay = np.load(args.overlay / f"{name}_chunk_0000.npz")
        positions[frames] = overlay["positions"][overlay_frames]
    np.savez_compressed(
        args.output / f"{name}_chunk_0000.npz",
        frames=base["frames"], positions=positions)
(args.output / ".done").touch()
