#!/usr/bin/env python3
"""Create a diverse quasistatic left-femur/knee collision benchmark BVH.

All channels except LeftUpLeg and LeftLeg are copied from frame zero of the
existing five-pose benchmark.  The knee is deliberately a one-DOF hinge:
LeftLeg Z/Y rotations are always zero and X flexion is always non-negative.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.make_quasistatic_left_thigh_bvh import parse_channels, set_zxy


DEFAULT_SOURCE = Path("data/motion/left_thigh_quasistatic_5pose.bvh")
DEFAULT_OUTPUT = Path("data/motion/left_thigh_quasistatic_diverse_16pose.bvh")

# label, hip flexion, hip abduction, hip internal rotation, knee flexion
POSES = (
    ("neutral",                 0.0,   0.0,   0.0,   0.0),
    ("hip_flex_30",            30.0,   0.0,   0.0,   0.0),
    ("hip_flex_60_knee_30",    60.0,   0.0,   0.0,  30.0),
    ("hip_flex_90_knee_60",    90.0,   0.0,   0.0,  60.0),
    ("deep_flexion",           90.0,   0.0,   0.0,  90.0),
    ("deeper_flexion",         90.0,   0.0,   0.0, 120.0),
    ("hip_extension",         -20.0,   0.0,   0.0,   0.0),
    ("abduction",               0.0,  30.0,   0.0,  20.0),
    ("adduction",               0.0, -20.0,   0.0,  20.0),
    ("internal_rotation",       0.0,   0.0,  25.0,  30.0),
    ("external_rotation",       0.0,   0.0, -25.0,  30.0),
    ("flex_abduct",            55.0,  25.0,  15.0,  70.0),
    ("flex_adduct",            55.0, -15.0, -15.0,  70.0),
    ("cross_body_deep",        65.0, -20.0,  20.0,  85.0),
    ("open_hip_deep",          65.0,  30.0, -20.0,  85.0),
    ("neutral_repeat",          0.0,   0.0,   0.0,   0.0),
)


def build(source: Path, output: Path) -> None:
    lines = source.read_text().splitlines()
    motion_index = lines.index("MOTION")
    hierarchy = lines[:motion_index]
    channels = parse_channels(hierarchy)
    base = [float(value) for value in lines[motion_index + 3].split()]

    frames = []
    for _, flexion, abduction, rotation, knee_flexion in POSES:
        frame = base.copy()
        # Existing left_thigh benchmark convention: negative hip X is
        # flexion, positive knee X is anatomically valid backward flexion.
        set_zxy(frame, channels, "LeftUpLeg",
                z=abduction, x=-flexion, y=rotation)
        set_zxy(frame, channels, "LeftLeg",
                z=0.0, x=knee_flexion, y=0.0)
        frames.append(frame)

    knee = channels["LeftLeg"]
    assert all(frame[knee["Zrotation"]] == 0.0 for frame in frames)
    assert all(frame[knee["Yrotation"]] == 0.0 for frame in frames)
    assert all(frame[knee["Xrotation"]] >= 0.0 for frame in frames)
    assert frames[0] == frames[-1]

    output.parent.mkdir(parents=True, exist_ok=True)
    output_lines = [
        *hierarchy,
        "MOTION",
        f"Frames:\t{len(frames)}",
        lines[motion_index + 2],
        *(" ".join(f"{value:.6f}" for value in frame) for frame in frames),
    ]
    output.write_text("\n".join(output_lines) + "\n")
    print(f"Wrote {output} with {len(frames)} poses")
    for index, pose in enumerate(POSES):
        print(index, pose)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
