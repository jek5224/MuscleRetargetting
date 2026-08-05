#!/usr/bin/env python3
"""Insert linearly interpolated motion rows between two BVH frames."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--after-frame", type=int, required=True)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    lines = args.source.read_text().splitlines()
    marker = next(i for i, line in enumerate(lines) if line == "MOTION")
    frames_line = marker + 1
    frame_time_line = marker + 2
    motion = lines[frame_time_line + 1 :]
    if not 0 <= args.after_frame < len(motion) - 1:
        raise ValueError("--after-frame must have a following frame")

    first = [float(value) for value in motion[args.after_frame].split()]
    second = [float(value) for value in motion[args.after_frame + 1].split()]
    if len(first) != len(second):
        raise ValueError("adjacent BVH rows have different channel counts")
    inserted = []
    for step in range(1, args.count + 1):
        alpha = step / (args.count + 1)
        values = [(1.0 - alpha) * a + alpha * b
                  for a, b in zip(first, second)]
        inserted.append(" ".join(f"{value:.6f}" for value in values))

    output_motion = (
        motion[: args.after_frame + 1]
        + inserted
        + motion[args.after_frame + 1 :])
    lines[frames_line] = f"Frames:\t{len(output_motion)}"
    args.output.write_text("\n".join(
        lines[: frame_time_line + 1] + output_motion) + "\n")


if __name__ == "__main__":
    main()
