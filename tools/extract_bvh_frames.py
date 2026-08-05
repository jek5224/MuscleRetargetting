#!/usr/bin/env python3
"""Extract a contiguous inclusive frame range from a BVH motion."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    lines = args.source.read_text().splitlines()
    marker = next(i for i, line in enumerate(lines) if line == "MOTION")
    frames_line = marker + 1
    frame_time_line = marker + 2
    motion = lines[frame_time_line + 1 :]
    selected = motion[args.start : args.end + 1]
    if not selected or len(selected) != args.end - args.start + 1:
        raise ValueError("requested BVH frame range is out of bounds")
    lines[frames_line] = f"Frames:\t{len(selected)}"
    args.output.write_text("\n".join(
        lines[: frame_time_line + 1] + selected) + "\n")


if __name__ == "__main__":
    main()
