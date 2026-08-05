#!/usr/bin/env python3
"""Create the five-pose, history-independence BVH for the left thigh test.

The complete hierarchy and all non-left-leg channels come from
``run_vert_sternum_arm_forearm.bvh``.  The five output frames use one source
frame as a stationary whole-body reference and replace only LeftUpLeg and
LeftLeg rotations:

    frame 0: hip neutral, knee   0 degrees
    frame 1: hip flexed,  knee  30 degrees
    frame 2: hip flexed,  knee  60 degrees
    frame 3: hip flexed,  knee  90 degrees
    frame 4: exact duplicate of frame 0

The duplicate endpoints are an explicit quasistatic history-independence
check: a pose-only bake must return the same muscle vertices in frames 0 and 4.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_SOURCE = Path("data/motion/run_vert_sternum_arm_forearm.bvh")
DEFAULT_OUTPUT = Path("data/motion/left_thigh_quasistatic_5pose.bvh")


def parse_channels(hierarchy_lines: list[str]) -> dict[str, dict[str, int]]:
    """Return joint -> channel name -> absolute motion-column index."""
    stack: list[str] = []
    pending_joint: str | None = None
    channel_index = 0
    result: dict[str, dict[str, int]] = {}

    for line in hierarchy_lines:
        stripped = line.strip()
        if stripped.startswith(("ROOT ", "JOINT ")):
            pending_joint = stripped.split(None, 1)[1]
        elif stripped == "{":
            if pending_joint is not None:
                stack.append(pending_joint)
                pending_joint = None
        elif stripped == "}":
            if stack:
                stack.pop()
        elif stripped.startswith("CHANNELS "):
            if not stack:
                raise ValueError("CHANNELS entry outside a joint")
            fields = stripped.split()
            count = int(fields[1])
            names = fields[2:2 + count]
            result[stack[-1]] = {
                name: channel_index + offset
                for offset, name in enumerate(names)
            }
            channel_index += count

    return result


def set_zxy(frame: list[float], channels: dict[str, dict[str, int]],
            joint: str, z: float, x: float, y: float) -> None:
    mapping = channels[joint]
    frame[mapping["Zrotation"]] = z
    frame[mapping["Xrotation"]] = x
    frame[mapping["Yrotation"]] = y


def build(source: Path, output: Path, source_frame: int) -> None:
    lines = source.read_text().splitlines()
    try:
        motion_index = lines.index("MOTION")
    except ValueError as exc:
        raise ValueError(f"{source} has no MOTION section") from exc

    hierarchy = lines[:motion_index]
    channels = parse_channels(hierarchy)
    required = {"LeftUpLeg", "LeftLeg"}
    missing = required.difference(channels)
    if missing:
        raise ValueError(f"missing required joints: {sorted(missing)}")

    frame_count_line = motion_index + 1
    frame_time_line = motion_index + 2
    source_frames = [
        [float(value) for value in line.split()]
        for line in lines[motion_index + 3:]
        if line.strip()
    ]
    if not 0 <= source_frame < len(source_frames):
        raise IndexError(
            f"source frame {source_frame} outside 0..{len(source_frames) - 1}")

    base = source_frames[source_frame]
    expected_channels = sum(len(values) for values in channels.values())
    if len(base) != expected_channels:
        raise ValueError(
            f"motion row has {len(base)} values, hierarchy has "
            f"{expected_channels} channels")

    # ZXY Euler values are in BVH degrees. Hip ab/adduction and twist stay at
    # zero so this benchmark isolates sagittal hip and knee flexion.
    pose_degrees = (
        (0.0, 0.0),
        (20.0, 30.0),
        (40.0, 60.0),
        (60.0, 90.0),
        (0.0, 0.0),
    )
    frames: list[list[float]] = []
    for hip_flexion, knee_flexion in pose_degrees:
        frame = base.copy()
        set_zxy(frame, channels, "LeftUpLeg",
                z=0.0, x=-hip_flexion, y=0.0)
        set_zxy(frame, channels, "LeftLeg",
                z=0.0, x=knee_flexion, y=0.0)
        frames.append(frame)

    if frames[0] != frames[4]:
        raise AssertionError("history-independence endpoint poses differ")

    output.parent.mkdir(parents=True, exist_ok=True)
    output_lines = [
        *hierarchy,
        "MOTION",
        "Frames:\t5",
        lines[frame_time_line],
        *(" ".join(f"{value:.6f}" for value in frame) for frame in frames),
    ]
    output.write_text("\n".join(output_lines) + "\n")
    print(f"Wrote {output}")
    print("Poses (hip flexion, knee flexion): "
          "(0,0), (20,30), (40,60), (60,90), (0,0) degrees")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-frame", type=int, default=0,
        help="Reference frame supplying root and all untouched body channels.")
    args = parser.parse_args()
    build(args.source, args.output, args.source_frame)


if __name__ == "__main__":
    main()
