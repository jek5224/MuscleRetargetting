#!/usr/bin/env python3
"""Assemble the reviewed fast Smooth-ARAP VI frame segments."""

from pathlib import Path

import numpy as np


ROOT = Path(".bake_outputs")
NAME = "L_Vastus_Intermedius_Subdivided"
OUTPUT = (ROOT / "motion_cache"
          / "left_thigh_quasistatic_diverse_smooth_76frame"
          / f"{NAME}_fast_smooth_arap_full76_v38")


def chunk(directory):
    path = Path(directory) / f"{NAME}_chunk_0000.npz"
    data = np.load(path)
    return {int(frame): (path, row) for row, frame in enumerate(data["frames"])}


def main():
    selected = {}
    sources = [
        (ROOT / "smooth_arap_segments/frames0_12", range(0, 13)),
        (ROOT / "combined_shape_contact"
         / f"{NAME}_smooth_arap_frames13_26_v33", range(13, 26)),
        (ROOT / "combined_shape_contact"
         / f"{NAME}_frames13_27_v30", [26]),
        (ROOT / "smooth_arap_segments/frame27_restart_pose0_fast", [27]),
        (ROOT / "smooth_arap_segments/frame28_from29_fast", [28]),
        (ROOT / "smooth_arap_segments/frame29_restart_pose0_fast", [29]),
        (ROOT / "smooth_arap_segments/frames30_51_fast", range(30, 52)),
        (ROOT / "smooth_arap_segments/frames52_75_fast", range(52, 72)),
        (ROOT / "smooth_arap_segments/frame72_restart_pose0_fast", [72]),
        (ROOT / "smooth_arap_segments/frame73_restart_pose0_fast", [73]),
        (ROOT / "smooth_arap_segments/frame74_restart_pose0_fast", [74]),
        (ROOT / "smooth_arap_segments/frame75_restart_pose0_fast", [75]),
    ]
    for directory, frames in sources:
        available = chunk(directory)
        for frame in frames:
            selected[frame] = available[frame]
    if sorted(selected) != list(range(76)):
        raise RuntimeError("assembled cache does not cover exactly frames 0-75")

    keys = [
        "positions", "inverted_tets", "minimum_jacobian_ratio",
        "maximum_attachment_error", "remaining_contact_samples",
    ]
    output = {"frames": np.arange(76, dtype=np.int32)}
    opened = {}
    for key in keys:
        values = []
        for frame in range(76):
            path, row = selected[frame]
            data = opened.setdefault(path, np.load(path))
            values.append(data[key][row])
        output[key] = np.asarray(values)
    output["source_cache"] = np.asarray(
        [str(selected[f][0].parent) for f in range(76)])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / f"{NAME}_chunk_0000.npz", **output)
    (OUTPUT / ".done").touch()
    print(OUTPUT)


if __name__ == "__main__":
    main()
