#!/usr/bin/env python3
"""Assemble the reviewed direct-simulation VL frames 0-25."""

from pathlib import Path

import numpy as np


ROOT = Path(".bake_outputs")
NAME = "L_Vastus_Lateralis_Subdivded"
MOTION = "left_thigh_quasistatic_diverse_smooth_76frame"
OUTPUT = ROOT / "motion_cache" / MOTION / f"{NAME}_attached_frames0_25_v3"

SOURCES = [
    (ROOT / "vl_qualityreserve_tests/frames0_8_v1", range(0, 7)),
    (ROOT / "vl_qualityreserve_tests/frame7_minj01", [7]),
    (ROOT / "vl_qualityreserve_tests/frame8_from_healthy7", [8]),
    (ROOT / "vl_qualityreserve_tests/frame9_from_rebuilt8", [9]),
    (ROOT / "vl_qualityreserve_tests/frame10_from_rebuilt9", [10]),
    (ROOT / "vl_qualityreserve_tests/frame11_from_healthy10", [11]),
    (ROOT / "vl_qualityreserve_tests/frame12_flexiblefloor", [12]),
    (ROOT / "vl_qualityreserve_tests/frames13_14_flexible", range(13, 15)),
    (ROOT / "vl_qualityreserve_tests/frames15_23_dense", range(15, 24)),
    (ROOT / "vl_qualityreserve_tests/frames24_25_from_repaired23", range(24, 26)),
]


def read(directory):
    path = directory / f"{NAME}_chunk_0000.npz"
    data = np.load(path)
    return path, data, {int(frame): row for row, frame in enumerate(data["frames"])}


def main():
    selected = {}
    opened = []
    for directory, frames in SOURCES:
        path, data, rows = read(directory)
        opened.append(data)
        for frame in frames:
            selected[frame] = (path, data, rows[frame])
    if sorted(selected) != list(range(26)):
        raise RuntimeError("assembled VL cache must cover frames 0-25")

    keys = (
        "positions", "inverted_tets", "minimum_jacobian_ratio",
        "maximum_attachment_error", "remaining_contact_samples",
    )
    result = {"frames": np.arange(26, dtype=np.int32)}
    for key in keys:
        result[key] = np.asarray(
            [selected[f][1][key][selected[f][2]] for f in range(26)])
    result["source_cache"] = np.asarray(
        [str(selected[f][0].parent) for f in range(26)])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / f"{NAME}_chunk_0000.npz", **result)
    (OUTPUT / ".done").touch()
    print(OUTPUT)


if __name__ == "__main__":
    main()
