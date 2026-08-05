#!/usr/bin/env python3
"""Expose the reviewed VI cache with the direct-sim frame-25 replacement."""

from pathlib import Path

import numpy as np


ROOT = Path(".bake_outputs")
NAME = "L_Vastus_Intermedius_Subdivided"
MOTION = "left_thigh_quasistatic_diverse_smooth_76frame"
BASE = ROOT / "motion_cache" / MOTION / f"{NAME}_fast_smooth_arap_full76_v38"
REPLACEMENT = ROOT / "vi_qualityreserve_tests/frame25_from_v38_frame24"
OUTPUT = ROOT / "motion_cache" / MOTION / f"{NAME}_qualityreserve_full76_v45"


def load(directory):
    return np.load(directory / f"{NAME}_chunk_0000.npz")


def main():
    base = load(BASE)
    replacement = load(REPLACEMENT)
    if replacement["frames"].tolist() != [25]:
        raise RuntimeError("replacement must contain only frame 25")

    result = {key: np.array(base[key], copy=True) for key in base.files}
    row = int(np.flatnonzero(result["frames"] == 25)[0])
    for key in (
        "positions", "inverted_tets", "minimum_jacobian_ratio",
        "maximum_attachment_error", "remaining_contact_samples",
    ):
        result[key][row] = replacement[key][0]
    result["source_cache"][row] = str(REPLACEMENT)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / f"{NAME}_chunk_0000.npz", **result)
    (OUTPUT / ".done").touch()
    print(OUTPUT)


if __name__ == "__main__":
    main()
