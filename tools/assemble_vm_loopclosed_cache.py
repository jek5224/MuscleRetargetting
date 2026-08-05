#!/usr/bin/env python3
"""Expose VM with frame 75 solved from the identical frame-0 equilibrium."""

from pathlib import Path

import numpy as np


ROOT = Path(".bake_outputs")
NAME = "L_Vastus_Medialis_Subdivded"
MOTION = "left_thigh_quasistatic_diverse_smooth_76frame"
BASE = ROOT / "motion_cache" / MOTION / f"{NAME}_qualityreserve_full76_v3"
REPLACEMENT = ROOT / "vm_loopclosure_tests/frame75_from_frame0"
OUTPUT = ROOT / "motion_cache" / MOTION / f"{NAME}_qualityreserve_loopclosed_full76_v4"


def load(directory):
    return np.load(directory / f"{NAME}_chunk_0000.npz")


def main():
    base = load(BASE)
    replacement = load(REPLACEMENT)
    if replacement["frames"].tolist() != [75]:
        raise RuntimeError("replacement must contain only frame 75")
    result = {key: np.array(base[key], copy=True) for key in base.files}
    row = int(np.flatnonzero(result["frames"] == 75)[0])
    for key in (
        "positions", "inverted_tets", "minimum_jacobian_ratio",
        "maximum_attachment_error", "remaining_contact_samples",
    ):
        result[key][row] = replacement[key][0]
    if "source_cache" in result:
        result["source_cache"][row] = str(REPLACEMENT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / f"{NAME}_chunk_0000.npz", **result)
    (OUTPUT / ".done").touch()
    print(OUTPUT)


if __name__ == "__main__":
    main()
