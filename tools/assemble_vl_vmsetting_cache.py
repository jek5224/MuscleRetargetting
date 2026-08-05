#!/usr/bin/env python3
"""Assemble validated VL frames produced with the VM solver configuration."""
from pathlib import Path

import numpy as np


NAME = "L_Vastus_Lateralis_Subdivded"
ROOT = Path(".bake_outputs")
SOURCES = (
    ("vl_surface_attachment_tests/frame0_rigid_v28", range(0, 1)),
    ("vl_vmsetting_segments/frames1_10_v1", range(1, 8)),
    ("vl_vmsetting_segments/frames8_20_v2", range(8, 12)),
    ("vl_vmsetting_segments/frames12_25_v3", range(12, 15)),
    ("vl_vmsetting_segments/frame15_from13_v4", range(15, 16)),
    ("vl_vmsetting_segments/frames16_30_v5", range(16, 17)),
    ("vl_vmsetting_segments/frame17_from15_v6", range(17, 18)),
    ("vl_vmsetting_segments/frames18_35_v7", range(18, 22)),
    ("vl_vmsetting_segments/frames22_40_v8", range(22, 24)),
    ("vl_vmsetting_segments/frame24_from23_v15", range(24, 25)),
    ("vl_vmsetting_segments/frame25_qualityreserve_v17", range(25, 26)),
    ("vl_vmsetting_segments/frame26_qualityreserve_v19", range(26, 27)),
)
output = (ROOT / "motion_cache/left_thigh_quasistatic_diverse_smooth_76frame"
          / f"{NAME}_vmsetting_validated_frames0_26_v30")
selected = []
for relative, frames in SOURCES:
    data = np.load(ROOT / relative / f"{NAME}_chunk_0000.npz")
    rows = [int(np.where(data["frames"] == frame)[0][0]) for frame in frames]
    selected.append({key: data[key][rows] for key in data.files})
keys = selected[0].keys()
merged = {key: np.concatenate([part[key] for part in selected]) for key in keys}
if not np.array_equal(merged["frames"], np.arange(27)):
    raise RuntimeError(f"non-contiguous frames: {merged['frames']}")
output.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output / f"{NAME}_chunk_0000.npz", **merged)
(output / ".done").touch()
print(f"Saved validated frames 0-26 to {output}")
