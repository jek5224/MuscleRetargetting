#!/usr/bin/env python3
"""Assemble the validated VL surface-attached frame-zero and continuation."""
from pathlib import Path

import numpy as np


NAME = "L_Vastus_Lateralis_Subdivded"
ROOT = Path(".bake_outputs")
SOURCES = (
    (ROOT / "vl_surface_attachment_tests/frame0_v3", range(0, 1)),
    (ROOT / "vl_surface_attachment_tests/frames1_5_v2", range(1, 6)),
    (ROOT / "vl_surface_attachment_tests/frames6_7_v2", range(6, 8)),
    (ROOT / "vl_surface_attachment_tests/frame8_v3", range(8, 9)),
    (ROOT / "vl_surface_attachment_tests/frames9_onward_v4", range(9, 12)),
    (ROOT / "vl_surface_attachment_tests/frame12_v5", range(12, 13)),
    (ROOT / "vl_surface_attachment_tests/frame13_from11_v9", range(13, 14)),
    (ROOT / "vl_surface_attachment_tests/frame14_v11", range(14, 15)),
    (ROOT / "vl_surface_attachment_tests/frames15_onward_v12", range(15, 17)),
    (ROOT / "vl_surface_attachment_tests/frames17_onward_v13", range(17, 18)),
    (ROOT / "vl_surface_attachment_tests/frames18_25_local_v20", range(18, 19)),
    (ROOT / "vl_surface_attachment_tests/frames19_onward_local_v22", range(19, 20)),
    (ROOT / "vl_surface_attachment_tests/frames20_onward_local_v23", range(20, 23)),
    (ROOT / "vl_surface_attachment_tests/frames23_onward_local_v24", range(23, 28)),
)
OUTPUT = (ROOT / "motion_cache/left_thigh_quasistatic_diverse_smooth_76frame"
          / f"{NAME}_surface_attached_frames0_27_v26")

parts = []
for path, selected in SOURCES:
    source = np.load(path / f"{NAME}_chunk_0000.npz")
    rows = [int(np.where(source["frames"] == frame)[0][0])
            for frame in selected]
    parts.append({key: source[key][rows] for key in source.files})
keys = (
    "frames", "positions", "inverted_tets", "minimum_jacobian_ratio",
    "maximum_attachment_error", "remaining_contact_samples",
)
OUTPUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    OUTPUT / f"{NAME}_chunk_0000.npz",
    **{key: np.concatenate([part[key] for part in parts]) for key in keys})
(OUTPUT / ".done").touch()
frames = np.concatenate([part["frames"] for part in parts])
if not np.array_equal(frames, np.arange(28)):
    raise RuntimeError(f"assembled frames are not contiguous 0-27: {frames}")
print(f"Saved validated frames 0-27 to {OUTPUT}")
