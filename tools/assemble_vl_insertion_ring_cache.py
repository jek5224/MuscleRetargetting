#!/usr/bin/env python3
"""Assemble validated VL frames with the corrected soft insertion ring."""
from pathlib import Path

import numpy as np

NAME = "L_Vastus_Lateralis_Subdivded"
ROOT = Path(".bake_outputs")
sources = (
    ("vl_true_vm_ring_tests/frame0_ring1_v2", range(0, 1)),
    ("vl_true_vm_ring_tests/frames1_10_ring1_v3", range(1, 7)),
    ("vl_true_vm_ring_tests/frames7_10_ring1_v4", range(7, 11)),
    ("vl_true_vm_ring_tests/frames11_20_ring1_v5", range(11, 15)),
    ("vl_true_vm_ring_tests/frame15_from13_ring1_v7", range(15, 16)),
    ("vl_true_vm_ring_tests/frames16_20_ring1_v8", range(16, 17)),
    ("vl_true_vm_ring_tests/frame17_ring1_v9", range(17, 18)),
    ("vl_true_vm_ring_tests/frame18_from16_ring1_v10", range(18, 19)),
)
parts = []
for relative, frames in sources:
    data = np.load(ROOT / relative / f"{NAME}_chunk_0000.npz")
    rows = [int(np.where(data["frames"] == frame)[0][0]) for frame in frames]
    parts.append({key: data[key][rows] for key in data.files})
merged = {key: np.concatenate([part[key] for part in parts])
          for key in parts[0]}
if not np.array_equal(merged["frames"], np.arange(19)):
    raise RuntimeError(f"non-contiguous frames: {merged['frames']}")
output = (ROOT / "motion_cache/left_thigh_quasistatic_diverse_smooth_76frame"
          / f"{NAME}_insertion_ring_frames0_18_v35")
output.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output / f"{NAME}_chunk_0000.npz", **merged)
(output / ".done").touch()
print(f"Saved insertion-ring frames 0-18 to {output}")
