#!/usr/bin/env python3
"""Assemble the shape-clean VL cache using the true VM formulation."""
from pathlib import Path

import numpy as np


NAME = "L_Vastus_Lateralis_Subdivded"
ROOT = Path(".bake_outputs")
sources = (
    (ROOT / "vl_vmsetting_segments/frame0_true_vm_v32", range(0, 1)),
    (ROOT / "vl_true_vm_segments/frames1_10_v1", range(1, 11)),
    (ROOT / "vl_true_vm_segments/frames11_20_v2", range(11, 12)),
    (ROOT / "vl_true_vm_segments/frames12_20_v3", range(12, 18)),
    (ROOT / "vl_true_vm_segments/frame18_from16_v4", range(18, 19)),
    (ROOT / "vl_true_vm_segments/frames19_20_v5", range(19, 21)),
)
parts = []
for path, frames in sources:
    data = np.load(path / f"{NAME}_chunk_0000.npz")
    rows = [int(np.where(data["frames"] == frame)[0][0]) for frame in frames]
    parts.append({key: data[key][rows] for key in data.files})
keys = parts[0].keys()
merged = {key: np.concatenate([part[key] for part in parts]) for key in keys}
if not np.array_equal(merged["frames"], np.arange(21)):
    raise RuntimeError(f"non-contiguous frames: {merged['frames']}")
output = (ROOT / "motion_cache/left_thigh_quasistatic_diverse_smooth_76frame"
          / f"{NAME}_true_vm_frames0_20_v34")
output.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output / f"{NAME}_chunk_0000.npz", **merged)
(output / ".done").touch()
print(f"Saved true-VM frames 0-20 to {output}")
