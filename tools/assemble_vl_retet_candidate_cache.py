#!/usr/bin/env python3
"""Assemble the light insertion-band VL retet candidate frames 0-5."""
from pathlib import Path
import numpy as np

NAME = "L_Vastus_Lateralis_Subdivded"
ROOT = Path(".bake_outputs")
sources = (
    ("vl_retet_tests/insertion27_frame0_v4", range(0, 1)),
    ("vl_retet_tests/insertion27_frames1_5_v4", range(1, 6)),
)
parts = []
for relative, frames in sources:
    data = np.load(ROOT / relative / f"{NAME}_chunk_0000.npz")
    rows = [int(np.where(data["frames"] == frame)[0][0]) for frame in frames]
    parts.append({key: data[key][rows] for key in data.files})
merged = {key: np.concatenate([part[key] for part in parts])
          for key in parts[0]}
output = (ROOT / "motion_cache/left_thigh_quasistatic_diverse_smooth_76frame"
          / f"{NAME}_insertion27_frames0_5_v37")
output.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output / f"{NAME}_chunk_0000.npz", **merged)
(output / ".done").touch()
print(f"Saved light insertion-band candidate frames 0-5 to {output}")
