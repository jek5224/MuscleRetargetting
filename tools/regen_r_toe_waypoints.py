"""Regenerate cached waypoints for the 4 R-LowLeg muscles whose toe insertion
body names were broken by the mirror-script bug (now fixed in
tools/fix_r_toe_body_names.py).  Uses tools.bake_headless.patch_waypoints over
the existing tet positions in cache (no re-sim).

Usage: python tools/regen_r_toe_waypoints.py
"""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from tools.bake_headless import (
    load_skeleton, load_skeleton_meshes, load_muscle_meshes,
    load_tet_meshes, init_soft_bodies, patch_waypoints,
)
from core.bvhparser import MyBVH
from viewer.zygote_mesh_ui import _detect_bvh_tframe


AFFECTED = [
    "R_Extensor_Digitorum_Longus",
    "R_Extensor_Hallucis_Longus",
    "R_Flexor_Digitorum_Longus",
    "R_Flexor_Hallucis",
]
BVH_PATH = "data/motion/walk.bvh"
CACHE_DIR = "data/motion_cache/walk/R_LowLeg"
MUSCLE_LIST_PATH = "/tmp/_r_toe_affected.json"


def main():
    # Build minimal muscle JSON for the 4 affected muscles
    muscle_entries = [
        {"name": n, "path": f"Zygote_Meshes_251229/Muscle/LowLeg/{n}.obj"}
        for n in AFFECTED
    ]
    with open(MUSCLE_LIST_PATH, "w") as f:
        json.dump(muscle_entries, f)

    skel, bvh_info, mesh_info = load_skeleton()
    skeleton_meshes = load_skeleton_meshes()
    muscle_meshes = load_muscle_meshes(MUSCLE_LIST_PATH)
    load_tet_meshes(muscle_meshes, tet_dir="tet")
    skel.setPositions(np.zeros(skel.getNumDofs()))
    init_soft_bodies(muscle_meshes, skeleton_meshes, skel, mesh_info)
    active = {n: m for n, m in muscle_meshes.items() if m.soft_body is not None}
    print(f"Active muscles: {list(active.keys())}")

    t_frame = _detect_bvh_tframe(BVH_PATH)
    motion_bvh = MyBVH(BVH_PATH, bvh_info, skel, T_frame=t_frame)
    print(f"BVH frames: {motion_bvh.mocap_refs.shape[0]}")

    patch_waypoints(CACHE_DIR, active, motion_bvh, skel)


if __name__ == "__main__":
    main()
