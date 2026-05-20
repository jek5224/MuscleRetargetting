"""Regenerate cached waypoints from current tet metadata (waypoint_bary_coords)
for a given muscle JSON list and cache directory.

Usage:
  python tools/regen_waypoints.py --muscles .muscles_r_upleg.json \
      --cache data/motion_cache/walk/R_UpLeg
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--muscles', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--bvh', default='data/motion/walk.bvh')
    args = ap.parse_args()

    skel, bvh_info, mesh_info = load_skeleton()
    skeleton_meshes = load_skeleton_meshes()
    muscle_meshes = load_muscle_meshes(args.muscles)
    load_tet_meshes(muscle_meshes, tet_dir='tet')
    skel.setPositions(np.zeros(skel.getNumDofs()))
    init_soft_bodies(muscle_meshes, skeleton_meshes, skel, mesh_info)
    active = {n: m for n, m in muscle_meshes.items() if m.soft_body is not None}
    print(f'Active muscles: {list(active.keys())}')

    t_frame = _detect_bvh_tframe(args.bvh)
    motion_bvh = MyBVH(args.bvh, bvh_info, skel, T_frame=t_frame)
    print(f'BVH frames: {motion_bvh.mocap_refs.shape[0]}')

    patch_waypoints(args.cache, active, motion_bvh, skel)


if __name__ == '__main__':
    main()
