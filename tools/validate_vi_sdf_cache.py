#!/usr/bin/env python3
"""Report femur-SDF penetration for a VI motion cache."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.bvhparser import MyBVH
from tools import bake_emu
from tools.bake_surface_fast import load_tet, surface_faces
from tools.bake_stiff_tet_arap import BoneSDF, prepare_full_cap_group
from tools.bake_stiff_tet_arap import prepare_full_origin_group
from tools.bake_stiff_tet_pbd import build_surface_samples
import test_emu

ap = argparse.ArgumentParser()
ap.add_argument("cache", type=Path)
ap.add_argument("--frame", type=int)
ap.add_argument("--bvh", type=Path,
                default=Path("data/motion/left_thigh_quasistatic_5pose.bvh"))
ap.add_argument("--tet", type=Path,
                default=Path("tet/L_Vastus_Intermedius_tet.npz"))
ap.add_argument("--name", default="L_Vastus_Intermedius")
ap.add_argument("--anchor-caps", action="store_true")
ap.add_argument("--full-origin-cap", action="store_true")
ap.add_argument("--sdf", type=Path,
                default=Path(".bake_outputs/collision_sdf/L_Femur0_sdf.npz"))
a = ap.parse_args()
file = next(a.cache.glob(f"{a.name}_chunk_*.npz"))
positions = np.load(file)["positions"].astype(float)
skel, info, _ = bake_emu.load_skeleton()
data = load_tet(a.tet)
if a.anchor_caps and a.full_origin_cap:
    ap.error("--anchor-caps and --full-origin-cap are mutually exclusive")
if a.full_origin_cap:
    group = prepare_full_origin_group(
        data, a.name, a.tet, skel, test_emu._load_bone_trees())
elif a.anchor_caps:
    group = test_emu.prepare_group_data(
        data, a.name, skel, test_emu._load_bone_trees(),
        source_path=a.tet, attachment_rings=0)
else:
    group = prepare_full_cap_group(
        data, a.name, a.tet,
        skel, test_emu._load_bone_trees())
fixed = np.asarray(group["fixed_vertices"])
fixed_mask = np.zeros(positions.shape[1], bool)
fixed_mask[fixed] = True
samples_i, samples_w = build_surface_samples(
    surface_faces(np.asarray(group["tetrahedra"])))
attachment_supported = np.any(
    fixed_mask[samples_i] & (samples_w > 0), axis=1)
motion = MyBVH(
    str(a.bvh), info, skel,
    T_frame=bake_emu._detect_bvh_tframe(
        str(a.bvh)))
field = BoneSDF(a.sdf)
frames = range(len(positions)) if a.frame is None else [a.frame]
failed = False
for frame in frames:
    x = positions[frame]
    skel.setPositions(motion.mocap_refs[frame].copy())
    points = np.einsum("nij,ni->nj", x[samples_i], samples_w)
    wt = skel.getBodyNode(field.body).getWorldTransform()
    local = (wt.rotation().T @ (points - wt.translation()).T).T
    distance = map_coordinates(
        field.sdf, ((local - field.origin) / field.pitch).T,
        order=1, mode="constant", cval=1.0)
    test = distance[~attachment_supported]
    inside = int(np.sum(test < 0))
    failed |= inside > 0
    print(f"frame={frame} inside={inside} min={test.min():.9f} "
          f"below_1.5mm={np.sum(test < .0015)}")
raise SystemExit(1 if failed else 0)
