"""Re-patch walk + run motion cache chunks with 25-fiber waypoints.

Existing chunks (`*_chunk_*.npz`) hold tet vertex traces (`positions`) plus
stale `waypoints_flat` from a 100-fiber pipeline. After resampling fibers
(now 25 per stream) and saving tets via the viewer's Apply N=5 + Save Tet
flow, this script rebuilds `waypoint_bary_coords` against the unchanged tet
topology and re-derives `waypoints_flat` for each frame — no re-bake needed.

Combined output:
  walk × `data/motion_cache/walk/{REGION}_iter50_nobone`
  run  × `data/motion_cache/run/{REGION}_quads_tfl_skin`
Regions: L_UpLeg, R_UpLeg, L_LowLeg, R_LowLeg.
"""
import os
import sys
import json
import glob
import time

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH
from viewer.mesh_loader import MeshLoader
from viewer.zygote_mesh_ui import _detect_bvh_tframe, _flatten_waypoints
from tools.bake_headless_surfcoll import patch_waypoints


SKEL_XML = "data/zygote_skel.xml"
ZYGOTE_DIR = "Zygote_Meshes_251229/"
MESH_SCALE = 0.01

REGIONS = ("L_UpLeg", "R_UpLeg", "L_LowLeg", "R_LowLeg")
WALK_TEMPLATE = "data/motion_cache/walk/{REGION}_iter50_nobone"
RUN_TEMPLATE = "data/motion_cache/run/{REGION}_quads_tfl_skin"
WALK_BVH = "data/motion/walk.bvh"
RUN_BVH = "data/motion/run.bvh"


def load_skeleton():
    skel_info, root_name, bvh_info, _pd, mesh_info, _smpl = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)
    print(f"[1/5] Skeleton: {skel.getNumDofs()} DOFs, {skel.getNumBodyNodes()} bodies")
    return skel, bvh_info, mesh_info


def load_skeleton_meshes():
    skeleton_meshes = {}
    skel_dir = os.path.join(ZYGOTE_DIR, "Skeleton")
    for fname in sorted(os.listdir(skel_dir)):
        if not fname.endswith(".obj"):
            continue
        name = fname.split(".")[0]
        path = os.path.join(skel_dir, fname)
        skeleton_meshes[name] = MeshLoader()
        skeleton_meshes[name].load(path)
        skeleton_meshes[name].color = np.array([0.9, 0.9, 0.9])
        tri = trimesh.load_mesh(path)
        tri.vertices *= MESH_SCALE
        skeleton_meshes[name].trimesh = tri
    print(f"[2/5] Skeleton meshes: {len(skeleton_meshes)}")
    return skeleton_meshes


def discover_leg_muscles():
    muscle_dir = os.path.join(ZYGOTE_DIR, "Muscle")
    out = {}
    for sub in ("UpLeg", "LowLeg"):
        d = os.path.join(muscle_dir, sub)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".obj"):
                continue
            name = fname[:-4]
            out[name] = os.path.join(d, fname)
    return out


def load_muscle_meshes(muscle_paths):
    muscle_meshes = {}
    for name, path in muscle_paths.items():
        mobj = MeshLoader()
        mobj.load(path)
        mobj.color = np.array([0.8, 0.2, 0.2])
        tri = trimesh.load_mesh(path)
        tri.vertices *= MESH_SCALE
        mobj.trimesh = tri
        muscle_meshes[name] = mobj
    print(f"[3/5] Muscle meshes: {len(muscle_meshes)}")
    return muscle_meshes


def load_tets_and_init(muscle_meshes, skeleton_meshes, skel, mesh_info):
    print("[4/5] Loading tets + initializing soft bodies (builds 25-fiber bary)...")
    loaded = 0
    init_ok = 0
    for name, mobj in muscle_meshes.items():
        mobj.load_tetrahedron_mesh(name)
        if mobj.tet_vertices is None:
            print(f"       SKIP {name}: no tet")
            continue
        n_fib = len(mobj.waypoints[0][0]) if (
            getattr(mobj, "waypoints", None) and mobj.waypoints and mobj.waypoints[0]
        ) else None
        if n_fib != 25:
            print(f"       WARN {name}: wp count = {n_fib} (expected 25)")
        loaded += 1
        mobj.init_soft_body(
            skeleton_meshes=skeleton_meshes,
            skeleton=skel,
            mesh_info=mesh_info,
        )
        if mobj.soft_body is not None and getattr(mobj, "waypoint_bary_coords", None):
            init_ok += 1
        else:
            print(f"       WARN {name}: bary build failed")
    print(f"       Tets loaded: {loaded}, bary built: {init_ok}")
    return init_ok


def run_patch(skel, bvh_info, muscle_meshes, bvh_path, cache_template, regions, label):
    if not os.path.exists(bvh_path):
        print(f"[{label}] BVH not found: {bvh_path}")
        return
    t_frame = _detect_bvh_tframe(bvh_path)
    motion_bvh = MyBVH(bvh_path, bvh_info, skel, T_frame=t_frame)
    print(f"[{label}] BVH frames: {motion_bvh.mocap_refs.shape[0]}, T_frame: {t_frame}")
    for region in regions:
        cache_dir = cache_template.replace("{REGION}", region)
        if not os.path.isdir(cache_dir):
            print(f"[{label}] {region}: no cache dir {cache_dir}")
            continue
        chunks = glob.glob(os.path.join(cache_dir, "*_chunk_*.npz"))
        print(f"[{label}] {region}: {len(chunks)} chunks in {cache_dir}")
        patch_waypoints(cache_dir, muscle_meshes, motion_bvh, skel)


def main():
    skel, bvh_info, mesh_info = load_skeleton()
    skeleton_meshes = load_skeleton_meshes()
    muscle_paths = discover_leg_muscles()
    print(f"       Discovered {len(muscle_paths)} leg muscle obj files")
    muscle_meshes = load_muscle_meshes(muscle_paths)
    init_ok = load_tets_and_init(muscle_meshes, skeleton_meshes, skel, mesh_info)
    if init_ok == 0:
        print("No muscles ready for patch. Aborting.")
        sys.exit(1)

    print("[5/5] Patching walk caches...")
    run_patch(skel, bvh_info, muscle_meshes, WALK_BVH, WALK_TEMPLATE, REGIONS, "WALK")
    print("[5/5] Patching run caches...")
    run_patch(skel, bvh_info, muscle_meshes, RUN_BVH, RUN_TEMPLATE, REGIONS, "RUN")
    print("Done.")


if __name__ == "__main__":
    main()
