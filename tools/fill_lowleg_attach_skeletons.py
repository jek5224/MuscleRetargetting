#!/usr/bin/env python3
"""Patch attach_skeleton_names into LowLeg tet npz files via
auto_detect_attachments using loaded skeleton OBJs."""
import os
import sys
import json
import pickle
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import trimesh
from viewer.mesh_loader import MeshLoader

MESH_SCALE = 0.01
SKEL_DIR = "Zygote_Meshes_251229/Skeleton"


def load_skeleton_meshes():
    sk = {}
    for fname in sorted(os.listdir(SKEL_DIR)):
        if not fname.endswith(".obj"):
            continue
        name = fname.split(".")[0]
        m = MeshLoader()
        m.load(os.path.join(SKEL_DIR, fname))
        sk[name] = m
    return sk


def main():
    with open("tools/muscles_L_LowLeg.json") as f:
        muscles = json.load(f)

    skel_meshes = load_skeleton_meshes()
    print(f"Loaded {len(skel_meshes)} skeleton meshes")

    for entry in muscles:
        name = entry["name"]
        obj_path = entry["path"]
        tet_path = f"tet/{name}_tet.npz"
        if not os.path.exists(tet_path):
            print(f"{name}: tet missing, skip")
            continue

        m = MeshLoader()
        m.load(obj_path)
        try:
            m.load_tetrahedron_mesh(name)
        except Exception as e:
            print(f"{name}: load_tet failed: {e}")
            continue

        existing = getattr(m, "attach_skeleton_names", None)
        has_names = (
            existing is not None
            and len(existing) > 0
            and any(any(n for n in g) for g in existing)
        )
        if has_names:
            print(f"{name}: already has attach_skeleton_names, skip")
            continue

        try:
            ok = m.auto_detect_attachments(skel_meshes)
        except Exception as e:
            print(f"{name}: auto_detect failed: {e}")
            continue
        if not ok:
            print(f"{name}: auto_detect returned False")
            continue

        new_names = m.attach_skeleton_names
        # Patch the existing npz file in place
        d = dict(np.load(tet_path, allow_pickle=True))
        d["attach_skeleton_names"] = np.array(new_names, dtype=object)
        with open(tet_path, "wb") as f:
            pickle.dump(d, f)
        print(f"{name}: wrote attach_skeleton_names = {new_names}")


if __name__ == "__main__":
    main()
