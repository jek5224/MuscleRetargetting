#!/usr/bin/env python3
"""Headless test: build contour mesh + tet for all L_* lower-leg muscles.
Reports key ARAP-relevant metrics per muscle."""
import os
import sys
import gc
import io
import pickle
import contextlib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Suppress verbose mesh-loader output during scripted tests
from viewer.mesh_loader import MeshLoader

MUSCLES = [
    "L_Soleus", "L_Plantaris", "L_Peroneus_Brevis", "L_Peroneus_Longus",
    "L_Peroneus_Tertius",
    "L_Gastrocnemius", "L_Flexor_Hallucis", "L_Tibialis_Anterior",
    "L_Tibialis_Posterior", "L_Extensor_Hallucis_Longus",
    "L_Extensor_Digitorum_Longus", "L_Flexor_Digitorum_Longus",
]

MESH_DIR = "Zygote_Meshes_251229/Muscle/LowLeg"
PKL_DIR = "Zygote_Meshes_251229/Muscle"


def test_one(name):
    obj_path = os.path.join(MESH_DIR, f"{name}.obj")
    pkl_path = os.path.join(PKL_DIR, f"{name}.anim.pkl")
    if not os.path.exists(obj_path):
        return {"name": name, "error": "no_obj"}

    m = MeshLoader()
    m.load(obj_path)
    if os.path.exists(pkl_path):
        try:
            m.load_animation_state(pkl_path)
        except Exception as e:
            return {"name": name, "error": f"load_state: {e}"}

    if getattr(m, "contours_resampled", None) is None:
        # Try resample if state didn't have it cached
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                m.resample_contours(base_samples=32, defer=False)
        except Exception as e:
            return {"name": name, "error": f"resample: {e}"}
        if getattr(m, "contours_resampled", None) is None:
            return {"name": name, "error": "no_contours_resampled_after_resample"}

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            m.build_contour_mesh(defer=False)
            ok = m.tetrahedralize_contour_mesh()
    except Exception as e:
        return {"name": name, "error": f"tet: {e}"}

    if not ok or m.tet_vertices is None or m.tet_tetrahedra is None:
        return {"name": name, "error": "tet_failed"}

    n_verts = len(m.tet_vertices)
    n_tets = len(m.tet_tetrahedra)

    # Inverted check
    v0 = m.tet_vertices[m.tet_tetrahedra[:, 0]].astype(np.float64)
    cr = np.cross(m.tet_vertices[m.tet_tetrahedra[:, 1]].astype(np.float64) - v0,
                  m.tet_vertices[m.tet_tetrahedra[:, 2]].astype(np.float64) - v0)
    vol = np.einsum("ij,ij->i", cr,
                    m.tet_vertices[m.tet_tetrahedra[:, 3]].astype(np.float64) - v0) / 6.0
    n_inverted = int(np.sum(vol < 0))
    n_sliver = int(np.sum((vol >= 0) & (vol < 1e-12)))

    # Orphan verts (not used by any tet)
    used = np.unique(m.tet_tetrahedra.ravel())
    n_orphan_verts = n_verts - len(used)

    # Anchor connectivity
    incident = np.zeros(n_verts, dtype=np.int32)
    for tet in m.tet_tetrahedra:
        for v in tet:
            incident[int(v)] += 1
    anchors = getattr(m, "tet_anchor_vertices", []) or []
    n_orphan_anchors = sum(1 for ai in anchors if 0 <= ai < n_verts and incident[ai] == 0)
    n_weak_anchors = sum(1 for ai in anchors if 0 <= ai < n_verts and 0 < incident[ai] < 3)

    n_streams = len(m.contours_resampled) if m.contours_resampled is not None else 0

    return {
        "name": name,
        "streams": n_streams,
        "verts": n_verts,
        "tets": n_tets,
        "inverted": n_inverted,
        "slivers": n_sliver,
        "orphan_verts": n_orphan_verts,
        "anchors": len(anchors),
        "orphan_anchors": n_orphan_anchors,
        "weak_anchors": n_weak_anchors,
    }


def main():
    results = []
    for name in MUSCLES:
        print(f"Testing {name}...", flush=True)
        r = test_one(name)
        results.append(r)
        gc.collect()

    print("\n=== Summary ===")
    print(f"{'muscle':<32} {'str':>3} {'verts':>5} {'tets':>6} {'inv':>3} {'sliv':>4} "
          f"{'orph_v':>6} {'anch':>4} {'orph_a':>6} {'weak_a':>6}")
    for r in results:
        if "error" in r:
            print(f"{r['name']:<32} ERROR: {r['error']}")
            continue
        print(f"{r['name']:<32} {r['streams']:>3} {r['verts']:>5} {r['tets']:>6} "
              f"{r['inverted']:>3} {r['slivers']:>4} {r['orphan_verts']:>6} "
              f"{r['anchors']:>4} {r['orphan_anchors']:>6} {r['weak_anchors']:>6}")


if __name__ == "__main__":
    main()
