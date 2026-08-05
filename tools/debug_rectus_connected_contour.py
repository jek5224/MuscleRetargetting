#!/usr/bin/env python3
"""Headless rectus-femoris L/M connected contour diagnostic."""
import argparse
import contextlib
import io
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import trimesh
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from viewer.mesh_loader import MeshLoader
from viewer.zygote_mesh_ui import (
    _ensure_counterpart_scalar,
    _apply_master_contour_schedule_to_counterpart,
    _sync_counterpart_level_selection,
    _resample_contours_with_links,
)


MASTER_NAME = "L_Rectus_Femoris_L_Belly"
FOLLOWER_NAME = "L_Rectus_Femoris_M_Belly"
MESH_DIR = "Zygote_Meshes_251229/Muscle/UpLeg"


def load_obj(name):
    path = os.path.join(MESH_DIR, f"{name}.obj")
    m = MeshLoader()
    m.load(path)
    m.trimesh = trimesh.load_mesh(path)
    m.trimesh.vertices *= 0.01
    m.enable_tendon_extension = False
    m.origin_tendon_extension_name = ""
    m.insertion_tendon_extension_name = ""
    return m


def capture(label, fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        result = fn()
    text = buf.getvalue()
    print(f"\n--- {label} ---")
    print(text.rstrip())
    return result, text


def apply_recommended_selection(master, follower, v):
    master.select_levels()
    if getattr(master, "_level_select_checkboxes", None) is None:
        raise RuntimeError("master did not create level-select checkboxes")
    selected = [
        [i for i, checked in enumerate(row) if checked]
        for row in master._level_select_checkboxes
    ]
    master._apply_level_selection()
    master._save_level_select_post_state()
    _sync_counterpart_level_selection(v, MASTER_NAME, master, defer=False)
    return selected


def install_belly_connected_source(master, follower):
    master._connected_contour_mesh_source = [
        [np.asarray(c).copy() for c in master.contours_resampled[0]],
        [np.asarray(c).copy() for c in follower.contours_resampled[0]],
    ]
    master._connected_contour_mesh_params = [
        [np.asarray(p).copy() for p in master.contours_resampled_params[0]],
        [np.asarray(p).copy() for p in follower.contours_resampled_params[0]],
    ]
    master._connected_contour_mesh_fixed = [
        [list(f) for f in master.contours_resampled_fixed[0]],
        [list(f) for f in follower.contours_resampled_fixed[0]],
    ]
    master._connected_contour_mesh_types = [
        [list(t) for t in master.contours_resampled_types[0]],
        [list(t) for t in follower.contours_resampled_types[0]],
    ]
    master._connected_contour_mesh_provenance = [
        [{'part': 'belly', 'component': MASTER_NAME, 'local_level': i}
         for i in range(len(master.contours_resampled[0]))],
        [{'part': 'belly', 'component': FOLLOWER_NAME, 'local_level': i}
         for i in range(len(follower.contours_resampled[0]))],
    ]
    master._connected_contour_mesh_components = [
        {'component': MASTER_NAME, 'stream_start': 0, 'stream_end': 1,
         'origin_tendon': '', 'insertion_tendon': ''},
        {'component': FOLLOWER_NAME, 'stream_start': 1, 'stream_end': 2,
         'origin_tendon': '', 'insertion_tendon': ''},
    ]


def mesh_metrics(master, source_meshes):
    verts = np.asarray(master.contour_mesh_vertices, dtype=np.float64)
    faces = np.asarray(master.contour_mesh_faces, dtype=np.int64)
    tri = verts[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    edges = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            e = tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
            edges.setdefault(e, []).append(fi)
    boundary_edges = [e for e, refs in edges.items() if len(refs) == 1]
    seam_ids = getattr(master, "_connected_seam_vertex_ids", set()) or set()
    seam_boundary_edges = [
        e for e in boundary_edges if e[0] in seam_ids and e[1] in seam_ids
    ]
    rec_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    rec_pts, _ = trimesh.sample.sample_surface(rec_mesh, min(5000, max(1000, len(faces) * 8)))
    src_pts = []
    for src in source_meshes:
        pts, _ = trimesh.sample.sample_surface(src, 5000)
        src_pts.append(pts)
    src_pts = np.vstack(src_pts)
    src_tree = cKDTree(src_pts)
    rec_tree = cKDTree(rec_pts)
    rec_to_src, _ = src_tree.query(rec_pts, k=1)
    src_to_rec, _ = rec_tree.query(src_pts, k=1)
    return {
        "vertices": int(len(verts)),
        "faces": int(len(faces)),
        "area_min": float(np.min(areas)) if len(areas) else 0.0,
        "area_p01": float(np.percentile(areas, 1)) if len(areas) else 0.0,
        "area_median": float(np.median(areas)) if len(areas) else 0.0,
        "tiny_faces": int(np.sum(areas < 1e-10)),
        "boundary_edges": int(len(boundary_edges)),
        "seam_vertices": int(len(seam_ids)),
        "seam_boundary_edges": int(len(seam_boundary_edges)),
        "rec_to_src_mean": float(np.mean(rec_to_src)),
        "rec_to_src_p95": float(np.percentile(rec_to_src, 95)),
        "src_to_rec_mean": float(np.mean(src_to_rec)),
        "src_to_rec_p95": float(np.percentile(src_to_rec, 95)),
    }


def stream_surface_metrics(master, source_meshes):
    verts = np.asarray(master.contour_mesh_vertices, dtype=np.float64)
    faces = np.asarray(master.contour_mesh_faces, dtype=np.int64)
    face_stream = getattr(master, "_face_stream_map", None)
    if face_stream is None or len(face_stream) != len(faces):
        print("\n--- stream surface metrics ---")
        print("missing _face_stream_map")
        return
    print("\n--- stream surface metrics ---")
    for stream_idx, src in enumerate(source_meshes):
        stream_faces = faces[np.asarray(face_stream) == stream_idx]
        if len(stream_faces) == 0:
            print(f"stream={stream_idx} faces=0")
            continue
        rec_mesh = trimesh.Trimesh(vertices=verts, faces=stream_faces, process=False)
        rec_pts, _ = trimesh.sample.sample_surface(
            rec_mesh, min(4000, max(1000, len(stream_faces) * 8)))
        src_pts, _ = trimesh.sample.sample_surface(src, 5000)
        src_tree = cKDTree(src_pts)
        rec_tree = cKDTree(rec_pts)
        rec_to_src, _ = src_tree.query(rec_pts, k=1)
        src_to_rec, _ = rec_tree.query(src_pts, k=1)
        print(
            f"stream={stream_idx} faces={len(stream_faces)} "
            f"rec_to_src_p95={np.percentile(rec_to_src, 95):.6f} "
            f"src_to_rec_p95={np.percentile(src_to_rec, 95):.6f}"
        )


def _polygon_area_3d(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    center = pts.mean(axis=0)
    uu, ss, vh = np.linalg.svd(pts - center, full_matrices=False)
    basis = vh[:2].T
    xy = (pts - center) @ basis
    x = xy[:, 0]
    y = xy[:, 1]
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def _polyline_perimeter(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)))


def contour_metrics(label, original_stream, resampled_stream):
    print(f"\n--- contour metrics {label} ---")
    for level_idx, (orig, resamp) in enumerate(zip(original_stream, resampled_stream)):
        orig_area = _polygon_area_3d(orig)
        res_area = _polygon_area_3d(resamp)
        orig_perim = _polyline_perimeter(orig)
        res_perim = _polyline_perimeter(resamp)
        ratio = res_area / orig_area if orig_area > 1e-12 else 0.0
        print(
            f"level={level_idx} orig_n={len(orig)} res_n={len(resamp)} "
            f"area={orig_area:.8f}->{res_area:.8f} ratio={ratio:.3f} "
            f"perim={orig_perim:.5f}->{res_perim:.5f}"
        )


def export_mesh(master, out_path):
    mesh = trimesh.Trimesh(
        vertices=np.asarray(master.contour_mesh_vertices),
        faces=np.asarray(master.contour_mesh_faces),
        process=False,
    )
    mesh.export(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/rectus_connected_contour.obj")
    args = parser.parse_args()

    master = load_obj(MASTER_NAME)
    follower = load_obj(FOLLOWER_NAME)
    master.linked_counterpart_name = FOLLOWER_NAME
    follower.linked_counterpart_name = MASTER_NAME
    master.linked_drive_counterpart = True
    follower.linked_drive_counterpart = True
    master.linked_use_shared_scalar = True
    follower.linked_use_shared_scalar = True
    master.linked_pair_eps = 1e-5
    follower.linked_pair_eps = 1e-5

    v = SimpleNamespace(
        zygote_muscle_meshes={MASTER_NAME: master, FOLLOWER_NAME: follower},
        zygote_skeleton_meshes={},
    )

    capture("scalar", lambda: _ensure_counterpart_scalar(v, MASTER_NAME, master, defer=False))
    capture("find master contours", lambda: master.find_contours(
        skeleton_meshes={}, spacing_scale=master.contour_spacing_scale, defer=False))
    capture("copy contour schedule", lambda: _apply_master_contour_schedule_to_counterpart(
        v, MASTER_NAME, master, defer=False, label="headless contours"))
    capture("refine master", lambda: master.refine_contours(max_spacing_threshold=0.01, defer=False))
    capture("refine follower", lambda: follower.refine_contours(max_spacing_threshold=0.01, defer=False))

    for obj, name in ((master, MASTER_NAME), (follower, FOLLOWER_NAME)):
        scalar_min = float(obj.scalar_field.min())
        scalar_max = float(obj.scalar_field.max())
        capture(f"transitions {name}", lambda obj=obj, scalar_min=scalar_min, scalar_max=scalar_max:
                obj.find_all_transitions(scalar_min=scalar_min, scalar_max=scalar_max,
                                         num_samples=200,
                                         expected_origin=len(obj.contours[0]),
                                         expected_insertion=len(obj.contours[-1])))
        capture(f"add transitions {name}", lambda obj=obj:
                obj.add_transitions_to_contours(defer=False))
        capture(f"smooth {name}", lambda obj=obj: obj.smoothen_all(defer=False))
        capture(f"cut {name}", lambda obj=obj, name=name:
                obj.cut_streams(cut_method=obj.cutting_method, muscle_name=name))
        capture(f"stream smooth {name}", lambda obj=obj: obj.stream_smoothen_all(defer=False))

    selected, selection_log = capture("select/apply levels", lambda: apply_recommended_selection(master, follower, v))
    _, resample_log = capture("joint resample", lambda: _resample_contours_with_links(v, MASTER_NAME, master, defer=False))
    contour_metrics("master", master._selected_stream_contours[0], master.contours_resampled[0])
    contour_metrics("follower", follower._selected_stream_contours[0], follower.contours_resampled[0])
    install_belly_connected_source(master, follower)
    _, build_log = capture("build connected contour mesh", lambda: master.build_contour_mesh(defer=False))

    metrics = mesh_metrics(master, [master.trimesh, follower.trimesh])
    stream_surface_metrics(master, [master.trimesh, follower.trimesh])
    export_mesh(master, args.out)

    print("\n=== summary ===")
    print(f"selected={selected}")
    for key, value in metrics.items():
        print(f"{key}={value}")
    print(f"exported={args.out}")

    bad_patterns = [
        "Merged connected master/follower",
        "Closing small boundary artifact",
        "Closed gap:",
    ]
    for pattern in bad_patterns:
        if pattern in build_log:
            print(f"WARNING: build log contains {pattern!r}")
    if not re.search(r"Welded \d+ duplicate connected-source vertices", build_log):
        print("WARNING: no connected-source weld reported")


if __name__ == "__main__":
    main()
