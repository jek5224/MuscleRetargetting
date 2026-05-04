#!/usr/bin/env python3
"""Audit and rebuild empty `cap_attachments` arrays in tet npz files.

Some tet files (notably multi-stream LowLeg muscles where waypoints
weren't populated at tet build time) were saved with cap_attachments=[].
Without cap_attachments the bake's init_soft_body produces no body
attachments, so fixed verts at origin/insertion never receive bone
targets and stay at the rest pose.

For each tet npz with empty cap_attachments:
  1. Read cap_face_indices + tet_render_faces (or 'faces') -> build
     cap-face vertex adjacency.
  2. Find connected components in that adjacency (one component per
     anatomical cap end).
  3. Match each component centroid to the nearest contour-stream
     endpoint (origin = first contour, insertion = last contour) to
     recover (stream_idx, end_type).
  4. Rebuild cap_attachments as [(anchor_idx, stream_idx, end_type, 0, 0)]
     and pickle the npz back in place.

Usage:
    python tools/fix_cap_attachments.py [--dry-run] [tet/<name>_tet.npz ...]

With no path args, processes every tet npz under tet/.
"""
import argparse
import os
import pickle
import sys
from collections import deque

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def derive_cap_attachments(data):
    """Return list of (anchor, stream, end_type, 0, 0) or None if cannot."""
    cap_face_indices = data.get('cap_face_indices')
    if cap_face_indices is None:
        return None
    cap_face_indices = np.asarray(cap_face_indices, dtype=np.int64)
    if len(cap_face_indices) == 0:
        return None

    face_arr = data.get('render_faces')
    if face_arr is None:
        face_arr = data.get('faces')
    if face_arr is None:
        return None
    face_arr = np.asarray(face_arr, dtype=np.int64)

    rest = data.get('vertices')
    if rest is None:
        rest = data.get('tet_vertices')
    if rest is None:
        return None
    rest = np.asarray(rest, dtype=np.float64)

    contours = data.get('contours')
    if contours is None:
        return None

    # Cap-face vertex adjacency
    cap_adj = {}
    for fi in cap_face_indices:
        if fi >= len(face_arr):
            continue
        f = face_arr[fi]
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            cap_adj.setdefault(a, set()).add(b)
            cap_adj.setdefault(b, set()).add(a)
    if not cap_adj:
        return None

    # Connected components
    components = []
    visited = set()
    for seed in cap_adj:
        if seed in visited:
            continue
        comp = []
        q = deque([seed])
        while q:
            u = q.popleft()
            if u in visited:
                continue
            visited.add(u)
            comp.append(u)
            for nb in cap_adj.get(u, ()):
                if nb not in visited:
                    q.append(nb)
        if comp:
            components.append(comp)

    # Stream endpoint catalog (centroid per (stream, end_type))
    stream_ends = []
    for stream_idx, contour_list in enumerate(contours):
        if contour_list is None or len(contour_list) == 0:
            continue
        for end_type, contour in ((0, contour_list[0]), (1, contour_list[-1])):
            cpts = np.asarray(contour, dtype=np.float64).reshape(-1, 3)
            if len(cpts) == 0:
                continue
            stream_ends.append((stream_idx, end_type, cpts.mean(axis=0)))
    if not stream_ends:
        return None

    derived = []
    for comp in components:
        comp_arr = np.asarray(comp, dtype=np.int64)
        centroid = rest[comp_arr].mean(axis=0)
        best = None
        best_dist = float('inf')
        for stream_idx, end_type, ep in stream_ends:
            d = float(np.linalg.norm(ep - centroid))
            if d < best_dist:
                best_dist = d
                best = (int(comp_arr[0]), stream_idx, end_type)
        if best is not None:
            derived.append((best[0], best[1], best[2], 0, 0))
    return derived


def process(path, dry_run):
    try:
        d = dict(np.load(path, allow_pickle=True))
    except Exception as e:
        return f"  load error: {e}"
    cap = d.get('cap_attachments')
    cap_arr = np.asarray(cap)
    if cap_arr.size > 0:
        return f"  ok ({cap_arr.shape[0] if cap_arr.ndim else 0} attachments)"
    derived = derive_cap_attachments(d)
    if not derived:
        return "  empty cap_attachments AND derivation failed (no cap_face_indices/contours/render_faces)"
    if dry_run:
        return f"  would derive {len(derived)} cap_attachments"
    d['cap_attachments'] = np.array(derived)
    with open(path, 'wb') as f:
        pickle.dump(d, f)
    return f"  derived + wrote {len(derived)} cap_attachments"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='*',
                    help='Tet npz paths (default: every tet/*_tet.npz)')
    ap.add_argument('--dry-run', action='store_true',
                    help='Audit only; do not modify files.')
    args = ap.parse_args()

    if args.paths:
        paths = args.paths
    else:
        tet_dir = os.path.join(PROJECT_ROOT, 'tet')
        paths = sorted(
            os.path.join(tet_dir, f)
            for f in os.listdir(tet_dir)
            if f.endswith('_tet.npz')
        )

    n_ok = n_fix = n_fail = 0
    for path in paths:
        name = os.path.basename(path).replace('_tet.npz', '')
        result = process(path, args.dry_run)
        if 'derived' in result and 'wrote' in result:
            n_fix += 1
        elif 'would derive' in result:
            n_fix += 1
        elif 'failed' in result or 'error' in result:
            n_fail += 1
        else:
            n_ok += 1
        print(f"{name:42s} {result}")

    print()
    print(f"summary: {n_ok} already-ok, {n_fix} {'would-fix' if args.dry_run else 'fixed'}, {n_fail} failed")


if __name__ == '__main__':
    main()
