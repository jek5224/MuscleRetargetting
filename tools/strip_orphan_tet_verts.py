#!/usr/bin/env python3
"""Remove vertices that are in the tet npz vertex array but referenced
by no tetrahedron.

These "tet-orphan" verts cannot be deformed by ARAP (no tet edges).
On muscles like L_Peroneus_Longus 14.5% of verts are tet-orphans, and
they show up as visibly stuck side surface points after baking. Strip
them entirely: drop from vertices, drop any render/sim face that
references them, and remap every vert-index field to the surviving
verts.

Re-runnable: idempotent on already-clean tet files.

Usage:
    python tools/strip_orphan_tet_verts.py [--dry-run] [tet/<name>_tet.npz ...]

With no path args, processes every tet npz under tet/.
"""
import argparse
import os
import pickle
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _vert_keys(d):
    """Field names whose entries are vertex indices (1D)."""
    return [
        'soft_body_fixed_vertices',
        'tet_anchor_vertices',
    ]


def strip(path, dry_run):
    try:
        d = dict(np.load(path, allow_pickle=True))
    except Exception as e:
        return f"  load error: {e}"

    if 'vertices' not in d or 'tetrahedra' not in d:
        return "  missing vertices/tetrahedra"

    verts = np.asarray(d['vertices'])
    n_v = len(verts)
    tets = np.asarray(d['tetrahedra'], dtype=np.int64)

    used = np.zeros(n_v, dtype=bool)
    used[tets.reshape(-1)] = True
    n_orph = int((~used).sum())
    if n_orph == 0:
        return "  ok (no orphans)"

    if dry_run:
        return f"  would strip {n_orph} orphan verts ({100*n_orph/n_v:.1f}%)"

    # Build remap: old idx -> new idx (or -1 if dropped).
    keep_old = np.where(used)[0]
    remap = np.full(n_v, -1, dtype=np.int64)
    remap[keep_old] = np.arange(len(keep_old))

    # Update verts.
    d['vertices'] = verts[keep_old]
    if 'tet_vertices' in d and d['tet_vertices'] is not None:
        d['tet_vertices'] = np.asarray(d['tet_vertices'])[keep_old]

    # Tets: every index is in keep_old (by construction).
    d['tetrahedra'] = remap[tets].astype(tets.dtype)

    # Render/sim faces: drop any face that references an orphan, then remap.
    for face_key in ('render_faces', 'faces', 'sim_faces'):
        if face_key not in d or d[face_key] is None:
            continue
        f_arr = np.asarray(d[face_key], dtype=np.int64)
        if len(f_arr) == 0:
            continue
        valid_face = used[f_arr].all(axis=1)
        # Map old face index -> new face index (for cap_face_indices).
        new_face_idx = np.full(len(f_arr), -1, dtype=np.int64)
        new_face_idx[valid_face] = np.arange(int(valid_face.sum()))
        d[face_key] = remap[f_arr[valid_face]].astype(f_arr.dtype)

        if face_key == 'render_faces' and 'cap_face_indices' in d and d['cap_face_indices'] is not None:
            cfi = np.asarray(d['cap_face_indices'], dtype=np.int64)
            cfi_valid = cfi[(cfi >= 0) & (cfi < len(f_arr))]
            cfi_mapped = new_face_idx[cfi_valid]
            cfi_mapped = cfi_mapped[cfi_mapped >= 0]
            d['cap_face_indices'] = cfi_mapped.astype(cfi.dtype)
        # If surface_face_count exists alongside render_faces, recompute.
        if face_key == 'render_faces' and 'surface_face_count' in d:
            old_count = int(d['surface_face_count'])
            old_count = min(old_count, len(f_arr))
            d['surface_face_count'] = int(valid_face[:old_count].sum())

    # Vertex-keyed fields. Skip if size doesn't match vert array (legacy
    # files where the array drifted out of sync with vertices).
    if 'vertex_contour_level' in d and d['vertex_contour_level'] is not None:
        vcl = np.asarray(d['vertex_contour_level'])
        if vcl.ndim == 1 and len(vcl) == n_v:
            d['vertex_contour_level'] = vcl[keep_old]

    # Fixed-vertex / anchor index lists.
    for key in _vert_keys(d):
        if key not in d or d[key] is None:
            continue
        old = np.asarray(d[key], dtype=np.int64)
        new = remap[old]
        new = new[new >= 0]
        d[key] = new.astype(np.int64)

    # cap_attachments rows: (anchor_idx, stream, end_type, ...)
    if 'cap_attachments' in d and d['cap_attachments'] is not None:
        ca = np.asarray(d['cap_attachments'])
        if ca.ndim == 2 and ca.shape[0] > 0:
            new_anchors = remap[ca[:, 0].astype(np.int64)]
            mask = new_anchors >= 0
            ca = ca[mask].copy()
            ca[:, 0] = new_anchors[mask]
            d['cap_attachments'] = ca

    # Bary-coord refs: waypoint_bary_coords stores ('tet', tet_idx, bary4) –
    # tet indices remap below, but bary doesn't reference vert idx, so the
    # only thing that needs updating is _tet_neighbor_bary which we don't
    # save. Skip.

    # Write back.
    with open(path, 'wb') as f:
        pickle.dump(d, f)
    return f"  stripped {n_orph} orphans ({100*n_orph/n_v:.1f}%) -> {len(keep_old)} verts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='*',
                    help='Tet npz paths (default: every tet/*_tet.npz)')
    ap.add_argument('--dry-run', action='store_true')
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

    n_clean = n_strip = n_fail = 0
    for path in paths:
        name = os.path.basename(path).replace('_tet.npz', '')
        result = strip(path, args.dry_run)
        if 'no orphans' in result:
            n_clean += 1
        elif 'stripped' in result or 'would strip' in result:
            n_strip += 1
        else:
            n_fail += 1
        print(f"{name:42s} {result}")
    print()
    print(f"summary: {n_clean} clean, {n_strip} {'would-strip' if args.dry_run else 'stripped'}, {n_fail} failed")


if __name__ == '__main__':
    main()
