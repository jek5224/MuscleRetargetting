#!/usr/bin/env python3
"""Strip render_faces that fold under ARAP deformation.

After bake, scan every frame in the cache and drop any render face that
participates in an adjacent pair with normal dot < -threshold. Updates the
tet npz so viewer/future bakes use the reduced render_faces set.

Usage:
    python tools/strip_fold_faces.py \
        --tet tet_orig_open/L_Vastus_Lateralis_tet.npz \
        --cache data/motion_cache/walk/layered_coll_indep/L_Vastus_Lateralis_chunk_0000.npz \
        --threshold 0.3
"""
import argparse
import glob
import os
import pickle

import numpy as np
import trimesh


def fold_faces(vertices, faces, threshold, protected=None):
    """Return indices of render faces in adjacent-fold pairs (normal dot < -thr).
    `protected` — face indices that must never be dropped (e.g. cap faces)."""
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    fa = tm.face_adjacency
    fn = tm.face_normals
    dots = np.einsum('ij,ij->i', fn[fa[:, 0]], fn[fa[:, 1]])
    bad_pairs = np.where(dots < -threshold)[0]
    protected = protected or set()
    drop = set()
    for i in bad_pairs:
        a, b = int(fa[i, 0]), int(fa[i, 1])
        # prefer dropping face not in protected set
        if b not in protected:
            drop.add(b)
        elif a not in protected:
            drop.add(a)
        # else both protected — can't drop either, skip
    return drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tet', required=True)
    ap.add_argument('--cache', required=True, help='glob pattern for *_chunk_*.npz')
    ap.add_argument('--threshold', type=float, default=0.7)
    ap.add_argument('--max-iters', type=int, default=3)
    args = ap.parse_args()

    with open(args.tet, 'rb') as f:
        td = pickle.load(f)
    rest = td['vertices']
    rf = np.asarray(td['render_faces'], dtype=np.int32)
    n0 = len(rf)
    # Cap faces render green; never drop them — holes would expose raw cap.
    protected = set(int(x) for x in td.get('cap_face_indices', []))

    chunks = sorted(glob.glob(args.cache))
    if not chunks:
        print(f'No cache files matching {args.cache}')
        return
    all_pos = []
    for c in chunks:
        d = np.load(c, allow_pickle=True)
        for i in range(len(d['positions'])):
            all_pos.append(d['positions'][i].astype(np.float64))
    print(f'Loaded {len(all_pos)} baked frames from {len(chunks)} chunk(s)')

    for it in range(args.max_iters):
        keep = np.ones(len(rf), dtype=bool)
        drop_rest = fold_faces(rest, rf, args.threshold, protected)
        total_drop = set(drop_rest)
        for pos in all_pos:
            total_drop |= fold_faces(pos, rf, args.threshold, protected)
        if not total_drop:
            print(f'iter {it}: no folds; done')
            break
        for fi in total_drop:
            keep[fi] = False
        rf = rf[keep]
        print(f'iter {it}: dropped {len(total_drop)} faces; {len(rf)} remain')

    td['render_faces'] = rf
    td['faces'] = rf
    td['surface_face_count'] = len(rf)
    # cap_face_indices must index render_faces; recompute trivially (empty for OBJ)
    anch = set(int(x) for x in td.get('anchor_vertices', []))
    td['cap_face_indices'] = np.array(
        [fi for fi, f in enumerate(rf) if all(int(v) in anch for v in f)],
        dtype=np.int32)
    with open(args.tet, 'wb') as f:
        pickle.dump(td, f)
    print(f'Wrote {args.tet}: {n0} → {len(rf)} render_faces')


if __name__ == '__main__':
    main()
