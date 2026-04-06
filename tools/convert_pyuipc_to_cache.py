#!/usr/bin/env python3
"""Convert pyuipc bake output to viewer motion cache format.

The pyuipc bake uses remapped vertices (unused removed) and mm units.
The viewer cache uses original vertex indices and meter-scale units.
This script un-remaps and converts units.

Usage:
    python tools/convert_pyuipc_to_cache.py \
        --input bake_pyuipc/pyuipc_walk1_subject1_f0-131.npz \
        --tet-dir tet_sim \
        --cache-dir data/motion_cache/walk1_subject1
"""
import argparse
import os
import pickle

import numpy as np


SCALE = 1000.0  # bake uses mm, viewer uses meters (well, 0.01-scale)


def compute_remap(tet_path):
    """Reproduce the same vertex remapping as bake_pyuipc.py."""
    with open(tet_path, 'rb') as f:
        data = pickle.load(f)

    verts = data['vertices'].astype(np.float64) * SCALE
    tets = data['tetrahedra'].astype(np.int32)

    # Fix tet orientation
    v0 = verts[tets[:, 0]]
    cross = np.cross(verts[tets[:, 1]] - v0, verts[tets[:, 2]] - v0)
    vol = np.einsum('ij,ij->i', cross, verts[tets[:, 3]] - v0) / 6.0
    neg = vol < 0
    if np.any(neg):
        tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy()

    # Remove degenerate tets
    v0 = verts[tets[:, 0]]
    cross = np.cross(verts[tets[:, 1]] - v0, verts[tets[:, 2]] - v0)
    vol = np.einsum('ij,ij->i', cross, verts[tets[:, 3]] - v0) / 6.0
    good = vol > 0.01
    tets = tets[good]

    # Remove unused vertices
    used = np.unique(tets.ravel())
    n_orig = len(data['vertices'])
    if len(used) < n_orig:
        # remap[old_idx] = new_idx  (-1 if unused)
        remap = np.full(n_orig, -1, dtype=np.int32)
        remap[used] = np.arange(len(used), dtype=np.int32)
        # inverse: inv_remap[new_idx] = old_idx
        inv_remap = used
        return remap, inv_remap, n_orig
    return None, None, n_orig


def load_arap_cache(arap_dirs, mname):
    """Load ARAP cache for a muscle from subdirectories. Returns {frame: positions}."""
    import glob as _glob
    cache = {}
    for arap_dir in arap_dirs:
        for pattern in [f'{mname}_chunk_*.npz', f'{mname}.npz']:
            for path in sorted(_glob.glob(os.path.join(arap_dir, pattern))):
                data = np.load(path)
                for i, f in enumerate(data['frames']):
                    cache[int(f)] = data['positions'][i]
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='pyuipc/IPC bake npz file')
    parser.add_argument('--tet-dir', default='tet_sim', help='Directory with tet files')
    parser.add_argument('--cache-dir', required=True, help='Output cache directory')
    parser.add_argument('--arap-dir', nargs='*', default=None,
                        help='ARAP cache subdirs for unmapped vertex positions '
                             '(e.g. data/motion_cache/walk/L_UpLeg data/motion_cache/walk/R_UpLeg)')
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # Load bake data
    bake = dict(np.load(args.input))

    # Group by muscle name
    muscle_frames = {}
    for key in bake:
        parts = key.rsplit('_f', 1)
        if len(parts) != 2:
            continue
        mname = parts[0]
        frame = int(parts[1])
        if mname not in muscle_frames:
            muscle_frames[mname] = {}
        muscle_frames[mname][frame] = bake[key]

    print(f"Found {len(muscle_frames)} muscles")

    for mname, frames in sorted(muscle_frames.items()):
        tet_path = os.path.join(args.tet_dir, f'{mname}_tet.npz')
        if not os.path.exists(tet_path):
            print(f"  {mname}: tet file not found, skipping")
            continue

        remap, inv_remap, n_orig = compute_remap(tet_path)

        # Load fallback positions for unmapped vertices
        rest_mm = None
        arap_cache = None
        if inv_remap is not None:
            with open(tet_path, 'rb') as fp:
                rest_mm = pickle.load(fp)['vertices'].astype(np.float64) * SCALE
            # Try ARAP cache for better unmapped vertex positions
            if args.arap_dir:
                arap_cache = load_arap_cache(args.arap_dir, mname)
                if arap_cache:
                    print(f"  {mname}: using ARAP cache for {n_orig - len(inv_remap)} unmapped verts")

        sorted_frames = sorted(frames.keys())
        positions_list = []

        for f in sorted_frames:
            pos_mm = frames[f].astype(np.float64)  # remapped positions in mm

            if inv_remap is not None:
                # Un-remap: expand back to original vertex count
                # Use ARAP positions for unmapped vertices if available
                if arap_cache and f in arap_cache:
                    full_pos = arap_cache[f].astype(np.float64) * SCALE
                else:
                    full_pos = rest_mm.copy()
                full_pos[inv_remap] = pos_mm
                pos_mm = full_pos

            # Convert from mm back to viewer scale (same as tet file)
            pos_viewer = (pos_mm / SCALE).astype(np.float32)
            positions_list.append(pos_viewer)

        out_path = os.path.join(args.cache_dir, f'{mname}_chunk_0000.npz')
        np.savez(out_path,
                 frames=np.array(sorted_frames, dtype=np.int32),
                 positions=np.array(positions_list, dtype=np.float32))
        print(f"  {mname}: {len(sorted_frames)} frames, {positions_list[0].shape[0]} verts -> {out_path}")

    print(f"\nDone. Cache saved to {args.cache_dir}")


if __name__ == '__main__':
    main()
