#!/usr/bin/env python3
"""Post-process baked cache to resolve muscle-muscle penetrations.

Loads existing chunk NPZs, detects inter-muscle surface penetrations
using trimesh, pushes penetrating vertices apart, saves back.

Usage:
    python tools/fix_muscle_collision.py data/motion_cache/walk/emu_L_UpLeg
    python tools/fix_muscle_collision.py data/motion_cache/walk/emu_L_UpLeg --margin 0.002 --iters 5
"""
import argparse
import os
import sys
import pickle
import time
from collections import Counter

import numpy as np
import trimesh
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def extract_surface_triangles(tet_elements):
    """Extract boundary faces from tet mesh."""
    face_count = Counter()
    face_orient = {}
    for t in tet_elements:
        v0, v1, v2, v3 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        for f in [(v0, v2, v1), (v0, v1, v3), (v1, v2, v3), (v0, v3, v2)]:
            key = tuple(sorted(f))
            face_count[key] += 1
            if key not in face_orient:
                face_orient[key] = f
    return np.array([face_orient[k] for k, c in face_count.items() if c == 1],
                    dtype=np.int32)


def load_muscle_data(cache_dir, tet_dir='tet'):
    """Load all muscle chunk data and surface info."""
    muscles = {}

    for f in sorted(os.listdir(cache_dir)):
        if not f.endswith('.npz'):
            continue
        mname = f.rsplit('_chunk_', 1)[0]
        if mname not in muscles:
            muscles[mname] = {'chunks': [], 'frames': {}}
        chunk_path = os.path.join(cache_dir, f)
        d = np.load(chunk_path, allow_pickle=True)
        frames = d['frames']
        positions = d['positions']
        muscles[mname]['chunks'].append({
            'path': chunk_path,
            'frames': frames,
            'positions': positions.copy(),
        })
        for fi, frame_num in enumerate(frames):
            muscles[mname]['frames'][int(frame_num)] = (len(muscles[mname]['chunks']) - 1, fi)

    # Load surface triangles and fixed vertices from tet files
    for mname in muscles:
        tet_path = os.path.join(tet_dir, f'{mname}_tet.npz')
        if not os.path.exists(tet_path):
            muscles[mname]['surf_faces'] = None
            muscles[mname]['fixed_verts'] = set()
            continue
        with open(tet_path, 'rb') as f:
            td = pickle.load(f)
        tets = td['tetrahedra'].astype(np.int32)
        sf = extract_surface_triangles(tets)
        muscles[mname]['surf_faces'] = sf
        muscles[mname]['surf_verts'] = sorted(set(np.unique(sf).tolist()))

        # Fixed vertices (cap/anchor)
        fixed = set()
        cap_faces = td.get('cap_face_indices', [])
        sim_faces = td.get('sim_faces', td.get('faces'))
        if sim_faces is not None and len(cap_faces) > 0:
            for fi in cap_faces:
                if fi < len(sim_faces):
                    for vi in sim_faces[fi]:
                        fixed.add(int(vi))
        for vi in td.get('anchor_vertices', []):
            fixed.add(int(vi))
        muscles[mname]['fixed_verts'] = fixed

    print(f"Loaded {len(muscles)} muscles")
    return muscles


def resolve_frame(muscles, frame, margin=0.002, max_iters=5):
    """Resolve muscle-muscle penetrations for a single frame.

    Returns total number of vertices pushed.
    """
    # Collect positions for this frame
    positions = {}
    for mname, mdata in muscles.items():
        if frame not in mdata['frames']:
            continue
        chunk_idx, pos_idx = mdata['frames'][frame]
        positions[mname] = mdata['chunks'][chunk_idx]['positions'][pos_idx].copy()

    if len(positions) < 2:
        return 0

    total_pushed = 0

    for iteration in range(max_iters):
        n_pushed = 0

        # Build trimesh for each muscle
        meshes = {}
        for mname in positions:
            sf = muscles[mname].get('surf_faces')
            if sf is None:
                continue
            pos = positions[mname]
            try:
                tm = trimesh.Trimesh(vertices=pos, faces=sf, process=False)
                meshes[mname] = tm
            except Exception:
                continue

        # Check each pair
        mnames = list(meshes.keys())
        for i in range(len(mnames)):
            for j in range(i + 1, len(mnames)):
                ma, mb = mnames[i], mnames[j]
                mesh_a, mesh_b = meshes[ma], meshes[mb]
                fixed_a = muscles[ma]['fixed_verts']
                fixed_b = muscles[mb]['fixed_verts']

                # AABB pre-filter
                try:
                    if not mesh_a.bounds[0].any() or not mesh_b.bounds[0].any():
                        continue
                    bmin_a, bmax_a = mesh_a.bounds
                    bmin_b, bmax_b = mesh_b.bounds
                    if np.any(bmin_a > bmax_b + margin) or np.any(bmin_b > bmax_a + margin):
                        continue
                except Exception:
                    continue

                # Check A's surface verts inside B
                sv_a = muscles[ma].get('surf_verts', [])
                if sv_a:
                    pos_a = positions[ma][sv_a]
                    try:
                        inside = mesh_b.contains(pos_a)
                        if np.any(inside):
                            inside_idx = np.array(sv_a)[inside]
                            inside_pos = positions[ma][inside_idx]
                            closest, _, face_ids = trimesh.proximity.closest_point(
                                mesh_b, inside_pos)
                            normals = mesh_b.face_normals[face_ids]
                            for k in range(len(inside_idx)):
                                vi = int(inside_idx[k])
                                if vi in fixed_a:
                                    continue
                                positions[ma][vi] = closest[k] + normals[k] * margin
                                n_pushed += 1
                    except Exception:
                        pass

                # Check B's surface verts inside A
                sv_b = muscles[mb].get('surf_verts', [])
                if sv_b:
                    pos_b = positions[mb][sv_b]
                    try:
                        inside = mesh_a.contains(pos_b)
                        if np.any(inside):
                            inside_idx = np.array(sv_b)[inside]
                            inside_pos = positions[mb][inside_idx]
                            closest, _, face_ids = trimesh.proximity.closest_point(
                                mesh_a, inside_pos)
                            normals = mesh_a.face_normals[face_ids]
                            for k in range(len(inside_idx)):
                                vi = int(inside_idx[k])
                                if vi in fixed_b:
                                    continue
                                positions[mb][vi] = closest[k] + normals[k] * margin
                                n_pushed += 1
                    except Exception:
                        pass

        total_pushed += n_pushed
        if n_pushed == 0:
            break

    # Write back corrected positions
    for mname, pos in positions.items():
        if frame not in muscles[mname]['frames']:
            continue
        chunk_idx, pos_idx = muscles[mname]['frames'][frame]
        muscles[mname]['chunks'][chunk_idx]['positions'][pos_idx] = pos.astype(np.float32)

    return total_pushed


def main():
    parser = argparse.ArgumentParser(description="Fix muscle-muscle collisions in baked cache")
    parser.add_argument('cache_dir', help='Path to cache directory (e.g. data/motion_cache/walk/emu_L_UpLeg)')
    parser.add_argument('--tet-dir', default='tet', help='Tet mesh directory')
    parser.add_argument('--margin', type=float, default=0.002, help='Collision margin (m)')
    parser.add_argument('--iters', type=int, default=5, help='Max collision iterations per frame')
    args = parser.parse_args()

    if not os.path.isdir(args.cache_dir):
        print(f"ERROR: {args.cache_dir} not found")
        sys.exit(1)

    muscles = load_muscle_data(args.cache_dir, args.tet_dir)

    # Get all frames
    all_frames = sorted(set(
        f for mdata in muscles.values() for f in mdata['frames'].keys()
    ))
    print(f"{len(all_frames)} frames to process")

    t_total = time.time()
    for frame in all_frames:
        t0 = time.time()
        n_pushed = resolve_frame(muscles, frame, margin=args.margin, max_iters=args.iters)
        dt = time.time() - t0
        print(f"  Frame {frame}: {n_pushed} verts pushed, {dt:.2f}s", flush=True)

    # Save modified chunks back to disk
    print("\nSaving corrected chunks...")
    for mname, mdata in muscles.items():
        for chunk in mdata['chunks']:
            np.savez_compressed(
                chunk['path'],
                frames=chunk['frames'],
                positions=chunk['positions'])
    print(f"Done in {time.time() - t_total:.1f}s")


if __name__ == '__main__':
    main()
