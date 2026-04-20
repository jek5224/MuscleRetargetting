#!/usr/bin/env python3
"""Build barycentric mapping from contour mesh vertices to original mesh tets.

For each contour mesh vertex, finds which original mesh tet contains it
(at rest pose) and computes barycentric coordinates. Saves mapping to
each contour tet file.

Then converts original mesh bake cache to contour mesh positions using
the mapping.

Usage:
    # Step 1: Build and save mappings
    python tools/build_contour_orig_mapping.py build

    # Step 2: Convert original bake to contour positions
    python tools/build_contour_orig_mapping.py convert \
        --orig-cache data/motion_cache/walk/orig_L_UpLeg \
        --output-dir data/motion_cache/walk/orig_mapped
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

UPLEG_MUSCLES = [
    "Adductor_Brevis", "Adductor_Longus", "Adductor_Magnus",
    "Biceps_Femoris", "Gluteus_Maximus", "Gluteus_Medius",
    "Gluteus_Minimus", "Gracilis", "Iliacus",
    "Inferior_Gemellus", "Obturator_Externus", "Obturator_Internus",
    "Pectineus", "Piriformis", "Popliteus",
    "Quadratus_Femoris", "Rectus_Femoris", "Sartorius",
    "Semimembranosus", "Semitendinosus", "Superior_Gemellus",
    "Tensor_Fascia_Lata", "Vastus_Intermedius", "Vastus_Lateralis",
    "Vastus_Medialis",
]


def compute_barycentric(point, tet_verts):
    """Compute barycentric coordinates of point in tetrahedron.

    tet_verts: (4, 3) array of tet vertex positions.
    Returns (4,) barycentric coordinates. Sum = 1 if inside.
    """
    v0, v1, v2, v3 = tet_verts
    T = np.column_stack([v0 - v3, v1 - v3, v2 - v3])
    try:
        bary3 = np.linalg.solve(T, point - v3)
    except np.linalg.LinAlgError:
        return np.array([-1, -1, -1, -1])
    bary4 = np.array([bary3[0], bary3[1], bary3[2], 1.0 - bary3.sum()])
    return bary4


def find_containing_tet(point, orig_verts, orig_tets, kdtree, k=20):
    """Find which original tet contains the point.

    Uses KDTree to find nearby vertices, then checks their incident tets.
    Returns (tet_index, barycentric_coords) or (-1, None) if not found.
    """
    _, nearby_idx = kdtree.query(point, k=k)

    # Collect candidate tets from nearby vertices
    candidate_tets = set()
    for vi in nearby_idx:
        # Find all tets containing this vertex
        # (precomputed in vert_to_tet map)
        pass  # Will use precomputed map

    return -1, None


def build_vert_to_tet_map(tets, n_verts):
    """Build mapping from vertex index to list of incident tet indices."""
    v2t = [[] for _ in range(n_verts)]
    for ti, t in enumerate(tets):
        for vi in t:
            v2t[int(vi)].append(ti)
    return v2t


def build_mapping_for_muscle(contour_verts, orig_verts, orig_tets):
    """Build barycentric mapping from contour vertices to original tets.

    Returns list of (tet_idx, bary_coords) for each contour vertex.
    If a vertex can't be mapped, uses nearest-tet fallback.
    """
    n_contour = len(contour_verts)
    n_orig = len(orig_verts)
    m_orig = len(orig_tets)

    # Build KDTree on original vertices
    kdtree = cKDTree(orig_verts)

    # Build vertex-to-tet map
    v2t = build_vert_to_tet_map(orig_tets, n_orig)

    # For each contour vertex, find containing tet
    mapping = []  # [(tet_idx, bary_coords), ...]
    n_inside = 0
    n_nearest = 0

    for ci in range(n_contour):
        point = contour_verts[ci]

        # Find nearby original vertices
        _, nearby_idx = kdtree.query(point, k=min(30, n_orig))

        # Collect candidate tets
        candidate_tets = set()
        for vi in nearby_idx:
            candidate_tets.update(v2t[int(vi)])

        # Check each candidate tet
        best_tet = -1
        best_bary = None
        best_min_bary = -float('inf')

        for ti in candidate_tets:
            tet_verts = orig_verts[orig_tets[ti]]
            bary = compute_barycentric(point, tet_verts)
            min_bary = bary.min()

            if min_bary >= -1e-6:
                # Inside this tet (within tolerance)
                best_tet = ti
                best_bary = bary
                n_inside += 1
                break
            elif min_bary > best_min_bary:
                # Track closest tet for fallback
                best_tet = ti
                best_bary = bary
                best_min_bary = min_bary

        if best_bary is None:
            # Absolute fallback: nearest vertex's first tet
            _, nearest = kdtree.query(point, k=1)
            if v2t[int(nearest)]:
                ti = v2t[int(nearest)][0]
                best_tet = ti
                best_bary = compute_barycentric(point, orig_verts[orig_tets[ti]])
            else:
                best_tet = 0
                best_bary = np.array([1.0, 0.0, 0.0, 0.0])
            n_nearest += 1
        elif best_min_bary < -1e-6:
            n_nearest += 1

        mapping.append((best_tet, best_bary))

    return mapping, n_inside, n_nearest


def cmd_build(args):
    """Build and save contour→original mappings."""
    contour_dir = args.contour_dir
    orig_dir = args.orig_dir

    for mname in UPLEG_MUSCLES:
        name = f"L_{mname}"
        contour_path = os.path.join(contour_dir, f"{name}_tet.npz")
        orig_path = os.path.join(orig_dir, f"{name}_tet.npz")

        if not os.path.exists(contour_path) or not os.path.exists(orig_path):
            print(f"  {name}: SKIP (missing files)")
            continue

        # Load contour mesh
        with open(contour_path, 'rb') as f:
            contour_data = pickle.load(f)
        contour_verts = contour_data['vertices'].astype(np.float64)

        # Load original mesh
        with open(orig_path, 'rb') as f:
            orig_data = pickle.load(f)
        orig_verts = orig_data['tet_vertices'].astype(np.float64)
        orig_tets = orig_data['tet_elements'].astype(np.int32)

        t0 = time.time()
        mapping, n_inside, n_nearest = build_mapping_for_muscle(
            contour_verts, orig_verts, orig_tets)
        dt = time.time() - t0

        print(f"  {name}: {len(contour_verts)} contour → {len(orig_verts)} orig, "
              f"{n_inside} inside, {n_nearest} nearest, {dt:.2f}s")

        # Save mapping into contour tet file
        contour_data['contour_to_tet_mapping'] = mapping
        with open(contour_path, 'wb') as f:
            pickle.dump(contour_data, f)

    print("Done. Mappings saved to contour tet files.")


def cmd_convert(args):
    """Convert original mesh bake cache to contour mesh positions."""
    import glob as glob_mod

    contour_dir = args.contour_dir
    orig_cache = args.orig_cache
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting {orig_cache} → {output_dir}")

    for mname in UPLEG_MUSCLES:
        name = f"L_{mname}"
        contour_path = os.path.join(contour_dir, f"{name}_tet.npz")
        if not os.path.exists(contour_path):
            continue

        with open(contour_path, 'rb') as f:
            contour_data = pickle.load(f)

        mapping = contour_data.get('contour_to_tet_mapping')
        if mapping is None:
            print(f"  {name}: SKIP (no mapping, run 'build' first)")
            continue

        n_contour = len(contour_data['vertices'])

        # Load original mesh tet connectivity (for barycentric interpolation)
        orig_path = os.path.join(args.orig_dir, f"{name}_tet.npz")
        with open(orig_path, 'rb') as f:
            orig_data = pickle.load(f)
        orig_tets = orig_data['tet_elements'].astype(np.int32)

        # Load original bake chunks
        chunks = sorted(glob_mod.glob(os.path.join(orig_cache, f"{name}_chunk_*.npz")))
        if not chunks:
            print(f"  {name}: SKIP (no cache chunks)")
            continue

        # Convert each chunk
        for chunk_path in chunks:
            d = np.load(chunk_path)
            frames = d['frames']
            orig_positions = d['positions']  # (n_frames, n_orig_verts, 3)

            contour_positions = np.zeros((len(frames), n_contour, 3), dtype=np.float32)

            for fi in range(len(frames)):
                orig_pos = orig_positions[fi]
                for ci, (tet_idx, bary) in enumerate(mapping):
                    if tet_idx < 0 or tet_idx >= len(orig_tets):
                        continue
                    tet_vi = orig_tets[tet_idx]
                    contour_positions[fi, ci] = (
                        bary[0] * orig_pos[tet_vi[0]] +
                        bary[1] * orig_pos[tet_vi[1]] +
                        bary[2] * orig_pos[tet_vi[2]] +
                        bary[3] * orig_pos[tet_vi[3]]
                    )

            # Save converted chunk
            chunk_name = os.path.basename(chunk_path)
            out_path = os.path.join(output_dir, chunk_name)
            np.savez_compressed(out_path, frames=frames, positions=contour_positions)

        print(f"  {name}: {len(chunks)} chunks converted")

    print(f"Done. Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Contour↔Original mesh mapping")
    sub = parser.add_subparsers(dest='command')

    p_build = sub.add_parser('build', help='Build and save mappings')
    p_build.add_argument('--contour-dir', default='tet')
    p_build.add_argument('--orig-dir', default='tet_orig')

    p_convert = sub.add_parser('convert', help='Convert original bake to contour')
    p_convert.add_argument('--contour-dir', default='tet')
    p_convert.add_argument('--orig-dir', default='tet_orig')
    p_convert.add_argument('--orig-cache', required=True,
                           help='Original mesh bake cache dir')
    p_convert.add_argument('--output-dir', required=True,
                           help='Output directory for contour mesh cache')

    args = parser.parse_args()
    if args.command == 'build':
        cmd_build(args)
    elif args.command == 'convert':
        cmd_convert(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
