#!/usr/bin/env python3
"""Build one volumetric cage around all left upper-leg contour tet meshes.

The cage is deliberately independent of the per-muscle tetrahedralizations.
Each muscle vertex is embedded into one cage tetrahedron with four barycentric
weights, so a single cage deformation can drive the whole anatomical group.
"""
import argparse
import json
import os
import pickle

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree
import tetgen


def load_tet(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def orient_tets(vertices, tets):
    q = vertices[tets]
    det = np.einsum(
        "ij,ij->i", q[:, 0] - q[:, 3],
        np.cross(q[:, 1] - q[:, 3], q[:, 2] - q[:, 3]))
    negative = det < 0
    if np.any(negative):
        tets[negative, 1], tets[negative, 2] = (
            tets[negative, 2].copy(), tets[negative, 1].copy())
    return tets


def embed_points(points, cage_v, cage_t):
    """Return containing tet indices and barycentric coordinates."""
    q = cage_v[cage_t]
    centers = q.mean(axis=1)
    tree = cKDTree(centers)
    # The center-nearest tet is not always the containing tet, especially
    # near the boundary. Testing 64 nearby elements remains cheap for this
    # coarse cage and avoids a dependency on exact point-location packages.
    _, candidates = tree.query(points, k=min(64, len(cage_t)))
    candidates = np.atleast_2d(candidates)
    out_tet = np.full(len(points), -1, dtype=np.int32)
    out_w = np.zeros((len(points), 4), dtype=np.float64)
    best_violation = np.full(len(points), np.inf)
    for ci in range(candidates.shape[1]):
        ti = candidates[:, ci]
        tet = q[ti]
        matrix = np.stack(
            (tet[:, 0] - tet[:, 3], tet[:, 1] - tet[:, 3],
             tet[:, 2] - tet[:, 3]), axis=2)
        rhs = points - tet[:, 3]
        try:
            w012 = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            continue
        weights = np.column_stack((w012, 1.0 - w012.sum(axis=1)))
        violation = np.maximum(-weights.min(axis=1),
                               weights.max(axis=1) - 1.0)
        improve = violation < best_violation
        out_tet[improve] = ti[improve]
        out_w[improve] = weights[improve]
        best_violation[improve] = violation[improve]
    return out_tet, out_w, best_violation


def embed_delaunay(points, triangulation):
    simplex = triangulation.find_simplex(points, tol=1e-8)
    weights = np.zeros((len(points), 4), dtype=np.float64)
    valid = simplex >= 0
    transform = triangulation.transform[simplex[valid]]
    relative = points[valid] - transform[:, 3]
    first = np.einsum("nij,nj->ni", transform[:, :3], relative)
    weights[valid, :3] = first
    weights[valid, 3] = 1.0 - first.sum(axis=1)
    violation = np.full(len(points), np.inf)
    violation[valid] = np.maximum(
        -weights[valid].min(axis=1),
        weights[valid].max(axis=1) - 1.0)
    return simplex.astype(np.int32), weights, violation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muscles", default=".muscles_L_UpLeg.json")
    ap.add_argument("--tet-dir", default="tet")
    ap.add_argument("--output", default="cage/L_UpLeg_contour_cage.npz")
    ap.add_argument("--margin-mm", type=float, default=6.0)
    ap.add_argument("--target-tets", type=int, default=1800)
    args = ap.parse_args()

    with open(args.muscles) as f:
        names = [entry["name"] for entry in json.load(f)]
    muscle_data = {}
    all_points = []
    for name in names:
        path = os.path.join(args.tet_dir, f"{name}_tet.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        data = load_tet(path)
        vertices = np.asarray(data["vertices"], dtype=np.float64)
        muscle_data[name] = vertices
        all_points.append(vertices)
    all_points = np.vstack(all_points)

    hull = ConvexHull(all_points)
    surface_v = all_points[hull.vertices].copy()
    # Uniform radial expansion preserves convexity and guarantees that the
    # original hull remains inside. Scale by the requested physical margin.
    center = surface_v.mean(axis=0)
    radial = surface_v - center
    radius = np.linalg.norm(radial, axis=1)
    surface_v = center + radial * (
        1.0 + (args.margin_mm / 1000.0) / np.maximum(radius, 1e-8))[:, None]
    expanded_hull = ConvexHull(surface_v)
    surface_f = expanded_hull.simplices.astype(np.int32)

    mesh_volume = float(expanded_hull.volume)
    max_volume = max(mesh_volume / float(args.target_tets), 1e-9)
    generator = tetgen.TetGen(surface_v, surface_f)
    generator.tetrahedralize(
        order=1, mindihedral=8, minratio=1.3,
        maxvolume=max_volume, steinerleft=max(1000, args.target_tets))
    cage_v = np.asarray(generator.node, dtype=np.float64)
    # Re-triangulate all boundary and Steiner vertices with scipy so its
    # exact find_simplex transform can be reused for embedding. Both
    # triangulations fill the same convex cage; this avoids unreliable
    # nearest-tet-center point location for long boundary elements.
    triangulation = Delaunay(cage_v)
    cage_t = triangulation.simplices.astype(np.int32).copy()

    embeddings = {}
    worst = 0.0
    outside = 0
    for name, points in muscle_data.items():
        tet_index, weights, violation = embed_delaunay(
            points, triangulation)
        embeddings[name] = {
            "tet_index": tet_index,
            "weights": weights.astype(np.float32),
            "vertex_count": len(points),
        }
        worst = max(worst, float(violation.max()))
        outside += int(np.sum(violation > 1e-5))
        print(f"{name}: {len(points)} embedded, "
              f"outside={np.sum(violation > 1e-5)}, "
              f"worst={violation.max():.2e}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({
            "vertices": cage_v.astype(np.float32),
            "tetrahedra": cage_t,
            "surface_faces": surface_f,
            "muscles": embeddings,
            "muscle_names": names,
            "source_tet_dir": args.tet_dir,
            "margin_mm": args.margin_mm,
        }, f)
    print(f"Cage: {len(cage_v)} vertices, {len(cage_t)} tets, "
          f"{outside} outside embeddings, worst={worst:.2e}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
