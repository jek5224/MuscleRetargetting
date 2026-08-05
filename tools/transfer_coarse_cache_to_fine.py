"""Transfer a stable coarse tet bake to a fine anatomical tet mesh.

The coarse tetrahedra act as a simulation cage.  Fine vertices inside (or
slightly outside) that cage receive its piecewise-affine displacement.  Points
outside the cage use a smooth inverse-distance displacement blend, preserving
the original fine surface instead of projecting it onto the coarse surface.
"""
import argparse
import glob
import os

import numpy as np
from scipy.spatial import cKDTree


def load_tet(path):
    data = np.load(path, allow_pickle=True)
    return data if isinstance(data, dict) else {key: data[key] for key in data.files}


def build_mapping(coarse_vertices, coarse_tets, fine_vertices,
                  candidate_tets=48, outside_neighbors=12):
    tet_points = coarse_vertices[coarse_tets]
    origins = tet_points[:, 3]
    basis = np.stack((tet_points[:, 0] - origins,
                      tet_points[:, 1] - origins,
                      tet_points[:, 2] - origins), axis=-1)
    inverse = np.linalg.inv(basis)
    centroid_tree = cKDTree(tet_points.mean(axis=1))
    _, candidates = centroid_tree.query(
        fine_vertices, k=min(candidate_tets, len(coarse_tets)))
    if candidates.ndim == 1:
        candidates = candidates[:, None]

    chosen_tet = np.full(len(fine_vertices), -1, dtype=np.int32)
    chosen_bary = np.zeros((len(fine_vertices), 4), dtype=np.float64)
    batch = 4096
    for start in range(0, len(fine_vertices), batch):
        stop = min(start + batch, len(fine_vertices))
        ids = candidates[start:stop]
        points = fine_vertices[start:stop, None, :]
        local = points - origins[ids]
        bary3 = np.einsum('nkij,nkj->nki', inverse[ids], local)
        bary = np.concatenate(
            (bary3, 1.0 - bary3.sum(axis=2, keepdims=True)), axis=2)
        quality = bary.min(axis=2)
        best = quality.argmax(axis=1)
        row = np.arange(stop - start)
        best_quality = quality[row, best]
        # A small extrapolation tolerance avoids a seam when the two tet
        # surfaces differ by sub-millimetre remeshing error.
        inside = best_quality >= -0.05
        chosen_tet[start:stop][inside] = ids[row, best][inside]
        chosen_bary[start:stop][inside] = bary[row, best][inside]

    outside = np.where(chosen_tet < 0)[0]
    neighbor_count = min(outside_neighbors, len(coarse_vertices))
    outside_vertex = np.zeros((len(outside), neighbor_count), dtype=np.int32)
    outside_weight = np.zeros((len(outside), neighbor_count), dtype=np.float64)
    if len(outside):
        vertex_tree = cKDTree(coarse_vertices)
        distance, outside_vertex = vertex_tree.query(
            fine_vertices[outside], k=neighbor_count)
        if outside_vertex.ndim == 1:
            outside_vertex = outside_vertex[:, None]
            distance = distance[:, None]
        scale = np.maximum(distance[:, -1:], 1e-8)
        weight = np.exp(-4.0 * (distance / scale) ** 2)
        outside_weight = weight / np.maximum(weight.sum(axis=1, keepdims=True),
                                              1e-12)
    return chosen_tet, chosen_bary, outside, outside_vertex, outside_weight


def build_smooth_mapping(coarse_vertices, fine_vertices, neighbors=96):
    """Smooth cage displacement weights, intended for render-quality transfer.

    Fine tetrahedra are much smaller than the simulation cage elements, so a
    literal per-tet transfer makes cage gradient jumps visible. A broad smooth
    kernel is more stable for the anatomical surface and retains fine detail.
    """
    count = min(neighbors, len(coarse_vertices))
    distance, indices = cKDTree(coarse_vertices).query(
        fine_vertices, k=count)
    if indices.ndim == 1:
        indices = indices[:, None]
        distance = distance[:, None]
    scale = np.maximum(distance[:, -1:], 1e-8)
    weight = np.exp(-6.0 * (distance / scale) ** 2)
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
    return indices.astype(np.int32), weight


def transfer_positions(coarse_rest, coarse_tets, fine_rest, mapping,
                       coarse_positions):
    chosen_tet, bary, outside, outside_vertex, outside_weight = mapping
    coarse_displacement = coarse_positions - coarse_rest
    result = fine_rest.copy()
    inside = np.where(chosen_tet >= 0)[0]
    if len(inside):
        cage_ids = coarse_tets[chosen_tet[inside]]
        result[inside] += np.einsum(
            'ni,nij->nj', bary[inside], coarse_displacement[cage_ids])
    if len(outside):
        result[outside] += np.einsum(
            'ni,nij->nj', outside_weight,
            coarse_displacement[outside_vertex])
    return result


def transfer_smooth(coarse_rest, fine_rest, mapping, coarse_positions):
    indices, weight = mapping
    displacement = coarse_positions - coarse_rest
    return fine_rest + np.einsum(
        'ni,nij->nj', weight, displacement[indices])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-cache', required=True)
    parser.add_argument('--coarse-tet-dir', default='tet')
    parser.add_argument('--fine-tet-dir', default='tet_orig_quality_v2')
    parser.add_argument('--output-cache', required=True)
    parser.add_argument('--muscles-json')
    parser.add_argument('--piecewise-tet', action='store_true',
                        help='Use exact piecewise tet displacement instead of '
                             'the default smooth render transfer.')
    args = parser.parse_args()

    if args.muscles_json:
        import json
        with open(args.muscles_json) as stream:
            muscles = [item['name'] for item in json.load(stream)]
    else:
        muscles = sorted({
            os.path.basename(path).split('_chunk_')[0]
            for path in glob.glob(os.path.join(
                args.source_cache, '*_chunk_*.npz'))
        })
    os.makedirs(args.output_cache, exist_ok=True)

    for muscle in muscles:
        coarse_path = os.path.join(
            args.coarse_tet_dir, f'{muscle}_tet.npz')
        fine_path = os.path.join(args.fine_tet_dir, f'{muscle}_tet.npz')
        chunks = sorted(glob.glob(os.path.join(
            args.source_cache, f'{muscle}_chunk_*.npz')))
        if not chunks or not os.path.isfile(coarse_path) or not os.path.isfile(fine_path):
            print(f'{muscle}: skipped (missing cache or tet)')
            continue
        coarse = load_tet(coarse_path)
        fine = load_tet(fine_path)
        coarse_rest = np.asarray(coarse['vertices'], dtype=np.float64)
        coarse_tets = np.asarray(coarse['tetrahedra'], dtype=np.int32)
        fine_rest = np.asarray(fine['vertices'], dtype=np.float64)
        if args.piecewise_tet:
            mapping = build_mapping(coarse_rest, coarse_tets, fine_rest)
            outside_count = len(mapping[2])
            print(f'{muscle}: {len(fine_rest)} fine verts, '
                  f'{outside_count} outside cage')
        else:
            mapping = build_smooth_mapping(coarse_rest, fine_rest)
            print(f'{muscle}: {len(fine_rest)} fine verts, smooth cage transfer')
        for chunk in chunks:
            with np.load(chunk) as data:
                frames = np.asarray(data['frames'])
                coarse_anim = np.asarray(data['positions'], dtype=np.float64)
            if args.piecewise_tet:
                fine_anim = np.stack([
                    transfer_positions(coarse_rest, coarse_tets, fine_rest,
                                       mapping, pose)
                    for pose in coarse_anim
                ])
            else:
                fine_anim = np.stack([
                    transfer_smooth(coarse_rest, fine_rest, mapping, pose)
                    for pose in coarse_anim
                ])
            output = os.path.join(args.output_cache, os.path.basename(chunk))
            np.savez_compressed(output, frames=frames,
                                positions=fine_anim.astype(np.float32))


if __name__ == '__main__':
    main()
