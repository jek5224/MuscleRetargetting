#!/usr/bin/env python3
"""Independent acceptance gate for a staged TFL frame."""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import trimesh

from core.bvhparser import MyBVH
from core.dartHelper import buildFromInfo, saveSkeletonInfo
from tools.fix_medial_tibia_frame0 import _minimal_rotation, signed_tet_ratios


NAME = 'L_Tensor_Fascia_Lata'
BONES = ('L_Femur', 'L_Tibia_Fibula', 'L_Patella', 'L_Os_Coxae')
NEIGHBORS = ('L_Gluteus_Medius', 'L_Gluteus_Minimus')


def load_frame(directory, name, frame):
    data = np.load(os.path.join(directory, f'{name}_chunk_0000.npz'))
    frames = [int(value) for value in data['frames']]
    return np.asarray(data['positions'][frames.index(frame)], dtype=np.float64)


def posed_context(frame, bvh_path):
    info, root, bvh_info, _, _, _ = saveSkeletonInfo('data/zygote_skel.xml')
    skeleton = buildFromInfo(info, root)
    skeleton.setPositions(np.zeros(skeleton.getNumDofs()))
    rest_transforms = {}
    for bone in BONES:
        body = skeleton.getBodyNode(bone + '0')
        transform = body.getWorldTransform()
        rest_transforms[bone] = (transform.rotation().copy(),
                                 transform.translation().copy())
    motion = MyBVH(bvh_path, bvh_info, skeleton, T_frame=None)
    skeleton.setPositions(motion.mocap_refs[frame])
    meshes = {}
    for bone, (rest_rotation, rest_translation) in rest_transforms.items():
        source = trimesh.load(
            os.path.join('Zygote_Meshes_251229', 'Skeleton', bone + '.obj'),
            process=False)
        rest_vertices = np.asarray(source.vertices, dtype=np.float64) * 0.01
        body = skeleton.getBodyNode(bone + '0')
        transform = body.getWorldTransform()
        local = (rest_rotation.T
                 @ (rest_vertices - rest_translation).T).T
        posed = (transform.rotation() @ local.T).T + transform.translation()
        meshes[bone] = trimesh.Trimesh(posed, source.faces, process=True)
    return skeleton, rest_transforms, meshes


def attachment_targets(rest, levels, skeleton, rest_transforms):
    targets = {}
    for level, bone in ((int(levels.min()), 'L_Os_Coxae'),
                        (int(levels.max()), 'L_Tibia_Fibula')):
        indices = np.where(levels == level)[0]
        r0, t0 = rest_transforms[bone]
        transform = skeleton.getBodyNode(bone + '0').getWorldTransform()
        local = (r0.T @ (rest[indices] - t0).T).T
        targets.update(zip(indices, (transform.rotation() @ local.T).T
                           + transform.translation()))
    return targets


def max_adjacent_twist(positions, levels):
    unique = np.asarray(sorted(int(v) for v in np.unique(levels) if v >= 0))
    centers = np.stack([positions[levels == level].mean(0) for level in unique])
    radial = []
    tangent = []
    for i, level in enumerate(unique):
        indices = np.sort(np.where(levels == level)[0])
        t = (centers[min(i + 1, len(centers) - 1)]
             - centers[max(i - 1, 0)])
        t /= np.linalg.norm(t) + 1e-12
        r = positions[indices[0]] - centers[i]
        r -= t * np.dot(r, t)
        r /= np.linalg.norm(r) + 1e-12
        tangent.append(t)
        radial.append(r)
    angles = []
    for i in range(len(unique) - 1):
        transported = _minimal_rotation(tangent[i], tangent[i + 1]) @ radial[i]
        dot = np.clip(np.dot(transported, radial[i + 1]), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(dot)))
    return float(max(angles, default=0.0))


def validate(cache_dir, neighbor_cache, frame, bvh_path):
    with open(os.path.join('tet', f'{NAME}_tet.npz'), 'rb') as stream:
        tet = pickle.load(stream)
    positions = load_frame(cache_dir, NAME, frame)
    rest = np.asarray(tet['vertices'], dtype=np.float64)
    tets = np.asarray(tet['tetrahedra'], dtype=np.int64)
    levels = np.asarray(tet['vertex_contour_level'], dtype=np.int32)
    skeleton, rest_transforms, bones = posed_context(frame, bvh_path)
    targets = attachment_targets(rest, levels, skeleton, rest_transforms)
    attachment_error = max(np.linalg.norm(positions[index] - target)
                           for index, target in targets.items())
    ratios = signed_tet_ratios(rest, positions, tets)
    q_rest = rest[tets]
    q_posed = positions[tets]
    rest_volume = np.abs(np.einsum(
        'ij,ij->i', q_rest[:, 1] - q_rest[:, 0],
        np.cross(q_rest[:, 2] - q_rest[:, 0],
                 q_rest[:, 3] - q_rest[:, 0])) / 6.0)
    posed_volume = np.abs(np.einsum(
        'ij,ij->i', q_posed[:, 1] - q_posed[:, 0],
        np.cross(q_posed[:, 2] - q_posed[:, 0],
                 q_posed[:, 3] - q_posed[:, 0])) / 6.0)
    total_volume_ratio = float(posed_volume.sum() / rest_volume.sum())
    surface = np.unique(np.asarray(tet['render_faces'], dtype=np.int64))
    fixed = (levels == levels.min()) | (levels == levels.max())
    free_surface = surface[~fixed[surface]]
    bone_inside = {}
    for name, mesh in bones.items():
        candidates = free_surface
        # The first free ring at an attachment is part of the embedded
        # enthesis, not a free collision surface.
        if name == 'L_Os_Coxae':
            candidates = candidates[levels[candidates] >= levels.min() + 2]
        elif name == 'L_Tibia_Fibula':
            candidates = candidates[levels[candidates] <= levels.max() - 2]
        bone_inside[name] = int(mesh.contains(positions[candidates]).sum())

    # Muscle contact excludes fixed and the two attachment-adjacent rings,
    # matching ProjectedNewtonSolver._collision_region.
    belly = (levels >= levels.min() + 2) & (levels <= levels.max() - 2)
    muscle_inside = {}
    tfl_mesh = trimesh.Trimesh(positions, tet['render_faces'], process=True)
    for neighbor in NEIGHBORS:
        with open(os.path.join('tet', f'{neighbor}_tet.npz'), 'rb') as stream:
            neighbor_tet = pickle.load(stream)
        neighbor_positions = load_frame(neighbor_cache, neighbor, frame)
        neighbor_mesh = trimesh.Trimesh(
            neighbor_positions, neighbor_tet['render_faces'], process=True)
        forward = int(np.sum(neighbor_mesh.contains(positions) & belly))
        neighbor_levels = np.asarray(
            neighbor_tet['vertex_contour_level'], dtype=np.int32)
        neighbor_belly = ((neighbor_levels >= neighbor_levels.min() + 2)
                          & (neighbor_levels <= neighbor_levels.max() - 2))
        reverse = int(np.sum(tfl_mesh.contains(neighbor_positions)
                             & neighbor_belly))
        muscle_inside[neighbor] = (forward, reverse)

    report = {
        'attachment_max_mm': attachment_error * 1000.0,
        'inversions': int(np.sum(ratios <= 0)),
        'min_j': float(ratios.min()),
        'max_j': float(ratios.max()),
        'total_volume_ratio': total_volume_ratio,
        'bone_inside': bone_inside,
        'muscle_belly_inside': muscle_inside,
        'max_adjacent_twist_deg': max_adjacent_twist(positions, levels),
    }
    failures = []
    if attachment_error > 1e-7:
        failures.append('attachment error')
    if np.any(ratios <= 0) or float(ratios.min()) < 1e-4:
        failures.append('non-positive/near-singular tets')
    if not 0.98 <= total_volume_ratio <= 1.02:
        failures.append('total volume drift')
    if any(bone_inside.values()):
        failures.append('bone penetration')
    if any(a or b for a, b in muscle_inside.values()):
        failures.append('muscle belly penetration')
    if report['max_adjacent_twist_deg'] > 25.0:
        failures.append('longitudinal twist')
    if failures:
        raise RuntimeError(f'TFL frame {frame} rejected ({", ".join(failures)}): '
                           f'{report}')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--neighbor-cache', required=True)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--bvh', default='data/motion/run.bvh')
    args = parser.parse_args()
    print(f'TFL frame accepted: {validate(args.cache_dir, args.neighbor_cache, args.frame, args.bvh)}')


if __name__ == '__main__':
    main()
