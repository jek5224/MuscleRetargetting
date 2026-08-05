#!/usr/bin/env python3
"""Repair medial tibial insertion routing for run.bvh frame 0.

Creates a viewer-readable local overlay; source caches and tet files are never
modified.  Each distal contour band is translated smoothly toward the tibia
exterior, avoiding the sharp cap-only kink produced by embedded attachment
targets. Remaining penetrations receive a local closest-surface correction.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.dartHelper import buildFromInfo, saveSkeletonInfo
from core.bvhparser import MyBVH

MUSCLES = ('L_Sartorius', 'L_Gracilis', 'L_Semitendinosus',
           'L_Semimembranosus')
TFL = 'L_Tensor_Fascia_Lata'
SOURCE = 'data/motion_cache/run/L_UpLeg_headless_full_corrected'
OUTPUT = '.bake_outputs/motion_cache/run/L_UpLeg_medial_tibia_repair'
MARGIN = 0.0015


def _minimal_rotation(v_from, v_to):
    a = v_from / (np.linalg.norm(v_from) + 1e-12)
    b = v_to / (np.linalg.norm(v_to) + 1e-12)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-10:
        return np.eye(3) if c > 0 else -np.eye(3) + 2.0 * np.outer(a, a)
    k = axis / s
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + K * s + (K @ K) * (1.0 - c)


def _rmf(centers, seed):
    n = len(centers)
    tangents = np.zeros((n, 3))
    tangents[0] = centers[1] - centers[0]
    tangents[-1] = centers[-1] - centers[-2]
    if n > 2:
        tangents[1:-1] = centers[2:] - centers[:-2]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12
    e1 = np.zeros_like(tangents)
    e1[0] = seed - tangents[0] * np.dot(seed, tangents[0])
    if np.linalg.norm(e1[0]) < 1e-8:
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(fallback, tangents[0])) > 0.9:
            fallback = np.array([0.0, 0.0, 1.0])
        e1[0] = fallback - tangents[0] * np.dot(fallback, tangents[0])
    e1[0] /= np.linalg.norm(e1[0])
    for i in range(1, n):
        e1[i] = _minimal_rotation(tangents[i - 1], tangents[i]) @ e1[i - 1]
        e1[i] -= tangents[i] * np.dot(e1[i], tangents[i])
        e1[i] /= np.linalg.norm(e1[i]) + 1e-12
    e2 = np.cross(tangents, e1)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True) + 1e-12
    return tangents, e1, e2


def untwist_body(rest, current, levels):
    """Remove axial corkscrew and high-frequency centerline bending."""
    unique = np.array(sorted(int(v) for v in np.unique(levels) if v >= 0))
    if len(unique) < 3:
        return current
    rc = np.stack([rest[levels == lv].mean(axis=0) for lv in unique])
    cc = np.stack([current[levels == lv].mean(axis=0) for lv in unique])
    # Fair the longitudinal axis while fixing both endpoint centers. A strong
    # second-difference penalty removes the S/Z-shaped centerline inherited
    # from blended skinning without forcing the muscle to become a straight
    # chord through the knee.
    n_levels = len(unique)
    d2 = np.zeros((max(n_levels - 2, 0), n_levels))
    for i in range(n_levels - 2):
        d2[i, i:i + 3] = (1.0, -2.0, 1.0)
    A = np.eye(n_levels) + 35.0 * (d2.T @ d2)
    rhs = cc.copy()
    endpoint_weight = 1e6
    A[0, 0] += endpoint_weight
    A[-1, -1] += endpoint_weight
    rhs[0] += endpoint_weight * cc[0]
    rhs[-1] += endpoint_weight * cc[-1]
    cc_smooth = np.linalg.solve(A, rhs)
    first_ring = np.where(levels == unique[0])[0]
    seed_r = rest[first_ring[0]] - rc[0]
    tr, r1, r2 = _rmf(rc, seed_r)
    seed_c = _minimal_rotation(tr[0], (cc_smooth[1] - cc_smooth[0])) @ r1[0]
    tc, c1, c2 = _rmf(cc_smooth, seed_c)

    rest_len = np.linalg.norm(np.diff(rc, axis=0), axis=1).sum()
    cur_len = np.linalg.norm(np.diff(cc_smooth, axis=0), axis=1).sum()
    bulge = float(np.clip(np.sqrt(rest_len / max(cur_len, 1e-8)), 0.85, 1.35))
    result = current.copy()
    for li, lv in enumerate(unique):
        idx = np.where(levels == lv)[0]
        off = rest[idx] - rc[li]
        u = off @ r1[li]
        v = off @ r2[li]
        axial = off @ tr[li]
        reconstructed = (cc_smooth[li] + bulge * u[:, None] * c1[li]
                         + bulge * v[:, None] * c2[li]
                         + axial[:, None] * tc[li])
        # Reconstruct every ring, including cap orientation. Only endpoint
        # *centers* are fixed; retaining the old cap orientation concentrated
        # the entire residual twist into the terminal contour bands.
        result[idx] = reconstructed
    return result


def attach_cap_with_taper(x, levels, cap_level, bone, span=1):
    """Seat a rigid attachment cap on bone and blend its translation backward.

    The terminal cross-section represents an embedded attachment interface,
    not a free collision surface. Keeping its footprint rigid avoids angular
    column crossing and tet collapse; allowing its edges to enter the curved
    bone avoids the artificial gap caused by forcing the whole planar cap out.
    """
    result = x.copy()
    cap_idx = np.sort(np.where(levels == cap_level)[0])
    pts = result[cap_idx]
    center = pts.mean(axis=0)
    center_cp, _, _ = trimesh.proximity.closest_point(bone, center[None])
    toward_surface = center_cp[0] - center
    toward = toward_surface / (np.linalg.norm(toward_surface) + 1e-12)
    # A planar footprint cannot coincide pointwise with a curved tibia. Search
    # only one rigid seating DOF and choose the depth with the smallest mean
    # surface gap; the attachment may straddle the bone boundary by design.
    centered_footprint = pts + toward_surface
    best_shift = toward_surface
    best_gap = np.inf
    for depth in np.linspace(-0.004, 0.004, 81):
        candidate = centered_footprint + depth * toward
        _, distances, _ = trimesh.proximity.closest_point(bone, candidate)
        gap = float(np.mean(distances))
        if gap < best_gap:
            best_gap = gap
            best_shift = toward_surface + depth * toward
    toward_surface = best_shift
    cap_displacement = np.repeat(
        toward_surface[None, :], len(cap_idx), axis=0)

    first_level = max(int(levels[levels >= 0].min()), cap_level - span)
    denom = max(cap_level - first_level, 1)
    for level in range(first_level, cap_level + 1):
        ring = np.sort(np.where(levels == level)[0])
        if len(ring) != len(cap_idx):
            continue
        u = (level - first_level) / denom
        weight = u * u * (3.0 - 2.0 * u)
        result[ring] += weight * cap_displacement

    # Eliminate accumulated floating-point error at the actual attachment.
    result[cap_idx] = pts + toward_surface
    return result


def conform_cap_with_taper(x, levels, cap_level, bone, span=4):
    """Conform a broad attachment footprint to bone, blending vertexwise."""
    result = x.copy()
    cap_idx = np.sort(np.where(levels == cap_level)[0])
    closest, _, face_ids = trimesh.proximity.closest_point(bone, result[cap_idx])
    target = closest + 0.00005 * bone.face_normals[face_ids]
    displacement = target - result[cap_idx]
    first_level = max(int(levels[levels >= 0].min()), cap_level - span)
    denom = max(cap_level - first_level, 1)
    for level in range(first_level, cap_level + 1):
        ring = np.sort(np.where(levels == level)[0])
        if len(ring) != len(cap_idx):
            continue
        weight = (level - first_level) / denom
        weight = weight * weight * (3.0 - 2.0 * weight)
        result[ring] += weight * displacement
    result[cap_idx] = target
    return result


def posed_tibia(frame, motion_path='data/motion/run.bvh'):
    info, root, bvh_info, _, _, _ = saveSkeletonInfo('data/zygote_skel.xml')
    skel = buildFromInfo(info, root)
    skel.setPositions(np.zeros(skel.getNumDofs()))
    body = skel.getBodyNode('L_Tibia_Fibula0')
    wt = body.getWorldTransform()
    r0, t0 = wt.rotation().copy(), wt.translation().copy()

    mesh = trimesh.load(
        'Zygote_Meshes_251229/Skeleton/L_Tibia_Fibula.obj', process=False)
    rest = np.asarray(mesh.vertices, dtype=np.float64) * 0.01
    motion = MyBVH(motion_path, bvh_info, skel, T_frame=None)
    skel.setPositions(motion.mocap_refs[frame])
    wt = body.getWorldTransform()
    local = (r0.T @ (rest - t0).T).T
    posed = (wt.rotation() @ local.T).T + wt.translation()
    return trimesh.Trimesh(posed, mesh.faces, process=True)


def signed_tet_ratios(rest, posed, tets):
    def volumes(x):
        q = x[tets]
        return np.einsum('ij,ij->i', q[:, 1] - q[:, 0],
                         np.cross(q[:, 2] - q[:, 0],
                                  q[:, 3] - q[:, 0])) / 6.0
    v0 = volumes(rest)
    good = np.abs(v0) > 1e-14
    return volumes(posed)[good] / v0[good]


def repair(name, bone, frame, source=SOURCE, output=OUTPUT):
    source_path = os.path.join(source, f'{name}_chunk_0000.npz')
    cache = np.load(source_path, allow_pickle=True)
    frames = [int(f) for f in cache['frames']]
    frame_idx = frames.index(frame)
    x = np.asarray(cache['positions'][frame_idx], dtype=np.float64).copy()
    source_x = x.copy()

    with open(os.path.join('tet', f'{name}_tet.npz'), 'rb') as f:
        tet = pickle.load(f)
    rest = np.asarray(tet['vertices'], dtype=np.float64)
    tets = np.asarray(tet['tetrahedra'], dtype=np.int64)
    levels = np.asarray(tet.get('vertex_contour_level'), dtype=np.int32)
    if levels.shape != (len(x),):
        raise RuntimeError(f'{name}: missing vertex_contour_level')

    x = untwist_body(rest, x, levels)
    inside_before = bone.contains(x)
    ratio_before = signed_tet_ratios(rest, x, tets)
    max_level = int(levels.max())
    insertion_cap = levels == max_level
    band_start = max(0, max_level - 10)
    distal = levels >= band_start

    # Estimate a single anatomically coherent outward displacement from all
    # penetrated distal samples. Translation of whole rings preserves their
    # cross-section and is much less prone to a visible hook than cap pushes.
    distal_inside = np.where(distal & inside_before)[0]
    if len(distal_inside):
        cp, dist, _ = trimesh.proximity.closest_point(bone, x[distal_inside])
        away = cp - x[distal_inside]
        n = np.linalg.norm(away, axis=1, keepdims=True)
        unit = away / np.maximum(n, 1e-12)
        corrections = away + unit * MARGIN
        # Robust central correction: reject extreme closest-point directions.
        direction = np.median(unit, axis=0)
        direction /= np.linalg.norm(direction) + 1e-12
        required = np.max(np.einsum('ij,j->i', corrections, direction))
        required = max(required, MARGIN)
        band_correction = direction * required
    else:
        band_correction = np.zeros(3)

    u = np.clip((levels - band_start) / max(max_level - band_start, 1), 0.0, 1.0)
    taper = u * u * (3.0 - 2.0 * u)
    x += taper[:, None] * band_correction[None, :]

    # Exact cleanup. Apply the residual to the containing contour ring rather
    # than only one vertex, retaining a coherent cross-section.
    for _ in range(4):
        inside = bone.contains(x) & distal
        if not np.any(inside):
            break
        for level in np.unique(levels[inside]):
            bad = np.where(inside & (levels == level))[0]
            ring = np.where(levels == level)[0]
            cp, _, _ = trimesh.proximity.closest_point(bone, x[bad])
            away = cp - x[bad]
            length = np.linalg.norm(away, axis=1, keepdims=True)
            unit = away / np.maximum(length, 1e-12)
            corr = np.median(away + unit * MARGIN, axis=0)
            x[ring] += corr

    inv_before = int(np.sum(ratio_before <= 0))
    # Backtrack the complete repair displacement if it creates additional tet
    # inversions. Usually the full correction is unnecessary because the
    # safety margin deliberately overshoots the surface.
    repaired = x.copy()
    accepted = None
    rejected = []
    for scale in np.linspace(1.0, 0.1, 19):
        trial = source_x + scale * (repaired - source_x)
        # The insertion footprint is an attachment, not a collision-margin
        # surface. Snap its complete terminal contour onto the tibia with only
        # a numerical 0.05 mm exterior offset. The preceding bands retain the
        # smooth no-twist/nonpenetrating routing.
        if name == TFL:
            trial = conform_cap_with_taper(trial, levels, max_level, bone)
        else:
            trial = attach_cap_with_taper(trial, levels, max_level, bone)
        trial_inside = bone.contains(trial)
        trial_ratio = signed_tet_ratios(rest, trial, tets)
        rejected.append((float(scale),
                         int(np.sum(trial_inside & distal & ~insertion_cap)),
                         int(np.sum(trial_inside & insertion_cap)),
                         int(np.sum(trial_ratio <= 0))))
        # The legacy source already contains inverted tets. This is a viewer
        # repair, so permit at most five additional inversions while requiring
        # exact distal nonpenetration; never present it as training-quality FEM.
        # The cap is deliberately embedded in the bone. All non-attachment
        # distal contours remain subject to strict collision exclusion.
        free_distal_inside = trial_inside & distal & ~insertion_cap
        if (not np.any(free_distal_inside)
                and int(np.sum(trial_ratio <= 0)) <= inv_before + 5):
            accepted = (trial, trial_inside, trial_ratio, scale)
            break
    if accepted is None:
        raise RuntimeError(
            f'{name} frame {frame}: no penetration-free, inversion-safe '
            f'line-search step; '
            f'candidates={rejected}')
    x, inside_after, ratio_after, accepted_scale = accepted
    inv_after = int(np.sum(ratio_after <= 0))
    distal_after = int(np.sum(inside_after & distal & ~insertion_cap))
    if distal_after:
        raise RuntimeError(
            f'{name} frame {frame}: {distal_after} distal vertices remain inside tibia')
    if inv_after > inv_before + 5:
        raise RuntimeError(
            f'{name} frame {frame}: inversions increased {inv_before} -> {inv_after}')

    os.makedirs(output, exist_ok=True)
    output_path = os.path.join(output, f'{name}_chunk_0000.npz')
    output_frames = {frame: x.astype(np.float32)}
    if os.path.exists(output_path):
        old = np.load(output_path, allow_pickle=True)
        output_frames.update({int(f): np.asarray(p, dtype=np.float32)
                              for f, p in zip(old['frames'], old['positions'])
                              if 0 <= int(f) <= 4})
        output_frames[frame] = x.astype(np.float32)
    ordered_frames = sorted(output_frames)
    np.savez_compressed(
        output_path,
        frames=np.asarray(ordered_frames, dtype=np.int32),
        positions=np.stack([output_frames[f] for f in ordered_frames]),
    )
    cap_idx = np.where(insertion_cap)[0]
    _, cap_distance, _ = trimesh.proximity.closest_point(bone, x[cap_idx])
    print(f'{name} frame {frame}: '
          f'inside {int(inside_before.sum())} -> {int(inside_after.sum())}, '
          f'distal -> {distal_after}, inversions {inv_before} -> {inv_after}, '
          f'band shift={np.linalg.norm(band_correction)*accepted_scale*1000:.2f}mm, '
          f'cap distance mean/max={cap_distance.mean()*1000:.3f}/'
          f'{cap_distance.max()*1000:.3f}mm')


def repair_tfl(name, frame, source=SOURCE, output=OUTPUT):
    """Remove the TFL belly's skinning corkscrew while retaining both caps."""
    source_path = os.path.join(source, f'{name}_chunk_0000.npz')
    cache = np.load(source_path, allow_pickle=True)
    source_frames = [int(f) for f in cache['frames']]
    source_x = np.asarray(cache['positions'][source_frames.index(frame)],
                          dtype=np.float64)
    with open(os.path.join('tet', f'{name}_tet.npz'), 'rb') as f:
        tet = pickle.load(f)
    rest = np.asarray(tet['vertices'], dtype=np.float64)
    tets = np.asarray(tet['tetrahedra'], dtype=np.int64)
    levels = np.asarray(tet['vertex_contour_level'], dtype=np.int32)
    repaired = untwist_body(rest, source_x, levels)
    # The skeleton constraint specifies every cap vertex, not just its center.
    # Restore the complete insertion footprint exactly and distribute its
    # orientation residual smoothly through the distal rings.
    max_level = int(levels.max())
    cap = np.sort(np.where(levels == max_level)[0])
    cap_residual = source_x[cap] - repaired[cap]
    first_level = max(int(levels[levels >= 0].min()), max_level - 8)
    denom = max(max_level - first_level, 1)
    for level in range(first_level, max_level + 1):
        ring = np.sort(np.where(levels == level)[0])
        if len(ring) != len(cap):
            continue
        weight = (level - first_level) / denom
        weight = weight * weight * (3.0 - 2.0 * weight)
        repaired[ring] += weight * cap_residual
    repaired[cap] = source_x[cap]
    before = int(np.sum(signed_tet_ratios(rest, source_x, tets) <= 0))
    after = int(np.sum(signed_tet_ratios(rest, repaired, tets) <= 0))
    if after > before:
        raise RuntimeError(f'{name} frame {frame}: untwist inversions '
                           f'increased {before} -> {after}')

    os.makedirs(output, exist_ok=True)
    output_path = os.path.join(output, f'{name}_chunk_0000.npz')
    output_frames = {}
    if os.path.exists(output_path):
        old = np.load(output_path, allow_pickle=True)
        output_frames.update({int(f): np.asarray(p, dtype=np.float32)
                              for f, p in zip(old['frames'], old['positions'])
                              if 0 <= int(f) <= 4})
    output_frames[frame] = repaired.astype(np.float32)
    ordered = sorted(output_frames)
    np.savez_compressed(
        output_path, frames=np.asarray(ordered, dtype=np.int32),
        positions=np.stack([output_frames[f] for f in ordered]))
    displacement = np.linalg.norm(repaired - source_x, axis=1)
    print(f'{name} frame {frame}: untwisted {int(np.sum(displacement > 1e-7))} '
          f'verts, max={displacement.max()*1000:.2f}mm, '
          f'inversions {before} -> {after}')


def repair_frames(source=SOURCE, output=OUTPUT,
                  motion_path='data/motion/run.bvh', frames=range(5)):
    frames = tuple(int(frame) for frame in frames)
    for frame in frames:
        bone = posed_tibia(frame, motion_path)
        for name in MUSCLES:
            repair(name, bone, frame, source=source, output=output)
    print(f'Wrote medial-tibia repair for frames {list(frames)}: {output}')


def main():
    repair_frames()


if __name__ == '__main__':
    main()
