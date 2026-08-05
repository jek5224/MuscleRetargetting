#!/usr/bin/env python3
"""Minimal stiff tetrahedral-muscle PBD baker.

The solver intentionally contains only:
  * hard BVH-driven attachment constraints,
  * rest-length constraints on every tet edge,
  * signed rest-volume constraints on every tet,
  * local projection out of posed bone meshes.

There is no gravity, inertia, activation, fiber model, or muscle contact.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import trimesh
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.bvhparser import MyBVH
from tools import bake_emu
from tools.bake_surface_fast import build_bones, load_tet, surface_faces
import test_emu


def unique_edges(tets):
    edges = np.vstack([
        tets[:, [i, j]] for i, j in itertools.combinations(range(4), 2)
    ])
    return np.unique(np.sort(edges, axis=1), axis=0).astype(np.int32)


def signed_volumes(x, tets):
    a, b, c, d = (x[tets[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def project_edges(x, edges, rest_lengths, inverse_mass, stiffness):
    i, j = edges[:, 0], edges[:, 1]
    delta = x[j] - x[i]
    length = np.linalg.norm(delta, axis=1)
    valid = length > 1e-12
    direction = np.zeros_like(delta)
    direction[valid] = delta[valid] / length[valid, None]
    error = length - rest_lengths
    wi, wj = inverse_mass[i], inverse_mass[j]
    denom = wi + wj
    valid &= denom > 0
    correction = np.zeros_like(delta)
    correction[valid] = (
        stiffness * error[valid, None] * direction[valid]
        / denom[valid, None]
    )
    accumulated = np.zeros_like(x)
    counts = np.zeros(len(x))
    np.add.at(accumulated, i, wi[:, None] * correction)
    np.add.at(accumulated, j, -wj[:, None] * correction)
    np.add.at(counts, i, wi)
    np.add.at(counts, j, wj)
    movable = counts > 0
    x[movable] += accumulated[movable] / counts[movable, None]


def project_volumes(x, tets, rest_volumes, inverse_mass, stiffness):
    a, b, c, d = (x[tets[:, i]] for i in range(4))
    grads = (
        np.cross(b - d, c - d) / 6.0,
        np.cross(c - d, a - d) / 6.0,
        np.cross(a - d, b - d) / 6.0,
        np.cross(b - a, c - a) / 6.0,
    )
    current = signed_volumes(x, tets)
    # A small positive floor prevents an inverted tet from being accepted as
    # an equally valid negative-volume solution.
    target = np.sign(rest_volumes) * np.maximum(
        np.abs(rest_volumes), 1e-12)
    weights = [inverse_mass[tets[:, k]] for k in range(4)]
    denom = sum(w * np.einsum("ij,ij->i", g, g)
                for w, g in zip(weights, grads))
    valid = denom > 1e-18
    lam = np.zeros(len(tets))
    lam[valid] = stiffness * (current[valid] - target[valid]) / denom[valid]
    lam = np.clip(lam, -0.01, 0.01)
    accumulated = np.zeros_like(x)
    counts = np.zeros(len(x))
    for k, (w, grad) in enumerate(zip(weights, grads)):
        np.add.at(
            accumulated, tets[:, k],
            -w[:, None] * lam[:, None] * grad)
        np.add.at(counts, tets[:, k], w)
    movable = counts > 0
    x[movable] += accumulated[movable] / counts[movable, None]


def project_edges_cuda(x, edges, rest_lengths, inverse_mass, stiffness):
    i, j = edges[:, 0], edges[:, 1]
    delta = x[j] - x[i]
    length = torch.linalg.vector_norm(delta, dim=1)
    direction = delta / length.clamp_min(1e-12)[:, None]
    wi, wj = inverse_mass[i], inverse_mass[j]
    denom = wi + wj
    correction = (
        stiffness * (length - rest_lengths)[:, None] * direction
        / denom.clamp_min(1e-12)[:, None])
    accumulated = torch.zeros_like(x)
    counts = torch.zeros(len(x), dtype=x.dtype, device=x.device)
    accumulated.index_add_(0, i, wi[:, None] * correction)
    accumulated.index_add_(0, j, -wj[:, None] * correction)
    counts.index_add_(0, i, wi)
    counts.index_add_(0, j, wj)
    movable = counts > 0
    x[movable] += accumulated[movable] / counts[movable, None]


def signed_volumes_cuda(x, tets):
    a, b, c, d = (x[tets[:, i]] for i in range(4))
    return torch.sum(torch.linalg.cross(b - a, c - a) * (d - a), dim=1) / 6.0


def project_volumes_cuda(
        x, tets, rest_volumes, inverse_mass, stiffness):
    a, b, c, d = (x[tets[:, i]] for i in range(4))
    grads = (
        torch.linalg.cross(b - d, c - d) / 6.0,
        torch.linalg.cross(c - d, a - d) / 6.0,
        torch.linalg.cross(a - d, b - d) / 6.0,
        torch.linalg.cross(b - a, c - a) / 6.0,
    )
    current = signed_volumes_cuda(x, tets)
    target = torch.sign(rest_volumes) * torch.abs(rest_volumes).clamp_min(1e-12)
    weights = [inverse_mass[tets[:, k]] for k in range(4)]
    denom = sum(
        w * torch.sum(g * g, dim=1) for w, g in zip(weights, grads))
    lam = stiffness * (current - target) / denom.clamp_min(1e-18)
    lam.clamp_(-0.01, 0.01)
    accumulated = torch.zeros_like(x)
    counts = torch.zeros(len(x), dtype=x.dtype, device=x.device)
    for k, (w, grad) in enumerate(zip(weights, grads)):
        accumulated.index_add_(
            0, tets[:, k], -w[:, None] * lam[:, None] * grad)
        counts.index_add_(0, tets[:, k], w)
    movable = counts > 0
    x[movable] += accumulated[movable] / counts[movable, None]


def build_surface_samples(faces):
    """Return vertex ids and barycentric weights for surface collision samples.

    Testing only mesh vertices misses an edge or face that crosses a bone
    while all its endpoints remain outside.  Include edge quarter-points and
    face centroids so those crossings produce positional constraints too.
    """
    faces = np.asarray(faces, dtype=np.int32)
    samples_i = []
    samples_w = []
    for face in faces:
        for k in range(3):
            w = np.zeros(3)
            w[k] = 1.0
            samples_i.append(face)
            samples_w.append(w)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            for fraction in (0.25, 0.5, 0.75):
                w = np.zeros(3)
                w[a] = 1.0 - fraction
                w[b] = fraction
                samples_i.append(face)
                samples_w.append(w)
        samples_i.append(face)
        samples_w.append(np.full(3, 1.0 / 3.0))
    return (np.asarray(samples_i, dtype=np.int32),
            np.asarray(samples_w, dtype=np.float64))


def surface_sample_positions(x, sample_vertices, sample_weights):
    return np.einsum(
        "nij,ni->nj", x[sample_vertices], sample_weights)


def project_bones(x, sample_vertices, sample_weights, bones, fixed_mask,
                  margin, max_step):
    """Push penetrating vertex/edge/face samples out of posed bones."""
    # Attachment caps are hard boundary conditions and are intentionally on
    # the bone surface (occasionally slightly embedded in the source asset).
    # Neither an attachment vertex nor an edge/face supported by one may
    # become a collision constraint; doing so transfers collision correction
    # into the first free ring and peels/distorts the cap.
    active_support = sample_weights > 0.0
    attachment_supported = np.all(
        fixed_mask[sample_vertices] | ~active_support, axis=1)
    for bone in bones:
        points = surface_sample_positions(
            x, sample_vertices, sample_weights)
        candidates = np.where(np.all(
            (points >= bone.bounds[0] - margin)
            & (points <= bone.bounds[1] + margin), axis=1)
            & ~attachment_supported)[0]
        if not len(candidates):
            continue
        try:
            inside = bone.contains(points[candidates])
        except Exception:
            continue
        sample_ids = candidates[inside]
        if not len(sample_ids):
            continue
        nearest, _, face_ids = trimesh.proximity.closest_point(
            bone, points[sample_ids])
        delta = nearest - points[sample_ids]
        length = np.linalg.norm(delta, axis=1)
        direction = np.zeros_like(delta)
        valid = length > 1e-12
        direction[valid] = delta[valid] / length[valid, None]
        direction[~valid] = bone.face_normals[face_ids[~valid]]
        correction = nearest + margin * direction - points[sample_ids]
        correction_length = np.linalg.norm(correction, axis=1)
        scale = np.minimum(
            1.0, max_step / np.maximum(correction_length, 1e-12))
        correction *= scale[:, None]

        # A sample correction is a constraint on its supporting primitive.
        # Scatter it to free vertices in proportion to barycentric influence,
        # averaging competing constraints to avoid collision-induced spikes.
        accumulated = np.zeros_like(x)
        counts = np.zeros(len(x))
        for local in range(3):
            vertex_ids = sample_vertices[sample_ids, local]
            weights = sample_weights[sample_ids, local]
            movable = ~fixed_mask[vertex_ids]
            np.add.at(
                accumulated, vertex_ids[movable],
                weights[movable, None] * correction[movable])
            np.add.at(counts, vertex_ids[movable], weights[movable])
        movable = counts > 0
        update = np.zeros_like(x)
        update[movable] = (
            accumulated[movable] / counts[movable, None])
        update_length = np.linalg.norm(update, axis=1)
        update_scale = np.minimum(
            1.0, max_step / np.maximum(update_length, 1e-12))
        x[movable] += update[movable] * update_scale[movable, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh", required=True, type=Path)
    ap.add_argument("--tet", default="tet/L_Vastus_Intermedius_tet.npz",
                    type=Path)
    ap.add_argument("--name", default="L_Vastus_Intermedius")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--edge-stiffness", type=float, default=0.95)
    ap.add_argument("--volume-stiffness", type=float, default=0.25)
    ap.add_argument("--collision-every", type=int, default=100)
    ap.add_argument("--collision-margin", type=float, default=0.0015)
    ap.add_argument("--max-collision-step", type=float, default=0.002)
    ap.add_argument("--final-collision-passes", type=int, default=6)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()

    skel, bvh_info, _ = bake_emu.load_skeleton()
    trees = test_emu._load_bone_trees()
    data = load_tet(args.tet)
    group = test_emu.prepare_group_data(
        data, args.name, skel, trees, source_path=args.tet,
        attachment_rings=0)
    contour_level = data.get("vertex_contour_level")
    if contour_level is not None:
        contour_level = np.asarray(contour_level, dtype=np.int32)
        if contour_level.shape == (len(group["vertices"]),):
            origin_fixed = np.where(
                contour_level == contour_level.min())[0].astype(np.int32)
            insertion_fixed = np.where(
                contour_level == contour_level.max())[0].astype(np.int32)
            if len(origin_fixed) and len(insertion_fixed):
                group["origin_fixed"] = origin_fixed.tolist()
                group["insertion_fixed"] = insertion_fixed.tolist()
                group["fixed_vertices"] = np.concatenate(
                    (origin_fixed, insertion_fixed)).tolist()
                # Rebuild the material coordinate and skeletal bindings from
                # the complete cap levels, not the endpoint-sampling fallback.
                u = test_emu._harmonic_axis_coordinate(
                    group["vertices"], group["tetrahedra"],
                    origin_fixed, insertion_fixed)
                u[origin_fixed] = 0.0
                u[insertion_fixed] = 1.0
                origin_body = skel.getBodyNode(group["origin_body"])
                insertion_body = skel.getBodyNode(group["insertion_body"])
                group["axis_coordinate"] = u
                group["lbs_bindings"] = test_emu._make_lbs_bindings(
                    group["vertices"], u, origin_body, insertion_body)
                print(
                    f"    Full contour caps: {len(origin_fixed)} origin + "
                    f"{len(insertion_fixed)} insertion hard vertices")
    rest = np.asarray(group["vertices"], dtype=np.float64)
    tets = np.asarray(group["tetrahedra"], dtype=np.int32)
    edges = unique_edges(tets)
    rest_lengths = np.linalg.norm(
        rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    rest_volumes = signed_volumes(rest, tets)
    fixed = np.asarray(group["fixed_vertices"], dtype=np.int32)
    fixed_mask = np.zeros(len(rest), dtype=bool)
    fixed_mask[fixed] = True
    inverse_mass = (~fixed_mask).astype(np.float64)
    faces = surface_faces(tets)
    sample_vertices, sample_weights = build_surface_samples(faces)

    motion = MyBVH(
        str(args.bvh), bvh_info, skel,
        T_frame=bake_emu._detect_bvh_tframe(str(args.bvh)))
    positions = []
    previous = rest.copy()
    previous_target = rest.copy()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    dtype = torch.float64
    edges_gpu = torch.as_tensor(edges, dtype=torch.long, device=device)
    tets_gpu = torch.as_tensor(tets, dtype=torch.long, device=device)
    rest_lengths_gpu = torch.as_tensor(
        rest_lengths, dtype=dtype, device=device)
    rest_volumes_gpu = torch.as_tensor(
        rest_volumes, dtype=dtype, device=device)
    inverse_mass_gpu = torch.as_tensor(
        inverse_mass, dtype=dtype, device=device)
    fixed_gpu = torch.as_tensor(fixed, dtype=torch.long, device=device)
    print(f"PBD device: {device}")
    for frame, pose in enumerate(motion.mocap_refs):
        skel.setPositions(pose.copy())
        target = bake_emu.compute_rigid_blend_positions(
            group["lbs_bindings"], skel, group["axis_coordinate"])
        # Transport every vertex with its skeleton guide. This avoids leaving
        # the free interior at the previous pose while the caps move away.
        x = torch.as_tensor(
            previous + target - previous_target,
            dtype=dtype, device=device).clone()
        target_gpu = torch.as_tensor(target, dtype=dtype, device=device)
        x[fixed_gpu] = target_gpu[fixed_gpu]
        bones = build_bones(skel)
        with torch.no_grad():
            for iteration in range(args.iterations):
                project_edges_cuda(
                    x, edges_gpu, rest_lengths_gpu, inverse_mass_gpu,
                    args.edge_stiffness)
                project_volumes_cuda(
                    x, tets_gpu, rest_volumes_gpu, inverse_mass_gpu,
                    args.volume_stiffness)
                x[fixed_gpu] = target_gpu[fixed_gpu]
                if (iteration + 1) % args.collision_every == 0:
                    x_cpu = x.cpu().numpy()
                    project_bones(
                        x_cpu, sample_vertices, sample_weights, bones,
                        fixed_mask, args.collision_margin,
                        args.max_collision_step)
                    x = torch.as_tensor(
                        x_cpu, dtype=dtype, device=device)
                    x[fixed_gpu] = target_gpu[fixed_gpu]
                if not torch.isfinite(x).all():
                    raise FloatingPointError(
                        f"non-finite PBD state at frame {frame}, "
                        f"iteration {iteration + 1}")
        x = x.cpu().numpy()
        # Two final face-aware collision passes catch constraints introduced
        # by the last stiffness projection without paying for dense CPU
        # containment queries on every GPU iteration.
        for _ in range(args.final_collision_passes):
            project_bones(
                x, sample_vertices, sample_weights, bones, fixed_mask,
                args.collision_margin, args.max_collision_step)
            x[fixed] = target[fixed]
        positions.append(x.astype(np.float32))
        previous = x
        previous_target = target
        v = signed_volumes(x, tets) * np.sign(rest_volumes)
        print(
            f"frame {frame}: inverted={(v <= 0).sum()}/{len(tets)}, "
            f"volume_ratio={np.abs(v).sum()/np.abs(rest_volumes).sum():.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.name}_chunk_0000.npz"
    np.savez_compressed(
        out, frames=np.arange(len(positions), dtype=np.int32),
        positions=np.stack(positions))
    (args.output_dir / ".done").touch()
    print(f"Saved {len(positions)} frames to {out}")


if __name__ == "__main__":
    main()
