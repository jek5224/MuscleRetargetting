#!/usr/bin/env python3
"""Cache-free local-global ARAP muscle baker with femur-SDF contact."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from core.bvhparser import MyBVH
import test_emu
from tools import bake_emu
from tools.bake_muscle_arap_sdf import dense_boundary_samples, relative_transform
from tools.bake_stiff_tet_arap import (
    BoneSDF, CUDAARAP, project_collision_constraints)
from tools.bake_stiff_tet_pbd import (
    project_volumes_cuda, signed_volumes, signed_volumes_cuda, unique_edges)
from tools.bake_surface_fast import load_tet, surface_faces


class MatrixFreeARAP:
    """GPU local-global ARAP with a Jacobi-PCG global step.

    This avoids the dense 10k-by-10k Laplacian and Cholesky factors used by
    the legacy CUDAARAP implementation. Only edge arrays and vertex vectors
    are resident, so memory is linear in mesh size.
    """
    def __init__(self, rest, edges, device, dtype=torch.float32):
        self.rest = torch.as_tensor(rest, dtype=dtype, device=device)
        edge = torch.as_tensor(edges, dtype=torch.long, device=device)
        self.i = torch.cat((edge[:, 0], edge[:, 1]))
        self.j = torch.cat((edge[:, 1], edge[:, 0]))
        self.rest_edge = self.rest[self.i] - self.rest[self.j]
        self.weight = 1.0 / torch.linalg.vector_norm(
            self.rest_edge, dim=1).clamp_min(1e-5)
        self.degree = torch.zeros(
            len(rest), dtype=dtype, device=device)
        self.degree.index_add_(0, self.i, self.weight)

    def laplacian(self, value):
        out = torch.zeros_like(value)
        out.index_add_(
            0, self.i, self.weight[:, None]
            * (value[self.i] - value[self.j]))
        return out

    @staticmethod
    def sample_transpose(value, ids, weights, vertex_count):
        out = torch.zeros(
            (vertex_count, 3), dtype=value.dtype, device=value.device)
        for corner in range(3):
            out.index_add_(
                0, ids[:, corner], weights[:, corner, None] * value)
        return out

    def step(self, x, anchors, arap_weight, collision=None,
             collision_weight=300.0, pcg_iterations=24):
        current = x[self.i] - x[self.j]
        covariance = torch.zeros(
            (len(x), 3, 3), dtype=x.dtype, device=x.device)
        covariance.index_add_(
            0, self.i, self.weight[:, None, None]
            * (current[:, :, None] * self.rest_edge[:, None, :]))
        u, _, vh = torch.linalg.svd(covariance)
        rotation = u @ vh
        negative = torch.linalg.det(rotation) < 0
        if torch.any(negative):
            u = u.clone()
            u[negative, :, -1] *= -1
            rotation = u @ vh
        wanted = 0.5 * (rotation[self.i] + rotation[self.j])
        wanted = (wanted @ self.rest_edge[:, :, None])[:, :, 0]
        rhs = torch.zeros_like(x)
        rhs.index_add_(0, self.i, self.weight[:, None] * wanted)
        rhs *= arap_weight
        diagonal = arap_weight * self.degree + 1e-6

        anchor_ids, anchor_targets, anchor_weights = anchors
        anchor_diagonal = torch.zeros_like(diagonal)
        anchor_rhs = torch.zeros_like(rhs)
        anchor_diagonal.index_add_(0, anchor_ids, anchor_weights)
        anchor_rhs.index_add_(
            0, anchor_ids, anchor_weights[:, None] * anchor_targets)
        diagonal = diagonal + anchor_diagonal
        rhs = rhs + anchor_rhs

        collision_data = None
        if collision is not None and len(collision[0]):
            ids, weights, targets = collision
            collision_data = ids, weights
            rhs += collision_weight * self.sample_transpose(
                targets, ids, weights, len(x))
            for corner in range(3):
                diagonal.index_add_(
                    0, ids[:, corner],
                    collision_weight * weights[:, corner] ** 2)

        def operator(value):
            result = arap_weight * self.laplacian(value) + 1e-6 * value
            result = result + anchor_diagonal[:, None] * value
            if collision_data is not None:
                ids, weights = collision_data
                sampled = torch.sum(
                    value[ids] * weights[:, :, None], dim=1)
                result += collision_weight * self.sample_transpose(
                    sampled, ids, weights, len(x))
            return result

        result = x.clone()
        residual = rhs - operator(result)
        direction = residual / diagonal[:, None]
        z = direction.clone()
        rz = torch.sum(residual * z)
        for _ in range(pcg_iterations):
            ad = operator(direction)
            alpha = rz / torch.sum(direction * ad).clamp_min(1e-20)
            result += alpha * direction
            residual -= alpha * ad
            new_z = residual / diagonal[:, None]
            new_rz = torch.sum(residual * new_z)
            if float(new_rz) < 1e-14:
                break
            direction = new_z + (new_rz / rz) * direction
            z, rz = new_z, new_rz
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="L_Vastus_Intermedius")
    ap.add_argument("--tet", type=Path)
    ap.add_argument("--bvh", type=Path, default=Path(
        "data/motion/left_thigh_quasistatic_diverse_smooth_76frame.bvh"))
    ap.add_argument("--sdf", type=Path, default=Path(
        ".bake_outputs/collision_sdf/L_Femur0_sdf.npz"))
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--end-frame", type=int)
    ap.add_argument("--substeps", type=int, default=6)
    ap.add_argument("--iterations", type=int, default=35)
    ap.add_argument("--maximum-iterations", type=int, default=140)
    ap.add_argument("--arap-weight", type=float, default=0.25)
    ap.add_argument("--insertion-weight", type=float, default=10000.0)
    ap.add_argument("--contact-weight", type=float, default=300.0)
    ap.add_argument("--quality-weight", type=float, default=1000.0)
    ap.add_argument("--collision-margin", type=float, default=0.0015)
    ap.add_argument("--collision-tolerance", type=float, default=0.00025)
    ap.add_argument("--attachment-tolerance", type=float, default=0.001)
    ap.add_argument("--volume-stiffness", type=float, default=0.15)
    ap.add_argument("--volume-passes", type=int, default=2)
    ap.add_argument("--final-contact-passes", type=int, default=100)
    args = ap.parse_args()

    if args.tet is None:
        args.tet = Path("tet") / f"{args.name}_tet.npz"
    if args.output_dir is None:
        args.output_dir = (
            Path(".bake_outputs/motion_cache") / args.bvh.stem
            / f"{args.name}_cacheless_local_global_arap_sdf_v1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    device, dtype = torch.device(args.device), torch.float32
    skel, info, _ = bake_emu.load_skeleton()
    skel.setPositions(np.zeros(skel.getNumDofs()))
    group = test_emu.prepare_group_data(
        load_tet(args.tet), args.name, skel, test_emu._load_bone_trees(),
        source_path=args.tet, attachment_rings=0)
    rest = np.asarray(group["vertices"], np.float64)
    tets = np.asarray(group["tetrahedra"], np.int32)
    origin = np.asarray(group["origin_fixed"], np.int32)
    insertion = np.asarray(group["insertion_fixed"], np.int32)
    attachment_mask = np.zeros(len(rest), bool)
    attachment_mask[np.r_[origin, insertion]] = True

    samples_i, samples_w = dense_boundary_samples(surface_faces(tets))
    active = samples_w > 1e-12
    pure_attachment = np.all(
        ~active | attachment_mask[samples_i], axis=1)
    samples_i, samples_w = (
        samples_i[~pure_attachment], samples_w[~pure_attachment])
    no_exclusion = np.zeros(len(rest), bool)
    sdf = BoneSDF(args.sdf)
    sdf.bind_rest_clearance(
        rest, samples_i, samples_w, skel,
        prevent_new_only=False, margin=args.collision_margin)
    valid = np.isfinite(sdf.rest_distance)
    sdf.rest_distance[valid & (sdf.rest_distance >= 0)] = args.collision_margin

    # The femoral origin removes rigid modes exactly. The patellar insertion
    # remains a strong soft target inside the global system.
    solver = MatrixFreeARAP(
        rest, unique_edges(tets), device, dtype=dtype)
    origin_t = torch.as_tensor(origin, dtype=torch.long, device=device)
    insertion_t = torch.as_tensor(
        insertion, dtype=torch.long, device=device)
    soft_weight = torch.full(
        (len(insertion),), args.insertion_weight,
        dtype=dtype, device=device)
    tet_t = torch.as_tensor(tets, dtype=torch.long, device=device)
    rest_volume = torch.as_tensor(
        signed_volumes(rest, tets), dtype=dtype, device=device)
    rest_sign = torch.sign(rest_volume)
    rest_abs = torch.abs(rest_volume).clamp_min(1e-14)
    inverse_mass = torch.ones(len(rest), dtype=dtype, device=device)
    inverse_mass[origin_t] = 0.0

    motion = MyBVH(
        str(args.bvh), info, skel,
        T_frame=bake_emu._detect_bvh_tframe(str(args.bvh)))
    poses = motion.mocap_refs
    if args.end_frame is not None:
        poses = poses[:args.end_frame + 1]
    origin_body = skel.getBodyNode(group["origin_body"])
    rest_tf = origin_body.getWorldTransform()
    rest_R = np.asarray(rest_tf.rotation()).copy()
    rest_t = np.asarray(rest_tf.translation()).copy()
    previous_R, previous_t = np.eye(3), np.zeros(3)
    previous_pose = np.zeros(skel.getNumDofs())
    x = torch.as_tensor(rest, dtype=dtype, device=device).clone()
    output, metrics = [], []

    for frame, frame_pose in enumerate(poses):
        for substep in range(1, args.substeps + 1):
            alpha = substep / args.substeps
            skel.setPositions(
                (1.0 - alpha) * previous_pose + alpha * frame_pose)
            R, t = relative_transform(origin_body, rest_R, rest_t)
            dR, dt = R @ previous_R.T, t - (R @ previous_R.T) @ previous_t
            x = (
                x @ torch.as_tensor(dR, dtype=dtype, device=device).T
                + torch.as_tensor(dt, dtype=dtype, device=device))
            previous_R, previous_t = R.copy(), t.copy()
            guide = bake_emu.compute_rigid_blend_positions(
                group["lbs_bindings"], skel, group["axis_coordinate"])
            origin_target = torch.as_tensor(
                guide[origin], dtype=dtype, device=device)
            insertion_target = torch.as_tensor(
                guide[insertion], dtype=dtype, device=device)
            anchor_ids = torch.cat((origin_t, insertion_t))
            anchor_targets = torch.cat((origin_target, insertion_target))
            anchor_weights = torch.cat((
                torch.full((len(origin),), args.insertion_weight,
                           dtype=dtype, device=device),
                soft_weight))
            insertion_error = float("inf")
            for iteration in range(args.maximum_iterations):
                # Projective-dynamics quality target. Adding this target to
                # the same global system prevents contact/attachment forces
                # from first collapsing a tet and relying on a later repair.
                quality_target = x.clone()
                project_volumes_cuda(
                    quality_target, tet_t, rest_volume, inverse_mass, 0.8)
                quality_target[origin_t] = origin_target
                quality_delta = torch.linalg.vector_norm(
                    quality_target - x, dim=1)
                quality_ids = torch.where(quality_delta > 1e-8)[0]
                if len(quality_ids):
                    quality_weights = torch.full(
                        (len(quality_ids),), args.quality_weight,
                        dtype=dtype, device=device)
                    iteration_ids = torch.cat((anchor_ids, quality_ids))
                    iteration_targets = torch.cat((
                        anchor_targets, quality_target[quality_ids]))
                    iteration_weights = torch.cat((
                        anchor_weights, quality_weights))
                else:
                    iteration_ids = anchor_ids
                    iteration_targets = anchor_targets
                    iteration_weights = anchor_weights
                anchors = (
                    iteration_ids, iteration_targets, iteration_weights)
                raw = sdf.constraints(
                    x.cpu().numpy(), samples_i, samples_w, no_exclusion,
                    skel, args.collision_margin, max_step=0.001,
                    rest_tolerance=args.collision_tolerance)
                collision = None if raw is None else tuple((
                    torch.as_tensor(raw[0], dtype=torch.long, device=device),
                    torch.as_tensor(raw[1], dtype=dtype, device=device),
                    torch.as_tensor(raw[2], dtype=dtype, device=device)))
                before_global = x.clone()
                proposed = solver.step(
                    x, anchors, args.arap_weight, collision,
                    collision_weight=args.contact_weight)
                before_ratio = (
                    signed_volumes_cuda(before_global, tet_t)
                    * rest_sign / rest_abs)
                before_minimum = float(torch.min(before_ratio))
                delta = proposed - before_global
                step_scale = 1.0
                x = before_global
                for _ in range(20):
                    candidate = before_global + step_scale * delta
                    candidate_ratio = (
                        signed_volumes_cuda(candidate, tet_t)
                        * rest_sign / rest_abs)
                    candidate_minimum = float(torch.min(candidate_ratio))
                    if ((before_minimum > 1e-4
                         and candidate_minimum > 1e-4)
                            or (before_minimum <= 1e-4
                                and candidate_minimum > before_minimum)):
                        x = candidate
                        break
                    step_scale *= 0.5
                for _ in range(args.volume_passes):
                    before_volume = x.clone()
                    project_volumes_cuda(
                        x, tet_t, rest_volume, inverse_mass,
                        args.volume_stiffness)
                    x[origin_t] = origin_target
                    projected_ratio = (
                        signed_volumes_cuda(x, tet_t)
                        * rest_sign / rest_abs)
                    if float(torch.min(projected_ratio)) <= 1e-4:
                        x = before_volume
                        break
                insertion_error = float(torch.max(
                    torch.linalg.vector_norm(
                        x[insertion_t] - insertion_target, dim=1)))
                if (
                    iteration + 1 >= args.iterations
                    and insertion_error <= args.attachment_tolerance
                    and raw is None
                ):
                    break
            print(
                f"  frame {frame} substep {substep}/{args.substeps}: "
                f"insertion={insertion_error:.6g}m "
                f"iterations={iteration + 1}", flush=True)

        previous_pose = frame_pose.copy()
        # Contact is the final saved-state constraint. ARAP is deliberately
        # not run after this stage, so it cannot pull the muscle back through
        # the femur. Both caps remain immovable during projection.
        projected_x = x.cpu().numpy()
        for _ in range(args.final_contact_passes):
            raw = sdf.constraints(
                projected_x, samples_i, samples_w, attachment_mask,
                skel, args.collision_margin, max_step=0.001,
                rest_tolerance=args.collision_tolerance)
            if raw is None:
                break
            before_contact = projected_x.copy()
            project_collision_constraints(
                projected_x, raw, attachment_mask, max_step=0.001)
            before_t = torch.as_tensor(
                before_contact, dtype=dtype, device=device)
            delta_t = torch.as_tensor(
                projected_x - before_contact, dtype=dtype, device=device)
            accepted = False
            contact_scale = 1.0
            for _ in range(20):
                candidate_t = before_t + contact_scale * delta_t
                candidate_ratio = (
                    signed_volumes_cuda(candidate_t, tet_t)
                    * rest_sign / rest_abs)
                if float(torch.min(candidate_ratio)) > 1e-4:
                    projected_x = candidate_t.cpu().numpy()
                    accepted = True
                    break
                contact_scale *= 0.5
            if not accepted:
                projected_x = before_contact
                break
        x = torch.as_tensor(
            projected_x, dtype=dtype, device=device)
        ratio = (
            signed_volumes_cuda(x, tet_t) * rest_sign / rest_abs)
        raw = sdf.constraints(
            x.cpu().numpy(), samples_i, samples_w, no_exclusion,
            skel, args.collision_margin, None,
            args.collision_tolerance)
        contacts = 0 if raw is None else len(raw[0])
        inverted = int(torch.sum(ratio <= 0))
        min_j = float(torch.min(ratio))
        print(
            f"frame {frame}: inverted={inverted}/{len(tets)} "
            f"minJ={min_j:.6g} insertion={insertion_error:.6g}m "
            f"contact={contacts}", flush=True)
        output.append(x.cpu().numpy().astype(np.float32))
        metrics.append((inverted, min_j, insertion_error, contacts))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    m = np.asarray(metrics)
    np.savez_compressed(
        args.output_dir / f"{args.name}_chunk_0000.npz",
        frames=np.arange(len(output), dtype=np.int32),
        positions=np.asarray(output, np.float32),
        inverted_tets=m[:, 0].astype(np.int32),
        minimum_jacobian_ratio=m[:, 1],
        maximum_attachment_error=m[:, 2],
        remaining_contact_samples=m[:, 3].astype(np.int32))
    (args.output_dir / ".done").touch()
    print(f"Saved {len(output)} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
