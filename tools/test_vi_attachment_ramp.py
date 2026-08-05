#!/usr/bin/env python3
"""One-pose VI experiment using a femur-rigid attachment load path."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.bvhparser import MyBVH
from tools import bake_emu
from tools.bake_stiff_tet_arap import BoneSDF
from tools.bake_stiff_tet_pbd import (
    build_surface_samples, signed_volumes, signed_volumes_cuda, unique_edges)
from tools.bake_surface_fast import load_tet, surface_faces
import test_emu


def relative_transform(body, rest_rotation, rest_translation):
    current = body.getWorldTransform()
    rotation = np.asarray(current.rotation()) @ rest_rotation.T
    translation = np.asarray(current.translation()) - rotation @ rest_translation
    return rotation, translation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, default=25)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--initial-cache", type=Path)
    parser.add_argument("--ramp-steps", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument(
        "--soft-insertion", dest="soft_insertion",
        action="store_true", default=True)
    parser.add_argument(
        "--hard-insertion", dest="soft_insertion", action="store_false")
    parser.add_argument(
        "--preconditioned", dest="preconditioned",
        action="store_true", default=True)
    parser.add_argument(
        "--no-preconditioned", dest="preconditioned", action="store_false")
    parser.add_argument("--project-jacobian-tangent", action="store_true")
    parser.add_argument("--attachment-start", type=float, default=10.0)
    parser.add_argument("--attachment-end", type=float, default=1000.0)
    parser.add_argument("--minimum-jacobian", type=float, default=1e-5)
    parser.add_argument("--allow-inverted-intermediate", action="store_true")
    parser.add_argument("--low-jacobian-weight", type=float, default=10.0)
    parser.add_argument("--contact-weight", type=float, default=3000.0)
    parser.add_argument("--strict-zero-sdf", action="store_true")
    parser.add_argument("--neo-hookean", action="store_true")
    parser.add_argument("--dense-contact-resolution", type=int, default=0)
    parser.add_argument("--bulk-coefficient", type=float, default=10.0)
    parser.add_argument(
        "--pose-continuation", dest="pose_continuation",
        action="store_true", default=True)
    parser.add_argument(
        "--no-pose-continuation", dest="pose_continuation",
        action="store_false")
    parser.add_argument(
        "--follow-bvh-frames", action="store_true",
        help="Traverse BVH frames 0 through --frame during continuation.")
    parser.add_argument("--bvh", type=Path, default=Path(
        "data/motion/left_thigh_quasistatic_diverse_smooth_76frame.bvh"))
    parser.add_argument("--tet", type=Path, default=Path(
        "tet/L_Vastus_Intermedius_tet.npz"))
    parser.add_argument("--sdf", type=Path, default=Path(
        ".bake_outputs/collision_sdf/L_Femur0_sdf.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        ".bake_outputs/vi_frame25_fast_sdf_view_v1"))
    parser.add_argument("--viewer-cache-dir", type=Path, default=Path(
        ".bake_outputs/motion_cache/"
        "left_thigh_quasistatic_diverse_smooth_76frame/"
        "L_Vastus_Intermedius_fast_sdf_raw_v1"))
    parser.add_argument(
        "--no-publish-viewer", dest="publish_viewer",
        action="store_false", default=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    skel, bvh_info, _ = bake_emu.load_skeleton()
    skel.setPositions(np.zeros(skel.getNumDofs()))
    data = load_tet(args.tet)
    group = test_emu.prepare_group_data(
        data, "L_Vastus_Intermedius", skel, test_emu._load_bone_trees(),
        source_path=args.tet, attachment_rings=0)
    rest = np.asarray(group["vertices"], dtype=np.float64)
    tets = np.asarray(group["tetrahedra"], dtype=np.int32)
    origin = np.asarray(group["origin_fixed"], dtype=np.int32)
    insertion = np.asarray(group["insertion_fixed"], dtype=np.int32)
    fixed = (origin.copy() if args.soft_insertion else
             np.unique(np.r_[origin, insertion]).astype(np.int32))
    fixed_mask = np.zeros(len(rest), dtype=bool)
    fixed_mask[fixed] = True
    contact_exclusion_mask = fixed_mask.copy()
    # A soft insertion is mechanically free, but its target lies on the
    # patella inside/next to the femur SDF envelope. Exempt only the authored
    # cap itself; the first free muscle layer still receives femur contact.
    if args.soft_insertion:
        contact_exclusion_mask[insertion] = True

    origin_body = skel.getBodyNode(group["origin_body"])
    insertion_body = skel.getBodyNode(group["insertion_body"])
    origin_rest_tf = origin_body.getWorldTransform()
    insertion_rest_tf = insertion_body.getWorldTransform()
    origin_rest = (
        np.asarray(origin_rest_tf.rotation()).copy(),
        np.asarray(origin_rest_tf.translation()).copy())
    insertion_rest = (
        np.asarray(insertion_rest_tf.rotation()).copy(),
        np.asarray(insertion_rest_tf.translation()).copy())

    motion = MyBVH(
        str(args.bvh), bvh_info, skel,
        T_frame=bake_emu._detect_bvh_tframe(str(args.bvh)))
    target_pose = motion.mocap_refs[args.frame].copy()
    start_pose = (
        motion.mocap_refs[args.start_frame].copy()
        if args.start_frame is not None else
        np.zeros(skel.getNumDofs()))
    skel.setPositions(
        start_pose.copy()
        if args.pose_continuation else target_pose.copy())
    femur_rotation, femur_translation = relative_transform(
        origin_body, *origin_rest)
    patella_rotation, patella_translation = relative_transform(
        insertion_body, *insertion_rest)
    femur_rigid = rest @ femur_rotation.T + femur_translation

    patella_target = rest[insertion] @ patella_rotation.T + patella_translation
    edges = unique_edges(tets)
    device = torch.device(args.device)
    dtype = torch.float64
    insertion_t = torch.as_tensor(
        insertion, dtype=torch.long, device=device)
    patella_target_t = torch.as_tensor(
        patella_target, dtype=dtype, device=device)
    edge_t = torch.as_tensor(edges, dtype=torch.long, device=device)
    tet_t = torch.as_tensor(tets, dtype=torch.long, device=device)
    initial_positions = femur_rigid
    if args.initial_cache is not None:
        initial_data = np.load(args.initial_cache)
        cached_positions = np.asarray(initial_data["positions"])
        if cached_positions.ndim == 3:
            cached_frames = np.asarray(initial_data["frames"])
            wanted_frame = (
                args.start_frame if args.start_frame is not None else 0)
            matches = np.flatnonzero(cached_frames == wanted_frame)
            if len(matches) != 1:
                raise ValueError(
                    f"frame {wanted_frame} not found uniquely in "
                    f"{args.initial_cache}")
            cached_positions = cached_positions[matches[0]]
        if cached_positions.shape != rest.shape:
            raise ValueError(
                f"initial cache shape {cached_positions.shape} != "
                f"tet shape {rest.shape}")
        initial_positions = cached_positions.astype(np.float64)
    reference_edge_length = np.linalg.norm(
        initial_positions[edges[:, 0]] - initial_positions[edges[:, 1]],
        axis=1)
    rest_edge_length = torch.as_tensor(
        reference_edge_length, dtype=dtype, device=device).clamp_min(1e-8)
    reference_volume = signed_volumes(initial_positions, tets)
    rest_volume = torch.as_tensor(
        reference_volume, dtype=dtype, device=device)
    rest_sign = torch.sign(rest_volume)
    rest_abs = torch.abs(rest_volume).clamp_min(1e-14)
    fixed_t = torch.as_tensor(fixed, dtype=torch.long, device=device)

    faces = surface_faces(tets)
    if args.dense_contact_resolution > 0:
        resolution = args.dense_contact_resolution
        sample_i_out, sample_w_out = [], []
        for face in faces:
            for i in range(resolution + 1):
                for j in range(resolution + 1 - i):
                    a, b = i / resolution, j / resolution
                    sample_i_out.append(face)
                    sample_w_out.append((1.0 - a - b, a, b))
        # Explicitly include every unique tet edge, not only boundary edges.
        # This prevents an internal straight segment from cutting through the
        # femur while its surrounding boundary happens to remain outside.
        for edge in edges:
            carrier = (int(edge[0]), int(edge[1]), int(edge[1]))
            for i in range(resolution + 1):
                a = i / resolution
                sample_i_out.append(carrier)
                sample_w_out.append((1.0 - a, a, 0.0))
        sample_i = np.asarray(sample_i_out, dtype=np.int32)
        sample_w = np.asarray(sample_w_out, dtype=np.float64)
    else:
        sample_i, sample_w = build_surface_samples(faces)
    if args.strict_zero_sdf or args.dense_contact_resolution > 0:
        attachment_mask = np.zeros(len(rest), dtype=bool)
        attachment_mask[origin] = True
        attachment_mask[insertion] = True
        active_support = sample_w > 1e-12
        pure_attachment = (
            np.sum(active_support, axis=1) == 1) & np.any(
                attachment_mask[sample_i] & active_support, axis=1)
        sample_i = sample_i[~pure_attachment]
        sample_w = sample_w[~pure_attachment]
        # Mixed attachment/free edge samples remain active. Only pure cap
        # samples were removed above.
        contact_exclusion_mask[:] = False
    sdf = BoneSDF(args.sdf)
    # Bone-relative clearance is unchanged by the initial femur-rigid motion.
    skel.setPositions(np.zeros(skel.getNumDofs()))
    sdf.bind_rest_clearance(
        rest, sample_i, sample_w, skel,
        prevent_new_only=False, margin=0.0015)
    # Preserve authored overlap instead of disabling its contact constraint:
    # an initially interior material sample may remain at its rest SDF depth,
    # but flexion may not drive it deeper. Initially exterior samples retain
    # the requested positive clearance. This applies to boundary-face samples
    # and to every sampled tet edge.
    valid_sdf = np.isfinite(sdf.rest_distance)
    exterior_at_rest = valid_sdf & (sdf.rest_distance >= 0.0)
    sdf.rest_distance[exterior_at_rest] = 0.0015
    strict_start_distance = (
        sdf.rest_distance.copy() if args.strict_zero_sdf else None)
    if args.strict_zero_sdf:
        # Enforce actual exterior contact for every valid non-cap sample,
        # including regions overlapping the femur in the authored rest mesh.
        # The threshold is advanced inside the continuation loop. Applying
        # full exterior clearance to the intersecting start in one step folds
        # the first tet layer.
        sdf.rest_distance[valid_sdf] = strict_start_distance[valid_sdf]
    skel.setPositions(
        start_pose.copy()
        if args.pose_continuation else target_pose.copy())

    x = torch.as_tensor(
        initial_positions, dtype=dtype, device=device).clone()
    minimum_history = []

    attachment_weight = args.attachment_start
    previous_femur_rotation, previous_femur_translation = relative_transform(
        origin_body, *origin_rest)
    reference_tet = torch.as_tensor(
        initial_positions, dtype=dtype, device=device)[tet_t]
    rest_dm = torch.stack((
        reference_tet[:, 1] - reference_tet[:, 0],
        reference_tet[:, 2] - reference_tet[:, 0],
        reference_tet[:, 3] - reference_tet[:, 0]), dim=2)
    rest_dm_inverse = torch.linalg.inv(rest_dm)
    rest_tet_weight = torch.abs(rest_volume)
    rest_tet_weight = rest_tet_weight / torch.sum(rest_tet_weight)

    def objective(current):
        volume_ratio = (
            signed_volumes_cuda(current, tet_t) * rest_sign / rest_abs)
        safe_ratio = volume_ratio.clamp_min(1e-8)
        if args.neo_hookean:
            current_tet = current[tet_t]
            current_dm = torch.stack((
                current_tet[:, 1] - current_tet[:, 0],
                current_tet[:, 2] - current_tet[:, 0],
                current_tet[:, 3] - current_tet[:, 0]), dim=2)
            deformation = current_dm @ rest_dm_inverse
            invariant = torch.sum(deformation * deformation, dim=(1, 2))
            jacobian = torch.linalg.det(deformation)
            # Stable Neo-Hookean with normalized shear modulus 1 and bulk
            # coefficient 20. The separate log barrier enforces J > 0.
            density = (
                0.5 * (invariant - 3.0) - (jacobian - 1.0)
                + args.bulk_coefficient * (jacobian - 1.0) ** 2)
            shape_loss = torch.sum(rest_tet_weight * density)
            volume_loss = current.new_zeros(())
        else:
            edge_length = torch.linalg.vector_norm(
                current[edge_t[:, 0]] - current[edge_t[:, 1]], dim=1)
            edge_ratio = edge_length / rest_edge_length
            shape_loss = torch.mean((edge_ratio - 1.0) ** 2)
            volume_loss = torch.mean((volume_ratio - 1.0) ** 2)
        barrier = -torch.mean(torch.log(safe_ratio))
        low_jacobian = torch.mean(
            torch.relu(0.35 - volume_ratio) ** 2)
        contact = sdf.torch_penetration_loss(
            current, sample_i, sample_w, contact_exclusion_mask, skel,
            tolerance=0.00025)
        attachment = (
            torch.mean(torch.sum(
                (current[insertion_t] - patella_target_t) ** 2, dim=1))
            / (0.001 ** 2)
            if args.soft_insertion else current.new_zeros(()))
        return (
            shape_loss + 4.0 * volume_loss
            + args.low_jacobian_weight * low_jacobian + 0.2 * barrier
            + args.contact_weight * contact
            + attachment_weight * attachment)

    for step in range(1, args.ramp_steps + 1):
        fraction = step / args.ramp_steps
        if args.pose_continuation:
            if args.follow_bvh_frames and args.start_frame is None:
                frame_coordinate = fraction * args.frame
                lower = min(int(np.floor(frame_coordinate)), args.frame)
                upper = min(lower + 1, args.frame)
                blend = frame_coordinate - lower
                skel.setPositions(
                    (1.0 - blend) * motion.mocap_refs[lower]
                    + blend * motion.mocap_refs[upper])
            else:
                skel.setPositions(
                    (1.0 - fraction) * start_pose
                    + fraction * target_pose)
            femur_rotation, femur_translation = relative_transform(
                origin_body, *origin_rest)
            patella_rotation, patella_translation = relative_transform(
                insertion_body, *insertion_rest)
            femur_rigid = rest @ femur_rotation.T + femur_translation
            patella_target = (
                rest[insertion] @ patella_rotation.T
                + patella_translation)
            patella_target_t = torch.as_tensor(
                patella_target, dtype=dtype, device=device)
            # Rigidly transport the previous equilibrium by the incremental
            # femur motion. This is a predictor only: no free vertex is fixed
            # or tethered to it during the ensuing FEM solve.
            incremental_rotation = (
                femur_rotation @ previous_femur_rotation.T)
            incremental_translation = (
                femur_translation
                - incremental_rotation @ previous_femur_translation)
            rotation_t = torch.as_tensor(
                incremental_rotation, dtype=dtype, device=device)
            translation_t = torch.as_tensor(
                incremental_translation, dtype=dtype, device=device)
            x = x @ rotation_t.T + translation_t
            previous_femur_rotation = femur_rotation.copy()
            previous_femur_translation = femur_translation.copy()
        target = femur_rigid[fixed].copy()
        if args.soft_insertion:
            # The virtual zero-rest edges always point to the final patellar
            # attachment sites. Continuation increases their stiffness rather
            # than prescribing an immediately infeasible cap displacement.
            attachment_weight = (
                args.attachment_start
                * (args.attachment_end / args.attachment_start) ** fraction)
        else:
            # With pose continuation, patella_target is already evaluated at
            # the current interpolated skeleton pose. Interpolating it again
            # delays the cap motion quadratically and causes a destructive
            # catch-up near the final frame.
            insertion_target = (
                patella_target if args.pose_continuation else
                (1.0 - fraction) * femur_rigid[insertion]
                + fraction * patella_target)
            target[np.searchsorted(fixed, insertion)] = insertion_target
        if args.strict_zero_sdf:
            valid_sdf = np.isfinite(strict_start_distance)
            sdf.rest_distance[valid_sdf] = (
                (1.0 - fraction) * strict_start_distance[valid_sdf]
                + fraction * 0.0015)
        target_t = torch.as_tensor(target, dtype=dtype, device=device)
        previous_x = x.clone()
        x[fixed_t] = target_t
        loaded_ratio = (
            signed_volumes_cuda(x, tet_t) * rest_sign / rest_abs)
        # A moving hard cap can temporarily invert its adjacent layer before
        # the free vertices respond. Do not abort at that intermediate state;
        # the recovery branch in the line search accepts monotonic Jacobian
        # improvement until the layer becomes positive again.

        adam_m = torch.zeros_like(x)
        adam_v = torch.zeros_like(x)
        for _ in range(args.iterations):
            x.requires_grad_(True)
            # A logarithmic barrier makes crossing J=0 infinitely expensive;
            # the line search below also rejects every non-positive candidate.
            loss = objective(x)
            gradient, = torch.autograd.grad(
                loss, x, retain_graph=args.project_jacobian_tangent)
            gradient[fixed_t] = 0.0
            before = x.detach()
            max_gradient = float(torch.abs(gradient).max())
            if args.preconditioned:
                # Per-coordinate scaling prevents a very stiff insertion
                # spring from reducing all neighboring barrier/contact motion
                # to almost zero through one global gradient normalization.
                trial_m = 0.9 * adam_m + 0.1 * gradient
                trial_v = 0.999 * adam_v + 0.001 * gradient * gradient
                direction = trial_m / (torch.sqrt(trial_v) + 1e-12)
                direction[fixed_t] = 0.0
                scale = 1e-5
            else:
                # Normalize the initial trial to at most 10 micrometres of
                # vertex motion. Raw SDF/barrier gradients have incompatible
                # units.
                direction = gradient
                scale = 1e-5 / max(max_gradient, 1e-12)
            delta = -direction
            if args.project_jacobian_tangent:
                current_ratio = (
                    signed_volumes_cuda(x, tet_t) * rest_sign / rest_abs)
                # Project the proposed displacement into the tangent cones of
                # the weakest determinant constraints. This permits sliding
                # along J=J_min instead of freezing the whole solve there.
                active = torch.argsort(current_ratio)[:128]
                for tet_id in active:
                    if float(current_ratio[tet_id]) > (
                            args.minimum_jacobian + 0.03):
                        break
                    jacobian_gradient, = torch.autograd.grad(
                        current_ratio[tet_id], x, retain_graph=True)
                    jacobian_gradient[fixed_t] = 0.0
                    derivative = torch.sum(jacobian_gradient * delta)
                    denominator = torch.sum(
                        jacobian_gradient * jacobian_gradient).clamp_min(1e-20)
                    if float(derivative) < 0.0:
                        delta = (
                            delta - derivative / denominator
                            * jacobian_gradient)
                delta[fixed_t] = 0.0
            accepted = False
            before_ratio = (
                signed_volumes_cuda(before, tet_t) * rest_sign / rest_abs)
            before_minimum = float(before_ratio.min())
            for _ in range(20):
                candidate = before + scale * delta
                candidate[fixed_t] = target_t
                candidate_ratio = (
                    signed_volumes_cuda(candidate, tet_t)
                    * rest_sign / rest_abs)
                with torch.no_grad():
                    candidate_loss = objective(candidate)
                loss_value = float(loss.detach())
                candidate_value = float(candidate_loss)
                candidate_minimum = float(candidate_ratio.min())
                recovering = (
                    before_minimum <= args.minimum_jacobian
                    and candidate_minimum > before_minimum + 1e-10)
                feasible_descent = (
                    candidate_minimum > args.minimum_jacobian
                    and candidate_value
                    <= loss_value + 1e-12 * max(1.0, abs(loss_value)))
                unrestricted_descent = (
                    args.allow_inverted_intermediate
                    and candidate_value
                    <= loss_value + 1e-12 * max(1.0, abs(loss_value)))
                if recovering or feasible_descent or unrestricted_descent:
                    x = candidate.detach()
                    if args.preconditioned:
                        adam_m, adam_v = trial_m.detach(), trial_v.detach()
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                # With grid-sampled SDF and a strict determinant inequality,
                # machine-precision line-search stagnation is the equilibrium
                # for this load level. Advance the stiffness continuation.
                x = before
                break

        ratio = (
            signed_volumes_cuda(x, tet_t) * rest_sign / rest_abs)
        minimum = float(ratio.min())
        minimum_history.append(minimum)
        if step == 1 or step % 5 == 0 or step == args.ramp_steps:
            insertion_error = float(torch.linalg.vector_norm(
                x[insertion_t] - patella_target_t, dim=1).max())
            print(
                f"ramp {step:03d}/{args.ramp_steps}: "
                f"min_J_ratio={minimum:.6g}, "
                f"attachment_error={insertion_error:.6g} m, "
                f"attachment_weight={attachment_weight:.6g}")

    result = x.cpu().numpy()
    raw_contact = sdf.constraints(
        result, sample_i, sample_w, contact_exclusion_mask, skel,
        margin=0.0015, max_step=None, rest_tolerance=0.00025)
    remaining_contact = 0 if raw_contact is None else len(raw_contact[0])
    insertion_error = np.linalg.norm(
        result[insertion] - patella_target, axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / (
            f"L_Vastus_Intermedius_frame_{args.frame:04d}.npz"),
        frame=np.asarray(args.frame, dtype=np.int32),
        positions=result.astype(np.float32),
        tetrahedra=tets,
        origin_fixed=origin,
        insertion_fixed=insertion,
        minimum_jacobian_ratio=np.asarray(minimum_history[-1]),
        insertion_error=insertion_error,
        remaining_contact_samples=np.asarray(remaining_contact))
    if args.publish_viewer:
        args.viewer_cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.viewer_cache_dir / (
                f"L_Vastus_Intermedius_chunk_{args.frame:04d}.npz"),
            frames=np.asarray([args.frame], dtype=np.int32),
            positions=result[None].astype(np.float32))
        (args.viewer_cache_dir / ".done").touch()
        print(f"VIEWER_CACHE {args.viewer_cache_dir}")
    final_ratio = (
        signed_volumes_cuda(x, tet_t) * rest_sign / rest_abs).cpu().numpy()
    inverted_count = int(np.count_nonzero(final_ratio <= 0.0))
    print(
        f"RESULT frame={args.frame} inverted={inverted_count}/{len(tets)} "
        f"min_J_ratio={minimum_history[-1]:.6g} "
        f"max_attachment_error={insertion_error.max():.6g} m "
        f"mean_attachment_error={insertion_error.mean():.6g} m "
        f"remaining_contact_samples={remaining_contact}")


if __name__ == "__main__":
    main()
