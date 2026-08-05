#!/usr/bin/env python3
"""Bake one fiber-informed quasistatic tet muscle over BVH poses."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.bvhparser import MyBVH
from core.dartHelper import buildFromInfo, saveSkeletonInfo
from viewer.isolated_muscle import (
    attachment_targets, bind_attachment_patches, deformation_gradients,
    fiber_diagnostics, harmonic_attachment_update, load_muscle_data,
    precompute_energy,
    total_energy_gradient,
)


def detect_bvh_tframe(path):
    # Match the convention already used by the viewer and existing bakers.
    from tools.bake_emu import _detect_bvh_tframe
    return _detect_bvh_tframe(str(path))


def world_matrix(body):
    transform = body.getWorldTransform()
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(transform.rotation())
    matrix[:3, 3] = np.asarray(transform.translation())
    return matrix


def capture_bones(skeleton, names):
    result = {}
    for name in names:
        body = skeleton.getBodyNode(name)
        if body is None:
            raise ValueError(f"skeleton body not found: {name}")
        result[name] = world_matrix(body)
    return result


def interpolate_transforms(first, second, fraction):
    result = {}
    for name in first:
        rest_rotation = first[name][:3, :3]
        relative_end = second[name][:3, :3] @ rest_rotation.T
        relative_rotation = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix(np.stack((np.eye(3), relative_end))))(
                [fraction]).as_matrix()[0]
        rotation = relative_rotation @ rest_rotation
        affine_offset = (
            second[name][:3, 3]
            - relative_end @ first[name][:3, 3])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = (
            relative_rotation @ first[name][:3, 3]
            + fraction * affine_offset)
        result[name] = transform
    return result


def solve_substep(initial, target_by_vertex, muscle, precomputed, config,
                  frame, substep, iteration_rows):
    fixed_ids = np.asarray(sorted(target_by_vertex), dtype=np.int32)
    fixed_mask = np.zeros(len(initial), dtype=bool)
    fixed_mask[fixed_ids] = True
    free_ids = np.where(~fixed_mask)[0]
    fixed_targets = np.asarray(
        [target_by_vertex[int(i)] for i in fixed_ids])
    base = harmonic_attachment_update(
        initial, target_by_vertex, precomputed)
    base[fixed_ids] = fixed_targets
    evaluations = 0
    previous_x = None

    def unpack(free_flat):
        current = base.copy()
        current[free_ids] = free_flat.reshape(-1, 3)
        current[fixed_ids] = fixed_targets
        return current

    def objective(free_flat):
        nonlocal evaluations
        evaluations += 1
        current = unpack(free_flat)
        energy, gradient, _ = total_energy_gradient(
            current, muscle, precomputed, config)
        if not np.isfinite(energy):
            return 1e40, np.zeros_like(free_flat)
        return energy, gradient[free_ids].ravel()

    callback_iteration = 0

    def callback(free_flat):
        nonlocal callback_iteration, previous_x
        callback_iteration += 1
        current = unpack(free_flat)
        energy, gradient, terms = total_energy_gradient(
            current, muscle, precomputed, config)
        step_norm = (
            0.0 if previous_x is None
            else float(np.linalg.norm(free_flat - previous_x)))
        previous_x = free_flat.copy()
        attachment_error = float(np.max(np.linalg.norm(
            current[fixed_ids] - fixed_targets, axis=1)))
        iteration_rows.append({
            "frame": frame, "pose_substep": substep,
            "nonlinear_iteration": callback_iteration,
            "total_energy": energy, "matrix_energy": terms["matrix"],
            "volume_energy": terms["volume"],
            "fiber_energy": terms["fiber"],
            "fiber_bending_energy": terms["bending"],
            "gradient_norm": float(np.linalg.norm(gradient[free_ids])),
            "step_norm": step_norm, "line_search_alpha": "",
            "minimum_J": terms["minimum_J"],
            "maximum_J": terms["maximum_J"],
            "volume_ratio": terms["volume_ratio"],
            "maximum_attachment_error": attachment_error,
        })

    solver = config["solver"]
    coordinate_scale = float(solver.get("coordinate_scale", 0.001))
    optimization_energy_scale = float(
        solver.get("optimization_energy_scale", 1.0))

    # L-BFGS occasionally declares relative-energy convergence immediately on
    # SI-scale muscle energies. A few explicit free-DOF Armijo steps move the
    # state into the feasible interior before invoking its curvature model.
    free_flat = base[free_ids].ravel()
    for _ in range(int(solver.get("pre_relaxation_iterations", 24))):
        energy, gradient = objective(free_flat)
        if not np.isfinite(energy):
            break
        gradient_vertices = gradient.reshape(-1, 3)
        maximum_force = float(np.max(
            np.linalg.norm(gradient_vertices, axis=1)))
        if maximum_force < float(solver["gradient_tolerance"]):
            break
        direction = -gradient
        directional_derivative = float(np.dot(gradient, direction))
        alpha = min(
            1.0, 1e-4 / max(maximum_force, 1e-12))
        accepted = False
        for _line_search in range(
                int(solver["max_line_search_iterations"])):
            trial = free_flat + alpha * direction
            trial_energy, _ = objective(trial)
            if (np.isfinite(trial_energy)
                    and trial_energy <= energy
                    + 1e-4 * alpha * directional_derivative):
                free_flat = trial
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break

    def scaled_objective(scaled_free):
        energy, gradient = objective(scaled_free * coordinate_scale)
        return (
            optimization_energy_scale * energy,
            optimization_energy_scale * gradient * coordinate_scale)

    def scaled_callback(scaled_free):
        callback(scaled_free * coordinate_scale)

    result = minimize(
        scaled_objective, free_flat / coordinate_scale,
        method="L-BFGS-B", jac=True, callback=scaled_callback,
        options={
            "maxiter": int(solver["max_newton_iterations"]),
            "gtol": float(solver["gradient_tolerance"]),
            "ftol": float(solver.get("energy_tolerance", 1e-12)),
            "maxls": int(solver["max_line_search_iterations"]),
            "maxcor": 20,
        })
    lbfgs_free = result.x * coordinate_scale
    lbfgs_vertices = unpack(lbfgs_free)
    _, lbfgs_gradient, _ = total_energy_gradient(
        lbfgs_vertices, muscle, precomputed, config)
    if (np.linalg.norm(lbfgs_gradient[free_ids])
            >= float(solver["gradient_tolerance"])):
        # Matrix-free Newton fallback. The analytic gradient has its own
        # finite-difference test; differentiating it along a Krylov vector
        # avoids a dense 3N x 3N Hessian while retaining second-order search
        # directions and a trust region near the determinant boundary.
        def hessian_vector(scaled_free, vector):
            vector_norm = max(float(np.linalg.norm(vector)), 1e-12)
            epsilon = 1e-4 / vector_norm
            _, plus = scaled_objective(scaled_free + epsilon * vector)
            _, minus = scaled_objective(scaled_free - epsilon * vector)
            if not (np.all(np.isfinite(plus))
                    and np.all(np.isfinite(minus))):
                return vector
            return (plus - minus) / (2.0 * epsilon)

        newton_result = minimize(
            scaled_objective, result.x, method="trust-krylov", jac=True,
            hessp=hessian_vector, callback=scaled_callback,
            options={
                "maxiter": int(solver.get(
                    "max_trust_region_iterations", 80)),
                "gtol": (
                    optimization_energy_scale * coordinate_scale
                    * float(solver["gradient_tolerance"])),
                "initial_trust_radius": float(
                    solver.get("initial_trust_radius", 0.1)),
                "max_trust_radius": float(
                    solver.get("maximum_trust_radius", 10.0)),
            })
        newton_vertices = unpack(newton_result.x * coordinate_scale)
        newton_energy = total_energy_gradient(
            newton_vertices, muscle, precomputed, config)[0]
        lbfgs_energy = total_energy_gradient(
            lbfgs_vertices, muscle, precomputed, config)[0]
        if np.isfinite(newton_energy) and newton_energy <= lbfgs_energy:
            result = newton_result
    solved = unpack(result.x * coordinate_scale)
    energy, gradient, terms = total_energy_gradient(
        solved, muscle, precomputed, config)
    attachment_error = float(np.max(np.linalg.norm(
        solved[fixed_ids] - fixed_targets, axis=1)))
    converged = (
        np.isfinite(energy)
        and terms["minimum_J"] > float(config["material"]["minimum_J"])
        and attachment_error < 1e-10
        and np.linalg.norm(gradient[free_ids])
        < float(solver["gradient_tolerance"]))
    info = {
        "success": bool(converged), "optimizer_success": bool(result.success),
        "message": str(result.message), "iterations": int(result.nit),
        "evaluations": evaluations, "energy": float(energy),
        "gradient_norm": float(np.linalg.norm(gradient[free_ids])),
        "attachment_error": attachment_error, **terms,
    }
    return solved, info


def export_obj(path, vertices, faces, fibers, patches):
    with Path(path).open("w") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            handle.write(
                f"f {int(face[0])+1} {int(face[1])+1} {int(face[2])+1}\n")
        offset = len(vertices)
        for diagnostic in fibers:
            points = diagnostic["points"]
            for point in points:
                handle.write(f"v {point[0]} {point[1]} {point[2]}\n")
            indices = [str(offset + index + 1) for index in range(len(points))]
            handle.write("l " + " ".join(indices) + "\n")
            offset += len(points)
        for patch in patches:
            handle.write(
                "# attachment " + patch.bone_name + " "
                + " ".join(str(int(i) + 1) for i in patch.vertex_ids) + "\n")


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def run(args):
    with Path(args.config).open() as handle:
        config = yaml.safe_load(handle)
    if args.configuration == "A":
        config["material"]["bulk_modulus"] = 0.0
        config["fiber"]["anisotropy_enabled"] = False
        config["fiber"]["bending_enabled"] = False
    elif args.configuration == "B":
        config["fiber"]["anisotropy_enabled"] = False
        config["fiber"]["bending_enabled"] = False
    elif args.configuration == "C":
        config["fiber"]["anisotropy_enabled"] = True
        config["fiber"]["bending_enabled"] = False
    elif args.configuration == "D":
        config["fiber"]["anisotropy_enabled"] = True
        config["fiber"]["bending_enabled"] = True

    muscle, embedding_report = load_muscle_data(
        Path(args.tet_dir) / f"{args.muscle}_tet.npz")
    precomputed = precompute_energy(
        muscle, nearby_radius=float(config["fiber"]["nearby_radius"]))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "embedding_report.json").open("w") as handle:
        json.dump(json_ready({
            "muscle": muscle.name,
            **embedding_report,
            "tet_count": len(muscle.tetrahedra),
            "surface_face_count": len(muscle.surface_faces),
            "fiber_direction_tets": int(np.sum(
                precomputed["fiber_valid"])),
            "isotropic_tets": int(np.sum(
                ~precomputed["fiber_valid"])),
            "attachment_patches": [{
                "bone": patch.bone_name,
                "vertex_count": len(patch.vertex_ids),
                "narrow_boundary_loop": len(patch.vertex_ids) < 8,
            } for patch in muscle.attachment_patches],
        }), handle, indent=2)
    np.savez_compressed(
        output / "rest_preprocessing.npz",
        vertices=muscle.vertices, tetrahedra=muscle.tetrahedra,
        surface_faces=muscle.surface_faces,
        fiber_directions=precomputed["fiber_directions"],
        fiber_direction_valid=precomputed["fiber_valid"])

    skeleton_info, root_name, bvh_info, _, _, _ = saveSkeletonInfo(
        "data/zygote_skel.xml")
    skeleton = buildFromInfo(skeleton_info, root_name)
    skeleton.setPositions(np.zeros(skeleton.getNumDofs()))
    bone_names = sorted({
        patch.bone_name for patch in muscle.attachment_patches})
    rest_bones = capture_bones(skeleton, bone_names)
    bind_attachment_patches(
        muscle.attachment_patches, muscle.vertices, rest_bones)
    motion = MyBVH(
        args.bvh, bvh_info, skeleton,
        T_frame=detect_bvh_tframe(args.bvh))
    end = min(
        args.end_frame if args.end_frame is not None
        else len(motion.mocap_refs) - 1,
        len(motion.mocap_refs) - 1)

    positions = muscle.vertices.copy()
    previous_bones = rest_bones
    baked = []
    frame_numbers = []
    summary_rows = []
    iteration_rows = []
    for frame in range(args.start_frame, end + 1):
        skeleton.setPositions(motion.mocap_refs[frame])
        frame_bones = capture_bones(skeleton, bone_names)
        frame_start = positions.copy()
        initial_substeps = int(config["pose"]["initial_substeps"])
        pending_fractions = [
            index / initial_substeps
            for index in range(1, initial_substeps + 1)]
        candidate = frame_start.copy()
        last_fraction = 0.0
        accepted_substeps = 0
        retry_depth = 0
        substep_info = None
        while pending_fractions:
            fraction = pending_fractions.pop(0)
            transforms = interpolate_transforms(
                previous_bones, frame_bones, fraction)
            targets = attachment_targets(
                muscle.attachment_patches, transforms)
            trial, substep_info = solve_substep(
                candidate, targets, muscle, precomputed, config,
                frame, accepted_substeps + 1, iteration_rows)
            print(
                f"frame={frame} fraction={fraction:.6f} "
                f"success={substep_info['success']} "
                f"E={substep_info['energy']:.6e} "
                f"|g|={substep_info['gradient_norm']:.3e} "
                f"minJ={substep_info['minimum_J']:.5f} "
                f"V/V0={substep_info['volume_ratio']:.6f} "
                f"attach={substep_info['attachment_error']:.3e} "
                f"iters={substep_info['iterations']} "
                f"status={substep_info['message']}")
            if substep_info["success"]:
                candidate = trial
                last_fraction = fraction
                accepted_substeps += 1
                continue
            increment = fraction - last_fraction
            minimum_fraction = float(
                config["pose"]["minimum_pose_fraction"])
            if (not config["pose"].get("adaptive_subdivision", True)
                    or 0.5 * increment < minimum_fraction):
                raise RuntimeError(
                    f"frame {frame} failed at pose fraction {fraction}: "
                    f"{substep_info['message']}")
            midpoint = 0.5 * (last_fraction + fraction)
            pending_fractions.insert(0, fraction)
            pending_fractions.insert(0, midpoint)
            retry_depth += 1
            print(
                f"retry frame={frame} interval="
                f"[{last_fraction:.6f},{fraction:.6f}] "
                f"via midpoint={midpoint:.6f}")
        positions = candidate
        substeps = accepted_substeps

        fibers = fiber_diagnostics(positions, muscle, precomputed)
        F = deformation_gradients(positions, muscle, precomputed)
        J = np.linalg.det(F)
        stretch = np.linalg.norm(np.einsum(
            "nij,nj->ni", F, precomputed["fiber_directions"]), axis=1)
        stretch[~precomputed["fiber_valid"]] = np.nan
        bone_transform_array = np.asarray(
            [frame_bones[name] for name in bone_names])
        np.savez_compressed(
            output / f"frame_{frame:04d}_debug.npz",
            positions=positions, J=J, fiber_stretch=stretch,
            bone_names=np.asarray(bone_names),
            bone_world_transforms=bone_transform_array,
            fiber_points=np.asarray(
                [entry["points"] for entry in fibers], dtype=object),
            fiber_turning_angles=np.asarray(
                [entry["turning_angles"] for entry in fibers], dtype=object))
        fiber_rows = [{
            "stream": entry["stream"],
            "fiber": entry["fiber"],
            "rest_length": entry["rest_length"],
            "current_length": entry["current_length"],
            "length_ratio": entry["length_ratio"],
            "minimum_segment_ratio": entry["minimum_segment_ratio"],
            "maximum_segment_ratio": entry["maximum_segment_ratio"],
            "maximum_turning_angle": entry["maximum_turning_angle"],
            "maximum_turning_sample": entry["maximum_turning_sample"],
        } for entry in fibers]
        if fiber_rows:
            with (output / f"frame_{frame:04d}_fibers.csv").open(
                    "w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(fiber_rows[0]))
                writer.writeheader()
                writer.writerows(fiber_rows)
        if config["debug"].get("export_surface", True):
            export_obj(
                output / f"frame_{frame:04d}.obj", positions,
                muscle.surface_faces, fibers, muscle.attachment_patches)
        summary_rows.append({
            "frame": frame, "configuration": args.configuration,
            "pose_substeps": substeps,
            "minimum_J": float(J.min()), "maximum_J": float(J.max()),
            "inverted_tets": int(np.sum(J <= 0.0)),
            "volume_ratio": float(np.sum(
                precomputed["volumes"] * J)
                / np.sum(precomputed["volumes"])),
            "maximum_attachment_error": substep_info["attachment_error"],
            "minimum_fiber_length_ratio": min(
                entry["length_ratio"] for entry in fibers),
            "maximum_fiber_length_ratio": max(
                entry["length_ratio"] for entry in fibers),
            "maximum_fiber_turning_angle": max(
                entry["maximum_turning_angle"] for entry in fibers),
            "converged": substep_info["success"],
        })
        baked.append(positions.astype(np.float32))
        frame_numbers.append(frame)
        previous_bones = frame_bones

    np.savez_compressed(
        output / f"{muscle.name}_chunk_0000.npz",
        frames=np.asarray(frame_numbers, dtype=np.int32),
        positions=np.asarray(baked))
    iteration_fields = [
        "frame", "pose_substep", "nonlinear_iteration", "total_energy",
        "matrix_energy", "volume_energy", "fiber_energy",
        "fiber_bending_energy", "gradient_norm", "step_norm",
        "line_search_alpha", "minimum_J", "maximum_J", "volume_ratio",
        "maximum_attachment_error",
    ]
    for filename, rows, fields in (
            ("summary.csv", summary_rows, list(summary_rows[0])),
            ("iterations.csv", iteration_rows, iteration_fields)):
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with (output / ".done").open("w") as handle:
        handle.write("complete\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", default="L_Rectus_Femoris")
    parser.add_argument("--tet-dir", default="tet")
    parser.add_argument(
        "--bvh", default="data/motion/left_thigh_quasistatic_5pose.bvh")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument(
        "--config", default="config/isolated_muscle.yaml")
    parser.add_argument(
        "--configuration", choices=("A", "B", "C", "D"), default="D")
    parser.add_argument(
        "--output", default=".bake_outputs/isolated_muscle")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
