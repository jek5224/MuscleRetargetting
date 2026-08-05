#!/usr/bin/env python3
"""Guarded augmented-Lagrangian rest correction for selected muscle pairs."""
import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import trimesh
import yaml

from muscle_sim.check_rest_feasibility import build_rest_system
from muscle_sim.rest_contact import attachment_patch_report
from viewer.augmented_contact import (
    PersistentActiveSet, build_deformable_candidates,
    diagnose_infeasible_contacts)
from viewer.isolated_muscle import (
    deformation_gradients, total_energy_gradient)


def correction_material_config(body_config, config):
    result = copy.deepcopy(body_config)
    result["material"]["shear_modulus"] = float(
        config["rest_initialization"]["shape_weight"])
    result["material"]["bulk_modulus"] = float(
        config["rest_initialization"]["volume_weight"])
    result["material"]["minimum_J"] = float(
        config["material"]["contact_safe_minimum_J"])
    result["fiber"]["anisotropy_enabled"] = False
    result["fiber"]["bending_enabled"] = False
    return result


def evaluate_correction(global_vertices, original, system, active_contacts,
                        rho, config):
    positions = system.unpack(global_vertices)
    originals = system.unpack(original)
    gradient = np.zeros_like(global_vertices)
    terms = {"shape_volume": 0.0, "surface": 0.0, "contact": 0.0}
    terms["transition_attachment"] = 0.0
    minimum_j = np.inf
    body_terms = {}
    surface_weight = float(
        config["rest_initialization"]["surface_weight"])
    for body_id, (body, current, rest) in enumerate(
            zip(system.bodies, positions, originals)):
        material_config = correction_material_config(body.config, config)
        energy, local_gradient, material = total_energy_gradient(
            current, body.muscle, body.precomputed, material_config)
        if not np.isfinite(energy):
            return np.inf, np.full_like(global_vertices, np.nan), {
                **terms, "minimum_J": material["minimum_J"]}
        surface_ids = np.unique(body.muscle.surface_faces)
        displacement = current[surface_ids] - rest[surface_ids]
        surface_energy = (
            0.5 * surface_weight * np.sum(displacement ** 2))
        local_gradient[surface_ids] += surface_weight * displacement
        terms["shape_volume"] += energy
        terms["surface"] += surface_energy
        transition = np.asarray(
            getattr(body, "transition_vertices", []), dtype=np.int32)
        if len(transition):
            stiffness = float(
                config["attachment"]["transition_stiffness"])
            transition_displacement = (
                current[transition] - rest[transition])
            terms["transition_attachment"] += (
                0.5 * stiffness
                * np.sum(transition_displacement ** 2))
            local_gradient[transition] += (
                stiffness * transition_displacement)
        minimum_j = min(minimum_j, material["minimum_J"])
        body_terms[body.name] = material
        gradient[body.offset:body.offset + body.vertex_count] += (
            local_gradient)
    for contact in active_contacts.values():
        energy, local_gradients, _, _ = contact.energy_gradient(
            positions, rho)
        terms["contact"] += energy
        for body, local_gradient in zip(
                system.bodies, local_gradients):
            gradient[body.offset:body.offset + body.vertex_count] += (
                local_gradient)
    terms["minimum_J"] = minimum_j
    terms["bodies"] = body_terms
    return sum(terms[key] for key in (
        "shape_volume", "surface", "transition_attachment",
        "contact")), gradient, terms


def solve_inner(current, original, system, active, rho, config):
    fixed_ids = np.asarray(sorted({
        body.offset + int(vertex)
        for body in system.bodies
        for vertex in body.fixed_vertices}), dtype=np.int32)
    fixed_values = original[fixed_ids]
    fixed_mask = np.zeros(system.vertex_count, dtype=bool)
    fixed_mask[fixed_ids] = True
    free_ids = np.where(~fixed_mask)[0]
    base = current.copy()
    base[fixed_ids] = fixed_values
    coordinate_scale, energy_scale = 1e-5, 1e6

    def unpack(values):
        result = base.copy()
        result[free_ids] = values.reshape(-1, 3) * coordinate_scale
        result[fixed_ids] = fixed_values
        return result

    def objective(values):
        vertices = unpack(values)
        energy, gradient, _ = evaluate_correction(
            vertices, original, system, active.contacts, rho, config)
        if not np.isfinite(energy):
            return 1e40, np.zeros_like(values)
        return (energy * energy_scale,
                gradient[free_ids].ravel()
                * coordinate_scale * energy_scale)

    result = minimize(
        objective, (base[free_ids] / coordinate_scale).ravel(),
        jac=True, method="L-BFGS-B",
        options={
            "maxiter": int(config["solver"]["max_newton_iterations"]),
            "maxls": int(config["solver"]["max_line_search_iterations"]),
            "gtol": (
                float(config["solver"]["gradient_tolerance"])
                * coordinate_scale * energy_scale),
            "ftol": 1e-15, "maxcor": 20,
        })
    lbfgs_state = unpack(result.x)
    lbfgs_energy, lbfgs_gradient, _ = evaluate_correction(
        lbfgs_state, original, system, active.contacts, rho, config)
    if (not np.isfinite(lbfgs_energy)
            or np.linalg.norm(lbfgs_gradient[free_ids])
            > float(config["solver"]["gradient_tolerance"])):
        def hessian_vector(values, vector):
            norm = max(float(np.linalg.norm(vector)), 1e-12)
            epsilon = 1e-4 / norm
            _, plus = objective(values + epsilon * vector)
            _, minus = objective(values - epsilon * vector)
            if not (np.all(np.isfinite(plus))
                    and np.all(np.isfinite(minus))):
                return vector
            return (plus - minus) / (2.0 * epsilon)

        newton = minimize(
            objective, result.x, jac=True, hessp=hessian_vector,
            method="trust-krylov",
            options={
                "maxiter": 80,
                "gtol": (
                    float(config["solver"]["gradient_tolerance"])
                    * coordinate_scale * energy_scale),
                "initial_trust_radius": 0.1,
                "max_trust_radius": 10.0,
            })
        newton_state = unpack(newton.x)
        newton_energy = evaluate_correction(
            newton_state, original, system, active.contacts,
            rho, config)[0]
        if (np.isfinite(newton_energy)
                and (not np.isfinite(lbfgs_energy)
                     or newton_energy <= lbfgs_energy)):
            result = newton
    solved = unpack(result.x)
    energy, gradient, terms = evaluate_correction(
        solved, original, system, active.contacts, rho, config)
    return solved, {
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "energy": energy,
        "mechanical_residual": float(
            np.linalg.norm(gradient[free_ids])),
        "minimum_J": terms["minimum_J"],
        "attachment_error": float(np.max(np.linalg.norm(
            solved[fixed_ids] - fixed_values, axis=1))),
        "terms": terms,
    }


def tet_quality_report(body, positions, active_points, threshold):
    F = deformation_gradients(
        positions, body.muscle, body.precomputed)
    J = np.linalg.det(F)
    rows = []
    attachment_vertices = np.asarray(sorted(body.fixed_vertices))
    contact_tree = (
        None if not len(active_points)
        else __import__("scipy").spatial.cKDTree(active_points))
    for tet_id in np.where(J < threshold)[0]:
        tet = body.muscle.tetrahedra[tet_id]
        rest = body.muscle.vertices[tet]
        edges = np.asarray([
            np.linalg.norm(rest[i] - rest[j])
            for i in range(4) for j in range(i + 1, 4)])
        centroid = positions[tet].mean(axis=0)
        face_normals = []
        for face in ((0, 2, 1), (0, 1, 3),
                     (1, 2, 3), (0, 3, 2)):
            triangle = rest[np.asarray(face)]
            normal = np.cross(
                triangle[1] - triangle[0],
                triangle[2] - triangle[0])
            normal /= max(np.linalg.norm(normal), 1e-20)
            face_normals.append(normal)
        dihedral = []
        for first in range(4):
            for second in range(first + 1, 4):
                cosine = np.clip(np.dot(
                    face_normals[first], face_normals[second]), -1.0, 1.0)
                dihedral.append(float(np.pi - np.arccos(cosine)))
        distance_attachment = (
            float(np.min(np.linalg.norm(
                body.muscle.vertices[attachment_vertices]
                - centroid, axis=1)))
            if len(attachment_vertices) else np.inf)
        distance_contact = (
            float(contact_tree.query(centroid)[0])
            if contact_tree is not None else np.inf)
        rows.append({
            "muscle": body.name, "tet_id": int(tet_id),
            "J": float(J[tet_id]),
            "singular_values": np.linalg.svd(
                F[tet_id], compute_uv=False).tolist(),
            "rest_volume": float(body.precomputed["volumes"][tet_id]),
            "edge_aspect_ratio": float(edges.max() / edges.min()),
            "minimum_dihedral_angle_degrees": float(
                np.degrees(min(dihedral))),
            "maximum_dihedral_angle_degrees": float(
                np.degrees(max(dihedral))),
            "distance_to_attachment": distance_attachment,
            "distance_to_active_contact": distance_contact,
            "nearby_active_contact_count": (
                len(contact_tree.query_ball_point(centroid, 0.003))
                if contact_tree is not None else 0),
        })
    return rows


def export_low_j(path, body, positions, tet_ids):
    if not tet_ids:
        return
    faces = []
    for tet_id in tet_ids:
        tet = body.muscle.tetrahedra[tet_id]
        faces.extend((
            [tet[0], tet[2], tet[1]], [tet[0], tet[1], tet[3]],
            [tet[1], tet[2], tet[3]], [tet[0], tet[3], tet[2]]))
    trimesh.Trimesh(
        vertices=positions, faces=np.asarray(faces),
        process=False).export(path)


def run(args):
    with Path(args.config).open() as handle:
        config = yaml.safe_load(handle)
    selected = list(args.pair)
    config["muscles"]["selected"] = selected
    config["contact_pairs"] = [selected]
    config["fascia"]["enabled"] = False
    overrides = {}
    for entry in getattr(args, "override_muscle", []) or []:
        name, separator, path = entry.partition("=")
        if not separator or not name or not path:
            raise ValueError("--override-muscle requires MUSCLE=PATH")
        overrides[name] = Path(path)
    system = build_rest_system(config, args.tet_dir, overrides)
    if args.attachment_mode is not None:
        config["attachment"]["mode"] = args.attachment_mode
    if config["attachment"]["mode"] == "core_hard_transition_soft":
        for body in system.bodies:
            proposals = attachment_patch_report(
                body, body.muscle.vertices, system.bones)
            body.fixed_vertices = set(
                vertex for proposal in proposals
                for vertex in proposal["candidate_central_hard_patch"])
            body.transition_vertices = sorted(set(
                vertex for proposal in proposals
                for vertex in proposal["candidate_transition_ring"]))
    elif config["attachment"]["mode"] == "full_patch":
        for body in system.bodies:
            body.transition_vertices = []
    elif config["attachment"]["mode"] == "manually_curated":
        raise ValueError(
            "manually_curated mode requires explicit curated vertex lists; "
            "none are configured")
    original_positions = [
        body.muscle.vertices.copy() for body in system.bodies]
    original = system.pack(original_positions)
    current = original.copy()
    contact_cfg = config["contact"]
    active = PersistentActiveSet(
        float(contact_cfg["activation_distance"]),
        float(contact_cfg["release_distance"]))
    initial_candidates = build_deformable_candidates(
        original_positions, system.bodies, 0, 1,
        float(contact_cfg["release_distance"]))
    fixed = [body.fixed_vertices for body in system.bodies]
    infeasible = diagnose_infeasible_contacts(
        initial_candidates, fixed,
        float(contact_cfg["penetration_tolerance"]))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if infeasible:
        with (output / "infeasible_constraints.json").open("w") as handle:
            json.dump(infeasible, handle, indent=2)
        raise RuntimeError(
            f"{len(infeasible)} fully constrained infeasible contacts")

    rho = float(contact_cfg["initial_rho"])
    maximum_rho = float(contact_cfg["maximum_rho"])
    growth = float(contact_cfg["rho_growth"])
    maximum_outer = int(contact_cfg["maximum_outer_iterations"])
    safe_j = float(config["material"]["contact_safe_minimum_J"])
    rows = []
    accepted = False
    last_valid = current.copy()
    diagnostic_state = current.copy()
    for outer in range(maximum_outer):
        positions = system.unpack(current)
        candidates = build_deformable_candidates(
            positions, system.bodies, 0, 1,
            float(contact_cfg["release_distance"]),
            previous=active.contacts)
        active_change = active.update(candidates)
        solved, inner = solve_inner(
            current, original, system, active, rho, config)
        diagnostic_state = solved.copy()
        solved_positions = system.unpack(solved)
        refreshed = build_deformable_candidates(
            solved_positions, system.bodies, 0, 1,
            float(contact_cfg["release_distance"]),
            previous=active.contacts)
        active_change += active.update(refreshed)
        for contact in active.contacts.values():
            contact.last_gap = contact.gap(solved_positions)
        multiplier_change = active.multiplier_update(rho)
        maximum_penetration = active.maximum_penetration()
        complementarity = active.complementarity_norm()
        row = {
            "outer_iteration": outer, "rho": rho,
            "active_contacts": len(active.contacts),
            "active_set_change_count": active_change,
            "inner_iterations": inner["iterations"],
            "inner_gradient_norm": inner["mechanical_residual"],
            "inner_step_norm": float(np.linalg.norm(solved - current)),
            "maximum_penetration": maximum_penetration,
            "rms_penetration": float(np.sqrt(np.mean([
                max(0.0, -contact.last_gap) ** 2
                for contact in active.contacts.values()] or [0.0]))),
            "complementarity_residual": complementarity,
            "multiplier_change": multiplier_change,
            "minimum_J": inner["minimum_J"],
            "attachment_error": inner["attachment_error"],
            "optimizer_success": inner["optimizer_success"],
            "optimizer_message": inner["message"],
        }
        rows.append(row)
        print(" ".join(f"{key}={value}" for key, value in row.items()
                       if key not in ("optimizer_message",)))
        inner_ok = (
            inner["mechanical_residual"]
            <= float(config["solver"]["gradient_tolerance"])
            and inner["minimum_J"] >= safe_j
            and inner["attachment_error"] <= 1e-10)
        if inner_ok:
            last_valid = solved.copy()
            current = solved
        else:
            current = last_valid.copy()
            break
        accepted = (
            maximum_penetration
            <= float(contact_cfg["penetration_tolerance"])
            and complementarity
            <= float(contact_cfg["complementarity_tolerance"])
            and inner_ok)
        if accepted:
            break
        if rho < maximum_rho:
            rho = min(maximum_rho, rho * growth)

    with (output / "al_iterations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    corrected_positions = system.unpack(
        last_valid if accepted else diagnostic_state)
    displacement_report = {}
    for body, corrected, rest in zip(
            system.bodies, corrected_positions, original_positions):
        displacement = np.linalg.norm(corrected - rest, axis=1)
        displacement_report[body.name] = {
            "maximum_displacement": float(displacement.max()),
            "rms_displacement": float(np.sqrt(np.mean(displacement ** 2))),
            "fraction_over_0.5mm": float(np.mean(displacement > 0.0005)),
            "fraction_over_1.0mm": float(np.mean(displacement > 0.001)),
            "attachment_patch_modified": False,
            "state_kind": (
                "accepted" if accepted
                else "rejected_last_trial_for_diagnostics"),
        }
        trimesh.Trimesh(
            vertices=corrected, faces=body.muscle.surface_faces,
            process=False).export(output / f"{body.name}_corrected.obj")
    active_points = np.asarray([
        corrected_positions[contact.source_body][contact.source_vertex]
        for contact in active.contacts.values()])
    active_rows = [{
        "constraint_id": str(contact.constraint_id),
        "source_body": system.bodies[contact.source_body].name,
        "source_vertex": contact.source_vertex,
        "target_body": system.bodies[contact.target_body].name,
        "target_triangle": list(contact.target_triangle),
        "gap": contact.last_gap,
        "penetration": max(0.0, -contact.last_gap),
        "multiplier": contact.multiplier,
        "scalar_force_magnitude": contact.weight * contact.multiplier,
        "normal": contact.normal.tolist(),
        "weight": contact.weight,
    } for contact in active.contacts.values()]
    with (output / "active_contact_multipliers.json").open("w") as handle:
        json.dump(active_rows, handle, indent=2)
    if len(active_points):
        colors = np.tile(
            np.array([[255, 40, 30, 255]], dtype=np.uint8),
            (len(active_points), 1))
        trimesh.points.PointCloud(
            active_points, colors=colors).export(
                output / "active_contacts.ply")
    low_j = []
    for body, corrected in zip(system.bodies, corrected_positions):
        entries = tet_quality_report(
            body, corrected, active_points, max(safe_j, 0.4))
        low_j.extend(entries)
        export_low_j(
            output / f"{body.name}_low_J.obj", body, corrected,
            [entry["tet_id"] for entry in entries])
    with (output / "rest_correction_report.json").open("w") as handle:
        json.dump({
            "accepted": accepted,
            "acceptance_criteria": {
                "penetration_tolerance":
                    contact_cfg["penetration_tolerance"],
                "complementarity_tolerance":
                    contact_cfg["complementarity_tolerance"],
                "mechanical_tolerance":
                    config["solver"]["gradient_tolerance"],
                "contact_safe_minimum_J": safe_j,
            },
            "displacements": displacement_report,
            "attachment_mode": config["attachment"]["mode"],
            "source_meshes_overwritten": False,
        }, handle, indent=2)
    with (output / "low_J_tet_report.json").open("w") as handle:
        json.dump(low_j, handle, indent=2)
    with (output / "local_remeshing_recommendation.json").open(
            "w") as handle:
        json.dump({
            "recommended": bool(low_j),
            "reason": (
                "rest correction reached the contact-safe J guard in tets "
                "with sub-degree minimum dihedral angles and/or high edge "
                "aspect ratio; further contact-parameter tuning is not "
                "recommended before local retetrahedralization"
                if low_j else "no repeated low-J region detected"),
            "regions": low_j,
            "suggested_scope": [
                "proximal Semitendinosus contact transition",
                "Gracilis-Semitendinosus overlapping surface neighborhood",
            ] if any(
                entry["muscle"] == "L_Semitendinosus"
                for entry in low_j) else [],
            "full_remeshing_implemented": False,
        }, handle, indent=2)
    if accepted:
        corrected_root = Path("preprocessed/rest_contact_corrected")
        corrected_root.mkdir(parents=True, exist_ok=True)
        for body, corrected in zip(system.bodies, corrected_positions):
            np.savez_compressed(
                corrected_root / f"{body.name}_corrected.npz",
                vertices=corrected,
                tetrahedra=body.muscle.tetrahedra,
                surface_faces=body.muscle.surface_faces)
        (output / ".done").write_text("accepted\n")
    else:
        raise RuntimeError(
            "rest correction rejected; see al_iterations.csv and "
            "rest_correction_report.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config/pes_anserinus_multimuscle.yaml")
    parser.add_argument("--tet-dir", default="tet")
    parser.add_argument(
        "--override-muscle", action="append", default=[],
        help="load one derived mesh as MUSCLE=PATH; contact state is rebuilt")
    parser.add_argument(
        "--pair", nargs=2,
        default=["L_Gracilis", "L_Semitendinosus"])
    parser.add_argument(
        "--output",
        default=".bake_outputs/gracilis_semitendinosus_rest_fix")
    parser.add_argument(
        "--attachment-mode",
        choices=["full_patch", "core_hard_transition_soft",
                 "manually_curated"])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
