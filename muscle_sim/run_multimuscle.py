#!/usr/bin/env python3
"""Coupled passive medial-thigh / pes-anserinus muscle experiment."""
import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import trimesh
import yaml

from core.bvhparser import MyBVH
from core.dartHelper import buildFromInfo, saveSkeletonInfo
from tools.bake_isolated_muscle import (
    capture_bones, detect_bvh_tframe, interpolate_transforms)
from viewer.isolated_muscle import (
    attachment_targets, bind_attachment_patches, deformation_gradients,
    fiber_diagnostics,
    harmonic_attachment_update, load_muscle_data, precompute_energy)
from viewer.multimuscle import (
    ContactDiagnostics, MultiMuscleSystem, MuscleBody, RigidSurface,
    compactness_metrics, multi_energy_gradient)


BONE_BODY = {
    "L_Femur": "L_Femur0",
    "L_Tibia_Fibula": "L_Tibia_Fibula0",
    "L_Os_Coxae": "L_Os_Coxae0",
    "L_Patella": "L_Patella0",
}
BONE_MESH_DIR = Path("Zygote_Meshes_251229/Skeleton")
MESH_SCALE = 0.01


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, ContactDiagnostics):
        return {
            "energy": value.energy, "active_count": value.active_count,
            "minimum_gap": value.minimum_gap,
            "maximum_penetration": value.maximum_penetration,
        }
    return value


def boundary_ring_exclusion(body, rings):
    excluded = set(body.fixed_vertices)
    adjacency = {i: set() for i in range(body.vertex_count)}
    for face in body.muscle.surface_faces:
        for i in face:
            adjacency[int(i)].update(int(j) for j in face if j != i)
    frontier = set(excluded)
    for _ in range(rings):
        frontier = set().union(*(adjacency[i] for i in frontier)) - excluded
        excluded.update(frontier)
    body.attachment_exclusion = excluded


def load_bone_rest_surfaces(skeleton, names):
    surfaces = {}
    rest_transforms = {}
    for short_name in names:
        body_name = BONE_BODY.get(short_name, short_name)
        path = BONE_MESH_DIR / f"{short_name}.obj"
        if not path.exists():
            raise FileNotFoundError(f"missing collision bone mesh: {path}")
        mesh = trimesh.load_mesh(path, process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64) * MESH_SCALE
        faces = np.asarray(mesh.faces, dtype=np.int32)
        transform = capture_bones(skeleton, [body_name])[body_name]
        rest_transforms[body_name] = transform
        surfaces[short_name] = RigidSurface(
            short_name, vertices, faces, parent_bone=body_name)
    return surfaces, rest_transforms


def pose_bone_surfaces(rest_surfaces, rest_transforms, frame_transforms):
    posed = {}
    for name, surface in rest_surfaces.items():
        rest = rest_transforms[surface.parent_bone]
        frame = frame_transforms[surface.parent_bone]
        local = (
            surface.vertices - rest[:3, 3]) @ rest[:3, :3]
        vertices = local @ frame[:3, :3].T + frame[:3, 3]
        posed[name] = RigidSurface(
            name, vertices, surface.faces, surface.parent_bone)
    return posed


def generate_sectional_fascia(points, clearance, section_count,
                              radial_count=24):
    """Generate a reproducible non-convex longitudinal enclosing shell."""
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    axis, radial_u, radial_v = vh
    coordinate = (points - center) @ axis
    lo, hi = coordinate.min() - clearance, coordinate.max() + clearance
    sections = np.linspace(lo, hi, section_count)
    spacing = (hi - lo) / max(section_count - 1, 1)
    rings = []
    angles = np.linspace(0.0, 2.0 * np.pi, radial_count, endpoint=False)
    for section in sections:
        mask = np.abs(coordinate - section) <= 2.0 * spacing
        local = points[mask] if np.any(mask) else points[
            np.argsort(np.abs(coordinate - section))[:32]]
        uv = np.column_stack((
            (local - center) @ radial_u,
            (local - center) @ radial_v))
        uv_center = np.median(uv, axis=0)
        centered = uv - uv_center
        # A circular local support is deliberately conservative. An elliptic
        # covariance fit crossed the sharply curving Sartorius path between
        # adjacent rings. Overlapping two-section support windows guarantee
        # that the loft contains the supplied surfaces without reverting to a
        # global convex hull.
        radius = np.max(np.linalg.norm(centered, axis=1)) + clearance
        ellipse = (
            radius * np.column_stack((np.cos(angles), np.sin(angles)))
            + uv_center)
        ring_center = center + section * axis
        rings.append(
            ring_center + ellipse[:, :1] * radial_u
            + ellipse[:, 1:] * radial_v)
    vertices = np.vstack(rings)
    faces = []
    for section in range(section_count - 1):
        for radial in range(radial_count):
            nxt = (radial + 1) % radial_count
            a = section * radial_count + radial
            b = section * radial_count + nxt
            c = (section + 1) * radial_count + radial
            d = (section + 1) * radial_count + nxt
            faces.extend(((a, b, c), (b, d, c)))
    vertices = np.vstack((vertices, rings[0].mean(axis=0),
                          rings[-1].mean(axis=0)))
    first_center, last_center = len(vertices) - 2, len(vertices) - 1
    for radial in range(radial_count):
        nxt = (radial + 1) % radial_count
        faces.append((first_center, nxt, radial))
        base = (section_count - 1) * radial_count
        faces.append((last_center, base + radial, base + nxt))
    return RigidSurface(
        "fascia", vertices, np.asarray(faces, dtype=np.int32))


def fascia_bone_weights(vertices, rest_transforms):
    names = ["L_Os_Coxae0", "L_Femur0", "L_Tibia_Fibula0"]
    centers = np.asarray([rest_transforms[name][:3, 3] for name in names])
    distances = np.linalg.norm(
        vertices[:, None, :] - centers[None, :, :], axis=2)
    inverse = 1.0 / np.maximum(distances, 0.02) ** 2
    return names, inverse / inverse.sum(axis=1, keepdims=True)


def pose_fascia(rest_fascia, names, weights, rest_transforms,
                frame_transforms):
    posed = np.zeros_like(rest_fascia.vertices)
    for column, name in enumerate(names):
        rest, frame = rest_transforms[name], frame_transforms[name]
        local = (
            rest_fascia.vertices - rest[:3, 3]) @ rest[:3, :3]
        transformed = local @ frame[:3, :3].T + frame[:3, 3]
        posed += weights[:, column, None] * transformed
    return RigidSurface("fascia", posed, rest_fascia.faces)


def initial_intersection_report(system, positions):
    report = []
    for first_id, second_id in system.contact_pairs:
        first, second = system.bodies[first_id], system.bodies[second_id]
        x1, x2 = positions[first_id], positions[second_id]
        mesh1 = trimesh.Trimesh(
            vertices=x1, faces=first.muscle.surface_faces, process=False)
        mesh2 = trimesh.Trimesh(
            vertices=x2, faces=second.muscle.surface_faces, process=False)
        ids1, ids2 = (np.unique(first.muscle.surface_faces),
                      np.unique(second.muscle.surface_faces))
        inside1, inside2 = mesh2.contains(x1[ids1]), mesh1.contains(x2[ids2])
        depth1 = (
            trimesh.proximity.closest_point(
                mesh2, x1[ids1[inside1]])[1]
            if np.any(inside1) else np.empty(0))
        depth2 = (
            trimesh.proximity.closest_point(
                mesh1, x2[ids2[inside2]])[1]
            if np.any(inside2) else np.empty(0))
        penetrations = np.r_[depth1, depth2]
        report.append({
            "pair": f"{first.name}|{second.name}",
            "candidate_triangle_pairs": (
                len(first.muscle.surface_faces)
                * len(second.muscle.surface_faces)),
            "intersecting_triangle_pairs": "unavailable_without_python_fcl",
            "penetrated_vertices": int(inside1.sum() + inside2.sum()),
            "maximum_penetration": float(
                penetrations.max()) if len(penetrations) else 0.0,
            "mean_penetration": float(
                penetrations.mean()) if len(penetrations) else 0.0,
            "attachment_region_intersections": int(
                sum(int(vertex) in first.fixed_vertices
                    for vertex in ids1[inside1])
                + sum(int(vertex) in second.fixed_vertices
                      for vertex in ids2[inside2])),
        })
    for body, current in zip(system.bodies, positions):
        surface_ids = np.unique(body.muscle.surface_faces)
        for bone_name in system.body_bones.get(body.name, []):
            bone = system.bones[bone_name]
            mesh = trimesh.Trimesh(
                vertices=bone.vertices, faces=bone.faces, process=False)
            inside = mesh.contains(current[surface_ids])
            depths = (
                trimesh.proximity.closest_point(
                    mesh, current[surface_ids[inside]])[1]
                if np.any(inside) else np.empty(0))
            report.append({
                "pair": f"{body.name}|{bone_name}",
                "candidate_triangle_pairs": (
                    len(body.muscle.surface_faces) * len(bone.faces)),
                "intersecting_triangle_pairs":
                    "unavailable_without_python_fcl",
                "penetrated_vertices": int(inside.sum()),
                "maximum_penetration": float(
                    depths.max()) if len(depths) else 0.0,
                "mean_penetration": float(
                    depths.mean()) if len(depths) else 0.0,
                "attachment_region_intersections": int(sum(
                    int(vertex) in body.fixed_vertices
                    for vertex in surface_ids[inside])),
            })
    if system.fascia is not None:
        shell = trimesh.Trimesh(
            vertices=system.fascia.vertices, faces=system.fascia.faces,
            process=False)
        for body, current in zip(system.bodies, positions):
            surface_ids = np.unique(body.muscle.surface_faces)
            outside = ~shell.contains(current[surface_ids])
            distances = (
                trimesh.proximity.closest_point(
                    shell, current[surface_ids[outside]])[1]
                if np.any(outside) else np.empty(0))
            report.append({
                "pair": f"{body.name}|fascia",
                "candidate_triangle_pairs": (
                    len(body.muscle.surface_faces)
                    * len(system.fascia.faces)),
                "intersecting_triangle_pairs":
                    "unavailable_without_python_fcl",
                "penetrated_vertices": int(outside.sum()),
                "maximum_penetration": float(
                    distances.max()) if len(distances) else 0.0,
                "mean_penetration": float(
                    distances.mean()) if len(distances) else 0.0,
                "attachment_region_intersections": int(sum(
                    int(vertex) in body.fixed_vertices
                    for vertex in surface_ids[outside])),
            })
    return report


def apply_experiment(config, experiment):
    cfg = copy.deepcopy(config)
    cfg["contact"]["fascia_muscle_enabled"] = False
    if experiment == "no_contact":
        cfg["contact"]["bone_muscle_enabled"] = False
        cfg["contact"]["muscle_muscle_enabled"] = False
        cfg["fascia"]["enabled"] = False
    elif experiment == "bone_only":
        cfg["contact"]["muscle_muscle_enabled"] = False
        cfg["fascia"]["enabled"] = False
    elif experiment == "muscle_contact":
        cfg["fascia"]["enabled"] = False
    elif experiment in ("fascia", "cohesion", "wrapping"):
        cfg["contact"]["fascia_muscle_enabled"] = True
        cfg["fascia"]["enabled"] = True
        cfg["cohesion"]["enabled"] = experiment in ("cohesion", "wrapping")
        cfg["wrapping"]["medial_knee_enabled"] = experiment == "wrapping"
    return cfg


def solve_global(initial, target_maps, system, contact_config, solver,
                 enforce_penetration=True):
    fixed_targets = {}
    base_parts = []
    for body, current, targets in zip(
            system.bodies, system.unpack(initial), target_maps):
        moved = harmonic_attachment_update(
            current, targets, body.precomputed)
        base_parts.append(moved)
        fixed_targets.update({
            body.offset + int(vertex): target
            for vertex, target in targets.items()})
    base = system.pack(base_parts)
    fixed_ids = np.asarray(sorted(fixed_targets), dtype=np.int32)
    fixed_values = np.asarray([fixed_targets[int(i)] for i in fixed_ids])
    fixed_mask = np.zeros(system.vertex_count, dtype=bool)
    fixed_mask[fixed_ids] = True
    free_ids = np.where(~fixed_mask)[0]
    base[fixed_ids] = fixed_values
    coordinate_scale = 1e-5
    energy_scale = 1e6

    def unpack_physical(values):
        current = base.copy()
        current[free_ids] = values.reshape(-1, 3)
        current[fixed_ids] = fixed_values
        return current

    evaluations = 0

    def physical_objective(values):
        nonlocal evaluations
        evaluations += 1
        current = unpack_physical(values)
        energy, gradient, _, _ = multi_energy_gradient(
            current, system, contact_config)
        if not np.isfinite(energy):
            return 1e40, np.zeros_like(values)
        return energy, gradient[free_ids].ravel()

    free_flat = base[free_ids].ravel()
    for _ in range(40):
        energy, gradient = physical_objective(free_flat)
        maximum_force = float(np.max(np.linalg.norm(
            gradient.reshape(-1, 3), axis=1)))
        if maximum_force <= float(solver["gradient_tolerance"]):
            break
        direction = -gradient
        directional = float(np.dot(gradient, direction))
        alpha = min(1.0, 1e-4 / max(maximum_force, 1e-12))
        accepted = False
        for _ in range(int(solver["max_line_search_iterations"])):
            trial = free_flat + alpha * direction
            trial_energy, _ = physical_objective(trial)
            if (np.isfinite(trial_energy)
                    and trial_energy <= energy
                    + 1e-4 * alpha * directional):
                free_flat = trial
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break

    def objective(values):
        energy, gradient = physical_objective(
            values * coordinate_scale)
        return (energy * energy_scale,
                gradient * coordinate_scale * energy_scale)

    x0 = free_flat / coordinate_scale
    result = minimize(
        objective, x0, jac=True, method="L-BFGS-B",
        options={"maxiter": int(solver["max_newton_iterations"]),
                 "maxls": int(solver["max_line_search_iterations"]),
                 "gtol": (float(solver["gradient_tolerance"])
                          * coordinate_scale * energy_scale),
                 "ftol": 1e-15, "maxcor": 20})
    lbfgs_vertices = unpack_physical(result.x * coordinate_scale)
    lbfgs_energy, lbfgs_gradient, _, _ = multi_energy_gradient(
        lbfgs_vertices, system, contact_config)
    if (not np.isfinite(lbfgs_energy)
            or np.linalg.norm(lbfgs_gradient[free_ids])
            >= float(solver["gradient_tolerance"])):
        def hessian_vector(values, vector):
            vector_norm = max(float(np.linalg.norm(vector)), 1e-12)
            epsilon = 1e-4 / vector_norm
            _, plus = objective(values + epsilon * vector)
            _, minus = objective(values - epsilon * vector)
            if not (np.all(np.isfinite(plus))
                    and np.all(np.isfinite(minus))):
                return vector
            return (plus - minus) / (2.0 * epsilon)

        newton = minimize(
            objective, result.x, method="trust-krylov", jac=True,
            hessp=hessian_vector,
            options={
                "maxiter": 80,
                "gtol": (
                    energy_scale * coordinate_scale
                    * float(solver["gradient_tolerance"])),
                "initial_trust_radius": 0.1,
                "max_trust_radius": 10.0,
            })
        newton_vertices = unpack_physical(
            newton.x * coordinate_scale)
        newton_energy = multi_energy_gradient(
            newton_vertices, system, contact_config)[0]
        if (np.isfinite(newton_energy)
                and (not np.isfinite(lbfgs_energy)
                     or newton_energy <= lbfgs_energy)):
            result = newton
    solved = unpack_physical(result.x * coordinate_scale)
    energy, gradient, terms, diagnostics = multi_energy_gradient(
        solved, system, contact_config)
    maximum_penetration = max([
        value.maximum_penetration
        for value in diagnostics.values()
        if isinstance(value, ContactDiagnostics)] or [0.0])
    material_minima = [
        value["material"]["minimum_J"]
        for value in diagnostics.values()
        if isinstance(value, dict) and "material" in value]
    if material_minima:
        minimum_j = min(material_minima)
    else:
        minimum_j = min(float(np.min(np.linalg.det(
            deformation_gradients(current, body.muscle,
                                  body.precomputed))))
                        for current, body in zip(
                            system.unpack(solved), system.bodies))
    attachment_error = float(np.max(np.linalg.norm(
        solved[fixed_ids] - fixed_values, axis=1)))
    gradient_norm = float(np.linalg.norm(gradient[free_ids]))
    success = (
        np.isfinite(energy)
        and minimum_j > float(solver["minimum_J"])
        and (not enforce_penetration
             or maximum_penetration <= float(
                 contact_config["maximum_penetration"]))
        and attachment_error <= 1e-10
        and gradient_norm <= float(solver["gradient_tolerance"]))
    info = {
        "success": success, "optimizer_success": bool(result.success),
        "message": str(result.message), "iterations": int(result.nit),
        "evaluations": evaluations, "energy": energy,
        "gradient_norm": gradient_norm, "minimum_J": minimum_j,
        "maximum_penetration": maximum_penetration,
        "attachment_error": attachment_error, "terms": terms,
        "contacts": diagnostics,
    }
    return solved, info


def export_frame(output, frame, system, positions, info, compactness):
    output.mkdir(parents=True, exist_ok=True)
    contact_points, closest_points, normals = [], [], []
    for value in info["contacts"].values():
        if isinstance(value, ContactDiagnostics):
            contact_points.extend(value.points)
            closest_points.extend(value.closest_points)
            normals.extend(value.normals)
    np.savez_compressed(
        output / f"frame_{frame:04d}_contacts.npz",
        contact_points=np.asarray(contact_points),
        closest_points=np.asarray(closest_points),
        contact_normals=np.asarray(normals))
    scene = trimesh.Scene()
    for body, current in zip(system.bodies, system.unpack(positions)):
        scene.add_geometry(trimesh.Trimesh(
            vertices=current, faces=body.muscle.surface_faces,
            process=False), node_name=body.name, geom_name=body.name)
    for name, bone in system.bones.items():
        scene.add_geometry(trimesh.Trimesh(
            vertices=bone.vertices, faces=bone.faces, process=False),
            node_name=f"bone_{name}", geom_name=f"bone_{name}")
    if system.fascia is not None:
        scene.add_geometry(trimesh.Trimesh(
            vertices=system.fascia.vertices, faces=system.fascia.faces,
            process=False), node_name="fascia", geom_name="fascia")
    scene.export(output / f"frame_{frame:04d}.glb")
    with (output / f"frame_{frame:04d}_metrics.json").open("w") as handle:
        json.dump(json_ready({
            "solve": info, "compactness": compactness}), handle, indent=2)


def run(args):
    with Path(args.config).open() as handle:
        config = apply_experiment(yaml.safe_load(handle), args.experiment)
    if args.contact_solver is not None:
        config["contact"]["solver"] = args.contact_solver
    if (args.experiment != "no_contact"
            and config["contact"].get("solver")
            == "augmented_lagrangian"):
        missing = [
            name for name in config["muscles"]["selected"]
            if not (Path("preprocessed/rest_contact_corrected")
                    / f"{name}_corrected.npz").exists()]
        if missing:
            raise RuntimeError(
                "augmented-Lagrangian pose contact is gated on an accepted "
                "corrected rest state; missing: " + ", ".join(missing))
    with Path("config/isolated_muscle.yaml").open() as handle:
        isolated_config = yaml.safe_load(handle)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    bodies, reports = [], {}
    selected = config["muscles"]["selected"]
    for name in selected:
        path = Path(args.tet_dir) / f"{name}_tet.npz"
        if not path.exists():
            print(f"WARNING: requested muscle missing: {name}")
            continue
        muscle, report = load_muscle_data(path)
        precomputed = precompute_energy(
            muscle, nearby_radius=float(
                isolated_config["fiber"]["nearby_radius"]))
        body = MuscleBody(
            muscle, precomputed, copy.deepcopy(isolated_config))
        bodies.append(body)
        reports[name] = report
    if len(bodies) < 2:
        raise RuntimeError("fewer than two requested muscles are valid")

    name_to_id = {body.name: index for index, body in enumerate(bodies)}
    pairs = [
        (name_to_id[a], name_to_id[b])
        for a, b in config["contact_pairs"]
        if a in name_to_id and b in name_to_id]
    system = MultiMuscleSystem(bodies, pairs)
    for body in bodies:
        boundary_ring_exclusion(
            body, int(config["contact"]["attachment_exclusion_rings"]))

    skeleton_info, root_name, bvh_info, _, _, _ = saveSkeletonInfo(
        "data/zygote_skel.xml")
    skeleton = buildFromInfo(skeleton_info, root_name)
    skeleton.setPositions(np.zeros(skeleton.getNumDofs()))
    attachment_bones = sorted({
        patch.bone_name for body in bodies
        for patch in body.muscle.attachment_patches})
    collision_short_names = sorted({
        name for body in bodies
        for name in config["muscles"]["collision_bones"][body.name]})
    collision_body_names = sorted(
        {BONE_BODY[name] for name in collision_short_names})
    transform_names = sorted(
        set(attachment_bones + collision_body_names
            + ["L_Os_Coxae0", "L_Femur0", "L_Tibia_Fibula0"]))
    rest_transforms = capture_bones(skeleton, transform_names)
    for body in bodies:
        bind_attachment_patches(
            body.muscle.attachment_patches, body.muscle.vertices,
            rest_transforms)
    rest_bone_surfaces, bone_rest_transforms = load_bone_rest_surfaces(
        skeleton, collision_short_names)
    system.body_bones = {
        body.name: config["muscles"]["collision_bones"][body.name]
        for body in bodies}
    system.bones = rest_bone_surfaces

    rest_positions = [body.muscle.vertices.copy() for body in bodies]
    rest_fascia = fascia_names = fascia_weights = None
    if config["fascia"]["enabled"]:
        rest_fascia = generate_sectional_fascia(
            np.vstack([
                x[np.unique(body.muscle.surface_faces)]
                for x, body in zip(rest_positions, bodies)]),
            float(config["fascia"]["clearance"]),
            int(config["fascia"]["section_count"]))
        fascia_names, fascia_weights = fascia_bone_weights(
            rest_fascia.vertices, rest_transforms)
        system.fascia = rest_fascia

    with (output / "input_validation.json").open("w") as handle:
        json.dump(json_ready(reports), handle, indent=2)
    intersections = initial_intersection_report(system, rest_positions)
    with (output / "initial_intersections.json").open("w") as handle:
        json.dump(intersections, handle, indent=2)
    if args.inspect_only:
        placeholder = {
            "contacts": {}, "energy": 0.0, "gradient_norm": 0.0,
            "minimum_J": 1.0, "maximum_penetration": max(
                (row["maximum_penetration"] for row in intersections),
                default=0.0),
            "attachment_error": 0.0, "terms": {},
            "success": False, "message": "inspection only",
        }
        fascia_volume = (
            abs(trimesh.Trimesh(
                vertices=system.fascia.vertices,
                faces=system.fascia.faces, process=False).volume)
            if system.fascia is not None else None)
        compactness = compactness_metrics(
            rest_positions, bodies,
            float(config["debug"]["neighbor_distance"]), fascia_volume)
        export_frame(
            output, 0, system, system.pack(rest_positions),
            placeholder, compactness)
        return

    contact_requested = (
        config["contact"]["bone_muscle_enabled"]
        or config["contact"]["muscle_muscle_enabled"]
        or config["contact"].get("fascia_muscle_enabled", False))
    if contact_requested:
        rest_targets = [
            attachment_targets(
                body.muscle.attachment_patches, rest_transforms)
            for body in bodies]
        depenetration_rows = []
        full_stiffness = float(config["contact"]["stiffness"])
        for scale in (0.01, 0.03, 0.1, 0.3, 1.0):
            ramp_contact = copy.deepcopy(config["contact"])
            ramp_contact["stiffness"] = full_stiffness * scale
            trial, depenetration = solve_global(
                system.pack(rest_positions), rest_targets, system,
                ramp_contact, config["solver"],
                enforce_penetration=(scale == 1.0))
            depenetration_rows.append({
                "stiffness_scale": scale,
                "success": depenetration["success"],
                "minimum_J": depenetration["minimum_J"],
                "maximum_penetration":
                    depenetration["maximum_penetration"],
                "gradient_norm": depenetration["gradient_norm"],
                "attachment_error": depenetration["attachment_error"],
                "message": depenetration["message"],
            })
            if not (np.isfinite(depenetration["energy"])
                    and depenetration["minimum_J"]
                    > float(config["solver"]["minimum_J"])):
                break
            rest_positions = system.unpack(trial)
        with (output / "depenetration_report.json").open("w") as handle:
            json.dump(json_ready(depenetration_rows), handle, indent=2)
        if not depenetration_rows[-1]["success"]:
            corrected = system.pack(rest_positions)
            for body, current in zip(bodies, rest_positions):
                trimesh.Trimesh(
                    vertices=current,
                    faces=body.muscle.surface_faces,
                    process=False).export(
                        output / f"depenetrated_{body.name}.obj")
            raise RuntimeError(
                "rest-pose depenetration did not satisfy the configured "
                "penetration/J/gradient tolerances; see "
                f"{output / 'depenetration_report.json'}")

    motion = MyBVH(
        args.bvh, bvh_info, skeleton,
        T_frame=detect_bvh_tframe(args.bvh))
    end = min(args.end_frame, len(motion.mocap_refs) - 1)
    global_positions = system.pack(rest_positions)
    previous_transforms = rest_transforms
    baked = {body.name: [] for body in bodies}
    frame_numbers, summary, muscle_summary, pair_summary = [], [], [], []

    for frame in range(args.start_frame, end + 1):
        skeleton.setPositions(motion.mocap_refs[frame])
        frame_transforms = capture_bones(skeleton, transform_names)
        initial_substeps = int(config["solver"]["initial_substeps"])
        pending = [
            index / initial_substeps
            for index in range(1, initial_substeps + 1)]
        last_fraction = 0.0
        accepted = 0
        while pending:
            fraction = pending.pop(0)
            transforms = interpolate_transforms(
                previous_transforms, frame_transforms, fraction)
            system.bones = pose_bone_surfaces(
                rest_bone_surfaces, bone_rest_transforms, transforms)
            if rest_fascia is not None:
                system.fascia = pose_fascia(
                    rest_fascia, fascia_names, fascia_weights,
                    rest_transforms, transforms)
            targets = [
                attachment_targets(
                    body.muscle.attachment_patches, transforms)
                for body in bodies]
            trial, info = solve_global(
                global_positions, targets, system, config["contact"],
                config["solver"])
            print(
                f"frame={frame} fraction={fraction:.6f} "
                f"success={info['success']} E={info['energy']:.6e} "
                f"|g|={info['gradient_norm']:.3e} "
                f"minJ={info['minimum_J']:.5f} "
                f"penetration={info['maximum_penetration']:.6e} "
                f"attach={info['attachment_error']:.3e}")
            if info["success"]:
                global_positions = trial
                last_fraction = fraction
                accepted += 1
                continue
            increment = fraction - last_fraction
            if (not config["solver"]["adaptive_pose_subdivision"]
                    or 0.5 * increment < float(
                        config["solver"]["minimum_pose_fraction"])):
                raise RuntimeError(
                    f"frame {frame} failed at fraction {fraction}: "
                    f"{info['message']}; penetration="
                    f"{info['maximum_penetration']}")
            midpoint = 0.5 * (last_fraction + fraction)
            pending.insert(0, fraction)
            pending.insert(0, midpoint)
            print(f"retry [{last_fraction:.6f},{fraction:.6f}] "
                  f"via {midpoint:.6f}")

        positions = system.unpack(global_positions)
        fascia_volume = None
        if system.fascia is not None:
            fascia_volume = abs(trimesh.Trimesh(
                vertices=system.fascia.vertices,
                faces=system.fascia.faces, process=False).volume)
        compactness = compactness_metrics(
            positions, bodies,
            float(config["debug"]["neighbor_distance"]), fascia_volume)
        export_frame(
            output, frame, system, global_positions, info, compactness)
        for body, current in zip(bodies, positions):
            baked[body.name].append(current.astype(np.float32))
            material = info["contacts"][body.name]["material"]
            fibers = fiber_diagnostics(
                current, body.muscle, body.precomputed)
            F = deformation_gradients(
                current, body.muscle, body.precomputed)
            valid = body.precomputed["fiber_valid"]
            stretches = np.linalg.norm(np.einsum(
                "nij,nj->ni", F[valid],
                body.precomputed["fiber_directions"][valid]), axis=1)
            muscle_summary.append({
                "frame": frame, "experiment": args.experiment,
                "muscle": body.name,
                "minimum_J": material["minimum_J"],
                "maximum_J": material["maximum_J"],
                "volume_ratio": material["volume_ratio"],
                "matrix_energy": material["matrix"],
                "volume_energy": material["volume"],
                "fiber_energy": material["fiber"],
                "fiber_bending_energy": material["bending"],
                "minimum_fiber_stretch": float(np.min(stretches)),
                "maximum_fiber_stretch": float(np.max(stretches)),
                "maximum_fiber_turn": max(
                    value["maximum_turning_angle"] for value in fibers),
                "attachment_error": info["attachment_error"],
            })
        for first_id, second_id in pairs:
            first, second = bodies[first_id], bodies[second_id]
            x1 = positions[first_id][
                np.unique(first.muscle.surface_faces)]
            x2 = positions[second_id][
                np.unique(second.muscle.surface_faces)]
            d12 = cKDTree(x2).query(x1)[0]
            d21 = cKDTree(x1).query(x2)[0]
            contact_diag = info["contacts"].get(
                f"{first.name}|{second.name}", ContactDiagnostics())
            pair_summary.append({
                "frame": frame, "experiment": args.experiment,
                "first": first.name, "second": second.name,
                "minimum_surface_distance": min(
                    float(np.min(d12)), float(np.min(d21))),
                "mean_nearest_surface_distance": float(
                    0.5 * (np.mean(d12) + np.mean(d21))),
                "contact_area_proxy": contact_diag.active_count,
                "maximum_penetration":
                    contact_diag.maximum_penetration,
                "relative_tangential_sliding":
                    "not_available_without_rest_neighbor_tracking",
            })
        frame_numbers.append(frame)
        summary.append({
            "frame": frame, "experiment": args.experiment,
            "accepted_substeps": accepted,
            "minimum_J": info["minimum_J"],
            "maximum_penetration": info["maximum_penetration"],
            "attachment_error": info["attachment_error"],
            **compactness,
        })
        previous_transforms = frame_transforms

    for body in bodies:
        np.savez_compressed(
            output / f"{body.name}_chunk_0000.npz",
            frames=np.asarray(frame_numbers, dtype=np.int32),
            positions=np.asarray(baked[body.name]))
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for filename, rows in (
            ("muscle_summary.csv", muscle_summary),
            ("pair_summary.csv", pair_summary)):
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
    (output / ".done").write_text("complete\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config/pes_anserinus_multimuscle.yaml")
    parser.add_argument(
        "--bvh", default="data/motion/left_thigh_quasistatic_5pose.bvh")
    parser.add_argument("--tet-dir", default="tet")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=3)
    parser.add_argument(
        "--experiment",
        choices=["no_contact", "bone_only", "muscle_contact",
                 "fascia", "cohesion", "wrapping"],
        default="fascia")
    parser.add_argument(
        "--contact-solver",
        choices=["augmented_lagrangian", "semismooth_newton"])
    parser.add_argument(
        "--output", default=".bake_outputs/pes_anserinus_fascia")
    parser.add_argument("--inspect-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
