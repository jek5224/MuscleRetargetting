"""Produce the three-pose, three-muscle visible MVP preview."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from core.bvhparser import MyBVH
from tools.bake_contour_sim import (
    _detect_bvh_tframe, build_bone_trimeshes, compute_bone_rest_transforms,
    compute_lbs_positions, compute_multibone_lbs, extract_surface_triangles,
    load_bone_meshes, load_muscle, load_skeleton)
from muscle_sim.mvp.core import (
    centerline_from_endpoints, guide_volume, orient_tets,
    pathological_tets, pose_substeps, project_bone_collision,
    project_global_volume, project_pair_collision, signed_tet_volumes,
    sweep_deform, wrap_polyline_capsule)
from muscle_sim.mvp.io import write_json, write_obj, write_scene_obj
from muscle_sim.mvp.scope import (
    build_neighbor_pairs, discover_upper_leg, infer_attachment_sets,
    wrapping_regions)


def render_snapshot(path, muscles, posed, guides, bones, label,
                    camera_bounds, transparent=False, show_bones=True,
                    debug=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    palette = ("#e45756", "#f2cf5b", "#54a24b", "#4c78a8", "#b279a2",
               "#ff9da6", "#9d755d", "#72b7b2", "#f58518", "#bab0ac")
    figure = plt.figure(figsize=(8, 8), dpi=140)
    axes = figure.add_subplot(111, projection="3d")
    all_points = []
    for bone in bones if show_bones else []:
        faces = bone.faces[::max(1, len(bone.faces) // 1800)]
        collection = Poly3DCollection(
            bone.vertices[faces], facecolor="#c8c8c8", edgecolor="none",
            alpha=.24)
        axes.add_collection3d(collection)
    for muscle_index, muscle in enumerate(muscles):
        name = muscle["name"]
        vertices = posed[name]
        faces = muscle["surface_faces"]
        collection = Poly3DCollection(
            vertices[faces], facecolor=palette[muscle_index % len(palette)],
            edgecolor="none", alpha=.28 if transparent else .88)
        axes.add_collection3d(collection)
        guide = guides[name]
        if transparent or debug:
            axes.plot(guide[:, 0], guide[:, 1], guide[:, 2],
                      color="black", linewidth=1.2)
        all_points.extend((vertices, guide))
    low, high = camera_bounds
    center, radius = .5 * (low + high), .52 * np.max(high - low)
    axes.set_xlim(center[0] - radius, center[0] + radius)
    axes.set_ylim(center[1] - radius, center[1] + radius)
    axes.set_zlim(center[2] - radius, center[2] + radius)
    axes.view_init(elev=12, azim=-72)
    axes.set_title(label.replace("_", " ").title())
    axes.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, transparent=False)
    plt.close(figure)


def endpoint_sets(vertices, fixed):
    vertices = np.asarray(vertices)
    candidates = np.asarray(fixed, dtype=int)
    if len(candidates) < 2:
        candidates = np.arange(len(vertices))
    cloud = vertices[candidates]
    _, _, vt = np.linalg.svd(cloud - cloud.mean(axis=0), full_matrices=False)
    coordinate = (cloud - cloud.mean(axis=0)) @ vt[0]
    count = max(1, min(16, len(candidates) // 4))
    order = np.argsort(coordinate)
    return candidates[order[:count]], candidates[order[-count:]]


def fallback_bindings(muscle, skeleton, crossing):
    rest = muscle["rest_vertices"]
    start, end = muscle["start_ids"], muscle["end_ids"]
    start_center, end_center = rest[start].mean(axis=0), rest[end].mean(axis=0)
    axis = end_center - start_center
    u = np.clip(((rest - start_center) @ axis) / max(
        np.dot(axis, axis), 1e-12), 0., 1.)
    end_body = "L_Tibia_Fibula0" if crossing in (
        "KNEE_ONLY", "HIP_AND_KNEE") else "L_Femur0"
    body_names = ("L_Os_Coxae0", end_body)
    transforms = {}
    for name in body_names:
        node = skeleton.getBodyNode(name)
        if node is None:
            return None
        world = node.getWorldTransform()
        transforms[name] = (
            world.rotation().copy(), world.translation().copy())
    bindings = []
    for point, parameter in zip(rest, u):
        weights = []
        for name, weight in (
                (body_names[0], 1. - parameter),
                (body_names[1], parameter)):
            rotation, translation = transforms[name]
            weights.append((name, float(weight), rotation, translation))
        bindings.append((point.copy(), weights))
    muscle["fixed_vertices"] = sorted(set(map(int, np.r_[start, end])))
    return bindings


def deformation_diagnostics(rest, current, tets, fixed, target):
    rest_volume = np.sum(np.abs(signed_tet_volumes(rest, tets)))
    current_volumes = signed_tet_volumes(current, tets)
    current_volume = np.sum(np.abs(current_volumes))
    rest_tet = np.maximum(np.abs(signed_tet_volumes(rest, tets)), 1e-18)
    jacobian = current_volumes / rest_tet
    attachment_error = (float(np.max(np.linalg.norm(
        current[np.asarray(fixed)] - target[np.asarray(fixed)], axis=1)))
        if len(fixed) else 0.)
    return {
        "minimum_J": float(np.min(jacobian)),
        "number_of_inverted_tets": int(np.sum(jacobian <= 0.)),
        "volume_ratio": float(current_volume / max(rest_volume, 1e-18)),
        "attachment_error_m": attachment_error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--muscles", nargs="+")
    parser.add_argument("--all-left-upper-leg-muscles", action="store_true")
    parser.add_argument("--frames", nargs="+",
                        default=["neutral", "moderate_flexion",
                                 "deep_flexion"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())[
        "muscle_animation_mvp"]
    manifest = json.loads(Path(args.manifest).read_text())
    skeleton, bvh_info, _ = load_skeleton()
    motion = MyBVH(manifest["bvh_path"], bvh_info, skeleton,
                   T_frame=_detect_bvh_tframe(manifest["bvh_path"]))
    bone_data = load_bone_meshes("L")
    bone_rest = compute_bone_rest_transforms(skeleton, bone_data)
    scope_rows = discover_upper_leg(manifest, config["muscle_root"])
    write_json(Path(args.output) / "all_left_upper_leg_muscles.json",
               {"muscles": scope_rows})
    if args.all_left_upper_leg_muscles:
        requested = [row["canonical_name"] for row in scope_rows
                     if row["preview_eligibility"]]
    elif args.muscles:
        requested = [name.replace("Left_", "L_") for name in args.muscles]
    else:
        raise ValueError(
            "provide --all-left-upper-leg-muscles or --muscles")
    muscles = []
    muscle_status = {}
    scope_by_name = {row["canonical_name"]: row for row in scope_rows}
    skeleton.setPositions(np.zeros(skeleton.getNumDofs()))
    for name in requested:
        try:
            muscle = load_muscle(config["muscle_root"], name)
            if muscle is None:
                raise ValueError("missing MVP tet mesh")
            muscle["tetrahedra"] = orient_tets(
                muscle["vertices"], muscle["tetrahedra"])
            muscle["surface_faces"] = extract_surface_triangles(
                muscle["tetrahedra"])
            if not len(muscle["surface_faces"]):
                raise ValueError("no extractable tet boundary")
            muscle["surface_ids"] = np.unique(muscle["surface_faces"])
            inferred = False
            if len(muscle["fixed_vertices"]) < 2:
                muscle["start_ids"], muscle["end_ids"] = (
                    infer_attachment_sets(
                        muscle["rest_vertices"], muscle["surface_ids"]))
                muscle["fixed_vertices"] = sorted(set(map(
                    int, np.r_[muscle["start_ids"], muscle["end_ids"]])))
                inferred = True
            else:
                muscle["start_ids"], muscle["end_ids"] = endpoint_sets(
                    muscle["rest_vertices"], muscle["fixed_vertices"])
            muscle["bindings"] = compute_multibone_lbs(muscle, skeleton)
            if muscle["bindings"] is None:
                muscle["bindings"] = fallback_bindings(
                    muscle, skeleton,
                    scope_by_name[name]["joint_crossing"])
                inferred = True
            if muscle["bindings"] is None:
                raise ValueError("attachment inference failed")
            muscle["rest_guide"] = centerline_from_endpoints(
                muscle["rest_vertices"], muscle["start_ids"],
                muscle["end_ids"])
            poor, _ = pathological_tets(
                muscle["rest_vertices"], muscle["tetrahedra"])
            muscle["regularized_tets"] = poor
            muscle["wrapping_regions"] = wrapping_regions(name)
            muscles.append(muscle)
            muscle_status[name] = {
                "status": "APPROXIMATE" if inferred else "SUCCESS",
                "reason": ("AUTOMATIC_ATTACHMENT_FALLBACK"
                           if inferred else None)}
        except Exception as error:
            muscle_status[name] = {
                "status": "FAILED",
                "reason": f"{type(error).__name__}: {error}"}
    if not muscles:
        raise ValueError("no requested muscle could be prepared")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    neighbor_pairs = build_neighbor_pairs(muscles)
    write_json(output / "muscle_neighbor_pairs.json", {
        "method": "rest_aabb_overlap_with_padding",
        "pairs": [{"muscle_a": a, "muscle_b": b}
                  for a, b in neighbor_pairs]})
    rest_points = np.vstack([muscle["rest_vertices"] for muscle in muscles])
    camera_bounds = (rest_points.min(axis=0), rest_points.max(axis=0))
    pose_diagnostics, cache = [], {}
    previous = {muscle["name"]: muscle["rest_vertices"].copy()
                for muscle in muscles}
    for label in args.frames:
        frame = int(config["preview_frames"][label])
        frame = min(frame, len(motion.mocap_refs) - 1)
        skeleton.setPositions(motion.mocap_refs[frame].copy())
        bones = build_bone_trimeshes(skeleton, bone_data, bone_rest)
        bone_vertices = (np.vstack([bone.vertices for bone in bones])
                         if bones else np.empty((0, 3)))
        knee_node = skeleton.getBodyNode("L_Tibia_Fibula0")
        knee_center = (knee_node.getWorldTransform().translation()
                       if knee_node is not None else bone_vertices.mean(axis=0))
        angle = {"neutral": 0., "moderate_flexion": 60.,
                 "deep_flexion": 90.}.get(label, 0.)
        substeps = pose_substeps(
            angle, config["knee_flexion"]["moderate_angle_degrees"],
            config["knee_flexion"]["deep_angle_degrees"],
            config["frame_substeps"],
            config["knee_flexion"]["deep_flexion_substeps"])
        posed = {}
        guides = {}
        collision_rows = {}
        for muscle in muscles:
            target = compute_lbs_positions(
                muscle["bindings"], skeleton, len(muscle["vertices"]))
            fixed = muscle["fixed_vertices"]
            rest_guide = muscle["rest_guide"]
            posed_guide = centerline_from_endpoints(
                target, muscle["start_ids"], muscle["end_ids"])
            knee_regions = {
                region for region in muscle["wrapping_regions"]
                if region.endswith("_knee")}
            if (knee_regions and angle >=
                    config["knee_flexion"]["moderate_angle_degrees"]):
                direction = (
                    np.array([-1., 0., 0.])
                    if "medial_knee" in knee_regions else
                    np.array([1., 0., 0.])
                    if "lateral_knee" in knee_regions else
                    np.array([0., 0., 1.])
                    if "anterior_knee" in knee_regions else
                    np.array([0., 0., -1.]))
                posed_guide = wrap_polyline_capsule(
                    posed_guide, knee_center, config["wrapping_radius_m"],
                    direction)
            weight = config["centerline_weight"] * (
                config["knee_flexion"]["centerline_weight_multiplier"]
                if angle >= config["knee_flexion"]["deep_angle_degrees"]
                else 1.)
            bone_moves = bone_depth = 0
            x = muscle["rest_vertices"].copy()
            for step in range(substeps):
                alpha = (step + 1.) / substeps
                intermediate_guide = (
                    (1. - alpha) * rest_guide + alpha * posed_guide)
                intermediate_target = (
                    (1. - alpha) * muscle["rest_vertices"] + alpha * target)
                x = sweep_deform(
                    muscle["rest_vertices"], rest_guide,
                    intermediate_guide, intermediate_target, fixed)
                x[np.asarray(fixed, dtype=int)] = target[
                    np.asarray(fixed, dtype=int)]
                for _ in range(config["collision_iterations"]):
                    x, moved, depth = project_bone_collision(
                        x, muscle["surface_ids"], bone_vertices,
                        config["bone_collision_margin_m"], fixed)
                    bone_moves += moved
                    bone_depth = max(bone_depth, depth)
                    x[np.asarray(fixed, dtype=int)] = target[
                        np.asarray(fixed, dtype=int)]
            deviation = x - target
            distance = np.linalg.norm(deviation, axis=1)
            limit = config["maximum_sweep_deviation_from_lbs_m"]
            excessive = distance > limit
            if np.any(excessive):
                x[excessive] = (
                    target[excessive] + deviation[excessive]
                    * (limit / distance[excessive])[:, None])
                x[np.asarray(fixed, dtype=int)] = target[
                    np.asarray(fixed, dtype=int)]
            posed[muscle["name"]] = x
            guides[muscle["name"]] = posed_guide
            collision_rows[muscle["name"]] = {
                "bone_penetration_count": bone_moves,
                "bone_penetration_maximum_m": bone_depth}
        pair_count = pair_depth = 0
        muscles_by_name = {muscle["name"]: muscle for muscle in muscles}
        for first, second in neighbor_pairs:
                a, b = muscles_by_name[first], muscles_by_name[second]
                xa, xb, count, depth = project_pair_collision(
                    posed[first], a["surface_ids"],
                    posed[second], b["surface_ids"],
                    config["muscle_spacing_m"])
                posed[first], posed[second] = xa, xb
                pair_count += count
                pair_depth = max(pair_depth, depth)
        objects = []
        angles = np.linspace(0., 2. * np.pi, 48, endpoint=False)
        radius = config["wrapping_radius_m"]
        debug_points = np.vstack([
            knee_center + radius * np.column_stack(
                (np.cos(angles), np.sin(angles), np.zeros_like(angles))),
            knee_center + radius * np.column_stack(
                (np.cos(angles), np.zeros_like(angles), np.sin(angles))),
            knee_center + radius * np.column_stack(
                (np.zeros_like(angles), np.cos(angles), np.sin(angles)))])
        for muscle in muscles:
            name, x = muscle["name"], posed[muscle["name"]]
            target = compute_lbs_positions(
                muscle["bindings"], skeleton, len(x))
            fixed = muscle["fixed_vertices"]
            if len(fixed):
                x[np.asarray(fixed, dtype=int)] = target[
                    np.asarray(fixed, dtype=int)]
            write_obj(output / label / f"{name}.obj", x,
                      muscle["surface_faces"], object_name=name)
            guide = guides[name]
            write_obj(output / label / f"{name}_centerline.obj", guide,
                      lines=[np.arange(len(guide))],
                      object_name=name + "_centerline")
            objects.append((name, x, muscle["surface_faces"], []))
            objects.append((name + "_centerline", guide, [],
                            [np.arange(len(guide))]))
            diagnostics = deformation_diagnostics(
                muscle["rest_vertices"], x, muscle["tetrahedra"],
                fixed, target)
            guide_lengths = np.linalg.norm(np.diff(guide, axis=0), axis=1)
            rest_lengths = np.linalg.norm(
                np.diff(muscle["rest_guide"], axis=0), axis=1)
            diagnostics.update(collision_rows[name])
            diagnostics.update({
                "muscle": name,
                "centerline_length_ratio": float(
                    guide_lengths.sum() / max(rest_lengths.sum(), 1e-12)),
                "solver_iterations": int(substeps),
                "regularized_tet_count": int(
                    len(muscle["regularized_tets"])),
                "approximate_frame": True})
            pose_diagnostics.append({
                "pose": label, "bvh_frame": frame, **diagnostics})
            previous[name] = x.copy()
            cache.setdefault(name, []).append(x.copy())
        objects.extend((f"bone_{i}", bone.vertices, bone.faces, [])
                       for i, bone in enumerate(bones))
        write_scene_obj(
            output / label / "preview_all_muscles_scene.obj", objects)
        write_obj(output / label / "collision_debug.obj",
                  np.asarray(debug_points).reshape((-1, 3)))
        write_scene_obj(output / label / "bones.obj", [
            (f"bone_{i}", bone.vertices, bone.faces, [])
            for i, bone in enumerate(bones)])
        render_snapshot(
            output / label / "preview_all_muscles.png", muscles, posed,
            guides, bones, label, camera_bounds)
        render_snapshot(
            output / label / "preview_all_muscles_transparent.png",
            muscles, posed, guides, bones, label, camera_bounds,
            transparent=True)
        render_snapshot(
            output / label / "preview_bones_plus_muscles.png",
            muscles, posed, guides, bones, label, camera_bounds,
            transparent=True, show_bones=True)
        render_snapshot(
            output / label / "preview_muscles_without_bones.png",
            muscles, posed, guides, bones, label, camera_bounds,
            show_bones=False)
        render_snapshot(
            output / label / "preview_collision_debug.png",
            muscles, posed, guides, bones, label, camera_bounds,
            transparent=True, debug=True)
        current_rows = [
            row for row in pose_diagnostics
            if row.get("pose") == label and "muscle" in row]
        pose_diagnostics.append({
            "pose": label, "bvh_frame": frame,
            "muscle_pair_proximity_projection_count": pair_count,
            "muscle_pair_maximum_correction_m": pair_depth,
            "knee_flexion_degrees": angle,
            "muscles_discovered": requested,
            "muscles_successfully_exported": [
                name for name in requested if muscle_status.get(
                    name, {}).get("status") == "SUCCESS"],
            "approximate_muscles": [
                name for name in requested if muscle_status.get(
                    name, {}).get("status") == "APPROXIMATE"],
            "skipped_muscles": [
                name for name in requested if muscle_status.get(
                    name, {}).get("status") == "SKIPPED"],
            "failed_muscles": [
                name for name in requested if muscle_status.get(
                    name, {}).get("status") == "FAILED"],
            "total_inverted_tets": int(sum(
                row["number_of_inverted_tets"] for row in current_rows)),
            "muscles_with_visible_bone_penetration": [
                row["muscle"] for row in current_rows
                if row["bone_penetration_count"] > 0],
            "maximum_attachment_error_m": max(
                (row["attachment_error_m"] for row in current_rows),
                default=0.)})
    if config["output"]["export_npz_cache"]:
        np.savez_compressed(output / "preview_all_muscles_cache.npz",
                            **{name: np.asarray(frames)
                               for name, frames in cache.items()})
    write_json(output / "mvp_preview_summary.json", {
        "branch": "muscle_animation_mvp",
        "bvh": manifest["bvh_path"],
        "muscle_root": config["muscle_root"],
        "requested_muscles": requested,
        "muscle_status": muscle_status,
        "eligible_muscle_count": len(requested),
        "exported_muscle_count": len(muscles),
        "poses": pose_diagnostics,
        "assumptions": [
            "Simple tet assets are used directly.",
            "LBS supplies posed attachment targets and robust initialization.",
            "PCA/end-patch centerlines guide the belly without requiring fibers.",
            "Bone and muscle collision are approximate normal projections.",
            "Frames are visual MVP outputs, not strict FEM equilibria."]})
    print(output)


if __name__ == "__main__":
    main()
