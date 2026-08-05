"""Rest-contact feasibility and attachment-footprint diagnostics."""
from collections import defaultdict
import heapq

import numpy as np
import trimesh


CLASS_NAMES = (
    "FREE_FREE",
    "CONSTRAINED_FREE",
    "FREE_CONSTRAINED",
    "CONSTRAINED_CONSTRAINED_SAME_BONE",
    "CONSTRAINED_CONSTRAINED_DIFFERENT_BONES",
)


def patch_membership(body):
    membership = {}
    for patch_id, patch in enumerate(body.muscle.attachment_patches):
        for vertex in patch.vertex_ids:
            membership[int(vertex)] = {
                "patch_id": patch_id, "bone_id": patch.bone_name}
    return membership


def surface_adjacency(vertices, faces):
    adjacency = [dict() for _ in range(len(vertices))]
    for face in faces:
        for edge in ((face[0], face[1]), (face[1], face[2]),
                     (face[2], face[0])):
            a, b = int(edge[0]), int(edge[1])
            length = float(np.linalg.norm(vertices[a] - vertices[b]))
            old = adjacency[a].get(b, np.inf)
            adjacency[a][b] = min(old, length)
            adjacency[b][a] = min(adjacency[b].get(a, np.inf), length)
    return adjacency


def patch_boundary(patch_vertices, adjacency):
    patch = set(int(vertex) for vertex in patch_vertices)
    boundary = sorted(
        vertex for vertex in patch
        if any(neighbor not in patch for neighbor in adjacency[vertex]))
    # Some tet files duplicate the cap seam, so the cap is a disconnected
    # surface component and has no graph neighbor outside the patch. In that
    # representation every seam vertex is a footprint boundary vertex.
    return boundary if boundary else sorted(patch)


def graph_distances(adjacency, sources, allowed=None):
    distance = np.full(len(adjacency), np.inf)
    queue = []
    for source in sources:
        distance[int(source)] = 0.0
        heapq.heappush(queue, (0.0, int(source)))
    while queue:
        value, vertex = heapq.heappop(queue)
        if value != distance[vertex]:
            continue
        for neighbor, length in adjacency[vertex].items():
            if allowed is not None and neighbor not in allowed:
                continue
            candidate = value + length
            if candidate < distance[neighbor]:
                distance[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distance


def incident_tets(body):
    result = [[] for _ in range(body.vertex_count)]
    for tet_id, tet in enumerate(body.muscle.tetrahedra):
        for vertex in tet:
            result[int(vertex)].append(int(tet_id))
    return result


def classify_constraint(source_info, target_infos, target_is_rigid=False,
                        target_bone=None):
    source_fixed = source_info is not None
    target_fixed_infos = [
        value for value in target_infos if value is not None]
    target_fixed = target_is_rigid or bool(target_fixed_infos)
    if not source_fixed and not target_fixed:
        return "FREE_FREE"
    if source_fixed and not target_fixed:
        return "CONSTRAINED_FREE"
    if not source_fixed and target_fixed:
        return "FREE_CONSTRAINED"
    source_bone = source_info["bone_id"]
    target_bones = (
        [target_bone] if target_is_rigid
        else [value["bone_id"] for value in target_fixed_infos])
    if target_bones and all(bone == source_bone for bone in target_bones):
        return "CONSTRAINED_CONSTRAINED_SAME_BONE"
    return "CONSTRAINED_CONSTRAINED_DIFFERENT_BONES"


def _allowed_pair(config, source_name, target_name, source_patch,
                  target_patches):
    exclusions = config["contact"].get("pair_exclusions", {})
    key = f"{source_name}__{target_name}"
    reverse = f"{target_name}__{source_name}"
    entry = exclusions.get(key)
    reversed_entry = False
    if entry is None:
        entry = exclusions.get(reverse)
        reversed_entry = entry is not None
    if not entry:
        return False, None
    source_regions = entry.get(
        "target_patch_regions" if reversed_entry
        else "source_patch_regions", [])
    target_regions = entry.get(
        "source_patch_regions" if reversed_entry
        else "target_patch_regions", [])
    source_match = (
        source_patch is not None
        and source_patch["patch_id"] in source_regions)
    target_match = any(
        value is not None and value["patch_id"] in target_regions
        for value in target_patches)
    return bool(source_match and target_match), (
        reverse if reversed_entry else key)


def directed_muscle_constraints(source_body, target_body, source_vertices,
                                target_vertices, config):
    source_surface = np.unique(source_body.muscle.surface_faces)
    target_mesh = trimesh.Trimesh(
        vertices=target_vertices,
        faces=target_body.muscle.surface_faces, process=False)
    inside = target_mesh.contains(source_vertices[source_surface])
    penetrating = source_surface[inside]
    if not len(penetrating):
        return []
    closest, depths, face_ids = trimesh.proximity.closest_point(
        target_mesh, source_vertices[penetrating])
    source_members = patch_membership(source_body)
    target_members = patch_membership(target_body)
    source_adjacency = surface_adjacency(
        source_vertices, source_body.muscle.surface_faces)
    boundaries = [
        vertex
        for patch in source_body.muscle.attachment_patches
        for vertex in patch_boundary(patch.vertex_ids, source_adjacency)]
    boundary_distance = graph_distances(source_adjacency, boundaries)
    source_incident = incident_tets(source_body)
    thickness = float(config["contact"]["muscle_muscle_thickness"])
    constraints = []
    for vertex, depth, face_id, point in zip(
            penetrating, depths, face_ids, closest):
        triangle = target_body.muscle.surface_faces[int(face_id)]
        source_info = source_members.get(int(vertex))
        target_infos = [
            target_members.get(int(item)) for item in triangle]
        category = classify_constraint(source_info, target_infos)
        allowed, exclusion_key = _allowed_pair(
            config, source_body.name, target_body.name,
            source_info, target_infos)
        constraints.append({
            "constraint_id": (
                f"{source_body.name}:v{int(vertex)}->"
                f"{target_body.name}:f{int(face_id)}"),
            "source_body": source_body.name,
            "target_body": target_body.name,
            "source_vertex_id": int(vertex),
            "target_triangle_id": int(face_id),
            "target_triangle_vertices": [int(item) for item in triangle],
            "exact_signed_gap": -float(depth),
            "exact_penetration_depth": float(depth),
            "solver_proxy_penetration": float(thickness + depth),
            "contact_thickness": thickness,
            "source_hard_constrained": source_info is not None,
            "target_triangle_has_hard_vertex": any(
                value is not None for value in target_infos),
            "source_attachment_patch_id": (
                source_info["patch_id"] if source_info else None),
            "target_attachment_patch_ids": sorted({
                value["patch_id"] for value in target_infos
                if value is not None}),
            "source_associated_bone_id": (
                source_info["bone_id"] if source_info else None),
            "target_associated_bone_ids": sorted({
                value["bone_id"] for value in target_infos
                if value is not None}),
            "incident_tet_ids": source_incident[int(vertex)],
            "local_tet_minimum_J": 1.0,
            "distance_to_nearest_attachment_patch_boundary":
                float(boundary_distance[int(vertex)]),
            "allowed_anatomical_overlap": allowed,
            "exclusion_rule": exclusion_key,
            "classification": category,
            "closest_point": point.tolist(),
        })
    return constraints


def directed_bone_constraints(body, vertices, bone_name, bone_surface,
                              config):
    surface_ids = np.unique(body.muscle.surface_faces)
    mesh = trimesh.Trimesh(
        vertices=bone_surface.vertices, faces=bone_surface.faces,
        process=False)
    inside = mesh.contains(vertices[surface_ids])
    penetrating = surface_ids[inside]
    if not len(penetrating):
        return []
    closest, depths, face_ids = trimesh.proximity.closest_point(
        mesh, vertices[penetrating])
    members = patch_membership(body)
    adjacency = surface_adjacency(vertices, body.muscle.surface_faces)
    boundaries = [
        vertex
        for patch in body.muscle.attachment_patches
        for vertex in patch_boundary(patch.vertex_ids, adjacency)]
    boundary_distance = graph_distances(adjacency, boundaries)
    source_incident = incident_tets(body)
    thickness = float(config["contact"]["bone_muscle_thickness"])
    constraints = []
    for vertex, depth, face_id, point in zip(
            penetrating, depths, face_ids, closest):
        source_info = members.get(int(vertex))
        category = classify_constraint(
            source_info, [], target_is_rigid=True,
            target_bone=bone_surface.parent_bone)
        constraints.append({
            "constraint_id": (
                f"{body.name}:v{int(vertex)}->{bone_name}:f{int(face_id)}"),
            "source_body": body.name,
            "target_body": bone_name,
            "source_vertex_id": int(vertex),
            "target_triangle_id": int(face_id),
            "target_triangle_vertices": [
                int(item) for item in bone_surface.faces[int(face_id)]],
            "exact_signed_gap": -float(depth),
            "exact_penetration_depth": float(depth),
            "solver_proxy_penetration": float(thickness + depth),
            "contact_thickness": thickness,
            "source_hard_constrained": source_info is not None,
            "target_triangle_has_hard_vertex": True,
            "source_attachment_patch_id": (
                source_info["patch_id"] if source_info else None),
            "target_attachment_patch_ids": [],
            "source_associated_bone_id": (
                source_info["bone_id"] if source_info else None),
            "target_associated_bone_ids": [bone_surface.parent_bone],
            "incident_tet_ids": source_incident[int(vertex)],
            "local_tet_minimum_J": 1.0,
            "distance_to_nearest_attachment_patch_boundary":
                float(boundary_distance[int(vertex)]),
            "allowed_anatomical_overlap": False,
            "exclusion_rule": None,
            "classification": category,
            "closest_point": point.tolist(),
        })
    return constraints


def duplicate_diagnostics(constraints, tolerance=5e-4):
    """Find reverse-direction constraints acting on the same local region."""
    by_pair = defaultdict(list)
    for constraint in constraints:
        if constraint["target_body"].startswith("L_") and (
                "Femur" not in constraint["target_body"]
                and "Tibia" not in constraint["target_body"]
                and "Os_Coxae" not in constraint["target_body"]):
            key = tuple(sorted((
                constraint["source_body"], constraint["target_body"])))
            by_pair[key].append(constraint)
    groups = []
    for pair, entries in by_pair.items():
        for index, first in enumerate(entries):
            for second in entries[index + 1:]:
                if first["source_body"] == second["source_body"]:
                    continue
                p = np.asarray(first["closest_point"])
                q = np.asarray(second["closest_point"])
                if np.linalg.norm(p - q) <= tolerance:
                    groups.append({
                        "pair": list(pair),
                        "first_constraint": first["constraint_id"],
                        "second_constraint": second["constraint_id"],
                        "closest_point_distance": float(
                            np.linalg.norm(p - q)),
                        "current_weight_per_direction": 0.5,
                        "full_weight_duplicate": False,
                    })
    return groups


def summarize_constraints(constraints):
    categories = {
        name: {"count": 0, "maximum_exact_depth": 0.0,
               "constraint_ids": []}
        for name in CLASS_NAMES}
    for constraint in constraints:
        entry = categories[constraint["classification"]]
        entry["count"] += 1
        entry["maximum_exact_depth"] = max(
            entry["maximum_exact_depth"],
            constraint["exact_penetration_depth"])
        entry["constraint_ids"].append(constraint["constraint_id"])
    exact_max = max(
        (value["exact_penetration_depth"] for value in constraints),
        default=0.0)
    proxy_max = max(
        (value["solver_proxy_penetration"] for value in constraints),
        default=0.0)
    return {
        "sign_convention": (
            "g >= 0 is feasible; contained vertices have g=-depth"),
        "proxy_definition": (
            "max(0, contact_thickness - exact_signed_gap); therefore an "
            "inside vertex reports thickness + exact depth"),
        "constraint_count": len(constraints),
        "maximum_exact_initial_penetration": exact_max,
        "maximum_initial_solver_proxy_penetration": proxy_max,
        "categories": categories,
    }


def attachment_patch_report(body, vertices, bone_surfaces):
    adjacency = surface_adjacency(vertices, body.muscle.surface_faces)
    reports = []
    for patch_id, patch in enumerate(body.muscle.attachment_patches):
        patch_set = set(int(vertex) for vertex in patch.vertex_ids)
        boundary = patch_boundary(patch.vertex_ids, adjacency)
        inward = graph_distances(adjacency, boundary, allowed=patch_set)
        # These simplified caps often consist entirely of a duplicated seam,
        # leaving no topological interior. Use central Euclidean rank to make
        # an explicit 60% proposal rather than silently declaring the whole
        # loop a core.
        patch_positions = vertices[patch.vertex_ids]
        centroid = patch_positions.mean(axis=0)
        radial = np.linalg.norm(patch_positions - centroid, axis=1)
        core_count = max(1, int(round(0.60 * len(patch.vertex_ids))))
        order = np.argsort(radial, kind="stable")
        hard_core = sorted(
            int(patch.vertex_ids[index]) for index in order[:core_count])
        hard_set = set(hard_core)
        transition = sorted(
            vertex for vertex in patch_set - hard_set
            if any(neighbor in hard_set for neighbor in adjacency[vertex]))
        free_region = sorted(patch_set - hard_set - set(transition))
        patch_faces = [
            face for face in body.muscle.surface_faces
            if all(int(vertex) in patch_set for vertex in face)]
        area = 0.0
        for face in patch_faces:
            triangle = vertices[face]
            area += 0.5 * np.linalg.norm(np.cross(
                triangle[1] - triangle[0], triangle[2] - triangle[0]))
        geodesic_extent = 0.0
        for source in patch.vertex_ids:
            distances = graph_distances(
                adjacency, [int(source)], allowed=patch_set)
            finite = distances[list(patch_set)]
            geodesic_extent = max(
                geodesic_extent, float(np.max(finite[np.isfinite(finite)])))
        short_bone = patch.bone_name.rstrip("01")
        bone = bone_surfaces.get(short_bone)
        bone_distances = np.full(len(patch.vertex_ids), np.nan)
        if bone is not None:
            mesh = trimesh.Trimesh(
                vertices=bone.vertices, faces=bone.faces, process=False)
            bone_distances = trimesh.proximity.closest_point(
                mesh, vertices[patch.vertex_ids])[1]
        median = float(np.nanmedian(bone_distances))
        mad = float(np.nanmedian(np.abs(bone_distances - median)))
        outlier_limit = median + max(3.0 * mad, 1e-3)
        reports.append({
            "muscle": body.name, "patch_id": patch_id,
            "bone_id": patch.bone_name,
            "current_patch": [int(vertex) for vertex in patch.vertex_ids],
            "vertex_count": len(patch.vertex_ids),
            "surface_area": area,
            "geodesic_extent": geodesic_extent,
            "distance_to_bone_minimum": float(np.nanmin(bone_distances)),
            "distance_to_bone_median": median,
            "distance_to_bone_maximum": float(np.nanmax(bone_distances)),
            "vertices_beyond_heuristic_footprint": [
                int(vertex) for vertex, distance in zip(
                    patch.vertex_ids, bone_distances)
                if distance > outlier_limit],
            "candidate_central_hard_patch": hard_core,
            "candidate_transition_ring": transition,
            "candidate_free_contact_enabled_region": free_region,
            "proposal_only": True,
        })
    return reports
