#!/usr/bin/env python3
"""Quasistatically bake all L upper-leg contour muscles through one tet cage."""
import argparse
import glob
import os
import pickle
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree

from tools.bake_headless import (
    load_skeleton, load_skeleton_meshes, load_muscle_meshes,
    load_tet_meshes, init_soft_bodies, _detect_bvh_tframe, MyBVH)
from viewer.arap_backends import ARAPBackendCPU


def cage_edges(tets, n):
    edges = set()
    for tet in tets:
        for i in range(4):
            for j in range(i + 1, 4):
                edges.add(tuple(sorted((int(tet[i]), int(tet[j])))))
    neighbors = [[] for _ in range(n)]
    weights = {}
    rest_edges = [dict() for _ in range(n)]
    return edges, neighbors, weights, rest_edges


def boundary_faces(tets):
    from collections import Counter
    count = Counter()
    oriented = {}
    for tet in tets:
        for face in ((tet[0], tet[2], tet[1]),
                     (tet[0], tet[1], tet[3]),
                     (tet[1], tet[2], tet[3]),
                     (tet[0], tet[3], tet[2])):
            key = tuple(sorted(int(i) for i in face))
            count[key] += 1
            oriented.setdefault(key, tuple(int(i) for i in face))
    return np.asarray(
        [oriented[key] for key, value in count.items() if value == 1],
        dtype=np.int32)


def harmonic_bone_weights(rest, edges, fixed_indices, control):
    """Diffuse attachment bone labels through the cage graph harmonically."""
    bones = sorted({control[int(i)][1] for i in fixed_indices})
    bone_index = {bone: i for i, bone in enumerate(bones)}
    row, col, value = [], [], []
    degree = np.zeros(len(rest), dtype=np.float64)
    for a, b in edges:
        # Cotangent weights are unavailable on a voxel tet graph; inverse
        # length conductance is stable and respects physical edge scale.
        w = 1.0 / max(float(np.linalg.norm(rest[a] - rest[b])), 1e-8)
        row.extend((a, b))
        col.extend((b, a))
        value.extend((-w, -w))
        degree[a] += w
        degree[b] += w
    row.extend(range(len(rest)))
    col.extend(range(len(rest)))
    value.extend(degree.tolist())
    laplacian = sparse.csr_matrix(
        (value, (row, col)), shape=(len(rest), len(rest)))
    fixed_mask = np.zeros(len(rest), dtype=bool)
    fixed_mask[fixed_indices] = True
    free = np.where(~fixed_mask)[0]
    labels = np.zeros((len(fixed_indices), len(bones)), dtype=np.float64)
    for local_i, cage_i in enumerate(fixed_indices):
        labels[local_i, bone_index[control[int(cage_i)][1]]] = 1.0
    solve = factorized(
        laplacian[free][:, free].tocsc()
        + sparse.eye(len(free), format="csc") * 1e-10)
    weights = np.zeros((len(rest), len(bones)), dtype=np.float64)
    weights[fixed_indices] = labels
    rhs = -laplacian[free][:, fixed_indices] @ labels
    for bi in range(len(bones)):
        weights[free, bi] = solve(np.asarray(rhs[:, bi]).ravel())
    weights = np.maximum(weights, 0.0)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return bones, weights


def project_tet_volumes(positions, rest, tets, fixed_mask, fixed_targets,
                        sweeps=8, stiffness=0.65, max_step=0.003):
    """Jacobi projection to positive rest-oriented tet volumes."""
    x = positions.copy()
    rq = rest[tets]
    rest_det = np.einsum(
        "ij,ij->i", rq[:, 0] - rq[:, 3],
        np.cross(rq[:, 1] - rq[:, 3], rq[:, 2] - rq[:, 3]))
    # Keep a small positive barrier in the rest orientation.
    target = rest_det
    for _ in range(sweeps):
        q = x[tets]
        e0 = q[:, 0] - q[:, 3]
        e1 = q[:, 1] - q[:, 3]
        e2 = q[:, 2] - q[:, 3]
        det = np.einsum("ij,ij->i", e0, np.cross(e1, e2))
        g0 = np.cross(e1, e2)
        g1 = np.cross(e2, e0)
        g2 = np.cross(e0, e1)
        g3 = -(g0 + g1 + g2)
        grad = np.stack((g0, g1, g2, g3), axis=1)
        denom = np.sum(grad * grad, axis=(1, 2)) + 1e-14
        lagrange = -stiffness * (det - target) / denom
        correction = lagrange[:, None, None] * grad
        length = np.linalg.norm(correction, axis=2, keepdims=True)
        correction *= np.minimum(
            1.0, max_step / np.maximum(length, 1e-12))
        accum = np.zeros_like(x)
        count = np.zeros((len(x), 1), dtype=np.float64)
        for corner in range(4):
            np.add.at(accum, tets[:, corner], correction[:, corner])
            np.add.at(count, tets[:, corner], 1.0)
        free = ~fixed_mask
        x[free] += accum[free] / np.maximum(count[free], 1.0)
        x[fixed_mask] = fixed_targets
    return x


def untangle_tets_gs(positions, rest, tets, fixed_mask=None,
                     fixed_targets=None, sweeps=20,
                     minimum_ratio=0.08, max_step=0.0015):
    """Targeted sequential positive-Jacobian barrier for local residuals."""
    x = positions.copy()
    if fixed_mask is None:
        fixed_mask = np.zeros(len(x), dtype=bool)
    rq = rest[tets]
    rest_det = np.einsum(
        "ij,ij->i", rq[:, 0] - rq[:, 3],
        np.cross(rq[:, 1] - rq[:, 3], rq[:, 2] - rq[:, 3]))
    target = minimum_ratio * rest_det
    for _ in range(sweeps):
        q = x[tets]
        det = np.einsum(
            "ij,ij->i", q[:, 0] - q[:, 3],
            np.cross(q[:, 1] - q[:, 3], q[:, 2] - q[:, 3]))
        bad = np.where(det * np.sign(rest_det)
                       < np.abs(target))[0]
        if not len(bad):
            break
        # Worst oriented volumes first prevents a corrected fold from being
        # immediately recreated by a more severe neighbour.
        order = bad[np.argsort(
            det[bad] * np.sign(rest_det[bad]))]
        for ti in order:
            ids = tets[ti]
            p0, p1, p2, p3 = x[ids]
            e0, e1, e2 = p0 - p3, p1 - p3, p2 - p3
            current = float(np.dot(e0, np.cross(e1, e2)))
            gradients = np.stack((
                np.cross(e1, e2), np.cross(e2, e0),
                np.cross(e0, e1)))
            gradients = np.vstack((
                gradients, -np.sum(gradients, axis=0)))
            gradients[fixed_mask[ids]] = 0.0
            denom = float(np.sum(gradients * gradients)) + 1e-14
            correction = (
                -(current - target[ti]) / denom) * gradients
            length = np.linalg.norm(correction, axis=1)
            correction *= min(
                1.0, max_step / max(float(length.max()), 1e-12))
            x[ids] += correction
        if fixed_targets is not None:
            x[fixed_mask] = fixed_targets
    return x


def smooth_deformation_gradients(positions, rest, tets, fixed_mask,
                                 fixed_targets, strength=0.18):
    """One descent-like projection for neighboring-tet F smoothness.

    This deliberately does not compare any edge with its rest length.  It
    penalizes discontinuities of the deformation gradient while the separate
    volume barrier prevents collapse.
    """
    x = positions.copy()
    rq = rest[tets]
    xq = x[tets]
    dm = np.stack((
        rq[:, 0] - rq[:, 3], rq[:, 1] - rq[:, 3],
        rq[:, 2] - rq[:, 3]), axis=2)
    ds = np.stack((
        xq[:, 0] - xq[:, 3], xq[:, 1] - xq[:, 3],
        xq[:, 2] - xq[:, 3]), axis=2)
    inv_dm = np.linalg.inv(dm)
    gradient = np.einsum("nij,njk->nik", ds, inv_dm)

    # Vertex-mediated tet adjacency is a stable, inexpensive approximation
    # of the pairwise neighboring-tet gradient energy.
    vertex_sum = np.zeros((len(x), 3, 3), dtype=np.float64)
    vertex_count = np.zeros(len(x), dtype=np.float64)
    for corner in range(4):
        np.add.at(vertex_sum, tets[:, corner], gradient)
        np.add.at(vertex_count, tets[:, corner], 1.0)
    vertex_gradient = vertex_sum / np.maximum(
        vertex_count[:, None, None], 1.0)
    smooth_gradient = np.mean(vertex_gradient[tets], axis=1)

    rest_center = rq.mean(axis=1)
    current_center = xq.mean(axis=1)
    local = rq - rest_center[:, None, :]
    target = current_center[:, None, :] + np.einsum(
        "nij,nkj->nki", smooth_gradient, local)
    correction = target - xq
    accum = np.zeros_like(x)
    count = np.zeros((len(x), 1), dtype=np.float64)
    for corner in range(4):
        np.add.at(accum, tets[:, corner], correction[:, corner])
        np.add.at(count, tets[:, corner], 1.0)
    free = ~fixed_mask
    x[free] += strength * accum[free] / np.maximum(count[free], 1.0)
    x[fixed_mask] = fixed_targets
    return x


def build_tracked_cage_bone_contacts(rest, tets, fixed_mask,
                                     skeleton_meshes, skel,
                                     bind_distance=0.035,
                                     clearance=0.0015):
    """Bind nearby cage-boundary vertices to one-sided moving bone planes."""
    import trimesh
    surface_vertices = np.unique(boundary_faces(tets))
    surface_vertices = surface_vertices[~fixed_mask[surface_vertices]]
    points = rest[surface_vertices]
    best_distance = np.full(len(points), np.inf)
    best = [None] * len(points)
    for mesh_name in ("L_Os_Coxae", "L_Femur",
                      "L_Patella", "L_Tibia_Fibula"):
        loader = skeleton_meshes.get(mesh_name)
        body_name = mesh_name + "0"
        body = skel.getBodyNode(body_name)
        mesh = getattr(loader, "trimesh", None) if loader is not None else None
        if body is None or mesh is None:
            continue
        try:
            closest, distance, _ = trimesh.proximity.closest_point(
                mesh, points)
        except Exception:
            closest, distance, _ = trimesh.proximity.closest_point_naive(
                mesh, points)
        take = distance < best_distance
        transform = body.getWorldTransform()
        rotation = np.asarray(transform.rotation())
        translation = np.asarray(transform.translation())
        for local_i in np.where(take)[0]:
            delta = points[local_i] - closest[local_i]
            length = float(np.linalg.norm(delta))
            if length < 1e-8:
                continue
            normal = delta / length
            best_distance[local_i] = distance[local_i]
            best[local_i] = (
                body_name,
                rotation.T @ (closest[local_i] - translation),
                rotation.T @ normal)
    contacts = []
    for local_i, item in enumerate(best):
        if item is not None and best_distance[local_i] <= bind_distance:
            contacts.append((
                int(surface_vertices[local_i]), item[0], item[1], item[2],
                min(clearance, 0.5 * float(best_distance[local_i]))))
    print(f"Tracked cage-bone contacts: {len(contacts)} "
          f"of {len(surface_vertices)} free boundary vertices")
    return contacts


def project_tracked_bone_contacts(positions, contacts, skel):
    """Enforce normal clearance while leaving tangential motion unchanged."""
    x = positions.copy()
    projected = 0
    for vertex, body_name, point_local, normal_local, clearance in contacts:
        transform = skel.getBodyNode(body_name).getWorldTransform()
        rotation = np.asarray(transform.rotation())
        translation = np.asarray(transform.translation())
        point = rotation @ point_local + translation
        normal = rotation @ normal_local
        gap = float(np.dot(x[vertex] - point, normal))
        if gap < clearance:
            x[vertex] += (clearance - gap) * normal
            projected += 1
    return x, projected


def build_embedded_muscle_bone_contacts(muscles, cage, skeleton_meshes,
                                        skel, bind_distance=0.05,
                                        clearance=0.0015):
    """Rest-side bone planes for all non-attachment muscle tet vertices."""
    import trimesh
    bone_data = []
    for mesh_name in ("L_Os_Coxae", "L_Femur",
                      "L_Patella", "L_Tibia_Fibula"):
        loader = skeleton_meshes.get(mesh_name)
        body_name = mesh_name + "0"
        body = skel.getBodyNode(body_name)
        mesh = getattr(loader, "trimesh", None) if loader is not None else None
        if body is not None and mesh is not None:
            bone_data.append((body_name, body, mesh))
    contacts = []
    for name, mobj in muscles.items():
        points = np.asarray(mobj.soft_body.rest_positions, dtype=np.float64)
        protected = set(int(i) for i in (
            getattr(mobj, "soft_body_local_anchors", {}) or {}))
        best_distance = np.full(len(points), np.inf)
        best = [None] * len(points)
        for body_name, body, mesh in bone_data:
            try:
                closest, distance, _ = trimesh.proximity.closest_point(
                    mesh, points)
            except Exception:
                closest, distance, _ = trimesh.proximity.closest_point_naive(
                    mesh, points)
            take = distance < best_distance
            transform = body.getWorldTransform()
            rotation = np.asarray(transform.rotation())
            translation = np.asarray(transform.translation())
            for vi in np.where(take)[0]:
                delta = points[vi] - closest[vi]
                length = float(np.linalg.norm(delta))
                if length < 1e-8:
                    continue
                best_distance[vi] = distance[vi]
                best[vi] = (
                    body_name,
                    rotation.T @ (closest[vi] - translation),
                    rotation.T @ (delta / length))
        binding = cage["muscles"][name]
        binding_tet = np.asarray(binding["tet_index"], dtype=np.int32)
        binding_weight = np.asarray(binding["weights"], dtype=np.float64)
        for vi, item in enumerate(best):
            if (vi in protected or item is None
                    or best_distance[vi] > bind_distance):
                continue
            contacts.append((
                name, vi, int(binding_tet[vi]), binding_weight[vi],
                item[0], item[1], item[2],
                min(clearance, 0.5 * float(best_distance[vi]))))
    print(f"Tracked embedded muscle-bone contacts: {len(contacts)}")
    return contacts


def project_embedded_contacts_to_cage(positions, cage_tets, contacts,
                                      skel, fixed_mask,
                                      max_step=0.003):
    """Lift embedded-muscle contact corrections back to shared cage DOFs."""
    x = positions.copy()
    accum = np.zeros_like(x)
    denom = np.zeros((len(x), 1), dtype=np.float64)
    projected = 0
    worst = 0.0
    for _, _, tet_i, bary, body_name, point_local, normal_local, clearance \
            in contacts:
        ids = cage_tets[tet_i]
        point_x = np.sum(bary[:, None] * x[ids], axis=0)
        transform = skel.getBodyNode(body_name).getWorldTransform()
        rotation = np.asarray(transform.rotation())
        point = rotation @ point_local + np.asarray(transform.translation())
        normal = rotation @ normal_local
        violation = clearance - float(np.dot(point_x - point, normal))
        if violation <= 0.0:
            continue
        correction = min(violation, max_step) * normal
        for corner in range(4):
            weight = float(bary[corner])
            accum[ids[corner]] += weight * correction
            denom[ids[corner], 0] += weight * weight
        projected += 1
        worst = max(worst, violation)
    free = ~fixed_mask
    delta = accum / np.maximum(denom, 1e-10)
    delta_length = np.linalg.norm(delta, axis=1, keepdims=True)
    delta *= np.minimum(
        1.0, max_step / np.maximum(delta_length, 1e-12))
    x[free] += delta[free]
    return x, projected, worst


def project_muscle_vertices_from_bones(positions, contacts, skel,
                                       clearance_slop=1e-6,
                                       max_correction=0.004):
    """Bounded unilateral projection for tracked muscle/bone contacts.

    A rest-side plane is only a local contact model.  Once a joint rotates
    substantially, blindly satisfying that infinite plane can pull a vertex
    through the whole limb.  Treat it as a trust-region correction instead:
    small violations are removed, while stale/non-local planes are capped.
    """
    x = positions.copy()
    projected = 0
    worst = 0.0
    capped = 0
    for entry in contacts:
        _, vertex, _, _, body_name, point_local, normal_local, clearance = entry
        transform = skel.getBodyNode(body_name).getWorldTransform()
        rotation = np.asarray(transform.rotation())
        point = rotation @ point_local + np.asarray(transform.translation())
        normal = rotation @ normal_local
        violation = clearance - float(np.dot(x[vertex] - point, normal))
        if violation > clearance_slop:
            correction = min(violation, max_correction)
            x[vertex] += correction * normal
            projected += 1
            worst = max(worst, violation)
            capped += int(violation > max_correction)
    return x, projected, worst, capped


def build_posed_bone_surface_data(skeleton_meshes, skel,
                                  mesh_names=("L_Femur",)):
    """Store skeleton meshes in body-local coordinates for posed contact."""
    bone_data = []
    for mesh_name in mesh_names:
        loader = skeleton_meshes.get(mesh_name)
        mesh = getattr(loader, "trimesh", None) if loader is not None else None
        body_name = mesh_name + "0"
        body = skel.getBodyNode(body_name)
        if mesh is None or body is None:
            continue
        transform = body.getWorldTransform()
        rotation = np.asarray(transform.rotation(), dtype=np.float64)
        translation = np.asarray(transform.translation(), dtype=np.float64)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        local_vertices = (vertices - translation) @ rotation
        bone_data.append((
            body_name, local_vertices,
            np.asarray(mesh.faces, dtype=np.int32)))
    print("Posed surface contacts: " + ", ".join(
        item[0] for item in bone_data))
    return bone_data


def project_vertices_from_posed_bones(positions, bone_data, skel,
                                      clearance=0.001,
                                      max_step=0.002):
    """Project vertices from the actual current-pose closed bone surfaces."""
    import trimesh
    x = positions.copy()
    projected = 0
    inside_total = 0
    worst = 0.0
    for body_name, local_vertices, faces in bone_data:
        transform = skel.getBodyNode(body_name).getWorldTransform()
        rotation = np.asarray(transform.rotation(), dtype=np.float64)
        translation = np.asarray(transform.translation(), dtype=np.float64)
        world_vertices = local_vertices @ rotation.T + translation
        mesh = trimesh.Trimesh(
            vertices=world_vertices, faces=faces, process=False)
        try:
            closest, distance, _ = trimesh.proximity.closest_point(mesh, x)
        except Exception:
            closest, distance, _ = trimesh.proximity.closest_point_naive(
                mesh, x)
        try:
            inside = np.asarray(mesh.contains(x), dtype=bool)
        except Exception:
            # Signed distance is positive inside in trimesh.
            inside = trimesh.proximity.signed_distance(mesh, x) > 0.0
        inside_total += int(np.sum(inside))
        near = (~inside) & (distance < clearance)
        active = inside | near
        if not np.any(active):
            continue
        delta = x[active] - closest[active]
        length = np.linalg.norm(delta, axis=1)
        # Outside vertices move away from the surface. Inside vertices must
        # move toward the closest surface and then through it.
        direction = delta / np.maximum(length[:, None], 1e-12)
        direction[inside[active]] *= -1.0
        required = np.where(
            inside[active], length + clearance, clearance - length)
        step = np.minimum(required, max_step)
        x[active] += step[:, None] * direction
        projected += int(np.sum(active))
        worst = max(worst, float(required.max(initial=0.0)))
    return x, projected, inside_total, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cage", default="cage/L_UpLeg_contour_tissue_cage.npz")
    ap.add_argument("--muscles", default=".muscles_L_UpLeg.json")
    ap.add_argument("--tet-dir", default="tet")
    ap.add_argument("--bvh", default="data/motion/run_vert_sternum_arm_forearm.bvh")
    ap.add_argument("--output-root", default=".bake_outputs/motion_cache")
    ap.add_argument("--region-tag", default="L_UpLeg_contour_shared_cage_v1")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument(
        "--cage-energy", choices=("arap", "muscle"), default="arap",
        help="'muscle' removes rest-edge-length preservation and uses only "
             "F-field smoothness, volume preservation and a J>0 barrier.")
    ap.add_argument("--gradient-smooth-strength", type=float, default=0.18)
    ap.add_argument("--bone-contact", action="store_true")
    ap.add_argument("--bone-contact-distance", type=float, default=0.035)
    ap.add_argument("--bone-contact-margin", type=float, default=0.0015)
    ap.add_argument(
        "--embedded-bone-contact", action="store_true",
        help="Enforce muscle vertex/bone separation through shared cage DOFs.")
    ap.add_argument(
        "--final-muscle-bone-contact", action="store_true",
        help="Guarantee tracked non-attachment muscle vertices remain on "
             "their rest side of moving bones.")
    ap.add_argument(
        "--max-final-contact-correction-mm", type=float, default=4.0,
        help="Trust-region cap for a rest-side bone plane. Large violations "
             "usually mean the local plane became stale after joint rotation.")
    ap.add_argument(
        "--posed-femur-contact", action="store_true",
        help="Use the actual current-pose femur surface instead of its "
             "rest-side tangent planes.")
    ap.add_argument("--posed-contact-clearance-mm", type=float, default=1.0)
    ap.add_argument("--posed-contact-passes", type=int, default=12)
    ap.add_argument(
        "--collision-substeps", type=int, default=0,
        help="Deterministic rest-to-pose collision continuation. This catches "
             "muscles that otherwise tunnel completely through the femur.")
    ap.add_argument("--volume-sweeps", type=int, default=8)
    ap.add_argument(
        "--cage-untangle-sweeps", type=int, default=20,
        help="Positive-Jacobian barrier passes on residual cage folds.")
    ap.add_argument(
        "--controls-per-bone", type=int, default=24,
        help="Maximum spatially distributed hard cage controls per bone.")
    ap.add_argument(
        "--muscle-volume-sweeps", type=int, default=12,
        help="Local positive-volume residual after cage embedding.")
    ap.add_argument(
        "--muscle-untangle-sweeps", type=int, default=20)
    ap.add_argument(
        "--muscle-controls-per-bone", type=int, default=4,
        help="Distributed exact pins retained per muscle attachment bone.")
    ap.add_argument(
        "--smooth-exact-attachments", action="store_true",
        help="Keep every anatomical cap vertex exact after first diffusing "
             "its bone correction through a short tendon neighbourhood.")
    ap.add_argument("--attachment-blend-radius", type=float, default=0.04)
    ap.add_argument(
        "--harmonic-guide", type=float, default=0.12,
        help="Softly retain the bone-harmonic tissue field after ARAP. "
             "Small values suppress remote cage drift without making the "
             "interior rigidly follow a bone.")
    ap.add_argument(
        "--cage-only", action="store_true",
        help="Save solved cage positions without reconstructing muscles.")
    args = ap.parse_args()

    with open(args.cage, "rb") as f:
        cage = pickle.load(f)
    rest = np.asarray(cage["vertices"], dtype=np.float64)
    tets = np.asarray(cage["tetrahedra"], dtype=np.int32)

    skel, bvh_info, mesh_info = load_skeleton()
    skeleton_meshes = load_skeleton_meshes()
    muscles = load_muscle_meshes(args.muscles)
    load_tet_meshes(muscles, args.tet_dir)
    skel.setPositions(np.zeros(skel.getNumDofs()))
    init_soft_bodies(
        muscles, skeleton_meshes, skel, mesh_info, smooth_skinning=None)

    # Transfer all anatomical muscle attachments to nearby cage vertices.
    samples = []
    for mobj in muscles.values():
        for _, (bone, local) in (
                getattr(mobj, "soft_body_local_anchors", {}) or {}).items():
            body = skel.getBodyNode(bone)
            if body is None:
                continue
            wt = body.getWorldTransform()
            world = np.asarray(wt.rotation()) @ np.asarray(local) + np.asarray(
                wt.translation())
            samples.append((world, bone, np.asarray(local)))
    sample_p = np.asarray([x[0] for x in samples])
    cage_tree = cKDTree(rest)
    _, cage_index = cage_tree.query(sample_p)
    # When several muscle anchors choose one cage vertex, retain the closest
    # source. This gives one unambiguous bone constraint per cage DOF.
    control = {}
    for sample_i, cage_i in enumerate(cage_index):
        distance = np.linalg.norm(rest[int(cage_i)] - sample_p[sample_i])
        if int(cage_i) not in control or distance < control[int(cage_i)][0]:
            control[int(cage_i)] = (
                distance, samples[sample_i][1], samples[sample_i][2])
    # A shared cage must not inherit every per-fiber cap vertex as a hard
    # constraint. Spatially downsample independently per bone; otherwise
    # hundreds of nearly coincident but incompatible pins collapse the cage.
    by_bone = {}
    for cage_i, value in control.items():
        by_bone.setdefault(value[1], []).append(cage_i)
    selected_control = {}
    for bone, candidates in by_bone.items():
        candidates = np.asarray(sorted(candidates), dtype=np.int32)
        if len(candidates) <= args.controls_per_bone:
            chosen = candidates
        else:
            points = rest[candidates]
            chosen_local = [int(np.argmin(points[:, 1]))]
            nearest_distance = np.linalg.norm(
                points - points[chosen_local[0]], axis=1)
            while len(chosen_local) < args.controls_per_bone:
                next_local = int(np.argmax(nearest_distance))
                chosen_local.append(next_local)
                nearest_distance = np.minimum(
                    nearest_distance,
                    np.linalg.norm(points - points[next_local], axis=1))
            chosen = candidates[np.asarray(chosen_local)]
        for cage_i in chosen:
            selected_control[int(cage_i)] = control[int(cage_i)]
        print(f"  {bone}: {len(candidates)} candidate controls -> "
              f"{len(chosen)} hard controls")
    control = selected_control
    fixed_indices = np.asarray(sorted(control), dtype=np.int32)
    fixed_mask = np.zeros(len(rest), dtype=bool)
    fixed_mask[fixed_indices] = True
    print(f"Cage controls: {len(fixed_indices)} from {len(samples)} "
          f"muscle attachment samples")

    edges, neighbors, weights, rest_edges = cage_edges(
        tets, len(rest))
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
        length = np.linalg.norm(rest[a] - rest[b])
        weight = 1.0 / max(length, 1e-5)
        weights[(a, b)] = weight
        rest_edges[a][b] = rest[a] - rest[b]
        rest_edges[b][a] = rest[b] - rest[a]
    backend = ARAPBackendCPU()
    if not backend.build_system(
            len(rest), neighbors, weights, fixed_mask, regularization=1e-6):
        raise RuntimeError("Cage ARAP factorization failed")

    control_rest = rest[fixed_indices]
    harmonic_bones, harmonic_weights = harmonic_bone_weights(
        rest, edges, fixed_indices, control)
    rest_bone_transform = {}
    for bone in harmonic_bones:
        wt = skel.getBodyNode(bone).getWorldTransform()
        rest_bone_transform[bone] = (
            np.asarray(wt.rotation()), np.asarray(wt.translation()))
    print("Harmonic cage field bones: " + ", ".join(harmonic_bones))
    bone_contacts = []
    if args.bone_contact:
        bone_contacts = build_tracked_cage_bone_contacts(
            rest, tets, fixed_mask, skeleton_meshes, skel,
            bind_distance=args.bone_contact_distance,
            clearance=args.bone_contact_margin)
    embedded_contacts = []
    if args.embedded_bone_contact or args.final_muscle_bone_contact:
        embedded_contacts = build_embedded_muscle_bone_contacts(
            muscles, cage, skeleton_meshes, skel,
            bind_distance=args.bone_contact_distance,
            clearance=args.bone_contact_margin)
    contacts_by_muscle = {}
    for entry in embedded_contacts:
        contacts_by_muscle.setdefault(entry[0], []).append(entry)
    posed_bone_data = []
    if args.posed_femur_contact:
        posed_bone_data = build_posed_bone_surface_data(
            skeleton_meshes, skel, mesh_names=("L_Femur",))

    t_frame = _detect_bvh_tframe(args.bvh)
    motion = MyBVH(args.bvh, bvh_info, skel, T_frame=t_frame)
    end = min(args.end_frame, motion.mocap_refs.shape[0] - 1)
    bvh_stem = os.path.splitext(os.path.basename(args.bvh))[0]
    output = os.path.join(args.output_root, bvh_stem, args.region_tag)
    os.makedirs(output, exist_ok=True)
    for old in glob.glob(os.path.join(output, "*_chunk_*.npz")):
        os.remove(old)

    baked = {name: [] for name in muscles}
    baked_cage = []
    frames = []
    started = time.time()
    for frame in range(args.start_frame, end + 1):
        frame_pose = np.asarray(motion.mocap_refs[frame], dtype=np.float64)
        skel.setPositions(frame_pose)
        targets = []
        for cage_i in fixed_indices:
            _, bone, local = control[int(cage_i)]
            wt = skel.getBodyNode(bone).getWorldTransform()
            targets.append(
                np.asarray(wt.rotation()) @ local
                + np.asarray(wt.translation()))
        targets = np.asarray(targets)
        transformed = []
        for bone in harmonic_bones:
            body_transform = skel.getBodyNode(bone).getWorldTransform()
            rotation = np.asarray(body_transform.rotation())
            translation = np.asarray(body_transform.translation())
            rest_rotation, rest_translation = rest_bone_transform[bone]
            local = (rest - rest_translation) @ rest_rotation
            transformed.append(local @ rotation.T + translation)
        transformed = np.stack(transformed, axis=1)
        initial = np.sum(
            harmonic_weights[:, :, None] * transformed, axis=1)
        initial[fixed_indices] = targets
        if args.cage_energy == "arap":
            posed, iterations, residual = backend.solve(
                initial, rest, neighbors, weights, rest_edges,
                fixed_mask, targets, max_iterations=args.iterations,
                tolerance=1e-4)
            if args.harmonic_guide > 0.0:
                guide = float(np.clip(args.harmonic_guide, 0.0, 1.0))
                posed = (1.0 - guide) * posed + guide * initial
                posed[fixed_indices] = targets
            posed = project_tet_volumes(
                posed, rest, tets, fixed_mask, targets,
                sweeps=args.volume_sweeps)
        else:
            posed = initial
            residual = 0.0
            iterations = args.iterations
            # Alternating projections approximate the stated energy without
            # ever introducing a rest-length term.
            for _ in range(args.iterations):
                posed = smooth_deformation_gradients(
                    posed, rest, tets, fixed_mask, targets,
                    strength=args.gradient_smooth_strength)
                if args.volume_sweeps > 0:
                    posed = project_tet_volumes(
                        posed, rest, tets, fixed_mask, targets,
                        sweeps=max(
                            1, args.volume_sweeps // args.iterations),
                        stiffness=0.55, max_step=0.002)
                if bone_contacts:
                    posed, _ = project_tracked_bone_contacts(
                        posed, bone_contacts, skel)
            # Converge the actual incompressibility term after smoothing.
            # This still imposes no constraint on individual edge lengths.
            if args.volume_sweeps > 0:
                posed = project_tet_volumes(
                    posed, rest, tets, fixed_mask, targets,
                    sweeps=args.volume_sweeps,
                    stiffness=0.8, max_step=0.003)
            if bone_contacts:
                # Alternate contact and incompressibility so the last
                # operation cannot silently push tissue back through bone.
                for _ in range(4):
                    posed, projected_contacts = (
                        project_tracked_bone_contacts(
                            posed, bone_contacts, skel))
                    if args.volume_sweeps > 0:
                        posed = project_tet_volumes(
                            posed, rest, tets, fixed_mask, targets,
                            sweeps=2, stiffness=0.7, max_step=0.002)
                posed, projected_contacts = project_tracked_bone_contacts(
                    posed, bone_contacts, skel)
                print(f"  frame {frame}: bone contacts projected="
                      f"{projected_contacts}")
            if args.embedded_bone_contact and embedded_contacts:
                # Correct the common cage until embedded muscle vertices
                # respect their moving rest-side bone planes.
                embedded_projected = 0
                embedded_worst = 0.0
                for _ in range(10):
                    posed, embedded_projected, embedded_worst = (
                        project_embedded_contacts_to_cage(
                            posed, tets, embedded_contacts, skel, fixed_mask))
                    posed[fixed_indices] = targets
                    posed = project_tet_volumes(
                        posed, rest, tets, fixed_mask, targets,
                        sweeps=2, stiffness=0.7, max_step=0.002)
                    if embedded_projected == 0:
                        break
                posed, embedded_projected, embedded_worst = (
                    project_embedded_contacts_to_cage(
                        posed, tets, embedded_contacts, skel, fixed_mask))
                posed[fixed_indices] = targets
                print(f"  frame {frame}: embedded contacts projected="
                      f"{embedded_projected}, worst="
                      f"{embedded_worst * 1000.0:.2f}mm")
        if args.cage_untangle_sweeps > 0:
            posed = untangle_tets_gs(
                posed, rest, tets, fixed_mask=fixed_mask,
                fixed_targets=targets,
                sweeps=args.cage_untangle_sweeps,
                minimum_ratio=0.08, max_step=0.002)
        if args.embedded_bone_contact and embedded_contacts:
            # Untangling may move a contacted cage DOF back across its bone
            # plane. Contact therefore owns the final state.
            for _ in range(30):
                posed, embedded_projected, embedded_worst = (
                    project_embedded_contacts_to_cage(
                        posed, tets, embedded_contacts, skel, fixed_mask,
                        max_step=0.003))
                posed[fixed_indices] = targets
                if embedded_projected == 0 or embedded_worst < 1e-5:
                    break
            print(f"  frame {frame}: final embedded violations="
                  f"{embedded_projected}, worst="
                  f"{embedded_worst * 1000.0:.3f}mm")
            posed[fixed_indices] = targets
        baked_cage.append(posed.astype(np.float32))
        if args.cage_only:
            frames.append(frame)
            print(f"frame {frame}: {iterations} iterations, "
                  f"residual={residual:.2e} (cage only)")
            continue
        frame_direct_contacts = 0
        frame_direct_worst = 0.0
        frame_capped_contacts = 0
        for name in baked:
            binding = cage["muscles"][name]
            tet = tets[np.asarray(binding["tet_index"])]
            weight = np.asarray(binding["weights"], dtype=np.float64)
            embedded = np.einsum(
                "ni,nij->nj", weight, posed[tet])
            if args.final_muscle_bone_contact:
                embedded, direct_count, direct_worst, direct_capped = (
                    project_muscle_vertices_from_bones(
                        embedded, contacts_by_muscle.get(name, []), skel,
                        max_correction=(
                            args.max_final_contact_correction_mm * 0.001)))
                frame_direct_contacts += direct_count
                frame_direct_worst = max(frame_direct_worst, direct_worst)
                frame_capped_contacts += direct_capped
            if args.muscle_volume_sweeps > 0:
                mobj = muscles[name]
                muscle_rest = np.asarray(
                    mobj.soft_body.rest_positions, dtype=np.float64)
                muscle_tets = np.asarray(
                    mobj.soft_body.tetrahedra, dtype=np.int32)
                muscle_fixed = np.zeros(len(embedded), dtype=bool)
                local_anchors = (
                    getattr(mobj, "soft_body_local_anchors", {}) or {})
                anchor_by_bone = {}
                for anchor_i, (bone, _) in local_anchors.items():
                    anchor_by_bone.setdefault(bone, []).append(int(anchor_i))
                selected_anchors = []
                for _, candidates in anchor_by_bone.items():
                    candidates = np.asarray(
                        sorted(candidates), dtype=np.int32)
                    if args.smooth_exact_attachments:
                        chosen = candidates
                    elif args.muscle_controls_per_bone <= 0:
                        chosen = np.zeros(0, dtype=np.int32)
                    elif len(candidates) <= args.muscle_controls_per_bone:
                        chosen = candidates
                    else:
                        points = muscle_rest[candidates]
                        chosen_local = [0]
                        nearest_distance = np.linalg.norm(
                            points - points[0], axis=1)
                        while (len(chosen_local)
                               < args.muscle_controls_per_bone):
                            next_local = int(np.argmax(nearest_distance))
                            chosen_local.append(next_local)
                            nearest_distance = np.minimum(
                                nearest_distance,
                                np.linalg.norm(
                                    points - points[next_local], axis=1))
                        chosen = candidates[np.asarray(chosen_local)]
                    selected_anchors.extend(chosen.tolist())
                muscle_fixed_indices = np.asarray(
                    sorted(set(selected_anchors)), dtype=np.int32)
                muscle_fixed[muscle_fixed_indices] = True
                muscle_targets = []
                for fixed_i in muscle_fixed_indices:
                    bone, local = local_anchors[int(fixed_i)]
                    wt = skel.getBodyNode(bone).getWorldTransform()
                    muscle_targets.append(
                        np.asarray(wt.rotation()) @ np.asarray(local)
                        + np.asarray(wt.translation()))
                muscle_targets = np.asarray(
                    muscle_targets, dtype=np.float64).reshape(-1, 3)
                if (args.smooth_exact_attachments
                        and len(muscle_fixed_indices)):
                    # Diffuse the exact cap correction before imposing the
                    # Dirichlet values. This creates a short tendon transition
                    # instead of a one-ring snap at the attachment boundary.
                    anchor_rest = muscle_rest[muscle_fixed_indices]
                    anchor_correction = (
                        muscle_targets
                        - embedded[muscle_fixed_indices])
                    anchor_tree = cKDTree(anchor_rest)
                    distance, nearest = anchor_tree.query(
                        muscle_rest,
                        k=min(8, len(muscle_fixed_indices)))
                    if distance.ndim == 1:
                        distance = distance[:, None]
                        nearest = nearest[:, None]
                    blend_weight = np.exp(
                        -(distance / 0.012) ** 2)
                    blend_weight /= np.maximum(
                        blend_weight.sum(axis=1, keepdims=True), 1e-12)
                    correction = np.sum(
                        blend_weight[:, :, None]
                        * anchor_correction[nearest], axis=1)
                    closest = distance[:, 0]
                    support = np.clip(
                        1.0 - closest / args.attachment_blend_radius,
                        0.0, 1.0)
                    support = support * support * (3.0 - 2.0 * support)
                    embedded += support[:, None] * correction
                embedded[muscle_fixed_indices] = muscle_targets
                embedded = project_tet_volumes(
                    embedded, muscle_rest, muscle_tets,
                    muscle_fixed, muscle_targets,
                    sweeps=args.muscle_volume_sweeps,
                    stiffness=0.75, max_step=0.002)
                embedded = untangle_tets_gs(
                    embedded, muscle_rest, muscle_tets,
                    fixed_mask=muscle_fixed,
                    fixed_targets=muscle_targets,
                    sweeps=args.muscle_untangle_sweeps)
                embedded[muscle_fixed_indices] = muscle_targets
            if args.posed_femur_contact and posed_bone_data:
                # Alternate current-surface contact with local volume repair.
                # Unlike tracked tangent planes this remains meaningful under
                # large femur rotations and permits tangential sliding.
                surface_projected = 0
                surface_inside = 0
                surface_worst = 0.0
                mobj = muscles[name]
                muscle_rest = np.asarray(
                    mobj.soft_body.rest_positions, dtype=np.float64)
                muscle_tets = np.asarray(
                    mobj.soft_body.tetrahedra, dtype=np.int32)
                no_fixed = np.zeros(len(embedded), dtype=bool)
                if args.collision_substeps > 1:
                    # Follow the same canonical rest-to-target route for every
                    # requested pose.  Carrying collision-corrected positions
                    # along this route prevents end-pose SDF tunnelling while
                    # preserving the same-pose/same-result property.
                    desired_final = embedded.copy()
                    carried = muscle_rest.copy()
                    previous_desired = muscle_rest.copy()
                    for substep in range(1, args.collision_substeps + 1):
                        alpha = substep / args.collision_substeps
                        desired = (
                            muscle_rest
                            + alpha * (desired_final - muscle_rest))
                        carried += desired - previous_desired
                        previous_desired = desired
                        skel.setPositions(alpha * frame_pose)
                        for _ in range(4):
                            carried, _, inside_now, _ = (
                                project_vertices_from_posed_bones(
                                    carried, posed_bone_data, skel,
                                    clearance=(
                                        args.posed_contact_clearance_mm
                                        * 0.001),
                                    max_step=0.002))
                            if inside_now == 0:
                                break
                        carried = untangle_tets_gs(
                            carried, muscle_rest, muscle_tets,
                            fixed_mask=no_fixed, sweeps=4,
                            minimum_ratio=0.05, max_step=0.001)
                    embedded = carried
                    skel.setPositions(frame_pose)
                    embedded = untangle_tets_gs(
                        embedded, muscle_rest, muscle_tets,
                        fixed_mask=no_fixed,
                        sweeps=args.muscle_untangle_sweeps,
                        minimum_ratio=0.05, max_step=0.0015)
                for _ in range(args.posed_contact_passes):
                    embedded, surface_projected, surface_inside, \
                        surface_worst = project_vertices_from_posed_bones(
                            embedded, posed_bone_data, skel,
                            clearance=args.posed_contact_clearance_mm * 0.001,
                            max_step=0.002)
                    embedded = untangle_tets_gs(
                        embedded, muscle_rest, muscle_tets,
                        fixed_mask=no_fixed, sweeps=16,
                        minimum_ratio=0.05, max_step=0.001)
                    if surface_inside == 0:
                        break
                # Contact owns the final state; this final query also reports
                # whether the iterative projection genuinely cleared bone.
                embedded, surface_projected, surface_inside, \
                    surface_worst = project_vertices_from_posed_bones(
                        embedded, posed_bone_data, skel,
                        clearance=args.posed_contact_clearance_mm * 0.001,
                        max_step=0.002)
                print(f"    {name}: posed femur inside={surface_inside}, "
                      f"projected={surface_projected}, worst="
                      f"{surface_worst * 1000.0:.2f}mm")
            baked[name].append(embedded.astype(np.float32))
        if args.final_muscle_bone_contact:
            print(f"  frame {frame}: final muscle contacts="
                  f"{frame_direct_contacts}, capped={frame_capped_contacts}, "
                  f"initial worst="
                  f"{frame_direct_worst * 1000.0:.2f}mm")
        frames.append(frame)
        print(f"frame {frame}: {iterations} iterations, "
              f"residual={residual:.2e}")

    frame_array = np.asarray(frames, dtype=np.int32)
    for name, positions in baked.items():
        if not positions:
            continue
        np.savez_compressed(
            os.path.join(output, f"{name}_chunk_0000.npz"),
            frames=frame_array, positions=np.asarray(positions))
    np.savez_compressed(
        os.path.join(output, "__tissue_cage_chunk_0000.npz"),
        frames=frame_array,
        positions=np.asarray(baked_cage),
        rest_positions=rest.astype(np.float32),
        surface_faces=boundary_faces(tets))
    with open(os.path.join(output, ".done"), "w") as f:
        f.write("complete\n")
    print(f"Saved {len(frames)} frames for {len(baked)} muscles to {output}")
    print(f"Elapsed {time.time() - started:.2f}s")


if __name__ == "__main__":
    main()
