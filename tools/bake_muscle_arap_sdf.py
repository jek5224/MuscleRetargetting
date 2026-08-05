#!/usr/bin/env python3
"""Cache-free volumetric ARAP muscle bake with moving-bone SDF contact.

The solver reads only rest tetrahedra, skeleton/BVH motion, and an SDF.  It
does not accept a deformation cache.  Origin and insertion attachments are
soft constraints so contact and element quality can redistribute difficult
motion instead of forcing a single infeasible boundary displacement.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh

from core.bvhparser import MyBVH
import test_emu
from tools import bake_emu
from tools.bake_stiff_tet_arap import BoneSDF
from tools.bake_stiff_tet_pbd import (
    signed_volumes, signed_volumes_cuda, unique_edges)
from tools.bake_surface_fast import load_tet, surface_faces


def relative_transform(body, rest_rotation, rest_translation):
    current = body.getWorldTransform()
    rotation = np.asarray(current.rotation()) @ rest_rotation.T
    translation = np.asarray(current.translation()) - rotation @ rest_translation
    return rotation, translation


def closest_surface_points(mesh_name: str, points: np.ndarray) -> np.ndarray:
    """Return exact closest points on a rest-pose Zygote bone surface."""
    path = Path(bake_emu.SKEL_MESH_DIR) / f"{mesh_name}.obj"
    mesh = trimesh.load(path, process=False)
    mesh.vertices = np.asarray(mesh.vertices) * bake_emu.MESH_SCALE
    try:
        closest, _, _ = trimesh.proximity.closest_point(mesh, points)
    except (ModuleNotFoundError, ImportError):
        # The naive implementation has no spatial-index dependency and is only
        # used once for the relatively small attachment contours.
        closest, _, _ = trimesh.proximity.closest_point_naive(mesh, points)
    return np.asarray(closest, dtype=np.float64)


def rigid_surface_fit(points: np.ndarray, closest: np.ndarray) -> np.ndarray:
    """Move a complete attachment contour rigidly toward its bone surface."""
    source_center = np.mean(points, axis=0)
    target_center = np.mean(closest, axis=0)
    covariance = (points - source_center).T @ (closest - target_center)
    u, _, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1] *= -1.0
        rotation = vh.T @ u.T
    return (points - source_center) @ rotation.T + target_center


def dense_boundary_samples(faces: np.ndarray):
    """Vertices, edge midpoints, and centroid of every boundary triangle."""
    indices, weights = [], []
    barycentrics = (
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        (0.5, 0.5, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5),
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    )
    for face in faces:
        for barycentric in barycentrics:
            indices.append(face)
            weights.append(barycentric)
    return (
        np.asarray(indices, dtype=np.int32),
        np.asarray(weights, dtype=np.float64),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="L_Vastus_Intermedius")
    parser.add_argument("--tet", type=Path)
    parser.add_argument("--bvh", type=Path, default=Path(
        "data/motion/left_thigh_quasistatic_diverse_smooth_76frame.bvh"))
    parser.add_argument("--sdf", type=Path, default=Path(
        ".bake_outputs/collision_sdf/L_Femur0_sdf.npz"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float32",
        help="Solver arithmetic precision. float32 is substantially faster "
             "on consumer GPUs; float64 retains the historical path.")
    parser.add_argument(
        "--end-frame", type=int,
        help="Inclusive final BVH frame; defaults to the complete motion.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--initialization-cache", type=Path,
        help="Matching-topology cache containing start-frame minus one.")
    parser.add_argument("--initialization-cache-frame", type=int)
    parser.add_argument("--initialization-pose-frame", type=int)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument(
        "--maximum-attachment-step", type=float, default=0.0,
        help="If positive, automatically increase pose substeps so no "
             "origin/insertion target advances farther than this many metres.")
    parser.add_argument("--difficult-substeps", type=int, default=0)
    parser.add_argument(
        "--difficult-frame-ranges", default="",
        help="Comma-separated inclusive ranges, e.g. 6-27,52-72.")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--maximum-iterations", type=int, default=30,
        help="Adaptive per-substep ceiling when attachments have not converged.")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--maximum-step", type=float, default=2.5e-4)
    parser.add_argument("--collision-margin", type=float, default=0.0015)
    parser.add_argument("--collision-tolerance", type=float, default=0.00025)
    parser.add_argument("--arap-weight", type=float, default=0.25)
    parser.add_argument(
        "--muscle-material", action="store_true",
        help="Use an objective transversely-isotropic muscle material: an "
             "isochoric Neo-Hookean matrix, active/passive fiber stretch, "
             "near-incompressibility, and a positive-J barrier. ARAP remains "
             "available but is normally set to zero in this mode.")
    parser.add_argument(
        "--matrix-weight", type=float, default=0.0,
        help="Isochoric Neo-Hookean matrix weight used by "
             "--muscle-material.")
    parser.add_argument(
        "--inversion-barrier-weight", type=float, default=0.0,
        help="Weight of the -log(J) positive-volume barrier used by "
             "--muscle-material.")
    parser.add_argument(
        "--inversion-barrier-start", type=float, default=0.25,
        help="Jacobian ratio below which the shifted log barrier activates.")
    parser.add_argument(
        "--endpoint-arap-scale", type=float, default=1.0,
        help="ARAP stiffness multiplier at origin/insertion tets; smoothly "
             "decays to one through --endpoint-arap-width.")
    parser.add_argument(
        "--endpoint-arap-width", type=float, default=0.12,
        help="Fraction of the attachment axis occupied by each endpoint "
             "stiffness transition.")
    parser.add_argument(
        "--director-weight", type=float, default=0.0,
        help="Material-sheet director weight. Unlike ARAP, this directly "
             "resists axial corkscrew rotation while allowing fiber stretch.")
    parser.add_argument(
        "--fiber-weight", type=float, default=0.0,
        help="Per-tet active fiber stretch weight using saved volumetric "
             "Laplace-gradient directions.")
    parser.add_argument(
        "--fiber-target-stretch", type=float, default=1.0,
        help="Target stretch along the saved material fiber direction; "
             "values below one produce active contraction.")
    parser.add_argument(
        "--fiber-target-from-attachments", action="store_true",
        help="Set active fiber stretch from the current origin-to-insertion "
             "distance ratio instead of a constant target.")
    parser.add_argument(
        "--fiber-bending-weight", type=float, default=0.0,
        help="Penalize sharp fiberwise tangent changes relative to the "
             "bone-carrier reference, suppressing folds without axial "
             "rotation locking.")
    parser.add_argument(
        "--longitudinal-arap-scale", type=float, default=1.0,
        help="Relative ARAP stiffness along the harmonic origin-to-insertion "
             "axis. Values below one permit distributed muscle shortening "
             "without weakening transverse shape preservation.")
    parser.add_argument(
        "--smooth-arap-weight", type=float, default=0.0,
        help="Smooth-ARAP rotated Laplacian-vector weight, adapted from "
             "Oehri et al. (2025) to the volumetric tet edge graph.")
    parser.add_argument(
        "--smooth-arap-ramp-frames", type=int, default=0,
        help="Ramp Smooth ARAP after --start-frame to avoid an abrupt "
             "higher-order force on an initialized deformation cache.")
    parser.add_argument(
        "--fast-smooth-arap", action="store_true",
        help="Use one exact incident-tet rotation per vertex and avoid the "
             "additional per-vertex SVD pass.")
    parser.add_argument(
        "--rotation-update-interval", type=int, default=8,
        help="Reuse detached local ARAP rotations for this many optimizer "
             "iterations. Values above one avoid repeated tet SVDs while "
             "retaining the same energy and global vertex updates.")
    parser.add_argument("--volume-weight", type=float, default=0.1)
    parser.add_argument("--quality-weight", type=float, default=20.0)
    parser.add_argument(
        "--minimum-jacobian", type=float, default=1e-4,
        help="Hard positive-J line-search floor; quality energy remains active "
             "well above this value.")
    parser.add_argument(
        "--local-jacobian-line-search", action="store_true",
        help="Limit only vertices incident to endangered tets instead of "
             "shrinking the global optimizer step.")
    parser.add_argument(
        "--localized-quality-barrier", action="store_true",
        help="Concentrate the positive-Jacobian penalty on the worst 256 "
             "tets instead of diluting it over the complete mesh.")
    parser.add_argument("--contact-weight", type=float, default=300.0)
    parser.add_argument("--origin-weight", type=float, default=10000.0)
    parser.add_argument("--insertion-weight", type=float, default=10000.0)
    parser.add_argument(
        "--origin-body",
        help="Override the skeleton body carrying the origin attachment "
             "(for example Saccrum_Coccyx0 for rectus femoris).")
    parser.add_argument(
        "--insertion-body",
        help="Override the skeleton body carrying the insertion attachment.")
    parser.add_argument(
        "--transport-body",
        help="Rigid predictor for unconstrained vertices. Defaults to the "
             "origin body; use L_Femur0 for biarticular thigh muscles while "
             "keeping their anatomical endpoint bodies.")
    parser.add_argument(
        "--bone-carrier-endpoint-predictor", action="store_true",
        help="Use the carrier bone across the belly, with local origin- and "
             "insertion-bone transport near their attachment contours.")
    parser.add_argument(
        "--rotation-aware-origin-transport", action="store_true",
        help="Interpolate origin-to-carrier rotations on SO(3) instead of "
             "linearly blending transformed proximal positions.")
    parser.add_argument(
        "--endpoint-transport-width", type=float, default=0.2,
        help="Fraction of the attachment-axis length used for each local "
             "endpoint-to-carrier transition.")
    parser.add_argument(
        "--origin-transport-width", type=float,
        help="Origin-specific endpoint transition width; defaults to "
             "--endpoint-transport-width.")
    parser.add_argument(
        "--insertion-transport-width", type=float,
        help="Insertion-specific endpoint transition width; defaults to "
             "--endpoint-transport-width.")
    parser.add_argument(
        "--insertion-edge-weight", type=float, default=0.0,
        help="Dimensionless stretch penalty on tet edges joining insertion "
             "anchors to their first free neighbors. Prevents positive-"
             "Jacobian needle tets without constraining the muscle belly.")
    parser.add_argument(
        "--hard-attachments", action="store_true",
        help="Treat origin and insertion positions as Dirichlet constraints "
             "inside every ARAP/Jacobian line-search iteration. This avoids "
             "needle elements from a soft solve followed by cap repair.")
    parser.add_argument(
        "--rigid-attachment-tet-rings", type=int, default=0,
        help="With --hard-attachments, rigidly carry this many incident-tet "
             "rings beyond each contour with its attachment bone.")
    parser.add_argument(
        "--rigid-origin-tet-rings", type=int,
        help="Origin-specific rigid tet-ring count; overrides "
             "--rigid-attachment-tet-rings.")
    parser.add_argument(
        "--rigid-insertion-tet-rings", type=int,
        help="Insertion-specific rigid tet-ring count; overrides "
             "--rigid-attachment-tet-rings.")
    parser.add_argument(
        "--attachment-surface-blend", type=float, default=1.0,
        help="Blend attachment targets from their authored rest locations "
             "onto the exact nearest bone surfaces (0=legacy offsets, "
             "1=surface attached).")
    parser.add_argument(
        "--attachment-surface-mode", choices=("rigid", "pointwise"),
        default="rigid",
        help="Rigid preserves each open contour's shape while fitting it to "
             "the bone; pointwise projects every vertex independently.")
    parser.add_argument(
        "--origin-attachment-rings", type=int, default=0,
        help="Additional soft surface-topology rings around origin anchors.")
    parser.add_argument(
        "--insertion-attachment-rings", type=int, default=0,
        help="Additional soft surface-topology rings around insertion anchors.")
    parser.add_argument(
        "--insertion-soft-ring-weights", default="",
        help="Comma-separated insertion spring weights for successive free "
             "surface rings, relative to --insertion-weight (for example "
             "0.35,0.12). These rings stay out of the hard cap repair.")
    parser.add_argument(
        "--attachment-tolerance", type=float, default=0.001,
        help="Maximum accepted origin/insertion error in metres.")
    parser.add_argument(
        "--attachment-strength-scale", type=float, default=0.00025,
        help="Error scale for adaptive spring strengthening; independent of "
             "the final acceptance tolerance.")
    parser.add_argument(
        "--allow-attachment-error", action="store_true",
        help="Write frames exceeding --attachment-tolerance instead of failing.")
    parser.add_argument(
        "--disable-cap-repair", action="store_true",
        help="Disable the post-optimization vertex-only attachment advance. "
             "Useful when that repair would create needle tets after ARAP.")
    parser.add_argument(
        "--attachment-repair-neighbor-blend", type=float, default=0.0,
        help="Fraction of insertion cap-repair displacement propagated to "
             "its first free tet neighbors (0=legacy vertex-only repair).")
    parser.add_argument(
        "--independent-frames", action="store_true",
        help="Reset every requested frame to the initialization cache state "
             "before its quasistatic continuation. Useful when sequential "
             "states accumulate into a wrong attachment branch.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.tet is None:
        args.tet = Path("tet") / f"{args.name}_tet.npz"
    if args.output_dir is None:
        args.output_dir = (
            Path(".bake_outputs/motion_cache")
            / args.bvh.stem
            / f"{args.name}_cacheless_corotated_arap_sdf_v1")

    device = torch.device(args.device)
    dtype = (torch.float32 if args.precision == "float32"
             else torch.float64)
    skeleton, bvh_info, _ = bake_emu.load_skeleton()
    skeleton.setPositions(np.zeros(skeleton.getNumDofs()))
    data = load_tet(args.tet)
    group = test_emu.prepare_group_data(
        data, args.name, skeleton, test_emu._load_bone_trees(),
        source_path=args.tet, attachment_rings=0)
    rest = np.asarray(group["vertices"], dtype=np.float64)
    tets = np.asarray(group["tetrahedra"], dtype=np.int32)
    if ("origin_attachment_vertices" in data
            and "insertion_attachment_vertices" in data):
        origin = np.asarray(
            data["origin_attachment_vertices"], dtype=np.int32)
        insertion = np.asarray(
            data["insertion_attachment_vertices"], dtype=np.int32)
        group["origin_body"] = "L_Femur0"
        group["insertion_body"] = "L_Patella0"
    else:
        origin = np.asarray(group["origin_fixed"], dtype=np.int32)
        insertion = np.asarray(group["insertion_fixed"], dtype=np.int32)
    if args.origin_body:
        group["origin_body"] = args.origin_body
    if args.insertion_body:
        group["insertion_body"] = args.insertion_body
    attachment_surface_faces = np.asarray(
        data.get("collision_faces", surface_faces(tets)), dtype=np.int32)
    if args.origin_attachment_rings > 0:
        origin = np.asarray(test_emu._expand_surface_vertex_rings(
            origin, attachment_surface_faces,
            args.origin_attachment_rings), dtype=np.int32)
    if args.insertion_attachment_rings > 0:
        insertion = np.asarray(test_emu._expand_surface_vertex_rings(
            insertion, attachment_surface_faces,
            args.insertion_attachment_rings), dtype=np.int32)
    soft_insertion_weights = [
        float(value) for value in args.insertion_soft_ring_weights.split(",")
        if value.strip()]
    soft_insertion_rings = []
    reached_insertion = np.asarray(insertion, dtype=np.int32)
    for ring_weight in soft_insertion_weights:
        expanded = np.asarray(test_emu._expand_surface_vertex_rings(
            reached_insertion, attachment_surface_faces, 1), dtype=np.int32)
        ring = np.setdiff1d(expanded, reached_insertion).astype(np.int32)
        soft_insertion_rings.append((ring, ring_weight))
        reached_insertion = np.union1d(reached_insertion, ring).astype(np.int32)
    print(
        f"solver attachments: origin={len(origin)} insertion={len(insertion)}"
        + (" soft insertion rings=" + ",".join(
            f"{len(ring)}@{weight:g}" for ring, weight
            in soft_insertion_rings) if soft_insertion_rings else ""),
        flush=True)
    if ("origin_attachment_vertices" in data
            and "insertion_attachment_vertices" in data):
        axis = test_emu._harmonic_axis_coordinate(
            rest, tets, origin, insertion)
        axis[origin] = 0.0
        axis[insertion] = 1.0
        group["axis_coordinate"] = axis
        group["lbs_bindings"] = test_emu._make_lbs_bindings(
            rest, axis,
            skeleton.getBodyNode(group["origin_body"]),
            skeleton.getBodyNode(group["insertion_body"]))
    attachments = np.unique(np.r_[origin, insertion])

    def expand_tet_vertex_rings(seed, rings):
        reached = np.unique(np.asarray(seed, dtype=np.int32))
        for _ in range(max(0, int(rings))):
            incident = np.any(np.isin(tets, reached), axis=1)
            reached = np.unique(np.r_[reached, tets[incident].reshape(-1)])
        return reached.astype(np.int32)

    rigid_origin_rings = (
        args.rigid_origin_tet_rings
        if args.rigid_origin_tet_rings is not None
        else args.rigid_attachment_tet_rings)
    rigid_insertion_rings = (
        args.rigid_insertion_tet_rings
        if args.rigid_insertion_tet_rings is not None
        else args.rigid_attachment_tet_rings)
    rigid_origin = expand_tet_vertex_rings(origin, rigid_origin_rings)
    rigid_insertion = expand_tet_vertex_rings(
        insertion, rigid_insertion_rings)
    overlap = np.intersect1d(rigid_origin, rigid_insertion)
    if len(overlap):
        overlap_axis = np.asarray(group["axis_coordinate"])[overlap]
        rigid_origin = np.setdiff1d(
            rigid_origin, overlap[overlap_axis >= 0.5])
        rigid_insertion = np.setdiff1d(
            rigid_insertion, overlap[overlap_axis < 0.5])

    motion = MyBVH(
        str(args.bvh), bvh_info, skeleton,
        T_frame=bake_emu._detect_bvh_tframe(str(args.bvh)))

    origin_body = skeleton.getBodyNode(group["origin_body"])
    insertion_body = skeleton.getBodyNode(group["insertion_body"])
    transport_body = skeleton.getBodyNode(
        args.transport_body or group["origin_body"])
    origin_rest_tf = origin_body.getWorldTransform()
    origin_rest_rotation = np.asarray(origin_rest_tf.rotation()).copy()
    origin_rest_translation = np.asarray(origin_rest_tf.translation()).copy()
    insertion_rest_tf = insertion_body.getWorldTransform()
    insertion_rest_rotation = np.asarray(
        insertion_rest_tf.rotation()).copy()
    insertion_rest_translation = np.asarray(
        insertion_rest_tf.translation()).copy()
    transport_rest_tf = transport_body.getWorldTransform()
    transport_rest_rotation = np.asarray(
        transport_rest_tf.rotation()).copy()
    transport_rest_translation = np.asarray(
        transport_rest_tf.translation()).copy()

    surface_blend = float(np.clip(args.attachment_surface_blend, 0.0, 1.0))
    origin_surface_mesh = (
        "L_Os_Coxae" if group["origin_body"] in {
            "Saccrum_Coccyx0", "L_Os_Coxae0"}
        else "L_Femur")
    origin_surface = closest_surface_points(
        origin_surface_mesh, rest[origin])
    insertion_surface = closest_surface_points(
        "L_Patella", rest[insertion])
    if args.attachment_surface_mode == "rigid":
        origin_surface = rigid_surface_fit(rest[origin], origin_surface)
        insertion_surface = rigid_surface_fit(
            rest[insertion], insertion_surface)
    origin_rest_target = (
        (1.0 - surface_blend) * rest[origin]
        + surface_blend * origin_surface)
    insertion_rest_target = (
        (1.0 - surface_blend) * rest[insertion]
        + surface_blend * insertion_surface)
    rest_attachment_span = max(
        np.linalg.norm(
            np.mean(insertion_rest_target, axis=0)
            - np.mean(origin_rest_target, axis=0)), 1e-8)

    print(
        "attachment surface correction: "
        f"origin mean={np.mean(np.linalg.norm(origin_surface - rest[origin], axis=1)):.6g}m "
        f"insertion mean={np.mean(np.linalg.norm(insertion_surface - rest[insertion], axis=1)):.6g}m "
        f"blend={surface_blend:g}", flush=True)

    def attachment_targets(surface_factor=1.0):
        origin_rotation, origin_translation = relative_transform(
            origin_body, origin_rest_rotation, origin_rest_translation)
        insertion_rotation, insertion_translation = relative_transform(
            insertion_body, insertion_rest_rotation,
            insertion_rest_translation)
        origin_bind = ((1.0 - surface_factor) * rest[origin]
                       + surface_factor * origin_rest_target)
        insertion_bind = ((1.0 - surface_factor) * rest[insertion]
                          + surface_factor * insertion_rest_target)
        return (
            origin_bind @ origin_rotation.T + origin_translation,
            insertion_bind @ insertion_rotation.T
            + insertion_translation)

    axis_coordinate = np.clip(
        np.asarray(group["axis_coordinate"], dtype=np.float64), 0.0, 1.0)
    origin_transport_width = float(np.clip(
        args.origin_transport_width
        if args.origin_transport_width is not None
        else args.endpoint_transport_width, 1e-3, 0.5))
    insertion_transport_width = float(np.clip(
        args.insertion_transport_width
        if args.insertion_transport_width is not None
        else args.endpoint_transport_width, 1e-3, 0.5))

    def _smoothstep01(value):
        value = np.clip(value, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    origin_transport_weight = _smoothstep01(
        (origin_transport_width - axis_coordinate) / origin_transport_width)
    insertion_transport_weight = _smoothstep01(
        (axis_coordinate - (1.0 - insertion_transport_width))
        / insertion_transport_width)

    def bone_carrier_reference():
        """Rest mesh transported by origin/carrier/insertion anatomy.

        Endpoint influence is compactly supported.  Consequently a vastus
        whose origin and carrier are both the femur reduces to its existing
        femur transport, while a biarticular RF follows the femur through its
        belly without detaching its pelvis and patella ends.
        """
        carrier_rotation, carrier_translation = relative_transform(
            transport_body, transport_rest_rotation,
            transport_rest_translation)
        origin_rotation, origin_translation = relative_transform(
            origin_body, origin_rest_rotation, origin_rest_translation)
        insertion_rotation, insertion_translation = relative_transform(
            insertion_body, insertion_rest_rotation,
            insertion_rest_translation)
        carrier_position = rest @ carrier_rotation.T + carrier_translation
        origin_position = rest @ origin_rotation.T + origin_translation
        insertion_position = (
            rest @ insertion_rotation.T + insertion_translation)
        if args.rotation_aware_origin_transport:
            from scipy.spatial.transform import Rotation, Slerp
            origin_blended_rotation = Slerp(
                [0.0, 1.0], Rotation.from_matrix(np.stack((
                    carrier_rotation, origin_rotation))))(
                        origin_transport_weight).as_matrix()
            origin_blended_translation = (
                (1.0 - origin_transport_weight[:, None])
                * carrier_translation
                + origin_transport_weight[:, None] * origin_translation)
            proximal_position = (
                np.einsum("nij,nj->ni", origin_blended_rotation, rest)
                + origin_blended_translation)
        else:
            proximal_position = (
                carrier_position
                + origin_transport_weight[:, None]
                * (origin_position - carrier_position))
        return (proximal_position
                + insertion_transport_weight[:, None]
                * (insertion_position - carrier_position))

    contact_faces = np.asarray(
        data.get("collision_faces", surface_faces(tets)), dtype=np.int32)
    sample_i, sample_w = dense_boundary_samples(contact_faces)
    attachment_mask = np.zeros(len(rest), dtype=bool)
    attachment_mask[attachments] = True
    active = sample_w > 1e-12
    pure_attachment = np.all(
        ~active | attachment_mask[sample_i], axis=1)
    sample_i = sample_i[~pure_attachment]
    sample_w = sample_w[~pure_attachment]
    no_contact_exclusion = np.zeros(len(rest), dtype=bool)

    sdf = BoneSDF(args.sdf)
    sdf.bind_rest_clearance(
        rest, sample_i, sample_w, skeleton,
        prevent_new_only=True, margin=args.collision_margin)

    tet_t = torch.as_tensor(tets, dtype=torch.long, device=device)
    rest_tet = torch.as_tensor(rest, dtype=dtype, device=device)[tet_t]
    rest_dm = torch.stack((
        rest_tet[:, 1] - rest_tet[:, 0],
        rest_tet[:, 2] - rest_tet[:, 0],
        rest_tet[:, 3] - rest_tet[:, 0]), dim=2)
    rest_dm_inverse = torch.linalg.inv(rest_dm)
    axis_coordinate = np.asarray(group["axis_coordinate"], dtype=np.float64)
    tet_axis_delta = np.stack((
        axis_coordinate[tets[:, 1]] - axis_coordinate[tets[:, 0]],
        axis_coordinate[tets[:, 2]] - axis_coordinate[tets[:, 0]],
        axis_coordinate[tets[:, 3]] - axis_coordinate[tets[:, 0]]), axis=1)
    rest_dm_inverse_np = np.linalg.inv(np.asarray(rest_dm.cpu()))
    tet_longitudinal_axis = np.einsum(
        "nji,nj->ni", rest_dm_inverse_np, tet_axis_delta)
    tet_axis_length = np.linalg.norm(tet_longitudinal_axis, axis=1)
    valid_tet_axis = tet_axis_length > 1e-12
    tet_longitudinal_axis[valid_tet_axis] /= tet_axis_length[
        valid_tet_axis, None]
    tet_longitudinal_axis[~valid_tet_axis] = np.array([0.0, 1.0, 0.0])
    saved_fiber_directions = data.get("fiber_directions")
    if saved_fiber_directions is not None:
        saved_fiber_directions = np.asarray(
            saved_fiber_directions, dtype=np.float64)
        if saved_fiber_directions.shape != (len(tets), 3):
            raise RuntimeError(
                "saved fiber_directions shape does not match tetrahedra: "
                f"{saved_fiber_directions.shape} != {(len(tets), 3)}")
        saved_fiber_norm = np.linalg.norm(
            saved_fiber_directions, axis=1)
        saved_fiber_valid = np.isfinite(saved_fiber_norm) & (
            saved_fiber_norm > 1e-10)
        tet_longitudinal_axis[saved_fiber_valid] = (
            saved_fiber_directions[saved_fiber_valid]
            / saved_fiber_norm[saved_fiber_valid, None])
        print(
            "material fibers: volumetric Laplace gradients "
            f"{int(np.sum(saved_fiber_valid))}/{len(tets)} tets",
            flush=True)
    tet_longitudinal_axis_t = torch.as_tensor(
        tet_longitudinal_axis, dtype=dtype, device=device)
    # A coherent transverse material director distinguishes a thin muscle
    # sheet's rotational phase around its fiber axis. Isotropic ARAP has no
    # such long-range phase reference and can accumulate a corkscrew while
    # every individual tet remains nearly rigid.
    centered_rest = rest - np.mean(rest, axis=0)
    _, _, rest_vh = np.linalg.svd(centered_rest, full_matrices=False)
    sheet_normal = rest_vh[-1]
    tet_transverse_axis = np.cross(
        np.broadcast_to(sheet_normal, tet_longitudinal_axis.shape),
        tet_longitudinal_axis)
    transverse_length = np.linalg.norm(tet_transverse_axis, axis=1)
    degenerate_transverse = transverse_length < 1e-8
    if np.any(degenerate_transverse):
        fallback = np.cross(
            np.broadcast_to(np.array([1.0, 0.0, 0.0]),
                            tet_longitudinal_axis.shape),
            tet_longitudinal_axis)
        fallback_length = np.linalg.norm(fallback, axis=1)
        second_fallback = fallback_length < 1e-8
        fallback[second_fallback] = np.cross(
            np.broadcast_to(np.array([0.0, 1.0, 0.0]),
                            (np.sum(second_fallback), 3)),
            tet_longitudinal_axis[second_fallback])
        tet_transverse_axis[degenerate_transverse] = fallback[
            degenerate_transverse]
        transverse_length = np.linalg.norm(tet_transverse_axis, axis=1)
    tet_transverse_axis /= np.maximum(transverse_length[:, None], 1e-12)
    tet_transverse_axis_t = torch.as_tensor(
        tet_transverse_axis, dtype=dtype, device=device)
    rest_volume_np = signed_volumes(rest, tets)
    rest_volume = torch.as_tensor(
        rest_volume_np, dtype=dtype, device=device)
    rest_sign = torch.sign(rest_volume)
    rest_abs = torch.abs(rest_volume).clamp_min(1e-14)
    tet_weight = rest_abs / torch.sum(rest_abs)
    tet_axis_coordinate = np.mean(axis_coordinate[tets], axis=1)
    endpoint_arap_width = float(np.clip(
        args.endpoint_arap_width, 1e-3, 0.5))
    origin_tet_stiffness = _smoothstep01(
        (endpoint_arap_width - tet_axis_coordinate)
        / endpoint_arap_width)
    insertion_tet_stiffness = _smoothstep01(
        (tet_axis_coordinate - (1.0 - endpoint_arap_width))
        / endpoint_arap_width)
    endpoint_tet_stiffness = np.maximum(
        origin_tet_stiffness, insertion_tet_stiffness)
    arap_tet_weight = tet_weight * torch.as_tensor(
        1.0 + (max(1.0, args.endpoint_arap_scale) - 1.0)
        * endpoint_tet_stiffness,
        dtype=dtype, device=device)
    graph_edges = unique_edges(tets)
    fiber_bend_center = np.empty(0, dtype=np.int32)
    fiber_bend_lower = np.empty(0, dtype=np.int32)
    fiber_bend_upper = np.empty(0, dtype=np.int32)
    saved_laplace_field = data.get("laplace_field")
    if args.fiber_bending_weight > 0.0:
        if saved_laplace_field is None:
            raise RuntimeError(
                "--fiber-bending-weight requires saved laplace_field")
        saved_laplace_field = np.asarray(
            saved_laplace_field, dtype=np.float64)
        if saved_laplace_field.shape != (len(rest),):
            raise RuntimeError("saved laplace_field vertex count mismatch")
        adjacency = [[] for _ in range(len(rest))]
        for edge_a, edge_b in graph_edges:
            adjacency[int(edge_a)].append(int(edge_b))
            adjacency[int(edge_b)].append(int(edge_a))
        centers, lowers, uppers = [], [], []
        for center, neighbors in enumerate(adjacency):
            if not neighbors:
                continue
            neighbor = np.asarray(neighbors, dtype=np.int32)
            delta = saved_laplace_field[neighbor] - saved_laplace_field[center]
            lower_mask = delta < -1e-8
            upper_mask = delta > 1e-8
            if not np.any(lower_mask) or not np.any(upper_mask):
                continue
            lower_candidates = neighbor[lower_mask]
            upper_candidates = neighbor[upper_mask]
            lower = lower_candidates[np.argmax(
                (saved_laplace_field[center]
                 - saved_laplace_field[lower_candidates])
                / np.maximum(np.linalg.norm(
                    rest[lower_candidates] - rest[center], axis=1), 1e-8))]
            upper = upper_candidates[np.argmax(
                (saved_laplace_field[upper_candidates]
                 - saved_laplace_field[center])
                / np.maximum(np.linalg.norm(
                    rest[upper_candidates] - rest[center], axis=1), 1e-8))]
            centers.append(center)
            lowers.append(int(lower))
            uppers.append(int(upper))
        fiber_bend_center = np.asarray(centers, dtype=np.int32)
        fiber_bend_lower = np.asarray(lowers, dtype=np.int32)
        fiber_bend_upper = np.asarray(uppers, dtype=np.int32)
        print(
            f"fiber bending guides: {len(fiber_bend_center)} vertices",
            flush=True)
    insertion_edge_mask = np.logical_xor(
        np.isin(graph_edges[:, 0], insertion),
        np.isin(graph_edges[:, 1], insertion))
    insertion_transition_edges = graph_edges[insertion_edge_mask]
    transition_first_attached = np.isin(
        insertion_transition_edges[:, 0], insertion)
    insertion_transition_anchor = np.where(
        transition_first_attached,
        insertion_transition_edges[:, 0],
        insertion_transition_edges[:, 1]).astype(np.int32)
    insertion_transition_free = np.where(
        transition_first_attached,
        insertion_transition_edges[:, 1],
        insertion_transition_edges[:, 0]).astype(np.int32)
    insertion_transition_anchor_t = torch.as_tensor(
        insertion_transition_anchor, dtype=torch.long, device=device)
    insertion_transition_free_t = torch.as_tensor(
        insertion_transition_free, dtype=torch.long, device=device)
    insertion_transition_free_degree = torch.zeros(
        len(rest), dtype=dtype, device=device)
    insertion_transition_free_degree.index_add_(
        0, insertion_transition_free_t,
        torch.ones(len(insertion_transition_free_t),
                   dtype=dtype, device=device))
    insertion_transition_free_degree.clamp_min_(1.0)
    insertion_transition_edge_t = torch.as_tensor(
        insertion_transition_edges, dtype=torch.long, device=device)
    insertion_transition_rest_t = torch.as_tensor(
        rest, dtype=dtype, device=device)
    insertion_transition_rest_length = torch.linalg.vector_norm(
        insertion_transition_rest_t[insertion_transition_edge_t[:, 1]]
        - insertion_transition_rest_t[insertion_transition_edge_t[:, 0]],
        dim=1).clamp_min(1e-8)
    graph_i = torch.as_tensor(
        graph_edges[:, 0], dtype=torch.long, device=device)
    graph_j = torch.as_tensor(
        graph_edges[:, 1], dtype=torch.long, device=device)
    rest_t = torch.as_tensor(rest, dtype=dtype, device=device)
    graph_rest_edge = rest_t[graph_i] - rest_t[graph_j]
    graph_weight = 1.0 / torch.linalg.vector_norm(
        graph_rest_edge, dim=1).clamp_min(1e-8)
    graph_degree = torch.zeros(len(rest), dtype=dtype, device=device)
    graph_degree.index_add_(0, graph_i, graph_weight)
    graph_degree.index_add_(0, graph_j, graph_weight)
    graph_degree = graph_degree.clamp_min(1e-12)
    fiber_bend_center_t = torch.as_tensor(
        fiber_bend_center, dtype=torch.long, device=device)
    fiber_bend_lower_t = torch.as_tensor(
        fiber_bend_lower, dtype=torch.long, device=device)
    fiber_bend_upper_t = torch.as_tensor(
        fiber_bend_upper, dtype=torch.long, device=device)

    def graph_laplacian(values):
        edge_value = values[graph_i] - values[graph_j]
        result = torch.zeros_like(values)
        result.index_add_(0, graph_i, graph_weight[:, None] * edge_value)
        result.index_add_(0, graph_j, -graph_weight[:, None] * edge_value)
        return result / graph_degree[:, None]

    rest_laplacian = graph_laplacian(rest_t)
    laplacian_scale = torch.mean(
        torch.linalg.vector_norm(graph_rest_edge, dim=1) ** 2
    ).clamp_min(1e-12)
    vertex_tet_mass = torch.zeros(len(rest), dtype=dtype, device=device)
    for corner in range(4):
        vertex_tet_mass.index_add_(0, tet_t[:, corner], rest_abs)
    vertex_tet_mass = vertex_tet_mass.clamp_min(1e-14)
    representative_tet = np.zeros(len(rest), dtype=np.int32)
    representative_volume = np.full(len(rest), -np.inf)
    rest_abs_np = np.abs(rest_volume_np)
    for tet_index, tet_vertices in enumerate(tets):
        for vertex in tet_vertices:
            if rest_abs_np[tet_index] > representative_volume[vertex]:
                representative_volume[vertex] = rest_abs_np[tet_index]
                representative_tet[vertex] = tet_index
    representative_tet_t = torch.as_tensor(
        representative_tet, dtype=torch.long, device=device)
    origin_t = torch.as_tensor(origin, dtype=torch.long, device=device)
    insertion_t = torch.as_tensor(
        insertion, dtype=torch.long, device=device)
    rigid_origin_t = torch.as_tensor(
        rigid_origin, dtype=torch.long, device=device)
    rigid_insertion_t = torch.as_tensor(
        rigid_insertion, dtype=torch.long, device=device)
    soft_insertion_ring_t = [
        (torch.as_tensor(ring, dtype=torch.long, device=device), weight)
        for ring, weight in soft_insertion_rings]

    x = torch.as_tensor(rest, dtype=dtype, device=device).clone()
    previous_pose = np.zeros(skeleton.getNumDofs())
    previous_rotation = np.eye(3)
    previous_translation = np.zeros(3)
    stable_x = x.clone()
    stable_pose = previous_pose.copy()
    stable_rotation = previous_rotation.copy()
    stable_translation = previous_translation.copy()
    needs_rollback = False
    frames_out = []
    frame_ids_out = []
    metrics = []

    def save_checkpoint(done=False):
        if not frames_out:
            return
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_np = np.asarray(metrics)
        np.savez_compressed(
            args.output_dir / f"{args.name}_chunk_0000.npz",
            frames=np.asarray(frame_ids_out, dtype=np.int32),
            positions=np.asarray(frames_out, dtype=np.float32),
            inverted_tets=metrics_np[:, 0].astype(np.int32),
            minimum_jacobian_ratio=metrics_np[:, 1],
            maximum_attachment_error=metrics_np[:, 2],
            remaining_contact_samples=metrics_np[:, 3].astype(np.int32))
        if done:
            (args.output_dir / ".done").touch()

    all_poses = motion.mocap_refs
    poses = all_poses
    if args.end_frame is not None:
        poses = poses[:args.end_frame + 1]
    if args.start_frame > 0:
        if args.initialization_cache is None and not args.muscle_material:
            parser.error("--start-frame requires --initialization-cache")
        if args.initialization_cache is not None:
            cache_path = args.initialization_cache
            if cache_path.is_dir():
                candidates = sorted(cache_path.glob("*_chunk_*.npz"))
                if not candidates:
                    parser.error("initialization cache has no chunk")
                cache_path = candidates[0]
            cache = np.load(cache_path)
            wanted = (args.initialization_cache_frame
                      if args.initialization_cache_frame is not None
                      else args.start_frame - 1)
            rows = np.where(np.asarray(cache["frames"]) == wanted)[0]
            if not len(rows):
                parser.error(f"initialization cache has no frame {wanted}")
            initial = np.asarray(cache["positions"][rows[0]], dtype=np.float64)
            if initial.shape != rest.shape:
                parser.error("initialization cache topology does not match tet")
            x = torch.as_tensor(initial, dtype=dtype, device=device).clone()
            pose_frame = (args.initialization_pose_frame
                          if args.initialization_pose_frame is not None
                          else wanted)
            if pose_frame < 0 or pose_frame >= len(all_poses):
                parser.error(
                    f"initialization pose frame {pose_frame} is outside motion")
            previous_pose = np.asarray(all_poses[pose_frame]).copy()
            skeleton.setPositions(previous_pose.copy())
            previous_rotation, previous_translation = relative_transform(
                transport_body, transport_rest_rotation,
                transport_rest_translation)
            stable_x = x.clone()
            stable_pose = previous_pose.copy()
            stable_rotation = previous_rotation.copy()
            stable_translation = previous_translation.copy()
        else:
            print(
                f"frame {args.start_frame}: cache-free continuation from "
                "rest pose", flush=True)
    independent_x = x.clone()
    independent_pose = previous_pose.copy()
    independent_rotation = previous_rotation.copy()
    independent_translation = previous_translation.copy()
    difficult_frames = set()
    if args.difficult_frame_ranges:
        for item in args.difficult_frame_ranges.split(","):
            lo, hi = (int(v) for v in item.split("-", 1))
            difficult_frames.update(range(lo, hi + 1))
    for frame in range(args.start_frame, len(poses)):
        if args.independent_frames:
            x = independent_x.clone()
            previous_pose = independent_pose.copy()
            previous_rotation = independent_rotation.copy()
            previous_translation = independent_translation.copy()
            skeleton.setPositions(previous_pose.copy())
            needs_rollback = False
        frame_pose = poses[frame]
        frame_smooth_weight = args.smooth_arap_weight
        if args.smooth_arap_ramp_frames > 0:
            frame_smooth_weight *= min(
                1.0, (frame - args.start_frame + 1)
                / args.smooth_arap_ramp_frames)
        if needs_rollback:
            x = stable_x.clone()
            previous_pose = stable_pose.copy()
            previous_rotation = stable_rotation.copy()
            previous_translation = stable_translation.copy()
            print(
                f"frame {frame}: rollback to last healthy state before solve",
                flush=True)
        frame_substeps = (
            args.difficult_substeps
            if frame in difficult_frames and args.difficult_substeps > 0
            else args.substeps)
        if args.maximum_attachment_step > 0.0 and frame > 0:
            skeleton.setPositions(previous_pose.copy())
            previous_origin_target, previous_insertion_target = (
                attachment_targets())
            skeleton.setPositions(frame_pose.copy())
            current_origin_target, current_insertion_target = (
                attachment_targets())
            target_delta = np.linalg.norm(
                np.concatenate((current_origin_target, current_insertion_target))
                - np.concatenate((previous_origin_target,
                                  previous_insertion_target)), axis=1)
            displacement_substeps = int(np.ceil(
                float(np.max(target_delta))
                / args.maximum_attachment_step))
            frame_substeps = max(frame_substeps, displacement_substeps)
            if displacement_substeps > args.substeps:
                print(
                    f"frame {frame}: adaptive attachment continuation "
                    f"uses {frame_substeps} substeps",
                    flush=True)
        substep_previous_pose = previous_pose.copy()
        for substep in range(1, frame_substeps + 1):
            alpha = substep / frame_substeps
            pose = (1.0 - alpha) * previous_pose + alpha * frame_pose
            if args.bone_carrier_endpoint_predictor:
                skeleton.setPositions(substep_previous_pose.copy())
                previous_reference = bone_carrier_reference()
            skeleton.setPositions(pose.copy())
            rotation, translation = relative_transform(
                transport_body, transport_rest_rotation,
                transport_rest_translation)
            incremental_rotation = rotation @ previous_rotation.T
            incremental_translation = (
                translation - incremental_rotation @ previous_translation)
            rotation_t = torch.as_tensor(
                incremental_rotation, dtype=dtype, device=device)
            translation_t = torch.as_tensor(
                incremental_translation, dtype=dtype, device=device)
            x = (x @ rotation_t.T + translation_t).detach()
            if args.bone_carrier_endpoint_predictor:
                current_reference = bone_carrier_reference()
                transported_previous_reference = (
                    previous_reference @ incremental_rotation.T
                    + incremental_translation)
                reference_correction = torch.as_tensor(
                    current_reference - transported_previous_reference,
                    dtype=dtype, device=device)
                x = (x + reference_correction).detach()
                substep_previous_pose = pose.copy()
            previous_rotation = rotation.copy()
            previous_translation = translation.copy()

            # Introduce the rest-to-surface correction quasistatically on the
            # first frame. Applying the full correction in its first optimizer
            # iteration creates exactly the cap pull-away this option fixes.
            surface_factor = alpha if frame == 0 else 1.0
            origin_target_np, insertion_target_np = attachment_targets(
                surface_factor)
            origin_target = torch.as_tensor(
                origin_target_np, dtype=dtype, device=device)
            insertion_target = torch.as_tensor(
                insertion_target_np, dtype=dtype, device=device)
            active_fiber_target = float(args.fiber_target_stretch)
            if args.fiber_target_from_attachments:
                current_attachment_span = np.linalg.norm(
                    np.mean(insertion_target_np, axis=0)
                    - np.mean(origin_target_np, axis=0))
                active_fiber_target = float(np.clip(
                    current_attachment_span / rest_attachment_span,
                    0.6, 1.1))
            if args.hard_attachments:
                rigid_origin_rotation, rigid_origin_translation = (
                    relative_transform(
                        origin_body, origin_rest_rotation,
                        origin_rest_translation))
                rigid_insertion_rotation, rigid_insertion_translation = (
                    relative_transform(
                        insertion_body, insertion_rest_rotation,
                        insertion_rest_translation))
                rigid_origin_target = torch.as_tensor(
                    rest[rigid_origin] @ rigid_origin_rotation.T
                    + rigid_origin_translation,
                    dtype=dtype, device=device)
                rigid_insertion_target = torch.as_tensor(
                    rest[rigid_insertion] @ rigid_insertion_rotation.T
                    + rigid_insertion_translation,
                    dtype=dtype, device=device)
                # Advance the endpoint before evaluating deformation.  The
                # free solve therefore sees the real Dirichlet boundary for
                # the whole iteration instead of a soft, lagging endpoint.
                x = x.clone()
                x[rigid_origin_t] = rigid_origin_target
                x[rigid_insertion_t] = rigid_insertion_target
            insertion_rotation, insertion_translation = relative_transform(
                insertion_body, insertion_rest_rotation,
                insertion_rest_translation)
            soft_insertion_targets = [torch.as_tensor(
                rest[ring] @ insertion_rotation.T + insertion_translation,
                dtype=dtype, device=device)
                for ring, _ in soft_insertion_rings]
            desired_transverse = None
            if args.director_weight > 0.0:
                if not args.bone_carrier_endpoint_predictor:
                    raise RuntimeError(
                        "--director-weight requires "
                        "--bone-carrier-endpoint-predictor")
                reference_t = torch.as_tensor(
                    current_reference, dtype=dtype, device=device)
                reference_tet = reference_t[tet_t]
                reference_dm = torch.stack((
                    reference_tet[:, 1] - reference_tet[:, 0],
                    reference_tet[:, 2] - reference_tet[:, 0],
                    reference_tet[:, 3] - reference_tet[:, 0]), dim=2)
                reference_deformation = reference_dm @ rest_dm_inverse
                reference_u, _, reference_vh = torch.linalg.svd(
                    reference_deformation)
                reference_orientation = torch.linalg.det(
                    reference_u @ reference_vh)
                reference_correction = torch.ones(
                    (len(tets), 3), dtype=dtype, device=device)
                reference_correction[:, 2] = reference_orientation
                reference_rotation = (
                    reference_u @ torch.diag_embed(reference_correction)
                    @ reference_vh).detach()
                desired_transverse = torch.einsum(
                    "nij,nj->ni", reference_rotation,
                    tet_transverse_axis_t).detach()
            desired_fiber_bend = None
            if args.fiber_bending_weight > 0.0:
                reference_t = torch.as_tensor(
                    current_reference, dtype=dtype, device=device)
                reference_upper = reference_t[fiber_bend_upper_t]
                reference_center = reference_t[fiber_bend_center_t]
                reference_lower = reference_t[fiber_bend_lower_t]
                reference_forward = torch.nn.functional.normalize(
                    reference_upper - reference_center, dim=1, eps=1e-8)
                reference_backward = torch.nn.functional.normalize(
                    reference_center - reference_lower, dim=1, eps=1e-8)
                desired_fiber_bend = (
                    reference_forward - reference_backward).detach()
            x.requires_grad_(True)
            optimizer = torch.optim.Adam(
                [x], lr=args.learning_rate,
                fused=(device.type == "cuda"))

            attachment_error_now = float("inf")
            rotation_fit = None
            for iteration in range(args.maximum_iterations):
                before = x.detach().clone()
                optimizer.zero_grad()
                current_tet = x[tet_t]
                current_dm = torch.stack((
                    current_tet[:, 1] - current_tet[:, 0],
                    current_tet[:, 2] - current_tet[:, 0],
                    current_tet[:, 3] - current_tet[:, 0]), dim=2)
                deformation = current_dm @ rest_dm_inverse
                if (rotation_fit is None or iteration
                        % max(1, args.rotation_update_interval) == 0):
                    u, _, vh = torch.linalg.svd(deformation)
                    orientation = torch.linalg.det(u @ vh)
                    correction = torch.ones(
                        (len(tets), 3), dtype=dtype, device=device)
                    correction[:, 2] = orientation
                    rotation_fit = (
                        u @ torch.diag_embed(correction) @ vh).detach()
                arap_error = deformation - rotation_fit
                longitudinal_error = torch.einsum(
                    "nij,nj->ni", arap_error, tet_longitudinal_axis_t)
                longitudinal_scale = float(np.clip(
                    args.longitudinal_arap_scale, 0.0, 1.0))
                arap = torch.sum(
                    arap_tet_weight[:, None, None] * arap_error ** 2)
                arap = arap - (1.0 - longitudinal_scale) * torch.sum(
                    arap_tet_weight[:, None] * longitudinal_error ** 2)
                fiber = torch.zeros((), dtype=dtype, device=device)
                if args.fiber_weight > 0.0:
                    deformed_fiber = torch.einsum(
                        "nij,nj->ni", deformation,
                        tet_longitudinal_axis_t)
                    fiber_stretch = torch.linalg.vector_norm(
                        deformed_fiber, dim=1)
                    fiber = torch.sum(
                        tet_weight * (
                            fiber_stretch
                            - active_fiber_target) ** 2)
                fiber_bending = torch.zeros(
                    (), dtype=dtype, device=device)
                if desired_fiber_bend is not None:
                    current_forward = torch.nn.functional.normalize(
                        x[fiber_bend_upper_t] - x[fiber_bend_center_t],
                        dim=1, eps=1e-8)
                    current_backward = torch.nn.functional.normalize(
                        x[fiber_bend_center_t] - x[fiber_bend_lower_t],
                        dim=1, eps=1e-8)
                    current_fiber_bend = (
                        current_forward - current_backward)
                    fiber_bending = torch.mean(torch.sum(
                        (current_fiber_bend - desired_fiber_bend) ** 2,
                        dim=1))
                director = torch.zeros((), dtype=dtype, device=device)
                if desired_transverse is not None:
                    current_transverse = torch.einsum(
                        "nij,nj->ni", deformation,
                        tet_transverse_axis_t)
                    current_transverse = (
                        current_transverse
                        / torch.linalg.vector_norm(
                            current_transverse, dim=1,
                            keepdim=True).clamp_min(1e-8))
                    # Squared cross-product magnitude is insensitive to
                    # stretch but penalizes angular material-frame drift.
                    director_cross = torch.linalg.cross(
                        current_transverse, desired_transverse, dim=1)
                    director = torch.sum(
                        tet_weight[:, None] * director_cross ** 2)
                smooth_arap = torch.zeros((), dtype=dtype, device=device)
                if frame_smooth_weight > 0.0:
                    # Smooth ARAP: retain rotated Laplacian vectors while
                    # deriving rotations solely from the ordinary ARAP term.
                    if args.fast_smooth_arap:
                        vertex_rotation = rotation_fit[
                            representative_tet_t].detach()
                    else:
                        vertex_rotation = torch.zeros(
                            (len(rest), 3, 3), dtype=dtype, device=device)
                        for corner in range(4):
                            vertex_rotation.index_add_(
                                0, tet_t[:, corner],
                                rest_abs[:, None, None] * rotation_fit)
                        vertex_rotation = (
                            vertex_rotation / vertex_tet_mass[:, None, None])
                        vu, _, vvh = torch.linalg.svd(vertex_rotation)
                        vertex_rotation = (vu @ vvh).detach()
                        negative_vertex = (
                            torch.linalg.det(vertex_rotation) < 0.0)
                        if torch.any(negative_vertex):
                            vu = vu.clone()
                            vu[negative_vertex, :, -1] *= -1.0
                            vertex_rotation = (vu @ vvh).detach()
                    wanted_laplacian = torch.einsum(
                        "nij,nj->ni", vertex_rotation, rest_laplacian)
                    laplacian_error = (
                        graph_laplacian(x) - wanted_laplacian)
                    smooth_arap = torch.mean(
                        torch.sum(laplacian_error ** 2, dim=1)
                    ) / laplacian_scale

                ratio = (
                    signed_volumes_cuda(x, tet_t)
                    * rest_sign / rest_abs)
                matrix = torch.zeros((), dtype=dtype, device=device)
                inversion_barrier = torch.zeros(
                    (), dtype=dtype, device=device)
                if args.muscle_material:
                    # Objective isochoric matrix energy. C=F^T F enters only
                    # through tr(C), so rigid body motion costs exactly zero.
                    # Removing J^(2/3) leaves volume response to the separate
                    # near-incompressibility term below.
                    safe_j = ratio.clamp_min(args.minimum_jacobian)
                    i1 = torch.sum(deformation * deformation, dim=(1, 2))
                    isochoric_i1 = i1 * torch.pow(safe_j, -2.0 / 3.0)
                    matrix = torch.sum(
                        tet_weight * torch.relu(isochoric_i1 - 3.0))

                    # Shifted barrier is zero with zero slope at its activation
                    # threshold, but diverges as J approaches zero. The hard
                    # line search below remains the final inversion safeguard.
                    barrier_start = max(
                        args.inversion_barrier_start,
                        args.minimum_jacobian * 2.0)
                    normalized_j = safe_j / barrier_start
                    barrier_value = torch.where(
                        ratio < barrier_start,
                        -torch.log(normalized_j) + normalized_j - 1.0,
                        torch.zeros_like(ratio))
                    inversion_barrier = torch.sum(
                        tet_weight * barrier_value)
                volume = torch.sum(tet_weight * (ratio - 1.0) ** 2)
                quality_error = torch.relu(0.1 - ratio) ** 2
                if args.localized_quality_barrier:
                    quality = torch.mean(torch.topk(
                        quality_error,
                        min(256, len(tets)), largest=True).values)
                else:
                    quality = torch.mean(quality_error)
                contact = sdf.torch_penetration_loss(
                    x, sample_i, sample_w, no_contact_exclusion,
                    skeleton, args.collision_tolerance)
                origin_distance = torch.linalg.vector_norm(
                    x[origin_t] - origin_target, dim=1)
                insertion_distance = torch.linalg.vector_norm(
                    x[insertion_t] - insertion_target, dim=1)
                origin_error_now = torch.max(origin_distance)
                insertion_error_now = torch.max(insertion_distance)
                attachment_error_now = float(torch.max(
                    origin_error_now, insertion_error_now).detach())
                strength_scale = max(
                    args.attachment_strength_scale, 1e-8)
                origin_multiplier = min(
                    1e4, max(1.0, float(
                        origin_error_now.detach()) / strength_scale) ** 2)
                insertion_multiplier = min(
                    1e4, max(1.0, float(
                        insertion_error_now.detach()) / strength_scale) ** 2)
                origin_loss = torch.mean(origin_distance ** 2) / 1e-6
                insertion_loss = torch.mean(insertion_distance ** 2) / 1e-6
                insertion_edge_loss = torch.zeros(
                    (), dtype=dtype, device=device)
                if (args.insertion_edge_weight > 0.0
                        and len(insertion_transition_edges) > 0):
                    insertion_edge_length = torch.linalg.vector_norm(
                        x[insertion_transition_edge_t[:, 1]]
                        - x[insertion_transition_edge_t[:, 0]], dim=1)
                    insertion_edge_stretch = (
                        insertion_edge_length
                        / insertion_transition_rest_length)
                    insertion_edge_loss = torch.mean(
                        torch.relu(insertion_edge_stretch - 1.5) ** 2)
                soft_insertion_loss = torch.zeros(
                    (), dtype=dtype, device=device)
                for ((ring_t, ring_weight), ring_target) in zip(
                        soft_insertion_ring_t, soft_insertion_targets):
                    if len(ring_t) > 0:
                        ring_distance = torch.linalg.vector_norm(
                            x[ring_t] - ring_target, dim=1)
                        soft_insertion_loss = soft_insertion_loss + (
                            ring_weight * torch.mean(ring_distance ** 2) / 1e-6)
                loss = (
                    args.arap_weight * arap
                    + args.matrix_weight * matrix
                    + args.inversion_barrier_weight * inversion_barrier
                    + args.director_weight * director
                    + args.fiber_weight * fiber
                    + args.fiber_bending_weight * fiber_bending
                    + frame_smooth_weight * smooth_arap
                    + args.volume_weight * volume
                    + args.quality_weight * quality
                    + args.contact_weight * contact
                    + args.origin_weight * origin_multiplier * origin_loss
                    + args.insertion_weight * insertion_multiplier
                    * insertion_loss
                    + args.insertion_edge_weight * insertion_edge_loss
                    + args.insertion_weight * soft_insertion_loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([x], 1e6)
                optimizer.step()
                with torch.no_grad():
                    delta = x - before
                    length = torch.linalg.vector_norm(
                        delta, dim=1).clamp_min(1e-20)
                    scale = torch.clamp(
                        args.maximum_step / length, max=1.0)
                    delta = delta * scale[:, None]
                    before_ratio = (
                        signed_volumes_cuda(before, tet_t)
                        * rest_sign / rest_abs)
                    before_minimum = float(torch.min(before_ratio))
                    accepted = False
                    step_scale = 1.0
                    vertex_scale = torch.ones(
                        len(x), dtype=dtype, device=device)
                    for _ in range(20):
                        if args.local_jacobian_line_search:
                            candidate = (
                                before + vertex_scale[:, None] * delta)
                        else:
                            candidate = before + step_scale * delta
                        if args.hard_attachments:
                            candidate[rigid_origin_t] = rigid_origin_target
                            candidate[rigid_insertion_t] = (
                                rigid_insertion_target)
                        candidate_ratio = (
                            signed_volumes_cuda(candidate, tet_t)
                            * rest_sign / rest_abs)
                        candidate_minimum = float(
                            torch.min(candidate_ratio))
                        if (
                            (before_minimum > args.minimum_jacobian
                             and candidate_minimum > args.minimum_jacobian)
                            or
                            (before_minimum <= args.minimum_jacobian
                             and candidate_minimum > before_minimum)
                        ):
                            x[:] = candidate
                            accepted = True
                            break
                        if args.local_jacobian_line_search:
                            if before_minimum > args.minimum_jacobian:
                                bad_tets = torch.where(
                                    candidate_ratio
                                    <= args.minimum_jacobian)[0]
                            else:
                                bad_tets = torch.where(
                                    candidate_ratio <= before_ratio)[0]
                            if not len(bad_tets):
                                break
                            bad_vertices = torch.unique(
                                tet_t[bad_tets].reshape(-1))
                            vertex_scale[bad_vertices] *= 0.5
                        else:
                            step_scale *= 0.5
                    if not accepted:
                        x[:] = before
                if (
                    iteration + 1 >= args.iterations
                    and attachment_error_now <= args.attachment_tolerance
                ):
                    break
            x = x.detach()
            # If an element reaches the hard floor, restore local Jacobian
            # slack before advancing the motion. This prevents one ordinary
            # pose from poisoning every later frame. The repair is generated
            # from the current tet state only and uses no deformation cache.
            for _ in range(200):
                x.requires_grad_(True)
                repair_ratio = (
                    signed_volumes_cuda(x, tet_t)
                    * rest_sign / rest_abs)
                repair_minimum = float(torch.min(repair_ratio))
                if repair_minimum >= 0.01:
                    x = x.detach()
                    break
                repair_loss = torch.mean(
                    torch.relu(0.02 - repair_ratio) ** 2)
                repair_gradient, = torch.autograd.grad(repair_loss, x)
                direction = -repair_gradient
                if args.hard_attachments:
                    direction[rigid_origin_t] = 0.0
                    direction[rigid_insertion_t] = 0.0
                maximum = torch.max(torch.linalg.vector_norm(
                    direction, dim=1)).clamp_min(1e-20)
                direction = direction * (1e-4 / maximum)
                before_repair = x.detach()
                accepted_repair = False
                repair_scale = 1.0
                for _ in range(20):
                    repair_candidate = (
                        before_repair + repair_scale * direction)
                    candidate_ratio = (
                        signed_volumes_cuda(repair_candidate, tet_t)
                        * rest_sign / rest_abs)
                    if float(torch.min(candidate_ratio)) > repair_minimum:
                        x = repair_candidate.detach()
                        accepted_repair = True
                        break
                    repair_scale *= 0.5
                if not accepted_repair:
                    x = before_repair
                    break
            # Alternate cap advancement with Jacobian-slack restoration.
            # This resolves the case where either constraint is individually
            # feasible but a one-shot solve leaves the other at its boundary.
            cap_ids = torch.cat((origin_t, insertion_t))
            cap_targets = torch.cat((origin_target, insertion_target))
            for _ in range(
                    0 if (args.disable_cap_repair
                          or args.hard_attachments) else 20):
                cap_error = torch.linalg.vector_norm(
                    x[cap_ids] - cap_targets, dim=1)
                if float(torch.max(cap_error)) <= args.attachment_tolerance:
                    break
                before_cap = x
                cap_delta = torch.zeros_like(x)
                cap_delta[cap_ids] = (
                    0.25 * (cap_targets - x[cap_ids]))
                repair_neighbor_blend = float(np.clip(
                    args.attachment_repair_neighbor_blend, 0.0, 1.0))
                if (repair_neighbor_blend > 0.0
                        and len(insertion_transition_edges) > 0):
                    neighbor_delta = torch.zeros_like(x)
                    neighbor_delta.index_add_(
                        0, insertion_transition_free_t,
                        cap_delta[insertion_transition_anchor_t])
                    neighbor_delta = (
                        neighbor_delta
                        / insertion_transition_free_degree[:, None])
                    free_vertices = torch.unique(insertion_transition_free_t)
                    cap_delta[free_vertices] = (
                        repair_neighbor_blend
                        * neighbor_delta[free_vertices])
                cap_scale = 1.0
                accepted_cap = False
                for _ in range(20):
                    cap_candidate = before_cap + cap_scale * cap_delta
                    cap_ratio = (
                        signed_volumes_cuda(cap_candidate, tet_t)
                        * rest_sign / rest_abs)
                    if float(torch.min(cap_ratio)) > args.minimum_jacobian:
                        x = cap_candidate
                        accepted_cap = True
                        break
                    cap_scale *= 0.5
                if not accepted_cap:
                    break
                # Make room for the next cap increment by lifting the weakest
                # Jacobians without applying an ARAP tether.
                for _ in range(50):
                    x.requires_grad_(True)
                    polish_ratio = (
                        signed_volumes_cuda(x, tet_t)
                        * rest_sign / rest_abs)
                    polish_minimum = float(torch.min(polish_ratio))
                    if polish_minimum >= 0.01:
                        x = x.detach()
                        break
                    polish_loss = torch.mean(
                        torch.relu(0.02 - polish_ratio) ** 2)
                    polish_gradient, = torch.autograd.grad(polish_loss, x)
                    polish_direction = -polish_gradient
                    if args.hard_attachments:
                        polish_direction[rigid_origin_t] = 0.0
                        polish_direction[rigid_insertion_t] = 0.0
                    polish_maximum = torch.max(
                        torch.linalg.vector_norm(
                            polish_direction, dim=1)).clamp_min(1e-20)
                    polish_direction = (
                        polish_direction * (1e-4 / polish_maximum))
                    polish_candidate = (
                        x.detach() + polish_direction)
                    candidate_ratio = (
                        signed_volumes_cuda(polish_candidate, tet_t)
                        * rest_sign / rest_abs)
                    if float(torch.min(candidate_ratio)) <= polish_minimum:
                        x = x.detach()
                        break
                    x = polish_candidate.detach()
            with torch.no_grad():
                substep_attachment_error = float(torch.max(torch.cat((
                    torch.linalg.vector_norm(
                        x[origin_t] - origin_target, dim=1),
                    torch.linalg.vector_norm(
                        x[insertion_t] - insertion_target, dim=1)))))
            print(
                f"  frame {frame} substep {substep}/{frame_substeps}: "
                f"attachment={substep_attachment_error:.6g}m "
                f"iterations={iteration + 1}",
                flush=True)
            if (
                substep_attachment_error > args.attachment_tolerance
                and not args.allow_attachment_error
            ):
                raise RuntimeError(
                    f"attachment constraint failed at frame {frame}, "
                    f"substep {substep}/{frame_substeps}: "
                    f"{substep_attachment_error:.6g} m > "
                    f"{args.attachment_tolerance:.6g} m")

        previous_pose = frame_pose.copy()
        with torch.no_grad():
            ratio = (
                signed_volumes_cuda(x, tet_t)
                * rest_sign / rest_abs)
            origin_error = torch.linalg.vector_norm(
                x[origin_t] - origin_target, dim=1)
            insertion_error = torch.linalg.vector_norm(
                x[insertion_t] - insertion_target, dim=1)
            constraints = sdf.constraints(
                x.cpu().numpy(), sample_i, sample_w,
                no_contact_exclusion, skeleton, args.collision_margin,
                max_step=None, rest_tolerance=args.collision_tolerance)
            contact_count = 0 if constraints is None else len(constraints[0])
            inverted = int(torch.sum(ratio <= 0.0))
            min_j = float(torch.min(ratio))
            max_attachment = float(torch.max(torch.cat(
                (origin_error, insertion_error))))
            print(
                f"frame {frame}: inverted={inverted}/{len(tets)} "
                f"minJ={min_j:.6g} attachment={max_attachment:.6g}m "
                f"contact={contact_count}", flush=True)
            frames_out.append(x.cpu().numpy().astype(np.float32))
            frame_ids_out.append(frame)
            metrics.append((inverted, min_j, max_attachment, contact_count))
            # A difficult later pose must not discard already validated frames.
            save_checkpoint()
            if min_j >= 0.01:
                stable_x = x.clone()
                stable_pose = frame_pose.copy()
                stable_rotation = previous_rotation.copy()
                stable_translation = previous_translation.copy()
                needs_rollback = False
            else:
                needs_rollback = True

    save_checkpoint(done=True)
    print(f"Saved {len(frames_out)} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
