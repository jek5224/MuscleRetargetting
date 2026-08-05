#!/usr/bin/env python3
"""Stable Neo-Hookean quasistatic bake for the bone-conforming upper-leg cage."""
import argparse
import json
import os
import pickle
import time

import numpy as np
from scipy.spatial import cKDTree

from tools.bake_emu import (
    build_gradient_operator, relax_positions_with_jacobian_barrier)
from tools.bake_headless import (
    MyBVH, _detect_bvh_tframe, init_soft_bodies, load_muscle_meshes,
    load_skeleton, load_skeleton_meshes, load_tet_meshes)
from tools.bake_upperleg_contour_cage import (
    boundary_faces, build_tracked_cage_bone_contacts,
    cage_edges, harmonic_bone_weights, project_tracked_bone_contacts)


def load_pickle(path):
    with open(path, "rb") as stream:
        return pickle.load(stream)


def select_controls(rest, muscles, skel, per_bone):
    samples = []
    for muscle in muscles.values():
        anchors = getattr(muscle, "soft_body_local_anchors", {}) or {}
        for _, (bone, local) in anchors.items():
            body = skel.getBodyNode(bone)
            if body is None:
                continue
            transform = body.getWorldTransform()
            world = (np.asarray(transform.rotation()) @ np.asarray(local)
                     + np.asarray(transform.translation()))
            samples.append((world, bone, np.asarray(local)))
    tree = cKDTree(rest)
    _, nearest = tree.query(np.asarray([entry[0] for entry in samples]))
    controls = {}
    for sample_i, cage_i in enumerate(nearest):
        distance = np.linalg.norm(rest[cage_i] - samples[sample_i][0])
        if cage_i not in controls or distance < controls[cage_i][0]:
            controls[int(cage_i)] = (
                distance, samples[sample_i][1], samples[sample_i][2])
    by_bone = {}
    for cage_i, value in controls.items():
        by_bone.setdefault(value[1], []).append(cage_i)
    selected = {}
    for bone, candidates in by_bone.items():
        candidates = np.asarray(candidates, dtype=np.int32)
        points = rest[candidates]
        chosen = [int(np.argmin(points[:, 1]))]
        nearest_distance = np.linalg.norm(points - points[chosen[0]], axis=1)
        while len(chosen) < min(per_bone, len(candidates)):
            next_i = int(np.argmax(nearest_distance))
            chosen.append(next_i)
            nearest_distance = np.minimum(
                nearest_distance,
                np.linalg.norm(points - points[next_i], axis=1))
        for local_i in chosen:
            cage_i = int(candidates[local_i])
            _, selected_bone, _ = controls[cage_i]
            transform = skel.getBodyNode(selected_bone).getWorldTransform()
            rotation = np.asarray(transform.rotation())
            translation = np.asarray(transform.translation())
            cage_local = rotation.T @ (rest[cage_i] - translation)
            selected[cage_i] = (0.0, selected_bone, cage_local)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cage", default="cage/L_UpLeg_bone_conforming_cage_v2.npz")
    parser.add_argument("--muscles", default=".muscles_L_UpLeg.json")
    parser.add_argument("--tet-dir", default="tet")
    parser.add_argument(
        "--bvh", default="data/motion/run_vert_sternum_arm_forearm.bvh")
    parser.add_argument("--frame", type=int, default=41)
    parser.add_argument("--continuation-steps", type=int, default=20)
    parser.add_argument("--fem-iterations", type=int, default=12)
    parser.add_argument("--controls-per-bone", type=int, default=12)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--lambda-volume", type=float, default=20.0)
    parser.add_argument("--project-contact", action="store_true")
    parser.add_argument("--predictor-weight", type=float, default=0.0)
    parser.add_argument("--attachment-weight", type=float, default=500.0)
    parser.add_argument("--hard-attachments", action="store_true")
    parser.add_argument("--jacobian-floor", type=float, default=0.005)
    parser.add_argument("--barrier-start", type=float, default=0.20)
    parser.add_argument(
        "--optimization-coordinate-scale", type=float, default=0.01)
    parser.add_argument(
        "--region-tag", default="L_UpLeg_bone_conforming_fem_review_f41")
    args = parser.parse_args()

    cage = load_pickle(args.cage)
    rest = np.asarray(cage["vertices"], dtype=np.float64)
    tets = np.asarray(cage["tetrahedra"], dtype=np.int32)
    skel, bvh_info, mesh_info = load_skeleton()
    skeleton_meshes = load_skeleton_meshes()
    muscles = load_muscle_meshes(args.muscles)
    load_tet_meshes(muscles, args.tet_dir)
    skel.setPositions(np.zeros(skel.getNumDofs()))
    init_soft_bodies(
        muscles, skeleton_meshes, skel, mesh_info, smooth_skinning=None)

    controls = select_controls(
        rest, muscles, skel, args.controls_per_bone)
    fixed_indices = np.asarray(sorted(controls), dtype=np.int32)
    fixed_mask = np.zeros(len(rest), dtype=bool)
    fixed_mask[fixed_indices] = True
    edges, _, _, _ = cage_edges(tets, len(rest))
    harmonic_bones, harmonic_weights = harmonic_bone_weights(
        rest, edges, fixed_indices, controls)
    rest_transforms = {}
    for bone in harmonic_bones:
        transform = skel.getBodyNode(bone).getWorldTransform()
        rest_transforms[bone] = (
            np.asarray(transform.rotation()),
            np.asarray(transform.translation()))
    contacts = build_tracked_cage_bone_contacts(
        rest, tets, fixed_mask, skeleton_meshes, skel,
        bind_distance=0.04, clearance=0.0015)

    print("Building FEM gradient operator...")
    gradient, dm_inverse, volumes = build_gradient_operator(rest, tets)
    precomputed = {
        "Gt": gradient.T.tocsc(),
        "Dm_inv": dm_inverse,
        "volumes": volumes,
        "tetrahedra": tets,
    }

    motion = MyBVH(
        args.bvh, bvh_info, skel,
        T_frame=_detect_bvh_tframe(args.bvh))
    target_dofs = np.asarray(motion.mocap_refs[args.frame])
    positions = rest.copy()
    previous_harmonic = rest.copy()
    started = time.time()
    last_info = {}
    for step in range(1, args.continuation_steps + 1):
        alpha = step / args.continuation_steps
        skel.setPositions(alpha * target_dofs)
        fixed_targets = positions.copy()
        for cage_i in fixed_indices:
            _, bone, local = controls[int(cage_i)]
            transform = skel.getBodyNode(bone).getWorldTransform()
            fixed_targets[cage_i] = (
                np.asarray(transform.rotation()) @ local
                + np.asarray(transform.translation()))

        # Continue from the previous equilibrium and transport it only by this
        # step's harmonic increment. Blending toward the absolute prediction
        # would erase FEM relaxation accumulated by earlier steps.
        predictor = []
        for bone in harmonic_bones:
            transform = skel.getBodyNode(bone).getWorldTransform()
            rotation = np.asarray(transform.rotation())
            translation = np.asarray(transform.translation())
            rest_rotation, rest_translation = rest_transforms[bone]
            local = (rest - rest_translation) @ rest_rotation
            predictor.append(local @ rotation.T + translation)
        predictor = np.stack(predictor, axis=1)
        harmonic = np.sum(
            harmonic_weights[:, :, None] * predictor, axis=1)
        predictor_weight = float(np.clip(args.predictor_weight, 0.0, 1.0))
        predictor_mask = (
            ~fixed_mask if args.hard_attachments
            else np.ones(len(rest), dtype=bool))
        positions[predictor_mask] += predictor_weight * (
            harmonic[predictor_mask] - previous_harmonic[predictor_mask])
        previous_harmonic = harmonic.copy()
        if args.hard_attachments:
            positions[fixed_mask] = fixed_targets[fixed_mask]
        projected = 0
        if args.project_contact:
            positions, projected = project_tracked_bone_contacts(
                positions, contacts, skel)

        # Soft FEM attachments avoid instant inversion of thin boundary tets.
        fem_fixed_mask = (
            fixed_mask if args.hard_attachments
            else np.zeros_like(fixed_mask))
        solved, last_info = relax_positions_with_jacobian_barrier(
            positions, precomputed, fem_fixed_mask,
            fixed_targets if args.hard_attachments else positions,
            mu=args.mu, lam=args.lambda_volume,
            max_iters=args.fem_iterations,
            jacobian_floor=args.jacobian_floor,
            barrier_start=args.barrier_start, barrier_scale=8.0,
            stretch_frobenius_limit=16.0,
            soft_attachment_indices=(
                None if args.hard_attachments else fixed_indices),
            soft_attachment_targets=(
                None if args.hard_attachments
                else fixed_targets[fixed_indices]),
            soft_attachment_weight=(
                0.0 if args.hard_attachments
                else args.attachment_weight),
            optimization_coordinate_scale=(
                args.optimization_coordinate_scale))
        if solved is None:
            raise RuntimeError(
                f"Continuation failed at {step}/{args.continuation_steps}: "
                f"{last_info}")
        positions = solved
        if args.project_contact:
            positions, projected = project_tracked_bone_contacts(
                positions, contacts, skel)
        print(f"step {step:02d}/{args.continuation_steps}: "
              f"minJ={last_info['min_j']:.4f}, contacts={projected}, "
              f"iters={last_info['iterations']}")

    bvh_name = os.path.splitext(os.path.basename(args.bvh))[0]
    output = os.path.join(
        ".bake_outputs/motion_cache", bvh_name, args.region_tag)
    os.makedirs(output, exist_ok=True)
    frames = np.asarray([args.frame], dtype=np.int32)
    for name in muscles:
        binding = cage["muscles"][name]
        ids = tets[np.asarray(binding["tet_index"], dtype=np.int32)]
        weight = np.asarray(binding["weights"], dtype=np.float64)
        embedded = np.einsum("ni,nij->nj", weight, positions[ids])
        np.savez_compressed(
            os.path.join(output, name + "_chunk_0000.npz"),
            frames=frames, positions=embedded[None].astype(np.float32))
    np.savez_compressed(
        os.path.join(output, "__tissue_cage_chunk_0000.npz"),
        frames=frames, positions=positions[None].astype(np.float32),
        rest_positions=rest.astype(np.float32),
        surface_faces=boundary_faces(tets))
    with open(os.path.join(output, ".done"), "w") as stream:
        stream.write("complete\n")
    print(f"Saved {output}; elapsed={time.time() - started:.2f}s; "
          f"final={last_info}")


if __name__ == "__main__":
    main()
