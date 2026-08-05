#!/usr/bin/env python3
"""Joint GPU refinement of one VI/VL pose with SDF and pair contact."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from core.bvhparser import MyBVH
from tools import bake_emu
from tools.bake_surface_fast import load_tet, surface_faces
from tools.bake_stiff_tet_arap import BoneSDF, prepare_full_cap_group
from tools.bake_stiff_tet_pbd import signed_volumes, unique_edges
from tools.bake_stiff_tet_pbd import build_surface_samples
import test_emu


def volume_torch(x, tets):
    v = x[tets]
    return torch.sum(
        (v[:, 1] - v[:, 0])
        * torch.linalg.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]),
        dim=1) / 6.0


def sdf_torch(x, field, rotation, translation):
    local = (x - translation) @ rotation
    coordinate = (local - field["origin"]) / field["pitch"]
    shape = torch.as_tensor(
        field["shape"], dtype=x.dtype, device=x.device)
    normalized = 2.0 * coordinate / (shape - 1.0) - 1.0
    # NumPy field axes are (i,j,k); grid_sample consumes (x=W,y=H,z=D).
    grid = normalized[:, [2, 1, 0]].reshape(1, -1, 1, 1, 3)
    return F.grid_sample(
        field["tensor"], grid, mode="bilinear",
        padding_mode="border", align_corners=True).reshape(-1)


def load_body(name, tet_path, cache_path, frame, skeleton, trees, device):
    data = load_tet(tet_path)
    group = prepare_full_cap_group(
        data, name, tet_path, skeleton, trees)
    rest = np.asarray(group["vertices"], dtype=np.float64)
    tets_np = np.asarray(group["tetrahedra"], dtype=np.int32)
    cache = np.load(cache_path)
    initial = cache["positions"][frame].astype(np.float64)
    edges_np = unique_edges(tets_np)
    surface = np.unique(surface_faces(tets_np))
    fixed = np.asarray(group["fixed_vertices"], dtype=np.int32)
    fixed_mask = np.zeros(len(rest), dtype=bool)
    fixed_mask[fixed] = True
    sample_ids_np, sample_weights_np = build_surface_samples(
        surface_faces(tets_np))
    sample_active = ~np.any(
        fixed_mask[sample_ids_np] & (sample_weights_np > 0.0), axis=1)
    return {
        "name": name, "cache": cache, "initial_np": initial, "group": group,
        "x": torch.tensor(initial, dtype=torch.float64, device=device,
                          requires_grad=True),
        "initial": torch.tensor(initial, dtype=torch.float64, device=device),
        "rest": torch.tensor(rest, dtype=torch.float64, device=device),
        "tets": torch.tensor(tets_np, dtype=torch.long, device=device),
        "tets_np": tets_np,
        "edges": torch.tensor(edges_np, dtype=torch.long, device=device),
        "surface": torch.tensor(surface, dtype=torch.long, device=device),
        "free_surface": torch.tensor(
            surface[~fixed_mask[surface]], dtype=torch.long, device=device),
        "sample_ids": torch.tensor(
            sample_ids_np[sample_active], dtype=torch.long, device=device),
        "sample_weights": torch.tensor(
            sample_weights_np[sample_active], dtype=torch.float64,
            device=device),
        "fixed": torch.tensor(fixed, dtype=torch.long, device=device),
        "fixed_target": torch.tensor(
            initial[fixed], dtype=torch.float64, device=device),
        "rest_volume": torch.tensor(
            signed_volumes(rest, tets_np), dtype=torch.float64,
            device=device),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bvh", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--bone-weight", type=float, default=300.0)
    parser.add_argument("--include-vm", action="store_true")
    parser.add_argument("--vi-only", action="store_true")
    parser.add_argument("--vm-only", action="store_true",
                        help="Optimize VM while retaining VI/VL as contact bodies")
    parser.add_argument("--volume-weight", type=float, default=4.0)
    parser.add_argument("--inversion-weight", type=float, default=80.0)
    parser.add_argument("--max-displacement", type=float, default=0.0,
                        help="Hard trust radius from the input pose (metres)")
    parser.add_argument("--guide-weight", type=float, default=0.05,
                        help="Material-guide tether weight")
    parser.add_argument("--bone-attraction-weight", type=float, default=0.0,
                        help="Penalty for moving farther from bone than rest")
    parser.add_argument("--bone-max-slack", type=float, default=0.005,
                        help="Allowed extra bone distance beyond rest (metres)")
    parser.add_argument("--manifold-weight", type=float, default=0.0,
                        help="Rest-pose per-vertex SDF manifold weight")
    parser.add_argument("--manifold-tolerance", type=float, default=0.002,
                        help="Half-width of the material SDF band (metres)")
    parser.add_argument("--cohesion-weight", type=float, default=20.0)
    parser.add_argument("--cohesion-radius", type=float, default=0.008)
    parser.add_argument(
        "--sdf", type=Path,
        default=Path(".bake_outputs/collision_sdf/L_Femur0_sdf.npz"))
    args = parser.parse_args()
    device = torch.device("cuda")

    skeleton, bvh_info, _ = bake_emu.load_skeleton()
    motion = MyBVH(
        str(args.bvh), bvh_info, skeleton,
        T_frame=bake_emu._detect_bvh_tframe(str(args.bvh)))
    trees = test_emu._load_bone_trees()
    bodies = [
        load_body(
            "L_Vastus_Intermedius",
            Path("tet/L_Vastus_Intermedius_tet.npz"),
            args.cache_dir / "L_Vastus_Intermedius_chunk_0000.npz",
            args.frame, skeleton, trees, device),
    ]
    if not args.vi_only:
        bodies.append(load_body(
            "L_Vastus_Lateralis",
            Path("tet/L_Vastus_Lateralis_tet.npz"),
            args.cache_dir / "L_Vastus_Lateralis_chunk_0000.npz",
            args.frame, skeleton, trees, device))
    if args.include_vm:
        bodies.append(load_body(
            "L_Vastus_Medialis",
            Path("tet/L_Vastus_Medialis_tet.npz"),
            args.cache_dir / "L_Vastus_Medialis_chunk_0000.npz",
            args.frame, skeleton, trees, device))
    # Bindings above are built in the skeleton rest pose. Pose only after all
    # groups exist, then derive the true per-frame skeletal targets.
    skeleton.setPositions(motion.mocap_refs[args.frame].copy())
    for body in bodies:
        group = body["group"]
        guide = bake_emu.compute_rigid_blend_positions(
            group["lbs_bindings"], skeleton, group["axis_coordinate"])
        fixed_np = body["fixed"].cpu().numpy()
        initial_np = body["initial_np"].copy()
        initial_np[fixed_np] = guide[fixed_np]
        body["initial_np"] = initial_np
        body["x"] = torch.tensor(
            initial_np, dtype=torch.float64, device=device,
            requires_grad=True)
        body["initial"] = torch.tensor(
            initial_np, dtype=torch.float64, device=device)
        body["fixed_target"] = torch.tensor(
            guide[fixed_np], dtype=torch.float64, device=device)
    if args.vm_only and not args.include_vm:
        parser.error("--vm-only requires --include-vm")
    active_bodies = bodies[-1:] if args.vm_only else bodies
    if args.vm_only:
        for body in bodies[:-1]:
            body["x"].requires_grad_(False)
    neighbor_pairs = []
    for first in range(len(bodies)):
        for second in range(first + 1, len(bodies)):
            first_surface = bodies[first]["surface"].cpu().numpy()
            second_surface = bodies[second]["surface"].cpu().numpy()
            first_rest = bodies[first]["rest"].detach().cpu().numpy()
            second_rest = bodies[second]["rest"].detach().cpu().numpy()
            pairs = set()
            distance, nearest = cKDTree(
                second_rest[second_surface]).query(first_rest[first_surface])
            for local, (gap, target) in enumerate(zip(distance, nearest)):
                if gap <= args.cohesion_radius:
                    pairs.add((
                        int(first_surface[local]),
                        int(second_surface[int(target)])))
            distance, nearest = cKDTree(
                first_rest[first_surface]).query(second_rest[second_surface])
            for local, (gap, target) in enumerate(zip(distance, nearest)):
                if gap <= args.cohesion_radius:
                    pairs.add((
                        int(first_surface[int(target)]),
                        int(second_surface[local])))
            pair_array = np.asarray(sorted(pairs), dtype=np.int32)
            rest_gap = np.linalg.norm(
                first_rest[pair_array[:, 0]]
                - second_rest[pair_array[:, 1]], axis=1)
            neighbor_pairs.append((
                first, second,
                torch.tensor(
                    pair_array[:, 0], dtype=torch.long, device=device),
                torch.tensor(
                    pair_array[:, 1], dtype=torch.long, device=device),
                torch.tensor(rest_gap, dtype=torch.float64, device=device)))
            print(
                f"cohesion {bodies[first]['name']}|"
                f"{bodies[second]['name']}: {len(pair_array)} rest neighbors")
    raw_sdf = BoneSDF(args.sdf)
    field = {
        "tensor": torch.tensor(
            raw_sdf.sdf, dtype=torch.float64, device=device
        ).reshape(1, 1, *raw_sdf.sdf.shape),
        "origin": torch.tensor(
            raw_sdf.origin, dtype=torch.float64, device=device),
        "pitch": raw_sdf.pitch, "shape": raw_sdf.sdf.shape,
    }
    rest_skeleton, _, _ = bake_emu.load_skeleton()
    rest_transform = rest_skeleton.getBodyNode(
        raw_sdf.body).getWorldTransform()
    rest_rotation = torch.tensor(
        rest_transform.rotation(), dtype=torch.float64, device=device)
    rest_translation = torch.tensor(
        rest_transform.translation(), dtype=torch.float64, device=device)
    with torch.no_grad():
        for body in bodies:
            rest_points = torch.sum(
                body["rest"][body["sample_ids"]]
                * body["sample_weights"][:, :, None], dim=1)
            body["rest_bone_distance"] = sdf_torch(
                rest_points, field, rest_rotation, rest_translation)
            rest_local = (
                body["rest"] - rest_translation) @ rest_rotation
            coordinate = (rest_local - field["origin"]) / field["pitch"]
            shape = torch.as_tensor(
                field["shape"], dtype=torch.float64, device=device)
            valid = torch.all(
                (coordinate >= 1.0) & (coordinate <= shape - 2.0), dim=1)
            free_mask = torch.ones(
                len(body["rest"]), dtype=torch.bool, device=device)
            free_mask[body["fixed"]] = False
            body["manifold_ids"] = torch.where(valid & free_mask)[0]
            body["rest_vertex_sdf"] = sdf_torch(
                body["rest"][body["manifold_ids"]],
                field, rest_rotation, rest_translation)
    transform = skeleton.getBodyNode(raw_sdf.body).getWorldTransform()
    rotation = torch.tensor(
        transform.rotation(), dtype=torch.float64, device=device)
    translation = torch.tensor(
        transform.translation(), dtype=torch.float64, device=device)
    optimizer = torch.optim.Adam(
        [body["x"] for body in active_bodies], lr=args.lr)

    for iteration in range(args.iterations):
        optimizer.zero_grad()
        loss = torch.zeros((), dtype=torch.float64, device=device)
        for body in active_bodies:
            edge = body["edges"]
            current_length = torch.linalg.vector_norm(
                body["x"][edge[:, 0]] - body["x"][edge[:, 1]], dim=1)
            rest_length = torch.linalg.vector_norm(
                body["rest"][edge[:, 0]] - body["rest"][edge[:, 1]],
                dim=1).clamp_min(1e-8)
            loss = loss + torch.mean((current_length / rest_length - 1) ** 2)
            volume = volume_torch(body["x"], body["tets"])
            ratio = (
                volume * torch.sign(body["rest_volume"])
                / torch.abs(body["rest_volume"]).clamp_min(1e-12))
            loss = loss + args.volume_weight * torch.mean((ratio - 1.0) ** 2)
            loss = loss + args.inversion_weight * torch.mean(
                F.relu(0.35 - ratio) ** 2)
            sample_points = torch.sum(
                body["x"][body["sample_ids"]]
                * body["sample_weights"][:, :, None], dim=1)
            distance = sdf_torch(
                sample_points, field, rotation, translation)
            loss = loss + args.bone_weight * torch.mean(
                F.relu(0.0015 - distance) ** 2 / 0.0015 ** 2)
            if args.bone_attraction_weight > 0.0:
                maximum_distance = (
                    body["rest_bone_distance"] + args.bone_max_slack)
                loss = loss + args.bone_attraction_weight * torch.mean(
                    F.relu(distance - maximum_distance) ** 2
                    / max(args.bone_max_slack, 1e-6) ** 2)
            if args.manifold_weight > 0.0:
                ids = body["manifold_ids"]
                vertex_distance = sdf_torch(
                    body["x"][ids], field, rotation, translation)
                error = torch.abs(
                    vertex_distance - body["rest_vertex_sdf"])
                loss = loss + args.manifold_weight * torch.mean(
                    F.relu(error - args.manifold_tolerance) ** 2
                    / max(args.manifold_tolerance, 1e-6) ** 2)
            loss = loss + args.guide_weight * torch.mean(torch.sum(
                (body["x"] - body["initial"]) ** 2, dim=1) / 0.01 ** 2)

        for first in range(len(bodies)):
            for second in range(first + 1, len(bodies)):
                a = bodies[first]["x"][bodies[first]["surface"]]
                b = bodies[second]["x"][bodies[second]["surface"]]
                pair_distance = torch.cdist(a, b)
                nearest_a = torch.min(pair_distance, dim=1).values
                nearest_b = torch.min(pair_distance, dim=0).values
                pair_loss = 0.5 * (
                    torch.mean(F.relu(0.0008 - nearest_a) ** 2)
                    + torch.mean(F.relu(0.0008 - nearest_b) ** 2))
                loss = loss + 20.0 * pair_loss / 0.0008 ** 2
        for first, second, ids_a, ids_b, rest_gap in neighbor_pairs:
            gap = torch.linalg.vector_norm(
                bodies[first]["x"][ids_a] - bodies[second]["x"][ids_b],
                dim=1)
            allowed = rest_gap + 0.0015
            cohesion = torch.mean(F.relu(gap - allowed) ** 2)
            loss = loss + args.cohesion_weight * cohesion / 0.003 ** 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [body["x"] for body in active_bodies], max_norm=200.0)
        optimizer.step()
        with torch.no_grad():
            for body in active_bodies:
                if args.max_displacement > 0.0:
                    delta = body["x"] - body["initial"]
                    length = torch.linalg.vector_norm(
                        delta, dim=1).clamp_min(1e-12)
                    scale = torch.clamp(
                        args.max_displacement / length, max=1.0)
                    body["x"].copy_(
                        body["initial"] + delta * scale[:, None])
                body["x"][body["fixed"]] = body["fixed_target"]
        if iteration % 100 == 0 or iteration + 1 == args.iterations:
            print(f"iteration={iteration} loss={float(loss):.8f}")

    args.output.mkdir(parents=True, exist_ok=True)
    for body in bodies:
        result = body["cache"]["positions"].astype(np.float32).copy()
        solved = body["x"].detach().cpu().numpy()
        result[args.frame] = solved.astype(np.float32)
        oriented = signed_volumes(
            solved, body["tets_np"]) * np.sign(
                body["rest_volume"].detach().cpu().numpy())
        attachment_error = float(np.max(np.linalg.norm(
            solved[body["fixed"].cpu().numpy()]
            - body["fixed_target"].cpu().numpy(), axis=1)))
        print(
            f"{body['name']}: inverted={np.sum(oriented <= 0)}/"
            f"{len(oriented)} attachment_error={attachment_error:.3e}")
        np.savez_compressed(
            args.output / f"{body['name']}_chunk_0000.npz",
            frames=body["cache"]["frames"], positions=result)
    (args.output / ".done").touch()


if __name__ == "__main__":
    main()
