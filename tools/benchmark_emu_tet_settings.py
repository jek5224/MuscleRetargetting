#!/usr/bin/env python3
"""Benchmark isotropic tetrahedralization and EMU pose stability."""

import argparse
import time
from pathlib import Path

import numpy as np
import tetgen
from scipy.spatial import cKDTree

import test_emu
from tools import bake_emu
from viewer.zygote_mesh_ui import (
    _isotropic_voxel_surface_for_tet,
    _tet_volume_quality_stats,
)


POSES = {
    "rest": {},
    "femur_x_050": {"L_Femur0_x": 0.50},
    "femur_y_040": {"L_Femur0_y": 0.40},
    "femur_z_040": {"L_Femur0_z": 0.40},
    "tibia_x_070": {"L_Tibia_Fibula0_x": 0.70},
    "combined": {
        "L_Femur0_x": 0.40, "L_Femur0_y": 0.20,
        "L_Femur0_z": 0.20, "L_Tibia_Fibula0_x": 0.50,
    },
    "mixed_sign": {
        "L_Femur0_x": -0.30, "L_Femur0_y": 0.25,
        "L_Femur0_z": -0.25, "L_Tibia_Fibula0_x": 0.60,
    },
    "large_combo": {
        "L_Femur0_x": 0.60, "L_Femur0_y": -0.30,
        "L_Femur0_z": 0.30, "L_Tibia_Fibula0_x": 0.70,
    },
}


def make_candidate(data, target):
    sv, sf, _ = _isotropic_voxel_surface_for_tet(
        f"target_{target}", data["vertices"], data["render_faces"], target)
    started = time.time()
    tg = tetgen.TetGen(sv, sf)
    tg.tetrahedralize(
        order=1, quality=True, minratio=1.2, mindihedral=10,
        nobisect=False)
    vertices = np.asarray(tg.node, dtype=np.float64)
    tetrahedra = np.asarray(tg.elem, dtype=np.int32)
    _, quality = _tet_volume_quality_stats(
        f"target_{target}", vertices, tetrahedra)
    return vertices, tetrahedra, quality, time.time() - started


def transfer_material(data, vertices, tetrahedra):
    old_labels = np.asarray(data.get("tet_region_labels", []), dtype=str)
    if len(old_labels) != len(data["tetrahedra"]):
        return np.full(len(tetrahedra), "belly", dtype=str)
    old_centers = np.asarray(data["vertices"])[np.asarray(data["tetrahedra"])].mean(axis=1)
    new_centers = vertices[tetrahedra].mean(axis=1)
    nearest = cKDTree(old_centers).query(new_centers, k=1)[1]
    return old_labels[nearest]


def run_pose(group, precomp, skel, pose, mu, lam, activation, alpha, steps, iterations):
    dofs = {skel.getDof(i).getName(): i for i in range(skel.getNumDofs())}
    q_dart = np.zeros(skel.getNumDofs())
    for name, value in pose.items():
        q_dart[dofs[name]] = value
    skel.setPositions(q_dart)
    fixed = np.zeros(len(group["vertices"]), dtype=bool)
    fixed[group["fixed_vertices"]] = True
    q = group["vertices"].copy()
    warm = precomp["G"] @ q.ravel()
    info = None
    reached = 0
    for step in range(1, steps + 1):
        fraction = step / steps
        targets = bake_emu.compute_rigid_blend_positions_at_fraction(
            group["lbs_bindings"], skel, group["axis_coordinate"], fraction)
        q, info = bake_emu.emu_solve(
            q, precomp, fixed, targets, mu, lam, alpha,
            max_iters=iterations, use_gpu=False, warm_F=warm,
            activation=activation * fraction)
        warm = precomp["G"] @ q.ravel()
        F = bake_emu._deformation_gradients_from_q(
            q, group["tetrahedra"], precomp["Dm_inv"])
        J = np.linalg.det(F)
        stretch = np.linalg.svd(F, compute_uv=False)[:, 0]
        reached = step
        if (np.min(J) <= 0.02 or not np.all(np.isfinite(stretch)) or
                np.max(stretch) > 8.0):
            return False, reached, J, stretch, info, q
    return True, reached, J, stretch, info, q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tet", type=Path,
                        default=Path("tet/L_Rectus_Femoris_tet.npz"))
    parser.add_argument("--targets", type=int, nargs="+",
                        default=[12000, 20000, 33000])
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--rings", type=int, default=1)
    parser.add_argument("--muscle-youngs", type=float, default=6e6)
    parser.add_argument("--tendon-youngs", type=float, default=4.5e8)
    parser.add_argument("--activation", type=float, default=0.0)
    parser.add_argument("--max-active-stress", type=float, default=6e6)
    parser.add_argument("--alpha", type=float, default=1e3)
    parser.add_argument("--pose", action="append", default=[])
    args = parser.parse_args()

    data = test_emu._load_saved_tet(args.tet)
    skel, _, _ = bake_emu.load_skeleton()
    trees = test_emu._load_bone_trees()
    selected = [(n, p) for n, p in POSES.items()
                if not args.pose or n in args.pose]
    print("BENCH target tets verts sj_p01 mr_p01 critical pose ok reached "
          "Jmin stretch_p99 stretch_max fiber_stretch_mean energy seconds", flush=True)

    for target in args.targets:
        skel.setPositions(np.zeros(skel.getNumDofs()))
        vertices, tetrahedra, quality, tet_seconds = make_candidate(data, target)
        candidate = dict(data)
        candidate.update(vertices=vertices, tetrahedra=tetrahedra)
        group = test_emu.prepare_group_data(
            candidate, "L_Rectus_Femoris", skel, trees,
            source_path=f"target {target}", attachment_rings=args.rings)
        labels = transfer_material(data, vertices, tetrahedra)
        tendon = np.char.find(labels.astype(str), "tendon") >= 0
        youngs = np.where(tendon, args.tendon_youngs, args.muscle_youngs)
        mu, lam = bake_emu.lame_parameters(youngs, 0.49)
        activation = np.where(
            tendon, 0.0,
            np.clip(args.activation, 0.0, 1.0) * args.max_active_stress)
        precomp = bake_emu.precompute_emu(
            vertices, tetrahedra, group["fixed_vertices"],
            group["axis_coordinate"], k_modes=args.modes)
        for pose_name, pose in selected:
            started = time.time()
            ok, reached, J, stretch, info, q = run_pose(
                group, precomp, skel, pose, mu, lam, activation, args.alpha,
                args.steps, args.iterations)
            elapsed = time.time() - started
            final_F = bake_emu._deformation_gradients_from_q(
                q,
                group["tetrahedra"], precomp["Dm_inv"])
            fiber_stretch = np.linalg.norm(np.einsum(
                'mij,mj->mi', final_F, precomp['fiber_dirs']), axis=1)
            print(
                f"BENCH {target} {len(tetrahedra)} {len(vertices)} "
                f"{quality['scaled_jacobian_p01']:.6g} "
                f"{quality['mean_ratio_p01']:.6g} "
                f"{quality['critical_slivers']} {pose_name} {int(ok)} "
                f"{reached}/{args.steps} {float(np.min(J)):.6g} "
                f"{float(np.quantile(stretch, .99)):.6g} "
                f"{float(np.max(stretch)):.6g} "
                f"{float(np.mean(fiber_stretch[~tendon])):.6g} "
                f"{float(info['energy']):.6g} {elapsed:.2f}", flush=True)


if __name__ == "__main__":
    main()
