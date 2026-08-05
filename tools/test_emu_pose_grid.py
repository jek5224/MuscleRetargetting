#!/usr/bin/env python3
"""Sweep DART leg poses and EMU continuation settings headlessly."""

import argparse
import time
from pathlib import Path

import numpy as np

import test_emu
from tools import bake_emu


POSES = {
    "femur_x_010": {"L_Femur0_x": 0.10},
    "femur_x_030": {"L_Femur0_x": 0.30},
    "femur_x_050": {"L_Femur0_x": 0.50},
    "femur_y_030": {"L_Femur0_y": 0.30},
    "femur_z_030": {"L_Femur0_z": 0.30},
    "tibia_x_050": {"L_Tibia_Fibula0_x": 0.50},
    "femur_xy": {"L_Femur0_x": 0.35, "L_Femur0_y": 0.25},
    "femur_xyz_tibia": {
        "L_Femur0_x": 0.40,
        "L_Femur0_y": 0.20,
        "L_Femur0_z": 0.20,
        "L_Tibia_Fibula0_x": 0.50,
    },
    "femur_mixed_sign": {
        "L_Femur0_x": -0.30,
        "L_Femur0_y": 0.20,
        "L_Femur0_z": -0.20,
        "L_Tibia_Fibula0_x": 0.50,
    },
    "femur_large_combo": {
        "L_Femur0_x": 0.60,
        "L_Femur0_y": -0.30,
        "L_Femur0_z": 0.30,
        "L_Tibia_Fibula0_x": 0.70,
    },
}


SETTINGS = (
    ("ring1_10x2", 1, 10, 2),
    ("ring1_20x2", 1, 20, 2),
    ("ring1_20x4", 1, 20, 4),
    ("ring0_10x2", 0, 10, 2),
    ("ring0_20x2", 0, 20, 2),
    ("ring0_20x4", 0, 20, 4),
)


def quality(q, group, precomp):
    F = bake_emu._deformation_gradients_from_q(
        q, group["tetrahedra"], precomp["Dm_inv"])
    J = np.linalg.det(F)
    stretch = np.linalg.svd(F, compute_uv=False)[:, 0]
    return int(np.sum(J <= 0.0)), float(np.min(J)), float(np.max(stretch))


def solve_pose(group, precomp, skel, steps, iterations, mu, lam, alpha):
    targets = bake_emu.compute_rigid_blend_positions(
        group["lbs_bindings"], skel, group["axis_coordinate"])
    preview_F = bake_emu._deformation_gradients_from_q(
        targets, group["tetrahedra"], precomp["Dm_inv"])
    preview_J = np.linalg.det(preview_F)
    preview_stretch = np.linalg.svd(preview_F, compute_uv=False)[:, 0]
    severity = max(
        float(np.max(preview_stretch)) / 2.0,
        float(np.sum(preview_J <= 0.0)) /
        max(1.0, 0.01 * len(group["tetrahedra"])))
    steps = max(int(steps), int(np.clip(np.ceil(4.0 + 2.0 * severity), 4, 24)))
    fixed_mask = np.zeros(len(group["vertices"]), dtype=bool)
    fixed_mask[group["fixed_vertices"]] = True
    q = group["vertices"].copy()
    warm_F = precomp["G"] @ q.ravel()
    last_info = None
    for step in range(1, steps + 1):
        previous_q = q.copy()
        previous_fraction = (step - 1) / steps
        fraction = step / steps
        step_targets = bake_emu.compute_rigid_blend_positions_at_fraction(
            group["lbs_bindings"], skel, group["axis_coordinate"], fraction)
        q, last_info = bake_emu.emu_solve(
            q, precomp, fixed_mask, step_targets, mu, lam, alpha,
            max_iters=iterations, verbose=False, use_gpu=False, warm_F=warm_F)
        warm_F = precomp["G"] @ q.ravel()
        inverted, min_j, max_stretch = quality(q, group, precomp)
        if inverted or min_j <= 0.02 or max_stretch > 4.0:
            fallback_q = None
            if min_j > 0.02:
                direct_q, _direct_info = (
                    bake_emu.relax_positions_with_jacobian_barrier(
                        q, precomp, fixed_mask, step_targets, mu, lam,
                        max_iters=30))
                if direct_q is not None:
                    _inv, _min_j, _stretch = quality(direct_q, group, precomp)
                    if (not _inv and _min_j > 0.02 and
                            np.isfinite(_stretch) and _stretch <= 8.0):
                        fallback_q = direct_q
            if fallback_q is None:
                trial_q = previous_q.copy()
                previous_sub_targets = (
                    bake_emu.compute_rigid_blend_positions_at_fraction(
                        group["lbs_bindings"], skel,
                        group["axis_coordinate"], previous_fraction))
                current_fraction = previous_fraction
                interval = fraction - previous_fraction
                increment = interval / 4.0
                minimum_increment = interval / 1024.0
                fallback_attempts = 0
                while (current_fraction < fraction - 1e-12 and
                       fallback_attempts < 32):
                    fallback_attempts += 1
                    sub_fraction = min(fraction, current_fraction + increment)
                    sub_targets = bake_emu.compute_rigid_blend_positions_at_fraction(
                        group["lbs_bindings"], skel,
                        group["axis_coordinate"], sub_fraction)
                    predicted = trial_q + sub_targets - previous_sub_targets
                    relaxed, _fallback_info = (
                        bake_emu.relax_positions_with_jacobian_barrier(
                            predicted, precomp, fixed_mask, sub_targets, mu, lam,
                            max_iters=40))
                    if relaxed is None:
                        increment *= 0.5
                        if increment < minimum_increment:
                            break
                        continue
                    _inv, _min_j, _stretch = quality(
                        relaxed, group, precomp)
                    if (_inv or _min_j <= 0.02 or
                            not np.isfinite(_stretch) or _stretch > 8.0):
                        increment *= 0.5
                        if increment < minimum_increment:
                            break
                        continue
                    trial_q = relaxed
                    previous_sub_targets = sub_targets
                    current_fraction = sub_fraction
                    increment = min(increment * 1.5,
                                    fraction - current_fraction)
                if current_fraction >= fraction - 1e-12:
                    fallback_q = trial_q
            if fallback_q is None:
                print(f"FALLBACK_FAIL step={step} info={_fallback_info}")
                return False, step, inverted, min_j, max_stretch, last_info
            q = fallback_q
            warm_F = precomp["G"] @ q.ravel()
    return True, steps, *quality(q, group, precomp), last_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tet", type=Path, nargs="?",
                        default=Path("tet/L_Rectus_Femoris_tet.npz"))
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1e3)
    parser.add_argument("--setting", action="append", default=[],
                        help="Run only the named setting (repeatable)")
    parser.add_argument("--pose", action="append", default=[],
                        help="Run only the named pose (repeatable)")
    args = parser.parse_args()

    skel, _bvh, _mesh = bake_emu.load_skeleton()
    dofs = {skel.getDof(i).getName(): i for i in range(skel.getNumDofs())}
    trees = test_emu._load_bone_trees()
    data = test_emu._load_saved_tet(args.tet)
    mu, lam = bake_emu.lame_parameters(6e6, 0.49)

    prepared = {}
    precomputed = {}
    for ring in (0, 1):
        skel.setPositions(np.zeros(skel.getNumDofs()))
        group = test_emu.prepare_group_data(
            data, "L_Rectus_Femoris", skel, trees,
            source_path=args.tet, attachment_rings=ring)
        prepared[ring] = group
        precomputed[ring] = bake_emu.precompute_emu(
            group["vertices"], group["tetrahedra"], group["fixed_vertices"],
            group["axis_coordinate"], k_modes=args.modes)

    print("RESULT setting pose ok reached inverted minJ maxStretch energy seconds")
    selected_settings = [s for s in SETTINGS
                         if not args.setting or s[0] in args.setting]
    selected_poses = [(name, value) for name, value in POSES.items()
                      if not args.pose or name in args.pose]
    for setting, ring, steps, iterations in selected_settings:
        group = prepared[ring]
        precomp = precomputed[ring]
        for pose_name, values in selected_poses:
            q = np.zeros(skel.getNumDofs())
            for name, value in values.items():
                q[dofs[name]] = value
            skel.setPositions(q)
            started = time.time()
            ok, reached, inverted, min_j, stretch, info = solve_pose(
                group, precomp, skel, steps, iterations, mu, lam, args.alpha)
            elapsed = time.time() - started
            energy = float(info["energy"]) if info is not None else float("nan")
            print(f"RESULT {setting} {pose_name} {int(ok)} {reached}/{steps} "
                  f"{inverted} {min_j:.6g} {stretch:.6g} {energy:.6g} {elapsed:.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
