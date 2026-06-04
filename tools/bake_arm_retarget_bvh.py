"""Subtract frame 0's local rotation from ARM-only BVH joints (Shoulder,
Arm; per-side) so that BVH-driven N-pose skel has its arms hanging at rest
at frame 0. Leg/spine joints stay untouched (avoids the asymmetric leg
artifact from global T_frame=0).

For each target joint:
  new_R_local(f) = R_local_frame_0.inv @ R_local_frame_f

Writes a new BVH file with the arm joints' channels rewritten.
"""
import argparse
import os
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

# Joints to retarget (suffix match: handles Character1_LeftShoulder too).
TARGET_SUFFIXES = ["LeftShoulder", "RightShoulder", "LeftArm", "RightArm"]


def parse_bvh(bvh_path):
    with open(bvh_path) as f:
        lines = f.read().splitlines()
    # Build pre-order channel list as parser does.
    joints = []  # list of dict {name, parent, channels, indent}
    stack = []
    pending_name = None
    pending_type = None
    motion_idx = None
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped == "MOTION":
            motion_idx = i
            break
        m = re.match(r'^(ROOT|JOINT)\s+(\S+)\s*$', stripped)
        if m:
            pending_type = m.group(1)
            pending_name = m.group(2)
            continue
        if stripped == "End Site":
            pending_type = "End"
            pending_name = None
            continue
        if stripped == "{":
            j = {
                "type": pending_type, "name": pending_name,
                "parent": stack[-1] if stack else None,
                "channels": [],
            }
            joints.append(j)
            stack.append(j)
            pending_name = None
            continue
        if stripped == "}":
            if stack:
                stack.pop()
            continue
        m_ch = re.match(r'^CHANNELS\s+(\d+)\s+(.+)$', stripped)
        if m_ch and stack:
            stack[-1]["channels"] = m_ch.group(2).split()
            continue
    return lines, joints, motion_idx


def build_channel_layout(joints):
    """List of (joint_name, num_channels) in pre-order."""
    return [(j["name"], len(j["channels"])) for j in joints if j["channels"]]


def euler_zxy_from_matrix(M):
    return R.from_matrix(M).as_euler("ZXY", degrees=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="bvh_in", required=True)
    ap.add_argument("--out", dest="bvh_out", required=True)
    ap.add_argument("--no-subtract", action="store_true",
                    help="Skip frame-0 subtraction (preserve BVH initial arm pose).")
    ap.add_argument("--rest-align", action="store_true",
                    help="Multiply each frame by R_humerus_world_rest^-1 to align BVH arm motion to skel humerus joint-local frame.")
    ap.add_argument("--symmetric-rest-align", action="store_true",
                    help="When using --rest-align, force R-side parent rest = X-mirror of L-side. Fixes asymmetric skel L_Scapula vs R_Scapula.")
    ap.add_argument("--shoulder-scale", type=float, default=1.0,
                    help="Scale shoulder/clavicle rotation magnitude (1.0=full, 0.2=damped). Anatomical clavicle range ~10-15°; LaFAN sources may amplify.")
    ap.add_argument("--scapulohumeral", type=float, default=0.0,
                    help="Inman scapulohumeral rhythm. Value = clavicle share "
                         "(0.27 ≈ anatomical). Clavicle gets rotvec*share, "
                         "Humerus gets compound (rotvec*(1-share)) + LeftArm. "
                         "Preserves LeftHand world position. 0 = disabled.")
    ap.add_argument("--skel-xml", default="data/zygote_skel.xml")
    args = ap.parse_args()

    lines, joints, motion_idx = parse_bvh(args.bvh_in)
    if motion_idx is None:
        sys.exit("MOTION section not found")
    layout = build_channel_layout(joints)
    # Map channel-order joint name → (start_col, n_channels)
    col_map = {}
    col = 0
    for name, n in layout:
        col_map[name] = (col, n)
        col += n
    total_channels = col

    # Identify target joint names (suffix-match) and their channel order.
    target_names = []
    for name, _ in layout:
        for suf in TARGET_SUFFIXES:
            if name == suf or name.endswith("_" + suf):
                target_names.append(name)
                break
    print(f"Retarget targets: {target_names}")

    # Find each target joint's Euler order from BVH CHANNELS.
    joint_chs = {j["name"]: j["channels"] for j in joints}
    target_orders = {}
    for tn in target_names:
        chs = joint_chs[tn]
        order = "".join(c[0] for c in chs if c.lower().endswith("rotation"))
        target_orders[tn] = order
    print(f"Euler orders: {target_orders}")

    # Read motion rows.
    header = []
    frame_data = []
    for ln in lines[motion_idx:]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("MOTION") or s.startswith("Frames") or s.startswith("Frame Time"):
            header.append(ln)
            continue
        frame_data.append([float(x) for x in s.split()])
    print(f"Frames: {len(frame_data)}, channels per row: {len(frame_data[0])}")
    assert len(frame_data[0]) == total_channels, \
        f"channel mismatch: row={len(frame_data[0])} layout={total_channels}"

    # Frame 0 local rotation for each target → R_local_0
    R_local_0 = {}
    for tn in target_names:
        c, n = col_map[tn]
        # n should be 3 (rotations only) for JOINT entries
        euler_0 = frame_data[0][c:c + n]
        R_local_0[tn] = R.from_euler(target_orders[tn].upper(), euler_0, degrees=True)

    # Optional: build skel and compute per-joint rest rotations for alignment.
    R_align_inv = {}
    if args.rest_align:
        sys.path.insert(0, ".")
        from core.dartHelper import saveSkeletonInfo, buildFromInfo
        si, rn, *_ = saveSkeletonInfo(args.skel_xml)
        _skel = buildFromInfo(si, rn)
        bvh_to_skel = {"LeftArm": "L_Humerus0", "RightArm": "R_Humerus0",
                       "LeftShoulder": "L_Clavicle0", "RightShoulder": "R_Clavicle0"}
        for tn in target_names:
            suf = next((s for s in TARGET_SUFFIXES if tn == s or tn.endswith("_" + s)), None)
            if suf is None: continue
            skel_body_name = bvh_to_skel.get(suf)
            if skel_body_name is None: continue
            bn = _skel.getBodyNode(skel_body_name)
            if bn is None:
                # Try parent of joint: get the joint's parent body rest rotation
                continue
            Rw_rest = np.asarray(bn.getTransform().rotation())
            # Use parent body rest = R_world_rest of the body this joint outputs.
            # For rotation in joint-local frame: R_local = R_p^-1 * R_world_at_joint
            # We need R_humerus's PARENT body world rest, which is Scapula's body world rest.
            parent = bn.getParentBodyNode()
            if parent is not None:
                R_parent_rest = np.asarray(parent.getTransform().rotation())
            else:
                R_parent_rest = np.eye(3)
            # Symmetric mode for asymmetric skel R_Scapula: derive
            # R_align_inv_R = R_R_parent.T @ X-mirror(L_parent) so R-side
            # motion replicates L-side motion mirror via skel.
            if args.symmetric_rest_align and suf in ("RightArm", "RightShoulder"):
                l_skel_body = {"RightArm": "L_Humerus0",
                               "RightShoulder": "L_Clavicle0"}[suf]
                l_bn = _skel.getBodyNode(l_skel_body)
                l_parent = l_bn.getParentBodyNode() if l_bn else None
                if l_parent is not None:
                    L_parent_rest = np.asarray(l_parent.getTransform().rotation())
                    M = np.diag([-1.0, 1.0, 1.0])
                    mirror_L = M @ L_parent_rest @ M
                    R_align_inv[tn] = R.from_matrix(R_parent_rest.T @ mirror_L)
                    print(f"  [symmetric] {tn}: R_align_inv = R_p.T @ mirror(L_parent_rest)")
                    print(f"  align {tn}: parent_rest=\n{R_parent_rest.round(3)}")
                    continue
            R_align_inv[tn] = R.from_matrix(R_parent_rest.T)
            print(f"  align {tn}: parent_rest=\n{R_parent_rest.round(3)}")

    if not args.no_subtract:
        is_shoulder = lambda tn: any(s in tn for s in ("LeftShoulder", "RightShoulder"))
        is_arm = lambda tn: any(s in tn for s in ("LeftArm", "RightArm")) and not is_shoulder(tn)

        # Per side pairing for scapulohumeral rhythm.
        def side_of(name):
            return "L" if "Left" in name else ("R" if "Right" in name else None)
        shoulder_by_side = {}
        arm_by_side = {}
        for tn in target_names:
            s = side_of(tn)
            if s is None: continue
            if is_shoulder(tn): shoulder_by_side[s] = tn
            elif is_arm(tn): arm_by_side[s] = tn

        for row in frame_data:
            R_subtracted = {}
            for tn in target_names:
                c, n = col_map[tn]
                order = target_orders[tn].upper()
                euler_f = row[c:c + n]
                R_f = R.from_euler(order, euler_f, degrees=True)
                R_subtracted[tn] = R_local_0[tn].inv() * R_f

            # Scapulohumeral distribution: split shoulder between clavicle and
            # humerus. Share = args.scapulohumeral fraction goes to clavicle,
            # remainder composes onto humerus. Preserves LeftHand world pos.
            R_dist = dict(R_subtracted)
            if args.scapulohumeral > 0:
                share = float(args.scapulohumeral)
                for side, sh_tn in shoulder_by_side.items():
                    arm_tn = arm_by_side.get(side)
                    if arm_tn is None: continue
                    R_sh = R_subtracted[sh_tn]
                    R_arm = R_subtracted[arm_tn]
                    rv_sh = R_sh.as_rotvec()
                    R_clav = R.from_rotvec(rv_sh * share)
                    R_residual = R.from_rotvec(rv_sh * (1.0 - share))
                    R_dist[sh_tn] = R_clav
                    R_dist[arm_tn] = R_residual * R_arm

            for tn in target_names:
                c, n = col_map[tn]
                order = target_orders[tn].upper()
                R_new = R_dist[tn]
                if tn in R_align_inv:
                    R_new = R_align_inv[tn] * R_new
                if is_shoulder(tn) and args.shoulder_scale != 1.0:
                    rv = R_new.as_rotvec() * args.shoulder_scale
                    R_new = R.from_rotvec(rv)
                new_euler = R_new.as_euler(order, degrees=True)
                row[c:c + n] = new_euler.tolist()
    else:
        print("Skipping frame-0 subtraction (--no-subtract)")
        if R_align_inv:
            for row in frame_data:
                for tn in target_names:
                    if tn not in R_align_inv: continue
                    c, n = col_map[tn]
                    order = target_orders[tn].upper()
                    R_f = R.from_euler(order, row[c:c + n], degrees=True)
                    R_new = R_align_inv[tn] * R_f
                    row[c:c + n] = R_new.as_euler(order, degrees=True).tolist()

    # Write output
    out = lines[:motion_idx] + header
    for row in frame_data:
        out.append(" ".join(f"{v:.6f}" for v in row))
    with open(args.bvh_out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"Wrote {args.bvh_out}")


if __name__ == "__main__":
    main()
