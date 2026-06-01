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

    # Rewrite each frame: new R_local(f) = R_local_0.inv * R_local(f)
    for row in frame_data:
        for tn in target_names:
            c, n = col_map[tn]
            order = target_orders[tn].upper()
            euler_f = row[c:c + n]
            R_f = R.from_euler(order, euler_f, degrees=True)
            R_new = R_local_0[tn].inv() * R_f
            new_euler = R_new.as_euler(order, degrees=True)
            row[c:c + n] = new_euler.tolist()

    # Write output
    out = lines[:motion_idx] + header
    for row in frame_data:
        out.append(" ".join(f"{v:.6f}" for v in row))
    with open(args.bvh_out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"Wrote {args.bvh_out}")


if __name__ == "__main__":
    main()
