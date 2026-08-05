"""Per-frame IK retarget for arm chain (Clavicle → Humerus → Ulna → Carpal).
Reads an input BVH (with normalized lower body / spine), computes BVH arm
joint world positions per frame, and solves skel arm DOFs to position skel
joints at scaled BVH targets. Writes Eulers back to output BVH.

Usage:
    python tools/ik_arm_retarget.py --in walk1_vert.bvh --out walk1_arm_ik.bvh
"""
import argparse
import os
import re
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R


def parse_bvh(path):
    with open(path) as f: lines = f.read().splitlines()
    joints = []
    stack = []
    pending_type = None; pending_name = None; motion_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s == "MOTION":
            motion_idx = i; break
        m = re.match(r"^(ROOT|JOINT)\s+(\S+)\s*$", s)
        if m:
            pending_type = m.group(1); pending_name = m.group(2); continue
        if s == "End Site":
            pending_type = "End"; pending_name = f"EndSite_{joints[-1]['name'] if joints else 'x'}"; continue
        if s == "{":
            j = {"type": pending_type, "name": pending_name,
                 "parent": stack[-1] if stack else -1,
                 "children": [], "channels": [], "offset": None}
            joints.append(j)
            new_idx = len(joints) - 1
            if stack: joints[stack[-1]]["children"].append(new_idx)
            stack.append(new_idx)
            pending_type = None; pending_name = None
            continue
        if s == "}":
            if stack: stack.pop()
            continue
        m = re.match(r"^OFFSET\s+(\S+)\s+(\S+)\s+(\S+)\s*$", s)
        if m and stack:
            joints[stack[-1]]["offset"] = tuple(float(x) for x in m.groups())
            continue
        m = re.match(r"^CHANNELS\s+(\d+)\s+(.+)$", s)
        if m and stack:
            joints[stack[-1]]["channels"] = m.group(2).split()
            continue
    return lines, joints, motion_idx


def channel_layout(joints):
    out = []
    def walk(i):
        if joints[i]["channels"]:
            out.append((i, len(joints[i]["channels"])))
        for c in joints[i]["children"]: walk(c)
    for i, j in enumerate(joints):
        if j["parent"] == -1: walk(i)
    return out


def bvh_fk_world(joints, row, n2c):
    """Compute world transforms for all joints at this frame."""
    Twr = [None] * len(joints)
    order = []
    def walk(i):
        order.append(i)
        for c in joints[i]["children"]: walk(c)
    for i, j in enumerate(joints):
        if j["parent"] == -1: walk(i)
    for ji in order:
        j = joints[ji]
        off = np.array(j["offset"]) if j["offset"] else np.zeros(3)
        Rl = R.identity(); pos_add = np.zeros(3)
        chs = j["channels"]
        if chs and j["name"] in n2c:
            c0, _ = n2c[j["name"]]; pi = 0; rc = []
            for ch in chs:
                v = row[c0 + pi]; pi += 1
                if ch.lower() == "xposition": pos_add[0] = v
                elif ch.lower() == "yposition": pos_add[1] = v
                elif ch.lower() == "zposition": pos_add[2] = v
                else: rc.append((ch[0].upper(), v))
            if rc:
                Rl = R.from_euler("".join(c for c, _ in rc), [v for _, v in rc], degrees=True)
        T = np.eye(4); T[:3, :3] = Rl.as_matrix(); T[:3, 3] = off + pos_add
        if j["parent"] >= 0: T = Twr[j["parent"]] @ T
        Twr[ji] = T
    return Twr


def find_joint(joints, suffix):
    for i, j in enumerate(joints):
        if j["type"] in ("ROOT", "JOINT") and (j["name"] == suffix or j["name"].endswith("_" + suffix)):
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="bvh_in", required=True)
    ap.add_argument("--out", dest="bvh_out", required=True)
    ap.add_argument("--skel-xml", default="data/zygote_skel.xml")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="If >0, only process first N frames (for testing).")
    args = ap.parse_args()

    sys.path.insert(0, ".")
    from core.dartHelper import saveSkeletonInfo, buildFromInfo
    from core.bvhparser import MyBVH

    lines, joints, motion_idx = parse_bvh(args.bvh_in)
    layout = channel_layout(joints)
    n2c = {}
    col = 0
    for ji, n in layout: n2c[joints[ji]["name"]] = (col, n); col += n
    rows = []
    header = []
    for ln in lines[motion_idx:]:
        s = ln.strip()
        if not s: continue
        if s.startswith("MOTION") or s.startswith("Frames") or s.startswith("Frame Time"):
            header.append(ln); continue
        rows.append([float(x) for x in s.split()])
    print(f"Frames: {len(rows)}, channels: {sum(n for _, n in layout)}")
    if args.max_frames > 0:
        rows = rows[:args.max_frames]
        print(f"Limiting to first {len(rows)} frames")

    # Find arm joint indices in BVH.
    arm_bvh = {}
    for suf in ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
                "RightShoulder", "RightArm", "RightForeArm", "RightHand"]:
        ji = find_joint(joints, suf)
        if ji is None:
            sys.exit(f"BVH missing {suf}")
        arm_bvh[suf] = ji

    # Build skel.
    si, rn, bi, *_ = saveSkeletonInfo(args.skel_xml)
    skel = buildFromInfo(si, rn)
    skel_body_names = {
        "LeftShoulder": "L_Clavicle0",
        "LeftArm": "L_Humerus0",
        "LeftForeArm": "L_Ulna0",
        "LeftHand": "L_Carpal0",
        "RightShoulder": "R_Clavicle0",
        "RightArm": "R_Humerus0",
        "RightForeArm": "R_Ulna0",
        "RightHand": "R_Carpal0",
    }
    # Build mocap baseline via MyBVH (gives leg/spine motion etc.).
    m = MyBVH(args.bvh_in, bi, skel, T_frame=0)

    # Identify DOF ranges for L/R arm joints.
    arm_dofs = {}
    for side_prefix in ("L_", "R_"):
        for body_name in ["Clavicle0", "Humerus0", "Ulna0", "Carpal0"]:
            full = side_prefix + body_name
            j = skel.getBodyNode(full).getParentJoint()
            idx0 = j.getIndexInSkeleton(0)
            nd = j.getNumDofs()
            arm_dofs[full] = (idx0, nd)

    # Compute BVH→skel scale via shoulder→wrist distance at f0.
    Twr0 = bvh_fk_world(joints, rows[0], n2c)
    bvh_left_shoulder = Twr0[arm_bvh["LeftShoulder"]][:3, 3]
    bvh_left_hand = Twr0[arm_bvh["LeftHand"]][:3, 3]
    bvh_arm_len = float(np.linalg.norm(bvh_left_hand - bvh_left_shoulder))
    # Skel arm length at rest.
    skel.setPositions(np.zeros(skel.getNumDofs()))
    skel_shoulder = skel.getBodyNode("L_Clavicle0").getCOM()
    skel_wrist = skel.getBodyNode("L_Carpal0").getCOM()
    skel_arm_len = float(np.linalg.norm(skel_wrist - skel_shoulder))
    scale = skel_arm_len / max(bvh_arm_len, 1e-6)
    print(f"BVH arm len: {bvh_arm_len:.3f}  Skel arm len: {skel_arm_len:.3f}  scale: {scale:.4f}")

    def solve_arm_ik(side, frame_idx, x0):
        """Solve arm DOFs for one side. side='L' or 'R'."""
        suf_prefix = "Left" if side == "L" else "Right"
        skel_prefix = side + "_"
        Twr = bvh_fk_world(joints, rows[frame_idx], n2c)
        # Target positions (relative to BVH shoulder, scaled, relative to skel shoulder).
        p_bvh_sh = Twr[arm_bvh[suf_prefix + "Shoulder"]][:3, 3]
        targets = {}
        for skel_body, suf in [(skel_prefix + "Humerus0", suf_prefix + "Arm"),
                                (skel_prefix + "Ulna0", suf_prefix + "ForeArm"),
                                (skel_prefix + "Carpal0", suf_prefix + "Hand")]:
            p_bvh = Twr[arm_bvh[suf]][:3, 3]
            rel = (p_bvh - p_bvh_sh) * scale
            skel_sh = skel.getBodyNode(skel_prefix + "Clavicle0").getCOM()
            targets[skel_body] = skel_sh + rel

        # DOFs to optimize for this side.
        dof_idx_lists = []
        for body in ["Clavicle0", "Humerus0", "Ulna0", "Carpal0"]:
            full = skel_prefix + body
            idx0, nd = arm_dofs[full]
            dof_idx_lists.append((idx0, nd))

        # Get current pose, replace arm DOFs with x.
        def err(x):
            pose = skel.getPositions().copy()
            xi = 0
            for idx0, nd in dof_idx_lists:
                for k in range(nd):
                    pose[idx0 + k] = x[xi]; xi += 1
            skel.setPositions(pose)
            e = 0.0
            for body_name, tgt in targets.items():
                p = skel.getBodyNode(body_name).getCOM()
                e += float(np.sum((p - tgt) ** 2))
            return e

        res = minimize(err, x0, method="L-BFGS-B",
                       options={"maxiter": 30, "ftol": 1e-5})
        return res.x, res.fun

    # For each frame, solve L then R arm IK.
    new_dofs = np.zeros((len(rows), skel.getNumDofs()))
    x0_L = np.zeros(3 + 3 + 1 + 3)
    x0_R = np.zeros(3 + 3 + 1 + 3)
    for fi in range(len(rows)):
        # Set skel pose to MyBVH baseline (legs/spine correct).
        pose = m.mocap_refs[fi].copy()
        skel.setPositions(pose)
        # L
        xL, eL = solve_arm_ik("L", fi, x0_L)
        x0_L = xL
        # Apply L DOFs to pose.
        xi = 0
        for body in ["Clavicle0", "Humerus0", "Ulna0", "Carpal0"]:
            idx0, nd = arm_dofs["L_" + body]
            for k in range(nd):
                pose[idx0 + k] = xL[xi]; xi += 1
        skel.setPositions(pose)
        # R
        xR, eR = solve_arm_ik("R", fi, x0_R)
        x0_R = xR
        xi = 0
        for body in ["Clavicle0", "Humerus0", "Ulna0", "Carpal0"]:
            idx0, nd = arm_dofs["R_" + body]
            for k in range(nd):
                pose[idx0 + k] = xR[xi]; xi += 1
        new_dofs[fi] = pose
        if fi % 50 == 0:
            print(f"  frame {fi}: errL={eL:.5f} errR={eR:.5f}")

    # Convert new DOFs back to BVH Eulers per joint, but only for arm joints.
    # The rest of the BVH (legs, spine) preserved as-is.
    # For each frame, modify the rows of BVH ForeArm/Hand/Shoulder/Arm channels.
    # We need to know the Euler order per joint and convert from rotvec (skel)
    # back to Euler.
    # Skel arm DOFs:
    #   L_Clavicle0 (Ball, 3 dof rotvec).
    #   L_Humerus0  (Ball, 3 dof rotvec).
    #   L_Ulna0     (Revolute, 1 dof scalar).
    #   L_Carpal0   (Ball, 3 dof rotvec).
    bvh_to_skel = {
        "LeftShoulder": ("L_Clavicle0", "ball"),
        "LeftArm": ("L_Humerus0", "ball"),
        "LeftForeArm": ("L_Ulna0", "revolute"),
        "LeftHand": ("L_Carpal0", "ball"),
        "RightShoulder": ("R_Clavicle0", "ball"),
        "RightArm": ("R_Humerus0", "ball"),
        "RightForeArm": ("R_Ulna0", "revolute"),
        "RightHand": ("R_Carpal0", "ball"),
    }
    for fi, row in enumerate(rows):
        for bvh_name, (skel_body, jt) in bvh_to_skel.items():
            j_idx = arm_bvh[bvh_name]
            chs = joints[j_idx]["channels"]
            rot_chans = [c for c in chs if c.lower().endswith("rotation")]
            if not rot_chans: continue
            order = "".join(c[0] for c in rot_chans).upper()
            c0 = n2c[joints[j_idx]["name"]][0]
            rot_offsets = [k for k, ch in enumerate(chs) if ch.lower().endswith("rotation")]
            idx0, nd = arm_dofs[skel_body]
            if jt == "ball":
                rv = new_dofs[fi, idx0:idx0 + 3]
                Rmat = R.from_rotvec(rv)
            else:
                ang = new_dofs[fi, idx0]
                # Use revolute axis from skel.
                ax = np.array(skel.getBodyNode(skel_body).getParentJoint().getAxis(),
                              dtype=np.float64, copy=True)
                ax = ax / max(np.linalg.norm(ax), 1e-12)
                Rmat = R.from_rotvec(ax * ang)
            eul = Rmat.as_euler(order, degrees=True)
            for k, off in enumerate(rot_offsets):
                row[c0 + off] = float(eul[k])

    out_lines = lines[:motion_idx] + header
    for row in rows:
        out_lines.append(" ".join(f"{v:.6f}" for v in row))
    with open(args.bvh_out, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Wrote {args.bvh_out}")


if __name__ == "__main__":
    main()
