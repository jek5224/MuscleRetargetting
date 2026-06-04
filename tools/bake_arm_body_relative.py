"""Body-relative arm retarget for LaFAN bone-aligned BVH.

Accounts for the actor's body orientation per frame (subtracts root yaw)
so BVH bone-aligned channel inflation from rest-pose encoding doesn't
propagate as huge skel mocap rotations.

Algorithm per arm joint (LeftShoulder, LeftArm + R mirrors):
  1. BVH FK → R_arm_world(f), R_hips_world(f).
  2. R_arm_body_rel(f) = R_hips_world(f).T @ R_arm_world(f).
  3. R_motion_body(f) = R_arm_body_rel(f) @ R_arm_body_rel(0).T.
     (Body-relative motion from f0; root yaw cancels.)
  4. Channel value = R_parent_world_bvh(0).T @ R_motion_body(f) @ R_parent_world_bvh(0).
     (Inverse-T_net assuming R_local(0) = identity; under MyBVH T_frame=0
     this produces T_net = R_motion_body(f) = skel joint local rotation.)

At f=0 R_motion = identity → channel(0) = identity → MyBVH T_frame=0
forces skel rest (N-pose). For f>0 channel = body-relative motion only,
magnitude = anatomical arm swing.

Note: assumes parent_world(f) ≈ parent_world(0) (clavicle motion small).
For walking with ~5-15° clavicle, error is bounded.
"""
import argparse
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_bvh(path):
    with open(path) as f:
        lines = f.read().splitlines()
    joints = []; stack = []; pending_type = None; pending_name = None; motion_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s == "MOTION":
            motion_idx = i; break
        m = re.match(r"^(ROOT|JOINT)\s+(\S+)\s*$", s)
        if m:
            pending_type = m.group(1); pending_name = m.group(2); continue
        if s == "End Site":
            pending_type = "End"
            pending_name = f"EndSite_{joints[-1]['name'] if joints else 'x'}"
            continue
        if s == "{":
            j = {"type": pending_type, "name": pending_name,
                 "parent": stack[-1] if stack else -1,
                 "children": [], "channels": [], "offset": None}
            joints.append(j); idx = len(joints) - 1
            if stack: joints[stack[-1]]["children"].append(idx)
            stack.append(idx); pending_type = None; pending_name = None; continue
        if s == "}":
            if stack: stack.pop()
            continue
        m = re.match(r"^OFFSET\s+(\S+)\s+(\S+)\s+(\S+)\s*$", s)
        if m and stack:
            joints[stack[-1]]["offset"] = tuple(float(x) for x in m.groups()); continue
        m = re.match(r"^CHANNELS\s+(\d+)\s+(.+)$", s)
        if m and stack:
            joints[stack[-1]]["channels"] = m.group(2).split(); continue
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


def find_joint(joints, suf):
    for i, j in enumerate(joints):
        if j["type"] in ("ROOT", "JOINT") and (j["name"] == suf or j["name"].endswith("_" + suf)):
            return i
    return None


def bvh_fk_world_R(joints, row, n2c):
    Twr = [None] * len(joints); order = []
    def walk(i):
        order.append(i)
        for c in joints[i]["children"]: walk(c)
    for i, j in enumerate(joints):
        if j["parent"] == -1: walk(i)
    for ji in order:
        j = joints[ji]; Rl = R.identity()
        if j["channels"] and j["name"] in n2c:
            chs = j["channels"]; c0, _ = n2c[j["name"]]; rc = []
            for k, ch in enumerate(chs):
                v = row[c0 + k]
                if ch.lower().endswith("rotation"):
                    rc.append((ch[0].upper(), v))
            if rc:
                Rl = R.from_euler("".join(c for c, _ in rc), [v for _, v in rc], degrees=True)
        M = Rl.as_matrix()
        if j["parent"] >= 0 and Twr[j["parent"]] is not None:
            M = Twr[j["parent"]] @ M
        Twr[ji] = M
    return Twr


def joint_local_R(j, row, n2c):
    chs = j["channels"]; c0, _ = n2c[j["name"]]; rc = []
    for k, ch in enumerate(chs):
        v = row[c0 + k]
        if ch.lower().endswith("rotation"):
            rc.append((ch[0].upper(), v))
    if not rc: return R.identity()
    return R.from_euler("".join(c for c, _ in rc), [v for _, v in rc], degrees=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="bvh_in", required=True)
    ap.add_argument("--out", dest="bvh_out", required=True)
    ap.add_argument("--scapulohumeral", type=float, default=0.27,
                    help="Inman clavicle share. 0.27 = anatomical SC:GH ratio.")
    args = ap.parse_args()

    lines, joints, mi = parse_bvh(args.bvh_in)
    layout = channel_layout(joints)
    n2c = {}; col = 0
    for ji, n in layout:
        n2c[joints[ji]["name"]] = (col, n); col += n

    motion_header = []; rows = []
    for ln in lines[mi:]:
        s = ln.strip()
        if not s: continue
        if s.startswith(("MOTION", "Frames", "Frame Time")):
            motion_header.append(ln); continue
        rows.append([float(x) for x in s.split()])

    hips_ji = find_joint(joints, "Hips")
    arm_joints = {}
    for suf in ["LeftShoulder", "LeftArm", "RightShoulder", "RightArm"]:
        arm_joints[suf] = find_joint(joints, suf)
    if any(v is None for v in arm_joints.values()) or hips_ji is None:
        sys.exit("missing arm joints or Hips")

    # f0 world rotations.
    Twr0 = bvh_fk_world_R(joints, rows[0], n2c)
    R_hips0 = Twr0[hips_ji]
    # BVH parent world rotation at f0 per arm joint.
    parent_world_0 = {}
    arm_body_rest = {}
    for suf, ji in arm_joints.items():
        par = joints[ji]["parent"]
        parent_world_0[suf] = Twr0[par].copy() if par >= 0 else np.eye(3)
        # Body-relative arm rest = root.T @ arm world rest.
        arm_body_rest[suf] = R_hips0.T @ Twr0[ji]

    # Per side, compute scapulohumeral share lookup.
    side_pairs = [("L", "LeftShoulder", "LeftArm"),
                  ("R", "RightShoulder", "RightArm")]

    # Channel layout: each arm joint has 3 rotation channels.
    arm_channels = {}
    for suf in arm_joints:
        chs = joints[arm_joints[suf]]["channels"]
        order = "".join(c[0] for c in chs if c.lower().endswith("rotation")).upper()
        c0 = n2c[joints[arm_joints[suf]]["name"]][0]
        rot_offs = [k for k, c in enumerate(chs) if c.lower().endswith("rotation")]
        arm_channels[suf] = (c0, order, rot_offs)

    share = float(args.scapulohumeral)

    # Find walking equilibrium reference frame: pick median frame where
    # actor's arm is far from T-pose (small +X-world component in body local).
    # Heuristic: choose first frame after f=100 where Hips→LeftHand body-local
    # Z component (lateral) is below 0.2 (arm not extended sideways).
    lh_ji = find_joint(joints, "LeftHand")
    ref_frame = 0
    for fi in range(min(len(rows), 500), min(len(rows), 100), -1):
        T = bvh_fk_world_R(joints, rows[fi], n2c)
        R_h = T[hips_ji]
        # body-local position of left hand
        lh_world_pos = None  # placeholder; using rotation as proxy
        break
    # Simpler: pick frame 200 (after T-pose intro for LaFAN walking clips).
    ref_frame = min(200, len(rows) - 1)
    print(f"  Using ref_frame={ref_frame} as walking equilibrium calibration")

    # Compute channel value per frame, store in arrays. Then overwrite
    # row[0]'s arm channels with row[ref_frame]'s computed channels so
    # MyBVH T_frame=0 calibrates skel rest = walking equilibrium (not T-pose).
    computed_channels = {suf: [None] * len(rows) for suf in arm_joints}
    for fi, row in enumerate(rows):
        Twr = bvh_fk_world_R(joints, row, n2c)
        R_hips_f = Twr[hips_ji]
        for side, sh_suf, arm_suf in side_pairs:
            R_sh_world_f = Twr[arm_joints[sh_suf]]
            R_arm_world_f = Twr[arm_joints[arm_suf]]
            R_sh_body_f = R_hips_f.T @ R_sh_world_f
            R_arm_body_f = R_hips_f.T @ R_arm_world_f
            R_motion_sh = R_sh_body_f @ arm_body_rest[sh_suf].T
            R_motion_arm = R_arm_body_f @ arm_body_rest[arm_suf].T
            rv_sh = R.from_matrix(R_motion_sh).as_rotvec()
            R_clav = R.from_rotvec(rv_sh * share).as_matrix()
            R_residual = R.from_rotvec(rv_sh * (1.0 - share)).as_matrix()
            R_humerus = R_residual @ R_motion_arm
            for suf, R_target in [(sh_suf, R_clav), (arm_suf, R_humerus)]:
                P = parent_world_0[suf]
                R_channel = P.T @ R_target @ P
                c0, order, rot_offs = arm_channels[suf]
                eul = R.from_matrix(R_channel).as_euler(order, degrees=True)
                computed_channels[suf][fi] = list(eul)

    # Write computed channels back to rows, but use ref_frame's value for f=0.
    for fi, row in enumerate(rows):
        for suf in arm_joints:
            c0, order, rot_offs = arm_channels[suf]
            src_fi = ref_frame if fi == 0 else fi
            eul = computed_channels[suf][src_fi]
            for k, off in enumerate(rot_offs):
                row[c0 + off] = float(eul[k])

    # Write out.
    out_lines = lines[:mi] + motion_header
    for row in rows:
        out_lines.append(" ".join(f"{v:.6f}" for v in row))
    with open(args.bvh_out, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Wrote {args.bvh_out}")


if __name__ == "__main__":
    main()
