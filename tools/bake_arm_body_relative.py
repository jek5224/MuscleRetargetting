"""World-direction arm retarget for LaFAN bone-aligned BVH.

Matches skel arm WORLD BONE DIRECTION to BVH arm world bone direction.
Avoids the asymmetric skel-body-local frame issue.

Algorithm per arm (LeftArm, RightArm):
  1. BVH FK → d_bvh(f) = direction from arm joint to hand joint, world.
  2. Skel rest: d_skel_rest = direction from humerus body to carpal body, world.
  3. R_motion_world(f) = rotation that takes d_skel_rest to d_bvh(f).
     This is the rotation skel humerus body needs to undergo.
  4. Skel humerus body world target = R_motion_world(f) @ R_humerus_rest_world.
  5. Skel humerus joint local = parent_skel_world_rest.T @ target.
  6. Inverse-T_net: channel = parent_world_bvh(0).T @ joint_local @ parent_world_bvh(0).

Inman scapulohumeral split: clavicle takes ~27% of humerus rotvec, residual
composes onto humerus.

f=0 calibration: T_frame=0 forces skel rest at f=0. For LaFAN clips with
T-pose intro, copy ref_frame's channel into row[0] so MyBVH calibrates
skel rest = walking equilibrium. f=0 visual is N-pose (intended; T-pose
mismatch only at first frame).
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


def bvh_fk_world_full(joints, row, n2c):
    """4x4 world transforms."""
    T = [None] * len(joints); order = []
    def walk(i):
        order.append(i)
        for c in joints[i]["children"]: walk(c)
    for i, j in enumerate(joints):
        if j["parent"] == -1: walk(i)
    for ji in order:
        j = joints[ji]; Rl = R.identity(); pos = np.zeros(3)
        if j["channels"] and j["name"] in n2c:
            chs = j["channels"]; c0, _ = n2c[j["name"]]; rc = []
            for k, ch in enumerate(chs):
                v = row[c0 + k]
                if ch.lower() == "xposition": pos[0] = v
                elif ch.lower() == "yposition": pos[1] = v
                elif ch.lower() == "zposition": pos[2] = v
                elif ch.lower().endswith("rotation"):
                    rc.append((ch[0].upper(), v))
            if rc:
                Rl = R.from_euler("".join(c for c, _ in rc), [v for _, v in rc], degrees=True)
        off = np.array(j["offset"]) if j["offset"] else np.zeros(3)
        M = np.eye(4); M[:3, :3] = Rl.as_matrix(); M[:3, 3] = off + pos
        par = j["parent"]
        T[ji] = M if par < 0 else T[par] @ M
    return T


def rotation_from_to(a, b):
    """Rotation matrix that takes unit vector a to unit vector b."""
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    c = np.cross(a, b)
    cn = np.linalg.norm(c)
    dot = float(a @ b)
    if cn < 1e-9:
        if dot > 0:
            return np.eye(3)
        else:
            # Antipodal — pick any perpendicular axis.
            perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            axis = np.cross(a, perp); axis /= np.linalg.norm(axis)
            return R.from_rotvec(axis * np.pi).as_matrix()
    ang = float(np.arctan2(cn, dot))
    return R.from_rotvec(c / cn * ang).as_matrix()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="bvh_in", required=True)
    ap.add_argument("--out", dest="bvh_out", required=True)
    ap.add_argument("--skel-xml", default="data/zygote_skel.xml")
    ap.add_argument("--scapulohumeral", type=float, default=0.27)
    ap.add_argument("--ref-frame", type=int, default=200,
                    help="Frame index treated as walking equilibrium; written "
                         "to output f=0 so T_frame=0 calibrates correctly.")
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

    arm_idx = {}
    for suf in ["LeftShoulder", "LeftArm", "LeftHand", "RightShoulder", "RightArm", "RightHand"]:
        arm_idx[suf] = find_joint(joints, suf)
    if any(v is None for v in arm_idx.values()):
        sys.exit("missing arm joints")

    # Skel rest arm world direction.
    sys.path.insert(0, ".")
    from core.dartHelper import saveSkeletonInfo, buildFromInfo
    si, rn, *_ = saveSkeletonInfo(args.skel_xml)
    skel = buildFromInfo(si, rn)
    skel.setPositions(np.zeros(skel.getNumDofs()))
    skel_arm_data = {}
    for side, hum, car, clav in [("L", "L_Humerus0", "L_Carpal0", "L_Clavicle0"),
                                   ("R", "R_Humerus0", "R_Carpal0", "R_Clavicle0")]:
        hum_bn = skel.getBodyNode(hum)
        car_bn = skel.getBodyNode(car)
        clav_bn = skel.getBodyNode(clav)
        d_rest = car_bn.getTransform().translation() - hum_bn.getTransform().translation()
        skel_arm_data[side] = {
            "d_rest_world": d_rest.copy(),
            "R_hum_body_rest": np.asarray(hum_bn.getTransform().rotation()).copy(),
            "R_clav_body_rest": np.asarray(clav_bn.getTransform().rotation()).copy(),
        }
    # BVH arm world rest rotation (from input BVH FK at f=0). Used to compute
    # alignment for full-rotation match.
    T0_bvh = bvh_fk_world_full(joints, rows[0], n2c)
    R_align_arm = {}
    for side, sh_suf, arm_suf, hand_suf in [("L", "LeftShoulder", "LeftArm", "LeftHand"),
                                              ("R", "RightShoulder", "RightArm", "RightHand")]:
        ji = arm_idx[arm_suf]
        R_bvh_arm_rest = T0_bvh[ji][:3, :3].copy()
        R_skel_hum_rest = skel_arm_data[side]["R_hum_body_rest"]
        # R_align: skel arm world(f) = R_align @ R_bvh_arm_world(f).
        # At rest: R_skel_hum_rest = R_align @ R_bvh_arm_rest → R_align = R_skel @ R_bvh.inv.
        R_align_arm[side] = R_skel_hum_rest @ R_bvh_arm_rest.T

    # Pass 1: compute parent_world_at_F0 from input rows[0] (initial estimate).
    # Pass 2 (after writing): recompute from updated rows[0] for self-consistency.
    def compute_parent_world(rows0):
        T0 = bvh_fk_world_full(joints, rows0, n2c)
        d = {}
        for suf in ["LeftShoulder", "LeftArm", "RightShoulder", "RightArm"]:
            par = joints[arm_idx[suf]]["parent"]
            d[suf] = T0[par][:3, :3].copy() if par >= 0 else np.eye(3)
        return d
    bvh_parent_world_at_outputF0 = compute_parent_world(rows[0])

    side_pairs = [("L", "LeftShoulder", "LeftArm", "LeftHand"),
                  ("R", "RightShoulder", "RightArm", "RightHand")]

    arm_channel_info = {}
    for suf in ["LeftShoulder", "LeftArm", "RightShoulder", "RightArm"]:
        ji = arm_idx[suf]
        chs = joints[ji]["channels"]
        order = "".join(c[0] for c in chs if c.lower().endswith("rotation")).upper()
        c0 = n2c[joints[ji]["name"]][0]
        rot_offs = [k for k, c in enumerate(chs) if c.lower().endswith("rotation")]
        arm_channel_info[suf] = (c0, order, rot_offs)

    share = float(args.scapulohumeral)

    def compute_all_channels(P_dict):
        computed = {suf: [None] * len(rows) for suf in arm_channel_info}
        for fi, row in enumerate(rows):
            Tf = bvh_fk_world_full(joints, row, n2c)
            for side, sh_suf, arm_suf, hand_suf in side_pairs:
                # Arm bone direction match (world).
                bvh_arm_pos = Tf[arm_idx[arm_suf]][:3, 3]
                bvh_hand_pos = Tf[arm_idx[hand_suf]][:3, 3]
                d_bvh_world = bvh_hand_pos - bvh_arm_pos
                d_bvh_world = d_bvh_world / max(np.linalg.norm(d_bvh_world), 1e-12)
                d_rest = skel_arm_data[side]["d_rest_world"]
                R_motion_world = rotation_from_to(d_rest, d_bvh_world)
                rv = R.from_matrix(R_motion_world).as_rotvec()
                R_clav_world = R.from_rotvec(rv * share).as_matrix()
                R_residual_world = R.from_rotvec(rv * (1.0 - share)).as_matrix()
                R_humerus_world = R_residual_world
                for suf, R_local_skel in [(sh_suf, R_clav_world), (arm_suf, R_humerus_world)]:
                    P = P_dict[suf]
                    R_channel = P.T @ R_local_skel @ P
                    c0, order, rot_offs = arm_channel_info[suf]
                    eul = R.from_matrix(R_channel).as_euler(order, degrees=True)
                    computed[suf][fi] = list(eul)
        return computed

    def write_channels(computed):
        for fi, row in enumerate(rows):
            for suf in arm_channel_info:
                c0, order, rot_offs = arm_channel_info[suf]
                eul = computed[suf][fi]
                for k, off in enumerate(rot_offs):
                    row[c0 + off] = float(eul[k])

    # 2-pass with channel(0)=identity forcing.
    # Pass 1: zero arm channels at row[0] so P from FK gives Sternum-direction
    # parent_world (the actual P that MyBVH will see).
    for suf in arm_channel_info:
        c0, order, rot_offs = arm_channel_info[suf]
        for k, off in enumerate(rot_offs):
            rows[0][c0 + off] = 0.0
    bvh_parent_world_at_outputF0 = compute_parent_world(rows[0])
    # Compute all channels using this P.
    computed = compute_all_channels(bvh_parent_world_at_outputF0)
    # Force channel(0) = identity again (compute_all_channels overwrites).
    for suf in arm_channel_info:
        computed[suf][0] = [0.0] * 3
    write_channels(computed)

    out_lines = lines[:mi] + motion_header
    for row in rows:
        out_lines.append(" ".join(f"{v:.6f}" for v in row))
    with open(args.bvh_out, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Wrote {args.bvh_out}")


if __name__ == "__main__":
    main()
