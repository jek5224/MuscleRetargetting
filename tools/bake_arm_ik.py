"""IK-based arm retarget. Per frame, solve for L/R_Clavicle (3-DoF) +
L/R_Humerus (3-DoF) + L/R_Ulna (1-DoF) angles that place skel wrist at
BVH actor's wrist position (scaled to skel proportions).

Bypasses conjugation-based arm bake which suffered from BVH-T-pose vs
skel-N-pose rest mismatch: BVH joint-local rotation expressed in skel-
parent-body frame mapped motion onto wrong joint axes.

Inputs:
  --in       : BVH after Spine+Sternum bake (pre-arm-bake)
  --orig-bvh : normalized BVH for BVH FK world positions
  --skel-xml : skel XML

Algorithm per frame, per side:
  1. BVH FK on --orig-bvh → shoulder, elbow, wrist world positions.
  2. target_wrist_rel = (wrist - shoulder) * (skel_arm_len / bvh_arm_len).
  3. Skel: set arm DOFs to zero, all other DOFs from MyBVH mocap.
  4. scipy.optimize.minimize on 7 vars (Clavicle xyz + Humerus xyz + Ulna).
     Cost = ||skel_carpal - (skel_humerus + target)||^2 + small elbow-bend match.
  5. Read DOF values. Convert to BVH channel via inverse-T_net formula.

Outputs new BVH with channels rewritten.
"""
import argparse
import os
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize


def parse_bvh(path):
    with open(path) as f:
        lines = f.read().splitlines()
    joints = []
    stack = []
    pending_type = None
    pending_name = None
    motion_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s == "MOTION":
            motion_idx = i
            break
        m = re.match(r"^(ROOT|JOINT)\s+(\S+)\s*$", s)
        if m:
            pending_type = m.group(1)
            pending_name = m.group(2)
            continue
        if s == "End Site":
            pending_type = "End"
            pending_name = f"EndSite_{joints[-1]['name'] if joints else 'x'}"
            continue
        if s == "{":
            j = {"type": pending_type, "name": pending_name,
                 "parent": stack[-1] if stack else -1,
                 "children": [], "channels": [], "offset": None}
            joints.append(j)
            idx = len(joints) - 1
            if stack:
                joints[stack[-1]]["children"].append(idx)
            stack.append(idx)
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
        for c in joints[i]["children"]:
            walk(c)
    for i, j in enumerate(joints):
        if j["parent"] == -1:
            walk(i)
    return out


def bvh_fk_world(joints, row, n2c):
    """Per-joint 4x4 world transform (rotation + translation)."""
    T = [None] * len(joints)
    order = []
    def walk(i):
        order.append(i)
        for c in joints[i]["children"]: walk(c)
    for i, j in enumerate(joints):
        if j["parent"] == -1: walk(i)
    for ji in order:
        j = joints[ji]
        Rl = R.identity()
        pos = np.zeros(3)
        if j["channels"] and j["name"] in n2c:
            chs = j["channels"]; c0, _ = n2c[j["name"]]; rc = []
            for k, ch in enumerate(chs):
                v = row[c0 + k]
                if ch.lower() == "xposition": pos[0] = v
                elif ch.lower() == "yposition": pos[1] = v
                elif ch.lower() == "zposition": pos[2] = v
                elif ch.lower().endswith("rotation"): rc.append((ch[0].upper(), v))
            if rc: Rl = R.from_euler("".join(c for c, _ in rc), [v for _, v in rc], degrees=True)
        off = np.array(j["offset"]) if j["offset"] else np.zeros(3)
        M = np.eye(4)
        M[:3, :3] = Rl.as_matrix()
        M[:3, 3] = off + pos
        par = j["parent"]
        T[ji] = M if par < 0 else T[par] @ M
    return T


def find_joint(joints, suf):
    for i, j in enumerate(joints):
        if j["type"] in ("ROOT", "JOINT") and (j["name"] == suf or j["name"].endswith("_" + suf)):
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="bvh_in", required=True)
    ap.add_argument("--out", dest="bvh_out", required=True)
    ap.add_argument("--orig-bvh", required=True,
                    help="Normalized BVH for BVH FK world positions.")
    ap.add_argument("--skel-xml", default="data/zygote_skel.xml")
    ap.add_argument("--limit-frames", type=int, default=0,
                    help="Process only first N frames (0 = all).")
    args = ap.parse_args()

    # Parse arm-input BVH (post-sternum, pre-arm).
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

    # Parse orig BVH for BVH wrist world positions.
    olines, ojoints, omi = parse_bvh(args.orig_bvh)
    olayout = channel_layout(ojoints)
    on2c = {}; ocol = 0
    for ji, n in olayout:
        on2c[ojoints[ji]["name"]] = (ocol, n); ocol += n
    orows = []
    for ln in olines[omi:]:
        s = ln.strip()
        if not s or s.startswith(("MOTION", "Frames", "Frame Time")): continue
        orows.append([float(x) for x in s.split()])

    # Build skel + load mocap.
    sys.path.insert(0, ".")
    from core.dartHelper import saveSkeletonInfo, buildFromInfo
    from core.bvhparser import MyBVH
    si, rn, bvh_info, *_ = saveSkeletonInfo(args.skel_xml)
    skel = buildFromInfo(si, rn)
    # Use T_frame=0 to match viewer convention.
    mbv = MyBVH(args.bvh_in, bvh_info, skel, T_frame=0)
    mocap = mbv.mocap_refs

    n_frames = mocap.shape[0]
    if args.limit_frames > 0:
        n_frames = min(n_frames, args.limit_frames)

    # Find skel DOF indices.
    def joint_dofs(name):
        for j in range(skel.getNumJoints()):
            jn = skel.getJoint(j)
            if jn.getName() == name:
                return jn.getIndexInSkeleton(0), jn.getNumDofs()
        return None, 0

    sides_skel = []
    for L_or_R in ["L", "R"]:
        clav_i, clav_n = joint_dofs(f"{L_or_R}_Clavicle0")
        hum_i, hum_n = joint_dofs(f"{L_or_R}_Humerus0")
        ulna_i, ulna_n = joint_dofs(f"{L_or_R}_Ulna0")
        sides_skel.append({
            "side": L_or_R,
            "clav_i": clav_i, "clav_n": clav_n,
            "hum_i": hum_i, "hum_n": hum_n,
            "ulna_i": ulna_i, "ulna_n": ulna_n,
            "humerus_body": f"{L_or_R}_Humerus0",
            "carpal_body": f"{L_or_R}_Carpal0",
            "ulna_body": f"{L_or_R}_Ulna0",
        })

    # Find BVH arm joint indices in --orig-bvh.
    def of(name):
        for i, j in enumerate(ojoints):
            if j["type"] in ("ROOT", "JOINT") and (j["name"] == name or j["name"].endswith("_" + name)):
                return i
        return None

    bvh_sides = []
    for L_or_R, prefix in [("L", "Left"), ("R", "Right")]:
        arm = of(f"{prefix}Arm")
        forearm = of(f"{prefix}ForeArm")
        hand = of(f"{prefix}Hand")
        bvh_sides.append({"side": L_or_R, "arm": arm, "forearm": forearm, "hand": hand})
    # BVH root for actor-local frame.
    bvh_hips_ji = of("Hips")

    # R_align_arm: rotation in BODY-LOCAL frame that maps BVH arm rest
    # direction (T-pose, in BVH-hips-local) to skel arm rest direction
    # (N-pose, in skel-root-local). Per side because T-pose mirrors (+X vs -X).
    T0_bvh = bvh_fk_world(ojoints, orows[0], on2c)
    R_bvh_hips_0 = T0_bvh[bvh_hips_ji][:3, :3]
    skel.setPositions(np.zeros(skel.getNumDofs()))
    R_skel_root_rest_for_align = np.asarray(
        skel.getBodyNode("Saccrum_Coccyx0").getTransform().rotation()).copy()
    R_align_arm = {}
    for bs, ss in zip(bvh_sides, sides_skel):
        bvh_dir_world = T0_bvh[bs["hand"]][:3, 3] - T0_bvh[bs["arm"]][:3, 3]
        bvh_dir_local = R_bvh_hips_0.T @ bvh_dir_world
        bvh_dir_local /= np.linalg.norm(bvh_dir_local)
        sk_dir_world = (skel.getBodyNode(ss["carpal_body"]).getTransform().translation()
                        - skel.getBodyNode(ss["humerus_body"]).getTransform().translation())
        sk_dir_local = R_skel_root_rest_for_align.T @ sk_dir_world
        sk_dir_local /= np.linalg.norm(sk_dir_local)
        cross = np.cross(bvh_dir_local, sk_dir_local)
        cn = np.linalg.norm(cross)
        ang = float(np.arctan2(cn, float(bvh_dir_local @ sk_dir_local)))
        if cn > 1e-9:
            R_align_arm[bs["side"]] = R.from_rotvec(cross / cn * ang).as_matrix()
        else:
            R_align_arm[bs["side"]] = np.eye(3)
    print(f"  R_align_arm L rotvec={R.from_matrix(R_align_arm['L']).as_rotvec().round(2)} "
          f"R rotvec={R.from_matrix(R_align_arm['R']).as_rotvec().round(2)}")

    # Direct world target: BVH arm direction in BVH world ≈ skel arm direction
    # in skel world (both Y-up, same axes). Scale magnitude by skel/BVH arm
    # length. Walking pose (arm hanging) maps to skel hanging — natural.
    # T-pose intro frames intentionally won't match (skel rest is N-pose).
    T0 = bvh_fk_world(ojoints, orows[0], on2c)
    skel.setPositions(np.zeros(skel.getNumDofs()))
    scales = {}
    for bs, ss in zip(bvh_sides, sides_skel):
        bvh_sh = T0[bs["arm"]][:3, 3]
        bvh_wr = T0[bs["hand"]][:3, 3]
        bvh_len = np.linalg.norm(bvh_wr - bvh_sh)
        sk_sh = skel.getBodyNode(ss["humerus_body"]).getTransform().translation().copy()
        sk_wr = skel.getBodyNode(ss["carpal_body"]).getTransform().translation().copy()
        sk_len = np.linalg.norm(sk_wr - sk_sh)
        scales[bs["side"]] = sk_len / max(bvh_len, 1e-9)
        print(f"  {bs['side']}: scale={scales[bs['side']]:.3f}")

    # Channel column maps.
    bvh_arm_names = {"L": ("LeftShoulder", "LeftArm", "LeftForeArm"),
                     "R": ("RightShoulder", "RightArm", "RightForeArm")}
    chan_cols = {}
    chan_orders = {}
    for side, names in bvh_arm_names.items():
        for nm in names:
            ji = find_joint(joints, nm)
            if ji is None: continue
            chs = joints[ji]["channels"]
            order = "".join(c[0] for c in chs if c.lower().endswith("rotation")).upper()
            c0 = n2c[joints[ji]["name"]][0]
            chan_cols[(side, nm)] = c0
            chan_orders[(side, nm)] = order

    # BVH parent world rotation at f0 for each arm BVH joint.
    # We force arm channels f=0 to be identity in OUTPUT — so chain through
    # arm joints passes parent rotation unchanged. parent_world_0 for any
    # arm joint = chain through trunk to that joint's parent in INPUT BVH,
    # but with ARM joints in the chain set to identity at f=0.
    arm_set = {nm for side in ("L", "R") for nm in bvh_arm_names[side]}
    rows0_zeroed = [list(rows[0])]
    for side in ("L", "R"):
        for nm in bvh_arm_names[side]:
            ji = find_joint(joints, nm)
            if ji is None: continue
            c0 = n2c[joints[ji]["name"]][0]
            chs = joints[ji]["channels"]
            for k, ch in enumerate(chs):
                if ch.lower().endswith("rotation"):
                    rows0_zeroed[0][c0 + k] = 0.0
    Twr_in_0 = bvh_fk_world(joints, rows0_zeroed[0], n2c)
    parent_world_0 = {}
    for side in ("L", "R"):
        for nm in bvh_arm_names[side]:
            ji = find_joint(joints, nm)
            if ji is None: continue
            par = joints[ji]["parent"]
            parent_world_0[(side, nm)] = (Twr_in_0[par][:3, :3].copy()
                                          if par >= 0 else np.eye(3))

    # IK loop.
    n_dofs = skel.getNumDofs()
    print(f"  IK over {n_frames} frames × 2 sides × 4 dofs...")
    progress_step = max(1, n_frames // 20)
    new_mocap = mocap.copy()
    # Warm-start cache: previous frame's solution per side for temporal
    # coherence (avoids per-frame local-minimum jitter).
    prev_x = {"L": None, "R": None}
    for f in range(n_frames):
        if f % progress_step == 0:
            print(f"    frame {f}/{n_frames}")
        # BVH FK at frame f.
        Tf = bvh_fk_world(ojoints, orows[f] if f < len(orows) else orows[-1], on2c)
        # Set base pose from mocap (everything else fixed).
        base_pose = mocap[f].copy()
        # Step 1: BVH offset in BVH-hips-local frame (removes actor yaw).
        # Step 2: per-side R_align_arm rotates from BVH arm rest direction
        #         to skel arm rest direction.
        # Step 3: rotate by skel root world (applies skel actor's facing).
        R_bvh_root = Tf[bvh_hips_ji][:3, :3] if bvh_hips_ji is not None else np.eye(3)
        skel.setPositions(base_pose)
        R_skel_root = np.asarray(skel.getBodyNode("Saccrum_Coccyx0").getTransform().rotation())
        for bs, ss in zip(bvh_sides, sides_skel):
            side = bs["side"]
            sc = scales[side]
            bvh_sh = Tf[bs["arm"]][:3, 3]
            bvh_el = Tf[bs["forearm"]][:3, 3]
            bvh_wr = Tf[bs["hand"]][:3, 3]
            bvh_rel_world = (bvh_wr - bvh_sh) * sc
            bvh_rel_local = R_bvh_root.T @ bvh_rel_world
            target_rel = R_skel_root @ (R_align_arm[side] @ bvh_rel_local)
            bvh_elbow_world = (bvh_el - bvh_sh) * sc
            target_elbow_rel = R_skel_root @ (R_align_arm[side] @ (R_bvh_root.T @ bvh_elbow_world))
            # IK vars: 7 = clavicle (3) + humerus (3) + ulna (1).
            # Initial humerus: rotate skel arm rest direction to target.
            # Skel rest: set zero pose, get current arm direction.
            init_pose = base_pose.copy()
            init_pose[ss["clav_i"]:ss["clav_i"]+3] = 0
            init_pose[ss["hum_i"]:ss["hum_i"]+3] = 0
            init_pose[ss["ulna_i"]] = 0
            skel.setPositions(init_pose)
            sh0 = skel.getBodyNode(ss["humerus_body"]).getTransform().translation()
            wr0 = skel.getBodyNode(ss["carpal_body"]).getTransform().translation()
            arm_rest_dir = (wr0 - sh0)
            arm_rest_dir = arm_rest_dir / max(np.linalg.norm(arm_rest_dir), 1e-9)
            tgt_dir = target_rel / max(np.linalg.norm(target_rel), 1e-9)
            cross = np.cross(arm_rest_dir, tgt_dir)
            cn = np.linalg.norm(cross)
            ang = float(np.arctan2(cn, float(arm_rest_dir @ tgt_dir)))
            init_humerus_world_rotvec = (cross / cn * ang) if cn > 1e-9 else np.zeros(3)
            # Convert world rotvec to humerus body-local rotvec.
            hum_bn = skel.getBodyNode(ss["humerus_body"])
            R_hum_rest = np.asarray(hum_bn.getTransform().rotation())
            init_humerus_local = R_hum_rest.T @ init_humerus_world_rotvec

            # 4 DOFs: Humerus (3) + Ulna (1). Clavicle stays at rest.
            # Init Ulna from BVH anatomical elbow bend (angle between arm and
            # forearm BVH vectors). Sign negative for skel flex direction.
            bvh_arm_v = bvh_el - bvh_sh
            bvh_fa_v = bvh_wr - bvh_el
            an = np.linalg.norm(bvh_arm_v); fn = np.linalg.norm(bvh_fa_v)
            if an > 1e-6 and fn > 1e-6:
                cos_a = float(np.clip(np.dot(bvh_arm_v, bvh_fa_v) / (an * fn), -1, 1))
                init_ulna = -float(np.arccos(cos_a))
            else:
                init_ulna = 0.0
            # Warm-start: previous frame's solution but Ulna replaced by
            # anatomical init (escapes "straight arm" local minimum).
            if prev_x[side] is not None:
                x0 = prev_x[side].copy()
                x0[3] = init_ulna
            else:
                x0 = np.zeros(4)
                x0[0:3] = init_humerus_local
                x0[3] = init_ulna

            # Use Ulna JOINT world transform for elbow position (not ulna body
            # center which is mid-forearm). Likewise Humerus joint = shoulder
            # joint, Carpal joint = wrist joint.
            sh_joint_idx = next(j for j in range(skel.getNumJoints())
                                if skel.getJoint(j).getName() == ss["humerus_body"])
            el_joint_idx = next(j for j in range(skel.getNumJoints())
                                if skel.getJoint(j).getName() == ss["ulna_body"])
            wr_joint_idx = next(j for j in range(skel.getNumJoints())
                                if skel.getJoint(j).getName() == ss["carpal_body"])

            # Normalize elbow direction (skel bone lengths don't match BVH
            # proportions exactly — match direction not absolute position).
            tgt_elbow_dir = target_elbow_rel / max(np.linalg.norm(target_elbow_rel), 1e-9)
            tgt_wrist_dir = target_rel / max(np.linalg.norm(target_rel), 1e-9)
            tgt_wrist_len = np.linalg.norm(target_rel)

            def _cost(x):
                pose = base_pose.copy()
                pose[ss["hum_i"]:ss["hum_i"]+3] = x[0:3]
                pose[ss["ulna_i"]] = x[3]
                skel.setPositions(pose)
                sh = skel.getBodyNode(ss["humerus_body"]).getTransform().translation()
                el = skel.getBodyNode(ss["ulna_body"]).getTransform().translation()
                wr = skel.getBodyNode(ss["carpal_body"]).getTransform().translation()
                wr_rel = wr - sh
                el_rel = el - sh
                el_dir = el_rel / max(np.linalg.norm(el_rel), 1e-9)
                # Wrist position + elbow direction match. Strong elbow weight
                # to force humerus to point correctly, then ulna bends for
                # wrist (was tuning to "straight arm" with low elbow weight).
                return float(np.sum((wr_rel - target_rel) ** 2)
                             + 5.0 * np.sum((el_dir - tgt_elbow_dir) ** 2))

            res = minimize(_cost, x0, method="L-BFGS-B",
                           options={"maxiter": 100, "ftol": 1e-9})
            x = res.x
            prev_x[side] = x.copy()
            new_mocap[f, ss["hum_i"]:ss["hum_i"]+3] = x[0:3]
            new_mocap[f, ss["ulna_i"]] = x[3]

    # Inverse-T_net: convert mocap rotvec → BVH channel value.
    # mocap = T_net = parent_world_bvh(0) @ R_local_bvh(f) @ R_local_bvh(0).T @ parent_world_bvh(0).T
    # With R_local_bvh(0) = identity (we control), R_local_bvh(f) =
    # parent_world_bvh(0).T @ mat(mocap rotvec) @ parent_world_bvh(0).
    side_skel_to_bvh = {
        "L": {"clav": ("LeftShoulder", "clav_i", 3),
              "hum":  ("LeftArm",      "hum_i",  3),
              "ulna": ("LeftForeArm",  "ulna_i", 1)},
        "R": {"clav": ("RightShoulder", "clav_i", 3),
              "hum":  ("RightArm",      "hum_i",  3),
              "ulna": ("RightForeArm",  "ulna_i", 1)},
    }
    for f in range(n_frames):
        row = rows[f]
        for side in ("L", "R"):
            ss = sides_skel[0 if side == "L" else 1]
            for kind, (bvh_name, dof_attr, nd) in side_skel_to_bvh[side].items():
                idx = ss[dof_attr]
                if (side, bvh_name) not in chan_cols: continue
                c0 = chan_cols[(side, bvh_name)]
                order = chan_orders[(side, bvh_name)]
                P = parent_world_0[(side, bvh_name)]
                if nd == 3:
                    rv = new_mocap[f, idx:idx+3]
                    R_target = R.from_rotvec(rv).as_matrix()
                    R_local = P.T @ R_target @ P
                    eul = R.from_matrix(R_local).as_euler(order, degrees=True)
                    rot_offs = [k for k, c in enumerate(joints[find_joint(joints, bvh_name)]["channels"])
                                if c.lower().endswith("rotation")]
                    for k, off in enumerate(rot_offs):
                        row[c0 + off] = float(eul[k])
                else:
                    # Ulna: 1-DoF, axis from skel XML. Build matrix from axis*angle.
                    j_idx_dart = next(j for j in range(skel.getNumJoints())
                                       if skel.getJoint(j).getName() == ss["ulna_body"])
                    axis = np.asarray(skel.getJoint(j_idx_dart).getAxis(), dtype=np.float64)
                    axis = axis / max(np.linalg.norm(axis), 1e-9)
                    ang = float(new_mocap[f, idx])
                    R_target = R.from_rotvec(axis * ang).as_matrix()
                    R_local = P.T @ R_target @ P
                    eul = R.from_matrix(R_local).as_euler(order, degrees=True)
                    rot_offs = [k for k, c in enumerate(joints[find_joint(joints, bvh_name)]["channels"])
                                if c.lower().endswith("rotation")]
                    for k, off in enumerate(rot_offs):
                        row[c0 + off] = float(eul[k])

    out_lines = lines[:mi] + motion_header
    for row in rows[:n_frames]:
        out_lines.append(" ".join(f"{v:.6f}" for v in row))
    # If we limited frames, only write those (truncate motion).
    if args.limit_frames > 0:
        # Update Frames: header.
        for k, ln in enumerate(out_lines):
            if ln.strip().startswith("Frames:"):
                out_lines[k] = f"Frames: {n_frames}"
                break
    with open(args.bvh_out, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Wrote {args.bvh_out}")


if __name__ == "__main__":
    main()
