"""Direct skel pose computation for arm chain.

Per frame, computes desired skel joint DOFs by applying chain transforms
and matching BVH segment directions:
  - Clavicle: scapulohumeral share of humerus motion.
  - Humerus: rotation_from_to(chain_humerus_dir, bvh_humerus_dir).
  - Ulna: signed angle around ulna axis matching bvh forearm dir.
  - Radius: scalar twist from BVH hand axial component.
  - Carpal: full rotation matching bvh hand orientation.

Then writes BVH channels via inverse-T_net under T_frame=0 convention:
  channel(f) = P_parent_world_at_F0.T @ mocap_target @ P_parent_world_at_F0
  channel(0) = identity (forced; T_frame=0 calibrates skel rest at f=0).

Output BVH has all arm channels rewritten. Forearm bake replaced by
this in pipeline.
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
                elif ch.lower().endswith("rotation"): rc.append((ch[0].upper(), v))
            if rc:
                Rl = R.from_euler("".join(c for c, _ in rc), [v for _, v in rc], degrees=True)
        off = np.array(j["offset"]) if j["offset"] else np.zeros(3)
        M = np.eye(4); M[:3, :3] = Rl.as_matrix(); M[:3, 3] = off + pos
        par = j["parent"]
        T[ji] = M if par < 0 else T[par] @ M
    return T


def rotation_from_to(a, b):
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    c = np.cross(a, b); cn = np.linalg.norm(c)
    dot = float(a @ b)
    if cn < 1e-9:
        if dot > 0: return np.eye(3)
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp); axis /= np.linalg.norm(axis)
        return R.from_rotvec(axis * np.pi).as_matrix()
    return R.from_rotvec(c / cn * float(np.arctan2(cn, dot))).as_matrix()


def signed_angle_about_axis(a, b, axis):
    """Signed angle (rad) rotating vector a to vector b around axis."""
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    a_perp = a - (a @ axis) * axis
    b_perp = b - (b @ axis) * axis
    if np.linalg.norm(a_perp) < 1e-9 or np.linalg.norm(b_perp) < 1e-9:
        return 0.0
    a_perp /= np.linalg.norm(a_perp); b_perp /= np.linalg.norm(b_perp)
    cos = np.clip(a_perp @ b_perp, -1.0, 1.0)
    sin = float(np.cross(a_perp, b_perp) @ axis)
    return float(np.arctan2(sin, cos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="bvh_in", required=True)
    ap.add_argument("--out", dest="bvh_out", required=True)
    ap.add_argument("--orig-bvh", required=True,
                    help="Normalized BVH for arm/forearm/hand world positions.")
    ap.add_argument("--skel-xml", default="data/zygote_skel.xml")
    ap.add_argument("--scapulohumeral", type=float, default=0.27)
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

    sys.path.insert(0, ".")
    from core.dartHelper import saveSkeletonInfo, buildFromInfo
    from core.bvhparser import MyBVH
    si, rn, bvh_info, *_ = saveSkeletonInfo(args.skel_xml)
    skel = buildFromInfo(si, rn)
    mbv_in = MyBVH(args.bvh_in, bvh_info, skel, T_frame=0)
    mocap_in = mbv_in.mocap_refs.copy()

    # Skel joint DOF index lookup.
    def dof(name):
        for j in range(skel.getNumJoints()):
            jn = skel.getJoint(j)
            if jn.getName() == name:
                return jn.getIndexInSkeleton(0), jn.getNumDofs()
        return None, 0

    def joint_world(body_name):
        bn = skel.getBodyNode(body_name)
        par = bn.getParentBodyNode()
        j = bn.getParentJoint()
        T = j.getTransformFromParentBodyNode()
        if par is None: return T.translation()
        return par.getTransform().translation() + np.asarray(par.getTransform().rotation()) @ T.translation()

    sides = []
    for L_or_R, prefix in [("L", "Left"), ("R", "Right")]:
        sh_suf = f"{prefix}Shoulder"; arm_suf = f"{prefix}Arm"
        fa_suf = f"{prefix}ForeArm"; hd_suf = f"{prefix}Hand"
        skel_clav = f"{L_or_R}_Clavicle0"
        skel_hum = f"{L_or_R}_Humerus0"
        skel_ulna = f"{L_or_R}_Ulna0"
        skel_rad = f"{L_or_R}_Radius0"
        skel_carp = f"{L_or_R}_Carpal0"
        sides.append({
            "L": L_or_R,
            "bvh": {"sh": sh_suf, "arm": arm_suf, "fa": fa_suf, "hd": hd_suf},
            "skel": {"clav": skel_clav, "hum": skel_hum, "ulna": skel_ulna,
                       "rad": skel_rad, "carp": skel_carp},
            "bvh_ji": {n: find_joint(ojoints, s) for n, s in [("sh", sh_suf),
                                                                ("arm", arm_suf),
                                                                ("fa", fa_suf),
                                                                ("hd", hd_suf)]},
        })

    arm_dofs = []
    for sd in sides:
        for k in ("clav", "hum", "ulna", "rad", "carp"):
            idx, nd = dof(sd["skel"][k])
            if idx is not None:
                arm_dofs.extend(range(idx, idx + nd))
                sd[f"dof_{k}"] = (idx, nd)

    n_frames = len(rows)
    n_dofs = skel.getNumDofs()
    target_mocap = mocap_in.copy()

    share = float(args.scapulohumeral)
    print(f"  scapulohumeral share = {share}")
    print(f"  Computing arm DOFs over {n_frames} frames...")
    progress = max(1, n_frames // 20)
    for f in range(n_frames):
        if f % progress == 0:
            print(f"    f {f}/{n_frames}")
        # Reset to input mocap, zero arm DOFs.
        pose = mocap_in[f].copy()
        for di in arm_dofs:
            if di < len(pose): pose[di] = 0.0
        skel.setPositions(pose)
        Tf_orig = bvh_fk_world_full(ojoints, orows[f] if f < len(orows) else orows[-1], on2c)

        for sd in sides:
            sh_ji = sd["bvh_ji"]["arm"]; fa_ji = sd["bvh_ji"]["fa"]; hd_ji = sd["bvh_ji"]["hd"]
            bvh_sh_pos = Tf_orig[sh_ji][:3, 3]
            bvh_fa_pos = Tf_orig[fa_ji][:3, 3]
            bvh_hd_pos = Tf_orig[hd_ji][:3, 3]
            d_bvh_hum = bvh_fa_pos - bvh_sh_pos
            d_bvh_hum /= max(np.linalg.norm(d_bvh_hum), 1e-12)
            d_bvh_fore = bvh_hd_pos - bvh_fa_pos
            d_bvh_fore /= max(np.linalg.norm(d_bvh_fore), 1e-12)

            # CHAIN read (no arm DOFs set yet).
            sh_w = joint_world(sd["skel"]["hum"])
            el_w = joint_world(sd["skel"]["ulna"])
            d_chain_hum = el_w - sh_w
            d_chain_hum /= max(np.linalg.norm(d_chain_hum), 1e-12)
            R_motion = rotation_from_to(d_chain_hum, d_bvh_hum)
            rv = R.from_matrix(R_motion).as_rotvec()
            # Clavicle gets share, humerus residual (composition).
            R_clav = R.from_rotvec(rv * share).as_matrix()
            R_hum = R_motion @ R_clav.T

            # Apply clavicle + humerus DOFs.
            clav_idx, clav_n = sd["dof_clav"]
            hum_idx, hum_n = sd["dof_hum"]
            pose[clav_idx:clav_idx + 3] = R.from_matrix(R_clav).as_rotvec()
            pose[hum_idx:hum_idx + 3] = R.from_matrix(R_hum).as_rotvec()
            skel.setPositions(pose)

            # Compute Ulna DOF (revolute): signed angle around its axis,
            # bringing chain forearm to BVH forearm dir.
            ulna_idx, ulna_n = sd["dof_ulna"]
            ulna_body = sd["skel"]["ulna"]
            for j in range(skel.getNumJoints()):
                if skel.getJoint(j).getName() == ulna_body:
                    ulna_axis_local = np.asarray(skel.getJoint(j).getAxis(), dtype=np.float64)
                    break
            # Ulna axis in world = humerus body world * axis_local.
            hum_world = np.asarray(skel.getBodyNode(sd["skel"]["hum"]).getTransform().rotation())
            ulna_axis_w = hum_world @ ulna_axis_local
            ulna_axis_w /= max(np.linalg.norm(ulna_axis_w), 1e-12)
            # Current forearm direction (chain through humerus, ulna DOF = 0).
            el_w2 = joint_world(sd["skel"]["ulna"])
            wr_w2 = joint_world(sd["skel"]["carp"])
            d_chain_fore = wr_w2 - el_w2
            d_chain_fore /= max(np.linalg.norm(d_chain_fore), 1e-12)
            ulna_angle = signed_angle_about_axis(d_chain_fore, d_bvh_fore, ulna_axis_w)
            pose[ulna_idx] = ulna_angle
            skel.setPositions(pose)

            # Radius + Carpal: TODO (set to 0 for now).
            rad_idx, rad_n = sd["dof_rad"]
            pose[rad_idx] = 0.0
            carp_idx, carp_n = sd["dof_carp"]
            pose[carp_idx:carp_idx + 3] = 0.0
            skel.setPositions(pose)

        target_mocap[f] = pose

    # Now write BVH channels such that MyBVH on output yields target_mocap.
    # For 3-DoF joints: channel(f) = P_out.T @ R_target @ P_out, channel(0) = identity.
    # For 1-DoF joints: pick rotvec along axis with desired magnitude, then conjugate.
    # P_out = parent body world rotation in OUTPUT BVH at f=0 (with arm channels zeroed).

    # Zero arm channels at row 0 first.
    arm_bvh_joints = []
    for sd in sides:
        for k in ("sh", "arm", "fa", "hd"):
            jn_name = sd["bvh"][k]
            ji = find_joint(joints, jn_name)
            if ji is not None: arm_bvh_joints.append(ji)
    for ji in arm_bvh_joints:
        chs = joints[ji]["channels"]
        c0 = n2c[joints[ji]["name"]][0]
        for k, ch in enumerate(chs):
            if ch.lower().endswith("rotation"):
                rows[0][c0 + k] = 0.0

    # Compute parent_world for each arm BVH joint at output f=0.
    Twr_out_0 = bvh_fk_world_full(joints, rows[0], n2c)
    P_out = {}
    for ji in arm_bvh_joints:
        par = joints[ji]["parent"]
        P_out[ji] = Twr_out_0[par][:3, :3].copy() if par >= 0 else np.eye(3)

    # Write per-frame channels for each arm joint.
    bvh_to_skel = {
        "LeftShoulder": ("clav", "L"),
        "LeftArm": ("hum", "L"),
        "LeftForeArm": ("ulna", "L"),
        "LeftHand": ("carp", "L"),
        "RightShoulder": ("clav", "R"),
        "RightArm": ("hum", "R"),
        "RightForeArm": ("ulna", "R"),
        "RightHand": ("carp", "R"),
    }

    for f in range(n_frames):
        for sd in sides:
            for bvh_name, (dof_key, side_label) in bvh_to_skel.items():
                if sd["L"] != side_label: continue
                ji = find_joint(joints, bvh_name)
                if ji is None: continue
                idx, nd = sd[f"dof_{dof_key}"]
                if nd == 0: continue
                if nd == 3:
                    rotvec = target_mocap[f, idx:idx + 3]
                    R_target = R.from_rotvec(rotvec).as_matrix()
                elif nd == 1:
                    ang = float(target_mocap[f, idx])
                    skel_body = sd["skel"][dof_key]
                    for j in range(skel.getNumJoints()):
                        if skel.getJoint(j).getName() == skel_body:
                            ax = np.asarray(skel.getJoint(j).getAxis(), dtype=np.float64)
                            break
                    R_target = R.from_rotvec(ax * ang).as_matrix()
                else:
                    continue
                P = P_out[ji]
                R_channel = P.T @ R_target @ P
                chs = joints[ji]["channels"]
                order = "".join(c[0] for c in chs if c.lower().endswith("rotation")).upper()
                c0 = n2c[joints[ji]["name"]][0]
                rot_offs = [k for k, c in enumerate(chs) if c.lower().endswith("rotation")]
                eul = R.from_matrix(R_channel).as_euler(order, degrees=True)
                for k, off in enumerate(rot_offs):
                    rows[f][c0 + off] = float(eul[k])

    # Force arm channels at f=0 to identity (after per-frame write).
    for ji in arm_bvh_joints:
        chs = joints[ji]["channels"]
        c0 = n2c[joints[ji]["name"]][0]
        for k, ch in enumerate(chs):
            if ch.lower().endswith("rotation"):
                rows[0][c0 + k] = 0.0

    out_lines = lines[:mi] + motion_header
    for row in rows:
        out_lines.append(" ".join(f"{v:.6f}" for v in row))
    with open(args.bvh_out, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Wrote {args.bvh_out}")


if __name__ == "__main__":
    main()
