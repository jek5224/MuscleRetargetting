"""Direct skel mocap_refs bake. Bypasses MyBVH conversion entirely.

For each frame, computes desired skel DOF values via DART FK:
  - Root (trunk + spine vertebrae): from input BVH MyBVH-derived mocap.
  - Clavicle: scapulohumeral share of humerus motion.
  - Humerus: rotation_from_to(chain_humerus_dir, bvh_humerus_dir), composed
    with clavicle so combined body_world rotation = R_motion @ body_rest.
  - Ulna: signed angle around current ulna joint axis to match bvh forearm dir.
  - Radius: 0 (or BVH twist component — TODO).
  - Carpal: 0 (or BVH hand orientation — TODO).

Saves mocap_refs as .npy alongside output BVH. Viewer loads .npy directly,
skipping MyBVH conversion → zero conversion error.
"""
import argparse
import os
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
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    a_perp = a - (a @ axis) * axis
    b_perp = b - (b @ axis) * axis
    if np.linalg.norm(a_perp) < 1e-9 or np.linalg.norm(b_perp) < 1e-9:
        return 0.0
    a_perp /= np.linalg.norm(a_perp); b_perp /= np.linalg.norm(b_perp)
    cos = float(np.clip(a_perp @ b_perp, -1.0, 1.0))
    sin = float(np.cross(a_perp, b_perp) @ axis)
    return float(np.arctan2(sin, cos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="bvh_in", required=True,
                    help="Input BVH (post sternum bake).")
    ap.add_argument("--orig-bvh", required=True,
                    help="Normalized BVH for BVH-FK direction targets.")
    ap.add_argument("--out-bvh", required=True,
                    help="Output BVH (copied from input, just for compatibility).")
    ap.add_argument("--out-npy", required=True,
                    help="Output mocap_refs.npy path.")
    ap.add_argument("--skel-xml", default="data/zygote_skel.xml")
    ap.add_argument("--scapulohumeral", type=float, default=0.27)
    args = ap.parse_args()

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

    n_frames = mocap_in.shape[0]
    n_dofs = skel.getNumDofs()
    print(f"  Frames: {n_frames}, DOFs: {n_dofs}")

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
        sd = {
            "L": L_or_R,
            "bvh": {
                "sh": find_joint(ojoints, f"{prefix}Shoulder"),
                "arm": find_joint(ojoints, f"{prefix}Arm"),
                "fa": find_joint(ojoints, f"{prefix}ForeArm"),
                "hd": find_joint(ojoints, f"{prefix}Hand"),
            },
            "skel": {
                "clav": f"{L_or_R}_Clavicle0",
                "hum": f"{L_or_R}_Humerus0",
                "ulna": f"{L_or_R}_Ulna0",
                "rad": f"{L_or_R}_Radius0",
                "carp": f"{L_or_R}_Carpal0",
            },
        }
        for k in ("clav", "hum", "ulna", "rad", "carp"):
            idx, nd = dof(sd["skel"][k])
            sd[f"dof_{k}"] = (idx, nd)
        sides.append(sd)

    arm_dof_set = []
    for sd in sides:
        for k in ("clav", "hum", "ulna", "rad", "carp"):
            idx, nd = sd[f"dof_{k}"]
            if idx is not None:
                arm_dof_set.extend(range(idx, idx + nd))

    share = float(args.scapulohumeral)
    out_mocap = mocap_in.copy()

    def compute_hum_target(sd, pose_in, frame_idx):
        """Two-vector basis humerus body world target for given frame.
        Sets skel to pose_in with arm DOFs zero, reads d_chain_hum,
        builds target from d_bvh_hum and d_bvh_fore."""
        p = pose_in.copy()
        for di in arm_dof_set:
            if di < len(p): p[di] = 0.0
        skel.setPositions(p)
        Tf_o = bvh_fk_world_full(ojoints, orows[frame_idx] if frame_idx < len(orows) else orows[-1], on2c)
        d_bvh_h = Tf_o[sd["bvh"]["fa"]][:3, 3] - Tf_o[sd["bvh"]["arm"]][:3, 3]
        d_bvh_h /= max(np.linalg.norm(d_bvh_h), 1e-12)
        d_bvh_f = Tf_o[sd["bvh"]["hd"]][:3, 3] - Tf_o[sd["bvh"]["fa"]][:3, 3]
        d_bvh_f /= max(np.linalg.norm(d_bvh_f), 1e-12)
        j_h = skel.getJoint(sd["skel"]["hum"])
        j_u = None
        for jj in range(skel.getNumJoints()):
            if skel.getJoint(jj).getName() == sd["skel"]["ulna"]:
                j_u = skel.getJoint(jj); break
        T_p2j_u = np.asarray(j_u.getTransformFromParentBodyNode().rotation())
        ax_lj = np.array(j_u.getAxis(), dtype=np.float64, copy=True)
        ax_lj /= max(np.linalg.norm(ax_lj), 1e-12)
        trans_p2j_u = np.asarray(j_u.getTransformFromParentBodyNode().translation())
        X_loc = trans_p2j_u / max(np.linalg.norm(trans_p2j_u), 1e-12)
        Z_loc = T_p2j_u @ ax_lj
        hum_init = np.asarray(skel.getBodyNode(sd["skel"]["hum"]).getTransform().rotation())
        n_pl = np.cross(d_bvh_h, d_bvh_f)
        npn = np.linalg.norm(n_pl)
        if npn < 1e-6:
            Rm = rotation_from_to(np.asarray(j_h.getTransformFromChildBodyNode().rotation()) @ X_loc, d_bvh_h)
            return Rm @ hum_init, d_bvh_h, d_bvh_f
        Zt = n_pl / npn
        if Zt @ (hum_init @ Z_loc) < 0:
            Zt = -Zt
        Yt = np.cross(Zt, d_bvh_h)
        Yn = np.linalg.norm(Yt)
        if Yn < 1e-9:
            Rm = rotation_from_to(hum_init @ X_loc, d_bvh_h)
            return Rm @ hum_init, d_bvh_h, d_bvh_f
        Yt /= Yn
        Zt = np.cross(d_bvh_h, Yt); Zt /= max(np.linalg.norm(Zt), 1e-12)
        Mtw = np.column_stack([d_bvh_h, Yt, Zt])
        bx = X_loc / np.linalg.norm(X_loc)
        bz = Z_loc - (Z_loc @ bx) * bx; bz /= max(np.linalg.norm(bz), 1e-12)
        by = np.cross(bz, bx)
        Mbl = np.column_stack([bx, by, bz])
        return Mtw @ Mbl.T, d_bvh_h, d_bvh_f

    # Baseline f=0: hum_body_world_target at rest motion (T-pose for LaFAN).
    hum_target_0 = {}
    for sd in sides:
        ht0, _, _ = compute_hum_target(sd, mocap_in[0], 0)
        hum_target_0[sd["L"]] = ht0

    progress = max(1, n_frames // 20)
    for f in range(n_frames):
        if f % progress == 0:
            print(f"    frame {f}/{n_frames}")
        pose = mocap_in[f].copy()
        for di in arm_dof_set:
            if di < len(pose): pose[di] = 0.0
        skel.setPositions(pose)
        Tf_orig = bvh_fk_world_full(ojoints, orows[f] if f < len(orows) else orows[-1], on2c)

        for sd in sides:
            sh_ji = sd["bvh"]["arm"]; fa_ji = sd["bvh"]["fa"]; hd_ji = sd["bvh"]["hd"]
            d_bvh_hum = Tf_orig[fa_ji][:3, 3] - Tf_orig[sh_ji][:3, 3]
            d_bvh_hum /= max(np.linalg.norm(d_bvh_hum), 1e-12)
            d_bvh_fore = Tf_orig[hd_ji][:3, 3] - Tf_orig[fa_ji][:3, 3]
            d_bvh_fore /= max(np.linalg.norm(d_bvh_fore), 1e-12)

            # Chain-only humerus dir (arm DOFs zero).
            sh_w = joint_world(sd["skel"]["hum"])
            el_w = joint_world(sd["skel"]["ulna"])
            d_chain_hum = el_w - sh_w
            d_chain_hum /= max(np.linalg.norm(d_chain_hum), 1e-12)

            # Two-vector humerus orientation: bone direction = d_bvh_hum,
            # ulna flex axis world = normal of (d_bvh_hum, d_bvh_fore) plane.
            # This constrains humerus twist so ulna_angle can reach d_bvh_fore.
            j_hum = skel.getJoint(sd["skel"]["hum"])
            j_ulna_obj = None
            for jj in range(skel.getNumJoints()):
                if skel.getJoint(jj).getName() == sd["skel"]["ulna"]:
                    j_ulna_obj = skel.getJoint(jj); break
            T_p2j_hum = np.asarray(j_hum.getTransformFromParentBodyNode().rotation())
            T_c2j_hum = np.asarray(j_hum.getTransformFromChildBodyNode().rotation())
            T_p2j_ulna = np.asarray(j_ulna_obj.getTransformFromParentBodyNode().rotation())
            ulna_axis_local_joint = np.array(j_ulna_obj.getAxis(), dtype=np.float64, copy=True)
            ulna_axis_local_joint /= max(np.linalg.norm(ulna_axis_local_joint), 1e-12)

            # X_local_hum: humerus body-local direction toward ulna joint
            # origin (bone direction = shoulder→elbow in body frame).
            T_p2j_ulna_trans = np.asarray(j_ulna_obj.getTransformFromParentBodyNode().translation())
            X_local_hum = T_p2j_ulna_trans / max(np.linalg.norm(T_p2j_ulna_trans), 1e-12)
            # Z_local_hum: ulna flex axis in humerus body frame.
            Z_local_hum = T_p2j_ulna @ ulna_axis_local_joint

            # Target world directions for humerus body local X and Z axes.
            X_target_w = d_bvh_hum.copy()
            n_plane = np.cross(d_bvh_hum, d_bvh_fore)
            n_plane_norm = np.linalg.norm(n_plane)
            if n_plane_norm < 1e-6:
                # Forearm collinear with humerus → degenerate. Use shortest rot.
                R_motion_world = rotation_from_to(d_chain_hum, d_bvh_hum)
                hum_body_world_init = np.asarray(skel.getBodyNode(sd["skel"]["hum"]).getTransform().rotation())
                hum_body_world_target = R_motion_world @ hum_body_world_init
            else:
                Z_target_w = n_plane / n_plane_norm
                # Pick sign so Z_target_w consistent with Z_local_hum at rest
                # (avoid 180° flip).
                hum_body_world_init = np.asarray(skel.getBodyNode(sd["skel"]["hum"]).getTransform().rotation())
                Z_local_w_init = hum_body_world_init @ Z_local_hum
                if Z_target_w @ Z_local_w_init < 0:
                    Z_target_w = -Z_target_w
                # Build hum_body_world_target via orthonormal basis from
                # (X_target_w, Z_target_w → orthogonalized).
                Y_target_w = np.cross(Z_target_w, X_target_w)
                Yn = np.linalg.norm(Y_target_w)
                if Yn < 1e-9:
                    R_motion_world = rotation_from_to(d_chain_hum, d_bvh_hum)
                    hum_body_world_target = R_motion_world @ hum_body_world_init
                else:
                    Y_target_w /= Yn
                    Z_target_w = np.cross(X_target_w, Y_target_w)
                    Z_target_w /= max(np.linalg.norm(Z_target_w), 1e-12)
                    # body_world maps body-local axes to world axes.
                    M_target_w = np.column_stack([X_target_w, Y_target_w, Z_target_w])
                    M_body_local = np.column_stack([X_local_hum,
                                                   np.cross(Z_local_hum, X_local_hum) / max(np.linalg.norm(np.cross(Z_local_hum, X_local_hum)), 1e-12),
                                                   Z_local_hum])
                    # Re-orthonormalize body local basis.
                    bx = M_body_local[:, 0]; bz = M_body_local[:, 2]
                    bx /= np.linalg.norm(bx)
                    bz = bz - (bz @ bx) * bx; bz /= max(np.linalg.norm(bz), 1e-12)
                    by = np.cross(bz, bx)
                    M_body_local = np.column_stack([bx, by, bz])
                    hum_body_world_target = M_target_w @ M_body_local.T

            # Clavicle share applies to MOTION delta only (target relative
            # to f=0 target), not to BVH-rest vs skel-rest offset. At f=0
            # R_motion=I → clav stays at rest → no spread artifact.
            R_motion = hum_body_world_target @ hum_target_0[sd["L"]].T
            rv_motion = R.from_matrix(R_motion).as_rotvec()
            R_clav_world = R.from_rotvec(rv_motion * share).as_matrix()

            j_clav = skel.getJoint(sd["skel"]["clav"])
            clav_parent_world = np.asarray(skel.getBodyNode(sd["skel"]["clav"]).getParentBodyNode().getTransform().rotation())
            scap_world_init = np.asarray(skel.getBodyNode(sd["skel"]["hum"]).getParentBodyNode().getTransform().rotation())
            J_clav_world = clav_parent_world @ np.asarray(j_clav.getTransformFromParentBodyNode().rotation())
            J_hum_world = scap_world_init @ T_p2j_hum

            R_clav_local = J_clav_world.T @ R_clav_world @ J_clav_world

            # Humerus local DOF: hum_body_world_after = par_after @ T_p2j @ expmap(rv) @ T_c2j.T
            # par_after = R_clav_world @ scap_world_init (scapula rigid with clavicle).
            # → expmap(rv) = T_p2j.T @ par_after.T @ hum_body_world_target @ T_c2j
            par_after = R_clav_world @ scap_world_init
            expmap_rv_hum = T_p2j_hum.T @ par_after.T @ hum_body_world_target @ T_c2j_hum
            R_hum_local = expmap_rv_hum

            clav_idx, _ = sd["dof_clav"]; hum_idx, _ = sd["dof_hum"]
            pose[clav_idx:clav_idx + 3] = R.from_matrix(R_clav_local).as_rotvec()
            pose[hum_idx:hum_idx + 3] = R.from_matrix(R_hum_local).as_rotvec()
            skel.setPositions(pose)

            # Ulna: signed angle to align forearm dir with bvh_fore.
            j_ulna = None
            for j in range(skel.getNumJoints()):
                if skel.getJoint(j).getName() == sd["skel"]["ulna"]:
                    j_ulna = skel.getJoint(j); break
            ulna_axis_local = np.asarray(j_ulna.getAxis(), dtype=np.float64)
            hum_world = np.asarray(skel.getBodyNode(sd["skel"]["hum"]).getTransform().rotation())
            T_p2j_ulna = np.asarray(j_ulna.getTransformFromParentBodyNode().rotation())
            ulna_axis_w = hum_world @ T_p2j_ulna @ ulna_axis_local
            ulna_axis_w /= max(np.linalg.norm(ulna_axis_w), 1e-12)
            el_w2 = joint_world(sd["skel"]["ulna"])
            wr_w2 = joint_world(sd["skel"]["carp"])
            d_chain_fore = wr_w2 - el_w2
            d_chain_fore /= max(np.linalg.norm(d_chain_fore), 1e-12)
            ulna_angle = signed_angle_about_axis(d_chain_fore, d_bvh_fore, ulna_axis_w)
            ulna_idx, _ = sd["dof_ulna"]
            pose[ulna_idx] = ulna_angle
            skel.setPositions(pose)

            # Radius + Carpal: 0 (TODO: BVH hand twist + orientation).
            rad_idx, _ = sd["dof_rad"]
            pose[rad_idx] = 0.0
            carp_idx, _ = sd["dof_carp"]
            pose[carp_idx:carp_idx + 3] = 0.0
            skel.setPositions(pose)

        out_mocap[f] = pose

    # Save .npy alongside output BVH.
    np.save(args.out_npy, out_mocap)
    print(f"Wrote {args.out_npy}")

    # Also copy input BVH to output for compatibility (channels unchanged).
    with open(args.bvh_in) as fin, open(args.out_bvh, "w") as fout:
        fout.write(fin.read())
    print(f"Wrote {args.out_bvh}")


if __name__ == "__main__":
    main()
