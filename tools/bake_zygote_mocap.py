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
    ap.add_argument("--palm-down-l-deg", type=float, default=90.0,
                    help="L_Radius angle (deg) — pronation to make palm face "
                         "down at T-pose. Applied per-frame as constant offset.")
    ap.add_argument("--palm-down-r-deg", type=float, default=-90.0,
                    help="R_Radius angle (deg) — symmetric for right arm.")
    ap.add_argument("--clav-scale", type=float, default=0.4,
                    help="Scale BVH-driven clavicle rotation (slerp from "
                         "identity). <1 restricts clavicle motion.")
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

    # Zero hum/ulna/rad/carp only. KEEP clavicle DOF from MyBVH-converted
    # input (BVH LeftShoulder → L_Clavicle anatomical mapping). Letting
    # clavicle ride bvh source avoids both T-pose over-spread (Inman full
    # delta) and N-pose collapse (Inman zero at f=0).
    arm_dof_set = []
    for sd in sides:
        for k in ("hum", "ulna", "rad", "carp"):
            idx, nd = sd[f"dof_{k}"]
            if idx is not None:
                arm_dof_set.extend(range(idx, idx + nd))

    share = float(args.scapulohumeral)
    clav_scale = float(args.clav_scale)
    # Scale BVH-driven clavicle DOFs (rotvec slerp from identity by factor).
    # Linear rotvec scaling = exact slerp when axis fixed, approximate else.
    if abs(clav_scale - 1.0) > 1e-6:
        for sd in sides:
            ci, nd = sd["dof_clav"]
            if ci is not None and nd == 3:
                mocap_in[:, ci:ci+3] *= clav_scale
    out_mocap = mocap_in.copy()
    prev_Z_target = {sd["L"]: None for sd in sides}

    # Cache skel-rest orthonormal humerus-body basis: bone (X), natural
    # elbow bend direction in plane perpendicular to bone (bend), normal
    # = cross(X, bend). Bend derived from skel rest forearm direction.
    skel.setPositions(np.zeros(skel.getNumDofs()))
    rest_basis = {}
    for sd in sides:
        for jj in range(skel.getNumJoints()):
            if skel.getJoint(jj).getName() == sd["skel"]["ulna"]:
                j_u = skel.getJoint(jj)
                T_p2j_u_trans = np.asarray(j_u.getTransformFromParentBodyNode().translation())
                break
        X_loc = T_p2j_u_trans / max(np.linalg.norm(T_p2j_u_trans), 1e-12)
        hum_rest_w = np.asarray(skel.getBodyNode(sd["skel"]["hum"]).getTransform().rotation())
        el_r = joint_world(sd["skel"]["ulna"])
        wr_r = joint_world(sd["skel"]["carp"])
        d_fore_w_rest = wr_r - el_r; d_fore_w_rest /= max(np.linalg.norm(d_fore_w_rest), 1e-12)
        d_fore_loc = hum_rest_w.T @ d_fore_w_rest
        # bend_loc = perpendicular component of d_fore_loc against X_loc.
        bend_loc = d_fore_loc - (d_fore_loc @ X_loc) * X_loc
        bend_loc /= max(np.linalg.norm(bend_loc), 1e-12)
        N_loc = np.cross(X_loc, bend_loc)
        N_loc /= max(np.linalg.norm(N_loc), 1e-12)
        rest_basis[sd["L"]] = (X_loc, bend_loc, N_loc)

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

            # Two-vector humerus orientation:
            #   body-local bone direction → d_bvh_hum (shoulder→elbow).
            #   body-local NATURAL forearm bend direction → BVH forearm bend.
            # This makes the natural skel rest elbow bend (~25° in d_fore_local)
            # follow the BVH bend direction. Ulna angle then corrects only
            # the magnitude difference, never flips the elbow backward.
            j_hum = skel.getJoint(sd["skel"]["hum"])
            T_p2j_hum = np.asarray(j_hum.getTransformFromParentBodyNode().rotation())
            T_c2j_hum = np.asarray(j_hum.getTransformFromChildBodyNode().rotation())

            X_local_hum, bend_local_axis, N_local_hum = rest_basis[sd["L"]]
            hum_body_world_init = np.asarray(skel.getBodyNode(sd["skel"]["hum"]).getTransform().rotation())

            # Build target world basis. Orthogonalize d_bvh_fore against
            # d_bvh_hum to get the bend-direction perpendicular component.
            X_target_w = d_bvh_hum.copy()
            bend_target = d_bvh_fore - (d_bvh_fore @ d_bvh_hum) * d_bvh_hum
            bend_norm = np.linalg.norm(bend_target)
            if bend_norm < 0.17:
                # Arm near-straight: ill-defined bend direction. Use previous
                # frame's bend axis if available, else fall back to shortest
                # rotation (twist undefined → ulna_angle picks small value).
                R_motion_world = rotation_from_to(d_chain_hum, d_bvh_hum)
                hum_body_world_target = R_motion_world @ hum_body_world_init
            else:
                bend_target /= bend_norm
                N_target_w = np.cross(X_target_w, bend_target)
                N_target_w /= max(np.linalg.norm(N_target_w), 1e-12)
                # Continuity: anchor bend sign to previous frame's N_target.
                if prev_Z_target[sd["L"]] is not None and N_target_w @ prev_Z_target[sd["L"]] < 0:
                    N_target_w = -N_target_w
                    bend_target = -bend_target
                M_target_w = np.column_stack([X_target_w, bend_target, N_target_w])
                M_body_local = np.column_stack([X_local_hum, bend_local_axis, N_local_hum])
                hum_body_world_target = M_target_w @ M_body_local.T
                prev_Z_target[sd["L"]] = N_target_w.copy()

            # Clavicle stays at BVH-converted MyBVH value (loaded from
            # mocap_in via L_Clavicle bvh=LeftShoulder mapping). Humerus
            # absorbs all remaining delta needed for d_bvh_hum direction.
            # Scapula stays rigid with current clavicle pose.
            scap_world_init = np.asarray(skel.getBodyNode(sd["skel"]["hum"]).getParentBodyNode().getTransform().rotation())
            J_hum_world = scap_world_init @ T_p2j_hum

            # par_after = current scapula world (already reflects BVH clav).
            par_after = scap_world_init
            expmap_rv_hum = T_p2j_hum.T @ par_after.T @ hum_body_world_target @ T_c2j_hum
            R_hum_local = expmap_rv_hum

            # Clavicle DOF preserved from mocap_in (not overwritten).
            hum_idx, _ = sd["dof_hum"]
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

            # Radius: constant pronation offset (palm-down at T-pose).
            # TODO: per-frame BVH hand twist component.
            rad_idx, _ = sd["dof_rad"]
            palm_deg = args.palm_down_l_deg if sd["L"] == "L" else args.palm_down_r_deg
            pose[rad_idx] = float(np.deg2rad(palm_deg))
            # Carpal: 0 (TODO: BVH hand orientation).
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
