"""Retarget BVH forearm chain (LeftForeArm + LeftHand, RightForeArm +
RightHand) into the skel's Ulna (revolute X) / Radius (revolute around
forearm long axis) / Carpal (Ball, no axial twist) structure.

Decomposition per frame:
  * Total forearm-axis twist comes from BOTH BVH ForeArm and BVH Hand
    rotations (pronation can be authored at either joint). Sum → Radius
    revolute angle.
  * Elbow flex = swing component of BVH ForeArm projected onto the
    Ulna's local X axis → Ulna revolute angle.
  * Hand non-axial rotation (flex/ext + abd/add) → Carpal Ball joint.

BVH structure modification: insert `LeftForeArm_Twist`/`RightForeArm_Twist`
as new BVH joints between ForeArm and Hand. ForeArm channels carry the
Ulna angle (pure rotation around an axis we declare, currently world X
for the Euler we write), ForeArm_Twist carries the Radius angle (around
forearm axis), Hand carries the carpal swing. Frame-0 subtraction applied
to all three so the chain rests at N-pose at frame 0.

XML patch (when --skip-xml not passed):
  L/R_Ulna0  → bvh="LeftForeArm" / "RightForeArm"
  L/R_Radius0 → bvh="LeftForeArm_Twist" / "RightForeArm_Twist"
  L/R_Carpal0 → bvh="LeftHand" / "RightHand"
"""
import argparse
import os
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R


LEFT_FOREARM = "LeftForeArm"
RIGHT_FOREARM = "RightForeArm"
LEFT_HAND = "LeftHand"
RIGHT_HAND = "RightHand"


def _suffix_match(name, suf):
    return name == suf or name.endswith("_" + suf)


def parse_bvh(lines):
    """Pre-order parse. Returns (joints, motion_idx). Each joint: dict with
    name, type, parent_idx, children (idx list), channels, offset,
    brace_open/close line indices, indent depth."""
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
            pending_name = f"EndSite_{(joints[-1]['name'] if joints else 'x')}"
            continue
        if s == "{":
            j = {
                "type": pending_type,
                "name": pending_name,
                "parent_idx": stack[-1] if stack else -1,
                "children": [],
                "channels": [],
                "offset": None,
                "brace_open": i,
                "brace_close": None,
                "indent": len(ln) - len(ln.lstrip("\t ")),
            }
            joints.append(j)
            new_idx = len(joints) - 1
            if stack:
                joints[stack[-1]]["children"].append(new_idx)
            stack.append(new_idx)
            pending_type = None
            pending_name = None
            continue
        if s == "}":
            if stack:
                joints[stack[-1]]["brace_close"] = i
                stack.pop()
            continue
        m_off = re.match(r"^OFFSET\s+(\S+)\s+(\S+)\s+(\S+)\s*$", s)
        if m_off and stack:
            joints[stack[-1]]["offset"] = tuple(float(x) for x in m_off.groups())
            continue
        m_ch = re.match(r"^CHANNELS\s+(\d+)\s+(.+)$", s)
        if m_ch and stack:
            joints[stack[-1]]["channels"] = m_ch.group(2).split()
            continue
    return joints, motion_idx


def find_joint(joints, suffix):
    for i, j in enumerate(joints):
        if j["type"] in ("ROOT", "JOINT") and _suffix_match(j["name"], suffix):
            return i
    return None


def channel_layout(joints):
    out = []
    def walk(i):
        j = joints[i]
        if j["channels"]:
            out.append((i, len(j["channels"])))
        for c in j["children"]:
            walk(c)
    for i, j in enumerate(joints):
        if j["parent_idx"] == -1:
            walk(i)
    return out


def twist_swing(q_wxyz, axis):
    """Decompose unit quaternion (w, x, y, z) into (R_twist, R_swing) with
    R = R_swing @ R_twist about `axis` (normalized)."""
    w = q_wxyz[0]
    v = np.asarray(q_wxyz[1:], dtype=np.float64)
    a = np.asarray(axis, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    v_twist = np.dot(v, a) * a
    twist_q = np.array([w, *v_twist])
    n = np.linalg.norm(twist_q)
    if n < 1e-12:
        twist_q = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        twist_q /= n
    qq = R.from_quat([v[0], v[1], v[2], w])
    tq = R.from_quat([twist_q[1], twist_q[2], twist_q[3], twist_q[0]])
    swing = qq * tq.inv()
    return tq, swing


def twist_angle(rot, axis):
    """Signed angle of rotation around `axis` extracted via twist-swing."""
    q = rot.as_quat()  # x, y, z, w
    w, v = q[3], q[:3]
    a = np.asarray(axis, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    # twist quaternion ~ (w, (v·a)*a) normalized
    vt = np.dot(v, a)
    twist_q = np.array([w, vt * a[0], vt * a[1], vt * a[2]])
    n = np.linalg.norm(twist_q)
    if n < 1e-12:
        return 0.0
    twist_q /= n
    # signed angle = 2*atan2(|vec part along axis|, w) with sign from vt
    s = np.sign(vt) if abs(vt) > 1e-12 else 1.0
    ang = 2.0 * np.arctan2(np.linalg.norm(twist_q[1:]), twist_q[0])
    return float(s * ang)


def _read_axis_from_xml(xml_path, node_name):
    import xml.etree.ElementTree as ET
    tr = ET.parse(xml_path)
    for nd in tr.getroot().findall("Node"):
        if nd.attrib.get("name") != node_name:
            continue
        j = nd.find("Joint")
        if j is None:
            return None
        if j.attrib.get("type") != "Revolute":
            return None
        a = j.attrib.get("axis", "1 0 0").split()
        v = np.array([float(x) for x in a], dtype=np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="bvh_in", required=True)
    ap.add_argument("--out", dest="bvh_out", required=True)
    ap.add_argument("--skip-xml", action="store_true")
    ap.add_argument("--skel-xml", default="data/zygote_skel.xml")
    ap.add_argument("--l-radius-offset-deg", type=float, default=0.0,
                    help="Constant supination offset for L_Radius (palm-front→palm-medial). Applied to all frames (not removed by frame-0 subtract).")
    ap.add_argument("--r-radius-offset-deg", type=float, default=0.0,
                    help="Same for R_Radius.")
    ap.add_argument("--no-frame0-subtract", action="store_true",
                    help="Skip frame-0 subtraction (preserves BVH's initial pose, e.g. elbow flex at frame 0).")
    ap.add_argument("--ulna-scale", type=float, default=1.0,
                    help="Scale factor for ulna flex angle (tune visual bend vs BVH magnitude).")
    ap.add_argument("--orig-bvh",
                    help="Original BVH for joint-position comparison after bake.")
    ap.add_argument("--const-radius-twist", action="store_true",
                    help="Skip per-frame Radius twist (only constant offset "
                         "from --l/r-radius-offset-deg). Palm orientation "
                         "stays fixed relative to forearm across all frames.")
    ap.add_argument("--use-bvh-bend", action="store_true",
                    help="Replace ulna angle calc with BVH elbow bend angle (arccos of arm·forearm vec).")
    ap.add_argument("--ik-ulna", action="store_true",
                    help="IK: find ulna angle that minimizes |skel_wrist - target_wrist| where target_wrist = skel_shoulder + (BVH_wrist - BVH_shoulder).")
    args = ap.parse_args()

    with open(args.bvh_in) as f:
        lines = f.read().splitlines()
    joints, motion_idx = parse_bvh(lines)
    if motion_idx is None:
        sys.exit("MOTION section not found")
    layout = channel_layout(joints)
    name_to_col = {}
    col = 0
    for jidx, n in layout:
        name_to_col[joints[jidx]["name"]] = (col, n)
        col += n
    total_cols = col

    # Read motion frames
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
    assert len(frame_data[0]) == total_cols
    print(f"Frames: {len(frame_data)}, channels: {total_cols}")

    # Build skel via DART to query parent body world rest orientations.
    # Conjugation by R_p^-1 maps BVH-local rotation (assuming BVH parent
    # world rest = identity post-arm-subtract) into skel joint-local.
    sys.path.insert(0, ".")
    from core.dartHelper import saveSkeletonInfo, buildFromInfo
    _si, _rn, *_ = saveSkeletonInfo(args.skel_xml)
    _skel = buildFromInfo(_si, _rn)
    def _body_world_R(name):
        bn = _skel.getBodyNode(name)
        if bn is None: return np.eye(3)
        return np.asarray(bn.getTransform().rotation())

    # Parent body rest rotations for each retargeted joint:
    #   Ulna's parent body = Humerus
    #   Radius's parent body = Ulna
    #   Carpal's parent body = Radius
    sides = []
    for fa_suf, hand_suf, ulna_node, radius_node, carpal_node, humerus_node in [
        (LEFT_FOREARM, LEFT_HAND, "L_Ulna0", "L_Radius0", "L_Carpal0", "L_Humerus0"),
        (RIGHT_FOREARM, RIGHT_HAND, "R_Ulna0", "R_Radius0", "R_Carpal0", "R_Humerus0"),
    ]:
        fa_idx = find_joint(joints, fa_suf)
        hd_idx = find_joint(joints, hand_suf)
        if fa_idx is None or hd_idx is None:
            print(f"  WARN: missing pair {fa_suf}/{hand_suf}")
            continue
        fa_name = joints[fa_idx]["name"]
        twist_name = (fa_suf + "_Twist") if fa_name == fa_suf else (fa_name + "_Twist")
        ch_order = "".join(c[0] for c in joints[fa_idx]["channels"]
                           if c.lower().endswith("rotation")).upper()
        # Skel humerus body world rest rotation. Needed to express the BVH
        # forearm world rotation in skel Ulna joint-local frame:
        #   R_ulna_local = R_humerus_world^-1 * R_forearm_world_bvh
        R_humerus_world = _body_world_R(humerus_node)
        R_p = R.from_matrix(R_humerus_world)
        R_p_inv = R.from_matrix(R_humerus_world.T)
        ulna_axis = _read_axis_from_xml(args.skel_xml, ulna_node)
        radius_axis = _read_axis_from_xml(args.skel_xml, radius_node)
        if ulna_axis is None: ulna_axis = np.array([1.0, 0.0, 0.0])
        if radius_axis is None: radius_axis = np.array([0.0, -1.0, 0.0])
        sides.append({
            "fa_idx": fa_idx,
            "hd_idx": hd_idx,
            "twist_name": twist_name,
            "order": ch_order,
            "fa_name": fa_name,
            "hd_name": joints[hd_idx]["name"],
            "ulna_axis": ulna_axis,
            "radius_axis": radius_axis,
            "R_p_inv": R_p_inv,
            "R_p": R_p,
        })
        print(f"  {fa_name}: humerus_world_rest=\n{R_humerus_world.round(3)}")
        print(f"    ulna_axis={ulna_axis}  radius_axis={radius_axis}")

    if not sides:
        sys.exit("no forearm pairs")

    # Per-frame outputs (ZXY Euler arrays, in degrees)
    new_fa_euler = {s["fa_name"]: [] for s in sides}        # for Ulna (X rot only)
    new_tw_euler = {s["twist_name"]: [] for s in sides}     # for Radius (forearm-axis rot only)
    new_hd_euler = {s["hd_name"]: [] for s in sides}        # for Carpal (no axial twist)

    # Pre-compute BVH shoulder/elbow/wrist world positions per frame for
    # --use-bvh-bend (anatomical elbow angle).
    bvh_arm_pos = {s["fa_name"]: [] for s in sides}
    bvh_elbow_pos = {s["fa_name"]: [] for s in sides}
    bvh_wrist_pos = {s["fa_name"]: [] for s in sides}
    if args.use_bvh_bend or args.ik_ulna:
        # Use original BVH (if provided) for bend angle / IK target, else current.
        bend_bvh = args.orig_bvh or args.bvh_in
        with open(bend_bvh) as fbf:
            obvh_lines = fbf.read().splitlines()
        obvh_joints, obvh_mi = parse_bvh(obvh_lines)
        obvh_layout = channel_layout(obvh_joints)
        obvh_n2c = {}; ocol = 0
        for ji, n in obvh_layout: obvh_n2c[obvh_joints[ji]['name']] = (ocol, n); ocol += n
        obvh_rows = [list(map(float, ln.split())) for ln in obvh_lines[obvh_mi:]
                     if ln.strip() and not ln.strip().startswith(('MOTION','Frame'))]
        order = []
        def _walk(i):
            order.append(i)
            for c in obvh_joints[i]['children']: _walk(c)
        for i, j in enumerate(obvh_joints):
            if j['parent_idx'] == -1: _walk(i)
        # Map side fa_name → obvh joint indices.
        side_obvh = {}
        for s in sides:
            for jname in [s["fa_name"], s["fa_name"].replace("Character1_", ""),
                          "Character1_" + s["fa_name"]]:
                obvh_fi = next((idx for idx, jj in enumerate(obvh_joints)
                                if jj['name'] == jname or jj['name'].endswith("_"+jname)), None)
                if obvh_fi is not None:
                    obvh_hi = next((idx for idx, jj in enumerate(obvh_joints)
                                    if (jj['name'] == s["hd_name"]
                                        or jj['name'].endswith("_"+s["hd_name"].replace("Character1_","")))), None)
                    if obvh_hi is None:
                        # fallback by suffix
                        suf = "LeftHand" if "Left" in s["fa_name"] else "RightHand"
                        obvh_hi = next((idx for idx, jj in enumerate(obvh_joints)
                                        if jj['name'] == suf or jj['name'].endswith("_"+suf)), None)
                    side_obvh[s["fa_name"]] = (obvh_fi, obvh_hi); break
        n_frames = min(len(frame_data), len(obvh_rows))
        for f_idx in range(n_frames):
            row = obvh_rows[f_idx]
            Twr = [None] * len(obvh_joints)
            for ji in order:
                j = obvh_joints[ji]
                off = np.array(j['offset'])
                Rl = R.identity(); pos_add = np.zeros(3)
                chs = j['channels']
                if chs:
                    c0, _ = obvh_n2c[j['name']]; pi = 0; rc = []
                    for ch in chs:
                        v = row[c0 + pi]; pi += 1
                        if ch.lower() == 'xposition': pos_add[0] = v
                        elif ch.lower() == 'yposition': pos_add[1] = v
                        elif ch.lower() == 'zposition': pos_add[2] = v
                        else: rc.append((ch[0].upper(), v))
                    if rc: Rl = R.from_euler(''.join(c for c, _ in rc), [v for _, v in rc], degrees=True)
                T = np.eye(4); T[:3,:3] = Rl.as_matrix(); T[:3,3] = off + pos_add
                if j['parent_idx'] >= 0: T = Twr[j['parent_idx']] @ T
                Twr[ji] = T
            for s in sides:
                if s["fa_name"] not in side_obvh: continue
                obvh_fi, obvh_hi = side_obvh[s["fa_name"]]
                arm_idx = obvh_joints[obvh_fi]['parent_idx']
                bvh_arm_pos[s["fa_name"]].append(Twr[arm_idx][:3, 3])
                bvh_elbow_pos[s["fa_name"]].append(Twr[obvh_fi][:3, 3])
                bvh_wrist_pos[s["fa_name"]].append(Twr[obvh_hi][:3, 3])

    # For IK ulna: build skel + load arm-baked mocap to fix humerus per frame.
    ik_skel = None
    ik_bi = None
    ik_mocap = None
    ik_ulna_body_names = None
    ik_ulna_dof_idx = None
    if args.ik_ulna:
        from core.dartHelper import saveSkeletonInfo, buildFromInfo
        from core.bvhparser import MyBVH
        _si, _rn, _bi, *_ = saveSkeletonInfo(args.skel_xml)
        ik_skel = buildFromInfo(_si, _rn)
        ik_bi = _bi
        ik_mocap_obj = MyBVH(args.bvh_in, ik_bi, ik_skel)
        ik_mocap = ik_mocap_obj.mocap_refs
        ik_ulna_body_names = {"Character1_LeftForeArm": "L_Ulna0",
                              "LeftForeArm": "L_Ulna0",
                              "Character1_RightForeArm": "R_Ulna0",
                              "RightForeArm": "R_Ulna0"}
        ik_ulna_dof_idx = {}
        for skel_name in ["L_Ulna0", "R_Ulna0"]:
            j = ik_skel.getBodyNode(skel_name).getParentJoint()
            ik_ulna_dof_idx[skel_name] = j.getIndexInSkeleton(0)

    for f in range(len(frame_data)):
        row = frame_data[f]
        for s in sides:
            order = s["order"]
            c_fa, n_fa = name_to_col[s["fa_name"]]
            c_hd, n_hd = name_to_col[s["hd_name"]]
            R_fa_bvh = R.from_euler(order, row[c_fa:c_fa + n_fa], degrees=True)
            R_hd_bvh = R.from_euler(order, row[c_hd:c_hd + n_hd], degrees=True)
            # BVH rotvec in BVH local frame (= world at rest post-subtract).
            # Project onto ulna_axis transformed to world via parent body
            # rest rotation: ulna_axis_world = R_p * ulna_axis_xml.
            # BVH actor: L arm bone = BVH local +X, R = BVH local -X.
            # Elbow flex axis in BVH local = Y (perpendicular to bone in
            # sagittal plane). Decompose BVH motion: twist along bone (X) +
            # swing (rest). Project swing onto Y for Ulna flex magnitude.
            rv_fa = R_fa_bvh.as_rotvec()
            rv_hd = R_hd_bvh.as_rotvec()
            bvh_bone_sign = +1.0 if "Left" in s["fa_name"] else -1.0
            bone_bvh = np.array([bvh_bone_sign, 0.0, 0.0])
            # ForeArm decomp: twist around bone + swing.
            q_fa = R_fa_bvh.as_quat()
            q_fa_wxyz = np.array([q_fa[3], q_fa[0], q_fa[1], q_fa[2]])
            tw_fa, sw_fa = twist_swing(q_fa_wxyz, bone_bvh)
            alpha_r_fa = twist_angle(tw_fa, bone_bvh)
            # Ulna flex = swing rotvec projected onto BVH Y axis (signed).
            alpha_u = float(sw_fa.as_rotvec()[1]) * args.ulna_scale
            if args.use_bvh_bend:
                # Anatomical bend = angle between arm and forearm vectors.
                # Skel Ulna XML axes mirror: L=+X, R=-X. To produce mirrored
                # anatomical flexion (joint range upper=0 lower=-2.5 means
                # signed value must be negative for both), sign per side:
                # L → -bend (axis +X gives rotvec -X*bend after MyBVH chain),
                # R → +bend (axis -X gives rotvec +X*bend, projects to -bend).
                p_arm = bvh_arm_pos[s["fa_name"]][f]
                p_elb = bvh_elbow_pos[s["fa_name"]][f]
                p_wri = bvh_wrist_pos[s["fa_name"]][f]
                arm_v = p_elb - p_arm
                fa_v = p_wri - p_elb
                an = np.linalg.norm(arm_v); fn = np.linalg.norm(fa_v)
                if an > 1e-6 and fn > 1e-6:
                    cos_a = float(np.clip(np.dot(arm_v, fa_v) / (an * fn), -1, 1))
                    bend = float(np.arccos(cos_a))
                else:
                    bend = 0.0
                is_left = "Left" in s["fa_name"]
                alpha_u = (-bend if is_left else bend) * args.ulna_scale
            if args.ik_ulna:
                # IK: find α minimizing |skel_wrist - target_wrist| in WORLD.
                # target_wrist = skel_shoulder + (bvh_wrist - bvh_shoulder).
                skel_body = ik_ulna_body_names.get(s["fa_name"])
                if skel_body is not None:
                    sk_bn = ik_skel.getBodyNode(skel_body)
                    shoulder_skel_name = "L_Humerus0" if "Left" in s["fa_name"] else "R_Humerus0"
                    carpal_skel_name = "L_Carpal0" if "Left" in s["fa_name"] else "R_Carpal0"
                    # Set skel to f's mocap (gives humerus rotation etc.).
                    pose = np.zeros(ik_skel.getNumDofs())
                    nn = min(len(pose), ik_mocap.shape[1])
                    pose[:nn] = ik_mocap[f, :nn]
                    # Target wrist world from BVH.
                    p_arm = bvh_arm_pos[s["fa_name"]][f]
                    p_wri = bvh_wrist_pos[s["fa_name"]][f]
                    rel = p_wri - p_arm
                    # Search α.
                    dof_i = ik_ulna_dof_idx[skel_body]
                    def _err(a):
                        pose[dof_i] = float(a)
                        ik_skel.setPositions(pose)
                        sh = ik_skel.getBodyNode(shoulder_skel_name).getTransform().translation()
                        wr = ik_skel.getBodyNode(carpal_skel_name).getTransform().translation()
                        return float(np.linalg.norm((wr - sh) - rel))
                    # Coarse grid for global minimum, then refine.
                    grid = np.linspace(-np.pi, np.pi, 37)
                    best_a = min(grid, key=_err)
                    from scipy.optimize import minimize_scalar
                    res = minimize_scalar(_err, bracket=(best_a - 0.2, best_a, best_a + 0.2),
                                          method="brent", options={"xtol": 1e-5})
                    alpha_u = float(res.x)
            R_ulna_only = R.from_rotvec(s["ulna_axis"] * alpha_u)
            # Hand decomp: twist around bone + swing (Carpal).
            q_hd = R_hd_bvh.as_quat()
            q_hd_wxyz = np.array([q_hd[3], q_hd[0], q_hd[1], q_hd[2]])
            tw_hd, sw_hd = twist_swing(q_hd_wxyz, bone_bvh)
            alpha_r_hd = twist_angle(tw_hd, bone_bvh)
            # Total Radius twist = ForeArm twist + Hand twist (palm axial).
            alpha_r = alpha_r_fa + alpha_r_hd
            if args.const_radius_twist:
                alpha_r = 0.0  # zero per-frame twist, only constant offset applied later
            R_radius_only = R.from_rotvec(s["radius_axis"] * alpha_r)
            # Carpal: hand swing only (no axial).
            R_carpal = sw_hd
            new_fa_euler[s["fa_name"]].append(
                R_ulna_only.as_euler(order, degrees=True).tolist())
            new_tw_euler[s["twist_name"]].append(
                R_radius_only.as_euler(order, degrees=True).tolist())
            new_hd_euler[s["hd_name"]].append(
                R_carpal.as_euler(order, degrees=True).tolist())

    # Frame-0 subtraction for swing/twist/hand (so they land at identity
    # at frame 0).
    def _subtract_frame0(eulers_list, order):
        R0 = R.from_euler(order, eulers_list[0], degrees=True)
        out = []
        for e in eulers_list:
            Rf = R.from_euler(order, e, degrees=True)
            R_new = R0.inv() * Rf
            out.append(R_new.as_euler(order, degrees=True).tolist())
        return out

    if not args.no_frame0_subtract:
        for s in sides:
            order = s["order"]
            new_fa_euler[s["fa_name"]] = _subtract_frame0(new_fa_euler[s["fa_name"]], order)
            new_tw_euler[s["twist_name"]] = _subtract_frame0(new_tw_euler[s["twist_name"]], order)
            new_hd_euler[s["hd_name"]] = _subtract_frame0(new_hd_euler[s["hd_name"]], order)

    # Constant supination offset (palm-front → palm-medial). Applied around
    # XML radius_axis (joint-local) so MyBVH projection picks it up fully.
    for s in sides:
        order = s["order"]
        is_left = "Left" in s["fa_name"]
        off_deg = args.l_radius_offset_deg if is_left else args.r_radius_offset_deg
        if abs(off_deg) < 1e-9: continue
        R_off = R.from_rotvec(s["radius_axis"] * np.radians(off_deg))
        out = []
        for e in new_tw_euler[s["twist_name"]]:
            Rf = R.from_euler(order, e, degrees=True)
            R_new = R_off * Rf
            out.append(R_new.as_euler(order, degrees=True).tolist())
        new_tw_euler[s["twist_name"]] = out

    # Write Eulers back to motion rows.
    for f in range(len(frame_data)):
        for s in sides:
            c_fa, n_fa = name_to_col[s["fa_name"]]
            c_hd, n_hd = name_to_col[s["hd_name"]]
            for k in range(n_fa):
                frame_data[f][c_fa + k] = float(new_fa_euler[s["fa_name"]][f][k])
            for k in range(n_hd):
                frame_data[f][c_hd + k] = float(new_hd_euler[s["hd_name"]][f][k])

    # Insert ForeArm_Twist channels in motion rows at the column position
    # right after ForeArm's channels. Sort by descending col so insertions
    # don't shift earlier indices.
    insert_specs = []
    for s in sides:
        c_fa, n_fa = name_to_col[s["fa_name"]]
        insert_specs.append((c_fa + n_fa, s["twist_name"]))
    insert_specs.sort(key=lambda t: -t[0])
    for col_pos, tw_name in insert_specs:
        eulers = new_tw_euler[tw_name]
        for f, row in enumerate(frame_data):
            row[col_pos:col_pos] = [float(x) for x in eulers[f]]

    # Rewrite hierarchy: wrap ForeArm's existing children inside a new
    # JOINT block named ForeArm_Twist.
    hier_lines = lines[:motion_idx]
    sides_sorted = sorted(sides, key=lambda s: -joints[s["fa_idx"]]["brace_open"])
    for s in sides_sorted:
        fa = joints[s["fa_idx"]]
        fa_open = fa["brace_open"]
        fa_close = fa["brace_close"]
        indent = fa["indent"]
        head_lines = []
        body_lines = []
        in_head = True
        for k in range(fa_open + 1, fa_close):
            ln = hier_lines[k]
            sline = ln.strip()
            if in_head and (sline.startswith("OFFSET") or sline.startswith("CHANNELS")):
                head_lines.append(ln)
                continue
            in_head = False
            body_lines.append(ln)
        tw_indent = "\t" * (indent + 1)
        tw_open = [
            tw_indent + f"JOINT {s['twist_name']}",
            tw_indent + "{",
            tw_indent + "\tOFFSET 0 0 0",
            tw_indent + "\tCHANNELS 3 Zrotation Xrotation Yrotation",
        ]
        tw_close = [tw_indent + "}"]
        hier_lines[fa_open + 1:fa_close] = head_lines + tw_open + body_lines + tw_close

    out_lines = list(hier_lines) + header
    for row in frame_data:
        out_lines.append(" ".join(f"{v:.6f}" for v in row))
    with open(args.bvh_out, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Wrote {args.bvh_out}")

    if args.skip_xml:
        _run_verify_if_requested(args)
        return
    import xml.etree.ElementTree as ET
    SKEL_XML = "data/zygote_skel.xml"
    MAP = {
        "L_Ulna0": "LeftForeArm",
        "R_Ulna0": "RightForeArm",
        "L_Radius0": "LeftForeArm_Twist",
        "R_Radius0": "RightForeArm_Twist",
        "L_Carpal0": "LeftHand",
        "R_Carpal0": "RightHand",
    }
    tree = ET.parse(SKEL_XML)
    patched = 0
    for node in tree.getroot().findall("Node"):
        n = node.attrib["name"]
        if n not in MAP:
            continue
        j = node.find("Joint")
        if j is None:
            continue
        j.attrib["bvh"] = MAP[n]
        patched += 1
        print(f"  {n} → bvh=\"{MAP[n]}\"")
    tree.write(SKEL_XML)
    print(f"Patched {patched} XML joints")

    _run_verify_if_requested(args)


def _run_verify_if_requested(args):
    if getattr(args, "orig_bvh", None):
        verify_bake(args.orig_bvh, args.bvh_out, args.skel_xml)


def verify_bake(orig_bvh, baked_bvh, skel_xml):
    """Compare original BVH joint world positions vs skel positions after
    applying baked mocap. Report shoulder→hand vector per side at f0 and
    a few representative frames."""
    from core.dartHelper import saveSkeletonInfo, buildFromInfo
    from core.bvhparser import MyBVH

    def bvh_fk_f0(path, suffixes):
        with open(path) as f: lines = f.read().splitlines()
        joints, mi = parse_bvh(lines)
        layout = channel_layout(joints)
        n2c = {}; col = 0
        for ji, n in layout: n2c[joints[ji]['name']] = (col, n); col += n
        rows = [list(map(float, ln.split())) for ln in lines[mi:]
                if ln.strip() and not ln.strip().startswith(('MOTION','Frame'))]
        order = []
        def walk(i):
            order.append(i)
            for c in joints[i]['children']: walk(c)
        for i, j in enumerate(joints):
            if j['parent_idx'] == -1: walk(i)
        Twr = [None] * len(joints)
        for ji in order:
            j = joints[ji]
            off = np.array(j['offset'])
            chs = j['channels']
            Rl = R.identity(); pos_add = np.zeros(3)
            if chs:
                c0, _ = n2c[j['name']]; pi = 0; rc = []
                for ch in chs:
                    v = rows[0][c0 + pi]; pi += 1
                    if ch.lower() == 'xposition': pos_add[0] = v
                    elif ch.lower() == 'yposition': pos_add[1] = v
                    elif ch.lower() == 'zposition': pos_add[2] = v
                    else: rc.append((ch[0].upper(), v))
                if rc: Rl = R.from_euler(''.join(c for c, _ in rc), [v for _, v in rc], degrees=True)
            T = np.eye(4); T[:3,:3] = Rl.as_matrix(); T[:3,3] = off + pos_add
            if j['parent_idx'] >= 0: T = Twr[j['parent_idx']] @ T
            Twr[ji] = T
        out = {}
        for suf in suffixes:
            ji = find_joint(joints, suf)
            if ji is not None:
                out[suf] = Twr[ji][:3, 3]
        return out

    bvh_pos = bvh_fk_f0(orig_bvh, ['LeftArm','LeftForeArm','LeftHand','RightArm','RightForeArm','RightHand'])
    # Determine BVH scale (root offset magnitude)
    bvh_scale = max(np.linalg.norm(bvh_pos.get('LeftArm', np.zeros(3))), 1.0)

    si, rn, bi, *_ = saveSkeletonInfo(skel_xml)
    skel = buildFromInfo(si, rn)
    m = MyBVH(baked_bvh, bi, skel)
    pose = np.zeros(skel.getNumDofs())
    nn = min(len(pose), m.mocap_refs.shape[1])
    pose[:nn] = m.mocap_refs[0, :nn]
    skel.setPositions(pose)
    skel_pos = {bn: skel.getBodyNode(bn).getCOM() for bn in
                ['L_Humerus0','L_Ulna0','L_Carpal0','R_Humerus0','R_Ulna0','R_Carpal0']}

    print("\n=== Bake verification (frame 0) ===")
    pairs = [('LeftArm','L_Humerus0'),('LeftForeArm','L_Ulna0'),('LeftHand','L_Carpal0'),
             ('RightArm','R_Humerus0'),('RightForeArm','R_Ulna0'),('RightHand','R_Carpal0')]
    for bn, sn in pairs:
        b = bvh_pos.get(bn)
        s = skel_pos.get(sn)
        if b is None or s is None: continue
        print(f"  {bn:14s} BVH={np.round(b,2)}  |  {sn:11s} skel={np.round(s,3)}")
    # Relative shoulder→hand
    for arm, hand, sh_n, hd_n in [('LeftArm','LeftHand','L_Humerus0','L_Carpal0'),
                                   ('RightArm','RightHand','R_Humerus0','R_Carpal0')]:
        bv = bvh_pos[hand] - bvh_pos[arm]
        sv = skel_pos[hd_n] - skel_pos[sh_n]
        print(f"  {arm}→{hand} BVH rel={np.round(bv,2)} (norm={np.linalg.norm(bv):.2f})")
        print(f"  {sh_n}→{hd_n} skel rel={np.round(sv,3)} (norm={np.linalg.norm(sv):.3f})")


if __name__ == "__main__":
    main()
