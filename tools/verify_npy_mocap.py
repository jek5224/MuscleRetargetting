"""Verify mocap_refs.npy: load skel, setPositions(mocap[f]), compare
body world positions against BVH FK targets."""
import argparse
import os
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_bvh(path):
    with open(path) as f:
        lines = f.read().splitlines()
    joints = []; stack = []; pt = None; pn = None; mi = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s == "MOTION": mi = i; break
        m = re.match(r"^(ROOT|JOINT)\s+(\S+)\s*$", s)
        if m: pt, pn = m.group(1), m.group(2); continue
        if s == "End Site": pt = "End"; pn = f"End_{joints[-1]['name']}"; continue
        if s == "{":
            j = {"type": pt, "name": pn, "parent": stack[-1] if stack else -1,
                 "children": [], "channels": [], "offset": None}
            joints.append(j); idx = len(joints) - 1
            if stack: joints[stack[-1]]["children"].append(idx)
            stack.append(idx); pt = pn = None; continue
        if s == "}":
            if stack: stack.pop()
            continue
        m = re.match(r"^OFFSET\s+(\S+)\s+(\S+)\s+(\S+)\s*$", s)
        if m and stack: joints[stack[-1]]["offset"] = tuple(float(x) for x in m.groups()); continue
        m = re.match(r"^CHANNELS\s+(\d+)\s+(.+)$", s)
        if m and stack: joints[stack[-1]]["channels"] = m.group(2).split(); continue
    return lines, joints, mi


def channel_layout(joints):
    out = []
    def walk(i):
        if joints[i]["channels"]: out.append((i, len(joints[i]["channels"])))
        for c in joints[i]["children"]: walk(c)
    for i, j in enumerate(joints):
        if j["parent"] == -1: walk(i)
    return out


def find_joint(joints, suf):
    for i, j in enumerate(joints):
        if j["type"] in ("ROOT", "JOINT") and (j["name"] == suf or j["name"].endswith("_" + suf)):
            return i
    return None


def bvh_fk(joints, row, n2c):
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


def joint_world_pos(skel, body_name):
    bn = skel.getBodyNode(body_name)
    par = bn.getParentBodyNode()
    j = bn.getParentJoint()
    T = j.getTransformFromParentBodyNode()
    if par is None: return T.translation()
    return par.getTransform().translation() + np.asarray(par.getTransform().rotation()) @ T.translation()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-bvh", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--skel-xml", default="data/zygote_skel.xml")
    args = ap.parse_args()

    olines, ojoints, omi = parse_bvh(args.orig_bvh)
    olayout = channel_layout(ojoints)
    on2c = {}; col = 0
    for ji, n in olayout:
        on2c[ojoints[ji]["name"]] = (col, n); col += n
    orows = []
    for ln in olines[omi:]:
        s = ln.strip()
        if not s or s.startswith(("MOTION", "Frames", "Frame Time")): continue
        orows.append([float(x) for x in s.split()])

    sys.path.insert(0, ".")
    from core.dartHelper import saveSkeletonInfo, buildFromInfo
    si, rn, *_ = saveSkeletonInfo(args.skel_xml)
    skel = buildFromInfo(si, rn)

    mocap = np.load(args.npy)
    print(f"mocap shape: {mocap.shape}")

    sides = []
    for L_or_R, p in [("L", "Left"), ("R", "Right")]:
        sd = {"L": L_or_R}
        sd["bvh_arm"] = find_joint(ojoints, f"{p}Arm")
        sd["bvh_fa"] = find_joint(ojoints, f"{p}ForeArm")
        sd["bvh_hd"] = find_joint(ojoints, f"{p}Hand")
        sd["skel_hum"] = f"{L_or_R}_Humerus0"
        sd["skel_ulna"] = f"{L_or_R}_Ulna0"
        sd["skel_carp"] = f"{L_or_R}_Carpal0"
        sides.append(sd)

    # Pelvis → Neck for spine direction.
    bvh_hips = find_joint(ojoints, "Hips")
    bvh_neck = find_joint(ojoints, "Neck")
    # Skel: Pelvis joint to C70 (neck body) or T10 (top trunk).
    spine_top_body = None
    for cand in ("C70", "Neck0", "T10"):
        for jj in range(skel.getNumJoints()):
            if skel.getBodyNode(jj).getName() == cand:
                spine_top_body = cand; break
        if spine_top_body: break

    N = mocap.shape[0]
    sample = [0, N // 4, N // 2, 3 * N // 4, N - 1]
    for f in sample:
        skel.setPositions(mocap[f])
        Tf = bvh_fk(ojoints, orows[f], on2c)
        # Spine direction.
        if spine_top_body and bvh_hips is not None and bvh_neck is not None:
            d_bvh_sp = Tf[bvh_neck][:3, 3] - Tf[bvh_hips][:3, 3]
            d_bvh_sp /= max(np.linalg.norm(d_bvh_sp), 1e-9)
            pel = joint_world_pos(skel, "Pelvis0") if any(skel.getBodyNode(jj).getName() == "Pelvis0" for jj in range(skel.getNumJoints())) else np.asarray(skel.getRootBodyNode().getTransform().translation())
            top = joint_world_pos(skel, spine_top_body)
            d_sk_sp = top - pel; d_sk_sp /= max(np.linalg.norm(d_sk_sp), 1e-9)
            csp = float(d_bvh_sp @ d_sk_sp)
            print(f"  f{f:5d} spine Hips→Neck cosθ={csp:+.4f}")
        for sd in sides:
            d_bvh_h = Tf[sd["bvh_fa"]][:3, 3] - Tf[sd["bvh_arm"]][:3, 3]
            d_bvh_h /= np.linalg.norm(d_bvh_h)
            d_bvh_f = Tf[sd["bvh_hd"]][:3, 3] - Tf[sd["bvh_fa"]][:3, 3]
            d_bvh_f /= np.linalg.norm(d_bvh_f)
            sh = joint_world_pos(skel, sd["skel_hum"])
            el = joint_world_pos(skel, sd["skel_ulna"])
            wr = joint_world_pos(skel, sd["skel_carp"])
            d_sk_h = el - sh; d_sk_h /= max(np.linalg.norm(d_sk_h), 1e-9)
            d_sk_f = wr - el; d_sk_f /= max(np.linalg.norm(d_sk_f), 1e-9)
            ch = float(d_bvh_h @ d_sk_h); cf = float(d_bvh_f @ d_sk_f)
            print(f"  f{f:5d} {sd['L']} humerus cosθ={ch:+.4f} forearm cosθ={cf:+.4f}")


if __name__ == "__main__":
    main()
