"""Per-frame arm retarget with anatomical rest alignment.

Anatomical basis:
- Clavicle (sternoclavicular joint): saddle joint, ~30° elevation/depression
  + ~30° protraction/retraction. Walking range ~5-15°.
- Humerus (glenohumeral joint): ball joint, ~180° flex/ext, ~180° abd/add,
  ~90° internal/external rotation. Walking arm swing ~10-30°.
- Scapulohumeral rhythm: scapula rotates ~1° per 2° humerus elevation
  during shoulder flexion/abduction. Not retargeted here (L_Scapula0 has
  no bvh attr in skel XML).

Why parent-local delta (not world delta):
- World delta from f0 carries root yaw drift. Actor walking forward
  accumulates 180° root yaw mid-clip → all child world rotations carry
  the same 180°. Pollutes arm motion.
- Parent-local channel delta (R_local_0.inv * R_local_f) cancels chain
  motion above the joint, capturing only joint-intrinsic motion.

Algorithm per arm joint (LeftShoulder, LeftArm, RightShoulder, RightArm):
  1. BVH parent world rest Q via BVH FK at f0.
  2. Skel parent body world rest P from DART.
  3. Rest-alignment rotation M = Q^T @ P (maps skel-parent frame to
     BVH-parent frame at rest).
  4. Per frame: ΔL_bvh = R_local_0.inv @ R_local_f (channel-based,
     parent-local — root yaw cancels).
  5. ΔL_skel = M^T @ ΔL_bvh @ M (conjugation — magnitude-preserving).
  6. Write ΔL_skel back as Euler.

Result: skel arm joint rotates in skel-parent-local frame by the SAME
angle the BVH arm joint rotates in BVH-parent-local frame. No scaling.
Clavicle stays in anatomical range because input does.
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
            j = {
                "type": pending_type,
                "name": pending_name,
                "parent": stack[-1] if stack else -1,
                "children": [],
                "channels": [],
                "offset": None,
            }
            joints.append(j)
            idx = len(joints) - 1
            if stack:
                joints[stack[-1]]["children"].append(idx)
            stack.append(idx)
            pending_type = None
            pending_name = None
            continue
        if s == "}":
            if stack:
                stack.pop()
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


def bvh_fk_world_R(joints, row, n2c):
    """Per-joint world rotation matrix (rotation only, ignores translation)."""
    Twr = [None] * len(joints)
    order_list = []

    def walk(i):
        order_list.append(i)
        for c in joints[i]["children"]:
            walk(c)

    for i, j in enumerate(joints):
        if j["parent"] == -1:
            walk(i)
    for ji in order_list:
        j = joints[ji]
        Rl = R.identity()
        if j["channels"] and j["name"] in n2c:
            chs = j["channels"]
            c0, _ = n2c[j["name"]]
            rc = []
            pi = 0
            for ch in chs:
                v = row[c0 + pi]
                pi += 1
                if ch.lower().endswith("rotation"):
                    rc.append((ch[0].upper(), v))
            if rc:
                Rl = R.from_euler("".join(c for c, _ in rc),
                                   [v for _, v in rc], degrees=True)
        M = Rl.as_matrix()
        if j["parent"] >= 0 and Twr[j["parent"]] is not None:
            M = Twr[j["parent"]] @ M
        Twr[ji] = M
    return Twr


def find_joint(joints, suf):
    for i, j in enumerate(joints):
        if j["type"] in ("ROOT", "JOINT") and (j["name"] == suf or j["name"].endswith("_" + suf)):
            return i
    return None


def joint_local_R(j, row, n2c):
    chs = j["channels"]
    c0, _ = n2c[j["name"]]
    rc = []
    pi = 0
    for ch in chs:
        v = row[c0 + pi]
        pi += 1
        if ch.lower().endswith("rotation"):
            rc.append((ch[0].upper(), v))
    if not rc:
        return R.identity()
    return R.from_euler("".join(c for c, _ in rc),
                         [v for _, v in rc], degrees=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="bvh_in", required=True)
    ap.add_argument("--out", dest="bvh_out", required=True)
    ap.add_argument("--skel-xml", default="data/zygote_skel.xml")
    args = ap.parse_args()

    lines, joints, mi = parse_bvh(args.bvh_in)
    layout = channel_layout(joints)
    n2c = {}
    col = 0
    for ji, n in layout:
        n2c[joints[ji]["name"]] = (col, n)
        col += n

    rows = []
    motion_header = []
    for ln in lines[mi:]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("MOTION") or s.startswith("Frames") or s.startswith("Frame Time"):
            motion_header.append(ln)
            continue
        rows.append([float(x) for x in s.split()])

    sys.path.insert(0, ".")
    from core.dartHelper import saveSkeletonInfo, buildFromInfo
    si, rn, *_ = saveSkeletonInfo(args.skel_xml)
    skel = buildFromInfo(si, rn)
    skel.setPositions(np.zeros(skel.getNumDofs()))

    bvh_to_skel = {
        "LeftShoulder": "L_Clavicle0",
        "LeftArm": "L_Humerus0",
        "RightShoulder": "R_Clavicle0",
        "RightArm": "R_Humerus0",
    }
    # Skel parent body world rest (P).
    P_skel = {}
    for bvh_suf, body_name in bvh_to_skel.items():
        bn = skel.getBodyNode(body_name)
        if bn is None:
            continue
        p = bn.getParentBodyNode()
        P_skel[bvh_suf] = (np.asarray(p.getTransform().rotation()).copy()
                            if p is not None else np.eye(3))

    # Locate BVH arm joints.
    arm_ji = {}
    for suf in bvh_to_skel:
        ji = find_joint(joints, suf)
        if ji is not None:
            arm_ji[suf] = ji
    print(f"  Retargeting arm joints: {list(arm_ji.keys())}")

    # BVH parent world rest (Q) via FK at f0.
    Twr0 = bvh_fk_world_R(joints, rows[0], n2c)
    Q_bvh = {}
    for suf, ji in arm_ji.items():
        par = joints[ji]["parent"]
        Q_bvh[suf] = Twr0[par].copy() if par >= 0 and Twr0[par] is not None else np.eye(3)

    M_align = {suf: Q_bvh[suf].T @ P_skel[suf] for suf in arm_ji}
    for suf, M in M_align.items():
        mag = np.linalg.norm(R.from_matrix(M).as_rotvec()) * 180 / np.pi
        print(f"  {suf}: rest-align M mag={mag:.2f}°")

    # Per frame: ΔL_skel = M^T @ R_local_bvh @ M.
    for fi, row in enumerate(rows):
        for suf, ji in arm_ji.items():
            j = joints[ji]
            Rl_f = joint_local_R(j, row, n2c).as_matrix()
            M = M_align[suf]
            dL_skel = M.T @ Rl_f @ M
            chs = j["channels"]
            order = "".join(c[0] for c in chs if c.lower().endswith("rotation")).upper()
            c0 = n2c[j["name"]][0]
            rot_offs = [k for k, c in enumerate(chs) if c.lower().endswith("rotation")]
            eul = R.from_matrix(dL_skel).as_euler(order, degrees=True)
            for k, off in enumerate(rot_offs):
                row[c0 + off] = float(eul[k])

    out_lines = lines[:mi] + motion_header
    for row in rows:
        out_lines.append(" ".join(f"{v:.6f}" for v in row))
    with open(args.bvh_out, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Wrote {args.bvh_out}")


if __name__ == "__main__":
    main()
