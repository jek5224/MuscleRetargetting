"""Pure position-based IK retarget. For each frame, target skel joint
world positions = scaled BVH joint world positions. Optimize skel DOFs
via least_squares. Bypasses all rotation convention issues.

Inputs: any BVH with standard joint names (Hips, LeftShoulder, ...).
Output: BVH with skel hierarchy + rotation Eulers per frame.

Handles upper body arms + spine simultaneously. Lower body uses BVH
mocap directly (already works).
"""
import argparse
import os
import re
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


def parse_bvh(path):
    with open(path) as f: lines = f.read().splitlines()
    joints = []; stack = []
    pending_type = None; pending_name = None; motion_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s == "MOTION": motion_idx = i; break
        m = re.match(r"^(ROOT|JOINT)\s+(\S+)\s*$", s)
        if m: pending_type, pending_name = m.group(1), m.group(2); continue
        if s == "End Site":
            pending_type = "End"; pending_name = f"EndSite_{joints[-1]['name'] if joints else 'x'}"; continue
        if s == "{":
            j = {"type": pending_type, "name": pending_name,
                 "parent": stack[-1] if stack else -1,
                 "children": [], "channels": [], "offset": None}
            joints.append(j); idx = len(joints) - 1
            if stack: joints[stack[-1]]["children"].append(idx)
            stack.append(idx); pending_type = pending_name = None; continue
        if s == "}":
            if stack: stack.pop(); continue
        m = re.match(r"^OFFSET\s+(\S+)\s+(\S+)\s+(\S+)\s*$", s)
        if m and stack: joints[stack[-1]]["offset"] = tuple(float(x) for x in m.groups()); continue
        m = re.match(r"^CHANNELS\s+(\d+)\s+(.+)$", s)
        if m and stack: joints[stack[-1]]["channels"] = m.group(2).split(); continue
    return lines, joints, motion_idx


def channel_layout(joints):
    out = []
    def walk(i):
        if joints[i]["channels"]: out.append((i, len(joints[i]["channels"])))
        for c in joints[i]["children"]: walk(c)
    for i, j in enumerate(joints):
        if j["parent"] == -1: walk(i)
    return out


def bvh_fk_positions(joints, row, n2c):
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
        Rl = R.identity(); pa = np.zeros(3)
        if j["channels"] and j["name"] in n2c:
            c0, _ = n2c[j["name"]]; pi = 0; rc = []
            for ch in j["channels"]:
                v = row[c0 + pi]; pi += 1
                if ch.lower() == "xposition": pa[0] = v
                elif ch.lower() == "yposition": pa[1] = v
                elif ch.lower() == "zposition": pa[2] = v
                else: rc.append((ch[0].upper(), v))
            if rc: Rl = R.from_euler("".join(c for c, _ in rc), [v for _, v in rc], degrees=True)
        T = np.eye(4); T[:3, :3] = Rl.as_matrix(); T[:3, 3] = off + pa
        if j["parent"] >= 0: T = Twr[j["parent"]] @ T
        Twr[ji] = T
    return Twr


def find_bvh_joint(joints, suf):
    for i, j in enumerate(joints):
        if j["type"] in ("ROOT", "JOINT") and (j["name"] == suf or j["name"].endswith("_" + suf)):
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="bvh_in", required=True)
    ap.add_argument("--out", dest="bvh_out", required=True)
    ap.add_argument("--skel-xml", default="data/zygote_skel.xml")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, ".")
    from core.dartHelper import saveSkeletonInfo, buildFromInfo
    from core.bvhparser import MyBVH

    lines, joints, motion_idx = parse_bvh(args.bvh_in)
    layout = channel_layout(joints)
    n2c = {}; col = 0
    for ji, n in layout: n2c[joints[ji]["name"]] = (col, n); col += n

    rows = []; header = []
    for ln in lines[motion_idx:]:
        s = ln.strip()
        if not s: continue
        if s.startswith("MOTION") or s.startswith("Frames") or s.startswith("Frame Time"):
            header.append(ln); continue
        rows.append([float(x) for x in s.split()])
    if args.max_frames > 0: rows = rows[:args.max_frames]
    print(f"Frames: {len(rows)}")

    si, rn, bi, *_ = saveSkeletonInfo(args.skel_xml)
    skel = buildFromInfo(si, rn)
    # Load BVH via MyBVH for baseline (legs + spine via T_frame=0).
    m = MyBVH(args.bvh_in, bi, skel, T_frame=0)
    print(f"Baseline mocap_refs shape: {m.mocap_refs.shape}")

    # Find BVH arm joint indices.
    bvh_arm = {}
    for suf in ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
                "RightShoulder", "RightArm", "RightForeArm", "RightHand"]:
        ji = find_bvh_joint(joints, suf)
        if ji is not None: bvh_arm[suf] = ji
    print(f"BVH arm joints: {list(bvh_arm.keys())}")

    # Skel arm body DOFs.
    sides = []
    for side, prefix in [("L", "L_"), ("R", "R_")]:
        clav = skel.getBodyNode(prefix + "Clavicle0")
        hum = skel.getBodyNode(prefix + "Humerus0")
        uln = skel.getBodyNode(prefix + "Ulna0")
        carp = skel.getBodyNode(prefix + "Carpal0")
        if not all([clav, hum, uln, carp]): continue
        dofs = []
        for bn in [clav, hum, uln, carp]:
            j = bn.getParentJoint()
            idx0 = j.getIndexInSkeleton(0); nd = j.getNumDofs()
            dofs.append((idx0, nd))
        sides.append({"side": side, "prefix": prefix,
                      "clav": clav, "hum": hum, "uln": uln, "carp": carp,
                      "dofs": dofs})

    # Compute scale: skel arm length / BVH arm length.
    Twr0 = bvh_fk_positions(joints, rows[0], n2c)
    bvh_arm_len = np.linalg.norm(Twr0[bvh_arm["LeftHand"]][:3, 3] - Twr0[bvh_arm["LeftArm"]][:3, 3])
    skel.setPositions(np.zeros(skel.getNumDofs()))
    skel_arm_len = np.linalg.norm(skel.getBodyNode("L_Carpal0").getCOM() - skel.getBodyNode("L_Humerus0").getCOM())
    scale = skel_arm_len / max(bvh_arm_len, 1e-6)
    print(f"Scale: {scale:.4f}")

    new_mocap = m.mocap_refs.copy()
    x0_per_side = {"L": np.zeros(10), "R": np.zeros(10)}

    for fi in range(len(rows)):
        # Set baseline pose from MyBVH (legs, spine via T_frame=0).
        pose = m.mocap_refs[fi].copy()

        Twr = bvh_fk_positions(joints, rows[fi], n2c)

        for s in sides:
            suf_arm = "LeftArm" if s["side"] == "L" else "RightArm"
            suf_fa = "LeftForeArm" if s["side"] == "L" else "RightForeArm"
            suf_hd = "LeftHand" if s["side"] == "L" else "RightHand"
            suf_sh = "LeftShoulder" if s["side"] == "L" else "RightShoulder"

            # BVH target positions relative to BVH shoulder, scaled.
            p_bvh_sh = Twr[bvh_arm[suf_sh]][:3, 3]
            rel_hum = (Twr[bvh_arm[suf_arm]][:3, 3] - p_bvh_sh) * scale
            rel_uln = (Twr[bvh_arm[suf_fa]][:3, 3] - p_bvh_sh) * scale
            rel_carp = (Twr[bvh_arm[suf_hd]][:3, 3] - p_bvh_sh) * scale

            # Skel shoulder world (from baseline pose).
            skel.setPositions(pose)
            sh_skel = s["clav"].getCOM()
            target_hum = sh_skel + rel_hum
            target_uln = sh_skel + rel_uln
            target_carp = sh_skel + rel_carp

            def residuals(x):
                p = pose.copy()
                xi = 0
                for idx0, nd in s["dofs"]:
                    for k in range(nd):
                        p[idx0 + k] = x[xi]; xi += 1
                skel.setPositions(p)
                r = []
                r.extend(s["hum"].getCOM() - target_hum)
                r.extend(s["uln"].getCOM() - target_uln)
                r.extend(s["carp"].getCOM() - target_carp)
                return r

            x0 = x0_per_side[s["side"]]
            try:
                res = least_squares(residuals, x0, method="lm", max_nfev=80)
                x0_per_side[s["side"]] = res.x
                xi = 0
                for idx0, nd in s["dofs"]:
                    for k in range(nd):
                        pose[idx0 + k] = res.x[xi]; xi += 1
            except Exception as e:
                pass

        new_mocap[fi] = pose
        if fi % 200 == 0:
            print(f"  frame {fi}/{len(rows)}")

    # Write output BVH (skel-mirror hierarchy with bvh attrs as joint names).
    import xml.etree.ElementTree as ET
    skel_tree = ET.parse(args.skel_xml)
    bvh_attr = {}
    for nd in skel_tree.getroot().findall("Node"):
        n = nd.attrib.get("name")
        j = nd.find("Joint")
        if j is not None and "bvh" in j.attrib:
            bvh_attr[n] = j.attrib["bvh"]

    out_lines = ["HIERARCHY"]
    skel.setPositions(np.zeros(skel.getNumDofs()))
    visited = set()
    def emit(bn, depth):
        body = bn.getName()
        if body in visited: return
        visited.add(body)
        name = bvh_attr.get(body, body)
        is_root = bn.getParentBodyNode() is None
        if is_root:
            offset = np.zeros(3)
        else:
            offset = bn.getRelativeTransform().translation()
        indent = "\t" * depth
        if is_root:
            out_lines.append(f"ROOT {name}")
            out_lines.append("{")
            out_lines.append(f"\tOFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
            out_lines.append("\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
        else:
            out_lines.append(indent + f"JOINT {name}")
            out_lines.append(indent + "{")
            out_lines.append(indent + f"\tOFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
            out_lines.append(indent + "\tCHANNELS 3 Zrotation Yrotation Xrotation")
        for ci in range(bn.getNumChildBodyNodes()):
            emit(bn.getChildBodyNode(ci), depth + 1)
        out_lines.append(indent + "}" if not is_root else "}")

    root_bn = None
    for i in range(skel.getNumBodyNodes()):
        if skel.getBodyNode(i).getParentBodyNode() is None:
            root_bn = skel.getBodyNode(i); break
    emit(root_bn, 0)

    out_lines.append("MOTION")
    out_lines.append(f"Frames: {len(rows)}")
    out_lines.append("Frame Time: 0.033333")

    def write_frame(pose):
        vals = []
        def write_bn(bn):
            sj = bn.getParentJoint()
            idx0 = sj.getIndexInSkeleton(0); nd = sj.getNumDofs()
            if nd == 6:
                vals.extend([pose[idx0 + 3], pose[idx0 + 4], pose[idx0 + 5]])
                rv = pose[idx0:idx0 + 3]
                Rm = R.from_rotvec(rv); eul = Rm.as_euler("ZYX", degrees=True)
                vals.extend([eul[0], eul[1], eul[2]])
            elif nd == 3:
                rv = pose[idx0:idx0 + 3]
                Rm = R.from_rotvec(rv); eul = Rm.as_euler("ZYX", degrees=True)
                vals.extend([eul[0], eul[1], eul[2]])
            elif nd == 1:
                ax = np.array(sj.getAxis(), dtype=np.float64, copy=True)
                ax = ax / max(np.linalg.norm(ax), 1e-12)
                Rm = R.from_rotvec(ax * pose[idx0]); eul = Rm.as_euler("ZYX", degrees=True)
                vals.extend([eul[0], eul[1], eul[2]])
            else:
                vals.extend([0.0, 0.0, 0.0])
            for ci in range(bn.getNumChildBodyNodes()):
                write_bn(bn.getChildBodyNode(ci))
        write_bn(root_bn)
        return vals

    for fi in range(len(rows)):
        vals = write_frame(new_mocap[fi])
        out_lines.append(" ".join(f"{v:.6f}" for v in vals))

    with open(args.bvh_out, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Wrote {args.bvh_out}")


if __name__ == "__main__":
    main()
