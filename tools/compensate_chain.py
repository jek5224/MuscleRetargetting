"""Compensate BVH joint rotations for skel chain intermediate body TLs.

Skel has joints (Sternum, L_Scapula, L_Radius) between BVH-mapped joints.
These bodies have non-identity rest TransLinear, creating chain rotation
offsets that MyBVH doesn't see.

For each mapped BVH joint, write rotation:
    X = body_TL_intermediate.T @ R_bvh_local @ body_TL_self.T

So that when MyBVH applies via skel chain, the resulting skel body world
rotation matches the BVH body world rotation.
"""
import argparse
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_bvh(path):
    with open(path) as f: lines = f.read().splitlines()
    joints = []
    stack = []
    pending_type = None; pending_name = None; motion_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s == "MOTION":
            motion_idx = i; break
        m = re.match(r"^(ROOT|JOINT)\s+(\S+)\s*$", s)
        if m:
            pending_type = m.group(1); pending_name = m.group(2); continue
        if s == "End Site":
            pending_type = "End"; pending_name = f"EndSite_{joints[-1]['name'] if joints else 'x'}"; continue
        if s == "{":
            j = {"type": pending_type, "name": pending_name,
                 "parent": stack[-1] if stack else -1,
                 "children": [], "channels": [], "offset": None}
            joints.append(j)
            new_idx = len(joints) - 1
            if stack: joints[stack[-1]]["children"].append(new_idx)
            stack.append(new_idx)
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


def get_body_TLs(skel_xml):
    tr = ET.parse(skel_xml)
    out = {}
    for nd in tr.getroot().findall("Node"):
        n = nd.attrib.get("name")
        body = nd.find("Body")
        if body is None: continue
        t = body.find("Transformation")
        if t is None:
            out[n] = np.eye(3); continue
        arr = [float(x) for x in t.attrib.get("linear", "1 0 0 0 1 0 0 0 1").split()]
        out[n] = np.array(arr).reshape(3, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="bvh_in", required=True)
    ap.add_argument("--out", dest="bvh_out", required=True)
    ap.add_argument("--skel-xml", default="data/zygote_skel.xml")
    args = ap.parse_args()

    lines, joints, motion_idx = parse_bvh(args.bvh_in)
    layout = []
    def walk(i):
        if joints[i]["channels"]:
            layout.append((i, len(joints[i]["channels"])))
        for c in joints[i]["children"]: walk(c)
    for i, j in enumerate(joints):
        if j["parent"] == -1: walk(i)
    n2c = {}
    col = 0
    for ji, n in layout: n2c[joints[ji]["name"]] = (col, n); col += n

    rows = []
    header = []
    for ln in lines[motion_idx:]:
        s = ln.strip()
        if not s: continue
        if s.startswith("MOTION") or s.startswith("Frames") or s.startswith("Frame Time"):
            header.append(ln); continue
        rows.append([float(x) for x in s.split()])

    body_TLs = get_body_TLs(args.skel_xml)

    # Compensation map: BVH joint suffix → (intermediate body name, self body name)
    comp_map = {
        "LeftShoulder":   ("Sternum0",   "L_Clavicle0"),
        "LeftArm":        ("L_Scapula0", "L_Humerus0"),
        "LeftForeArm":    (None,         "L_Ulna0"),
        "LeftHand":       ("L_Radius0",  "L_Carpal0"),
        "RightShoulder":  ("Sternum0",   "R_Clavicle0"),
        "RightArm":       ("R_Scapula0", "R_Humerus0"),
        "RightForeArm":   (None,         "R_Ulna0"),
        "RightHand":      ("R_Radius0",  "R_Carpal0"),
    }

    def find_joint(suf):
        for ji, j in enumerate(joints):
            if j["type"] in ("ROOT", "JOINT") and (j["name"] == suf or j["name"].endswith("_" + suf)):
                return ji
        return None

    # For each mapped joint, build target TLs and apply compensation.
    for suf, (inter, slf) in comp_map.items():
        ji = find_joint(suf)
        if ji is None:
            print(f"  skip {suf} (not in BVH)")
            continue
        chs = joints[ji]["channels"]
        rot_chans = [c for c in chs if c.lower().endswith("rotation")]
        if not rot_chans: continue
        order = "".join(c[0] for c in rot_chans).upper()
        c0 = n2c[joints[ji]["name"]][0]
        rot_offsets = [k for k, ch in enumerate(chs) if ch.lower().endswith("rotation")]
        TL_inter = body_TLs.get(inter, np.eye(3)) if inter else np.eye(3)
        TL_self = body_TLs.get(slf, np.eye(3))
        R_inter = R.from_matrix(TL_inter)
        R_self = R.from_matrix(TL_self)
        print(f"  {suf}: inter={inter} self={slf}")
        for row in rows:
            e_f = [row[c0 + k] for k in rot_offsets]
            R_bvh = R.from_euler(order, e_f, degrees=True)
            R_x = R_inter.inv() * R_bvh * R_self.inv()
            new_e = R_x.as_euler(order, degrees=True).tolist()
            for k, off in enumerate(rot_offsets):
                row[c0 + off] = new_e[k]

    out_lines = lines[:motion_idx] + header
    for row in rows:
        out_lines.append(" ".join(f"{v:.6f}" for v in row))
    with open(args.bvh_out, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Wrote {args.bvh_out}")


if __name__ == "__main__":
    main()
