"""Full walk1 retarget: rest-pose alignment + per-frame body world rotation
match via DART forward kinematics.

For each frame f, for each skel body with BVH mapping, compute target
body world rotation from BVH FK, then solve joint local rotation given
parent body world (computed dynamically via DART setPositions for
upstream joints first).

Writes output BVH with skel-matching hierarchy (Sternum + Scapula + Radius
dummy joints added) so MyBVH chain matches skel chain.
"""
import argparse
import os
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_bvh(path):
    with open(path) as f: lines = f.read().splitlines()
    joints = []
    stack = []
    pending_type = None; pending_name = None; motion_idx = None
    line_open = {}
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


def channel_layout(joints):
    out = []
    def walk(i):
        if joints[i]["channels"]:
            out.append((i, len(joints[i]["channels"])))
        for c in joints[i]["children"]: walk(c)
    for i, j in enumerate(joints):
        if j["parent"] == -1: walk(i)
    return out


def bvh_fk(joints, row, n2c):
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


# Skel body name → BVH joint suffix
SKEL_TO_BVH = {
    "Saccrum_Coccyx0": "Hips",
    "L_Femur0": "LeftUpLeg", "L_Tibia_Fibula0": "LeftLeg",
    "L_Talus0": "LeftFoot", "L_Toe10": "LeftToeBase",
    "R_Femur0": "RightUpLeg", "R_Tibia_Fibula0": "RightLeg",
    "R_Talus0": "RightFoot", "R_Toe10": "RightToeBase",
    "L_Clavicle0": "LeftShoulder", "L_Humerus0": "LeftArm",
    "L_Ulna0": "LeftForeArm", "L_Carpal0": "LeftHand",
    "R_Clavicle0": "RightShoulder", "R_Humerus0": "RightArm",
    "R_Ulna0": "RightForeArm", "R_Carpal0": "RightHand",
}
# Spine vertebrae driven by Spine/Spine1 (composite). Sub-distribute.
SPINE_BVH_FOR_SKEL = {
    "L5": ("Spine", 5, 0), "L4": ("Spine", 5, 1), "L3": ("Spine", 5, 2),
    "L2": ("Spine", 5, 3), "L1": ("Spine", 5, 4),
    "T12": ("Spine1", 12, 0), "T11": ("Spine1", 12, 1), "T10": ("Spine1", 12, 2),
    "T9": ("Spine1", 12, 3), "T8": ("Spine1", 12, 4), "T7": ("Spine1", 12, 5),
    "T6": ("Spine1", 12, 6), "T5": ("Spine1", 12, 7), "T4": ("Spine1", 12, 8),
    "T3": ("Spine1", 12, 9), "T2": ("Spine1", 12, 10), "T1": ("Spine1", 12, 11),
    "C7": ("Neck", 5, 0), "C6": ("Neck", 5, 1), "C5": ("Neck", 5, 2),
    "C4": ("Neck", 5, 3), "C3": ("Neck", 5, 4),
    "Axis0": ("Head", 2, 0), "Atlas0": ("Head", 2, 1),
    "Head0": ("Head", 1, 0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="bvh_in", required=True)
    ap.add_argument("--out", dest="bvh_out", required=True)
    ap.add_argument("--skel-xml", default="data/zygote_skel.xml")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, ".")
    from core.dartHelper import saveSkeletonInfo, buildFromInfo

    lines, joints, motion_idx = parse_bvh(args.bvh_in)
    layout = channel_layout(joints)
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
    if args.max_frames > 0: rows = rows[:args.max_frames]
    print(f"Frames: {len(rows)}")

    si, rn, bi, *_ = saveSkeletonInfo(args.skel_xml)
    skel = buildFromInfo(si, rn)

    # Read skel XML to get bvh attrs per joint.
    import xml.etree.ElementTree as ET
    skel_tree = ET.parse(args.skel_xml)
    bvh_attr_for_body = {}
    for nd in skel_tree.getroot().findall("Node"):
        n = nd.attrib.get("name")
        j = nd.find("Joint")
        if j is not None and "bvh" in j.attrib:
            bvh_attr_for_body[n] = j.attrib["bvh"]

    # Compute skel rest body world rotations.
    skel.setPositions(np.zeros(skel.getNumDofs()))
    skel_rest_world = {}
    for i in range(skel.getNumBodyNodes()):
        bn = skel.getBodyNode(i)
        skel_rest_world[bn.getName()] = np.asarray(bn.getTransform().rotation()).copy()
    # For BVH-mapped joints, also store BVH→skel mapping.
    mappings = {}
    for sk_body, bv_suf in SKEL_TO_BVH.items():
        bvh_idx = find_bvh_joint(joints, bv_suf)
        if bvh_idx is None: continue
        mappings[sk_body] = bvh_idx
    print(f"Direct mappings: {len(mappings)} joints")

    # Spine vertebra fractional rotations from Spine/Spine1/Neck/Head BVH joints.
    spine_chains = {}
    for sk_body, (bv_suf, n_chain, idx_in_chain) in SPINE_BVH_FOR_SKEL.items():
        bvh_idx = find_bvh_joint(joints, bv_suf)
        if bvh_idx is None: continue
        spine_chains[sk_body] = (bvh_idx, n_chain)

    # Body name → DART body node + joint.
    body_to_joint = {}
    for i in range(skel.getNumBodyNodes()):
        bn = skel.getBodyNode(i)
        body_to_joint[bn.getName()] = (bn, bn.getParentJoint())

    # Topological order of skel joints (root first).
    skel_order = []
    visited = set()
    def visit(bn):
        name = bn.getName()
        if name in visited: return
        parent = bn.getParentBodyNode()
        if parent is not None: visit(parent)
        visited.add(name)
        skel_order.append(name)
    for i in range(skel.getNumBodyNodes()):
        visit(skel.getBodyNode(i))

    # Compute mocap_refs per frame.
    mocap = np.zeros((len(rows), skel.getNumDofs()))
    for fi, row in enumerate(rows):
        # BVH FK at this frame.
        Twr = bvh_fk(joints, row, n2c)

        pose = np.zeros(skel.getNumDofs())

        # Root translation: scale BVH root xyz to skel coords.
        root_j = skel.getJoint(0)
        if root_j.getNumDofs() == 6:
            chs = joints[0]["channels"]
            c0 = n2c[joints[0]["name"]][0]
            posv = np.zeros(3)
            for k, ch in enumerate(chs):
                chl = ch.lower()
                if chl == "xposition": posv[0] = row[c0 + k]
                elif chl == "yposition": posv[1] = row[c0 + k]
                elif chl == "zposition": posv[2] = row[c0 + k]
            if fi == 0:
                main.scale = 0.01 if np.linalg.norm(posv) > 10 else 1.0
                main.root_pos_f0 = posv.copy() * main.scale
            pose[3:6] = posv * main.scale - np.array([main.root_pos_f0[0], 0, main.root_pos_f0[2]])

        # Process joints in skel topological order.
        skel.setPositions(pose)
        for body_name in skel_order:
            if body_name not in body_to_joint: continue
            bn, sj = body_to_joint[body_name]

            # Determine target world rotation for this body.
            target_world = None
            if body_name in mappings:
                bvh_idx = mappings[body_name]
                target_world = Twr[bvh_idx][:3, :3]
            elif body_name in spine_chains:
                bvh_idx, n_chain = spine_chains[body_name]
                # Use parent BVH joint world rotation directly, fractional already handled by chain.
                # For spine vertebrae: each vertebra contributes a fraction. Use q^(1/n) of parent's local rotation.
                # Read parent's BVH local rotation from row.
                parent_bvh = joints[bvh_idx]
                chs = parent_bvh["channels"]
                rc = []
                if chs and parent_bvh["name"] in n2c:
                    c0, _ = n2c[parent_bvh["name"]]
                    pi = 0
                    for ch in chs:
                        v = row[c0 + pi]; pi += 1
                        if ch.lower().endswith("rotation"):
                            rc.append((ch[0].upper(), v))
                if rc:
                    R_parent_local_bvh = R.from_euler("".join(c for c, _ in rc),
                                                      [v for _, v in rc], degrees=True)
                    R_frac = R.from_rotvec(R_parent_local_bvh.as_rotvec() / n_chain)
                else:
                    R_frac = R.identity()
                # Get parent's world rotation in BVH chain.
                parent_world_bvh = Twr[joints[bvh_idx]["parent"]][:3, :3] if joints[bvh_idx]["parent"] >= 0 else np.eye(3)
                # Distribute: target = parent_world * R_frac^idx_in_chain.
                idx_in = SPINE_BVH_FOR_SKEL[body_name][2]
                target_world = parent_world_bvh @ R.from_rotvec(R_parent_local_bvh.as_rotvec() * (idx_in + 1) / n_chain).as_matrix() if rc else parent_world_bvh
            else:
                # No BVH driver — keep at rest.
                continue

            # Get current skel parent body world rotation.
            parent_bn = bn.getParentBodyNode()
            if parent_bn is None:
                parent_world_skel = np.eye(3)
            else:
                parent_world_skel = np.asarray(parent_bn.getTransform().rotation())

            # Body TL: at rest, body world / parent world = joint_TL * body_TL_rest.
            # Use skel REST body world and parent world to derive.
            body_rest_in_parent = skel_rest_world.get(parent_bn.getName(), np.eye(3)).T @ skel_rest_world[body_name] if parent_bn is not None else skel_rest_world[body_name]

            # Solve joint local rotation: target = parent_world_skel * R_local * body_rest_in_parent.
            # → R_local = parent_world_skel.T @ target_world @ body_rest_in_parent.T
            R_local = parent_world_skel.T @ target_world @ body_rest_in_parent.T

            # Write to pose.
            idx0 = sj.getIndexInSkeleton(0)
            nd = sj.getNumDofs()
            try:
                rotvec = R.from_matrix(R_local).as_rotvec()
            except Exception:
                rotvec = np.zeros(3)
            if nd == 6:
                pose[idx0:idx0 + 3] = rotvec
            elif nd == 3:
                pose[idx0:idx0 + 3] = rotvec
            elif nd == 1:
                axis = np.array(sj.getAxis(), dtype=np.float64, copy=True)
                axis = axis / max(np.linalg.norm(axis), 1e-12)
                pose[idx0] = float(np.dot(rotvec, axis))

            skel.setPositions(pose)

        mocap[fi] = pose
        if fi % 200 == 0: print(f"  frame {fi}/{len(rows)}")

    # Convert mocap (skel local rotations) → output BVH.
    # Use a SKEL-mirror BVH structure: write one joint per skel body, with
    # OFFSET and CHANNELS. Joints not driven by BVH have CHANNELS but zero rotations.
    out_hier_lines = ["HIERARCHY"]
    visited_emit = set()
    skel_to_bvh_name = {}  # skel body → emitted BVH joint name
    # Build hierarchy by traversing skel chain.
    def emit_node(bn, depth):
        body_name = bn.getName()
        # Use bvh attr if exists, else skel body name.
        name = bvh_attr_for_body.get(body_name, body_name)
        if body_name in visited_emit: return
        visited_emit.add(body_name)
        indent = "\t" * depth
        is_root = (bn.getParentBodyNode() is None)
        # OFFSET: difference between this body world rest position and parent body world rest position.
        if is_root:
            offset = np.zeros(3)
        else:
            offset = bn.getRelativeTransform().translation()
        if is_root:
            out_hier_lines.append(f"ROOT {name}")
            out_hier_lines.append("{")
            out_hier_lines.append(f"\tOFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
            out_hier_lines.append("\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
        else:
            out_hier_lines.append(indent + f"JOINT {name}")
            out_hier_lines.append(indent + "{")
            out_hier_lines.append(indent + f"\tOFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
            out_hier_lines.append(indent + "\tCHANNELS 3 Zrotation Yrotation Xrotation")
        skel_to_bvh_name[body_name] = name
        # Emit children.
        for ci in range(bn.getNumChildBodyNodes()):
            emit_node(bn.getChildBodyNode(ci), depth + 1)
        out_hier_lines.append(indent + "}" if not is_root else "}")

    # Build skel hierarchy traversal.
    root_bn = None
    for i in range(skel.getNumBodyNodes()):
        if skel.getBodyNode(i).getParentBodyNode() is None:
            root_bn = skel.getBodyNode(i); break
    skel.setPositions(np.zeros(skel.getNumDofs()))  # reset for clean offsets
    emit_node(root_bn, 0)

    # Compose motion section.
    out_lines = out_hier_lines + ["MOTION", f"Frames: {len(rows)}", f"Frame Time: 0.033333"]
    # For each frame, write channels in skel body emission order.
    # First channels: root 6 (xyz + rot zyx). Then each non-root joint 3 (rot zyx).
    # We need rotvec → ZYX Euler for each body.
    for fi in range(len(rows)):
        pose = mocap[fi]
        out_vals = []
        # Iterate visited_emit in insertion order (preserve hierarchy).
        # Re-traverse skel.
        def write_node(bn):
            name = bn.getName()
            sj = bn.getParentJoint()
            idx0 = sj.getIndexInSkeleton(0)
            nd = sj.getNumDofs()
            if nd == 6:
                # Root: translation + rotation.
                out_vals.extend([pose[idx0 + 3], pose[idx0 + 4], pose[idx0 + 5]])  # xyz
                rv = pose[idx0:idx0 + 3]
                Rm = R.from_rotvec(rv)
                eul = Rm.as_euler("ZYX", degrees=True)
                out_vals.extend([eul[0], eul[1], eul[2]])
            elif nd == 3:
                rv = pose[idx0:idx0 + 3]
                Rm = R.from_rotvec(rv)
                eul = Rm.as_euler("ZYX", degrees=True)
                out_vals.extend([eul[0], eul[1], eul[2]])
            elif nd == 1:
                axis = np.array(sj.getAxis(), dtype=np.float64, copy=True)
                axis = axis / max(np.linalg.norm(axis), 1e-12)
                Rm = R.from_rotvec(axis * pose[idx0])
                eul = Rm.as_euler("ZYX", degrees=True)
                out_vals.extend([eul[0], eul[1], eul[2]])
            else:
                # No DOF (Weld): emit zeros for 3 rotation channels (Euler 0,0,0).
                out_vals.extend([0.0, 0.0, 0.0])
            for ci in range(bn.getNumChildBodyNodes()):
                write_node(bn.getChildBodyNode(ci))
        write_node(root_bn)
        out_lines.append(" ".join(f"{v:.6f}" for v in out_vals))

    with open(args.bvh_out, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Wrote {args.bvh_out}")


if __name__ == "__main__":
    main()
