"""Per-frame dynamic retarget: for each BVH joint with a skel mapping,
compute target body WORLD rotation via BVH FK at frame f, then solve
the skel joint local rotation that places skel body at that world
rotation. Handles dynamic chain (parent rotates → child frame changes)
correctly by traversing skel in topological order with DART FK.

Output BVH has same hierarchy as input plus spine expansion, but joint
rotations replaced with computed skel-local equivalents. Loaded with
T_frame=None — no further correction needed.
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


def find_joint(joints, name):
    for i, j in enumerate(joints):
        if j["type"] in ("ROOT", "JOINT") and (j["name"] == name or j["name"].endswith("_" + name)):
            return i
    return None


def bvh_fk_world_rotations(joints, row, n2c, order_list):
    """Return dict joint_idx → world rotation matrix at this frame."""
    Twr = [None] * len(joints)
    for ji in order_list:
        j = joints[ji]
        Rl = R.identity()
        chs = j["channels"]
        if chs and j["name"] in n2c:
            c0, _ = n2c[j["name"]]; pi = 0; rc = []
            for ch in chs:
                v = row[c0 + pi]; pi += 1
                if ch.lower().endswith("rotation"):
                    rc.append((ch[0].upper(), v))
                else:
                    pi  # position channels skipped (rotation only)
            # Reset pi and re-scan, only rotation values matter
            pi2 = 0
            rc = []
            for ch in chs:
                if ch.lower().endswith("rotation"):
                    rc.append((ch[0].upper(), row[c0 + pi2]))
                pi2 += 1
            if rc:
                Rl = R.from_euler("".join(c for c, _ in rc), [v for _, v in rc], degrees=True)
        T = Rl.as_matrix()
        if j["parent"] >= 0 and Twr[j["parent"]] is not None:
            T = Twr[j["parent"]] @ T
        Twr[ji] = T
    return Twr


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
    if args.max_frames > 0:
        rows = rows[:args.max_frames]
    print(f"Frames: {len(rows)}")

    # Topological order of BVH joints.
    order_list = []
    def walk_bvh(i):
        order_list.append(i)
        for c in joints[i]["children"]: walk_bvh(c)
    for i, j in enumerate(joints):
        if j["parent"] == -1: walk_bvh(i)

    # Build skel.
    si, rn, bi, *_ = saveSkeletonInfo(args.skel_xml)
    skel = buildFromInfo(si, rn)
    print(f"Skel DOFs: {skel.getNumDofs()}, joints: {skel.getNumJoints()}")

    # Map skel body name → BVH joint suffix.
    skel_to_bvh = {
        "Saccrum_Coccyx0": "Hips",
        "L5": "L5", "L4": "L4", "L3": "L3", "L2": "L2", "L1": "L1",
        "T12": "T12", "T11": "T11", "T10": "T10", "T9": "T9", "T8": "T8",
        "T7": "T7", "T6": "T6", "T5": "T5", "T4": "T4", "T3": "T3",
        "T2": "T2", "T1": "T1",
        "C7": "C7", "C6": "C6", "C5": "C5", "C4": "C4", "C3": "C3",
        "Axis0": "Axis0", "Atlas0": "Atlas0", "Head0": "Head",
        "L_Clavicle0": "LeftShoulder", "L_Humerus0": "LeftArm",
        "L_Ulna0": "LeftForeArm", "L_Carpal0": "LeftHand",
        "R_Clavicle0": "RightShoulder", "R_Humerus0": "RightArm",
        "R_Ulna0": "RightForeArm", "R_Carpal0": "RightHand",
        "L_Femur0": "LeftUpLeg", "L_Tibia_Fibula0": "LeftLeg",
        "L_Talus0": "LeftFoot", "L_Toe10": "LeftToeBase",
        "R_Femur0": "RightUpLeg", "R_Tibia_Fibula0": "RightLeg",
        "R_Talus0": "RightFoot", "R_Toe10": "RightToeBase",
    }
    # Strip "0" suffix variants? L5 vs L50.
    extra = {}
    for sk, bv in skel_to_bvh.items():
        if sk.endswith("0") and not sk.endswith("00"):
            extra[sk[:-1] + "0"] = bv  # leave as is for "L50"
    # Find skel body name → DART body. Some skel bodies have "0" suffix differently.

    # Build BVH joint index lookup.
    bvh_lookup = {}  # name → index
    for ji, j in enumerate(joints):
        bvh_lookup[j["name"]] = ji
    def find_bvh_joint(suf):
        for ji, j in enumerate(joints):
            if j["type"] in ("ROOT", "JOINT") and (j["name"] == suf or j["name"].endswith("_" + suf)):
                return ji
        return None

    # Skel joint → bvh joint idx mapping.
    skel_to_bvh_idx = {}
    skel_body_objs = {}
    for sk_name, bv_suf in skel_to_bvh.items():
        # Skel body name candidates (with/without trailing 0).
        for candidate in [sk_name, sk_name + "0", sk_name[:-1] if sk_name.endswith("0") else sk_name]:
            bn = skel.getBodyNode(candidate)
            if bn is not None:
                skel_body_objs[candidate] = bn
                bvh_idx = find_bvh_joint(bv_suf)
                if bvh_idx is not None:
                    skel_to_bvh_idx[candidate] = bvh_idx
                break
    print(f"Mapped {len(skel_to_bvh_idx)} skel→bvh joints")

    # Skel topological order via DART (joint index).
    skel_joint_order = []
    for ji in range(skel.getNumJoints()):
        sj = skel.getJoint(ji)
        bn = sj.getChildBodyNode()
        skel_joint_order.append((ji, sj, bn))

    # For each frame, compute skel pose.
    new_dofs = np.zeros((len(rows), skel.getNumDofs()))
    for fi, row in enumerate(rows):
        # BVH world rotations.
        bvh_wr = bvh_fk_world_rotations(joints, row, n2c, order_list)

        # Reset skel.
        pose = np.zeros(skel.getNumDofs())
        skel.setPositions(pose)

        # Root translation: BVH root xyz → skel root translation.
        root_j = skel.getJoint(0)
        if root_j.getNumDofs() == 6:
            root_chs = joints[0]["channels"]
            c0 = n2c[joints[0]["name"]][0]
            pos_idx = {"xposition": None, "yposition": None, "zposition": None}
            for k, ch in enumerate(root_chs):
                chl = ch.lower()
                if chl in pos_idx: pos_idx[chl] = k
            if all(v is not None for v in pos_idx.values()):
                # Use BVH cm → meter scale (auto-detect via magnitude)
                bvh_root_pos = np.array([
                    row[c0 + pos_idx["xposition"]],
                    row[c0 + pos_idx["yposition"]],
                    row[c0 + pos_idx["zposition"]],
                ])
                if fi == 0:
                    main.scale_factor = 0.01 if np.linalg.norm(bvh_root_pos) > 10 else 1.0
                pose[3:6] = bvh_root_pos * main.scale_factor

        # Traverse skel joints in order, set each one.
        for ji, sj, bn in skel_joint_order:
            sj_name = sj.getName()
            bn_name = bn.getName() if bn else sj_name
            if bn_name not in skel_to_bvh_idx:
                continue
            bvh_idx = skel_to_bvh_idx[bn_name]
            target_world = bvh_wr[bvh_idx]
            # Get parent body world rotation (from skel after current pose).
            parent_bn = bn.getParentBodyNode()
            if parent_bn is None:
                parent_world = np.eye(3)
            else:
                parent_world = np.asarray(parent_bn.getTransform().rotation())
            # Solve joint local rotation: R_local = parent_world.T @ target_world @ body_TL.T
            # body_TL = body's rest rotation in joint frame.
            body_TL = np.asarray(bn.getRelativeTransform().rotation())
            # body's REST rotation: when joint_local = identity, body world = parent_world * joint_TL * body_TL_rest
            # We computed parent_world at current state. We want body_world = target_world.
            # body_world = parent_world * R_local * body_TL_rest
            # → R_local = parent_world.T * target_world * body_TL_rest.T
            # But body_TL_rest changes per skel pose? No — it's fixed (body's transform in joint frame at rest).
            # DART's getRelativeTransform gives current relative transform (post joint rotation).
            # At rest (joint_local=identity), this = body's rest TL.
            # Since we've set joint_local=identity so far (zeros), getRelativeTransform = body's rest TL. OK.
            R_local_target = parent_world.T @ target_world @ body_TL.T

            # Set in skel pose.
            idx0 = sj.getIndexInSkeleton(0)
            nd = sj.getNumDofs()
            try:
                rotvec = R.from_matrix(R_local_target).as_rotvec()
            except Exception:
                rotvec = np.zeros(3)
            if nd == 6:
                pose[idx0:idx0 + 3] = rotvec
            elif nd == 3:
                pose[idx0:idx0 + 3] = rotvec
            elif nd == 1:
                # Revolute: project onto axis.
                axis = np.array(sj.getAxis(), dtype=np.float64, copy=True)
                axis = axis / max(np.linalg.norm(axis), 1e-12)
                pose[idx0] = float(np.dot(rotvec, axis))

            skel.setPositions(pose)

        new_dofs[fi] = pose
        if fi % 200 == 0:
            print(f"  frame {fi}/{len(rows)}")

    # Convert new DOFs back into BVH joint Eulers.
    # For each BVH joint that has a mapping, write its rotation from corresponding skel joint local.
    # First, build reverse map: bvh_joint_idx → skel body name.
    bvh_idx_to_skel = {v: k for k, v in skel_to_bvh_idx.items()}
    for fi, row in enumerate(rows):
        for bvh_idx, sk_body in bvh_idx_to_skel.items():
            j = joints[bvh_idx]
            chs = j["channels"]
            rot_chans = [c for c in chs if c.lower().endswith("rotation")]
            if not rot_chans: continue
            order = "".join(c[0] for c in rot_chans).upper()
            c0 = n2c[j["name"]][0]
            rot_offsets = [k for k, ch in enumerate(chs) if ch.lower().endswith("rotation")]
            bn = skel_body_objs[sk_body]
            sj = bn.getParentJoint()
            idx0 = sj.getIndexInSkeleton(0)
            nd = sj.getNumDofs()
            if nd >= 3:
                rv = new_dofs[fi, idx0:idx0 + 3]
                Rmat = R.from_rotvec(rv)
            elif nd == 1:
                ax = np.array(sj.getAxis(), dtype=np.float64, copy=True)
                ax = ax / max(np.linalg.norm(ax), 1e-12)
                Rmat = R.from_rotvec(ax * new_dofs[fi, idx0])
            else:
                continue
            eul = Rmat.as_euler(order, degrees=True)
            for k, off in enumerate(rot_offsets):
                row[c0 + off] = float(eul[k])

    out_lines = lines[:motion_idx]
    # Rebuild header with correct frame count
    for h in header:
        if h.strip().startswith("Frames"):
            out_lines.append(f"Frames: {len(rows)}")
        else:
            out_lines.append(h)
    for row in rows:
        out_lines.append(" ".join(f"{v:.6f}" for v in row))
    with open(args.bvh_out, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Wrote {args.bvh_out}")


if __name__ == "__main__":
    main()
