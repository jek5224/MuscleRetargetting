"""General retarget: any BVH → BVH driving zygote_skel.xml.

Pipeline:
1. Normalize joint names to canonical (Character1_Hips, Spine, Spine1, Neck,
   Head, LeftShoulder, LeftArm, LeftForeArm, LeftHand, etc.).
2. Collapse extra spine joints (Spine2 etc.) into Spine1 (rotation composed).
3. Run vert expansion (lumbar/thoracic/cervical/cranial chains via q^(1/n)).
4. Add sternum joint baked via Kabsch.
5. Arm retarget (frame-0 subtract for Shoulder/Arm).
6. Forearm retarget (IK ulna for wrist position + radius offset for palm).

Usage:
    python tools/retarget_bvh.py --in any.bvh --out any_retargeted.bvh
"""
import argparse
import os
import re
import sys
import subprocess
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation as R


# Canonical names expected by downstream pipeline.
CANONICAL = {
    "hips": "Character1_Hips",
    "spine": "Character1_Spine",
    "spine1": "Character1_Spine1",
    "neck": "Character1_Neck",
    "head": "Character1_Head",
    "leftshoulder": "Character1_LeftShoulder",
    "leftarm": "Character1_LeftArm",
    "leftforearm": "Character1_LeftForeArm",
    "lefthand": "Character1_LeftHand",
    "rightshoulder": "Character1_RightShoulder",
    "rightarm": "Character1_RightArm",
    "rightforearm": "Character1_RightForeArm",
    "righthand": "Character1_RightHand",
    "leftupleg": "Character1_LeftUpLeg",
    "leftleg": "Character1_LeftLeg",
    "leftfoot": "Character1_LeftFoot",
    "lefttoebase": "Character1_LeftToeBase",
    "rightupleg": "Character1_RightUpLeg",
    "rightleg": "Character1_RightLeg",
    "rightfoot": "Character1_RightFoot",
    "righttoebase": "Character1_RightToeBase",
}


def _norm_key(name):
    """Lower-case, strip prefix, strip trailing digits/underscores."""
    n = name
    for pref in ("Character1_", "mixamorig:", "mixamorig_", "Bip01_", "Bip001_"):
        if n.startswith(pref):
            n = n[len(pref):]
    n = n.lower().replace("_", "").replace(" ", "")
    # Map "toe" → "toebase" if not already.
    if n == "lefttoe": n = "lefttoebase"
    if n == "righttoe": n = "righttoebase"
    return n


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
                 "children": [], "channels": [], "offset": None,
                 "line_open": i, "line_close": None,
                 "indent": len(ln) - len(ln.lstrip("\t "))}
            joints.append(j)
            new_idx = len(joints) - 1
            if stack: joints[stack[-1]]["children"].append(new_idx)
            stack.append(new_idx)
            pending_type = None; pending_name = None
            continue
        if s == "}":
            if stack:
                joints[stack[-1]]["line_close"] = i
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
        for c in joints[i]["children"]: walk(c)
    for i, j in enumerate(joints):
        if j["parent"] == -1: walk(i)
    return out


def build_name_map(joints):
    """Map source joint name → canonical (Character1_*) name."""
    name_map = {}
    for j in joints:
        if j["type"] not in ("ROOT", "JOINT"): continue
        k = _norm_key(j["name"])
        if k in CANONICAL:
            name_map[j["name"]] = CANONICAL[k]
    return name_map


def merge_extra_spines(joints, frame_rows, src_layout):
    """If hierarchy has Spine2 (between Spine1 and Neck), collapse its
    rotation into Spine1 and remove Spine2 joint. Returns (new joints,
    new frame rows, new layout, removed_indices).
    """
    # Find Spine1 and Spine2 by canonical key.
    spine1_idx = None
    spine2_idx = None
    for i, j in enumerate(joints):
        k = _norm_key(j["name"])
        if k == "spine1": spine1_idx = i
        elif k == "spine2": spine2_idx = i
    if spine2_idx is None:
        return joints, frame_rows, src_layout
    if spine1_idx is None:
        # No Spine1: rename Spine2 → Spine1
        joints[spine2_idx]["name"] = CANONICAL["spine1"]
        return joints, frame_rows, src_layout

    # Compose Spine2's rotation into Spine1, then remove Spine2.
    col_lookup = {}
    col = 0
    for ji, n in src_layout: col_lookup[joints[ji]["name"]] = (col, n); col += n
    s1_c, s1_n = col_lookup[joints[spine1_idx]["name"]]
    s2_c, s2_n = col_lookup[joints[spine2_idx]["name"]]
    s1_order = "".join(c[0] for c in joints[spine1_idx]["channels"]
                       if c.lower().endswith("rotation")).upper()
    s2_order = "".join(c[0] for c in joints[spine2_idx]["channels"]
                       if c.lower().endswith("rotation")).upper()
    for row in frame_rows:
        R1 = R.from_euler(s1_order, row[s1_c:s1_c+s1_n], degrees=True)
        R2 = R.from_euler(s2_order, row[s2_c:s2_c+s2_n], degrees=True)
        Rc = R1 * R2
        row[s1_c:s1_c+s1_n] = Rc.as_euler(s1_order, degrees=True).tolist()

    # Move Spine2's offset into Spine1 (sum) and reparent Spine2's children.
    spine2 = joints[spine2_idx]
    spine1 = joints[spine1_idx]
    new_off = tuple(a + b for a, b in zip(spine1["offset"], spine2["offset"]))
    spine1["offset"] = new_off
    # Reparent Spine2's children to Spine1.
    for ch_idx in spine2["children"]:
        joints[ch_idx]["parent"] = spine1_idx
        spine1["children"].append(ch_idx)
    # Remove Spine2 from Spine1's children.
    spine1["children"].remove(spine2_idx)
    # Mark Spine2 as deleted (will skip in output).
    spine2["_deleted"] = True
    return joints, frame_rows, src_layout


def rename_and_normalize(joints, name_map):
    """Rename joints according to name_map. Add prefix to children for consistency."""
    for j in joints:
        if j["name"] in name_map:
            j["name"] = name_map[j["name"]]
    return joints


def write_bvh(out_path, joints, motion_idx_orig, original_lines, frame_rows,
              src_layout):
    """Reconstruct BVH file from joints + frame_rows. Skips _deleted joints."""
    out_lines = ["HIERARCHY"]

    def emit_node(idx, depth):
        j = joints[idx]
        if j.get("_deleted"): return
        indent = "\t" * depth
        if j["type"] == "ROOT":
            out_lines.append(f"ROOT {j['name']}")
            out_lines.append("{")
            out_lines.append(f"\tOFFSET {j['offset'][0]} {j['offset'][1]} {j['offset'][2]}")
            out_lines.append(f"\tCHANNELS {len(j['channels'])} " + " ".join(j["channels"]))
            for c in j["children"]: emit_node(c, 1)
            out_lines.append("}")
        elif j["type"] == "End":
            out_lines.append(indent + "End Site")
            out_lines.append(indent + "{")
            out_lines.append(indent + f"\tOFFSET {j['offset'][0]} {j['offset'][1]} {j['offset'][2]}")
            out_lines.append(indent + "}")
        else:
            out_lines.append(indent + f"JOINT {j['name']}")
            out_lines.append(indent + "{")
            out_lines.append(indent + f"\tOFFSET {j['offset'][0]} {j['offset'][1]} {j['offset'][2]}")
            if j["channels"]:
                out_lines.append(indent + f"\tCHANNELS {len(j['channels'])} " + " ".join(j["channels"]))
            for c in j["children"]: emit_node(c, depth + 1)
            out_lines.append(indent + "}")

    for i, j in enumerate(joints):
        if j["parent"] == -1: emit_node(i, 0)

    # Motion header: preserve from original.
    motion_header = []
    for ln in original_lines[motion_idx_orig:]:
        s = ln.strip()
        if s.startswith("MOTION") or s.startswith("Frames") or s.startswith("Frame Time"):
            motion_header.append(ln)
        else:
            break
    out_lines.extend(motion_header)

    # Build new column layout (matches reorganized hierarchy).
    new_layout = []
    def walk(i):
        if joints[i].get("_deleted"): return
        if joints[i]["channels"]:
            new_layout.append((i, len(joints[i]["channels"])))
        for c in joints[i]["children"]: walk(c)
    for i, j in enumerate(joints):
        if j["parent"] == -1: walk(i)

    # Map old layout col → new col by joint identity (joint indices unchanged).
    old_col_lookup = {}
    col = 0
    for ji, n in src_layout:
        old_col_lookup[ji] = (col, n); col += n
    for row in frame_rows:
        new_row = []
        for ji, n in new_layout:
            if ji in old_col_lookup:
                c, k = old_col_lookup[ji]
                new_row.extend(row[c:c+k])
            else:
                new_row.extend([0.0] * n)
        out_lines.append(" ".join(f"{v:.6f}" for v in new_row))

    with open(out_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")


def _fk_world_positions(joints, frame_row, src_layout, target_names):
    pos_by_name = {}
    col_lookup = {}
    col = 0
    for ji, n in src_layout: col_lookup[joints[ji]["name"]] = (col, n); col += n
    order_idx = []
    def walk(i):
        order_idx.append(i)
        for c in joints[i]["children"]: walk(c)
    for i, j in enumerate(joints):
        if j["parent"] == -1: walk(i)
    Twr = [None] * len(joints)
    for ji in order_idx:
        j = joints[ji]
        off = np.array(j["offset"]) if j["offset"] else np.zeros(3)
        Rl = R.identity(); pos_add = np.zeros(3)
        if j["channels"] and j["name"] in col_lookup:
            c0, _ = col_lookup[j["name"]]; pi = 0; rc = []
            for ch in j["channels"]:
                v = frame_row[c0 + pi]; pi += 1
                if ch.lower() == "xposition": pos_add[0] = v
                elif ch.lower() == "yposition": pos_add[1] = v
                elif ch.lower() == "zposition": pos_add[2] = v
                else: rc.append((ch[0].upper(), v))
            if rc: Rl = R.from_euler("".join(c for c, _ in rc), [v for _, v in rc], degrees=True)
        T = np.eye(4); T[:3, :3] = Rl.as_matrix(); T[:3, 3] = off + pos_add
        if j["parent"] >= 0: T = Twr[j["parent"]] @ T
        Twr[ji] = T
    for tn in target_names:
        for i, j in enumerate(joints):
            if j["name"] == tn or j["name"].endswith("_" + tn):
                pos_by_name[tn] = Twr[i][:3, 3].copy(); break
    return pos_by_name


def _R_align_vec(v_from, v_to):
    a = v_from / max(np.linalg.norm(v_from), 1e-12)
    b = v_to / max(np.linalg.norm(v_to), 1e-12)
    c = float(np.dot(a, b))
    if c > 0.99999: return R.identity()
    if c < -0.99999:
        ax = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, np.array([0.0, 0.0, 1.0]))
        ax /= np.linalg.norm(ax)
        return R.from_rotvec(ax * np.pi)
    ax = np.cross(a, b)
    ax /= np.linalg.norm(ax)
    return R.from_rotvec(ax * np.arccos(c))


CANONICAL_TO_SKEL = {
    # Upper body only — lower body (legs) already aligns naturally with
    # skel via direct joint mapping. Conjugating legs breaks working baseline.
    "Character1_Spine": "L50",
    "Character1_Spine1": "T120",
    "Character1_Neck": "C70",
    "Character1_Head": "Head0",
    "Character1_LeftShoulder": "L_Clavicle0",
    "Character1_LeftArm": "L_Humerus0",
    "Character1_LeftForeArm": "L_Ulna0",
    "Character1_LeftHand": "L_Carpal0",
    "Character1_RightShoulder": "R_Clavicle0",
    "Character1_RightArm": "R_Humerus0",
    "Character1_RightForeArm": "R_Ulna0",
    "Character1_RightHand": "R_Carpal0",
}


def compute_rest_alignment(joints, src_layout, skel_xml):
    """Per joint, compute R_align mapping BVH-local rest bone direction
    to skel-joint-local rest bone direction. Used to conjugate per-frame
    BVH local rotations into skel-compatible rotations."""
    from core.dartHelper import saveSkeletonInfo, buildFromInfo
    si, rn, *_ = saveSkeletonInfo(skel_xml)
    skel = buildFromInfo(si, rn)

    def _R_world_of(body_name):
        bn = skel.getBodyNode(body_name)
        if bn is None: return np.eye(3)
        return np.asarray(bn.getTransform().rotation())

    def _world_pos(body_name):
        bn = skel.getBodyNode(body_name)
        if bn is None: return np.zeros(3)
        return np.asarray(bn.getTransform().translation())

    # Build name lookup tolerant of Character1_-prefixed and bare names.
    name_to_skel = {}
    for canon, sb in CANONICAL_TO_SKEL.items():
        name_to_skel[canon] = sb
        # bare name (strip Character1_ prefix)
        if canon.startswith("Character1_"):
            name_to_skel[canon[len("Character1_"):]] = sb
    R_align = {}
    for j_idx, j in enumerate(joints):
        if j["type"] not in ("ROOT", "JOINT"):
            continue
        skel_body = name_to_skel.get(j["name"])
        if skel_body is None:
            R_align[j_idx] = R.identity()
            continue
        # R_align_self = PARENT body world rest rotation (= skel joint
        # frame at rest, since joint frame = parent body frame).
        bn = skel.getBodyNode(skel_body)
        if bn is None:
            R_align[j_idx] = R.identity(); continue
        pn = bn.getParentBodyNode()
        if pn is None:
            R_align[j_idx] = R.identity()
        else:
            Rp = np.asarray(pn.getTransform().rotation())
            R_align[j_idx] = R.from_matrix(Rp)
        print(f"  align {j['name']:30s} → {skel_body:20s} parent_rest_rotvec_deg={np.degrees(R_align[j_idx].as_rotvec()).round(1)}")
    return R_align


def apply_rest_alignment(joints, frame_rows, src_layout, R_align):
    """For each frame, conjugate joint local rotations:
    R_skel = R_align_parent^-1 * R_bvh * R_align_self.
    Skip joints not in R_align (treated as identity alignment)."""
    col_lookup = {}
    col = 0
    for ji, n in src_layout: col_lookup[ji] = (col, n); col += n
    for row in frame_rows:
        for ji, n in src_layout:
            chs = joints[ji]["channels"]
            rot_chans = [c for c in chs if c.lower().endswith("rotation")]
            if len(rot_chans) != 3:
                continue
            order = "".join(c[0] for c in rot_chans).upper()
            c, _ = col_lookup[ji]
            rot_offsets = [k for k, ch in enumerate(chs) if ch.lower().endswith("rotation")]
            e_f = [row[c + k] for k in rot_offsets]
            R_bvh = R.from_euler(order, e_f, degrees=True)
            R_self = R_align.get(ji, R.identity())
            parent_idx = joints[ji]["parent"]
            R_parent = R_align.get(parent_idx, R.identity()) if parent_idx >= 0 else R.identity()
            # Self-conjugation: R_skel = R_self^-1 * R_bvh * R_self
            # (converts BVH local rotation to skel joint-local rotation,
            # both expressed in their own joint frames).
            R_new = R_self.inv() * R_bvh * R_self
            new_e = R_new.as_euler(order, degrees=True).tolist()
            for k, off in enumerate(rot_offsets):
                row[c + off] = new_e[k]


def align_up_vector(joints, frame_rows, src_layout):
    """Compute actor's up direction at f0 (Hips→Head). Rotate root rotation
    and translation per-frame so up = world +Y, putting actor upright."""
    if not frame_rows: return frame_rows
    wp = _fk_world_positions(joints, frame_rows[0], src_layout, ["Hips", "Head"])
    if "Hips" not in wp or "Head" not in wp:
        print("[normalize] Cannot find Hips/Head for up alignment, skipping")
        return frame_rows
    up_actor = wp["Head"] - wp["Hips"]
    R_corr = _R_align_vec(up_actor, np.array([0.0, 1.0, 0.0]))
    print(f"[normalize] Up vector correction: actor up={up_actor.round(2)} → world +Y")
    print(f"             R_correction rotvec deg={np.degrees(R_corr.as_rotvec()).round(2)}")
    if np.allclose(R_corr.as_matrix(), np.eye(3), atol=1e-6):
        return frame_rows

    # Find root joint and its channels.
    root_idx = next(i for i, j in enumerate(joints) if j["parent"] == -1)
    col_lookup = {}
    col = 0
    for ji, n in src_layout: col_lookup[ji] = (col, n); col += n
    rc_idx, _ = col_lookup[root_idx]
    chs = joints[root_idx]["channels"]
    pos_idx = {ch.lower(): k for k, ch in enumerate(chs) if ch.lower().endswith("position")}
    rot_offsets = [k for k, ch in enumerate(chs) if ch.lower().endswith("rotation")]
    rot_chans = [chs[k] for k in rot_offsets]
    rot_order = "".join(c[0] for c in rot_chans).upper()

    # Capture f0 position to subtract (so actor starts near world origin).
    p0_raw = np.array([frame_rows[0][rc_idx + pos_idx["xposition"]],
                       frame_rows[0][rc_idx + pos_idx["yposition"]],
                       frame_rows[0][rc_idx + pos_idx["zposition"]]])
    p0_corrected = R_corr.apply(p0_raw)
    # Keep f0 Y (so actor stands on its rest height); zero X/Z so actor at origin.
    p0_subtract = np.array([p0_corrected[0], 0.0, p0_corrected[2]])
    for row in frame_rows:
        pos = np.array([row[rc_idx + pos_idx["xposition"]],
                        row[rc_idx + pos_idx["yposition"]],
                        row[rc_idx + pos_idx["zposition"]]])
        new_pos = R_corr.apply(pos) - p0_subtract
        row[rc_idx + pos_idx["xposition"]] = new_pos[0]
        row[rc_idx + pos_idx["yposition"]] = new_pos[1]
        row[rc_idx + pos_idx["zposition"]] = new_pos[2]
        eul = [row[rc_idx + k] for k in rot_offsets]
        R_root = R.from_euler(rot_order, eul, degrees=True)
        R_new = R_corr * R_root
        new_eul = R_new.as_euler(rot_order, degrees=True).tolist()
        for k, off in enumerate(rot_offsets):
            row[rc_idx + off] = new_eul[k]
    return frame_rows


def subtract_frame0_all(joints, frame_rows, src_layout):
    """Subtract frame 0's local rotation from ALL non-root joints (sets
    rest pose = first frame pose). Necessary for BVHs with bone-aligned
    local frames (LaFAN, Mixamo) where rotation channels don't represent
    rotation around world axes."""
    if not frame_rows:
        return frame_rows
    col_lookup = {}
    col = 0
    for ji, n in src_layout: col_lookup[ji] = (col, n); col += n
    R0_per_joint = {}
    for ji, n in src_layout:
        chs = joints[ji]["channels"]
        rot_chans = [c for c in chs if c.lower().endswith("rotation")]
        if len(rot_chans) != 3:
            continue
        order = "".join(c[0] for c in rot_chans).upper()
        c, _ = col_lookup[ji]
        rot_offsets = [k for k, ch in enumerate(chs) if ch.lower().endswith("rotation")]
        e0 = [frame_rows[0][c + k] for k in rot_offsets]
        R0_per_joint[ji] = (R.from_euler(order, e0, degrees=True), order, c, rot_offsets)
    for row in frame_rows:
        for ji, (R0, order, c, rot_offsets) in R0_per_joint.items():
            e_f = [row[c + k] for k in rot_offsets]
            R_f = R.from_euler(order, e_f, degrees=True)
            R_new = R0.inv() * R_f
            new_e = R_new.as_euler(order, degrees=True).tolist()
            for k, off in enumerate(rot_offsets):
                row[c + off] = new_e[k]
    print(f"[normalize] Subtracted frame-0 rotation from {len(R0_per_joint)} joints")
    return frame_rows


def compensate_scapula(joints, frame_rows, src_layout, skel_xml):
    """Skel chain Clavicle → Scapula → Humerus. L_Scapula0 has no bvh attr,
    so it stays at rest rotation. Its body Trans linear is non-identity,
    introducing a rest-pose rotation offset between Clavicle and Humerus
    frames. BVH LeftArm rotation (expressed in LeftShoulder frame) needs
    conjugation by R_Scapula_body_TL^-1 to apply correctly in L_Humerus
    parent frame (= L_Scapula body frame)."""
    import xml.etree.ElementTree as ET
    tr = ET.parse(skel_xml)
    scap_tl = {}
    for nd in tr.getroot().findall("Node"):
        n = nd.attrib.get("name")
        if n not in ("L_Scapula0", "R_Scapula0"): continue
        body = nd.find("Body")
        if body is None: continue
        t = body.find("Transformation")
        if t is None: continue
        arr = [float(x) for x in t.attrib.get("linear", "1 0 0 0 1 0 0 0 1").split()]
        M = np.array(arr).reshape(3, 3)
        scap_tl[n] = M
    if not scap_tl:
        return frame_rows

    side_to_scap = {"LeftArm": "L_Scapula0", "RightArm": "R_Scapula0"}
    col_lookup = {}
    col = 0
    for ji, n in src_layout: col_lookup[ji] = (col, n); col += n
    n_modified = 0
    for j_idx, j in enumerate(joints):
        for suf in ("LeftArm", "RightArm"):
            if j["name"] == suf or j["name"].endswith("_" + suf):
                scap_M = scap_tl.get(side_to_scap[suf])
                if scap_M is None: continue
                chs = j["channels"]
                rot_chans = [c for c in chs if c.lower().endswith("rotation")]
                if len(rot_chans) != 3: continue
                order = "".join(c[0] for c in rot_chans).upper()
                c0, _ = col_lookup[j_idx]
                rot_offsets = [k for k, ch in enumerate(chs) if ch.lower().endswith("rotation")]
                R_scap = R.from_matrix(scap_M)
                for row in frame_rows:
                    e_f = [row[c0 + k] for k in rot_offsets]
                    R_old = R.from_euler(order, e_f, degrees=True)
                    R_new = R_scap.inv() * R_old * R_scap
                    new_e = R_new.as_euler(order, degrees=True).tolist()
                    for k, off in enumerate(rot_offsets):
                        row[c0 + off] = new_e[k]
                n_modified += 1
                print(f"[normalize] Scapula-compensated {j['name']} via {side_to_scap[suf]}")
    return frame_rows


def normalize_bvh(in_path, out_path, subtract_f0=False, rest_align=True,
                  scapula_compensate=False, skel_xml="data/zygote_skel.xml"):
    """Normalize input BVH: rename joints to canonical, collapse Spine2,
    optionally subtract frame-0 rotations to put rest pose at first
    motion frame (handles bone-aligned local frame conventions)."""
    lines, joints, motion_idx = parse_bvh(in_path)
    layout = channel_layout(joints)
    rows = []
    for ln in lines[motion_idx:]:
        s = ln.strip()
        if not s or s.startswith("MOTION") or s.startswith("Frames") or s.startswith("Frame Time"):
            continue
        rows.append([float(x) for x in s.split()])

    name_map = build_name_map(joints)
    print(f"[normalize] Mapped {len(name_map)} joint names to canonical")

    joints, rows, layout = merge_extra_spines(joints, rows, layout)
    print("[normalize] Collapsed extra spine joints (if any)")

    # Only subtract root XZ translation at f0 so actor starts near world
    # origin (otherwise far world positions can make skel appear to fly
    # away). Keep Y so actor stays at correct height. Don't touch rotations
    # — viewer's MyBVH T_frame=0 detection handles BVH-side T-pose correction.
    if rows:
        root_idx = next(i for i, j in enumerate(joints) if j["parent"] == -1)
        col_lookup = {}
        col = 0
        for ji, n in layout: col_lookup[ji] = (col, n); col += n
        rc_idx, _ = col_lookup[root_idx]
        chs = joints[root_idx]["channels"]
        pos_offs = {}
        for k, ch in enumerate(chs):
            chl = ch.lower()
            if chl in ("xposition", "yposition", "zposition"):
                pos_offs[chl] = k
        if "xposition" in pos_offs and "zposition" in pos_offs:
            x0 = rows[0][rc_idx + pos_offs["xposition"]]
            z0 = rows[0][rc_idx + pos_offs["zposition"]]
            for row in rows:
                row[rc_idx + pos_offs["xposition"]] -= x0
                row[rc_idx + pos_offs["zposition"]] -= z0
    if subtract_f0:
        rows = subtract_frame0_all(joints, rows, layout)
    if scapula_compensate:
        rows = compensate_scapula(joints, rows, layout, skel_xml)

    # Don't rename to Character1_*. Keep original names so viewer's
    # _detect_bvh_tframe regex (JOINT LeftLeg ...) can match. MyBVH's
    # auto-remap handles Character1_-prefixed skel bvh attrs by stripping
    # prefix and looking up base names in BVH joints.
    # joints = rename_and_normalize(joints, name_map)
    print(f"[normalize] Skipping rename (preserving source joint names for viewer T_frame detection)")

    if rest_align:
        print("[normalize] Computing rest pose alignment per joint...")
        R_align = compute_rest_alignment(joints, layout, skel_xml)
        apply_rest_alignment(joints, rows, layout, R_align)
        print(f"[normalize] Applied conjugation to {sum(1 for v in R_align.values() if not np.allclose(v.as_matrix(), np.eye(3), atol=1e-6))} non-identity alignments")

    write_bvh(out_path, joints, motion_idx, lines, rows, layout)
    print(f"[normalize] Wrote {out_path}")


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout); print(res.stderr); sys.exit(f"Failed: {cmd}")
    print(res.stdout.splitlines()[-1] if res.stdout else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="bvh_in", required=True)
    ap.add_argument("--out", dest="bvh_out", required=True)
    ap.add_argument("--no-pipeline", action="store_true",
                    help="Only normalize; skip downstream sternum/arm/forearm.")
    ap.add_argument("--minimal", action="store_true",
                    help="Minimal pipeline: just normalize + spine expand. Skip sternum/arm/forearm bake (for LaFAN-style BVHs where viewer's T_frame=0 handles convention).")
    ap.add_argument("--subtract-f0", action="store_true",
                    help="Subtract f0 rotations (zero out first-frame pose). Off by default — preserves source motion intact.")
    ap.add_argument("--no-rest-align", action="store_true",
                    help="Skip per-joint rest-pose alignment via conjugation. On by default for upper-body joints (clavicle, arm, hand).")
    ap.add_argument("--no-ik-ulna", action="store_true",
                    help="Skip IK ulna step in forearm bake.")
    ap.add_argument("--l-radius-offset-deg", type=float, default=90.0)
    ap.add_argument("--r-radius-offset-deg", type=float, default=-90.0)
    args = ap.parse_args()

    workdir = tempfile.mkdtemp(prefix="retarget_")
    base = os.path.splitext(os.path.basename(args.bvh_in))[0]
    norm = os.path.join(workdir, f"{base}_normalized.bvh")
    vert = os.path.join(workdir, f"{base}_vert.bvh")
    stern = os.path.join(workdir, f"{base}_vert_sternum.bvh")
    armed = os.path.join(workdir, f"{base}_vert_sternum_arm.bvh")

    normalize_bvh(args.bvh_in, norm, subtract_f0=args.subtract_f0,
                  rest_align=not args.no_rest_align)
    if args.no_pipeline:
        os.replace(norm, args.bvh_out)
        print(f"Wrote {args.bvh_out}")
        return

    run(["python", "tools/make_run_vert_bvh.py", "--in", norm, "--out", vert,
         "--skip-xml"])
    if args.minimal:
        os.replace(vert, args.bvh_out)
        print(f"\nRetarget (minimal) complete: {args.bvh_out}")
        return
    run(["python", "tools/add_sternum_to_bvh.py", "--in", vert, "--out", stern])
    run(["python", "tools/bake_arm_retarget_bvh.py", "--in", stern, "--out", armed])
    fa_cmd = ["python", "tools/bake_forearm_retarget_bvh.py",
              "--in", armed, "--out", args.bvh_out, "--skip-xml",
              "--no-frame0-subtract",
              "--l-radius-offset-deg", str(args.l_radius_offset_deg),
              "--r-radius-offset-deg", str(args.r_radius_offset_deg),
              "--orig-bvh", norm]
    if not args.no_ik_ulna:
        fa_cmd.append("--ik-ulna")
    run(fa_cmd)
    print(f"\nRetarget complete: {args.bvh_out}")


if __name__ == "__main__":
    main()
