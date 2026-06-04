"""General BVH → zygote_skel retarget orchestrator.

Stage 0 (in-process): normalize input BVH — strip joint name prefixes,
collapse Spine2 into Spine1, detect axis convention, scale to meters,
center root at world origin.

Stages 1–4 (subprocess): invoke existing pipeline scripts.
  1. tools/make_run_vert_bvh.py    — spine vertebra expansion (q^(1/n))
  2. tools/add_sternum_to_bvh.py   — Kabsch-fit sternum joint
  3. tools/bake_arm_retarget_bvh.py --rest-align  — shoulder/arm baking
  4. tools/bake_forearm_retarget_bvh.py --ik-ulna — ulna/radius/carpal decomp

Stage 5 (in-process): verify joint positions vs original BVH FK.

Usage:
  source pyMAC/bin/activate
  python tools/retarget_to_zygote.py --in any.bvh --out data/motion/x.bvh
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation as R


PREFIXES = ["Character1_", "mixamorig:", "mixamorig_", "Bip01_", "Bip001_"]


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
                "line_open": i,
                "line_close": None,
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
    """List of (joint_idx, n_channels) in pre-order."""
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


def strip_prefix(name):
    for p in PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def find_joint(joints, suf):
    for i, j in enumerate(joints):
        if j["type"] in ("ROOT", "JOINT") and (j["name"] == suf or j["name"].endswith("_" + suf)):
            return i
    return None


def detect_tpose(joints):
    """T-pose if shoulder→hand vector at rest (cumulative OFFSETs) is
    X-dominant (arms extend laterally). Returns True if T-pose."""
    sh = find_joint(joints, "LeftShoulder")
    hd = find_joint(joints, "LeftHand")
    if sh is None or hd is None:
        return False
    v = np.zeros(3)
    cur = hd
    safety = 0
    while cur != sh and cur >= 0 and safety < 50:
        if joints[cur]["offset"] is not None:
            v = v + np.array(joints[cur]["offset"])
        cur = joints[cur]["parent"]
        safety += 1
    absv = np.abs(v)
    if absv.max() < 1e-9:
        return False
    return int(np.argmax(absv)) == 0  # X-dominant = lateral = T-pose


def detect_rig_style(joints):
    """Return 'bone_aligned' (LaFAN-style; bones extend along local +X) or
    'world_aligned' (Character1/Maya; bone direction baked into world).
    Decision based on LeftLeg OFFSET dominant axis."""
    ji = find_joint(joints, "LeftLeg")
    if ji is None:
        return "world_aligned"
    off = joints[ji]["offset"]
    if off is None:
        return "world_aligned"
    absv = np.abs(np.array(off))
    if absv.max() < 1e-9:
        return "world_aligned"
    dom = int(np.argmax(absv))
    # Y-dominant (axis 1) and negative → world-aligned (leg points -Y down).
    if dom == 1 and absv[1] / absv.sum() > 0.7:
        return "world_aligned"
    return "bone_aligned"


def stable_layout(joints, orig_children):
    """Pre-order layout using ORIGINAL child lists (pre-reparent)."""
    out = []
    visited = set()

    def walk(i):
        if i in visited:
            return
        visited.add(i)
        if joints[i]["channels"]:
            out.append((i, len(joints[i]["channels"])))
        for c in orig_children.get(i, joints[i]["children"]):
            walk(c)

    for i, j in enumerate(joints):
        if j["parent"] == -1:
            walk(i)
    return out


BVH_TO_SKEL_BODY = {
    "Hips": "Saccrum_Coccyx0",
    "Spine": "L50",
    "Spine1": "T120",
    "Neck": "C70",
    "Head": "Skull0",
    "LeftUpLeg": "L_Femur0", "LeftLeg": "L_Tibia_Fibula0",
    "LeftFoot": "L_Talus0", "LeftToeBase": "L_Toe10", "LeftToe": "L_Toe10",
    "RightUpLeg": "R_Femur0", "RightLeg": "R_Tibia_Fibula0",
    "RightFoot": "R_Talus0", "RightToeBase": "R_Toe10", "RightToe": "R_Toe10",
}


def compute_skel_rest_world_rotations(skel_xml):
    """Build skel via DART, return {bvh_joint_name: skel_body_world_rest_R}."""
    sys.path.insert(0, ".")
    from core.dartHelper import saveSkeletonInfo, buildFromInfo
    si, rn, *_ = saveSkeletonInfo(skel_xml)
    skel = buildFromInfo(si, rn)
    skel.setPositions(np.zeros(skel.getNumDofs()))
    out = {}
    for bvh_name, body_name in BVH_TO_SKEL_BODY.items():
        bn = skel.getBodyNode(body_name)
        if bn is None:
            out[bvh_name] = R.identity()
            continue
        out[bvh_name] = R.from_matrix(np.asarray(bn.getTransform().rotation()))
    return out


def normalize_bvh(in_path, out_path, skel_xml="data/zygote_skel.xml"):
    """Stage 0: read input BVH, apply normalizations, write out_path.
    Returns rig style string ('bone_aligned' / 'world_aligned')."""
    lines, joints, motion_idx = parse_bvh(in_path)
    if motion_idx is None:
        sys.exit("MOTION section not found in input BVH")
    # Snapshot ORIGINAL child lists before any reparenting (Spine2 collapse).
    orig_children = {i: list(j["children"]) for i, j in enumerate(joints)}
    layout = channel_layout(joints)
    name_to_col = {}
    col = 0
    for ji, n in layout:
        name_to_col[joints[ji]["name"]] = (col, n)
        col += n

    rows = []
    motion_header = []
    for ln in lines[motion_idx:]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("MOTION") or s.startswith("Frames") or s.startswith("Frame Time"):
            motion_header.append(ln)
            continue
        rows.append([float(x) for x in s.split()])

    # 1. Joint name canonicalization: strip prefixes.
    rename_count = 0
    for j in joints:
        new = strip_prefix(j["name"])
        if new != j["name"]:
            j["name"] = new
            rename_count += 1
    if rename_count:
        print(f"  [normalize] stripped prefixes from {rename_count} joints")
    # Rebuild name_to_col after rename.
    name_to_col = {}
    col = 0
    for ji, n in layout:
        name_to_col[joints[ji]["name"]] = (col, n)
        col += n

    # 2. Spine2 collapse: fold Spine2 rotation into Spine1.
    spine1_idx = find_joint(joints, "Spine1")
    spine2_idx = find_joint(joints, "Spine2")
    if spine1_idx is not None and spine2_idx is not None:
        s1 = joints[spine1_idx]
        s2 = joints[spine2_idx]
        # Channel order of each.
        for jr in (s1, s2):
            if jr["name"] not in name_to_col:
                spine2_idx = None
                break
        if spine2_idx is not None:
            c1, n1 = name_to_col[s1["name"]]
            c2, n2 = name_to_col[s2["name"]]
            order1 = "".join(c[0] for c in s1["channels"] if c.lower().endswith("rotation")).upper()
            order2 = "".join(c[0] for c in s2["channels"] if c.lower().endswith("rotation")).upper()
            r1_offs = [k for k, ch in enumerate(s1["channels"]) if ch.lower().endswith("rotation")]
            r2_offs = [k for k, ch in enumerate(s2["channels"]) if ch.lower().endswith("rotation")]
            for row in rows:
                e1 = [row[c1 + k] for k in r1_offs]
                e2 = [row[c2 + k] for k in r2_offs]
                R1 = R.from_euler(order1, e1, degrees=True)
                R2 = R.from_euler(order2, e2, degrees=True)
                Rc = R1 * R2
                new_e = Rc.as_euler(order1, degrees=True).tolist()
                for k, off in enumerate(r1_offs):
                    row[c1 + off] = new_e[k]
            # Keep Spine1 OFFSET unchanged (modifying it disturbs the
            # downstream LeftShoulder OFFSET which is in Spine2's frame).
            # Reparent Spine2's children to Spine1.
            for child_idx in s2["children"]:
                joints[child_idx]["parent"] = spine1_idx
                s1["children"].append(child_idx)
            s1["children"].remove(spine2_idx)
            s2["_deleted"] = True
            print("  [normalize] collapsed Spine2 into Spine1")

    # 3. Up-vector detection: compute Hips → Spine direction at rest (sum offsets).
    hips_idx = find_joint(joints, "Hips")
    spine_idx = find_joint(joints, "Spine")
    if hips_idx is not None and spine_idx is not None:
        # Walk parent chain from Spine back to Hips, summing offsets.
        v = np.zeros(3)
        cur = spine_idx
        while cur != hips_idx and cur >= 0:
            v = v + np.array(joints[cur]["offset"])
            cur = joints[cur]["parent"]
        abs_v = np.abs(v)
        dom = int(np.argmax(abs_v))
        # 0=X, 1=Y, 2=Z. Y-up (1) is expected.
        if dom == 2:
            # Z-up source: rotate 90° around X to make Y-up.
            print(f"  [normalize] Z-up source detected (Hips→Spine={v.round(3)}); rotating 90° around X")
            R_axis = R.from_euler("X", -90.0, degrees=True).as_matrix()
            # Rotate every joint's OFFSET.
            for j in joints:
                if j.get("_deleted"):
                    continue
                if j["offset"] is None:
                    continue
                ov = np.array(j["offset"])
                nv = R_axis @ ov
                j["offset"] = tuple(nv.tolist())
            # Rotate root position channels per frame.
            root_idx = next(i for i, j in enumerate(joints) if j["parent"] == -1)
            r_chs = joints[root_idx]["channels"]
            c0 = name_to_col[joints[root_idx]["name"]][0]
            xi = next((k for k, ch in enumerate(r_chs) if ch.lower() == "xposition"), None)
            yi = next((k for k, ch in enumerate(r_chs) if ch.lower() == "yposition"), None)
            zi = next((k for k, ch in enumerate(r_chs) if ch.lower() == "zposition"), None)
            for row in rows:
                if xi is not None and yi is not None and zi is not None:
                    p = np.array([row[c0 + xi], row[c0 + yi], row[c0 + zi]])
                    pn = R_axis @ p
                    row[c0 + xi] = float(pn[0])
                    row[c0 + yi] = float(pn[1])
                    row[c0 + zi] = float(pn[2])

    # 4. Scale to meters if root OFFSET large.
    root_idx = next(i for i, j in enumerate(joints) if j["parent"] == -1)
    root_off = np.array(joints[root_idx]["offset"]) if joints[root_idx]["offset"] else np.zeros(3)
    if np.linalg.norm(root_off) > 10.0:
        print(f"  [normalize] cm-scale detected (root |OFFSET|={np.linalg.norm(root_off):.2f}); scaling x0.01")
        for j in joints:
            if j.get("_deleted"):
                continue
            if j["offset"] is not None:
                j["offset"] = tuple(v * 0.01 for v in j["offset"])
        # Scale per-frame root positions.
        r_chs = joints[root_idx]["channels"]
        c0 = name_to_col[joints[root_idx]["name"]][0]
        for k, ch in enumerate(r_chs):
            if ch.lower().endswith("position"):
                for row in rows:
                    row[c0 + k] *= 0.01

    # 5. Root XZ centering: subtract f0 root X and Z position.
    root_idx = next(i for i, j in enumerate(joints) if j["parent"] == -1)
    r_chs = joints[root_idx]["channels"]
    c0 = name_to_col[joints[root_idx]["name"]][0]
    xi = next((k for k, ch in enumerate(r_chs) if ch.lower() == "xposition"), None)
    zi = next((k for k, ch in enumerate(r_chs) if ch.lower() == "zposition"), None)
    if xi is not None and zi is not None and rows:
        x0 = rows[0][c0 + xi]
        z0 = rows[0][c0 + zi]
        for row in rows:
            row[c0 + xi] -= x0
            row[c0 + zi] -= z0
        print(f"  [normalize] centered root XZ (subtracted f0 X={x0:.3f}, Z={z0:.3f})")

    # 5b. Bone-aligned → world-aligned conversion for non-arm joints.
    # LaFAN bone-aligned: each joint's local frame has X along its bone, so
    # rest pose encoded in non-zero channels. Skel rest is N-pose with arms
    # hanging. Under viewer's T_frame=0 auto-trigger (X-dominant leg
    # offsets), MyBVH subtracts f0 channels entirely → skel f0 = rest. BVH
    # actor f0 = T-pose, so arm directions cannot match.
    # Solution: rotate non-arm bone offsets to world directions (Y-down legs,
    # Y-up spine) and zero f0 channels. After this, leg offsets become
    # Y-dominant → viewer picks T_frame=None. Arms keep bone-aligned values
    # so arm-bake's M_align conjugation places skel arms in T-pose at f0.
    # Stage 0 bone-aligned → world-aligned conversion DISABLED. Subtracting
    # f0 from trunk joints orphans arm children whose channels are still
    # interpreted in old parent frame (Spine1 used to be bone-aligned, now
    # identity at f0). Arm direction goes wrong and root rotates in air.
    # Would need to also rotate arm channels by parent f0 world rotation
    # (not just OFFSET) to preserve world motion. Non-trivial.
    TRUNK_LEG_NAMES = set()
    rig_style_check = detect_rig_style(joints)
    if False and rig_style_check == "bone_aligned" and rows:
        joint_order = []
        def _walk(i):
            if joints[i].get("_deleted"):
                return
            joint_order.append(i)
            for c in joints[i]["children"]:
                if not joints[c].get("_deleted"):
                    _walk(c)
        for i, j in enumerate(joints):
            if j["parent"] == -1:
                _walk(i)

        col_map_loc = {}
        c2 = 0
        for ji, n in layout:
            col_map_loc[ji] = (c2, n)
            c2 += n

        # f0 local rotation per joint.
        local_R0 = {}
        for ji in joint_order:
            j = joints[ji]
            if not j["channels"] or ji not in col_map_loc:
                local_R0[ji] = R.identity()
                continue
            chs = j["channels"]
            rot_offs = [k for k, ch in enumerate(chs) if ch.lower().endswith("rotation")]
            if not rot_offs:
                local_R0[ji] = R.identity()
                continue
            order_j = "".join(chs[k][0] for k in rot_offs).upper()
            c0_j = col_map_loc[ji][0]
            e0 = [rows[0][c0_j + k] for k in rot_offs]
            local_R0[ji] = R.from_euler(order_j, e0, degrees=True)

        # f0 world cumulative rotation per joint.
        world_R0 = {}
        for ji in joint_order:
            j = joints[ji]
            parent = j["parent"]
            if parent == -1 or parent not in world_R0:
                world_R0[ji] = local_R0[ji]
            else:
                world_R0[ji] = world_R0[parent] * local_R0[ji]

        # Pre-rotate OFFSET of every joint whose parent is in TRUNK_LEG.
        # This preserves world position when parent's f0 rotation gets zeroed.
        n_offsets = 0
        for ji in joint_order:
            j = joints[ji]
            if j["offset"] is None or j["parent"] == -1:
                continue
            parent_name = joints[j["parent"]]["name"]
            if parent_name not in TRUNK_LEG_NAMES:
                continue
            parent_world_R = world_R0.get(j["parent"], R.identity())
            new_off = parent_world_R.apply(np.array(j["offset"]))
            j["offset"] = tuple(new_off.tolist())
            n_offsets += 1

        # Subtract f0 from TRUNK_LEG joints' channels.
        n_subs = 0
        for ji in joint_order:
            j = joints[ji]
            if j["name"] not in TRUNK_LEG_NAMES:
                continue
            if not j["channels"] or ji not in col_map_loc:
                continue
            chs = j["channels"]
            rot_offs = [k for k, ch in enumerate(chs) if ch.lower().endswith("rotation")]
            if not rot_offs:
                continue
            order_j = "".join(chs[k][0] for k in rot_offs).upper()
            c0_j = col_map_loc[ji][0]
            R0_inv = local_R0[ji].inv()
            for row in rows:
                e_f = [row[c0_j + k] for k in rot_offs]
                R_f = R.from_euler(order_j, e_f, degrees=True)
                R_new = R0_inv * R_f
                new_e = R_new.as_euler(order_j, degrees=True).tolist()
                for k, off in enumerate(rot_offs):
                    row[c0_j + off] = new_e[k]
            n_subs += 1
        print(f"  [normalize] LaFAN bone→world: pre-rotated {n_offsets} offsets, "
              f"f0-subtracted {n_subs} non-arm joints")

    # 6. Detect rig style + T-pose.
    rig_style = detect_rig_style(joints)
    is_tpose = detect_tpose(joints)
    print(f"  [normalize] rig style: {rig_style}, T-pose: {is_tpose}")

    # 7. Write normalized BVH.
    _write_bvh(out_path, joints, motion_idx, lines, rows, motion_header,
               orig_children=orig_children)
    return rig_style, is_tpose


def _write_bvh(out_path, joints, motion_idx_orig, original_lines, rows, motion_header,
               orig_children=None):
    """Reconstruct BVH from modified joints + rows. Skips deleted joints.
    orig_children: pre-reparent child lists, used to lookup row column
    positions correctly even after Spine2 collapse reparented children."""
    out_lines = ["HIERARCHY"]

    def emit(jidx, depth):
        j = joints[jidx]
        if j.get("_deleted"):
            return
        ind = "\t" * depth
        if j["type"] == "ROOT":
            out_lines.append(f"ROOT {j['name']}")
            out_lines.append("{")
            out_lines.append(f"\tOFFSET {j['offset'][0]} {j['offset'][1]} {j['offset'][2]}")
            out_lines.append(f"\tCHANNELS {len(j['channels'])} " + " ".join(j["channels"]))
            for c in j["children"]:
                emit(c, 1)
            out_lines.append("}")
        elif j["type"] == "End":
            out_lines.append(ind + "End Site")
            out_lines.append(ind + "{")
            out_lines.append(ind + f"\tOFFSET {j['offset'][0]} {j['offset'][1]} {j['offset'][2]}")
            out_lines.append(ind + "}")
        else:
            out_lines.append(ind + f"JOINT {j['name']}")
            out_lines.append(ind + "{")
            out_lines.append(ind + f"\tOFFSET {j['offset'][0]} {j['offset'][1]} {j['offset'][2]}")
            if j["channels"]:
                out_lines.append(ind + f"\tCHANNELS {len(j['channels'])} " + " ".join(j["channels"]))
            for c in j["children"]:
                emit(c, depth + 1)
            out_lines.append(ind + "}")

    for i, j in enumerate(joints):
        if j["parent"] == -1:
            emit(i, 0)

    # Build new layout (skipping deleted) and remap row columns.
    new_layout = []

    def walk_new(i):
        j = joints[i]
        if j.get("_deleted"):
            return
        if j["channels"]:
            new_layout.append((i, len(j["channels"])))
        for c in j["children"]:
            walk_new(c)

    for i, j in enumerate(joints):
        if j["parent"] == -1:
            walk_new(i)

    # Original column lookup: walk using ORIGINAL children snapshot (pre
    # Spine2 collapse). Motion rows use original BVH column order; post-
    # collapse walk would misalign LeftShoulder ↔ Spine2 column data.
    old_col = {}
    def walk_orig(i, col_acc):
        if joints[i]["channels"]:
            old_col[i] = (col_acc, len(joints[i]["channels"]))
            col_acc += len(joints[i]["channels"])
        for c in orig_children[i]:
            col_acc = walk_orig(c, col_acc)
        return col_acc
    col_acc = 0
    for i, j in enumerate(joints):
        if j["parent"] == -1:
            col_acc = walk_orig(i, col_acc)

    out_lines.append("MOTION")
    out_lines.append(f"Frames: {len(rows)}")
    # Try to preserve Frame Time from motion_header.
    ft = "0.033333"
    for h in motion_header:
        if h.strip().startswith("Frame Time"):
            ft = h.split(":")[1].strip()
            break
    out_lines.append(f"Frame Time: {ft}")
    for row in rows:
        new_row = []
        for ji, n in new_layout:
            c, k = old_col[ji]
            new_row.extend(row[c:c + k])
        out_lines.append(" ".join(f"{v:.6f}" for v in new_row))

    with open(out_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")


def run_stage(cmd, name):
    print(f"\n[stage] {name}: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        for ln in res.stdout.splitlines()[-5:]:
            print(f"  {ln}")
    if res.returncode != 0:
        print(res.stderr)
        sys.exit(f"Stage '{name}' failed (exit {res.returncode})")


def bvh_fk_world(joints, row, n2c):
    """Compute world transforms for all joints at this frame."""
    Twr = [None] * len(joints)
    order = []

    def walk(i):
        order.append(i)
        for c in joints[i]["children"]:
            walk(c)

    for i, j in enumerate(joints):
        if j["parent"] == -1:
            walk(i)
    for ji in order:
        j = joints[ji]
        if j.get("_deleted"):
            continue
        off = np.array(j["offset"]) if j["offset"] else np.zeros(3)
        Rl = R.identity()
        pa = np.zeros(3)
        if j["channels"] and j["name"] in n2c:
            c0, _ = n2c[j["name"]]
            pi = 0
            rc = []
            for ch in j["channels"]:
                v = row[c0 + pi]
                pi += 1
                if ch.lower() == "xposition":
                    pa[0] = v
                elif ch.lower() == "yposition":
                    pa[1] = v
                elif ch.lower() == "zposition":
                    pa[2] = v
                else:
                    rc.append((ch[0].upper(), v))
            if rc:
                Rl = R.from_euler("".join(c for c, _ in rc), [v for _, v in rc], degrees=True)
        T = np.eye(4)
        T[:3, :3] = Rl.as_matrix()
        T[:3, 3] = off + pa
        if j["parent"] >= 0 and Twr[j["parent"]] is not None:
            T = Twr[j["parent"]] @ T
        Twr[ji] = T
    return Twr


def verify(final_bvh, norm_bvh, skel_xml):
    """Compare baked output skel positions vs original BVH FK at sample frames."""
    print("\n[verify] joint position comparison")
    sys.path.insert(0, ".")
    from core.dartHelper import saveSkeletonInfo, buildFromInfo
    from core.bvhparser import MyBVH

    si, rn, bi, *_ = saveSkeletonInfo(skel_xml)
    skel = buildFromInfo(si, rn)
    # Use T_frame=0 to match viewer auto-detect for non-upright BVHs.
    # Pass T_frame=None for already-upright (Character1) — heuristic via leg.
    leg_y_ok = False
    try:
        with open(final_bvh) as _f:
            _c = _f.read()
        import re as _re
        _m = _re.search(r'JOINT\s+\S*LeftLeg\s*\{[^}]*?OFFSET\s+(\S+)\s+(\S+)\s+(\S+)', _c, _re.DOTALL)
        if _m:
            _x, _y, _z = abs(float(_m.group(1))), abs(float(_m.group(2))), abs(float(_m.group(3)))
            _max = max(_x, _y, _z)
            leg_y_ok = (_max > 1e-6) and (_y / _max > 0.8)
    except Exception:
        pass
    t_frame_use = None if leg_y_ok else 0
    m = MyBVH(final_bvh, bi, skel, T_frame=t_frame_use)
    print(f"  [verify] using T_frame={t_frame_use}")
    n_frames = m.mocap_refs.shape[0]
    dofs = skel.getNumDofs()

    # BVH FK of normalized source.
    lines, joints, mi = parse_bvh(norm_bvh)
    layout = channel_layout(joints)
    n2c = {}
    col = 0
    for ji, n in layout:
        n2c[joints[ji]["name"]] = (col, n)
        col += n
    rows = []
    for ln in lines[mi:]:
        s = ln.strip()
        if not s or s.startswith("MOTION") or s.startswith("Frames") or s.startswith("Frame Time"):
            continue
        rows.append([float(x) for x in s.split()])

    pairs = [
        ("LeftArm", "L_Humerus0"),
        ("LeftHand", "L_Carpal0"),
        ("RightArm", "R_Humerus0"),
        ("RightHand", "R_Carpal0"),
        ("Head", "Head0"),
    ]

    n_sample = min(n_frames, len(rows))
    sample_frames = [0, n_sample // 4, n_sample // 2, 3 * n_sample // 4, n_sample - 1]
    sample_frames = [f for f in sample_frames if f < n_sample]

    pairs_arms = [
        ("LeftArm", "LeftHand", "L_Humerus0", "L_Carpal0"),
        ("RightArm", "RightHand", "R_Humerus0", "R_Carpal0"),
    ]
    pairs_trunk = [("Hips", "Head", "Saccrum_Coccyx0", "Skull0")]
    pairs_legs = [
        ("LeftUpLeg", "LeftFoot", "L_Femur0", "L_Talus0"),
        ("RightUpLeg", "RightFoot", "R_Femur0", "R_Talus0"),
        ("LeftUpLeg", "LeftLeg", "L_Femur0", "L_Tibia_Fibula0"),
        ("RightUpLeg", "RightLeg", "R_Femur0", "R_Tibia_Fibula0"),
    ]

    for f in sample_frames:
        Twr = bvh_fk_world(joints, rows[f], n2c)
        pose = np.zeros(dofs)
        nn = min(dofs, m.mocap_refs.shape[1])
        pose[:nn] = m.mocap_refs[f, :nn]
        skel.setPositions(pose)

        for suf_a, suf_b, skel_a, skel_b in pairs_arms + pairs_trunk + pairs_legs:
            a_idx = find_joint(joints, suf_a)
            b_idx = find_joint(joints, suf_b)
            if a_idx is None or b_idx is None:
                continue
            rel_b_vec = Twr[b_idx][:3, 3] - Twr[a_idx][:3, 3]
            p_a = skel.getBodyNode(skel_a).getCOM()
            p_b = skel.getBodyNode(skel_b).getCOM()
            rel_s_vec = p_b - p_a
            bn = np.linalg.norm(rel_b_vec)
            sn = np.linalg.norm(rel_s_vec)
            cos = float(np.dot(rel_b_vec, rel_s_vec) / max(bn * sn, 1e-9))
            len_err = abs(bn - sn) / max(sn, 1e-9)
            print(f"  f{f:5d} {suf_a:9s}→{suf_b:9s}  "
                  f"bvh_len={bn:.3f} skel_len={sn:.3f} (Δ={len_err*100:.1f}%)  "
                  f"cosθ={cos:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="bvh_in", required=True)
    ap.add_argument("--out", dest="bvh_out", required=True)
    ap.add_argument("--skel-xml", default="data/zygote_skel.xml")
    ap.add_argument("--keep-intermediates", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--tpose-palm-deg", type=float, default=90.0,
                    help="Radius offset (deg) for T-pose sources. L=+val, R=-val.")
    ap.add_argument("--tpose-palm-l", type=float, default=None,
                    help="Override L_Radius offset (overrides --tpose-palm-deg sign).")
    ap.add_argument("--tpose-palm-r", type=float, default=None,
                    help="Override R_Radius offset.")
    ap.add_argument("--head-scale", type=float, default=1.0,
                    help="Scale BVH Head rotation before distributing across Atlas/Axis. Default 1.0. Use <1 to damp head wave.")
    ap.add_argument("--neck-scale", type=float, default=1.0,
                    help="Scale BVH Neck rotation. Default 1.0.")
    args = ap.parse_args()

    work = tempfile.mkdtemp(prefix="retarget_zygote_")
    base = os.path.splitext(os.path.basename(args.bvh_in))[0]
    norm = os.path.join(work, f"{base}.norm.bvh")
    vert = os.path.join(work, f"{base}.vert.bvh")
    vert_st = os.path.join(work, f"{base}.vert_st.bvh")
    armed = os.path.join(work, f"{base}.arm.bvh")

    print(f"[Stage 0] normalize: {args.bvh_in} → {norm}")
    rig_style, is_tpose = normalize_bvh(args.bvh_in, norm)

    py = sys.executable
    # For LaFAN-style sources, head BVH rotations encode large rest-pose
    # offsets (similar to clavicle). Auto-damp to 0.3 if user didn't override.
    auto_head = 0.3 if (rig_style == "bone_aligned" and args.head_scale == 1.0) else args.head_scale
    auto_neck = 0.4 if (rig_style == "bone_aligned" and args.neck_scale == 1.0) else args.neck_scale
    vert_cmd = [py, "tools/make_run_vert_bvh.py", "--in", norm, "--out", vert, "--skip-xml",
                "--head-scale", str(auto_head), "--neck-scale", str(auto_neck)]
    run_stage(vert_cmd, "spine expand")
    run_stage(
        [py, "tools/add_sternum_to_bvh.py", "--in", vert, "--out", vert_st],
        "sternum",
    )
    if rig_style == "bone_aligned":
        # World-direction arm bake: skel arm world direction = BVH arm world
        # direction per frame. Bypasses bone-aligned encoding inflation and
        # skel body-local frame asymmetry. Inman scapulohumeral split.
        arm_cmd = [py, "tools/bake_arm_body_relative.py",
                   "--in", vert_st, "--out", armed,
                   "--skel-xml", args.skel_xml,
                   "--scapulohumeral", "0.27"]
        run_stage(arm_cmd, "arm (world-direction + scapulohumeral)")
    else:
        arm_cmd = [py, "tools/bake_arm_retarget_bvh.py", "--in", vert_st, "--out", armed,
                   "--skel-xml", args.skel_xml]
        run_stage(arm_cmd, f"arm ({rig_style})")
    fa_cmd = [py, "tools/bake_forearm_retarget_bvh.py",
              "--in", armed, "--out", args.bvh_out,
              "--skip-xml", "--orig-bvh", norm,
              "--skel-xml", args.skel_xml]
    # Bone-aligned LaFAN: use anatomical elbow bend angle (from arm/forearm
    # vectors) instead of IK target. IK fails when skel rest (N-pose) and
    # BVH rest (T-pose) put arms in different world directions — IK target
    # unreachable, picks max-bend. Anatomical bend = pure angle, sign-safe.
    if rig_style == "bone_aligned":
        fa_cmd += ["--use-bvh-bend"]
    else:
        fa_cmd += ["--ik-ulna"]
    # T-pose source: palm faces forward at rest; rotate via radius (forearm
    # axial twist) so palm faces ground. L_Radius axis ≈ -Y joint-local;
    # +90° around it = pronation → palm-down for L. R mirrored.
    if is_tpose:
        l_off = args.tpose_palm_l if args.tpose_palm_l is not None else args.tpose_palm_deg
        r_off = args.tpose_palm_r if args.tpose_palm_r is not None else -args.tpose_palm_deg
        fa_cmd += ["--l-radius-offset-deg", str(l_off),
                   "--r-radius-offset-deg", str(r_off),
                   "--const-radius-twist"]
        print(f"  [forearm] T-pose detected → L_offset={l_off}° R_offset={r_off}° + const twist")
    run_stage(fa_cmd, "forearm")

    print(f"\n[done] wrote {args.bvh_out}")

    if not args.no_verify:
        try:
            verify(args.bvh_out, norm, args.skel_xml)
        except Exception as e:
            print(f"[verify] skipped: {e}")

    if not args.keep_intermediates:
        for p in (norm, vert, vert_st, armed):
            try:
                os.remove(p)
            except Exception:
                pass
        try:
            os.rmdir(work)
        except Exception:
            pass
    else:
        print(f"[intermediates kept in {work}]")


if __name__ == "__main__":
    main()
