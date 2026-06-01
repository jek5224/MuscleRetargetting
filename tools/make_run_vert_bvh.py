"""Generate data/motion/run_vert.bvh from data/motion/run.bvh by inserting
the full spine vertebra chain between Hips and Spine1's children, and
between Neck and Head. Rotations from the original Spine/Spine1/Neck/Head
pivots are distributed across the new vertebrae via quaternion fractional
power (q^(1/n)), so the cumulative chain rotation matches the original
single-pivot rotation.

Also patches data/zygote_skel.xml so every spine bone is a Ball joint with
the matching bvh="..." attribute. Run from project root."""
import os
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

import argparse
SKEL_XML = "data/zygote_skel.xml"

# Replacement chains: source-pivot → list of vertebrae (parent → child order)
LUMBAR = ["L5", "L4", "L3", "L2", "L1"]
THORACIC = ["T12", "T11", "T10", "T9", "T8", "T7", "T6", "T5", "T4", "T3", "T2", "T1"]
CERVICAL = ["C7", "C6", "C5", "C4", "C3"]
CRANIAL = ["Axis0", "Atlas0"]  # 2 in chain; Axis1 stays Weld (compound second body)

# Skel-XML body name → new BVH joint name
SKEL_TO_BVH = {}
for name in LUMBAR + THORACIC + CERVICAL:
    SKEL_TO_BVH[name + "0"] = name
SKEL_TO_BVH["Axis0"] = "Axis0"
SKEL_TO_BVH["Atlas0"] = "Atlas0"


def parse_bvh_hierarchy(lines):
    """Return list of joint dicts {name, offset, channels, parent_idx, depth} in pre-order,
    plus index where MOTION section begins."""
    joints = []
    stack = []  # stack of (joint_idx, expecting_brace)
    pending_name = None
    pending_type = None  # 'ROOT' / 'JOINT' / 'End'
    depth = 0
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped == "MOTION":
            return joints, i
        m = re.match(r'^(ROOT|JOINT)\s+(\S+)\s*$', stripped)
        if m:
            pending_type = m.group(1)
            pending_name = m.group(2)
            continue
        if stripped == "End Site":
            pending_type = "End"
            pending_name = "EndSite_" + (joints[-1]["name"] if joints else "x")
            continue
        if stripped == "{":
            j = {
                "type": pending_type,
                "name": pending_name,
                "depth": depth,
                "parent": stack[-1] if stack else -1,
                "offset": None,
                "channels": [],
                "children": [],
                "start_line": i,
                "end_line": None,
            }
            joints.append(j)
            this_idx = len(joints) - 1
            if stack:
                joints[stack[-1]]["children"].append(this_idx)
            stack.append(this_idx)
            depth += 1
            pending_type = None
            pending_name = None
            continue
        if stripped == "}":
            if stack:
                joints[stack[-1]]["end_line"] = i
                stack.pop()
            depth -= 1
            continue
        m = re.match(r'^OFFSET\s+(\S+)\s+(\S+)\s+(\S+)\s*$', stripped)
        if m and stack:
            joints[stack[-1]]["offset"] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            continue
        m = re.match(r'^CHANNELS\s+(\d+)\s+(.+)$', stripped)
        if m and stack:
            joints[stack[-1]]["channels"] = m.group(2).split()
            continue
    return joints, len(lines)


def euler_to_quat(zxy_deg):
    """run.bvh CHANNELS order Zrotation Xrotation Yrotation → intrinsic ZXY in degrees."""
    return R.from_euler("ZXY", zxy_deg, degrees=True)


def quat_to_euler_zxy(rot):
    return rot.as_euler("ZXY", degrees=True)


def fractional_rotation(rot, n):
    """Return rot^(1/n) such that (rot^(1/n))^n ≈ rot. Uses rotvec scaling."""
    rv = rot.as_rotvec()
    return R.from_rotvec(rv / float(n))


def build_chain(parent_name, joint_names, offset_total, base_indent):
    """Generate BVH HIERARCHY text for a chain of joints starting from parent.
    Returns (text_lines_before_inner, text_lines_inner_open, text_lines_after_inner)
    Simpler: returns the indented JOINT block string that the caller wraps.
    offset_total is split evenly across the chain."""
    n = len(joint_names)
    if n == 0:
        return ""
    seg = tuple(o / n for o in offset_total)
    # nested chain: each JOINT contains the next
    lines = []
    indent = base_indent

    def emit_open(name, off, ind):
        lines.append("\t" * ind + f"JOINT {name}")
        lines.append("\t" * ind + "{")
        lines.append("\t" * (ind + 1) + f"OFFSET {off[0]} {off[1]} {off[2]}")
        lines.append("\t" * (ind + 1) + "CHANNELS 3 Zrotation Xrotation Yrotation")

    def emit_close(ind):
        lines.append("\t" * ind + "}")

    for nm in joint_names:
        emit_open(nm, seg, indent)
        indent += 1
    return lines, indent  # caller inserts children at current indent then closes


def emit_block_close(lines, indent, count):
    for _ in range(count):
        indent -= 1
        lines.append("\t" * indent + "}")
    return indent


def _resolve_pivot(by_name, base):
    """Resolve 'Spine'/'Spine1'/'Neck'/'Head' to actual BVH joint name (handles
    Character1_Spine etc. via prefix tolerance)."""
    if base in by_name:
        return base
    for candidate in by_name:
        if candidate.endswith("_" + base) or candidate == "Character1_" + base:
            return candidate
    return None


def render_hierarchy(joints, motion_line_idx, original_lines):
    """Rewrite hierarchy lines, expanding Spine/Spine1 chain and Neck chain."""
    # Indices for key joints
    by_name = {j["name"]: i for i, j in enumerate(joints) if j["type"] in ("ROOT", "JOINT")}
    spine_name = _resolve_pivot(by_name, "Spine")
    spine1_name = _resolve_pivot(by_name, "Spine1")
    neck_name = _resolve_pivot(by_name, "Neck")
    head_name = _resolve_pivot(by_name, "Head")
    if not all([spine_name, spine1_name, neck_name, head_name]):
        sys.exit(f"Could not resolve all pivots: Spine={spine_name} Spine1={spine1_name} Neck={neck_name} Head={head_name}")
    print(f"[pivots] {spine_name}, {spine1_name}, {neck_name}, {head_name}")
    spine_idx = by_name[spine_name]
    spine1_idx = by_name[spine1_name]
    neck_idx = by_name[neck_name]
    head_idx = by_name[head_name]

    spine_offset = joints[spine_idx]["offset"]
    spine1_offset = joints[spine1_idx]["offset"]
    neck_offset = joints[neck_idx]["offset"]
    head_offset = joints[head_idx]["offset"]

    # Spine1's children list (idx) — what we keep below the new T1
    spine1_children = list(joints[spine1_idx]["children"])
    # Neck's children — what we keep below the new Atlas0
    neck_children = list(joints[neck_idx]["children"])

    # We'll emit the hierarchy by walking the joint tree, inserting our
    # chain replacements at Spine and Neck.
    out_lines = ["HIERARCHY"]
    out_lines.append("ROOT " + joints[0]["name"])
    out_lines.append("{")
    out_lines.append(f"\tOFFSET {joints[0]['offset'][0]} {joints[0]['offset'][1]} {joints[0]['offset'][2]}")
    out_lines.append("\tCHANNELS 6 " + " ".join(joints[0]["channels"]))

    # Channel column tracking: list of (joint_name, num_channels) in emit order.
    # First joint is root with 6 channels.
    channel_order = [(joints[0]["name"], 6)]

    def write_joint_recursive(jidx, indent):
        j = joints[jidx]
        if j["name"] == spine_name:
            # Replace Spine subtree with lumbar+thoracic chain.
            # Open L5 at current indent...
            opened = 0
            for nm, off in zip(LUMBAR, [tuple(o / len(LUMBAR) for o in spine_offset)] * len(LUMBAR)):
                out_lines.append("\t" * indent + f"JOINT {nm}")
                out_lines.append("\t" * indent + "{")
                out_lines.append("\t" * (indent + 1) + f"OFFSET {off[0]} {off[1]} {off[2]}")
                out_lines.append("\t" * (indent + 1) + "CHANNELS 3 Zrotation Xrotation Yrotation")
                channel_order.append((nm, 3))
                indent += 1
                opened += 1
            for nm, off in zip(THORACIC, [tuple(o / len(THORACIC) for o in spine1_offset)] * len(THORACIC)):
                out_lines.append("\t" * indent + f"JOINT {nm}")
                out_lines.append("\t" * indent + "{")
                out_lines.append("\t" * (indent + 1) + f"OFFSET {off[0]} {off[1]} {off[2]}")
                out_lines.append("\t" * (indent + 1) + "CHANNELS 3 Zrotation Xrotation Yrotation")
                channel_order.append((nm, 3))
                indent += 1
                opened += 1
            # Now emit Spine1's children at this indent
            for child in spine1_children:
                write_joint_recursive(child, indent)
            # Close all opened
            for _ in range(opened):
                indent -= 1
                out_lines.append("\t" * indent + "}")
            return
        if j["name"] == neck_name:
            # Replace Neck subtree with cervical+cranial chain.
            opened = 0
            for nm, off in zip(CERVICAL, [tuple(o / len(CERVICAL) for o in neck_offset)] * len(CERVICAL)):
                out_lines.append("\t" * indent + f"JOINT {nm}")
                out_lines.append("\t" * indent + "{")
                out_lines.append("\t" * (indent + 1) + f"OFFSET {off[0]} {off[1]} {off[2]}")
                out_lines.append("\t" * (indent + 1) + "CHANNELS 3 Zrotation Xrotation Yrotation")
                channel_order.append((nm, 3))
                indent += 1
                opened += 1
            for nm, off in zip(CRANIAL, [tuple(o / len(CRANIAL) for o in head_offset)] * len(CRANIAL)):
                out_lines.append("\t" * indent + f"JOINT {nm}")
                out_lines.append("\t" * indent + "{")
                out_lines.append("\t" * (indent + 1) + f"OFFSET {off[0]} {off[1]} {off[2]}")
                out_lines.append("\t" * (indent + 1) + "CHANNELS 3 Zrotation Xrotation Yrotation")
                channel_order.append((nm, 3))
                indent += 1
                opened += 1
            # Now emit Neck's children EXCEPT Head (replaced)
            for child in neck_children:
                if joints[child]["name"] == head_name:
                    # Emit Head's children (e.g., End Site) at current indent
                    for hc in joints[child]["children"]:
                        write_joint_recursive(hc, indent)
                    continue
                write_joint_recursive(child, indent)
            for _ in range(opened):
                indent -= 1
                out_lines.append("\t" * indent + "}")
            return
        # Normal joint
        if j["type"] == "End":
            out_lines.append("\t" * indent + "End Site")
            out_lines.append("\t" * indent + "{")
            out_lines.append("\t" * (indent + 1) + f"OFFSET {j['offset'][0]} {j['offset'][1]} {j['offset'][2]}")
            out_lines.append("\t" * indent + "}")
            return
        if j["type"] == "ROOT":
            # already emitted; just recurse into children
            for child in j["children"]:
                write_joint_recursive(child, indent + 1)
            return
        # JOINT
        out_lines.append("\t" * indent + f"JOINT {j['name']}")
        out_lines.append("\t" * indent + "{")
        out_lines.append("\t" * (indent + 1) + f"OFFSET {j['offset'][0]} {j['offset'][1]} {j['offset'][2]}")
        if j["channels"]:
            out_lines.append("\t" * (indent + 1) + f"CHANNELS {len(j['channels'])} " + " ".join(j['channels']))
            channel_order.append((j["name"], len(j["channels"])))
        for child in j["children"]:
            write_joint_recursive(child, indent + 1)
        out_lines.append("\t" * indent + "}")

    for child in joints[0]["children"]:
        write_joint_recursive(child, 1)
    out_lines.append("}")
    return out_lines, channel_order, {
        'spine': spine_name, 'spine1': spine1_name,
        'neck': neck_name, 'head': head_name,
    }


def rewrite_motion(motion_rows, src_channel_layout, dst_channel_order,
                   pivots):
    """Each row in motion_rows is list of floats matching src_channel_layout.
    src_channel_layout = list of (joint_name, num_channels).
    dst_channel_order   = list of (joint_name, num_channels).
    pivots = dict with keys 'spine','spine1','neck','head' mapping to actual
    BVH joint names (handles Character1_ prefix)."""
    # Build name -> (start_col, n) lookup for source
    src_lookup = {}
    col = 0
    for name, n in src_channel_layout:
        src_lookup[name] = (col, n)
        col += n
    src_total = col

    new_rows = []
    for row in motion_rows:
        if len(row) != src_total:
            sys.exit(f"motion row size {len(row)} != channel layout total {src_total}")
        spine_zxy = row[src_lookup[pivots['spine']][0]:src_lookup[pivots['spine']][0] + 3]
        spine1_zxy = row[src_lookup[pivots['spine1']][0]:src_lookup[pivots['spine1']][0] + 3]
        neck_zxy = row[src_lookup[pivots['neck']][0]:src_lookup[pivots['neck']][0] + 3]
        head_zxy = row[src_lookup[pivots['head']][0]:src_lookup[pivots['head']][0] + 3]

        # Compute fractional per-vertebra rotation
        spine_rot = euler_to_quat(spine_zxy)
        spine1_rot = euler_to_quat(spine1_zxy)
        neck_rot = euler_to_quat(neck_zxy)
        head_rot = euler_to_quat(head_zxy)

        lumbar_seg = quat_to_euler_zxy(fractional_rotation(spine_rot, len(LUMBAR)))
        thoracic_seg = quat_to_euler_zxy(fractional_rotation(spine1_rot, len(THORACIC)))
        cervical_seg = quat_to_euler_zxy(fractional_rotation(neck_rot, len(CERVICAL)))
        cranial_seg = quat_to_euler_zxy(fractional_rotation(head_rot, len(CRANIAL)))

        per_joint_rot = {}
        for nm in LUMBAR:
            per_joint_rot[nm] = lumbar_seg
        for nm in THORACIC:
            per_joint_rot[nm] = thoracic_seg
        for nm in CERVICAL:
            per_joint_rot[nm] = cervical_seg
        for nm in CRANIAL:
            per_joint_rot[nm] = cranial_seg

        # Emit new row in dst_channel_order
        new_row = []
        for name, n in dst_channel_order:
            if name in per_joint_rot:
                new_row.extend(per_joint_rot[name])
            elif name in src_lookup:
                c, k = src_lookup[name]
                new_row.extend(row[c:c + k])
            else:
                new_row.extend([0.0] * n)
        new_rows.append(new_row)
    return new_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="bvh_in", default="data/motion/run.bvh")
    ap.add_argument("--out", dest="bvh_out", default="data/motion/run_vert.bvh")
    ap.add_argument("--skip-xml", action="store_true",
                    help="Skip patching zygote_skel.xml (use after first run)")
    args = ap.parse_args()

    with open(args.bvh_in) as f:
        lines = f.read().splitlines()

    # Parse hierarchy
    joints, motion_idx = parse_bvh_hierarchy(lines)
    # Build src channel layout in pre-order
    src_layout = []

    def walk_src(jidx):
        j = joints[jidx]
        if j["channels"]:
            src_layout.append((j["name"], len(j["channels"])))
        for c in j["children"]:
            walk_src(c)
    walk_src(0)

    # Parse motion frames
    frames_header = []
    frame_data = []
    for ln in lines[motion_idx:]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("MOTION") or s.startswith("Frames") or s.startswith("Frame Time"):
            frames_header.append(ln)
            continue
        # data row
        vals = [float(x) for x in s.split()]
        frame_data.append(vals)

    # Build new hierarchy + channel order
    hierarchy_lines, dst_channel_order, pivots = render_hierarchy(joints, motion_idx, lines)

    # Rewrite motion
    new_rows = rewrite_motion(frame_data, src_layout, dst_channel_order, pivots)

    # Write run_vert.bvh
    out = list(hierarchy_lines)
    out.extend(frames_header)
    for row in new_rows:
        out.append(" ".join(f"{v:.6f}" for v in row))
    with open(args.bvh_out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"Wrote {args.bvh_out}: {len(new_rows)} frames, "
          f"{sum(n for _, n in dst_channel_order)} channels "
          f"(was {sum(n for _, n in src_layout)})")
    # Sanity
    src_joint_count = len(src_layout)
    dst_joint_count = len(dst_channel_order)
    print(f"Joint count: {src_joint_count} → {dst_joint_count} (+{dst_joint_count - src_joint_count})")

    if args.skip_xml:
        return
    # Patch zygote_skel.xml: change spine bones to Ball + add bvh= attr
    import xml.etree.ElementTree as ET
    tree = ET.parse(SKEL_XML)
    skel_root = tree.getroot()
    patched = 0
    for node in skel_root.findall("Node"):
        name = node.attrib["name"]
        if name not in SKEL_TO_BVH:
            continue
        j = node.find("Joint")
        if j is None:
            continue
        j.attrib.clear()
        j.attrib["type"] = "Ball"
        j.attrib["bvh"] = SKEL_TO_BVH[name]
        j.attrib["lower"] = "-1.57 -1.57 -1.57"
        j.attrib["upper"] = "1.57 1.57 1.57"
        patched += 1
    tree.write(SKEL_XML)
    print(f"Patched {patched} spine joints in {SKEL_XML} (Ball + bvh=)")


if __name__ == "__main__":
    main()
