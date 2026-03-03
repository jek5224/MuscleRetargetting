"""Convert BVH file coordinate system by rotating skeleton and motion data.

Applies a rotation to bone offsets, root positions, and all joint rotations
so the rest pose matches the target convention (Y-up, legs in -Y).

Usage:
    python tools/convert_bvh_axes.py input.bvh output.bvh --rot X -90
    python tools/convert_bvh_axes.py input.bvh output.bvh --rot Z 90 --rot X -90
"""
import sys
import argparse
import re
import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_bvh(filepath):
    """Parse BVH into hierarchy + motion data."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    hierarchy_lines = []
    motion_lines = []
    num_frames = 0
    frame_time = 0.0
    in_motion = False
    header_done = False

    # Track joints in order, with their channel info
    joints = []  # list of {name, channels, parent_idx, offset, is_end}
    joint_stack = []  # stack of indices
    channel_layout = []  # (joint_idx, channel_names)

    for line in lines:
        stripped = line.strip()

        if stripped == "MOTION":
            in_motion = True
            continue

        if not in_motion:
            hierarchy_lines.append(line)

            upper = stripped.upper()
            if "ROOT" in upper or "JOINT" in upper:
                name = stripped.split()[-1]
                parent = joint_stack[-1] if joint_stack else -1
                idx = len(joints)
                joints.append({
                    'name': name, 'channels': [], 'parent_idx': parent,
                    'offset': None, 'is_end': False
                })
                joint_stack.append(idx)
            elif "END" in upper and "SITE" in upper:
                parent = joint_stack[-1] if joint_stack else -1
                idx = len(joints)
                joints.append({
                    'name': f"End_{joints[parent]['name']}",
                    'channels': [], 'parent_idx': parent,
                    'offset': None, 'is_end': True
                })
                joint_stack.append(idx)
            elif "OFFSET" in upper:
                vals = [float(x) for x in stripped.split()[1:]]
                joints[joint_stack[-1]]['offset'] = np.array(vals)
            elif "CHANNELS" in upper:
                parts = stripped.split()
                chans = [c.lower() for c in parts[2:]]
                joints[joint_stack[-1]]['channels'] = chans
                channel_layout.append((joint_stack[-1], chans))
            elif "}" in stripped:
                joint_stack.pop()
        else:
            if "Frames:" in stripped:
                num_frames = int(stripped.split(":")[1].strip())
            elif "Frame Time:" in stripped:
                frame_time = float(stripped.split(":")[1].strip())
                header_done = True
            elif header_done:
                motion_lines.append(stripped)

    # Parse motion data
    frames = []
    for mline in motion_lines:
        vals = [float(x) for x in mline.split()]
        frames.append(vals)
    frames = np.array(frames)

    return {
        'hierarchy_lines': hierarchy_lines,
        'joints': joints,
        'channel_layout': channel_layout,
        'num_frames': num_frames,
        'frame_time': frame_time,
        'frames': frames,
    }


def get_euler_order(channels):
    """Extract euler order string from channel names."""
    rot_chans = [c for c in channels if 'rotation' in c]
    return "".join([c[0].upper() for c in rot_chans])


def apply_rotation_to_bvh(bvh, rot_matrix):
    """Apply a global rotation to all offsets, root positions, and rotations."""
    joints = bvh['joints']
    channel_layout = bvh['channel_layout']
    frames = bvh['frames'].copy()
    rot_global = R.from_matrix(rot_matrix)

    # Rotate all bone offsets
    for j in joints:
        if j['offset'] is not None:
            j['offset'] = rot_matrix @ j['offset']

    # Rotate motion data
    col = 0
    for joint_idx, channels in channel_layout:
        n_ch = len(channels)
        is_root = (joints[joint_idx]['parent_idx'] == -1)

        if is_root and n_ch == 6:
            # Root: has position + rotation
            pos_map = {}
            rot_chans = []
            for i, c in enumerate(channels):
                if 'position' in c:
                    pos_map[c[0]] = col + i
                elif 'rotation' in c:
                    rot_chans.append((col + i, c))

            # Rotate positions
            px, py, pz = pos_map['x'], pos_map['y'], pos_map['z']
            positions = np.column_stack([frames[:, px], frames[:, py], frames[:, pz]])
            positions_rot = (rot_matrix @ positions.T).T
            frames[:, px] = positions_rot[:, 0]
            frames[:, py] = positions_rot[:, 1]
            frames[:, pz] = positions_rot[:, 2]

            # Rotate rotations: R_new = R_global @ R_original
            euler_order = get_euler_order(channels)
            rot_cols = [rc[0] for rc in rot_chans]
            rot_data = frames[:, rot_cols]
            for i in range(len(frames)):
                r_orig = R.from_euler(euler_order, rot_data[i], degrees=True)
                r_new = rot_global * r_orig
                frames[i, rot_cols] = r_new.as_euler(euler_order, degrees=True)

        elif n_ch == 3:
            # Non-root joint: rotations are relative to parent, no change needed
            # (The offset rotation handles the coordinate change)
            pass

        col += n_ch

    bvh['frames'] = frames
    return bvh


def write_bvh(bvh, filepath):
    """Write BVH back to file with updated offsets and motion data."""
    joints = bvh['joints']

    # Rebuild hierarchy with updated offsets
    with open(filepath, 'w') as f:
        joint_stack = []
        joint_iter = iter(joints)

        for line in bvh['hierarchy_lines']:
            stripped = line.strip().upper()

            if 'OFFSET' in stripped:
                # Find which joint this offset belongs to
                # Match it by position in the hierarchy
                current_joint = joints[joint_stack[-1]] if joint_stack else None
                if current_joint and current_joint['offset'] is not None:
                    off = current_joint['offset']
                    indent = line[:len(line) - len(line.lstrip())]
                    f.write(f"{indent}OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}\n")
                    continue

            # Track joint stack for offset matching
            if 'ROOT' in stripped or ('JOINT' in stripped and 'END' not in stripped):
                name = line.strip().split()[-1]
                idx = next(i for i, j in enumerate(joints)
                           if j['name'] == name and i not in joint_stack)
                joint_stack.append(idx)
            elif 'END' in stripped and 'SITE' in stripped:
                parent = joint_stack[-1]
                idx = next(i for i, j in enumerate(joints)
                           if j['is_end'] and j['parent_idx'] == parent and i not in joint_stack)
                joint_stack.append(idx)
            elif '}' in stripped and joint_stack:
                joint_stack.pop()

            f.write(line)

        # Write motion
        f.write("MOTION\n")
        f.write(f"Frames: {bvh['num_frames']}\n")
        f.write(f"Frame Time: {bvh['frame_time']:.6f}\n")
        for frame in bvh['frames']:
            f.write(" ".join(f"{v:.6f}" for v in frame) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Rotate BVH coordinate system")
    parser.add_argument("input", help="Input BVH file")
    parser.add_argument("output", help="Output BVH file")
    parser.add_argument("--rot", nargs=2, action='append', metavar=('AXIS', 'DEGREES'),
                        help="Rotation to apply (e.g. --rot X -90). Can be repeated.")

    args = parser.parse_args()

    if not args.rot:
        print("No rotations specified. Use --rot AXIS DEGREES")
        sys.exit(1)

    # Build combined rotation matrix
    rot_combined = np.eye(3)
    for axis, degrees in args.rot:
        angle = float(degrees)
        axis = axis.upper()
        r = R.from_euler(axis, angle, degrees=True)
        rot_combined = r.as_matrix() @ rot_combined
        print(f"  Rotation: {angle}° around {axis}")

    print(f"Loading {args.input}...")
    bvh = parse_bvh(args.input)
    print(f"  {bvh['num_frames']} frames, {len(bvh['joints'])} joints")

    print("Applying rotation...")
    bvh = apply_rotation_to_bvh(bvh, rot_combined)

    # Show root offset and first-frame root position for verification
    root = bvh['joints'][0]
    print(f"  Root offset after: {root['offset']}")
    print(f"  Frame 0 root pos: {bvh['frames'][0, :3]}")

    print(f"Writing {args.output}...")
    write_bvh(bvh, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
