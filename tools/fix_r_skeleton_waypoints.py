"""Recompute R-side skeleton-attached waypoint local_pos from L-side tet.

The mirror script blindly negated local_pos[0], which only works for body
frames with mirror-symmetric rotations.  Limb bones (L_Femur, R_Femur, etc.)
have non-mirror-symmetric rotation matrices, so the naive negation produces
wrong world positions (e.g. Gluteus Medius insertion ends up 32cm below the
greater trochanter).

For each L tet file, find R counterpart.  At T-pose, walk skeleton-attached
waypoint_bary_coords entries on L; compute world position via L body
transform; mirror X; inverse-transform through the corresponding R body to
get the correct R local_pos.  Write fixed R tet.
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.dartHelper import saveSkeletonInfo, buildFromInfo


# Map L body name -> R body name (after digit-aware suffix handling).
SKELETON_NAME_MAP = {
    "L_Os_Coxae": "R_Os_Coxae",
    "L_Femur": "R_Femur",
    "L_Tibia_Fibula": "R_Tibia_Fibula",
    "L_Patella": "R_Patella",
    "L_Talus": "R_Talus",
    "L_Calcaneus": "R_Calcaneus",
    "L_Metatarsal": "R_Metatarsal",
    "L_Toe1": "R_Toe1",
    "L_Toe2": "R_Toe2",
    "L_Toe3": "R_Toe3",
    "L_Toe4": "R_Toe4",
    "L_Toe5": "R_Toe5",
}


def mirror_body_name(name):
    if not name.startswith("L_"):
        return name
    suffix = ""
    base = name
    for i in range(len(name) - 1, -1, -1):
        if name[i].isdigit():
            suffix = name[i] + suffix
        else:
            base = name[:i + 1]
            break
    mapped = SKELETON_NAME_MAP.get(base, "R_" + base[2:])
    return mapped + suffix


def get_body_transform(skel, name):
    b = skel.getBodyNode(name)
    if b is None:
        return None
    T = b.getWorldTransform().matrix()
    return np.array(T[:3, :3]), np.array(T[:3, 3])


def fix_r_tet(l_tet_path, r_tet_path, skel, dry_run=False):
    with open(l_tet_path, 'rb') as f:
        l_data = pickle.load(f)
    with open(r_tet_path, 'rb') as f:
        r_data = pickle.load(f)
    l_wbc = l_data.get('waypoint_bary_coords')
    r_wbc = r_data.get('waypoint_bary_coords')
    if l_wbc is None or r_wbc is None:
        return 0
    fixed = 0
    for s_idx, l_stream in enumerate(l_wbc):
        if s_idx >= len(r_wbc):
            break
        r_stream = r_wbc[s_idx]
        if l_stream is None or r_stream is None:
            continue
        for c_idx, l_contour in enumerate(l_stream):
            if c_idx >= len(r_stream):
                break
            r_contour = r_stream[c_idx]
            if l_contour is None or r_contour is None:
                continue
            for f_idx, l_fiber in enumerate(l_contour):
                if f_idx >= len(r_contour):
                    break
                r_fiber = r_contour[f_idx]
                if l_fiber is None or r_fiber is None:
                    continue
                if l_fiber[0] != 'skeleton' or r_fiber[0] != 'skeleton':
                    continue
                _, l_body_name, l_local = l_fiber
                r_body_name = mirror_body_name(l_body_name)
                lt = get_body_transform(skel, l_body_name)
                rt = get_body_transform(skel, r_body_name)
                if lt is None or rt is None:
                    continue
                L_R, L_t = lt
                R_R, R_t = rt
                # world from L body
                world_L = L_R @ np.asarray(l_local) + L_t
                # mirror across YZ plane
                world_R = world_L.copy()
                world_R[0] *= -1
                # inverse-transform through R body
                r_local = R_R.T @ (world_R - R_t)
                # Replace R entry
                r_contour[f_idx] = ('skeleton', r_body_name, r_local.astype(np.float32))
                fixed += 1
    if fixed and not dry_run:
        with open(r_tet_path, 'wb') as f:
            pickle.dump(r_data, f)
    return fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tet-dir', default='tet')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', nargs='*', help='restrict to muscle name(s) without L_/R_ prefix')
    args = ap.parse_args()

    skel_info, root_name, _, _, _, _ = saveSkeletonInfo('data/zygote_skel.xml')
    skel = buildFromInfo(skel_info, root_name)
    skel.resetPositions()

    paths = sorted(glob.glob(os.path.join(args.tet_dir, 'L_*_tet.npz')))
    total = 0
    for lp in paths:
        base = os.path.basename(lp).replace('L_', '').replace('_tet.npz', '')
        if args.only and base not in args.only:
            continue
        rp = os.path.join(args.tet_dir, 'R_' + base + '_tet.npz')
        if not os.path.exists(rp):
            continue
        n = fix_r_tet(lp, rp, skel, dry_run=args.dry_run)
        if n > 0:
            print(f'  R_{base}: fixed {n} skeleton entries')
            total += n
    print(f'Total: {total} entries fixed across all R tet files')


if __name__ == '__main__':
    main()
