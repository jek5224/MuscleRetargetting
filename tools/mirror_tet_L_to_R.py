"""Mirror L tet data to create R tet data.

Mirrors vertices/waypoints along sagittal plane (negate X),
fixes face winding and tet orientation, remaps skeleton attachment names.

Usage: python tools/mirror_tet_L_to_R.py [--dry-run]
"""
import argparse
import glob
import os
import pickle
import re

import numpy as np


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

# Skeleton mesh list offset: R meshes = L meshes + 12
# (Saccrum_Coccyx0 at index 0, L starts at 1, R starts at 13)
L_TO_R_SKEL_MESH_OFFSET = 12


def mirror_x(arr):
    """Negate X component of Nx3 array."""
    out = arr.copy()
    out[..., 0] *= -1
    return out


def mirror_skeleton_indices(skel_list):
    """Remap L body node indices to R by adding offset."""
    mirrored = []
    for group in skel_list:
        mirrored.append([idx + L_TO_R_SKEL_MESH_OFFSET for idx in group])
    return mirrored


def mirror_skeleton_names(names_list):
    """Remap L_ skeleton names to R_."""
    mirrored = []
    for group in names_list:
        new_group = []
        for name in group:
            mapped = SKELETON_NAME_MAP.get(name, name)
            if mapped == name and name.startswith("L_"):
                # Fallback: replace L_ prefix with R_
                mapped = "R_" + name[2:]
            new_group.append(mapped)
        mirrored.append(new_group)
    return mirrored


def _mirror_body_name(name):
    """Remap a single L_ body name to R_."""
    mapped = SKELETON_NAME_MAP.get(name.replace("0", ""), name)
    if mapped == name and name.startswith("L_"):
        mapped = "R_" + name[2:]
    elif not name.startswith("L_"):
        mapped = name  # non-L names stay as-is
    else:
        # Re-add suffix (e.g. "0") if present in original
        suffix = ""
        for i in range(len(name) - 1, -1, -1):
            if name[i].isdigit():
                suffix = name[i] + suffix
            else:
                break
        if suffix and not mapped.endswith(suffix):
            mapped = mapped + suffix
    return mapped


def _mirror_bary_coords(bary_coords):
    """Mirror waypoint barycentric coordinates.

    - ('skeleton', body_name, local_pos) → remap name L→R, negate local_pos X
    - ('tet', tet_idx, bary, was_inside) → swap bary weights [0],[1] (tet verts swapped)
    """
    mirrored = []
    for stream in bary_coords:
        if stream is None:
            mirrored.append(None)
            continue
        stream_m = []
        for contour in stream:
            if contour is None:
                stream_m.append(None)
                continue
            contour_m = []
            for fiber in contour:
                if fiber is None:
                    contour_m.append(None)
                    continue
                if fiber[0] == 'skeleton':
                    _, body_name, local_pos = fiber
                    new_pos = local_pos.copy()
                    new_pos[0] *= -1  # negate X
                    contour_m.append(('skeleton', _mirror_body_name(body_name), new_pos))
                elif fiber[0] == 'tet':
                    if len(fiber) >= 4:
                        _, tet_idx, bary, was_inside = fiber
                    else:
                        _, tet_idx, bary = fiber
                        was_inside = True
                    # Swap bary weights 0,1 to match swapped tet vertex indices
                    new_bary = bary.copy()
                    new_bary[0], new_bary[1] = bary[1], bary[0]
                    contour_m.append(('tet', tet_idx, new_bary, was_inside))
                else:
                    contour_m.append(fiber)
            stream_m.append(contour_m)
        mirrored.append(stream_m)
    return mirrored


def mirror_tet_file(src_path, dst_path, dry_run=False):
    """Mirror a single L tet file to R. Files are pickle despite .npz extension."""
    with open(src_path, 'rb') as f:
        data = pickle.load(f)

    # Start with a copy, stripping contour/anim data (not valid after mirror)
    STRIP_KEYS = {
        'contours', 'fiber_architecture', 'bounding_planes',
        'draw_contour_stream', 'contour_to_tet_mapping',
        'mvc_weights', '_stream_endpoints',
        'stream_contours', 'stream_bounding_planes', 'stream_groups',
    }
    mirrored = {k: v for k, v in data.items() if k not in STRIP_KEYS}

    # Vertices: negate X
    mirrored["vertices"] = mirror_x(data["vertices"])

    # Faces: reverse winding order to fix normals
    for face_key in ["faces", "render_faces", "sim_faces"]:
        if face_key in data and data[face_key] is not None:
            mirrored[face_key] = data[face_key][:, ::-1].copy()

    # Tetrahedra: swap indices 0,1 to fix orientation after mirror
    tets = data["tetrahedra"].copy()
    tets[:, 0], tets[:, 1] = data["tetrahedra"][:, 1].copy(), data["tetrahedra"][:, 0].copy()
    mirrored["tetrahedra"] = tets

    # Waypoints: negate X of all positions
    if data.get("waypoints") is not None:
        mirrored_wp = []
        for stream in data["waypoints"]:
            stream_mirrored = []
            for wp in stream:
                wp = np.array(wp)
                if wp.ndim >= 1 and wp.shape[-1] == 3:
                    stream_mirrored.append(mirror_x(wp))
                else:
                    stream_mirrored.append(wp)
            mirrored_wp.append(stream_mirrored)
        mirrored["waypoints"] = mirrored_wp

    # Skeleton attachment names: L_ → R_
    if data.get("attach_skeleton_names") is not None:
        mirrored["attach_skeleton_names"] = mirror_skeleton_names(data["attach_skeleton_names"])

    # Skeleton attachment indices: index into skeleton_meshes list.
    # R meshes are at L index + 12 in the skeleton mesh list.
    if data.get("attach_skeletons") is not None:
        mirrored["attach_skeletons"] = mirror_skeleton_indices(data["attach_skeletons"])

    # Waypoint barycentric coords: mirror skeleton refs and swap tet bary weights 0,1
    if data.get("waypoint_bary_coords") is not None:
        mirrored["waypoint_bary_coords"] = _mirror_bary_coords(data["waypoint_bary_coords"])

    if dry_run:
        print(f"  [DRY RUN] Would write {dst_path}")
        return

    with open(dst_path, 'wb') as f:
        pickle.dump(mirrored, f)


def main():
    parser = argparse.ArgumentParser(description="Mirror L tet files to R")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files")
    args = parser.parse_args()

    tet_dir = "tet"
    l_files = sorted(glob.glob(os.path.join(tet_dir, "L_*_tet.npz")))
    print(f"Found {len(l_files)} L tet files")

    mirrored = 0
    skipped = 0
    for l_path in l_files:
        basename = os.path.basename(l_path)
        r_basename = "R_" + basename[2:]
        r_path = os.path.join(tet_dir, r_basename)

        # Check vertex count match with existing R obj mesh
        l_name = basename.replace("_tet.npz", "")
        r_name = "R_" + l_name[2:]

        print(f"{l_name} → {r_name}")
        mirror_tet_file(l_path, r_path, dry_run=args.dry_run)
        mirrored += 1

    print(f"\nMirrored: {mirrored}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
