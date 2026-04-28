#!/usr/bin/env python3
"""Render iron-man-style convergence snapshots dumped by bake_layered.py.

Reads npz files from `<cache_dir>/_anim/frame*_layer*.npz` and, for each iter
snapshot, writes a PNG showing the L_UpLeg muscles + femur / pelvis / patella
bones at the corresponding BVH frame pose.

Usage:
    python tools/render_iron_man.py \
        --cache-dir data/motion_cache/walk/_layered_nc \
        --bvh data/motion/walk.bvh \
        --frame 60 \
        --out-dir /tmp/iron_man_pngs
"""
import argparse
import os
import sys
import glob
import numpy as np
import trimesh
import pyrender

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dartHelper import buildFromInfo, saveSkeletonInfo
from core.bvhparser import MyBVH


SKEL_XML = "data/zygote_skel.xml"
BONE_OBJ_DIR = "Zygote_Meshes_251229/Skeleton"
MESH_SCALE = 0.01  # cm → m (matches bake_layered convention)


def _detect_bvh_tframe(bvh_path):
    with open(bvh_path, 'r') as f:
        for line in f:
            if 'JOINT' in line or 'ROOT' in line:
                continue
            if 'OFFSET' in line and abs(float(line.split()[-1])) > 0.5:
                return 0  # heuristic
    return None


def load_bone_trimeshes(skel):
    """Load Zygote skeleton OBJs scaled by MESH_SCALE; resolve to DART body node."""
    out = []
    for fname in os.listdir(BONE_OBJ_DIR):
        if not fname.lower().endswith(('.obj', '.stl', '.ply')):
            continue
        stem = os.path.splitext(fname)[0]
        try:
            tm = trimesh.load(os.path.join(BONE_OBJ_DIR, fname), process=False)
        except Exception:
            continue
        if not isinstance(tm, trimesh.Trimesh):
            continue
        tm.vertices = tm.vertices * MESH_SCALE
        # DART body node names append a trailing '0'
        bn = None
        for cand in (stem + '0', stem, stem + '1'):
            try:
                bn = skel.getBodyNode(cand)
                if bn is not None:
                    break
            except Exception:
                continue
        if bn is None:
            continue
        out.append((bn, tm))
    return out


def shape_world_transform(body_node):
    """Combined body_world * shape_relative — same composition viewer uses."""
    bw = body_node.getWorldTransform().matrix()
    sn_iter = body_node.getShapeNodes()
    if sn_iter is not None and len(sn_iter) > 0:
        sr = sn_iter[0].getRelativeTransform().matrix()
        return bw @ sr
    return bw


def transform_bone(tm, body_node):
    """Apply DART body's shape-world transform to shape-local OBJ verts."""
    M = shape_world_transform(body_node)
    R = M[:3, :3]
    t = M[:3, 3]
    verts = (R @ tm.vertices.T).T + t
    return trimesh.Trimesh(vertices=verts, faces=tm.faces.copy(), process=False)


def build_scene(snapshot_positions, anim, bone_meshes_world, muscle_color, bone_color,
                only_muscle=None):
    """Build pyrender scene with muscle meshes at given snapshot + bones."""
    scene = pyrender.Scene(bg_color=[0.07, 0.07, 0.10, 1.0],
                           ambient_light=[0.3, 0.3, 0.3])
    names = anim['muscle_names']
    offsets = anim['offsets']
    n_verts = anim['n_verts']
    faces_flat = anim['surf_faces_flat']
    faces_lens = anim['surf_faces_lens']
    f_off = 0
    for i, n in enumerate(names):
        nm = str(n)
        n_v = int(n_verts[i])
        v0 = int(offsets[i])
        f_len = int(faces_lens[i])
        if only_muscle is not None and nm != only_muscle:
            f_off += f_len
            continue
        verts = snapshot_positions[v0:v0 + n_v]
        if f_len == 0:
            continue
        faces = faces_flat[f_off:f_off + f_len].reshape(-1, 3)
        f_off += f_len
        try:
            tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        except Exception:
            continue
        m = pyrender.Mesh.from_trimesh(
            tm, smooth=True,
            material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=muscle_color, metallicFactor=0.1, roughnessFactor=0.6,
            ),
        )
        scene.add(m)
    for tm in bone_meshes_world:
        m = pyrender.Mesh.from_trimesh(
            tm, smooth=True,
            material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=bone_color, metallicFactor=0.05, roughnessFactor=0.85,
            ),
        )
        scene.add(m)
    return scene


def fit_camera(scene, scale=1.4, azimuth_deg=30.0, elev_deg=10.0):
    """Position perspective camera looking at scene center."""
    bounds = scene.bounds  # (2, 3)
    center = (bounds[0] + bounds[1]) * 0.5
    extent = float(np.linalg.norm(bounds[1] - bounds[0]))
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elev_deg)
    dist = extent * scale
    eye = center + dist * np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    forward = (center - eye)
    forward /= (np.linalg.norm(forward) + 1e-9)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    scene.add(cam, pose=pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    scene.add(light, pose=pose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--bvh', required=True)
    ap.add_argument('--frame', type=int, required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--width', type=int, default=900)
    ap.add_argument('--height', type=int, default=900)
    ap.add_argument('--azimuth', type=float, default=35.0)
    ap.add_argument('--elev', type=float, default=8.0)
    ap.add_argument('--only-muscle', default=None,
                    help='Render only this muscle (e.g. L_Vastus_Intermedius).')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

    # Load skeleton + apply BVH frame pose so bones are at the right place.
    skel_info, root_name, bvh_info, _pd, mesh_info, _smpl = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)

    # OBJ verts are in shape-local (per <Body><Transformation> in zygote_skel.xml).
    # bake_layered scales them by 0.01 — keep that. The shape's rest-world
    # position is body_world * shape_relative at zero pose; converting OBJ
    # verts to shape-local just means dividing out that rest-world transform.
    skel.setPositions(np.zeros(skel.getNumDofs()))
    bone_pairs = load_bone_trimeshes(skel)  # (body_node, world-rest trimesh)
    rest_local_pairs = []
    for bn, tm in bone_pairs:
        M0 = shape_world_transform(bn)
        R0, t0 = M0[:3, :3], M0[:3, 3]
        local_verts = (R0.T @ (tm.vertices - t0).T).T
        rest_local_pairs.append((bn, trimesh.Trimesh(vertices=local_verts,
                                                      faces=tm.faces.copy(),
                                                      process=False)))

    # Now pose the skeleton at the target frame.
    t_frame = _detect_bvh_tframe(args.bvh)
    motion = MyBVH(args.bvh, bvh_info, skel, T_frame=t_frame)
    skel.setPositions(motion.mocap_refs[args.frame])

    bones = [transform_bone(tm, bn) for bn, tm in rest_local_pairs]
    print(f'Loaded {len(bones)} bone meshes')

    # Quick skeleton-only sanity render at this frame.
    sanity_scene = pyrender.Scene(bg_color=[0.07, 0.07, 0.10, 1.0],
                                  ambient_light=[0.3, 0.3, 0.3])
    for tm in bones:
        m = pyrender.Mesh.from_trimesh(
            tm, smooth=True,
            material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.92, 0.90, 0.84, 1.0],
                metallicFactor=0.05, roughnessFactor=0.85))
        sanity_scene.add(m)
    fit_camera(sanity_scene, azimuth_deg=args.azimuth, elev_deg=args.elev)
    sanity_r = pyrender.OffscreenRenderer(viewport_width=args.width,
                                           viewport_height=args.height)
    sanity_color, _ = sanity_r.render(sanity_scene)
    sanity_path = os.path.join(args.out_dir,
                                f'_sanity_skel_frame{args.frame:04d}.png')
    from PIL import Image
    Image.fromarray(sanity_color).save(sanity_path)
    sanity_r.delete()
    print(f'  wrote {sanity_path}')

    anim_files = sorted(glob.glob(os.path.join(args.cache_dir, '_anim',
                                                f'frame{args.frame:04d}_layer*.npz')))
    if not anim_files:
        print(f'No anim files in {args.cache_dir}/_anim for frame {args.frame}')
        return
    print(f'Found {len(anim_files)} layer anim files')

    r = pyrender.OffscreenRenderer(viewport_width=args.width,
                                   viewport_height=args.height)
    muscle_color = [0.85, 0.25, 0.25, 1.0]   # red muscle
    bone_color = [0.92, 0.90, 0.84, 1.0]     # bone ivory

    for anim_path in anim_files:
        anim = np.load(anim_path, allow_pickle=False)
        snapshots = anim['snapshots']  # (K, total_verts, 3)
        layer_tag = os.path.basename(anim_path).replace('.npz', '')
        for k in range(snapshots.shape[0]):
            scene = build_scene(snapshots[k], anim, bones, muscle_color, bone_color,
                                only_muscle=args.only_muscle)
            fit_camera(scene, azimuth_deg=args.azimuth, elev_deg=args.elev)
            color, _ = r.render(scene)
            png = os.path.join(args.out_dir, f'{layer_tag}_iter{k:02d}.png')
            from PIL import Image
            Image.fromarray(color).save(png)
            print(f'  wrote {png}')
    r.delete()


if __name__ == '__main__':
    main()
