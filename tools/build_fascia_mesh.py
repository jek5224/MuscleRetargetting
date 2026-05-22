"""Build per-region fascia rest meshes from the anatomical surfaces of
the lower-body muscles at A-pose.

Produces one NPZ per region (L_UpLeg, R_UpLeg, L_LowLeg, R_LowLeg) at:
    data/zygote_fascia_rest_{REGION}.npz

NPZ keys:
    vertices:    (V_f, 3) float32, pelvis-local A-pose coords (meters)
    faces:       (F_f, 3) int32
    edges:       (E, 2) int32 (unique undirected)
    src_muscle:  (V_f,) int32 — index into `muscle_names` of bound muscle
    src_tri:     (V_f,) int32 — local anatomical tri index in that muscle
    bary:        (V_f, 3) float32 — barycentric weights at A-pose
    muscle_names: (M,) <U str — local muscle name list for this region

Pipeline:
  1. For each region, load every muscle's tet rest geometry.
  2. Extract anatomical surface = render_faces minus cap_face_indices.
  3. Concatenate region surfaces in pelvis-local A-pose frame.
  4. Outward-dilate each vertex by OFFSET (default 3 mm) along its
     area-weighted vertex normal.
  5. Voxel-fill the dilated union → marching_cubes → apply voxel transform
     → watertight shell.
  6. Decimate to ~TARGET_VERTS.
  7. Per fascia vert: closest anatomical tri (KDTree), barycentric weights.
  8. Save NPZ.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


REGIONS = {
    'L_UpLeg':  '.muscles_L_UpLeg.json',
    'R_UpLeg':  '.muscles_R_UpLeg.json',
    'L_LowLeg': '.muscles_L_LowLeg.json',
    'R_LowLeg': '.muscles_R_LowLeg.json',
}
TET_DIR = 'tet'
DEFAULT_OFFSET = 3e-3
DEFAULT_TARGET_VERTS = 5000   # per region (4 regions x 5k ≈ 20k total)


def load_region_muscles(json_path):
    with open(json_path) as f:
        return [m['name'] for m in json.load(f)]


def vertex_normals_area_weighted(V, F):
    tri = V[F]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    face_n = cross / (2.0 * area[:, None] + 1e-12)
    Vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(Vn, F[:, k], face_n * area[:, None])
    norm = np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-12
    return Vn / norm


def build_region(region, json_path, offset, voxel_pitch, target_verts, out_path):
    muscles = load_region_muscles(json_path)
    print(f'\n=== {region}: {len(muscles)} muscles ===')

    muscle_data = []  # list of (name, V, F_anat)
    for name in muscles:
        p = os.path.join(TET_DIR, f'{name}_tet.npz')
        if not os.path.exists(p):
            print(f'  skip (no tet): {name}')
            continue
        with open(p, 'rb') as f:
            d = pickle.load(f)
        V = np.asarray(d['vertices'], dtype=np.float32)
        F_render = np.asarray(d['render_faces'], dtype=np.int32)
        cap = np.asarray(d['cap_face_indices'], dtype=np.int64)
        mask = np.ones(len(F_render), dtype=bool)
        mask[cap] = False
        muscle_data.append((name, V, F_render[mask]))
    print(f'  loaded {len(muscle_data)} muscles')

    big_V, big_F, tri_to_muscle, tri_to_local = [], [], [], []
    v_offset = 0
    for m_idx, (name, V, F_anat) in enumerate(muscle_data):
        big_V.append(V)
        big_F.append(F_anat + v_offset)
        tri_to_muscle.extend([m_idx] * len(F_anat))
        tri_to_local.extend(range(len(F_anat)))
        v_offset += len(V)
    big_V = np.concatenate(big_V, axis=0)
    big_F = np.concatenate(big_F, axis=0)
    tri_to_muscle = np.array(tri_to_muscle, dtype=np.int32)
    tri_to_local = np.array(tri_to_local, dtype=np.int32)
    print(f'  union surface: {len(big_V)} verts, {len(big_F)} tris')

    Vn = vertex_normals_area_weighted(big_V, big_F)
    dilated = big_V + offset * Vn
    print(f'  dilated by {offset*1000:.1f} mm')

    raw = trimesh.Trimesh(vertices=dilated, faces=big_F, process=False)
    voxel = raw.voxelized(pitch=voxel_pitch).fill()
    fascia = voxel.marching_cubes
    fascia.apply_transform(voxel.transform)  # voxel-index -> world coords
    print(f'  voxel mesh: {len(fascia.vertices)} verts, {len(fascia.faces)} tris '
          f'(watertight={fascia.is_watertight})')

    if len(fascia.vertices) > target_verts:
        try:
            frac = max(0.01, min(0.99, target_verts / len(fascia.vertices)))
            fascia = fascia.simplify_quadric_decimation(percent=frac)
            print(f'  simplified to: {len(fascia.vertices)} verts, '
                  f'{len(fascia.faces)} tris')
        except Exception as e:
            print(f'  simplify skipped ({e})')

    Vf = np.asarray(fascia.vertices, dtype=np.float32)
    Ff = np.asarray(fascia.faces, dtype=np.int32)

    # KDTree binding: fascia vert -> nearest anatomical tri centroid
    tri_centroids = big_V[big_F].mean(axis=1)
    tree = cKDTree(tri_centroids)
    _, nearest = tree.query(Vf, k=1)
    src_muscle = tri_to_muscle[nearest]
    src_tri_local = tri_to_local[nearest]

    a = big_V[big_F[nearest, 0]]
    b = big_V[big_F[nearest, 1]]
    c = big_V[big_F[nearest, 2]]
    n = np.cross(b - a, c - a)
    n_unit = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    p_proj = Vf - np.sum((Vf - a) * n_unit, axis=1, keepdims=True) * n_unit
    v0 = b - a; v1 = c - a; v2 = p_proj - a
    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)
    denom = d00 * d11 - d01 * d01 + 1e-12
    v_b = (d11 * d20 - d01 * d21) / denom
    w_b = (d00 * d21 - d01 * d20) / denom
    u_b = 1.0 - v_b - w_b
    bary = np.clip(np.stack([u_b, v_b, w_b], axis=1), -0.5, 1.5).astype(np.float32)

    edges_set = set()
    for tri in Ff:
        for k in range(3):
            i, j = int(tri[k]), int(tri[(k + 1) % 3])
            edges_set.add((min(i, j), max(i, j)))
    edges = np.array(sorted(edges_set), dtype=np.int32)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.savez(out_path,
             vertices=Vf, faces=Ff, edges=edges,
             src_muscle=src_muscle.astype(np.int32),
             src_tri=src_tri_local.astype(np.int32),
             bary=bary,
             muscle_names=np.array([n for n, _, _ in muscle_data]),
             offset_m=np.array([offset], dtype=np.float32),
             voxel_pitch_m=np.array([voxel_pitch], dtype=np.float32),
             )
    print(f'  saved {out_path}  '
          f'(verts={len(Vf)} faces={len(Ff)} edges={len(edges)} bbox={Vf.min(0)}..{Vf.max(0)})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='data')
    ap.add_argument('--offset', type=float, default=DEFAULT_OFFSET)
    ap.add_argument('--target-verts', type=int, default=DEFAULT_TARGET_VERTS)
    ap.add_argument('--voxel-pitch', type=float, default=5e-3)
    ap.add_argument('--regions', nargs='*', default=list(REGIONS.keys()))
    args = ap.parse_args()

    for region in args.regions:
        if region not in REGIONS:
            print(f'unknown region: {region}'); continue
        out = os.path.join(args.out_dir, f'zygote_fascia_rest_{region}.npz')
        build_region(region, REGIONS[region], args.offset,
                     args.voxel_pitch, args.target_verts, out)


if __name__ == '__main__':
    main()
