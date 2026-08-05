"""Regenerate LowLeg tetrahedra in-place with stricter quality.

Existing tet npz files use loose tetgen params (minratio=2.0, mindihedral=5)
which produced sliver-rich LowLeg meshes (aspect_p99 up to 408) that prevented
ARAP from converging during bake (max_disp oscillates at ~6e-4 instead of
hitting the threshold). Same flags work fine on UpLeg because contour-mesh
geometry there yields cleaner PLCs.

This tool runs tetgen with `minratio=1.2 mindihedral=10` (matching
`tools/tet_from_original_obj.py`) on the already-built closed surface stored
in each tet npz. Surface vertices are preserved (`nobisect=True`,
`steinerleft=0`) so all index arrays — anchor_vertices, cap_face_indices,
cap_attachments, waypoint_bary_coords — stay valid.

Run after editing viewer/tetrahedron_mesh.py to match the same quality, so
future contour-mesh tetrahedralizations from the viewer also avoid slivers.

Usage:  python tools/regen_lowleg_tets.py [--dry-run]
"""
import argparse
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np


LOW_LEG_MUSCLES_L = [
    "L_Extensor_Digitorum_Longus",
    "L_Extensor_Hallucis_Longus",
    "L_Flexor_Digitorum_Longus",
    "L_Flexor_Hallucis",
    "L_Gastrocnemius",
    "L_Peroneus_Brevis",
    "L_Peroneus_Longus",
    "L_Peroneus_Tertius",
    "L_Plantaris",
    "L_Soleus",
    "L_Tibialis_Anterior",
    "L_Tibialis_Posterior",
]


def quality_stats(verts, tets):
    v = np.asarray(verts, dtype=np.float64)
    t = np.asarray(tets, dtype=np.int64)
    p0 = v[t[:, 0]]; p1 = v[t[:, 1]]; p2 = v[t[:, 2]]; p3 = v[t[:, 3]]
    edges = np.stack([p1 - p0, p2 - p0, p3 - p0,
                      p2 - p1, p3 - p1, p3 - p2], axis=1)
    el = np.linalg.norm(edges, axis=2)
    aspect = el.max(axis=1) / np.maximum(el.min(axis=1), 1e-12)
    vol = np.einsum('ij,ij->i', np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0
    return {
        'n_tets': len(t),
        'aspect_p99': float(np.percentile(aspect, 99)),
        'aspect_med': float(np.median(aspect)),
        'inverted': int((vol < 0).sum()),
        'min_vol': float(np.abs(vol).min()),
    }


def regenerate(in_path, out_path, dry_run=False):
    with open(in_path, 'rb') as f:
        d = pickle.load(f)
    verts = np.asarray(d['vertices'], dtype=np.float64)
    faces = np.asarray(d['faces'], dtype=np.int32)
    n_verts_in = len(verts)

    old_stats = quality_stats(verts, d['tetrahedra'])

    rv = verts * 1000.0  # tetgen prefers mm scale
    rf = faces.copy()

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as inp:
        np.savez(inp, v=rv, f=rf)
        inp_path = inp.name
    out_npz = inp_path.replace('.npz', '_out.npz')

    script = f'''
import numpy as np, tetgen, trimesh, pymeshfix, sys
d = np.load("{inp_path}")
rv = d["v"].astype(np.float64); rf = d["f"].astype(np.int32)
n_in = len(rv)
# pymeshfix repair to make surface watertight + manifold for strict tetgen.
fixer = pymeshfix.MeshFix(rv.copy(), rf.copy())
fixer.repair(verbose=False)
fv, ff = fixer.v.astype(np.float64), fixer.f.astype(np.int32)
mesh = trimesh.Trimesh(vertices=fv, faces=ff, process=False)
max_vol = max(abs(mesh.volume) / 1500.0, 1e-6)
def run_tet(quality, verts, faces):
    t = tetgen.TetGen(verts.copy(), faces.copy())
    if quality:
        t.tetrahedralize(order=1, mindihedral=10, minratio=1.2,
                         maxvolume=max_vol, nobisect=True, steinerleft=0)
    else:
        t.tetrahedralize(quality=False, nobisect=True)
    return t
try:
    t = run_tet(True, fv, ff)
except Exception as e1:
    sys.stderr.write(f"strict failed: {{e1}}\\n")
    t = run_tet(False, fv, ff)
np.savez("{out_npz}", node=np.asarray(t.node),
         elem=np.asarray(t.elem).astype(np.int32),
         n_in=np.array([n_in]))
'''
    # Use python3.10 explicitly — tetgen is installed in user-site for 3.10,
    # not for the default `python3` (which is also 3.10 but ignores user-site
    # in some shell setups). Explicit path mirrors what pip3 / pip3 show says.
    py = '/usr/bin/python3.10'
    try:
        proc = subprocess.run([py, '-c', script],
                              capture_output=True, timeout=180, text=True)
    except subprocess.TimeoutExpired:
        os.unlink(inp_path)
        if os.path.exists(out_npz):
            os.unlink(out_npz)
        print(f'  TIMEOUT for {in_path}')
        return False

    if proc.returncode != 0 or not os.path.exists(out_npz):
        os.unlink(inp_path)
        if os.path.exists(out_npz):
            os.unlink(out_npz)
        print(f'  FAIL tetgen rc={proc.returncode} stderr={proc.stderr[-300:]}')
        return False

    out = np.load(out_npz)
    new_node = np.asarray(out['node'], dtype=np.float64) / 1000.0
    new_tets = np.asarray(out['elem'], dtype=np.int32)
    os.unlink(inp_path); os.unlink(out_npz)

    if len(new_node) != n_verts_in:
        print(f'  WARN tetgen returned {len(new_node)} verts vs input {n_verts_in} '
              f'— index arrays would break, skipping')
        return False

    new_stats = quality_stats(new_node, new_tets)
    print(f'  tets: {old_stats["n_tets"]} -> {new_stats["n_tets"]}'
          f'  aspect_p99: {old_stats["aspect_p99"]:.1f} -> {new_stats["aspect_p99"]:.1f}'
          f'  median: {old_stats["aspect_med"]:.1f} -> {new_stats["aspect_med"]:.1f}'
          f'  inv: {new_stats["inverted"]}')

    if dry_run:
        return True

    d['vertices'] = new_node.astype(np.float32)
    d['tetrahedra'] = new_tets
    with open(out_path, 'wb') as f:
        pickle.dump(d, f)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--names', nargs='+', default=None,
                        help='Override which muscle tets to regen')
    args = parser.parse_args()

    names = args.names if args.names else LOW_LEG_MUSCLES_L
    n_ok = n_fail = 0
    for name in names:
        in_path = f'tet/{name}_tet.npz'
        if not os.path.exists(in_path):
            print(f'{name}: missing')
            n_fail += 1
            continue
        print(f'{name}:')
        ok = regenerate(in_path, in_path, dry_run=args.dry_run)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
    print(f'\nOK: {n_ok}  FAIL: {n_fail}')


if __name__ == '__main__':
    main()
