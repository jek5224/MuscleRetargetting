#!/usr/bin/env python
"""Test FEM gradient correctness on the full system."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['TAICHI_LOG_LEVEL'] = 'warn'

import numpy as np
import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64, debug=False)

from viewer.fem_sim import UnifiedFEMSolver, _compute_gradient_and_energy, _zero_field

# Load skeleton + muscles through the standard pipeline
from core.dartHelper import DartHelper
from viewer.muscle_mesh import MuscleMesh
import json

print("Loading skeleton...")
dart = DartHelper()
skel = dart.load_skeleton('data/zygote_skel.xml')
print(f"  DOFs: {skel.getNumDofs()}, Bodies: {skel.getNumBodyNodes()}")

# Load skeleton meshes
from viewer.mesh_loader import MeshLoader
skel_meshes = {}
skel_xml_path = 'data/zygote_skel.xml'
import xml.etree.ElementTree as ET
tree = ET.parse(skel_xml_path)
for body in tree.iter('Body'):
    bname = body.get('name', '')
    vis = body.find('.//Visualization')
    if vis is not None:
        mesh_path = vis.get('mesh', '')
        if mesh_path:
            full_path = os.path.join('data', mesh_path)
            if os.path.exists(full_path):
                ml = MeshLoader()
                ml.load(full_path)
                skel_meshes[bname] = ml

print(f"  Loaded {len(skel_meshes)} skeleton meshes")

# Load muscles
with open('.last_loaded_muscles.json') as f:
    muscle_config = json.load(f)

print("Loading muscles...")
muscles = {}
for mc in muscle_config:
    name = mc['name']
    mobj = MuscleMesh()
    tet_path = mc.get('tet_path', f'data/tet/{name}_tet.npz')
    if os.path.exists(tet_path):
        mobj.load_tet_mesh(tet_path)
        if hasattr(mobj, 'tet_vertices') and mobj.tet_vertices is not None:
            mobj.init_soft_body(skel, skel_meshes)
            if mobj.soft_body is not None:
                muscles[name] = mobj

print(f"  Loaded {len(muscles)} muscles with soft bodies")

# Set frame 100 pose
from core.bvh_loader import BVHMotion
motion = BVHMotion()
motion.load('data/motion/walk1_subject1.bvh', skel)
pose = motion.mocap_refs[100].copy()
skel.setPositions(pose)

# Update skeleton bindings
for name, mobj in muscles.items():
    if hasattr(mobj, '_update_tet_positions_from_skeleton'):
        mobj._update_tet_positions_from_skeleton(skel)
    if hasattr(mobj, '_update_fixed_targets_from_skeleton'):
        mobj._update_fixed_targets_from_skeleton(skel_meshes, skel)

# Build solver
solver = UnifiedFEMSolver()
muscles_data = {}
for name, mobj in muscles.items():
    sb = mobj.soft_body
    muscles_data[name] = {
        'rest_positions': sb.rest_positions,
        'tetrahedra': sb.tetrahedra if hasattr(sb, 'tetrahedra') else mobj.tet_tetrahedra,
        'fixed_mask': sb.fixed_mask,
        'surface_faces': getattr(mobj, 'tet_faces', None),
    }
solver.build(muscles_data)

# Only update fixed targets
muscles_update = {}
for name, mobj in muscles.items():
    sb = mobj.soft_body
    muscles_update[name] = {
        'positions': sb.positions,
        'fixed_targets': sb.fixed_targets,
    }
solver.update_targets_and_positions(muscles_update, use_lbs_init=False)

# Reset free verts to rest
solver.positions[solver.free_indices] = solver.rest_positions[solver.free_indices]

mu, lam = 1678.0, 82215.0
vol_penalty = 100.0
empty = np.array([], dtype=np.int64)
empty3 = np.zeros((0,3))
emptyf = np.array([])

x0 = solver.positions[solver.free_indices].ravel().copy()
E0, g0 = solver._energy_gradient(x0, mu, lam, vol_penalty, empty, empty3, emptyf)
print(f'\nE0={E0:.6e}, |g|={np.linalg.norm(g0):.6e}, dim={len(g0)}')
print(f'grad NaN: {np.any(np.isnan(g0))}, Inf: {np.any(np.isinf(g0))}')
print(f'grad max component: {np.max(np.abs(g0)):.6e}')

# Count inverted tets
X = solver.positions
T = solver.tetrahedra
v0 = X[T[:, 0]]
Ds = np.stack([X[T[:, 1]] - v0, X[T[:, 2]] - v0, X[T[:, 3]] - v0], axis=-1)
Js = np.linalg.det(Ds @ solver.Dm_inv)
n_inv = int(np.sum(Js <= 0))
print(f'Inverted tets: {n_inv}/{len(T)}, J range: [{np.min(Js):.4f}, {np.max(Js):.4f}]')

# Fixed target displacement
ft_disp = np.linalg.norm(solver.fixed_targets - solver.rest_positions[solver.fixed_indices], axis=1)
print(f'Fixed target disp: max={np.max(ft_disp):.6f}m, mean={np.mean(ft_disp):.6f}m')

# Try gradient descent with various step sizes
print('\nGradient descent test:')
for alpha in [1e-8, 1e-10, 1e-12, 1e-14, 1e-16]:
    x1 = x0 - alpha * g0
    E1, _ = solver._energy_gradient(x1, mu, lam, vol_penalty, empty, empty3, emptyf)
    print(f'  alpha={alpha:.0e}: E={E1:.10e}, dE={E1-E0:.6e}')
