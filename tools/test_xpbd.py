#!/usr/bin/env python
"""Quick test: XPBD Neo-Hookean solver on the full muscle system."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['TAICHI_LOG_LEVEL'] = 'warn'

import numpy as np
import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64, debug=False)

from viewer.fem_sim import UnifiedFEMSolver, _greedy_graph_color
from core.dartHelper import saveSkeletonInfo, buildFromInfo
from viewer.muscle_mesh import MuscleMesh
from viewer.mesh_loader import MeshLoader
import json, time

# --- Load skeleton ---
print("Loading skeleton...")
skel_info, root_name, bvh_info, _pd, mesh_info, _smpl = saveSkeletonInfo('data/zygote_skel.xml')
skel = buildFromInfo(skel_info, root_name)
print(f"  DOFs: {skel.getNumDofs()}, Bodies: {skel.getNumBodyNodes()}")

# Load skeleton meshes
import xml.etree.ElementTree as ET
skel_meshes = {}
tree = ET.parse('data/zygote_skel.xml')
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

# --- Load muscles ---
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

# --- Set frame 100 pose ---
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

# --- Build solver ---
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

# Update targets
muscles_update = {}
for name, mobj in muscles.items():
    sb = mobj.soft_body
    muscles_update[name] = {
        'positions': sb.positions,
        'fixed_targets': sb.fixed_targets,
    }
solver.update_targets_and_positions(muscles_update)

# --- Material params ---
E, nu = 500.0, 0.4
mu = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
vol_penalty = 5000.0

print(f"\nMaterial: E={E}, nu={nu}, mu={mu:.1f}, lam={lam:.1f}, vol_penalty={vol_penalty}")

# --- Test: varying iteration counts ---
for n_iters in [20, 50, 100, 200]:
    # Reset to rest positions for free verts
    solver.positions[solver.free_indices] = solver.rest_positions[solver.free_indices]
    # Keep fixed at targets
    solver.positions[solver.fixed_indices] = solver.fixed_targets

    t0 = time.time()
    iters, residual = solver.solve(
        mu=mu, lam=lam, vol_penalty=vol_penalty,
        max_lbfgs_iters=n_iters, verbose=True
    )
    dt = time.time() - t0
    print(f"  {n_iters} iters: {dt:.3f}s ({dt/n_iters*1000:.1f}ms/iter), ||dx||={residual:.4e}\n")
