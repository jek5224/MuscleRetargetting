#!/usr/bin/env python3
"""Cache tet for a single muscle. Run as separate process to avoid OOM accumulation."""
import gc
import os
import pickle
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

mname = sys.argv[1]
cache_path = os.path.join('tet_orig', f"L_{mname}_tet.npz")
if os.path.exists(cache_path):
    print(f"  {mname}: already cached")
    sys.exit(0)

MESH_DIR = "Zygote_Meshes_251229/Muscle/UpLeg"
obj_path = os.path.join(MESH_DIR, f"L_{mname}.obj")
if not os.path.exists(obj_path):
    print(f"  {mname}: OBJ not found")
    sys.exit(1)

from tools.bake_original_mesh import parse_muscle_xml, load_and_identify_boundaries, tetrahedralize_mesh
from scipy.spatial import cKDTree

xml_data = parse_muscle_xml()
mxml = xml_data.get(f"L_{mname}", [])

print(f"  {mname}: loading...", end='', flush=True)
verts, faces, boundary_verts = load_and_identify_boundaries(obj_path, mxml)
print(f" {len(verts)} verts, tetgen...", end='', flush=True)
tet_v, tet_e, surf_v, surf_f = tetrahedralize_mesh(verts, faces)

cap_verts = {}
if boundary_verts:
    orig_tree = cKDTree(tet_v)
    for orig_vi, end_type in boundary_verts.items():
        if orig_vi < len(verts):
            d, tet_vi = orig_tree.query(verts[orig_vi])
            if d < 0.001:
                cap_verts[int(tet_vi)] = end_type

os.makedirs('tet_orig', exist_ok=True)
with open(cache_path, 'wb') as f:
    pickle.dump({
        'orig_vertices': verts,
        'tet_vertices': tet_v,
        'tet_elements': tet_e,
        'cap_vertices': cap_verts,
        'boundary_vertices': boundary_verts,
    }, f)

print(f" → {len(tet_v)} tet verts, {len(tet_e)} tets, {len(cap_verts)} caps. Saved.")
