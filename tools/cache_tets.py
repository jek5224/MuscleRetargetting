#!/usr/bin/env python3
"""Pre-cache tetrahedralization for original meshes, one muscle at a time."""
import gc
import os
import pickle
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MESH_DIR = "Zygote_Meshes_251229/Muscle/UpLeg"
MESH_SCALE = 0.01

UPLEG_MUSCLES = [
    "Adductor_Brevis", "Adductor_Longus", "Adductor_Magnus",
    "Biceps_Femoris", "Gluteus_Maximus", "Gluteus_Medius",
    "Gluteus_Minimus", "Gracilis", "Iliacus",
    "Inferior_Gemellus", "Obturator_Externus", "Obturator_Internus",
    "Pectineus", "Piriformis", "Popliteus",
    "Quadratus_Femoris", "Rectus_Femoris", "Sartorius",
    "Semimembranosus", "Semitendinosus", "Superior_Gemellus",
    "Tensor_Fascia_Lata", "Vastus_Intermedius", "Vastus_Lateralis",
    "Vastus_Medialis",
]

# Import from bake_original_mesh
from tools.bake_original_mesh import (
    parse_muscle_xml, load_and_identify_boundaries, tetrahedralize_mesh
)

os.makedirs('tet_orig', exist_ok=True)
xml_data = parse_muscle_xml()

for mname in UPLEG_MUSCLES:
    cache_path = os.path.join('tet_orig', f"L_{mname}_tet.npz")
    if os.path.exists(cache_path):
        print(f"  {mname}: cached, skip")
        continue

    full_name = f"L_{mname}"
    obj_path = os.path.join(MESH_DIR, f"L_{mname}.obj")
    if not os.path.exists(obj_path):
        print(f"  {mname}: OBJ not found, skip")
        continue

    xml_key = f"L_{mname}"
    mxml = xml_data.get(xml_key, [])

    print(f"  {mname}: loading...", end='', flush=True)
    try:
        verts, faces, boundary_verts = load_and_identify_boundaries(obj_path, mxml)
    except Exception as e:
        print(f" load failed: {e}")
        continue

    print(f" {len(verts)} verts, tetgen...", end='', flush=True)
    try:
        tet_v, tet_e, surf_v, surf_f = tetrahedralize_mesh(verts, faces)
    except Exception as e:
        print(f" tetgen failed: {e}")
        continue

    # Map boundary vertices to tet mesh
    cap_verts = {}
    if boundary_verts:
        orig_tree = cKDTree(tet_v)
        for orig_vi, end_type in boundary_verts.items():
            if orig_vi < len(verts):
                d, tet_vi = orig_tree.query(verts[orig_vi])
                if d < 0.001:
                    cap_verts[int(tet_vi)] = end_type

    with open(cache_path, 'wb') as f:
        pickle.dump({
            'orig_vertices': verts,
            'tet_vertices': tet_v,
            'tet_elements': tet_e,
            'cap_vertices': cap_verts,
            'boundary_vertices': boundary_verts,
        }, f)

    print(f" → {len(tet_v)} tet verts, {len(tet_e)} tets, {len(cap_verts)} caps. Saved.")
    del verts, faces, boundary_verts, tet_v, tet_e, surf_v, surf_f, cap_verts
    gc.collect()

print("Done!")
