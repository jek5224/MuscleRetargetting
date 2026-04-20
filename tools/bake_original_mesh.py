#!/usr/bin/env python3
"""ARAP bake using original Zygote OBJ meshes (fine resolution).

Replaces contour-based tet meshes (480-857 verts) with original OBJs (2K-18K verts)
for better collision handling. Finer edges can't penetrate bones.

Pipeline per muscle:
  1. Load OBJ → close boundaries (CDT cap) → pymeshfix → TetGen
  2. Identify fixed vertices (cap) → assign to bones from XML
  3. Compute skinning weights

Per frame:
  1. Set skeleton pose from BVH
  2. Update fixed targets from bone transforms
  3. LBS initial guess + ARAP solve
  4. Save chunk progressively

Usage (on A6000 server):
    python tools/bake_original_mesh.py --bvh data/motion/walk.bvh --sides L --end-frame 5
"""
import argparse
import gc
import json
import os
import pickle
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from types import SimpleNamespace

import numpy as np
import trimesh
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH
from viewer.arap_backends import check_taichi_available, check_gpu_available, get_backend

SKEL_XML = "data/zygote_skel.xml"
MUSCLE_XML = "data/zygote_muscle.xml"
MESH_DIR = "Zygote_Meshes_251229/Muscle/UpLeg"
SKEL_MESH_DIR = "Zygote_Meshes_251229/Skeleton"
MESH_SCALE = 0.01  # cm → m
FLUSH_INTERVAL = 20

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

SKELETON_NAME_MAP = {
    "L_Os_Coxae": "R_Os_Coxae", "L_Femur": "R_Femur",
    "L_Tibia_Fibula": "R_Tibia_Fibula", "L_Patella": "R_Patella",
}


# ---------------------------------------------------------------------------
# Step 1: XML Parsing
# ---------------------------------------------------------------------------
def parse_muscle_xml(xml_path=MUSCLE_XML):
    """Parse zygote_muscle.xml for per-muscle origin/insertion bone info.
    Returns: {muscle_name: [(origin_bone, insertion_bone, origin_pts, insertion_pts), ...]}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result = {}
    for unit in root.findall('.//Unit'):
        name = unit.get('name', '')
        fibers = unit.findall('Fiber')
        if not fibers:
            continue
        # Group fibers by (origin_body, insertion_body)
        groups = defaultdict(lambda: ([], []))
        for fiber in fibers:
            wps = fiber.findall('Waypoint')
            if len(wps) < 2:
                continue
            origin_body = wps[0].get('body')
            insertion_body = wps[-1].get('body')
            origin_pos = np.array([float(x) for x in wps[0].get('p').split()])
            insertion_pos = np.array([float(x) for x in wps[-1].get('p').split()])
            key = (origin_body, insertion_body)
            groups[key][0].append(origin_pos)
            groups[key][1].append(insertion_pos)
        streams = []
        for (ob, ib), (o_pts, i_pts) in groups.items():
            streams.append((ob, ib, np.array(o_pts), np.array(i_pts)))
        result[name] = streams
    return result


# ---------------------------------------------------------------------------
# Step 2: Load OBJ and close boundaries
# ---------------------------------------------------------------------------
def find_boundary_loops(vertices, faces):
    """Find boundary loops using face-consistent edge traversal."""
    edge_count = defaultdict(list)
    for fi, face in enumerate(faces):
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge = (min(v0, v1), max(v0, v1))
            edge_count[edge].append(fi)

    open_edges = [e for e, fl in edge_count.items() if len(fl) == 1]
    if not open_edges:
        return []

    open_edge_set = set(open_edges)
    edge_to_face = defaultdict(list)
    for fi, face in enumerate(faces):
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            e = (min(a, b), max(a, b))
            if e in open_edge_set:
                edge_to_face[(a, b)].append(fi)
                edge_to_face[(b, a)].append(fi)

    vertex_open_neighbors = defaultdict(list)
    for edge in open_edges:
        vertex_open_neighbors[edge[0]].append(edge[1])
        vertex_open_neighbors[edge[1]].append(edge[0])

    visited_edges = set()
    loops = []
    for start_edge in open_edges:
        if start_edge in visited_edges:
            continue
        loop = []
        prev_v, curr_v = start_edge
        loop.append(prev_v)
        visited_edges.add((min(prev_v, curr_v), max(prev_v, curr_v)))
        loop_closed = False

        for _ in range(len(open_edges) + 1):
            loop.append(curr_v)
            incoming_faces = set(edge_to_face.get((prev_v, curr_v), []))
            candidates = []
            for nbr in vertex_open_neighbors[curr_v]:
                if nbr == prev_v:
                    continue
                e_key = (min(curr_v, nbr), max(curr_v, nbr))
                if e_key in visited_edges:
                    continue
                nbr_faces = set(edge_to_face.get((curr_v, nbr), []))
                shared = incoming_faces & nbr_faces
                candidates.append((nbr, e_key, len(shared) > 0))

            next_v = None
            next_key = None
            for nbr, e_key, face_shared in candidates:
                if face_shared:
                    next_v, next_key = nbr, e_key
                    break
            if next_v is None and candidates:
                next_v, next_key = candidates[0][0], candidates[0][1]

            if next_v is None:
                start_v = loop[0]
                e_close = (min(curr_v, start_v), max(curr_v, start_v))
                if e_close in open_edge_set and e_close not in visited_edges:
                    visited_edges.add(e_close)
                    loop_closed = True
                break

            visited_edges.add(next_key)
            prev_v = curr_v
            curr_v = next_v
            if curr_v == loop[0]:
                loop_closed = True
                break

        if loop_closed and len(loop) > 1 and loop[-1] == loop[0]:
            loop = loop[:-1]
        if len(loop) >= 3 and loop_closed:
            loops.append(loop)
    return loops


def cap_boundary_loop(loop_indices, vertices):
    """Cap a boundary loop using CDT (constrained Delaunay triangulation)."""
    import triangle as tr
    n = len(loop_indices)
    if n < 3:
        return [], None
    if n == 3:
        return [[loop_indices[0], loop_indices[1], loop_indices[2]]], None

    pts_3d = np.array([vertices[vi] for vi in loop_indices])
    centroid = pts_3d.mean(axis=0)
    centered = pts_3d - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)

    # Check for self-intersection
    def segments_cross(pts_2d, n):
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = pts_2d[i], pts_2d[(i + 1) % n]
                c, d = pts_2d[j], pts_2d[(j + 1) % n]
                d1 = (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
                d2 = (d[0] - a[0]) * (b[1] - a[1]) - (d[1] - a[1]) * (b[0] - a[0])
                d3 = (a[0] - c[0]) * (d[1] - c[1]) - (a[1] - c[1]) * (d[0] - c[0])
                d4 = (b[0] - c[0]) * (d[1] - c[1]) - (b[1] - c[1]) * (d[0] - c[0])
                if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                   ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                    return True
        return False

    projections = [
        ('best-fit', centered @ Vt[:2].T),
        ('XY', pts_3d[:, :2]),
        ('XZ', pts_3d[:, [0, 2]]),
        ('YZ', pts_3d[:, 1:3]),
    ]
    segments = np.array([[i, (i + 1) % n] for i in range(n)], dtype=np.int32)

    for proj_name, pts_2d in projections:
        if segments_cross(pts_2d, n):
            continue
        try:
            result = tr.triangulate({'vertices': pts_2d, 'segments': segments}, 'p')
            faces = []
            for tri_idx in result['triangles']:
                faces.append([loop_indices[tri_idx[0]], loop_indices[tri_idx[1]], loop_indices[tri_idx[2]]])
            if len(faces) >= n - 2:
                return faces, None
        except:
            continue

    # Fan fallback
    center_idx = len(vertices)  # will be appended
    faces = []
    for i in range(n):
        faces.append([loop_indices[i], loop_indices[(i + 1) % n], center_idx])
    return faces, centroid


def load_and_identify_boundaries(obj_path, muscle_xml_data):
    """Load OBJ, identify boundary vertices as origin/insertion.
    pymeshfix will close the mesh during tetrahedralization."""
    mesh = trimesh.load(obj_path, process=False)
    vertices = np.array(mesh.vertices, dtype=np.float64) * MESH_SCALE  # cm → m
    faces = np.array(mesh.faces, dtype=np.int32)

    loops = find_boundary_loops(vertices, faces)
    if not loops:
        return vertices, faces, {}

    # Assign loops to origin/insertion using XML waypoints
    origin_pts = np.vstack([s[2] for s in muscle_xml_data]) if muscle_xml_data else np.zeros((1, 3))
    insertion_pts = np.vstack([s[3] for s in muscle_xml_data]) if muscle_xml_data else np.zeros((1, 3))
    origin_mean = origin_pts.mean(axis=0)
    insertion_mean = insertion_pts.mean(axis=0)

    boundary_vertices = {}  # vi -> 'origin' or 'insertion'
    # Per-vertex classification: each boundary vertex assigned to nearest
    # XML waypoint (origin or insertion). More robust than per-loop centroid
    # which fails when loops are in unexpected locations.
    all_origin = origin_pts  # (N, 3) from XML
    all_insertion = insertion_pts  # (M, 3) from XML
    origin_tree = cKDTree(all_origin) if len(all_origin) > 0 else None
    insertion_tree = cKDTree(all_insertion) if len(all_insertion) > 0 else None

    for loop in loops:
        if len(loop) < 3:
            continue
        for vi in loop:
            pos = vertices[vi]
            d_o = origin_tree.query(pos)[0] if origin_tree else float('inf')
            d_i = insertion_tree.query(pos)[0] if insertion_tree else float('inf')
            boundary_vertices[vi] = 'origin' if d_o < d_i else 'insertion'

    return vertices, faces, boundary_vertices


# ---------------------------------------------------------------------------
# Step 3: Tetrahedralize
# ---------------------------------------------------------------------------
def tetrahedralize_mesh(vertices, faces):
    """pymeshfix repair + TetGen, with Delaunay+inside fallback."""
    import pymeshfix

    fixer = pymeshfix.MeshFix(vertices, faces)
    try:
        fixer.repair(verbose=False)
    except TypeError:
        fixer.repair()
    try:
        v_fixed, f_fixed = fixer.v, fixer.f
    except AttributeError:
        v_fixed, f_fixed = fixer._return_arrays()

    # Try TetGen first
    tet_verts, tet_elems = None, None
    try:
        import tetgen
        tg = tetgen.TetGen(v_fixed, f_fixed)
        tg.tetrahedralize(order=1, mindihedral=0, quality=False)
        tet_verts = np.array(tg.node, dtype=np.float64)
        tet_elems = np.array(tg.elem, dtype=np.int32)
    except Exception:
        pass

    # Fallback: Delaunay + inside filter
    if tet_verts is None:
        from scipy.spatial import Delaunay
        mesh_check = trimesh.Trimesh(vertices=v_fixed, faces=f_fixed, process=False)
        dl = Delaunay(v_fixed)
        centroids = v_fixed[dl.simplices].mean(axis=1)
        try:
            inside = mesh_check.contains(centroids)
        except:
            inside = np.ones(len(centroids), dtype=bool)
        tet_elems = dl.simplices[inside]
        tet_verts = v_fixed

    # Fix orientation
    v = tet_verts[tet_elems]
    vols = np.einsum('ij,ij->i', v[:, 1] - v[:, 0], np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]))
    neg = vols < 0
    tet_elems[neg, 1], tet_elems[neg, 2] = tet_elems[neg, 2].copy(), tet_elems[neg, 1].copy()

    return tet_verts, tet_elems, v_fixed, f_fixed


# ---------------------------------------------------------------------------
# Step 4: Fixed vertices and bone assignment
# ---------------------------------------------------------------------------
def identify_fixed_and_assign_bones(tet_verts, cap_vertices_orig, orig_verts,
                                     muscle_xml_data, skel, bone_name_map=None):
    """Map cap vertices to tet mesh, assign to bones."""
    # Match original cap vertices to TetGen vertices
    tree = cKDTree(tet_verts)
    fixed_verts = {}  # tet_vi -> (bone_name, 'origin'/'insertion')

    for orig_vi, end_type in cap_vertices_orig.items():
        if orig_vi < len(orig_verts):
            pos = orig_verts[orig_vi]
        else:
            continue
        dist, tet_vi = tree.query(pos)
        if dist < 0.005:  # 5mm tolerance (TetGen can move vertices)
            # Determine bone from XML
            if end_type == 'origin':
                bone = muscle_xml_data[0][0] if muscle_xml_data else 'L_Femur0'
            else:
                bone = muscle_xml_data[0][1] if muscle_xml_data else 'L_Femur0'
            if bone_name_map:
                bone = bone_name_map.get(bone, bone)
            fixed_verts[int(tet_vi)] = (bone, end_type)

    # Compute local positions at rest
    local_anchors = {}
    initial_transforms = {}
    for vi, (bone_name, end_type) in fixed_verts.items():
        body_node = skel.getBodyNode(bone_name)
        if body_node is None:
            continue
        wt = body_node.getWorldTransform()
        R = wt.rotation()
        t = wt.translation()
        if bone_name not in initial_transforms:
            initial_transforms[bone_name] = (R.copy(), t.copy())
        local_pos = R.T @ (tet_verts[vi] - t)
        local_anchors[vi] = (bone_name, local_pos.copy())

    return fixed_verts, local_anchors, initial_transforms


# ---------------------------------------------------------------------------
# Step 5: Skinning weights
# ---------------------------------------------------------------------------
def compute_skinning_weights(tet_verts, local_anchors, initial_transforms):
    """Distance-based skinning weights."""
    n_verts = len(tet_verts)
    bones = list(initial_transforms.keys())
    n_bones = len(bones)
    if n_bones == 0:
        return np.ones((n_verts, 1)) / 1, ['unknown']

    weights = np.zeros((n_verts, n_bones))

    # Collect anchor positions per bone
    bone_anchors = {b: [] for b in bones}
    for vi, (bone_name, local_pos) in local_anchors.items():
        if bone_name in bone_anchors:
            bone_anchors[bone_name].append(tet_verts[vi])

    bbox = np.ptp(tet_verts, axis=0)
    falloff = np.max(bbox) * 0.5

    for bi, bone in enumerate(bones):
        anchors = np.array(bone_anchors.get(bone, []))
        if len(anchors) == 0:
            continue
        dists = np.linalg.norm(tet_verts[:, None, :] - anchors[None, :, :], axis=2).min(axis=1)
        nd = dists / falloff
        weights[:, bi] = np.where(nd < 1.0, (1.0 - nd ** 2) ** 2, 0.0)

    # Normalize
    sums = weights.sum(axis=1, keepdims=True)
    sums[sums < 1e-10] = 1.0
    weights /= sums

    # Override: fixed vertices 100% to their bone
    for vi, (bone_name, _) in local_anchors.items():
        if bone_name in bones:
            bi = bones.index(bone_name)
            weights[vi, :] = 0.0
            weights[vi, bi] = 1.0

    return weights, bones


# ---------------------------------------------------------------------------
# Step 6: Build edges from tets
# ---------------------------------------------------------------------------
def build_edges(tetrahedra):
    """Extract unique edges from tetrahedra."""
    edge_set = set()
    for tet in tetrahedra:
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = int(tet[i]), int(tet[j])
                edge_set.add((min(a, b), max(a, b)))
    edges = np.array(list(edge_set), dtype=np.int32)
    return edges


# ---------------------------------------------------------------------------
# Step 7: IPC collision resolution
# ---------------------------------------------------------------------------
SKELETON_BONES = {
    'L': ['L_Os_Coxae', 'L_Femur', 'L_Tibia_Fibula', 'L_Patella'],
    'R': ['R_Os_Coxae', 'R_Femur', 'R_Tibia_Fibula', 'R_Patella'],
}
BONE_NAME_TO_BODY = {
    'L_Os_Coxae': 'L_Os_Coxae0', 'L_Femur': 'L_Femur0',
    'L_Tibia_Fibula': 'L_Tibia_Fibula0', 'L_Patella': 'L_Patella0',
    'R_Os_Coxae': 'R_Os_Coxae0', 'R_Femur': 'R_Femur0',
    'R_Tibia_Fibula': 'R_Tibia_Fibula0', 'R_Patella': 'R_Patella0',
    'Saccrum_Coccyx': 'Saccrum_Coccyx0',
}


def get_bone_world_verts(skel, bone_name, bone_verts_cm, bone_rest_transforms):
    """Transform bone mesh vertices to world coords (meters) for current skeleton pose."""
    body_name = BONE_NAME_TO_BODY.get(bone_name, bone_name + '0')
    body_node = skel.getBodyNode(body_name)
    if body_node is None:
        return None
    if body_name not in bone_rest_transforms:
        return None
    R_rest, t_rest = bone_rest_transforms[body_name]
    R_cur = body_node.getWorldTransform().rotation()
    t_cur = body_node.getWorldTransform().translation()
    v_m = bone_verts_cm * 0.01  # cm → m
    local = (R_rest.T @ (v_m - t_rest).T).T
    return (R_cur @ local.T).T + t_cur


def run_ipc_frame(muscles, frame_positions, skel, bone_meshes, bone_rest_transforms,
                  side, d_hat_mm=0.5, elastic_kpa=10.0):
    """Run IPC collision resolution for one frame. Returns updated positions."""
    from uipc import view
    import uipc.builtin as builtin
    from uipc.core import Engine, World, Scene
    from uipc.geometry import tetmesh, label_surface, label_triangle_orient
    from uipc.constitution import StableNeoHookean, ElasticModuli
    from uipc.unit import kPa, MPa

    SCALE = 1000.0

    # No pre-push for original mesh — fine resolution handles most penetrations.
    # Pre-push on dense meshes creates inverted tets → NaN in IPC.

    # Build IPC scene
    engine = Engine('cuda')
    world = World(engine)
    config = Scene.default_config()
    config['dt'] = 0.01
    config['gravity'] = [[0.0], [0.0], [0.0]]
    config['contact'] = {'enable': True, 'friction': {'enable': False}, 'd_hat': d_hat_mm}
    config['sanity_check'] = {'enable': False}
    config['newton'] = {'max_iter': 512}
    scene = Scene(config)
    scene.contact_tabular().default_model(0.0, 1.0 * MPa)

    snh = StableNeoHookean()
    moduli = ElasticModuli.youngs_poisson(elastic_kpa * kPa, 0.45)

    geo_slots = []
    valid_muscles = []
    for m in muscles:
        arap_pos = frame_positions[m['name']] * SCALE  # m → mm
        tets = m['tetrahedra'].copy()
        # Fix orientation for current positions
        v = arap_pos[tets]
        vols = np.einsum('ij,ij->i', v[:, 1] - v[:, 0], np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]))
        neg = vols < 0
        tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy()
        # Remove degenerate tets
        v2 = arap_pos[tets]
        vols2 = np.abs(np.einsum('ij,ij->i', v2[:, 1] - v2[:, 0], np.cross(v2[:, 2] - v2[:, 0], v2[:, 3] - v2[:, 0])))
        tets = tets[vols2 > 0.0]

        # Check tet quality — skip muscles with degenerate tets
        v_check = arap_pos[tets]
        vols_check = np.einsum('ij,ij->i', v_check[:, 1] - v_check[:, 0],
                               np.cross(v_check[:, 2] - v_check[:, 0], v_check[:, 3] - v_check[:, 0]))
        if np.any(vols_check <= 0):
            n_bad = np.sum(vols_check <= 0)
            # Try removing bad tets
            tets = tets[vols_check > 0]
            if len(tets) == 0:
                continue

        try:
            mesh = tetmesh(arap_pos, tets)
            label_surface(mesh)
            label_triangle_orient(mesh)
            snh.apply_to(mesh, moduli, mass_density=1060.0)
            is_fixed = view(mesh.vertices().find(builtin.is_fixed))
            for vi in m['fixed_vertices']:
                if vi < len(is_fixed):
                    is_fixed[vi] = 1
        except Exception as e:
            continue

        obj = scene.objects().create(m['name'])
        geo_slot, _ = obj.geometries().create(mesh)
        geo_slots.append(geo_slot)
        valid_muscles.append(m)

    # Solve
    result = {}
    if len(valid_muscles) == 0:
        return {m['name']: frame_positions[m['name']].astype(np.float32) for m in muscles}

    try:
        world.init(scene)
        world.advance()
        world.retrieve()
        for i, m in enumerate(valid_muscles):
            geo = geo_slots[i].geometry()
            pos = np.array(geo.positions().view()).reshape(-1, 3) / SCALE  # mm → m
            result[m['name']] = pos.astype(np.float32)
    except Exception as e:
        print(f"    IPC failed: {e}", flush=True)

    # Fill missing muscles with ARAP positions
    for m in muscles:
        if m['name'] not in result:
            result[m['name']] = frame_positions[m['name']].astype(np.float32)

    del world, engine
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ARAP bake using original OBJ meshes")
    parser.add_argument('--bvh', required=True, help='BVH motion file')
    parser.add_argument('--sides', default='L', help='L, R, or LR')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int, default=None)
    parser.add_argument('--settle-iters', type=int, default=150)
    parser.add_argument('--backend', default='auto', choices=['auto', 'taichi', 'gpu', 'cpu'])
    parser.add_argument('--output-dir', default='data/motion_cache')
    parser.add_argument('--ipc', action='store_true', help='Run IPC collision after ARAP (experimental, may fail on dense meshes)')
    parser.add_argument('--d-hat', type=float, default=0.5, help='IPC barrier distance (mm)')
    parser.add_argument('--elastic', type=float, default=10.0, help='IPC elastic modulus (kPa)')
    args = parser.parse_args()

    t_start = time.time()

    # ── Load skeleton ─────────────────────────────────────────────────────
    print("[1] Loading skeleton...")
    skel_info, root_name, bvh_info, _, mesh_info, _ = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)
    skel.setPositions(np.zeros(skel.getNumDofs()))
    print(f"    DOFs: {skel.getNumDofs()}, Bodies: {skel.getNumBodyNodes()}")

    # ── Load skeleton meshes ──────────────────────────────────────────────
    print("[2] Loading skeleton meshes...")
    skeleton_meshes = {}
    for f in sorted(os.listdir(SKEL_MESH_DIR)):
        if f.endswith('.obj'):
            name = f.replace('.obj', '')
            m = trimesh.load(os.path.join(SKEL_MESH_DIR, f), process=False)
            skeleton_meshes[name] = m

    # ── Parse muscle XML ──────────────────────────────────────────────────
    print("[3] Parsing muscle XML...")
    xml_data = parse_muscle_xml()
    print(f"    {len(xml_data)} muscles found")

    # ── Load bone meshes for collision ──────────────────────────────────
    bone_meshes = {}
    bone_rest_transforms = {}
    if True:  # always load bones for ARAP collision
        print("[3b] Loading bone meshes...")
        for bone_name in ['L_Os_Coxae', 'L_Femur', 'L_Tibia_Fibula', 'L_Patella',
                          'R_Os_Coxae', 'R_Femur', 'R_Tibia_Fibula', 'R_Patella',
                          'Saccrum_Coccyx']:
            path = os.path.join(SKEL_MESH_DIR, f'{bone_name}.obj')
            if os.path.exists(path):
                m = trimesh.load(path, process=False)
                bone_meshes[bone_name] = {
                    'vertices': np.array(m.vertices, dtype=np.float64),
                    'faces': np.array(m.faces, dtype=np.int32),
                }
        # Compute rest transforms
        skel.setPositions(np.zeros(skel.getNumDofs()))
        for bone_name in bone_meshes:
            body_name = BONE_NAME_TO_BODY.get(bone_name, bone_name + '0')
            bn = skel.getBodyNode(body_name)
            if bn is not None:
                wt = bn.getWorldTransform()
                bone_rest_transforms[body_name] = (wt.rotation().copy(), wt.translation().copy())
        print(f"    {len(bone_meshes)} bones, {len(bone_rest_transforms)} transforms")

    # ── Load BVH ──────────────────────────────────────────────────────────
    print("[4] Loading BVH...")
    from viewer.zygote_mesh_ui import _detect_bvh_tframe
    t_frame = _detect_bvh_tframe(args.bvh)
    motion_bvh = MyBVH(args.bvh, bvh_info, skel, T_frame=t_frame)
    n_frames = motion_bvh.mocap_refs.shape[0]
    end_frame = args.end_frame if args.end_frame is not None else n_frames - 1
    end_frame = min(end_frame, n_frames - 1)
    total_frames = end_frame - args.start_frame + 1
    print(f"    Frames: {n_frames}, baking {args.start_frame}-{end_frame} ({total_frames})")

    # ── Process each side ─────────────────────────────────────────────────
    for side in args.sides:
        print(f"\n{'='*60}")
        print(f"  Processing {side} side")
        print(f"{'='*60}")

        # Reset skeleton to rest
        skel.setPositions(np.zeros(skel.getNumDofs()))

        # ── Load and prepare muscles ──────────────────────────────────────
        print("[5] Loading and tetrahedralizing muscles...")
        muscles = []
        total_verts = 0
        total_tets = 0

        for mname in UPLEG_MUSCLES:
            full_name = f"{side}_{mname}"
            obj_path = os.path.join(MESH_DIR, f"L_{mname}.obj")  # always load L, mirror for R
            if not os.path.exists(obj_path):
                print(f"    {full_name}: OBJ not found, skipping")
                continue

            xml_key = f"L_{mname}"
            mxml = xml_data.get(xml_key, [])

            # Load from cache or tetrahedralize
            cache_path = os.path.join('tet_orig', f"L_{mname}_tet.npz")
            os.makedirs('tet_orig', exist_ok=True)

            if os.path.exists(cache_path) and side == 'L':
                # Load cached tet
                with open(cache_path, 'rb') as _f:
                    _cd = pickle.load(_f)
                verts = _cd['orig_vertices']
                tet_v = _cd['tet_vertices']
                tet_e = _cd['tet_elements']
                cap_verts = _cd['cap_vertices']  # {vi: 'origin'/'insertion'}
                boundary_verts = _cd['boundary_vertices']
            else:
                # Load and identify boundaries
                try:
                    verts, faces, boundary_verts = load_and_identify_boundaries(obj_path, mxml)
                except Exception as e:
                    print(f"    {full_name}: load failed: {e}")
                    continue

                # Tetrahedralize (pymeshfix closes boundaries)
                try:
                    tet_v, tet_e, surf_v, surf_f = tetrahedralize_mesh(verts, faces)
                except Exception as e:
                    print(f"    {full_name}: tetgen failed: {e}")
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

                # Save cache (L side only, R is mirrored)
                with open(cache_path, 'wb') as _f:
                    pickle.dump({
                        'orig_vertices': verts,
                        'tet_vertices': tet_v,
                        'tet_elements': tet_e,
                        'cap_vertices': cap_verts,
                        'boundary_vertices': boundary_verts,
                    }, _f)

            # Mirror for R side
            if side == 'R':
                verts = verts.copy()
                verts[:, 0] *= -1
                tet_v = tet_v.copy()
                tet_v[:, 0] *= -1
                tet_e = tet_e.copy()
                tet_e[:, 1], tet_e[:, 2] = tet_e[:, 2].copy(), tet_e[:, 1].copy()

            # Mirror bone names for R
            bone_map = None
            if side == 'R':
                bone_map = {}
                for k, v_name in SKELETON_NAME_MAP.items():
                    bone_map[k + '0'] = v_name + '0'
                bone_map['Saccrum_Coccyx0'] = 'Saccrum_Coccyx0'

            # Assign bones
            fixed_verts, local_anchors, init_transforms = identify_fixed_and_assign_bones(
                tet_v, cap_verts, verts, mxml, skel, bone_map)

            # Skinning weights
            skin_w, skin_bones = compute_skinning_weights(tet_v, local_anchors, init_transforms)

            # Build edges
            edges = build_edges(tet_e)

            n_fixed = len(local_anchors)
            total_verts += len(tet_v)
            total_tets += len(tet_e)

            muscles.append({
                'name': full_name,
                'vertices': tet_v,
                'tetrahedra': tet_e,
                'edges': edges,
                'fixed_vertices': sorted(fixed_verts.keys()),
                'local_anchors': local_anchors,
                'initial_transforms': init_transforms,
                'skinning_weights': skin_w,
                'skinning_bones': skin_bones,
                'n_orig': len(verts),
                'xml_streams': mxml,  # for IPC attachment bone check
            })
            print(f"    {full_name}: {len(tet_v)} verts, {len(tet_e)} tets, {n_fixed} fixed")

        print(f"    Total: {len(muscles)} muscles, {total_verts} verts, {total_tets} tets")

        if len(muscles) == 0:
            print("    No muscles loaded, skipping side")
            continue

        # ── Build unified ARAP system ─────────────────────────────────────
        print("[6] Building ARAP system...")
        # Global indexing
        global_offset = {}
        offset = 0
        for m in muscles:
            global_offset[m['name']] = offset
            offset += len(m['vertices'])

        n_total = offset
        global_rest = np.zeros((n_total, 3))
        global_fixed_mask = np.zeros(n_total, dtype=bool)
        global_fixed_targets = {}
        all_local_anchors = {}  # global_vi -> (bone, local_pos)

        neighbors = [[] for _ in range(n_total)]
        edge_weights = {}
        rest_edge_vectors = [{} for _ in range(n_total)]

        for m in muscles:
            off = global_offset[m['name']]
            n = len(m['vertices'])
            global_rest[off:off + n] = m['vertices']

            for vi in m['fixed_vertices']:
                global_fixed_mask[off + vi] = True

            for vi, (bone, lpos) in m['local_anchors'].items():
                all_local_anchors[off + vi] = (bone, lpos)

            for e in m['edges']:
                gi, gj = off + int(e[0]), off + int(e[1])
                neighbors[gi].append(gj)
                neighbors[gj].append(gi)
                edge_weights[(gi, gj)] = 1.0
                edge_weights[(gj, gi)] = 1.0
                rest_edge_vectors[gi][gj] = global_rest[gi] - global_rest[gj]
                rest_edge_vectors[gj][gi] = global_rest[gj] - global_rest[gi]

        print(f"    Unified: {n_total} verts, {sum(len(n) for n in neighbors)//2} edges, "
              f"{np.sum(global_fixed_mask)} fixed")

        # ── Select ARAP backend ───────────────────────────────────────────
        backend_name = args.backend
        if backend_name == 'auto':
            if check_taichi_available():
                backend_name = 'taichi'
            elif check_gpu_available():
                backend_name = 'gpu'
            else:
                backend_name = 'cpu'
        print(f"    Backend: {backend_name}")
        backend = get_backend(backend_name)
        backend.build_system(n_total, neighbors, edge_weights, global_fixed_mask, regularization=1e-6)
        print(f"    System built")

        # ── Frame loop ────────────────────────────────────────────────────
        print(f"[7] Baking frames {args.start_frame}-{end_frame}...")
        bvh_stem = os.path.splitext(os.path.basename(args.bvh))[0]
        cache_dir = os.path.join(args.output_dir, bvh_stem, f"orig_{side}_UpLeg")
        os.makedirs(cache_dir, exist_ok=True)

        bake_data = {m['name']: {} for m in muscles}
        global_positions = global_rest.copy()
        bake_start = time.time()

        for frame in range(args.start_frame, end_frame + 1):
            frame_start = time.time()
            fi = frame - args.start_frame

            # Set skeleton pose
            skel.setPositions(motion_bvh.mocap_refs[frame])

            # Update fixed targets
            fixed_indices = np.where(global_fixed_mask)[0]
            fixed_targets = np.zeros((len(fixed_indices), 3))
            for idx, gi in enumerate(fixed_indices):
                if gi in all_local_anchors:
                    bone_name, local_pos = all_local_anchors[gi]
                    body_node = skel.getBodyNode(bone_name)
                    if body_node is not None:
                        wt = body_node.getWorldTransform()
                        fixed_targets[idx] = wt.rotation() @ local_pos + wt.translation()
                    else:
                        fixed_targets[idx] = global_rest[gi]
                else:
                    fixed_targets[idx] = global_rest[gi]

            # LBS initial guess
            for m in muscles:
                off = global_offset[m['name']]
                n = len(m['vertices'])
                rest = m['vertices']
                lbs = np.zeros((n, 3))
                for bi, bone in enumerate(m['skinning_bones']):
                    body_node = skel.getBodyNode(bone)
                    if body_node is None:
                        continue
                    wt = body_node.getWorldTransform()
                    R_cur = wt.rotation()
                    t_cur = wt.translation()
                    if bone in m['initial_transforms']:
                        R0, t0 = m['initial_transforms'][bone]
                    else:
                        continue
                    local = (R0.T @ (rest - t0).T).T
                    deformed = (R_cur @ local.T).T + t_cur
                    w = m['skinning_weights'][:, bi:bi + 1]
                    lbs += w * deformed
                global_positions[off:off + n] = lbs

            # First frame: more iterations
            solve_iters = args.settle_iters * 2 if fi == 0 else args.settle_iters

            # Build bone surfaces for collision projection
            import trimesh as _trimesh
            bone_surfs_frame = []  # [(trimesh, body_name, KDTree)]
            if bone_meshes:
                for bone_name, bm in bone_meshes.items():
                    body_name = BONE_NAME_TO_BODY.get(bone_name, bone_name + '0')
                    wv = get_bone_world_verts(skel, bone_name, bm['vertices'], bone_rest_transforms)
                    if wv is not None:
                        bone_surfs_frame.append((
                            _trimesh.Trimesh(vertices=wv, faces=bm['faces'], process=False),
                            body_name, cKDTree(wv)))

            # Build collision projection function (projective dynamics)
            def collision_projection(positions):
                """Project penetrating vertices to bone surface + margin.
                Uses closest_point + normal dot product instead of ray-casting contains.
                Fast enough to run every ARAP iteration.
                """
                proj_count = 0
                for bone_surf, body_name, bone_tree in bone_surfs_frame:
                    for m in muscles:
                        attach_bones = set()
                        for s in m.get('xml_streams', []):
                            attach_bones.add(s[0])
                            attach_bones.add(s[1])
                        if body_name in attach_bones:
                            continue
                        off = global_offset[m['name']]
                        n = len(m['vertices'])
                        pos = positions[off:off + n]
                        fixed_set = set(m['fixed_vertices'])

                        # KDTree pre-filter: only check vertices near bone
                        dists, _ = bone_tree.query(pos)
                        close_mask = dists < 0.030  # 30mm radius (deep penetrations can be >15mm)
                        if not np.any(close_mask):
                            continue
                        close_idx = np.where(close_mask)[0]
                        free_close = np.array([vi for vi in close_idx if vi not in fixed_set])
                        if len(free_close) == 0:
                            continue

                        # Closest point on bone surface (BVH, fast)
                        closest, _, face_ids = _trimesh.proximity.closest_point(
                            bone_surf, pos[free_close])
                        normals = bone_surf.face_normals[face_ids]
                        # Inside test: dot(vertex - closest, face_normal) < 0
                        to_vert = pos[free_close] - closest
                        dots = np.sum(to_vert * normals, axis=1)
                        inside = dots < 0

                        if not np.any(inside):
                            continue
                        pen = free_close[inside]
                        # Depth-dependent margin: deeper penetration → push further out
                        depths = -dots[inside]  # positive values in meters
                        margins = np.clip(depths + 0.005, 0.005, 0.020)  # 5-20mm
                        positions[off + pen] = closest[inside] + normals[inside] * margins[:, None]
                        proj_count += len(pen)
                return positions

            # ARAP solve with collision projection inside iteration loop
            global_positions, iterations, max_disp = backend.solve(
                global_positions, global_rest, neighbors, edge_weights, rest_edge_vectors,
                global_fixed_mask, fixed_targets,
                max_iterations=solve_iters, tolerance=1e-4, verbose=(fi % 20 == 0),
                collision_projection=collision_projection if bone_surfs_frame else None
            )

            # Capture
            for m in muscles:
                off = global_offset[m['name']]
                n = len(m['vertices'])
                bake_data[m['name']][frame] = global_positions[off:off + n].astype(np.float32)

            # Progress
            dt = time.time() - frame_start
            elapsed = time.time() - bake_start
            avg = elapsed / (fi + 1)
            eta = avg * (total_frames - fi - 1)
            print(f"  Frame {frame}/{end_frame} ({fi+1}/{total_frames}) "
                  f"{dt:.2f}s avg {avg:.2f}s/frame ETA {eta:.0f}s", flush=True)

            # Progressive save
            if (fi + 1) % FLUSH_INTERVAL == 0 or frame == end_frame:
                chunk_start = fi - (fi % FLUSH_INTERVAL)
                chunk_idx = chunk_start // FLUSH_INTERVAL
                chunk_frames = list(range(args.start_frame + chunk_start,
                                         min(args.start_frame + chunk_start + FLUSH_INTERVAL, end_frame + 1)))
                for m in muscles:
                    chunk_pos = []
                    chunk_f = []
                    for cf in chunk_frames:
                        if cf in bake_data[m['name']]:
                            chunk_pos.append(bake_data[m['name']][cf])
                            chunk_f.append(cf)
                    if chunk_pos:
                        out_path = os.path.join(cache_dir, f"{m['name']}_chunk_{chunk_idx:04d}.npz")
                        np.savez(out_path,
                                 frames=np.array(chunk_f, dtype=np.int32),
                                 positions=np.array(chunk_pos, dtype=np.float32))

                # Free flushed frames
                for cf in chunk_frames:
                    for m in muscles:
                        bake_data[m['name']].pop(cf, None)
                gc.collect()
                print(f"  Flushed chunk {chunk_idx}", flush=True)

        total_dt = time.time() - t_start
        print(f"\nDone in {total_dt:.1f}s. {len(muscles)} muscles, {total_frames} frames.")
        print(f"Output: {cache_dir}/")


if __name__ == '__main__':
    main()
