#!/usr/bin/env python3
"""Muscle bake via PolyFEM: quasi-static Neo-Hookean + IPC contact.

Uses polyfem CLI (PolyFEM_bin) for proper Newton + CCD solve.
Each frame: export meshes → write JSON config → run polyfem → read results.

Architecture:
  Sim mesh:     original tet mesh (MSH format)
  Obstacles:    bone surface meshes (OBJ, rigid)
  Material:     Neo-Hookean (high Poisson for volume preservation)
  Contact:      IPC barrier + CCD (built into polyfem)
  Solver:       Newton with backtracking line search
  BCs:          Dirichlet on origin/insertion vertices

Usage:
    python tools/bake_polyfem.py --bvh data/motion/walk.bvh --sides L \\
        --start-frame 66 --end-frame 66 --polyfem-bin ~/polyfem/build/PolyFEM_bin
"""
import argparse
import gc
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH
from tools.bake_original_mesh import (
    parse_muscle_xml, identify_fixed_and_assign_bones,
    compute_skinning_weights, build_edges,
    UPLEG_MUSCLES, SKELETON_NAME_MAP, SKEL_XML, MUSCLE_XML, MESH_DIR,
    SKEL_MESH_DIR, MESH_SCALE, FLUSH_INTERVAL, BONE_NAME_TO_BODY,
    get_bone_world_verts,
)

COLLISION_BONES = {
    'L': ['L_Os_Coxae', 'L_Femur', 'L_Tibia_Fibula', 'L_Patella'],
    'R': ['R_Os_Coxae', 'R_Femur', 'R_Tibia_Fibula', 'R_Patella'],
}


def write_msh(path, vertices, tetrahedra, fixed_vertex_ids=None):
    """Write Gmsh MSH 2.2 format (ASCII) with physical groups for BCs."""
    with open(path, 'w') as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        # Nodes
        n = len(vertices)
        f.write(f"$Nodes\n{n}\n")
        for i, v in enumerate(vertices):
            f.write(f"{i + 1} {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n")
        f.write("$EndNodes\n")

        # Elements: tets (type 4 = linear tet)
        n_tets = len(tetrahedra)
        f.write(f"$Elements\n{n_tets}\n")
        for i, t in enumerate(tetrahedra):
            # tag format: num_tags phys_tag elem_tag
            f.write(f"{i + 1} 4 2 1 1 {t[0]+1} {t[1]+1} {t[2]+1} {t[3]+1}\n")
        f.write("$EndElements\n")


def write_obj(path, vertices, faces):
    """Write OBJ surface mesh."""
    with open(path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def read_vtu_positions(vtu_path):
    """Read vertex positions from VTU output."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(vtu_path)
    root = tree.getroot()

    # Find Points DataArray
    for da in root.iter('DataArray'):
        if da.get('Name') == 'Points' or (
                da.getparent() is not None and da.getparent().tag == 'Points'):
            text = da.text.strip()
            values = np.array([float(x) for x in text.split()])
            n_verts = len(values) // 3
            return values.reshape(n_verts, 3)

    # Fallback: try to find any 3-component point array
    for piece in root.iter('Piece'):
        n_points = int(piece.get('NumberOfPoints', 0))
        if n_points == 0:
            continue
        points = piece.find('.//Points')
        if points is not None:
            da = points.find('DataArray')
            if da is not None:
                text = da.text.strip()
                values = np.array([float(x) for x in text.split()])
                return values.reshape(n_points, 3)

    raise RuntimeError(f"Could not read positions from {vtu_path}")


def build_polyfem_config(muscle_msh, bone_objs, fixed_selections,
                         dirichlet_bcs, youngs, poisson, dhat,
                         output_dir):
    """Build PolyFEM JSON config for one frame.

    Args:
        muscle_msh: path to unified muscle tet mesh
        bone_objs: list of (path, name) for bone OBJ obstacles
        fixed_selections: list of {"id": N, "box": [[x0,y0,z0],[x1,y1,z1]]}
        dirichlet_bcs: list of {"id": N, "value": [x,y,z]}
        youngs: Young's modulus
        poisson: Poisson ratio
        dhat: barrier distance
        output_dir: where to write results
    """
    geometry = [{
        "mesh": muscle_msh,
        "volume_selection": 1,
        "point_selection": fixed_selections,
    }]

    # Add bone obstacles
    for bone_path, bone_name in bone_objs:
        geometry.append({
            "mesh": bone_path,
            "is_obstacle": True,
        })

    config = {
        "geometry": geometry,
        "materials": {
            "type": "NeoHookean",
            "E": youngs,
            "nu": poisson,
            "rho": 1000,
        },
        "time": {
            "quasistatic": True,
            "tend": 1.0,
            "dt": 1.0,  # single step: rest → target in one go
        },
        "contact": {
            "enabled": True,
            "dhat": dhat,
            "friction_coefficient": 0.0,
        },
        "boundary_conditions": {
            "rhs": [0, 0, 0],
            "dirichlet_boundary": dirichlet_bcs,
        },
        "solver": {
            "linear": {
                "solver": ["Eigen::CholmodSupernodalLLT", "Eigen::SimplicialLDLT"],
            },
            "nonlinear": {
                "solver": "Newton",
                "line_search": {"method": "Backtracking"},
                "grad_norm": 1e-4,
                "max_iterations": 100,
            },
            "contact": {
                "barrier_stiffness": "adaptive",
            },
        },
        "output": {
            "directory": output_dir,
            "json": "sim.json",
            "paraview": {
                "file_name": "result.vtu",
                "volume": True,
                "surface": False,
            },
        },
    }
    return config


def main():
    parser = argparse.ArgumentParser(description="Muscle bake via PolyFEM")
    parser.add_argument('--bvh', required=True)
    parser.add_argument('--sides', default='L')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int, default=None)
    parser.add_argument('--output-dir', default='data/motion_cache')
    parser.add_argument('--polyfem-bin', default='PolyFEM_bin',
                        help='Path to PolyFEM binary')
    parser.add_argument('--youngs', type=float, default=5e3)
    parser.add_argument('--poisson', type=float, default=0.45)
    parser.add_argument('--dhat', type=float, default=5.0, help='Barrier distance (mm)')
    args = parser.parse_args()

    dhat_m = args.dhat * 0.001

    # ── Load skeleton ─────────────────────────────────────────────────────
    print("[1] Loading skeleton...")
    skel_info, root_name, bvh_info, _, mesh_info, _ = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)
    skel.setPositions(np.zeros(skel.getNumDofs()))
    print(f"    DOFs: {skel.getNumDofs()}")

    # ── Load bone meshes ──────────────────────────────────────────────────
    print("[2] Loading bone meshes...")
    import trimesh
    bone_meshes = {}
    bone_rest_transforms = {}
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
    skel.setPositions(np.zeros(skel.getNumDofs()))
    for bone_name in bone_meshes:
        body_name = BONE_NAME_TO_BODY.get(bone_name, bone_name + '0')
        bn = skel.getBodyNode(body_name)
        if bn is not None:
            wt = bn.getWorldTransform()
            bone_rest_transforms[body_name] = (wt.rotation().copy(), wt.translation().copy())
    print(f"    {len(bone_meshes)} bones")

    # ── Load BVH ──────────────────────────────────────────────────────────
    print("[3] Parsing muscle XML + BVH...")
    xml_data = parse_muscle_xml()
    from viewer.zygote_mesh_ui import _detect_bvh_tframe
    t_frame = _detect_bvh_tframe(args.bvh)
    motion_bvh = MyBVH(args.bvh, bvh_info, skel, T_frame=t_frame)
    n_frames = motion_bvh.mocap_refs.shape[0]
    end_frame = args.end_frame if args.end_frame is not None else n_frames - 1
    total_frames = end_frame - args.start_frame + 1
    print(f"    Frames: {n_frames}, baking {args.start_frame}-{end_frame}")

    # ── Per side ──────────────────────────────────────────────────────────
    for side in args.sides:
        print(f"\n{'='*60}\n  Processing {side} side\n{'='*60}")
        side_bones = COLLISION_BONES.get(side, [])

        # Load muscles from tet cache
        print("[4] Loading muscles...")
        muscles = []
        for mname in UPLEG_MUSCLES:
            full_name = f"{side}_{mname}"
            cache_path = os.path.join('tet_orig', f"L_{mname}_tet.npz")
            if not os.path.exists(cache_path):
                continue
            with open(cache_path, 'rb') as f:
                cd = pickle.load(f)
            verts = cd['tet_vertices'].copy()
            tet_e = cd['tet_elements'].copy()
            cap_verts = cd['cap_vertices']
            mxml = xml_data.get(f"L_{mname}", [])

            if side == 'R':
                verts[:, 0] *= -1
                tet_e[:, 1], tet_e[:, 2] = tet_e[:, 2].copy(), tet_e[:, 1].copy()

            bone_map = None
            if side == 'R':
                bone_map = {k + '0': v + '0' for k, v in SKELETON_NAME_MAP.items()}
                bone_map['Saccrum_Coccyx0'] = 'Saccrum_Coccyx0'

            fixed_verts, local_anchors, init_transforms = identify_fixed_and_assign_bones(
                verts, cap_verts,
                cd['orig_vertices'] if side == 'L'
                else cd['orig_vertices'].copy() * np.array([-1, 1, 1]),
                mxml, skel, bone_map)

            skin_w, skin_bones = compute_skinning_weights(verts, local_anchors, init_transforms)

            muscles.append({
                'name': full_name,
                'vertices': verts,
                'tetrahedra': tet_e,
                'fixed_vertices': sorted(fixed_verts.keys()),
                'local_anchors': local_anchors,
                'initial_transforms': init_transforms,
                'skinning_weights': skin_w,
                'skinning_bones': skin_bones,
                'xml_streams': mxml,
            })
            print(f"    {full_name}: {len(verts)} verts, {len(tet_e)} tets, "
                  f"{len(local_anchors)} fixed")

        if not muscles:
            continue

        # Build unified mesh
        global_offset = {}
        offset = 0
        for m in muscles:
            global_offset[m['name']] = offset
            offset += len(m['vertices'])
        n_total = offset

        global_rest = np.zeros((n_total, 3))
        global_fixed_mask = np.zeros(n_total, dtype=bool)
        all_local_anchors = {}
        all_tets = []

        for m in muscles:
            off = global_offset[m['name']]
            n = len(m['vertices'])
            global_rest[off:off + n] = m['vertices']
            for vi in m['fixed_vertices']:
                global_fixed_mask[off + vi] = True
            for vi, (bone, lpos) in m['local_anchors'].items():
                all_local_anchors[off + vi] = (bone, lpos)
            all_tets.append(m['tetrahedra'] + off)

        global_tets = np.vstack(all_tets)
        print(f"    Unified: {n_total} verts, {len(global_tets)} tets, "
              f"{np.sum(global_fixed_mask)} fixed")

        # ── Frame loop ────────────────────────────────────────────────────
        print(f"[5] Baking frames {args.start_frame}-{end_frame}...")
        bvh_stem = os.path.splitext(os.path.basename(args.bvh))[0]
        cache_dir = os.path.join(args.output_dir, bvh_stem, f"orig_{side}_UpLeg")
        os.makedirs(cache_dir, exist_ok=True)

        bake_data = {m['name']: {} for m in muscles}
        global_positions = global_rest.copy()
        bake_start = time.time()

        for fi, frame in enumerate(range(args.start_frame, end_frame + 1)):
            frame_start = time.time()
            skel.setPositions(motion_bvh.mocap_refs[frame])

            # Fixed targets
            fixed_indices = np.where(global_fixed_mask)[0]
            fixed_targets = np.zeros_like(global_rest)
            fixed_targets[:] = global_rest
            for gi, (bone_name, local_pos) in all_local_anchors.items():
                body_node = skel.getBodyNode(bone_name)
                if body_node is not None:
                    R = body_node.getWorldTransform().rotation()
                    t = body_node.getWorldTransform().translation()
                    fixed_targets[gi] = R @ local_pos + t

            # LBS initial guess
            for m in muscles:
                off = global_offset[m['name']]
                n = len(m['vertices'])
                rest = m['vertices']
                lbs = np.zeros((n, 3))
                for bi, bone_name in enumerate(m['skinning_bones']):
                    body_node = skel.getBodyNode(bone_name)
                    if body_node is None:
                        continue
                    R_cur = body_node.getWorldTransform().rotation()
                    t_cur = body_node.getWorldTransform().translation()
                    if bone_name in m['initial_transforms']:
                        R0, t0 = m['initial_transforms'][bone_name]
                    else:
                        continue
                    local = (R0.T @ (rest - t0).T).T
                    deformed = (R_cur @ local.T).T + t_cur
                    w = m['skinning_weights'][:, bi:bi + 1]
                    lbs += w * deformed
                global_positions[off:off + n] = lbs

            # Write meshes to persistent temp directory
            work_dir = os.path.abspath(os.path.join(args.output_dir, 'polyfem_work'))
            os.makedirs(work_dir, exist_ok=True)

            # Write REST mesh (polyfem deforms from rest → target via BCs)
            msh_path = os.path.join(work_dir, 'muscles.msh')
            if fi == 0:  # only write once (rest shape doesn't change)
                write_msh(msh_path, global_rest, global_tets)

            # Bone obstacle OBJs
            bone_objs = []
            for bone_name in side_bones:
                if bone_name not in bone_meshes:
                    continue
                bm = bone_meshes[bone_name]
                wv = get_bone_world_verts(
                    skel, bone_name, bm['vertices'], bone_rest_transforms)
                if wv is None:
                    continue
                obj_path = os.path.join(work_dir, f'{bone_name}.obj')
                write_obj(obj_path, wv, bm['faces'])
                bone_objs.append((obj_path, bone_name))

            # Dirichlet BCs: select fixed vertices by bounding box,
            # zero displacement (mesh already at target positions)
            # Group all fixed verts into one BC with a big bounding box
            # containing all of them, then use zero displacement.
            # Problem: this selects ALL verts in the box, not just fixed ones.
            #
            # Better: use MSH physical groups to tag fixed nodes.
            # Rewrite MSH with node sets.
            #
            # Simplest for now: select bottom/top surfaces by axis position.
            # But our fixed verts are scattered...
            #
            # PolyFEM supports "node" selection type for individual nodes:
            # Not documented but might work. Let's try grouping by bone.
            #
            # Actually, use sphere selection with tiny radius around each fixed vert.
            # But 4888 selections is too many.
            #
            # Practical approach: tag the MSH file with $PhysicalGroups
            # and $NodeData to mark fixed vertices.

            # Dirichlet BCs: use bounding box selections grouping fixed verts by bone
            # Group fixed vertices by their attachment bone
            from collections import defaultdict
            bone_groups = defaultdict(list)
            for gi in fixed_indices:
                if gi in all_local_anchors:
                    bone_name = all_local_anchors[gi][0]
                    bone_groups[bone_name].append(gi)
                else:
                    bone_groups['_rest'].append(gi)

            point_selections = []
            dirichlet_bcs = []
            bc_id = 1
            for bone_name, verts_gi in bone_groups.items():
                if not verts_gi:
                    continue
                verts_gi = np.array(verts_gi)
                # Bounding box around REST positions of these fixed verts
                rest_pos = global_rest[verts_gi]
                bbox_min = rest_pos.min(axis=0) - 1e-4
                bbox_max = rest_pos.max(axis=0) + 1e-4
                point_selections.append({
                    "id": bc_id,
                    "box": [bbox_min.tolist(), bbox_max.tolist()],
                })
                # Average displacement: rest → target for this bone group
                # (all verts in group move with same bone, so avg is accurate)
                disp = fixed_targets[verts_gi] - global_rest[verts_gi]
                avg_disp = disp.mean(axis=0)
                # Use "t" to interpolate: at t=1 apply full displacement
                dirichlet_bcs.append({
                    "id": bc_id,
                    "value": [
                        f"{avg_disp[0]:.8f} * t",
                        f"{avg_disp[1]:.8f} * t",
                        f"{avg_disp[2]:.8f} * t",
                    ],
                })
                bc_id += 1

            out_dir = os.path.join(work_dir, 'output')
            os.makedirs(out_dir, exist_ok=True)

            config = build_polyfem_config(
                msh_path, bone_objs, point_selections, dirichlet_bcs,
                args.youngs, args.poisson, dhat_m, out_dir)

            config_path = os.path.join(work_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

                # Run PolyFEM
                print(f"  Frame {frame}: running PolyFEM ({config_path})...",
                      end='', flush=True)
                try:
                    log_path = os.path.join(work_dir, 'polyfem.log')
                    cmd = f'{args.polyfem_bin} -j {config_path} --log_level 2 > {log_path} 2>&1'
                    result = subprocess.run(
                        cmd, shell=True, timeout=600)
                    result.stderr = open(log_path).read()[-500:]
                    if result.returncode != 0:
                        print(f" FAILED (exit {result.returncode})")
                        print(f"    stderr: {result.stderr[:500]}")
                        continue
                except subprocess.TimeoutExpired:
                    print(" TIMEOUT (600s)")
                    continue
                except FileNotFoundError:
                    print(f" ERROR: {args.polyfem_bin} not found")
                    return

                # Read results
                vtu_path = os.path.join(out_dir, 'result.vtu')
                if not os.path.exists(vtu_path):
                    # Check for timestep output
                    for f_name in os.listdir(out_dir):
                        if f_name.endswith('.vtu'):
                            vtu_path = os.path.join(out_dir, f_name)
                            break

                if os.path.exists(vtu_path):
                    try:
                        new_positions = read_vtu_positions(vtu_path)
                        if len(new_positions) == n_total:
                            global_positions = new_positions
                            print(f" OK", end='')
                        else:
                            print(f" vertex count mismatch ({len(new_positions)} vs {n_total})")
                    except Exception as e:
                        print(f" read error: {e}")
                else:
                    print(f" no output VTU found")

            # Capture
            for m in muscles:
                off = global_offset[m['name']]
                n = len(m['vertices'])
                bake_data[m['name']][frame] = global_positions[off:off + n].astype(np.float32)

            dt = time.time() - frame_start
            elapsed = time.time() - bake_start
            avg = elapsed / (fi + 1)
            eta = avg * (total_frames - fi - 1)
            print(f" {dt:.1f}s (avg {avg:.1f}s ETA {eta:.0f}s)")

            # Progressive save
            if (fi + 1) % FLUSH_INTERVAL == 0 or frame == end_frame:
                chunk_start = fi - (fi % FLUSH_INTERVAL)
                chunk_idx = chunk_start // FLUSH_INTERVAL
                chunk_frames = list(range(
                    args.start_frame + chunk_start,
                    min(args.start_frame + chunk_start + FLUSH_INTERVAL, end_frame + 1)))
                for m in muscles:
                    chunk_pos, chunk_f = [], []
                    for cf in chunk_frames:
                        if cf in bake_data[m['name']]:
                            chunk_pos.append(bake_data[m['name']][cf])
                            chunk_f.append(cf)
                    if chunk_pos:
                        out_path = os.path.join(
                            cache_dir, f"{m['name']}_chunk_{chunk_idx:04d}.npz")
                        np.savez(out_path,
                                 frames=np.array(chunk_f, dtype=np.int32),
                                 positions=np.array(chunk_pos, dtype=np.float32))
                for cf in chunk_frames:
                    for m in muscles:
                        bake_data[m['name']].pop(cf, None)
                gc.collect()
                print(f"  Flushed chunk {chunk_idx}")

        elapsed_total = time.time() - bake_start
        print(f"\nDone in {elapsed_total:.1f}s. {len(muscles)} muscles, {total_frames} frames.")


if __name__ == '__main__':
    main()
