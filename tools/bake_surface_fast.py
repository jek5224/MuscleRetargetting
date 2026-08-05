#!/usr/bin/env python3
"""Fast collision-aware bake for one muscle or a complete muscle region.

This intentionally has no tendon/group/EMU material path. It transports the
whole tet volume with DART LBS, projects bone and inter-muscle contacts on the
visible tet boundary, then propagates contact corrections through the tet
graph. It is designed for quick, robust motion-cache generation rather than a
full constitutive solve.

Example:
  python tools/bake_surface_fast.py --region L_UpLeg \
      --bvh data/motion/run.bvh --end-frame 20
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import splu
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import bake_emu
import test_emu
from core.bvhparser import MyBVH


REGION_CONFIGS = {
    'L_UpLeg': '.muscles_L_UpLeg.json',
    'R_UpLeg': '.muscles_R_UpLeg.json',
    'L_LowLeg': '.muscles_L_LowLeg.json',
    'R_LowLeg': '.muscles_R_LowLeg.json',
}


def load_tet(path):
    with open(path, 'rb') as f:
        try:
            data = pickle.load(f)
        except Exception:
            f.seek(0); z = np.load(f, allow_pickle=True)
            data = {k: z[k] for k in z.files}
    return data


def surface_faces(tets):
    faces = {}
    counts = {}
    for a, b, c, d in np.asarray(tets, dtype=np.int32):
        for f in ((a, c, b), (a, b, d), (b, c, d), (a, d, c)):
            k = tuple(sorted(map(int, f))); counts[k] = counts.get(k, 0) + 1
            faces.setdefault(k, f)
    return np.asarray([faces[k] for k, n in counts.items() if n == 1], dtype=np.int32)


def graph_laplacian(tets, n):
    edges = set()
    for tet in np.asarray(tets, dtype=np.int32):
        for i in range(4):
            for j in range(i + 1, 4): edges.add(tuple(sorted((int(tet[i]), int(tet[j])))))
    e = np.asarray(sorted(edges), dtype=np.int32)
    rows = np.r_[e[:, 0], e[:, 1]]; cols = np.r_[e[:, 1], e[:, 0]]
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return diags(np.asarray(A.sum(1)).ravel()) - A


def resolve_tet_paths(explicit_paths, region=None, tet_dir=Path('tet'),
                      muscles_manifest=None):
    """Resolve explicit tet paths or every muscle listed for a region."""
    paths = [Path(p) for p in explicit_paths]
    if muscles_manifest is not None:
        config_path = Path(muscles_manifest)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        if not config_path.exists():
            raise FileNotFoundError(f'No muscle manifest found: {config_path}')
        with config_path.open() as f:
            entries = json.load(f)
        paths.extend(Path(tet_dir) / f"{entry['name']}_tet.npz"
                     for entry in entries)
    elif region:
        config_path = ROOT / REGION_CONFIGS[region]
        if not config_path.exists():
            fallback = ROOT / 'tools' / f'muscles_{region}.json'
            config_path = fallback if fallback.exists() else config_path
        if not config_path.exists():
            raise FileNotFoundError(f'No muscle list found for {region}')
        with config_path.open() as f:
            entries = json.load(f)
        paths.extend(Path(tet_dir) / f"{entry['name']}_tet.npz" for entry in entries)

    unique = []
    seen = set()
    for path in paths:
        resolved = path if path.is_absolute() else ROOT / path
        key = str(resolved.resolve())
        if key in seen:
            continue
        if not resolved.exists():
            raise FileNotFoundError(f'Tet file not found: {resolved}')
        seen.add(key)
        unique.append(resolved)
    if not unique:
        raise ValueError('Pass one or more tet files, or use --region')
    return unique


def transport_with_lbs(q, old_target, new_target, fixed):
    """Carry the entire previous shape by the per-vertex LBS target delta.

    The old implementation moved only fixed cap vertices. Free vertices stayed
    behind at the previous/rest pose, stretching the muscle across the scene.
    Transporting every vertex preserves any previous contact correction while
    making the whole volume follow the skeleton.
    """
    q = np.asarray(q, dtype=np.float64).copy()
    q += np.asarray(new_target) - np.asarray(old_target)
    q[np.asarray(fixed, dtype=np.int32)] = np.asarray(new_target)[fixed]
    return q


def flush_cache(buffers, frame_buffer, groups, output_dir, chunk_index):
    """Write viewer-compatible, per-muscle chunks with heterogeneous sizes."""
    if not frame_buffer:
        return chunk_index
    frames = np.asarray(frame_buffer, dtype=np.int32)
    for group, positions in zip(groups, buffers):
        if not positions:
            continue
        out = output_dir / f"{group['name']}_chunk_{chunk_index:04d}.npz"
        np.savez_compressed(out, frames=frames,
                            positions=np.stack(positions).astype(np.float32))
        positions.clear()
    frame_buffer.clear()
    return chunk_index + 1


def build_bones(skel):
    meshes = []
    rest = {}
    old = skel.getPositions().copy(); skel.setPositions(np.zeros(skel.getNumDofs()))
    for i in range(skel.getNumBodyNodes()):
        b = skel.getBodyNode(i); wt = b.getWorldTransform()
        rest[b.getName()] = (wt.rotation().copy(), wt.translation().copy())
    skel.setPositions(old)
    mesh_dir = ROOT / bake_emu.SKEL_MESH_DIR
    for p in sorted(mesh_dir.glob('*.obj')):
        stem = p.stem; body = skel.getBodyNode(stem) or skel.getBodyNode(stem + '0')
        if body is None or stem not in rest and body.getName() not in rest: continue
        raw = trimesh.load(p, process=False)
        body_name = body.getName(); R0, t0 = rest[body_name]
        R = body.getWorldTransform().rotation(); t = body.getWorldTransform().translation()
        local = (R0.T @ (np.asarray(raw.vertices) * bake_emu.MESH_SCALE - t0).T).T
        posed = (R @ local.T).T + t
        meshes.append(trimesh.Trimesh(vertices=posed, faces=raw.faces, process=True))
    return meshes


def project_contacts(q, rest, faces, bones, other_surfaces, fixed, fixed_targets,
                     lap, margin, passes, max_contact_step=0.02,
                     local_projection=False):
    q = q.copy(); n = len(q); fixed = set(map(int, fixed))
    surface_ids = np.unique(faces).astype(np.int32)
    for _ in range(max(1, passes)):
        targets = {}
        surf = q[surface_ids]
        for bone in bones:
            ids = np.where(np.all((surf >= bone.bounds[0] - margin) &
                                  (surf <= bone.bounds[1] + margin), axis=1))[0]
            if not len(ids): continue
            try: inside = bone.contains(surf[ids])
            except Exception: continue
            ids = ids[inside]
            if not len(ids): continue
            pts = q[surface_ids[ids]]
            near, _, face_id = trimesh.proximity.closest_point(bone, pts)
            for k, vi in enumerate(surface_ids[ids]):
                if int(vi) in fixed: continue
                d = near[k] - q[vi]; l = np.linalg.norm(d)
                if l > 1e-12: d /= l
                else: d = bone.face_normals[face_id[k]]
                targets[int(vi)] = near[k] + margin * d - q[vi]
        for other in other_surfaces:
            if other is None: continue
            ids = np.where(np.all((surf >= other.bounds[0] - margin) &
                                  (surf <= other.bounds[1] + margin), axis=1))[0]
            if not len(ids): continue
            try: inside = other.contains(surf[ids])
            except Exception: continue
            ids = ids[inside]
            if not len(ids): continue
            pts = q[surface_ids[ids]]
            near, _, _ = trimesh.proximity.closest_point(other, pts)
            for k, vi in enumerate(surface_ids[ids]):
                if int(vi) not in fixed: targets[int(vi)] = near[k] - q[vi]
        if not targets: break
        if local_projection:
            for vi, delta in targets.items():
                length = float(np.linalg.norm(delta))
                if length > max_contact_step:
                    delta = delta * (max_contact_step / length)
                q[vi] += delta
            q[list(fixed)] = fixed_targets
            continue
        constrained = sorted(fixed | set(targets)); free = np.asarray(
            [i for i in range(n) if i not in set(constrained)], dtype=np.int32)
        c = np.asarray(constrained, dtype=np.int32); disp = np.zeros((n, 3))
        prescribed = np.zeros((len(c), 3)); lookup = {int(v): i for i, v in enumerate(c)}
        for vi, d in targets.items(): prescribed[lookup[vi]] = d
        disp[c] = prescribed
        if len(free):
            solve = splu((lap[free][:, free] + 1e-10 * diags(np.ones(len(free)))).tocsc())
            rhs = -(lap[free][:, c] @ prescribed)
            for a in range(3): disp[free, a] = solve.solve(rhs[:, a])
        # Clamp only catastrophic graph propagation. A 2 mm cap required many
        # expensive passes and left ordinary 10-17 mm penetrations unresolved
        # in the fast one-pass configuration.
        length = np.linalg.norm(disp, axis=1)
        scale = min(1.0, max_contact_step /
                    max(float(np.max(length)), 1e-12))
        q += scale * disp
        q[list(fixed)] = fixed_targets

    # Harmonic propagation can pull a different boundary vertex a small
    # distance into a collider after the queried vertices have been pushed
    # out.  A few cheap local sweeps remove those residuals without paying for
    # another sparse factorization (or leaving sub-millimetre penetrations in
    # the default fast path).
    cleanup_colliders = [(mesh, True) for mesh in bones]
    cleanup_colliders.extend(
        (mesh, False) for mesh in other_surfaces if mesh is not None)
    for _ in range(0 if local_projection else 1):
        changed = False
        for collider, add_margin in cleanup_colliders:
            surf = q[surface_ids]
            ids = np.where(np.all((surf >= collider.bounds[0] - margin) &
                                  (surf <= collider.bounds[1] + margin), axis=1))[0]
            if not len(ids):
                continue
            try:
                ids = ids[collider.contains(surf[ids])]
            except Exception:
                continue
            ids = np.asarray([i for i in ids
                              if int(surface_ids[i]) not in fixed], dtype=np.int32)
            if not len(ids):
                continue
            vertex_ids = surface_ids[ids]
            near, _, face_id = trimesh.proximity.closest_point(
                collider, q[vertex_ids])
            if add_margin:
                direction = near - q[vertex_ids]
                length = np.linalg.norm(direction, axis=1)
                valid = length > 1e-12
                direction[valid] /= length[valid, None]
                direction[~valid] = collider.face_normals[face_id[~valid]]
                near += margin * direction
            q[vertex_ids] = near
            changed = True
        q[list(fixed)] = fixed_targets
        if not changed:
            break
    return q


def main():
    program_start = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument('tets', nargs='*', type=Path,
                    help='Tet files to bake (optional when --region is used)')
    ap.add_argument('--region', choices=sorted(REGION_CONFIGS),
                    help='Load every tet in this anatomical region')
    ap.add_argument('--muscles-manifest', type=Path,
                    help='JSON muscle list to load instead of a region preset')
    ap.add_argument('--tet-dir', type=Path, default=Path('tet'))
    ap.add_argument('--bvh', type=Path)
    ap.add_argument('--start-frame', type=int, default=0)
    ap.add_argument('--end-frame', type=int, default=None)
    ap.add_argument('--steps', type=int, default=1,
                    help='LBS/contact substeps per frame (default: 1; use 4-8 '
                         'for poses that tunnel through bones)')
    ap.add_argument('--passes', type=int, default=1,
                    help='Bone-contact projection passes per substep')
    ap.add_argument('--margin', type=float, default=.0015)
    ap.add_argument('--max-contact-step', type=float, default=.02,
                    help='Maximum propagated contact correction per pass in '
                         'meters (default: 0.02)')
    ap.add_argument('--local-contact', action='store_true',
                    help='Project only penetrating surface vertices instead '
                         'of harmonically propagating corrections. This '
                         'prevents contact-induced whole-muscle spikes.')
    ap.add_argument('--inter-passes', type=int, default=1,
                    help='Sequential all-muscle contact sweeps per frame '
                         '(default: 1; use 2 for tighter contact resolution)')
    ap.add_argument('--chunk-size', type=int, default=20)
    ap.add_argument('--output-dir', type=Path, default=None)
    args = ap.parse_args(); skel, bvh_info, _ = bake_emu.load_skeleton()
    paths = resolve_tet_paths(
        args.tets, args.region, args.tet_dir, args.muscles_manifest)
    trees = test_emu._load_bone_trees(); groups = []
    for path in paths:
        data = load_tet(path); g = test_emu.prepare_group_data(data, path.stem.replace('_tet',''), skel, trees, source_path=path, attachment_rings=0)
        g['faces'] = surface_faces(g['tetrahedra']); g['lap'] = graph_laplacian(g['tetrahedra'], len(g['vertices']))
        groups.append(g)
    motion = None
    if args.bvh:
        motion = MyBVH(str(args.bvh), bvh_info, skel,
                       T_frame=bake_emu._detect_bvh_tframe(str(args.bvh)))
        last_frame = len(motion.mocap_refs) - 1
    else:
        last_frame = args.end_frame if args.end_frame is not None else 0
    end_frame = last_frame if args.end_frame is None else min(args.end_frame, last_frame)
    if args.start_frame < 0 or end_frame < args.start_frame:
        raise ValueError(f'Invalid frame range {args.start_frame}..{end_frame}')
    frames = range(args.start_frame, end_frame + 1)

    output_dir = args.output_dir
    if output_dir is None:
        if args.bvh and args.region:
            output_dir = (ROOT / 'data' / 'motion_cache' / args.bvh.stem /
                          f'{args.region}_surface_fast')
        else:
            output_dir = ROOT / 'surface_fast_cache'
    elif not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob('*_chunk_*.npz'):
        stale.unlink()
    done_marker = output_dir / '.done'
    if done_marker.exists():
        done_marker.unlink()

    previous = [g['vertices'].copy() for g in groups]
    previous_targets = [g['vertices'].copy() for g in groups]
    buffers = [[] for _ in groups]
    frame_buffer = []
    chunk_index = 0
    setup_elapsed = time.perf_counter() - program_start
    frame_times = []
    print(f'Baking {len(groups)} muscles, frames {args.start_frame}..{end_frame}')
    for frame in frames:
        frame_start = time.perf_counter()
        if motion is not None: skel.setPositions(motion.mocap_refs[frame].copy())
        bones = build_bones(skel); posed = []
        for gi, g in enumerate(groups):
            target = bake_emu.compute_rigid_blend_positions(g['lbs_bindings'], skel, g['axis_coordinate'])
            fixed = np.asarray(g['fixed_vertices'], dtype=np.int32)
            q = previous[gi].copy()
            old_target = previous_targets[gi]
            step_target = old_target
            for step in range(max(1, args.steps)):
                f = (step + 1) / max(1, args.steps)
                next_target = (1.0 - f) * old_target + f * target
                q = transport_with_lbs(q, step_target, next_target, fixed)
                q = project_contacts(q, g['vertices'], g['faces'], bones, [], fixed,
                                     next_target[fixed], g['lap'], args.margin,
                                     args.passes, args.max_contact_step,
                                     args.local_contact)
                step_target = next_target
            q[fixed] = target[fixed]
            posed.append(q)
            previous_targets[gi] = target

        # Resolve contacts against current-frame shapes for every muscle.
        # Rebuild surfaces after each correction so later sweeps do not use
        # stale previous-frame geometry.
        for _ in range(max(0, args.inter_passes)):
            surfaces = [trimesh.Trimesh(vertices=q, faces=g['faces'], process=False)
                        for q, g in zip(posed, groups)]
            for gi, g in enumerate(groups):
                fixed = np.asarray(g['fixed_vertices'], dtype=np.int32)
                others = surfaces[:gi] + surfaces[gi + 1:]
                posed[gi] = project_contacts(
                    posed[gi], g['vertices'], g['faces'], [], others, fixed,
                    previous_targets[gi][fixed], g['lap'], args.margin,
                    args.passes, args.max_contact_step, args.local_contact)
                posed[gi][fixed] = previous_targets[gi][fixed]
                surfaces[gi] = trimesh.Trimesh(
                    vertices=posed[gi], faces=g['faces'], process=False)

        for gi, q in enumerate(posed):
            previous[gi] = q
            buffers[gi].append(q.astype(np.float32))
        frame_buffer.append(frame)
        if len(frame_buffer) >= max(1, args.chunk_size):
            chunk_index = flush_cache(
                buffers, frame_buffer, groups, output_dir, chunk_index)
        frame_elapsed = time.perf_counter() - frame_start
        frame_times.append(frame_elapsed)
        print(f'frame {frame}/{end_frame}: {len(groups)} muscles, '
              f'pure bake {frame_elapsed:.3f}s')

    chunk_index = flush_cache(buffers, frame_buffer, groups, output_dir, chunk_index)
    done_marker.write_text(
        f'{end_frame - args.start_frame + 1} frames, {len(groups)} muscles\n')
    pure_total = float(np.sum(frame_times))
    pure_avg = pure_total / max(len(frame_times), 1)
    print(f'Done: wrote {chunk_index} chunk(s) per muscle to {output_dir}')
    print(f'Timing: setup={setup_elapsed:.3f}s, pure bake={pure_total:.3f}s, '
          f'average={pure_avg:.3f}s/frame')

if __name__ == '__main__': main()
