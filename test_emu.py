#!/usr/bin/env python3
"""Headless EMU smoke test on the Zygote DART skeleton.

This runner accepts the group tet files written by the viewer.  In particular,
it handles connected tendon/belly tets whose top-level cap attachment arrays are
empty: the outer endpoints are recovered from the saved component fiber
waypoints, matched to the closest Zygote skeleton OBJ, and then resolved to the
corresponding DART BodyNode.

Examples
--------
Rest-pose smoke test (CPU, no window/OpenGL)::

    python test_emu.py tet/L_Rectus_Femoris_tet.npz --max-iters 3

Test a posed skeleton from a BVH frame::

    python test_emu.py tet/L_Rectus_Femoris_tet.npz \
        --bvh data/motion/walk1_subject1.bvh --frame 10 --max-iters 10
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.bvhparser import MyBVH
from tools import bake_emu


def _load_saved_tet(path: Path) -> dict:
    """Read the viewer's pickle format, with npz compatibility."""
    with path.open("rb") as f:
        try:
            data = pickle.load(f)
        except Exception:
            f.seek(0)
            npz = np.load(f, allow_pickle=True)
            data = {key: npz[key] for key in npz.files}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a tet dictionary, got {type(data).__name__}")
    return data


def _orient_and_filter_tets(vertices: np.ndarray, tetrahedra: np.ndarray):
    tets = np.asarray(tetrahedra, dtype=np.int32).copy()
    verts = np.asarray(vertices, dtype=np.float64)
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"tetrahedra must have shape (N, 4), got {tets.shape}")
    v0 = verts[tets[:, 0]]
    signed6 = np.einsum(
        "ij,ij->i",
        np.cross(verts[tets[:, 1]] - v0, verts[tets[:, 2]] - v0),
        verts[tets[:, 3]] - v0,
    )
    negative = signed6 < 0.0
    if np.any(negative):
        tmp = tets[negative, 1].copy()
        tets[negative, 1] = tets[negative, 2]
        tets[negative, 2] = tmp
        signed6[negative] *= -1.0
    scale = max(float(np.linalg.norm(np.ptp(verts, axis=0))), 1.0)
    good = np.isfinite(signed6) & (signed6 > 1e-14 * scale**3)
    if not np.all(good):
        print(f"    Removed {int(np.sum(~good))} degenerate tetrahedra")
        tets = tets[good]
    return tets


def _surface_faces(tetrahedra: np.ndarray) -> np.ndarray:
    counts: Counter[tuple[int, int, int]] = Counter()
    oriented = {}
    for a, b, c, d in np.asarray(tetrahedra, dtype=np.int32):
        for face in ((a, c, b), (a, b, d), (b, c, d), (a, d, c)):
            key = tuple(sorted(int(x) for x in face))
            counts[key] += 1
            oriented.setdefault(key, tuple(int(x) for x in face))
    return np.asarray([oriented[k] for k, count in counts.items() if count == 1],
                      dtype=np.int32)


def _component_snapshots(data: dict) -> list[dict]:
    return [x for x in (data.get("connected_component_fibers") or [])
            if isinstance(x, dict)]


def _snapshot_value(entry: dict, key: str):
    state = entry.get("component_state")
    if isinstance(state, dict) and state.get(key) is not None:
        return state[key]
    return entry.get(key)


def _fiber_endpoint_clouds(data: dict):
    """Return origin points, insertion points, and centerline samples."""
    origins, insertions, center_points, center_u = [], [], [], []
    for entry in _component_snapshots(data):
        # The belly snapshot contains the tendon-extended stream, so its first
        # and last levels are the actual outer group endpoints.
        if entry.get("part") != "belly":
            continue
        streams = _snapshot_value(entry, "waypoints")
        if not streams:
            streams = _snapshot_value(entry, "waypoints_original")
        for stream in streams or []:
            if stream is None or len(stream) < 2:
                continue
            levels = [np.asarray(level, dtype=np.float64).reshape(-1, 3)
                      for level in stream]
            origins.append(levels[0])
            insertions.append(levels[-1])
            denom = max(len(levels) - 1, 1)
            for li, level in enumerate(levels):
                center_points.append(np.mean(level, axis=0))
                center_u.append(li / denom)
    if origins and insertions:
        return (np.vstack(origins), np.vstack(insertions),
                np.asarray(center_points), np.asarray(center_u))
    return None, None, None, None


def _region_endpoint_clouds(data: dict, vertices: np.ndarray,
                            tetrahedra: np.ndarray, surface_vertices: np.ndarray):
    """Fallback for older group saves without component fiber snapshots."""
    labels = data.get("tet_region_labels")
    if labels is None or len(labels) != len(tetrahedra):
        axis = vertices[:, int(np.argmax(np.ptp(vertices, axis=0)))]
        lo, hi = np.quantile(axis[surface_vertices], [0.03, 0.97])
        return vertices[surface_vertices[axis[surface_vertices] >= hi]], \
            vertices[surface_vertices[axis[surface_vertices] <= lo]]

    labels = np.asarray(labels).astype(str)
    belly_ids = np.where(labels == "belly")[0]
    belly_center = (vertices[np.unique(tetrahedra[belly_ids])].mean(axis=0)
                    if len(belly_ids) else vertices.mean(axis=0))
    clouds = []
    surface_set = set(int(x) for x in surface_vertices)
    for part in ("origin_tendon", "insertion_tendon"):
        ids = np.where(labels == part)[0]
        region_vertices = np.unique(tetrahedra[ids]) if len(ids) else surface_vertices
        region_surface = np.asarray([v for v in region_vertices if int(v) in surface_set],
                                    dtype=np.int32)
        if len(region_surface) == 0:
            region_surface = surface_vertices
        dist = np.linalg.norm(vertices[region_surface] - belly_center, axis=1)
        cutoff = np.quantile(dist, 0.96)
        clouds.append(vertices[region_surface[dist >= cutoff]])
    return clouds[0], clouds[1]


def _nearest_surface_cap(points: np.ndarray, vertices: np.ndarray,
                         surface_faces: np.ndarray) -> list[int]:
    surface_vertices = np.unique(surface_faces)
    surface_points = vertices[surface_vertices]
    tree = cKDTree(surface_points)
    _, nearest = tree.query(points)
    seeds = set(int(surface_vertices[i]) for i in np.atleast_1d(nearest))
    # Do not radius-expand these seeds across the whole terminal surface.  On
    # a wide insertion (Rectus Femoris), 50 fiber endpoint samples previously
    # expanded to 485 hard-pinned vertices.  The stiff tendon then transferred
    # all mismatch into the first free tet row and produced local extrusion.
    # The unique nearest samples are already distributed over the full cap;
    # tendon elasticity carries their rigid-body motion to intervening verts.
    return sorted(seeds)


def _stored_attachment_caps(data: dict, vertices: np.ndarray,
                            surface_faces: np.ndarray):
    """Read authoritative endpoint caps from saved tet metadata."""
    raw = data.get("anchor_vertices")
    levels = data.get("vertex_contour_level")
    if raw is None or levels is None:
        return None
    anchors = np.unique(np.asarray(raw, dtype=np.int64).reshape(-1))
    levels = np.asarray(levels)
    anchors = anchors[(anchors >= 0) & (anchors < len(vertices))]
    if levels.shape != (len(vertices),) or len(anchors) < 2:
        return None
    surface = set(int(value) for value in np.unique(surface_faces))
    anchors = np.asarray(
        [value for value in anchors if int(value) in surface],
        dtype=np.int64)
    if len(anchors) < 2:
        return None
    anchor_levels = levels[anchors]
    finite = np.isfinite(anchor_levels)
    anchors, anchor_levels = anchors[finite], anchor_levels[finite]
    if len(anchors) < 2:
        return None
    lo, hi = np.min(anchor_levels), np.max(anchor_levels)
    if lo == hi:
        return None
    origin = sorted(
        int(value) for value in anchors[anchor_levels == lo])
    insertion = sorted(
        int(value) for value in anchors[anchor_levels == hi])
    return (origin, insertion) if origin and insertion else None


def _expand_surface_vertex_rings(seeds, surface_faces, rings=1):
    """Expand attachment anchors only along surface topology."""
    selected = set(int(v) for v in seeds)
    if rings <= 0:
        return sorted(selected)
    adjacency = {}
    for face in np.asarray(surface_faces, dtype=np.int32):
        for vi in face:
            adjacency.setdefault(int(vi), set()).update(
                int(vj) for vj in face if vj != vi)
    frontier = set(selected)
    for _ in range(int(rings)):
        frontier = {
            neighbor for vi in frontier
            for neighbor in adjacency.get(vi, ()) if neighbor not in selected
        }
        selected.update(frontier)
        if not frontier:
            break
    return sorted(selected)


def _load_bone_trees():
    """Load Zygote bone geometry in the same metre units as the tet mesh."""
    result = {}
    skel_dir = ROOT / bake_emu.SKEL_MESH_DIR
    import trimesh
    for path in sorted(skel_dir.glob("*.obj")):
        mesh = trimesh.load(path, process=False)
        verts = np.asarray(mesh.vertices, dtype=np.float64) * bake_emu.MESH_SCALE
        if len(verts):
            result[path.stem] = cKDTree(verts)
    return result


def _closest_bone(points: np.ndarray, bone_trees: dict[str, cKDTree]):
    scores = []
    for name, tree in bone_trees.items():
        distance, _ = tree.query(points)
        # A trimmed multi-point distance is less sensitive to a single contour
        # sample than a centroid-only query.
        distance = np.sort(np.asarray(distance))
        keep = max(1, int(np.ceil(0.8 * len(distance))))
        scores.append((float(np.mean(distance[:keep])), name))
    if not scores:
        raise RuntimeError("No Zygote skeleton OBJ meshes were loaded")
    return min(scores)


def _resolve_body(skel, mesh_name: str):
    body, body_name = bake_emu._find_body(skel, mesh_name)
    if body is None:
        raise RuntimeError(f"Zygote bone '{mesh_name}' has no matching DART BodyNode")
    return body, body_name


def _axis_coordinate(vertices: np.ndarray, origin_points: np.ndarray,
                     insertion_points: np.ndarray, center_points, center_u):
    if center_points is not None and len(center_points) >= 2:
        # Project onto centerline *segments* and interpolate u continuously.
        # Nearest-level assignment quantizes u to the saved contour count
        # (typically 15), which makes posed LBS and the resulting tet surface
        # look like rigid stairs.
        centers = np.asarray(center_points, dtype=np.float64)
        values = np.asarray(center_u, dtype=np.float64)
        best_dist2 = np.full(len(vertices), np.inf, dtype=np.float64)
        best_u = np.zeros(len(vertices), dtype=np.float64)
        for i in range(len(centers) - 1):
            # A reset/decrease marks the boundary between two component
            # centerlines; never connect those as one segment.
            if values[i + 1] <= values[i]:
                continue
            a, b = centers[i], centers[i + 1]
            segment = b - a
            denom = float(np.dot(segment, segment))
            if denom < 1e-15:
                continue
            t = np.clip(((vertices - a) @ segment) / denom, 0.0, 1.0)
            projected = a + t[:, None] * segment
            dist2 = np.einsum('ij,ij->i', vertices - projected,
                              vertices - projected)
            better = dist2 < best_dist2
            best_dist2[better] = dist2[better]
            best_u[better] = values[i] + t[better] * (values[i + 1] - values[i])
        if np.any(np.isfinite(best_dist2)):
            missing = ~np.isfinite(best_dist2)
            if np.any(missing):
                tree = cKDTree(centers)
                _, ids = tree.query(vertices[missing])
                best_u[missing] = values[ids]
            return np.clip(best_u, 0.0, 1.0)
    origin = np.mean(origin_points, axis=0)
    insertion = np.mean(insertion_points, axis=0)
    axis = insertion - origin
    denom = float(np.dot(axis, axis))
    if denom < 1e-15:
        raise RuntimeError("Origin and insertion endpoint centers coincide")
    return np.clip(((vertices - origin) @ axis) / denom, 0.0, 1.0)


def _make_lbs_bindings(vertices, u, origin_body, insertion_body):
    bindings = []
    body_data = []
    for body in (origin_body, insertion_body):
        tf = body.getWorldTransform()
        body_data.append((body.getName(), tf.rotation().copy(), tf.translation().copy()))
    for vi, rest in enumerate(vertices):
        weights = []
        for weight, (name, rotation, translation) in zip(
                (1.0 - u[vi], u[vi]), body_data):
            if weight > 1e-10:
                weights.append((name, float(weight), rotation, translation))
        bindings.append((rest.copy(), weights))
    return bindings


def _harmonic_axis_coordinate(vertices, tetrahedra, origin_fixed,
                              insertion_fixed):
    """Paper-style heat/Laplace coordinate with endpoint Dirichlet values."""
    n = len(vertices)
    edge_set = set()
    for tet in np.asarray(tetrahedra, dtype=np.int32):
        for i in range(4):
            for j in range(i + 1, 4):
                edge_set.add(tuple(sorted((int(tet[i]), int(tet[j])))))
    edges = np.asarray(sorted(edge_set), dtype=np.int32)
    if len(edges) == 0:
        raise RuntimeError("Cannot build harmonic axis on a tet mesh with no edges")
    lengths = np.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    weights = 1.0 / np.maximum(lengths, 1e-8)
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    vals = np.concatenate((weights, weights))
    adjacency = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    laplacian = sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency

    origin = set(int(v) for v in origin_fixed)
    insertion = set(int(v) for v in insertion_fixed) - origin
    constrained = np.asarray(sorted(origin | insertion), dtype=np.int32)
    constrained_set = set(int(v) for v in constrained)
    free = np.asarray([v for v in range(n) if v not in constrained_set], dtype=np.int32)
    boundary = np.asarray(
        [1.0 if int(v) in insertion else 0.0 for v in constrained],
        dtype=np.float64)
    u = np.zeros(n, dtype=np.float64)
    u[constrained] = boundary
    if len(free):
        Lff = laplacian[free][:, free].tocsc()
        Lfc = laplacian[free][:, constrained].tocsc()
        solver = splu(Lff + 1e-10 * sp.eye(len(free), format='csc'))
        u[free] = solver.solve(-(Lfc @ boundary))
    return np.clip(u, 0.0, 1.0)


def prepare_group_data(data: dict, name: str, skel, bone_trees,
                       source_path=None, attachment_rings=0):
    """Prepare a saved or live viewer group for EMU at DART rest pose."""
    display_name = str(source_path or name)
    vertices = np.asarray(data.get("vertices"), dtype=np.float64)
    tetrahedra = _orient_and_filter_tets(vertices, data.get("tetrahedra"))
    faces = _surface_faces(tetrahedra)
    surface_vertices = np.unique(faces)

    stored_caps = _stored_attachment_caps(data, vertices, faces)
    if stored_caps is not None:
        # anchor_vertices is the mesh author's explicit attachment selection.
        # Preserve the complete endpoint caps instead of reducing them to the
        # nearest vertices of a sparse fiber sample.
        origin_seeds, insertion_seeds = stored_caps
        origin_points = vertices[np.asarray(origin_seeds, dtype=np.int32)]
        insertion_points = vertices[np.asarray(insertion_seeds, dtype=np.int32)]
        source = "saved anchor_vertices endpoint caps"
    else:
        origin_points, insertion_points, _, _ = _fiber_endpoint_clouds(data)
        source = "saved component fiber endpoints"
        if origin_points is None:
            origin_points, insertion_points = _region_endpoint_clouds(
                data, vertices, tetrahedra, surface_vertices)
            source = "tet region endpoint fallback"
        origin_seeds = _nearest_surface_cap(origin_points, vertices, faces)
        insertion_seeds = _nearest_surface_cap(
            insertion_points, vertices, faces)
    origin_fixed = _expand_surface_vertex_rings(
        origin_seeds, faces, rings=attachment_rings)
    insertion_fixed = _expand_surface_vertex_rings(
        insertion_seeds, faces, rings=attachment_rings)
    fixed = sorted(set(origin_fixed) | set(insertion_fixed))
    if not origin_fixed or not insertion_fixed:
        raise RuntimeError(f"{display_name}: failed to recover both outer caps")

    origin_distance, origin_mesh = _closest_bone(origin_points, bone_trees)
    insertion_distance, insertion_mesh = _closest_bone(insertion_points, bone_trees)
    origin_body, origin_body_name = _resolve_body(skel, origin_mesh)
    insertion_body, insertion_body_name = _resolve_body(skel, insertion_mesh)

    # Use the same harmonic construction described in EMU Sec. 3.6 rather
    # than a nearest-centerline approximation.  This is globally smooth even
    # where lateral and medial component centerlines meet.
    u = _harmonic_axis_coordinate(
        vertices, tetrahedra, origin_fixed, insertion_fixed)
    u[np.asarray(origin_fixed)] = 0.0
    u[np.asarray(insertion_fixed)] = 1.0
    bindings = _make_lbs_bindings(vertices, u, origin_body, insertion_body)

    print(f"[{name}] {len(vertices)} vertices, {len(tetrahedra)} tets")
    print(f"    Endpoint source: {source}")
    print(f"    Origin: {len(origin_seeds)} anchors + {attachment_rings} surface ring(s) "
          f"= {len(origin_fixed)} fixed -> {origin_mesh} -> "
          f"{origin_body_name} (mean distance {origin_distance:.6g} m)")
    print(f"    Insertion: {len(insertion_seeds)} anchors + {attachment_rings} surface ring(s) "
          f"= {len(insertion_fixed)} fixed -> {insertion_mesh} -> "
          f"{insertion_body_name} (mean distance {insertion_distance:.6g} m)")

    return {
        "name": name,
        "path": source_path,
        "vertices": vertices,
        "tetrahedra": tetrahedra,
        "surface_faces": faces,
        "fixed_vertices": fixed,
        "origin_fixed": origin_fixed,
        "insertion_fixed": insertion_fixed,
        "origin_mesh": origin_mesh,
        "insertion_mesh": insertion_mesh,
        "origin_body": origin_body_name,
        "insertion_body": insertion_body_name,
        "axis_coordinate": u,
        "lbs_bindings": bindings,
    }


def prepare_group(path: Path, skel, bone_trees):
    data = _load_saved_tet(path)
    stem = path.stem
    name = stem[:-4] if stem.endswith("_tet") else stem
    return prepare_group_data(data, name, skel, bone_trees, source_path=path)


def _set_pose(args, skel, bvh_info):
    skel.setPositions(np.zeros(skel.getNumDofs()))
    if args.bvh is None:
        print("Skeleton pose: Zygote rest pose")
        return
    t_frame = bake_emu._detect_bvh_tframe(str(args.bvh))
    motion = MyBVH(str(args.bvh), bvh_info, skel, T_frame=t_frame)
    if args.frame < 0 or args.frame >= len(motion.mocap_refs):
        raise IndexError(f"BVH frame {args.frame} is outside 0..{len(motion.mocap_refs)-1}")
    skel.setPositions(motion.mocap_refs[args.frame].copy())
    print(f"Skeleton pose: {args.bvh}, frame {args.frame}, T_frame={t_frame}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the deformation-space EMU solver headlessly on Zygote/DART")
    parser.add_argument("tet", nargs="+", type=Path,
                        help="Viewer-saved group tet file(s)")
    parser.add_argument("--bvh", type=Path, help="Optional BVH pose source")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("emu_test_output"))
    parser.add_argument("--youngs", type=float, default=6e6)
    parser.add_argument("--poisson", type=float, default=0.49)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--k-modes", type=int, default=8,
                        help="Woodbury modes (small default for a smoke test)")
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--gpu", action="store_true",
                        help="Use Taichi CUDA; CPU is the reliable headless default")
    args = parser.parse_args()

    for path in args.tet:
        if not path.is_file():
            parser.error(f"tet file not found: {path}")
    if args.bvh is not None and not args.bvh.is_file():
        parser.error(f"BVH file not found: {args.bvh}")

    print("Loading Zygote DART skeleton...")
    skel, bvh_info, _mesh_info = bake_emu.load_skeleton()
    skel.setPositions(np.zeros(skel.getNumDofs()))
    print(f"    {skel.getNumBodyNodes()} bodies, {skel.getNumDofs()} DOFs")
    print("Loading Zygote skeleton geometry for automatic attachment...")
    bone_trees = _load_bone_trees()
    print(f"    {len(bone_trees)} bone meshes")

    # Attachment/local coordinates must be made in the rest pose.
    groups = [prepare_group(path, skel, bone_trees) for path in args.tet]
    _set_pose(args, skel, bvh_info)

    mu, lam = bake_emu.lame_parameters(args.youngs, args.poisson)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for group in groups:
        print(f"[{group['name']}] EMU precompute...")
        precomp = bake_emu.precompute_emu(
            group["vertices"], group["tetrahedra"], group["fixed_vertices"],
            group["axis_coordinate"], k_modes=args.k_modes)
        posed = bake_emu.compute_rigid_blend_positions(
            group["lbs_bindings"], skel, group["axis_coordinate"])
        fixed_mask = np.zeros(len(group["vertices"]), dtype=bool)
        fixed_mask[group["fixed_vertices"]] = True

        print(f"[{group['name']}] EMU solve ({'GPU' if args.gpu else 'CPU'})...")
        positions, info = bake_emu.emu_solve(
            posed, precomp, fixed_mask, posed, mu, lam, args.alpha,
            max_iters=args.max_iters, verbose=True, use_gpu=args.gpu)
        finite = bool(np.all(np.isfinite(positions)))
        fixed_error = float(np.max(np.linalg.norm(
            positions[group["fixed_vertices"]] - posed[group["fixed_vertices"]], axis=1)))
        print(f"    finite={finite}, fixed_error={fixed_error:.3e} m, "
              f"energy={info['energy']:.6e}, iterations={info['iterations']}")
        failures += int(not finite or fixed_error > 1e-7)

        output = args.output_dir / f"{group['name']}_frame_{args.frame:04d}.npz"
        np.savez_compressed(
            output,
            rest_positions=group["vertices"].astype(np.float32),
            posed_targets=posed.astype(np.float32),
            positions=positions.astype(np.float32),
            tetrahedra=group["tetrahedra"],
            surface_faces=group["surface_faces"],
            fixed_vertices=np.asarray(group["fixed_vertices"], dtype=np.int32),
            origin_fixed=np.asarray(group["origin_fixed"], dtype=np.int32),
            insertion_fixed=np.asarray(group["insertion_fixed"], dtype=np.int32),
            origin_mesh=np.asarray(group["origin_mesh"]),
            insertion_mesh=np.asarray(group["insertion_mesh"]),
            origin_body=np.asarray(group["origin_body"]),
            insertion_body=np.asarray(group["insertion_body"]),
            energy=np.asarray(info["energy"]),
            grad_norm=np.asarray(info["grad_norm"]),
            iterations=np.asarray(info["iterations"]),
        )
        print(f"    Saved {output}")

    if failures:
        print(f"FAILED: {failures}/{len(groups)} result(s) failed validation")
        return 1
    print(f"PASS: {len(groups)} headless EMU result(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
