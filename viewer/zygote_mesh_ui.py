"""Zygote mesh UI and mesh-related functions — extracted from viewer.py.

Free functions that take the viewer instance (v) as first argument.
"""
try:
    import imgui
except ImportError:
    pass  # Headless mode — imgui not needed for simulation functions
import numpy as np
import os
import traceback
import glob
import trimesh
import json
import time
import copy
import pickle
from functools import lru_cache
try:
    import glfw
except ImportError:
    pass  # Headless mode

try:
    from OpenGL.GL import *
    import viewer.gl_function as mygl
except ImportError:
    pass  # Headless mode
from viewer.mesh_loader import MeshLoader
from viewer.muscle_mesh import COLOR_MAP, cotangent_weight_matrix
from viewer.tetrahedron_mesh import _derive_tet_region_labels
from viewer.arap_backends import get_backend
from core.bvhparser import MyBVH

# UI dimension constants (mirrored from viewer.py to avoid circular imports)
wide_button_width = 308
button_width = 150
MIN_EYE_DISTANCE = 0.5
ZYGOTE_TENDON_COLOR = np.array([1.0, 0.82, 0.78], dtype=np.float32)
# Benchmarked on the Rectus Femoris group.  With the isotropic voxel boundary
# this request produces about 30k TetGen elements after anatomical projection:
# the best tested surface-fidelity, stability, and runtime compromise.
RECTUS_FEMORIS_BASELINE_TETS = 20000


def _clip_polygon_to_half_plane(polygon, normal, offset, eps=1e-12):
    """Clip a convex 2D polygon to ``dot(normal, p) <= offset``."""
    if len(polygon) == 0:
        return polygon

    clipped = []
    previous = np.asarray(polygon[-1], dtype=np.float64)
    previous_value = float(np.dot(normal, previous) - offset)
    previous_inside = previous_value <= eps

    for current_raw in polygon:
        current = np.asarray(current_raw, dtype=np.float64)
        current_value = float(np.dot(normal, current) - offset)
        current_inside = current_value <= eps

        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > eps:
                t = previous_value / denominator
                clipped.append(previous + t * (current - previous))
        if current_inside:
            clipped.append(current)

        previous = current
        previous_value = current_value
        previous_inside = current_inside

    return clipped


@lru_cache(maxsize=64)
def _unit_square_voronoi_cells_cached(sample_key):
    """Return Voronoi cells clipped to [0, 1]^2 for a hashable sample key.

    Intersecting pairwise nearest-site half-planes avoids the unbounded-cell
    handling required by scipy.spatial.Voronoi and also handles a single site.
    The inspector cache makes the O(N^3) construction a one-time cost whenever
    the fiber samples change.
    """
    samples = np.asarray(sample_key, dtype=np.float64)
    if samples.size == 0:
        return ()
    samples = samples.reshape((-1, 2))
    cells = []

    for i, site in enumerate(samples):
        polygon = [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([1.0, 1.0]),
            np.array([0.0, 1.0]),
        ]
        for j, other in enumerate(samples):
            if i == j:
                continue
            normal = 2.0 * (other - site)
            if np.dot(normal, normal) < 1e-24:
                continue
            offset = float(np.dot(other, other) - np.dot(site, site))
            polygon = _clip_polygon_to_half_plane(polygon, normal, offset)
            if not polygon:
                break
        cells.append(tuple((float(p[0]), float(p[1])) for p in polygon))

    return tuple(cells)


def _unit_square_voronoi_cells(samples):
    """Compute stable, cached unit-square Voronoi cells for fiber samples."""
    points = np.asarray(samples, dtype=np.float64)
    if points.size == 0:
        return ()
    points = points.reshape((-1, points.shape[-1]))[:, :2]
    # Rounded coordinates prevent insignificant reconstruction noise from
    # defeating the render cache while retaining far more precision than UI
    # sampling needs.
    key = tuple((round(float(p[0]), 12), round(float(p[1]), 12)) for p in points)
    return _unit_square_voronoi_cells_cached(key)


def _map_voronoi_cells_to_contour(obj, plane_info, samples, cells):
    """Map labeled unit-square Voronoi cells with the waypoint MVC chart.

    Each returned cell contains the mapped fiber site first, followed by its
    mapped boundary vertices.  Keeping the site lets the renderer use the same
    triangle fan in parameter and contour space and expose local foldovers.
    """
    if not cells or plane_info is None:
        return ()

    sites = np.asarray(samples, dtype=np.float64)
    if sites.size == 0:
        return ()
    sites = sites.reshape((-1, sites.shape[-1]))[:, :2]

    query_points = []
    cell_sizes = []
    for site, cell in zip(sites, cells):
        if len(cell) < 3:
            cell_sizes.append(0)
            continue
        query_points.append(site)
        query_points.extend(cell)
        cell_sizes.append(1 + len(cell))
    if not query_points:
        return ()

    query = np.asarray(query_points, dtype=np.float64)
    contour_match = plane_info.get('contour_match', ())
    try:
        chart_values = np.asarray(
            [np.asarray(value, dtype=np.float64)
             for pair in contour_match for value in pair], dtype=np.float64)
        bp_values = np.asarray(
            plane_info.get('bounding_plane', ()), dtype=np.float64)
        chart_signature = (
            chart_values.shape, hash(chart_values.tobytes()),
            bp_values.shape, hash(bp_values.tobytes()),
        )
    except (TypeError, ValueError):
        chart_signature = (len(contour_match), id(contour_match))
    cache_key = (
        hash(query.tobytes()), query.shape, chart_signature,
        getattr(obj, 'fiber_positioning_method', 'mvc'),
    )

    cache = getattr(obj, '_inspect_voronoi_transfer_cache', None)
    if cache is None:
        cache = {}
        obj._inspect_voronoi_transfer_cache = cache
    mapped = cache.get(cache_key)
    if mapped is None:
        # find_waypoints is the P<-Q MVC map used by the existing grid-fiber
        # pipeline.  Calling it for all cell vertices in one batch keeps the
        # displayed transfer consistent with the red waypoint locations.
        _, mapped, _ = obj.find_waypoints(plane_info, query)
        mapped = np.asarray(mapped, dtype=np.float64)

        # A Q polygon can contain two adjacent samples at effectively the same
        # unit-square corner (especially in older/closest-edge contour_match
        # data).  Generic MVC then selects the first matching Q entry, which
        # can be one contour index away from the explicit corner correspondence
        # shown by Inspect 2D.  Unit-square corners are not ambiguous: map them
        # directly through the stored corner_indices.
        corner_indices = plane_info.get('corner_indices')
        if corner_indices is not None and len(corner_indices) >= 4:
            unit_corners = np.array([
                [0.0, 0.0], [1.0, 0.0],
                [1.0, 1.0], [0.0, 1.0],
            ], dtype=np.float64)
            for query_i, uv in enumerate(query):
                distances = np.linalg.norm(unit_corners - uv, axis=1)
                corner_i = int(np.argmin(distances))
                if distances[corner_i] > 1e-10:
                    continue
                contour_i = int(corner_indices[corner_i])
                if 0 <= contour_i < len(contour_match):
                    mapped[query_i] = np.asarray(
                        contour_match[contour_i][0], dtype=np.float64)

        if len(cache) >= 64:
            cache.pop(next(iter(cache)))
        cache[cache_key] = mapped

    mapped_cells = []
    offset = 0
    for size in cell_sizes:
        if size == 0:
            mapped_cells.append(None)
            continue
        mapped_cells.append(mapped[offset:offset + size])
        offset += size
    return tuple(mapped_cells)


def _is_zygote_tendon_mesh(name, path=None):
    """Classify Zygote tendon OBJs by filename/display name."""
    tokens = [str(name or '')]
    if path:
        tokens.append(os.path.basename(str(path)))
    return any('tendon' in token.lower() for token in tokens)


def _is_pennation_path(path):
    """Whether an OBJ belongs to a Pennation multi-part muscle group."""
    return 'pennation' in str(path or '').lower()


def _tendon_match_keys(muscle_name):
    name = str(muscle_name or '').lower()
    keys = [name]
    parts = name.split('_')
    if len(parts) > 1 and parts[-1] == 'belly':
        parts = parts[:-1]
        keys.append('_'.join(parts))
    if len(parts) > 1 and parts[-1] in ('m', 'l', 'medial', 'lateral'):
        keys.append('_'.join(parts[:-1]))
    return [k for k in dict.fromkeys(keys) if len(k) >= 4]


def _zygote_component_group_name(name):
    parts = str(name or '').split('_')
    if len(parts) >= 2 and parts[-1].lower() == 'tendon':
        parts = parts[:-1]
        if parts and parts[-1].lower() in ('origin', 'insertion'):
            parts = parts[:-1]
    if len(parts) > 1 and parts[-1].lower() == 'belly':
        parts = parts[:-1]
    if len(parts) > 1 and parts[-1].lower() in ('m', 'l', 'medial', 'lateral'):
        parts = parts[:-1]
    return '_'.join(parts) if parts else str(name or '')


def _leading_side_token(name):
    parts = str(name or '').lower().split('_')
    return parts[0] if parts and parts[0] in ('l', 'r', 'm') else None


def _is_component_belly_name(name):
    parts = str(name or '').lower().split('_')
    if not parts:
        return False
    if parts[-1] == 'belly':
        return True
    return parts[-1] in ('m', 'l', 'medial', 'lateral')


def _find_tendon_extension_candidate(v, muscle_name, role):
    role = role.lower()
    keys = _tendon_match_keys(muscle_name)
    muscle_side = _leading_side_token(muscle_name)
    _, muscle_comp = _zygote_component_role(muscle_name)
    matches = []
    all_role_tendons = []
    muscle_obj = getattr(v, 'zygote_muscle_meshes', {}).get(muscle_name)
    muscle_dir = None
    if muscle_obj is not None and getattr(muscle_obj, 'obj', None):
        muscle_dir = os.path.dirname(os.path.abspath(str(muscle_obj.obj)))
    for candidate, candidate_obj in getattr(v, 'zygote_muscle_meshes', {}).items():
        cand = candidate.lower()
        if candidate == muscle_name:
            continue
        if 'tendon' not in cand or role not in cand:
            continue
        cand_side = _leading_side_token(candidate)
        if muscle_side is not None and cand_side is not None and cand_side != muscle_side:
            continue
        _, cand_comp = _zygote_component_role(candidate)
        if muscle_comp is not None and cand_comp is not None and cand_comp != muscle_comp:
            continue
        all_role_tendons.append(candidate)
        score = None
        for key_i, key in enumerate(keys):
            if key in cand:
                # Full-name match wins over stripped-name match.
                score = key_i
                break
        if score is None and muscle_dir is not None and getattr(candidate_obj, 'obj', None):
            cand_dir = os.path.dirname(os.path.abspath(str(candidate_obj.obj)))
            if cand_dir == muscle_dir:
                # Generic names like Origin_Tendon are safe when colocated
                # with the belly OBJ in the same component folder.
                score = len(keys)
        if score is not None:
            matches.append((score, len(candidate), candidate))
    if matches:
        matches.sort()
        return matches[0][2]
    if len(all_role_tendons) == 1:
        # Last-resort convenience for a loaded one-muscle workspace.
        return all_role_tendons[0]
    return ''


def _auto_fill_tendon_extension_names(v, muscle_name, obj):
    if not _is_component_belly_name(muscle_name):
        return
    if not getattr(obj, 'origin_tendon_extension_name', ''):
        obj.origin_tendon_extension_name = _find_tendon_extension_candidate(
            v, muscle_name, 'origin')
    if not getattr(obj, 'insertion_tendon_extension_name', ''):
        obj.insertion_tendon_extension_name = _find_tendon_extension_candidate(
            v, muscle_name, 'insertion')


def _zygote_loaded_component_groups(v):
    groups = {}
    for name in getattr(v, 'zygote_muscle_meshes', {}).keys():
        groups.setdefault(_zygote_component_group_name(name), []).append(name)
    for group_name in groups:
        groups[group_name].sort()
    return dict(sorted(groups.items()))


def _zygote_component_role(name):
    low = str(name or '').lower()
    parts = low.split('_')
    if 'origin' in low and 'tendon' in low:
        kind = 'origin_tendon'
    elif 'insertion' in low and 'tendon' in low:
        kind = 'insertion_tendon'
    else:
        kind = 'belly'
    comp = None
    role_tokens = {'origin', 'insertion', 'tendon', 'belly'}
    for token in reversed(parts):
        if token in role_tokens:
            continue
        if token in ('m', 'l', 'medial', 'lateral'):
            comp = 'm' if token == 'medial' else 'l' if token == 'lateral' else token
            break
    if comp is None and len(parts) > 1 and parts[-1] in ('m', 'l', 'medial', 'lateral'):
        comp = parts[-1]
    if comp is None and 'medial' in parts:
        comp = 'm'
    elif comp is None and 'lateral' in parts:
        comp = 'l'
    elif comp is None and len(parts) > 0 and parts[0] in ('m', 'l'):
        comp = parts[0]
    return kind, comp


def _configure_zygote_component_group(v, group_name):
    names = _zygote_loaded_component_groups(v).get(group_name, [])
    if not names:
        return
    bellies, tendons = _zygote_group_component_tables(v, group_name)

    # Auto-link medial/lateral bellies and matching medial/lateral tendons.
    for table in (bellies, tendons['origin_tendon'], tendons['insertion_tendon']):
        if 'm' in table and 'l' in table:
            a = v.zygote_muscle_meshes[table['m']]
            b = v.zygote_muscle_meshes[table['l']]
            a.linked_counterpart_name = table['l']
            b.linked_counterpart_name = table['m']
            a.linked_drive_counterpart = True
            b.linked_drive_counterpart = True
            a.linked_use_shared_scalar = True
            b.linked_use_shared_scalar = True

    for comp_key, belly_name in bellies.items():
        belly = v.zygote_muscle_meshes[belly_name]
        origin = tendons['origin_tendon'].get(comp_key) or tendons['origin_tendon'].get('main')
        insertion = tendons['insertion_tendon'].get(comp_key) or tendons['insertion_tendon'].get('main')
        if origin:
            belly.origin_tendon_extension_name = origin
        if insertion:
            belly.insertion_tendon_extension_name = insertion
    print(f"[{group_name}] Configured component group: {len(names)} components")


def _zygote_group_component_tables(v, group_name):
    names = _zygote_loaded_component_groups(v).get(group_name, [])
    bellies = {}
    tendons = {'origin_tendon': {}, 'insertion_tendon': {}}
    for name in names:
        kind, comp = _zygote_component_role(name)
        comp_key = comp or 'main'
        if kind == 'belly':
            bellies[comp_key] = name
        else:
            tendons[kind][comp_key] = name
    return bellies, tendons


def _component_process_names(table, master_comp=None):
    if 'm' in table and 'l' in table:
        key = master_comp if master_comp in table else 'm'
        return [table[key]]
    return list(dict.fromkeys(table.values()))


def _run_component_pipeline_quick(v, name, obj, max_step, defer=False):
    if max_step >= 1 and len(getattr(obj, 'edge_groups', []) or []) > 0 and len(getattr(obj, 'edge_classes', []) or []) > 0:
        has_counterpart = bool(getattr(obj, 'linked_counterpart_name', ''))
        if (has_counterpart
                and getattr(obj, 'linked_drive_counterpart', False)
                and getattr(obj, 'linked_use_shared_scalar', True)):
            _ensure_counterpart_scalar(v, name, obj, defer=defer)
        else:
            obj.compute_scalar_field(defer=defer)
            if has_counterpart and getattr(obj, 'linked_drive_counterpart', False):
                _ensure_counterpart_scalar(v, name, obj, defer=defer)

    if max_step >= 2 and obj.scalar_field is not None:
        obj.find_contours(skeleton_meshes=v.zygote_skeleton_meshes,
                          spacing_scale=obj.contour_spacing_scale, defer=defer)
        if getattr(obj, 'linked_drive_counterpart', False):
            _apply_master_contour_schedule_to_counterpart(
                v, name, obj, defer=defer, label="group contours")
        if not defer:
            obj.is_draw_bounding_box = True

    if max_step >= 3 and obj.contours is not None and len(obj.contours) > 0:
        obj.refine_contours(max_spacing_threshold=0.01, defer=defer)
        if getattr(obj, 'linked_drive_counterpart', False):
            _apply_master_contour_schedule_to_counterpart(
                v, name, obj, defer=defer, label="group gap-filled contours")

    if max_step >= 4 and obj.scalar_field is not None:
        field_min = float(obj.scalar_field.min())
        field_max = float(obj.scalar_field.max())
        scalar_min, scalar_max = field_min, field_max
        exp_origin = len(obj.contours[0]) if obj.contours and len(obj.contours) > 0 else None
        exp_insertion = len(obj.contours[-1]) if obj.contours and len(obj.contours) > 0 else None
        obj.find_all_transitions(scalar_min=scalar_min, scalar_max=scalar_max, num_samples=200,
                                 expected_origin=exp_origin, expected_insertion=exp_insertion)
        if obj.contours is not None and len(obj.contours) > 0:
            obj.add_transitions_to_contours(defer=defer)
        if getattr(obj, 'linked_drive_counterpart', False):
            _apply_master_contour_schedule_to_counterpart(
                v, name, obj, defer=defer, label="group transition contours")

    if max_step >= 5 and obj.contours is not None and len(obj.contours) > 0:
        obj.smoothen_all(defer=defer)
        _run_counterpart_step(v, name, obj, 5, defer=defer)

    if max_step >= 6 and obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None:
        obj.cut_streams_animated(defer=defer, cut_method=obj.cutting_method, muscle_name=name)
        _run_counterpart_step(v, name, obj, 6, defer=defer)
        if getattr(obj, '_manual_cut_pending', False) or getattr(obj, '_manual_cut_data', None) is not None:
            print(f"[{name}] Group process paused: manual cut pending")
            return False

    if max_step >= 7 and getattr(obj, 'stream_contours', None) is not None:
        obj.stream_smoothen_all(defer=defer)
        _run_counterpart_step(v, name, obj, 7, defer=defer)

    if max_step >= 8 and getattr(obj, 'stream_contours', None) is not None:
        obj.select_levels()
        if getattr(obj, '_level_select_window_open', False):
            obj._process_step = int(max_step)
            obj._pipeline_paused_at = 9 if int(max_step) >= 9 else None
            print(f"[{name}] Group process paused: choose contours in Level Select window")
            return False
        _run_counterpart_step(v, name, obj, 8, defer=defer)

    if max_step >= 9 and getattr(obj, 'stream_contours', None) is not None:
        _ensure_level_selection_applied(v, name, obj, defer=defer)
        obj._belly_waypoints_before_tendon_extension = None
        obj.build_fibers(skeleton_meshes=v.zygote_skeleton_meshes, defer=defer)
        if (getattr(obj, 'enable_tendon_extension', True)
                and (getattr(obj, 'origin_tendon_extension_name', '')
                     or getattr(obj, 'insertion_tendon_extension_name', ''))):
            _extend_belly_fibers_with_tendons(v, name, obj)
        _run_counterpart_step(v, name, obj, 9, defer=defer)

    if max_step >= 10 and obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None:
        _resample_contours_with_links(v, name, obj, defer=defer)
        _resample_linked_tendon_extensions(v, name, obj, defer=defer)

    if max_step >= 11 and obj.contours is not None and len(obj.contours) > 0 and obj.draw_contour_stream is not None:
        if _prepare_owned_connected_contour_mesh_source(v, name, obj):
            obj.build_contour_mesh(defer=defer)
            if not _connected_source_has_linked_components(obj):
                _run_counterpart_step(v, name, obj, 11, defer=defer)

    if max_step >= 12:
        if _skip_non_owner_connected_mesh(v, name, obj, "Tetrahedralize"):
            return True
        tet_ok = _tetrahedralize_single_contour_mesh(v, name, obj, defer=defer)
        if tet_ok and not _connected_source_has_linked_components(obj):
            _run_counterpart_step(v, name, obj, 12, defer=defer)
        if tet_ok and obj.tet_vertices is not None and not defer:
            obj.is_draw_contours = False
            obj.is_draw_tet_mesh = True
            obj._tetrahedralize_replayed = True
        if not tet_ok:
            return False
    return True


def _process_zygote_component_group(v, group_name, max_step, tendon_step=None,
                                    master_comp=None, process_tendons_full=False,
                                    extend_fibers=True, auto_wire=True, defer=False):
    if auto_wire:
        _configure_zygote_component_group(v, group_name)
    bellies, tendons = _zygote_group_component_tables(v, group_name)

    if master_comp is None:
        master_comp = 'm' if 'm' in bellies else 'main' if 'main' in bellies else None
    tendon_step = int(max_step) if process_tendons_full else min(
        int(max_step), int(tendon_step if tendon_step is not None else 5))
    for table in (tendons['origin_tendon'], tendons['insertion_tendon']):
        for comp_name in _component_process_names(table, master_comp=master_comp):
            obj = v.zygote_muscle_meshes.get(comp_name)
            if obj is not None:
                print(f"[{group_name}] Processing tendon component {comp_name} to step {tendon_step}")
                _run_component_pipeline_quick(v, comp_name, obj, tendon_step, defer=defer)

    for comp_name in _component_process_names(bellies, master_comp=master_comp):
        obj = v.zygote_muscle_meshes.get(comp_name)
        if obj is not None:
            obj.enable_tendon_extension = bool(extend_fibers)
            print(f"[{group_name}] Processing belly component {comp_name} to step {max_step}")
            _run_component_pipeline_quick(v, comp_name, obj, int(max_step), defer=defer)


def _zygote_group_state(v, group_name):
    if not hasattr(v, 'zygote_muscle_group_state'):
        v.zygote_muscle_group_state = {}
    state = v.zygote_muscle_group_state.setdefault(group_name, {})
    state.setdefault('show_parts', False)
    state.setdefault('belly_step', 9)
    state.setdefault('tendon_step', 8)
    state.setdefault('draw_parts', True)
    state.setdefault('draw_surface', True)
    state.setdefault('draw_surface_tet', True)
    state.setdefault('draw_surface_tet_edges', False)
    state.setdefault('draw_surface_tet_internals', False)
    state.setdefault('draw_scalar_field', False)
    state.setdefault('draw_contours', False)
    state.setdefault('draw_fibers', False)
    state.setdefault('draw_bounding_boxes', False)
    state.setdefault('use_emu_volume_target', True)
    return state


def _zygote_group_loaded_names(v, group_name):
    names = [
        name for name in getattr(v, 'zygote_muscle_meshes', {}).keys()
        if _zygote_component_group_name(name) == group_name
    ]
    return sorted(names)


def _zygote_loaded_ui_group_names(v):
    groups = _zygote_loaded_component_groups(v)
    out = []
    for group_name, names in groups.items():
        # Filename suffixes alone do not make a group. Only meshes loaded from
        # a Pennation directory get the grouped pipeline UI; ordinary muscles
        # remain in the regular single-muscle controls.
        is_pennation = any(
            _is_pennation_path(getattr(v.zygote_muscle_meshes.get(name), 'obj', ''))
            for name in names)
        if is_pennation:
            out.append(group_name)
    return sorted(out)


def _zygote_group_name_side(name):
    _, comp = _zygote_component_role(name)
    return comp


def _zygote_group_auto_mapping(v, group_name):
    names = _zygote_group_loaded_names(v, group_name)
    mapping = {
        'surface': group_name if group_name in getattr(v, 'zygote_muscle_meshes', {}) else '',
        'master': '',
        'follower': '',
        'master_origin': '',
        'master_insertion': '',
        'follower_origin': '',
        'follower_insertion': '',
    }

    for name in names:
        if name == group_name:
            continue
        kind, side = _zygote_component_role(name)
        if kind == 'belly':
            if side == 'l' and not mapping['master']:
                mapping['master'] = name
            elif side == 'm' and not mapping['follower']:
                mapping['follower'] = name
        elif kind == 'origin_tendon':
            if side == 'l' and not mapping['master_origin']:
                mapping['master_origin'] = name
            elif side == 'm' and not mapping['follower_origin']:
                mapping['follower_origin'] = name
        elif kind == 'insertion_tendon':
            if side == 'l' and not mapping['master_insertion']:
                mapping['master_insertion'] = name
            elif side == 'm' and not mapping['follower_insertion']:
                mapping['follower_insertion'] = name

    # Fall back to any available belly/tendon if only one side exists.
    for name in names:
        if name == group_name:
            continue
        kind, side = _zygote_component_role(name)
        if kind == 'belly':
            mapping['master'] = mapping['master'] or name
        elif kind == 'origin_tendon':
            if side == _zygote_group_name_side(mapping.get('master', '')):
                mapping['master_origin'] = mapping['master_origin'] or name
            elif side == _zygote_group_name_side(mapping.get('follower', '')):
                mapping['follower_origin'] = mapping['follower_origin'] or name
            else:
                mapping['master_origin'] = mapping['master_origin'] or name
        elif kind == 'insertion_tendon':
            if side == _zygote_group_name_side(mapping.get('master', '')):
                mapping['master_insertion'] = mapping['master_insertion'] or name
            elif side == _zygote_group_name_side(mapping.get('follower', '')):
                mapping['follower_insertion'] = mapping['follower_insertion'] or name
            else:
                mapping['master_insertion'] = mapping['master_insertion'] or name
    return mapping


def _ensure_zygote_group_mapping(v, group_name):
    state = _zygote_group_state(v, group_name)
    auto = _zygote_group_auto_mapping(v, group_name)
    for key, val in auto.items():
        if not state.get(key) or state.get(key) not in getattr(v, 'zygote_muscle_meshes', {}):
            state[key] = val
    return state


def _zygote_group_candidates(v, group_name, key):
    names = _zygote_group_loaded_names(v, group_name)
    candidates = ['None']
    for name in names:
        kind, side = _zygote_component_role(name)
        if key == 'surface':
            ok = name == group_name
        elif key in ('master', 'follower'):
            ok = name != group_name and kind == 'belly'
        elif key.endswith('_origin'):
            ok = kind == 'origin_tendon'
        elif key.endswith('_insertion'):
            ok = kind == 'insertion_tendon'
        else:
            ok = True
        if ok:
            candidates.append(name)
    return candidates


def _draw_group_combo(v, group_name, key, label):
    state = _ensure_zygote_group_mapping(v, group_name)
    candidates = _zygote_group_candidates(v, group_name, key)
    current = state.get(key, '')
    try:
        idx = candidates.index(current if current else 'None')
    except ValueError:
        idx = 0
    changed, new_idx = imgui.combo(f"{label}##{group_name}_{key}", idx, candidates)
    if changed:
        state[key] = '' if candidates[new_idx] == 'None' else candidates[new_idx]
    return changed


def _apply_zygote_group_links(v, group_name):
    state = _ensure_zygote_group_mapping(v, group_name)
    master_name = state.get('master', '')
    follower_name = state.get('follower', '')
    master = v.zygote_muscle_meshes.get(master_name)
    follower = v.zygote_muscle_meshes.get(follower_name)
    if master is not None and follower is not None:
        master.linked_drive_counterpart = True
        master.linked_use_shared_scalar = True
        follower.linked_drive_counterpart = True
        follower.linked_use_shared_scalar = True
        _set_symmetric_counterpart_link(v, master_name, follower_name)

    for a_key, b_key in (
            ('master_origin', 'follower_origin'),
            ('master_insertion', 'follower_insertion')):
        a_name = state.get(a_key, '')
        b_name = state.get(b_key, '')
        a_obj = v.zygote_muscle_meshes.get(a_name)
        b_obj = v.zygote_muscle_meshes.get(b_name)
        if a_obj is None or b_obj is None:
            continue
        a_obj.linked_drive_counterpart = True
        a_obj.linked_use_shared_scalar = True
        b_obj.linked_drive_counterpart = True
        b_obj.linked_use_shared_scalar = True
        _set_symmetric_counterpart_link(v, a_name, b_name)

    for prefix, belly_name in (('master', master_name), ('follower', follower_name)):
        belly = v.zygote_muscle_meshes.get(belly_name)
        if belly is None:
            continue
        belly.enable_tendon_extension = True
        belly.origin_tendon_extension_name = state.get(f'{prefix}_origin', '') or ''
        belly.insertion_tendon_extension_name = state.get(f'{prefix}_insertion', '') or ''


def _zygote_group_guide_names(v, group_name):
    state = _ensure_zygote_group_mapping(v, group_name)
    keys = [
        'master', 'follower',
        'master_origin', 'master_insertion',
        'follower_origin', 'follower_insertion',
    ]
    out = []
    for key in keys:
        name = state.get(key, '')
        if name and name in getattr(v, 'zygote_muscle_meshes', {}) and name not in out:
            out.append(name)
    return out


def _invalidate_tet_draw_cache(obj):
    for attr in (
            '_tet_surface_verts', '_tet_surface_normals', '_tet_cap_verts',
            '_tet_cap_normals', '_tet_edge_verts', '_tet_surface_vidx',
            '_tet_cap_vidx', '_tet_edge_vidx', '_tet_internal_verts',
            '_tet_internal_normals', '_tet_internal_colors',
            '_tet_internal_vidx', '_tet_internal_stride_cached',
            '_tet_region_surface_colors', '_tet_region_cap_colors'):
        if hasattr(obj, attr):
            setattr(obj, attr, None)
    if hasattr(obj, '_tet_edge_source'):
        obj._tet_edge_source = None


def _ensure_zygote_group_render_embedding(group_name, surface):
    """Upgrade an older saved group tet with the exact original render skin."""
    required = (
        'tet_render_vertices_rest', 'tet_render_vertex_indices',
        'tet_render_vertex_weights', 'tet_render_tet_rest_vertices')
    if all(getattr(surface, field, None) is not None for field in required):
        return False
    if (getattr(surface, 'tet_vertices', None) is None or
            getattr(surface, 'tet_tetrahedra', None) is None):
        return False
    anatomical_vertices, anatomical_faces = _component_original_surface(surface)
    if (anatomical_vertices is None or anatomical_faces is None or
            len(anatomical_faces) == 0):
        return False

    import pyvista as pv
    tet_vertices = np.asarray(surface.tet_vertices, dtype=np.float64)
    tetrahedra = np.asarray(surface.tet_tetrahedra, dtype=np.int32)
    anatomical_vertices = np.asarray(anatomical_vertices, dtype=np.float64)
    vtk_cells = np.hstack((
        np.full((len(tetrahedra), 1), 4, dtype=np.int64),
        tetrahedra.astype(np.int64))).ravel()
    grid = pv.UnstructuredGrid(
        vtk_cells,
        np.full(len(tetrahedra), pv.CellType.TETRA, dtype=np.uint8),
        tet_vertices)
    containing = np.asarray(
        grid.find_containing_cell(anatomical_vertices), dtype=np.int64)
    inside = containing >= 0
    cells = containing.copy()
    if np.any(~inside):
        cells[~inside] = np.asarray(
            grid.find_closest_cell(anatomical_vertices[~inside]),
            dtype=np.int64)
    skin_indices = tetrahedra[cells]
    x = tet_vertices[skin_indices]
    dm = np.stack((x[:, 0] - x[:, 3], x[:, 1] - x[:, 3],
                   x[:, 2] - x[:, 3]), axis=-1)
    rhs = anatomical_vertices - x[:, 3]
    bary012 = np.linalg.solve(dm, rhs[..., None])[..., 0]
    weights = np.empty((len(anatomical_vertices), 4), dtype=np.float64)
    weights[:, :3] = bary012
    weights[:, 3] = 1.0 - np.sum(bary012, axis=1)
    inside_fraction = float(np.mean(inside))
    max_abs_weight = float(np.max(np.abs(weights)))
    if (inside_fraction < 0.10 or not np.isfinite(max_abs_weight) or
            max_abs_weight > 100.0):
        print(f"[{group_name}] Saved-tet anatomical skin upgrade skipped: "
              f"coordinate spaces do not match (inside={inside_fraction:.1%}, "
              f"max |bary|={max_abs_weight:.3g})")
        return False

    surface.tet_render_vertices_rest = anatomical_vertices.copy()
    surface.tet_render_faces = np.asarray(anatomical_faces, dtype=np.int32)
    surface.tet_faces = surface.tet_render_faces
    surface.tet_render_vertex_indices = skin_indices.astype(np.int32)
    surface.tet_render_vertex_weights = weights
    surface.tet_render_tet_rest_vertices = tet_vertices.copy()
    surface.tet_cap_face_indices = []
    surface.tet_surface_face_count = len(surface.tet_render_faces)
    _invalidate_tet_draw_cache(surface)
    print(f"[{group_name}] Upgraded saved tet with anatomical skin: "
          f"{int(np.sum(inside))}/{len(inside)} embedded inside, "
          f"bary min={float(np.min(weights)):.3g}")
    return True


def _relabel_zygote_group_tet_materials(v, group_name, surface):
    """Assign belly/tendon labels to both simulation tets and render skin."""
    guide_names = _zygote_group_guide_names(v, group_name)
    if not guide_names:
        return False
    changed = False
    if (getattr(surface, 'tet_vertices', None) is not None and
            getattr(surface, 'tet_tetrahedra', None) is not None):
        try:
            labels = _classify_tets_by_component_containment(
                v, guide_names, surface.tet_vertices, surface.tet_tetrahedra)
            (surface.tet_vertex_regions,
             surface.tet_region_labels,
             surface.tet_component_labels,
             surface.tet_region_mixed) = labels
            changed = True
            print(f"[{group_name}] Relabeled surface tet from "
                  f"{len(guide_names)} guide component(s): "
                  f"{_count_simple_labels(surface.tet_region_labels)}")
        except Exception as exc:
            print(f"[{group_name}] Guide tet labeling failed: {exc}")

    # The anatomical render skin has independent vertices, so volume labels
    # cannot color it by tet index. Transfer the nearest guide component's
    # part explicitly (belly/origin tendon/insertion tendon).
    render_rest = getattr(surface, 'tet_render_vertices_rest', None)
    if render_rest is not None:
        try:
            from scipy.spatial import cKDTree
            guide_points = []
            guide_regions = []
            for guide_name in guide_names:
                guide = v.zygote_muscle_meshes.get(guide_name)
                guide_vertices, _ = _component_original_surface(guide)
                if guide_vertices is None:
                    continue
                region = {
                    'part': _mesh_part_from_name(guide_name),
                    'component': guide_name,
                }
                guide_points.extend(np.asarray(guide_vertices).tolist())
                guide_regions.extend(
                    [region.copy() for _ in range(len(guide_vertices))])
            if guide_points:
                _, nearest = cKDTree(np.asarray(guide_points)).query(
                    np.asarray(render_rest), k=1)
                surface.tet_render_vertex_regions = [
                    guide_regions[int(i)] for i in nearest]
                changed = True
                print(f"[{group_name}] Anatomical skin regions: "
                      f"{_count_simple_labels([r['part'] for r in surface.tet_render_vertex_regions])}")
        except Exception as exc:
            print(f"[{group_name}] Anatomical skin region transfer failed: {exc}")
    if changed:
        surface._tet_surface_colors = None
        _invalidate_tet_draw_cache(surface)
    return changed


def _tetrahedralize_zygote_group_surface(v, group_name, defer=False):
    state = _ensure_zygote_group_mapping(v, group_name)
    surface_name = state.get('surface', '')
    surface = v.zygote_muscle_meshes.get(surface_name)
    if surface is None:
        print(f"[{group_name}] Select an uncut surface mesh before tetrahedralizing")
        return False

    _apply_zygote_group_links(v, group_name)
    # Group volumes are EMU domains, not open contour pieces. Always use the
    # closed boundary-remesh path for them, regardless of the source component
    # UI setting; individual component processing may still preserve its own
    # open surface when requested.
    surface.enable_tet_boundary_remesh = True
    surface.preserve_tet_surface = False
    surface._group_closed_default = True
    print(f"[{group_name}] Group tetrahedralization: closed domain default")
    ok = _tetrahedralize_original_surface_for_sim(v, surface_name, surface, defer=defer)
    if not ok or surface.tet_vertices is None or surface.tet_tetrahedra is None:
        return False

    guide_names = _zygote_group_guide_names(v, group_name)
    _relabel_zygote_group_tet_materials(v, group_name, surface)

    surface._zygote_group_name = group_name
    surface._connected_original_tet_components = [surface_name]
    surface._connected_component_fibers = _collect_connected_component_fibers(v, guide_names)
    surface._connected_contour_mesh_components = [
        {'component': name, 'part': _mesh_part_from_name(name)}
        for name in guide_names
    ]
    for name in guide_names:
        comp = v.zygote_muscle_meshes.get(name)
        if comp is not None:
            comp._connected_mesh_owner_name = surface_name
            comp.is_draw_tet_mesh = False
    surface.is_draw_tet_mesh = True
    surface.is_draw = bool(state.get('draw_surface', True))
    return True


def _estimate_group_surface_volume(v, group_name):
    state = _ensure_zygote_group_mapping(v, group_name)
    surface_name = state.get('surface', '')
    surface = v.zygote_muscle_meshes.get(surface_name)
    if surface is None:
        return 0.0
    def _safe_len(value):
        try:
            return len(value) if value is not None else 0
        except Exception:
            return 0
    cache_key = (
        surface_name,
        id(getattr(surface, 'vertices', None)),
        id(getattr(surface, 'faces_3', None)),
        id(getattr(surface, 'faces_4', None)),
        _safe_len(getattr(surface, 'vertices', None)),
        _safe_len(getattr(surface, 'faces_3', None)),
        _safe_len(getattr(surface, 'faces_4', None)),
    )
    cached_key = state.get('_surface_volume_cache_key')
    if cached_key == cache_key and '_surface_volume_cache' in state:
        return float(state.get('_surface_volume_cache', 0.0) or 0.0)
    verts, faces = _component_original_surface(surface)
    if verts is None or faces is None:
        return 0.0
    all_vertices = verts.tolist()
    all_faces = faces.tolist()
    regions = [{'part': 'belly', 'component': surface_name} for _ in range(len(all_vertices))]
    _cap_component_boundary_loops(
        surface, surface_name, verts, faces,
        all_vertices, all_faces, regions,
        {'part': 'belly', 'component': surface_name},
        offset=0, cap_face_indices=None, verbose=False)
    try:
        closed_vertices, closed_faces, _ = _dedupe_surface_vertices(
            all_vertices, all_faces, regions, eps=2e-5)
        mesh = trimesh.Trimesh(vertices=closed_vertices, faces=closed_faces, process=False)
        if not mesh.is_winding_consistent:
            trimesh.repair.fix_normals(mesh, multibody=False)
        volume = _estimate_closed_surface_volume(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    except Exception:
        volume = _estimate_closed_surface_volume(all_vertices, all_faces)
    state['_surface_volume_cache_key'] = cache_key
    state['_surface_volume_cache'] = float(volume)
    return float(volume)


def _draw_emu_tet_target_controls(v, group_name, surface):
    state = _ensure_zygote_group_mapping(v, group_name)
    volume = _estimate_group_surface_volume(v, group_name)
    baseline_group, baseline_volume = _rectus_femoris_baseline_volume(v)
    emu_tets = RECTUS_FEMORIS_BASELINE_TETS
    density = (float(emu_tets) / baseline_volume) if baseline_volume > 0.0 else 0.0
    desired = int(round(volume * density)) if density > 0.0 else 0
    imgui.text(f"Current closed volume: {volume:.9f} m^3")
    if density > 0.0:
        desired = int(np.clip(desired, 1, 1000000))
        surface.target_tet_count = desired
        surface._target_tet_count_source = 'emu_volume'
        surface.enable_tet_boundary_remesh = True
        surface.preserve_tet_surface = False
        imgui.text(
            f"Baseline: {baseline_group} = {emu_tets} tets, "
            f"volume {baseline_volume:.9f} m^3")
        imgui.text(f"Rectus-based density: {density:.3g} tets/m^3")
        imgui.text(f"Auto desired tets: {desired}")
    else:
        surface._target_tet_count_source = 'manual_fallback'
        imgui.text("Approx target inactive: load/designate Rectus Femoris surface first.")
    return desired


def _rectus_femoris_baseline_volume(v):
    group_names = _zygote_loaded_ui_group_names(v)
    preferred = []
    for group_name in group_names:
        low = group_name.lower()
        if 'rectus' in low and 'femoris' in low:
            score = 0 if group_name.startswith('L_') else 1
            preferred.append((score, group_name))
    preferred.sort()
    for _score, group_name in preferred:
        volume = _estimate_group_surface_volume(v, group_name)
        if volume > 0.0:
            return group_name, volume
    return '', 0.0


def _read_tet_file_volume_and_count(path):
    try:
        with open(path, 'rb') as f:
            try:
                data = pickle.load(f)
            except Exception:
                f.seek(0)
                data = np.load(f, allow_pickle=True)
                data = {k: data[k] for k in data.files}
        verts = data.get('vertices', data.get('tet_vertices', None))
        tets = data.get('tetrahedra', data.get('tet_tetrahedra', None))
        if verts is None or tets is None:
            return 0.0, 0
        verts = np.asarray(verts, dtype=np.float64)
        tets = np.asarray(tets, dtype=np.int64)
        if len(verts) == 0 or len(tets) == 0:
            return 0.0, 0
        tv = verts[tets]
        vol = np.abs(np.einsum(
            'ij,ij->i',
            tv[:, 1] - tv[:, 0],
            np.cross(tv[:, 2] - tv[:, 0], tv[:, 3] - tv[:, 0]))) / 6.0
        good = np.isfinite(vol) & (vol > 1e-15)
        return float(np.sum(vol[good])), int(np.sum(good))
    except Exception as exc:
        print(f"[EMU baseline] Could not read {path}: {exc}")
        return 0.0, 0


def _scan_emu_baseline_tet_files(v):
    if hasattr(v, '_emu_baseline_tet_file_cache'):
        return v._emu_baseline_tet_file_cache
    paths = []
    for pattern in ('tet/*_tet.npz', 'tet_*/*_tet.npz', 'emu*/*_tet.npz', 'EMU*/*_tet.npz'):
        paths.extend(glob.glob(pattern))
    paths = sorted(dict.fromkeys(paths))
    labels = ['All tet/*.npz'] + [os.path.relpath(p) for p in paths]
    v._emu_baseline_tet_file_cache = (labels, paths)
    return v._emu_baseline_tet_file_cache


def _draw_emu_baseline_from_tet_files(v, group_name, state):
    labels, paths = _scan_emu_baseline_tet_files(v)
    if not labels:
        return
    idx = int(state.get('emu_baseline_tet_file_idx', 0) or 0)
    idx = max(0, min(idx, len(labels) - 1))
    changed, idx = imgui.combo(f"Baseline tet source##{group_name}_emu_tet_source", idx, labels)
    if changed:
        state['emu_baseline_tet_file_idx'] = int(idx)
    if imgui.button(f"Compute Baseline Density From Tet Source##{group_name}_emu_tet_compute",
                    width=wide_button_width):
        total_volume = 0.0
        total_tets = 0
        selected = []
        if idx == 0:
            selected = [p for p in paths if os.path.dirname(p) == 'tet']
        elif idx - 1 < len(paths):
            selected = [paths[idx - 1]]
        for path in selected:
            vol, n_tets = _read_tet_file_volume_and_count(path)
            total_volume += vol
            total_tets += n_tets
        if total_volume > 0.0 and total_tets > 0:
            state['emu_baseline_muscle_volume'] = float(total_volume)
            state['emu_baseline_muscle_tets'] = int(total_tets)
            print(f"[{group_name}] EMU baseline density from {len(selected)} tet file(s): "
                  f"volume={total_volume:.9g} m^3, tets={total_tets}, "
                  f"density={total_tets / total_volume:.6g} tets/m^3")
        else:
            print(f"[{group_name}] EMU baseline source has no valid tet volume")


def _reset_zygote_group_process(v, group_name):
    state = _ensure_zygote_group_mapping(v, group_name)
    names = _zygote_group_loaded_names(v, group_name)
    mapped_names = [
        state.get(key, '') for key in (
            'surface', 'master', 'follower',
            'master_origin', 'master_insertion',
            'follower_origin', 'follower_insertion')
    ]
    for name in mapped_names:
        if name and name not in names:
            names.append(name)

    for name in names:
        _clear_connected_mesh_ownership(v, name)
    for obj in getattr(v, 'zygote_muscle_meshes', {}).values():
        if getattr(obj, '_connected_mesh_owner_name', '') in names:
            obj._connected_mesh_owner_name = ''

    reset_count = 0
    for name in names:
        obj = v.zygote_muscle_meshes.get(name)
        if obj is None:
            continue
        if hasattr(obj, 'reset_process'):
            obj.reset_process()
        else:
            # Defensive fallback for non-MuscleMesh objects.
            for attr in (
                    'scalar_field', 'vertex_colors', 'contours', 'bounding_planes',
                    'contours_resampled', 'contours_resampled_params',
                    'contours_resampled_fixed', 'contours_resampled_types',
                    'contour_mesh_vertices', 'contour_mesh_faces',
                    'contour_mesh_normals', 'tet_vertices', 'tet_tetrahedra',
                    'tet_faces', 'tet_render_faces', 'tet_sim_faces',
                    'tet_vertex_regions', 'tet_region_labels',
                    'tet_component_labels', 'tet_region_mixed',
                    'tet_quality_stats'):
                if hasattr(obj, attr):
                    setattr(obj, attr, None)
            obj.bounding_planes = []
            obj.waypoints = []
            obj.fiber_architecture = [obj._sobol_sampling_barycentric_default(16)] if hasattr(obj, '_sobol_sampling_barycentric_default') else []
            obj.soft_body = None
        for attr in (
                '_belly_waypoints_before_tendon_extension',
                '_tendon_extended_inspect_contours',
                '_connected_original_tet_components',
                '_connected_component_fibers',
                '_connected_contour_mesh_components',
                '_emu_rest_vertices', '_emu_viewer_cache',
                '_tet_region_surface_colors',
                'tet_quality_stats',
                'last_actual_tet_count',
                'vertex_contour_level', '_emu_rest_waypoints'):
            if hasattr(obj, attr):
                if attr in ('_emu_rest_vertices', '_emu_viewer_cache',
                            '_emu_rest_waypoints'):
                    delattr(obj, attr)
                else:
                    setattr(obj, attr, None)
        obj._connected_mesh_owner_name = ''
        obj.is_draw = True
        obj.is_draw_scalar_field = False
        obj.is_draw_contours = False
        obj.is_draw_bounding_box = False
        obj.is_draw_contour_mesh = False
        obj.is_draw_tet_mesh = False
        obj.is_draw_tet_edges = False
        if hasattr(obj, 'is_draw_tet_internal_faces'):
            obj.is_draw_tet_internal_faces = False
        obj.is_draw_fiber_architecture = False
        _invalidate_tet_draw_cache(obj)
        reset_count += 1

    for store_name in ('inspect_2d_open', 'inspect_2d_stream_idx', 'inspect_2d_contour_idx'):
        store = getattr(v, store_name, None)
        if isinstance(store, dict):
            for name in names:
                store.pop(name, None)

    state['draw_parts'] = True
    state['draw_surface'] = True
    state['draw_surface_tet'] = False
    state['draw_surface_tet_edges'] = False
    state['draw_surface_tet_internals'] = False
    state['draw_scalar_field'] = False
    state['draw_contours'] = False
    state['draw_fibers'] = False
    state['draw_bounding_boxes'] = False
    state['belly_step'] = 9
    state['tendon_step'] = 8
    v.zygote_group_belly_step = 9
    _apply_zygote_group_links(v, group_name)
    print(f"[{group_name}] Reset process state for {reset_count} group mesh(es)")


def _draw_zygote_group_pipeline_controls(v, group_name):
    state = _ensure_zygote_group_mapping(v, group_name)
    _apply_zygote_group_links(v, group_name)

    if imgui.button(f"Reset Group Process##{group_name}_reset_process", width=wide_button_width):
        _reset_zygote_group_process(v, group_name)

    master_name = state.get('master', '')
    master = v.zygote_muscle_meshes.get(master_name)
    imgui.text("Belly Fiber Pipeline")
    if master is None:
        imgui.text("Select a master scaffold")
    else:
        if not hasattr(v, 'zygote_group_belly_step'):
            v.zygote_group_belly_step = int(state.get('belly_step', 9))
        changed, step = imgui.slider_int(
            f"Belly target step##{group_name}_belly_step",
            int(state.get('belly_step', v.zygote_group_belly_step)), 1, 11)
        if changed:
            state['belly_step'] = int(step)
            v.zygote_group_belly_step = int(step)
        belly_step_names = ['', 'Scalar', 'Contours', 'Fill Gap', 'Transitions',
                            'Smooth', 'Cut', 'Stream Smooth', 'Select Contours',
                            'Build Fibers', 'Resample', 'Build Contour Mesh']
        step_i = int(state.get('belly_step', 9))
        imgui.text(f"Runs master belly from Scalar through: {belly_step_names[step_i]}")
        if imgui.button(f"Run Belly Pipeline to {belly_step_names[step_i]}##{group_name}_belly_proc",
                        width=wide_button_width):
            _apply_zygote_group_links(v, group_name)
            _run_component_pipeline_quick(
                v, master_name, master, step_i,
                defer=getattr(master, 'animate_process', False))
        if imgui.button(f"Extend Belly Fibers Through Tendons##{group_name}_extend",
                        width=wide_button_width):
            _extend_belly_and_linked_counterpart_fibers(v, master_name, master)
        _draw_zygote_group_inspect_buttons(v, group_name)

    imgui.separator()
    imgui.text("Tendon Contour Pipeline")
    changed, step = imgui.slider_int(
        f"Tendon target step##{group_name}_tendon_step",
        int(state.get('tendon_step', 8)), 1, 8)
    if changed:
        state['tendon_step'] = int(step)
    tendon_step_names = ['', 'Scalar', 'Contours', 'Fill Gap', 'Transitions',
                         'Smooth', 'Cut', 'Stream Smooth', 'Select Contours']
    tendon_step = int(state.get('tendon_step', 8))
    imgui.text(f"Runs each selected tendon through: {tendon_step_names[tendon_step]}")
    if imgui.button(f"Run Tendon Pipeline to {tendon_step_names[tendon_step]}##{group_name}_tendon_proc",
                    width=wide_button_width):
        _apply_zygote_group_links(v, group_name)
        for key, fallback_key in (
                ('master_origin', 'follower_origin'),
                ('master_insertion', 'follower_insertion')):
            name = state.get(key, '')
            if not name:
                name = state.get(fallback_key, '')
            obj = v.zygote_muscle_meshes.get(name)
            if obj is None:
                continue
            other_name = getattr(obj, 'linked_counterpart_name', '')
            linked_msg = f" with linked follower {other_name}" if other_name else ""
            print(f"[{group_name}] Processing master tendon {name}{linked_msg} to step {tendon_step}")
            _run_component_pipeline_quick(
                v, name, obj, tendon_step,
                defer=getattr(obj, 'animate_process', False))

    imgui.separator()
    surface_name = state.get('surface', '')
    surface = v.zygote_muscle_meshes.get(surface_name)
    imgui.text("Uncut Surface Tet")
    if surface is None:
        imgui.text("Select a surface mesh")
    else:
        if not hasattr(surface, 'target_tet_count'):
            surface.target_tet_count = 30000
        if not hasattr(surface, 'enable_tet_boundary_remesh'):
            surface.enable_tet_boundary_remesh = True
            surface.preserve_tet_surface = False
        changed_approx, surface.enable_tet_boundary_remesh = imgui.checkbox(
            f"Isotropic EMU surface remesh##{group_name}_approx_surface",
            bool(surface.enable_tet_boundary_remesh))
        if changed_approx:
            surface.preserve_tet_surface = not bool(surface.enable_tet_boundary_remesh)
        desired = _draw_emu_tet_target_controls(v, group_name, surface)
        if desired > 0:
            imgui.text(f"Tetrahedralize target: {int(surface.target_tet_count)} stable tets")
            if bool(surface.enable_tet_boundary_remesh):
                imgui.text("Voxel/SDF boundary remesh removes sliver-forcing triangles.")
        elif bool(surface.enable_tet_boundary_remesh):
            imgui.text("Tetrahedralize requires EMU-derived target.")
        if imgui.button(f"Tetrahedralize Surface##{group_name}_tet", width=wide_button_width):
            if (bool(surface.enable_tet_boundary_remesh)
                    and getattr(surface, '_target_tet_count_source', '') != 'emu_volume'):
                print(f"[{group_name}] Tetrahedralize blocked: approximate target must be computed "
                      f"from Rectus Femoris baseline density. Load/designate Rectus Femoris "
                      f"surface so its volume can define "
                      f"{RECTUS_FEMORIS_BASELINE_TETS} tets.")
            else:
                if _tetrahedralize_zygote_group_surface(
                        v, group_name,
                        defer=getattr(surface, 'animate_process', False)):
                    for attr in ('_emu_rest_vertices', '_emu_viewer_cache'):
                        if hasattr(surface, attr):
                            delattr(surface, attr)
        if imgui.button(f"Save Surface Tet##{group_name}_save_tet", width=wide_button_width):
            guide_names = _zygote_group_guide_names(v, group_name)
            surface._zygote_group_name = group_name
            surface._connected_original_tet_components = [surface_name]
            surface._connected_component_fibers = _collect_connected_component_fibers(
                v, guide_names)
            surface._connected_contour_mesh_components = [
                {'component': name, 'part': _mesh_part_from_name(name)}
                for name in guide_names]
            surface.save_tetrahedron_mesh(group_name)
        if imgui.button(f"Load Surface Tet##{group_name}_load_tet", width=wide_button_width):
            if surface.load_tetrahedron_mesh(group_name):
                surface.tet_render_contact_offsets = None
                for attr in ('_emu_rest_vertices', '_emu_viewer_cache'):
                    if hasattr(surface, attr):
                        delattr(surface, attr)
                _restore_connected_component_fibers(
                    v, group_name, surface_name, surface)
                _ensure_zygote_group_render_embedding(group_name, surface)
                # Recompute material designation from the currently loaded
                # group guides. This upgrades old/transitional files whose
                # anatomical render skin had no tendon labels.
                _relabel_zygote_group_tet_materials(
                    v, group_name, surface)
                surface.is_draw_tet_mesh = True

        imgui.separator()
        imgui.text("EMU Current-Pose Bake")
        imgui.text("EMU examples: muscle 6e6 Pa, tendon 4.5e8-1.2e9 Pa, nu=0.49")
        defaults = {
            'emu_muscle_youngs': 6e6,
            'emu_tendon_youngs': 4.5e8,
            'emu_poisson': 0.49,
            'emu_activation': 0.0,
            'emu_max_active_stress': 6e6,
            'emu_alpha': 1.0,
            'emu_k_modes': 16,
            'emu_max_iters': 5,
            'emu_load_steps': 10,
            'emu_attachment_rings': 1,
            'emu_auto_smooth': True,
            'emu_bone_collision': True,
            'emu_ignore_rest_inside': True,
            'emu_collision_margin': 0.003,
            'emu_collision_stiffness': 1e7,
            'emu_collision_iterations': 1,
        }
        for key, value in defaults.items():
            state.setdefault(key, value)
        imgui.push_item_width(150)
        _, state['emu_muscle_youngs'] = imgui.input_float(
            f"Muscle E (Pa)##{group_name}_emu_muscle_E",
            float(state['emu_muscle_youngs']), 0.0, 0.0, "%.3g")
        _, state['emu_tendon_youngs'] = imgui.input_float(
            f"Tendon E (Pa)##{group_name}_emu_tendon_E",
            float(state['emu_tendon_youngs']), 0.0, 0.0, "%.3g")
        _, state['emu_poisson'] = imgui.input_float(
            f"Poisson##{group_name}_emu_poisson",
            float(state['emu_poisson']), 0.0, 0.0, "%.3f")
        _, state['emu_activation'] = imgui.slider_float(
            f"Belly activation [0,1]##{group_name}_emu_activation",
            float(state['emu_activation']), 0.0, 1.0)
        _, state['emu_max_active_stress'] = imgui.input_float(
            f"Max active coefficient (Pa)##{group_name}_emu_active_stress",
            float(state['emu_max_active_stress']), 0.0, 0.0, "%.3g")
        _, state['emu_alpha'] = imgui.input_float(
            f"ACAP alpha##{group_name}_emu_alpha",
            float(state['emu_alpha']), 0.0, 0.0, "%.3g")
        _, state['emu_k_modes'] = imgui.input_int(
            f"Woodbury modes##{group_name}_emu_modes",
            int(state['emu_k_modes']))
        _, state['emu_max_iters'] = imgui.input_int(
            f"Newton iterations/load step##{group_name}_emu_iters",
            int(state['emu_max_iters']))
        _, state['emu_load_steps'] = imgui.input_int(
            f"Attachment load steps##{group_name}_emu_load_steps",
            int(state['emu_load_steps']))
        _, state['emu_attachment_rings'] = imgui.slider_int(
            f"Fixed attachment rings##{group_name}_emu_attachment_rings",
            int(state['emu_attachment_rings']), 0, 2)
        _, state['emu_auto_smooth'] = imgui.checkbox(
            f"Smooth EMU quality##{group_name}_emu_smooth",
            bool(state['emu_auto_smooth']))
        _, state['emu_bone_collision'] = imgui.checkbox(
            f"Bone collision##{group_name}_emu_collision",
            bool(state['emu_bone_collision']))
        _, state['emu_ignore_rest_inside'] = imgui.checkbox(
            f"Ignore vertices inside bone at rest##{group_name}_emu_rest_inside",
            bool(state['emu_ignore_rest_inside']))
        _, state['emu_collision_margin'] = imgui.input_float(
            f"Bone margin (m)##{group_name}_emu_margin",
            float(state['emu_collision_margin']), 0.0, 0.0, "%.4f")
        _, state['emu_collision_stiffness'] = imgui.input_float(
            f"Contact stiffness##{group_name}_emu_contact_k",
            float(state['emu_collision_stiffness']), 0.0, 0.0, "%.3g")
        _, state['emu_collision_iterations'] = imgui.input_int(
            f"Contact passes/iteration##{group_name}_emu_contact_iters",
            int(state['emu_collision_iterations']))
        imgui.pop_item_width()
        imgui.text("Active coefficient = activation * max coefficient; belly only.")
        if state['emu_auto_smooth']:
            imgui.text("Quality mode uses selected alpha, modes>=16, >=5 iterations/step.")
        if imgui.button(f"EMU Bake Current Pose##{group_name}_emu_bake",
                        width=wide_button_width):
            try:
                _run_group_emu_current_pose(v, group_name, surface, state)
            except Exception as exc:
                state['emu_last_status'] = f"ERROR: {exc}"
                print(f"[{group_name}] EMU current-pose bake failed: {exc}")
                traceback.print_exc()
        if imgui.button(f"Reset EMU Tet to Rest##{group_name}_emu_reset",
                        width=wide_button_width):
            _reset_group_emu_pose(v, group_name, surface)
            state['emu_last_status'] = "Reset to saved/rest tet positions"
        if state.get('emu_last_status'):
            imgui.text_wrapped(str(state['emu_last_status']))


def _focus_zygote_points(v, pts, label):
    if pts is None or len(pts) == 0:
        print(f"[{label}] No vertices to focus on")
        return
    pts = np.asarray(pts, dtype=np.float64)
    min_pt = np.min(pts, axis=0)
    max_pt = np.max(pts, axis=0)
    center = (min_pt + max_pt) / 2.0
    bbox_size = float(np.linalg.norm(max_pt - min_pt))
    v.trans = -center * 1000.0
    eye_dir = v.eye / (np.linalg.norm(v.eye) + 1e-10)
    v.eye = eye_dir * max(bbox_size * 2.0, MIN_EYE_DISTANCE)


def _focus_zygote_group(v, group_name):
    state = _ensure_zygote_group_mapping(v, group_name)
    surface_name = state.get('surface', '')
    surface = v.zygote_muscle_meshes.get(surface_name)
    if surface is not None and getattr(surface, 'vertices', None) is not None:
        _focus_zygote_points(v, surface.vertices, group_name)
        return
    pts = []
    for name in _zygote_group_loaded_names(v, group_name):
        obj = v.zygote_muscle_meshes.get(name)
        verts = getattr(obj, 'vertices', None)
        if verts is not None and len(verts) > 0:
            pts.append(np.asarray(verts, dtype=np.float64))
    _focus_zygote_points(v, np.vstack(pts) if pts else None, group_name)


def _open_inspect_2d(v, name, stream_idx=0, contour_idx=0):
    obj = getattr(v, 'zygote_muscle_meshes', {}).get(name)
    if obj is None:
        return False
    has_contour_data = (
        hasattr(obj, 'contours')
        and obj.contours is not None
        and len(obj.contours) > 0
    )
    has_extended_data = (
        getattr(obj, '_tendon_extended_inspect_contours', None) is not None
        and len(getattr(obj, '_tendon_extended_inspect_contours', []) or []) > 0
    )
    if not has_contour_data and not has_extended_data:
        print(f"[{name}] No contour data. Run 'Find Contours' first.")
        return False
    if not hasattr(v, 'inspect_2d_open'):
        v.inspect_2d_open = {}
    if not hasattr(v, 'inspect_2d_stream_idx'):
        v.inspect_2d_stream_idx = {}
    if not hasattr(v, 'inspect_2d_contour_idx'):
        v.inspect_2d_contour_idx = {}
    v.inspect_2d_open[name] = True
    v.inspect_2d_stream_idx[name] = max(0, int(stream_idx))
    v.inspect_2d_contour_idx[name] = max(0, int(contour_idx))
    return True


def _draw_zygote_group_inspect_buttons(v, group_name):
    state = _ensure_zygote_group_mapping(v, group_name)
    entries = [
        ('Master Belly', state.get('master', '')),
        ('Follower Belly', state.get('follower', '')),
    ]
    if not any(name and name in getattr(v, 'zygote_muscle_meshes', {}) for _, name in entries):
        return

    if not imgui.tree_node(f"Inspect 2D##{group_name}_inspect_bellies"):
        return
    for label, name in entries:
        if not name or name not in getattr(v, 'zygote_muscle_meshes', {}):
            continue
        if imgui.button(f"{label}: Inspect 2D##{group_name}_inspect_{name}",
                        width=wide_button_width):
            _open_inspect_2d(v, name, stream_idx=0, contour_idx=0)
    imgui.tree_pop()


def _draw_tet_quality_stats(obj):
    stats = getattr(obj, 'tet_quality_stats', None)
    if not stats:
        return
    count = int(stats.get('count', 0) or 0)
    deg = int(stats.get('degenerate', 0) or 0)
    near = int(stats.get('near_zero', 0) or 0)
    edge_cv = float(stats.get('edge_cv', 0.0) or 0.0)
    imgui.text(f"Tet quality: total {count}, degenerate {deg}, near-zero {near}")
    imgui.text(f"Volume median {float(stats.get('median_volume', 0.0) or 0.0):.3g}, "
               f"p01 {float(stats.get('p01_volume', 0.0) or 0.0):.3g}, "
               f"edge CV {edge_cv:.3g}")
    imgui.text(
        f"Shape scaled-J p01 {float(stats.get('scaled_jacobian_p01', 0.0) or 0.0):.3g}, "
        f"mean-ratio p01 {float(stats.get('mean_ratio_p01', 0.0) or 0.0):.3g}")
    imgui.text(
        f"Slivers critical/weak: {int(stats.get('critical_slivers', 0) or 0)}/"
        f"{int(stats.get('weak_slivers', 0) or 0)}")


def _group_any_draw_flag(v, names, attr, default=False):
    values = [
        bool(getattr(v.zygote_muscle_meshes[name], attr, default))
        for name in names
        if name in getattr(v, 'zygote_muscle_meshes', {})
    ]
    return any(values) if values else bool(default)


def _draw_zygote_group_draw_controls(v, group_name):
    state = _ensure_zygote_group_mapping(v, group_name)
    names = _zygote_group_loaded_names(v, group_name)
    surface_name = state.get('surface', '')
    part_names = [n for n in names if n != surface_name]

    if imgui.button(f"Focus Group##{group_name}_focus", width=button_width):
        _focus_zygote_group(v, group_name)

    state['draw_scalar_field'] = _group_any_draw_flag(
        v, part_names, 'is_draw_scalar_field', False)
    changed, state['draw_scalar_field'] = imgui.checkbox(
        f"Draw scalar fields##{group_name}_draw_scalar",
        bool(state['draw_scalar_field']))
    if changed:
        for name in part_names:
            obj = v.zygote_muscle_meshes.get(name)
            if obj is not None:
                obj.is_draw_scalar_field = bool(state['draw_scalar_field'])

    state['draw_contours'] = _group_any_draw_flag(
        v, part_names, 'is_draw_contours', False)
    changed, state['draw_contours'] = imgui.checkbox(
        f"Draw contours##{group_name}_draw_contours",
        bool(state['draw_contours']))
    if changed:
        for name in part_names:
            obj = v.zygote_muscle_meshes.get(name)
            if obj is not None:
                obj.is_draw_contours = bool(state['draw_contours'])

    state['draw_fibers'] = _group_any_draw_flag(
        v, part_names, 'is_draw_fiber_architecture', False)
    changed, state['draw_fibers'] = imgui.checkbox(
        f"Draw fibers##{group_name}_draw_fibers",
        bool(state['draw_fibers']))
    if changed:
        for name in part_names:
            obj = v.zygote_muscle_meshes.get(name)
            if obj is not None:
                obj.is_draw_fiber_architecture = bool(state['draw_fibers'])

    state['draw_bounding_boxes'] = _group_any_draw_flag(
        v, part_names, 'is_draw_bounding_box', False)
    changed, state['draw_bounding_boxes'] = imgui.checkbox(
        f"Draw bounding boxes##{group_name}_draw_bbox",
        bool(state['draw_bounding_boxes']))
    if changed:
        for name in part_names:
            obj = v.zygote_muscle_meshes.get(name)
            if obj is not None:
                obj.is_draw_bounding_box = bool(state['draw_bounding_boxes'])

    state['draw_parts'] = _group_any_draw_flag(v, part_names, 'is_draw', True)
    changed, state['draw_parts'] = imgui.checkbox(
        f"Draw guide parts##{group_name}_draw_parts", bool(state['draw_parts']))
    if changed:
        for name in part_names:
            obj = v.zygote_muscle_meshes.get(name)
            if obj is not None:
                obj.is_draw = bool(state['draw_parts'])
    if surface_name in v.zygote_muscle_meshes:
        state['draw_surface'] = bool(getattr(v.zygote_muscle_meshes[surface_name], 'is_draw', True))
    changed, state['draw_surface'] = imgui.checkbox(
        f"Draw surface mesh##{group_name}_draw_surface", bool(state['draw_surface']))
    if changed and surface_name in v.zygote_muscle_meshes:
        v.zygote_muscle_meshes[surface_name].is_draw = bool(state['draw_surface'])
    if surface_name in v.zygote_muscle_meshes:
        state['draw_surface_tet'] = bool(getattr(v.zygote_muscle_meshes[surface_name], 'is_draw_tet_mesh', False))
    changed, state['draw_surface_tet'] = imgui.checkbox(
        f"Draw surface tet##{group_name}_draw_surface_tet", bool(state['draw_surface_tet']))
    if changed and surface_name in v.zygote_muscle_meshes:
        v.zygote_muscle_meshes[surface_name].is_draw_tet_mesh = bool(state['draw_surface_tet'])
    if surface_name in v.zygote_muscle_meshes:
        state['draw_surface_tet_edges'] = bool(getattr(v.zygote_muscle_meshes[surface_name], 'is_draw_tet_edges', False))
    changed, state['draw_surface_tet_edges'] = imgui.checkbox(
        f"Draw tet edges##{group_name}_draw_tet_edges", bool(state['draw_surface_tet_edges']))
    if changed and surface_name in v.zygote_muscle_meshes:
        v.zygote_muscle_meshes[surface_name].is_draw_tet_edges = bool(state['draw_surface_tet_edges'])
    if surface_name in v.zygote_muscle_meshes:
        state['draw_surface_tet_internals'] = bool(getattr(v.zygote_muscle_meshes[surface_name], 'is_draw_tet_internal_faces', False))
    changed, state['draw_surface_tet_internals'] = imgui.checkbox(
        f"Draw tet internals##{group_name}_draw_tet_internals", bool(state['draw_surface_tet_internals']))
    if changed and surface_name in v.zygote_muscle_meshes:
        v.zygote_muscle_meshes[surface_name].is_draw_tet_internal_faces = bool(state['draw_surface_tet_internals'])
    surface = v.zygote_muscle_meshes.get(surface_name)
    if surface is not None:
        _draw_tet_quality_stats(surface)


def _sync_zygote_group_draw_state(v, draw_value):
    for group_name in _zygote_loaded_ui_group_names(v):
        state = _ensure_zygote_group_mapping(v, group_name)
        state['draw_parts'] = bool(draw_value)
        state['draw_surface'] = bool(draw_value)


def _draw_zygote_group_part_panels(v, group_name):
    state = _ensure_zygote_group_mapping(v, group_name)
    # Establish counterpart links before rendering the per-part UI so that
    # processing a master part from here reliably drives its follower (the
    # follower's contours come from the master's schedule at step 2). Without
    # this the Process button in a part panel may run before group links are
    # applied and the follower is left with no contours.
    _apply_zygote_group_links(v, group_name)
    surface_name = state.get('surface', '')
    ordered = []
    for key in ('surface', 'master', 'follower', 'master_origin', 'master_insertion',
                'follower_origin', 'follower_insertion'):
        name = state.get(key, '')
        if name and name in getattr(v, 'zygote_muscle_meshes', {}) and name not in ordered:
            ordered.append(name)
    for name in _zygote_group_loaded_names(v, group_name):
        if name not in ordered:
            ordered.append(name)

    for name in ordered:
        obj = v.zygote_muscle_meshes.get(name)
        if obj is None:
            continue
        role = "surface" if name == surface_name else _mesh_part_from_name(name)
        if imgui.tree_node(f"{name} ({role})##{group_name}_part_{name}"):
            _draw_zygote_muscle_body(v, name, obj)
            imgui.tree_pop()


def _draw_zygote_group_ui(v):
    groups = _zygote_loaded_ui_group_names(v)
    if not groups:
        return
    if not imgui.tree_node("Muscle Groups", imgui.TREE_NODE_DEFAULT_OPEN):
        return
    for group_name in groups:
        state = _ensure_zygote_group_mapping(v, group_name)
        # Group headers start collapsed. Use a new ID namespace so an older
        # imgui.ini entry that remembered these headers open cannot override
        # the new default on the next launch.
        if imgui.tree_node(f"{group_name}##zygote_group_collapsed_{group_name}"):
            changed, state['show_parts'] = imgui.checkbox(
                f"Show part panels##{group_name}_show_parts",
                bool(state.get('show_parts', False)))
            if changed:
                state['show_parts'] = bool(state['show_parts'])

            if imgui.tree_node(f"Part Designation##{group_name}_designation"):
                any_changed = False
                any_changed |= _draw_group_combo(v, group_name, 'surface', 'Surface tet mesh')
                any_changed |= _draw_group_combo(v, group_name, 'master', 'Master scaffold')
                any_changed |= _draw_group_combo(v, group_name, 'follower', 'Follower scaffold')
                any_changed |= _draw_group_combo(v, group_name, 'master_origin', 'Master origin tendon')
                any_changed |= _draw_group_combo(v, group_name, 'master_insertion', 'Master insertion tendon')
                any_changed |= _draw_group_combo(v, group_name, 'follower_origin', 'Follower origin tendon')
                any_changed |= _draw_group_combo(v, group_name, 'follower_insertion', 'Follower insertion tendon')
                if any_changed:
                    _apply_zygote_group_links(v, group_name)
                if imgui.button(f"Apply Links##{group_name}_apply_links", width=wide_button_width):
                    _apply_zygote_group_links(v, group_name)
                    print(f"[{group_name}] Applied group links")
                imgui.tree_pop()

            imgui.separator()
            _draw_zygote_group_draw_controls(v, group_name)
            if bool(state.get('show_parts', False)):
                imgui.separator()
                if imgui.tree_node(f"Show Part Panels##{group_name}_show_part_tree"):
                    _draw_zygote_group_part_panels(v, group_name)
                    imgui.tree_pop()
            imgui.separator()
            _draw_zygote_group_pipeline_controls(v, group_name)
            imgui.tree_pop()
    imgui.tree_pop()


def _zygote_hidden_group_part_names(v):
    hidden = set()
    for group_name in _zygote_loaded_ui_group_names(v):
        hidden.update(_zygote_group_loaded_names(v, group_name))
    return hidden


def _apply_zygote_muscle_style(mesh, name, path, muscle_color, transparency, is_draw):
    is_tendon = _is_zygote_tendon_mesh(name, path)
    mesh.is_zygote_tendon = is_tendon
    mesh.color = ZYGOTE_TENDON_COLOR.copy() if is_tendon else np.array(muscle_color)
    mesh.transparency = transparency
    mesh.is_draw = is_draw


def _linked_counterpart(v, obj):
    name = getattr(obj, 'linked_counterpart_name', '')
    if not name:
        return None, None
    other = getattr(v, 'zygote_muscle_meshes', {}).get(name)
    if other is None or other is obj:
        return None, None
    return name, other


def _set_symmetric_counterpart_link(v, name, counterpart_name):
    obj = getattr(v, 'zygote_muscle_meshes', {}).get(name)
    if obj is None:
        return
    prev_name = getattr(obj, 'linked_counterpart_name', '')
    obj.linked_counterpart_name = counterpart_name or ''

    if prev_name and prev_name != counterpart_name:
        prev = v.zygote_muscle_meshes.get(prev_name)
        if prev is not None and getattr(prev, 'linked_counterpart_name', '') == name:
            prev.linked_counterpart_name = ''

    if counterpart_name:
        other = v.zygote_muscle_meshes.get(counterpart_name)
        if other is not None:
            other.linked_counterpart_name = name
            other.linked_drive_counterpart = bool(getattr(obj, 'linked_drive_counterpart', True))
            other.linked_use_shared_scalar = bool(getattr(obj, 'linked_use_shared_scalar', True))
            other.linked_pair_eps = float(getattr(obj, 'linked_pair_eps', 1e-5))


def _sync_counterpart_link_settings(v, name):
    obj = getattr(v, 'zygote_muscle_meshes', {}).get(name)
    if obj is None:
        return
    other_name = getattr(obj, 'linked_counterpart_name', '')
    other = v.zygote_muscle_meshes.get(other_name) if other_name else None
    if other is None:
        return
    other.linked_counterpart_name = name
    other.linked_drive_counterpart = bool(getattr(obj, 'linked_drive_counterpart', True))
    other.linked_use_shared_scalar = bool(getattr(obj, 'linked_use_shared_scalar', True))
    other.linked_pair_eps = float(getattr(obj, 'linked_pair_eps', 1e-5))


def _sync_counterpart_display_state(v, name):
    obj = getattr(v, 'zygote_muscle_meshes', {}).get(name)
    if obj is None:
        return
    other_name = getattr(obj, 'linked_counterpart_name', '')
    other = v.zygote_muscle_meshes.get(other_name) if other_name else None
    if other is None:
        return

    fields = [
        'transparency',
        'is_draw',
        'is_draw_open_edges',
        'is_draw_scalar_field',
        'is_draw_contours',
        'is_draw_contour_vertices',
        'is_draw_farthest_pair',
        'is_draw_edges',
        'is_draw_centroid',
        'is_draw_bounding_box',
        'bounding_box_draw_mode',
        'is_draw_discarded',
        'is_draw_fiber_architecture',
        'is_draw_resampled_vertices',
        'is_draw_contour_mesh',
        'is_draw_tet_mesh',
        'is_draw_tet_edges',
        'is_draw_tet_internal_faces',
        'tet_internal_face_stride',
        'is_draw_constraints',
    ]
    for field in fields:
        if hasattr(obj, field):
            setattr(other, field, getattr(obj, field))
    if getattr(other, 'vertex_colors', None) is not None:
        other.vertex_colors[:, 3] = float(getattr(other, 'transparency', 1.0))
    if (getattr(other, 'is_draw_scalar_field', False)
            and getattr(other, '_scalar_anim_target_colors', None) is not None):
        other.vertex_colors = other._scalar_anim_target_colors.copy()


def _collect_scalar_boundary_indices(obj):
    origin_indices = []
    insertion_indices = []
    for cls, edge_group in zip(getattr(obj, 'edge_classes', []) or [],
                               getattr(obj, 'edge_groups', []) or []):
        if cls == 'origin':
            origin_indices.extend(edge_group)
        else:
            insertion_indices.extend(edge_group)
    return np.asarray(origin_indices, dtype=np.int64), np.asarray(insertion_indices, dtype=np.int64)


def _install_scalar_field(obj, u, defer=False):
    obj.scalar_field = np.asarray(u, dtype=np.float64)
    obj._face_scalar_min = None
    obj._face_scalar_max = None

    u_min, u_max = np.min(obj.scalar_field), np.max(obj.scalar_field)
    normalized_u = ((obj.scalar_field - u_min) / (u_max - u_min)
                    if u_max > u_min else np.zeros_like(obj.scalar_field))
    target_colors = np.array(COLOR_MAP(1 - normalized_u)[:, :4], dtype=np.float32)
    target_colors[:, 3] = obj.transparency
    obj._scalar_anim_target_colors = target_colors[obj.faces_3[:, :, 0].flatten()]
    obj._scalar_anim_normalized_u = normalized_u[obj.faces_3[:, :, 0].flatten()]
    obj._scalar_anim_active = False
    obj._scalar_replayed = False
    if defer:
        obj.is_draw_scalar_field = False
    else:
        obj.vertex_colors = obj._scalar_anim_target_colors.copy()
        obj.is_draw_scalar_field = True
        obj._scalar_replayed = True


def _find_mutual_near_vertex_pairs(obj_a, obj_b, eps):
    from scipy.spatial import cKDTree
    va = np.asarray(obj_a.vertices, dtype=np.float64)
    vb = np.asarray(obj_b.vertices, dtype=np.float64)
    if len(va) == 0 or len(vb) == 0:
        return np.empty((0, 2), dtype=np.int64)
    tree_b = cKDTree(vb)
    dist_ab, idx_ab = tree_b.query(va, k=1)
    tree_a = cKDTree(va)
    _, idx_ba = tree_a.query(vb, k=1)
    pairs = []
    for ia, (d, ib) in enumerate(zip(dist_ab, idx_ab)):
        if d <= eps and idx_ba[int(ib)] == ia:
            pairs.append((ia, int(ib)))
    return np.asarray(pairs, dtype=np.int64)


def _compute_linked_scalar_field(master, other, defer=False):
    """Solve one Laplace field on two meshes coupled by near-identical vertices."""
    import scipy.sparse
    import scipy.sparse.linalg

    o_a, i_a = _collect_scalar_boundary_indices(master)
    o_b, i_b = _collect_scalar_boundary_indices(other)
    if len(o_a) == 0 or len(i_a) == 0 or len(o_b) == 0 or len(i_b) == 0:
        print("Linked scalar field needs origin and insertion edge groups on both meshes")
        return False

    n_a = len(master.vertices)
    n_b = len(other.vertices)
    w_a = cotangent_weight_matrix(master.vertices, master.faces_3)
    w_b = cotangent_weight_matrix(other.vertices, other.faces_3)
    l_a = -scipy.sparse.diags(w_a.sum(axis=1).A1) + w_a
    l_b = -scipy.sparse.diags(w_b.sum(axis=1).A1) + w_b
    L = scipy.sparse.block_diag((l_a, l_b), format='lil')

    eps = float(getattr(master, 'linked_pair_eps', 1e-5))
    pairs = _find_mutual_near_vertex_pairs(master, other, eps)
    if len(pairs) == 0 and eps < 1e-4:
        pairs = _find_mutual_near_vertex_pairs(master, other, 1e-4)
    if len(pairs) == 0:
        print("Linked scalar field: no near-zero counterpart vertex pairs found; using separate scalar fields")
        master.compute_scalar_field(defer=defer)
        other.compute_scalar_field(defer=defer)
        return False

    coupling_weight = float(getattr(master, 'linked_scalar_pair_weight', 1e4))
    for ia, ib_local in pairs:
        ib = n_a + int(ib_local)
        ia = int(ia)
        L[ia, ia] -= coupling_weight
        L[ib, ib] -= coupling_weight
        L[ia, ib] += coupling_weight
        L[ib, ia] += coupling_weight
    L = L.tocsr()

    n = n_a + n_b
    b = np.zeros(n)
    boundary_mask = np.zeros(n, dtype=bool)
    origin = np.concatenate([o_a, n_a + o_b])
    insertion = np.concatenate([i_a, n_a + i_b])
    b[origin] = 1.0
    b[insertion] = 10.0
    boundary_mask[origin] = True
    boundary_mask[insertion] = True

    free = ~boundary_mask
    A = L[free][:, free] + 1e-8 * scipy.sparse.eye(int(np.sum(free)))
    rhs = -L[free][:, boundary_mask] @ b[boundary_mask]
    u = np.zeros(n)
    u[free] = scipy.sparse.linalg.spsolve(A, rhs)
    u[boundary_mask] = b[boundary_mask]

    _install_scalar_field(master, u[:n_a], defer=defer)
    _install_scalar_field(other, u[n_a:], defer=defer)
    master.linked_counterpart_pairs = pairs
    other.linked_counterpart_pairs = pairs[:, ::-1].copy()
    print(f"Linked scalar field: coupled {len(pairs)} vertex pairs")
    return True


def _ensure_counterpart_scalar(v, master_name, master, defer=False):
    other_name, other = _linked_counterpart(v, master)
    if other is None:
        return None
    if len(getattr(other, 'edge_groups', []) or []) == 0 or len(getattr(other, 'edge_classes', []) or []) == 0:
        print(f"[{master_name}] Linked counterpart {other_name}: need edge_groups and edge_classes")
        return None
    if getattr(master, 'linked_use_shared_scalar', True):
        _compute_linked_scalar_field(master, other, defer=defer)
    elif other.scalar_field is None:
        other.compute_scalar_field(defer=defer)
        print(f"[{master_name}] Linked counterpart {other_name}: scalar field computed")
    return other


def _apply_master_contour_schedule_to_counterpart(v, master_name, master, defer=False, label="schedule"):
    other_name, other = _linked_counterpart(v, master)
    if other is None:
        return False
    if master.scalar_field is None or not getattr(master, 'bounding_planes', None):
        print(f"[{master_name}] Linked counterpart {other_name}: master has no contour schedule")
        return False
    if other.scalar_field is None:
        other.compute_scalar_field(defer=defer)
    values = master.get_contour_scalar_schedule() if hasattr(master, 'get_contour_scalar_schedule') else []
    if not values:
        print(f"[{master_name}] Linked counterpart {other_name}: empty master contour schedule")
        return False
    ok = other.find_contours_at_values(values, skeleton_meshes=v.zygote_skeleton_meshes, defer=defer)
    if ok:
        other.linked_master_name = master_name
        other.linked_contour_schedule_values = list(values)
        if not defer:
            other.is_draw_bounding_box = True
        print(f"[{master_name}] Linked counterpart {other_name}: applied {label} ({len(values)} levels)")
    return ok


def _sync_counterpart_level_selection(v, master_name, master, defer=False):
    other_name, other = _linked_counterpart(v, master)
    if other is None:
        return False
    if getattr(master, 'stream_selected_levels', None) is None:
        print(f"[{master_name}] Linked counterpart {other_name}: master has no selected levels")
        return False
    if getattr(other, 'stream_contours', None) is None:
        print(f"[{master_name}] Linked counterpart {other_name}: follower needs cut streams first")
        return False

    # Initialize follower level-select bookkeeping on its own geometry, then
    # overwrite the checkboxes with the master's selected level indices.
    other.select_levels()
    if getattr(other, '_level_select_checkboxes', None) is None:
        return False

    max_stream_count = int(getattr(other, 'max_stream_count', len(other.stream_contours)))
    master_sel = getattr(master, 'stream_selected_levels', [])
    master_bps = getattr(master, 'stream_bounding_planes', None)
    other_bps = getattr(other, 'stream_bounding_planes', None)

    def _scalar_at(bps, s, li):
        # Post-cut stream bounding planes are one dict per (stream, level);
        # tolerate the [contour] list form just in case.
        if bps is None or s >= len(bps) or li < 0 or li >= len(bps[s]):
            return None
        bp = bps[s][li]
        if isinstance(bp, dict):
            return bp.get('scalar_value')
        if isinstance(bp, (list, tuple)) and bp and isinstance(bp[0], dict):
            return bp[0].get('scalar_value')
        return None

    rows = []
    for s in range(max_stream_count):
        n = len(other.stream_contours[s])
        ms = min(s, len(master_sel) - 1) if master_sel else 0
        src = master_sel[ms] if master_sel else []
        row = [False] * n

        # The follower was cut independently, so its raw level indices need not
        # line up with the master's. Both realize the same master scalar
        # schedule, so match each master-selected level to the follower level
        # with the same scalar value instead of copying the raw index.
        follower_scalars = [_scalar_at(other_bps, s, li) for li in range(n)]
        master_scalars = [_scalar_at(master_bps, ms, int(li)) for li in src]
        matched_by_scalar = (
            n > 0
            and all(x is not None for x in follower_scalars)
            and all(x is not None for x in master_scalars)
        )
        if matched_by_scalar:
            for m_scalar in master_scalars:
                best_j = min(range(n), key=lambda j: abs(follower_scalars[j] - m_scalar))
                row[best_j] = True
        else:
            # Fallback: raw index copy (original behavior).
            for li in src:
                if 0 <= int(li) < n:
                    row[int(li)] = True

        if n > 0:
            row[0] = True
            row[-1] = True

        sel_count = sum(1 for b in row if b)
        want_count = len(src) if src else sel_count
        # Endpoints are force-added, so the master count may already include
        # them; only warn when the follower genuinely cannot realize as many
        # distinct levels as the master selected.
        if src and sel_count < want_count:
            print(f"[{master_name}] Linked counterpart {other_name}: stream {s} "
                  f"realized {sel_count} of {want_count} master levels "
                  f"(follower has {n} raw levels) — cut structure differs")
        rows.append(row)
    other._level_select_checkboxes = rows
    other._apply_level_selection()
    other._save_level_select_post_state()
    other._level_select_anim_active = False
    other._level_select_window_open = False
    other._level_select_replayed = not defer
    other.linked_master_name = master_name
    print(f"[{master_name}] Linked counterpart {other_name}: copied selected levels")
    return True


def _ensure_level_selection_applied(v, name, obj, defer=False):
    """Apply an open level-select session before building fibers."""
    if getattr(obj, '_selected_stream_contours', None) is not None:
        return False
    if getattr(obj, '_level_select_checkboxes', None) is None:
        return False
    obj._start_level_select_animation(defer=True)
    obj._level_select_anim_pending_resume = False
    if getattr(obj, 'linked_drive_counterpart', False):
        _run_counterpart_step(v, name, obj, 8, defer=defer)
    print(f"[{name}] Applied pending contour selection before Build Fiber")
    return True


def _resample_linked_tendon_extensions(v, belly_name, belly, defer=False, include_counterpart=True):
    """Resample origin/insertion tendon contours linked to a belly component."""
    names = []

    def add_tendon_names(comp_name, comp_obj):
        if comp_obj is None:
            return
        _auto_fill_tendon_extension_names(v, comp_name, comp_obj)
        for attr in ('origin_tendon_extension_name', 'insertion_tendon_extension_name'):
            tendon_name = getattr(comp_obj, attr, '')
            if tendon_name and tendon_name in v.zygote_muscle_meshes and tendon_name not in names:
                names.append(tendon_name)

    add_tendon_names(belly_name, belly)
    if include_counterpart and getattr(belly, 'linked_drive_counterpart', False):
        other_name, other = _linked_counterpart(v, belly)
        if other is not None:
            add_tendon_names(other_name, other)

    did_any = False
    for tendon_name in names:
        tendon = v.zygote_muscle_meshes.get(tendon_name)
        if tendon is None:
            continue
        if tendon.contours is not None and len(tendon.contours) > 0 and tendon.bounding_planes is not None:
            tendon.resample_contours(base_samples=32, defer=defer)
            did_any = True
            print(f"[{belly_name}] Linked tendon {tendon_name}: resample done")
        else:
            print(f"[{belly_name}] Linked tendon {tendon_name}: needs contours before resample")
    return did_any


def _resample_source_for_linked_pair(obj):
    contours = getattr(obj, '_selected_stream_contours', None)
    planes = getattr(obj, '_selected_stream_bounding_planes', None)
    if contours is None or planes is None:
        contours = getattr(obj, 'stream_contours', None)
        planes = getattr(obj, 'stream_bounding_planes', None)
    if contours is None or planes is None:
        contours = getattr(obj, 'contours', None)
        planes = getattr(obj, 'bounding_planes', None)
    if contours is None or planes is None or len(contours) == 0 or len(planes) == 0:
        return None, None
    return contours, planes


def _copy_first_stream_for_resample(contours, planes):
    if contours is None or planes is None or len(contours) == 0 or len(planes) == 0:
        return None, None
    if len(contours) > 0 and isinstance(contours[0], np.ndarray):
        return (
            [np.array(c).copy() for c in contours],
            copy.deepcopy(planes),
        )
    return (
        [np.array(c).copy() for c in contours[0]],
        copy.deepcopy(planes[0]),
    )


def _assign_resampled_single_stream(obj, contours, params, fixed, types, stream_idx, defer=False):
    obj.contours_resampled = [[np.array(c).copy() for c in contours[stream_idx]]]
    obj.contours_resampled_params = copy.deepcopy([params[stream_idx]]) if params is not None else None
    obj.contours_resampled_fixed = copy.deepcopy([fixed[stream_idx]]) if fixed is not None else None
    obj.contours_resampled_types = copy.deepcopy([types[stream_idx]]) if types is not None else None
    obj._resample_anim_data = [[np.array(c).copy() for c in obj.contours_resampled[0]]]
    obj._resample_replayed = not defer
    obj.is_draw_resampled_vertices = False
    if defer:
        obj._build_fibers_replayed = False


def _resample_linked_counterpart_pair(v, master_name, master, defer=False):
    """Resample linked belly/counterpart as one two-stream problem, then split output."""
    if not getattr(master, 'linked_drive_counterpart', False):
        return False
    other_name, other = _linked_counterpart(v, master)
    if other is None:
        return False

    master_contours, master_planes = _resample_source_for_linked_pair(master)
    other_contours, other_planes = _resample_source_for_linked_pair(other)
    master_stream, master_plane_stream = _copy_first_stream_for_resample(master_contours, master_planes)
    other_stream, other_plane_stream = _copy_first_stream_for_resample(other_contours, other_planes)
    if master_stream is None or other_stream is None:
        return False
    if len(master_stream) == 0 or len(other_stream) == 0:
        return False

    saved = {}
    for attr in (
            'contours', 'bounding_planes', 'draw_contour_stream',
            'stream_contours', 'stream_bounding_planes',
            '_selected_stream_contours', '_selected_stream_bounding_planes',
            'contours_resampled', 'contours_resampled_params',
            'contours_resampled_fixed', 'contours_resampled_types',
            '_resample_anim_data', '_resample_replayed'):
        saved[attr] = getattr(master, attr, None)

    assigned = False
    try:
        master.contours = [master_stream, other_stream]
        master.bounding_planes = [master_plane_stream, other_plane_stream]
        master.draw_contour_stream = [
            [True] * len(master_stream),
            [True] * len(other_stream),
        ]
        master.stream_contours = None
        master.stream_bounding_planes = None
        master._selected_stream_contours = None
        master._selected_stream_bounding_planes = None
        master.resample_contours(base_samples=32, defer=defer)

        contours = getattr(master, 'contours_resampled', None)
        params = getattr(master, 'contours_resampled_params', None)
        fixed = getattr(master, 'contours_resampled_fixed', None)
        types = getattr(master, 'contours_resampled_types', None)
        if contours is None or len(contours) < 2:
            return False

        _assign_resampled_single_stream(master, contours, params, fixed, types, 0, defer=defer)
        _assign_resampled_single_stream(other, contours, params, fixed, types, 1, defer=defer)
        assigned = True
        print(f"[{master_name}] Linked counterpart {other_name}: joint resample done")
        return True
    finally:
        restore_attrs = (
            'contours', 'bounding_planes', 'draw_contour_stream',
            'stream_contours', 'stream_bounding_planes',
            '_selected_stream_contours', '_selected_stream_bounding_planes')
        if not assigned:
            restore_attrs = tuple(saved.keys())
        for attr in restore_attrs:
            setattr(master, attr, saved[attr])


def _resample_contours_with_links(v, name, obj, defer=False):
    if _resample_linked_counterpart_pair(v, name, obj, defer=defer):
        return True
    obj.resample_contours(base_samples=32, defer=defer)
    _run_counterpart_step(v, name, obj, 10, defer=defer)
    return True


def _run_counterpart_step(v, master_name, master, step, defer=False):
    if not getattr(master, 'linked_drive_counterpart', False):
        return False
    other_name, other = _linked_counterpart(v, master)
    if other is None:
        return False

    try:
        if step == 5:
            if other.contours is not None and len(other.contours) > 0:
                other.smoothen_all(defer=defer)
                print(f"[{master_name}] Linked counterpart {other_name}: smooth done")
                return True
        elif step == 6:
            if other.contours is not None and len(other.contours) > 0 and other.bounding_planes is not None:
                if defer:
                    other.cut_streams_animated(
                        defer=True,
                        cut_method=other.cutting_method,
                        muscle_name=other_name,
                    )
                else:
                    other.cut_streams(cut_method=other.cutting_method, muscle_name=other_name)
                print(f"[{master_name}] Linked counterpart {other_name}: cut done")
                return True
        elif step == 7:
            if getattr(other, 'stream_contours', None) is not None:
                other.stream_smoothen_all(defer=defer)
                print(f"[{master_name}] Linked counterpart {other_name}: stream smooth done")
                return True
        elif step == 8:
            return _sync_counterpart_level_selection(v, master_name, master, defer=defer)
        elif step == 9:
            if getattr(other, 'stream_contours', None) is not None:
                if getattr(other, '_selected_stream_contours', None) is None and getattr(master, 'stream_selected_levels', None) is not None:
                    _sync_counterpart_level_selection(v, master_name, master, defer=defer)
                other._belly_waypoints_before_tendon_extension = None
                other.build_fibers(skeleton_meshes=v.zygote_skeleton_meshes, defer=defer)
                if (getattr(other, 'enable_tendon_extension', True)
                        and (getattr(other, 'origin_tendon_extension_name', '')
                             or getattr(other, 'insertion_tendon_extension_name', ''))):
                    _extend_belly_fibers_with_tendons(v, other_name, other)
                if defer:
                    other._level_select_replayed = False
                print(f"[{master_name}] Linked counterpart {other_name}: build fibers done")
                return True
        elif step == 10:
            if other.contours is not None and len(other.contours) > 0 and other.bounding_planes is not None:
                other.resample_contours(base_samples=32, defer=defer)
                if defer:
                    other._build_fibers_replayed = False
                print(f"[{master_name}] Linked counterpart {other_name}: resample done")
                return True
        elif step == 11:
            if other.contours is not None and len(other.contours) > 0 and other.draw_contour_stream is not None:
                if _skip_non_owner_connected_mesh(v, other_name, other, "Build Contour Mesh"):
                    return True
                _prepare_connected_contour_mesh_source(v, other_name, other, include_counterpart=False)
                other.build_contour_mesh(defer=defer)
                if defer:
                    other._resample_replayed = False
                print(f"[{master_name}] Linked counterpart {other_name}: contour mesh done")
                return True
        elif step == 12:
            if _skip_non_owner_connected_mesh(v, other_name, other, "Tetrahedralize"):
                return True
            tet_ok = _tetrahedralize_single_contour_mesh(v, other_name, other, defer=defer)
            if tet_ok and other.tet_vertices is not None:
                if defer:
                    other._extract_internal_tet_edges()
                    other._classify_tet_faces_into_bands()
                    other._tetrahedralize_replayed = False
                else:
                    other.is_draw_contours = False
                    other.is_draw_tet_mesh = True
                    other._tetrahedralize_replayed = True
            status = "done" if tet_ok else "failed"
            print(f"[{master_name}] Linked counterpart {other_name}: tetrahedralize {status}")
            return tet_ok
    except Exception as e:
        print(f"[{master_name}] Linked counterpart {other_name}: step {step} error: {e}")
        traceback.print_exc()
    return False


def _copy_fiber_samples_for_stream(target_count, source_samples):
    if source_samples is None or len(source_samples) == 0:
        return None
    samples = []
    for i in range(target_count):
        samples.append(np.asarray(source_samples[min(i, len(source_samples) - 1)], dtype=np.float64).copy())
    return samples


def _prepare_tendon_waypoints_from_belly(tendon, belly, reverse=False, attach=None):
    if tendon is None:
        return None
    if getattr(tendon, 'bounding_planes', None) is None or len(tendon.bounding_planes) == 0:
        print("Tendon extension: tendon has no bounding planes")
        return None
    if getattr(tendon, 'contours', None) is None or len(tendon.contours) == 0:
        print("Tendon extension: tendon has no contours")
        return None
    if getattr(belly, 'fiber_architecture', None) is None or len(belly.fiber_architecture) == 0:
        print("Tendon extension: belly has no fiber architecture")
        return None

    # Unlike a belly, a tendon never runs build_fibers, so its contours/
    # bounding_planes still hold the FULL set of cut levels while the rest of
    # the extension reads _selected_stream_* (the level-selected subset). Reduce
    # the tendon to the selected levels here so the regenerated waypoints have
    # the same level count as the contours pulled by _stream_major_contour_source
    # — otherwise the inspect region boundary (built from waypoints) lands at the
    # full-level count and diverges between two linked tendons with equal
    # selected counts but different total cut levels.
    sel_c = getattr(tendon, '_selected_stream_contours', None)
    sel_bp = getattr(tendon, '_selected_stream_bounding_planes', None)
    work_contours = sel_c if sel_c and sel_bp else tendon.contours
    work_planes = sel_bp if sel_c and sel_bp else tendon.bounding_planes

    # NOTE: seam alignment is no longer done per-tendon here (smoothing a tendon
    # in isolation gave weird frames). Alignment now happens on the combined
    # origin-tendon + belly + insertion-tendon sequence in
    # _extend_belly_fibers_with_tendons. These waypoints are only used for
    # orientation (flip detection) and region sizing.

    # Regeneration writes several caches. Work on copied selected data and put
    # every touched field back afterward so extending a belly cannot change the
    # standalone tendon object or its later visualization/reprocessing state.
    touched = (
        'contours', 'bounding_planes', 'fiber_architecture', 'waypoints',
        'normalized_Qs', 'mvc_weights', '_stream_endpoints',
        'unit_circle_triangulations', 'fiber_embeddings',
        'triangulated_deformed_2d')
    missing = object()
    saved = {name: getattr(tendon, name, missing) for name in touched}
    try:
        tendon.contours = _copy_stream_levels(work_contours, len(work_contours))
        tendon.bounding_planes = _copy_stream_levels(work_planes, len(work_planes))
        tendon.fiber_architecture = _copy_fiber_samples_for_stream(
            len(tendon.bounding_planes), belly.fiber_architecture)
        if tendon.fiber_architecture is None:
            return None
        tendon._regenerate_waypoints_from_fibers(
            skeleton_meshes=None, propagate_corners=False)
        streams = []
        for stream in tendon.waypoints:
            levels = [np.asarray(wp, dtype=np.float64).copy() for wp in stream]
            if reverse:
                levels = list(reversed(levels))
            streams.append(levels)
        return streams
    finally:
        for name, value in saved.items():
            if value is missing:
                if hasattr(tendon, name):
                    delattr(tendon, name)
            else:
                setattr(tendon, name, value)


def _orient_tendon_streams_to_belly(tendon_streams, belly_streams, attach):
    """Flip tendon streams when their boundary nearest to belly is on the wrong end."""
    if not tendon_streams or not belly_streams:
        return tendon_streams, []
    oriented = []
    flip_mask = []
    flips = 0
    for s, levels in enumerate(tendon_streams):
        levels = [np.asarray(wp, dtype=np.float64).copy() for wp in levels]
        if len(levels) < 2:
            oriented.append(levels)
            flip_mask.append(False)
            continue
        belly_levels = belly_streams[min(s, len(belly_streams) - 1)]
        if not belly_levels:
            oriented.append(levels)
            flip_mask.append(False)
            continue
        belly_endpoint = np.asarray(
            belly_levels[0] if attach == 'origin' else belly_levels[-1],
            dtype=np.float64)
        d_first = float(np.linalg.norm(np.mean(levels[0], axis=0) - np.mean(belly_endpoint, axis=0)))
        d_last = float(np.linalg.norm(np.mean(levels[-1], axis=0) - np.mean(belly_endpoint, axis=0)))
        should_flip = d_first < d_last if attach == 'origin' else d_last < d_first
        if should_flip:
            levels = list(reversed(levels))
            flips += 1
        oriented.append(levels)
        flip_mask.append(should_flip)
    if flips:
        print(f"Tendon extension: auto-flipped {flips}/{len(tendon_streams)} {attach} tendon stream(s)")
    return oriented, flip_mask


def _flip_stream_levels_by_mask(streams, flip_mask):
    if streams is None or not flip_mask:
        return streams
    out = []
    for s, levels in enumerate(streams):
        copied = [copy.deepcopy(item) for item in levels]
        if flip_mask[min(s, len(flip_mask) - 1)]:
            copied = list(reversed(copied))
        out.append(copied)
    return out


def _trim_tendon_seam_level(streams, end):
    """Drop the tendon-side junction level that duplicates the belly boundary.

    After orientation the origin tendon attaches at its last level and the
    insertion tendon at its first level; that level coincides with the belly
    origin/insertion boundary. Drop it (keep the belly copy) so the extended
    stream has no repeated level at the seam. ``end`` is 'last' for the origin
    tendon, 'first' for the insertion tendon.
    """
    if streams is None:
        return None
    out = []
    for levels in streams:
        if levels is not None and len(levels) > 1:
            levels = levels[:-1] if end == 'last' else levels[1:]
        out.append(levels)
    return out


def _copy_stream_levels(src, target_count, reverse=False):
    if src is None:
        return None
    copied = []
    for i in range(target_count):
        stream = src[min(i, len(src) - 1)] if len(src) > 0 else []
        levels = []
        for item in stream:
            if isinstance(item, np.ndarray):
                levels.append(item.copy())
            elif isinstance(item, dict):
                levels.append(dict(item))
            else:
                levels.append(copy.deepcopy(item))
        if reverse:
            levels = list(reversed(levels))
        copied.append(levels)
    return copied


def _stream_major_contour_source(obj, prefer_live=False):
    if obj is None:
        return None, None
    if prefer_live:
        contours = getattr(obj, 'contours', None)
        planes = getattr(obj, 'bounding_planes', None)
        if contours is not None and len(contours) > 0:
            return contours, planes
    contours = getattr(obj, '_selected_stream_contours', None)
    planes = getattr(obj, '_selected_stream_bounding_planes', None)
    if contours is None or len(contours) == 0:
        contours = getattr(obj, 'stream_contours', None)
        planes = getattr(obj, 'stream_bounding_planes', None)
    if contours is None or len(contours) == 0:
        contours = getattr(obj, 'contours', None)
        planes = getattr(obj, 'bounding_planes', None)
    return contours, planes


def _clear_tendon_extension(obj):
    base = getattr(obj, '_belly_waypoints_before_tendon_extension', None)
    if base is not None:
        obj.waypoints = [[np.asarray(wp, dtype=np.float64).copy() for wp in stream]
                         for stream in base]
        obj.waypoints_original = [[np.asarray(wp, dtype=np.float64).copy() for wp in stream]
                                  for stream in obj.waypoints]
    obj._belly_waypoints_before_tendon_extension = None
    obj._tendon_extended_inspect_contours = None
    obj._tendon_extended_inspect_bounding_planes = None
    obj._tendon_extended_inspect_waypoints = None
    obj._connected_contour_mesh_source = None
    obj._connected_contour_mesh_params = None
    obj._connected_contour_mesh_fixed = None
    obj._connected_contour_mesh_types = None
    obj._connected_contour_mesh_provenance = None
    obj._connected_contour_mesh_components = None
    obj._connected_component_fibers = None
    obj._tendon_origin_auto_flip_mask = None
    obj._tendon_insertion_auto_flip_mask = None
    for tendon_obj in getattr(obj, '_tendon_alignment_display_objects', []) or []:
        if getattr(tendon_obj, '_aligned_extension_owner_id', None) == id(obj):
            tendon_obj._aligned_extension_bounding_planes = None
            tendon_obj._aligned_extension_owner_id = None
    obj._tendon_alignment_display_objects = None
    obj.waypoint_level_regions = None
    obj.tendon_extended_fibers = False
    obj._fiber_draw_dirty = True
    obj._fiber_draw_pts = None
    obj._fiber_draw_lines = None


def _part_mesh_source(obj, n_streams, reverse=False):
    contours = getattr(obj, 'contours_resampled', None)
    params = getattr(obj, 'contours_resampled_params', None)
    fixed = getattr(obj, 'contours_resampled_fixed', None)
    types = getattr(obj, 'contours_resampled_types', None)
    if contours is None or len(contours) == 0:
        contours = getattr(obj, 'contours', None)
        params = fixed = types = None
    if contours is None or len(contours) == 0:
        return None, None, None, None
    out_c = _copy_stream_levels(contours, n_streams, reverse=reverse)
    out_p = _copy_stream_levels(params, n_streams, reverse=reverse) if params is not None else None
    out_f = _copy_stream_levels(fixed, n_streams, reverse=reverse) if fixed is not None else None
    out_t = _copy_stream_levels(types, n_streams, reverse=reverse) if types is not None else None
    return out_c, out_p, out_f, out_t


def _append_part_source(dst_c, dst_p, dst_f, dst_t, provenance,
                        src_c, src_p, src_f, src_t, part_name, component_name,
                        stream_offset=0):
    if not src_c:
        return
    for s, stream in enumerate(src_c):
        dst_s = stream_offset + s
        dst_c[dst_s].extend(stream)
        n = len(stream)
        if dst_p is not None:
            dst_p[dst_s].extend(src_p[s] if src_p and s < len(src_p) else [None] * n)
        if dst_f is not None:
            dst_f[dst_s].extend(src_f[s] if src_f and s < len(src_f) else [None] * n)
        if dst_t is not None:
            dst_t[dst_s].extend(src_t[s] if src_t and s < len(src_t) else [None] * n)
        provenance[dst_s].extend([
            {'part': part_name, 'component': component_name, 'local_level': i}
            for i in range(n)
        ])


def _prepare_connected_contour_mesh_source(v, belly_name, belly, include_counterpart=True):
    """Prepare tendon+belly contour source for contour mesh/tet build.

    This does not overwrite belly.contours or belly.contours_resampled. The
    contour mesh builder reads these temporary connected source fields.
    """
    if not getattr(belly, 'tendon_extended_fibers', False):
        belly._connected_contour_mesh_source = None
        belly._connected_contour_mesh_params = None
        belly._connected_contour_mesh_fixed = None
        belly._connected_contour_mesh_types = None
        belly._connected_contour_mesh_provenance = None
        belly._connected_contour_mesh_components = None
        belly._connected_component_fibers = None
        belly.waypoint_level_regions = None
        return False

    def gather_component(comp_name, comp_obj):
        _auto_fill_tendon_extension_names(v, comp_name, comp_obj)
        base_src = getattr(comp_obj, 'contours_resampled', None)
        if base_src is None or len(base_src) == 0:
            base_src = getattr(comp_obj, 'contours', None)
        if base_src is None or len(base_src) == 0:
            return None
        n_streams = len(base_src)
        base_c, base_p, base_f, base_t = _part_mesh_source(comp_obj, n_streams)
        if not base_c:
            return None
        origin_name = getattr(comp_obj, 'origin_tendon_extension_name', '')
        insertion_name = getattr(comp_obj, 'insertion_tendon_extension_name', '')
        origin = v.zygote_muscle_meshes.get(origin_name) if origin_name else None
        insertion = v.zygote_muscle_meshes.get(insertion_name) if insertion_name else None
        origin_rev = bool(getattr(comp_obj, 'origin_tendon_reverse', False))
        insertion_rev = bool(getattr(comp_obj, 'insertion_tendon_reverse', False))
        origin_c, origin_p, origin_f, origin_t = (
            _part_mesh_source(origin, n_streams, reverse=origin_rev)
            if origin is not None else (None, None, None, None)
        )
        insertion_c, insertion_p, insertion_f, insertion_t = (
            _part_mesh_source(insertion, n_streams, reverse=insertion_rev)
            if insertion is not None else (None, None, None, None)
        )
        origin_mask = getattr(comp_obj, '_tendon_origin_auto_flip_mask', None)
        insertion_mask = getattr(comp_obj, '_tendon_insertion_auto_flip_mask', None)
        origin_c = _flip_stream_levels_by_mask(origin_c, origin_mask)
        origin_p = _flip_stream_levels_by_mask(origin_p, origin_mask)
        origin_f = _flip_stream_levels_by_mask(origin_f, origin_mask)
        origin_t = _flip_stream_levels_by_mask(origin_t, origin_mask)
        insertion_c = _flip_stream_levels_by_mask(insertion_c, insertion_mask)
        insertion_p = _flip_stream_levels_by_mask(insertion_p, insertion_mask)
        insertion_f = _flip_stream_levels_by_mask(insertion_f, insertion_mask)
        insertion_t = _flip_stream_levels_by_mask(insertion_t, insertion_mask)
        # Drop the tendon-side seam level that duplicates the belly boundary.
        origin_c = _trim_tendon_seam_level(origin_c, 'last')
        origin_p = _trim_tendon_seam_level(origin_p, 'last')
        origin_f = _trim_tendon_seam_level(origin_f, 'last')
        origin_t = _trim_tendon_seam_level(origin_t, 'last')
        insertion_c = _trim_tendon_seam_level(insertion_c, 'first')
        insertion_p = _trim_tendon_seam_level(insertion_p, 'first')
        insertion_f = _trim_tendon_seam_level(insertion_f, 'first')
        insertion_t = _trim_tendon_seam_level(insertion_t, 'first')
        return {
            'name': comp_name,
            'obj': comp_obj,
            'n_streams': n_streams,
            'origin_name': origin_name,
            'insertion_name': insertion_name,
            'base': (base_c, base_p, base_f, base_t),
            'origin': (origin_c, origin_p, origin_f, origin_t),
            'insertion': (insertion_c, insertion_p, insertion_f, insertion_t),
        }

    components = []
    primary = gather_component(belly_name, belly)
    if primary is not None:
        components.append(primary)

    if include_counterpart and getattr(belly, 'linked_drive_counterpart', False):
        other_name, other = _linked_counterpart(v, belly)
        if other is not None:
            if not getattr(other, 'tendon_extended_fibers', False):
                _extend_belly_and_linked_counterpart_fibers(v, belly_name, belly)
            other_component = gather_component(other_name, other)
            if other_component is not None:
                components.append(other_component)

    if not components:
        return False

    total_streams = sum(comp['n_streams'] for comp in components)
    have_params = all(
        comp['base'][1] is not None
        and (comp['origin'][0] is None or comp['origin'][1] is not None)
        and (comp['insertion'][0] is None or comp['insertion'][1] is not None)
        for comp in components
    )
    have_fixed = have_params and all(
        comp['base'][2] is not None
        and (comp['origin'][0] is None or comp['origin'][2] is not None)
        and (comp['insertion'][0] is None or comp['insertion'][2] is not None)
        for comp in components
    )
    have_types = all(
        comp['base'][3] is not None
        and (comp['origin'][0] is None or comp['origin'][3] is not None)
        and (comp['insertion'][0] is None or comp['insertion'][3] is not None)
        for comp in components
    )

    connected_c = [[] for _ in range(total_streams)]
    connected_p = [[] for _ in range(total_streams)] if have_params else None
    connected_f = [[] for _ in range(total_streams)] if have_fixed else None
    connected_t = [[] for _ in range(total_streams)] if have_types else None
    provenance = [[] for _ in range(total_streams)]

    stream_offset = 0
    component_ranges = []
    component_fibers = []
    for comp in components:
        origin_c, origin_p, origin_f, origin_t = comp['origin']
        base_c, base_p, base_f, base_t = comp['base']
        insertion_c, insertion_p, insertion_f, insertion_t = comp['insertion']
        _append_part_source(connected_c, connected_p, connected_f, connected_t, provenance,
                            origin_c, origin_p, origin_f, origin_t,
                            'origin_tendon', comp['origin_name'] or '',
                            stream_offset=stream_offset)
        _append_part_source(connected_c, connected_p, connected_f, connected_t, provenance,
                            base_c, base_p, base_f, base_t,
                            'belly', comp['name'],
                            stream_offset=stream_offset)
        _append_part_source(connected_c, connected_p, connected_f, connected_t, provenance,
                            insertion_c, insertion_p, insertion_f, insertion_t,
                            'insertion_tendon', comp['insertion_name'] or '',
                            stream_offset=stream_offset)
        component_ranges.append({
            'component': comp['name'],
            'stream_start': stream_offset,
            'stream_end': stream_offset + comp['n_streams'],
            'origin_tendon': comp['origin_name'] or '',
            'insertion_tendon': comp['insertion_name'] or '',
        })
        comp_obj = comp['obj']
        component_fibers.append({
            'component': comp['name'],
            'stream_start': stream_offset,
            'stream_end': stream_offset + comp['n_streams'],
            'waypoints': _copy_stream_levels(
                getattr(comp_obj, 'waypoints', None), comp['n_streams']),
            'waypoints_original': _copy_stream_levels(
                getattr(comp_obj, 'waypoints_original', None), comp['n_streams']),
            'waypoint_level_regions': copy.deepcopy(
                getattr(comp_obj, 'waypoint_level_regions', None)),
            'fiber_architecture': [
                np.asarray(f, dtype=np.float64).copy()
                for f in getattr(comp_obj, 'fiber_architecture', []) or []
            ],
        })
        stream_offset += comp['n_streams']

    if not all(len(stream) >= 2 for stream in connected_c):
        print(f"[{belly_name}] Connected contour mesh needs at least 2 levels per stream")
        return False

    belly._connected_contour_mesh_source = connected_c
    belly._connected_contour_mesh_params = connected_p
    belly._connected_contour_mesh_fixed = connected_f
    belly._connected_contour_mesh_types = connected_t
    belly._connected_contour_mesh_provenance = provenance
    belly._connected_contour_mesh_components = component_ranges
    belly._connected_component_fibers = component_fibers
    belly.waypoint_level_regions = provenance
    if len(component_ranges) > 1:
        for comp in component_ranges:
            comp_name = comp.get('component')
            if comp_name and comp_name != belly_name:
                comp_obj = v.zygote_muscle_meshes.get(comp_name)
                if comp_obj is not None:
                    comp_obj.is_draw_contour_mesh = False
                    comp_obj.is_draw_tet_mesh = False
    print(f"[{belly_name}] Connected contour mesh source: "
          f"{len(components)} component(s), {total_streams} stream(s)")
    return True


def _smooth_and_regenerate_combined(belly, combined_contours, combined_planes,
                                    belly_anchor_ranges=None,
                                    belly_waypoints=None):
    """Align the whole origin-tendon + belly + insertion-tendon sequence as one
    chain and regenerate its fibers in a single pass.

    Keeps the built belly frames fixed, aligns each tendon outward from its
    adjacent belly seam, then regenerates waypoints with
    corner correspondence re-propagated across the full sequence — no per-tendon
    frame guessing, no seam twist. Operates on deep copies and restores the
    belly's own contour state afterward; returns (waypoints, smoothed_contours,
    smoothed_planes) for the combined sequence.
    """
    if len(combined_contours) != len(combined_planes):
        raise ValueError("combined contour/plane stream counts differ")
    for stream_i, (contours, planes) in enumerate(zip(combined_contours, combined_planes)):
        if len(contours) != len(planes):
            raise ValueError(
                f"combined stream {stream_i} has {len(contours)} contours but "
                f"{len(planes)} bounding planes")

    combined_contours = [[np.asarray(c, dtype=np.float64).copy() for c in s]
                         for s in combined_contours]
    combined_planes = [[copy.deepcopy(bp) for bp in s] for s in combined_planes]

    belly_snapshots = []
    for stream_i, (contours, planes) in enumerate(zip(combined_contours, combined_planes)):
        start, end = (belly_anchor_ranges[stream_i]
                      if belly_anchor_ranges and stream_i < len(belly_anchor_ranges)
                      else (0, len(planes)))
        belly_snapshots.append((
            start, end,
            [np.asarray(c, dtype=np.float64).copy() for c in contours[start:end]],
            [copy.deepcopy(bp) for bp in planes[start:end]]))
    for si, stream in enumerate(combined_planes):
        for li, bp in enumerate(stream):
            if isinstance(bp, dict) and li < len(combined_contours[si]):
                bp['contour_vertices'] = combined_contours[si][li]

    saved = {a: getattr(belly, a, None) for a in (
        'contours', 'bounding_planes', 'stream_contours', 'stream_bounding_planes',
        'max_stream_count', 'draw_contour_stream', 'normalized_Qs', 'mvc_weights',
        'waypoints', '_stream_endpoints')}
    try:
        belly.stream_contours = combined_contours
        belly.stream_bounding_planes = combined_planes
        belly.contours = combined_contours
        belly.bounding_planes = combined_planes
        belly.max_stream_count = len(combined_contours)
        belly.draw_contour_stream = [[True] * len(s) for s in combined_contours]
        # Both copies of a physically shared seam contour must use exactly the
        # same chart. The tendon copy remains in the fiber sequence as a zero-gap
        # registration level; downstream connected mesh construction removes it.
        for stream_i, (start, end, _, _) in enumerate(belly_snapshots):
            if start > 0:
                belly.stream_contours[stream_i][start - 1] = np.asarray(
                    belly.stream_contours[stream_i][start], dtype=np.float64).copy()
                belly.stream_bounding_planes[stream_i][start - 1] = copy.deepcopy(
                    belly.stream_bounding_planes[stream_i][start])
            if end < len(belly.stream_bounding_planes[stream_i]):
                belly.stream_contours[stream_i][end] = np.asarray(
                    belly.stream_contours[stream_i][end - 1], dtype=np.float64).copy()
                belly.stream_bounding_planes[stream_i][end] = copy.deepcopy(
                    belly.stream_bounding_planes[stream_i][end - 1])
        # Preserve the already-built belly frames. Make tendon z signs continuous
        # and continuously project the fixed belly x axis into each tendon plane,
        # working outward from each seam.
        for stream_i, bp_stream in enumerate(belly.stream_bounding_planes):
            start, end = belly_snapshots[stream_i][:2]

            def _align_level_to_reference(level_i, reference_i, forward_hint):
                bp = bp_stream[level_i]
                ref = bp_stream[reference_i]
                z_axis = np.asarray(bp['basis_z'], dtype=np.float64)
                z_axis /= np.linalg.norm(z_axis) + 1e-10
                forward_hint = np.asarray(forward_hint, dtype=np.float64)
                if (np.linalg.norm(forward_hint) > 1e-10
                        and np.dot(z_axis, forward_hint) < 0):
                    z_axis = -z_axis

                ref_z = np.asarray(ref['basis_z'], dtype=np.float64)
                ref_z /= np.linalg.norm(ref_z) + 1e-10
                ref_x = np.asarray(ref['basis_x'], dtype=np.float64)

                # Rotation-minimizing transport: rotate the complete reference
                # frame by the smallest 3D rotation carrying ref_z onto z_axis.
                # Simple projection can introduce visible in-plane drift on the
                # sharply bending origin tendon.
                cross_z = np.cross(ref_z, z_axis)
                sin_angle = np.linalg.norm(cross_z)
                cos_angle = float(np.clip(np.dot(ref_z, z_axis), -1.0, 1.0))
                if sin_angle > 1e-8:
                    rot_axis = cross_z / sin_angle
                    x_axis = (ref_x * cos_angle
                              + np.cross(rot_axis, ref_x) * sin_angle
                              + rot_axis * np.dot(rot_axis, ref_x)
                              * (1.0 - cos_angle))
                elif cos_angle >= 0.0:
                    x_axis = ref_x.copy()
                else:
                    # Rare 180-degree case: rotate around the reference y axis.
                    rot_axis = np.asarray(ref['basis_y'], dtype=np.float64)
                    rot_axis /= np.linalg.norm(rot_axis) + 1e-10
                    x_axis = -ref_x + 2.0 * rot_axis * np.dot(rot_axis, ref_x)

                x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
                if np.linalg.norm(x_axis) < 1e-8:
                    ref_y = np.asarray(ref['basis_y'], dtype=np.float64)
                    x_axis = np.cross(ref_y, z_axis)
                x_axis /= np.linalg.norm(x_axis) + 1e-10
                y_axis = np.cross(z_axis, x_axis)
                y_axis /= np.linalg.norm(y_axis) + 1e-10
                x_axis = np.cross(y_axis, z_axis)
                x_axis /= np.linalg.norm(x_axis) + 1e-10
                bp['basis_x'] = x_axis
                bp['basis_y'] = y_axis
                bp['basis_z'] = z_axis

            # Shared seam levels (start-1 and end) are exact belly copies. Keep
            # them fixed; orient every other tendon normal toward the next level
            # in the origin-to-insertion sequence.
            for level_i in range(start - 2, -1, -1):
                forward = (np.asarray(bp_stream[level_i + 1]['mean'])
                           - np.asarray(bp_stream[level_i]['mean']))
                _align_level_to_reference(level_i, level_i + 1, forward)
            for level_i in range(end + 1, len(bp_stream)):
                if level_i + 1 < len(bp_stream):
                    forward = (np.asarray(bp_stream[level_i + 1]['mean'])
                               - np.asarray(bp_stream[level_i]['mean']))
                else:
                    forward = (np.asarray(bp_stream[level_i]['mean'])
                               - np.asarray(bp_stream[level_i - 1]['mean']))
                _align_level_to_reference(level_i, level_i - 1, forward)

            # Refit only tendon planes after their axes change. Belly contours,
            # planes, matches, and corner labels remain byte-for-byte untouched.
            tendon_levels = (list(range(0, max(0, start - 1)))
                             + list(range(min(len(bp_stream), end + 1), len(bp_stream))))
            for level_i in tendon_levels:
                new_contour = belly._recompute_bounding_plane_after_axis_change(
                    bp_stream[level_i], belly.stream_contours[stream_i][level_i])
                belly.stream_contours[stream_i][level_i] = new_contour
                bp_stream[level_i].pop('corner_indices', None)

            for label, levels in (
                    ('origin', range(0, max(0, start - 1))),
                    ('insertion', range(min(len(bp_stream), end + 1), len(bp_stream)))):
                forward_dots = []
                for level_i in levels:
                    next_i = min(level_i + 1, len(bp_stream) - 1)
                    if next_i == level_i:
                        continue
                    forward = (np.asarray(bp_stream[next_i]['mean'])
                               - np.asarray(bp_stream[level_i]['mean']))
                    norm = np.linalg.norm(forward)
                    if norm > 1e-10:
                        forward_dots.append(float(np.dot(
                            bp_stream[level_i]['basis_z'], forward / norm)))
                if forward_dots:
                    print(f"  [Tendon z forward] Stream {stream_i} {label}: "
                          f"min_dot={min(forward_dots):.4f}")

            for label, first, last in (
                    ('origin', 0, start),
                    ('insertion', end, len(bp_stream))):
                xy_dots = []
                for level_i in range(first, max(first, last - 1)):
                    xy_dots.append((
                        float(np.dot(bp_stream[level_i]['basis_x'],
                                     bp_stream[level_i + 1]['basis_x'])),
                        float(np.dot(bp_stream[level_i]['basis_y'],
                                     bp_stream[level_i + 1]['basis_y']))))
                if xy_dots:
                    print(f"  [Tendon xy transport] Stream {stream_i} {label}: "
                          f"min_dot(x,y)=({min(d[0] for d in xy_dots):.4f}, "
                          f"{min(d[1] for d in xy_dots):.4f})")

            for label, left, right in (
                    ('origin/belly', start - 1, start),
                    ('belly/insertion', end - 1, end)):
                if 0 <= left < len(bp_stream) and 0 <= right < len(bp_stream):
                    dots = [float(np.dot(bp_stream[left][key], bp_stream[right][key]))
                            for key in ('basis_x', 'basis_y', 'basis_z')]
                    print(f"  [Seam align] Stream {stream_i} {label}: "
                          f"dot(x,y,z)=({dots[0]:.4f}, {dots[1]:.4f}, {dots[2]:.4f})")

        # Reassert the authoritative belly snapshots defensively.
        for stream_i, (start, end, snap_contours, snap_planes) in enumerate(belly_snapshots):
            belly.stream_contours[stream_i][start:end] = snap_contours
            belly.stream_bounding_planes[stream_i][start:end] = snap_planes
        belly.contours = belly.stream_contours
        belly.bounding_planes = belly.stream_bounding_planes
        # Generate all levels once without global corner propagation. Register
        # each tendon chart from its *actual adjacent belly contour* so origin
        # and insertion do not share an unrelated remote reference.
        print("  [Tendon chart align v2] Registering origin and insertion from their belly seams")
        belly._regenerate_waypoints_from_fibers(
            skeleton_meshes=None, propagate_corners=False)
        origin_refs = []
        origin_targets = []
        insertion_refs = []
        insertion_targets = []
        for stream_i, (start, end, _, _) in enumerate(belly_snapshots):
            stream_len = len(belly.stream_bounding_planes[stream_i])
            origin_refs.append(start if start > 0 else -1)
            origin_targets.append((0, max(0, start - 1)))
            insertion_refs.append(end - 1 if end < stream_len else -1)
            insertion_targets.append((min(stream_len, end + 1), stream_len))
        if any(ref >= 0 for ref in origin_refs):
            belly._propagate_corner_correspondences(
                explicit_reference_levels=origin_refs,
                target_level_ranges=origin_targets,
                full_uv_matching=True)
        if any(ref >= 0 for ref in insertion_refs):
            belly._propagate_corner_correspondences(
                explicit_reference_levels=insertion_refs,
                target_level_ranges=insertion_targets,
                full_uv_matching=True)
        wp = [[np.asarray(w, dtype=np.float64).copy() for w in stream]
              for stream in belly.waypoints]
        if belly_waypoints is not None:
            for stream_i, stream_waypoints in enumerate(belly_waypoints):
                if stream_i >= len(wp):
                    break
                start, end = belly_snapshots[stream_i][:2]
                if end - start == len(stream_waypoints):
                    wp[stream_i][start:end] = [
                        np.asarray(w, dtype=np.float64).copy()
                        for w in stream_waypoints]
                    if start > 0 and stream_waypoints:
                        wp[stream_i][start - 1] = np.asarray(
                            stream_waypoints[0], dtype=np.float64).copy()
                    if end < len(wp[stream_i]) and stream_waypoints:
                        wp[stream_i][end] = np.asarray(
                            stream_waypoints[-1], dtype=np.float64).copy()
        smoothed_contours = [[np.asarray(c, dtype=np.float64).copy() for c in s]
                             for s in belly.stream_contours]
        smoothed_planes = [[copy.deepcopy(bp) for bp in s]
                           for s in belly.stream_bounding_planes]
    finally:
        for a, val in saved.items():
            setattr(belly, a, val)
    return wp, smoothed_contours, smoothed_planes


def _extend_belly_fibers_with_tendons(v, belly_name, belly):
    if getattr(belly, 'waypoints', None) is None or len(belly.waypoints) == 0:
        print(f"[{belly_name}] Tendon extension needs belly fibers first")
        return False

    _auto_fill_tendon_extension_names(v, belly_name, belly)
    origin_name = getattr(belly, 'origin_tendon_extension_name', '')
    insertion_name = getattr(belly, 'insertion_tendon_extension_name', '')
    origin = v.zygote_muscle_meshes.get(origin_name) if origin_name else None
    insertion = v.zygote_muscle_meshes.get(insertion_name) if insertion_name else None
    if origin is None and insertion is None:
        print(f"[{belly_name}] Tendon extension: choose origin and/or insertion tendon")
        return False

    # Restore unextended belly waypoints before rebuilding the composite.
    base = getattr(belly, '_belly_waypoints_before_tendon_extension', None)
    if base is None:
        base = [[np.asarray(wp, dtype=np.float64).copy() for wp in stream]
                for stream in belly.waypoints]
        belly._belly_waypoints_before_tendon_extension = base
    else:
        base = [[np.asarray(wp, dtype=np.float64).copy() for wp in stream]
                for stream in base]

    origin_streams = _prepare_tendon_waypoints_from_belly(
        origin, belly, reverse=bool(getattr(belly, 'origin_tendon_reverse', False)),
        attach='origin')
    insertion_streams = _prepare_tendon_waypoints_from_belly(
        insertion, belly, reverse=bool(getattr(belly, 'insertion_tendon_reverse', False)),
        attach='insertion')

    n_streams = len(base)
    origin_streams, origin_flip_mask = _orient_tendon_streams_to_belly(origin_streams, base, 'origin')
    insertion_streams, insertion_flip_mask = _orient_tendon_streams_to_belly(insertion_streams, base, 'insertion')
    belly._tendon_origin_auto_flip_mask = list(origin_flip_mask)
    belly._tendon_insertion_auto_flip_mask = list(insertion_flip_mask)
    # The built belly's live stream is authoritative: it is what its fibers and
    # 3D coordinate axes currently use. `_selected_stream_bounding_planes` can
    # be an older, separate snapshot, so aligning against it produces perfect
    # internal diagnostics but six visibly different axes at the live junction.
    base_src_contours, base_src_planes = _stream_major_contour_source(
        belly, prefer_live=True)
    origin_src_contours, origin_src_planes = _stream_major_contour_source(origin)
    insertion_src_contours, insertion_src_planes = _stream_major_contour_source(insertion)
    base_contours = _copy_stream_levels(base_src_contours, n_streams)
    base_planes = _copy_stream_levels(base_src_planes, n_streams)
    origin_reverse = bool(getattr(belly, 'origin_tendon_reverse', False))
    insertion_reverse = bool(getattr(belly, 'insertion_tendon_reverse', False))
    origin_contours = _copy_stream_levels(
        origin_src_contours, n_streams, reverse=origin_reverse) if origin is not None else None
    origin_planes = _copy_stream_levels(
        origin_src_planes, n_streams, reverse=origin_reverse) if origin is not None else None
    insertion_contours = _copy_stream_levels(
        insertion_src_contours, n_streams, reverse=insertion_reverse) if insertion is not None else None
    insertion_planes = _copy_stream_levels(
        insertion_src_planes, n_streams, reverse=insertion_reverse) if insertion is not None else None
    origin_contours = _flip_stream_levels_by_mask(origin_contours, origin_flip_mask)
    origin_planes = _flip_stream_levels_by_mask(origin_planes, origin_flip_mask)
    insertion_contours = _flip_stream_levels_by_mask(insertion_contours, insertion_flip_mask)
    insertion_planes = _flip_stream_levels_by_mask(insertion_planes, insertion_flip_mask)

    # Keep the tendon-side seam levels for fiber-coordinate registration. They
    # are replaced by exact copies of the matching belly boundary in the combined
    # sequence. Connected contour-mesh construction removes duplicates separately.

    def _validate_part_levels(label, contours, planes):
        if contours is None:
            return True
        if planes is None:
            print(f"[{belly_name}] Tendon extension: {label} has contours but no bounding planes")
            return False
        if len(contours) != len(planes):
            print(f"[{belly_name}] Tendon extension: {label} has {len(contours)} contour "
                  f"streams but {len(planes)} bounding-plane streams")
            return False
        for stream_i, (stream_contours, stream_planes) in enumerate(zip(contours, planes)):
            if len(stream_contours) != len(stream_planes):
                print(f"[{belly_name}] Tendon extension: {label} stream {stream_i} has "
                      f"{len(stream_contours)} contours but {len(stream_planes)} "
                      "bounding planes")
                return False
        return True

    if not all((
            _validate_part_levels('origin tendon', origin_contours, origin_planes),
            _validate_part_levels('belly', base_contours, base_planes),
            _validate_part_levels('insertion tendon', insertion_contours, insertion_planes))):
        return False

    # Assemble the combined origin-tendon + belly + insertion-tendon contour and
    # bounding-plane sequence per stream, including both shared seam copies.
    inspect_contours = []
    inspect_planes = []
    origin_len = []
    insertion_len = []
    belly_anchor_ranges = []
    for s in range(n_streams):
        stream_contours = []
        stream_planes = []
        o_n = 0
        i_n = 0
        if origin_contours:
            oc = origin_contours[min(s, len(origin_contours) - 1)]
            stream_contours.extend(oc)
            o_n = len(oc)
            if origin_planes:
                stream_planes.extend(origin_planes[min(s, len(origin_planes) - 1)])
        if base_contours:
            bc = base_contours[min(s, len(base_contours) - 1)]
            stream_contours.extend(bc)
            belly_anchor_ranges.append((o_n, o_n + len(bc)))
        else:
            belly_anchor_ranges.append((o_n, o_n))
        if base_planes:
            stream_planes.extend(base_planes[min(s, len(base_planes) - 1)])
        if insertion_contours:
            ic = insertion_contours[min(s, len(insertion_contours) - 1)]
            stream_contours.extend(ic)
            i_n = len(ic)
            if insertion_planes:
                stream_planes.extend(insertion_planes[min(s, len(insertion_planes) - 1)])
        inspect_contours.append(stream_contours)
        inspect_planes.append(stream_planes)
        origin_len.append(o_n)
        insertion_len.append(i_n)

    # Align the whole sequence as one chain and regenerate every fiber in one
    # pass — global frame continuity, no per-tendon seam guessing or twist.
    extended, inspect_contours, inspect_planes = _smooth_and_regenerate_combined(
        belly, inspect_contours, inspect_planes,
        belly_anchor_ranges=belly_anchor_ranges,
        belly_waypoints=base)

    belly.waypoints = extended
    belly.waypoints_original = [[np.asarray(wp, dtype=np.float64).copy() for wp in stream]
                                for stream in extended]
    belly._tendon_extended_inspect_contours = inspect_contours
    belly._tendon_extended_inspect_bounding_planes = inspect_planes
    belly._tendon_extended_inspect_waypoints = [[np.asarray(wp, dtype=np.float64).copy()
                                                 for wp in stream]
                                                for stream in extended]

    live_belly_planes = getattr(belly, 'bounding_planes', None)
    if live_belly_planes is not None:
        for s in range(min(len(inspect_planes), len(live_belly_planes))):
            checks = []
            if origin_len[s] > 0 and live_belly_planes[s]:
                checks.append(('origin/belly', inspect_planes[s][origin_len[s] - 1],
                               live_belly_planes[s][0]))
            if insertion_len[s] > 0 and live_belly_planes[s]:
                checks.append(('belly/insertion',
                               inspect_planes[s][len(inspect_planes[s]) - insertion_len[s]],
                               live_belly_planes[s][-1]))
            for label, tendon_bp, live_bp in checks:
                dots = [float(np.dot(tendon_bp[key], live_bp[key]))
                        for key in ('basis_x', 'basis_y', 'basis_z')]
                mean_delta = float(np.linalg.norm(
                    np.asarray(tendon_bp['mean']) - np.asarray(live_bp['mean'])))
                print(f"  [Live seam check] Stream {s} {label}: "
                      f"dot(x,y,z)=({dots[0]:.4f}, {dots[1]:.4f}, {dots[2]:.4f}), "
                      f"mean_delta={mean_delta:.6g}")

    # Standalone tendon objects retain their original processing data, but their
    # 3D BP visualization must show the aligned copies actually used by this
    # extension. Otherwise the viewer displays the old unaligned frames and makes
    # a correct extension look broken.
    for old_tendon in getattr(belly, '_tendon_alignment_display_objects', []) or []:
        if getattr(old_tendon, '_aligned_extension_owner_id', None) == id(belly):
            old_tendon._aligned_extension_bounding_planes = None
            old_tendon._aligned_extension_owner_id = None
    display_objects = []
    if origin is not None:
        origin._aligned_extension_bounding_planes = [
            [copy.deepcopy(bp) for bp in inspect_planes[s][:origin_len[s]]]
            for s in range(len(inspect_planes))]
        for stream in origin._aligned_extension_bounding_planes:
            if stream:
                stream[-1]['_suppress_duplicate_seam_axes'] = True
        origin._aligned_extension_owner_id = id(belly)
        display_objects.append(origin)
    if insertion is not None:
        insertion._aligned_extension_bounding_planes = [
            [copy.deepcopy(bp) for bp in inspect_planes[s][
                len(inspect_planes[s]) - insertion_len[s]:]]
            for s in range(len(inspect_planes))]
        for stream in insertion._aligned_extension_bounding_planes:
            if stream:
                stream[0]['_suppress_duplicate_seam_axes'] = True
        insertion._aligned_extension_owner_id = id(belly)
        display_objects.append(insertion)
    belly._tendon_alignment_display_objects = display_objects
    belly.waypoint_level_regions = [[
        {'part': 'origin_tendon' if li < origin_len[s]
         else 'insertion_tendon' if li >= len(stream) - insertion_len[s]
         else 'belly',
         'level': li}
        for li in range(len(stream))
    ] for s, stream in enumerate(extended)]
    belly._fiber_draw_dirty = True
    belly._fiber_draw_pts = None
    belly._fiber_draw_lines = None
    belly.is_draw_fiber_architecture = True
    belly.tendon_extended_fibers = True

    print(f"[{belly_name}] Tendon extension applied: "
          f"origin={origin_name or 'None'}, insertion={insertion_name or 'None'}")
    return True


def _extend_belly_and_linked_counterpart_fibers(v, belly_name, belly):
    ok = _extend_belly_fibers_with_tendons(v, belly_name, belly)
    if not getattr(belly, 'linked_drive_counterpart', False):
        return ok
    other_name, other = _linked_counterpart(v, belly)
    if other is None:
        return ok
    if getattr(other, 'waypoints', None) is None or len(other.waypoints) == 0:
        print(f"[{belly_name}] Linked counterpart {other_name}: build fibers first")
        return ok
    other._belly_waypoints_before_tendon_extension = None
    _auto_fill_tendon_extension_names(v, other_name, other)
    if (getattr(other, 'origin_tendon_extension_name', '')
            or getattr(other, 'insertion_tendon_extension_name', '')):
        ok_other = _extend_belly_fibers_with_tendons(v, other_name, other)
        return ok and ok_other
    print(f"[{belly_name}] Linked counterpart {other_name}: no tendons selected")
    return ok


def _connected_source_has_linked_components(obj):
    comps = getattr(obj, '_connected_contour_mesh_components', None)
    return comps is not None and len(comps) > 1


def _original_tet_component_names(v, name, obj):
    names = []

    def add(comp_name):
        if comp_name and comp_name in v.zygote_muscle_meshes and comp_name not in names:
            names.append(comp_name)

    add(name)
    other_name, other = _linked_counterpart(v, obj)
    if other is not None:
        add(other_name)

    for comp_name in list(names):
        comp = v.zygote_muscle_meshes.get(comp_name)
        if comp is None:
            continue
        _auto_fill_tendon_extension_names(v, comp_name, comp)
        add(getattr(comp, 'origin_tendon_extension_name', ''))
        add(getattr(comp, 'insertion_tendon_extension_name', ''))
    return names


def _mesh_part_from_name(name):
    lowered = str(name or '').lower()
    if 'origin' in lowered and 'tendon' in lowered:
        return 'origin_tendon'
    if 'insertion' in lowered and 'tendon' in lowered:
        return 'insertion_tendon'
    if 'tendon' in lowered:
        return 'tendon'
    return 'belly'


def _component_original_surface(comp_obj):
    if comp_obj is None:
        return None, None
    verts = getattr(comp_obj, 'vertices', None)
    if verts is None or len(verts) == 0:
        return None, None
    tri_faces = []
    faces_3 = getattr(comp_obj, 'faces_3', None)
    if faces_3 is not None and len(faces_3) > 0:
        tri_faces.extend(np.asarray(faces_3[:, :, 0], dtype=np.int32).tolist())
    faces_4 = getattr(comp_obj, 'faces_4', None)
    if faces_4 is not None and len(faces_4) > 0:
        for face in np.asarray(faces_4[:, :, 0], dtype=np.int32):
            tri_faces.append([int(face[0]), int(face[1]), int(face[2])])
            tri_faces.append([int(face[0]), int(face[2]), int(face[3])])
    for face in getattr(comp_obj, 'faces_other', []) or []:
        f = np.asarray(face, dtype=np.int32)
        if f.ndim != 2 or len(f) < 3:
            continue
        ids = f[:, 0]
        for i in range(1, len(ids) - 1):
            tri_faces.append([int(ids[0]), int(ids[i]), int(ids[i + 1])])
    if len(tri_faces) == 0:
        return None, None
    return np.asarray(verts, dtype=np.float64), np.asarray(tri_faces, dtype=np.int32)


def _triangulate_cap_loop(verts, loop):
    loop = [int(i) for i in loop]
    n = len(loop)
    if n < 3:
        return [], None
    if n == 3:
        return [[loop[0], loop[1], loop[2]]], None
    pts_3d = np.asarray(verts, dtype=np.float64)[loop]
    centroid = pts_3d.mean(axis=0)
    try:
        centered = pts_3d - centroid
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        pts_2d = centered @ vt[:2].T
        segments = np.array([[i, (i + 1) % n] for i in range(n)], dtype=np.int32)
        import triangle as tr
        result = tr.triangulate({'vertices': pts_2d, 'segments': segments}, 'p')
        tris = result.get('triangles', None)
        if tris is not None and len(tris) > 0:
            if int(np.max(tris)) >= len(loop):
                return None, centroid
            return [[loop[int(a)], loop[int(b)], loop[int(c)]] for a, b, c in tris], None
    except Exception:
        pass
    return None, centroid


def _surface_boundary_loops(faces):
    from collections import defaultdict
    faces = np.asarray(faces, dtype=np.int32)
    edge_count = defaultdict(int)
    for face in faces:
        if len(face) != 3:
            continue
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            if a == b:
                continue
            edge_count[(min(a, b), max(a, b))] += 1

    open_edges = [edge for edge, count in edge_count.items() if count == 1]
    if not open_edges:
        return []

    adjacency = defaultdict(list)
    open_set = set(open_edges)
    for a, b in open_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    visited = set()
    loops = []
    for start in open_edges:
        if start in visited:
            continue
        a0, b0 = start
        loop = [a0]
        prev, cur = a0, b0
        visited.add(start)
        for _ in range(len(open_edges) + 2):
            loop.append(cur)
            candidates = []
            for nxt in adjacency[cur]:
                if nxt == prev:
                    continue
                key = (min(cur, nxt), max(cur, nxt))
                if key not in visited:
                    candidates.append((nxt, key))
            if not candidates:
                close_key = (min(cur, loop[0]), max(cur, loop[0]))
                if close_key in open_set and close_key not in visited:
                    visited.add(close_key)
                break
            nxt, key = candidates[0]
            visited.add(key)
            prev, cur = cur, nxt
            if cur == loop[0]:
                break
        if len(loop) > 1 and loop[-1] == loop[0]:
            loop = loop[:-1]
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _fallback_edge_group_loops(comp_obj):
    loops = []
    for group in getattr(comp_obj, 'edge_groups', []) or []:
        loop = [int(i) for i in group]
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _cap_component_boundary_loops(comp_obj, comp_name, vertices, faces,
                                  all_vertices, all_faces, vertex_regions,
                                  region, offset=0, cap_face_indices=None,
                                  verbose=True):
    loops = _surface_boundary_loops(faces)
    if not loops:
        loops = _fallback_edge_group_loops(comp_obj)
    if not loops:
        return 0

    added = 0
    cap_face_indices = cap_face_indices if cap_face_indices is not None else []
    verts_arr = np.asarray(all_vertices, dtype=np.float64)
    for loop in loops:
        loop = [int(i) for i in loop]
        if len(loop) < 3:
            continue
        loop_global = [idx + int(offset) for idx in loop]
        cap_faces, centroid = _triangulate_cap_loop(verts_arr, loop_global)
        if cap_faces is None:
            center_idx = len(all_vertices)
            all_vertices.append(np.asarray(centroid, dtype=np.float64).tolist())
            vertex_regions.append(region.copy())
            cap_faces = [
                [loop_global[i], loop_global[(i + 1) % len(loop_global)], center_idx]
                for i in range(len(loop_global))
            ]
            verts_arr = np.asarray(all_vertices, dtype=np.float64)
        for tri in cap_faces:
            if cap_face_indices is not None:
                cap_face_indices.append(len(all_faces))
            all_faces.append([int(tri[0]), int(tri[1]), int(tri[2])])
            added += 1
    if added and verbose:
        print(f"[{comp_name}] Capped {len(loops)} original-surface boundary loop(s), {added} faces")
    return added


def _dedupe_surface_vertices(vertices, faces, regions, eps=2e-5):
    from scipy.spatial import cKDTree
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if len(vertices) == 0:
        return vertices, faces, regions
    parent = list(range(len(vertices)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    try:
        tree = cKDTree(vertices)
        for i, j in tree.query_pairs(r=float(eps)):
            union(i, j)
    except Exception:
        pass

    clusters = {}
    for i in range(len(vertices)):
        clusters.setdefault(find(i), []).append(i)
    old_to_new = {}
    new_vertices = []
    new_regions = []
    for members in clusters.values():
        new_idx = len(new_vertices)
        new_vertices.append(np.mean(vertices[members], axis=0))
        first = regions[members[0]] if regions and members[0] < len(regions) else None
        new_regions.append(first)
        for old in members:
            old_to_new[old] = new_idx

    remapped = np.array([[old_to_new[int(i)] for i in face] for face in faces], dtype=np.int32)
    face_counts = {}
    valid_faces = []
    for face in remapped:
        if len(set(int(i) for i in face)) < 3:
            continue
        key = tuple(sorted(int(i) for i in face))
        face_counts[key] = face_counts.get(key, 0) + 1
        valid_faces.append((key, face))

    clean_faces = []
    for key, face in valid_faces:
        # Coincident opposite-side faces are internal walls after welding the
        # split belly/tendon components. Drop all copies so TetGen sees one
        # connected volume instead of adjacent closed volumes.
        if face_counts.get(key, 0) > 1:
            continue
        clean_faces.append(face.tolist())
    return np.asarray(new_vertices, dtype=np.float64), np.asarray(clean_faces, dtype=np.int32), new_regions


def _component_closed_surface_for_labels(comp_obj, part, component_name):
    verts, faces = _component_original_surface(comp_obj)
    if verts is None or faces is None:
        return None, None, None
    all_vertices = verts.tolist()
    all_faces = faces.tolist()
    regions = [{'part': part, 'component': component_name} for _ in range(len(all_vertices))]
    _cap_component_boundary_loops(
        comp_obj, component_name, verts, faces,
        all_vertices, all_faces, regions,
        {'part': part, 'component': component_name},
        offset=0, cap_face_indices=None)
    if not all_vertices or not all_faces:
        return None, None, None
    eps = 2e-5
    return _dedupe_surface_vertices(all_vertices, all_faces, regions, eps=eps)


def _count_simple_labels(labels):
    counts = {}
    for label in labels or []:
        counts[label] = counts.get(label, 0) + 1
    return counts


def _estimate_closed_surface_volume(vertices, faces):
    try:
        mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int32),
            process=False)
        vol = abs(float(mesh.volume))
        if np.isfinite(vol) and vol > 1e-12:
            return vol
    except Exception:
        pass
    verts = np.asarray(vertices, dtype=np.float64)
    if len(verts) == 0:
        return 0.0
    extent = np.max(verts, axis=0) - np.min(verts, axis=0)
    vol = float(np.prod(np.maximum(extent, 1e-9)))
    return vol if np.isfinite(vol) else 0.0


def _simplify_closed_surface_for_tet(name, vertices, faces, target_tet_count):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    target_tet_count = max(0, int(target_tet_count or 0))
    if target_tet_count <= 0 or len(faces) == 0:
        return vertices, faces, False


    # In coarse non-refined TetGen mode, this muscle family produces roughly
    # 1.5-1.7 tets per simplified surface face. Use surface count, not TetGen
    # volume refinement, as the main tet-count control.
    target_faces = int(np.clip(round(target_tet_count * 0.6), 120, max(120, len(faces))))
    if target_faces >= len(faces):
        # Still reduce a little in approximate mode so the log and behavior are
        # consistent, but do not crush the surface when the requested tet count
        # is already close to what the current surface will produce.
        target_faces = max(120, int(round(len(faces) * 0.9)))
    if target_faces >= len(faces):
        print(f"[{name}] Approx surface tet: surface already at/below target face budget")
        return vertices, faces, False

    try:
        import warnings
        import pyvista as pv
        import fast_simplification
        faces_flat = np.hstack([
            np.full((len(faces), 1), 3, dtype=np.int64),
            faces.astype(np.int64)
        ]).ravel()
        poly = pv.PolyData(vertices, faces_flat)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simp = fast_simplification.simplify_mesh(
                poly, target_count=target_faces, agg=5, verbose=False)
        simp_faces_flat = np.asarray(simp.faces, dtype=np.int64)
        if len(simp_faces_flat) == 0:
            raise RuntimeError("simplifier returned no faces")
        simp_faces = simp_faces_flat.reshape((-1, 4))[:, 1:4].astype(np.int32)
        simp_vertices = np.asarray(simp.points, dtype=np.float64)

        # Simplification can leave a watertight but self-intersecting surface
        # near thin tendon holes. Always run MeshFix in approximate mode because
        # this mode explicitly allows the boundary to change.
        import pymeshfix
        fixer = pymeshfix.MeshFix(simp_vertices.copy(), simp_faces.copy())
        try:
            fixer.repair(verbose=False)
        except TypeError:
            fixer.repair()
        try:
            simp_vertices, simp_faces = fixer.v, fixer.f
        except AttributeError:
            simp_vertices, simp_faces = fixer._return_arrays()
        mesh = trimesh.Trimesh(vertices=simp_vertices, faces=simp_faces, process=False)
        if not mesh.is_winding_consistent:
            trimesh.repair.fix_normals(mesh, multibody=False)
        simp_vertices = np.asarray(mesh.vertices, dtype=np.float64)
        simp_faces = np.asarray(mesh.faces, dtype=np.int32)
        if len(simp_vertices) == 0 or len(simp_faces) == 0:
            raise RuntimeError("simplified surface is empty")
        print(f"[{name}] Approx surface tet: simplified {len(vertices)}v/{len(faces)}f "
              f"-> {len(simp_vertices)}v/{len(simp_faces)}f "
              f"(target tets {target_tet_count})")
        return simp_vertices, simp_faces, True
    except Exception as exc:
        print(f"[{name}] Approx surface tet simplification failed, using capped surface: {exc}")
        return vertices, faces, False


def _isotropic_voxel_surface_for_tet(name, vertices, faces, target_tet_count):
    """Create a watertight, near-isotropic boundary for stable EMU tets.

    Decimation preserves pathological skinny input triangles, which then force
    TetGen slivers regardless of its volume quality settings. A filled voxel
    level set removes sub-grid defects and marching cubes supplies uniformly
    sized boundary triangles. The 5.5 calibration is the measured number of
    quality tets per filled voxel for this muscle family.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    target_tet_count = max(1000, int(target_tet_count or 0))
    try:
        source = trimesh.Trimesh(
            vertices=vertices, faces=faces, process=False)
        if not source.is_winding_consistent:
            trimesh.repair.fix_normals(source, multibody=False)
        volume = abs(float(source.volume))
        if not np.isfinite(volume) or volume <= 1e-12:
            volume = _estimate_closed_surface_volume(vertices, faces)
        pitch = float(np.power(
            max(5.5 * volume / target_tet_count, 1e-18), 1.0 / 3.0))
        extent = np.maximum(np.asarray(source.extents, dtype=np.float64), 1e-9)
        pitch = float(np.clip(pitch, np.min(extent) / 80.0,
                              np.min(extent) / 6.0))
        voxel_grid = source.voxelized(pitch).fill()
        remeshed = voxel_grid.marching_cubes
        remeshed.apply_transform(voxel_grid.transform)
        # Marching cubes over binary occupancy has isotropic connectivity but
        # retains visible voxel-scale terraces.  Non-shrinking Taubin passes
        # round those terraces before TetGen; unlike decimation this preserves
        # the uniform topology which prevents boundary-forced slivers.
        trimesh.smoothing.filter_taubin(
            remeshed, lamb=0.5, nu=0.53, iterations=8)
        projected, source_distance, _ = source.nearest.on_surface(
            np.asarray(remeshed.vertices, dtype=np.float64))
        projection_blend = 0.65
        remeshed.vertices = (
            (1.0 - projection_blend) *
            np.asarray(remeshed.vertices, dtype=np.float64) +
            projection_blend * projected)
        volume_after_smooth = abs(float(remeshed.volume))
        if not remeshed.is_winding_consistent:
            trimesh.repair.fix_normals(remeshed, multibody=False)
        remesh_vertices = np.asarray(remeshed.vertices, dtype=np.float64)
        remesh_faces = np.asarray(remeshed.faces, dtype=np.int32)
        if len(remesh_vertices) == 0 or len(remesh_faces) == 0:
            raise RuntimeError("voxel marching cubes returned an empty surface")
        print(f"[{name}] EMU isotropic boundary: pitch={pitch*1000.0:.3g} mm, "
              f"{len(vertices)}v/{len(faces)}f -> "
              f"{len(remesh_vertices)}v/{len(remesh_faces)}f, "
              f"target={target_tet_count} tets, Taubin=8, "
              f"source projection={projection_blend:.0%}, "
              f"boundary deviation={np.mean(source_distance)*(1-projection_blend)*1000.0:.3g} mm, "
              f"anatomical volume ratio={volume_after_smooth / max(volume, 1e-30):.4f}")
        return remesh_vertices, remesh_faces, True
    except Exception as exc:
        print(f"[{name}] EMU isotropic boundary failed ({exc}); "
              f"trying decimation/repair fallback")
        return _simplify_closed_surface_for_tet(
            name, vertices, faces, target_tet_count)


def _tet_volume_quality_stats(name, tet_vertices, tetrahedra):
    tet_vertices = np.asarray(tet_vertices, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32)
    if len(tet_vertices) == 0 or len(tetrahedra) == 0:
        return tetrahedra, {
            'count': 0, 'degenerate': 0, 'near_zero': 0,
            'min_volume': 0.0, 'median_volume': 0.0,
            'edge_cv': 0.0, 'scaled_jacobian_min': 0.0,
            'scaled_jacobian_p01': 0.0, 'mean_ratio_min': 0.0,
            'mean_ratio_p01': 0.0, 'critical_slivers': 0,
        }

    tv = tet_vertices[tetrahedra]
    signed = np.einsum(
        'ij,ij->i',
        tv[:, 1] - tv[:, 0],
        np.cross(tv[:, 2] - tv[:, 0], tv[:, 3] - tv[:, 0])) / 6.0
    vol = np.abs(signed)
    finite = np.isfinite(vol)
    finite_vol = vol[finite]
    median = float(np.median(finite_vol)) if len(finite_vol) else 0.0
    mean = float(np.mean(finite_vol)) if len(finite_vol) else 0.0
    min_v = float(np.min(finite_vol)) if len(finite_vol) else 0.0
    p01 = float(np.percentile(finite_vol, 1.0)) if len(finite_vol) else 0.0
    p99 = float(np.percentile(finite_vol, 99.0)) if len(finite_vol) else 0.0
    deg_eps = max(1e-18, median * 1e-10)
    near_eps = max(1e-16, median * 1e-4)
    degenerate = (~finite) | (vol <= deg_eps)
    if np.any(degenerate):
        # TetGen can leave zero-volume artifacts around capped contour seams.
        # They cannot contribute to a valid volume and poison the percentile
        # quality gate, so remove them and recompute all metrics on the filled
        # nondegenerate volume before deciding whether to accept the mesh.
        kept = tetrahedra[~degenerate]
        print(f"[{name}] Tet quality repair: removing "
              f"{int(np.sum(degenerate))} degenerate tetrahedra")
        if len(kept) == 0:
            return tetrahedra, {
                'count': 0, 'degenerate': int(np.sum(degenerate)),
                'near_zero': 0, 'removed_degenerate': int(np.sum(degenerate)),
                'scaled_jacobian_p01': 0.0, 'mean_ratio_p01': 0.0,
                'critical_slivers': 0,
            }
        repaired, repaired_stats = _tet_volume_quality_stats(
            name, tet_vertices, kept)
        repaired_stats['removed_degenerate'] = int(np.sum(degenerate))
        return repaired, repaired_stats
    near_zero = finite & (vol > deg_eps) & (vol <= near_eps)

    edge_lens = []
    for i in range(4):
        for j in range(i + 1, 4):
            edge_lens.append(np.linalg.norm(tv[:, i] - tv[:, j], axis=1))
    edge_lens = np.asarray(edge_lens, dtype=np.float64)
    avg_edge = np.mean(edge_lens, axis=0)
    edge_cv = float(np.std(avg_edge) / (np.mean(avg_edge) + 1e-30)) if len(avg_edge) else 0.0

    # Shape metrics are scale-independent and expose slivers that volume-only
    # diagnostics miss. Scaled Jacobian and mean ratio equal 1 for a regular
    # tetrahedron and approach zero for a collapsed/sliver element.
    corner_scaled = []
    for corner in range(4):
        others = [i for i in range(4) if i != corner]
        e0 = tv[:, others[0]] - tv[:, corner]
        e1 = tv[:, others[1]] - tv[:, corner]
        e2 = tv[:, others[2]] - tv[:, corner]
        numerator = np.abs(np.einsum('ij,ij->i', e0, np.cross(e1, e2)))
        denominator = (np.linalg.norm(e0, axis=1) *
                       np.linalg.norm(e1, axis=1) *
                       np.linalg.norm(e2, axis=1))
        corner_scaled.append(
            np.sqrt(2.0) * numerator / np.maximum(denominator, 1e-30))
    scaled_jacobian = np.clip(np.min(np.asarray(corner_scaled), axis=0), 0.0, 1.0)
    edge_sq_sum = np.sum(edge_lens ** 2, axis=0)
    mean_ratio = np.clip(
        12.0 * np.power(np.maximum(3.0 * vol, 0.0), 2.0 / 3.0) /
        np.maximum(edge_sq_sum, 1e-30), 0.0, 1.0)
    sj_min = float(np.min(scaled_jacobian)) if len(scaled_jacobian) else 0.0
    sj_p01 = float(np.percentile(scaled_jacobian, 1.0)) if len(scaled_jacobian) else 0.0
    mr_min = float(np.min(mean_ratio)) if len(mean_ratio) else 0.0
    mr_p01 = float(np.percentile(mean_ratio, 1.0)) if len(mean_ratio) else 0.0
    critical_slivers = int(np.sum((scaled_jacobian < 1e-3) | (mean_ratio < 1e-3)))
    weak_slivers = int(np.sum((scaled_jacobian < 0.03) | (mean_ratio < 0.03)))

    stats = {
        'count': int(len(tetrahedra)),
        'degenerate': int(np.sum(degenerate)),
        'near_zero': int(np.sum(near_zero)),
        'removed_degenerate': 0,
        'min_volume': min_v,
        'p01_volume': p01,
        'median_volume': median,
        'mean_volume': mean,
        'p99_volume': p99,
        'near_zero_threshold': near_eps,
        'degenerate_threshold': deg_eps,
        'edge_cv': edge_cv,
        'scaled_jacobian_min': sj_min,
        'scaled_jacobian_p01': sj_p01,
        'mean_ratio_min': mr_min,
        'mean_ratio_p01': mr_p01,
        'critical_slivers': critical_slivers,
        'weak_slivers': weak_slivers,
    }
    print(f"[{name}] Tet quality: degenerate={stats['degenerate']} "
          f"(removed 0; volume kept filled), near_zero={stats['near_zero']} "
          f"(vol <= {near_eps:.3g}), min={min_v:.3g}, p01={p01:.3g}, "
          f"median={median:.3g}, p99={p99:.3g}, edge_cv={edge_cv:.3g}")
    print(f"[{name}] Tet shape: scaled-J min/p01={sj_min:.3g}/{sj_p01:.3g}, "
          f"mean-ratio min/p01={mr_min:.3g}/{mr_p01:.3g}, "
          f"critical/weak slivers={critical_slivers}/{weak_slivers}")
    return tetrahedra, stats


def _collect_connected_component_fibers(v, component_names):
    state_fields = (
        'contours', 'bounding_planes', 'stream_contours', 'stream_bounding_planes',
        '_selected_stream_contours', '_selected_stream_bounding_planes',
        'draw_contour_stream', 'max_stream_count', 'mvc_weights',
        'waypoints', 'waypoints_original', 'waypoint_bary_coords',
        'waypoint_level_regions', 'fiber_architecture',
        '_belly_waypoints_before_tendon_extension',
        '_tendon_extended_inspect_contours',
        '_tendon_extended_inspect_bounding_planes',
        '_tendon_extended_inspect_waypoints',
        '_tendon_origin_auto_flip_mask', '_tendon_insertion_auto_flip_mask',
        '_aligned_extension_bounding_planes',
        'tendon_extended_fibers', 'origin_tendon_extension_name',
        'insertion_tendon_extension_name', 'origin_tendon_reverse',
        'insertion_tendon_reverse', 'enable_tendon_extension',
        'fiber_positioning_method', 'sampling_method', 'fiber_sampling_seed',
        'attach_skeletons', 'attach_skeletons_sub', 'attach_skeleton_names',
    )
    out = []
    for comp_name in component_names:
        comp = v.zygote_muscle_meshes.get(comp_name)
        if comp is None:
            continue
        waypoints = getattr(comp, 'waypoints', None)
        waypoints_original = getattr(comp, 'waypoints_original', None)
        n_streams = len(waypoints) if waypoints is not None else 0
        out.append({
            'snapshot_version': 2,
            'component': comp_name,
            'part': _mesh_part_from_name(comp_name),
            'stream_start': None,
            'stream_end': None,
            'waypoints': _copy_stream_levels(waypoints, n_streams) if n_streams else None,
            'waypoints_original': (
                _copy_stream_levels(waypoints_original, len(waypoints_original))
                if waypoints_original is not None else None
            ),
            'waypoint_level_regions': copy.deepcopy(
                getattr(comp, 'waypoint_level_regions', None)),
            'fiber_architecture': [
                np.asarray(f, dtype=np.float64).copy()
                for f in getattr(comp, 'fiber_architecture', []) or []
            ],
            'tendon_extended_fibers': bool(getattr(comp, 'tendon_extended_fibers', False)),
            'origin_tendon_extension_name': getattr(comp, 'origin_tendon_extension_name', ''),
            'insertion_tendon_extension_name': getattr(comp, 'insertion_tendon_extension_name', ''),
            'component_state': {
                field: copy.deepcopy(getattr(comp, field))
                for field in state_fields if hasattr(comp, field)
            },
        })
    return out


def _restore_connected_component_fibers(v, group_name, surface_name, surface):
    """Restore saved group component state into the live component objects."""
    snapshots = getattr(surface, '_connected_component_fibers', None) or []
    restored = []
    missing = []
    for entry in snapshots:
        comp_name = entry.get('component', '')
        comp = v.zygote_muscle_meshes.get(comp_name)
        if comp is None:
            missing.append(comp_name)
            continue

        state = entry.get('component_state') or {}
        if state:
            for field, value in state.items():
                setattr(comp, field, copy.deepcopy(value))
        else:
            # Backward compatibility with the original lightweight snapshot.
            for field in ('waypoints', 'waypoints_original',
                          'waypoint_level_regions', 'fiber_architecture'):
                if entry.get(field) is not None:
                    setattr(comp, field, copy.deepcopy(entry[field]))
            comp.tendon_extended_fibers = bool(
                entry.get('tendon_extended_fibers', False))
            comp.origin_tendon_extension_name = entry.get(
                'origin_tendon_extension_name', '')
            comp.insertion_tendon_extension_name = entry.get(
                'insertion_tendon_extension_name', '')

        comp._connected_mesh_owner_name = surface_name
        if hasattr(comp, '_emu_rest_waypoints'):
            delattr(comp, '_emu_rest_waypoints')
        comp._fiber_draw_dirty = True
        comp._fiber_draw_pts = None
        comp._fiber_draw_lines = None
        comp.is_draw_fiber_architecture = bool(getattr(comp, 'waypoints', None))
        comp.is_draw_tet_mesh = False
        restored.append(comp_name)

    # Reconnect saved aligned tendon display planes to their owning bellies.
    for entry in snapshots:
        belly_name = entry.get('component', '')
        belly = v.zygote_muscle_meshes.get(belly_name)
        if belly is None or not getattr(belly, 'tendon_extended_fibers', False):
            continue
        display_objects = []
        for field in ('origin_tendon_extension_name',
                      'insertion_tendon_extension_name'):
            tendon_name = getattr(belly, field, '')
            tendon = v.zygote_muscle_meshes.get(tendon_name) if tendon_name else None
            if (tendon is not None
                    and getattr(tendon, '_aligned_extension_bounding_planes', None)):
                tendon._aligned_extension_owner_id = id(belly)
                display_objects.append(tendon)
        belly._tendon_alignment_display_objects = display_objects

    _apply_zygote_group_links(v, group_name)
    surface._connected_original_tet_components = [surface_name]
    surface.is_draw_tet_mesh = True
    if missing:
        print(f"[{group_name}] Group fiber restore missing component(s): {missing}")
    print(f"[{group_name}] Restored current state for {len(restored)} group component(s)")
    return len(restored) > 0


def _deform_group_fibers_from_emu(v, group_name, rest_vertices, positions):
    """Move saved component fiber samples with the nearest EMU tet vertex."""
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(rest_vertices, dtype=np.float64))
    displacement = np.asarray(positions, dtype=np.float64) - rest_vertices
    state = _ensure_zygote_group_mapping(v, group_name)
    for name in _zygote_group_guide_names(v, group_name):
        obj = v.zygote_muscle_meshes.get(name)
        if obj is None or not getattr(obj, 'waypoints', None):
            continue
        if getattr(obj, '_emu_rest_waypoints', None) is None:
            obj._emu_rest_waypoints = copy.deepcopy(obj.waypoints)
        deformed = copy.deepcopy(obj._emu_rest_waypoints)
        for si, stream in enumerate(deformed):
            for li, points in enumerate(stream):
                points = np.asarray(points, dtype=np.float64)
                _, nearest = tree.query(points.reshape(-1, 3))
                deformed[si][li] = points + displacement[nearest].reshape(points.shape)
        obj.waypoints = deformed
        obj._fiber_draw_dirty = True
        obj._fiber_draw_pts = None
        obj._fiber_draw_lines = None


def _reset_group_emu_pose(v, group_name, surface):
    rest = getattr(surface, '_emu_rest_vertices', None)
    if rest is not None:
        surface.tet_vertices = np.asarray(rest, dtype=np.float32).copy()
        surface.tet_render_contact_offsets = None
        _invalidate_tet_draw_cache(surface)
        surface.is_draw_tet_mesh = True
    for name in _zygote_group_guide_names(v, group_name):
        obj = v.zygote_muscle_meshes.get(name)
        if obj is not None and hasattr(obj, '_emu_rest_waypoints'):
            obj.waypoints = copy.deepcopy(obj._emu_rest_waypoints)
            obj._fiber_draw_dirty = True
            obj._fiber_draw_pts = None
            obj._fiber_draw_lines = None


def _emu_rest_inside_bone_vertices(v, prepared, surface):
    """Find muscle surface vertices anatomically embedded in bones at rest."""
    from viewer.fem_sim import _build_bone_trimeshes
    rest = np.asarray(prepared['vertices'], dtype=np.float64)
    surface_vertices = np.unique(prepared['surface_faces']).astype(np.int32)
    rest_min = np.min(rest, axis=0) - 0.02
    rest_max = np.max(rest, axis=0) + 0.02
    rest_bones = _build_bone_trimeshes(
        v, {prepared['name']: surface}, v.env.skel, verbose=False)
    nearby = [
        mesh for mesh in rest_bones
        if np.all(mesh.bounds[1] >= rest_min) and np.all(mesh.bounds[0] <= rest_max)
    ]
    exempt = set()
    for bone in nearby:
        points = rest[surface_vertices]
        in_bbox = np.all(
            (points >= bone.bounds[0]) & (points <= bone.bounds[1]), axis=1)
        if not np.any(in_bbox):
            continue
        ids = surface_vertices[in_bbox]
        try:
            inside = bone.contains(rest[ids])
            exempt.update(int(vi) for vi in ids[inside])
        except Exception as exc:
            print(f"[{prepared['name']}] Rest-inside collision query failed: {exc}")
    # The displayed anatomical skin is independently embedded in the tet
    # volume.  Preserve its own rest intersections instead of inferring them
    # from a nearby regularized boundary vertex.
    render_rest = getattr(surface, 'tet_render_vertices_rest', None)
    render_exempt = set()
    if render_rest is not None:
        render_rest = np.asarray(render_rest, dtype=np.float64)
        for bone in nearby:
            in_bbox = np.all(
                (render_rest >= bone.bounds[0]) &
                (render_rest <= bone.bounds[1]), axis=1)
            ids = np.where(in_bbox)[0]
            if len(ids) == 0:
                continue
            try:
                inside = bone.contains(render_rest[ids])
                render_exempt.update(int(i) for i in ids[inside])
            except Exception as exc:
                print(f"[{prepared['name']}] Rest-skin collision query failed: {exc}")
    prepared['collision_exempt_skin_vertices'] = render_exempt
    print(f"[{prepared['name']}] EMU rest collision exemptions: {len(exempt)} "
          f"simulation / {len(render_exempt)} anatomical-skin vertices "
          f"inside {len(nearby)} nearby rest bone meshes")
    return exempt


def _select_emu_collision_bones(group_name, all_bones, reference_points):
    """Select anatomically relevant posed bones without a swept AABB."""
    from scipy.spatial import cKDTree
    lowered = str(group_name).lower()
    side = 'L_' if lowered.startswith('l_') else (
        'R_' if lowered.startswith('r_') else '')
    # Rectus and the other thigh muscles can contact this compact chain.  A
    # large posed-muscle AABB previously admitted 17--23 unrelated bones.
    thigh_tokens = (
        'femor', 'vastus', 'adductor', 'sartorius', 'gracilis',
        'glute', 'tensor_fascia')
    if side and any(token in lowered for token in thigh_tokens):
        allowed = {
            side + 'Os_Coxae', side + 'Femur', side + 'Patella',
            side + 'Tibia_Fibula', 'Saccrum_Coccyx',
        }
        selected = [
            mesh for mesh in all_bones
            if mesh.metadata.get('bone_name', '') in allowed]
        if selected:
            print(f"[{group_name}] EMU collision bones: "
                  f"{[m.metadata.get('bone_name') for m in selected]}")
            return selected

    points = np.asarray(reference_points, dtype=np.float64)
    tree = cKDTree(points)
    selected = []
    for mesh in all_bones:
        name = str(mesh.metadata.get('bone_name', ''))
        if side and not (name.startswith(side) or name == 'Saccrum_Coccyx'):
            continue
        vertices = np.asarray(mesh.vertices)
        if len(vertices) > 2000:
            vertices = vertices[::max(1, len(vertices) // 2000)]
        distance = float(np.min(tree.query(vertices, k=1)[0]))
        if distance <= 0.04:
            selected.append(mesh)
    print(f"[{group_name}] EMU collision bones: "
          f"{[m.metadata.get('bone_name') for m in selected]}")
    return selected


def _run_group_emu_current_pose(v, group_name, surface, state):
    """Solve one EMU quasistatic step using the DART skeleton's current pose."""
    if (getattr(surface, 'tet_vertices', None) is None or
            getattr(surface, 'tet_tetrahedra', None) is None):
        raise RuntimeError("Load or tetrahedralize the group surface tet first")

    # Import lazily: Taichi/EMU startup should not affect ordinary viewer use.
    import test_emu as emu_view
    from tools import bake_emu
    # Contact offsets are transient display corrections for embedded skin
    # vertices that cannot be moved through a fixed attachment tet.
    surface.tet_render_contact_offsets = None

    if getattr(surface, '_emu_rest_vertices', None) is None:
        surface._emu_rest_vertices = np.asarray(
            surface.tet_vertices, dtype=np.float64).copy()
    rest = np.asarray(surface._emu_rest_vertices, dtype=np.float64)
    tets = np.asarray(surface.tet_tetrahedra, dtype=np.int32)
    auto_smooth = bool(state.get('emu_auto_smooth', True))
    requested_modes = max(1, int(state.get('emu_k_modes', 16)))
    # 32 modes was slower, gave no combined-pose improvement, and failed the
    # large benchmark one load step earlier.  The best tested balance is 16.
    modes = max(requested_modes, 16) if auto_smooth else requested_modes
    attachment_rings = int(np.clip(state.get('emu_attachment_rings', 1), 0, 2))
    cache_key = (len(rest), len(tets), float(np.sum(rest)),
                 id(surface.tet_tetrahedra), attachment_rings)
    cache = getattr(surface, '_emu_viewer_cache', None)

    current_pose = v.env.skel.getPositions().copy()
    try:
        if cache is None or cache.get('key') != cache_key:
            # Endpoint local coordinates are defined in the DART rest pose.
            v.env.skel.setPositions(np.zeros(v.env.skel.getNumDofs()))
            if not hasattr(v, '_emu_bone_trees'):
                v._emu_bone_trees = emu_view._load_bone_trees()
            data = {
                'vertices': rest,
                'tetrahedra': tets,
                'tet_region_labels': getattr(surface, 'tet_region_labels', None),
                'tet_component_labels': getattr(surface, 'tet_component_labels', None),
                'connected_component_fibers': getattr(
                    surface, '_connected_component_fibers', None),
            }
            prepared = emu_view.prepare_group_data(
                data, group_name, v.env.skel, v._emu_bone_trees,
                source_path='live viewer group',
                attachment_rings=attachment_rings)
            prepared['collision_exempt_vertices'] = _emu_rest_inside_bone_vertices(
                v, prepared, surface)
            precomp = bake_emu.precompute_emu(
                prepared['vertices'], prepared['tetrahedra'],
                prepared['fixed_vertices'], prepared['axis_coordinate'],
                k_modes=modes)
            cache = {'key': cache_key, 'prepared': prepared, 'precomp': precomp,
                     'k_modes': modes}
            surface._emu_viewer_cache = cache
        elif cache.get('k_modes') != modes:
            prepared = cache['prepared']
            cache['precomp'] = bake_emu.precompute_emu(
                prepared['vertices'], prepared['tetrahedra'],
                prepared['fixed_vertices'], prepared['axis_coordinate'],
                k_modes=modes)
            cache['k_modes'] = modes
    finally:
        # The requested pose is authoritative; never leave DART in rest pose.
        v.env.skel.setPositions(current_pose)

    prepared = cache['prepared']
    precomp = cache['precomp']
    rigid_targets = bake_emu.compute_rigid_blend_positions(
        prepared['lbs_bindings'], v.env.skel,
        prepared['axis_coordinate'])
    preview_F = bake_emu._deformation_gradients_from_q(
        rigid_targets, prepared['tetrahedra'], precomp['Dm_inv'])
    preview_J = np.linalg.det(preview_F)
    preview_stretch = np.linalg.svd(preview_F, compute_uv=False)[:, 0]
    print(f"[{group_name}] EMU direct-pose preview (diagnostic only): "
          f"inverted={int(np.sum(preview_J <= 0.0))}, "
          f"J min={float(np.min(preview_J)):.3g}, "
          f"stretch max={float(np.max(preview_stretch)):.3g}")
    direct_inverted = int(np.sum(preview_J <= 0.0))
    # Empirical sweep boundary: poses above this direct-target inversion count
    # required dozens of q-space substeps and only exchanged inversions for
    # 10x--45x local stretch. Do not spend minutes producing an unusable mesh.
    severe_pose_limit = max(1000, int(0.05 * len(prepared['tetrahedra'])))
    severe_pose = direct_inverted > severe_pose_limit
    if severe_pose:
        print(f"[{group_name}] EMU warning: direct target has "
              f"{direct_inverted} inverted tets (preview limit "
              f"{severe_pose_limit}); attempting adaptive continuation")

    labels = getattr(surface, 'tet_region_labels', None)
    if labels is None or len(labels) != len(prepared['tetrahedra']):
        labels = np.full(len(prepared['tetrahedra']), 'belly', dtype=object)
        print(f"[{group_name}] EMU warning: tet region labels unavailable; treating all tets as belly")
    else:
        labels = np.asarray(labels).astype(str)
    tendon_mask = np.char.find(labels, 'tendon') >= 0
    belly_mask = ~tendon_mask
    region_counts = _count_simple_labels(labels.tolist())
    print(f"[{group_name}] EMU tet material regions: {region_counts}")
    has_tendon_guides = any(
        'tendon' in _mesh_part_from_name(name)
        for name in _zygote_group_guide_names(v, group_name))
    if has_tendon_guides and not np.any(tendon_mask):
        raise RuntimeError(
            "group has tendon guides but the loaded tet has no tendon-labeled "
            "elements; retetrahedralize or reload it to refresh material labels")

    muscle_E = max(float(state.get('emu_muscle_youngs', 6e6)), 1.0)
    tendon_E = max(float(state.get('emu_tendon_youngs', 4.5e8)), 1.0)
    poisson = float(np.clip(state.get('emu_poisson', 0.49), 0.0, 0.499))
    youngs = np.where(tendon_mask, tendon_E, muscle_E)
    mu, lam = bake_emu.lame_parameters(youngs, poisson)
    activation_value = float(np.clip(state.get('emu_activation', 0.0), 0.0, 1.0))
    max_active_stress = max(
        float(state.get('emu_max_active_stress', 6e6)), 0.0)
    active_coefficient = activation_value * max_active_stress
    activation = np.where(belly_mask, active_coefficient, 0.0)
    print(f"[{group_name}] EMU materials: belly E={muscle_E:.3g} Pa, "
          f"tendon E={tendon_E:.3g} Pa, ratio={tendon_E/muscle_E:.3g}; "
          f"activation={activation_value:.3g}, "
          f"active coefficient={active_coefficient:.3g} Pa")

    fixed_mask = np.zeros(len(prepared['vertices']), dtype=bool)
    fixed_mask[prepared['fixed_vertices']] = True
    bone_trimeshes = []
    muscle_surfaces = None
    if bool(state.get('emu_bone_collision', True)):
        from viewer.fem_sim import _build_bone_trimeshes
        all_bones = _build_bone_trimeshes(
            v, {group_name: surface}, v.env.skel, verbose=True)
        bone_trimeshes = _select_emu_collision_bones(
            group_name, all_bones, rigid_targets)
        surface_vertices = np.unique(prepared['surface_faces']).astype(np.int32)
        collision_exempt = (
            set(prepared.get('collision_exempt_vertices', set()))
            if bool(state.get('emu_ignore_rest_inside', True)) else set()
        )
        muscle_surfaces = [{
            'surf_verts': surface_vertices,
            'fixed_set': set(prepared['fixed_vertices']),
            'collision_exempt_set': collision_exempt,
            'offset': 0,
        }]
        print(f"[{group_name}] EMU collision: {len(bone_trimeshes)} nearby posed bone meshes, "
              f"{len(surface_vertices)} surface vertices, "
              f"{len(collision_exempt)} rest-inside exempt")

    effective_alpha = max(float(state.get('emu_alpha', 1.0)), 0.0)
    effective_iters = max(int(state.get('emu_max_iters', 5)), 5) \
        if auto_smooth else max(1, int(state.get('emu_max_iters', 5)))
    requested_load_steps = max(1, int(state.get('emu_load_steps', 10)))
    # Adaptive continuation: the user value is the minimum resolution, while
    # difficult poses get finer attachment increments automatically. The
    # direct preview is only used to choose step density; EMU still solves the
    # actual path and validates every accepted increment.
    preview_severity = max(
        float(np.max(preview_stretch)) / 2.0,
        float(direct_inverted) / max(1.0, 0.01 * len(prepared['tetrahedra'])))
    adaptive_cap = 32 if severe_pose else 24
    adaptive_steps = int(np.clip(
        np.ceil(4.0 + 2.0 * preview_severity), 4, adaptive_cap))
    load_steps = max(requested_load_steps, adaptive_steps)
    print(f"[{group_name}] EMU adaptive attachment steps: "
          f"requested={requested_load_steps}, selected={load_steps}, "
          f"severity={preview_severity:.3g}")
    started = time.time()
    positions = prepared['vertices'].copy()
    warm_F = precomp['G'] @ positions.ravel()
    info = None
    qspace_fallbacks = 0
    for load_step in range(1, load_steps + 1):
        previous_positions = positions.copy()
        previous_fraction = (load_step - 1) / load_steps
        fraction = load_step / load_steps
        step_targets = bake_emu.compute_rigid_blend_positions_at_fraction(
            prepared['lbs_bindings'], v.env.skel,
            prepared['axis_coordinate'], fraction)
        is_final_step = load_step == load_steps
        print(f"[{group_name}] EMU attachment continuation "
              f"{load_step}/{load_steps} ({fraction:.0%})")
        positions, info = bake_emu.emu_solve(
            positions, precomp, fixed_mask, step_targets, mu, lam,
            effective_alpha, max_iters=effective_iters,
            verbose=(load_step == 1 or is_final_step), use_gpu=False,
            activation=activation * fraction,
            warm_F=warm_F,
            # Bone meshes are at the final DART pose, so contact is physically
            # meaningful only after the attachment continuation reaches 100%.
            bone_trimeshes=(bone_trimeshes or None) if is_final_step else None,
            muscle_surfaces=muscle_surfaces if is_final_step else None,
            margin=max(float(state.get('emu_collision_margin', 0.003)), 0.0),
            collision_stiffness=max(
                float(state.get('emu_collision_stiffness', 1e7)), 0.0),
            collision_iterations=max(
                int(state.get('emu_collision_iterations', 1)), 1))
        # Start the next load increment from the deformation gradients of the
        # reconstructed continuous mesh. Carrying discontinuous independent F
        # across increments accumulated a huge, non-physical ACAP residual.
        warm_F = precomp['G'] @ positions.ravel()
        step_F = bake_emu._deformation_gradients_from_q(
            positions, prepared['tetrahedra'], precomp['Dm_inv'])
        step_J = np.linalg.det(step_F)
        step_stretch = np.linalg.svd(step_F, compute_uv=False)[:, 0]
        print(f"[{group_name}] EMU continuation quality {load_step}/{load_steps}: "
              f"inverted={int(np.sum(step_J <= 0.0))}, "
              f"J min={float(np.min(step_J)):.3g}, "
              f"stretch max={float(np.max(step_stretch)):.3g}")
        if np.max(step_stretch) >= 3.0:
            print(f"[{group_name}] EMU high-stretch warning at load step "
                  f"{load_step}/{load_steps}: max={np.max(step_stretch):.3g}")
        step_max_stretch = float(np.max(step_stretch))
        unsafe_jacobian = bool(np.any(step_J <= 0.02))
        unsafe_stretch = step_max_stretch > 4.0
        if unsafe_jacobian or unsafe_stretch:
            print(f"[{group_name}] Reduced EMU unsafe at load step "
                  f"{load_step}/{load_steps} "
                  f"(J min={np.min(step_J):.3g}, stretch={step_max_stretch:.3g}); "
                  f"trying exact q-space quality fallback")
            fallback_positions = None
            fallback_info = None
            # If the reduced reconstruction is still orientation-preserving,
            # it is the closest and best initial state for exact relaxation.
            if np.min(step_J) > 0.02:
                direct_relaxed, direct_info = (
                    bake_emu.relax_positions_with_jacobian_barrier(
                        positions, precomp, fixed_mask, step_targets,
                        mu, lam, max_iters=30,
                        activation=activation * fraction))
                if direct_relaxed is not None:
                    direct_F = bake_emu._deformation_gradients_from_q(
                        direct_relaxed, prepared['tetrahedra'], precomp['Dm_inv'])
                    direct_stretch = np.linalg.svd(
                        direct_F, compute_uv=False)[:, 0]
                    direct_min_j = float(np.min(np.linalg.det(direct_F)))
                    direct_max_stretch = float(np.max(direct_stretch))
                    if (np.all(np.isfinite(direct_stretch)) and
                            direct_min_j > 0.02 and
                            direct_max_stretch <= 8.0 and
                            (not unsafe_stretch or
                             direct_max_stretch < 0.95 * step_max_stretch)):
                        fallback_positions = direct_relaxed
                        fallback_info = direct_info
                        print(f"[{group_name}] q-space direct fallback accepted, "
                              f"J min={direct_info['min_j']:.3g}, "
                              f"stretch max={direct_max_stretch:.3g}")
            # Adaptively traverse only this unsafe interval. Keep accepted
            # progress, halve a failed increment, and grow it after success.
            if fallback_positions is None:
                trial_positions = previous_positions.copy()
                previous_sub_targets = (
                    bake_emu.compute_rigid_blend_positions_at_fraction(
                        prepared['lbs_bindings'], v.env.skel,
                        prepared['axis_coordinate'], previous_fraction))
                current_fraction = previous_fraction
                interval = fraction - previous_fraction
                increment = interval / 4.0
                minimum_increment = interval / 1024.0
                accepted_substeps = 0
                rejected_substeps = 0
                fallback_attempts = 0
                while (current_fraction < fraction - 1e-12 and
                        fallback_attempts < 32):
                    fallback_attempts += 1
                    sub_fraction = min(fraction, current_fraction + increment)
                    sub_targets = bake_emu.compute_rigid_blend_positions_at_fraction(
                        prepared['lbs_bindings'], v.env.skel,
                        prepared['axis_coordinate'], sub_fraction)
                    # Predictor: carry every vertex with the incremental
                    # blended rigid field before snapping the hard cap. This
                    # prevents 3-fixed/1-free sliver tets from leaving their
                    # free vertex behind during a cap rotation.
                    predicted = trial_positions + sub_targets - previous_sub_targets
                    relaxed, fallback_info = (
                        bake_emu.relax_positions_with_jacobian_barrier(
                            predicted, precomp, fixed_mask, sub_targets,
                            mu, lam, max_iters=40,
                            activation=activation * sub_fraction))
                    if relaxed is None:
                        rejected_substeps += 1
                        increment *= 0.5
                        if increment < minimum_increment:
                            break
                        continue
                    relaxed_F = bake_emu._deformation_gradients_from_q(
                        relaxed, prepared['tetrahedra'], precomp['Dm_inv'])
                    relaxed_J = np.linalg.det(relaxed_F)
                    relaxed_stretch = np.linalg.svd(
                        relaxed_F, compute_uv=False)[:, 0]
                    if (np.min(relaxed_J) <= 0.02 or
                            not np.all(np.isfinite(relaxed_stretch)) or
                            float(np.max(relaxed_stretch)) > 8.0):
                        rejected_substeps += 1
                        increment *= 0.5
                        if increment < minimum_increment:
                            break
                        continue
                    trial_positions = relaxed
                    previous_sub_targets = sub_targets
                    current_fraction = sub_fraction
                    accepted_substeps += 1
                    increment = min(increment * 1.5,
                                    fraction - current_fraction)
                if current_fraction >= fraction - 1e-12:
                    fallback_positions = trial_positions
                    print(f"[{group_name}] adaptive q-space fallback accepted: "
                          f"{accepted_substeps} accepted / "
                          f"{rejected_substeps} rejected substeps, "
                          f"J min={fallback_info['min_j']:.3g}")
            if fallback_positions is None:
                reason = fallback_info.get('reason', fallback_info.get('message', 'unsafe')) \
                    if fallback_info else 'unsafe'
                raise RuntimeError(
                    f"EMU continuation became invalid at load step "
                    f"{load_step}/{load_steps}; q-space fallback failed: {reason}")
            positions = fallback_positions
            warm_F = precomp['G'] @ positions.ravel()
            qspace_fallbacks += 1

    # Contact must be checked on the exact anatomical shell that the viewer
    # displays.  That shell is embedded in the regularized simulation volume
    # and can penetrate a bone even when the hidden tet boundary does not.
    skin_projected = 0
    skin_residual = 0
    render_rest = getattr(surface, 'tet_render_vertices_rest', None)
    render_indices = getattr(surface, 'tet_render_vertex_indices', None)
    render_weights = getattr(surface, 'tet_render_vertex_weights', None)
    render_tet_rest = getattr(surface, 'tet_render_tet_rest_vertices', None)
    if (bone_trimeshes and render_rest is not None and
            render_indices is not None and render_weights is not None and
            render_tet_rest is not None):
        positions, skin_projected, skin_residual, skin_offsets = (
            bake_emu.project_embedded_skin_out_of_bones(
                positions, precomp, bone_trimeshes,
                render_rest, render_indices, render_weights, render_tet_rest,
                prepared['fixed_vertices'],
                exempt_tet_vertices=(
                    prepared.get('collision_exempt_vertices', set())
                    if bool(state.get('emu_ignore_rest_inside', True)) else set()),
                exempt_skin_vertices=(
                    prepared.get('collision_exempt_skin_vertices', set())
                    if bool(state.get('emu_ignore_rest_inside', True)) else set()),
                margin=max(float(state.get('emu_collision_margin', 0.003)), 0.0),
                passes=max(int(state.get('emu_collision_iterations', 1)) + 3, 4)))
        if skin_residual:
            surface.tet_render_contact_offsets = np.asarray(
                skin_offsets, dtype=np.float64)
            _invalidate_tet_draw_cache(surface)
        print(f"[{group_name}] EMU anatomical-skin collision: "
              f"projected={skin_projected}, residual={skin_residual}")
        eligible_skin_count = max(
            len(render_rest) - len(prepared.get(
                'collision_exempt_skin_vertices', set())), 1)
        residual_limit = max(20, int(0.005 * eligible_skin_count))
        if skin_residual > residual_limit:
            print(f"[{group_name}] EMU warning: {skin_residual} residual "
                  f"skin contacts use transient display offsets "
                  f"(preferred limit {residual_limit})")
    elapsed = time.time() - started
    if not np.all(np.isfinite(positions)):
        raise RuntimeError("EMU returned non-finite tet positions")
    safety_scale = 1.0

    deformation = bake_emu._deformation_gradients_from_q(
        positions, prepared['tetrahedra'], precomp['Dm_inv'])
    det_f = np.linalg.det(deformation)
    singular_values = np.linalg.svd(deformation, compute_uv=False)
    max_stretch = singular_values[:, 0]
    quality_text = (
        f"inverted={int(np.sum(det_f <= 0.0))}, "
        f"J min/p01={float(np.min(det_f)):.3g}/"
        f"{float(np.quantile(det_f, 0.01)):.3g}, "
        f"stretch p99/max={float(np.quantile(max_stretch, 0.99)):.3g}/"
        f"{float(np.max(max_stretch)):.3g}")
    print(f"[{group_name}] EMU tet quality: {quality_text}")

    if (np.any(det_f <= 0.02) or not np.all(np.isfinite(max_stretch)) or
            float(np.max(max_stretch)) > 8.0):
        raise RuntimeError(
            f"EMU final quality rejected; result was not displayed: {quality_text}")

    surface.tet_vertices = np.asarray(positions, dtype=np.float32)
    surface.is_draw_tet_mesh = True
    _invalidate_tet_draw_cache(surface)
    _deform_group_fibers_from_emu(
        v, group_name, prepared['vertices'], positions)
    terms = info.get('energy_terms', {})
    state['emu_last_status'] = (
        f"{info['iterations']} iter, E={info['energy']:.4g}, {elapsed:.2f}s; "
        f"{int(np.sum(belly_mask))} belly / {int(np.sum(tendon_mask))} tendon tets; "
        f"iso/fiber/aACAP={terms.get('isotropic', 0.0):.3g}/"
        f"{terms.get('fiber', 0.0):.3g}/{terms.get('acap_weighted', 0.0):.3g}; "
        f"activation={activation_value:.3g}, active={active_coefficient:.3g}Pa; "
        f"alpha={effective_alpha:.3g}, modes={modes}, load-steps={load_steps}, "
        f"q-space-fallbacks={qspace_fallbacks}, "
        f"collision peak/final={info.get('collisions', 0)}/"
        f"{info.get('final_collisions', 0)}, "
        f"projected={info.get('projected_collisions', 0)}, "
        f"skin-projected/residual={skin_projected}/{skin_residual}, "
        f"rest-exempt={len(prepared.get('collision_exempt_vertices', set())) if bool(state.get('emu_ignore_rest_inside', True)) else 0}; "
        f"quality-scale={safety_scale:.3g}; "
        f"{quality_text}")
    print(f"[{group_name}] EMU current-pose bake complete: {state['emu_last_status']}")
    return True


def _classify_tets_by_component_containment(v, component_names, tet_vertices, tetrahedra):
    tet_vertices = np.asarray(tet_vertices, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32)
    if len(tet_vertices) == 0 or len(tetrahedra) == 0:
        return None, None, None, None

    centroids = tet_vertices[tetrahedra].mean(axis=1)
    tet_regions = [None] * len(tetrahedra)
    tet_mixed = np.zeros(len(tetrahedra), dtype=bool)
    component_surfaces = []
    for comp_name in component_names:
        comp = v.zygote_muscle_meshes.get(comp_name)
        part = _mesh_part_from_name(comp_name)
        verts, faces, regions = _component_closed_surface_for_labels(comp, part, comp_name)
        if verts is None or faces is None or len(faces) == 0:
            continue
        priority = 0 if 'tendon' in part else 1
        component_surfaces.append({
            'name': comp_name,
            'part': part,
            'vertices': verts,
            'faces': faces,
            'regions': regions,
            'priority': priority,
        })

    # Tendon wins in overlaps/junctions. Belly fills the remaining volume.
    component_surfaces.sort(key=lambda item: item['priority'])

    try:
        import pyvista as pv
        query = pv.PolyData(centroids)
        for surface in component_surfaces:
            faces_flat = np.hstack([
                np.full((len(surface['faces']), 1), 3, dtype=np.int64),
                np.asarray(surface['faces'], dtype=np.int64)
            ]).ravel()
            poly = pv.PolyData(np.asarray(surface['vertices'], dtype=np.float64), faces_flat)
            selected = query.select_enclosed_points(poly, tolerance=1e-8, check_surface=False)
            inside = np.asarray(selected.point_data['SelectedPoints'], dtype=bool)
            for i, flag in enumerate(inside):
                if not flag:
                    continue
                if tet_regions[i] is None:
                    tet_regions[i] = {'part': surface['part'], 'component': surface['name']}
                else:
                    tet_mixed[i] = True
    except Exception as exc:
        print(f"  Tet component containment classification failed: {exc}")

    # Fallback for boundary or failed containment cases: nearest component
    # surface vertex. This keeps every tet labeled, but successful containment
    # labels above are preferred.
    try:
        from scipy.spatial import cKDTree
        all_surface_vertices = []
        all_surface_regions = []
        for surface in component_surfaces:
            all_surface_vertices.extend(np.asarray(surface['vertices'], dtype=np.float64).tolist())
            all_surface_regions.extend([
                {'part': surface['part'], 'component': surface['name']}
                for _ in range(len(surface['vertices']))
            ])
        if all_surface_vertices:
            tree = cKDTree(np.asarray(all_surface_vertices, dtype=np.float64))
            _, nearest = tree.query(centroids, k=1)
            for i, region in enumerate(tet_regions):
                if region is None:
                    tet_regions[i] = all_surface_regions[int(nearest[i])]
                    tet_mixed[i] = True
    except Exception:
        pass

    if any(region is None for region in tet_regions):
        tet_regions = [
            region if region is not None else {'part': 'unknown', 'component': 'unknown'}
            for region in tet_regions
        ]

    tet_region_labels = [region.get('part', 'unknown') for region in tet_regions]
    tet_component_labels = [region.get('component', 'unknown') for region in tet_regions]

    # Per-vertex regions are still useful for surface coloring. Use nearest
    # component surface vertices; per-tet material labels above are authoritative.
    tet_vertex_regions = None
    try:
        from scipy.spatial import cKDTree
        all_surface_vertices = []
        all_surface_regions = []
        for surface in component_surfaces:
            all_surface_vertices.extend(np.asarray(surface['vertices'], dtype=np.float64).tolist())
            all_surface_regions.extend([
                {'part': surface['part'], 'component': surface['name']}
                for _ in range(len(surface['vertices']))
            ])
        if all_surface_vertices:
            tree = cKDTree(np.asarray(all_surface_vertices, dtype=np.float64))
            _, nearest = tree.query(tet_vertices, k=1)
            tet_vertex_regions = [all_surface_regions[int(i)] for i in nearest]
    except Exception:
        tet_vertex_regions = None

    print(f"  Tet regions by component volume: {_count_simple_labels(tet_region_labels)}")
    print(f"  Tet components by component volume: {_count_simple_labels(tet_component_labels)}")
    if np.any(tet_mixed):
        print(f"  Tet classification fallback/interface tets: {int(np.sum(tet_mixed))}")
    return tet_vertex_regions, tet_region_labels, tet_component_labels, tet_mixed


def _tetrahedralize_single_contour_mesh(v, name, obj, defer=False):
    """Standalone-muscle tet path: use the original contour implementation."""
    if getattr(obj, 'contour_mesh_vertices', None) is None:
        print(f"[{name}] No contour mesh to tetrahedralize")
        return False
    try:
        result = obj.tetrahedralize_contour_mesh(
            skeleton_meshes=getattr(v, 'zygote_skeleton_meshes', None))
        ok = bool(result) if result is not None else bool(
            getattr(obj, 'tet_vertices', None) is not None and
            getattr(obj, 'tet_tetrahedra', None) is not None)
        if ok:
            obj.tet_quality_rejected = False
            print(f"[{name}] Single-muscle contour tetrahedralization applied")
        return ok
    except Exception as exc:
        print(f"[{name}] Single-muscle contour tetrahedralization failed: {exc}")
        traceback.print_exc()
        return False


def _tetrahedralize_original_surface_for_sim(v, name, obj, defer=False):
    component_names = _original_tet_component_names(v, name, obj)
    other_name, other = _linked_counterpart(v, obj)
    if other is not None:
        _connected_mesh_owner_name(v, name, obj, create=True)
    all_vertices = []
    all_faces = []
    vertex_regions = []
    cap_face_indices = []
    surface_face_count = 0
    component_ranges = []
    capped_hole_face_count = 0

    for comp_name in component_names:
        comp = v.zygote_muscle_meshes.get(comp_name)
        verts, faces = _component_original_surface(comp)
        if verts is None or faces is None:
            print(f"[{name}] Original tet: skipping {comp_name}, no source surface")
            continue
        part = _mesh_part_from_name(comp_name)
        region = {'part': part, 'component': comp_name}
        offset = len(all_vertices)
        face_start = len(all_faces)
        all_vertices.extend(verts.tolist())
        vertex_regions.extend([region.copy() for _ in range(len(verts))])
        all_faces.extend((faces + offset).tolist())
        surface_face_count += len(faces)

        capped_hole_face_count += _cap_component_boundary_loops(
            comp, comp_name, verts, faces,
            all_vertices, all_faces, vertex_regions, region,
            offset=offset, cap_face_indices=cap_face_indices)
        component_ranges.append({
            'component': comp_name,
            'part': part,
            'surface_face_start': face_start,
            'surface_face_end': face_start + len(faces),
        })

    if len(all_vertices) == 0 or len(all_faces) == 0:
        print(f"[{name}] Original tet: no assembled surface")
        return False

    eps = max(float(getattr(obj, 'linked_pair_eps', 1e-5)) * 5.0, 2e-5)
    closed_vertices, closed_faces, closed_regions = _dedupe_surface_vertices(
        all_vertices, all_faces, vertex_regions, eps=eps)
    # Face indices can change during weld/deduplication, so stale cap indices
    # from the assembled pre-weld surface are not safe to expose.
    cap_face_indices = []
    if len(closed_faces) == 0:
        print(f"[{name}] Original tet: no valid faces after weld")
        return False
    try:
        orient_mesh = trimesh.Trimesh(
            vertices=np.asarray(closed_vertices, dtype=np.float64),
            faces=np.asarray(closed_faces, dtype=np.int32),
            process=False)
        if not orient_mesh.is_winding_consistent:
            trimesh.repair.fix_normals(orient_mesh, multibody=False)
            closed_faces = np.asarray(orient_mesh.faces, dtype=np.int32)
            print(f"[{name}] Original tet: fixed inconsistent surface winding")
    except Exception as exc:
        print(f"[{name}] Original tet: winding check skipped ({exc})")

    # Preserve the assembled anatomical shell independently from the simulation
    # boundary.  The quality tet mesh may use a regularized level-set boundary,
    # while this exact shell is embedded over it for rendering.
    anatomical_render_vertices = np.asarray(closed_vertices, dtype=np.float64).copy()
    anatomical_render_faces = np.asarray(closed_faces, dtype=np.int32).copy()
    anatomical_render_regions = copy.deepcopy(closed_regions)
    anatomical_cap_face_indices = list(cap_face_indices)
    anatomical_surface_face_count = int(surface_face_count)

    if not hasattr(obj, 'target_tet_count'):
        obj.target_tet_count = 30000
    if not hasattr(obj, 'enable_tet_boundary_remesh'):
        obj.enable_tet_boundary_remesh = False
    if not hasattr(obj, 'allow_tet_meshfix_fallback'):
        obj.allow_tet_meshfix_fallback = False
    remesh_boundary = bool(getattr(obj, 'enable_tet_boundary_remesh', False))
    target_tet_count = (
        max(0, int(getattr(obj, 'target_tet_count', 30000) or 0))
        if remesh_boundary else 0
    )
    preserve_tet_surface = not remesh_boundary
    allow_meshfix_fallback = (
        bool(getattr(obj, 'allow_tet_meshfix_fallback', False))
        if remesh_boundary else False
    )
    if capped_hole_face_count > 0:
        allow_meshfix_fallback = True
    obj.preserve_tet_surface = preserve_tet_surface

    if remesh_boundary and target_tet_count > 0:
        source_vertices_for_regions = np.asarray(closed_vertices, dtype=np.float64)
        source_regions_for_regions = list(closed_regions)
        closed_vertices, closed_faces, did_simplify = _isotropic_voxel_surface_for_tet(
            name, closed_vertices, closed_faces, target_tet_count)
        if did_simplify:
            cap_face_indices = []
            surface_face_count = int(len(closed_faces))
            allow_meshfix_fallback = True
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(source_vertices_for_regions)
                _, nearest = tree.query(np.asarray(closed_vertices, dtype=np.float64), k=1)
                closed_regions = [
                    source_regions_for_regions[int(i)]
                    if int(i) < len(source_regions_for_regions) else None
                    for i in nearest
                ]
            except Exception:
                default_region = {'part': _mesh_part_from_name(name), 'component': name}
                closed_regions = [default_region.copy() for _ in range(len(closed_vertices))]

    estimated_volume = _estimate_closed_surface_volume(closed_vertices, closed_faces)
    target_maxvolume = 0.0
    if preserve_tet_surface:
        repair_msg = "hole-cap repair fallback=True" if allow_meshfix_fallback else "repair_fallback=False"
        print(f"[{name}] Clean tet: preserve_surface=True, {repair_msg}")
    elif target_tet_count > 0 and estimated_volume > 0.0:
        target_maxvolume = 0.0
        print(f"[{name}] Approx surface tet target: desired={target_tet_count}, "
              f"volume={estimated_volume:.6g}, control=surface_simplification")

    import subprocess
    import sys
    import tempfile
    tet_vertices = None
    tetrahedra = None
    render_source_verts = closed_vertices
    render_source_faces = closed_faces
    used_meshfix_fallback = False
    script = r'''
import numpy as np
import sys

inp, outp = sys.argv[1], sys.argv[2]
data = np.load(inp)
verts = np.asarray(data["vertices"], dtype=np.float64)
faces = np.asarray(data["faces"], dtype=np.int32)
maxvolume = float(data["maxvolume"][0]) if "maxvolume" in data else 0.0
preserve_surface = bool(int(data["preserve_surface"][0])) if "preserve_surface" in data else True
allow_repair = bool(int(data["allow_repair"][0])) if "allow_repair" in data else False
render_verts = verts
render_faces = faces
used_fix = False
quality_profile = "none"

def run_tet(v, f):
    import tetgen
    errors = []
    # Boundary Steiner points subdivide existing surface triangles without
    # changing the surface geometry. Forbidding them forces slivers near thin
    # tendons and attachment caps, so EMU meshes always allow them.
    for profile, ratio, dihedral in (
            ("strict_q1.2_d10", 1.2, 10.0),
            ("relaxed_q1.35_d7", 1.35, 7.0)):
        tg = tetgen.TetGen(v.copy(), f.copy())
        try:
            kwargs = dict(order=1, quality=True, minratio=ratio,
                          mindihedral=dihedral, nobisect=False)
            if maxvolume > 0:
                kwargs["maxvolume"] = maxvolume
            tg.tetrahedralize(**kwargs)
            return (np.asarray(tg.node, dtype=np.float64),
                    np.asarray(tg.elem, dtype=np.int32), profile)
        except Exception as exc:
            errors.append(f"{profile}: {exc}")
    raise RuntimeError("quality-constrained TetGen failed; " + " | ".join(errors))

try:
    tet_v, tet_t, quality_profile = run_tet(verts, faces)
except Exception as direct_exc:
    if not allow_repair:
        raise RuntimeError(
            "direct TetGen failed and pymeshfix repair fallback is disabled; "
            "enable Allow Tet Repair Fallback if you want repaired/remeshed input"
        ) from direct_exc
    import pymeshfix
    fixer = pymeshfix.MeshFix(verts.copy(), faces.copy())
    try:
        fixer.repair(verbose=False)
    except TypeError:
        fixer.repair()
    try:
        render_verts, render_faces = fixer.v, fixer.f
    except AttributeError:
        render_verts, render_faces = fixer._return_arrays()
    render_verts = np.asarray(render_verts, dtype=np.float64)
    render_faces = np.asarray(render_faces, dtype=np.int32)
    tet_v, tet_t, quality_profile = run_tet(render_verts, render_faces)
    used_fix = True

if tet_t is None or len(tet_t) == 0:
    raise RuntimeError("tetgen produced no tetrahedra")
np.savez(outp, tet_vertices=tet_v, tetrahedra=tet_t,
         render_vertices=render_verts, render_faces=render_faces,
         used_fix=np.array([1 if used_fix else 0], dtype=np.int32),
         quality_profile=np.asarray(quality_profile))
'''
    with tempfile.TemporaryDirectory(prefix="orig_tet_") as td:
        in_path = os.path.join(td, "input.npz")
        out_path = os.path.join(td, "output.npz")
        np.savez(in_path, vertices=closed_vertices, faces=closed_faces,
                 maxvolume=np.array([target_maxvolume], dtype=np.float64),
                 preserve_surface=np.array([1 if preserve_tet_surface else 0], dtype=np.int32),
                 allow_repair=np.array([1 if allow_meshfix_fallback else 0], dtype=np.int32))
        timeout_sec = 180 if target_tet_count <= 30000 else 600
        proc = subprocess.run(
            [sys.executable, "-c", script, in_path, out_path],
            capture_output=True, text=True, timeout=timeout_sec)
        if proc.returncode != 0 or not os.path.exists(out_path):
            print(f"[{name}] Original tet subprocess failed (rc={proc.returncode})")
            if proc.stdout:
                print(proc.stdout[-1000:])
            if proc.stderr:
                print(proc.stderr[-2000:])
            return False
        data = np.load(out_path)
        tet_vertices = np.asarray(data["tet_vertices"], dtype=np.float64)
        tetrahedra = np.asarray(data["tetrahedra"], dtype=np.int32)
        render_source_verts = np.asarray(data["render_vertices"], dtype=np.float64)
        render_source_faces = np.asarray(data["render_faces"], dtype=np.int32)
        if int(data["used_fix"][0]) != 0:
            used_meshfix_fallback = True
            cap_face_indices = []
            print(f"[{name}] Original tet: used pymeshfix fallback in subprocess")
        tet_quality_profile = str(np.asarray(data["quality_profile"]).item()) \
            if "quality_profile" in data else "unknown"
        print(f"[{name}] TetGen EMU quality profile: {tet_quality_profile}")

    if tet_vertices is None or tetrahedra is None or len(tetrahedra) == 0:
        print(f"[{name}] Original tet: no tetrahedra produced")
        return False
    actual_tet_count = int(len(tetrahedra))
    obj.last_actual_tet_count = actual_tet_count
    if used_meshfix_fallback and target_tet_count > 0:
        print(f"[{name}] Tet target calibration skipped: pymeshfix changed the surface")
    elif target_tet_count > 0 and target_maxvolume > 0.0 and actual_tet_count > 0:
        prev_calib = float(getattr(obj, 'tet_target_calibration', 1.0) or 1.0)
        new_calib = float(np.clip(prev_calib * actual_tet_count / float(target_tet_count),
                                  0.05, 20.0))
        obj.tet_target_calibration = new_calib
        print(f"[{name}] Tet target calibration updated: actual={actual_tet_count}, "
              f"desired={target_tet_count}, next calibration={new_calib:.4g}")
    elif target_tet_count > 0 and actual_tet_count > 0:
        print(f"[{name}] Approx surface tet count: actual={actual_tet_count}, "
              f"desired={target_tet_count}")
        if actual_tet_count > target_tet_count * 2:
            print(f"[{name}] Approx surface tet warning: actual tets are much higher than target; "
                  f"surface complexity or repair likely dominates TetGen output")

    tv = tet_vertices[tetrahedra]
    vols = np.einsum('ij,ij->i', tv[:, 1] - tv[:, 0],
                     np.cross(tv[:, 2] - tv[:, 0], tv[:, 3] - tv[:, 0]))
    neg = vols < 0
    if np.any(neg):
        tetrahedra[neg, 1], tetrahedra[neg, 2] = tetrahedra[neg, 2].copy(), tetrahedra[neg, 1].copy()
    tetrahedra, candidate_quality = _tet_volume_quality_stats(
        name, tet_vertices, tetrahedra)
    quality_failures = []
    if int(candidate_quality.get('degenerate', 0)) > 0:
        quality_failures.append(
            f"{int(candidate_quality['degenerate'])} degenerate tets")
    if int(candidate_quality.get('critical_slivers', 0)) > 0:
        quality_failures.append(
            f"{int(candidate_quality['critical_slivers'])} critical slivers")
    if float(candidate_quality.get('scaled_jacobian_p01', 0.0)) < 0.05:
        quality_failures.append(
            f"scaled-J p01={candidate_quality.get('scaled_jacobian_p01', 0.0):.3g} < 0.05")
    if float(candidate_quality.get('mean_ratio_p01', 0.0)) < 0.08:
        quality_failures.append(
            f"mean-ratio p01={candidate_quality.get('mean_ratio_p01', 0.0):.3g} < 0.08")
    if quality_failures:
        obj.tet_quality_rejected = True
        obj.last_rejected_tet_quality_stats = candidate_quality
        print(f"[{name}] EMU tet rejected: {'; '.join(quality_failures)}. "
              f"The previous tet mesh was left unchanged.")
        return False
    obj.tet_quality_rejected = False
    obj.tet_quality_stats = candidate_quality
    obj.tet_quality_profile = tet_quality_profile
    actual_tet_count = int(len(tetrahedra))
    obj.last_actual_tet_count = actual_tet_count

    from scipy.spatial import cKDTree
    # Embed the exact anatomical shell in the robust tet volume. Rendering the
    # regularized TetGen boundary directly caused the visible "Lego" result;
    # forcing it onto the source surface reintroduced slivers. Piecewise-linear
    # tet barycentrics exactly reproduce rigid and affine motion.
    import pyvista as pv
    vtk_cells = np.hstack((
        np.full((len(tetrahedra), 1), 4, dtype=np.int64),
        tetrahedra.astype(np.int64))).ravel()
    tet_grid = pv.UnstructuredGrid(
        vtk_cells,
        np.full(len(tetrahedra), pv.CellType.TETRA, dtype=np.uint8),
        tet_vertices)
    containing = np.asarray(
        tet_grid.find_containing_cell(anatomical_render_vertices),
        dtype=np.int64)
    inside = containing >= 0
    embedding_cells = containing.copy()
    if np.any(~inside):
        # Preserve affine precision for the handful of points just outside the
        # regularized shell by extrapolating from their closest tet.
        embedding_cells[~inside] = np.asarray(
            tet_grid.find_closest_cell(anatomical_render_vertices[~inside]),
            dtype=np.int64)
    skin_indices = tetrahedra[embedding_cells].astype(np.int32)
    x = tet_vertices[skin_indices]
    dm = np.stack((x[:, 0] - x[:, 3], x[:, 1] - x[:, 3],
                   x[:, 2] - x[:, 3]), axis=-1)
    rhs = anatomical_render_vertices - x[:, 3]
    bary012 = np.linalg.solve(dm, rhs[..., None])[..., 0]
    skin_weights = np.empty((len(anatomical_render_vertices), 4), dtype=np.float64)
    skin_weights[:, :3] = bary012
    skin_weights[:, 3] = 1.0 - np.sum(bary012, axis=1)
    print(f"[{name}] Anatomical skin embedding: {int(np.sum(inside))}/"
          f"{len(inside)} vertices inside simulation tets, "
          f"bary min={float(np.min(skin_weights)):.3g}")
    render_faces = anatomical_render_faces

    obj.soft_body = None
    obj.tet_vertices = tet_vertices.copy()
    obj.tet_tetrahedra = tetrahedra.copy()
    obj.tet_render_faces = render_faces.copy()
    obj.tet_render_vertices_rest = anatomical_render_vertices.copy()
    obj.tet_render_vertex_indices = skin_indices
    obj.tet_render_vertex_weights = skin_weights
    obj.tet_render_tet_rest_vertices = tet_vertices.copy()
    obj.tet_render_vertex_regions = anatomical_render_regions
    obj.tet_faces = obj.tet_render_faces
    obj.tet_sim_faces = obj._extract_tet_boundary_faces(obj.tet_tetrahedra)
    obj.tet_cap_face_indices = anatomical_cap_face_indices
    obj.tet_anchor_vertices = []
    obj.tet_surface_face_count = anatomical_surface_face_count
    obj._tet_surface_verts = None
    obj._tet_surface_normals = None
    obj._tet_cap_verts = None
    obj._tet_cap_normals = None
    obj._tet_edge_verts = None
    obj._tet_edge_source = None
    obj._tet_surface_vidx = None
    obj._tet_cap_vidx = None
    obj._tet_edge_vidx = None
    obj._tet_internal_verts = None
    obj._tet_internal_normals = None
    obj._tet_internal_colors = None
    obj._tet_internal_vidx = None
    obj._tet_internal_stride_cached = None
    obj.vertex_contour_level = getattr(obj, 'vertex_contour_level', None)

    try:
        labels = _classify_tets_by_component_containment(
            v, component_names, obj.tet_vertices, obj.tet_tetrahedra)
        (obj.tet_vertex_regions,
         obj.tet_region_labels,
         obj.tet_component_labels,
         obj.tet_region_mixed) = labels
    except Exception as exc:
        print(f"[{name}] Tet volume labeling failed, using nearest surface labels: {exc}")
        try:
            region_tree = cKDTree(closed_vertices)
            _, nearest = region_tree.query(tet_vertices, k=1)
            obj.tet_vertex_regions = [closed_regions[int(i)] for i in nearest]
            obj.tet_region_labels, obj.tet_component_labels, obj.tet_region_mixed = (
                _derive_tet_region_labels(obj.tet_tetrahedra, obj.tet_vertex_regions)
            )
        except Exception:
            obj.tet_vertex_regions = None
            obj.tet_region_labels = None
            obj.tet_component_labels = None
            obj.tet_region_mixed = None

    obj._connected_original_tet_components = component_names
    obj._connected_contour_mesh_components = component_ranges
    obj._connected_component_fibers = _collect_connected_component_fibers(v, component_names)
    for comp_name in component_names:
        comp_obj = v.zygote_muscle_meshes.get(comp_name)
        if comp_obj is not None:
            comp_obj._connected_mesh_owner_name = name
            if comp_name != name:
                comp_obj.is_draw_tet_mesh = False
    obj.is_draw_tet_mesh = True
    if not defer:
        obj.is_draw_contours = False
        obj._tetrahedralize_replayed = True
    print(f"[{name}] Tetrahedralized original surface: {len(tet_vertices)} vertices, "
          f"{len(tetrahedra)} tets from {len(component_names)} component(s); "
          f"embedded anatomical skin={len(anatomical_render_vertices)} vertices")
    return True


def _find_tendon_mesh_owner(v, tendon_name):
    if 'tendon' not in tendon_name.lower():
        return None
    for belly_name, belly in getattr(v, 'zygote_muscle_meshes', {}).items():
        if belly_name == tendon_name or not _is_component_belly_name(belly_name):
            continue
        _auto_fill_tendon_extension_names(v, belly_name, belly)
        if tendon_name in (
                getattr(belly, 'origin_tendon_extension_name', ''),
                getattr(belly, 'insertion_tendon_extension_name', '')):
            return belly_name, belly
    return None


def _connected_mesh_owner_name(v, name, obj, create=False):
    existing = getattr(obj, '_connected_mesh_owner_name', '')
    if existing and existing in getattr(v, 'zygote_muscle_meshes', {}):
        return existing

    tendon_owner = _find_tendon_mesh_owner(v, name)
    if tendon_owner is not None:
        belly_name, belly = tendon_owner
        owner = getattr(belly, '_connected_mesh_owner_name', '') or belly_name
        return owner if owner in v.zygote_muscle_meshes else belly_name

    other_name, other = _linked_counterpart(v, obj)
    if other is None:
        return None

    other_owner = getattr(other, '_connected_mesh_owner_name', '')
    if other_owner and other_owner in v.zygote_muscle_meshes:
        return other_owner
    if create:
        obj._connected_mesh_owner_name = name
        other._connected_mesh_owner_name = name
        return name
    return None


def _skip_non_owner_connected_mesh(v, name, obj, stage):
    owner = _connected_mesh_owner_name(v, name, obj, create=False)
    if owner and owner != name:
        print(f"[{name}] Skipping {stage}: connected mesh is owned by {owner}")
        obj.is_draw_contour_mesh = False
        obj.is_draw_tet_mesh = False
        return True
    return False


def _clear_connected_mesh_ownership(v, owner_name):
    for obj in getattr(v, 'zygote_muscle_meshes', {}).values():
        if getattr(obj, '_connected_mesh_owner_name', '') == owner_name:
            obj._connected_mesh_owner_name = ''


def _prepare_owned_connected_contour_mesh_source(v, name, obj):
    if _skip_non_owner_connected_mesh(v, name, obj, "Build Contour Mesh"):
        return False
    _connected_mesh_owner_name(v, name, obj, create=True)
    ok = _prepare_connected_contour_mesh_source(v, name, obj)
    if ok:
        owner = name
        for comp in getattr(obj, '_connected_contour_mesh_components', []) or []:
            for comp_name in (
                    comp.get('component'),
                    comp.get('origin_tendon'),
                    comp.get('insertion_tendon')):
                if comp_name and comp_name in v.zygote_muscle_meshes:
                    comp_obj = v.zygote_muscle_meshes[comp_name]
                    comp_obj._connected_mesh_owner_name = owner
                    if comp_name != owner:
                        comp_obj.is_draw_contour_mesh = False
                        comp_obj.is_draw_tet_mesh = False
    return True


def _dti_init(v):
    """Initialize DTI viewer state lazily."""
    if hasattr(v, 'dti_meshes'):
        return

    v.dti_root_dir = 'DTI/Subject 01'
    v.dti_obj_dir = os.path.join(v.dti_root_dir, 'BuiltMeshes')
    v.dti_legacy_obj_dir = os.path.join(v.dti_root_dir, 'Muscle Bone objs')
    v.dti_tract_dir = os.path.join(v.dti_root_dir, 'DTI fibres')
    v.dti_meshes = {}
    v.dti_tracts = {}
    v.dti_available_meshes = []
    v.dti_available_tracts = []
    v.dti_mesh_selected = 0
    v.dti_loaded_mesh_selected = 0
    v.dti_tract_selected = 0
    v.dti_loaded_tract_selected = 0
    v.dti_draw_meshes = True
    v.dti_draw_tracts = True
    v.dti_mesh_color = np.array([0.35, 0.55, 0.9], dtype=np.float32)
    v.dti_mesh_transparency = 0.35
    v.dti_tract_color = np.array([0.95, 0.15, 0.05], dtype=np.float32)
    v.dti_tract_transparency = 0.9
    v.dti_tract_line_width = 1.5
    v.dti_scale = 0.001
    v.dti_offset = np.zeros(3, dtype=np.float32)
    # Best global center correction over same-named DTI OBJ/tract pairs.
    # Per-muscle residual remains because tracts are belly-like subsets while
    # OBJs often include a larger segmented structure.
    v.dti_tract_offset = np.array([-0.04395, -0.00448, -0.03656], dtype=np.float32)
    v.dti_obj_z_origin = 1100.0
    v.dti_tract_max_fibers = 1000
    v.dti_tract_resample_points = 32
    v.mri_converted_dir = os.path.join(v.dti_root_dir, 'ConvertedNPZ')
    v.mri_available_volumes = []
    v.mri_volume_selected = 0
    v.mri_loaded = None
    v.mri_loaded_path = None
    v.mri_slice_idx = 0
    v.mri_time_idx = 0
    v.mri_window = [0.01, 0.99]
    v.mri_texture_id = None
    v.mri_texture_key = None
    v.mri_draw_3d_slice = True
    v.mri_draw_3d_stack = False
    v.mri_stack_stride = 6
    v.mri_plane_alpha = 0.75
    v.mri_scale = 0.001
    v.mri_offset = np.zeros(3, dtype=np.float32)
    v.mri_modality_mode = 0  # DTI: 0=b0 mean, 1=selected volume
    update_available_dti(v)


def _strip_dti_tract_name(path):
    name = os.path.splitext(os.path.basename(path))[0]
    if '_tracts_ext_' in name:
        base, extra = name.split('_tracts_ext_', 1)
        return f"{base}_{extra}"
    for suffix in ('_tracts_ext', '_tracts'):
        if suffix in name:
            return name.split(suffix)[0]
    return name


def update_available_dti(v):
    _dti_init(v) if not hasattr(v, 'dti_meshes') else None

    loaded_meshes = set(v.dti_meshes.keys())
    mesh_paths = sorted(glob.glob(os.path.join(v.dti_obj_dir, '*.obj')))
    v.dti_available_meshes = [
        (os.path.splitext(os.path.basename(p))[0], p)
        for p in mesh_paths
        if os.path.splitext(os.path.basename(p))[0] not in loaded_meshes
    ]

    loaded_tracts = set(v.dti_tracts.keys())
    tract_paths = sorted(glob.glob(os.path.join(v.dti_tract_dir, '*_tracts_ext*.mat')))
    v.dti_available_tracts = [
        (_strip_dti_tract_name(p), p)
        for p in tract_paths
        if _strip_dti_tract_name(p) not in loaded_tracts
    ]

    if v.dti_mesh_selected >= len(v.dti_available_meshes):
        v.dti_mesh_selected = max(0, len(v.dti_available_meshes) - 1)
    if v.dti_tract_selected >= len(v.dti_available_tracts):
        v.dti_tract_selected = max(0, len(v.dti_available_tracts) - 1)

    volume_paths = sorted(glob.glob(os.path.join(getattr(v, 'mri_converted_dir', ''), '*.npz')))
    v.mri_available_volumes = [
        (os.path.splitext(os.path.basename(p))[0], p)
        for p in volume_paths
        if not os.path.basename(p).startswith('manifest')
    ]
    if v.mri_volume_selected >= len(v.mri_available_volumes):
        v.mri_volume_selected = max(0, len(v.mri_available_volumes) - 1)


def _mri_load_volume(v, path):
    data = np.load(path)
    volume = np.asarray(data['volume'])
    is_dti = volume.ndim == 4
    affine = np.asarray(data['affine_zyx'], dtype=np.float32) if 'affine_zyx' in data.files else np.eye(4, dtype=np.float32)
    loaded = {
        'path': path,
        'name': os.path.splitext(os.path.basename(path))[0],
        'volume': volume,
        'is_dti': is_dti,
        'affine_zyx': affine,
        'spacing_zyx': np.asarray(data['spacing_zyx'], dtype=np.float32) if 'spacing_zyx' in data.files else np.ones(3, dtype=np.float32),
    }
    if is_dti:
        loaded['bvals'] = np.asarray(data['bvals'], dtype=np.float32)
        loaded['bvecs'] = np.asarray(data['bvecs'], dtype=np.float32)
    if 'gap_slices' in data.files:
        loaded['gap_slices'] = set(int(x) for x in np.asarray(data['gap_slices']).reshape(-1))
    if 'gap_after_slices' in data.files:
        loaded['gap_after_slices'] = set(int(x) for x in np.asarray(data['gap_after_slices']).reshape(-1))
    v.mri_loaded = loaded
    v.mri_loaded_path = path
    v.mri_slice_idx = min(int(v.mri_slice_idx), _mri_slice_count(v) - 1)
    v.mri_time_idx = 0
    v.mri_texture_key = None
    print(f"[MRI] Loaded {loaded['name']}: shape={volume.shape}")


def _mri_slice_count(v):
    loaded = getattr(v, 'mri_loaded', None)
    if not loaded:
        return 0
    volume = loaded['volume']
    return int(volume.shape[1] if loaded['is_dti'] else volume.shape[0])


def _mri_current_volume(v):
    loaded = getattr(v, 'mri_loaded', None)
    if not loaded:
        return None
    volume = loaded['volume']
    if not loaded['is_dti']:
        return volume
    if getattr(v, 'mri_modality_mode', 0) == 0:
        bvals = loaded.get('bvals', np.zeros(volume.shape[0], dtype=np.float32))
        mask = bvals == 0
        if np.any(mask):
            return volume[mask].mean(axis=0)
        return volume[0]
    idx = int(np.clip(getattr(v, 'mri_time_idx', 0), 0, volume.shape[0] - 1))
    return volume[idx]


def _mri_slice_image_u8(v, z_override=None):
    vol = _mri_current_volume(v)
    if vol is None:
        return None
    z_src = getattr(v, 'mri_slice_idx', 0) if z_override is None else z_override
    z = int(np.clip(z_src, 0, vol.shape[0] - 1))
    sl = np.asarray(vol[z], dtype=np.float32)
    lo_q, hi_q = getattr(v, 'mri_window', [0.01, 0.99])
    vals = sl[np.isfinite(sl)]
    vals = vals[vals > 0]
    if vals.size == 0:
        norm = np.zeros_like(sl, dtype=np.float32)
    else:
        lo, hi = np.quantile(vals, [lo_q, hi_q])
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((sl - lo) / (hi - lo), 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    rgba = np.empty((gray.shape[0], gray.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = gray
    rgba[..., 1] = gray
    rgba[..., 2] = gray
    rgba[..., 3] = 255
    return np.ascontiguousarray(rgba)


def _mri_update_texture(v, z_override=None):
    image = _mri_slice_image_u8(v, z_override=z_override)
    if image is None:
        return None, (0, 0)
    key = (
        getattr(v, 'mri_loaded_path', None),
        int(getattr(v, 'mri_slice_idx', 0) if z_override is None else z_override),
        int(getattr(v, 'mri_time_idx', 0)),
        int(getattr(v, 'mri_modality_mode', 0)),
        float(getattr(v, 'mri_window', [0.01, 0.99])[0]),
        float(getattr(v, 'mri_window', [0.01, 0.99])[1]),
    )
    h, w = image.shape[:2]
    if getattr(v, 'mri_texture_id', None) is None:
        v.mri_texture_id = glGenTextures(1)
    if getattr(v, 'mri_texture_key', None) != key:
        glBindTexture(GL_TEXTURE_2D, v.mri_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
        glBindTexture(GL_TEXTURE_2D, 0)
        v.mri_texture_key = key
    return v.mri_texture_id, (w, h)


def _mri_patient_to_viewer(v, pts):
    pts = np.asarray(pts, dtype=np.float32)
    out = np.empty_like(pts, dtype=np.float32)
    out[..., 0] = -pts[..., 0]
    out[..., 1] = pts[..., 2]
    out[..., 2] = -pts[..., 1]
    return out * float(getattr(v, 'mri_scale', 0.001)) + np.asarray(getattr(v, 'mri_offset', np.zeros(3)), dtype=np.float32)


def _mri_slice_corners(v, z):
    loaded = getattr(v, 'mri_loaded', None)
    vol = _mri_current_volume(v)
    if not loaded or vol is None:
        return None
    h, w = vol.shape[1], vol.shape[2]
    affine = loaded['affine_zyx']
    corners_zyx = np.array([
        [z, 0, 0, 1],
        [z, 0, w - 1, 1],
        [z, h - 1, w - 1, 1],
        [z, h - 1, 0, 1],
    ], dtype=np.float32).T
    patient = (affine @ corners_zyx)[:3].T
    return _mri_patient_to_viewer(v, patient)


def _dti_transform_obj_points(v, points):
    """DTI OBJ raw coords to viewer coords: Z-up MRI -> Y-up viewer."""
    pts = np.asarray(points, dtype=np.float32)
    out = np.empty_like(pts, dtype=np.float32)
    out[:, 0] = -pts[:, 0]
    out[:, 1] = pts[:, 2] + float(v.dti_obj_z_origin)
    out[:, 2] = -pts[:, 1]
    return out * float(v.dti_scale) + np.asarray(v.dti_offset, dtype=np.float32)


def _dti_transform_tract_points(v, points):
    """DTI tract raw coords to viewer coords, registered to DTI OBJ space."""
    pts = np.asarray(points, dtype=np.float32)
    out = np.empty_like(pts, dtype=np.float32)
    out[:, 0] = pts[:, 0]
    out[:, 1] = pts[:, 2]
    out[:, 2] = -pts[:, 1]
    return (out * float(v.dti_scale)
            + np.asarray(v.dti_offset, dtype=np.float32)
            + np.asarray(v.dti_tract_offset, dtype=np.float32))


def _set_dti_mesh_arrays(mesh, vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    mesh.vertices = vertices
    mesh.faces = faces
    mesh.faces_3 = mesh.faces
    mesh.faces_4 = np.empty((0, 4), dtype=np.int64)
    mesh.faces_other = []
    mesh.normals = np.asarray(tri.vertex_normals, dtype=np.float32)
    mesh.vertices_3 = np.ascontiguousarray(vertices[mesh.faces.reshape(-1)], dtype=np.float32)
    face_normals = np.asarray(tri.face_normals, dtype=np.float32)
    mesh.normals_3 = np.ascontiguousarray(np.repeat(face_normals, 3, axis=0), dtype=np.float32)
    mesh.vertices_4 = np.array([])
    mesh.normals_4 = np.array([])
    mesh.vertices_other = []
    mesh.normals_other = []
    mesh.new_vertices_3 = mesh.vertices_3.copy()
    mesh.new_vertices_4 = mesh.vertices_4.copy()
    mesh.new_vertices_other = []
    mesh.open_edges = []
    mesh.edge_groups = []
    mesh.edge_classes = []
    mesh.centroids = [np.mean(vertices, axis=0)] if len(vertices) else []
    mesh.trimesh = tri


def _set_dti_mesh_arrays_from_raw(v, mesh, raw_vertices, faces):
    vertices = _dti_transform_obj_points(v, raw_vertices)
    _set_dti_mesh_arrays(mesh, vertices, faces)


def _update_dti_mesh_transforms(v):
    for mesh in getattr(v, 'dti_meshes', {}).values():
        raw_vertices = getattr(mesh, 'dti_raw_vertices', None)
        raw_faces = getattr(mesh, 'dti_raw_faces', None)
        if raw_vertices is not None and raw_faces is not None and not getattr(mesh, 'dti_prebuilt', False):
            _set_dti_mesh_arrays_from_raw(v, mesh, raw_vertices, raw_faces)


def _dti_mesh_center(mesh):
    verts = getattr(mesh, 'vertices', None)
    if verts is None or len(verts) == 0:
        return None
    return np.asarray(verts, dtype=np.float32).mean(axis=0)


def _dti_tract_center(v, entry):
    raw = entry.get('raw_lines', None)
    if raw is None or len(raw) == 0:
        return None
    pts = _dti_transform_tract_points(v, raw)
    return np.asarray(pts, dtype=np.float32).mean(axis=0)


def align_dti_selected_tract_to_obj(v, name):
    mesh = getattr(v, 'dti_meshes', {}).get(name)
    tract = getattr(v, 'dti_tracts', {}).get(name)
    if mesh is None or tract is None:
        print(f"[DTI] Need both OBJ and tract loaded with same name: {name}")
        return
    mc = _dti_mesh_center(mesh)
    tc = _dti_tract_center(v, tract)
    if mc is None or tc is None:
        print(f"[DTI] Cannot align {name}: empty OBJ or tract")
        return
    delta = mc - tc
    v.dti_tract_offset = np.asarray(v.dti_tract_offset, dtype=np.float32) + delta.astype(np.float32)
    print(f"[DTI] Aligned {name}: added tract offset {delta}")


def _load_dti_obj_mesh(v, path):
    """Load DTI OBJ meshes, including VTK-style OBJ faces without normals."""
    mesh = MeshLoader()
    tri = trimesh.load_mesh(path, process=False)
    if not isinstance(tri, trimesh.Trimesh):
        tri = trimesh.util.concatenate(tuple(tri.geometry.values()))

    raw_vertices = np.asarray(tri.vertices, dtype=np.float32)
    faces = np.asarray(tri.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"DTI OBJ needs triangular faces: {path}")

    mesh.obj = path
    is_prebuilt = os.path.abspath(path).startswith(os.path.abspath(v.dti_obj_dir))
    mesh.dti_prebuilt = is_prebuilt
    mesh.dti_raw_vertices = raw_vertices
    mesh.dti_raw_faces = faces
    if is_prebuilt:
        _set_dti_mesh_arrays(mesh, raw_vertices * 0.01, faces)
    else:
        _set_dti_mesh_arrays_from_raw(v, mesh, raw_vertices, faces)
    return mesh


def add_dti_mesh(v, name, path):
    _dti_init(v)
    if name in v.dti_meshes:
        return
    try:
        mesh = _load_dti_obj_mesh(v, path)
        mesh.color = np.array(v.dti_mesh_color)
        mesh.transparency = v.dti_mesh_transparency
        mesh.is_draw = v.dti_draw_meshes
        v.dti_meshes[name] = mesh
        v.dti_meshes = dict(sorted(v.dti_meshes.items()))
        update_available_dti(v)
        print(f"[DTI] Loaded OBJ {name}")
    except Exception as e:
        print(f"[DTI] Failed to load OBJ {path}: {e}")


def remove_dti_mesh(v, name):
    if not hasattr(v, 'dti_meshes') or name not in v.dti_meshes:
        return
    del v.dti_meshes[name]
    if v.dti_loaded_mesh_selected >= len(v.dti_meshes):
        v.dti_loaded_mesh_selected = max(0, len(v.dti_meshes) - 1)
    update_available_dti(v)


def _resample_polyline(points, n):
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return None
    if len(points) == 1 or n <= 1:
        return np.repeat(points[:1], max(1, n), axis=0)
    seg = np.linalg.norm(points[1:] - points[:-1], axis=1)
    total = float(np.sum(seg))
    if total <= 1e-8:
        return np.repeat(points[:1], n, axis=0)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.linspace(0.0, total, n)
    out = np.empty((n, 3), dtype=np.float32)
    for c in range(3):
        out[:, c] = np.interp(target, dist, points[:, c])
    return out


def _load_dti_tract_lines(path, max_fibers, resample_points):
    from scipy.io import loadmat

    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    xyz = np.asarray(data['tracts_xyz'], dtype=np.float32)
    if xyz.shape[0] == 3:
        xyz = xyz.T
    fibindex = np.asarray(data['fibindex'], dtype=np.int64)
    if fibindex.ndim == 1:
        fibindex = fibindex.reshape(1, 2)

    num_fibers = len(fibindex)
    take = min(num_fibers, max(1, int(max_fibers)))
    rows = np.linspace(0, num_fibers - 1, take, dtype=np.int64)
    segments = []
    sampled = 0
    for row in rows:
        a, b = fibindex[row]
        a -= 1
        b -= 1
        lo = max(0, min(a, b))
        hi = min(len(xyz) - 1, max(a, b))
        if hi <= lo:
            continue
        pts = xyz[lo:hi + 1]
        if a > b:
            pts = pts[::-1]
        pts = _resample_polyline(pts, int(resample_points))
        if pts is None or len(pts) < 2:
            continue
        line = np.empty(((len(pts) - 1) * 2, 3), dtype=np.float32)
        line[0::2] = pts[:-1]
        line[1::2] = pts[1:]
        segments.append(line)
        sampled += 1

    if not segments:
        return np.empty((0, 3), dtype=np.float32), num_fibers, 0
    return np.ascontiguousarray(np.concatenate(segments, axis=0), dtype=np.float32), num_fibers, sampled


def add_dti_tract(v, name, path):
    _dti_init(v)
    if name in v.dti_tracts:
        return
    try:
        raw_lines, total_count, sampled_count = _load_dti_tract_lines(
            path, v.dti_tract_max_fibers, v.dti_tract_resample_points)
        v.dti_tracts[name] = {
            'path': path,
            'raw_lines': raw_lines,
            'total_count': total_count,
            'sampled_count': sampled_count,
            'is_draw': v.dti_draw_tracts,
        }
        v.dti_tracts = dict(sorted(v.dti_tracts.items()))
        update_available_dti(v)
        print(f"[DTI] Loaded tract {name}: {sampled_count}/{total_count} fibers")
    except Exception as e:
        print(f"[DTI] Failed to load tract {path}: {e}")
        traceback.print_exc()


def remove_dti_tract(v, name):
    if not hasattr(v, 'dti_tracts') or name not in v.dti_tracts:
        return
    del v.dti_tracts[name]
    if v.dti_loaded_tract_selected >= len(v.dti_tracts):
        v.dti_loaded_tract_selected = max(0, len(v.dti_tracts) - 1)
    update_available_dti(v)


def draw_dti_overlays(v):
    if not hasattr(v, 'dti_meshes'):
        return

    draw_mri_volume_planes(v)

    # Draw transparent subject meshes first, then draw tracts without depth
    # writes/testing so internal fibers remain visible through the surface.
    for name, obj in v.dti_meshes.items():
        if obj.is_draw:
            obj.draw([obj.color[0], obj.color[1], obj.color[2], obj.transparency])

    visible = [
        entry for entry in v.dti_tracts.values()
        if entry.get('is_draw', True) and len(entry.get('raw_lines', [])) > 0
    ]
    if not visible:
        return

    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDisable(GL_DEPTH_TEST)
    glDepthMask(GL_FALSE)
    glEnableClientState(GL_VERTEX_ARRAY)
    glLineWidth(v.dti_tract_line_width)
    glColor4f(v.dti_tract_color[0], v.dti_tract_color[1],
              v.dti_tract_color[2], v.dti_tract_transparency)
    for entry in visible:
        verts = _dti_transform_tract_points(v, entry['raw_lines'])
        verts = np.ascontiguousarray(verts, dtype=np.float32)
        entry['_draw_keepalive'] = verts
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glVertexPointer(3, GL_FLOAT, 0, verts)
        glDrawArrays(GL_LINES, 0, len(verts))
    glDisableClientState(GL_VERTEX_ARRAY)
    glDepthMask(GL_TRUE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)


def _draw_mri_textured_slice(v, z, alpha):
    tex_id, _ = _mri_update_texture(v, z_override=z)
    corners = _mri_slice_corners(v, z)
    if tex_id is None or corners is None:
        return

    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glColor4f(1.0, 1.0, 1.0, float(alpha))
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0); glVertex3fv(corners[0])
    glTexCoord2f(1.0, 0.0); glVertex3fv(corners[1])
    glTexCoord2f(1.0, 1.0); glVertex3fv(corners[2])
    glTexCoord2f(0.0, 1.0); glVertex3fv(corners[3])
    glEnd()
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)
    glEnable(GL_LIGHTING)

    glDisable(GL_LIGHTING)
    glColor4f(0.1, 0.8, 1.0, min(1.0, float(alpha) + 0.2))
    glBegin(GL_LINE_LOOP)
    for p in corners:
        glVertex3fv(p)
    glEnd()
    glEnable(GL_LIGHTING)


def draw_mri_volume_planes(v):
    loaded = getattr(v, 'mri_loaded', None)
    if not loaded:
        return
    if getattr(v, 'mri_draw_3d_stack', False):
        n = _mri_slice_count(v)
        stride = max(1, int(getattr(v, 'mri_stack_stride', 6)))
        for z in range(0, n, stride):
            _draw_mri_textured_slice(v, z, getattr(v, 'mri_plane_alpha', 0.75) * 0.35)
    if getattr(v, 'mri_draw_3d_slice', True):
        _draw_mri_textured_slice(v, int(getattr(v, 'mri_slice_idx', 0)), getattr(v, 'mri_plane_alpha', 0.75))


def draw_mri_ui(v):
    if imgui.tree_node("MRI/DTI Images", imgui.TREE_NODE_DEFAULT_OPEN):
        imgui.text(f"Volume source: {v.mri_converted_dir}")
        if imgui.button("Rescan Volumes", width=button_width):
            update_available_dti(v)

        names = [n for n, _ in getattr(v, 'mri_available_volumes', [])]
        imgui.text(f"Available volumes: {len(names)}")
        if names:
            _, v.mri_volume_selected = imgui.listbox(
                "##mri_available_volumes", v.mri_volume_selected, names, 6)
            if imgui.button("Load Volume", width=button_width):
                _, path = v.mri_available_volumes[v.mri_volume_selected]
                _mri_load_volume(v, path)

        loaded = getattr(v, 'mri_loaded', None)
        if loaded:
            vol = loaded['volume']
            imgui.separator()
            imgui.text(f"Loaded: {loaded['name']}")
            imgui.text(f"Shape: {vol.shape}")
            imgui.text(f"Spacing z/y/x: {loaded['spacing_zyx']}")
            gaps = loaded.get('gap_slices', set())
            if gaps:
                imgui.text_colored(f"Empty gap slices: {len(gaps)}", 1.0, 0.65, 0.1, 1.0)
            gap_after = loaded.get('gap_after_slices', set())
            if gap_after:
                imgui.text_colored(f"Physical gaps between slices: {len(gap_after)}", 1.0, 0.65, 0.1, 1.0)
            if loaded['is_dti']:
                if imgui.radio_button("b0 mean", v.mri_modality_mode == 0):
                    v.mri_modality_mode = 0
                    v.mri_texture_key = None
                imgui.same_line()
                if imgui.radio_button("Volume", v.mri_modality_mode == 1):
                    v.mri_modality_mode = 1
                    v.mri_texture_key = None
                if v.mri_modality_mode == 1:
                    max_t = vol.shape[0] - 1
                    changed, v.mri_time_idx = imgui.slider_int("DTI Volume", int(v.mri_time_idx), 0, max_t)
                    if changed:
                        v.mri_texture_key = None
                    bvals = loaded.get('bvals', None)
                    if bvals is not None and 0 <= int(v.mri_time_idx) < len(bvals):
                        imgui.text(f"b-value: {float(bvals[int(v.mri_time_idx)]):.1f}")

            max_z = max(0, _mri_slice_count(v) - 1)
            changed, v.mri_slice_idx = imgui.slider_int("Slice", int(v.mri_slice_idx), 0, max_z)
            if changed:
                v.mri_texture_key = None
            if int(v.mri_slice_idx) in loaded.get('gap_slices', set()):
                imgui.text_colored("Current slice is an inter-slab gap", 1.0, 0.45, 0.1, 1.0)
            if int(v.mri_slice_idx) in loaded.get('gap_after_slices', set()):
                imgui.text_colored("Physical gap after this slice", 1.0, 0.45, 0.1, 1.0)
            changed_lo, v.mri_window[0] = imgui.slider_float("Low Quantile", float(v.mri_window[0]), 0.0, 0.4, "%.3f")
            changed_hi, v.mri_window[1] = imgui.slider_float("High Quantile", float(v.mri_window[1]), 0.6, 1.0, "%.3f")
            if v.mri_window[1] <= v.mri_window[0]:
                v.mri_window[1] = min(1.0, v.mri_window[0] + 0.01)
            if changed_lo or changed_hi:
                v.mri_texture_key = None

            tex_id, size = _mri_update_texture(v)
            if tex_id is not None:
                avail_w = max(120.0, imgui.get_content_region_available_width() - 10)
                w, h = size
                scale = min(avail_w / max(1, w), 320.0 / max(1, h))
                imgui.image(tex_id, w * scale, h * scale, uv0=(0, 1), uv1=(1, 0))

            _, v.mri_draw_3d_slice = imgui.checkbox("Draw Current Slice in 3D", v.mri_draw_3d_slice)
            _, v.mri_draw_3d_stack = imgui.checkbox("Draw Slice Stack in 3D", v.mri_draw_3d_stack)
            _, v.mri_plane_alpha = imgui.slider_float("3D Slice Alpha", float(v.mri_plane_alpha), 0.05, 1.0, "%.2f")
            changed_stride, stride = imgui.input_int("3D Stack Stride", int(v.mri_stack_stride), 1, 4)
            if changed_stride:
                v.mri_stack_stride = max(1, stride)
            changed_scale, v.mri_scale = imgui.input_float("MRI Scale", float(v.mri_scale), 0.0001, 0.001, "%.6f")
            _, v.mri_offset = imgui.input_float3("MRI Offset", *v.mri_offset)
        imgui.tree_pop()


def draw_dti_ui(v):
    _dti_init(v)
    if imgui.tree_node("DTI"):
        imgui.begin_child("DTIScroll", width=0, height=420, border=True,
                          flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)

        draw_mri_ui(v)
        imgui.separator()

        if imgui.button("Rescan DTI", width=button_width):
            update_available_dti(v)
        imgui.text(f"Mesh source: {v.dti_obj_dir}")

        changed, v.dti_draw_meshes = imgui.checkbox("Draw DTI OBJs", v.dti_draw_meshes)
        if changed:
            for obj in v.dti_meshes.values():
                obj.is_draw = v.dti_draw_meshes
        changed, v.dti_mesh_color = imgui.color_edit3("OBJ Color", *v.dti_mesh_color)
        if changed:
            for obj in v.dti_meshes.values():
                obj.color = np.array(v.dti_mesh_color)
        changed, v.dti_mesh_transparency = imgui.slider_float(
            "OBJ Transparency", v.dti_mesh_transparency, 0.0, 1.0)
        if changed:
            for obj in v.dti_meshes.values():
                obj.transparency = v.dti_mesh_transparency

        if imgui.tree_node("OBJ Meshes", imgui.TREE_NODE_DEFAULT_OPEN):
            names = [n for n, _ in v.dti_available_meshes]
            imgui.text(f"Available: {len(names)}")
            if names:
                _, v.dti_mesh_selected = imgui.listbox(
                    "##dti_available_meshes", v.dti_mesh_selected, names, 8)
                if imgui.button("Load OBJ", width=button_width):
                    name, path = v.dti_available_meshes[v.dti_mesh_selected]
                    add_dti_mesh(v, name, path)
            loaded = list(v.dti_meshes.keys())
            imgui.text(f"Loaded: {len(loaded)}")
            if loaded:
                _, v.dti_loaded_mesh_selected = imgui.listbox(
                    "##dti_loaded_meshes", v.dti_loaded_mesh_selected, loaded, 8)
                if imgui.button("Delete OBJ", width=button_width):
                    remove_dti_mesh(v, loaded[v.dti_loaded_mesh_selected])
                imgui.same_line()
                if imgui.button("Delete All OBJ", width=button_width):
                    for name in list(v.dti_meshes.keys()):
                        remove_dti_mesh(v, name)
            imgui.tree_pop()

        changed, v.dti_draw_tracts = imgui.checkbox("Draw DTI Tracts", v.dti_draw_tracts)
        if changed:
            for entry in v.dti_tracts.values():
                entry['is_draw'] = v.dti_draw_tracts
        _, v.dti_tract_color = imgui.color_edit3("Tract Color", *v.dti_tract_color)
        _, v.dti_tract_transparency = imgui.slider_float(
            "Tract Transparency", v.dti_tract_transparency, 0.0, 1.0)
        _, v.dti_tract_line_width = imgui.slider_float(
            "Tract Line Width", v.dti_tract_line_width, 0.1, 6.0)
        changed_scale, v.dti_scale = imgui.input_float("DTI Scale", v.dti_scale, 0.0001, 0.001, "%.6f")
        changed_offset, v.dti_offset = imgui.input_float3("DTI Offset", *v.dti_offset)
        _, v.dti_tract_offset = imgui.input_float3("Tract Reg Offset", *v.dti_tract_offset)
        changed_origin, v.dti_obj_z_origin = imgui.input_float(
            "OBJ Z Origin", v.dti_obj_z_origin, 10.0, 100.0, "%.3f")
        if changed_scale or changed_offset or changed_origin:
            _update_dti_mesh_transforms(v)
        changed_max, max_f = imgui.input_int("Max Fibers/Tract", int(v.dti_tract_max_fibers), 100, 1000)
        if changed_max:
            v.dti_tract_max_fibers = max(1, max_f)
        changed_pts, pts = imgui.input_int("Resample Points", int(v.dti_tract_resample_points), 1, 8)
        if changed_pts:
            v.dti_tract_resample_points = max(2, pts)

        if imgui.tree_node("Fiber Tracts", imgui.TREE_NODE_DEFAULT_OPEN):
            names = [n for n, _ in v.dti_available_tracts]
            imgui.text(f"Available: {len(names)}")
            if names:
                _, v.dti_tract_selected = imgui.listbox(
                    "##dti_available_tracts", v.dti_tract_selected, names, 8)
                if imgui.button("Load Tract", width=button_width):
                    name, path = v.dti_available_tracts[v.dti_tract_selected]
                    add_dti_tract(v, name, path)
            loaded = list(v.dti_tracts.keys())
            imgui.text(f"Loaded: {len(loaded)}")
            if loaded:
                _, v.dti_loaded_tract_selected = imgui.listbox(
                    "##dti_loaded_tracts", v.dti_loaded_tract_selected, loaded, 8)
                selected = loaded[v.dti_loaded_tract_selected]
                entry = v.dti_tracts[selected]
                _, entry['is_draw'] = imgui.checkbox(f"Draw Selected##{selected}", entry.get('is_draw', True))
                imgui.text(f"{entry['sampled_count']} shown / {entry['total_count']} total")
                if imgui.button("Align Selected Tract to OBJ", width=wide_button_width):
                    align_dti_selected_tract_to_obj(v, selected)
                if imgui.button("Delete Tract", width=button_width):
                    remove_dti_tract(v, selected)
                imgui.same_line()
                if imgui.button("Delete All Tracts", width=button_width):
                    for name in list(v.dti_tracts.keys()):
                        remove_dti_tract(v, name)
            imgui.tree_pop()

        imgui.end_child()
        imgui.tree_pop()


def draw_zygote_ui(v):
    """Top-level Zygote tree node with scrollable child region."""
    if imgui.tree_node("Zygote", imgui.TREE_NODE_DEFAULT_OPEN):
        # Scrollable child region for Zygote menu
        imgui.begin_child("ZygoteScroll", width=0, height=500, border=True, flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
        draw_zygote_muscle_ui(v)
        draw_zygote_skeleton_ui(v)
        imgui.end_child()  # End ZygoteScroll
        imgui.tree_pop()


def _draw_zygote_muscle_body(v, name, obj):
    # Two-column layout: "Process All" button on left, individual buttons on right
    imgui.columns(2, f"cols##{name}", border=False)
    imgui.set_column_width(0, 120)

    # Left column: Process button with vertical slider
    num_process_buttons = 12  # Match number of buttons on right
    process_all_height = num_process_buttons * imgui.get_frame_height() + (num_process_buttons - 1) * imgui.get_style().item_spacing[1]

    # Process step slider is global so adjusting it on any
    # muscle propagates to every other muscle's tree.
    if not hasattr(v, 'global_process_step'):
        v.global_process_step = getattr(obj, '_process_step', 12)
    if not hasattr(obj, '_process_step'):
        obj._process_step = v.global_process_step

    # Vertical slider for step selection (top=1, bottom=12)
    # Reversed min/max (12, 1) makes value increase downward
    changed, new_val = imgui.v_slider_int(
        f"##step{name}", 20, process_all_height, obj._process_step, 12, 1)
    if changed:
        obj._process_step = new_val
        v.global_process_step = new_val
        for _other in v.zygote_muscle_meshes.values():
            _other._process_step = new_val
    imgui.same_line()

    # Step names matching button order (1=top, 12=bottom)
    # 1:Scalar, 2:Contours, 3:FillGap, 4:Transitions, 5:Smooth, 6:Cut, 7:StreamSmooth, 8:Select, 9:Build, 10:Resample, 11:Mesh, 12:Tet
    step_names = ['', 'Scalar', 'Contours', 'Fill Gap', 'Transitions', 'Smooth', 'Cut', 'StreamSmooth', 'Select', 'Build', 'Resample', 'Mesh', 'Tet']

    # Check if pipeline is paused waiting for manual step
    pipeline_paused = hasattr(obj, '_pipeline_paused_at') and obj._pipeline_paused_at is not None

    # Button label changes if paused
    if pipeline_paused:
        btn_label = f"Resume\nfrom {obj._pipeline_paused_at}\n({step_names[obj._pipeline_paused_at]})"
    else:
        btn_label = f"Process\n1 to {obj._process_step}\n({step_names[obj._process_step]})"

    if imgui.button(f"{btn_label}##{name}", width=75, height=process_all_height):
        try:
            max_step = obj._process_step
            # If resuming, start from paused step
            if pipeline_paused:
                start_step = obj._pipeline_paused_at
                obj._pipeline_paused_at = None
                print(f"[{name}] Resuming pipeline from step {start_step} to {max_step}...")
            else:
                start_step = 1
                print(f"[{name}] Running pipeline steps 1 to {max_step}...")

            # Step 1: Scalar Field
            defer = getattr(obj, 'animate_process', False)
            if start_step <= 1 <= max_step and len(obj.edge_groups) > 0 and len(obj.edge_classes) > 0:
                print(f"  [1/{max_step}] Computing Scalar Field...")
                _t0 = time.time()
                has_counterpart = bool(getattr(obj, 'linked_counterpart_name', ''))
                if (has_counterpart
                        and getattr(obj, 'linked_drive_counterpart', False)
                        and getattr(obj, 'linked_use_shared_scalar', True)):
                    _ensure_counterpart_scalar(v, name, obj, defer=defer)
                else:
                    obj.compute_scalar_field(defer=defer)
                    if has_counterpart and getattr(obj, 'linked_drive_counterpart', False):
                        _ensure_counterpart_scalar(v, name, obj, defer=defer)
                print(f"  [1/{max_step}] Done in {time.time()-_t0:.3f}s")

            # Step 2: Find Contours
            if start_step <= 2 <= max_step and obj.scalar_field is not None:
                print(f"  [2/{max_step}] Finding Contours...")
                _t0 = time.time()
                obj.find_contours(skeleton_meshes=v.zygote_skeleton_meshes, spacing_scale=obj.contour_spacing_scale, defer=defer)
                if getattr(obj, 'linked_drive_counterpart', False):
                    _apply_master_contour_schedule_to_counterpart(
                        v, name, obj, defer=defer, label="contours")
                print(f"  [2/{max_step}] Done in {time.time()-_t0:.3f}s")
                if not defer:
                    obj.is_draw_bounding_box = True

            # Step 3: Fill Gaps
            if start_step <= 3 <= max_step and obj.contours is not None and len(obj.contours) > 0:
                print(f"  [3/{max_step}] Filling Gaps...")
                _t0 = time.time()
                obj.refine_contours(max_spacing_threshold=0.01, defer=defer)
                if getattr(obj, 'linked_drive_counterpart', False):
                    _apply_master_contour_schedule_to_counterpart(
                        v, name, obj, defer=defer, label="gap-filled contours")
                print(f"  [3/{max_step}] Done in {time.time()-_t0:.3f}s")

            # Step 4: Find Transitions
            if start_step <= 4 <= max_step and obj.scalar_field is not None:
                print(f"  [4/{max_step}] Finding Transitions...")
                _t0 = time.time()
                field_min = float(obj.scalar_field.min())
                field_max = float(obj.scalar_field.max())
                scalar_min, scalar_max = field_min, field_max
                if hasattr(obj, 'origin_contour_value') and hasattr(obj, 'insertion_contour_value'):
                    o_val, i_val = obj.origin_contour_value, obj.insertion_contour_value
                    if o_val != i_val:
                        proposed_min, proposed_max = min(o_val, i_val), max(o_val, i_val)
                        if proposed_min >= field_min and proposed_max <= field_max:
                            scalar_min, scalar_max = proposed_min, proposed_max
                exp_origin = len(obj.contours[0]) if obj.contours and len(obj.contours) > 0 else None
                exp_insertion = len(obj.contours[-1]) if obj.contours and len(obj.contours) > 0 else None
                obj.find_all_transitions(scalar_min=scalar_min, scalar_max=scalar_max, num_samples=200,
                                        expected_origin=exp_origin, expected_insertion=exp_insertion)
                if obj.contours is not None and len(obj.contours) > 0:
                    obj.add_transitions_to_contours(defer=defer)
                if getattr(obj, 'linked_drive_counterpart', False):
                    _apply_master_contour_schedule_to_counterpart(
                        v, name, obj, defer=defer, label="transition contours")
                print(f"  [4/{max_step}] Done in {time.time()-_t0:.3f}s")

            # Step 5: Smooth (z, x, bp - before cut)
            if start_step <= 5 <= max_step and obj.contours is not None and len(obj.contours) > 0:
                print(f"  [5/{max_step}] Smoothening (z, x, bp)...")
                _t0 = time.time()
                obj.smoothen_all(defer=defer)
                _run_counterpart_step(v, name, obj, 5, defer=defer)
                print(f"  [5/{max_step}] Done in {time.time()-_t0:.3f}s")

            # Step 6: Cut
            if start_step <= 6 <= max_step and obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None:
                print(f"  [6/{max_step}] Cutting streams...")
                _t0 = time.time()
                obj.cut_streams_animated(defer=defer, cut_method=obj.cutting_method, muscle_name=name)
                _run_counterpart_step(v, name, obj, 6, defer=defer)
                print(f"  [6/{max_step}] Done in {time.time()-_t0:.3f}s")
                # Check if waiting for manual cut
                if hasattr(obj, '_manual_cut_pending') and obj._manual_cut_pending or hasattr(obj, '_manual_cut_data') and obj._manual_cut_data is not None:
                    obj._pipeline_paused_at = 7  # Resume from step 7 after cut is complete
                    print(f"  [6/{max_step}] Waiting for manual cut - pipeline paused")
                    raise StopIteration("Manual cut pending")

            # Step 7: Stream Smooth (z, x, bp - after cut)
            if start_step <= 7 <= max_step and hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                print(f"  [7/{max_step}] Stream Smoothening (z, x, bp)...")
                _t0 = time.time()
                obj.stream_smoothen_all(defer=defer)
                _run_counterpart_step(v, name, obj, 7, defer=defer)
                print(f"  [7/{max_step}] Done in {time.time()-_t0:.3f}s")

            # Step 8: Contour Select
            if start_step <= 8 <= max_step and hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                print(f"  [8/{max_step}] Selecting contours...")
                _t0 = time.time()
                obj.select_levels()
                if not (hasattr(obj, '_level_select_window_open') and obj._level_select_window_open):
                    _run_counterpart_step(v, name, obj, 8, defer=defer)
                print(f"  [8/{max_step}] Done in {time.time()-_t0:.3f}s")
                # Check if waiting for manual level selection
                if hasattr(obj, '_level_select_window_open') and obj._level_select_window_open:
                    obj._pipeline_paused_at = 9  # Resume from step 9 after selection
                    print(f"  [8/{max_step}] Waiting for level selection - pipeline paused")
                    raise StopIteration("Level selection pending")

            # Step 9: Build Fiber
            if start_step <= 9 <= max_step and hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                print(f"  [9/{max_step}] Building fibers...")
                _t0 = time.time()
                _ensure_level_selection_applied(v, name, obj, defer=defer)
                obj._belly_waypoints_before_tendon_extension = None
                obj.build_fibers(skeleton_meshes=v.zygote_skeleton_meshes, defer=defer)
                if (getattr(obj, 'origin_tendon_extension_name', '')
                        or getattr(obj, 'insertion_tendon_extension_name', '')):
                    _extend_belly_fibers_with_tendons(v, name, obj)
                _run_counterpart_step(v, name, obj, 9, defer=defer)
                print(f"  [9/{max_step}] Done in {time.time()-_t0:.3f}s")
                if defer:
                    obj._level_select_replayed = False

            # Step 10: Resample Contours
            if start_step <= 10 <= max_step and obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None:
                print(f"  [10/{max_step}] Resampling Contours...")
                _t0 = time.time()
                _resample_contours_with_links(v, name, obj, defer=defer)
                _resample_linked_tendon_extensions(v, name, obj, defer=defer)
                print(f"  [10/{max_step}] Done in {time.time()-_t0:.3f}s")
                if defer:
                    obj._build_fibers_replayed = False

            # Step 11: Build Contour Mesh
            if start_step <= 11 <= max_step and obj.contours is not None and len(obj.contours) > 0 and obj.draw_contour_stream is not None:
                print(f"  [11/{max_step}] Building Contour Mesh...")
                _t0 = time.time()
                if _prepare_owned_connected_contour_mesh_source(v, name, obj):
                    obj.build_contour_mesh(defer=defer)
                    if not _connected_source_has_linked_components(obj):
                        _run_counterpart_step(v, name, obj, 11, defer=defer)
                print(f"  [11/{max_step}] Done in {time.time()-_t0:.3f}s")
                if defer:
                    obj._resample_replayed = False

            # Step 12: Tetrahedralize
            if start_step <= 12 <= max_step:
                print(f"  [12/{max_step}] Tetrahedralizing...")
                _t0 = time.time()
                if _skip_non_owner_connected_mesh(v, name, obj, "Tetrahedralize"):
                    print(f"  [12/{max_step}] Skipped in {time.time()-_t0:.3f}s")
                    tet_ok = False
                else:
                    tet_ok = _tetrahedralize_single_contour_mesh(v, name, obj, defer=defer)
                    if tet_ok and not _connected_source_has_linked_components(obj):
                        _run_counterpart_step(v, name, obj, 12, defer=defer)
                if tet_ok and obj.tet_vertices is not None and not _skip_non_owner_connected_mesh(v, name, obj, "Tetrahedralize display"):
                    if defer:
                        obj._extract_internal_tet_edges()
                        obj._classify_tet_faces_into_bands()
                        obj._tetrahedralize_replayed = False
                    else:
                        obj.is_draw_contours = False
                        obj.is_draw_tet_mesh = True
                        obj._tetrahedralize_replayed = True
                status = "Done" if tet_ok else "Failed"
                print(f"  [12/{max_step}] {status} in {time.time()-_t0:.3f}s")

            obj._pipeline_paused_at = None  # Clear pause state on completion
            print(f"[{name}] Pipeline complete (steps {start_step}-{max_step})!")
        except StopIteration:
            pass  # Manual cut pending - pipeline paused gracefully
        except Exception as e:
            print(f"[{name}] Pipeline error: {e}")
            traceback.print_exc()

    # Right column: Individual buttons (use -1 to auto-fill column width)
    imgui.next_column()
    col_button_width = 180  # Fits in the right column

    # Helper for button coloring based on process step
    def colored_button(label, step_num, width):
        will_run = step_num <= obj._process_step
        if will_run:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.6, 0.2, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.7, 0.3, 1.0)
        clicked = imgui.button(label, width=width)
        if will_run:
            imgui.pop_style_color(2)
        return clicked

    animate = getattr(obj, 'animate_process', False)
    replay_w = 25
    proc_w = col_button_width - replay_w - imgui.get_style().item_spacing[0] if animate else col_button_width

    if colored_button(f"Scalar Field##{name}", 1, proc_w):
        if len(obj.edge_groups) > 0 and len(obj.edge_classes) > 0:
            try:
                _t0 = time.time()
                has_counterpart = bool(getattr(obj, 'linked_counterpart_name', ''))
                if (has_counterpart
                        and getattr(obj, 'linked_drive_counterpart', False)
                        and getattr(obj, 'linked_use_shared_scalar', True)):
                    _ensure_counterpart_scalar(v, name, obj, defer=animate)
                else:
                    obj.compute_scalar_field(defer=animate)
                    if has_counterpart and getattr(obj, 'linked_drive_counterpart', False):
                        _ensure_counterpart_scalar(v, name, obj, defer=animate)
                print(f"[{name}] Scalar Field done in {time.time()-_t0:.3f}s")
            except Exception as e:
                print(f"[{name}] Scalar Field error: {e}")
        else:
            print(f"[{name}] Need edge_groups and edge_classes")
    if animate and obj._scalar_anim_target_colors is not None:
        imgui.same_line()
        if imgui.button(f">##{name}_scalar_replay", width=replay_w):
            obj.replay_scalar_animation()

    if colored_button(f"Find Contours##{name}", 2, proc_w):
        if obj.scalar_field is not None:
            try:
                _t0 = time.time()
                obj.find_contours(skeleton_meshes=v.zygote_skeleton_meshes, spacing_scale=obj.contour_spacing_scale, defer=animate)
                if getattr(obj, 'linked_drive_counterpart', False):
                    _apply_master_contour_schedule_to_counterpart(
                        v, name, obj, defer=animate, label="contours")
                print(f"[{name}] Find Contours done in {time.time()-_t0:.3f}s")
                if not animate:
                    obj.is_draw_bounding_box = True
            except Exception as e:
                print(f"[{name}] Find Contours error: {e}")
        else:
            print(f"[{name}] Prerequisites: Run 'Scalar Field' first")
    # Contour replay: only available after scalar replay has been played
    if animate and obj.contours is not None and len(obj.contours) > 0 and getattr(obj, '_scalar_replayed', False):
        imgui.same_line()
        if imgui.button(f">##{name}_contour_replay", width=replay_w):
            obj.replay_contour_animation()
    if colored_button(f"Fill Gaps##{name}", 3, proc_w):
        if obj.contours is not None and len(obj.contours) > 0:
            try:
                _t0 = time.time()
                obj.refine_contours(max_spacing_threshold=0.01, defer=animate)
                if getattr(obj, 'linked_drive_counterpart', False):
                    _apply_master_contour_schedule_to_counterpart(
                        v, name, obj, defer=animate, label="gap-filled contours")
                print(f"[{name}] Fill Gaps done in {time.time()-_t0:.3f}s")
            except Exception as e:
                print(f"[{name}] Fill Gaps error: {e}")
        else:
            print(f"[{name}] Prerequisites: Run 'Find Contours' first")
    # Fill gaps replay: only available after contour replay has been played
    if animate and getattr(obj, '_fill_gaps_inserted_indices', None) is not None and getattr(obj, '_contour_replayed', False):
        imgui.same_line()
        if imgui.button(f">##{name}_fillgaps_replay", width=replay_w):
            obj.replay_fill_gaps_animation()

    # Find Transitions button - fast scan for contour count changes (step 4)
    if colored_button(f"Find Transitions##{name}", 4, proc_w):
        if hasattr(obj, 'scalar_field') and obj.scalar_field is not None:
            try:
                _t0 = time.time()
                # Use actual scalar field range
                field_min = float(obj.scalar_field.min())
                field_max = float(obj.scalar_field.max())
                scalar_min = field_min
                scalar_max = field_max
                # Optionally narrow to origin/insertion if set AND within field range
                if hasattr(obj, 'origin_contour_value') and hasattr(obj, 'insertion_contour_value'):
                    o_val = obj.origin_contour_value
                    i_val = obj.insertion_contour_value
                    # Only use if different AND within the actual scalar field range
                    if o_val != i_val:
                        proposed_min = min(o_val, i_val)
                        proposed_max = max(o_val, i_val)
                        if proposed_min >= field_min and proposed_max <= field_max:
                            scalar_min = proposed_min
                            scalar_max = proposed_max
                        else:
                            print(f"[{name}] Origin/insertion values ({o_val}, {i_val}) outside field range [{field_min:.4f}, {field_max:.4f}], using field range")
                print(f"[{name}] Scanning scalar range: {scalar_min:.4f} to {scalar_max:.4f}")
                # Try to get expected origin/insertion counts from existing contours
                exp_origin = None
                exp_insertion = None
                if hasattr(obj, 'contours') and obj.contours is not None and len(obj.contours) > 0:
                    exp_origin = len(obj.contours[0])
                    exp_insertion = len(obj.contours[-1])
                    print(f"[{name}] Expected counts from contours: origin={exp_origin}, insertion={exp_insertion}")
                obj.find_all_transitions(scalar_min=scalar_min, scalar_max=scalar_max, num_samples=200,
                                        expected_origin=exp_origin, expected_insertion=exp_insertion)
                # Auto-add transitions to contours if contours exist
                if obj.contours is not None and len(obj.contours) > 0:
                    obj.add_transitions_to_contours(defer=animate)
                if getattr(obj, 'linked_drive_counterpart', False):
                    _apply_master_contour_schedule_to_counterpart(
                        v, name, obj, defer=animate, label="transition contours")
                print(f"[{name}] Find Transitions done in {time.time()-_t0:.3f}s")
            except Exception as e:
                print(f"[{name}] Find Transitions error: {e}")
                traceback.print_exc()
        else:
            print(f"[{name}] Prerequisites: Run 'Scalar Field' first")
    # Transitions replay: only available after fill gaps replay has been played
    if animate and getattr(obj, '_transitions_inserted_indices', None) is not None and (getattr(obj, '_fill_gaps_replayed', False) or getattr(obj, '_fill_gaps_inserted_indices', None) is None):
        imgui.same_line()
        if imgui.button(f">##{name}_transitions_replay", width=replay_w):
            obj.replay_transitions_animation()

    # Step 5: Smoothen buttons
    sub_button_width = (col_button_width - 8) // 3  # 3 buttons with small margins
    if animate:
        # Single "Smooth" button with replay when animate is on
        if colored_button(f"Smooth##{name}", 5, proc_w):
            if obj.contours is not None and len(obj.contours) > 0:
                try:
                    _t0 = time.time()
                    obj.smoothen_all(defer=True)
                    _run_counterpart_step(v, name, obj, 5, defer=True)
                    print(f"[{name}] Smooth done in {time.time()-_t0:.3f}s")
                except Exception as e:
                    print(f"[{name}] Smooth error: {e}")
            else:
                print(f"[{name}] Prerequisites: Run 'Find Contours' first")
        if getattr(obj, '_smooth_bp_after', None) is not None and (getattr(obj, '_transitions_replayed', False) or getattr(obj, '_transitions_inserted_indices', None) is None):
            imgui.same_line()
            if imgui.button(f">##{name}_smooth_replay", width=replay_w):
                obj.replay_smooth_animation()
    else:
        # Individual z, x, bp buttons when animate is off
        if colored_button(f"z##{name}", 5, sub_button_width):
            if obj.contours is not None and len(obj.contours) > 0:
                try:
                    _t0 = time.time()
                    obj.smoothen_contours_z()
                    _run_counterpart_step(v, name, obj, 5, defer=False)
                    print(f"[{name}] Smooth Z done in {time.time()-_t0:.3f}s")
                except Exception as e:
                    print(f"[{name}] Smoothen Z error: {e}")
            else:
                print(f"[{name}] Prerequisites: Run 'Find Contours' first")
        imgui.same_line(spacing=4)
        if colored_button(f"x##{name}", 5, sub_button_width):
            if obj.contours is not None and len(obj.contours) > 0:
                try:
                    _t0 = time.time()
                    obj.smoothen_contours_x()
                    _run_counterpart_step(v, name, obj, 5, defer=False)
                    print(f"[{name}] Smooth X done in {time.time()-_t0:.3f}s")
                except Exception as e:
                    print(f"[{name}] Smoothen X error: {e}")
            else:
                print(f"[{name}] Prerequisites: Run 'Find Contours' first")
        imgui.same_line(spacing=4)
        if colored_button(f"bp##{name}", 5, sub_button_width):
            if obj.contours is not None and len(obj.contours) > 0:
                try:
                    _t0 = time.time()
                    obj.smoothen_contours_bp()
                    _run_counterpart_step(v, name, obj, 5, defer=False)
                    print(f"[{name}] Smooth BP done in {time.time()-_t0:.3f}s")
                except Exception as e:
                    print(f"[{name}] Smoothen BP error: {e}")
            else:
                print(f"[{name}] Prerequisites: Run 'Find Contours' first")

    # Step 6: Cut (standalone button)
    cut_w = col_button_width - replay_w - imgui.get_style().item_spacing[0] if animate else col_button_width
    if colored_button(f"Cut##{name}", 6, cut_w):
        if obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None and len(obj.bounding_planes) > 0:
            try:
                _t0 = time.time()
                if animate:
                    obj.cut_streams_animated(defer=True, cut_method=obj.cutting_method, muscle_name=name)
                else:
                    obj.cut_streams(cut_method=obj.cutting_method, muscle_name=name)
                _run_counterpart_step(v, name, obj, 6, defer=animate)
                print(f"[{name}] Cut done in {time.time()-_t0:.3f}s")
            except Exception as e:
                print(f"[{name}] Cut Streams error: {e}")
                traceback.print_exc()
        else:
            print(f"[{name}] Prerequisites: Run 'Find Contours' first")
    if animate and getattr(obj, '_cut_color_after', None) is not None and (getattr(obj, '_smooth_replayed', False) or getattr(obj, '_smooth_bp_after', None) is None):
        imgui.same_line()
        if imgui.button(f">##{name}_cut_replay", width=replay_w):
            obj.replay_cut_animation()

    # Step 7: Stream Smoothen buttons: z, x, bp (3 buttons in same row - after cut)
    if animate:
        # Single "Stream Smooth" button with replay when animate is on
        if colored_button(f"Stream Smooth##{name}", 7, proc_w):
            if hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                try:
                    _t0 = time.time()
                    obj.stream_smoothen_all(defer=True)
                    _run_counterpart_step(v, name, obj, 7, defer=True)
                    print(f"[{name}] Stream Smooth done in {time.time()-_t0:.3f}s")
                except Exception as e:
                    print(f"[{name}] Stream Smooth error: {e}")
            else:
                print(f"[{name}] Prerequisites: Run 'Cut' first")
        if getattr(obj, '_stream_smooth_bp_after', None) is not None and (getattr(obj, '_cut_replayed', False) or getattr(obj, '_cut_color_after', None) is None):
            imgui.same_line()
            if imgui.button(f">##{name}_stream_smooth_replay", width=replay_w):
                obj.replay_stream_smooth_animation()
    else:
        # Individual z, x, bp buttons when animate is off
        if colored_button(f"z##stream{name}", 7, sub_button_width):
            if hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                try:
                    _t0 = time.time()
                    obj.smoothen_contours_z()
                    _run_counterpart_step(v, name, obj, 7, defer=False)
                    print(f"[{name}] Stream Smooth Z done in {time.time()-_t0:.3f}s")
                except Exception as e:
                    print(f"[{name}] Stream Smoothen Z error: {e}")
            else:
                print(f"[{name}] Prerequisites: Run 'Cut' first")
        imgui.same_line(spacing=4)
        if colored_button(f"x##stream{name}", 7, sub_button_width):
            if hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                try:
                    _t0 = time.time()
                    obj.smoothen_contours_x()
                    _run_counterpart_step(v, name, obj, 7, defer=False)
                    print(f"[{name}] Stream Smooth X done in {time.time()-_t0:.3f}s")
                except Exception as e:
                    print(f"[{name}] Stream Smoothen X error: {e}")
            else:
                print(f"[{name}] Prerequisites: Run 'Cut' first")
        imgui.same_line(spacing=4)
        if colored_button(f"bp##stream{name}", 7, sub_button_width):
            if hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                try:
                    _t0 = time.time()
                    obj.smoothen_contours_bp()
                    _run_counterpart_step(v, name, obj, 7, defer=False)
                    print(f"[{name}] Stream Smooth BP done in {time.time()-_t0:.3f}s")
                except Exception as e:
                    print(f"[{name}] Stream Smoothen BP error: {e}")
            else:
                print(f"[{name}] Prerequisites: Run 'Cut' first")

    # Step 8: Contour Select
    if colored_button(f"Contour Select##{name}", 8, proc_w if animate else col_button_width):
        if hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
            try:
                _t0 = time.time()
                obj.select_levels()
                print(f"[{name}] Contour Select done in {time.time()-_t0:.3f}s")
            except Exception as e:
                print(f"[{name}] Select Levels error: {e}")
                traceback.print_exc()
        else:
            print(f"[{name}] Prerequisites: Run 'Cut' first")
    if animate and getattr(obj, '_level_select_anim_original', None) is not None and (getattr(obj, '_stream_smooth_replayed', False) or getattr(obj, '_stream_smooth_bp_after', None) is None):
        imgui.same_line()
        if imgui.button(f">##{name}_level_select_replay", width=replay_w):
            obj.replay_level_select_animation()

    # Step 9: Build Fiber (standalone button)
    if colored_button(f"Build Fiber##{name}", 9, proc_w if animate else col_button_width):
        if hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
            try:
                _t0 = time.time()
                _ensure_level_selection_applied(v, name, obj, defer=animate)
                obj._belly_waypoints_before_tendon_extension = None
                obj.build_fibers(skeleton_meshes=v.zygote_skeleton_meshes, defer=animate)
                if (getattr(obj, 'origin_tendon_extension_name', '')
                        or getattr(obj, 'insertion_tendon_extension_name', '')):
                    _extend_belly_fibers_with_tendons(v, name, obj)
                _run_counterpart_step(v, name, obj, 9, defer=animate)
                print(f"[{name}] Build Fiber done in {time.time()-_t0:.3f}s")
            except Exception as e:
                print(f"[{name}] Build Fibers error: {e}")
                traceback.print_exc()
        else:
            print(f"[{name}] Prerequisites: Run 'Cut' first")
    if animate and getattr(obj, '_fiber_anim_waypoints', None) is not None and getattr(obj, '_level_select_replayed', False) and not getattr(obj, '_level_select_anim_active', False):
        imgui.same_line()
        if imgui.button(f">##{name}_fiber_replay", width=replay_w):
            obj.replay_fiber_animation()

    # Step 10: Resample Contours
    if colored_button(f"Resample Contours##{name}", 10, proc_w if animate else col_button_width):
        if obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None:
            try:
                _t0 = time.time()
                _resample_contours_with_links(v, name, obj, defer=animate)
                _resample_linked_tendon_extensions(v, name, obj, defer=animate)
                print(f"[{name}] Resample Contours done in {time.time()-_t0:.3f}s")
            except Exception as e:
                print(f"[{name}] Resample Contours error: {e}")
        else:
            print(f"[{name}] Prerequisites: Run 'Smoothen Contours' first")
    if animate and getattr(obj, '_resample_anim_data', None) is not None and (getattr(obj, '_build_fibers_replayed', False) or getattr(obj, '_fiber_anim_waypoints', None) is None) and not getattr(obj, '_fiber_anim_active', False):
        imgui.same_line()
        if imgui.button(f">##{name}_resample_replay", width=replay_w):
            obj.replay_resample_animation()

    # Step 11: Build Contour Mesh
    if colored_button(f"Build Contour Mesh##{name}", 11, proc_w if animate else col_button_width):
        if obj.contours is not None and len(obj.contours) > 0 and obj.draw_contour_stream is not None:
            try:
                _t0 = time.time()
                if _prepare_owned_connected_contour_mesh_source(v, name, obj):
                    obj.build_contour_mesh(defer=animate)
                    if not _connected_source_has_linked_components(obj):
                        _run_counterpart_step(v, name, obj, 11, defer=animate)
                print(f"[{name}] Build Contour Mesh done in {time.time()-_t0:.3f}s")
            except Exception as e:
                print(f"[{name}] Build Contour Mesh error: {e}")
                traceback.print_exc()
        else:
            print(f"[{name}] Prerequisites: Run 'Build Fiber' first")
    if animate and getattr(obj, '_mesh_anim_face_bands', None) is not None and (getattr(obj, '_resample_replayed', False) or getattr(obj, '_resample_anim_data', None) is None) and not getattr(obj, '_resample_anim_active', False):
        imgui.same_line()
        if imgui.button(f">##{name}_mesh_replay", width=replay_w):
            obj.replay_mesh_animation()

    if not hasattr(obj, 'target_tet_count'):
        obj.target_tet_count = 30000
    if not hasattr(obj, 'enable_tet_boundary_remesh'):
        obj.enable_tet_boundary_remesh = False
    if not hasattr(obj, 'allow_tet_meshfix_fallback'):
        obj.allow_tet_meshfix_fallback = False
    # Boundary-remesh controls and tet-count diagnostics are intentionally
    # kept out of the process-button strip. Group/EMU controls own those
    # advanced settings; ordinary component pipelines should stay compact.
    obj.preserve_tet_surface = not bool(obj.enable_tet_boundary_remesh)

    # Step 12: Tetrahedralize
    if colored_button(f"Tetrahedralize##{name}", 12, proc_w if animate else col_button_width):
        try:
            _t0 = time.time()
            if _skip_non_owner_connected_mesh(v, name, obj, "Tetrahedralize"):
                print(f"[{name}] Tetrahedralize skipped in {time.time()-_t0:.3f}s")
                tet_ok = False
            else:
                tet_ok = _tetrahedralize_single_contour_mesh(v, name, obj, defer=animate)
                if tet_ok and not _connected_source_has_linked_components(obj):
                    _run_counterpart_step(v, name, obj, 12, defer=animate)
            status = "done" if tet_ok else "failed"
            print(f"[{name}] Tetrahedralize {status} in {time.time()-_t0:.3f}s")
            if tet_ok and obj.tet_vertices is not None and not _skip_non_owner_connected_mesh(v, name, obj, "Tetrahedralize display"):
                if animate:
                    obj._extract_internal_tet_edges()
                    obj._classify_tet_faces_into_bands()
                    obj._tetrahedralize_replayed = False
                else:
                    obj.is_draw_contours = False
                    obj.is_draw_tet_mesh = True
                    obj._tetrahedralize_replayed = True
        except Exception as e:
            print(f"[{name}] Tetrahedralize error: {e}")
            traceback.print_exc()
    if animate and getattr(obj, '_tet_anim_internal_edges', None) is not None and (getattr(obj, '_build_mesh_replayed', False) or getattr(obj, '_mesh_anim_face_bands', None) is None) and not getattr(obj, '_tet_anim_active', False):
        imgui.same_line()
        if imgui.button(f">##{name}_tet_replay", width=replay_w):
            obj.replay_tet_animation()

    # End two-column layout - back to full width for remaining GUI elements
    imgui.columns(1)

    # Reset process button
    reset_width = button_width * 2 + imgui.get_style().item_spacing[0]
    if imgui.button(f"Reset Process##{name}", width=reset_width):
        _clear_connected_mesh_ownership(v, name)
        obj.reset_process()

    # Save/Load contours buttons
    contour_filepath = f"{v.zygote_muscle_dir}{name}.contours.json"
    if imgui.button(f"Save Contour##{name}", width=button_width):
        if obj.contours is not None and len(obj.contours) > 0:
            try:
                obj.save_contours(contour_filepath)
            except Exception as e:
                print(f"[{name}] Save Contours error: {e}")
        else:
            print(f"[{name}] No contours to save")
    imgui.same_line()
    if imgui.button(f"Load Contour##{name}", width=button_width):
        try:
            obj.load_contours(contour_filepath)
        except Exception as e:
            print(f"[{name}] Load Contours error: {e}")

    if imgui.button(f"Save Tet##{name}", width=button_width):
        if hasattr(obj, 'tet_vertices') and obj.tet_vertices is not None:
            try:
                obj.save_tetrahedron_mesh(name)
            except Exception as e:
                print(f"[{name}] Save Tet error: {e}")
        else:
            print(f"[{name}] No tetrahedron mesh to save")
    imgui.same_line()
    if imgui.button(f"Load Tet##{name}", width=button_width):
        try:
            obj.soft_body = None  # Reset soft body when loading new tet
            obj.load_tetrahedron_mesh(name)
            if obj.tet_vertices is not None:
                obj.is_draw = False  # Disable mesh draw
                obj.is_draw_contours = False
                obj.is_draw_tet_mesh = True
                obj.is_draw_fiber_architecture = True  # Enable fiber draw
                # Resolve skeleton attachments from names to current indices
                skeleton_names = list(v.zygote_skeleton_meshes.keys())
                obj.resolve_skeleton_attachments(skeleton_names)
        except Exception as e:
            print(f"[{name}] Load Tet error: {e}")

    # Inspect 2D button - opens visualization window for contours (and fiber samples if available)
    inspect_width = button_width * 2 + imgui.get_style().item_spacing[0]
    has_contour_data = (hasattr(obj, 'contours') and obj.contours is not None and len(obj.contours) > 0)
    if not has_contour_data:
        imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
    if imgui.button(f"Inspect 2D##{name}", width=inspect_width):
        _open_inspect_2d(v, name, stream_idx=0, contour_idx=0)
    if not has_contour_data:
        imgui.pop_style_var()

    # Focus camera on muscle button
    if imgui.button(f"Focus##{name}", width=button_width):
        if obj.vertices is not None and len(obj.vertices) > 0:
            # Compute bounding box center
            min_pt = np.min(obj.vertices, axis=0)
            max_pt = np.max(obj.vertices, axis=0)
            center = (min_pt + max_pt) / 2
            bbox_size = np.linalg.norm(max_pt - min_pt)
            # Set camera target (trans is scaled by 0.001 in render, so multiply by 1000)
            v.trans = -center * 1000.0
            # Adjust eye distance based on bounding box size
            distance = bbox_size * 2.0
            eye_dir = v.eye / (np.linalg.norm(v.eye) + 1e-10)
            v.eye = eye_dir * max(distance, MIN_EYE_DISTANCE)
        else:
            print(f"[{name}] No vertices to focus on")

    # Rotate toggle button (auto-orbit around focused muscle)
    imgui.same_line()
    is_rotating = v.auto_rotate
    if is_rotating:
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.4, 0.8, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.5, 0.9, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.1, 0.3, 0.7, 1.0)
    if imgui.button(f"Rotate##{name}", width=button_width):
        if v.auto_rotate:
            v.auto_rotate = False
        else:
            v.auto_rotate = True
    if is_rotating:
        imgui.pop_style_color(3)

    if not hasattr(v, 'global_animate_process'):
        v.global_animate_process = getattr(obj, 'animate_process', True)
    if not hasattr(obj, 'animate_process'):
        obj.animate_process = v.global_animate_process
    changed_anim, new_anim = imgui.checkbox("Animate", obj.animate_process)
    obj.animate_process = new_anim
    if changed_anim:
        v.global_animate_process = new_anim
        for _other in v.zygote_muscle_meshes.values():
            _other.animate_process = new_anim

    # Save/Load animation state + Play All
    imgui.same_line()
    avail = imgui.get_content_region_available_width()
    anim_btn_w = (avail - imgui.get_style().item_spacing[0] * 2) / 3
    anim_filepath = f"{v.zygote_muscle_dir}{name}.anim.pkl"
    if imgui.button(f"Save Anim##{name}", width=anim_btn_w):
        try:
            obj.save_animation_state(anim_filepath)
        except Exception as e:
            print(f"[{name}] Save Animation error: {e}")
    imgui.same_line()
    if imgui.button(f"Load Anim##{name}", width=anim_btn_w):
        try:
            obj.load_animation_state(anim_filepath)
        except Exception as e:
            print(f"[{name}] Load Animation error: {e}")
    imgui.same_line()
    if imgui.button(f"Play All##{name}", width=anim_btn_w):
        obj._play_all_active = True
        obj._play_all_step = 0
        # Reset visibility to pre-scalar start state
        obj.is_draw = True
        obj.is_draw_scalar_field = False
        obj.is_draw_contours = False
        obj.is_draw_bounding_box = False
        obj.is_draw_contour_mesh = False
        obj.is_draw_tet_mesh = False
        obj.is_draw_fiber_architecture = False
        obj.is_draw_resampled_vertices = False
        # Reset vertex colors to default muscle color
        if obj.vertex_colors is not None:
            n = len(obj.vertex_colors)
            obj.vertex_colors = np.tile(
                np.array([obj.color[0], obj.color[1], obj.color[2], obj.transparency], dtype=np.float32),
                (n, 1)
            )
        # Reset all replayed flags
        obj._scalar_replayed = False
        obj._contour_replayed = False
        obj._fill_gaps_replayed = False
        obj._transitions_replayed = False
        obj._smooth_replayed = False
        obj._cut_replayed = False
        obj._stream_smooth_replayed = False
        obj._level_select_replayed = False
        obj._build_fibers_replayed = False
        obj._resample_replayed = False
        obj._build_mesh_replayed = False
        obj._tetrahedralize_replayed = False

    if not hasattr(obj, 'linked_counterpart_name'):
        obj.linked_counterpart_name = ''
    if not hasattr(obj, 'linked_drive_counterpart'):
        obj.linked_drive_counterpart = True
    if not hasattr(obj, 'linked_use_shared_scalar'):
        obj.linked_use_shared_scalar = True
    if not hasattr(obj, 'linked_pair_eps'):
        obj.linked_pair_eps = 1e-5
    if imgui.tree_node(f"Linked Counterpart##{name}"):
        linked_names = ['None'] + [n for n in v.zygote_muscle_meshes.keys() if n != name]
        current_label = obj.linked_counterpart_name if obj.linked_counterpart_name in linked_names else 'None'
        current_idx = linked_names.index(current_label)
        changed_link, new_idx = imgui.combo(f"Counterpart##linked_{name}", current_idx, linked_names)
        if changed_link:
            _set_symmetric_counterpart_link(
                v, name, '' if new_idx == 0 else linked_names[new_idx])
        changed_drive, obj.linked_drive_counterpart = imgui.checkbox(
            f"Drive counterpart schedule##linked_{name}", bool(obj.linked_drive_counterpart))
        changed_scalar, obj.linked_use_shared_scalar = imgui.checkbox(
            f"Shared scalar solve##linked_{name}", bool(obj.linked_use_shared_scalar))
        changed_eps, obj.linked_pair_eps = imgui.input_float(
            f"Pair epsilon##linked_{name}", float(obj.linked_pair_eps), 1e-6, 1e-5, "%.7f")
        obj.linked_pair_eps = max(0.0, float(obj.linked_pair_eps))
        if changed_link or changed_drive or changed_scalar or changed_eps:
            _sync_counterpart_link_settings(v, name)
        if obj.linked_counterpart_name:
            imgui.text(f"Master: {name}")
            imgui.text(f"Follower: {obj.linked_counterpart_name}")
            if getattr(obj, 'linked_counterpart_pairs', None) is not None:
                imgui.text(f"Pairs: {len(obj.linked_counterpart_pairs)}")
        imgui.tree_pop()

    if not hasattr(obj, 'origin_tendon_extension_name'):
        obj.origin_tendon_extension_name = ''
    if not hasattr(obj, 'insertion_tendon_extension_name'):
        obj.insertion_tendon_extension_name = ''
    if not hasattr(obj, 'origin_tendon_reverse'):
        obj.origin_tendon_reverse = False
    if not hasattr(obj, 'insertion_tendon_reverse'):
        obj.insertion_tendon_reverse = False
    if _is_component_belly_name(name):
        _auto_fill_tendon_extension_names(v, name, obj)
    elif ('tendon' not in name.lower()
          and (obj.origin_tendon_extension_name or obj.insertion_tendon_extension_name)):
        obj.origin_tendon_extension_name = ''
        obj.insertion_tendon_extension_name = ''
        _clear_tendon_extension(obj)
    if imgui.tree_node(f"Tendon Fiber Extension##{name}"):
        tendon_names = ['None'] + [
            n for n in v.zygote_muscle_meshes.keys()
            if n != name and 'tendon' in n.lower()
        ]
        origin_label = obj.origin_tendon_extension_name if obj.origin_tendon_extension_name in tendon_names else 'None'
        insertion_label = obj.insertion_tendon_extension_name if obj.insertion_tendon_extension_name in tendon_names else 'None'
        changed_o, idx_o = imgui.combo(
            f"Origin Tendon##tendon_ext_o_{name}",
            tendon_names.index(origin_label),
            tendon_names)
        if changed_o:
            obj.origin_tendon_extension_name = '' if idx_o == 0 else tendon_names[idx_o]
        changed_i, idx_i = imgui.combo(
            f"Insertion Tendon##tendon_ext_i_{name}",
            tendon_names.index(insertion_label),
            tendon_names)
        if changed_i:
            obj.insertion_tendon_extension_name = '' if idx_i == 0 else tendon_names[idx_i]
        _, obj.origin_tendon_reverse = imgui.checkbox(
            f"Reverse Origin Tendon##tendon_ext_o_rev_{name}",
            bool(obj.origin_tendon_reverse))
        _, obj.insertion_tendon_reverse = imgui.checkbox(
            f"Reverse Insertion Tendon##tendon_ext_i_rev_{name}",
            bool(obj.insertion_tendon_reverse))
        if imgui.button(f"Extend Fibers Through Tendons##tendon_ext_apply_{name}", width=wide_button_width):
            _extend_belly_and_linked_counterpart_fibers(v, name, obj)
        if getattr(obj, 'tendon_extended_fibers', False):
            imgui.text("Extended fibers: active")
        imgui.tree_pop()

    # Min-spacing threshold for length-density auto-selection.
    # Search increases N until any consecutive pair drops below
    # this distance along the muscle axis.  Value is global on
    # `v` so adjusting from any muscle's tree propagates to
    # every muscle on the next render.
    if not hasattr(v, 'global_level_select_min_spacing'):
        v.global_level_select_min_spacing = 0.04
    changed_sp, v.global_level_select_min_spacing = imgui.slider_float(
        "Min Spacing (m)", v.global_level_select_min_spacing,
        0.005, 0.5, "%.3f")
    obj.level_select_min_spacing = v.global_level_select_min_spacing
    if changed_sp:
        for _other in v.zygote_muscle_meshes.values():
            _other.level_select_min_spacing = v.global_level_select_min_spacing

    imgui.text(obj.link_mode)
    changed1, obj.specific_contour_value = imgui.slider_float(f"Ori##{name}", obj.specific_contour_value, 1.0, obj.contour_value_min, flags=imgui.SLIDER_FLAGS_NO_ROUND_TO_FORMAT)
    changed2, obj.specific_contour_value = imgui.slider_float(f"Mid##{name}", obj.specific_contour_value, obj.contour_value_min, obj.contour_value_max, flags=imgui.SLIDER_FLAGS_NO_ROUND_TO_FORMAT)
    changed3, obj.specific_contour_value = imgui.slider_float(f"Ins##{name}", obj.specific_contour_value, obj.contour_value_max, 10.0, flags=imgui.SLIDER_FLAGS_NO_ROUND_TO_FORMAT)
    if imgui.tree_node(f"Epic##{name}"):
        if not hasattr(obj, 'epic_fiber_count'):
            obj.epic_fiber_count = 100
        if not hasattr(obj, 'epic_gradient_direction_limit'):
            obj.epic_gradient_direction_limit = 8000
        imgui.push_item_width(160)
        _, obj.epic_fiber_count = imgui.slider_int(
            f"Fiber Count##epic_fiber_count_{name}",
            int(obj.epic_fiber_count), 1, 1000)
        _, obj.epic_gradient_direction_limit = imgui.slider_int(
            f"Direction Draw Limit##epic_dir_limit_{name}",
            int(obj.epic_gradient_direction_limit), 100, 50000)
        imgui.pop_item_width()

        if imgui.button(f"1. Tetrahedralize Original Mesh##epic_tet_{name}", width=wide_button_width):
            try:
                _t0 = time.time()
                ok = obj.epic_tetrahedralize_original_mesh()
                status = "done" if ok else "failed"
                print(f"[{name}] Epic tetrahedralize {status} in {time.time()-_t0:.3f}s")
            except Exception as e:
                print(f"[{name}] Epic tetrahedralize error: {e}")
                traceback.print_exc()

        if imgui.button(f"2. Find Volume Laplace Field##epic_field_{name}", width=wide_button_width):
            try:
                _t0 = time.time()
                ok = obj.epic_solve_volume_laplace_field()
                if ok:
                    v.zygote_tet_transparency = 0.5
                status = "done" if ok else "failed"
                print(f"[{name}] Epic volume field {status} in {time.time()-_t0:.3f}s")
            except Exception as e:
                print(f"[{name}] Epic volume field error: {e}")
                traceback.print_exc()

        if imgui.button(f"3. Show Tet Gradient Directions##epic_dirs_{name}", width=wide_button_width):
            try:
                _t0 = time.time()
                ok = obj.epic_show_laplace_gradient_directions(
                    max_segments=obj.epic_gradient_direction_limit)
                status = "done" if ok else "failed"
                print(f"[{name}] Epic tet gradient directions {status} in {time.time()-_t0:.3f}s")
            except Exception as e:
                print(f"[{name}] Epic tet gradient directions error: {e}")
                traceback.print_exc()

        if imgui.button(f"4. Sample Shape-Coordinate Fibers##epic_shape_fibers_{name}", width=wide_button_width):
            try:
                _t0 = time.time()
                ok = obj.epic_sample_shape_coordinate_fibers(count=obj.epic_fiber_count)
                status = "done" if ok else "failed"
                print(f"[{name}] Epic shape-coordinate fibers {status} in {time.time()-_t0:.3f}s")
            except Exception as e:
                print(f"[{name}] Epic shape-coordinate fibers error: {e}")
                traceback.print_exc()

        tet_v = 0 if getattr(obj, 'epic_tet_vertices', None) is None else len(obj.epic_tet_vertices)
        tet_n = 0 if getattr(obj, 'epic_tetrahedra', None) is None else len(obj.epic_tetrahedra)
        imgui.text(f"Tet: {tet_v} verts / {tet_n} tets")
        if getattr(obj, 'epic_laplace_field', None) is not None:
            imgui.text("Field: ready")
        if getattr(obj, 'epic_gradient_direction_segments', None) is not None:
            imgui.text(f"Directions: {len(obj.epic_gradient_direction_segments)}")
        if getattr(obj, 'epic_fibers', None) is not None:
            imgui.text(f"Fibers: {obj.epic_fibers.shape[1]}")
        if getattr(obj, 'epic_error', ""):
            imgui.text_colored(str(obj.epic_error)[:90], 1.0, 0.35, 0.2, 1.0)
        imgui.tree_pop()

    if imgui.tree_node(f"MinMax##{name}"):
        _, obj.contour_value_min = imgui.input_float(f"Min##{name}", obj.contour_value_min)
        _, obj.contour_value_max = imgui.input_float(f"Max##{name}", obj.contour_value_max)
        imgui.tree_pop()
    if changed1 or changed2 or changed3:
        obj.find_contour_with_value(obj.specific_contour_value)
    # if imgui.button(f"Find Value Contour##{name}"):
    #     obj.find_contour_with_value()

    # if imgui.button("Find Interesecting Bones"):
    #     for other_name, other_obj in v.zygote_muscle_meshes.items():
    #         other_obj.is_draw = False
    #     obj.is_draw = True

    #     intersecting_meshes = obj.find_intersections(v.zygote_skeleton_meshes)
    #     # print(bb_intersect)
    #     for skel_name, skel_obj in v.zygote_skeleton_meshes.items():
    #         if skel_name in intersecting_meshes:
    #             skel_obj.color = np.array([0.0, 0.0, 1.0])
    #         else:
    #             skel_obj.color = np.array([0.9, 0.9, 0.9])

    #     v.zygote_muscle_meshes_intersection_bones[name] = intersecting_meshes

    changed, obj.transparency = imgui.slider_float(f"Transparency##{name}", obj.transparency, 0.0, 1.0)
    if changed and obj.vertex_colors is not None:
        obj.vertex_colors[:, 3] = obj.transparency
    if changed:
        _sync_counterpart_display_state(v, name)

    if imgui.tree_node("Edge Classes"):
        for i in range(len(obj.edge_classes)):
            # Fixed width: "insertion" is 9 chars, pad "origin" to match
            label = f"{obj.edge_classes[i]:9s}"
            imgui.text(label)
            imgui.same_line()
            if imgui.button(f"Flip class##{name}_{i}"):
                obj.edge_classes[i] = 'insertion' if obj.edge_classes[i] == 'origin' else 'origin'
        imgui.tree_pop()
    if obj.draw_contour_stream is not None:
        if imgui.tree_node("Contour Stream"):
            if imgui.button("All Stream Off"):
                for i in range(len(obj.draw_contour_stream)):
                    obj.draw_contour_stream[i] = False
            imgui.same_line()
            if imgui.button(f"Auto Detect##{name}"):
                obj.auto_detect_attachments(v.zygote_skeleton_meshes)
            # Ensure attach_skeletons arrays are properly sized
            num_streams = len(obj.draw_contour_stream)
            while len(obj.attach_skeletons) < num_streams:
                obj.attach_skeletons.append([0, 0])
            while len(obj.attach_skeletons_sub) < num_streams:
                obj.attach_skeletons_sub.append([0, 0])
            for i in range(num_streams):
                _, obj.draw_contour_stream[i] = imgui.checkbox(f"Stream {i}", obj.draw_contour_stream[i])
                imgui.push_item_width(100)
                changed, obj.attach_skeletons[i][0] = imgui.input_int(f"Origin##{name}_stream{i}_origin", obj.attach_skeletons[i][0])
                if changed:
                    if obj.attach_skeletons[i][0] < 0:
                        obj.attach_skeletons[i][0] = 0
                    elif obj.attach_skeletons[i][0] > len(v.zygote_skeleton_meshes) - 1:
                        obj.attach_skeletons[i][0] = len(v.zygote_skeleton_meshes) - 1
                changed, obj.attach_skeletons_sub[i][0] = imgui.input_int(f"Subpart##{name}_stream{i}_origin_sub", obj.attach_skeletons_sub[i][0])
                if changed:
                    if obj.attach_skeletons_sub[i][0] < 0:
                        obj.attach_skeletons_sub[i][0] = 0
                    elif obj.attach_skeletons_sub[i][0] > 1:
                        obj.attach_skeletons_sub[i][0] = 1

                imgui.text(list(v.zygote_skeleton_meshes.keys())[obj.attach_skeletons[i][0]] + f"{obj.attach_skeletons_sub[i][0]}")
                changed, obj.attach_skeletons[i][1] = imgui.input_int(f"Insertion##{name}_stream{i}_insertion", obj.attach_skeletons[i][1])
                if changed:
                    if obj.attach_skeletons[i][1] < 0:
                        obj.attach_skeletons[i][1] = 0
                    elif obj.attach_skeletons[i][1] > len(v.zygote_skeleton_meshes) - 1:
                        obj.attach_skeletons[i][1] = len(v.zygote_skeleton_meshes) - 1
                changed, obj.attach_skeletons_sub[i][1] = imgui.input_int(f"Subpart##{name}_stream{i}_insertion_sub", obj.attach_skeletons_sub[i][1])
                if changed:
                    if obj.attach_skeletons_sub[i][1] < 0:
                        obj.attach_skeletons_sub[i][1] = 0
                    elif obj.attach_skeletons_sub[i][1] > 1:
                        obj.attach_skeletons_sub[i][1] = 1
                imgui.text(list(v.zygote_skeleton_meshes.keys())[obj.attach_skeletons[i][1]] + f"{obj.attach_skeletons_sub[i][1]}")
                imgui.pop_item_width()

                # if imgui.button(f"Print Contour points##{i}"):
                #     print(f"Print {i}th contour stream")
                #     for j, contour in enumerate(obj.contours[i]):
                #         print(f"Contour {j}")
                #         for v in contour:
                #             print(v)
                #         print()
            imgui.tree_pop()

    display_changed = False
    changed_draw, obj.is_draw = imgui.checkbox("Draw", obj.is_draw)
    display_changed = display_changed or changed_draw
    changed_open, obj.is_draw_open_edges = imgui.checkbox("Draw Open Edges", obj.is_draw_open_edges)
    display_changed = display_changed or changed_open
    changed_scalar_draw, obj.is_draw_scalar_field = imgui.checkbox("Draw Scalar Field", obj.is_draw_scalar_field)
    display_changed = display_changed or changed_scalar_draw
    if changed_scalar_draw:
        if obj.is_draw_scalar_field:
            if getattr(obj, '_scalar_anim_target_colors', None) is not None:
                obj.vertex_colors = obj._scalar_anim_target_colors.copy()
    changed_contours, obj.is_draw_contours = imgui.checkbox("Draw Contours", obj.is_draw_contours)
    display_changed = display_changed or changed_contours
    imgui.same_line()
    changed_vertices, obj.is_draw_contour_vertices = imgui.checkbox("Vertices", obj.is_draw_contour_vertices)
    display_changed = display_changed or changed_vertices
    imgui.same_line()
    changed_pair, obj.is_draw_farthest_pair = imgui.checkbox("Farthest Pair", obj.is_draw_farthest_pair)
    display_changed = display_changed or changed_pair
    changed_edges, obj.is_draw_edges = imgui.checkbox("Draw Edges", obj.is_draw_edges)
    display_changed = display_changed or changed_edges
    changed_centroid, obj.is_draw_centroid = imgui.checkbox("Draw Centroid", obj.is_draw_centroid)
    display_changed = display_changed or changed_centroid
    changed_bbox, obj.is_draw_bounding_box = imgui.checkbox("Draw Bounding Box", obj.is_draw_bounding_box)
    display_changed = display_changed or changed_bbox
    if obj.is_draw_bounding_box:
        imgui.same_line()
        bb_mode = getattr(obj, 'bounding_box_draw_mode', 0)
        bb_labels = ["Planes", "Boxes"]
        imgui.push_item_width(80)
        changed, new_mode = imgui.combo(f"##bb_mode_{name}", bb_mode, bb_labels)
        imgui.pop_item_width()
        if changed:
            obj.bounding_box_draw_mode = new_mode
            display_changed = True
    changed_discarded, obj.is_draw_discarded = imgui.checkbox("Draw Discarded", obj.is_draw_discarded)
    display_changed = display_changed or changed_discarded
    changed_fiber, obj.is_draw_fiber_architecture = imgui.checkbox("Draw Fiber Architecture", obj.is_draw_fiber_architecture)
    display_changed = display_changed or changed_fiber
    if getattr(obj, 'contours_resampled', None) is not None:
        changed_resamp, obj.is_draw_resampled_vertices = imgui.checkbox("Draw Resampled Vertices", obj.is_draw_resampled_vertices)
        display_changed = display_changed or changed_resamp
    changed_mesh, obj.is_draw_contour_mesh = imgui.checkbox("Draw Contour Mesh", obj.is_draw_contour_mesh)
    display_changed = display_changed or changed_mesh
    changed_tet, obj.is_draw_tet_mesh = imgui.checkbox("Draw Tet Mesh", obj.is_draw_tet_mesh)
    display_changed = display_changed or changed_tet
    imgui.same_line()
    changed_tet_edges, obj.is_draw_tet_edges = imgui.checkbox("Tet Edges", obj.is_draw_tet_edges)
    display_changed = display_changed or changed_tet_edges
    if changed_tet_edges:
        obj._tet_edge_verts = None
        obj._tet_edge_vidx = None
        obj._tet_edge_source = None
    # Internal tet faces/stride controls are intentionally hidden.
    obj.is_draw_tet_internal_faces = False
    changed_constraints, obj.is_draw_constraints = imgui.checkbox("Constraints", obj.is_draw_constraints)
    display_changed = display_changed or changed_constraints
    if display_changed:
        _sync_counterpart_display_state(v, name)
    tet_labels = getattr(obj, 'tet_region_labels', None)
    if tet_labels is not None:
        _draw_tet_quality_stats(obj)
        tet_region_counts = {}
        for label in tet_labels:
            tet_region_counts[label] = tet_region_counts.get(label, 0) + 1
        region_text = ", ".join(
            f"{label}:{count}" for label, count in sorted(tet_region_counts.items())
        )
        imgui.text(f"Tet regions: {region_text}")
        tet_mixed = getattr(obj, 'tet_region_mixed', None)
        if tet_mixed is not None:
            imgui.text(f"Mixed/interface tets: {int(np.sum(tet_mixed))}")

    if imgui.button("Export Muscle Waypoints", width=wide_button_width):
        pass
        from core.dartHelper import exportMuscleWaypoints
        exportMuscleWaypoints(v.zygote_muscle_meshes, list(v.zygote_skeleton_meshes.keys()))
    if imgui.button("Import zygote_muscle", width=wide_button_width):
        muscle_file = "data/zygote_muscle.xml"
        if not os.path.exists(muscle_file):
            print(f"Error: Muscle file not found: {muscle_file}")
            print("  Run 'Export Muscle Waypoints' first to create it.")
        else:
            try:
                v.env.muscle_info = v.env.saveZygoteMuscleInfo(muscle_file)
                if not v.env.muscle_info:
                    print("No muscles loaded from file (empty or invalid)")
                else:
                    v.env.loading_zygote_muscle_info(v.env.muscle_info)
                    v.env.muscle_activation_levels = np.zeros(v.env.muscles.getNumMuscles())

                    v.draw_obj = True
                    # Disable skeleton drawing when importing muscle waypoints
                    v.is_draw_zygote_skeleton = False
                    for sname, sobj in v.zygote_skeleton_meshes.items():
                        sobj.is_draw = False
                    print(f"Imported {v.env.muscles.getNumMuscles()} muscles from {muscle_file}")
            except Exception as e:
                print(f"Error importing muscle waypoints: {e}")

    # End column layout
    imgui.columns(1)


def draw_zygote_muscle_ui(v):
    """Muscle section inside the Zygote tree node."""
    if imgui.tree_node("Muscle", imgui.TREE_NODE_DEFAULT_OPEN):
        changed, v.is_draw_zygote_muscle = imgui.checkbox("Draw", v.is_draw_zygote_muscle)
        if changed:
            for name, obj in v.zygote_muscle_meshes.items():
                obj.is_draw = v.is_draw_zygote_muscle
            _sync_zygote_group_draw_state(v, v.is_draw_zygote_muscle)
        changed, v.is_draw_zygote_muscle_open_edges = imgui.checkbox("Draw Open Edges", v.is_draw_zygote_muscle_open_edges)
        if changed:
            for name, obj in v.zygote_muscle_meshes.items():
                obj.is_draw_open_edges = v.is_draw_zygote_muscle_open_edges

        changed, v.zygote_muscle_color = imgui.color_edit3("Color", *v.zygote_muscle_color)
        if changed:
            for name, obj in v.zygote_muscle_meshes.items():
                if not getattr(obj, 'is_zygote_tendon', False):
                    obj.color = v.zygote_muscle_color

        changed, v.zygote_muscle_transparency = imgui.slider_float("Transparency", v.zygote_muscle_transparency, 0.0, 1.0)
        if changed:
            for name, obj in v.zygote_muscle_meshes.items():
                obj.transparency = v.zygote_muscle_transparency
                if obj.vertex_colors is not None:
                    obj.vertex_colors[:, 3] = obj.transparency

        changed, v.is_draw_zygote_muscle_tet = imgui.checkbox("##draw_tet", v.is_draw_zygote_muscle_tet)
        if changed:
            for name, obj in v.zygote_muscle_meshes.items():
                obj.is_draw_tet_mesh = v.is_draw_zygote_muscle_tet
        imgui.same_line()
        _, v.zygote_tet_transparency = imgui.slider_float("Tet Mesh Transparency", v.zygote_tet_transparency, 0.0, 1.0)
        # Internal tet faces are intentionally hidden; the surface/tet toggle
        # above remains available without exposing the internal-face controls.
        v.is_draw_zygote_tet_internal_faces = False
        for obj in v.zygote_muscle_meshes.values():
            obj.is_draw_tet_internal_faces = False

        changed, v.is_draw_zygote_muscle_fibers = imgui.checkbox("##draw_fibers", v.is_draw_zygote_muscle_fibers)
        if changed:
            for name, obj in v.zygote_muscle_meshes.items():
                obj.is_draw_fiber_architecture = v.is_draw_zygote_muscle_fibers
        imgui.same_line()
        _, v.zygote_fiber_transparency = imgui.slider_float("Fiber Transparency", v.zygote_fiber_transparency, 0.0, 1.0)

        # Reverse-LBS waypoint override: shows fiber positions computed from
        # solved 2/3-bone LBS local positions + current skeleton pose, instead
        # of cache- or NN-derived waypoints. Tet mesh rendering is unaffected.
        if not hasattr(v, 'reverse_lbs_enabled'):
            v.reverse_lbs_enabled = False
        changed_rlbs, v.reverse_lbs_enabled = imgui.checkbox(
            "Show Reverse-LBS Waypoints##fiber_rlbs", v.reverse_lbs_enabled)
        if changed_rlbs and v.reverse_lbs_enabled:
            _reverse_lbs_load(v)
        if changed_rlbs:
            if v.reverse_lbs_enabled:
                _reverse_lbs_apply(v)
            else:
                # Restore cached/NN waypoints for current frame.
                cur = getattr(v, 'motion_current_frame', 0)
                if not _motion_apply_cached_deformation(v, cur):
                    for mobj in v.zygote_muscle_meshes.values():
                        if hasattr(mobj, '_update_waypoints_from_tet'):
                            mobj._update_waypoints_from_tet(v.env.skel, verbose=False)
                        mobj._fiber_draw_dirty = True

        # Muscle Add/Remove UI
        if imgui.tree_node("Add/Remove Muscles"):
            imgui.text("Available:")
            # Calculate total available count
            total_available = sum(
                len(muscles) for muscles in v.available_muscle_by_category.values())

            if total_available > 0:
                # Draw category-based listbox using child region
                # Group categories by body part (first part before '/')
                from collections import OrderedDict
                body_parts = OrderedDict()
                for category, muscles in v.available_muscle_by_category.items():
                    if len(muscles) == 0:
                        continue
                    if '/' in category:
                        body_part, side = category.rsplit('/', 1)
                    else:
                        body_part, side = category, ''
                    if body_part not in body_parts:
                        body_parts[body_part] = []
                    body_parts[body_part].append((side, category, muscles))

                imgui.begin_child("##available_muscles_child", width=0, height=150, border=True)

                for body_part, sides in body_parts.items():
                    # Count total muscles in this body part
                    bp_total = sum(len(muscles) for _, _, muscles in sides)
                    bp_key = f"_bp_{body_part}"
                    is_bp_expanded = v.available_category_expanded.get(bp_key, False)
                    arrow = "v" if is_bp_expanded else ">"
                    bp_label = f"{arrow} {body_part} ({bp_total})"

                    if imgui.selectable(bp_label, False)[0]:
                        v.available_category_expanded[bp_key] = not is_bp_expanded

                    if is_bp_expanded:
                        for side, category, muscles in sides:
                            imgui.indent(15)
                            is_expanded = v.available_category_expanded.get(category, False)
                            arrow2 = "v" if is_expanded else ">"
                            side_label = f"{arrow2} {side} ({len(muscles)})"

                            if imgui.selectable(f"{side_label}##{category}", False)[0]:
                                v.available_category_expanded[category] = not is_expanded

                            if is_expanded:
                                component_groups = getattr(v, 'available_muscle_groups_by_category', {}).get(category, {})
                                grouped_component_names = set()
                                for group_name, comps in component_groups.items():
                                    grouped_component_names.update(name for name, _path in comps)
                                    imgui.indent(15)
                                    is_selected = (
                                        getattr(v, 'available_selected_group', None) == group_name
                                        and v.available_selected_category == category
                                    )
                                    clicked, _ = imgui.selectable(
                                        f"  [G] {group_name} ({len(comps)})", is_selected)
                                    if clicked:
                                        v.available_selected_category = category
                                        v.available_selected_group = group_name
                                        v.available_selected_muscle = None
                                    if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                                        add_muscle_group(v, category, group_name)
                                    imgui.unindent(15)
                                for muscle_name, muscle_path in muscles:
                                    if muscle_name in grouped_component_names:
                                        continue
                                    imgui.indent(15)
                                    is_selected = (v.available_selected_muscle == muscle_name)
                                    clicked, _ = imgui.selectable(
                                        f"  {muscle_name}", is_selected)
                                    if clicked:
                                        v.available_selected_category = category
                                        v.available_selected_muscle = muscle_name
                                        v.available_selected_group = None
                                    if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                                        add_muscle_mesh(v, muscle_name, muscle_path)
                                    imgui.unindent(15)
                            imgui.unindent(15)

                imgui.end_child()
            else:
                imgui.text("(none)")

            # Arrow buttons for add/remove
            if imgui.button("Add", width=button_width):
                selected_group = getattr(v, 'available_selected_group', None)
                if selected_group and v.available_selected_category:
                    add_muscle_group(v, v.available_selected_category, selected_group)
                elif v.available_selected_muscle and v.available_selected_category:
                    selected_path = None
                    for muscle_name, muscle_path in v.available_muscle_by_category.get(v.available_selected_category, []):
                        if muscle_name == v.available_selected_muscle:
                            selected_path = muscle_path
                            break
                    if selected_path is not None:
                        add_muscle_mesh(v, v.available_selected_muscle, selected_path)
            imgui.same_line()
            if imgui.button("Remove", width=button_width):
                loaded_groups = _zygote_loaded_ui_group_names(v)
                selected_group = getattr(v, 'loaded_muscle_selected_group', None)
                if selected_group in loaded_groups:
                    remove_muscle_group(v, selected_group)
                    v.loaded_muscle_selected_group = None
                else:
                    loaded_names = list(v.zygote_muscle_meshes.keys())
                    if loaded_names:
                        idx = min(v.loaded_muscle_selected, len(loaded_names) - 1)
                        remove_muscle_mesh(v, loaded_names[idx])

            imgui.text("Loaded:")
            loaded_groups = _zygote_loaded_ui_group_names(v)
            grouped_loaded_names = set()
            for group_name in loaded_groups:
                grouped_loaded_names.update(_zygote_group_loaded_names(v, group_name))
            loaded_names = [
                name for name in v.zygote_muscle_meshes.keys()
                if name not in grouped_loaded_names
            ]
            # Always show child region so layout stays stable
            imgui.begin_child("##loaded_muscles_child", width=0, height=150, border=True)
            for group_name in loaded_groups:
                is_selected = (getattr(v, 'loaded_muscle_selected_group', None) == group_name)
                clicked, _ = imgui.selectable(
                    f"[G] {group_name} ({len(_zygote_group_loaded_names(v, group_name))})",
                    is_selected)
                if clicked:
                    v.loaded_muscle_selected_group = group_name
                if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                    remove_muscle_group(v, group_name)
            for i, name in enumerate(loaded_names):
                is_selected = (v.loaded_muscle_selected == i)
                clicked, _ = imgui.selectable(name, is_selected)
                if clicked:
                    v.loaded_muscle_selected = i
                    v.loaded_muscle_selected_group = None
                # Double-click to remove
                if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                    remove_muscle_mesh(v, name)
            imgui.end_child()

            # Bulk add/remove buttons
            imgui.separator()

            if imgui.button("Remove All", width=button_width):
                remove_all_muscles(v)

            # Add L/R muscles by group
            groups = get_available_muscle_groups(v)
            if len(groups) > 0:
                imgui.text("Add by group:")
                for group in groups:
                    short_name = group[:8] if len(group) > 8 else group
                    if imgui.button(f"L {short_name}##L{group}", width=73):
                        add_muscles_by_group(v, group, "L_")
                    imgui.same_line()
                    if imgui.button(f"R {short_name}##R{group}", width=73):
                        add_muscles_by_group(v, group, "R_")

            imgui.tree_pop()

        if imgui.tree_node("Activation levels"):
            if v.env.zygote_activation_levels is not None:
                # Use shorter slider width to leave room for name
                slider_width = 120
                for i, (name, obj) in enumerate(v.env.muscle_info.items()):
                    # Bounds check for zygote_activation_levels
                    if i >= len(v.env.zygote_activation_levels):
                        continue
                    # Show name first (truncate if too long)
                    display_name = name[:25] + "..." if len(name) > 25 else name
                    imgui.text(f"{display_name}")
                    imgui.same_line(position=180)
                    imgui.push_item_width(slider_width)
                    changed, v.env.zygote_activation_levels[i] = imgui.slider_float(f"##zygote_act{i}", v.env.zygote_activation_levels[i], 0.0, 1.0)
                    imgui.pop_item_width()
                    if changed:
                        # Bounds check for activation indices
                        if i + 1 < len(v.env.zygote_activation_indices):
                            start_fiber = v.env.zygote_activation_indices[i]
                            end_fiber = v.env.zygote_activation_indices[i + 1]
                            if end_fiber <= len(v.env.muscle_activation_levels):
                                v.env.muscle_activation_levels[start_fiber:end_fiber] = v.env.zygote_activation_levels[i]
            imgui.tree_pop()

        # Motion Browser
        if v.env.skel is not None and imgui.tree_node("Motion Browser", imgui.TREE_NODE_DEFAULT_OPEN):
            _draw_motion_browser_ui(v)
            imgui.tree_pop()

        # Skeleton joint angle sliders
        if v.env.skel is not None and imgui.tree_node("Skeleton Joint Angles"):
            # Initialize DOF array if not present
            if not hasattr(v, '_skel_dofs'):
                v._skel_dofs = v.env.skel.getPositions().copy()
                v._skel_dof_names = []
                # Build DOF names from joints
                for jn_idx in range(v.env.skel.getNumJoints()):
                    joint = v.env.skel.getJoint(jn_idx)
                    jn_name = joint.getName()
                    num_dofs = joint.getNumDofs()
                    if num_dofs == 1:
                        v._skel_dof_names.append(jn_name)
                    elif num_dofs == 3:
                        v._skel_dof_names.extend([f"{jn_name}_x", f"{jn_name}_y", f"{jn_name}_z"])
                    elif num_dofs == 6:
                        v._skel_dof_names.extend([f"{jn_name}_tx", f"{jn_name}_ty", f"{jn_name}_tz",
                                                    f"{jn_name}_rx", f"{jn_name}_ry", f"{jn_name}_rz"])
                    else:
                        for d in range(num_dofs):
                            v._skel_dof_names.append(f"{jn_name}_{d}")

            # Sync DOF array size with skeleton
            num_dofs = v.env.skel.getNumDofs()
            if len(v._skel_dofs) != num_dofs:
                v._skel_dofs = v.env.skel.getPositions().copy()

            # Reset all button
            if imgui.button("Reset All##skel_dofs", width=100):
                v._skel_dofs = np.zeros(num_dofs)
                v.env.skel.setPositions(v._skel_dofs)
                # Update soft bodies and waypoints
                for mname, mobj in v.zygote_muscle_meshes.items():
                    if mobj.soft_body is not None:
                        mobj._update_tet_positions_from_skeleton(v.env.skel)
                        mobj._update_fixed_targets_from_skeleton(v.zygote_skeleton_meshes, v.env.skel)
            imgui.same_line()
            if imgui.button("Sync from Skel##skel_dofs", width=120):
                v._skel_dofs = v.env.skel.getPositions().copy()

            # Sliders for each DOF
            label_width = 140
            slider_width = 120
            reset_btn_width = 22
            any_changed = False
            for i in range(num_dofs):
                # Get DOF name (truncate if too long)
                dof_name = v._skel_dof_names[i] if i < len(v._skel_dof_names) else f"DOF {i}"
                display_name = dof_name[:15] if len(dof_name) > 15 else dof_name
                imgui.text(f"{i:2d} {display_name:<15}")
                imgui.same_line(position=label_width)
                imgui.push_item_width(slider_width)
                changed, v._skel_dofs[i] = imgui.slider_float(f"##skel_dof{i}", v._skel_dofs[i], -3.14, 3.14)
                imgui.pop_item_width()
                if changed:
                    any_changed = True
                imgui.same_line()
                if imgui.button(f"0##reset_dof{i}", width=reset_btn_width):
                    v._skel_dofs[i] = 0.0
                    any_changed = True

            # Apply changes to skeleton
            if any_changed:
                v.env.skel.setPositions(v._skel_dofs)
                # Update soft bodies and waypoints
                for mname, mobj in v.zygote_muscle_meshes.items():
                    if mobj.soft_body is not None:
                        mobj._update_tet_positions_from_skeleton(v.env.skel)
                        mobj._update_fixed_targets_from_skeleton(v.zygote_skeleton_meshes, v.env.skel)

            imgui.tree_pop()

        if imgui.button("Export Muscle Waypoints", width=wide_button_width):
            from core.dartHelper import exportMuscleWaypoints
            exportMuscleWaypoints(v.zygote_muscle_meshes, list(v.zygote_skeleton_meshes.keys()))
        if imgui.button("Import zygote_muscle", width=wide_button_width):
            muscle_file = "data/zygote_muscle.xml"
            if not os.path.exists(muscle_file):
                print(f"Error: Muscle file not found: {muscle_file}")
                print("  Run 'Export Muscle Waypoints' first to create it.")
            else:
                # Reload skel XML first so DART picks up bones added since the
                # last skel import. Without this, newly added bones (e.g. L5)
                # won't appear in draw_obj and muscles can't reference them.
                try:
                    _reimport_zygote_skel(v)
                except Exception as _e:
                    print(f"[Import zygote_muscle] skel reload failed: {_e}")
                try:
                    v.env.muscle_info = v.env.saveZygoteMuscleInfo(muscle_file)
                    if not v.env.muscle_info:
                        print("No muscles loaded from file (empty or invalid)")
                    else:
                        v.env.loading_zygote_muscle_info(v.env.muscle_info)
                        v.env.muscle_activation_levels = np.zeros(v.env.muscles.getNumMuscles())

                        v.draw_obj = True
                        # Disable skeleton drawing when importing muscle waypoints
                        v.is_draw_zygote_skeleton = False
                        for name, obj in v.zygote_skeleton_meshes.items():
                            obj.is_draw = False
                        print(f"Imported {v.env.muscles.getNumMuscles()} muscles from {muscle_file}")
                except Exception as e:
                    print(f"Error importing muscle waypoints: {e}")

        if imgui.button("Update DART LBS Muscles", width=wide_button_width):
            if hasattr(v.env, 'muscles') and v.env.muscles is not None \
                    and v.env.muscles.getNumMuscles() > 0:
                v.env.muscles.update()
                v.env.muscle_pos = v.env.muscles.getMusclePositions()
            else:
                print("Click 'Import zygote_muscle' first.")

        if imgui.button("Import Reverse-LBS muscles", width=wide_button_width):
            _import_reverse_lbs_into_dart(v)

        # Load all tet meshes and init soft bodies
        if imgui.button("Load All Tets", width=wide_button_width):
            import time as _t
            _t0 = _t.time()
            load_count = 0
            init_count = 0
            already_init_count = 0
            for mname, mobj in v.zygote_muscle_meshes.items():
                try:
                    # If tet already loaded, just check if soft body needs init
                    if mobj.tet_vertices is not None:
                        if mobj.soft_body is None:
                            # Tet loaded but soft body not initialized - init it
                            skeleton_names = list(v.zygote_skeleton_meshes.keys())
                            mobj.resolve_skeleton_attachments(skeleton_names)
                            mobj.init_soft_body(v.zygote_skeleton_meshes, v.env.skel, v.env.mesh_info)
                            if mobj.soft_body is not None:
                                init_count += 1
                        else:
                            already_init_count += 1
                        continue
                    # Load new tet
                    mobj.soft_body = None  # Reset soft body when loading new tet
                    if mobj.load_tetrahedron_mesh(mname):
                        if mobj.tet_vertices is not None:
                            mobj.is_draw = False  # Disable mesh draw
                            mobj.is_draw_contours = False
                            mobj.is_draw_tet_mesh = True
                            mobj.is_draw_fiber_architecture = True  # Enable fiber draw
                            load_count += 1
                            # Resolve skeleton attachments from names to current indices
                            skeleton_names = list(v.zygote_skeleton_meshes.keys())
                            mobj.resolve_skeleton_attachments(skeleton_names)
                            # Also init soft body
                            mobj.init_soft_body(v.zygote_skeleton_meshes, v.env.skel, v.env.mesh_info)
                            if mobj.soft_body is not None:
                                init_count += 1
                except Exception as e:
                    print(f"[{mname}] Load Tet error: {e}")
            print(f"Loaded {load_count} new tets, initialized {init_count} soft bodies ({already_init_count} already initialized) in {_t.time()-_t0:.1f}s")
        # Run soft body simulation for all muscles at once
        if imgui.button("Run All Tet Sim", width=wide_button_width):
            count = 0
            collision_count = 0
            for mname, mobj in v.zygote_muscle_meshes.items():
                if mobj.tet_vertices is not None:
                    if mobj.soft_body is None:
                        mobj.init_soft_body(v.zygote_skeleton_meshes, v.env.skel, v.env.mesh_info)
                    if mobj.soft_body is not None:
                        # Respect each muscle's individual collision setting
                        iterations, residual = mobj.run_soft_body_to_convergence(
                            v.zygote_skeleton_meshes,
                            v.env.skel,
                            max_iterations=100,
                            tolerance=1e-4,
                            enable_collision=mobj.soft_body_collision,
                            collision_margin=mobj.soft_body_collision_margin,
                            verbose=False,
                            use_arap=mobj.use_arap
                        )
                        count += 1
                        if mobj.soft_body_collision:
                            collision_count += 1
            print(f"Ran tet sim for {count} muscles ({collision_count} with collision)")

        # Inter-muscle constraints section
        imgui.separator()
        imgui.text("Inter-Muscle Constraints")

        # Threshold slider
        imgui.push_item_width(100)
        changed, v.inter_muscle_constraint_threshold = imgui.slider_float(
            "Threshold (m)", v.inter_muscle_constraint_threshold, 0.001, 0.05
        )
        imgui.pop_item_width()
        imgui.same_line()
        imgui.text(f"({v.inter_muscle_constraint_threshold*100:.1f}cm)")

        # Find constraints button
        if imgui.button("Find Constraints", width=wide_button_width):
            count = find_inter_muscle_constraints(v)

        imgui.text(f"{len(v.inter_muscle_constraints)} constraints")
        _, v.draw_inter_muscle_constraints = imgui.checkbox(
            "Draw##inter_constraints", getattr(v, 'draw_inter_muscle_constraints', False)
        )

        # Unified-volume + muscle-aware ARAP are the only path now (Taichi
        # backend, no per-muscle/FEM/CPU/GPU alternatives).
        _, v.coupled_as_unified_volume = imgui.checkbox(
            "Unified Volume", v.coupled_as_unified_volume
        )

        # Run coupled simulation button
        if imgui.button("Run Coupled Tet Sim", width=wide_button_width):
            run_all_tet_sim_with_constraints(v)

        imgui.separator()

        _draw_zygote_group_ui(v)
        hidden_group_parts = _zygote_hidden_group_part_names(v)

        for name, obj in v.zygote_muscle_meshes.items():
            if name in hidden_group_parts:
                continue
            if imgui.tree_node(name):
                _draw_zygote_muscle_body(v, name, obj)
                imgui.tree_pop()
        imgui.tree_pop()


def _reimport_zygote_skel(v):
    """Reload data/zygote_skel.xml into DART skel + rebuild dependent state
    (muscles, soft bodies, attach indices). Used by Import zygote_skel button
    and Import zygote_muscle button so muscle import always sees the latest
    skeleton."""
    from core.dartHelper import saveSkeletonInfo, buildFromInfo
    from copy import deepcopy
    import trimesh as _trimesh

    # Rescan Zygote_Meshes_251229/Skeleton/ and load any OBJ file not already
    # in v.zygote_skeleton_meshes. Skips files already present so existing
    # state (corners, hierarchy edits, is_draw flags) isn't lost.
    skel_dir = 'Zygote_Meshes_251229/Skeleton'
    if os.path.isdir(skel_dir):
        new_loaded = []
        for fname in sorted(os.listdir(skel_dir)):
            if not fname.endswith('.obj'):
                continue
            mesh_name = fname[:-4]
            if mesh_name in v.zygote_skeleton_meshes:
                continue
            path = os.path.join(skel_dir, fname)
            try:
                ml = MeshLoader()
                ml.load(path)
                ml.color = np.array([0.9, 0.9, 0.9])
                tri = _trimesh.load_mesh(path)
                tri.vertices *= 0.01  # MESH_SCALE
                ml.trimesh = tri
                v.zygote_skeleton_meshes[mesh_name] = ml
                new_loaded.append(mesh_name)
            except Exception as _e:
                print(f"[Import zygote_skel] failed to load {fname}: {_e}")
        if new_loaded:
            v.zygote_skeleton_meshes = dict(sorted(v.zygote_skeleton_meshes.items()))
            # Refresh cand_parent_index so dropdown ordering is stable.
            for _i, (_n, _m) in enumerate(v.zygote_skeleton_meshes.items()):
                _m.cand_parent_index = _i
            print(f"[Import zygote_skel] loaded {len(new_loaded)} new mesh(es): {new_loaded}")

    if v.env.skel is not None:
        v.env.world.removeSkeleton(v.env.skel)

    skel_info, root_name, bvh_info, _, mesh_info, smpl_jn_idx = saveSkeletonInfo(
        "data/zygote_skel.xml")
    v.env.skel_info = skel_info
    v.env.new_skel_info = deepcopy(skel_info)
    v.env.root_name = root_name
    v.env.bvh_info = bvh_info
    v.env.mesh_info = mesh_info
    v.env.smpl_jn_idx = smpl_jn_idx
    v.env.skel = buildFromInfo(skel_info, root_name)
    v.env.target_skel = v.env.skel.clone()
    v.env.world.addSkeleton(v.env.skel)
    v.env.kp = 300.0 * np.ones(v.env.skel.getNumDofs())
    v.env.kv = 20.0 * np.ones(v.env.skel.getNumDofs())
    v.env.kp[:6] = 0.0
    v.env.kv[:6] = 0.0
    v.env.num_action = len(v.env.get_zero_action()) * (3 if v.env.learning_gain else 1)
    v.motion_skel = v.env.skel.clone()

    # Rebuild dependent state — see Import zygote_skel button history for
    # why each step matters.
    try:
        import dartpy as _dart_skel_rebuild
        if getattr(v.env, 'muscle_info', None):
            v.env.loading_zygote_muscle_info(v.env.muscle_info)
            v.env.muscle_activation_levels = np.zeros(
                v.env.muscles.getNumMuscles())
        else:
            v.env.muscles = _dart_skel_rebuild.dynamics.Muscles(v.env.skel)
            v.env.muscle_activation_levels = np.zeros(0)
    except Exception as _e:
        print(f"[Import zygote_skel] muscle rebuild failed: {_e}")
    for _mname, _mobj in v.zygote_muscle_meshes.items():
        if getattr(_mobj, 'soft_body', None) is None:
            continue
        try:
            _mobj.soft_body = None
            _mobj.init_soft_body(
                skeleton_meshes=v.zygote_skeleton_meshes,
                skeleton=v.env.skel,
                mesh_info=v.env.mesh_info,
            )
        except Exception as _e:
            print(f"[Import zygote_skel] re-init {_mname} failed: {_e}")
    _skel_names = list(v.zygote_skeleton_meshes.keys())
    for _mobj in v.zygote_muscle_meshes.values():
        if hasattr(_mobj, 'resolve_skeleton_attachments'):
            try:
                _mobj.resolve_skeleton_attachments(_skel_names)
            except Exception:
                pass
    # Reload self.meshes from updated mesh_info (new bones added mid-session
    # like Sternum0 won't be in self.meshes otherwise), then re-run the
    # body-local vertex shift so drawObj puts the OBJ at the right world
    # rest position. Without this, newly added bones either don't render at
    # all OR render at a stale offset.
    import trimesh as _tm
    for _k, _info in v.env.mesh_info.items():
        if _k in v.meshes:
            continue
        try:
            ml = MeshLoader()
            ml.load(_info)
            ml.use_two_pass_culling = False
            v.meshes[_k] = ml
            print(f"[Import zygote_skel] loaded {_k} mesh from {_info}")
        except Exception as _e:
            print(f"[Import zygote_skel] mesh load {_k} failed: {_e}")
    # Re-run vertex shift for every mesh (cheap; ensures shifts match
    # the current skel rest pose).
    for _bn in v.env.skel.getBodyNodes():
        _name = _bn.getName()
        if _name not in v.meshes:
            continue
        try:
            _xform = (np.asarray(_bn.getWorldTransform().matrix())
                      @ np.asarray(_bn.getParentJoint().getTransformFromChildBodyNode().matrix()))
            _t_parent = _xform[:3, 3]
            _mesh = v.meshes[_name]
            if hasattr(_mesh, 'vertices_3') and len(_mesh.vertices_3) > 0:
                # vertices_3 store offsets; reset to original then re-subtract.
                # If we don't track orig, re-load from file.
                # Simpler: reload mesh fresh, then subtract.
                fresh = MeshLoader()
                fresh.load(_mesh.obj)
                fresh.use_two_pass_culling = False
                fresh.vertices_3 -= _t_parent
                if hasattr(fresh, 'new_vertices_3') and len(fresh.new_vertices_3) > 0:
                    fresh.new_vertices_3 -= _t_parent
                if len(fresh.vertices_4) > 0:
                    fresh.vertices_4 -= _t_parent
                v.meshes[_name] = fresh
        except Exception as _e:
            print(f"[Import zygote_skel] vertex shift for {_name} failed: {_e}")
    print(f"Import zygote_skel: DART skel {v.env.skel.getNumBodyNodes()} bodies, "
          f"muscles {v.env.muscles.getNumMuscles() if v.env.muscles else 0}, "
          f"self.meshes {len(v.meshes)} entries")


def draw_zygote_skeleton_ui(v):
    """Skeleton section inside the Zygote tree node."""
    if imgui.tree_node("Skeleton"):
        changed, v.is_draw_zygote_skeleton = imgui.checkbox("Draw", v.is_draw_zygote_skeleton)
        if changed:
            for name, obj in v.zygote_skeleton_meshes.items():
                obj.is_draw = v.is_draw_zygote_skeleton
        _, v.is_draw_one_zygote_skeleton = imgui.checkbox("Draw One Skeleton", v.is_draw_one_zygote_skeleton)
        changed, v.zygote_skeleton_color = imgui.color_edit3("Color", *v.zygote_skeleton_color)
        if changed:
            for name, obj in v.zygote_skeleton_meshes.items():
                obj.color = v.zygote_skeleton_color
        if imgui.button("Draw All"):
            for name, obj in v.zygote_skeleton_meshes.items():
                obj.is_draw = True
        changed, v.zygote_skeleton_transparency = imgui.slider_float("Transparency##Skeleton", v.zygote_skeleton_transparency, 0.0, 1.0)
        if changed:
            for name, obj in v.zygote_skeleton_meshes.items():
                obj.transparency = v.zygote_skeleton_transparency

        for i, (name, obj) in enumerate(v.zygote_skeleton_meshes.items()):
            if imgui.tree_node(f"{i}: {name}"):
                changed, obj.transparency = imgui.slider_float(f"Transparency##{name}", obj.transparency, 0.0, 1.0)
                if changed and obj.vertex_colors is not None:
                    obj.vertex_colors[:, 3] = obj.transparency
                _, obj.is_draw = imgui.checkbox("Draw", obj.is_draw)
                _, obj.is_draw_corners = imgui.checkbox("Draw Corners", obj.is_draw_corners)
                _, obj.is_draw_edges = imgui.checkbox("Draw Edges", obj.is_draw_edges)
                _, obj.is_contact = imgui.checkbox("Contact", obj.is_contact)
                if obj.is_draw and v.is_draw_one_zygote_skeleton:
                    for other_name, other_obj in v.zygote_skeleton_meshes.items():
                        other_obj.is_draw = False
                        obj.is_draw = True
                imgui.tree_pop()

                # Auto checkbox and num boxes slider
                _, obj.auto_num_boxes = imgui.checkbox(f"Auto##{name}_auto", obj.auto_num_boxes)
                imgui.same_line()
                if obj.auto_num_boxes:
                    imgui.text(f"Boxes: {obj.num_boxes} (auto)")
                else:
                    imgui.push_item_width(100)
                    _, obj.num_boxes = imgui.slider_int(f"Boxes##{name}", obj.num_boxes, 1, 10)
                    imgui.pop_item_width()

                # Axis alignment checkboxes
                imgui.text("Align:")
                imgui.same_line()
                _, obj.bb_align_x = imgui.checkbox(f"X##{name}_bbx", obj.bb_align_x)
                imgui.same_line()
                _, obj.bb_align_y = imgui.checkbox(f"Y##{name}_bby", obj.bb_align_y)
                imgui.same_line()
                _, obj.bb_align_z = imgui.checkbox(f"Z##{name}_bbz", obj.bb_align_z)
                imgui.same_line()
                _, obj.bb_enforce_symmetry = imgui.checkbox(f"Sym##{name}_sym", obj.bb_enforce_symmetry)
                imgui.same_line()
                if not hasattr(obj, 'bb_midline'):
                    obj.bb_midline = False
                _, obj.bb_midline = imgui.checkbox(f"Mid##{name}_mid", obj.bb_midline)
                if obj.bb_midline:
                    # Mid forces x-align + disables L/R symmetry
                    obj.bb_enforce_symmetry = False

                # Build axis string from checkboxes
                axis_str = ''
                if obj.bb_align_x or obj.bb_midline:
                    axis_str += 'x'
                if obj.bb_align_y:
                    axis_str += 'y'
                if obj.bb_align_z:
                    axis_str += 'z'
                axis_param = axis_str if axis_str else None

                # Show current alignment
                align_label = axis_str.upper() if axis_str else "PCA"
                sym_label = "+Sym" if obj.bb_enforce_symmetry else ""
                mid_label = "+Mid" if obj.bb_midline else ""
                auto_label = "+Auto" if obj.auto_num_boxes else ""
                if imgui.button(f"Find BB ({align_label}{sym_label}{mid_label}{auto_label})##{name}", width=140):
                    obj.find_bounding_box(axis=axis_param,
                                          symmetry=obj.bb_enforce_symmetry,
                                          midline=obj.bb_midline)

                imgui.text(f"Parent: {obj.parent_name}")
                _skel_names = list(v.zygote_skeleton_meshes.keys())
                if obj.cand_parent_index >= len(_skel_names):
                    obj.cand_parent_index = max(0, len(_skel_names) - 1)
                elif obj.cand_parent_index < 0:
                    obj.cand_parent_index = 0
                imgui.push_item_width(180)
                changed, new_idx = imgui.combo(
                    f"Parent##{name}_parent", obj.cand_parent_index, _skel_names)
                imgui.pop_item_width()
                if changed:
                    obj.cand_parent_index = new_idx
                cand_name = _skel_names[obj.cand_parent_index]

                if imgui.button(f"Set as root##{name}"):
                    for other_name, other_obj in v.zygote_skeleton_meshes.items():
                        other_obj.is_root = False
                    obj.is_root = True
                    print(f"{name} set as root")
                if imgui.button(f"Connect to parent##{name}"):

                    if v.zygote_skeleton_meshes[cand_name].corners is None:
                        print("First find bounding boxes for parent mesh")
                    elif obj.corners is None:
                        print("First find bounding boxes for this mesh")
                    elif name == cand_name:
                        print("Self connection")
                    else:
                        parent_mesh = v.zygote_skeleton_meshes[cand_name]
                        parent_corners_list = parent_mesh.corners_list
                        cand_joints = []
                        for parent_corners in parent_corners_list:
                            for corners in obj.corners_list:
                                cand_joint = obj.find_overlap_point(parent_corners, corners)
                                if cand_joint is not None:
                                    cand_joints.append(cand_joint)

                        if len(cand_joints) == 0:
                            print("They can't be linked; No overlapping points")
                        else:
                            obj.parent_mesh = parent_mesh
                            obj.parent_name = cand_name
                            if not name in parent_mesh.children_names:
                                parent_mesh.children_names.append(name)
                            print(f"{name} connected to {cand_name}")
                            if len(cand_joints) == 1:
                                obj.joint_to_parent = cand_joints[0]
                            else:
                                mean = np.mean(obj.vertices)
                                distances = np.linalg.norm(cand_joints - mean, axis=1)
                                obj.joint_to_parent = cand_joints[np.argmax(distances)]
                            obj.is_weld = False
                imgui.same_line()
                if imgui.button(f"Connect to parent as Weld##{name}"):
                    if v.zygote_skeleton_meshes[cand_name].corners is None:
                        print("First find bounding boxes for parent mesh")
                    elif obj.corners is None:
                        print("First find bounding boxes for this mesh")
                    elif name == cand_name:
                        print("Self connection")
                    else:
                        parent_mesh = v.zygote_skeleton_meshes[cand_name]
                        parent_corners_list = parent_mesh.corners_list
                        cand_joints = []
                        for parent_corners in parent_corners_list:
                            for corners in obj.corners_list:
                                cand_joint = obj.find_overlap_point(parent_corners, corners)
                                if cand_joint is not None:
                                    cand_joints.append(cand_joint)

                        if len(cand_joints) == 0:
                            print("They can't be linked; No overlapping points")
                        else:
                            obj.parent_mesh = parent_mesh
                            obj.parent_name = cand_name
                            parent_mesh.children_names.append(name)
                            print(f"{name} connected to {cand_name} as Weld")
                            if len(cand_joints) == 1:
                                obj.joint_to_parent = cand_joints[0]
                            else:
                                mean = np.mean(obj.vertices)
                                distances = np.linalg.norm(cand_joints - mean, axis=1)
                                obj.joint_to_parent = cand_joints[np.argmax(distances)]
                            obj.is_weld = True

                # Revolute joint connection buttons. Try the OBB overlap
                # picker first (same logic as Weld Connect); fall back to
                # nearest vertex-pair midpoint when no overlap exists
                # (handles meshes like L 2nd distal phalanx ↔ middle
                # phalanx that don't geometrically intersect).
                def _pick_revolute_joint(child_obj, parent_obj):
                    """Overlap point if available, else nearest-pair midpoint."""
                    if (child_obj.corners_list is not None
                            and parent_obj.corners_list is not None):
                        cand_joints = []
                        for pc in parent_obj.corners_list:
                            for cc in child_obj.corners_list:
                                pt = child_obj.find_overlap_point(pc, cc)
                                if pt is not None:
                                    cand_joints.append(pt)
                        if cand_joints:
                            if len(cand_joints) == 1:
                                return np.asarray(cand_joints[0], dtype=np.float32), 'overlap'
                            cand_arr = np.asarray(cand_joints)
                            ctr = np.mean(child_obj.vertices, axis=0)
                            d = np.linalg.norm(cand_arr - ctr, axis=1)
                            return cand_arr[int(np.argmax(d))].astype(np.float32), 'overlap'
                    # Fallback: midpoint between the closest OBB CORNERS of
                    # the two meshes. More stable than nearest-vertex (which
                    # can pick anatomically wrong points on curved bones).
                    if (child_obj.corners_list is None
                            or parent_obj.corners_list is None):
                        return None, None
                    best_pair = None
                    best_d = float('inf')
                    for pc in parent_obj.corners_list:
                        pc_arr = np.asarray(pc)
                        for cc in child_obj.corners_list:
                            cc_arr = np.asarray(cc)
                            # pairwise distances between all 8x8 corner combos
                            diff = cc_arr[:, None, :] - pc_arr[None, :, :]
                            d2 = np.einsum('ijk,ijk->ij', diff, diff)
                            i_min, j_min = np.unravel_index(np.argmin(d2), d2.shape)
                            d = float(np.sqrt(d2[i_min, j_min]))
                            if d < best_d:
                                best_d = d
                                best_pair = ((cc_arr[i_min] + pc_arr[j_min]) / 2.0)
                    if best_pair is None:
                        return None, None
                    return best_pair.astype(np.float32), 'corner-pair'

                def _connect_revolute(child_obj, parent_obj, parent_name_,
                                      bends_backward, lower, upper, label):
                    pt, mode = _pick_revolute_joint(child_obj, parent_obj)
                    if pt is None:
                        print(f"  could not pick joint pivot; skipping")
                        return
                    child_obj.parent_mesh = parent_obj
                    child_obj.parent_name = parent_name_
                    if name not in parent_obj.children_names:
                        parent_obj.children_names.append(name)
                    child_obj.joint_to_parent = pt
                    child_obj.is_weld = False
                    child_obj.is_revolute = True
                    child_obj.bends_backward = bends_backward
                    child_obj.revolute_axis = np.array([1.0, 0.0, 0.0])
                    child_obj.revolute_lower = lower
                    child_obj.revolute_upper = upper
                    print(f"{name} → {parent_name_} as {label} "
                          f"(pivot via {mode} @ {pt})")

                def _revolute_preconditions():
                    """Both meshes must have Find BB run, otherwise nearest-
                    vertex fallback picks anatomically wrong points (meshes
                    in their unfit rest pose)."""
                    if name == cand_name:
                        print("Self connection")
                        return False
                    if v.zygote_skeleton_meshes[cand_name].corners is None:
                        print("First find bounding boxes for parent mesh")
                        return False
                    if obj.corners is None:
                        print("First find bounding boxes for this mesh")
                        return False
                    return True

                if imgui.button(f"Revolute (Knee/Toe)##{name}"):
                    if _revolute_preconditions():
                        _connect_revolute(
                            obj, v.zygote_skeleton_meshes[cand_name], cand_name,
                            bends_backward=True, lower=0.0, upper=2.5,
                            label="Revolute (Knee/Toe, bends backward)")

                imgui.same_line()
                if imgui.button(f"Revolute (Elbow/Finger)##{name}"):
                    if _revolute_preconditions():
                        _connect_revolute(
                            obj, v.zygote_skeleton_meshes[cand_name], cand_name,
                            bends_backward=False, lower=-2.5, upper=0.0,
                            label="Revolute (Elbow/Finger, bends forward)")

                # Show revolute joint settings if connected as revolute
                if obj.is_revolute:
                    label = 'Knee/Toe (backward)' if obj.bends_backward else 'Elbow/Finger (forward)'
                    imgui.text(f"Revolute: {label}")
                    imgui.push_item_width(80)
                    changed, obj.revolute_lower = imgui.input_float(f"Lower##{name}_rev", obj.revolute_lower, 0.1)
                    imgui.same_line()
                    changed, obj.revolute_upper = imgui.input_float(f"Upper##{name}_rev", obj.revolute_upper, 0.1)
                    imgui.pop_item_width()

                if imgui.button("Export Bounding boxes", width=wide_button_width):
                    from core.dartHelper import exportBoundingBoxes
                    exportBoundingBoxes(v.zygote_skeleton_meshes)
                if imgui.button("Import zygote_skel", width=wide_button_width):
                    _reimport_zygote_skel(v)

        # --- Skeleton XML Edit Mode ---
        imgui.separator()
        xml_loaded = hasattr(v.env, 'new_skel_info') and v.env.new_skel_info is not None
        if not hasattr(v, 'skel_edit_mode'):
            v.skel_edit_mode = False
        if not xml_loaded:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
            imgui.checkbox("Edit Mode##skel", False)
            imgui.pop_style_var()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Load zygote_skel XML first")
            if v.skel_edit_mode:
                # XML unloaded mid-session — drop edit mode + hide BBs
                v.skel_edit_mode = False
                for _sm in getattr(v, 'zygote_skeleton_meshes', {}).values():
                    _sm.is_draw_corners = False
                v.joint_edit_mode = False
        else:
            prev_edit_mode = v.skel_edit_mode
            _, v.skel_edit_mode = imgui.checkbox("Edit Mode##skel", v.skel_edit_mode)
            if v.skel_edit_mode != prev_edit_mode:
                # Edge: walk skel meshes once, toggle is_draw_corners.
                # Per-mesh Draw Corners toggle stays independent between edges.
                for _sm in getattr(v, 'zygote_skeleton_meshes', {}).values():
                    _sm.is_draw_corners = v.skel_edit_mode
                v.joint_edit_mode = False  # default OFF on Edit Mode toggle (either dir)
                if v.skel_edit_mode:
                    # Seed mesh hierarchy from XML so Save XML / exportBoundingBoxes
                    # has root + parent links. User can still edit after.
                    skel_meshes = v.zygote_skeleton_meshes
                    # Resolve body key → mesh name by picking the LONGEST mesh
                    # name that is a prefix of the key with a digits-only
                    # suffix. Plain re.sub(r'\d+$', '') breaks on names like
                    # 'L50' (mesh 'L5', body '0') — it strips both digits.
                    _sorted_mesh_names = sorted(skel_meshes.keys(), key=lambda x: -len(x))
                    def _resolve_mesh_name(body_key):
                        if not body_key:
                            return None
                        for _sn in _sorted_mesh_names:
                            if not body_key.startswith(_sn):
                                continue
                            _suf = body_key[len(_sn):]
                            if _suf and _suf.isdigit():
                                return _sn
                        return None
                    for _sm2 in skel_meshes.values():
                        _sm2.is_root = False
                        _sm2.parent_mesh = None
                        _sm2.parent_name = None
                        _sm2.children_names = []
                    # Per-mesh: track the lowest-digit body's joint_type so
                    # is_weld reflects XML state (Save XML writes Weld for
                    # the first body only when mesh.is_weld is True).
                    _mesh_first_jtype = {}
                    for _bkey, _info in v.env.new_skel_info.items():
                        _mesh_name = _resolve_mesh_name(_bkey)
                        if _mesh_name is None:
                            continue
                        _digit = int(_bkey[len(_mesh_name):])
                        _jt = _info.get('joint_type', 'Ball')
                        if _mesh_name not in _mesh_first_jtype or _digit < _mesh_first_jtype[_mesh_name][0]:
                            _mesh_first_jtype[_mesh_name] = (_digit, _jt)
                        _mesh = skel_meshes[_mesh_name]
                        _ps = _info.get('parent_str', 'None')
                        if _ps == 'None' or _ps is None:
                            _mesh.is_root = True
                            continue
                        _parent_mesh_name = _resolve_mesh_name(_ps)
                        if _parent_mesh_name is None or _parent_mesh_name == _mesh_name:
                            continue  # intra-mesh body link
                        _parent = skel_meshes[_parent_mesh_name]
                        _mesh.parent_mesh = _parent
                        _mesh.parent_name = _parent_mesh_name
                        if _mesh_name not in _parent.children_names:
                            _parent.children_names.append(_mesh_name)
                    # Apply is_weld from the lowest-digit body's joint_type
                    for _mn, (_dig, _jt) in _mesh_first_jtype.items():
                        skel_meshes[_mn].is_weld = (_jt == 'Weld')
                    # Seed Revolute mesh state from skel_info so the 3D axis
                    # overlay can show without a UI re-click after XML load.
                    for _bkey, _info in v.env.new_skel_info.items():
                        if _info.get('joint_type') != 'Revolute':
                            continue
                        _mesh_name2 = _resolve_mesh_name(_bkey)
                        if _mesh_name2 is None:
                            continue
                        _sm3 = skel_meshes[_mesh_name2]
                        _sm3.is_revolute = True
                        _ax = np.asarray(_info.get('axis', [1.0, 0.0, 0.0]), dtype=np.float32)
                        _sm3.revolute_axis = _ax
                        _sm3.revolute_lower = float(_info.get('lower', -2.5))
                        _sm3.revolute_upper = float(_info.get('upper', 2.5))
                        _sm3.bends_backward = _sm3.revolute_lower >= 0
                        # joint world position (XML stores world-frame translation)
                        _sm3.joint_to_parent = np.asarray(
                            _info.get('joint_t', [0.0, 0.0, 0.0]), dtype=np.float32)
            if v.skel_edit_mode:
                # Continuous: derive per-frame world-space OBBs + sizes/body_rs
                # /body_ts/weld_joints from XML body_r/body_t/size via DART
                # world transform so joint edits update geometry without
                # re-toggling, and exportBoundingBoxes finds the data it needs.
                _local_corners = np.array([
                    [-1, -1, -1], [ 1, -1, -1], [-1,  1, -1], [ 1,  1, -1],
                    [-1, -1,  1], [ 1, -1,  1], [-1,  1,  1], [ 1,  1,  1],
                ], dtype=np.float32) * 0.5
                # Group body keys by mesh, ordered by digit suffix.
                # Greedy '\d+$' breaks on names like 'L50' (mesh 'L5' + body
                # '0'). Pick the longest mesh name that prefixes the key with
                # a digits-only suffix.
                _sorted_mesh_names = sorted(
                    v.zygote_skeleton_meshes.keys(), key=lambda x: -len(x))
                _mesh_bodies = {}
                for key, info in v.env.new_skel_info.items():
                    mesh_name = None
                    digit = None
                    for _sn in _sorted_mesh_names:
                        if not key.startswith(_sn):
                            continue
                        suf = key[len(_sn):]
                        if suf and suf.isdigit():
                            mesh_name = _sn
                            digit = int(suf)
                            break
                    if mesh_name is None:
                        continue
                    _mesh_bodies.setdefault(mesh_name, []).append((digit, key, info))
                for lst in _mesh_bodies.values():
                    lst.sort(key=lambda t: t[0])
                for sname, _sm in v.zygote_skeleton_meshes.items():
                    bodies = _mesh_bodies.get(sname, [])
                    if not bodies:
                        continue
                    body_corners = []
                    sizes_list = []
                    body_rs_list = []
                    body_ts_list = []
                    weld_joints = []
                    joint_to_parent = None
                    for idx, (_digit, key, info) in enumerate(bodies):
                        sz = np.asarray(info['size'], dtype=np.float32)
                        local = _local_corners * sz
                        body_node = (v.env.skel.getBodyNode(key)
                                     if v.env.skel is not None else None)
                        if body_node is not None:
                            T = np.asarray(body_node.getWorldTransform().matrix(), dtype=np.float32)
                            R = T[:3, :3]
                            tw = T[:3, 3]
                        else:
                            R = np.asarray(info['body_r'], dtype=np.float32)
                            tw = np.asarray(info['body_t'], dtype=np.float32)
                        corners = (R @ local.T).T + tw
                        body_corners.append(corners)
                        sizes_list.append(sz)
                        body_rs_list.append(R)
                        body_ts_list.append(np.mean(corners, axis=0))
                        jt = np.asarray(info.get('joint_t', np.zeros(3)), dtype=np.float32)
                        if idx == 0:
                            joint_to_parent = jt
                        else:
                            weld_joints.append(jt)
                    _sm.corners = body_corners
                    # find_bounding_box keeps these two aliased; downstream
                    # ops (Connect to parent overlap check at line 1426/1429)
                    # read corners_list, so mirror the derived OBBs here.
                    _sm.corners_list = body_corners
                    _sm.sizes = sizes_list
                    _sm.body_rs = body_rs_list
                    _sm.body_ts = body_ts_list
                    _sm.weld_joints = weld_joints
                    if joint_to_parent is not None:
                        _sm.joint_to_parent = joint_to_parent
        if v.skel_edit_mode and xml_loaded:
            _, v.joint_edit_mode = imgui.checkbox("Edit Joint Positions", v.joint_edit_mode)
            if v.joint_edit_mode:
                _, v.joint_edit_symmetry = imgui.checkbox("Symmetry (L<>R)", v.joint_edit_symmetry)

                # Joint name list for combo
                joint_names = list(v.env.new_skel_info.keys())
                current_idx = joint_names.index(v.joint_edit_selected) if v.joint_edit_selected in joint_names else -1

                imgui.push_item_width(180)
                changed, new_idx = imgui.combo("Joint##jed", current_idx, joint_names)
                imgui.pop_item_width()
                if changed and new_idx >= 0:
                    v.joint_edit_selected = joint_names[new_idx]
                    mirror = v._get_mirror_name(joint_names[new_idx])
                    v.joint_edit_symmetry = mirror is not None and mirror in v.env.new_skel_info

                # XYZ input for selected joint
                if v.joint_edit_selected and v.joint_edit_selected in v.env.new_skel_info:
                    info = v.env.new_skel_info[v.joint_edit_selected]
                    jt = info['joint_t'].astype(np.float64)
                    any_changed = False
                    imgui.push_item_width(120)
                    c, jt[0] = imgui.input_float("X##jed", jt[0], 0.001, 0.01, "%.5f")
                    any_changed = any_changed or c
                    c, jt[1] = imgui.input_float("Y##jed", jt[1], 0.001, 0.01, "%.5f")
                    any_changed = any_changed or c
                    c, jt[2] = imgui.input_float("Z##jed", jt[2], 0.001, 0.01, "%.5f")
                    any_changed = any_changed or c
                    imgui.pop_item_width()

                    if any_changed:
                        info['joint_t'] = jt
                        # Mirror
                        if v.joint_edit_symmetry:
                            mirror = v._get_mirror_name(v.joint_edit_selected)
                            if mirror and mirror in v.env.new_skel_info:
                                mt = jt.copy()
                                mt[0] = -mt[0]
                                v.env.new_skel_info[mirror]['joint_t'] = mt
                        v.newSkeleton()

                # Revolute axis editor (spherical). Shows when selected joint
                # is either (a) saved as Revolute in new_skel_info OR (b)
                # mesh-flagged is_revolute (not yet saved). Axis stored in
                # JOINT-LOCAL frame. Slider angles:
                #   axis_local = (cos(el)*cos(az), sin(el), cos(el)*sin(az))
                # Az=0 → +X, Az=90° → +Z, El=+90° → +Y. Symmetric mode
                # mirrors across YZ plane (X-flip) onto the L/R counterpart.
                _sel = v.joint_edit_selected
                _info = v.env.new_skel_info.get(_sel) if _sel else None
                _is_saved_rev = _info is not None and _info.get('joint_type') == 'Revolute'
                _mesh_name = _sel[:-1] if _sel and _sel.endswith('0') else None
                _mesh = v.zygote_skeleton_meshes.get(_mesh_name) if _mesh_name else None
                _is_mesh_rev = bool(getattr(_mesh, 'is_revolute', False)) if _mesh else False
                if _is_saved_rev or _is_mesh_rev:
                    if _is_saved_rev:
                        ax = np.asarray(_info.get('axis', np.array([1.0, 0.0, 0.0])),
                                        dtype=np.float64)
                    else:
                        ax = np.asarray(_mesh.revolute_axis, dtype=np.float64)
                    n = np.linalg.norm(ax)
                    if n > 1e-9:
                        ax = ax / n
                    el = float(np.degrees(np.arcsin(np.clip(ax[1], -1.0, 1.0))))
                    az = float(np.degrees(np.arctan2(ax[2], ax[0])))
                    imgui.text(f"Revolute axis (local): ({ax[0]:.3f}, {ax[1]:.3f}, {ax[2]:.3f})")
                    imgui.push_item_width(120)
                    c1, new_az = imgui.input_float(
                        f"Azimuth°##jed_ax", az, 1.0, 10.0, "%.3f")
                    c2, new_el = imgui.input_float(
                        f"Elevation°##jed_el", el, 1.0, 10.0, "%.3f")
                    imgui.pop_item_width()
                    if c1 or c2:
                        az_r = np.radians(new_az)
                        el_r = np.radians(new_el)
                        new_axis = np.array([
                            np.cos(el_r) * np.cos(az_r),
                            np.sin(el_r),
                            np.cos(el_r) * np.sin(az_r),
                        ], dtype=np.float64)
                        if _is_saved_rev:
                            _info['axis'] = new_axis
                        if _mesh is not None:
                            _mesh.revolute_axis = new_axis.astype(np.float32)
                        # Symmetric mirror
                        if v.joint_edit_symmetry:
                            mirror = v._get_mirror_name(_sel)
                            if mirror:
                                m_axis = new_axis.copy()
                                m_axis[0] = -m_axis[0]
                                if mirror in v.env.new_skel_info and \
                                        v.env.new_skel_info[mirror].get('joint_type') == 'Revolute':
                                    v.env.new_skel_info[mirror]['axis'] = m_axis
                                m_mesh_name = mirror[:-1] if mirror.endswith('0') else None
                                m_mesh = v.zygote_skeleton_meshes.get(m_mesh_name) if m_mesh_name else None
                                if m_mesh is not None and getattr(m_mesh, 'is_revolute', False):
                                    m_mesh.revolute_axis = m_axis.astype(np.float32)
                        # Axis-only change: the 3D arrow re-reads
                        # mesh.revolute_axis next frame, so no skel rebuild
                        # needed. (Joint position change still rebuilds via
                        # the XYZ input handler above.) Skipping newSkeleton
                        # makes the +/- click respond instantly.

                # Save / Reset buttons
                if imgui.button("Save XML##jed", width=100):
                    from core.dartHelper import exportSkeleton
                    exportSkeleton(v.env.new_skel_info, v.env.root_name, 'zygote_skel.xml')
                    print("Saved joint positions to data/zygote_skel.xml")
                imgui.same_line()
                if imgui.button("Reset All##jed", width=100):
                    from copy import deepcopy
                    v.env.new_skel_info = deepcopy(v.env.skel_info)
                    v.joint_edit_selected = None
                    v.newSkeleton()

        imgui.tree_pop()


def _draw_motion_browser_ui(v):
    """Draw the Motion Browser imgui panel."""
    # Section 1: Motion file selection
    bvh_names = [os.path.basename(f) for f in v.motion_bvh_files]
    if len(bvh_names) == 0:
        imgui.text("No .bvh files in data/motion/")
        if imgui.button("Rescan##motion"):
            _scan_motion_files(v)
        return

    imgui.push_item_width(200)
    preview = bvh_names[v.motion_selected_idx] if 0 <= v.motion_selected_idx < len(bvh_names) else "-- Select BVH --"
    if imgui.begin_combo("Motion##bvh_select", preview):
        for i, bname in enumerate(bvh_names):
            is_selected = (i == v.motion_selected_idx)
            clicked, _ = imgui.selectable(bname, is_selected)
            if clicked:
                _load_motion_bvh(v, i)
            if is_selected:
                imgui.set_item_default_focus()
        imgui.end_combo()
    imgui.pop_item_width()

    # Reload cache from disk (useful while a bake is writing fresh chunks).
    if v.motion_bvh is not None:
        if imgui.button("Load Latest Cache##motion"):
            import time as _t
            _t0 = _t.time()
            _motion_load_cache(v, force=True, prefer_latest=True)
            _motion_load_tissue_cage_overlay(v)
            # Refresh the currently displayed geometry immediately. Reloading
            # only the dictionaries left the old cache positions onscreen
            # until the user changed frames.
            current_frame = int(getattr(v, "motion_current_frame", 0))
            _motion_apply_pose(v, current_frame)
            if v.motion_use_nn and v.motion_nn_model is not None:
                _motion_apply_nn_deformation(v, current_frame)
            else:
                _motion_apply_cached_deformation(v, current_frame)
            print(f"[Motion] Cache reload: {_t.time() - _t0:.2f}s, {len(v.motion_deform_cache)} muscles")
        imgui.same_line()
    if imgui.button("Reload BVH List##motion"):
        _scan_motion_files(v)
        print(f"[Motion] BVH list rescanned: {len(v.motion_bvh_files)} files")

    # Show info if loaded
    if v.motion_bvh is not None:
        fps = 1.0 / v.motion_bvh.frame_time
        duration = v.motion_total_frames * v.motion_bvh.frame_time
        imgui.text(f"Frames: {v.motion_total_frames}   FPS: {fps:.0f}   Duration: {duration:.2f}s")

        # Section 2: Sequential Playback Transport
        # Frame slider
        imgui.push_item_width(imgui.get_content_region_available_width())
        changed, new_frame = imgui.slider_int(
            "##frame_slider", v.motion_current_frame,
            0, v.motion_total_frames - 1,
            f"Frame {v.motion_current_frame} / {v.motion_total_frames - 1}")
        imgui.pop_item_width()
        if changed and new_frame != v.motion_current_frame:
            _motion_apply_pose(v, new_frame)
            if v.motion_use_nn and v.motion_nn_model is not None:
                _motion_apply_nn_deformation(v, new_frame)
                if v.motion_nn_error_heatmap:
                    _motion_update_nn_error_heatmap(v, new_frame)
                else:
                    _motion_clear_heatmap(v)
            else:
                _motion_clear_heatmap(v)
                _motion_apply_cached_deformation(v, new_frame)
            if getattr(v, 'reverse_lbs_enabled', False):
                _reverse_lbs_apply(v)

        # Convergence-anim scrubber: visible only if cache has positions_anim.
        anim_k = _motion_anim_steps(v, v.motion_current_frame)
        if anim_k > 1:
            if not hasattr(v, 'motion_anim_idx'):
                v.motion_anim_idx = anim_k - 1
            v.motion_anim_idx = min(v.motion_anim_idx, anim_k - 1)
            imgui.push_item_width(imgui.get_content_region_available_width())
            ch_a, new_a = imgui.slider_int(
                "##anim_slider", v.motion_anim_idx, 0, anim_k - 1,
                f"Iron-Man iter {v.motion_anim_idx} / {anim_k - 1}")
            imgui.pop_item_width()
            if ch_a:
                v.motion_anim_idx = new_a
                _motion_apply_cached_deformation(v, v.motion_current_frame)

        # Transport buttons
        if imgui.button("Reset##motion"):
            _motion_reset(v)
        imgui.same_line()
        if imgui.button("Step -1##motion"):
            new_frame = max(0, v.motion_current_frame - 1)
            if new_frame != v.motion_current_frame:
                _motion_apply_pose(v, new_frame)
                if v.motion_use_nn and v.motion_nn_model is not None:
                    _motion_apply_nn_deformation(v, new_frame)
                    if v.motion_nn_error_heatmap:
                        _motion_update_nn_error_heatmap(v, new_frame)
                    else:
                        _motion_clear_heatmap(v)
                else:
                    _motion_clear_heatmap(v)
                    _motion_apply_cached_deformation(v, new_frame)
                if getattr(v, 'reverse_lbs_enabled', False):
                    _reverse_lbs_apply(v)
        imgui.same_line()
        if imgui.button("Step +1##motion"):
            _motion_step_forward(v, 1, run_tet=v.motion_run_tet_sim)

        # Play/Pause + speed
        if v.motion_is_playing:
            if imgui.button("Pause##motion", width=80):
                v.motion_is_playing = False
        else:
            if imgui.button("Play##motion", width=80):
                if v.motion_current_frame < v.motion_total_frames - 1 or v.motion_repeat:
                    v.motion_is_playing = True
                    v.motion_play_accumulator = 0.0
        imgui.same_line()
        speed_options = [0.1, 0.25, 0.5, 1.0, 2.0]
        speed_labels = ["0.1x", "0.25x", "0.5x", "1.0x", "2.0x"]
        current_speed_idx = speed_options.index(v.motion_play_speed) if v.motion_play_speed in speed_options else 3
        imgui.push_item_width(80)
        changed, new_speed_idx = imgui.combo("Speed##motion", current_speed_idx, speed_labels)
        imgui.pop_item_width()
        if changed:
            v.motion_play_speed = speed_options[new_speed_idx]

        # Repeat option
        _, v.motion_repeat = imgui.checkbox("Repeat##motion", v.motion_repeat)

        # Fix root translation axes at rest position
        _, v.motion_fix_x = imgui.checkbox("Fix X##motion", v.motion_fix_x)
        imgui.same_line()
        _, v.motion_fix_y = imgui.checkbox("Fix Y##motion", v.motion_fix_y)
        imgui.same_line()
        _, v.motion_fix_z = imgui.checkbox("Fix Z##motion", v.motion_fix_z)
        imgui.same_line()
        _, v.motion_fix_rotation = imgui.checkbox("Fix Rotation##motion", v.motion_fix_rotation)

        # Coupled tet sim option
        _, v.motion_run_tet_sim = imgui.checkbox("Run Coupled Tet Sim##motion", v.motion_run_tet_sim)
        if v.motion_run_tet_sim:
            imgui.push_item_width(150)
            _, v.motion_settle_iters = imgui.slider_int("Settle Iters##motion", v.motion_settle_iters, 10, 200)
            imgui.pop_item_width()

        # --- Deformation Cache ---
        imgui.text("--- Deformation Cache ---")

        # Cache info: count how many frames are cached. Snapshot values() to a
        # list so the background loader thread can keep mutating the dict safely.
        cached_frames = set()
        for mname_cache in list(v.motion_deform_cache.values()):
            cached_frames.update(mname_cache.keys())
        num_cached = len(cached_frames)
        loading_tag = " [loading...]" if getattr(v, 'motion_cache_loading', False) else ""
        imgui.text(f"Cache: {num_cached}/{v.motion_total_frames} frames baked{loading_tag}")

        changed_cage, v.draw_tissue_cage = imgui.checkbox(
            "Draw Tissue Cage##motion_cage",
            getattr(v, "draw_tissue_cage", False))
        if changed_cage and v.draw_tissue_cage:
            _motion_load_tissue_cage_overlay(v)
        if getattr(v, "draw_tissue_cage", False):
            _, v.draw_tissue_cage_rest = imgui.checkbox(
                "Rest Cage##motion_cage_rest",
                getattr(v, "draw_tissue_cage_rest", False))
            overlay = getattr(v, "tissue_cage_overlay", None)
            if overlay is not None:
                frame = int(v.motion_current_frame)
                if (not v.draw_tissue_cage_rest
                        and frame not in overlay["frames"]):
                    imgui.text_colored(
                        f"No saved cage for frame {frame}; showing rest",
                        1.0, 0.65, 0.2)

        # Save Current Frame button
        has_soft_bodies = any(m.soft_body is not None for m in v.zygote_muscle_meshes.values()) if hasattr(v, 'zygote_muscle_meshes') else False
        if not has_soft_bodies or v.motion_baking:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
            imgui.button("Save Current Frame##motion_cache")
            imgui.pop_style_var()
        else:
            if imgui.button("Save Current Frame##motion_cache"):
                _motion_save_current_frame(v)

        # Bake End Frame slider
        imgui.push_item_width(150)
        _, v.motion_bake_end_frame = imgui.slider_int(
            "Bake End Frame##motion_cache",
            v.motion_bake_end_frame, 0, max(1, v.motion_total_frames - 1))
        imgui.pop_item_width()

        # Bake / Cancel buttons
        if v.motion_baking:
            # Show progress during bake
            bake_frac = v.motion_bake_current / max(1, v.motion_bake_end_frame)
            imgui.progress_bar(bake_frac, (imgui.get_content_region_available_width(), 0),
                              f"Baking: frame {v.motion_bake_current}/{v.motion_bake_end_frame}")
            if imgui.button("Cancel Bake##motion_cache"):
                _motion_bake_finish(v)
                print("Bake cancelled, partial results saved")
        else:
            if not has_soft_bodies:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
                imgui.button("Bake to End Frame##motion_cache")
                imgui.pop_style_var()
            else:
                if imgui.button("Bake to End Frame##motion_cache"):
                    _motion_start_bake(v)

        # Recompute/patch waypoints in cache
        if num_cached > 0 and has_soft_bodies and not v.motion_baking:
            if imgui.button("Recompute Waypoints in Cache##motion_cache"):
                _motion_patch_waypoints(v)

        # --- Neural Network ---
        imgui.separator()
        imgui.text("--- Neural Network ---")
        changed_nn, v.motion_use_nn = imgui.checkbox("Use NN Checkpoint##motion_nn", v.motion_use_nn)
        if changed_nn and v.motion_use_nn:
            # Reload checkpoint from disk (picks up training updates)
            _motion_load_nn_checkpoint(v)
            if v.motion_nn_model is None:
                v.motion_use_nn = False
            else:
                # Apply NN to current frame immediately
                _motion_apply_nn_deformation(v, v.motion_current_frame)
        nn_available = v.motion_nn_model is not None
        if not nn_available and not v.motion_use_nn:
            imgui.text("No checkpoint found")
        elif nn_available:
            val_str = f", val={v._motion_nn_val_loss:.6f}" if v._motion_nn_val_loss is not None else ""
            ver_str = f" [{v._motion_nn_model_version}]" if hasattr(v, '_motion_nn_model_version') else ""
            imgui.text(f"best.pt (epoch {v._motion_nn_epoch}{val_str}){ver_str}")
            if imgui.button("Reload NN##nn_reload"):
                _motion_load_nn_checkpoint(v)
                if v.motion_nn_model is not None and v.motion_use_nn:
                    _motion_apply_nn_deformation(v, v.motion_current_frame)
            _, v.motion_nn_error_heatmap = imgui.checkbox(
                "Error Heatmap (NN vs GT)##motion_heatmap", v.motion_nn_error_heatmap)


def _render_inspect_2d_windows(v):
    """Render 2D inspection windows for fiber samples and contour waypoints."""
    muscles_to_close = []

    for name, is_open in list(v.inspect_2d_open.items()):
        if not is_open:
            continue

        if name not in v.zygote_muscle_meshes:
            muscles_to_close.append(name)
            continue

        obj = v.zygote_muscle_meshes[name]

        # Check if data is available (only contours required, fiber_architecture optional)
        if (not hasattr(obj, 'contours') or obj.contours is None or len(obj.contours) == 0):
            muscles_to_close.append(name)
            continue

        # Check if fiber_architecture is available (optional - for fiber sample display)
        has_fiber = (hasattr(obj, 'fiber_architecture') and obj.fiber_architecture is not None and
                    len(obj.fiber_architecture) > 0)

        # Window setup - size to fit two 280px canvases with padding
        # Width: 2 * (280 + 2*20 + 20) + window padding = ~720
        # Height: sliders(~75) + labels(~40) + canvas(280+2*20) + margins = ~450
        imgui.set_next_window_size(720, 480, imgui.FIRST_USE_EVER)
        expanded, opened = imgui.begin(f"Inspect 2D: {name}", True)

        if not opened:
            muscles_to_close.append(name)
            imgui.end()
            continue

        # Detect data structure format:
        # - Pre-stream (before cutting): contours[level_idx][stream_idx], bounding_planes[level_idx][stream_idx]
        # - Post-stream (after build_fibers): contours[stream_idx][level_idx], bounding_planes[stream_idx][level_idx]
        #
        # Detection: If stream_contours exists and has data, we're in post-stream mode.
        # Also check if contours structure matches stream_contours (indicating build_fibers was called).
        has_stream_contours = (hasattr(obj, 'stream_contours') and
                               obj.stream_contours is not None and
                               len(obj.stream_contours) > 0)

        # If contours == stream_contours (same object or same structure), it's post-stream
        is_post_stream = False
        if has_stream_contours:
            # Check if contours is the same as stream_contours (build_fibers assigns directly)
            if obj.contours is obj.stream_contours:
                is_post_stream = True
            # Also check if structure matches: outer dim is small (num streams), inner is larger (num levels)
            elif len(obj.contours) > 0 and len(obj.contours) <= 10:  # Typically few streams
                # In post-stream, contours[stream][level], so inner should have many elements
                if isinstance(obj.contours[0], (list, np.ndarray)) and len(obj.contours[0]) > 0:
                    # Check if inner elements are contour arrays (have 3D points)
                    inner = obj.contours[0][0]
                    if isinstance(inner, np.ndarray) and inner.ndim == 2 and inner.shape[1] == 3:
                        is_post_stream = True

        is_pre_stream = not is_post_stream
        inspect_contours = getattr(obj, '_tendon_extended_inspect_contours', None)
        inspect_planes = getattr(obj, '_tendon_extended_inspect_bounding_planes', None)
        inspect_waypoints = getattr(obj, '_tendon_extended_inspect_waypoints', None)
        use_tendon_inspect = (
            is_post_stream
            and getattr(obj, 'tendon_extended_fibers', False)
            and inspect_contours is not None
            and inspect_planes is not None
        )

        # Helper functions to access data in correct format
        def get_num_levels():
            """Get number of levels for the current stream."""
            if is_pre_stream:
                return len(obj.contours)
            else:
                # Post-stream: use current stream's level count (streams may differ)
                s = v.inspect_2d_stream_idx.get(name, 0)
                if use_tendon_inspect and s < len(inspect_contours):
                    return len(inspect_contours[s])
                if s < len(obj.contours):
                    return len(obj.contours[s])
                elif len(obj.contours) > 0:
                    return max(len(stream) for stream in obj.contours)
                return 0

        def get_max_contours_at_level():
            """Get maximum number of contours at any level (for pre-stream)."""
            if is_pre_stream:
                if len(obj.contours) == 0:
                    return 0
                return max(len(level) for level in obj.contours)
            else:
                # Post-stream: streams are the outer index
                if use_tendon_inspect:
                    return len(inspect_contours)
                return len(obj.contours)

        def get_num_contours_at_level(level_idx):
            """Get number of contours at a specific level (for pre-stream)."""
            if is_pre_stream:
                if level_idx < len(obj.contours):
                    return len(obj.contours[level_idx])
                return 0
            else:
                if use_tendon_inspect:
                    return len(inspect_contours)
                return len(obj.contours)

        def get_contour(s_idx, level_idx):
            if is_pre_stream:
                # contours[level_idx][contour_idx] - s_idx is contour within level
                if level_idx < len(obj.contours) and s_idx < len(obj.contours[level_idx]):
                    return obj.contours[level_idx][s_idx]
                return None
            else:
                # contours[stream_idx][level_idx]
                if use_tendon_inspect:
                    if s_idx < len(inspect_contours) and level_idx < len(inspect_contours[s_idx]):
                        return inspect_contours[s_idx][level_idx]
                    return None
                if s_idx < len(obj.contours) and level_idx < len(obj.contours[s_idx]):
                    return obj.contours[s_idx][level_idx]
                return None

        def get_bounding_plane(s_idx, level_idx):
            if is_pre_stream:
                # bounding_planes[level_idx][contour_idx] - s_idx is contour within level
                if level_idx < len(obj.bounding_planes) and s_idx < len(obj.bounding_planes[level_idx]):
                    return obj.bounding_planes[level_idx][s_idx]
                return None
            else:
                # bounding_planes[stream_idx][level_idx]
                if use_tendon_inspect:
                    if s_idx < len(inspect_planes) and level_idx < len(inspect_planes[s_idx]):
                        return inspect_planes[s_idx][level_idx]
                    return None
                if s_idx < len(obj.bounding_planes) and level_idx < len(obj.bounding_planes[s_idx]):
                    return obj.bounding_planes[s_idx][level_idx]
                return None

        def get_waypoints_for_level(s_idx, level_idx):
            if use_tendon_inspect and inspect_waypoints is not None:
                if s_idx < len(inspect_waypoints) and level_idx < len(inspect_waypoints[s_idx]):
                    return inspect_waypoints[s_idx][level_idx]
                return None
            if (hasattr(obj, 'waypoints') and obj.waypoints is not None
                    and s_idx < len(obj.waypoints)
                    and level_idx < len(obj.waypoints[s_idx])):
                return obj.waypoints[s_idx][level_idx]
            return None

        # Different UI for pre-stream vs post-stream
        if is_pre_stream:
            # Pre-stream: Level slider (contour) + Contour-within-level slider (stream)
            num_levels = get_num_levels()
            max_contours = get_max_contours_at_level()

            # Level slider (labeled as "Level")
            level_idx = v.inspect_2d_contour_idx.get(name, 0)
            level_idx = min(level_idx, max(0, num_levels - 1))

            changed, new_level_idx = imgui.slider_int(f"Level##{name}_inspect", level_idx, 0, max(0, num_levels - 1))
            if changed:
                v.inspect_2d_contour_idx[name] = new_level_idx
                level_idx = new_level_idx

            # Get number of contours at current level for slider boundary
            num_at_level = get_num_contours_at_level(level_idx)

            # Contour-within-level slider (labeled as "Contour")
            # Slider max is based on contours at this level, not global max
            contour_in_level = v.inspect_2d_stream_idx.get(name, 0)
            contour_in_level = min(contour_in_level, max(0, num_at_level - 1))

            changed, new_contour_in_level = imgui.slider_int(f"Contour##{name}_inspect", contour_in_level, 0, max(0, num_at_level - 1))
            if changed:
                v.inspect_2d_stream_idx[name] = new_contour_in_level
                contour_in_level = new_contour_in_level

            # Show info about contours at this level
            imgui.text(f"Level {level_idx}: {num_at_level} contour(s)")

            imgui.separator()

            # Determine if we should show anything
            # Show nothing if contour_in_level >= num_at_level
            if contour_in_level < num_at_level:
                contour_indices = [(contour_in_level, level_idx)]  # (stream_idx, level_idx)
            else:
                contour_indices = []  # Show nothing

            # Pre-stream doesn't have show_all feature
            show_all = False

        else:
            # Post-stream: Stream slider + Contour (level) slider
            num_streams = get_max_contours_at_level()
            stream_idx = v.inspect_2d_stream_idx.get(name, 0)
            stream_idx = min(stream_idx, max(0, num_streams - 1))

            changed, new_stream_idx = imgui.slider_int(f"Stream##{name}_inspect", stream_idx, 0, max(0, num_streams - 1))
            if changed:
                v.inspect_2d_stream_idx[name] = new_stream_idx
                stream_idx = new_stream_idx

            num_levels = get_num_levels()
            contour_idx = v.inspect_2d_contour_idx.get(name, 0)
            contour_idx = min(contour_idx, max(0, num_levels - 1))

            # Show All checkbox
            if not hasattr(v, 'inspect_2d_show_all'):
                v.inspect_2d_show_all = {}
            show_all = v.inspect_2d_show_all.get(name, False)
            changed_show_all, show_all = imgui.checkbox(f"Show All##{name}", show_all)
            if changed_show_all:
                v.inspect_2d_show_all[name] = show_all
                # Exit Edit Fiber mode when switching to "Show All"
                if show_all and name in v.inspect_2d_edit_fiber_mode and v.inspect_2d_edit_fiber_mode[name]:
                    v.inspect_2d_edit_fiber_mode[name] = False
                    v.inspect_2d_edit_fiber_selected[name] = -1
                    v.inspect_2d_edit_fiber_preview[name] = None
                    v.inspect_2d_edit_fiber_test[name] = None
                    # Clear test fiber from object
                    obj.test_fiber_waypoints = None
                    obj.test_fiber_stream_idx = None

            if not show_all:
                changed, new_contour_idx = imgui.slider_int(f"Contour##{name}_inspect", contour_idx, 0, max(0, num_levels - 1))
                if changed:
                    v.inspect_2d_contour_idx[name] = new_contour_idx
                    contour_idx = new_contour_idx

            imgui.separator()

            # Grid fiber resample: slider (1..10) + Apply
            if getattr(obj, 'sampling_method', None) == 'grid':
                if name not in v.inspect_2d_grid_n:
                    current_fa = getattr(obj, 'fiber_architecture', None)
                    if current_fa and len(current_fa) > 0:
                        guess_n = int(round(np.sqrt(len(current_fa[0]))))
                        v.inspect_2d_grid_n[name] = max(1, min(10, guess_n))
                    else:
                        v.inspect_2d_grid_n[name] = 5
                changed_n, new_n = imgui.slider_int(
                    f"Grid N##{name}_resample",
                    v.inspect_2d_grid_n[name], 1, 10,
                )
                if changed_n:
                    v.inspect_2d_grid_n[name] = new_n
                imgui.same_line()
                if imgui.button(f"Apply##{name}_resample"):
                    obj.resample_grid_fibers(v.inspect_2d_grid_n[name])
                imgui.separator()
            else:
                imgui.text(f"Fiber resample: sampling_method={getattr(obj, 'sampling_method', '?')} (grid only)")
                imgui.separator()

            # Determine which contours to draw
            if show_all:
                contour_indices = [(stream_idx, i) for i in range(num_levels)]
            else:
                contour_indices = [(stream_idx, contour_idx)]

        # Always use child region for consistent layout (scrollbar space reserved)
        imgui.begin_child(f"contours_scroll##{name}", 0, 0, border=False)

        # Initialize correspondence mode state for this muscle
        if name not in v.inspect_2d_corr_mode:
            v.inspect_2d_corr_mode[name] = False
            v.inspect_2d_corr_corner[name] = -1
            v.inspect_2d_corr_vertex[name] = -1

        corr_mode = v.inspect_2d_corr_mode[name]
        corr_corner = v.inspect_2d_corr_corner[name]
        corr_vertex = v.inspect_2d_corr_vertex[name]

        # Initialize edit fiber mode state for this muscle
        if name not in v.inspect_2d_edit_fiber_mode:
            v.inspect_2d_edit_fiber_mode[name] = False
            v.inspect_2d_edit_fiber_selected[name] = -1
            v.inspect_2d_edit_fiber_preview[name] = None
            v.inspect_2d_edit_fiber_test[name] = None

        edit_fiber_mode = v.inspect_2d_edit_fiber_mode[name]
        edit_fiber_selected = v.inspect_2d_edit_fiber_selected[name]
        edit_fiber_preview = v.inspect_2d_edit_fiber_preview[name]
        edit_fiber_test = v.inspect_2d_edit_fiber_test[name]

        # Set inspector highlight on the object for 3D visualization
        # This will highlight the contour being inspected in the 3D view
        if is_post_stream:
            highlight_stream = v.inspect_2d_stream_idx.get(name, 0)
            highlight_level = v.inspect_2d_contour_idx.get(name, 0)
        else:
            highlight_stream = v.inspect_2d_stream_idx.get(name, 0)
            highlight_level = v.inspect_2d_contour_idx.get(name, 0)
        obj.inspector_highlight_stream = highlight_stream
        obj.inspector_highlight_level = highlight_level

        # obj.contours (drawn/highlighted in 3D) is belly-only, but the inspect
        # slider counts extended levels (origin tendon + belly + insertion tendon)
        # after tendon extension. Map the inspect level into belly-contour space
        # so the colored contour matches the highlighted waypoints. On a tendon
        # level there is no belly contour to color.
        contour_highlight_level = highlight_level
        if use_tendon_inspect:
            contour_highlight_level = None
            regions = getattr(obj, 'waypoint_level_regions', None)
            if regions is not None and highlight_stream < len(regions):
                stream_regions = regions[highlight_stream]
                if 0 <= highlight_level < len(stream_regions):
                    if stream_regions[highlight_level].get('part') == 'belly':
                        origin_count = sum(
                            1 for r in stream_regions
                            if r.get('part') == 'origin_tendon')
                        contour_highlight_level = highlight_level - origin_count
        obj.inspector_highlight_contour_level = contour_highlight_level

        # Initialize hover state (may not be set if contour_indices is empty)
        hovered_idx = -1
        hovered_type = None

        for stream_idx, level_idx in contour_indices:
            contour_idx = level_idx  # For display purposes

            imgui.text(f"=== Level {level_idx}, Contour {stream_idx} ===")

            # Show bounding plane corner angles (debug info)
            bp_info = get_bounding_plane(stream_idx, level_idx)
            if bp_info is not None:
                bp_corners = bp_info.get('bounding_plane', None)
                if bp_corners is not None and len(bp_corners) >= 4:
                    # Compute angles at each corner
                    angles = []
                    for i in range(4):
                        p0 = np.array(bp_corners[(i - 1) % 4])
                        p1 = np.array(bp_corners[i])
                        p2 = np.array(bp_corners[(i + 1) % 4])
                        v1 = p0 - p1
                        v2 = p2 - p1
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                        angles.append(angle_deg)
                    imgui.text(f"BP Angles: {angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f}, {angles[3]:.1f}")

            imgui.separator()

            # Get canvas dimensions
            canvas_size = 280
            padding = 20
            column_width = canvas_size + 2 * padding + 20  # Fixed column width

            # Two columns: unit square (left) and contour (right)
            imgui.columns(2, f"inspect_cols##{name}_{level_idx}_{stream_idx}", border=True)
            imgui.set_column_width(0, column_width)
            imgui.set_column_width(1, column_width)

            draw_list = imgui.get_window_draw_list()
            mouse_pos = imgui.get_mouse_pos()
            hovered_idx = -1
            hovered_type = None  # 'vertex', 'fiber', or 'waypoint'
            hover_radius = 8.0

            # Get contour data first (needed for both columns)
            contour_match = None
            plane_info = None
            mean = None
            basis_x = None
            basis_y = None
            bp = None
            p_screen_points = []
            q_screen_points = []
            fiber_screen_points = []
            waypoint_screen_points = []
            corner_screen_points_left = []  # Bounding plane corners on unit square
            corner_screen_points_right = []  # Bounding plane corners on contour
            corner_to_closest_vertex = []  # (corner_idx, closest_vertex_idx) pairs
            contour_2d_norm = None

            # Helper function to compute normalized [0,1] coordinates by solving linear system
            def point_to_unit_square_2d(point_3d, mean, basis_x, basis_y, bp_corners):
                """Convert 3D point to [0,1]x[0,1] by inverting the bounding plane formula.

                Q was created as: Q = bp[0] + u * (bp[1]-bp[0]) + v * (bp[3]-bp[0])
                So we solve: Q - bp[0] = u * edge_x + v * edge_y
                This gives proper [0,1] coordinates regardless of basis orthogonality.
                """
                v0 = bp_corners[0]
                edge_x = bp_corners[1] - bp_corners[0]  # horizontal edge
                edge_y = bp_corners[3] - bp_corners[0]  # vertical edge

                # Solve 2x2 system using least squares (works in 3D)
                # [edge_x | edge_y] * [u; v] = point - v0
                rel_p = point_3d - v0

                # Build matrix A = [edge_x, edge_y] as columns (3x2)
                A = np.column_stack([edge_x, edge_y])

                # Solve using least squares
                result, _, _, _ = np.linalg.lstsq(A, rel_p, rcond=None)
                u, v = result[0], result[1]

                return np.array([np.clip(u, 0, 1), np.clip(v, 0, 1)])

            plane_info = get_bounding_plane(stream_idx, level_idx)
            if plane_info is not None:
                contour_match = plane_info.get('contour_match', None)
                # For an extended tendon/belly stream, this is the registered
                # seam chart produced by the extension. Re-running
                # find_contour_match here independently per level discards that
                # registration and makes the inspector show the old mismatch.

                if contour_match is not None and len(contour_match) > 0 and 'basis_x' in plane_info:
                    mean = plane_info['mean']
                    basis_x = plane_info['basis_x']
                    basis_y = plane_info['basis_y']
                    bp = plane_info.get('bounding_plane', None)

            # Display rotation helper for unit-square coords
            _display_rot = v.inspect_2d_display_rot.get((name, level_idx), 0)
            def _rot_uv(u, v):
                """Rotate (u,v) in unit square by _display_rot * 90° CCW."""
                for _ in range(_display_rot % 4):
                    u, v = v, 1 - u  # CCW 90°
                return u, v

            # Left column: Unit square with Q points
            imgui.text("Unit Square (Q points)")
            cursor_pos_left = imgui.get_cursor_screen_pos()

            left_x0, left_y0 = cursor_pos_left[0] + padding, cursor_pos_left[1] + padding
            left_x1, left_y1 = left_x0 + canvas_size, left_y0 + canvas_size

            # Background
            draw_list.add_rect_filled(left_x0, left_y0, left_x1, left_y1, imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 1.0))
            draw_list.add_rect(left_x0, left_y0, left_x1, left_y1, imgui.get_color_u32_rgba(0.5, 0.5, 0.5, 1.0), thickness=2.0)

            # Draw the unit-square Voronoi partition behind the fiber samples.
            # It is derived from the live fiber architecture, so Grid Apply
            # automatically selects/recomputes the matching cached partition.
            fiber_samples = []
            if has_fiber and stream_idx < len(obj.fiber_architecture):
                fiber_samples = obj.fiber_architecture[stream_idx]
                voronoi_cells = _unit_square_voronoi_cells(fiber_samples)
                voronoi_fill = imgui.get_color_u32_rgba(0.10, 0.32, 0.22, 0.28)
                voronoi_edge = imgui.get_color_u32_rgba(0.30, 0.72, 0.48, 0.85)
                for cell in voronoi_cells:
                    if len(cell) < 3:
                        continue
                    screen_cell = []
                    for cell_u, cell_v in cell:
                        cell_u, cell_v = _rot_uv(cell_u, cell_v)
                        screen_cell.append((
                            left_x0 + cell_u * canvas_size,
                            left_y0 + (1 - cell_v) * canvas_size,
                        ))
                    draw_list.path_clear()
                    for cell_x, cell_y in screen_cell:
                        draw_list.path_line_to(cell_x, cell_y)
                    draw_list.path_fill_convex(voronoi_fill)
                    draw_list.path_clear()
                    for cell_x, cell_y in screen_cell:
                        draw_list.path_line_to(cell_x, cell_y)
                    draw_list.path_stroke(
                        voronoi_edge, flags=imgui.DRAW_CLOSED, thickness=1.0)

                # Draw fiber samples (green) over their Voronoi cells and check hover.
                for i, sample in enumerate(fiber_samples):
                    if len(sample) >= 2:
                        _fu, _fv = _rot_uv(sample[0], sample[1])
                        sx = left_x0 + _fu * canvas_size
                        sy = left_y0 + (1 - _fv) * canvas_size
                        fiber_screen_points.append((sx, sy))
                        draw_list.add_circle_filled(sx, sy, 4, imgui.get_color_u32_rgba(0.2, 0.8, 0.2, 1.0))
                        # Check hover on fiber samples
                        dist = np.sqrt((mouse_pos[0] - sx)**2 + (mouse_pos[1] - sy)**2)
                        if dist < hover_radius and hovered_idx < 0:
                            hovered_idx = i
                            hovered_type = 'fiber'

            # Compute and draw Q points from contour_match (cyan) - draw immediately so they're not covered
            # Use bounding plane parametric coordinates to match MVC computation in find_waypoints()
            _best_q_dist = hover_radius
            if contour_match is not None and bp is not None and len(bp) >= 4:
                for i, (p, q) in enumerate(contour_match):
                    p = np.array(p)
                    q = np.array(q)
                    # Compute Q's position on unit square using bounding plane parametric coords
                    q_norm = point_to_unit_square_2d(q, mean, basis_x, basis_y, bp)
                    _qu, _qv = _rot_uv(q_norm[0], q_norm[1])
                    qx = left_x0 + _qu * canvas_size
                    qy = left_y0 + (1 - _qv) * canvas_size
                    q_screen_points.append((qx, qy))
                    # Draw Q point immediately (cyan)
                    draw_list.add_circle_filled(qx, qy, 3, imgui.get_color_u32_rgba(0.0, 0.8, 0.8, 1.0))

                    # Check hover on Q points (pick closest, not first)
                    dist = np.sqrt((mouse_pos[0] - qx)**2 + (mouse_pos[1] - qy)**2)
                    if dist < hover_radius:
                        if hovered_idx < 0 or dist < _best_q_dist:
                            _best_q_dist = dist
                            hovered_idx = i
                            hovered_type = 'vertex'

                # Draw bounding plane corners on unit square (corners are at (0,0), (1,0), (1,1), (0,1))
                corner_uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
                for ci, (cu, cv) in enumerate(corner_uv):
                    cu, cv = _rot_uv(cu, cv)
                    cx = left_x0 + cu * canvas_size
                    cy = left_y0 + (1 - cv) * canvas_size
                    corner_screen_points_left.append((cx, cy))
                    # Draw corner (purple/magenta diamond)
                    draw_list.add_quad_filled(cx, cy - 5, cx + 5, cy, cx, cy + 5, cx - 5, cy,
                                             imgui.get_color_u32_rgba(0.8, 0.2, 0.8, 1.0))
                    # Check hover on corners
                    dist = np.sqrt((mouse_pos[0] - cx)**2 + (mouse_pos[1] - cy)**2)
                    if dist < hover_radius and hovered_idx < 0:
                        hovered_idx = ci
                        hovered_type = 'corner'

            # Reserve space
            imgui.dummy(canvas_size + 2 * padding, canvas_size + 2 * padding)

            # Right column: Contour with P points
            imgui.next_column()

            # Rotate display CW/CCW buttons (visual only) + Reset correspondence
            _rot_key = (name, level_idx)
            _display_rot = v.inspect_2d_display_rot.get(_rot_key, 0)
            if imgui.button(f"CCW##{name}_{level_idx}_{stream_idx}"):
                v.inspect_2d_display_rot[_rot_key] = (_display_rot + 1) % 4
            imgui.same_line()
            if imgui.button(f"CW##{name}_{level_idx}_{stream_idx}"):
                v.inspect_2d_display_rot[_rot_key] = (_display_rot - 1) % 4
            imgui.same_line()
            if imgui.button(f"Reset##{name}_{level_idx}_{stream_idx}"):
                _reset_corner_correspondence(obj, stream_idx, level_idx, is_post_stream)
            if _display_rot != 0:
                imgui.same_line()
                imgui.text(f"(rot {_display_rot * 90})")

            imgui.text(f"Level {level_idx} (P points)")
            cursor_pos_right = imgui.get_cursor_screen_pos()

            right_x0, right_y0 = cursor_pos_right[0] + padding, cursor_pos_right[1] + padding
            right_x1, right_y1 = right_x0 + canvas_size, right_y0 + canvas_size

            # Background
            draw_list.add_rect_filled(right_x0, right_y0, right_x1, right_y1, imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 1.0))

            # Draw contour and P points
            if contour_match is not None and 'basis_x' in plane_info:
                # Apply display rotation to basis (visual only, data unchanged)
                _display_rot = v.inspect_2d_display_rot.get((name, level_idx), 0)
                _dbx, _dby = basis_x, basis_y
                for _ in range(_display_rot % 4):
                    _dbx, _dby = _dby, -_dbx  # CCW 90°

                # Project P points to 2D using display-rotated basis
                p_2d_list = []
                for p, q in contour_match:
                    p = np.array(p)
                    p_2d = np.array([np.dot(p - mean, _dbx), np.dot(p - mean, _dby)])
                    p_2d_list.append(p_2d)
                p_2d_arr = np.array(p_2d_list)

                # Per-axis normalization (unit-square ratio)
                min_xy = p_2d_arr.min(axis=0)
                max_xy = p_2d_arr.max(axis=0)
                range_xy = max_xy - min_xy
                range_xy[range_xy < 1e-10] = 1.0
                margin = 0.02
                scale_xy = (1 - 2 * margin) / range_xy
                center_xy = (min_xy + max_xy) / 2

                def _p2d_to_screen(pt):
                    nx = (pt[0] - center_xy[0]) * scale_xy[0] + 0.5
                    ny = (pt[1] - center_xy[1]) * scale_xy[1] + 0.5
                    return (right_x0 + nx * canvas_size, right_y0 + (1 - ny) * canvas_size)

                p_screen_points = [_p2d_to_screen(p_2d) for p_2d in p_2d_list]

                # Transfer the shared unit-square Voronoi cells through the
                # same MVC chart used by the fiber waypoints.  A triangle fan
                # supports concave mapped cells; locally inverted triangles are
                # colored red so mapping defects remain visible in Inspect 2D.
                if has_fiber and stream_idx < len(obj.fiber_architecture):
                    right_samples = obj.fiber_architecture[stream_idx]
                    right_cells_uv = _unit_square_voronoi_cells(right_samples)
                    right_cells_3d = _map_voronoi_cells_to_contour(
                        obj, plane_info, right_samples, right_cells_uv)
                    valid_fill = imgui.get_color_u32_rgba(0.10, 0.32, 0.22, 0.25)
                    valid_edge = imgui.get_color_u32_rgba(0.30, 0.72, 0.48, 0.82)
                    flipped_fill = imgui.get_color_u32_rgba(0.75, 0.12, 0.12, 0.34)
                    flipped_edge = imgui.get_color_u32_rgba(0.95, 0.25, 0.20, 0.90)

                    for fiber_i, (cell_uv, cell_3d) in enumerate(
                            zip(right_cells_uv, right_cells_3d)):
                        if cell_3d is None or len(cell_3d) < 4:
                            continue
                        mapped_2d = [
                            np.array([
                                np.dot(point - mean, _dbx),
                                np.dot(point - mean, _dby),
                            ])
                            for point in cell_3d
                        ]
                        mapped_screen = [_p2d_to_screen(point) for point in mapped_2d]
                        # Use the actual fiber site, not the cell centroid, as
                        # the parameter-space triangle-fan center.
                        # cell_3d[0] was generated from the corresponding site.
                        site_uv = np.asarray(right_samples[fiber_i])[:2]

                        any_flipped = False
                        boundary_count = len(cell_uv)
                        for boundary_i in range(boundary_count):
                            next_i = (boundary_i + 1) % boundary_count
                            uv_a = np.asarray(cell_uv[boundary_i]) - site_uv
                            uv_b = np.asarray(cell_uv[next_i]) - site_uv
                            uv_cross = uv_a[0] * uv_b[1] - uv_a[1] * uv_b[0]
                            mapped_a = mapped_2d[1 + boundary_i] - mapped_2d[0]
                            mapped_b = mapped_2d[1 + next_i] - mapped_2d[0]
                            mapped_cross = (mapped_a[0] * mapped_b[1]
                                            - mapped_a[1] * mapped_b[0])
                            flipped = uv_cross * mapped_cross < -1e-12
                            any_flipped = any_flipped or flipped
                            center_screen = mapped_screen[0]
                            a_screen = mapped_screen[1 + boundary_i]
                            b_screen = mapped_screen[1 + next_i]
                            draw_list.add_triangle_filled(
                                center_screen[0], center_screen[1],
                                a_screen[0], a_screen[1],
                                b_screen[0], b_screen[1],
                                flipped_fill if flipped else valid_fill)

                        edge_color = flipped_edge if any_flipped else valid_edge
                        for boundary_i in range(boundary_count):
                            a_screen = mapped_screen[1 + boundary_i]
                            b_screen = mapped_screen[1 + ((boundary_i + 1) % boundary_count)]
                            draw_list.add_line(
                                a_screen[0], a_screen[1], b_screen[0], b_screen[1],
                                edge_color, 1.0)

                # Draw contour lines (yellow)
                for i in range(len(p_screen_points)):
                    p1 = p_screen_points[i]
                    p2 = p_screen_points[(i + 1) % len(p_screen_points)]
                    draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                      imgui.get_color_u32_rgba(0.8, 0.8, 0.2, 1.0), thickness=2.0)

                # Check hover on P points (pick closest, only if mouse is in right canvas)
                if (right_x0 - hover_radius <= mouse_pos[0] <= right_x1 + hover_radius and
                    right_y0 - hover_radius <= mouse_pos[1] <= right_y1 + hover_radius):
                    _best_p_dist = float('inf')
                    _best_p_idx = -1
                    for i, (px, py) in enumerate(p_screen_points):
                        dist = np.sqrt((mouse_pos[0] - px)**2 + (mouse_pos[1] - py)**2)
                        if dist < _best_p_dist:
                            _best_p_dist = dist
                            _best_p_idx = i
                    if _best_p_idx >= 0 and _best_p_dist < hover_radius:
                        hovered_idx = _best_p_idx
                        hovered_type = 'vertex'

                # Draw bounding plane (blue). If bp['bounding_plane'] corners are
                # inconsistent with bp['mean']/basis (happens after some load_anim
                # paths where the corner array was carried over from a different
                # cell of bounding_planes), the projected square explodes far off
                # the contour. Detect that case and re-derive corners from the
                # contour's projected extent so the square stays sane.
                if bp is not None and len(bp) >= 4:
                    bp_corners_2d = []
                    for corner_3d in bp[:4]:
                        corner_3d = np.array(corner_3d)
                        bp_corners_2d.append(np.array([
                            np.dot(corner_3d - mean, _dbx),
                            np.dot(corner_3d - mean, _dby),
                        ]))
                    bp_corners_2d_arr = np.array(bp_corners_2d)

                    # Sanity: BP should roughly enclose contour P points. If its
                    # extent is much bigger than P extent OR centroid is far from
                    # P centroid, treat corners as stale.
                    p_min = p_2d_arr.min(axis=0)
                    p_max = p_2d_arr.max(axis=0)
                    p_range = p_max - p_min
                    p_center = (p_min + p_max) / 2
                    bp_min = bp_corners_2d_arr.min(axis=0)
                    bp_max = bp_corners_2d_arr.max(axis=0)
                    bp_range = bp_max - bp_min
                    bp_center = (bp_min + bp_max) / 2
                    p_diag = float(np.linalg.norm(p_range)) or 1.0
                    extent_ratio = float(np.linalg.norm(bp_range)) / p_diag
                    center_drift = float(np.linalg.norm(bp_center - p_center)) / p_diag
                    bp_stale = extent_ratio > 5.0 or center_drift > 2.0
                    if bp_stale:
                        # Fallback: derive a rectangle aligned with (basis_x, basis_y)
                        # frame, sized to enclose contour with 5% margin. Better
                        # than rendering a far-off quad. Order matches BP convention
                        # (0=BL, 1=BR, 2=TR, 3=TL).
                        m = 0.05 * np.maximum(p_range, 1e-6)
                        bl = p_min - m
                        tr = p_max + m
                        bp_corners_2d_arr = np.array([
                            [bl[0], bl[1]],
                            [tr[0], bl[1]],
                            [tr[0], tr[1]],
                            [bl[0], tr[1]],
                        ])

                    bp_screen = [_p2d_to_screen(c) for c in bp_corners_2d_arr]
                    for i in range(4):
                        p1 = bp_screen[i]
                        p2 = bp_screen[(i + 1) % 4]
                        draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                          imgui.get_color_u32_rgba(0.3, 0.5, 0.9, 1.0), thickness=1.5)

                # Draw bounding plane corners and lines to closest contour vertices
                if bp is not None and len(bp) >= 4:
                    # Find corner-vertex correspondence based on Q positions (not ray-based)
                    # This reflects the actual correspondence used in fiber computation
                    bp_corners_3d = [np.array(c) for c in bp[:4]]
                    q_based_corner_indices = []
                    for ci, corner_3d in enumerate(bp_corners_3d):
                        # Find vertex whose Q is closest to this corner
                        min_dist = float('inf')
                        closest_vi = 0
                        for vi, (p, q) in enumerate(contour_match):
                            q_arr = np.array(q)
                            dist = np.linalg.norm(q_arr - corner_3d)
                            if dist < min_dist:
                                min_dist = dist
                                closest_vi = vi
                        q_based_corner_indices.append(closest_vi)

                    for ci, corner_3d in enumerate(bp[:4]):
                        corner_3d = np.array(corner_3d)
                        corner_2d = np.array([np.dot(corner_3d - mean, _dbx), np.dot(corner_3d - mean, _dby)])
                        cx, cy = _p2d_to_screen(corner_2d)
                        corner_screen_points_right.append((cx, cy))

                        # Use stored corner_indices if available, else Q-based fallback
                        stored_ci = plane_info.get('corner_indices') if plane_info else None
                        if stored_ci is not None and ci < len(stored_ci):
                            closest_vi = stored_ci[ci]
                        elif ci < len(q_based_corner_indices):
                            closest_vi = q_based_corner_indices[ci]
                        else:
                            closest_vi = -1

                        if closest_vi >= 0 and len(p_screen_points) > 0:
                            corner_to_closest_vertex.append((ci, closest_vi))

                            # Draw line from corner to corresponding vertex (purple, thin)
                            if closest_vi < len(p_screen_points):
                                px, py = p_screen_points[closest_vi]
                                draw_list.add_line(cx, cy, px, py,
                                                 imgui.get_color_u32_rgba(0.7, 0.3, 0.7, 0.6), thickness=1.0)

                        # Draw corner (purple/magenta diamond)
                        draw_list.add_quad_filled(cx, cy - 5, cx + 5, cy, cx, cy + 5, cx - 5, cy,
                                                 imgui.get_color_u32_rgba(0.8, 0.2, 0.8, 1.0))
                        # Corner index label
                        draw_list.add_text(cx + 7, cy - 7, imgui.get_color_u32_rgba(0.8, 0.2, 0.8, 1.0), str(ci))

                        # Check hover on corners (right side). Corners normally
                        # take priority over vertices, but in correspondence
                        # mode the user is trying to PICK a vertex to assign to
                        # the already-selected corner — so a near-corner vertex
                        # must remain selectable. Skip the override while
                        # corr_mode is active so vertex hover (set earlier) wins.
                        if not corr_mode:
                            dist = np.sqrt((mouse_pos[0] - cx)**2 + (mouse_pos[1] - cy)**2)
                            if dist < hover_radius:
                                hovered_idx = ci
                                hovered_type = 'corner'

                    # Show corner-vertex mapping as text
                    imgui.text(f"Corners: {q_based_corner_indices}")

                # Draw waypoints (red) and check hover
                waypoints_3d = get_waypoints_for_level(stream_idx, level_idx)
                # Extended waypoints were generated from the registered chart;
                # display them directly rather than independently regenerating
                # another coordinate field during rendering.
                if waypoints_3d is not None:
                    if waypoints_3d is not None and len(waypoints_3d) > 0:
                        for wi, wp in enumerate(waypoints_3d):
                            wp = np.array(wp)
                            wp_2d = np.array([np.dot(wp - mean, _dbx), np.dot(wp - mean, _dby)])
                            wpx, wpy = _p2d_to_screen(wp_2d)
                            waypoint_screen_points.append((wpx, wpy))
                            draw_list.add_circle_filled(wpx, wpy, 5, imgui.get_color_u32_rgba(0.9, 0.3, 0.3, 1.0))
                            # Check hover on waypoints
                            dist = np.sqrt((mouse_pos[0] - wpx)**2 + (mouse_pos[1] - wpy)**2)
                            if dist < hover_radius and hovered_idx < 0:
                                hovered_idx = wi
                                hovered_type = 'waypoint'

            # Draw P vertex points (non-highlighted)
            for i in range(len(p_screen_points)):
                px, py = p_screen_points[i]
                if not (hovered_type == 'vertex' and i == hovered_idx):
                    draw_list.add_circle_filled(px, py, 3, imgui.get_color_u32_rgba(1.0, 0.5, 0.0, 1.0))

            # Draw resampled contour vertices and edges (cyan) if available
            if (not use_tendon_inspect and
                hasattr(obj, 'contours_resampled') and obj.contours_resampled is not None and
                stream_idx < len(obj.contours_resampled) and
                level_idx < len(obj.contours_resampled[stream_idx])):
                resampled = obj.contours_resampled[stream_idx][level_idx]
                if resampled is not None and len(resampled) > 0:
                    # Project resampled vertices to 2D using same transformation
                    resampled_screen = []
                    for rv in resampled:
                        rv = np.array(rv)
                        rv_2d = np.array([np.dot(rv - mean, _dbx), np.dot(rv - mean, _dby)])
                        resampled_screen.append(_p2d_to_screen(rv_2d))
                    # Draw resampled contour edges (cyan)
                    for i in range(len(resampled_screen)):
                        p1 = resampled_screen[i]
                        p2 = resampled_screen[(i + 1) % len(resampled_screen)]
                        draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                          imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 0.9), thickness=1.5)
                    # Draw resampled vertices
                    for rvx, rvy in resampled_screen:
                        draw_list.add_circle_filled(rvx, rvy, 3, imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 1.0))

            # Reserve space
            imgui.dummy(canvas_size + 2 * padding, canvas_size + 2 * padding)

            imgui.columns(1)

            # Draw all highlights AFTER columns are done (ensures they're on top)
            # Re-get draw list to ensure we're drawing on top layer
            draw_list = imgui.get_window_draw_list()

            if hovered_type == 'vertex' and hovered_idx >= 0:
                # Highlight P on contour (orange with white border)
                if hovered_idx < len(p_screen_points):
                    px, py = p_screen_points[hovered_idx]
                    draw_list.add_circle_filled(px, py, 7, imgui.get_color_u32_rgba(1.0, 0.5, 0.0, 1.0))
                    draw_list.add_circle(px, py, 9, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                # Highlight corresponding Q on unit square (magenta with white border)
                if hovered_idx < len(q_screen_points):
                    qx, qy = q_screen_points[hovered_idx]
                    draw_list.add_circle_filled(qx, qy, 7, imgui.get_color_u32_rgba(1.0, 0.0, 1.0, 1.0))
                    draw_list.add_circle(qx, qy, 9, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)

            elif hovered_type == 'fiber' and hovered_idx >= 0:
                # Highlight fiber sample on unit square (bright green with white border)
                if hovered_idx < len(fiber_screen_points):
                    fx, fy = fiber_screen_points[hovered_idx]
                    draw_list.add_circle_filled(fx, fy, 7, imgui.get_color_u32_rgba(0.2, 1.0, 0.2, 1.0))
                    draw_list.add_circle(fx, fy, 9, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                # Highlight corresponding waypoint on contour (bright red with white border)
                if hovered_idx < len(waypoint_screen_points):
                    wpx, wpy = waypoint_screen_points[hovered_idx]
                    draw_list.add_circle_filled(wpx, wpy, 7, imgui.get_color_u32_rgba(1.0, 0.3, 0.3, 1.0))
                    draw_list.add_circle(wpx, wpy, 9, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                # Draw MVC weight-proportional vertices
                mvc_w = None
                if (hasattr(obj, 'mvc_weights') and stream_idx < len(obj.mvc_weights) and
                    level_idx < len(obj.mvc_weights[stream_idx])):
                    mvc_w = obj.mvc_weights[stream_idx][level_idx]
                # Fallback: compute on-the-fly if mvc_weights not available
                if mvc_w is None and contour_match is not None and len(fiber_samples) > 0:
                    try:
                        # Check if contour_match has valid 3D points (not 2D)
                        if len(contour_match) > 0 and len(contour_match[0]) >= 2:
                            first_q = np.array(contour_match[0][1])
                            if len(first_q) == 3:  # Valid 3D point
                                _, _, mvc_w = obj.find_waypoints(plane_info, fiber_samples)
                    except Exception:
                        mvc_w = None
                if mvc_w is not None and len(mvc_w) > hovered_idx:
                    weights = np.array(mvc_w[hovered_idx])
                    if len(weights) > 0 and np.isfinite(weights).all():
                        max_w = weights.max()
                        if max_w > 1e-8:
                            max_radius = 7.0  # Same as hover emphasis
                            # Sort by weight descending (draw largest first, smallest on top)
                            sorted_indices = np.argsort(weights)[::-1]
                            for vi in sorted_indices:
                                w = weights[vi]
                                if w > 1e-8:  # Only draw non-zero weights
                                    rel_size = w / max_w
                                    radius = max(1.0, max_radius * rel_size)  # Minimum radius of 1
                                    # Draw on P (contour) side - yellow
                                    if vi < len(p_screen_points):
                                        px, py = p_screen_points[vi]
                                        draw_list.add_circle_filled(px, py, radius, imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 0.8))
                                    # Draw on Q (unit square) side - yellow
                                    if vi < len(q_screen_points):
                                        qx, qy = q_screen_points[vi]
                                        draw_list.add_circle_filled(qx, qy, radius, imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 0.8))

            elif hovered_type == 'waypoint' and hovered_idx >= 0:
                # Highlight waypoint on contour (bright red with white border)
                if hovered_idx < len(waypoint_screen_points):
                    wpx, wpy = waypoint_screen_points[hovered_idx]
                    draw_list.add_circle_filled(wpx, wpy, 7, imgui.get_color_u32_rgba(1.0, 0.3, 0.3, 1.0))
                    draw_list.add_circle(wpx, wpy, 9, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                # Highlight corresponding fiber sample on unit square (bright green with white border)
                if hovered_idx < len(fiber_screen_points):
                    fx, fy = fiber_screen_points[hovered_idx]
                    draw_list.add_circle_filled(fx, fy, 7, imgui.get_color_u32_rgba(0.2, 1.0, 0.2, 1.0))
                    draw_list.add_circle(fx, fy, 9, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                # Draw MVC weight-proportional vertices
                mvc_w = None
                if (hasattr(obj, 'mvc_weights') and stream_idx < len(obj.mvc_weights) and
                    level_idx < len(obj.mvc_weights[stream_idx])):
                    mvc_w = obj.mvc_weights[stream_idx][level_idx]
                # Fallback: compute on-the-fly if mvc_weights not available
                if mvc_w is None and contour_match is not None and len(fiber_samples) > 0:
                    try:
                        # Check if contour_match has valid 3D points (not 2D)
                        if len(contour_match) > 0 and len(contour_match[0]) >= 2:
                            first_q = np.array(contour_match[0][1])
                            if len(first_q) == 3:  # Valid 3D point
                                _, _, mvc_w = obj.find_waypoints(plane_info, fiber_samples)
                    except Exception:
                        mvc_w = None
                if mvc_w is not None and len(mvc_w) > hovered_idx:
                    weights = np.array(mvc_w[hovered_idx])
                    if len(weights) > 0 and np.isfinite(weights).all():
                        max_w = weights.max()
                        if max_w > 1e-8:
                            max_radius = 7.0  # Same as hover emphasis
                            # Sort by weight descending (draw largest first, smallest on top)
                            sorted_indices = np.argsort(weights)[::-1]
                            for vi in sorted_indices:
                                w = weights[vi]
                                if w > 1e-8:  # Only draw non-zero weights
                                    rel_size = w / max_w
                                    radius = max(1.0, max_radius * rel_size)  # Minimum radius of 1
                                    # Draw on P (contour) side - yellow
                                    if vi < len(p_screen_points):
                                        px, py = p_screen_points[vi]
                                        draw_list.add_circle_filled(px, py, radius, imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 0.8))
                                    # Draw on Q (unit square) side - yellow
                                    if vi < len(q_screen_points):
                                        qx, qy = q_screen_points[vi]
                                        draw_list.add_circle_filled(qx, qy, radius, imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 0.8))

            elif hovered_type == 'corner' and hovered_idx >= 0:
                # Highlight corner on unit square (left side) - bright magenta with white border
                if hovered_idx < len(corner_screen_points_left):
                    cx, cy = corner_screen_points_left[hovered_idx]
                    draw_list.add_quad_filled(cx, cy - 8, cx + 8, cy, cx, cy + 8, cx - 8, cy,
                                             imgui.get_color_u32_rgba(1.0, 0.0, 1.0, 1.0))
                    draw_list.add_quad(cx, cy - 10, cx + 10, cy, cx, cy + 10, cx - 10, cy,
                                      imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                # Highlight corner on contour (right side) - bright magenta with white border
                if hovered_idx < len(corner_screen_points_right):
                    cx, cy = corner_screen_points_right[hovered_idx]
                    draw_list.add_quad_filled(cx, cy - 8, cx + 8, cy, cx, cy + 8, cx - 8, cy,
                                             imgui.get_color_u32_rgba(1.0, 0.0, 1.0, 1.0))
                    draw_list.add_quad(cx, cy - 10, cx + 10, cy, cx, cy + 10, cx - 10, cy,
                                      imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                # Highlight the closest contour vertex and draw emphasized line
                for corner_ci, closest_vi in corner_to_closest_vertex:
                    if corner_ci == hovered_idx:
                        # Draw emphasized line from corner to closest vertex
                        if hovered_idx < len(corner_screen_points_right) and closest_vi < len(p_screen_points):
                            cx, cy = corner_screen_points_right[hovered_idx]
                            px, py = p_screen_points[closest_vi]
                            draw_list.add_line(cx, cy, px, py,
                                             imgui.get_color_u32_rgba(1.0, 0.0, 1.0, 1.0), thickness=3.0)
                        # Highlight the closest P vertex
                        if closest_vi < len(p_screen_points):
                            px, py = p_screen_points[closest_vi]
                            draw_list.add_circle_filled(px, py, 7, imgui.get_color_u32_rgba(1.0, 0.5, 0.0, 1.0))
                            draw_list.add_circle(px, py, 9, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                        # Highlight the corresponding Q vertex
                        if closest_vi < len(q_screen_points):
                            qx, qy = q_screen_points[closest_vi]
                            draw_list.add_circle_filled(qx, qy, 7, imgui.get_color_u32_rgba(0.0, 0.8, 0.8, 1.0))
                            draw_list.add_circle(qx, qy, 9, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                        break

        # Set 3D highlights based on hover.  The pillar is ephemeral and must
        # be explicitly rebuilt by a fiber/waypoint hover every frame.
        obj.inspector_voronoi_pillar_3d = None
        if hovered_type == 'vertex' and hovered_idx >= 0 and contour_match is not None and hovered_idx < len(contour_match):
            obj.inspector_highlight_vertex_3d = np.array(contour_match[hovered_idx][0])
            obj.inspector_highlight_corner_vertices_3d = None
        elif hovered_type == 'corner' and hovered_idx >= 0:
            # Highlight this corner's corresponding vertex at EVERY contour level
            corner_idx = hovered_idx
            corner_pts = []
            other_stream_corner_pts = []  # Same corner on OTHER streams at same level
            if is_post_stream and hasattr(obj, 'bounding_planes') and obj.bounding_planes is not None:
                if stream_idx < len(obj.bounding_planes):
                    for lev_idx, bp_lev in enumerate(obj.bounding_planes[stream_idx]):
                        if bp_lev is None:
                            continue
                        cm = bp_lev.get('contour_match', None)
                        ci = bp_lev.get('corner_indices', None)
                        bp_c = bp_lev.get('bounding_plane', None)
                        if cm is None or bp_c is None:
                            continue
                        # Find corner vertex for this level
                        if ci is not None and corner_idx < len(ci):
                            vi = ci[corner_idx]
                        else:
                            # Fallback: find by Q proximity
                            bp_corner = np.array(bp_c[corner_idx])
                            vi = 0
                            min_d = float('inf')
                            for vvi, (p, q) in enumerate(cm):
                                d = np.linalg.norm(np.array(q) - bp_corner)
                                if d < min_d:
                                    min_d = d
                                    vi = vvi
                        if vi < len(cm):
                            corner_pts.append(np.array(cm[vi][0]))

                # Collect contour vertices from ALL OTHER streams at the current level
                # Use stream_contours (full multi-stream data)
                all_contours_hover = getattr(obj, 'stream_contours', None)
                if all_contours_hover is not None:
                    for other_s in range(len(all_contours_hover)):
                        if other_s == stream_idx:
                            continue
                        if level_idx >= len(all_contours_hover[other_s]):
                            continue
                        c_hover = np.asarray(all_contours_hover[other_s][level_idx])
                        for v_pt in c_hover:
                            other_stream_corner_pts.append(np.array(v_pt))

            # Current level vertex highlighted in main color
            stored_ci_curr = plane_info.get('corner_indices') if plane_info else None
            if contour_match is not None and stored_ci_curr is not None and corner_idx < len(stored_ci_curr):
                vi = stored_ci_curr[corner_idx]
            elif contour_match is not None and len(q_based_corner_indices) > corner_idx:
                vi = q_based_corner_indices[corner_idx]
            else:
                vi = -1
            if vi >= 0 and vi < len(contour_match):
                obj.inspector_highlight_vertex_3d = np.array(contour_match[vi][0])
            obj.inspector_highlight_corner_vertices_3d = corner_pts if corner_pts else None
            obj.inspector_highlight_other_stream_corners_3d = other_stream_corner_pts if other_stream_corner_pts else None
        elif hovered_type in ('fiber', 'waypoint') and hovered_idx >= 0:
            obj.inspector_highlight_fiber_idx = (stream_idx, hovered_idx)
            obj.inspector_highlight_corner_vertices_3d = None
            obj.inspector_highlight_other_stream_corners_3d = None
            obj.inspector_highlight_vertex_3d = None
            if corr_corner < 0:
                obj.inspector_highlight_other_level_contours = None

            # Map the hovered fiber's one labeled unit-square Voronoi cell
            # through every contour level.  Store level indices with rings so
            # the 3D renderer never bridges a missing/invalid contour level.
            obj.inspector_voronoi_pillar_3d = None
            if (has_fiber and stream_idx < len(obj.fiber_architecture)
                    and hovered_idx < len(obj.fiber_architecture[stream_idx])):
                pillar_samples = obj.fiber_architecture[stream_idx]
                pillar_cells = _unit_square_voronoi_cells(pillar_samples)
                if hovered_idx < len(pillar_cells):
                    selected_sample = [pillar_samples[hovered_idx]]
                    selected_cell = [pillar_cells[hovered_idx]]
                    pillar_rings = []
                    for pillar_level in range(get_num_levels()):
                        pillar_plane = get_bounding_plane(stream_idx, pillar_level)
                        if pillar_plane is None:
                            continue
                        try:
                            mapped_cell = _map_voronoi_cells_to_contour(
                                obj, pillar_plane, selected_sample, selected_cell)
                        except Exception:
                            continue
                        if (not mapped_cell or mapped_cell[0] is None
                                or len(mapped_cell[0]) < 4):
                            continue
                        # Element zero is the mapped fiber site; the remaining
                        # points are the consistently ordered cell boundary.
                        ring = np.asarray(mapped_cell[0][1:], dtype=np.float32)
                        if ring.ndim == 2 and ring.shape[1] == 3 \
                                and np.all(np.isfinite(ring)):
                            pillar_rings.append((pillar_level, ring))
                    if pillar_rings:
                        obj.inspector_voronoi_pillar_3d = pillar_rings
        else:
            obj.inspector_highlight_vertex_3d = None
            obj.inspector_highlight_other_stream_corners_3d = None
            obj.inspector_highlight_fiber_idx = None
            obj.inspector_highlight_corner_vertices_3d = None
            obj.inspector_voronoi_pillar_3d = None

        # When in corner edit mode, show the same corner's correspondence on
        # ALL OTHER contour levels in the 3D viewer. This shows the contour
        # outlines at every level with the corner position marked, so the user
        # can see how the correspondence at the current level relates to others.
        if corr_corner >= 0:
            other_level_data = []  # [(contour_verts, corner_3d_pos), ...]
            bps_src = obj.bounding_planes
            contours_src = obj.contours
            if bps_src is not None and contours_src is not None:
                if is_post_stream and stream_idx < len(bps_src):
                    bp_list = bps_src[stream_idx]
                    c_list = contours_src[stream_idx]
                else:
                    bp_list = bps_src
                    c_list = contours_src
                for lev_i in range(len(bp_list)):
                    if lev_i == level_idx:
                        continue
                    bp_lev = bp_list[lev_i]
                    if bp_lev is None:
                        continue
                    # Handle pre-stream: bp_lev might be a list of BPs
                    if isinstance(bp_lev, list):
                        if stream_idx < len(bp_lev):
                            bp_lev = bp_lev[stream_idx]
                        else:
                            continue
                    cm = bp_lev.get('contour_match')
                    ci = bp_lev.get('corner_indices')
                    bp_c = bp_lev.get('bounding_plane')
                    if cm is None:
                        continue
                    # Find the corner vertex index
                    vi = None
                    if ci is not None and corr_corner < len(ci):
                        vi = ci[corr_corner]
                    elif bp_c is not None and corr_corner < len(bp_c):
                        bp_corner = np.array(bp_c[corr_corner])
                        vi = 0
                        min_d = float('inf')
                        for vvi, (p, q) in enumerate(cm):
                            d = np.linalg.norm(np.array(q) - bp_corner)
                            if d < min_d:
                                min_d = d
                                vi = vvi
                    if vi is not None and vi < len(cm):
                        corner_pos = np.array(cm[vi][0])
                        # Get contour vertices
                        c_lev = c_list[lev_i]
                        if isinstance(c_lev, list) and len(c_lev) > 0 and isinstance(c_lev[0], (list, np.ndarray)) and np.asarray(c_lev[0]).ndim == 1:
                            # c_lev is a list of 3D points (single contour)
                            contour_verts = np.asarray(c_lev)
                        elif isinstance(c_lev, np.ndarray) and c_lev.ndim == 2:
                            contour_verts = c_lev
                        elif isinstance(c_lev, list) and stream_idx < len(c_lev):
                            # Pre-stream: c_lev is list of contours
                            contour_verts = np.asarray(c_lev[stream_idx])
                        else:
                            contour_verts = None
                        if contour_verts is not None and len(contour_verts) >= 3:
                            other_level_data.append((contour_verts, corner_pos))
            obj.inspector_highlight_other_level_contours = other_level_data if other_level_data else None

        # Show tooltip
        if hovered_idx >= 0:
            if hovered_type == 'vertex':
                imgui.set_tooltip(f"Vertex {hovered_idx}")
            elif hovered_type == 'fiber':
                imgui.set_tooltip(f"Fiber {hovered_idx}")
            elif hovered_type == 'waypoint':
                imgui.set_tooltip(f"Waypoint {hovered_idx}")
            elif hovered_type == 'corner':
                corner_names = ['Bottom-Left', 'Bottom-Right', 'Top-Right', 'Top-Left']
                corner_name = corner_names[hovered_idx] if hovered_idx < 4 else f"Corner {hovered_idx}"
                # Find closest vertex for this corner
                closest_vi = -1
                for corner_ci, vi in corner_to_closest_vertex:
                    if corner_ci == hovered_idx:
                        closest_vi = vi
                        break
                if closest_vi >= 0:
                    imgui.set_tooltip(f"{corner_name} Corner -> Vertex {closest_vi}")

            if show_all:
                imgui.separator()

        # Handle click for correspondence mode
        mouse_clicked = imgui.is_mouse_clicked(0)  # Left click
        if mouse_clicked and hovered_idx >= 0:
            if hovered_type == 'corner':
                # Clicking corner enters correspondence mode — save backup for hover preview
                v.inspect_2d_corr_mode[name] = True
                v.inspect_2d_corr_corner[name] = hovered_idx
                v.inspect_2d_corr_vertex[name] = -1
                corr_mode = True
                corr_corner = hovered_idx
                corr_vertex = -1
                # Save original contour_match + corner_indices + waypoints for hover restore
                if contour_match is not None:
                    _bp_ref = plane_info
                    v.inspect_2d_corr_backup_cm[name] = [((np.array(p).copy(), np.array(q).copy())) for p, q in contour_match]
                    v.inspect_2d_corr_backup_ci = _bp_ref.get('corner_indices')  # may be None
                    v.inspect_2d_corr_preview_active[name] = False
                    # Backup waypoints/mvc for this level
                    wp = get_waypoints_for_level(stream_idx, level_idx)
                    v.inspect_2d_corr_backup_wp[name] = (
                        [np.array(w).copy() for w in wp] if wp is not None else None)
                    if hasattr(obj, 'mvc_weights') and obj.mvc_weights is not None:
                        if is_post_stream and stream_idx < len(obj.mvc_weights) and level_idx < len(obj.mvc_weights[stream_idx]):
                            v.inspect_2d_corr_backup_mvc[name] = obj.mvc_weights[stream_idx][level_idx]
            elif hovered_type == 'vertex' and corr_mode and corr_corner >= 0:
                # Clicking vertex confirms — apply permanently, clear backup
                _apply_corner_correspondence(v, name, obj, stream_idx, level_idx,
                                             corr_corner, hovered_idx, is_post_stream,
                                             bp_info_override=plane_info,
                                             waypoint_store=(inspect_waypoints
                                                             if use_tendon_inspect
                                                             else None))
                v.inspect_2d_corr_mode[name] = False
                v.inspect_2d_corr_corner[name] = -1
                v.inspect_2d_corr_vertex[name] = -1
                v.inspect_2d_corr_backup_cm.pop(name, None)
                v.inspect_2d_corr_backup_wp.pop(name, None)
                v.inspect_2d_corr_backup_mvc.pop(name, None)
                v.inspect_2d_corr_preview_active.pop(name, None)
                corr_mode = False
                corr_corner = -1
            elif hovered_type == 'fiber' and edit_fiber_mode:
                # Clicking on existing fiber in edit mode - select it
                v.inspect_2d_edit_fiber_selected[name] = hovered_idx
                v.inspect_2d_edit_fiber_preview[name] = None
                edit_fiber_selected = hovered_idx
                edit_fiber_preview = None

        # Handle click on empty space in unit square for edit fiber mode
        if mouse_clicked and edit_fiber_mode and hovered_idx < 0:
            # Check if click is within the left canvas (unit square)
            if (left_x0 <= mouse_pos[0] <= left_x0 + canvas_size and
                left_y0 <= mouse_pos[1] <= left_y0 + canvas_size):
                # Convert screen position to unit square coordinates
                ucoord = (mouse_pos[0] - left_x0) / canvas_size
                vcoord = 1 - (mouse_pos[1] - left_y0) / canvas_size  # Flip Y
                # Clamp to [0, 1]
                ucoord = max(0.0, min(1.0, ucoord))
                vcoord = max(0.0, min(1.0, vcoord))
                # Set preview position and clear test data
                v.inspect_2d_edit_fiber_preview[name] = (ucoord, vcoord)
                v.inspect_2d_edit_fiber_selected[name] = -1
                v.inspect_2d_edit_fiber_test[name] = None  # Clear previous test
                # Clear test fiber from object
                obj.test_fiber_waypoints = None
                obj.test_fiber_stream_idx = None
                edit_fiber_preview = (ucoord, vcoord)
                edit_fiber_selected = -1
                edit_fiber_test = None

        # Hover preview for correspondence mode — temporarily apply when hovering vertex
        if corr_mode and corr_corner >= 0 and name in v.inspect_2d_corr_backup_cm:
            if use_tendon_inspect:
                bp_info_ref = get_bounding_plane(stream_idx, level_idx)
            elif is_post_stream:
                bp_info_ref = obj.bounding_planes[stream_idx][level_idx]
            else:
                bp_info_ref = obj.bounding_planes[level_idx][stream_idx]
            backup_cm = v.inspect_2d_corr_backup_cm[name]
            backup_ci = getattr(v, 'inspect_2d_corr_backup_ci', None)

            if hovered_type == 'vertex' and hovered_idx >= 0:
                # Restore backup first, then apply preview (contour_match only, no waypoints)
                bp_info_ref['contour_match'] = [(np.array(p).copy(), np.array(q).copy()) for p, q in backup_cm]
                if backup_ci is not None:
                    bp_info_ref['corner_indices'] = list(backup_ci)
                else:
                    bp_info_ref.pop('corner_indices', None)
                # Apply preview: only update contour_match and corner_indices, skip waypoints
                _apply_corner_correspondence_lightweight(obj, stream_idx, level_idx,
                                                         corr_corner, hovered_idx, is_post_stream,
                                                         bp_info_override=bp_info_ref)
                v.inspect_2d_corr_preview_active[name] = True
            elif v.inspect_2d_corr_preview_active.get(name, False):
                # Not hovering vertex — restore backup
                bp_info_ref['contour_match'] = [(np.array(p).copy(), np.array(q).copy()) for p, q in backup_cm]
                if backup_ci is not None:
                    bp_info_ref['corner_indices'] = list(backup_ci)
                else:
                    bp_info_ref.pop('corner_indices', None)
                v.inspect_2d_corr_preview_active[name] = False

        # Update cyan corner line strip AFTER preview has modified contour_match,
        # so the current level's corner position reflects the hovered vertex.
        # Re-read ALL levels' corner positions (including current, which may be preview-modified).
        if corr_corner >= 0:
            bps_src = inspect_planes if use_tendon_inspect else obj.bounding_planes
            if bps_src is not None:
                if is_post_stream and stream_idx < len(bps_src):
                    bp_list_all = bps_src[stream_idx]
                else:
                    bp_list_all = bps_src
                corner_pts_all = []
                for lev_i in range(len(bp_list_all)):
                    bp_lev = bp_list_all[lev_i]
                    if bp_lev is None:
                        continue
                    if isinstance(bp_lev, list):
                        if stream_idx < len(bp_lev):
                            bp_lev = bp_lev[stream_idx]
                        else:
                            continue
                    cm = bp_lev.get('contour_match')
                    ci = bp_lev.get('corner_indices')
                    bp_c = bp_lev.get('bounding_plane')
                    if cm is None:
                        continue
                    vi = None
                    if ci is not None and corr_corner < len(ci):
                        vi = ci[corr_corner]
                    elif bp_c is not None and corr_corner < len(bp_c):
                        bp_corner = np.array(bp_c[corr_corner])
                        vi = 0
                        min_d = float('inf')
                        for vvi, (p, q) in enumerate(cm):
                            d = np.linalg.norm(np.array(q) - bp_corner)
                            if d < min_d:
                                min_d = d
                                vi = vvi
                    if vi is not None and vi < len(cm):
                        corner_pts_all.append(np.array(cm[vi][0]))
                obj.inspector_highlight_corner_vertices_3d = corner_pts_all if corner_pts_all else None

        # Draw correspondence mode visual feedback
        if corr_mode and corr_corner >= 0:
            draw_list = imgui.get_window_draw_list()
            # Highlight selected corner with green
            if corr_corner < len(corner_screen_points_right):
                cx, cy = corner_screen_points_right[corr_corner]
                draw_list.add_quad_filled(cx, cy - 10, cx + 10, cy, cx, cy + 10, cx - 10, cy,
                                         imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 1.0))
                draw_list.add_quad(cx, cy - 12, cx + 12, cy, cx, cy + 12, cx - 12, cy,
                                  imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
            if corr_corner < len(corner_screen_points_left):
                cx, cy = corner_screen_points_left[corr_corner]
                draw_list.add_quad_filled(cx, cy - 10, cx + 10, cy, cx, cy + 10, cx - 10, cy,
                                         imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 1.0))
                draw_list.add_quad(cx, cy - 12, cx + 12, cy, cx, cy + 12, cx - 12, cy,
                                  imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)

            # Highlight selected vertex with green if selected
            if corr_vertex >= 0 and corr_vertex < len(p_screen_points):
                px, py = p_screen_points[corr_vertex]
                draw_list.add_circle_filled(px, py, 9, imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 1.0))
                draw_list.add_circle(px, py, 11, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                # Also highlight on Q side
                if corr_vertex < len(q_screen_points):
                    qx, qy = q_screen_points[corr_vertex]
                    draw_list.add_circle_filled(qx, qy, 9, imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 1.0))
                    draw_list.add_circle(qx, qy, 11, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                # Draw line from corner to selected vertex
                if corr_corner < len(corner_screen_points_right):
                    cx, cy = corner_screen_points_right[corr_corner]
                    draw_list.add_line(cx, cy, px, py,
                                      imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 1.0), thickness=3.0)

        # Draw edit fiber mode visual feedback
        if edit_fiber_mode:
            draw_list = imgui.get_window_draw_list()
            # Highlight selected fiber with yellow ring
            if edit_fiber_selected >= 0 and edit_fiber_selected < len(fiber_screen_points):
                fx, fy = fiber_screen_points[edit_fiber_selected]
                draw_list.add_circle_filled(fx, fy, 8, imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 1.0))
                draw_list.add_circle(fx, fy, 10, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
            # Draw preview marker (cyan) for new fiber position
            if edit_fiber_preview is not None:
                px = left_x0 + edit_fiber_preview[0] * canvas_size
                py = left_y0 + (1 - edit_fiber_preview[1]) * canvas_size
                draw_list.add_circle_filled(px, py, 6, imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 0.7))
                draw_list.add_circle(px, py, 8, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.0)
                # Draw crosshair
                draw_list.add_line(px - 12, py, px + 12, py, imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 0.5), 1.5)
                draw_list.add_line(px, py - 12, px, py + 12, imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 0.5), 1.5)

        # Correspondence mode UI
        if corr_mode:
            imgui.separator()
            corner_names = ['Bottom-Left', 'Bottom-Right', 'Top-Right', 'Top-Left']
            corner_name = corner_names[corr_corner] if 0 <= corr_corner < 4 else f"Corner {corr_corner}"
            if hovered_type == 'vertex' and hovered_idx >= 0:
                imgui.text(f"{corner_name} -> Vertex {hovered_idx} (click to apply)")
            else:
                imgui.text(f"Selected {corner_name}, hover vertex to preview")

            # "Find cor" operates on the belly-only live arrays and is therefore
            # not offered while editing an extended tendon chart.
            if not use_tendon_inspect:
                imgui.same_line()
                if imgui.button(f"Find cor (x)##{name}"):
                    _find_correspondence_all_levels(v, name, obj, stream_idx, level_idx,
                                                     corr_corner, is_post_stream, axis='x')
                imgui.same_line()
                if imgui.button(f"Find cor (y)##{name}"):
                    _find_correspondence_all_levels(v, name, obj, stream_idx, level_idx,
                                                     corr_corner, is_post_stream, axis='y')

            if imgui.button(f"Cancel##{name}_corr"):
                # Restore backup contour_match + waypoints and exit corr mode
                backup_cm = v.inspect_2d_corr_backup_cm.get(name)
                if backup_cm is not None:
                    if use_tendon_inspect:
                        bp_info_ref = get_bounding_plane(stream_idx, level_idx)
                    elif is_post_stream:
                        bp_info_ref = obj.bounding_planes[stream_idx][level_idx]
                    else:
                        bp_info_ref = obj.bounding_planes[level_idx][stream_idx]
                    bp_info_ref['contour_match'] = [(np.array(p).copy(), np.array(q).copy()) for p, q in backup_cm]
                # Restore waypoints
                backup_wp = v.inspect_2d_corr_backup_wp.get(name)
                if backup_wp is not None and hasattr(obj, 'waypoints') and obj.waypoints is not None:
                    if is_post_stream and stream_idx < len(obj.waypoints) and level_idx < len(obj.waypoints[stream_idx]):
                        obj.waypoints[stream_idx][level_idx] = backup_wp
                    if (use_tendon_inspect and inspect_waypoints is not None
                            and stream_idx < len(inspect_waypoints)
                            and level_idx < len(inspect_waypoints[stream_idx])):
                        inspect_waypoints[stream_idx][level_idx] = [
                            np.asarray(w, dtype=np.float64).copy() for w in backup_wp]
                backup_mvc = v.inspect_2d_corr_backup_mvc.get(name)
                if backup_mvc is not None and hasattr(obj, 'mvc_weights') and obj.mvc_weights is not None:
                    if is_post_stream and stream_idx < len(obj.mvc_weights) and level_idx < len(obj.mvc_weights[stream_idx]):
                        obj.mvc_weights[stream_idx][level_idx] = backup_mvc
                v.inspect_2d_corr_mode[name] = False
                v.inspect_2d_corr_corner[name] = -1
                v.inspect_2d_corr_vertex[name] = -1
                v.inspect_2d_corr_backup_cm.pop(name, None)
                v.inspect_2d_corr_backup_wp.pop(name, None)
                v.inspect_2d_corr_backup_mvc.pop(name, None)
                v.inspect_2d_corr_preview_active.pop(name, None)

        # 3D MVC button for saddle-shaped contours
        if not corr_mode and not show_all and is_post_stream and contour_match is not None:
            if imgui.button(f"3D MVC##{name}"):
                _apply_3d_mvc(obj, stream_idx, level_idx, is_post_stream)

        # Edit Fiber mode UI (only available when not in "Show All" mode)
        if has_fiber and not corr_mode and not show_all:
            imgui.separator()
            if not edit_fiber_mode:
                # Show Edit Fiber button only when fiber architecture exists
                if imgui.button(f"Edit Fiber##{name}"):
                    v.inspect_2d_edit_fiber_mode[name] = True
                    v.inspect_2d_edit_fiber_selected[name] = -1
                    v.inspect_2d_edit_fiber_preview[name] = None
            else:
                # In edit fiber mode
                imgui.text("Edit Fiber Mode (click unit square)")

                # Exit button
                if imgui.button(f"Exit Edit Mode##{name}"):
                    v.inspect_2d_edit_fiber_mode[name] = False
                    v.inspect_2d_edit_fiber_selected[name] = -1
                    v.inspect_2d_edit_fiber_preview[name] = None
                    v.inspect_2d_edit_fiber_test[name] = None
                    # Clear test fiber from object
                    obj.test_fiber_waypoints = None
                    obj.test_fiber_stream_idx = None

                imgui.same_line()

                # Delete button - enabled when existing fiber is selected
                if edit_fiber_selected >= 0:
                    if imgui.button(f"Delete Fiber {edit_fiber_selected}##{name}"):
                        _delete_fiber(v, name, obj, stream_idx, edit_fiber_selected, is_post_stream)
                        v.inspect_2d_edit_fiber_selected[name] = -1
                        v.inspect_2d_edit_fiber_preview[name] = None
                else:
                    imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
                    imgui.button(f"Delete##{name}")
                    imgui.pop_style_var()

                imgui.same_line()

                # Add button - enabled when preview position is set
                if edit_fiber_preview is not None:
                    if imgui.button(f"Add Fiber##{name}"):
                        _add_fiber(v, name, obj, stream_idx, edit_fiber_preview, is_post_stream)
                        v.inspect_2d_edit_fiber_selected[name] = -1
                        v.inspect_2d_edit_fiber_preview[name] = None
                        v.inspect_2d_edit_fiber_test[name] = None  # Clear test data
                        # Clear test fiber from object
                        obj.test_fiber_waypoints = None
                        obj.test_fiber_stream_idx = None
                else:
                    imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
                    imgui.button(f"Add##{name}")
                    imgui.pop_style_var()

                imgui.same_line()

                # Test button - enabled when preview position is set
                if edit_fiber_preview is not None:
                    if imgui.button(f"Test##{name}"):
                        _test_fiber(v, name, obj, stream_idx, edit_fiber_preview, is_post_stream)
                else:
                    imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
                    imgui.button(f"Test##{name}")
                    imgui.pop_style_var()

                # Show info text
                if edit_fiber_selected >= 0:
                    imgui.text(f"Selected: Fiber {edit_fiber_selected}")
                elif edit_fiber_preview is not None:
                    test_status = " (testing)" if edit_fiber_test is not None else ""
                    imgui.text(f"Preview: ({edit_fiber_preview[0]:.3f}, {edit_fiber_preview[1]:.3f}){test_status}")
                else:
                    imgui.text("Click on fiber to select, or empty space to add")

        # End scrollable region (always used now)
        imgui.end_child()

        imgui.end()

    # Close windows that were marked for closing
    for name in muscles_to_close:
        v.inspect_2d_open[name] = False
        # Clear inspector highlight when window closes
        if name in v.zygote_muscle_meshes:
            obj = v.zygote_muscle_meshes[name]
            obj.inspector_highlight_stream = None
            obj.inspector_highlight_level = None
            obj.inspector_highlight_contour_level = None
            obj.inspector_highlight_vertex_3d = None
            obj.inspector_highlight_fiber_idx = None
            obj.inspector_highlight_corner_vertices_3d = None
            obj.inspector_voronoi_pillar_3d = None
            obj.inspector_highlight_other_stream_corners_3d = None
            obj.inspector_highlight_other_level_contours = None

            # Rebuild 3D fiber draw arrays to reflect correspondence changes
            if hasattr(obj, '_rebuild_fiber_draw_arrays'):
                obj._rebuild_fiber_draw_arrays()


def _reset_corner_correspondence(obj, stream_idx, level_idx, is_post_stream):
    """Reset corner correspondence using ray-based method (find_contour_match)."""
    if is_post_stream:
        bp_info = obj.bounding_planes[stream_idx][level_idx]
    else:
        bp_info = obj.bounding_planes[level_idx][stream_idx]

    bp_corners = bp_info.get('bounding_plane')
    if bp_corners is None:
        return

    # Get contour vertices
    contour_vertices = None
    if is_post_stream:
        if stream_idx < len(obj.contours) and level_idx < len(obj.contours[stream_idx]):
            contour_vertices = obj.contours[stream_idx][level_idx]
    else:
        if level_idx < len(obj.contours) and stream_idx < len(obj.contours[level_idx]):
            contour_vertices = obj.contours[level_idx][stream_idx]

    if contour_vertices is None:
        cm = bp_info.get('contour_match')
        if cm:
            contour_vertices = np.array([np.array(p) for p, q in cm])

    if contour_vertices is None:
        return

    preserve = getattr(obj, '_contours_normalized', False)
    new_contour, contour_match = obj.find_contour_match(
        np.array(contour_vertices), np.array(bp_corners), preserve_order=preserve)
    bp_info['contour_match'] = contour_match
    bp_info.pop('corner_indices', None)  # Clear stored indices so they're re-detected

    if is_post_stream:
        if stream_idx < len(obj.contours) and level_idx < len(obj.contours[stream_idx]):
            obj.contours[stream_idx][level_idx] = new_contour
    else:
        if level_idx < len(obj.contours) and stream_idx < len(obj.contours[level_idx]):
            obj.contours[level_idx][stream_idx] = new_contour

    # Recompute waypoints if fiber architecture exists
    if hasattr(obj, 'fiber_architecture') and obj.fiber_architecture is not None:
        if is_post_stream and stream_idx < len(obj.fiber_architecture):
            fiber_samples = obj.fiber_architecture[stream_idx]
            if fiber_samples is not None and len(fiber_samples) > 0:
                _, waypoints_3d, mvc_weights = obj.find_waypoints(bp_info, fiber_samples)
                if hasattr(obj, 'waypoints') and obj.waypoints is not None:
                    if stream_idx < len(obj.waypoints) and level_idx < len(obj.waypoints[stream_idx]):
                        obj.waypoints[stream_idx][level_idx] = waypoints_3d
                if hasattr(obj, 'mvc_weights') and obj.mvc_weights is not None:
                    if stream_idx < len(obj.mvc_weights) and level_idx < len(obj.mvc_weights[stream_idx]):
                        obj.mvc_weights[stream_idx][level_idx] = mvc_weights
                obj._save_fiber_anim_data()

    print(f"[Reset] Corner correspondence reset at stream={stream_idx} level={level_idx}")


def _find_correspondence_all_levels(v, name, obj, stream_idx, level_idx, corner_idx, is_post_stream, axis='x'):
    """Find corner correspondence across ALL levels by matching unit-square coordinate.

    Takes the selected corner's (u, v) position on the current level's unit square,
    then for each other level, finds the contour vertex at the same x (or y) ratio.
    When two candidates exist at the same ratio (upper/lower for x, left/right for y),
    picks the one on the same side as the current corner.

    Args:
        axis: 'x' matches u-coordinate, 'y' matches v-coordinate
    """
    # Get current level's corner position in unit square
    if is_post_stream:
        bp_curr = obj.bounding_planes[stream_idx][level_idx]
    else:
        bp_curr = obj.bounding_planes[level_idx][stream_idx]

    contour_match = bp_curr.get('contour_match')
    bp_corners = bp_curr.get('bounding_plane')
    if contour_match is None or bp_corners is None:
        print("  [Find cor] No contour_match or bounding_plane at current level")
        return

    ci = bp_curr.get('corner_indices')
    if ci is None:
        # Detect corner indices from Q positions
        bp_c = [np.array(c) for c in bp_corners[:4]]
        ci = []
        for corner_3d in bp_c:
            dists = [np.linalg.norm(np.array(q) - corner_3d) for _, q in contour_match]
            ci.append(int(np.argmin(dists)))
        bp_curr['corner_indices'] = ci

    if corner_idx >= len(ci):
        return

    # Get the corner's corresponding vertex P position, project to unit square
    corner_vi = ci[corner_idx]
    if corner_vi >= len(contour_match):
        return
    corner_p = np.array(contour_match[corner_vi][0])  # Use P (3D contour vertex), not Q

    # Project P onto bounding plane to get unit square coordinates
    bp_c = [np.array(c) for c in bp_corners[:4]]
    edge_x = bp_c[1] - bp_c[0]
    edge_y = bp_c[3] - bp_c[0]
    A = np.column_stack([edge_x, edge_y])
    rel = corner_p - bp_c[0]
    result, _, _, _ = np.linalg.lstsq(A, rel, rcond=None)
    target_u, target_v = float(result[0]), float(result[1])

    print(f"  [Find cor ({axis})] Corner {corner_idx} at ({target_u:.3f}, {target_v:.3f})")

    # Target ratio = projected u/v of the selected corner's vertex (so any
    # displacement propagates). Side preference inferred from the SAME
    # projection: if target_v > 0.5 the reference vertex is in the upper half
    # of the unit square → prefer upper v on other levels too. Likewise for u.
    # Inferring from the actual vertex position avoids reliance on whether
    # corner_idx 0..3 corresponds to BL/BR/TR/TL on the user's BP, which can
    # vary per cell.
    if axis == 'x':
        target_ratio = target_u
        prefer_high_other = target_v > 0.5
    else:
        target_ratio = target_v
        prefer_high_other = target_u > 0.5
    print(f"  [Find cor ({axis})] target=({target_u:.3f},{target_v:.3f}) "
          f"prefer_high_other={prefer_high_other}")

    # Apply to all levels in this stream
    if is_post_stream:
        bp_list = obj.bounding_planes[stream_idx]
    else:
        bp_list = obj.bounding_planes

    n_modified = 0
    fiber_samples = None
    if hasattr(obj, 'fiber_architecture') and obj.fiber_architecture is not None:
        if is_post_stream and stream_idx < len(obj.fiber_architecture):
            fiber_samples = obj.fiber_architecture[stream_idx]

    for lev in range(len(bp_list)):
        # Skip the current level — it's the reference
        if lev == level_idx:
            continue
        bp_lev = bp_list[lev] if is_post_stream else bp_list[lev]
        if isinstance(bp_lev, list):
            if stream_idx < len(bp_lev):
                bp_lev = bp_lev[stream_idx]
            else:
                continue

        match_lev = bp_lev.get('contour_match')
        bp_corners_lev = bp_lev.get('bounding_plane')
        if match_lev is None or bp_corners_lev is None:
            continue

        P_verts = [np.array(p) for p, q in match_lev]
        n_verts = len(P_verts)
        bp_c_lev = [np.array(c) for c in bp_corners_lev[:4]]

        # Project P vertices (actual contour positions) to unit square
        edge_x_lev = bp_c_lev[1] - bp_c_lev[0]
        edge_y_lev = bp_c_lev[3] - bp_c_lev[0]
        A_lev = np.column_stack([edge_x_lev, edge_y_lev])

        vert_uv = np.zeros((n_verts, 2))
        for vi in range(n_verts):
            p, _ = match_lev[vi]
            rel_p = np.array(p) - bp_c_lev[0]
            res, _, _, _ = np.linalg.lstsq(A_lev, rel_p, rcond=None)
            vert_uv[vi] = [res[0], res[1]]

        # Find vertex closest to target ratio on the matching axis
        if axis == 'x':
            diffs = np.abs(vert_uv[:, 0] - target_ratio)
        else:
            diffs = np.abs(vert_uv[:, 1] - target_ratio)

        # Candidate window: include any vertex whose matched-axis distance to
        # target is within 0.05 (5% of unit square) OR within 2x the best
        # diff — whichever is larger. A closed contour at a given u typically
        # has TWO vertices (upper-half and lower-half), and they're rarely
        # equidistant from target_u, so the tie window must be loose enough
        # to admit both. Side disambig below then picks the half matching the
        # reference corner.
        best_diff = float(np.min(diffs))
        tie_eps = max(best_diff, 0.05)
        candidates = np.where(diffs <= best_diff + tie_eps)[0]

        if len(candidates) == 0:
            continue

        # If only one within tie window, take it. Otherwise pick by side in 2D:
        # for axis='x' tie-break on v (high if top corner, low if bottom);
        # for axis='y' tie-break on u (high if right corner, low if left).
        if len(candidates) == 1:
            best_vi = int(candidates[0])
            print(f"  [Find cor lev={lev}] single candidate vi={best_vi} "
                  f"uv={vert_uv[best_vi].tolist()}")
        else:
            other_axis = 1 if axis == 'x' else 0
            other_vals = vert_uv[candidates, other_axis]
            if prefer_high_other:
                pick = int(np.argmax(other_vals))
            else:
                pick = int(np.argmin(other_vals))
            best_vi = int(candidates[pick])
            cand_uvs = [vert_uv[c].tolist() for c in candidates]
            print(f"  [Find cor lev={lev}] {len(candidates)} cands uv={cand_uvs} "
                  f"prefer_high={prefer_high_other} → vi={best_vi} "
                  f"uv={vert_uv[best_vi].tolist()}")

        # Apply this corner correspondence
        ci_lev = bp_lev.get('corner_indices')
        if ci_lev is None:
            ci_lev = []
            for c_idx, c_3d in enumerate(bp_c_lev):
                d = [np.linalg.norm(np.array(q) - c_3d) for _, q in match_lev]
                ci_lev.append(int(np.argmin(d)))

        new_ci = list(ci_lev)
        old_vi = new_ci[corner_idx]
        new_ci[corner_idx] = int(best_vi)

        if len(set(new_ci)) < 4:
            continue

        if new_ci[corner_idx] == old_vi:
            continue

        # Recompute contour_match
        from viewer.contour_mesh import ContourMeshMixin
        new_match = ContourMeshMixin._recompute_contour_match(
            P_verts, bp_c_lev, new_ci, n_verts)
        bp_lev['contour_match'] = new_match
        bp_lev['corner_indices'] = new_ci

        # Recompute waypoints
        if fiber_samples is not None and len(fiber_samples) > 0:
            _, new_wp, new_mvc = obj.find_waypoints(bp_lev, fiber_samples)
            if hasattr(obj, 'waypoints') and obj.waypoints is not None:
                if is_post_stream and stream_idx < len(obj.waypoints) and lev < len(obj.waypoints[stream_idx]):
                    obj.waypoints[stream_idx][lev] = new_wp
            if hasattr(obj, 'mvc_weights') and obj.mvc_weights is not None:
                if is_post_stream and stream_idx < len(obj.mvc_weights) and lev < len(obj.mvc_weights[stream_idx]):
                    obj.mvc_weights[stream_idx][lev] = new_mvc

        n_modified += 1

    if n_modified > 0:
        # Resave animation data
        if hasattr(obj, '_save_fiber_anim_data'):
            obj._save_fiber_anim_data()
        print(f"  [Find cor ({axis})] Modified {n_modified} levels for corner {corner_idx}")

    # Exit corr mode
    v.inspect_2d_corr_mode[name] = False
    v.inspect_2d_corr_corner[name] = -1
    v.inspect_2d_corr_vertex[name] = -1
    v.inspect_2d_corr_backup_cm.pop(name, None)
    v.inspect_2d_corr_backup_wp.pop(name, None)
    v.inspect_2d_corr_backup_mvc.pop(name, None)
    v.inspect_2d_corr_preview_active.pop(name, None)


def _apply_3d_mvc(obj, stream_idx, level_idx, is_post_stream):
    """Compute waypoints using 3D MVC for saddle-shaped contours.

    Instead of projecting Q to bounding plane (which collapses for saddles),
    compute unit-square Q positions directly from 3D arc-length along segments.
    Then run MVC inline with these proper 2D Q coordinates.

    Unit-square corner mapping:
      corner 0 → (0,0), corner 1 → (1,0), corner 2 → (1,1), corner 3 → (0,1)
    """
    if is_post_stream:
        bp_info = obj.bounding_planes[stream_idx][level_idx]
    else:
        bp_info = obj.bounding_planes[level_idx][stream_idx]

    contour_match = bp_info.get('contour_match')
    bp_corners = bp_info.get('bounding_plane')
    ci = bp_info.get('corner_indices')
    if contour_match is None or bp_corners is None or ci is None:
        print("  [3D MVC] No contour_match or corner_indices")
        return

    # Use raw 3D contour vertices, not contour_match ordering
    if is_post_stream:
        raw_contour = obj.contours[stream_idx][level_idx]
    else:
        raw_contour = obj.contours[level_idx][stream_idx]
    P_verts = [np.array(v) for v in raw_contour]
    n_verts = len(P_verts)
    Ps = np.array(P_verts)
    bp_c = [np.array(c) for c in bp_corners[:4]]

    # Corner indices are into contour_match; find corresponding raw contour indices
    # by matching 3D positions
    raw_ci = []
    for c_idx in ci:
        if c_idx < len(contour_match):
            corner_pos = np.array(contour_match[c_idx][0])
            dists = np.linalg.norm(Ps - corner_pos, axis=1)
            raw_ci.append(int(np.argmin(dists)))
        else:
            raw_ci.append(c_idx % n_verts)
    ci = raw_ci

    # Sort corners by contour index for proper segmentation (no overlap, full coverage)
    # Map each sorted corner back to its BP corner for UV assignment
    uv_bp = [np.array([0.0, 0.0]), np.array([1.0, 0.0]),
             np.array([1.0, 1.0]), np.array([0.0, 1.0])]

    # (bp_corner_idx, contour_vertex_idx) sorted by contour vertex
    sorted_pairs = sorted(enumerate(ci), key=lambda x: x[1])
    sorted_bp = [p[0] for p in sorted_pairs]
    sorted_vi = [p[1] for p in sorted_pairs]

    # Check winding: if sorted BP order goes CCW around unit square, reverse
    # Standard BP winding is CW: 0→1→2→3 = (0,0)→(1,0)→(1,1)→(0,1)
    # Check if sorted_bp follows CW by checking cross product of first two UV edges
    uv0 = uv_bp[sorted_bp[0]]
    uv1 = uv_bp[sorted_bp[1]]
    uv2 = uv_bp[sorted_bp[2]]
    e1 = uv1 - uv0
    e2 = uv2 - uv1
    cross = e1[0] * e2[1] - e1[1] * e2[0]
    # Don't reverse sorted order (segments must go forward).
    # If winding is wrong, the user can adjust corners. Try both and pick best.

    def _build_qs_direct():
        qs = np.zeros((n_verts, 2))
        nm = [None] * n_verts
        for seg_idx in range(4):
            vs = sorted_vi[seg_idx]
            ve = sorted_vi[(seg_idx + 1) % 4]
            bp_s = sorted_bp[seg_idx]
            bp_e = sorted_bp[(seg_idx + 1) % 4]
            uv_s = uv_bp[bp_s]
            uv_e = uv_bp[bp_e]
            q_s = bp_c[bp_s]
            q_e = bp_c[bp_e]
            # Always forward (sorted order guarantees short arcs except last which wraps)
            if ve > vs:
                seg = list(range(vs, ve))
            elif ve < vs:
                seg = list(range(vs, n_verts)) + list(range(0, ve))
            else:
                seg = [vs]
            if not seg:
                continue
            arc = [0.0]
            for i in range(1, len(seg)):
                arc.append(arc[-1] + np.linalg.norm(P_verts[seg[i]] - P_verts[seg[i-1]]))
            total = arc[-1] if arc[-1] > 1e-10 else 1.0
            for i, vi in enumerate(seg):
                t = arc[i] / total
                qs[vi] = (1 - t) * uv_s + t * uv_e
                nm[vi] = (P_verts[vi], (1 - t) * q_s + t * q_e)
        for i in range(n_verts):
            if nm[i] is None:
                nm[i] = (P_verts[i], bp_c[0].copy())
        return qs, nm

    def _compute_waypoints(qs_norm):
        fs_list = []
        for v in fiber_samples:
            s_v = [Q - v for Q in qs_norm]
            f_found = False
            for i in range(n_verts):
                ip = (i + 1) % n_verts
                r_i = np.linalg.norm(s_v[i])
                if r_i < 1e-10:
                    f = np.zeros(n_verts); f[i] = 1; fs_list.append(f); f_found = True; break
                A_i = np.linalg.det(np.array([s_v[i], s_v[ip]])) / 2
                D_i = np.dot(s_v[i], s_v[ip])
                if abs(A_i) < 1e-10 and D_i < 0:
                    rip = np.linalg.norm(s_v[ip])
                    fi = np.zeros(n_verts); fi[i] = 1
                    fip = np.zeros(n_verts); fip[ip] = 1
                    d = r_i + rip
                    fs_list.append((rip * fi + r_i * fip) / d if d > 1e-10 else (fi + fip) / 2)
                    f_found = True; break
            if f_found:
                continue
            f = np.zeros(n_verts); W = 0
            for i in range(n_verts):
                ip = (i + 1) % n_verts; im = (i - 1) % n_verts
                r_i = np.linalg.norm(s_v[i])
                if r_i < 1e-10: continue
                w = 0
                Aim = np.linalg.det(np.array([s_v[im], s_v[i]])) / 2
                if abs(Aim) > 1e-10:
                    w += (np.linalg.norm(s_v[im]) - np.dot(s_v[im], s_v[i]) / r_i) / Aim
                Ai = np.linalg.det(np.array([s_v[i], s_v[ip]])) / 2
                if abs(Ai) > 1e-10:
                    w += (np.linalg.norm(s_v[ip]) - np.dot(s_v[i], s_v[ip]) / r_i) / Ai
                f[i] = w; W += w
            fs_list.append(f / W if abs(W) > 1e-10 else np.ones(n_verts) / n_verts)
        return np.dot(np.array(fs_list), Ps)

    # Get fiber samples
    fiber_samples_raw = None
    if hasattr(obj, 'fiber_architecture') and obj.fiber_architecture is not None:
        if is_post_stream and stream_idx < len(obj.fiber_architecture):
            fiber_samples_raw = obj.fiber_architecture[stream_idx]
    if fiber_samples_raw is None or len(fiber_samples_raw) == 0:
        print("  [3D MVC] No fiber samples"); return
    fiber_samples = np.array(fiber_samples_raw)
    if fiber_samples.ndim == 2 and fiber_samples.shape[1] > 2:
        fiber_samples = fiber_samples[:, :2]

    # Try both windings
    Qs_direct, nm_direct = _build_qs_direct()
    wp_direct = _compute_waypoints(Qs_direct)

    # Mirror x to try opposite winding
    Qs_mirror = Qs_direct.copy()
    Qs_mirror[:, 0] = 1.0 - Qs_mirror[:, 0]
    wp_mirror = _compute_waypoints(Qs_mirror)

    # Also try mirror y
    Qs_mirror_y = Qs_direct.copy()
    Qs_mirror_y[:, 1] = 1.0 - Qs_mirror_y[:, 1]
    wp_mirror_y = _compute_waypoints(Qs_mirror_y)

    # Also try mirror both
    Qs_mirror_xy = Qs_direct.copy()
    Qs_mirror_xy[:, 0] = 1.0 - Qs_mirror_xy[:, 0]
    Qs_mirror_xy[:, 1] = 1.0 - Qs_mirror_xy[:, 1]
    wp_mirror_xy = _compute_waypoints(Qs_mirror_xy)

    # Pick best match with adjacent level
    adj_lev = level_idx - 1 if level_idx > 0 else level_idx + 1
    adj_wp = None
    if hasattr(obj, 'waypoints') and obj.waypoints is not None:
        if stream_idx < len(obj.waypoints) and adj_lev < len(obj.waypoints[stream_idx]):
            adj_wp = np.array(obj.waypoints[stream_idx][adj_lev])

    candidates = [
        ('direct', Qs_direct, wp_direct),
        ('mirror_x', Qs_mirror, wp_mirror),
        ('mirror_y', Qs_mirror_y, wp_mirror_y),
        ('mirror_xy', Qs_mirror_xy, wp_mirror_xy),
    ]

    best_name = 'direct'
    best_qs = Qs_direct
    waypoints = wp_direct
    if adj_wp is not None and len(adj_wp) == len(wp_direct):
        best_err = float('inf')
        for name_c, qs_c, wp_c in candidates:
            err = np.sum(np.linalg.norm(wp_c - adj_wp, axis=1))
            if err < best_err:
                best_err = err
                best_name = name_c
                best_qs = qs_c
                waypoints = wp_c
        print(f"  [3D MVC] Best: {best_name} (err={best_err:.4f})")

    Qs_normalized = best_qs
    bp_info['contour_match'] = nm_direct

    # Debug
    print(f"  [3D MVC] ci={list(ci)}, sorted_vi={sorted_vi}, sorted_bp={sorted_bp}, n_verts={n_verts}")
    total_seg = 0
    for seg_idx in range(4):
        vs = sorted_vi[seg_idx]
        ve = sorted_vi[(seg_idx + 1) % 4]
        bp_s = sorted_bp[seg_idx]
        bp_e = sorted_bp[(seg_idx + 1) % 4]
        seg_len = (ve - vs) if ve > vs else (n_verts - vs + ve)
        total_seg += seg_len
        print(f"  [3D MVC] Seg {seg_idx}: {vs}→{ve} ({seg_len} verts), BP {bp_s}→{bp_e}, uv ({uv_bp[bp_s][0]:.0f},{uv_bp[bp_s][1]:.0f})→({uv_bp[bp_e][0]:.0f},{uv_bp[bp_e][1]:.0f})")
    print(f"  [3D MVC] Total seg verts: {total_seg} (should be {n_verts})")
    n_none = sum(1 for m in nm_direct if m is None)
    print(f"  [3D MVC] None: {n_none}, Qs at (0,0): {sum(1 for q in Qs_normalized if q[0]==0 and q[1]==0)}")
    print(f"  [3D MVC] Qs range: x=[{Qs_normalized[:,0].min():.3f},{Qs_normalized[:,0].max():.3f}] y=[{Qs_normalized[:,1].min():.3f},{Qs_normalized[:,1].max():.3f}]")

    # Clamp extreme waypoints
    centroid = np.mean(Ps, axis=0)
    max_dist_sq = max(np.sum((p - centroid) ** 2) for p in Ps)
    threshold_sq = max_dist_sq * 4.0
    for i in range(len(waypoints)):
        if np.sum((waypoints[i] - centroid) ** 2) > threshold_sq or not np.all(np.isfinite(waypoints[i])):
            waypoints[i] = centroid

    # Compute MVC weights for storage
    qs_used = Qs_normalized
    fs_list = []
    for v in fiber_samples:
        s_v = [Q - v for Q in qs_used]
        f_found = False
        for i in range(n_verts):
            ip = (i + 1) % n_verts
            r_i = np.linalg.norm(s_v[i])
            if r_i < 1e-10:
                f = np.zeros(n_verts); f[i] = 1; fs_list.append(f); f_found = True; break
            A_i = np.linalg.det(np.array([s_v[i], s_v[ip]])) / 2
            D_i = np.dot(s_v[i], s_v[ip])
            if abs(A_i) < 1e-10 and D_i < 0:
                rip = np.linalg.norm(s_v[ip])
                fi = np.zeros(n_verts); fi[i] = 1
                fip = np.zeros(n_verts); fip[ip] = 1
                d = r_i + rip
                fs_list.append((rip * fi + r_i * fip) / d if d > 1e-10 else (fi + fip) / 2)
                f_found = True; break
        if f_found: continue
        f = np.zeros(n_verts); W = 0
        for i in range(n_verts):
            ip = (i + 1) % n_verts; im = (i - 1) % n_verts
            r_i = np.linalg.norm(s_v[i])
            if r_i < 1e-10: continue
            w = 0
            Aim = np.linalg.det(np.array([s_v[im], s_v[i]])) / 2
            if abs(Aim) > 1e-10:
                w += (np.linalg.norm(s_v[im]) - np.dot(s_v[im], s_v[i]) / r_i) / Aim
            Ai = np.linalg.det(np.array([s_v[i], s_v[ip]])) / 2
            if abs(Ai) > 1e-10:
                w += (np.linalg.norm(s_v[ip]) - np.dot(s_v[i], s_v[ip]) / r_i) / Ai
            f[i] = w; W += w
        fs_list.append(f / W if abs(W) > 1e-10 else np.ones(n_verts) / n_verts)
    fs = np.array(fs_list)

    print(f"  [3D MVC] Qs range: x=[{qs_used[:,0].min():.3f},{qs_used[:,0].max():.3f}] y=[{qs_used[:,1].min():.3f},{qs_used[:,1].max():.3f}]")

    # Update waypoints and MVC weights
    if hasattr(obj, 'waypoints') and obj.waypoints is not None:
        if stream_idx < len(obj.waypoints) and level_idx < len(obj.waypoints[stream_idx]):
            obj.waypoints[stream_idx][level_idx] = waypoints
    if hasattr(obj, 'mvc_weights') and obj.mvc_weights is not None:
        if stream_idx < len(obj.mvc_weights) and level_idx < len(obj.mvc_weights[stream_idx]):
            obj.mvc_weights[stream_idx][level_idx] = fs
    if hasattr(obj, '_save_fiber_anim_data'):
        obj._save_fiber_anim_data()
    obj._fiber_draw_dirty = True

    print(f"  [3D MVC] Applied at stream={stream_idx} level={level_idx}, {len(waypoints)} waypoints")



def _apply_corner_correspondence_lightweight(obj, stream_idx, level_idx, corner_idx,
                                             vertex_idx, is_post_stream,
                                             bp_info_override=None):
    """Update contour_match and corner_indices only, without recomputing waypoints.
    Used for hover preview to avoid expensive MVC computation on every mouse move."""
    if bp_info_override is not None:
        bp_info = bp_info_override
    elif is_post_stream:
        bp_info = obj.bounding_planes[stream_idx][level_idx]
    else:
        bp_info = obj.bounding_planes[level_idx][stream_idx]

    bp_corners = bp_info.get('bounding_plane', None)
    contour_match = bp_info.get('contour_match', None)
    if bp_corners is None or contour_match is None:
        return

    P_vertices = [np.array(p) for p, q in contour_match]
    n = len(P_vertices)
    bp_corners_3d = [np.array(c) for c in bp_corners[:4]]

    current_corner_indices = bp_info.get('corner_indices', None)
    if current_corner_indices is None:
        current_corner_indices = []
        for ci, corner_3d in enumerate(bp_corners_3d):
            dists = [np.linalg.norm(np.array(q) - corner_3d) for _, q in contour_match]
            current_corner_indices.append(int(np.argmin(dists)))
    else:
        current_corner_indices = list(current_corner_indices)

    new_corner_indices = current_corner_indices.copy()
    new_corner_indices[corner_idx] = vertex_idx

    new_contour_match = [None] * n
    for edge_idx in range(4):
        vs = new_corner_indices[edge_idx]
        ve = new_corner_indices[(edge_idx + 1) % 4]
        q_start = bp_corners_3d[edge_idx]
        q_end = bp_corners_3d[(edge_idx + 1) % 4]

        if ve > vs:
            seg_indices = list(range(vs, ve))
        elif ve < vs:
            seg_indices = list(range(vs, n)) + list(range(0, ve))
        else:
            seg_indices = [vs]

        if len(seg_indices) == 0:
            continue

        arc_lengths = [0.0]
        for i in range(1, len(seg_indices)):
            arc_lengths.append(arc_lengths[-1] +
                np.linalg.norm(P_vertices[seg_indices[i]] - P_vertices[seg_indices[i - 1]]))
        # End corner (P_vertices[ve]) is NOT in this edge's segment (it is the next
        # edge's start), but the segment from seg_indices[-1] to it IS part of this
        # edge's arc length. Include it so the last in-segment vertex gets t<1.
        closing_seg = np.linalg.norm(P_vertices[ve] - P_vertices[seg_indices[-1]])
        total_arc = arc_lengths[-1] + closing_seg
        if total_arc < 1e-10:
            total_arc = 1.0

        for i, vi in enumerate(seg_indices):
            t = arc_lengths[i] / total_arc
            new_contour_match[vi] = (P_vertices[vi], (1 - t) * q_start + t * q_end)

    for i in range(n):
        if new_contour_match[i] is None:
            new_contour_match[i] = (P_vertices[i], np.array(contour_match[i][1]))

    bp_info['contour_match'] = new_contour_match
    bp_info['corner_indices'] = new_corner_indices


def _apply_corner_correspondence(v, name, obj, stream_idx, level_idx, corner_idx,
                                 vertex_idx, is_post_stream,
                                 bp_info_override=None,
                                 waypoint_store=None):
    """
    Apply manual corner-to-vertex correspondence.

    When a corner assignment changes, recompute Q positions based on arc-length
    interpolation between the new corner vertices.

    Args:
        name: Muscle name
        obj: ContourMesh object
        stream_idx: Stream index
        level_idx: Level index
        corner_idx: Selected corner (0-3)
        vertex_idx: Selected vertex index
        is_post_stream: Whether data is in post-stream format
    """
    # print(f"Applying correspondence: Corner {corner_idx} -> Vertex {vertex_idx}")
    # print(f"  Stream: {stream_idx}, Level: {level_idx}, Post-stream: {is_post_stream}")

    # Get current bounding plane info
    if bp_info_override is not None:
        bp_info = bp_info_override
    elif is_post_stream:
        bp_info = obj.bounding_planes[stream_idx][level_idx]
    else:
        bp_info = obj.bounding_planes[level_idx][stream_idx]

    bp_corners = bp_info.get('bounding_plane', None)
    contour_match = bp_info.get('contour_match', None)

    if bp_corners is None or contour_match is None:
        print("  Error: No bounding plane or contour_match found")
        return

    P_vertices = [np.array(p) for p, q in contour_match]
    n = len(P_vertices)
    bp_corners_3d = [np.array(c) for c in bp_corners[:4]]

    # Use stored corner indices if available, otherwise detect from Q
    current_corner_indices = bp_info.get('corner_indices', None)
    if current_corner_indices is None:
        current_corner_indices = []
        for ci, corner_3d in enumerate(bp_corners_3d):
            min_dist = float('inf')
            closest_vi = 0
            for vi, (p, q) in enumerate(contour_match):
                dist = np.linalg.norm(np.array(q) - corner_3d)
                if dist < min_dist:
                    min_dist = dist
                    closest_vi = vi
            current_corner_indices.append(closest_vi)
    else:
        current_corner_indices = list(current_corner_indices)

    # Only change the one corner being edited
    new_corner_indices = current_corner_indices.copy()
    new_corner_indices[corner_idx] = vertex_idx

    # Recompute Q for each P vertex in-place (same order as existing contour_match)
    # For each vertex, determine which BP edge segment it belongs to, then interpolate Q
    new_contour_match = [None] * n

    for edge_idx in range(4):
        corner_start = edge_idx
        corner_end = (edge_idx + 1) % 4

        vs = new_corner_indices[corner_start]
        ve = new_corner_indices[corner_end]

        q_start = bp_corners_3d[corner_start]
        q_end = bp_corners_3d[corner_end]

        # Collect vertex indices in this segment (vs inclusive, ve exclusive)
        if ve > vs:
            seg_indices = list(range(vs, ve))
        elif ve < vs:
            seg_indices = list(range(vs, n)) + list(range(0, ve))
        else:
            seg_indices = [vs]

        if len(seg_indices) == 0:
            continue

        # Arc-length parameterization. Include the closing segment from the last
        # in-segment vertex to the end corner P_vertices[ve] so the last vertex
        # before the corner gets t<1 instead of snapping to the corner's Q.
        arc_lengths = [0.0]
        for i in range(1, len(seg_indices)):
            arc_lengths.append(arc_lengths[-1] +
                np.linalg.norm(P_vertices[seg_indices[i]] - P_vertices[seg_indices[i - 1]]))
        closing_seg = np.linalg.norm(P_vertices[ve] - P_vertices[seg_indices[-1]])
        total_arc = arc_lengths[-1] + closing_seg
        if total_arc < 1e-10:
            total_arc = 1.0

        for i, vi in enumerate(seg_indices):
            t = arc_lengths[i] / total_arc
            new_contour_match[vi] = (P_vertices[vi], (1 - t) * q_start + t * q_end)

    # Fill any None entries (shouldn't happen, but safety)
    for i in range(n):
        if new_contour_match[i] is None:
            new_contour_match[i] = (P_vertices[i], np.array(contour_match[i][1]))

    bp_info['contour_match'] = new_contour_match
    bp_info['corner_indices'] = new_corner_indices

    # Recompute fiber structure if available
    if hasattr(obj, 'fiber_architecture') and obj.fiber_architecture is not None:
        if is_post_stream and stream_idx < len(obj.fiber_architecture):
            fiber_samples = obj.fiber_architecture[stream_idx]
            if fiber_samples is not None and len(fiber_samples) > 0:
                # Recompute waypoints and MVC weights for this level
                # find_waypoints returns (Qs_normalized_2d, waypoints_3d, mvc_weights)
                _, waypoints_3d, mvc_weights = obj.find_waypoints(bp_info, fiber_samples)

                # Update waypoints
                if hasattr(obj, 'waypoints') and obj.waypoints is not None:
                    if stream_idx < len(obj.waypoints) and level_idx < len(obj.waypoints[stream_idx]):
                        obj.waypoints[stream_idx][level_idx] = waypoints_3d
                if (waypoint_store is not None
                        and stream_idx < len(waypoint_store)
                        and level_idx < len(waypoint_store[stream_idx])):
                    waypoint_store[stream_idx][level_idx] = np.asarray(
                        waypoints_3d, dtype=np.float64).copy()

                # Update MVC weights
                if hasattr(obj, 'mvc_weights') and obj.mvc_weights is not None:
                    if stream_idx < len(obj.mvc_weights) and level_idx < len(obj.mvc_weights[stream_idx]):
                        obj.mvc_weights[stream_idx][level_idx] = mvc_weights

                # Resave animation data so replay uses updated waypoints
                obj._save_fiber_anim_data()


def _delete_fiber(v, name, obj, stream_idx, fiber_idx, is_post_stream):
    """
    Delete a fiber from the fiber architecture for a given stream.
    Removes the fiber from all levels and recomputes waypoints/MVC weights.

    Args:
        name: Muscle name
        obj: ContourMesh object
        stream_idx: Stream index
        fiber_idx: Index of fiber to delete
        is_post_stream: Whether data is in post-stream format
    """
    print(f"Deleting fiber {fiber_idx} from stream {stream_idx}")

    # Check if fiber_architecture exists
    if not hasattr(obj, 'fiber_architecture') or obj.fiber_architecture is None:
        print("  Error: No fiber architecture found")
        return

    if stream_idx >= len(obj.fiber_architecture):
        print(f"  Error: stream_idx {stream_idx} out of range")
        return

    fiber_samples = obj.fiber_architecture[stream_idx]
    if fiber_samples is None or fiber_idx >= len(fiber_samples):
        print(f"  Error: fiber_idx {fiber_idx} out of range")
        return

    # Remove the fiber sample (handle both list and numpy array)
    if isinstance(fiber_samples, np.ndarray):
        fiber_samples = np.delete(fiber_samples, fiber_idx, axis=0)
        obj.fiber_architecture[stream_idx] = fiber_samples
    else:
        fiber_samples.pop(fiber_idx)
    print(f"  Removed fiber sample, {len(fiber_samples)} remaining")

    # Recompute waypoints and MVC weights for all levels in this stream
    _recompute_fiber_data_for_stream(v, obj, stream_idx, is_post_stream)

    # Resave animation data so replay uses updated fiber structure
    obj._save_fiber_anim_data()

    print("  Fiber deleted successfully")


def _add_fiber(v, name, obj, stream_idx, position, is_post_stream):
    """
    Add a new fiber to the fiber architecture for a given stream.
    Adds the fiber to all levels and recomputes waypoints/MVC weights.

    Args:
        name: Muscle name
        obj: ContourMesh object
        stream_idx: Stream index
        position: (u, v) position on unit square
        is_post_stream: Whether data is in post-stream format
    """
    print(f"Adding fiber at ({position[0]:.3f}, {position[1]:.3f}) to stream {stream_idx}")

    # Check if fiber_architecture exists
    if not hasattr(obj, 'fiber_architecture') or obj.fiber_architecture is None:
        print("  Error: No fiber architecture found")
        return

    if stream_idx >= len(obj.fiber_architecture):
        print(f"  Error: stream_idx {stream_idx} out of range")
        return

    fiber_samples = obj.fiber_architecture[stream_idx]
    if fiber_samples is None:
        fiber_samples = np.array([[position[0], position[1]]])
        obj.fiber_architecture[stream_idx] = fiber_samples
    elif isinstance(fiber_samples, np.ndarray):
        # Append to numpy array
        new_sample = np.array([[position[0], position[1]]])
        fiber_samples = np.vstack([fiber_samples, new_sample])
        obj.fiber_architecture[stream_idx] = fiber_samples
    else:
        # Append to list
        fiber_samples.append([position[0], position[1]])
    print(f"  Added fiber sample, {len(fiber_samples)} total")

    # Recompute waypoints and MVC weights for all levels in this stream
    _recompute_fiber_data_for_stream(v, obj, stream_idx, is_post_stream)

    # Resave animation data so replay uses updated fiber structure
    obj._save_fiber_anim_data()

    print("  Fiber added successfully")


def _recompute_fiber_data_for_stream(v, obj, stream_idx, is_post_stream):
    """
    Recompute waypoints and MVC weights for all levels in a stream after fiber changes.

    Args:
        obj: ContourMesh object
        stream_idx: Stream index
        is_post_stream: Whether data is in post-stream format
    """
    fiber_samples = obj.fiber_architecture[stream_idx]
    if fiber_samples is None or len(fiber_samples) == 0:
        print("  No fiber samples to recompute")
        return

    # Get number of levels for this stream
    if is_post_stream:
        num_levels = len(obj.bounding_planes[stream_idx]) if stream_idx < len(obj.bounding_planes) else 0
    else:
        num_levels = len(obj.bounding_planes) if obj.bounding_planes else 0

    # Ensure waypoints and mvc_weights lists are properly sized
    if not hasattr(obj, 'waypoints') or obj.waypoints is None:
        obj.waypoints = []
    if not hasattr(obj, 'mvc_weights') or obj.mvc_weights is None:
        obj.mvc_weights = []

    # Extend lists if needed
    while len(obj.waypoints) <= stream_idx:
        obj.waypoints.append([])
    while len(obj.mvc_weights) <= stream_idx:
        obj.mvc_weights.append([])

    # Recompute for each level
    for level_idx in range(num_levels):
        # Get bounding plane info
        if is_post_stream:
            bp_info = obj.bounding_planes[stream_idx][level_idx]
        else:
            bp_info = obj.bounding_planes[level_idx][stream_idx]

        if bp_info is None:
            continue

        contour_match = bp_info.get('contour_match', None)
        if contour_match is None:
            continue

        # Recompute waypoints and MVC weights
        try:
            _, waypoints_3d, mvc_weights = obj.find_waypoints(bp_info, fiber_samples)

            # Ensure level lists are properly sized
            while len(obj.waypoints[stream_idx]) <= level_idx:
                obj.waypoints[stream_idx].append([])
            while len(obj.mvc_weights[stream_idx]) <= level_idx:
                obj.mvc_weights[stream_idx].append([])

            # Update data
            obj.waypoints[stream_idx][level_idx] = waypoints_3d
            obj.mvc_weights[stream_idx][level_idx] = mvc_weights
            print(f"  Recomputed level {level_idx}: {len(waypoints_3d)} waypoints")
        except Exception as e:
            print(f"  Error recomputing level {level_idx}: {e}")


def _test_fiber(v, name, obj, stream_idx, position, is_post_stream):
    """
    Compute test waypoints for a fiber at the given position.
    Stores the waypoints for visualization in 3D (shown in blue).

    Args:
        name: Muscle name
        obj: ContourMesh object
        stream_idx: Stream index
        position: (u, v) position on unit square
        is_post_stream: Whether data is in post-stream format
    """
    print(f"Testing fiber at ({position[0]:.3f}, {position[1]:.3f}) for stream {stream_idx}")

    # Create a single fiber sample for testing
    test_fiber_sample = np.array([[position[0], position[1]]])

    # Get number of levels for this stream
    if is_post_stream:
        num_levels = len(obj.bounding_planes[stream_idx]) if stream_idx < len(obj.bounding_planes) else 0
    else:
        num_levels = len(obj.bounding_planes) if obj.bounding_planes else 0

    # Compute waypoints for each level
    test_waypoints = []
    for level_idx in range(num_levels):
        # Get bounding plane info
        if is_post_stream:
            bp_info = obj.bounding_planes[stream_idx][level_idx]
        else:
            bp_info = obj.bounding_planes[level_idx][stream_idx]

        if bp_info is None:
            test_waypoints.append(None)
            continue

        contour_match = bp_info.get('contour_match', None)
        if contour_match is None:
            test_waypoints.append(None)
            continue

        # Compute waypoint for this level
        try:
            _, waypoints_3d, _ = obj.find_waypoints(bp_info, test_fiber_sample)
            if len(waypoints_3d) > 0:
                test_waypoints.append(waypoints_3d[0])  # Single waypoint
                print(f"  Level {level_idx}: waypoint at [{waypoints_3d[0][0]:.4f}, {waypoints_3d[0][1]:.4f}, {waypoints_3d[0][2]:.4f}]")
            else:
                test_waypoints.append(None)
        except Exception as e:
            print(f"  Error computing waypoint for level {level_idx}: {e}")
            test_waypoints.append(None)

    # Store test waypoints for visualization (both locally and on object for 3D drawing)
    v.inspect_2d_edit_fiber_test[name] = {
        'stream_idx': stream_idx,
        'waypoints': test_waypoints
    }
    # Also store on object for draw_fiber_architecture to access
    obj.test_fiber_waypoints = test_waypoints
    obj.test_fiber_stream_idx = stream_idx
    print(f"  Test fiber computed with {sum(1 for w in test_waypoints if w is not None)} valid waypoints")


def _render_neck_viz_windows(v):
    """Render transition visualization windows showing source and target contours."""
    if not hasattr(v, 'neck_viz_open'):
        return

    muscles_to_close = []

    for name, is_open in list(v.neck_viz_open.items()):
        if not is_open:
            continue

        if name not in v.zygote_muscle_meshes:
            muscles_to_close.append(name)
            continue

        obj = v.zygote_muscle_meshes[name]

        if not hasattr(obj, '_neck_viz_data') or not obj._neck_viz_data:
            muscles_to_close.append(name)
            continue

        viz_data = obj._neck_viz_data
        num_viz = len(viz_data)

        imgui.set_next_window_size(450, 520, imgui.FIRST_USE_EVER)
        expanded, opened = imgui.begin(f"Neck Viz: {name}", True)

        if not opened:
            muscles_to_close.append(name)
            imgui.end()
            continue

        # Slider to select visualization
        viz_idx = v.neck_viz_idx.get(name, 0)
        viz_idx = min(viz_idx, max(0, num_viz - 1))
        changed, new_idx = imgui.slider_int(f"Transition##{name}", viz_idx, 0, max(0, num_viz - 1))
        if changed:
            v.neck_viz_idx[name] = new_idx
            viz_idx = new_idx

        # Get current visualization data
        data = viz_data[viz_idx]
        large_count = data.get('large_count', 0)
        small_count = data.get('small_count', 0)
        transition_str = f"{large_count}→{small_count}"
        imgui.text(f"Transition {viz_idx + 1}/{num_viz}: {transition_str}")

        scalar_large = data.get('scalar_large', 0)
        scalar_small = data.get('scalar_small', 0)
        imgui.text(f"Source (after split): {large_count} contours @ scalar {scalar_large:.4f}")
        imgui.text(f"Target (before split): {small_count} contours @ scalar {scalar_small:.4f}")
        imgui.separator()

        # Get contour data
        target_contours = data.get('target_contours_2d', [])
        source_contours = data.get('source_contours_2d', [])

        if not target_contours and not source_contours:
            imgui.text("No contour data available")
            imgui.end()
            continue

        # Collect all points to compute bounds
        all_points = []
        for c in target_contours:
            if c is not None and len(c) > 0:
                all_points.extend(c)
        for c in source_contours:
            if c is not None and len(c) > 0:
                all_points.extend(c)

        if len(all_points) < 3:
            imgui.text("Insufficient contour data")
            imgui.end()
            continue

        all_points = np.array(all_points)
        min_xy = all_points.min(axis=0)
        max_xy = all_points.max(axis=0)
        range_xy = max_xy - min_xy
        range_xy[range_xy < 1e-10] = 1.0
        max_range = max(range_xy[0], range_xy[1])
        margin = 0.1
        scale_factor = (1 - 2 * margin) / max_range
        center_xy = (min_xy + max_xy) / 2

        def to_screen(pt_2d, x0, y0, canvas_size):
            norm = (pt_2d - center_xy) * scale_factor + 0.5
            sx = x0 + norm[0] * canvas_size
            sy = y0 + (1 - norm[1]) * canvas_size  # Flip Y to match matplotlib
            return (sx, sy)

        # Canvas setup
        canvas_size = 380
        padding = 15

        draw_list = imgui.get_window_draw_list()
        cursor_pos = imgui.get_cursor_screen_pos()
        x0, y0 = cursor_pos[0] + padding, cursor_pos[1] + padding

        # Background
        draw_list.add_rect_filled(x0, y0, x0 + canvas_size, y0 + canvas_size,
                                 imgui.get_color_u32_rgba(0.1, 0.1, 0.1, 1.0))

        # Draw TARGET contours (merged, before division) in BLUE
        target_color = imgui.get_color_u32_rgba(0.3, 0.5, 1.0, 1.0)
        target_fill = imgui.get_color_u32_rgba(0.2, 0.3, 0.6, 0.3)
        for contour_2d in target_contours:
            if contour_2d is None or len(contour_2d) < 3:
                continue
            contour_2d = np.array(contour_2d)
            contour_screen = [to_screen(p, x0, y0, canvas_size) for p in contour_2d]
            # Fill
            for i in range(1, len(contour_screen) - 1):
                draw_list.add_triangle_filled(
                    contour_screen[0][0], contour_screen[0][1],
                    contour_screen[i][0], contour_screen[i][1],
                    contour_screen[i+1][0], contour_screen[i+1][1],
                    target_fill)
            # Outline
            for i in range(len(contour_screen)):
                p1, p2 = contour_screen[i], contour_screen[(i+1) % len(contour_screen)]
                draw_list.add_line(p1[0], p1[1], p2[0], p2[1], target_color, 2.5)

        # Draw SOURCE contours (split, after division) in ORANGE/RED
        source_colors = [
            imgui.get_color_u32_rgba(1.0, 0.4, 0.1, 1.0),  # Orange
            imgui.get_color_u32_rgba(1.0, 0.8, 0.0, 1.0),  # Yellow
            imgui.get_color_u32_rgba(0.0, 1.0, 0.4, 1.0),  # Green
            imgui.get_color_u32_rgba(1.0, 0.0, 0.5, 1.0),  # Pink
            imgui.get_color_u32_rgba(0.6, 0.2, 1.0, 1.0),  # Purple
        ]
        for ci, contour_2d in enumerate(source_contours):
            if contour_2d is None or len(contour_2d) < 3:
                continue
            contour_2d = np.array(contour_2d)
            contour_screen = [to_screen(p, x0, y0, canvas_size) for p in contour_2d]
            color = source_colors[ci % len(source_colors)]
            # Outline only (no fill to see overlap with target)
            for i in range(len(contour_screen)):
                p1, p2 = contour_screen[i], contour_screen[(i+1) % len(contour_screen)]
                draw_list.add_line(p1[0], p1[1], p2[0], p2[1], color, 2.0)

        imgui.dummy(canvas_size + 2 * padding, canvas_size + 2 * padding)

        # Legend
        imgui.separator()
        imgui.text_colored("Target (before split)", 0.3, 0.5, 1.0, 1.0)
        imgui.same_line()
        imgui.text(" | ")
        imgui.same_line()
        imgui.text_colored("Source (after split)", 1.0, 0.4, 0.1, 1.0)

        imgui.end()

    for name in muscles_to_close:
        v.neck_viz_open[name] = False


def _resume_pipeline_after_cut(v, obj, name):
    """Resume pipeline from _pipeline_paused_at after manual cut completes.

    Returns True if pipeline paused again (e.g. for level selection), False otherwise.
    """
    max_step = obj._process_step if hasattr(obj, '_process_step') else 12
    start_step = obj._pipeline_paused_at
    if start_step > max_step:
        obj._pipeline_paused_at = None
        return False

    print(f"[{name}] Auto-resuming pipeline from step {start_step} to {max_step}...")
    try:
        _defer = getattr(obj, 'animate_process', False)
        # Step 7: Stream Smooth (z, x, bp - after cut)
        if start_step <= 7 <= max_step and hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
            print(f"  [7/{max_step}] Stream Smoothening (z, x, bp)...")
            _t0 = time.time()
            obj.stream_smoothen_all(defer=_defer)
            _run_counterpart_step(v, name, obj, 7, defer=_defer)
            print(f"  [7/{max_step}] Done in {time.time()-_t0:.3f}s")

        # Step 8: Contour Select
        if start_step <= 8 <= max_step and hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
            print(f"  [8/{max_step}] Selecting contours...")
            _t0 = time.time()
            obj.select_levels()
            if not (hasattr(obj, '_level_select_window_open') and obj._level_select_window_open):
                _run_counterpart_step(v, name, obj, 8, defer=_defer)
            print(f"  [8/{max_step}] Done in {time.time()-_t0:.3f}s")
            # Check if waiting for manual level selection
            if hasattr(obj, '_level_select_window_open') and obj._level_select_window_open:
                obj._pipeline_paused_at = 9  # Resume from step 9 after selection
                print(f"  [8/{max_step}] Waiting for level selection - pipeline paused")
                return True

        # Step 9: Build Fiber
        if start_step <= 9 <= max_step and hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
            print(f"  [9/{max_step}] Building fibers...")
            _t0 = time.time()
            _ensure_level_selection_applied(v, name, obj, defer=_defer)
            obj._belly_waypoints_before_tendon_extension = None
            obj.build_fibers(skeleton_meshes=v.zygote_skeleton_meshes, defer=_defer)
            if (getattr(obj, 'enable_tendon_extension', True)
                    and (getattr(obj, 'origin_tendon_extension_name', '')
                         or getattr(obj, 'insertion_tendon_extension_name', ''))):
                _extend_belly_fibers_with_tendons(v, name, obj)
            _run_counterpart_step(v, name, obj, 9, defer=_defer)
            print(f"  [9/{max_step}] Done in {time.time()-_t0:.3f}s")
            if _defer:
                obj._level_select_replayed = False

        # Step 10: Resample Contours
        if start_step <= 10 <= max_step and obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None:
            print(f"  [10/{max_step}] Resampling Contours...")
            _t0 = time.time()
            _resample_contours_with_links(v, name, obj, defer=_defer)
            _resample_linked_tendon_extensions(v, name, obj, defer=_defer)
            print(f"  [10/{max_step}] Done in {time.time()-_t0:.3f}s")
            if _defer:
                obj._build_fibers_replayed = False

        # Step 11: Build Contour Mesh
        if start_step <= 11 <= max_step and obj.contours is not None and len(obj.contours) > 0 and obj.draw_contour_stream is not None:
            print(f"  [11/{max_step}] Building Contour Mesh...")
            _t0 = time.time()
            if _prepare_owned_connected_contour_mesh_source(v, name, obj):
                obj.build_contour_mesh(defer=_defer)
                if not _connected_source_has_linked_components(obj):
                    _run_counterpart_step(v, name, obj, 11, defer=_defer)
            print(f"  [11/{max_step}] Done in {time.time()-_t0:.3f}s")
            if _defer:
                obj._resample_replayed = False

        # Step 12: Tetrahedralize
        if start_step <= 12 <= max_step:
            print(f"  [12/{max_step}] Tetrahedralizing...")
            _t0 = time.time()
            if _skip_non_owner_connected_mesh(v, name, obj, "Tetrahedralize"):
                print(f"  [12/{max_step}] Skipped in {time.time()-_t0:.3f}s")
                tet_ok = False
            else:
                tet_ok = _tetrahedralize_single_contour_mesh(v, name, obj, defer=_defer)
                if tet_ok and not _connected_source_has_linked_components(obj):
                    _run_counterpart_step(v, name, obj, 12, defer=_defer)
            status = "Done" if tet_ok else "Failed"
            print(f"  [12/{max_step}] {status} in {time.time()-_t0:.3f}s")
            if tet_ok and obj.tet_vertices is not None:
                if _defer:
                    obj._extract_internal_tet_edges()
                    obj._classify_tet_faces_into_bands()
                    obj._tetrahedralize_replayed = False
                else:
                    obj.is_draw_contours = False
                    obj.is_draw_tet_mesh = True
                    obj._tetrahedralize_replayed = True

        print(f"[{name}] Pipeline complete (steps {start_step}-{max_step})!")
    except Exception as e:
        print(f"[{name}] Pipeline error: {e}")
        import traceback
        traceback.print_exc()
    obj._pipeline_paused_at = None
    return False


def _render_manual_cut_windows(v):
    """Render manual cutting windows for muscles that need user input."""
    for name, obj in v.zygote_muscle_meshes.items():
        if not hasattr(obj, '_manual_cut_pending') or not obj._manual_cut_pending:
            continue

        if obj._manual_cut_data is None:
            continue

        # Initialize mouse state for this window
        if not hasattr(v, '_manual_cut_mouse'):
            v._manual_cut_mouse = {}
        if name not in v._manual_cut_mouse:
            v._manual_cut_mouse[name] = {
                'dragging': False,
                'start_pos': None,
                'end_pos': None,
                'zoom': 1.0,
                'pan': [0.0, 0.0],
                'panning': False,
            }

        mouse_state = v._manual_cut_mouse[name]

        # Window setup
        muscle_name = obj._manual_cut_data.get('muscle_name', name)
        target_i = obj._manual_cut_data.get('target_i', 0)
        source_indices = obj._manual_cut_data.get('source_indices', [])
        required_pieces_display = obj._manual_cut_data.get('required_pieces', 2)
        # Window size: main canvas (550+40) + gap (30) + source panel (280+30) + margins = ~960
        imgui.set_next_window_size(960, 780, imgui.FIRST_USE_EVER)
        # Show target index and source info in title for M->N cases
        title_suffix = f" (Target {target_i}, {len(source_indices)}->1)" if len(source_indices) > 0 else ""
        expanded, opened = imgui.begin(f"Manual Cut: {muscle_name}{title_suffix}", True)

        if not opened:
            obj._cancel_manual_cut()
            imgui.end()
            continue

        # Get data
        target_2d = obj._manual_cut_data['target_2d']
        source_2d_list = obj._manual_cut_data['source_2d_list']
        target_level = obj._manual_cut_data['target_level']
        source_level = obj._manual_cut_data['source_level']

        # Find initial line using obj._find_neck_in_contour (same as find_transitions)
        # Only for SEPARATE mode - COMMON mode uses source boundary line
        # Skip for sub-windows (original_source_indices set) - they don't need neck finding
        is_common_mode = obj._manual_cut_data.get('is_common_mode', False)
        original_source_indices = obj._manual_cut_data.get('original_source_indices', None)
        is_subwindow = original_source_indices is not None
        subcut_level = obj._manual_cut_data.get('subcut_level', 0)
        need_new_recommendation = 'initial_line' not in obj._manual_cut_data

        # Neck finding for SEPARATE mode only (multiple separate sources → one target)
        # Skip for: COMMON mode (sources share boundary), sub-windows, or sub-cuts
        skip_neck = is_common_mode or is_subwindow or subcut_level > 0
        if need_new_recommendation and not skip_neck:
            current_pieces = obj._manual_cut_data.get('current_pieces', [target_2d])
            contour_range = np.max(target_2d.max(axis=0) - target_2d.min(axis=0))

            # Find ALL neck candidates with distance < 3% of perimeter
            # Only print debug once when actually computing candidates
            if 'neck_candidates' not in obj._manual_cut_data:
                print(f"[NECK] Looking for neck candidates in {len(current_pieces)} pieces, target has {len(target_2d)} vertices")
                all_candidates = []
                for piece_idx, piece_2d in enumerate(current_pieces):
                    n = len(piece_2d)
                    if n < 10:
                        continue

                    # Compute perimeter
                    perimeter = 0
                    for i in range(n):
                        perimeter += np.linalg.norm(piece_2d[(i+1) % n] - piece_2d[i])

                    # Threshold: 3% of perimeter (wider to catch merge point necks)
                    neck_threshold = perimeter * 0.03
                    # Use smaller index separation (10%) to catch pinch points
                    min_sep_idx = int(n * 0.10)

                    # Find all pairs below threshold
                    for i in range(n):
                        for j in range(i + min_sep_idx, i + n - min_sep_idx):
                            j_mod = j % n
                            # Only add if i < j_mod to avoid duplicates (i,j) and (j,i)
                            if i >= j_mod:
                                continue
                            dist = np.linalg.norm(piece_2d[i] - piece_2d[j_mod])
                            if dist < neck_threshold:
                                all_candidates.append({
                                    'piece_idx': piece_idx,
                                    'idx_a': i,
                                    'idx_b': j_mod,
                                    'width': dist,
                                    'point': (piece_2d[i] + piece_2d[j_mod]) / 2,
                                    'pos_a': piece_2d[i].copy(),
                                    'pos_b': piece_2d[j_mod].copy(),
                                })

                # Sort by width (narrowest first)
                all_candidates.sort(key=lambda x: x['width'])

                # Remove near-duplicates (candidates within 5 vertices of each other)
                filtered = []
                for cand in all_candidates:
                    is_dup = False
                    for existing in filtered:
                        if existing['piece_idx'] == cand['piece_idx']:
                            # Check if indices are close (in same order or swapped)
                            n = len(current_pieces[cand['piece_idx']])
                            # Same order check
                            dist_a = min(abs(cand['idx_a'] - existing['idx_a']),
                                       n - abs(cand['idx_a'] - existing['idx_a']))
                            dist_b = min(abs(cand['idx_b'] - existing['idx_b']),
                                       n - abs(cand['idx_b'] - existing['idx_b']))
                            # Swapped order check
                            dist_a_swap = min(abs(cand['idx_a'] - existing['idx_b']),
                                            n - abs(cand['idx_a'] - existing['idx_b']))
                            dist_b_swap = min(abs(cand['idx_b'] - existing['idx_a']),
                                            n - abs(cand['idx_b'] - existing['idx_a']))
                            if (dist_a < 5 and dist_b < 5) or (dist_a_swap < 5 and dist_b_swap < 5):
                                is_dup = True
                                break
                    if not is_dup:
                        filtered.append(cand)

                obj._manual_cut_data['neck_candidates'] = filtered
                obj._manual_cut_data['selected_neck_idx'] = 0 if len(filtered) > 0 else -1
                print(f"[NECK] Found {len(filtered)} neck candidates (from {len(all_candidates)} raw)")

            # Get current selection
            candidates = obj._manual_cut_data.get('neck_candidates', [])
            selected_idx = obj._manual_cut_data.get('selected_neck_idx', 0)

            # Helper to find line-contour intersection points
            def find_contour_intersections(line_start, line_end, contour):
                """Find where a line intersects the contour, return the two intersection points."""
                intersections = []
                p1 = np.array(line_start)
                p2 = np.array(line_end)
                d = p2 - p1

                for i in range(len(contour)):
                    q1 = contour[i]
                    q2 = contour[(i + 1) % len(contour)]
                    e = q2 - q1

                    denom = d[0] * e[1] - d[1] * e[0]
                    if abs(denom) < 1e-10:
                        continue

                    t = ((q1[0] - p1[0]) * e[1] - (q1[1] - p1[1]) * e[0]) / denom
                    s = ((q1[0] - p1[0]) * d[1] - (q1[1] - p1[1]) * d[0]) / denom

                    if 0 <= s <= 1:  # Intersection on contour edge
                        pt = p1 + t * d
                        intersections.append((t, pt, i))

                intersections.sort(key=lambda x: x[0])
                return intersections

            if len(candidates) > 0 and selected_idx >= 0 and selected_idx < len(candidates):
                cand = candidates[selected_idx]
                piece_idx = cand['piece_idx']
                i0, i1 = cand['idx_a'], cand['idx_b']
                piece_2d = current_pieces[piece_idx]
                n = len(piece_2d)

                # Get neck vertices
                neck_a = piece_2d[i0]
                neck_b = piece_2d[i1]

                # Use neck vertices directly as cutting line endpoints
                # They are already on the contour (vertices of piece_2d)
                line_start = tuple(neck_a)
                line_end = tuple(neck_b)

                # Store neck info for zoomed view
                obj._manual_cut_data['current_neck_info'] = {
                    'neck_a': neck_a.copy(),
                    'neck_b': neck_b.copy(),
                    'line_start': neck_a.copy(),
                    'line_end': neck_b.copy(),
                    'piece_idx': piece_idx,
                    'idx_a': i0,
                    'idx_b': i1,
                }

                obj._manual_cut_data['initial_line'] = (line_start, line_end)
                # Only set is_neck_line = True if user hasn't drawn their own line
                # (i.e., if _manual_cut_line is None or equals initial_line)
                if obj._manual_cut_line is None:
                    obj._manual_cut_data['is_neck_line'] = True  # Flag for vertex-to-vertex cutting
                    obj._manual_cut_line = (line_start, line_end)
                elif obj._manual_cut_line == obj._manual_cut_data.get('initial_line'):
                    # User hasn't changed the line from initial neck recommendation
                    obj._manual_cut_data['is_neck_line'] = True
            else:
                # No candidates found - set flag to prevent re-computation every frame
                if 'neck_search_done' not in obj._manual_cut_data:
                    print(f"[NECK] No neck candidates found")
                    obj._manual_cut_data['neck_search_done'] = True

        current_pieces = obj._manual_cut_data.get('current_pieces', [target_2d])
        required_pieces = obj._manual_cut_data.get('required_pieces', 2)
        imgui.text(f"Target level: {target_level} | Source level: {source_level}")
        imgui.text(f"Pieces: {len(current_pieces)} / {required_pieces} required")
        imgui.text(f"Target verts: {len(target_2d)}")
        imgui.text_colored("WHITE = Target contour (cut this)", 1.0, 1.0, 1.0, 1.0)
        imgui.same_line()
        imgui.text_colored(" | FADED = Source ref (guide)", 0.6, 0.6, 0.6, 1.0)
        imgui.text("Draw a line to cut. Scroll to zoom, middle-drag to pan.")
        imgui.text(f"Zoom: {mouse_state['zoom']:.1f}x")

        # Neck candidate selector slider
        candidates = obj._manual_cut_data.get('neck_candidates', [])
        if len(candidates) > 0:
            selected_idx = obj._manual_cut_data.get('selected_neck_idx', 0)
            cand = candidates[selected_idx] if selected_idx < len(candidates) else None
            width_str = f"{cand['width']:.6f}" if cand else "N/A"
            imgui.text(f"Neck candidates: {len(candidates)} | Selected: {selected_idx} (width={width_str})")
            imgui.push_item_width(200)
            changed, new_idx = imgui.slider_int(f"##neck_slider_{name}", selected_idx, 0, len(candidates) - 1)
            if changed:
                obj._manual_cut_data['selected_neck_idx'] = new_idx
                # Update the cutting line to match selected candidate
                if new_idx < len(candidates):
                    cand = candidates[new_idx]
                    piece_idx = cand['piece_idx']
                    i0, i1 = cand['idx_a'], cand['idx_b']
                    current_pieces = obj._manual_cut_data.get('current_pieces', [target_2d])
                    piece_2d = current_pieces[piece_idx]
                    n = len(piece_2d)
                    contour_range = np.max(target_2d.max(axis=0) - target_2d.min(axis=0))

                    # Get neck vertices
                    neck_a = piece_2d[i0]
                    neck_b = piece_2d[i1]

                    # Determine line direction
                    direction = neck_b - neck_a
                    dir_len = np.linalg.norm(direction)
                    if dir_len > 1e-10:
                        direction = direction / dir_len
                    else:
                        # Zero-length (pinch): use direction from prev to prev
                        prev_i0 = (i0 - 1) % n
                        prev_i1 = (i1 - 1) % n
                        direction = piece_2d[prev_i1] - piece_2d[prev_i0]
                        d_len = np.linalg.norm(direction)
                        if d_len > 1e-10:
                            direction = direction / d_len
                        else:
                            direction = np.array([1.0, 0.0])

                    # Create extended line to find edge intersections
                    neck_width = np.linalg.norm(neck_b - neck_a)
                    extension = max(neck_width * 0.5, contour_range * 0.10)
                    ext_start = neck_a - direction * extension
                    ext_end = neck_b + direction * extension

                    # Find actual contour edge intersections
                    def find_inters_inline(ls, le, contour):
                        inters = []
                        p1, p2 = np.array(ls), np.array(le)
                        d = p2 - p1
                        for ci in range(len(contour)):
                            q1, q2 = contour[ci], contour[(ci + 1) % len(contour)]
                            e = q2 - q1
                            denom = d[0] * e[1] - d[1] * e[0]
                            if abs(denom) < 1e-10:
                                continue
                            t = ((q1[0] - p1[0]) * e[1] - (q1[1] - p1[1]) * e[0]) / denom
                            s = ((q1[0] - p1[0]) * d[1] - (q1[1] - p1[1]) * d[0]) / denom
                            if 0 <= s <= 1:
                                inters.append((t, p1 + t * d, ci))
                        return inters

                    intersections = find_inters_inline(ext_start, ext_end, piece_2d)

                    if len(intersections) >= 2:
                        # Find two intersections closest to neck center
                        neck_center = (neck_a + neck_b) / 2
                        inters_with_dist = [(np.linalg.norm(pt - neck_center), t, pt, ei)
                                           for t, pt, ei in intersections]
                        inters_with_dist.sort(key=lambda x: x[0])
                        if len(inters_with_dist) >= 2:
                            _, _, p_start, _ = inters_with_dist[0]
                            _, _, p_end, _ = inters_with_dist[1]
                        else:
                            p_start, p_end = ext_start, ext_end
                    else:
                        p_start, p_end = ext_start, ext_end

                    # Update cutting line
                    obj._manual_cut_line = (tuple(p_start), tuple(p_end))
                    obj._manual_cut_data['initial_line'] = obj._manual_cut_line

                    # Update neck info for zoomed view
                    obj._manual_cut_data['current_neck_info'] = {
                        'neck_a': neck_a.copy(),
                        'neck_b': neck_b.copy(),
                        'line_start': np.array(p_start).copy(),
                        'line_end': np.array(p_end).copy(),
                        'piece_idx': piece_idx,
                        'idx_a': i0,
                        'idx_b': i1,
                    }
                    # User selected a neck via slider - use neck-based cutting
                    obj._manual_cut_data['is_neck_line'] = True
            imgui.pop_item_width()
        imgui.separator()

        # Compute bounds for normalization (include target AND projected sources)
        all_points_2d = [target_2d]

        # Project all source contours onto target plane and include in bounds
        source_contours_3d_for_bounds = obj._manual_cut_data.get('source_contours', [])
        source_bps_for_bounds = obj._manual_cut_data.get('source_bps', [])
        target_bp = obj._manual_cut_data['target_bp']
        target_mean_bounds = target_bp['mean']
        # Negate basis vectors to rotate 180° - match cutting window orientation
        target_x_bounds = -target_bp['basis_x']
        target_y_bounds = -target_bp['basis_y']
        target_z_bounds = target_bp['basis_z']

        for src_contour_3d in source_contours_3d_for_bounds:
            src_projected = []
            for pt in src_contour_3d:
                diff = pt - target_mean_bounds
                proj_pt = pt - np.dot(diff, target_z_bounds) * target_z_bounds
                diff_proj = proj_pt - target_mean_bounds
                x_coord = np.dot(diff_proj, target_x_bounds)
                y_coord = np.dot(diff_proj, target_y_bounds)
                src_projected.append([x_coord, y_coord])
            if len(src_projected) > 0:
                all_points_2d.append(np.array(src_projected))

        # Include transformed sources from optimization in bounds (if available)
        transformed_sources = obj._manual_cut_data.get('transformed_sources_2d', None) or []
        for src_2d in transformed_sources:
            if src_2d is not None and len(src_2d) >= 3:
                all_points_2d.append(np.array(src_2d))

        # Compute combined bounds
        all_points_combined = np.vstack(all_points_2d)
        min_xy = all_points_combined.min(axis=0)
        max_xy = all_points_combined.max(axis=0)
        range_xy = max_xy - min_xy
        max_range = max(range_xy) * 1.1  # Small padding

        # Canvas setup
        canvas_size = 550
        padding = 20
        draw_list = imgui.get_window_draw_list()
        cursor_pos = imgui.get_cursor_screen_pos()
        x0, y0 = cursor_pos[0] + padding, cursor_pos[1] + padding

        # Get zoom and pan from mouse state
        zoom = mouse_state['zoom']
        pan = mouse_state['pan']

        # Coordinate transform functions (with zoom and pan)
        # Y flip required: matplotlib Y goes up, screen Y goes down
        def to_screen(p, x0, y0, canvas_size):
            center = (min_xy + max_xy) / 2
            normalized = (np.array(p) - center) / max_range + 0.5
            # Apply zoom and pan
            normalized = (normalized - 0.5) * zoom + 0.5 + np.array(pan)
            return (x0 + normalized[0] * canvas_size,
                    y0 + (1.0 - normalized[1]) * canvas_size)  # Flip Y to match matplotlib

        def from_screen(sx, sy, x0, y0, canvas_size):
            normalized_x = (sx - x0) / canvas_size
            normalized_y = 1.0 - (sy - y0) / canvas_size  # Flip Y back
            # Reverse zoom and pan
            normalized = np.array([normalized_x, normalized_y])
            normalized = (normalized - 0.5 - np.array(pan)) / zoom + 0.5
            center = (min_xy + max_xy) / 2
            return center + (normalized - 0.5) * max_range

        def find_line_contour_intersections(line_start, line_end, contour_2d):
            """Find intersection points of a line with a contour polygon."""
            intersections = []
            p1 = np.array(line_start)
            p2 = np.array(line_end)
            d = p2 - p1

            for i in range(len(contour_2d)):
                q1 = contour_2d[i]
                q2 = contour_2d[(i + 1) % len(contour_2d)]
                e = q2 - q1

                # Solve p1 + t*d = q1 + s*e
                denom = d[0] * e[1] - d[1] * e[0]
                if abs(denom) < 1e-10:
                    continue

                t = ((q1[0] - p1[0]) * e[1] - (q1[1] - p1[1]) * e[0]) / denom
                s = ((q1[0] - p1[0]) * d[1] - (q1[1] - p1[1]) * d[0]) / denom

                if 0 <= s <= 1:  # Intersection on contour edge
                    pt = p1 + t * d
                    intersections.append((t, pt))

            # Sort by t parameter and return points
            intersections.sort(key=lambda x: x[0])
            return [pt for _, pt in intersections]

        # Draw canvas background
        draw_list.add_rect_filled(x0 - padding, y0 - padding,
                                  x0 + canvas_size + padding, y0 + canvas_size + padding,
                                  imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 1.0))
        draw_list.add_rect(x0 - padding, y0 - padding,
                          x0 + canvas_size + padding, y0 + canvas_size + padding,
                          imgui.get_color_u32_rgba(0.5, 0.5, 0.5, 1.0))

        # Clip drawing to canvas area
        draw_list.push_clip_rect(x0 - padding, y0 - padding,
                                 x0 + canvas_size + padding, y0 + canvas_size + padding)

        # Draw all current pieces
        # Colors for different pieces: cycle through these
        piece_colors = [
            (1.0, 1.0, 1.0),  # White (first piece / uncut)
            (0.2, 0.6, 1.0),  # Blue
            (1.0, 0.4, 0.2),  # Orange
            (0.2, 1.0, 0.4),  # Green
            (1.0, 0.8, 0.2),  # Yellow
            (0.8, 0.2, 1.0),  # Purple
        ]
        current_pieces = obj._manual_cut_data.get('current_pieces', [target_2d])
        required_pieces = obj._manual_cut_data.get('required_pieces', 2)

        for piece_idx, piece_2d in enumerate(current_pieces):
            if len(piece_2d) >= 3:
                piece_screen = [to_screen(p, x0, y0, canvas_size) for p in piece_2d]
                color = piece_colors[piece_idx % len(piece_colors)]
                for i in range(len(piece_screen)):
                    p1, p2 = piece_screen[i], piece_screen[(i+1) % len(piece_screen)]
                    draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                      imgui.get_color_u32_rgba(*color, 1.0), 3.0)

                # Add label at centroid
                centroid = np.mean(piece_2d, axis=0)
                cx, cy = to_screen(centroid, x0, y0, canvas_size)
                if len(current_pieces) > 1:
                    # Multiple pieces - show piece index with colored background
                    label = f"P{piece_idx}"
                    # Draw background rect for visibility
                    draw_list.add_rect_filled(cx - 12, cy - 10, cx + 12, cy + 10,
                                              imgui.get_color_u32_rgba(*color, 0.8))
                    draw_list.add_text(cx - 8, cy - 7,
                                      imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 1.0), label)
                else:
                    # Single piece - show as target
                    draw_list.add_text(cx + 10, cy - 10,
                                      imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0),
                                      f"Target (Lv.{target_level})")

        # Draw transformed source contours after optimization (semi-transparent dashed outline)
        transformed_sources = obj._manual_cut_data.get('transformed_sources_2d', None) or []
        if len(transformed_sources) > 0:
            # Debug: print once when drawing
            if not hasattr(obj, '_debug_transformed_printed') or obj._debug_transformed_printed != len(transformed_sources):
                obj._debug_transformed_printed = len(transformed_sources)
                print(f"[Draw DEBUG] transformed_sources: {len(transformed_sources)}")
                for si, src in enumerate(transformed_sources):
                    if src is not None and len(src) > 0:
                        src_arr = np.array(src)
                        print(f"  S{si}: {len(src)} verts, first=[{src_arr[0,0]:.4f},{src_arr[0,1]:.4f}], centroid=[{src_arr.mean(axis=0)[0]:.4f},{src_arr.mean(axis=0)[1]:.4f}]")
            for src_idx, src_2d in enumerate(transformed_sources):
                if src_2d is not None and len(src_2d) >= 3:
                    src_color = piece_colors[src_idx % len(piece_colors)]
                    # Draw as dashed lines (semi-transparent)
                    for i in range(len(src_2d)):
                        p1 = to_screen(src_2d[i], x0, y0, canvas_size)
                        p2 = to_screen(src_2d[(i + 1) % len(src_2d)], x0, y0, canvas_size)
                        # Draw every other segment for dashed effect
                        if i % 2 == 0:
                            draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                              imgui.get_color_u32_rgba(*src_color, 0.5), 1.5)
                    # Draw centroid marker with S label
                    src_centroid = np.mean(src_2d, axis=0)
                    scx, scy = to_screen(src_centroid, x0, y0, canvas_size)
                    draw_list.add_circle(scx, scy, 8, imgui.get_color_u32_rgba(*src_color, 0.7), thickness=2.0)
                    draw_list.add_text(scx - 5, scy - 6,
                                      imgui.get_color_u32_rgba(*src_color, 0.9), f"S{src_idx}")

            # Draw actual cut points on target contour (where cutting lines cross target)
            # These are the real cut locations, not the source shared boundaries
            target_crossings = getattr(obj, '_target_cut_crossings_2d', [])

            line_colors = [
                imgui.get_color_u32_rgba(1.0, 0.0, 1.0, 1.0),  # Magenta
                imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 1.0),  # Cyan
                imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 1.0),  # Yellow
                imgui.get_color_u32_rgba(1.0, 0.5, 0.0, 1.0),  # Orange
            ]

            if target_crossings:
                # Group crossings by pair indices
                crossings_by_pair = {}
                for pt_2d, pair_indices in target_crossings:
                    if pair_indices not in crossings_by_pair:
                        crossings_by_pair[pair_indices] = []
                    crossings_by_pair[pair_indices].append(pt_2d)

                # Draw crossings for each pair
                for idx, (pair_indices, pts) in enumerate(crossings_by_pair.items()):
                    line_color = line_colors[idx % len(line_colors)]
                    # Draw each crossing point
                    for pt_2d in pts:
                        sp = to_screen(pt_2d, x0, y0, canvas_size)
                        draw_list.add_circle_filled(sp[0], sp[1], 6, line_color)
                    # If we have 2 crossings, draw line between them
                    if len(pts) >= 2:
                        p1 = to_screen(pts[0], x0, y0, canvas_size)
                        p2 = to_screen(pts[-1], x0, y0, canvas_size)
                        draw_list.add_line(p1[0], p1[1], p2[0], p2[1], line_color, 2.0)
                        mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                        draw_list.add_text(mid_x + 5, mid_y - 15, line_color, f"Cut {pair_indices[0]}-{pair_indices[1]}")
            else:
                # Fallback to old method using shared edges
                shared_edges_list = []
                if hasattr(obj, '_shared_cut_edges_2d') and obj._shared_cut_edges_2d:
                    shared_edges_list = obj._shared_cut_edges_2d
                elif hasattr(obj, '_shared_cut_edge_2d') and obj._shared_cut_edge_2d is not None:
                    shared_edges_list = [((0, 1), obj._shared_cut_edge_2d)]

                for idx, (pair_indices, shared_edge) in enumerate(shared_edges_list):
                    if shared_edge is not None and len(shared_edge) >= 2:
                        line_color = line_colors[idx % len(line_colors)]
                        p1 = to_screen(shared_edge[0], x0, y0, canvas_size)
                        p2 = to_screen(shared_edge[-1], x0, y0, canvas_size)
                        draw_list.add_line(p1[0], p1[1], p2[0], p2[1], line_color, 2.0)
                        draw_list.add_circle_filled(p1[0], p1[1], 4, line_color)
                        draw_list.add_circle_filled(p2[0], p2[1], 4, line_color)
                        mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                        draw_list.add_text(mid_x + 5, mid_y - 15, line_color, f"Cut {pair_indices[0]}-{pair_indices[1]}")

        # Draw all neck candidates (faded) and highlight selected one
        candidates = obj._manual_cut_data.get('neck_candidates', [])
        selected_neck_idx = obj._manual_cut_data.get('selected_neck_idx', 0)
        for cand_idx, cand in enumerate(candidates):
            pos_a = cand['pos_a']
            pos_b = cand['pos_b']
            screen_a = to_screen(pos_a, x0, y0, canvas_size)
            screen_b = to_screen(pos_b, x0, y0, canvas_size)
            mid_screen = ((screen_a[0] + screen_b[0]) / 2, (screen_a[1] + screen_b[1]) / 2)

            if cand_idx == selected_neck_idx:
                # Selected: bright cyan, thicker line
                color = imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 1.0)
                draw_list.add_line(screen_a[0], screen_a[1], screen_b[0], screen_b[1], color, 3.0)
                draw_list.add_circle_filled(screen_a[0], screen_a[1], 5, color)
                draw_list.add_circle_filled(screen_b[0], screen_b[1], 5, color)
                # Label
                draw_list.add_text(mid_screen[0] + 5, mid_screen[1] - 10,
                                  color, f"#{cand_idx} w={cand['width']:.4f}")
            else:
                # Non-selected: faded yellow
                color = imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 0.3)
                draw_list.add_line(screen_a[0], screen_a[1], screen_b[0], screen_b[1], color, 1.5)
                draw_list.add_circle_filled(screen_a[0], screen_a[1], 3, color)
                draw_list.add_circle_filled(screen_b[0], screen_b[1], 3, color)

        # Draw zoomed view indicator rectangle on main canvas
        neck_info_indicator = obj._manual_cut_data.get('current_neck_info', None)
        if neck_info_indicator is not None:
            line_start_ind = neck_info_indicator['line_start']
            line_end_ind = neck_info_indicator['line_end']
            piece_idx_ind = neck_info_indicator['piece_idx']
            current_pieces_ind = obj._manual_cut_data.get('current_pieces', [target_2d])

            if piece_idx_ind < len(current_pieces_ind):
                piece_2d_ind = current_pieces_ind[piece_idx_ind]

                # Calculate the same extent as the zoomed view
                target_line_pixels = 120.0
                zoom_canvas_size_ind = 200
                actual_line_length_ind = np.linalg.norm(line_end_ind - line_start_ind)
                if actual_line_length_ind > 1e-10:
                    world_extent_ind = actual_line_length_ind * (zoom_canvas_size_ind / target_line_pixels)
                else:
                    world_extent_ind = np.max(piece_2d_ind.max(axis=0) - piece_2d_ind.min(axis=0)) * 0.3

                neck_center_ind = (line_start_ind + line_end_ind) / 2
                half_extent = world_extent_ind / 2

                # Get the corners of the zoomed view in world coordinates
                corners_2d = [
                    neck_center_ind + np.array([-half_extent, -half_extent]),
                    neck_center_ind + np.array([half_extent, -half_extent]),
                    neck_center_ind + np.array([half_extent, half_extent]),
                    neck_center_ind + np.array([-half_extent, half_extent]),
                ]

                # Convert to screen coordinates and draw
                corners_screen = [to_screen(c, x0, y0, canvas_size) for c in corners_2d]
                indicator_color = imgui.get_color_u32_rgba(0.0, 1.0, 0.5, 0.7)
                for i in range(4):
                    p1 = corners_screen[i]
                    p2 = corners_screen[(i + 1) % 4]
                    draw_list.add_line(p1[0], p1[1], p2[0], p2[1], indicator_color, 2.0)

        # Draw selected source contour transparently on the main canvas
        source_contours_3d = obj._manual_cut_data.get('source_contours', [])
        source_bps = obj._manual_cut_data.get('source_bps', [])
        selected_sources = obj._manual_cut_data.get('selected_sources', list(range(len(source_contours_3d))))
        source_labels = obj._manual_cut_data.get('source_labels', list(range(len(source_contours_3d))))

        # Use pre-computed source_2d_list for drawing (same as PNG visualization)
        source_2d_list = obj._manual_cut_data.get('source_2d_list', [])

        if 'source_view_idx' in mouse_state and len(source_2d_list) > 0:
            src_idx = mouse_state.get('source_view_idx', 0)
            if src_idx < len(source_2d_list) and src_idx < len(source_bps):
                src_bp = source_bps[src_idx]
                # Use pre-computed 2D source contour (matches PNG visualization)
                src_on_target_2d = np.array(source_2d_list[src_idx])

                # Draw source contour transparently on main canvas
                src_color = piece_colors[src_idx % len(piece_colors)]
                is_selected = src_idx in selected_sources
                alpha = 0.6 if is_selected else 0.25
                thickness = 2.5 if is_selected else 1.5

                if len(src_on_target_2d) >= 3:
                    # Convert all points to screen coordinates
                    screen_pts = [to_screen(src_on_target_2d[i], x0, y0, canvas_size) for i in range(len(src_on_target_2d))]

                    # Fill polygon first (like PNG visualization does)
                    # Use triangulation for non-convex polygons
                    fill_alpha = 0.15 if is_selected else 0.05
                    try:
                        # Simple triangle fan from centroid for fill
                        src_centroid_screen = to_screen(src_on_target_2d.mean(axis=0), x0, y0, canvas_size)
                        for i in range(len(screen_pts)):
                            p1 = screen_pts[i]
                            p2 = screen_pts[(i + 1) % len(screen_pts)]
                            draw_list.add_triangle_filled(
                                src_centroid_screen[0], src_centroid_screen[1],
                                p1[0], p1[1], p2[0], p2[1],
                                imgui.get_color_u32_rgba(*src_color, fill_alpha))
                    except:
                        pass  # Skip fill if it fails

                    # Draw outline
                    for i in range(len(screen_pts)):
                        p1 = screen_pts[i]
                        p2 = screen_pts[(i + 1) % len(screen_pts)]
                        draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                          imgui.get_color_u32_rgba(*src_color, alpha), thickness)

                    # Draw centroid marker
                    src_centroid = src_on_target_2d.mean(axis=0)
                    cx, cy = to_screen(src_centroid, x0, y0, canvas_size)
                    draw_list.add_circle_filled(cx, cy, 6, imgui.get_color_u32_rgba(*src_color, alpha))
                    # Add X marker if not selected
                    if not is_selected:
                        draw_list.add_line(cx - 4, cy - 4, cx + 4, cy + 4, imgui.get_color_u32_rgba(1.0, 0.3, 0.3, 0.8), 2)
                        draw_list.add_line(cx - 4, cy + 4, cx + 4, cy - 4, imgui.get_color_u32_rgba(1.0, 0.3, 0.3, 0.8), 2)
                    # Add source label near centroid (use original index from source_labels)
                    src_label = source_labels[src_idx] if src_idx < len(source_labels) else src_idx
                    draw_list.add_text(cx + 10, cy + 5,
                                      imgui.get_color_u32_rgba(*src_color, 1.0),
                                      f"Source {src_label} (Lv.{source_level})")

        # End canvas clipping
        draw_list.pop_clip_rect()

        # Create invisible button to capture mouse input (prevents window dragging)
        imgui.set_cursor_screen_pos((x0 - padding, y0 - padding))
        imgui.invisible_button(f"canvas##{name}", canvas_size + 2 * padding, canvas_size + 2 * padding)
        canvas_hovered = imgui.is_item_hovered()
        canvas_active = imgui.is_item_active()

        # Handle mouse interaction for drawing line
        mouse_pos = imgui.get_mouse_pos()

        # Mouse wheel zoom
        if canvas_hovered:
            io = imgui.get_io()
            if io.mouse_wheel != 0:
                zoom_factor = 1.1 if io.mouse_wheel > 0 else 0.9
                # Zoom towards mouse position
                mouse_norm_x = (mouse_pos[0] - x0) / canvas_size
                mouse_norm_y = 1.0 - (mouse_pos[1] - y0) / canvas_size
                # Adjust pan to zoom towards mouse
                old_zoom = mouse_state['zoom']
                new_zoom = old_zoom * zoom_factor
                new_zoom = max(0.5, min(10.0, new_zoom))  # Clamp zoom
                if new_zoom != old_zoom:
                    # Pan adjustment to keep mouse point fixed
                    mouse_state['pan'][0] += (mouse_norm_x - 0.5) * (1 - zoom_factor / old_zoom * new_zoom) / new_zoom
                    mouse_state['pan'][1] += (mouse_norm_y - 0.5) * (1 - zoom_factor / old_zoom * new_zoom) / new_zoom
                    mouse_state['zoom'] = new_zoom

        # Middle mouse or right mouse pan
        if canvas_hovered and (imgui.is_mouse_clicked(2) or imgui.is_mouse_clicked(1)):  # Middle or right button
            mouse_state['panning'] = True
            mouse_state['pan_button'] = 2 if imgui.is_mouse_clicked(2) else 1  # Track which button
            mouse_state['pan_start'] = (mouse_pos[0], mouse_pos[1])
            mouse_state['pan_orig'] = mouse_state['pan'].copy()

        if mouse_state.get('panning', False):
            dx = (mouse_pos[0] - mouse_state['pan_start'][0]) / canvas_size
            dy = -(mouse_pos[1] - mouse_state['pan_start'][1]) / canvas_size  # Flip Y
            mouse_state['pan'][0] = mouse_state['pan_orig'][0] + dx
            mouse_state['pan'][1] = mouse_state['pan_orig'][1] + dy
            pan_button = mouse_state.get('pan_button', 2)
            if imgui.is_mouse_released(pan_button):
                mouse_state['panning'] = False

        # Left click for drawing cut line (disabled in optimization preview mode)
        is_preview_mode = obj._manual_cut_data.get('optimization_preview', False)
        if canvas_hovered and imgui.is_mouse_clicked(0) and not is_preview_mode:
            mouse_state['dragging'] = True
            mouse_state['start_pos'] = (mouse_pos[0], mouse_pos[1])
            mouse_state['end_pos'] = (mouse_pos[0], mouse_pos[1])

        if mouse_state['dragging'] and not is_preview_mode:
            end_x, end_y = mouse_pos[0], mouse_pos[1]
            # Check if shift is pressed for axis-aligned line
            shift_pressed = (glfw.get_key(v.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                             glfw.get_key(v.window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
            if shift_pressed:
                start_x, start_y = mouse_state['start_pos']
                dx = abs(end_x - start_x)
                dy = abs(end_y - start_y)
                # Snap to horizontal or vertical based on which direction is dominant
                if dx > dy:
                    end_y = start_y  # Horizontal line
                else:
                    end_x = start_x  # Vertical line
            mouse_state['end_pos'] = (end_x, end_y)
            if imgui.is_mouse_released(0):
                mouse_state['dragging'] = False
                # Convert to 2D coordinates and store the line
                start_2d = from_screen(mouse_state['start_pos'][0], mouse_state['start_pos'][1],
                                      x0, y0, canvas_size)
                end_2d = from_screen(mouse_state['end_pos'][0], mouse_state['end_pos'][1],
                                    x0, y0, canvas_size)
                obj._manual_cut_line = (tuple(start_2d), tuple(end_2d))
                obj._manual_cut_data['is_neck_line'] = False  # User drew this line manually

        # Clip cutting line drawing to canvas area
        draw_list.push_clip_rect(x0 - padding, y0 - padding,
                                 x0 + canvas_size + padding, y0 + canvas_size + padding)

        # Draw the cutting line (from click/drag points)
        if mouse_state['dragging'] and mouse_state['start_pos'] and mouse_state['end_pos']:
            # Draw line from drag start to current position
            draw_list.add_line(mouse_state['start_pos'][0], mouse_state['start_pos'][1],
                              mouse_state['end_pos'][0], mouse_state['end_pos'][1],
                              imgui.get_color_u32_rgba(1.0, 0.0, 1.0, 0.9), 3.0)
        elif obj._manual_cut_line is not None:
            # Draw stored line
            start_2d, end_2d = obj._manual_cut_line
            start_screen = to_screen(start_2d, x0, y0, canvas_size)
            end_screen = to_screen(end_2d, x0, y0, canvas_size)
            draw_list.add_line(start_screen[0], start_screen[1],
                              end_screen[0], end_screen[1],
                              imgui.get_color_u32_rgba(1.0, 0.0, 1.0, 0.9), 3.0)

            # Draw cut preview if we have a valid line
            piece0_2d, piece1_2d = obj._compute_cut_preview()
            if piece0_2d is not None and piece1_2d is not None:
                # Draw piece 0 (color 0)
                if len(piece0_2d) >= 3:
                    p0_screen = [to_screen(p, x0, y0, canvas_size) for p in piece0_2d]
                    for i in range(len(p0_screen)):
                        p1, p2 = p0_screen[i], p0_screen[(i+1) % len(p0_screen)]
                        draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                          imgui.get_color_u32_rgba(piece_colors[0][0], piece_colors[0][1], piece_colors[0][2], 1.0), 2.5)
                # Draw piece 1 (color 1)
                if len(piece1_2d) >= 3:
                    p1_screen = [to_screen(p, x0, y0, canvas_size) for p in piece1_2d]
                    for i in range(len(p1_screen)):
                        pt1, pt2 = p1_screen[i], p1_screen[(i+1) % len(p1_screen)]
                        draw_list.add_line(pt1[0], pt1[1], pt2[0], pt2[1],
                                          imgui.get_color_u32_rgba(piece_colors[1][0], piece_colors[1][1], piece_colors[1][2], 1.0), 2.5)

        # End cutting line clipping
        draw_list.pop_clip_rect()

        # ===== Source Contour Viewer Panel (right side) =====
        source_contours_3d = obj._manual_cut_data.get('source_contours', [])
        source_labels = obj._manual_cut_data.get('source_labels', list(range(len(source_contours_3d))))
        if len(source_contours_3d) > 0:
            # Initialize source viewer state
            if 'source_view_idx' not in mouse_state:
                mouse_state['source_view_idx'] = 0

            # Source viewer panel position (to the right of main canvas)
            src_panel_x = x0 + canvas_size + padding + 30
            src_panel_y = y0 - padding
            src_canvas_size = 280
            src_padding = 15

            # Draw panel background (taller to fit checkbox and summary)
            draw_list.add_rect_filled(
                src_panel_x - src_padding, src_panel_y - src_padding,
                src_panel_x + src_canvas_size + src_padding, src_panel_y + src_canvas_size + src_padding + 120,
                imgui.get_color_u32_rgba(0.12, 0.12, 0.12, 1.0))
            draw_list.add_rect(
                src_panel_x - src_padding, src_panel_y - src_padding,
                src_panel_x + src_canvas_size + src_padding, src_panel_y + src_canvas_size + src_padding + 120,
                imgui.get_color_u32_rgba(0.4, 0.4, 0.4, 1.0))

            # Title with level info
            draw_list.add_text(src_panel_x, src_panel_y - src_padding + 5,
                              imgui.get_color_u32_rgba(0.9, 0.9, 0.9, 1.0),
                              f"Source Contours (Lv.{source_level}, {len(source_contours_3d)} total)")

            # Adjust canvas position for title
            src_canvas_y = src_panel_y + 20

            # Get current source contour to display
            src_idx = mouse_state['source_view_idx']
            src_idx = max(0, min(src_idx, len(source_contours_3d) - 1))
            mouse_state['source_view_idx'] = src_idx

            # Project source contour to 2D using its bounding plane
            source_bps = obj._manual_cut_data.get('source_bps', [])
            if src_idx < len(source_bps):
                src_bp = source_bps[src_idx]
                src_contour_3d = source_contours_3d[src_idx]
                src_mean = src_bp['mean']
                src_basis_x = src_bp['basis_x']
                src_basis_y = src_bp['basis_y']

                # Project to 2D using TARGET's basis for consistent rotation with cutting panel
                target_bp_for_src = obj._manual_cut_data.get('target_bp')
                if target_bp_for_src is not None:
                    tgt_mean = target_bp_for_src['mean']
                    tgt_basis_x = target_bp_for_src['basis_x']
                    tgt_basis_y = target_bp_for_src['basis_y']
                    tgt_basis_z = target_bp_for_src['basis_z']
                    # Project source onto target plane, then to 2D
                    src_2d = []
                    for pt in src_contour_3d:
                        diff = pt - tgt_mean
                        dist_along_normal = np.dot(diff, tgt_basis_z)
                        pt_on_plane = pt - dist_along_normal * tgt_basis_z
                        diff_on_plane = pt_on_plane - tgt_mean
                        x = np.dot(diff_on_plane, tgt_basis_x)
                        y = np.dot(diff_on_plane, tgt_basis_y)
                        src_2d.append([x, y])
                    src_2d = np.array(src_2d)
                else:
                    # Fallback to source's own basis
                    src_2d = []
                    for pt in src_contour_3d:
                        diff = pt - src_mean
                        x = np.dot(diff, src_basis_x)
                        y = np.dot(diff, src_basis_y)
                        src_2d.append([x, y])
                    src_2d = np.array(src_2d)

                # Compute bounds for source contour
                src_min = src_2d.min(axis=0)
                src_max = src_2d.max(axis=0)
                src_range = src_max - src_min
                src_max_range = max(src_range) * 1.2 if max(src_range) > 0 else 1.0

                # Transform function for source canvas
                def src_to_screen(p):
                    center = (src_min + src_max) / 2
                    normalized = (np.array(p) - center) / src_max_range + 0.5
                    return (src_panel_x + normalized[0] * src_canvas_size,
                            src_canvas_y + (1.0 - normalized[1]) * src_canvas_size)

                # Draw source canvas background
                draw_list.add_rect_filled(
                    src_panel_x, src_canvas_y,
                    src_panel_x + src_canvas_size, src_canvas_y + src_canvas_size,
                    imgui.get_color_u32_rgba(0.08, 0.08, 0.08, 1.0))

                # Draw source contour with matching color
                src_color = piece_colors[src_idx % len(piece_colors)]
                if len(src_2d) >= 3:
                    for i in range(len(src_2d)):
                        p1 = src_to_screen(src_2d[i])
                        p2 = src_to_screen(src_2d[(i + 1) % len(src_2d)])
                        draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                          imgui.get_color_u32_rgba(*src_color, 1.0), 2.5)

                # Draw centroid marker
                src_centroid = src_2d.mean(axis=0)
                cx, cy = src_to_screen(src_centroid)
                draw_list.add_circle_filled(cx, cy, 5, imgui.get_color_u32_rgba(*src_color, 0.8))

            # Navigation controls below the source canvas
            nav_y = src_canvas_y + src_canvas_size + 10
            imgui.set_cursor_screen_pos((src_panel_x, nav_y))

            # Left/Right buttons and slider
            imgui.push_item_width(src_canvas_size)
            if len(source_contours_3d) > 1:
                # Left button
                if imgui.button(f"<##{name}_src_prev", 30, 25):
                    mouse_state['source_view_idx'] = max(0, src_idx - 1)
                imgui.same_line()

                # Slider
                imgui.push_item_width(src_canvas_size - 80)
                changed, new_idx = imgui.slider_int(f"##{name}_src_slider", src_idx, 0, len(source_contours_3d) - 1)
                if changed:
                    mouse_state['source_view_idx'] = new_idx
                imgui.pop_item_width()
                imgui.same_line()

                # Right button
                if imgui.button(f">##{name}_src_next", 30, 25):
                    mouse_state['source_view_idx'] = min(len(source_contours_3d) - 1, src_idx + 1)

            # Show source index label (use original index from source_labels)
            imgui.set_cursor_screen_pos((src_panel_x, nav_y + 30))
            src_label_color = piece_colors[src_idx % len(piece_colors)]
            src_orig_label = source_labels[src_idx] if src_idx < len(source_labels) else src_idx
            imgui.text_colored(f"Source {src_orig_label} ({src_idx + 1}/{len(source_contours_3d)})", *src_label_color, 1.0)

            # Checkbox to include/exclude this source
            imgui.set_cursor_screen_pos((src_panel_x, nav_y + 50))

            # Initialize selected_sources if not present
            if 'selected_sources' not in obj._manual_cut_data:
                obj._manual_cut_data['selected_sources'] = list(range(len(source_contours_3d)))

            selected_sources = obj._manual_cut_data['selected_sources']
            is_selected = src_idx in selected_sources

            changed, new_selected = imgui.checkbox(f"Include S{src_orig_label}##{name}", is_selected)
            if changed:
                if new_selected and src_idx not in selected_sources:
                    selected_sources.append(src_idx)
                    selected_sources.sort()
                elif not new_selected and src_idx in selected_sources:
                    # Don't allow deselecting if only 1 source remains
                    if len(selected_sources) > 1:
                        selected_sources.remove(src_idx)
                    else:
                        print("Cannot deselect: at least 1 source required")

                # Update required_pieces and reset cuts
                obj._manual_cut_data['selected_sources'] = selected_sources
                new_required = len(selected_sources)
                obj._manual_cut_data['required_pieces'] = new_required

                # Reset current pieces to original target
                obj._manual_cut_data['current_pieces'] = [target_2d.copy()]
                obj._manual_cut_data['current_pieces_3d'] = [obj._manual_cut_data['target_contour'].copy()]
                obj._manual_cut_data['cut_lines'] = []
                if 'initial_line' in obj._manual_cut_data:
                    del obj._manual_cut_data['initial_line']
                obj._manual_cut_line = None

                print(f"Source selection changed: {len(selected_sources)} sources selected, need {new_required} pieces")

            # Show selection summary
            imgui.set_cursor_screen_pos((src_panel_x, nav_y + 70))
            num_selected = len(obj._manual_cut_data.get('selected_sources', []))
            imgui.text(f"Selected: {num_selected} / {len(source_contours_3d)}")

            imgui.pop_item_width()

            # ===== Zoomed Neck View Panel (below source panel) =====
            neck_info = obj._manual_cut_data.get('current_neck_info', None)
            if neck_info is not None:
                # Panel position (below source panel)
                zoom_panel_x = src_panel_x - src_padding
                zoom_panel_y = src_panel_y + src_canvas_size + src_padding + 130
                zoom_canvas_size = 200
                zoom_padding = 10

                # Draw panel background
                draw_list.add_rect_filled(
                    zoom_panel_x, zoom_panel_y,
                    zoom_panel_x + src_canvas_size + src_padding * 2, zoom_panel_y + zoom_canvas_size + 50,
                    imgui.get_color_u32_rgba(0.1, 0.1, 0.1, 1.0))
                draw_list.add_rect(
                    zoom_panel_x, zoom_panel_y,
                    zoom_panel_x + src_canvas_size + src_padding * 2, zoom_panel_y + zoom_canvas_size + 50,
                    imgui.get_color_u32_rgba(0.5, 0.5, 0.5, 1.0))

                # Title
                draw_list.add_text(zoom_panel_x + zoom_padding, zoom_panel_y + 5,
                                  imgui.get_color_u32_rgba(0.9, 0.9, 0.9, 1.0),
                                  "Neck Detail (Zoomed)")

                # Get neck line info
                line_start = neck_info['line_start']
                line_end = neck_info['line_end']
                piece_idx = neck_info['piece_idx']
                current_pieces_zoom = obj._manual_cut_data.get('current_pieces', [target_2d])

                if piece_idx < len(current_pieces_zoom):
                    piece_2d = current_pieces_zoom[piece_idx]

                    # Calculate zoom level to keep line at fixed pixel length (e.g., 120 pixels)
                    target_line_pixels = 120.0
                    actual_line_length = np.linalg.norm(line_end - line_start)
                    if actual_line_length > 1e-10:
                        # Calculate the world space extent that should fit in target_line_pixels
                        world_extent = actual_line_length * (zoom_canvas_size / target_line_pixels)
                    else:
                        world_extent = np.max(piece_2d.max(axis=0) - piece_2d.min(axis=0)) * 0.3

                    # Center on neck midpoint
                    neck_center = (line_start + line_end) / 2

                    # Zoomed canvas position
                    zcanvas_x = zoom_panel_x + zoom_padding
                    zcanvas_y = zoom_panel_y + 25
                    zcanvas_size = zoom_canvas_size

                    # Draw zoom canvas background
                    draw_list.add_rect_filled(
                        zcanvas_x, zcanvas_y,
                        zcanvas_x + zcanvas_size, zcanvas_y + zcanvas_size,
                        imgui.get_color_u32_rgba(0.05, 0.05, 0.05, 1.0))

                    # Clip to zoom canvas
                    draw_list.push_clip_rect(zcanvas_x, zcanvas_y,
                                             zcanvas_x + zcanvas_size, zcanvas_y + zcanvas_size)

                    # Transform function for zoomed view
                    def to_zoom_screen(p):
                        offset = np.array(p) - neck_center
                        normalized = offset / world_extent + 0.5
                        return (zcanvas_x + normalized[0] * zcanvas_size,
                                zcanvas_y + (1.0 - normalized[1]) * zcanvas_size)

                    # Draw the piece contour (faded white)
                    if len(piece_2d) >= 3:
                        for i in range(len(piece_2d)):
                            p1 = to_zoom_screen(piece_2d[i])
                            p2 = to_zoom_screen(piece_2d[(i + 1) % len(piece_2d)])
                            draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                              imgui.get_color_u32_rgba(0.6, 0.6, 0.6, 0.8), 1.5)

                    # Draw cutting line (magenta)
                    zstart = to_zoom_screen(line_start)
                    zend = to_zoom_screen(line_end)
                    draw_list.add_line(zstart[0], zstart[1], zend[0], zend[1],
                                      imgui.get_color_u32_rgba(1.0, 0.0, 1.0, 1.0), 3.0)

                    # Draw endpoints (on contour)
                    draw_list.add_circle_filled(zstart[0], zstart[1], 5,
                                               imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 1.0))
                    draw_list.add_circle_filled(zend[0], zend[1], 5,
                                               imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 1.0))

                    # Draw neck vertices (cyan)
                    neck_a = neck_info['neck_a']
                    neck_b = neck_info['neck_b']
                    za = to_zoom_screen(neck_a)
                    zb = to_zoom_screen(neck_b)
                    draw_list.add_circle_filled(za[0], za[1], 4,
                                               imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 1.0))
                    draw_list.add_circle_filled(zb[0], zb[1], 4,
                                               imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 1.0))

                    # End clip
                    draw_list.pop_clip_rect()

                    # Show info below zoomed canvas
                    info_y = zcanvas_y + zcanvas_size + 5
                    draw_list.add_text(zcanvas_x, info_y,
                                      imgui.get_color_u32_rgba(0.7, 0.7, 0.7, 1.0),
                                      f"Line len: {actual_line_length:.4f}")

        # Get piece info (use selected_sources count for required_pieces)
        current_pieces = obj._manual_cut_data.get('current_pieces', [target_2d])
        selected_sources = obj._manual_cut_data.get('selected_sources', list(range(len(obj._manual_cut_data.get('source_contours', [])))))
        required_pieces = len(selected_sources) if len(selected_sources) > 0 else obj._manual_cut_data.get('required_pieces', 2)
        source_indices_display = obj._manual_cut_data.get('source_indices', [])

        # Show source->target info
        imgui.set_cursor_screen_pos((cursor_pos[0], y0 + canvas_size + padding + 10))
        imgui.separator()
        num_selected = len(selected_sources)
        imgui.text(f"Selected: {num_selected} sources -> 1 target")
        imgui.text(f"Pieces: {len(current_pieces)} / {required_pieces} needed")

        # Show preview mode indicator
        in_preview_mode = obj._manual_cut_data.get('optimization_preview', False)
        if in_preview_mode:
            edit_count = len(obj._manual_cut_data.get('edit_history', []))
            if edit_count > 0:
                imgui.text_colored(f"[PREVIEW MODE] {edit_count} edit(s) - Undo/Accept/Reset", 1.0, 0.8, 0.2, 1.0)
            else:
                imgui.text_colored("[PREVIEW MODE] Draw lines to edit, or Accept/Reset", 1.0, 0.8, 0.2, 1.0)

            # Show piece-to-source mapping
            current_pieces_3d_preview = obj._manual_cut_data.get('current_pieces_3d', [])
            source_contours_preview = obj._manual_cut_data.get('source_contours', [])

            if len(current_pieces_3d_preview) > 0 and len(source_contours_preview) > 0:
                imgui.separator()
                imgui.text("Piece -> Source mapping:")

                # Compute centroids for matching
                piece_centroids = [np.mean(p, axis=0) for p in current_pieces_3d_preview]
                source_centroids = [np.mean(s, axis=0) for s in source_contours_preview]

                # Piece colors
                piece_colors = [
                    (1.0, 1.0, 1.0, 1.0),  # White
                    (0.2, 0.6, 1.0, 1.0),  # Blue
                    (1.0, 0.4, 0.2, 1.0),  # Orange
                    (0.2, 1.0, 0.4, 1.0),  # Green
                    (1.0, 0.8, 0.2, 1.0),  # Yellow
                    (0.8, 0.2, 1.0, 1.0),  # Purple
                ]

                # Match each piece to closest source
                source_labels = obj._manual_cut_data.get('source_labels', list(range(len(source_contours_preview))))
                used_sources = set()
                for piece_idx, piece_centroid in enumerate(piece_centroids):
                    best_src = None
                    best_dist = float('inf')
                    for src_idx, src_centroid in enumerate(source_centroids):
                        if src_idx in used_sources:
                            continue
                        dist = np.linalg.norm(piece_centroid - src_centroid)
                        if dist < best_dist:
                            best_dist = dist
                            best_src = src_idx

                    if best_src is not None:
                        used_sources.add(best_src)
                        src_label = source_labels[best_src] if best_src < len(source_labels) else best_src
                        color = piece_colors[piece_idx % len(piece_colors)]
                        imgui.text_colored(f"  P{piece_idx} ({len(current_pieces_3d_preview[piece_idx])} verts) -> S{src_label}", *color)

            # Show optimization scales
            if hasattr(obj, '_bp_viz_data') and len(obj._bp_viz_data) > 0:
                latest_viz = obj._bp_viz_data[-1]
                opt_scales = latest_viz.get('scales', [])
                if len(opt_scales) > 0:
                    imgui.separator()
                    imgui.text("Optimization scales:")
                    for si, scale in enumerate(opt_scales):
                        if isinstance(scale, (list, tuple)) and len(scale) == 2:
                            imgui.text(f"  Source {si}: ({scale[0]:.3f}, {scale[1]:.3f})")
                        else:
                            imgui.text(f"  Source {si}: {scale:.3f}")

            imgui.separator()

        # Show hint for 1-to-1 case
        if num_selected == 1:
            imgui.text_colored("(1:1 mapping - no cutting needed, click Skip)", 0.5, 1.0, 0.5, 1.0)

        # Buttons: Optimize, Optimize All, Skip, Next Cut, OK, Reset, Cancel
        imgui.separator()
        button_width = 80

        # Check if auto-optimize-all mode is active
        auto_optimize_all = getattr(obj, '_auto_optimize_all', False)
        should_auto_optimize = auto_optimize_all and num_selected >= 2

        # Optimize button - run automatic optimization on current pieces
        # Can be used after manual cuts to optimize remaining subdivisions
        # Hide in preview mode (Accept/Reset are shown instead) and in assignment mode
        in_preview_for_button = obj._manual_cut_data.get('optimization_preview', False) if obj._manual_cut_data else False
        in_assignment_mode_for_button = obj._manual_cut_data.get('assignment_mode', False) if obj._manual_cut_data else False
        if num_selected >= 2 and not in_preview_for_button and not in_assignment_mode_for_button:
            # Determine if we should run optimization (button click or auto-optimize mode)
            run_optimization = False
            skip_preview = False  # For Optimize All: skip preview and auto-accept

            if imgui.button("Optimize", button_width, 30):
                run_optimization = True
            imgui.same_line()

            # Optimize All button - optimize and accept without preview
            if imgui.button("Opt All", button_width, 30):
                run_optimization = True
                skip_preview = True
                obj._auto_optimize_all = True  # Set flag for subsequent windows
                print(f"[OptAll] Starting Optimize All mode...")
            imgui.same_line()

            # Auto-trigger optimization if in auto-optimize-all mode
            if should_auto_optimize and not run_optimization:
                run_optimization = True
                skip_preview = True
                print(f"[OptAll] Auto-triggering optimization...")

            if run_optimization:
                # Get source data
                source_contours = obj._manual_cut_data.get('source_contours', [])
                source_bps = obj._manual_cut_data.get('source_bps', [])
                target_bp = obj._manual_cut_data.get('target_bp')
                stream_indices = obj._manual_cut_data.get('stream_indices', list(range(len(source_contours))))
                target_level = obj._manual_cut_data.get('target_level')
                source_level = obj._manual_cut_data.get('source_level')
                current_pieces_3d = obj._manual_cut_data.get('current_pieces_3d', [obj._manual_cut_data.get('target_contour')])

                print(f"Running optimization on {len(current_pieces_3d)} current pieces -> {required_pieces} needed...")

                # Filter by selected sources
                selected_source_contours = [source_contours[i] for i in selected_sources if i < len(source_contours)]
                selected_source_bps = [source_bps[i] for i in selected_sources if i < len(source_bps)]
                selected_stream_indices = [stream_indices[i] for i in selected_sources if i < len(stream_indices)]

                if len(selected_source_contours) >= 2:
                    # Get matched pairs for combining with optimized pieces
                    matched_pairs = obj._manual_cut_data.get('matched_pairs', [])

                    # Optimize remaining (unmatched) pieces
                    final_pieces = obj._optimize_remaining_pieces(
                        current_pieces_3d,
                        selected_source_contours, selected_source_bps,
                        selected_stream_indices, target_bp,
                        target_level, source_level,
                        matched_pairs=matched_pairs
                    )

                    if final_pieces is not None and len(final_pieces) > 0:
                        # Convert 3D pieces to 2D for display
                        target_mean = target_bp['mean']
                        # Negate basis vectors to rotate 180° - match initial cutting window
                        target_x = -target_bp['basis_x']
                        target_y = -target_bp['basis_y']

                        print(f"[Optimize DEBUG] Converting {len(final_pieces)} pieces to 2D")
                        optimized_2d = []
                        for pi, piece_3d in enumerate(final_pieces):
                            piece_2d = np.array([
                                [np.dot(v - target_mean, target_x), np.dot(v - target_mean, target_y)]
                                for v in piece_3d
                            ])
                            optimized_2d.append(piece_2d)
                            print(f"  Piece {pi}: {len(piece_3d)} verts 3D -> {len(piece_2d)} verts 2D")
                            if len(piece_2d) > 0:
                                centroid_2d = np.mean(piece_2d, axis=0)
                                print(f"    2D centroid: ({centroid_2d[0]:.4f}, {centroid_2d[1]:.4f})")

                        # Update current pieces with optimized result
                        obj._manual_cut_data['current_pieces'] = optimized_2d
                        obj._manual_cut_data['current_pieces_3d'] = list(final_pieces)
                        obj._manual_cut_data['pre_optimization_pieces'] = current_pieces.copy()
                        obj._manual_cut_data['pre_optimization_pieces_3d'] = [p.copy() for p in current_pieces_3d]
                        obj._manual_cut_data['cut_lines'] = []
                        obj._manual_cut_data['edit_history'] = []
                        obj._manual_cut_line = None

                        # Store transformed source contours for display on cutting panel
                        # Also update target_2d, source_2d_list, and current_pieces to match the coordinate system
                        if hasattr(obj, '_bp_viz_data') and len(obj._bp_viz_data) > 0:
                            latest_viz = obj._bp_viz_data[-1]
                            print(f"[Optimize DEBUG] _bp_viz_data has {len(obj._bp_viz_data)} entries, latest keys: {list(latest_viz.keys())}")
                            transformed_srcs = latest_viz.get('final_transformed', [])
                            obj._manual_cut_data['transformed_sources_2d'] = transformed_srcs

                            # Update target_2d to match optimization's coordinate system
                            opt_target_2d = latest_viz.get('target_2d', None)
                            old_target_2d = obj._manual_cut_data.get('target_2d', None)
                            if opt_target_2d is not None:
                                # Compare old vs new target_2d
                                if old_target_2d is not None and hasattr(old_target_2d, '__len__') and len(old_target_2d) > 0:
                                    old_centroid = np.mean(old_target_2d, axis=0)
                                    new_centroid = np.mean(opt_target_2d, axis=0)
                                    print(f"[Optimize DEBUG] target_2d comparison:")
                                    print(f"  old: {len(old_target_2d)} verts, centroid=({old_centroid[0]:.4f}, {old_centroid[1]:.4f}), first=({old_target_2d[0][0]:.4f}, {old_target_2d[0][1]:.4f})")
                                    print(f"  new: {len(opt_target_2d)} verts, centroid=({new_centroid[0]:.4f}, {new_centroid[1]:.4f}), first=({opt_target_2d[0][0]:.4f}, {opt_target_2d[0][1]:.4f})")
                                else:
                                    print(f"[Optimize DEBUG] old_target_2d is None or empty: {type(old_target_2d)}, hasattr len: {hasattr(old_target_2d, '__len__')}")
                                obj._manual_cut_data['target_2d'] = opt_target_2d
                                print(f"[Optimize DEBUG] Updated target_2d: {len(opt_target_2d)} verts")
                            else:
                                print(f"[Optimize DEBUG] WARNING: opt_target_2d is None!")

                            # Update source_2d_list to match (use source_2d_shapes + translations)
                            source_shapes = latest_viz.get('source_2d_shapes', [])
                            init_trans = latest_viz.get('initial_translations', [])
                            print(f"[Optimize DEBUG] source_shapes: {len(source_shapes)}, init_trans: {len(init_trans)}")
                            if len(source_shapes) > 0 and len(init_trans) == len(source_shapes):
                                new_source_2d_list = []
                                for i, (shape, trans) in enumerate(zip(source_shapes, init_trans)):
                                    # Reconstruct original source position (before any transform)
                                    src_2d = np.array(shape) + np.array(trans)
                                    new_source_2d_list.append(src_2d)
                                obj._manual_cut_data['source_2d_list'] = new_source_2d_list
                                print(f"[Optimize DEBUG] Updated source_2d_list: {len(new_source_2d_list)} sources")

                            # Use pieces_2d directly from optimization (same coordinate system)
                            opt_pieces_2d = latest_viz.get('pieces_2d', [])
                            if len(opt_pieces_2d) > 0:
                                obj._manual_cut_data['current_pieces'] = opt_pieces_2d
                                print(f"[Optimize DEBUG] Using pieces_2d from optimization ({len(opt_pieces_2d)} pieces)")
                                for pi, pc in enumerate(opt_pieces_2d):
                                    pc_centroid = np.mean(pc, axis=0)
                                    print(f"  Piece {pi}: {len(pc)} verts, centroid ({pc_centroid[0]:.4f}, {pc_centroid[1]:.4f})")
                            else:
                                # Fallback to projecting from 3D (old behavior)
                                obj._manual_cut_data['current_pieces'] = optimized_2d

                            print(f"[Optimize DEBUG] Transformed sources: {len(transformed_srcs)}")
                            for si, src in enumerate(transformed_srcs):
                                if src is not None and len(src) > 0:
                                    src_centroid = np.mean(src, axis=0)
                                    print(f"  Source {si}: {len(src)} verts, centroid ({src_centroid[0]:.4f}, {src_centroid[1]:.4f})")

                        if skip_preview:
                            # Optimize All: trigger auto-accept flag
                            obj._manual_cut_data['optimization_preview'] = True
                            obj._manual_cut_data['auto_accept'] = True
                            print(f"[OptAll] Optimization produced {len(final_pieces)} pieces - auto-accepting...")
                        else:
                            # Normal: enter preview mode
                            obj._manual_cut_data['optimization_preview'] = True
                            print(f"[Preview] Optimization produced {len(final_pieces)} pieces - enter preview mode")
                            print(f"[Preview] You can now Accept, Reset, or edit with cutting lines")
                    else:
                        print(f"[Error] Optimization failed to produce pieces")
                else:
                    print(f"Need at least 2 source contours for optimization")

        # ===== Preview Mode Buttons: Accept / Reset =====
        in_preview_mode = obj._manual_cut_data.get('optimization_preview', False) if obj._manual_cut_data else False
        if in_preview_mode:
            # Check for auto-accept (from Optimize All)
            auto_accept = obj._manual_cut_data.get('auto_accept', False)

            # Accept button - finalize the optimization result
            # Also triggered automatically in Optimize All mode
            # Don't show button if auto-accepting (cleaner UI)
            if auto_accept:
                accept_clicked = True
            else:
                accept_clicked = imgui.button("Accept", button_width, 30)
            if accept_clicked:
                if auto_accept:
                    obj._manual_cut_data['auto_accept'] = False  # Clear flag
                print(f"[Preview] Accepting optimization result...")

                # Get the current optimized pieces
                current_pieces_3d = obj._manual_cut_data.get('current_pieces_3d', [])
                # final_pieces will be set later to exclude matched pieces

                # Get context data
                source_contours = obj._manual_cut_data.get('source_contours', [])
                source_bps = obj._manual_cut_data.get('source_bps', [])
                target_bp = obj._manual_cut_data.get('target_bp')
                stream_indices = obj._manual_cut_data.get('stream_indices', list(range(len(source_contours))))
                target_level = obj._manual_cut_data.get('target_level')
                source_level = obj._manual_cut_data.get('source_level')
                matched_pairs = obj._manual_cut_data.get('matched_pairs', [])
                parent_finalized_pieces = obj._manual_cut_data.get('parent_finalized_pieces', {})
                original_source_indices = obj._manual_cut_data.get('original_source_indices', None)
                selected_sources = obj._manual_cut_data.get('selected_sources', list(range(len(source_contours))))

                # Exit preview mode
                obj._manual_cut_data['optimization_preview'] = False

                # Now run the finalization code
                # Use sub-window context if original_source_indices is set (indicating we're in a sub-cut)
                # Note: parent_finalized_pieces may be empty if no 1:1 matches from parent
                if original_source_indices:
                    # Sub-window context: need to combine with parent's 1:1 pieces
                    source_contours_full = obj._manual_cut_data.get('source_contours', [])
                    # Compute total_pieces from max stream index (NOT just count)
                    # Stream indices can be larger than count-1, e.g., streams [3,1,2] with count 3
                    # needs array size 4 to fit index 3
                    parent_context = obj._manual_cut_data.get('parent_context', None)
                    all_stream_indices_for_size = parent_context.get('stream_indices', []) if parent_context else []
                    max_parent_idx = max(parent_finalized_pieces.keys()) if parent_finalized_pieces else -1
                    max_current_idx = max(original_source_indices) if original_source_indices else -1
                    max_all_streams = max(all_stream_indices_for_size) if all_stream_indices_for_size else -1
                    total_pieces = max(max_parent_idx, max_current_idx, max_all_streams) + 1
                    all_pieces = [None] * total_pieces
                    print(f"[Accept] Using total_pieces={total_pieces} (from max stream index)")

                    # Fill in parent's finalized pieces (1:1 assignments from parent)
                    print(f"[Accept DEBUG] parent_finalized_pieces has {len(parent_finalized_pieces)} entries:")
                    for orig_src_idx, piece in parent_finalized_pieces.items():
                        if orig_src_idx < total_pieces:
                            all_pieces[orig_src_idx] = piece
                        piece_id = id(piece) if piece is not None else 0
                        piece_verts = len(piece) if piece is not None else 0
                        piece_centroid = np.mean(piece, axis=0) if piece is not None and len(piece) > 0 else [0,0,0]
                        print(f"[Accept DEBUG]   parent_finalized[{orig_src_idx}]: {piece_verts} verts, id={piece_id}, centroid={piece_centroid}")

                    # Fill in pre-matched pieces (mapped to original indices)
                    matched_piece_indices = set()
                    for piece_idx, local_src_idx in matched_pairs:
                        if piece_idx < len(current_pieces_3d) and local_src_idx < len(original_source_indices):
                            orig_src_idx = original_source_indices[local_src_idx]
                            if orig_src_idx < total_pieces:
                                all_pieces[orig_src_idx] = current_pieces_3d[piece_idx]
                                matched_piece_indices.add(piece_idx)

                    # Build list of unmatched pieces (excluding already-matched ones)
                    unmatched_pieces = [current_pieces_3d[i] for i in range(len(current_pieces_3d))
                                       if i not in matched_piece_indices]

                    # Fill in optimized pieces for remaining sources (mapped to original indices)
                    opt_idx = 0
                    for local_src_idx in selected_sources:
                        if local_src_idx < len(original_source_indices):
                            orig_src_idx = original_source_indices[local_src_idx]
                            if orig_src_idx < total_pieces and all_pieces[orig_src_idx] is None:
                                if opt_idx < len(unmatched_pieces):
                                    all_pieces[orig_src_idx] = unmatched_pieces[opt_idx]
                                    opt_idx += 1

                    # Get original stream indices from parent context
                    # IMPORTANT: Use the actual stream indices, not just [0, 1, ..., M-1]
                    parent_context = obj._manual_cut_data.get('parent_context', None)
                    if parent_context and 'stream_indices' in parent_context:
                        all_stream_indices = parent_context['stream_indices']
                    else:
                        # Fallback to initial stream_indices if parent_context not available
                        all_stream_indices = obj._manual_cut_data.get('stream_indices', list(range(total_pieces)))
                else:
                    # Normal context (not a sub-window)
                    source_contours_full = obj._manual_cut_data.get('source_contours', [])
                    all_pieces = [None] * len(source_contours_full)

                    # Fill in pre-matched pieces
                    matched_piece_indices = set()
                    for piece_idx, src_idx in matched_pairs:
                        if piece_idx < len(current_pieces_3d) and src_idx < len(all_pieces):
                            all_pieces[src_idx] = current_pieces_3d[piece_idx]
                            matched_piece_indices.add(piece_idx)

                    # Build list of unmatched pieces (excluding already-matched ones)
                    unmatched_pieces = [current_pieces_3d[i] for i in range(len(current_pieces_3d))
                                       if i not in matched_piece_indices]

                    # Fill in optimized pieces for remaining sources
                    opt_idx = 0
                    for src_idx in selected_sources:
                        if src_idx < len(all_pieces) and all_pieces[src_idx] is None and opt_idx < len(unmatched_pieces):
                            all_pieces[src_idx] = unmatched_pieces[opt_idx]
                            opt_idx += 1

                    all_stream_indices = obj._manual_cut_data.get('stream_indices', stream_indices)

                # Check if we have all pieces
                # Note: all_pieces may have extra None slots for unused indices
                # (e.g., streams [3,1,2] need array size 4 but only 3 pieces)
                filled_count = sum(1 for p in all_pieces if p is not None)
                # Get actual required count from parent context's stream indices
                required_pieces = len(all_stream_indices_for_size) if original_source_indices and all_stream_indices_for_size else len(all_pieces)
                print(f"[Accept] Filled {filled_count}/{required_pieces} pieces (array size {len(all_pieces)}): matched={len(matched_pairs)}, unmatched={len(unmatched_pieces)}")
                print(f"[Accept DEBUG] Final all_pieces state:")
                for i, p in enumerate(all_pieces):
                    if p is not None:
                        p_id = id(p)
                        p_centroid = np.mean(p, axis=0) if len(p) > 0 else [0,0,0]
                        print(f"  all_pieces[{i}]: {len(p)} verts, id={p_id}, centroid={p_centroid}")
                    else:
                        print(f"  all_pieces[{i}]: None")

                # Update parent_finalized_pieces with THIS sub-cut's results
                # Use all_pieces directly since it already has correct mappings
                updated_finalized = dict(parent_finalized_pieces) if parent_finalized_pieces else {}
                if original_source_indices:
                    for local_src_idx in range(len(original_source_indices)):
                        orig_src_idx = original_source_indices[local_src_idx]
                        if orig_src_idx < len(all_pieces) and all_pieces[orig_src_idx] is not None:
                            updated_finalized[orig_src_idx] = all_pieces[orig_src_idx]

                # Level-by-level processing: ALWAYS check pending sub-cuts first
                # This must happen regardless of whether all pieces are filled
                # IMPORTANT: Use subcut_level (where we ARE), not current_subcut_level
                subcut_level = obj._manual_cut_data.get('subcut_level', 0)
                pending_by_level = obj._manual_cut_data.get('pending_subcuts_by_level', {})

                print(f"[Accept] Subcut level: {subcut_level}")
                print(f"[Accept] Pending by level: {[(lvl, len(lst)) for lvl, lst in pending_by_level.items()]}")

                # Find lowest level with pending sub-cuts (breadth-first)
                next_pending = None
                for level in sorted(pending_by_level.keys()):
                    if len(pending_by_level[level]) > 0:
                        next_pending = (level, pending_by_level[level])
                        break

                if next_pending is not None:
                    # Open next pending sub-cut
                    level, pending_list = next_pending
                    print(f"[Accept] Opening pending sub-cut at level {level} ({len(pending_list)} remaining)")
                    next_subcut = pending_list[0]
                    obj._manual_cut_data['parent_finalized_pieces'] = updated_finalized
                    obj._open_subcut_for_piece(next_subcut)
                else:
                    # No more pending sub-cuts - check if ready to finalize
                    # Compare against required_pieces (actual source count), not array size
                    if filled_count >= required_pieces:
                        print(f"[Accept] Finalizing: {required_pieces} pieces ({len(matched_pairs)} pre-matched, {len(unmatched_pieces)} optimized)")

                        # All levels done - store final result and call cut_streams
                        if not hasattr(obj, '_manual_cut_results') or obj._manual_cut_results is None:
                            obj._manual_cut_results = {}

                        target_i = obj._manual_cut_data.get('target_i', 0)
                        result_key = (target_level, target_i)
                        print(f"[Accept] Storing result: target_level={target_level}, target_i={target_i}")
                        print(f"[Accept] all_pieces has {len(all_pieces)} entries, all_stream_indices={all_stream_indices}")
                        obj._manual_cut_results[result_key] = {
                            'cut_contours': all_pieces,
                            'source_indices': all_stream_indices,
                            'is_1to1': False,
                        }
                        print(f"[Accept] All levels complete, stored result with key {result_key}")

                        # Clear stale data before continuing
                        obj._manual_cut_data = None
                        obj._manual_cut_original_state = None
                        obj._manual_cut_pending = False
                        obj.cut_streams(cut_method='bp', muscle_name=muscle_name)
                        # Smoothening and alignment are now handled inside cut_streams
                        if not obj._manual_cut_pending and obj._manual_cut_data is None:
                            # Consume deferred cut animation backup
                            if hasattr(obj, '_cut_anim_deferred_backup') and obj._cut_anim_deferred_backup is not None:
                                obj._compute_cut_animation(*obj._cut_anim_deferred_backup)
                                obj._cut_anim_deferred_backup = None
                            # Clear auto-optimize-all flag when all cutting is done
                            if hasattr(obj, '_auto_optimize_all'):
                                obj._auto_optimize_all = False
                            if name in v._manual_cut_mouse:
                                del v._manual_cut_mouse[name]

                            # Auto-resume pipeline if it was paused waiting for manual cut
                            if hasattr(obj, '_pipeline_paused_at') and obj._pipeline_paused_at is not None:
                                paused_for_select = _resume_pipeline_after_cut(v, obj, name)
                                if paused_for_select:
                                    imgui.end()
                                    continue

                            imgui.end()
                            continue
                    else:
                        # No pending sub-cuts but pieces incomplete - this shouldn't happen
                        print(f"[Accept] WARNING: No pending sub-cuts but only {filled_count}/{required_pieces} pieces filled")

            # Skip remaining preview buttons if auto-accepting (cleaner UI)
            if not auto_accept:
                imgui.same_line()

                # Reset button - go back to pre-optimization state
                if imgui.button("Reset", button_width, 30):
                    print(f"[Preview] Resetting to pre-optimization state...")
                    pre_pieces = obj._manual_cut_data.get('pre_optimization_pieces', [])
                    pre_pieces_3d = obj._manual_cut_data.get('pre_optimization_pieces_3d', [])
                    if pre_pieces and pre_pieces_3d:
                        obj._manual_cut_data['current_pieces'] = pre_pieces
                        obj._manual_cut_data['current_pieces_3d'] = pre_pieces_3d
                    obj._manual_cut_data['optimization_preview'] = False
                    obj._manual_cut_data['cut_lines'] = []
                    obj._manual_cut_data['edit_history'] = []
                    obj._manual_cut_line = None
                    # Clear optimization display data
                    obj._manual_cut_data['transformed_sources_2d'] = None
                    # Clear _bp_viz_data to remove optimization results
                    if hasattr(obj, '_bp_viz_data'):
                        obj._bp_viz_data = []
                    print(f"[Preview] Reset complete - you can draw new cuts or optimize again")

                imgui.same_line()

                # Next Cut button - apply current cutting line to edit optimized pieces
                has_cut_line = obj._manual_cut_line is not None
                if has_cut_line:
                    if imgui.button("Cut", button_width, 30):
                        print(f"[Preview] Applying cut to edit optimized pieces...")

                        # Save current state for undo before applying cut
                        edit_history = obj._manual_cut_data.get('edit_history', [])
                        edit_history.append({
                            'pieces': [p.copy() for p in obj._manual_cut_data.get('current_pieces', [])],
                            'pieces_3d': [p.copy() for p in obj._manual_cut_data.get('current_pieces_3d', [])],
                        })
                        obj._manual_cut_data['edit_history'] = edit_history

                        # Apply the iterative cut to further subdivide pieces
                        success = obj._apply_iterative_cut()
                        if success:
                            print(f"[Preview] Cut applied - now have {len(obj._manual_cut_data.get('current_pieces', []))} pieces (edit #{len(edit_history)})")
                            obj._manual_cut_line = None  # Clear line after applying
                        else:
                            # Remove the history entry since cut failed
                            edit_history.pop()
                            print(f"[Preview] Cut failed - try drawing a different line")
                    imgui.same_line()

                # Undo button - revert to previous state
                edit_history = obj._manual_cut_data.get('edit_history', [])
                if len(edit_history) > 0:
                    if imgui.button("Undo", button_width, 30):
                        prev_state = edit_history.pop()
                        obj._manual_cut_data['current_pieces'] = prev_state['pieces']
                        obj._manual_cut_data['current_pieces_3d'] = prev_state['pieces_3d']
                        obj._manual_cut_data['edit_history'] = edit_history
                        obj._manual_cut_line = None
                        print(f"[Preview] Undid last edit - now have {len(prev_state['pieces'])} pieces")
                    imgui.same_line()

        # Skip button - for 1-to-1 case (only 1 source selected, no cutting needed)
        # Also auto-triggered in Optimize All mode
        if num_selected == 1:
            skip_clicked = imgui.button("Skip (1:1)", button_width, 30)
            # Auto-skip in Optimize All mode
            if auto_optimize_all and not skip_clicked:
                skip_clicked = True
                print(f"[OptAll] Auto-skipping 1:1 case...")
            if skip_clicked:
                print(f"Skipping cut for target (1:1 mapping with source {selected_sources[0]})")
                # Store the 1:1 result - the single selected source maps directly to target
                selected_src_idx = selected_sources[0]
                target_i = obj._manual_cut_data.get('target_i', 0)
                target_level = obj._manual_cut_data.get('target_level', 0)
                source_indices = obj._manual_cut_data.get('source_indices', [])

                # Store result in _manual_cut_results (persistent storage)
                # Use the actual source index from source_indices, not the local index
                actual_source_idx = source_indices[selected_src_idx] if selected_src_idx < len(source_indices) else selected_src_idx

                if not hasattr(obj, '_manual_cut_results') or obj._manual_cut_results is None:
                    obj._manual_cut_results = {}

                result_key = (target_level, target_i)
                obj._manual_cut_results[result_key] = {
                    'cut_contours': [obj._manual_cut_data['target_contour']],  # Single piece = target
                    'source_indices': [actual_source_idx],  # Maps to this source
                    'is_1to1': True,  # Flag for 1:1 mapping
                }
                print(f"Stored 1:1 result with key {result_key}, source {actual_source_idx}")

                # Continue with cut_streams - it will find the next cut point or finish
                obj._manual_cut_data = None
                obj._manual_cut_line = None
                obj._manual_cut_pending = False
                obj.cut_streams(cut_method='bp', muscle_name=muscle_name)

                # Smoothening and alignment are now handled inside cut_streams
                if not obj._manual_cut_pending and obj._manual_cut_data is None:
                    # Consume deferred cut animation backup
                    if hasattr(obj, '_cut_anim_deferred_backup') and obj._cut_anim_deferred_backup is not None:
                        obj._compute_cut_animation(*obj._cut_anim_deferred_backup)
                        obj._cut_anim_deferred_backup = None
                    # Clear auto-optimize-all flag when all cutting is done
                    if hasattr(obj, '_auto_optimize_all'):
                        obj._auto_optimize_all = False
                    if name in v._manual_cut_mouse:
                        del v._manual_cut_mouse[name]

                    # Auto-resume pipeline if it was paused waiting for manual cut
                    if hasattr(obj, '_pipeline_paused_at') and obj._pipeline_paused_at is not None:
                        _resume_pipeline_after_cut(v, obj, name)

                    imgui.end()
                    continue
                # cut_streams returned early for another manual cut - don't close window
            imgui.same_line()

        # Check if we're in assignment mode (after cutting)
        has_valid_line = obj._manual_cut_line is not None
        in_assignment_mode = obj._manual_cut_data.get('assignment_mode', False)
        piece_assignments = obj._manual_cut_data.get('piece_assignments', {})
        in_preview_mode_check = obj._manual_cut_data.get('optimization_preview', False)

        # Skip normal cutting/assignment UI if in preview mode (Accept/Reset/Cut shown above)
        if in_preview_mode_check:
            # In preview mode, show Save&Close and Cancel buttons
            if imgui.button("Save && Close", button_width, 30):
                obj._save_and_close_manual_cut(muscle_name=muscle_name)
                if hasattr(obj, '_auto_optimize_all'):
                    obj._auto_optimize_all = False
                if name in v._manual_cut_mouse:
                    del v._manual_cut_mouse[name]
                imgui.end()
                continue
            imgui.same_line()
            if imgui.button("Cancel", button_width, 30):
                obj._cancel_manual_cut()
                if name in v._manual_cut_mouse:
                    del v._manual_cut_mouse[name]
            imgui.end()
            continue

        if in_assignment_mode:
            # === ASSIGNMENT MODE UI ===
            imgui.text("Assign sources to pieces:")
            imgui.separator()

            # Piece colors (same as visualization)
            piece_colors = [
                (1.0, 1.0, 1.0),  # White
                (0.2, 0.6, 1.0),  # Blue
                (1.0, 0.4, 0.2),  # Orange
                (0.2, 1.0, 0.4),  # Green
                (1.0, 0.8, 0.2),  # Yellow
                (0.8, 0.2, 1.0),  # Purple
            ]

            # Get source labels (original indices for display)
            source_labels = obj._manual_cut_data.get('source_labels', list(range(len(source_contours_3d))))

            # Show each piece with source checkboxes
            for piece_idx in range(len(current_pieces)):
                assigned = piece_assignments.get(piece_idx, [])
                # Draw color indicator
                color = piece_colors[piece_idx % len(piece_colors)]
                imgui.push_style_color(imgui.COLOR_BUTTON, *color, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *color, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *color, 1.0)
                imgui.button(f"P{piece_idx}##color", 30, 20)
                imgui.pop_style_color(3)
                imgui.same_line()
                imgui.text("->")
                imgui.same_line()

                # Checkboxes for each source (use original labels for display)
                for src_idx in range(len(source_contours_3d)):
                    is_assigned = src_idx in assigned
                    # Display original source index from source_labels
                    src_label = source_labels[src_idx] if src_idx < len(source_labels) else src_idx
                    changed, new_val = imgui.checkbox(f"S{src_label}##p{piece_idx}", is_assigned)
                    if changed:
                        if new_val:
                            # Add source to this piece, remove from others
                            for p in piece_assignments:
                                if src_idx in piece_assignments[p]:
                                    piece_assignments[p].remove(src_idx)
                            if piece_idx not in piece_assignments:
                                piece_assignments[piece_idx] = []
                            piece_assignments[piece_idx].append(src_idx)
                        else:
                            if src_idx in piece_assignments.get(piece_idx, []):
                                piece_assignments[piece_idx].remove(src_idx)
                        obj._manual_cut_data['piece_assignments'] = piece_assignments
                    imgui.same_line()
                imgui.new_line()

            imgui.separator()

            # Confirm button - processes assignments
            if imgui.button("Confirm", button_width, 30):
                # Process assignments: 1:1 finalized, 2:1 opens sub-window
                obj._process_piece_assignments()

                # Check what to do next
                # IMPORTANT: Use subcut_level (where we ARE) not current_subcut_level (batch tracking)
                # _process_piece_assignments queues to subcut_level + 1
                subcut_level = obj._manual_cut_data.get('subcut_level', 0)
                pending_by_level = obj._manual_cut_data.get('pending_subcuts_by_level', {})
                newly_queued = pending_by_level.get(subcut_level + 1, [])

                print(f"[Confirm] Subcut level: {subcut_level}")
                print(f"[Confirm] Pending by level: {[(lvl, len(lst)) for lvl, lst in pending_by_level.items()]}")

                if len(newly_queued) > 0:
                    # Open sub-window for first pending N:1 case from THIS window
                    print(f"[Confirm] {len(newly_queued)} sub-cuts at level {subcut_level + 1}, opening next...")
                    obj._open_subcut_for_piece(newly_queued[0])
                else:
                    # Check if there are other pending sub-cuts at any level
                    # Find lowest level with pending sub-cuts (breadth-first)
                    next_pending = None
                    for level in sorted(pending_by_level.keys()):
                        if len(pending_by_level[level]) > 0:
                            next_pending = (level, pending_by_level[level])
                            break

                    if next_pending is not None:
                        # Open next pending sub-cut
                        level, pending_list = next_pending
                        print(f"[Confirm] Opening pending sub-cut at level {level} ({len(pending_list)} remaining)")
                        obj._open_subcut_for_piece(pending_list[0])
                    else:
                        # All done - finalize
                        # Extract values BEFORE finalize in case it modifies _manual_cut_data
                        target_level = obj._manual_cut_data.get('target_level', 0)
                        target_i = obj._manual_cut_data.get('target_i', 0)

                        # _finalize_manual_cuts() stores result with correct stream_indices from parent_context
                        cut_result, _ = obj._finalize_manual_cuts()
                        if cut_result is not None:
                            # NOTE: _finalize_manual_cuts() already stored the result with correct stream_indices
                            # No need to overwrite here - just verify it exists
                            result_key = (target_level, target_i)
                            if hasattr(obj, '_manual_cut_results') and result_key in obj._manual_cut_results:
                                print(f"[Confirm] Result already stored with key {result_key}")
                            else:
                                print(f"[Confirm] WARNING: Result not found for key {result_key}")

                            # Clear _manual_cut_data to prevent stale data issues
                            obj._manual_cut_data = None
                            obj._manual_cut_original_state = None
                            obj._manual_cut_pending = False
                            obj.cut_streams(cut_method='bp', muscle_name=muscle_name)
                            # Smoothening and alignment are now handled inside cut_streams
                            if not obj._manual_cut_pending and obj._manual_cut_data is None:
                                # Consume deferred cut animation backup
                                if hasattr(obj, '_cut_anim_deferred_backup') and obj._cut_anim_deferred_backup is not None:
                                    obj._compute_cut_animation(*obj._cut_anim_deferred_backup)
                                    obj._cut_anim_deferred_backup = None
                                # Clear auto-optimize-all flag when all cutting is done
                                if hasattr(obj, '_auto_optimize_all'):
                                    obj._auto_optimize_all = False
                                if name in v._manual_cut_mouse:
                                    del v._manual_cut_mouse[name]

                                # Auto-resume pipeline if it was paused waiting for manual cut
                                if hasattr(obj, '_pipeline_paused_at') and obj._pipeline_paused_at is not None:
                                    _resume_pipeline_after_cut(v, obj, name)

                                imgui.end()
                                continue
                            # cut_streams returned early for another manual cut - don't close window

            # NOTE: No "Back" button - only one cut allowed per window
            # User must use "Reset" to start over or "Done" to proceed
        else:
            # === CUTTING MODE UI ===
            # Cut button - applies cut and enters assignment mode
            cut_enabled = has_valid_line
            if not cut_enabled:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)

            if imgui.button("Cut", button_width, 30):
                if cut_enabled:
                    # Apply the cut
                    success = obj._apply_iterative_cut()
                    if success:
                        obj._manual_cut_line = None

                        # Enter assignment mode
                        obj._manual_cut_data['assignment_mode'] = True

                        # Initialize assignments by distance
                        obj._init_piece_assignments_by_distance()

                        print(f"Cut applied. Now assign sources to pieces.")
                    else:
                        print("Failed to apply cut - line must cross a piece twice")
                else:
                    print("Draw a cutting line first")

            if not cut_enabled:
                imgui.pop_style_var()

        # Hide Reset/Cancel when auto-optimizing (cleaner UI)
        if not should_auto_optimize:
            imgui.same_line()
            if imgui.button("Reset", button_width, 30):
                # Reset to original state (before any cuts or sub-windows)
                obj._reset_to_original_state()
                # Also clear auto-optimize-all flag
                if hasattr(obj, '_auto_optimize_all'):
                    obj._auto_optimize_all = False
                if name in v._manual_cut_mouse:
                    v._manual_cut_mouse[name]['dragging'] = False
                    v._manual_cut_mouse[name]['zoom'] = 1.0
                    v._manual_cut_mouse[name]['pan'] = [0.0, 0.0]

            imgui.same_line()
            if imgui.button("Save && Close", button_width, 30):
                obj._save_and_close_manual_cut(muscle_name=muscle_name)
                if hasattr(obj, '_auto_optimize_all'):
                    obj._auto_optimize_all = False
                if name in v._manual_cut_mouse:
                    del v._manual_cut_mouse[name]
                imgui.end()
                continue

            imgui.same_line()
            if imgui.button("Cancel", button_width, 30):
                obj._cancel_manual_cut()
                # Also clear auto-optimize-all flag
                if hasattr(obj, '_auto_optimize_all'):
                    obj._auto_optimize_all = False
                if name in v._manual_cut_mouse:
                    del v._manual_cut_mouse[name]

        imgui.end()


def _render_level_select_windows(v):
    """Render level selection windows for manual level selection after Contour Select."""
    for name, obj in v.zygote_muscle_meshes.items():
        # Check if level select animation just completed - resume pipeline
        if getattr(obj, '_level_select_anim_pending_resume', False):
            obj._level_select_anim_pending_resume = False
            if hasattr(obj, '_pipeline_paused_at') and obj._pipeline_paused_at is not None:
                max_step = obj._process_step if hasattr(obj, '_process_step') else 12
                start_step = obj._pipeline_paused_at
                if start_step <= max_step:
                    print(f"[{name}] Auto-resuming pipeline from step {start_step} to {max_step}...")
                    try:
                        _defer = getattr(obj, 'animate_process', False)
                        # Step 9: Build Fiber
                        if start_step <= 9 <= max_step and hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                            print(f"  [9/{max_step}] Building fibers...")
                            _t0 = time.time()
                            _ensure_level_selection_applied(v, name, obj, defer=_defer)
                            obj._belly_waypoints_before_tendon_extension = None
                            obj.build_fibers(skeleton_meshes=v.zygote_skeleton_meshes, defer=_defer)
                            if (getattr(obj, 'enable_tendon_extension', True)
                                    and (getattr(obj, 'origin_tendon_extension_name', '')
                                         or getattr(obj, 'insertion_tendon_extension_name', ''))):
                                _extend_belly_fibers_with_tendons(v, name, obj)
                            _run_counterpart_step(v, name, obj, 9, defer=_defer)
                            print(f"  [9/{max_step}] Done in {time.time()-_t0:.3f}s")
                            if _defer:
                                obj._level_select_replayed = False

                        # Step 10: Resample Contours
                        if start_step <= 10 <= max_step and obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None:
                            print(f"  [10/{max_step}] Resampling Contours...")
                            _t0 = time.time()
                            _resample_contours_with_links(v, name, obj, defer=_defer)
                            _resample_linked_tendon_extensions(v, name, obj, defer=_defer)
                            print(f"  [10/{max_step}] Done in {time.time()-_t0:.3f}s")
                            if _defer:
                                obj._build_fibers_replayed = False

                        # Step 11: Build Contour Mesh
                        if start_step <= 11 <= max_step and obj.contours is not None and len(obj.contours) > 0 and obj.draw_contour_stream is not None:
                            print(f"  [11/{max_step}] Building Contour Mesh...")
                            _t0 = time.time()
                            if _prepare_owned_connected_contour_mesh_source(v, name, obj):
                                obj.build_contour_mesh(defer=_defer)
                                if not _connected_source_has_linked_components(obj):
                                    _run_counterpart_step(v, name, obj, 11, defer=_defer)
                            print(f"  [11/{max_step}] Done in {time.time()-_t0:.3f}s")
                            if _defer:
                                obj._resample_replayed = False

                        # Step 12: Tetrahedralize
                        if start_step <= 12 <= max_step:
                            print(f"  [12/{max_step}] Tetrahedralizing...")
                            _t0 = time.time()
                            if _skip_non_owner_connected_mesh(v, name, obj, "Tetrahedralize"):
                                print(f"  [12/{max_step}] Skipped in {time.time()-_t0:.3f}s")
                                tet_ok = False
                            else:
                                tet_ok = _tetrahedralize_single_contour_mesh(v, name, obj, defer=_defer)
                                if tet_ok and not _connected_source_has_linked_components(obj):
                                    _run_counterpart_step(v, name, obj, 12, defer=_defer)
                            status = "Done" if tet_ok else "Failed"
                            print(f"  [12/{max_step}] {status} in {time.time()-_t0:.3f}s")
                            if tet_ok and obj.tet_vertices is not None:
                                if _defer:
                                    obj._extract_internal_tet_edges()
                                    obj._classify_tet_faces_into_bands()
                                    obj._tetrahedralize_replayed = False
                                else:
                                    obj.is_draw_contours = False
                                    obj.is_draw_tet_mesh = True
                                    obj._tetrahedralize_replayed = True

                        print(f"[{name}] Pipeline complete (steps {start_step}-{max_step})!")
                    except Exception as e:
                        print(f"[{name}] Pipeline error: {e}")
                        import traceback
                        traceback.print_exc()
                obj._pipeline_paused_at = None

        if not hasattr(obj, '_level_select_window_open') or not obj._level_select_window_open:
            continue

        if not hasattr(obj, '_level_select_checkboxes') or obj._level_select_checkboxes is None:
            continue

        if not hasattr(obj, '_level_select_original') or obj._level_select_original is None:
            continue

        # Get data
        orig = obj._level_select_original
        checkboxes = obj._level_select_checkboxes
        stream_groups = orig['stream_groups']
        max_stream_count = len(checkboxes)

        # Window setup
        imgui.set_next_window_size(500, 600, imgui.FIRST_USE_EVER)
        expanded, opened = imgui.begin(f"Level Select: {name}", True)

        if not opened:
            # User closed window - cancel selection
            obj._level_select_window_open = False
            obj._level_select_checkboxes = None
            obj._level_select_original = None
            # Clear pipeline pause so "Resume" doesn't skip level select
            obj._pipeline_paused_at = None
            imgui.end()
            continue

        imgui.text(f"Streams: {max_stream_count}")
        imgui.text("Check/uncheck levels. Linked levels toggle together.")

        # Total / desired-count / reselect controls.
        # Origin + insertion always selected → minimum 3.
        total_levels = len(checkboxes[0]) if max_stream_count > 0 else 0
        current_count = sum(1 for c in checkboxes[0] if c) if max_stream_count > 0 else 0
        if not hasattr(obj, '_level_select_desired_count'):
            obj._level_select_desired_count = max(3, current_count)
        # Clamp to [3, total_levels]
        obj._level_select_desired_count = int(np.clip(
            obj._level_select_desired_count, 3, max(3, total_levels)))
        imgui.text(f"Total: {total_levels}    Selected: {current_count}")
        imgui.text("Desired:")
        imgui.same_line()
        if imgui.button(f"<##{name}_lvl_dec"):
            obj._level_select_desired_count = max(3, obj._level_select_desired_count - 1)
        imgui.same_line()
        imgui.text(f"{obj._level_select_desired_count}")
        imgui.same_line()
        if imgui.button(f">##{name}_lvl_inc"):
            obj._level_select_desired_count = min(total_levels, obj._level_select_desired_count + 1)
        imgui.same_line()
        if imgui.button(f"Reselect##{name}_lvl_reselect"):
            try:
                obj.select_levels_count(obj._level_select_desired_count)
                vis_changed = True
            except Exception as _e:
                print(f"[{name}] Reselect error: {_e}")

        imgui.separator()

        # Create scrollable region for checkboxes
        imgui.begin_child("LevelCheckboxes", 0, -50, border=True)

        # Track if visualization needs update
        vis_changed = False

        # Find max levels across all streams
        max_levels = max(len(checkboxes[s]) for s in range(max_stream_count))

        # Column width for each stream
        col_width = 80

        # Compute display order so that streams from same merged contour are adjacent
        # Find levels with linked groups and use that structure for ordering
        stream_display_order = list(range(max_stream_count))

        # Find a level with multiple groups (most informative for ordering)
        # Prefer levels where streams are split into distinct groups
        best_level = 0
        best_score = 0
        for lvl_i, groups in enumerate(stream_groups):
            multi_groups = [g for g in groups if len(g) > 1]
            num_multi_groups = len(multi_groups)
            total_linked = sum(len(g) for g in multi_groups)
            # Score: prefer multiple groups over single big group
            # A level with [[0,4], [1,2,3]] scores higher than [[0,1,2,3,4]]
            score = num_multi_groups * 100 + total_linked
            if score > best_score:
                best_score = score
                best_level = lvl_i

        if best_score > 0 and best_level < len(stream_groups):
            # Use the linked structure at this level to order streams
            # Streams in same group should be adjacent
            groups_at_best = stream_groups[best_level]
            stream_display_order = []
            for group in groups_at_best:
                stream_display_order.extend(sorted(group))
            # Add any missing streams
            for s in range(max_stream_count):
                if s not in stream_display_order:
                    stream_display_order.append(s)
        elif hasattr(obj, 'stream_contours') and obj.stream_contours is not None and len(obj.stream_contours) > 0:
            # Fall back to spatial ordering at first level
            try:
                centroids = []
                for stream_i in range(max_stream_count):
                    if stream_i < len(obj.stream_contours) and len(obj.stream_contours[stream_i]) > 0:
                        first_contour = obj.stream_contours[stream_i][0]
                        if len(first_contour) > 0:
                            centroid = np.mean(first_contour, axis=0)
                            centroids.append((stream_i, centroid))
                        else:
                            centroids.append((stream_i, np.zeros(3)))
                    else:
                        centroids.append((stream_i, np.zeros(3)))

                if len(centroids) > 1:
                    positions = np.array([c[1] for c in centroids])
                    mean_pos = np.mean(positions, axis=0)
                    centered = positions - mean_pos

                    if centered.shape[0] >= 2:
                        cov = np.cov(centered.T)
                        if cov.shape == (3, 3):
                            eigenvalues, eigenvectors = np.linalg.eigh(cov)
                            principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
                            projections = [(s, np.dot(c - mean_pos, principal_axis)) for s, c in centroids]
                            projections.sort(key=lambda x: x[1])
                            stream_display_order = [p[0] for p in projections]
            except Exception:
                pass  # Keep default order on error

        # Header row with stream labels (in spatial order)
        imgui.text("Level")
        for col_i, stream_i in enumerate(stream_display_order):
            imgui.same_line(60 + col_i * col_width)
            imgui.text(f"S{stream_i}")
        imgui.separator()

        # Base colors - many distinct, bright colors
        base_colors = [
            (1.0, 0.2, 0.2),   # red
            (0.2, 0.8, 0.2),   # green
            (0.2, 0.4, 1.0),   # blue
            (1.0, 0.6, 0.1),   # orange
            (0.8, 0.2, 0.9),   # purple
            (0.1, 0.9, 0.9),   # cyan
            (0.9, 0.9, 0.2),   # yellow
            (1.0, 0.4, 0.7),   # pink
            (0.6, 0.4, 0.2),   # brown
            (0.4, 1.0, 0.6),   # mint
            (0.6, 0.6, 1.0),   # lavender
            (1.0, 0.8, 0.4),   # peach
        ]

        # Pre-compute colors for each stream at each level
        # - Same group: keep color
        # - Merge: blend colors
        # - Split: assign NEW distinct colors
        stream_colors_rgb = [[(0.5, 0.5, 0.5)] * max_stream_count for _ in range(max_levels)]
        group_to_color = {}  # group_key -> color
        next_base_color = 0

        for level_i in range(max_levels):
            if level_i < len(stream_groups):
                for group in stream_groups[level_i]:
                    group_key = tuple(sorted(group))

                    if group_key in group_to_color:
                        # Already assigned (same group seen before)
                        color = group_to_color[group_key]
                    elif level_i == 0:
                        # First level: assign base color
                        color = base_colors[next_base_color % len(base_colors)]
                        next_base_color += 1
                        group_to_color[group_key] = color
                    else:
                        # Check previous level colors of streams in this group
                        prev_colors = []
                        for s in group:
                            if s < max_stream_count:
                                prev_colors.append(stream_colors_rgb[level_i - 1][s])

                        unique_colors = list(set(prev_colors))

                        if len(unique_colors) > 1:
                            # Merge: blend all previous colors
                            r = sum(c[0] for c in unique_colors) / len(unique_colors)
                            g = sum(c[1] for c in unique_colors) / len(unique_colors)
                            b = sum(c[2] for c in unique_colors) / len(unique_colors)
                            color = (r, g, b)
                        else:
                            # New group (split or first appearance): new distinct color
                            color = base_colors[next_base_color % len(base_colors)]
                            next_base_color += 1

                        group_to_color[group_key] = color

                    # Assign color to all streams in group
                    for s in group:
                        if s < max_stream_count:
                            stream_colors_rgb[level_i][s] = color

        # Render rows (one per level)
        for level_i in range(max_levels):
            # Build stream-to-group mapping for this level
            stream_to_group = {}  # stream_i -> group list
            has_any_linked = False
            if level_i < len(stream_groups):
                for group in stream_groups[level_i]:
                    is_multi = len(group) > 1
                    if is_multi:
                        has_any_linked = True
                    for s in group:
                        stream_to_group[s] = group if is_multi else None

            # Level label with link indicator
            if has_any_linked:
                imgui.text_colored(f"{level_i:3d} *", 0.0, 1.0, 1.0, 1.0)
            else:
                imgui.text(f"{level_i:3d}  ")

            # Checkbox for each stream at this level (in spatial display order)
            for col_i, stream_i in enumerate(stream_display_order):
                imgui.same_line(60 + col_i * col_width)

                if level_i < len(checkboxes[stream_i]):
                    checkbox_id = f"##lvl_{stream_i}_{level_i}"
                    is_checked = checkboxes[stream_i][level_i]

                    # Find THIS stream's group
                    this_group = stream_to_group.get(stream_i, None)
                    is_this_linked = this_group is not None and len(this_group) > 1

                    # Always show color based on group identity
                    color = stream_colors_rgb[level_i][stream_i]
                    # Tint checkbox frame - use bright colors
                    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, color[0] * 0.8, color[1] * 0.8, color[2] * 0.8, 1.0)
                    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, color[0], color[1], color[2], 1.0)
                    # Checkmark color - use black or white based on background brightness
                    brightness = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
                    if brightness > 0.5:
                        imgui.push_style_color(imgui.COLOR_CHECK_MARK, 0.0, 0.0, 0.0, 1.0)  # black
                    else:
                        imgui.push_style_color(imgui.COLOR_CHECK_MARK, 1.0, 1.0, 1.0, 1.0)  # white

                    changed, new_value = imgui.checkbox(checkbox_id, is_checked)

                    imgui.pop_style_color(3)

                    if changed:
                        vis_changed = True
                        checkboxes[stream_i][level_i] = new_value

                        # Only propagate within an actually-linked group at
                        # this level.  Split levels of a cut muscle keep
                        # per-stream independence.
                        if is_this_linked:
                            for other_stream in this_group:
                                if other_stream != stream_i and other_stream < max_stream_count:
                                    if level_i < len(checkboxes[other_stream]):
                                        checkboxes[other_stream][level_i] = new_value
                else:
                    # This stream doesn't have this level - show placeholder
                    imgui.text("-")

        imgui.end_child()

        # Update visualization if checkboxes changed
        if vis_changed:
            obj._update_level_select_visualization()

        # Buttons
        button_width = 120
        if imgui.button("Undo Selection", button_width, 30):
            obj._undo_level_selection()

        imgui.same_line()
        if imgui.button("Finish Select", button_width, 30):
            # Always defer — apply selection immediately, replay animation later if needed
            obj._start_level_select_animation(defer=True)
            animate = getattr(obj, 'animate_process', False)
            if not animate:
                obj._level_select_replayed = True
            if getattr(obj, 'linked_drive_counterpart', False):
                _run_counterpart_step(v, name, obj, 8, defer=animate)

        imgui.end()


def update_available_muscles(v):
    """Scan muscle directory and subdirectories for available .obj files not yet loaded."""
    v.available_muscle_files = []
    v.available_muscle_by_category = {}
    v.available_muscle_groups_by_category = {}
    loaded_names = set(v.zygote_muscle_meshes.keys())

    # Scan root directory and subdirectories
    for root, dirs, files in os.walk(v.zygote_muscle_dir):
        for file in files:
            if file.endswith('.obj'):
                muscle_name = file.split('.')[0]
                if muscle_name not in loaded_names:
                    full_path = os.path.join(root, file)
                    v.available_muscle_files.append((muscle_name, full_path))

                    # Determine body part from relative path
                    rel_path = os.path.relpath(root, v.zygote_muscle_dir)
                    if rel_path == '.':
                        body_part = 'Root'
                    else:
                        body_part = rel_path.replace(os.sep, '/')

                    # Determine side (L/R/M) from muscle name prefix
                    if muscle_name.startswith('L_'):
                        side = 'L'
                    elif muscle_name.startswith('R_'):
                        side = 'R'
                    else:
                        side = 'M'

                    category = f"{body_part}/{side}"

                    if category not in v.available_muscle_by_category:
                        v.available_muscle_by_category[category] = []
                        v.available_muscle_groups_by_category[category] = {}
                        # Initialize expanded state (collapsed by default)
                        if category not in v.available_category_expanded:
                            v.available_category_expanded[category] = False

                    v.available_muscle_by_category[category].append((muscle_name, full_path))
                    group_name = _zygote_component_group_name(muscle_name)
                    v.available_muscle_groups_by_category[category].setdefault(group_name, [])
                    v.available_muscle_groups_by_category[category][group_name].append((muscle_name, full_path))

    # Sort categories and muscles within each category
    v.available_muscle_by_category = dict(sorted(v.available_muscle_by_category.items()))
    for category in v.available_muscle_by_category:
        v.available_muscle_by_category[category].sort(key=lambda x: x[0])
    for category in v.available_muscle_groups_by_category:
        v.available_muscle_groups_by_category[category] = dict(sorted(v.available_muscle_groups_by_category[category].items()))
        for group_name in v.available_muscle_groups_by_category[category]:
            v.available_muscle_groups_by_category[category][group_name].sort(key=lambda x: x[0])
        groups = v.available_muscle_groups_by_category[category]
        # Only Pennation folders represent true multi-OBJ groups. Collapse
        # ordinary component-looking filenames to their standalone base OBJ in
        # the available-muscle UI as well as in the loading function.
        normalized_groups = {}
        for group_name, components in groups.items():
            if any(_is_pennation_path(path) for _name, path in components):
                normalized_groups[group_name] = components
                continue
            base = next((item for item in components if item[0] == group_name), None)
            if base is None:
                base = next((item for item in components
                             if not _is_zygote_tendon_mesh(*item)), components[0])
            normalized_groups[base[0]] = [base]
        groups = normalized_groups
        v.available_muscle_groups_by_category[category] = dict(sorted(groups.items()))
        belly_groups = [
            group_name for group_name, components in groups.items()
            if any('tendon' not in comp_name.lower() for comp_name, _ in components)
        ]
        if len(belly_groups) == 1:
            parent = belly_groups[0]
            for group_name in list(groups.keys()):
                if group_name == parent:
                    continue
                components = groups[group_name]
                if components and all('tendon' in comp_name.lower() for comp_name, _ in components):
                    groups[parent].extend(components)
                    del groups[group_name]
            groups[parent].sort(key=lambda x: x[0])

    # Sort flat list by name (for backwards compatibility)
    v.available_muscle_files.sort(key=lambda x: x[0])

    # Reset selection if current selection is no longer valid
    if v.available_selected_muscle:
        # Check if selected muscle still exists in available list
        found = False
        for cat, muscles in v.available_muscle_by_category.items():
            if any(name == v.available_selected_muscle for name, _ in muscles):
                found = True
                break
        if not found:
            v.available_selected_category = None
            v.available_selected_muscle = None

    # Reset old index-based selection if out of bounds
    if v.available_muscle_selected >= len(v.available_muscle_files):
        v.available_muscle_selected = max(0, len(v.available_muscle_files) - 1)


def add_muscle_mesh(v, name, path):
    """Dynamically add a muscle mesh to the simulation."""
    if name in v.zygote_muscle_meshes:
        return  # Already loaded

    # Remember current position in category for cursor maintenance
    prev_category = v.available_selected_category
    prev_index = -1
    if prev_category and prev_category in v.available_muscle_by_category:
        muscles_in_cat = v.available_muscle_by_category[prev_category]
        for i, (mname, mpath) in enumerate(muscles_in_cat):
            if mname == name:
                prev_index = i
                break

    v.zygote_muscle_meshes[name] = MeshLoader()
    v.zygote_muscle_meshes[name].load(path)
    _apply_zygote_muscle_style(
        v.zygote_muscle_meshes[name],
        name,
        path,
        v.zygote_muscle_color,
        v.zygote_muscle_transparency,
        v.is_draw_zygote_muscle,
    )
    # Load trimesh and apply same scale as MeshLoader.load() uses
    muscle_trimesh = trimesh.load_mesh(path)
    muscle_trimesh.vertices *= 0.01  # MESH_SCALE
    v.zygote_muscle_meshes[name].trimesh = muscle_trimesh

    # Re-sort meshes by name
    v.zygote_muscle_meshes = dict(sorted(v.zygote_muscle_meshes.items()))

    # Update available list
    update_available_muscles(v)

    # Auto-save muscle list
    save_loaded_muscles(v)

    # Reload motion cache so newly added muscle picks up its cached frames
    if hasattr(v, 'motion_bvh') and v.motion_bvh is not None:
        _motion_load_cache(v)

    # Maintain cursor position in the same category
    if prev_category and prev_category in v.available_muscle_by_category:
        muscles_in_cat = v.available_muscle_by_category[prev_category]
        if len(muscles_in_cat) > 0:
            # Select same index or previous if at end
            new_index = min(prev_index, len(muscles_in_cat) - 1)
            v.available_selected_category = prev_category
            v.available_selected_muscle = muscles_in_cat[new_index][0]
        else:
            # Category is now empty, clear selection
            v.available_selected_category = None
            v.available_selected_muscle = None
    else:
        v.available_selected_category = None
        v.available_selected_muscle = None


def add_muscle_group(v, category, group_name):
    """Add Pennation components, or one standalone OBJ otherwise."""
    groups = getattr(v, 'available_muscle_groups_by_category', {})
    components = groups.get(category, {}).get(group_name, [])
    if not components:
        return
    if not any(_is_pennation_path(path) for _name, path in components):
        base = next((item for item in components if item[0] == group_name), None)
        if base is None:
            base = next((item for item in components
                         if not _is_zygote_tendon_mesh(*item)), components[0])
        components = [base]
        print(f"[{group_name}] Standalone OBJ group (non-Pennation): "
              f"loading {base[0]}")
    else:
        print(f"[{group_name}] Pennation group: loading {len(components)} components")
    for name, path in list(components):
        add_muscle_mesh(v, name, path)
    v.available_selected_category = None
    v.available_selected_muscle = None


def remove_muscle_mesh(v, name):
    """Dynamically remove a muscle mesh from the simulation."""
    if name not in v.zygote_muscle_meshes:
        return  # Not loaded

    del v.zygote_muscle_meshes[name]

    # Reset loaded selection if out of bounds
    if v.loaded_muscle_selected >= len(v.zygote_muscle_meshes):
        v.loaded_muscle_selected = max(0, len(v.zygote_muscle_meshes) - 1)

    # Update available list
    update_available_muscles(v)

    # Auto-save muscle list
    save_loaded_muscles(v)


def remove_muscle_group(v, group_name):
    """Remove every loaded component belonging to one grouped anatomical muscle."""
    names = [
        name for name in list(v.zygote_muscle_meshes.keys())
        if _zygote_component_group_name(name) == group_name
    ]
    for name in names:
        if name in v.zygote_muscle_meshes:
            del v.zygote_muscle_meshes[name]
    if v.loaded_muscle_selected >= len(v.zygote_muscle_meshes):
        v.loaded_muscle_selected = max(0, len(v.zygote_muscle_meshes) - 1)
    update_available_muscles(v)
    save_loaded_muscles(v)


def get_available_muscle_groups(v):
    """Get list of body part groups that have available muscles."""
    groups = set()
    for category in v.available_muscle_by_category.keys():
        if len(v.available_muscle_by_category[category]) > 0:
            if '/' in category:
                body_part = category.rsplit('/', 1)[0]
            else:
                body_part = category
            groups.add(body_part)
    return sorted(groups)


def add_muscles_by_group(v, group, prefix):
    """Add all muscles in a body part group that start with the given prefix (L_ or R_)."""
    to_add = []
    for category, muscles in v.available_muscle_by_category.items():
        # Match categories belonging to this body part
        if '/' in category:
            body_part = category.rsplit('/', 1)[0]
        else:
            body_part = category
        if body_part != group:
            continue
        for name, path in muscles:
            if name.startswith(prefix):
                to_add.append((name, path))
    for name, path in to_add:
        add_muscle_mesh(v, name, path)


def remove_all_muscles(v):
    """Remove all currently loaded muscles."""
    names = list(v.zygote_muscle_meshes.keys())
    for name in names:
        del v.zygote_muscle_meshes[name]
    v.loaded_muscle_selected = 0
    update_available_muscles(v)
    save_loaded_muscles(v)


def save_loaded_muscles(v):
    """Save current loaded muscle names to file for later reload."""
    import json
    try:
        # Build list of (name, path) for all loaded muscles
        muscle_list = []
        for name, mobj in v.zygote_muscle_meshes.items():
            # Get path from mesh object's stored filename
            path = getattr(mobj, 'obj', None)
            if path is None:
                # Fallback: try to reconstruct path
                path = v.zygote_muscle_dir + name + '.obj'
            muscle_list.append({'name': name, 'path': path})

        with open(v.last_muscles_file, 'w') as f:
            json.dump(muscle_list, f, indent=2)
        print(f"Saved {len(muscle_list)} muscle names to {v.last_muscles_file}")
    except Exception as e:
        print(f"Failed to save muscle list: {e}")


def _load_muscle_data(name, path, color, transparency, is_draw):
    """Load a single muscle's mesh data (I/O-heavy, no viewer state touched).

    Returns (name, MeshLoader) on success, or (name, None) on failure.
    """
    try:
        mesh = MeshLoader()
        mesh.load(path)
        _apply_zygote_muscle_style(mesh, name, path, color, transparency, is_draw)
        muscle_trimesh = trimesh.load_mesh(path)
        muscle_trimesh.vertices *= 0.01  # MESH_SCALE
        mesh.trimesh = muscle_trimesh
        return (name, mesh)
    except Exception as e:
        print(f"  Failed to load muscle {name}: {e}")
        return (name, None)


def load_previous_muscles(v):
    """Load muscles that were previously saved (parallel I/O)."""
    from concurrent.futures import ThreadPoolExecutor
    if not os.path.exists(v.last_muscles_file):
        print(f"No previous muscle list found at {v.last_muscles_file}")
        return 0

    try:
        with open(v.last_muscles_file, 'r') as f:
            muscle_list = json.load(f)

        # Collapse legacy multi-component lists for ordinary muscles. Only
        # Pennation directories represent true grouped anatomical parts.
        grouped = {}
        for entry in muscle_list:
            key = _zygote_component_group_name(entry['name'])
            grouped.setdefault(key, []).append(entry)
        load_entries = []
        for entries in grouped.values():
            if any(_is_pennation_path(e.get('path')) for e in entries):
                load_entries.extend(entries)
                continue
            group_key = _zygote_component_group_name(entries[0]['name'])
            base = next((e for e in entries if e['name'] == group_key), None)
            if base is None:
                base = next((e for e in entries
                             if not _is_zygote_tendon_mesh(e['name'], e.get('path'))),
                            entries[0])
            load_entries.append(base)

        # Collect work items, skipping already-loaded and missing files
        pairs = []
        for entry in load_entries:
            name = entry['name']
            path = entry['path']
            if name in v.zygote_muscle_meshes:
                continue
            if not os.path.exists(path):
                print(f"  Muscle file not found: {path}")
                continue
            pairs.append((name, path))

        if not pairs:
            print("No new muscles to load")
            return 0

        # Snapshot viewer state for thread-safe reads
        color = list(v.zygote_muscle_color)
        transparency = v.zygote_muscle_transparency
        is_draw = v.is_draw_zygote_muscle

        # Parallel I/O phase
        t0 = time.time()
        workers = min(len(pairs), os.cpu_count() or 4)
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_load_muscle_data, name, path, color, transparency, is_draw)
                for name, path in pairs
            ]
            for fut in futures:
                results.append(fut.result())

        # Sequential integration phase
        loaded_count = 0
        for name, mesh in results:
            if mesh is not None:
                v.zygote_muscle_meshes[name] = mesh
                loaded_count += 1

        if loaded_count > 0:
            v.zygote_muscle_meshes = dict(sorted(v.zygote_muscle_meshes.items()))
            update_available_muscles(v)
            save_loaded_muscles(v)
            if hasattr(v, 'motion_bvh') and v.motion_bvh is not None:
                _motion_load_cache(v)

        elapsed = time.time() - t0
        print(f"Loaded {loaded_count} muscles from previous session ({elapsed:.2f}s, {workers} workers)")
        return loaded_count
    except Exception as e:
        print(f"Failed to load previous muscles: {e}")
        return 0


def find_inter_muscle_constraints(v, threshold=None, k_cap=None):
    """
    Find distance constraints between vertices of different muscles.
    Uses REST positions (from soft_body.rest_positions) to find constraints.
    Only considers muscles with initialized soft bodies.

    Args:
        threshold: Maximum distance to create constraint (default: v.inter_muscle_constraint_threshold)
        k_cap: Per-vertex max number of cross-muscle neighbors (default:
            v.inter_muscle_k_cap, fallback 3). Bounds total inter-muscle
            edges to ~k_cap * N_verts; sparse-system non-zeros stop scaling
            quadratically with vert density.

    Returns:
        Number of constraints found
    """
    if threshold is None:
        threshold = v.inter_muscle_constraint_threshold
    if k_cap is None:
        k_cap = int(getattr(v, 'inter_muscle_k_cap', 3))

    v.inter_muscle_constraints = []

    # Get all muscles with soft body (need rest_positions).
    # Surface-only: interior tet verts cannot physically contact another
    # muscle, so restrict KD-tree query to surface verts. Surface vert
    # set = unique indices in tet_render_faces excluding cap faces (caps
    # are origin/insertion lids hidden inside bone, not in tissue contact).
    tet_muscles = {}
    for name, mobj in v.zygote_muscle_meshes.items():
        if hasattr(mobj, 'soft_body') and mobj.soft_body is not None:
            rest = mobj.soft_body.rest_positions.copy()
            fixed = mobj.soft_body.fixed_mask.copy()

            render_faces = getattr(mobj, 'tet_render_faces', None)
            surf_count = int(getattr(mobj, 'tet_surface_face_count', 0) or 0)
            if render_faces is not None and surf_count > 0:
                surf_faces = np.asarray(render_faces[:surf_count], dtype=np.int64)
                surface_vidx = np.unique(surf_faces.reshape(-1))
            else:
                # Fallback: every tet vert is candidate (shouldn't happen for
                # tet-baked muscles, but keeps behavior safe).
                surface_vidx = np.arange(len(rest), dtype=np.int64)

            tet_muscles[name] = {
                'rest_positions': rest,
                'fixed_mask': fixed,
                'surface_vidx': surface_vidx,
                'surface_rest': rest[surface_vidx],
                'surface_fixed': fixed[surface_vidx],
            }

    if len(tet_muscles) < 2:
        print(f"Inter-muscle constraints: need at least 2 muscles with soft body, found {len(tet_muscles)}")
        return 0

    muscle_names = list(tet_muscles.keys())
    print(f"Finding inter-muscle constraints for {len(muscle_names)} muscles "
          f"(threshold={threshold*100:.1f}cm, k_cap={k_cap})...")

    # For each pair of muscles
    from scipy.spatial import cKDTree

    for i in range(len(muscle_names)):
        name1 = muscle_names[i]
        data1 = tet_muscles[name1]
        # Surface-only: query and tree are over surface verts; map results
        # back to original tet-vert indices via surface_vidx.
        s_verts1 = data1['surface_rest']
        s_fixed1 = data1['surface_fixed']
        s_vidx1 = data1['surface_vidx']

        for j in range(i + 1, len(muscle_names)):
            name2 = muscle_names[j]
            data2 = tet_muscles[name2]
            s_verts2 = data2['surface_rest']
            s_fixed2 = data2['surface_fixed']
            s_vidx2 = data2['surface_vidx']

            if bool(getattr(v, 'inter_muscle_reciprocal', False)):
                # Mutual nearest neighbors define a sparse, approximately
                # one-to-one interface. One-sided k-NN overconstrains thin
                # muscles because many vertices on a thick neighbor can all
                # pull the same small cross-section.
                tree1 = cKDTree(s_verts1)
                tree2 = cKDTree(s_verts2)
                d12, j12 = tree2.query(
                    s_verts1, k=1, distance_upper_bound=threshold)
                _, i21 = tree1.query(
                    s_verts2, k=1, distance_upper_bound=threshold)
                for si, (dist, sj) in enumerate(zip(d12, j12)):
                    if not np.isfinite(dist) or sj >= len(s_verts2):
                        continue
                    if i21[int(sj)] != si:
                        continue
                    fixed1 = bool(s_fixed1[si])
                    fixed2 = bool(s_fixed2[int(sj)])
                    if fixed1 != fixed2:
                        continue
                    v.inter_muscle_constraints.append((
                        name1, int(s_vidx1[si]), fixed1,
                        name2, int(s_vidx2[int(sj)]), fixed2,
                        float(dist)))
                continue

            # Build KD-tree on muscle2 surface verts; query k_cap nearest
            # within threshold for every muscle1 surface vert. cKDTree.query
            # with k=k_cap returns INF/k for misses, which we filter out.
            tree2 = cKDTree(s_verts2)
            n2 = len(s_verts2)
            k_query = min(k_cap, n2) if k_cap > 0 else n2
            if k_query <= 0:
                continue
            dists, idxs = tree2.query(s_verts1, k=k_query,
                                      distance_upper_bound=threshold)
            # Normalize shape to (N1, K) regardless of k_query.
            if k_query == 1:
                dists = dists[:, None]
                idxs = idxs[:, None]
            # Iterate per source vert; skip misses (idx==n2 or inf dist).
            for s_v1_idx in range(len(s_verts1)):
                row_idx = idxs[s_v1_idx]
                row_d = dists[s_v1_idx]
                valid = (row_idx < n2) & np.isfinite(row_d)
                if not np.any(valid):
                    continue
                idx_arr = row_idx[valid].astype(np.int32)
                d_arr = row_d[valid]
                is_fixed1 = bool(s_fixed1[s_v1_idx])
                # Same fixed-status match (same fixed/same free)
                same_fixed_mask = s_fixed2[idx_arr] == is_fixed1
                idx_arr = idx_arr[same_fixed_mask]
                d_arr = d_arr[same_fixed_mask]
                if len(idx_arr) == 0:
                    continue
                orig_v1_idx = int(s_vidx1[s_v1_idx])
                orig_v2_indices = s_vidx2[idx_arr]
                for kk in range(len(idx_arr)):
                    is_fixed2 = bool(s_fixed2[idx_arr[kk]])
                    v.inter_muscle_constraints.append((
                        name1, orig_v1_idx, is_fixed1,
                        name2, int(orig_v2_indices[kk]), is_fixed2,
                        float(d_arr[kk])
                    ))

    print(f"Found {len(v.inter_muscle_constraints)} inter-muscle constraints")
    return len(v.inter_muscle_constraints)


def run_all_tet_sim_with_constraints(v, max_iterations=100, tolerance=1e-4, outer_iterations=20,
                                     snapshot_callback=None):
    """
    Run tet simulation for all muscles together, respecting inter-muscle constraints.
    Uses ARAP with collision detection integrated.

    If v.coupled_as_unified_volume is True, treats all muscles as one unified system.

    snapshot_callback(stage, active_muscles): if provided, called with stage="init"
    before any solve and stage=f"outer{i}" after each outer iter. Bake driver uses
    this to save per-iter positions for iron-man-style convergence playback.
    """
    # Get all muscles with soft body
    active_muscles = {}
    for name, mobj in v.zygote_muscle_meshes.items():
        if mobj.soft_body is not None:
            active_muscles[name] = mobj

    if len(active_muscles) == 0:
        print("No muscles with soft body initialized")
        return

    n_constraints = len(v.inter_muscle_constraints)

    if v.coupled_as_unified_volume:
        # Unified volume mode: treat all muscles as one system
        _run_unified_volume_sim(v, active_muscles, max_iterations, tolerance)
    else:
        # Standard mode: alternating individual solve + constraint enforcement
        print(f"Running coupled tet sim for {len(active_muscles)} muscles with {n_constraints} inter-muscle constraints...")

        # Build bone collision trimeshes ONCE per frame and share across all
        # muscles. Without this every muscle rebuilds them and trimesh
        # construction over the full skeleton is the dominant cost on dense
        # meshes (~hundreds of ms per muscle).
        shared_bone_meshes = None
        first_mobj = next(iter(active_muscles.values()))
        if getattr(first_mobj, 'soft_body_collision', False):
            shared_bone_meshes = first_mobj._build_transformed_collision_meshes(
                v.zygote_skeleton_meshes, v.env.skel, verbose=False
            )
            shared_bone_meshes.extend(
                first_mobj._build_dart_shape_collision_meshes(v.env.skel, verbose=False)
            )

        if snapshot_callback is not None:
            snapshot_callback("init", active_muscles)

        for outer_iter in range(outer_iterations):
            # Step 1: Run individual soft body solves
            total_residual = 0
            for name, mobj in active_muscles.items():
                iters, residual = mobj.run_soft_body_to_convergence(
                    v.zygote_skeleton_meshes,
                    v.env.skel,
                    max_iterations=max_iterations // outer_iterations,
                    tolerance=tolerance,
                    enable_collision=mobj.soft_body_collision,
                    collision_margin=mobj.soft_body_collision_margin,
                    collision_mesh_override=shared_bone_meshes,
                    verbose=False,
                    use_arap=mobj.use_arap
                )
                total_residual += residual

            # Step 2: Enforce inter-muscle constraints
            stop = False
            if n_constraints > 0:
                constraint_error = _enforce_inter_muscle_constraints(v, active_muscles)
                print(f"  Iter {outer_iter+1}/{outer_iterations}: residual={total_residual:.2e}, constraint_err={constraint_error:.6f}m")
                if constraint_error < tolerance:
                    print(f"  Constraints satisfied (error < {tolerance}), stopping")
                    stop = True
            else:
                print(f"  Iter {outer_iter+1}: residual={total_residual:.2e} (no constraints)")
                if total_residual < tolerance * len(active_muscles):
                    stop = True

            if snapshot_callback is not None:
                snapshot_callback(f"outer{outer_iter}", active_muscles)
            if stop:
                break

        print(f"Coupled tet sim complete ({len(active_muscles)} muscles)")


def _apply_tendon_elastic(cache, global_positions, scaled_rest):
    """Slack-only rest-length update for tendon-zone cross-contour edges.

    For each tendon cross-edge, if current edge length is shorter than the
    rest edge length, rescale the rest vector to match the current length
    (direction preserved).  Stretched edges keep their original rest.
    Result: tendon segments can collapse freely (buckle) in compression,
    behaving like elastic bands that go slack rather than fighting
    contraction.
    """
    tendon_mask = cache.get('csr_tendon_mask')
    if tendon_mask is None or not np.any(tendon_mask):
        return scaled_rest
    ei = cache['csr_edge_i']
    ej = cache['csr_edge_j']
    base_len = np.linalg.norm(scaled_rest, axis=1)
    cur_len = np.linalg.norm(global_positions[ej] - global_positions[ei], axis=1)
    slack = tendon_mask & (cur_len < base_len)
    if not np.any(slack):
        return scaled_rest
    base_safe = np.maximum(base_len, 1e-12)
    scale = np.where(slack, cur_len / base_safe, 1.0)
    out = scaled_rest.copy()
    out[slack] = scaled_rest[slack] * scale[slack, None]
    return out


def _apply_axial_pose_prior(v, cache, knee_angles):
    """Axial pose prior — pose-conditioned ARAP rest-edge rescaling.

    Replaces the old hardcoded "knee-angle contract" with a named, tunable
    primitive.  For each knee-crossing muscle (KNEE_CROSSING_MUSCLES from
    viewer/skin_prior.py), cross-contour rest edges shrink by `ratio` and
    intra-contour rest edges grow by sqrt(1/ratio) — volume-preserving.

    `ratio = 1 - (1 - min_ratio) * smoothstep(knee_angle / (π/2))` when
    v.axial_curve == 'smooth'; linear otherwise.  Parameters come from
    v.axial_min_ratio (default 0.65 — soft) and v.axial_max_bulge
    (default 2.0).  Returns (scaled_rest, crosser_strs) or (None, []).
    """
    from viewer.skin_prior import KNEE_CROSSING_MUSCLES
    if not v.use_muscle_aware_arap or cache.get('csr_cross_mask') is None:
        return None, []
    base_rest = cache['csr_rest_edges_base']
    cross_mask = cache['csr_cross_mask']
    intra_mask = cache.get('csr_intra_mask')
    csr_muscle_id = cache.get('csr_muscle_id')
    muscle_id_map = cache.get('muscle_id_map', {})
    if csr_muscle_id is None:
        return None, []
    min_ratio = float(getattr(v, 'axial_min_ratio', 0.65))
    max_bulge = float(getattr(v, 'axial_max_bulge', 2.0))
    curve = str(getattr(v, 'axial_curve', 'smooth'))
    HALF_PI = np.pi / 2.0
    scaled = base_rest.copy()
    notes = []
    any_change = False
    # Apply to all muscles in the unified system.  Per-muscle side ('L'/'R')
    # is taken from the muscle name prefix; non-prefixed muscles fall back
    # to max knee angle.
    for muscle_name, mid in muscle_id_map.items():
        if mid is None:
            continue
        side = muscle_name[0] if muscle_name and muscle_name[0] in 'LR' else None
        if side is not None:
            angle = knee_angles.get(side, 0.0)
        else:
            angle = max(knee_angles.values())
        a = float(np.clip(angle / HALF_PI, 0.0, 1.5))
        if curve == 'smooth':
            t = min(a, 1.0)
            s = t * t * (3.0 - 2.0 * t)
        else:
            s = min(a, 1.0)
        ratio = 1.0 - (1.0 - min_ratio) * s
        if abs(ratio - 1.0) < 0.01:
            continue
        any_change = True
        muscle_mask = (csr_muscle_id == mid)
        cross_edges = muscle_mask & cross_mask
        if np.any(cross_edges):
            scaled[cross_edges] = base_rest[cross_edges] * ratio
            notes.append(f"{muscle_name}={ratio:.2f}")
        if intra_mask is not None:
            perp_scale = float(np.clip(np.sqrt(1.0 / max(ratio, 0.05)),
                                       1.0, max_bulge))
            intra_edges = muscle_mask & intra_mask
            if np.any(intra_edges):
                scaled[intra_edges] = base_rest[intra_edges] * perp_scale
    if not any_change:
        return None, []
    return scaled, notes


def _project_positive_tet_volumes(positions, rest_positions, tets, fixed_mask,
                                  sweeps=12, stiffness=0.9,
                                  max_step=0.002):
    """XPBD-style rest-volume projection with a positive-Jacobian barrier.

    ARAP alone has no volumetric invariant.  This projection preserves each
    tet's signed rest volume and gives inverted/near-flat tets a stronger
    recovery target. Corrections are Jacobi-averaged per vertex and clamped so
    dense attachment rings cannot cause a one-iteration blowout.
    """
    if tets is None or len(tets) == 0:
        return positions, 0, 0.0
    x = np.asarray(positions, dtype=np.float64).copy()
    xr = np.asarray(rest_positions, dtype=np.float64)
    t = np.asarray(tets, dtype=np.int64)

    def signed_volume(p):
        q = p[t]
        return np.einsum('ij,ij->i',
                         q[:, 1] - q[:, 0],
                         np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 0])) / 6.0

    v0 = signed_volume(xr)
    usable = np.abs(v0) > 1e-14
    t = t[usable]
    v0 = v0[usable]
    if len(t) == 0:
        return x, 0, 0.0

    inv_mass = (~np.asarray(fixed_mask, dtype=bool)).astype(np.float64)
    for _ in range(max(int(sweeps), 0)):
        q = x[t]
        a, b, c, d = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        volume = np.einsum('ij,ij->i', b - a,
                           np.cross(c - a, d - a)) / 6.0
        gb = np.cross(c - a, d - a) / 6.0
        gc = np.cross(d - a, b - a) / 6.0
        gd = np.cross(b - a, c - a) / 6.0
        ga = -(gb + gc + gd)
        grads = np.stack((ga, gb, gc, gd), axis=1)
        w = inv_mass[t]
        denom = np.sum(w * np.sum(grads * grads, axis=2), axis=1) + 1e-18

        ratio = volume / v0
        # Ordinary incompressibility uses V-V0.  Once orientation is lost,
        # bias toward a small positive volume first instead of allowing the
        # element to remain on the wrong side of the singularity.
        target = v0.copy()
        bad = ratio < 0.05
        target[bad] = 0.20 * v0[bad]
        lam = np.clip((target - volume) / denom, -1e4, 1e4)
        corr = stiffness * w[:, :, None] * lam[:, None, None] * grads

        accum = np.zeros_like(x)
        count = np.zeros(len(x), dtype=np.float64)
        for corner in range(4):
            np.add.at(accum, t[:, corner], corr[:, corner])
            np.add.at(count, t[:, corner], w[:, corner])
        moving = count > 0
        delta = np.zeros_like(x)
        delta[moving] = accum[moving] / count[moving, None]
        length = np.linalg.norm(delta, axis=1)
        clamp = length > max_step
        delta[clamp] *= (max_step / (length[clamp] + 1e-18))[:, None]
        x[moving] += delta[moving]

    vf = signed_volume(xr)  # overwritten below only to retain simple shape
    q = x[t]
    vf = np.einsum('ij,ij->i', q[:, 1] - q[:, 0],
                   np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 0])) / 6.0
    ratio = vf / v0
    return x, int(np.sum(ratio <= 0.0)), float(np.max(np.abs(ratio - 1.0)))


def _run_unified_volume_sim(v, active_muscles, max_iterations=100, tolerance=1e-4):
    """
    Run simulation treating all muscles as one unified volume.
    Inter-muscle constraints become edges in a combined system.
    Supports GPU acceleration via PyTorch when v.use_gpu_arap is True.
    """
    import scipy.sparse
    import scipy.sparse.linalg
    import time

    if v.use_taichi_arap:
        backend_name = 'taichi'
    elif v.use_gpu_arap:
        backend_name = 'gpu'
    else:
        backend_name = 'cpu'
    print(f"Running UNIFIED volume sim for {len(active_muscles)} muscles... [{backend_name.upper()}]")

    # Step 1: Update each muscle's positions and fixed targets from skeleton
    print(f"  Updating positions from skeleton...")
    for name, mobj in active_muscles.items():
        # Update vertex positions based on skeleton bindings
        if hasattr(mobj, '_update_tet_positions_from_skeleton'):
            mobj._update_tet_positions_from_skeleton(v.env.skel)
        # Update fixed vertex targets (origins/insertions)
        if hasattr(mobj, '_update_fixed_targets_from_skeleton'):
            mobj._update_fixed_targets_from_skeleton(v.zygote_skeleton_meshes, v.env.skel)

    # Build per-frame bone trimeshes for the unified-path bone-contact
    # penalty.  Reuse from the per-muscle alternating branch's pattern
    # (line ~7078) — DART body world transforms applied to each bone
    # mesh, plus DART shape primitives.  Stashed on v for the closure
    # below.
    v._unified_bone_meshes = None
    if getattr(v, 'unified_bone_contact', True):
        first_mobj = next(iter(active_muscles.values()))
        bone_meshes = first_mobj._build_transformed_collision_meshes(
            v.zygote_skeleton_meshes, v.env.skel, verbose=False)
        bone_meshes.extend(
            first_mobj._build_dart_shape_collision_meshes(v.env.skel, verbose=False))
        v._unified_bone_meshes = bone_meshes

    # Check if we have valid cached topology from a previous frame
    muscle_names = list(active_muscles.keys())
    total_verts = sum(active_muscles[n].soft_body.num_vertices for n in muscle_names)

    cache = getattr(v, '_unified_sim_cache', None)
    cache_valid = (cache is not None
                   and cache['muscle_names'] == muscle_names
                   and cache['total_verts'] == total_verts)

    if cache_valid:
        # Reuse cached topology
        global_offset = cache['global_offset']
        global_rest_positions = cache['global_rest_positions']
        global_fixed_mask = cache['global_fixed_mask']
        global_tets = cache.get('global_tets')
        neighbors = cache['neighbors']
        edge_weights = cache['edge_weights']
        rest_edge_vectors = cache['rest_edge_vectors']
        print(f"  Using cached topology ({total_verts} verts)")
    else:
        # Build global vertex indexing
        global_offset = {}  # muscle_name -> starting index in global array
        offset_accum = 0
        for name in muscle_names:
            global_offset[name] = offset_accum
            offset_accum += active_muscles[name].soft_body.num_vertices

        print(f"  Total vertices: {total_verts}")

        # Collect rest positions and fixed mask (topology-invariant)
        global_rest_positions = np.zeros((total_verts, 3))
        global_fixed_mask = np.zeros(total_verts, dtype=bool)

        for name, mobj in active_muscles.items():
            offset = global_offset[name]
            n = mobj.soft_body.num_vertices
            global_rest_positions[offset:offset+n] = mobj.soft_body.rest_positions
            global_fixed_mask[offset:offset+n] = mobj.soft_body.fixed_mask

        global_tet_blocks = []
        for name, mobj in active_muscles.items():
            local_tets = np.asarray(getattr(mobj, 'tet_tetrahedra', []), dtype=np.int64)
            if local_tets.ndim == 2 and local_tets.shape[1] == 4 and len(local_tets):
                global_tet_blocks.append(local_tets + global_offset[name])
        global_tets = (np.concatenate(global_tet_blocks, axis=0)
                       if global_tet_blocks else np.zeros((0, 4), dtype=np.int64))

        # Build combined edge list (internal edges + inter-muscle constraints)
        all_edges = []  # (global_i, global_j, rest_length, weight)

        # Edge classification for muscle-aware ARAP
        edge_type_map = {}  # (gi, gj) -> 1=cross, 2=intra, 0=neutral
        edge_muscle_map = {}  # (gi, gj) -> muscle_name
        muscle_rest_axis_len = {}  # name -> rest distance between first/last fixed verts
        muscle_fixed_global = {}  # name -> list of global indices of fixed verts

        # Add internal edges from each muscle
        for name, mobj in active_muscles.items():
            offset = global_offset[name]
            sb = mobj.soft_body

            # Compute rest axis length from fixed vertices
            if hasattr(sb, 'fixed_indices') and sb.fixed_indices is not None and len(sb.fixed_indices) >= 2:
                rest_fixed = sb.rest_positions[sb.fixed_indices]
                muscle_rest_axis_len[name] = np.linalg.norm(rest_fixed[-1] - rest_fixed[0])
                muscle_fixed_global[name] = [offset + fi for fi in sb.fixed_indices]
            else:
                muscle_rest_axis_len[name] = 0.0
                muscle_fixed_global[name] = []

            has_edge_types = (hasattr(sb, 'cross_contour_edges') and sb.cross_contour_edges is not None and
                              hasattr(sb, 'intra_contour_edges') and sb.intra_contour_edges is not None)

            # Anisotropic ARAP: muscle-fiber-aware weighting.
            # Cross-contour edges run roughly along the fiber direction (between
            # adjacent contour rings); intra-contour edges run perpendicular
            # (around a ring).  Lower cross weight → fiber direction can
            # contract more freely; higher intra weight → cross-section keeps
            # its width, producing perpendicular bulging.  Disabled by default
            # (isotropic w=1.0 everywhere).
            arap_aniso = bool(getattr(v, 'arap_anisotropic', False))
            cross_w = float(getattr(v, 'arap_cross_w', 1.0))
            intra_w = float(getattr(v, 'arap_intra_w', 1.0))
            neutral_w = float(getattr(v, 'arap_neutral_w', 1.0))
            # Tracked contact can pull the belly while the terminal cap is
            # hard-fixed, concentrating stretch and twist in the first free
            # contour interval.  Preserve the attachment transition with a
            # smoothly tapered ARAP weight.  This is topology-driven and
            # applies uniformly to both ends of every muscle.
            attach_stiff = (float(getattr(v, 'attachment_rigidity', 1.0))
                            if getattr(v, 'tracked_bone_contact', False)
                            else 1.0)
            attach_rings = max(
                0, int(getattr(v, 'attachment_rigidity_rings', 0)))
            vcl = np.asarray(
                getattr(mobj, 'vertex_contour_level',
                        np.full(sb.num_vertices, -1)), dtype=np.int32)
            valid_vcl = (vcl.size == sb.num_vertices
                         and np.any(vcl >= 0) and attach_rings > 0
                         and attach_stiff > 1.0)
            max_vcl = int(vcl.max()) if valid_vcl else -1

            for edge_idx, (i, j) in enumerate(zip(sb.edge_i, sb.edge_j)):
                # Use stored rest_lengths if available, otherwise compute from rest positions
                if hasattr(sb, 'rest_lengths') and sb.rest_lengths is not None and edge_idx < len(sb.rest_lengths):
                    rest_len = sb.rest_lengths[edge_idx]
                else:
                    rest_len = np.linalg.norm(sb.rest_positions[j] - sb.rest_positions[i])
                etype = 0
                if has_edge_types:
                    if sb.cross_contour_edges[edge_idx]:
                        etype = 1
                    elif sb.intra_contour_edges[edge_idx]:
                        etype = 2
                if arap_aniso and has_edge_types:
                    w = cross_w if etype == 1 else (intra_w if etype == 2 else neutral_w)
                else:
                    w = 1.0
                if valid_vcl and vcl[i] >= 0 and vcl[j] >= 0:
                    edge_level = 0.5 * (float(vcl[i]) + float(vcl[j]))
                    cap_dist = min(edge_level, max_vcl - edge_level)
                    if cap_dist < attach_rings:
                        taper = 1.0 - max(0.0, cap_dist) / attach_rings
                        w *= 1.0 + (attach_stiff - 1.0) * taper * taper
                all_edges.append((offset + i, offset + j, rest_len, w))

                gi, gj = offset + i, offset + j
                edge_muscle_map[(gi, gj)] = name
                edge_muscle_map[(gj, gi)] = name
                edge_type_map[(gi, gj)] = etype
                edge_type_map[(gj, gi)] = etype

        n_internal = len(all_edges)

        # Add inter-muscle constraints as edges
        inter_w = float(getattr(v, 'inter_muscle_weight', 1.0))
        inter_degree = np.zeros(total_verts, dtype=np.int32)
        inter_global = []
        for constraint in v.inter_muscle_constraints:
            name1, v1_idx, _, name2, v2_idx, _, rest_dist = constraint
            if name1 not in active_muscles or name2 not in active_muscles:
                continue
            gi = global_offset[name1] + v1_idx
            gj = global_offset[name2] + v2_idx
            inter_global.append((gi, gj, rest_dist))
            inter_degree[gi] += 1
            inter_degree[gj] += 1
        for global_i, global_j, rest_dist in inter_global:
            degree_scale = np.sqrt(
                max(inter_degree[global_i], 1)
                * max(inter_degree[global_j], 1))
            all_edges.append((
                global_i, global_j, rest_dist,
                inter_w / degree_scale))

        n_inter = len(all_edges) - n_internal
        print(f"  Edges: {n_internal} internal + {n_inter} inter-muscle = {len(all_edges)} total")

        # Build neighbor list
        neighbors = [[] for _ in range(total_verts)]
        edge_weights = {}
        rest_edge_vectors = [{} for _ in range(total_verts)]

        for gi, gj, rest_len, weight in all_edges:
            neighbors[gi].append(gj)
            neighbors[gj].append(gi)
            edge_weights[(gi, gj)] = weight
            edge_weights[(gj, gi)] = weight
            # Store rest edge vectors
            rest_edge_vectors[gi][gj] = global_rest_positions[gj] - global_rest_positions[gi]
            rest_edge_vectors[gj][gi] = global_rest_positions[gi] - global_rest_positions[gj]

        # Debug: check neighbor distribution
        neighbor_counts = [len(neighbors[i]) for i in range(total_verts)]
        n0 = sum(1 for c in neighbor_counts if c == 0)
        n1 = sum(1 for c in neighbor_counts if c == 1)
        n2 = sum(1 for c in neighbor_counts if c == 2)
        if n0 > 0 or n1 > 0:
            print(f"  WARNING: {n0} isolated, {n1} single-neighbor, {n2} two-neighbor vertices")

        # Build CSR-ordered arrays for muscle-aware ARAP
        # Must iterate in same order as backend's build_system: for i, neighs in enumerate(neighbors): for j in neighs
        muscle_id_map = {name: mid for mid, name in enumerate(muscle_names)}
        n_csr_edges = sum(len(neighs) for neighs in neighbors)
        csr_cross_mask = np.zeros(n_csr_edges, dtype=bool)
        csr_intra_mask = np.zeros(n_csr_edges, dtype=bool)
        csr_muscle_id = np.full(n_csr_edges, -1, dtype=np.int32)
        csr_rest_edges_base = np.zeros((n_csr_edges, 3), dtype=np.float64)
        csr_edge_i = np.zeros(n_csr_edges, dtype=np.int32)
        csr_edge_j = np.zeros(n_csr_edges, dtype=np.int32)
        csr_idx = 0
        for i, neighs in enumerate(neighbors):
            for j in neighs:
                csr_rest_edges_base[csr_idx] = global_rest_positions[j] - global_rest_positions[i]
                csr_edge_i[csr_idx] = i
                csr_edge_j[csr_idx] = j
                edge_key = (i, j)
                etype = edge_type_map.get(edge_key, 0)
                csr_cross_mask[csr_idx] = (etype == 1)
                csr_intra_mask[csr_idx] = (etype == 2)
                mname = edge_muscle_map.get(edge_key, None)
                if mname is not None:
                    csr_muscle_id[csr_idx] = muscle_id_map[mname]
                csr_idx += 1

        n_cross = int(csr_cross_mask.sum())
        n_intra = int(csr_intra_mask.sum())
        print(f"  Muscle-aware ARAP: {n_cross} cross-contour, {n_intra} intra-contour CSR edges")

        # Tendon-zone per-edge mask.  A vertex is in the tendon zone if its
        # vertex_contour_level falls in {1..5} (5 levels after origin level 0)
        # or in {N-6..N-2} (5 levels before insertion level N-1).  A cross-
        # contour edge is "tendon" iff BOTH endpoints are in the tendon zone.
        # At solve time these edges use a slack-only rest update (elastic
        # band in tension, buckles in compression) to let tendon segments
        # collapse along the origin→insertion line when they cross a flexed
        # joint.
        # Tendon zone: 5 levels right after origin (1..5) AND 5 levels right
        # before insertion (max-5..max-1).  Belly stays pure ARAP; only
        # tendon edges get the fiber-spring (and slack-only) elastic energy.
        TENDON_LEVELS = 2
        tendon_vert_global = np.zeros(total_verts, dtype=bool)
        for name, mobj in active_muscles.items():
            offset = global_offset[name]
            n = mobj.soft_body.num_vertices
            vcl = np.asarray(getattr(mobj, 'vertex_contour_level',
                                     np.full(n, -1)), dtype=np.int32)
            if vcl.size != n:
                continue
            max_level = int(vcl.max())
            if max_level < 2 * TENDON_LEVELS:
                continue
            tend_local = (((vcl >= 1) & (vcl <= TENDON_LEVELS))
                          | ((vcl >= max_level - TENDON_LEVELS)
                             & (vcl <= max_level - 1)))
            tendon_vert_global[offset:offset + n] = tend_local
        csr_tendon_mask = (tendon_vert_global[csr_edge_i]
                           & tendon_vert_global[csr_edge_j]
                           & csr_cross_mask)
        n_tendon = int(csr_tendon_mask.sum())
        print(f"  Tendon zones: {int(tendon_vert_global.sum())} verts, "
              f"{n_tendon} tendon cross-edges")

        # Collision candidate vertices: non-fixed surface verts per muscle.
        # Used by the unified-path bone contact penalty.  Cap-attached / anchor
        # verts (already in global_fixed_mask) are excluded automatically.
        from viewer.bone_surface_collision import compute_surface_topology
        collision_vertex_set = set()
        for name, mobj in active_muscles.items():
            offset = global_offset[name]
            n = mobj.soft_body.num_vertices
            tet_faces = getattr(mobj, 'tet_faces', None)
            tet_tetrahedra = getattr(mobj, 'tet_tetrahedra', None)
            surf_vidx, _ = compute_surface_topology(
                tetrahedra=tet_tetrahedra, tet_faces=tet_faces)
            if len(surf_vidx) == 0:
                surf_vidx = np.arange(n, dtype=np.int64)
            for vi in surf_vidx:
                gi = offset + int(vi)
                if gi < total_verts and not global_fixed_mask[gi]:
                    collision_vertex_set.add(gi)

        # Per-muscle ANATOMICAL surface faces in global indices (cap faces
        # excluded).  Used for paper-§4.1.2 dynamic anisotropic contact.
        anat_face_global_list = []
        vert_owner_global = -np.ones(total_verts, dtype=np.int32)
        for m_id, name in enumerate(muscle_names):
            mobj = active_muscles[name]
            offset = global_offset[name]
            n = mobj.soft_body.num_vertices
            vert_owner_global[offset:offset + n] = m_id
            F_render = getattr(mobj, 'tet_render_faces', None)
            cap = getattr(mobj, 'tet_cap_face_indices', None)
            if F_render is None:
                continue
            F_render = np.asarray(F_render, dtype=np.int32)
            cap = np.asarray(cap if cap is not None else [], dtype=np.int64)
            mask = np.ones(len(F_render), dtype=bool)
            mask[cap] = False
            anat = F_render[mask].astype(np.int64) + offset  # global
            anat_face_global_list.append(anat)
        anat_face_global = (np.concatenate(anat_face_global_list, axis=0)
                            if anat_face_global_list
                            else np.zeros((0, 3), dtype=np.int64))

        # Cache topology for subsequent frames
        v._unified_sim_cache = {
            'global_offset': global_offset,
            'total_verts': total_verts,
            'global_rest_positions': global_rest_positions,
            'global_fixed_mask': global_fixed_mask,
            'global_tets': global_tets,
            'neighbors': neighbors,
            'edge_weights': edge_weights,
            'rest_edge_vectors': rest_edge_vectors,
            'muscle_names': muscle_names,
            'muscle_id_map': muscle_id_map,
            'muscle_rest_axis_len': muscle_rest_axis_len,
            'muscle_fixed_global': muscle_fixed_global,
            'csr_cross_mask': csr_cross_mask,
            'csr_intra_mask': csr_intra_mask,
            'csr_muscle_id': csr_muscle_id,
            'csr_rest_edges_base': csr_rest_edges_base,
            'csr_edge_i': csr_edge_i,
            'csr_edge_j': csr_edge_j,
            'csr_tendon_mask': csr_tendon_mask,
            'collision_vertex_set': collision_vertex_set,
            'anat_face_global': anat_face_global,
            'vert_owner_global': vert_owner_global,
        }

        # Paper §4.1.2 barycentric fascia constraints (pre-compute at A-pose).
        # For each anatomical surface vert, bind to barycentric position on
        # the nearest other-muscle anatomical triangle within threshold.
        if getattr(v, 'fascia_constraints_on', False) and len(anat_face_global) > 0:
            fc_threshold = float(getattr(v, 'fascia_constraint_threshold', 0.01))
            tri_centroids = global_rest_positions[anat_face_global].mean(axis=1)
            tri_owner = vert_owner_global[anat_face_global[:, 0]]
            anat_v_unique = np.unique(anat_face_global)
            from scipy.spatial import cKDTree as _cKDT_fc
            # A binding stores a *normal gap*, not a welded 3-D position.
            # Keeping the full barycentric point as a positional target locks
            # both tangent directions and makes neighbouring muscles tangle.
            # The normal-gap formulation below preserves the rest interface
            # thickness while leaving tangential motion unconstrained.
            fc_vi, fc_tri, fc_bary, fc_gap, fc_pair = [], [], [], [], []
            for m_id in np.unique(tri_owner):
                other_mask = tri_owner != m_id
                if not other_mask.any():
                    continue
                tree = _cKDT_fc(tri_centroids[other_mask])
                back = np.where(other_mask)[0]
                my_verts = anat_v_unique[vert_owner_global[anat_v_unique] == m_id]
                my_pos = global_rest_positions[my_verts]
                d, idx = tree.query(my_pos, k=1)
                keep = d < fc_threshold
                for k in range(len(my_verts)):
                    if not keep[k]:
                        continue
                    tri_g = back[idx[k]]
                    tri_v = anat_face_global[tri_g]
                    a = global_rest_positions[tri_v[0]]
                    b = global_rest_positions[tri_v[1]]
                    c = global_rest_positions[tri_v[2]]
                    nrm = np.cross(b - a, c - a)
                    n_unit = nrm / (np.linalg.norm(nrm) + 1e-12)
                    p = my_pos[k] - np.dot(my_pos[k] - a, n_unit) * n_unit
                    v0 = b - a; v1 = c - a; v2 = p - a
                    d00 = np.dot(v0, v0); d01 = np.dot(v0, v1); d11 = np.dot(v1, v1)
                    d20 = np.dot(v2, v0); d21 = np.dot(v2, v1)
                    denom = d00 * d11 - d01 * d01 + 1e-12
                    v_b = (d11 * d20 - d01 * d21) / denom
                    w_b = (d00 * d21 - d01 * d20) / denom
                    u_b = 1.0 - v_b - w_b
                    # Reject projections far outside the triangle.  A nearest
                    # centroid is only a broad-phase candidate; accepting an
                    # arbitrary extrapolated barycentric point creates long,
                    # non-anatomical cross-muscle constraints.
                    bary = np.array([u_b, v_b, w_b], dtype=np.float64)
                    if np.min(bary) < -0.15 or np.max(bary) > 1.15:
                        continue
                    closest = bary[0] * a + bary[1] * b + bary[2] * c
                    gap = float(np.dot(my_pos[k] - closest, n_unit))
                    if abs(gap) > fc_threshold:
                        continue
                    fc_vi.append(int(my_verts[k]))
                    fc_tri.append(tri_v.astype(np.int64))
                    fc_bary.append(bary)
                    fc_gap.append(gap)
                    fc_pair.append((int(m_id), int(tri_owner[tri_g])))

            # Keep only substantial, reciprocal rest-pose interfaces.  Tiny
            # proximity clusters are incidental contacts and must be handled
            # by collision, not converted into permanent fascia.
            if fc_pair:
                from collections import Counter as _Counter_fc
                pair_counts = _Counter_fc(fc_pair)
                min_patch = int(getattr(v, 'fascia_min_patch_vertices', 12))
                keep_idx = []
                for kk, pair in enumerate(fc_pair):
                    reverse = (pair[1], pair[0])
                    if pair_counts[pair] >= min_patch and pair_counts[reverse] >= min_patch:
                        keep_idx.append(kk)
                fc_vi = [fc_vi[k] for k in keep_idx]
                fc_tri = [fc_tri[k] for k in keep_idx]
                fc_bary = [fc_bary[k] for k in keep_idx]
                fc_gap = [fc_gap[k] for k in keep_idx]
            v._unified_sim_cache['fc_vi'] = np.array(fc_vi, dtype=np.int64) if fc_vi else np.zeros(0, dtype=np.int64)
            v._unified_sim_cache['fc_tri'] = np.stack(fc_tri, axis=0) if fc_tri else np.zeros((0, 3), dtype=np.int64)
            v._unified_sim_cache['fc_bary'] = np.stack(fc_bary, axis=0) if fc_bary else np.zeros((0, 3), dtype=np.float64)
            v._unified_sim_cache['fc_gap'] = np.asarray(fc_gap, dtype=np.float64) if fc_gap else np.zeros(0, dtype=np.float64)
            print(f"  Sliding fascia interfaces (normal-gap): {len(fc_vi)} bindings, "
                  f"threshold={fc_threshold*1000:.1f}mm")
        print(f"  Collision candidates: {len(collision_vertex_set)} non-fixed surface verts")

    # Compute LBS positions from skinning weights + skeleton transforms.
    # This gives ALL vertices skeleton-consistent positions as ARAP initial guess.
    def rigid_harmonic_coordinate(mobj):
        cached_coordinate = getattr(
            mobj, '_rigid_harmonic_coordinate', None)
        n_local = mobj.soft_body.num_vertices
        if (cached_coordinate is not None
                and len(cached_coordinate) == n_local):
            return cached_coordinate
        base = np.asarray([
            (float(binding[2]) if binding is not None else 0.5)
            for binding in mobj.tet_skeleton_bindings
        ], dtype=np.float64)
        anchors = getattr(mobj, 'soft_body_local_anchors', {}) or {}
        boundary = {}
        for anchor_vi, (anchor_bone, _) in anchors.items():
            binding = mobj.tet_skeleton_bindings[int(anchor_vi)]
            if binding is None:
                continue
            if anchor_bone == binding[0]:
                boundary[int(anchor_vi)] = 0.0
            elif anchor_bone == binding[1]:
                boundary[int(anchor_vi)] = 1.0
        if not boundary or not any(v == 0.0 for v in boundary.values()) \
                or not any(v == 1.0 for v in boundary.values()):
            return base
        tets_local = np.asarray(
            mobj.soft_body.tetrahedra, dtype=np.int64)
        edge_set = set()
        for tet in tets_local:
            for edge_a in range(4):
                for edge_b in range(edge_a + 1, 4):
                    edge_set.add(tuple(sorted(
                        (int(tet[edge_a]), int(tet[edge_b])))))
        edges_local = np.asarray(sorted(edge_set), dtype=np.int64)
        row = np.concatenate((edges_local[:, 0], edges_local[:, 1]))
        col = np.concatenate((edges_local[:, 1], edges_local[:, 0]))
        weight = np.ones(len(row), dtype=np.float64)
        adjacency_matrix = scipy.sparse.coo_matrix(
            (weight, (row, col)), shape=(n_local, n_local)).tocsr()
        degree = np.asarray(adjacency_matrix.sum(axis=1)).ravel()
        laplacian = scipy.sparse.diags(degree) - adjacency_matrix
        fixed_local = np.asarray(sorted(boundary), dtype=np.int64)
        fixed_value = np.asarray(
            [boundary[int(i)] for i in fixed_local], dtype=np.float64)
        fixed_mask_local = np.zeros(n_local, dtype=bool)
        fixed_mask_local[fixed_local] = True
        free_local = np.where(~fixed_mask_local)[0]
        coordinate = base.copy()
        coordinate[fixed_local] = fixed_value
        if len(free_local):
            lhs = laplacian[free_local][:, free_local].tocsc()
            rhs = -(laplacian[free_local][:, fixed_local]
                    @ fixed_value)
            coordinate[free_local] = scipy.sparse.linalg.spsolve(
                lhs, rhs)
        coordinate = np.clip(coordinate, 0.0, 1.0)
        mobj._rigid_harmonic_coordinate = coordinate
        return coordinate

    def rigid_material_coordinate(mobj):
        """Use the fiber binding coordinate for sweep deformation.

        The harmonic attachment coordinate is unsuitable for adductor magnus:
        its broad femoral insertion is a long surface, not a terminal cap.
        Treating every insertion vertex as u=1 creates a discontinuity through
        nearby tetrahedra.  The stored binding coordinate remains smooth.
        """
        base = np.asarray([
            (float(binding[2]) if binding is not None else 0.5)
            for binding in mobj.tet_skeleton_bindings
        ], dtype=np.float64)
        # A muscle with a broad femoral insertion is fan-shaped rather than
        # a simple end-to-end tube. Its imported fiber coordinate is strongly
        # femur-biased. The attachment-driven harmonic field lets the pelvic
        # portion actually follow the pelvis while smoothly approaching the
        # full femoral insertion surface.
        insertion_bones = {
            binding[1] for binding in mobj.tet_skeleton_bindings
            if binding is not None
        }
        if any('Femur' in bone for bone in insertion_bones):
            return rigid_harmonic_coordinate(mobj)
        return base

    global_lbs_positions = np.zeros((total_verts, 3))
    for name, mobj in active_muscles.items():
        offset = global_offset[name]
        n = mobj.soft_body.num_vertices
        rest = mobj.soft_body.rest_positions

        if (getattr(v, 'rigid_blend_init', False)
                and getattr(mobj, 'tet_skeleton_bindings', None)):
            from tools.bake_emu import compute_rigid_blend_positions
            initial = getattr(mobj, 'tet_initial_bone_transforms', {})
            rigid_bindings = []
            blend_coordinate = []
            harmonic_coordinate = rigid_material_coordinate(mobj)
            for binding_vi, binding in enumerate(
                    mobj.tet_skeleton_bindings):
                if binding is None:
                    rigid_bindings.append((np.zeros(3), []))
                    blend_coordinate.append(0.0)
                    continue
                origin, insertion, _, rest_vertex = binding
                weight = float(harmonic_coordinate[binding_vi])
                weighted_bones = []
                if origin in initial:
                    R0, t0 = initial[origin]
                    weighted_bones.append(
                        (origin, 1.0 - weight, R0, t0))
                if insertion in initial:
                    R0, t0 = initial[insertion]
                    weighted_bones.append(
                        (insertion, weight, R0, t0))
                rigid_bindings.append(
                    (np.asarray(rest_vertex, dtype=np.float64),
                     weighted_bones))
                blend_coordinate.append(weight)
            rigid = compute_rigid_blend_positions(
                rigid_bindings, v.env.skel,
                np.asarray(blend_coordinate, dtype=np.float64))
            if rigid.shape == (n, 3) and np.all(np.isfinite(rigid)):
                global_lbs_positions[offset:offset+n] = rigid
                continue

        if hasattr(mobj, 'skinning_weights') and mobj.skinning_weights is not None and len(mobj.skinning_bones) > 0:
            # Compute LBS: blend bone transforms weighted by skinning weights
            lbs = np.zeros((n, 3))
            for bone_idx, bone_name in enumerate(mobj.skinning_bones):
                body_node = v.env.skel.getBodyNode(bone_name)
                if body_node is None:
                    continue
                wt = body_node.getWorldTransform()
                R = wt.rotation()
                t = wt.translation()
                # Get rest transform
                if bone_name in mobj.soft_body_initial_transforms:
                    R0, t0 = mobj.soft_body_initial_transforms[bone_name]
                else:
                    continue
                # Deformation: p' = R_cur @ R_rest^T @ (p_rest - t_rest) + t_cur
                w = mobj.skinning_weights[:, bone_idx:bone_idx+1]  # (n, 1)
                local = (R0.T @ (rest - t0).T).T  # (n, 3) in rest bone frame
                deformed = (R @ local.T).T + t     # (n, 3) in current world
                lbs += w * deformed
            global_lbs_positions[offset:offset+n] = lbs
        else:
            # No skinning weights — use current positions
            global_lbs_positions[offset:offset+n] = mobj.soft_body.positions

    # Blend LBS with warm-start for temporal coherence
    prev_solution = cache.get('prev_solution', None) if cache_valid else None
    if prev_solution is not None and prev_solution.shape[0] == total_verts:
        lbs_weight = float(getattr(v, 'lbs_init_weight', 0.0))
        global_positions = lbs_weight * global_lbs_positions + (1 - lbs_weight) * prev_solution
        fixed_idx = np.where(global_fixed_mask)[0]
        global_positions[fixed_idx] = global_lbs_positions[fixed_idx]
        print(f"  LBS({lbs_weight:.0%}) + warm({1-lbs_weight:.0%})")
    else:
        global_positions = global_lbs_positions.copy()
        init_name = ("rigid-blend" if getattr(v, 'rigid_blend_init', False)
                     else "LBS")
        print(f"  {init_name} initial guess")

    if getattr(v, 'rigid_blend_init', False):
        init_ratios = []
        for init_name, mobj in active_muscles.items():
            init_offset = global_offset[init_name]
            init_n = mobj.soft_body.num_vertices
            init_tets = np.asarray(mobj.soft_body.tetrahedra, dtype=np.int64)
            init_rest = np.asarray(mobj.soft_body.rest_positions)
            init_pose = global_positions[init_offset:init_offset + init_n]
            rest_q = init_rest[init_tets]
            pose_q = init_pose[init_tets]
            rest_det = np.einsum(
                'ij,ij->i', rest_q[:, 0] - rest_q[:, 3],
                np.cross(rest_q[:, 1] - rest_q[:, 3],
                         rest_q[:, 2] - rest_q[:, 3]))
            pose_det = np.einsum(
                'ij,ij->i', pose_q[:, 0] - pose_q[:, 3],
                np.cross(pose_q[:, 1] - pose_q[:, 3],
                         pose_q[:, 2] - pose_q[:, 3]))
            init_ratios.append(pose_det / np.where(
                np.abs(rest_det) > 1e-15, rest_det, 1.0))
        init_ratios = np.concatenate(init_ratios)
        print(f"  Rigid-blend Jacobian: inv={np.sum(init_ratios <= 0)}/"
              f"{len(init_ratios)}, J=[{init_ratios.min():.3f},"
              f"{init_ratios.max():.3f}]")

    global_fixed_targets = {}  # global_idx -> target position
    for name, mobj in active_muscles.items():
        offset = global_offset[name]
        # Store fixed targets
        if mobj.soft_body.fixed_targets is not None and len(mobj.soft_body.fixed_indices) > 0:
            for local_idx, target in zip(mobj.soft_body.fixed_indices, mobj.soft_body.fixed_targets):
                global_fixed_targets[offset + local_idx] = target

    # Debug: check if fixed targets differ from rest
    max_fixed_diff = 0.0
    for gi, target in global_fixed_targets.items():
        diff = np.linalg.norm(target - global_rest_positions[gi])
        max_fixed_diff = max(max_fixed_diff, diff)

    n_fixed_mask = np.sum(global_fixed_mask)
    n_fixed_targets = len(global_fixed_targets)
    print(f"  Fixed: {n_fixed_mask} in mask, {n_fixed_targets} with targets, max displacement from rest: {max_fixed_diff:.4f}m")
    if n_fixed_mask != n_fixed_targets:
        print(f"  WARNING: Mismatch between fixed_mask ({n_fixed_mask}) and fixed_targets ({n_fixed_targets})")

    # Knee angles (used to switch between extension/flex backend bins and
    # to drive cross-contour edge contraction below).  Subtract the rest
    # baseline R_rel so T-pose = 0° (bones aren't exactly aligned in XML).
    from viewer.skin_prior import KNEE_CROSSING_MUSCLES
    skel_local = v.env.skel
    rest_R_rel = getattr(v, '_rest_knee_R_rel', {'L': np.eye(3), 'R': np.eye(3)})
    knee_angles = {'L': 0.0, 'R': 0.0}
    for side, body_pair in (('L', ('L_Femur0', 'L_Tibia_Fibula0')),
                            ('R', ('R_Femur0', 'R_Tibia_Fibula0'))):
        fem = skel_local.getBodyNode(body_pair[0])
        tib = skel_local.getBodyNode(body_pair[1])
        if fem is None or tib is None:
            continue
        fR = np.array(fem.getWorldTransform().rotation())
        tR = np.array(tib.getWorldTransform().rotation())
        rel_now = tR @ fR.T
        rel_baseline = rest_R_rel.get(side, np.eye(3))
        delta = rel_now @ rel_baseline.T
        tr = float(np.trace(delta))
        knee_angles[side] = float(np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0)))

    # Two-bin matrix cache: separate factorized backends for "extension"
    # (full skin prior on crossers) and "flex" (reduced skin prior on
    # crossers so contraction wins).  Threshold: knee angle ≥ 45° → flex.
    KNEE_FLEX_THRESHOLD = np.pi / 4.0      # 45°
    FLEX_CROSSER_SP_SCALE = 0.0            # scale skin prior weight on
                                           # crossers when in flex bin (0 = off)
    max_knee = max(knee_angles.values())
    in_flex_bin = max_knee >= KNEE_FLEX_THRESHOLD
    bin_name = 'flex' if in_flex_bin else 'ext'
    backend_attr = f'_unified_arap_backend_{bin_name}'

    cached = getattr(v, backend_attr, None)
    if cached is not None and getattr(cached, '_backend_name', None) == backend_name:
        backend = cached
    else:
        backend = get_backend(backend_name)
        backend._backend_name = backend_name
        setattr(v, backend_attr, backend)
    # Track which bin is currently active so downstream callers see it
    v._unified_arap_backend = backend

    fixed_indices = np.where(global_fixed_mask)[0]
    fixed_targets_array = np.array([global_fixed_targets.get(i, global_rest_positions[i]) for i in fixed_indices])

    if getattr(v, 'rigid_blend_only', False):
        rigid_coordinates = {}
        for rigid_name, mobj in active_muscles.items():
            rigid_offset = global_offset[rigid_name]
            rigid_n = mobj.soft_body.num_vertices
            local_fixed = np.asarray(
                mobj.soft_body.fixed_indices, dtype=np.int64)
            if not len(local_fixed):
                continue
            global_fixed = rigid_offset + local_fixed
            cap_delta = np.stack([
                global_fixed_targets[int(gi)] - global_positions[int(gi)]
                for gi in global_fixed])
            coordinate = rigid_material_coordinate(mobj)
            rigid_coordinates[rigid_name] = coordinate
            fixed_coordinate = coordinate[local_fixed]
            origin = fixed_coordinate < 0.5
            insertion = ~origin
            origin_delta = (np.mean(cap_delta[origin], axis=0)
                            if np.any(origin) else np.mean(cap_delta, axis=0))
            insertion_delta = (np.mean(cap_delta[insertion], axis=0)
                               if np.any(insertion) else np.mean(cap_delta, axis=0))
            rest_local = global_rest_positions[
                rigid_offset:rigid_offset + rigid_n]
            raw_local = global_positions[
                rigid_offset:rigid_offset + rigid_n].copy()

            # Build paired rest/posed centerlines from longitudinal material
            # bins, then sweep the native cross-sections along the corrected
            # posed centerline. This preserves thickness under flexion instead
            # of letting endpoint displacement become axial compression.
            sample_u = np.linspace(0.0, 1.0, 33)
            bin_id = np.clip(
                np.rint(coordinate * (len(sample_u) - 1)).astype(np.int32),
                0, len(sample_u) - 1)
            rest_center = np.full((len(sample_u), 3), np.nan)
            pose_center = np.full((len(sample_u), 3), np.nan)
            for center_i in range(len(sample_u)):
                selected = bin_id == center_i
                if np.any(selected):
                    rest_center[center_i] = np.mean(
                        rest_local[selected], axis=0)
                    pose_center[center_i] = np.mean(
                        raw_local[selected], axis=0)
            valid = np.where(np.isfinite(rest_center[:, 0]))[0]
            for axis in range(3):
                rest_center[:, axis] = np.interp(
                    np.arange(len(sample_u)), valid,
                    rest_center[valid, axis])
                pose_center[:, axis] = np.interp(
                    np.arange(len(sample_u)), valid,
                    pose_center[valid, axis])

            correction = (
                (1.0 - sample_u)[:, None] * origin_delta
                + sample_u[:, None] * insertion_delta)
            corrected_center = pose_center + correction
            rest_tangent = np.gradient(rest_center, sample_u, axis=0)
            pose_tangent = np.gradient(corrected_center, sample_u, axis=0)
            rest_speed = np.linalg.norm(rest_tangent, axis=1)
            pose_speed = np.linalg.norm(pose_tangent, axis=1)
            stretch = pose_speed / np.maximum(rest_speed, 1e-8)
            bulge = np.clip(1.0 / np.sqrt(np.maximum(stretch, 0.15)),
                            0.75, 1.8)
            # Tendon caps remain narrow; volume compensation acts primarily
            # on the contractile belly.
            belly = np.sin(np.pi * sample_u) ** 2
            bulge = 1.0 + belly * (bulge - 1.0)

            def sample_curve(values):
                return np.stack([
                    np.interp(coordinate, sample_u, values[:, axis])
                    for axis in range(3)
                ], axis=1)

            raw_center_at_v = sample_curve(pose_center)
            corrected_at_v = sample_curve(corrected_center)
            tangent_at_v = sample_curve(pose_tangent)
            tangent_at_v /= np.maximum(
                np.linalg.norm(tangent_at_v, axis=1, keepdims=True), 1e-12)
            offset = raw_local - raw_center_at_v
            axial = np.sum(offset * tangent_at_v, axis=1, keepdims=True)
            transverse = offset - axial * tangent_at_v
            bulge_at_v = np.interp(coordinate, sample_u, bulge)
            global_positions[rigid_offset:rigid_offset + rigid_n] = (
                corrected_at_v + axial * tangent_at_v
                + bulge_at_v[:, None] * transverse)

        # Pes muscles share fascia and converge toward one distal route, but
        # are not welded together. Blend only their centerline displacement
        # fields, retaining the distinct rest offsets and cross-sections.
        pes_names = [
            name for name in ('L_Sartorius', 'L_Gracilis',
                              'L_Semitendinosus')
            if name in active_muscles and name in rigid_coordinates]
        if len(pes_names) >= 2:
            sample_u = np.linspace(0.0, 1.0, 33)
            rest_centers = {}
            pose_centers = {}
            for pes_name in pes_names:
                pes_offset = global_offset[pes_name]
                pes_n = active_muscles[
                    pes_name].soft_body.num_vertices
                pes_u = rigid_coordinates[pes_name]
                ids = np.clip(
                    np.rint(pes_u * (len(sample_u) - 1)).astype(np.int32),
                    0, len(sample_u) - 1)
                rest = global_rest_positions[
                    pes_offset:pes_offset + pes_n]
                pose = global_positions[
                    pes_offset:pes_offset + pes_n]
                rest_curve = np.full((len(sample_u), 3), np.nan)
                pose_curve = np.full((len(sample_u), 3), np.nan)
                for curve_i in range(len(sample_u)):
                    selected = ids == curve_i
                    if np.any(selected):
                        rest_curve[curve_i] = np.mean(
                            rest[selected], axis=0)
                        pose_curve[curve_i] = np.mean(
                            pose[selected], axis=0)
                valid = np.where(np.isfinite(rest_curve[:, 0]))[0]
                for axis in range(3):
                    rest_curve[:, axis] = np.interp(
                        np.arange(len(sample_u)), valid,
                        rest_curve[valid, axis])
                    pose_curve[:, axis] = np.interp(
                        np.arange(len(sample_u)), valid,
                        pose_curve[valid, axis])
                rest_centers[pes_name] = rest_curve
                pose_centers[pes_name] = pose_curve
            support_weight = {name: 1.0 for name in pes_names}
            total_support = sum(support_weight.values())
            shared_displacement = sum(
                support_weight[name]
                * (pose_centers[name] - rest_centers[name])
                for name in pes_names) / total_support
            distal_weight = np.clip(
                (sample_u - 0.05) / 0.95, 0.0, 1.0)
            distal_weight = (
                distal_weight * distal_weight
                * (3.0 - 2.0 * distal_weight))
            for pes_name in pes_names:
                pes_offset = global_offset[pes_name]
                pes_n = active_muscles[
                    pes_name].soft_body.num_vertices
                pes_u = rigid_coordinates[pes_name]
                own_displacement = (
                    pose_centers[pes_name] - rest_centers[pes_name])
                center_correction = (
                    shared_displacement - own_displacement)
                sampled_correction = np.stack([
                    np.interp(pes_u, sample_u,
                              distal_weight * center_correction[:, axis])
                    for axis in range(3)
                ], axis=1)
                global_positions[
                    pes_offset:pes_offset + pes_n] += sampled_correction

        # Adductor magnus is the medial support, but it terminates broadly on
        # the femur rather than at the pes anserinus.  Couple it spatially:
        # nearby pes vertices inherit a fraction of the closest Magnus
        # material displacement, fading to zero beyond 3 cm.  This resists
        # whole-group drift without falsely matching longitudinal parameters.
        magnus_name = 'L_Adductor_Magnus'
        if magnus_name in active_muscles and magnus_name in rigid_coordinates:
            magnus_offset = global_offset[magnus_name]
            magnus_n = active_muscles[
                magnus_name].soft_body.num_vertices
            magnus_rest = global_rest_positions[
                magnus_offset:magnus_offset + magnus_n]
            magnus_pose = global_positions[
                magnus_offset:magnus_offset + magnus_n]
            magnus_tree = scipy.spatial.cKDTree(magnus_rest)
            magnus_delta = magnus_pose - magnus_rest
            for pes_name in pes_names:
                pes_offset = global_offset[pes_name]
                pes_n = active_muscles[pes_name].soft_body.num_vertices
                pes_rest = global_rest_positions[
                    pes_offset:pes_offset + pes_n]
                distance_k, nearest_k = magnus_tree.query(pes_rest, k=8)
                distance = distance_k[:, 0]
                # The medial thigh fascia has a wider zone of influence than
                # the old 3 cm point-contact band. A smooth 8 cm field keeps
                # Sartorius and Semitendinosus engaged with Magnus during hip
                # flexion while still permitting distal sliding.
                support = np.clip(1.0 - distance / 0.08, 0.0, 1.0)
                support = support * support * (3.0 - 2.0 * support)
                # Keep exact bone attachments; couple only the muscle body.
                material_u = rigid_coordinates[pes_name]
                belly = np.sin(np.pi * material_u) ** 2
                support *= belly
                transfer_weight = np.exp(
                    -(distance_k / 0.025) ** 2)
                transfer_weight /= np.maximum(
                    np.sum(transfer_weight, axis=1, keepdims=True), 1e-12)
                supported_delta = np.sum(
                    transfer_weight[:, :, None]
                    * magnus_delta[nearest_k], axis=1)
                correction = supported_delta - (
                    global_positions[pes_offset:pes_offset + pes_n]
                    - pes_rest)
                global_positions[
                    pes_offset:pes_offset + pes_n] += (
                        0.65 * support[:, None] * correction)
        global_positions[fixed_indices] = fixed_targets_array

    # Fiber-spring config (scalar Hookean spring on cross-contour edges,
    # applied per-iter via the collision_target_fn slot).  Reuses the
    # collision_vertices / collision_weight diag pipeline.  Mutually
    # exclusive with bone-contact for now (single closure slot).
    fiber_spring_on = bool(getattr(v, 'fiber_spring', False))
    fiber_spring_w = float(getattr(v, 'fiber_spring_weight', 10.0)) if fiber_spring_on else 0.0
    fiber_spring_scale = float(getattr(v, 'fiber_spring_rest_scale', 0.5))

    tracked_contact_global = {}
    if bool(getattr(v, 'tracked_bone_contact', False)) and not fiber_spring_on:
        for mname, entries in getattr(v, 'tracked_bone_contacts', {}).items():
            if mname not in global_offset:
                continue
            off = global_offset[mname]
            for vi, body_name, q_local, n_local, clearance in entries:
                gi = off + int(vi)
                if gi < total_verts and not global_fixed_mask[gi]:
                    tracked_contact_global[gi] = (
                        body_name, np.asarray(q_local), np.asarray(n_local),
                        float(clearance))
    tracked_contact_on = bool(tracked_contact_global)

    # Bone-contact penalty wiring: include collision_vertices in build_system
    # so the diagonal entry for each collision candidate gains the spring
    # weight.  Force rebuild if the contact weight changed since last call.
    bone_contact_on = bool(getattr(v, 'unified_bone_contact', True)) and v._unified_bone_meshes and not fiber_spring_on
    collision_vertex_set = cache.get('collision_vertex_set') if cache_valid else (
        v._unified_sim_cache.get('collision_vertex_set') if hasattr(v, '_unified_sim_cache') and v._unified_sim_cache else None
    )
    if tracked_contact_on:
        collision_vertex_set = set(tracked_contact_global)
    collision_weight = (
        float(getattr(v, 'tracked_bone_contact_weight', 3.0))
        if tracked_contact_on else
        float(getattr(v, 'unified_bone_collision_weight', 1.5))
        if bone_contact_on else 0.0)

    # For fiber spring: every vert that touches a cross-contour edge becomes
    # a "collision vertex" (gets diag weight) and per-iter target = position
    # that satisfies all its spring rest-lengths.  Spring weight overrides
    # bone-contact weight here (mutually exclusive above).
    if fiber_spring_on:
        _spc = cache if cache is not None else v._unified_sim_cache
        # Use tendon-zone cross edges only (belly stays pure ARAP).
        tmask = _spc.get('csr_tendon_mask')
        ei_arr = _spc.get('csr_edge_i')
        ej_arr = _spc.get('csr_edge_j')
        if tmask is not None and ei_arr is not None:
            sp_edge_mask = np.asarray(tmask, dtype=bool)
            sp_verts = np.unique(np.concatenate([
                ei_arr[sp_edge_mask], ej_arr[sp_edge_mask]
            ]).astype(np.int64))
            non_fixed = ~global_fixed_mask[sp_verts]
            sp_verts = sp_verts[non_fixed]
            collision_vertex_set = set(int(x) for x in sp_verts)
            collision_weight = fiber_spring_w
    weight_changed = (getattr(backend, '_collision_weight_built', None) != collision_weight)

    need_build = (not cache_valid
                  or weight_changed
                  or (getattr(backend, 'solver', None) is None
                      and getattr(backend, '_scipy_solver', None) is None
                      and getattr(backend, '_splu', None) is None))
    skin_prior_weights = None
    skin_binder = getattr(v, 'skin_prior_binder', None)
    if skin_binder is not None and getattr(skin_binder, 'bindings', None):
        skin_prior_weights = {}
        for name in muscle_names:
            offset = global_offset[name]
            for entry in skin_binder.bindings.get(name, []):
                gi = offset + entry['vi_local']
                skin_prior_weights[gi] = skin_prior_weights.get(gi, 0.0) + entry['weight']

    # Pes anserinus bundle prior — fold per-vert weights into skin_prior_weights
    # so build_system bakes them into the L diagonal.
    pes_bundle = getattr(v, 'pes_bundle_obj', None)
    if pes_bundle is not None and getattr(pes_bundle, 'per_muscle', None):
        if skin_prior_weights is None:
            skin_prior_weights = {}
        for mname, data in pes_bundle.per_muscle.items():
            if mname not in global_offset:
                continue
            base = global_offset[mname]
            vi = data['vi']
            w = data['w']
            for k in range(len(vi)):
                gi = base + int(vi[k])
                skin_prior_weights[gi] = skin_prior_weights.get(gi, 0.0) + float(w[k])

    if need_build:
        start_time = time.time()
        build_kwargs = dict(regularization=1e-6)
        if skin_prior_weights:
            build_kwargs['skin_prior_weights'] = skin_prior_weights
        if (bone_contact_on or tracked_contact_on or fiber_spring_on) and collision_vertex_set:
            build_kwargs['collision_vertices'] = collision_vertex_set
            build_kwargs['collision_weight'] = collision_weight
        backend.build_system(
            total_verts, neighbors, edge_weights, global_fixed_mask, **build_kwargs)
        backend._collision_weight_built = collision_weight
        sp_n = len(skin_prior_weights) if skin_prior_weights else 0
        cv_n = len(collision_vertex_set) if ((bone_contact_on or tracked_contact_on or fiber_spring_on) and collision_vertex_set) else 0
        diag_label = 'fiber-spring' if fiber_spring_on else 'bone-contact'
        print(f"  System built [{bin_name} bin] in {time.time() - start_time:.3f}s "
              f"(skin prior on {sp_n} verts, {diag_label} diag on {cv_n} verts @ w={collision_weight})")
    else:
        print(f"  Reusing cached system [{bin_name} bin]")

    # Skin-prior targets: bone-glued attractor (multi-bone DQS for DQS_MUSCLES,
    # single-bone for others).  For knee-crossing muscles, fade target toward
    # current position proportional to knee flex — skin prior keeps them
    # outside the bone at extension but stops fighting the cross-contour
    # contraction at flex.
    skin_prior_targets = None
    wt_sum = {}
    wtgt_sum = {}
    if skin_binder is not None and getattr(skin_binder, 'bindings', None):
        for name in muscle_names:
            offset = global_offset[name]
            res = skin_binder.compute_targets(
                name, offset, current_positions=global_positions,
            )
            if res is None:
                continue
            gi_arr, w_arr, tgt_arr = res
            for k in range(len(gi_arr)):
                gi = int(gi_arr[k])
                w = float(w_arr[k])
                wt_sum[gi] = wt_sum.get(gi, 0.0) + w
                if gi in wtgt_sum:
                    wtgt_sum[gi] = wtgt_sum[gi] + w * tgt_arr[k]
                else:
                    wtgt_sum[gi] = w * tgt_arr[k]
    # Pes anserinus bundle: per-frame target along the anatomical curve.
    if pes_bundle is not None and getattr(pes_bundle, 'per_muscle', None):
        shape_scale = float(getattr(v, 'pes_bundle_offset_scale', 1.0))
        res_b = pes_bundle.compute_targets(v.env.skel, global_offset,
                                            shape_scale=shape_scale)
        if res_b is not None:
            gi_arr, w_arr, tgt_arr = res_b
            for k in range(len(gi_arr)):
                gi = int(gi_arr[k])
                w = float(w_arr[k])
                wt_sum[gi] = wt_sum.get(gi, 0.0) + w
                if gi in wtgt_sum:
                    wtgt_sum[gi] = wtgt_sum[gi] + w * tgt_arr[k]
                else:
                    wtgt_sum[gi] = w * tgt_arr[k]
    if wt_sum:
        skin_prior_targets = {gi: wtgt_sum[gi] / wt_sum[gi] for gi in wt_sum}

    # Axial pose prior — pose-conditioned rest-edge rescaling for fiber-like
    # contraction.  Currently driven by knee flexion for KNEE_CROSSING_MUSCLES
    # (hamstrings + Gracilis + Sartorius).  Cross-contour rest shrinks under
    # flex, intra-contour rest grows to preserve cross-section volume.
    cache = v._unified_sim_cache
    target_edges = None
    scaled_rest, crosser_strs = _apply_axial_pose_prior(v, cache, knee_angles)
    # Tendon-zone slack-only update — runs even when axial prior is a no-op.
    if (scaled_rest is None
            and getattr(v, 'tendon_elastic', True)
            and cache.get('csr_tendon_mask') is not None):
        scaled_rest = cache['csr_rest_edges_base'].copy()
    if scaled_rest is None and fiber_spring_on:
        # Need a scaled_rest to apply intra-contour bulge for volume
        # preservation under spring contraction.
        scaled_rest = cache['csr_rest_edges_base'].copy()
    if scaled_rest is not None:
        if getattr(v, 'tendon_elastic', True):
            scaled_rest = _apply_tendon_elastic(
                cache, global_positions, scaled_rest)
        # Volume-preserving intra-contour expansion to compensate for the
        # fiber spring's axial contraction (target = rest * spring_rest_scale).
        # perp_scale = sqrt(1 / spring_rest_scale), capped at axial_max_bulge.
        if fiber_spring_on and cache.get('csr_intra_mask') is not None:
            # Tendon-zone bulge only.  perp_scale = 1/rest_scale (rest 0.5 →
            # 2.0).  Belly intra-contour edges keep their ARAP rest length —
            # belly stays pure ARAP per user direction.
            perp_scale = float(np.clip(
                1.0 / max(fiber_spring_scale, 0.05),
                1.0, float(getattr(v, 'axial_max_bulge', 2.0))))
            intra_mask = np.asarray(cache['csr_intra_mask'], dtype=bool)
            tendon_verts_mask = np.zeros(len(cache['csr_rest_edges_base']), dtype=bool)
            # Restrict to intra-edges whose endpoints are both in tendon zone
            _ei = cache['csr_edge_i']
            _ej = cache['csr_edge_j']
            tendon_v = cache.get('csr_tendon_mask')
            if tendon_v is not None:
                # csr_tendon_mask is per-edge cross-edges in tendon zone;
                # we want per-vert tendon flag — derive from edges
                tend_vset = np.zeros(total_verts, dtype=bool)
                tmask_arr = np.asarray(tendon_v, dtype=bool)
                tend_vset[_ei[tmask_arr]] = True
                tend_vset[_ej[tmask_arr]] = True
                tendon_intra = intra_mask & tend_vset[_ei] & tend_vset[_ej]
                scaled_rest[tendon_intra] = scaled_rest[tendon_intra] * perp_scale
        if hasattr(backend, 'update_rest_edges'):
            backend.update_rest_edges(scaled_rest)
        else:
            ei = cache['csr_edge_i']
            ej = cache['csr_edge_j']
            target_edges = [{} for _ in range(total_verts)]
            for k in range(len(ei)):
                target_edges[ei[k]][ej[k]] = scaled_rest[k]
        if crosser_strs:
            print(f"  Axial pose prior: {', '.join(crosser_strs)}")

    # One-sided bone-contact penalty closure.  Called per ARAP iter.
    # Outside-margin verts get target = current_pos (zero net force);
    # only inside-bone verts get target = closest_point + face_normal * margin.
    # Throttled — full bone-contact computation runs only every N iters;
    # in-between iters reuse the cached penetration targets and just
    # refresh outside-margin targets to current_pos.
    collision_target_fn = None
    if fiber_spring_on and collision_vertex_set:
        # Per-iter scalar Hookean spring on cross-contour edges.  For each
        # spring edge with current length L and target length L_rest *
        # spring_rest_scale, accumulate corrective displacement on both
        # endpoints; final target = position + accumulated displacement.
        _spc2 = cache if cache is not None else v._unified_sim_cache
        # Tendon cross-edges only, each undirected edge once.
        _ei_full = _spc2['csr_edge_i']
        _ej_full = _spc2['csr_edge_j']
        cross_mask_arr = (np.asarray(_spc2['csr_tendon_mask'], dtype=bool)
                          & (_ei_full < _ej_full))
        sp_ei = _ei_full[cross_mask_arr]
        sp_ej = _ej_full[cross_mask_arr]
        sp_base = _spc2['csr_rest_edges_base'][cross_mask_arr]
        sp_rest_len = np.linalg.norm(sp_base, axis=1) * fiber_spring_scale
        cv_arr = np.fromiter(collision_vertex_set, dtype=np.int64)

        cv_int = cv_arr.astype(np.int64)
        recompute_every = int(getattr(v, 'fiber_spring_recompute_every', 5))
        state = {'iter': 0, 'targets': None}

        def collision_target_fn(positions, _ei=sp_ei, _ej=sp_ej,
                                _rest=sp_rest_len, _cv=cv_int,
                                _every=recompute_every, _state=state):
            # Refresh corrective displacement every N iters; in-between iters
            # reuse the cached targets relative to the same neighbor structure
            # (effectively a damped Jacobi spring solve at coarser cadence).
            if _state['iter'] % _every == 0:
                p_i = positions[_ei]
                p_j = positions[_ej]
                d = p_j - p_i
                cur = np.linalg.norm(d, axis=1)
                inv_cur = np.zeros_like(cur)
                ok = cur > 1e-9
                inv_cur[ok] = 1.0 / cur[ok]
                half = np.clip(0.5 * (cur - _rest), -0.5 * cur, 0.5 * cur)
                delta = (half * inv_cur)[:, None] * d
                disp = np.zeros((len(positions), 3))
                count = np.zeros(len(positions), dtype=np.float64)
                np.add.at(disp, _ei, +delta)
                np.add.at(disp, _ej, -delta)
                np.add.at(count, _ei, 1.0)
                np.add.at(count, _ej, 1.0)
                nz = count > 0
                disp[nz] /= count[nz, None]
                disp_mag = np.linalg.norm(disp, axis=1)
                cap = 0.01
                scale = np.where(disp_mag > cap, cap / np.maximum(disp_mag, 1e-12), 1.0)
                disp *= scale[:, None]
                target_pos = positions[_cv] + disp[_cv]
                _state['targets'] = dict(zip(_cv.tolist(), target_pos))
            _state['iter'] += 1
            return _state['targets']
    elif tracked_contact_on:
        cv_arr = np.fromiter(tracked_contact_global, dtype=np.int64)
        state = {'iter': 0}
        tracked_max_step = max(
            0.0, float(getattr(
                v, 'tracked_bone_contact_max_step', np.inf)))

        def collision_target_fn(positions, _contacts=tracked_contact_global,
                                _cv=cv_arr, _state=state,
                                _max_step=tracked_max_step):
            targets = {int(vi): positions[int(vi)].copy() for vi in _cv}
            transforms = {}
            for vi, (body_name, q_local, n_local, clearance) in _contacts.items():
                if body_name not in transforms:
                    body = v.env.skel.getBodyNode(body_name)
                    if body is None:
                        continue
                    T = np.asarray(body.getWorldTransform().matrix())
                    transforms[body_name] = (T[:3, :3], T[:3, 3])
                R, t = transforms[body_name]
                q = R @ q_local + t
                n = R @ n_local
                n /= max(np.linalg.norm(n), 1e-12)
                d = float(np.dot(positions[vi] - q, n))
                if d < clearance:
                    correction = clearance - d
                    if np.isfinite(_max_step) and _max_step > 0.0:
                        correction = min(correction, _max_step)
                    targets[int(vi)] = (
                        positions[vi] + correction * n)
            _state['iter'] += 1
            return targets
    elif bone_contact_on and collision_vertex_set and v._unified_bone_meshes:
        from scipy.spatial import cKDTree as _cKDT_coll
        bone_meshes_for_fn = list(v._unified_bone_meshes)
        bone_kdtree = _cKDT_coll(np.vstack([bm.vertices for bm in bone_meshes_for_fn]))
        cv_arr = np.fromiter(collision_vertex_set, dtype=np.int64)
        margin = float(getattr(v, 'unified_bone_contact_margin', 0.005))
        recompute_every = int(getattr(v, 'unified_bone_contact_recompute_every', 1))
        state = {'iter': 0, 'pen_targets': {}}

        def collision_target_fn(positions, _bones=bone_meshes_for_fn,
                                _kdt=bone_kdtree, _cv=cv_arr,
                                _fmask=global_fixed_mask, _margin=margin,
                                _every=recompute_every, _state=state):
            targets = {int(vi): positions[int(vi)].copy() for vi in _cv}
            # Recompute penetration targets every N iters to amortize the
            # expensive contains() + closest_point queries.
            if _state['iter'] % _every == 0:
                new_pen = {}
                sv_pos = positions[_cv]
                d_kd, _ = _kdt.query(sv_pos)
                near = d_kd < (_margin + 0.025)
                if np.any(near):
                    near_sv = _cv[near]
                    near_pos = positions[near_sv]
                    for bm in _bones:
                        try:
                            bmin = bm.bounds[0] - _margin
                            bmax = bm.bounds[1] + _margin
                            in_bbox = np.all((near_pos >= bmin) & (near_pos <= bmax), axis=1)
                            if not np.any(in_bbox):
                                continue
                            bbox_pos = near_pos[in_bbox]
                            bbox_sv = near_sv[in_bbox]
                            inside = bm.contains(bbox_pos)
                            if not np.any(inside):
                                continue
                            ipos = bbox_pos[inside]
                            isv = bbox_sv[inside]
                            cp, _, fid = trimesh.proximity.closest_point(bm, ipos)
                            for k in range(len(isv)):
                                vi = int(isv[k])
                                if _fmask[vi]:
                                    continue
                                # Do not trust OBJ winding for the outward
                                # normal.  Use the closest-point direction and
                                # orient it away from the bone interior.
                                away = cp[k] - ipos[k]
                                norm = np.linalg.norm(away)
                                if norm > 1e-12:
                                    away /= norm
                                else:
                                    away = -np.asarray(bm.face_normals[fid[k]])
                                new_pen[vi] = cp[k] + away * _margin
                        except Exception:
                            continue
                _state['pen_targets'] = new_pen
            # Overlay cached penetration targets on the default current-pos targets
            for vi, tgt in _state['pen_targets'].items():
                targets[vi] = tgt
            _state['iter'] += 1
            return targets

    # Paper §4.1.2 dynamic anisotropic contact (muscle-muscle, muscle-bone)
    # + barycentric fascia constraints (always-on attractive coupling).
    # Activated by ctx.use_anisotropic_contact = True.
    if (getattr(v, 'use_anisotropic_contact', False)
            and collision_vertex_set
            and (cache or v._unified_sim_cache)):
        from scipy.spatial import cKDTree as _cKDT_aniso
        _spc = cache if cache_valid else v._unified_sim_cache
        anat_F = _spc.get('anat_face_global')
        vert_owner = _spc.get('vert_owner_global')
        if anat_F is None or vert_owner is None or len(anat_F) == 0:
            pass  # fall through; flag is no-op
        else:
            cv_arr = np.fromiter(collision_vertex_set, dtype=np.int64)
            mm_margin = float(getattr(v, 'inter_muscle_contact_margin', 0.002))
            bone_margin = float(getattr(v, 'unified_bone_contact_margin', 0.005))
            recompute_every = int(getattr(v, 'unified_bone_contact_recompute_every', 1))
            bone_meshes_for_fn = list(v._unified_bone_meshes) if v._unified_bone_meshes else []
            bone_kdtree = (_cKDT_aniso(np.vstack([bm.vertices for bm in bone_meshes_for_fn]))
                           if bone_meshes_for_fn else None)
            state = {'iter': 0, 'pen_targets': {}}
            # Unique anatomical surface verts (global) and their owner muscle
            anat_v_unique = np.unique(anat_F)
            owner_of_anat = vert_owner[anat_v_unique]
            # Barycentric fascia constraints (paper §4.1.2)
            fc_vi = _spc.get('fc_vi')
            fc_tri = _spc.get('fc_tri')
            fc_bary = _spc.get('fc_bary')
            fc_gap = _spc.get('fc_gap')

            def collision_target_fn(positions, _anatF=anat_F, _anatV=anat_v_unique,
                                    _owner=owner_of_anat, _vert_owner=vert_owner,
                                    _bones=bone_meshes_for_fn, _bone_kdt=bone_kdtree,
                                    _cv=cv_arr, _fmask=global_fixed_mask,
                                    _mmm=mm_margin, _bm=bone_margin,
                                    _every=recompute_every, _state=state,
                                    _fc_vi=fc_vi, _fc_tri=fc_tri, _fc_bary=fc_bary,
                                    _fc_gap=fc_gap):
                targets = {int(vi): positions[int(vi)].copy() for vi in _cv}
                # Always-on sliding fascia target.  Correct only signed normal
                # separation; never pull toward the stored tangential point.
                if (_fc_vi is not None and len(_fc_vi) > 0
                        and _fc_gap is not None and len(_fc_gap) == len(_fc_vi)):
                    tri_pos = positions[_fc_tri]
                    fc_pos = (_fc_bary[:, :, None] * tri_pos).sum(axis=1)
                    fc_n = np.cross(tri_pos[:, 1] - tri_pos[:, 0],
                                    tri_pos[:, 2] - tri_pos[:, 0])
                    fc_n /= np.linalg.norm(fc_n, axis=1, keepdims=True) + 1e-12
                    current_gap = np.einsum('ij,ij->i',
                                            positions[_fc_vi] - fc_pos, fc_n)
                    normal_correction = (_fc_gap - current_gap)[:, None] * fc_n
                    for k in range(len(_fc_vi)):
                        vi = int(_fc_vi[k])
                        if _fmask[vi]:
                            continue
                        targets[vi] = positions[vi] + normal_correction[k]
                if _state['iter'] % _every == 0:
                    new_pen = {}
                    # Per-vertex outward normals from current positions of
                    # anatomical faces (area-weighted).
                    tri = positions[_anatF]
                    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                    area = 0.5 * np.linalg.norm(cross, axis=1)
                    face_n = cross / (2.0 * area[:, None] + 1e-12)
                    Vn = np.zeros((len(positions), 3))
                    for kk in range(3):
                        np.add.at(Vn, _anatF[:, kk], face_n * area[:, None])
                    nn = np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-12
                    Vn = Vn / nn
                    # KDTree over current anatomical surface vert positions.
                    surf_pos = positions[_anatV]
                    surf_normals = Vn[_anatV]
                    tree = _cKDT_aniso(surf_pos)
                    # For every collision-candidate vert, query 4 nearest
                    # surface verts; pick first that belongs to a DIFFERENT
                    # muscle and is penetrating.
                    _, nn_idx = tree.query(positions[_cv], k=4)
                    cv_owner = _vert_owner[_cv]
                    for i, vi_global in enumerate(_cv):
                        vi_int = int(vi_global)
                        if _fmask[vi_int]:
                            continue
                        for kk in range(4):
                            j = int(nn_idx[i, kk])
                            if _owner[j] == cv_owner[i] or _owner[j] < 0:
                                continue
                            other_pos = surf_pos[j]
                            other_n = surf_normals[j]
                            signed = float(np.dot(positions[vi_int] - other_pos, other_n))
                            if signed < 0.0:
                                # Penetrating other muscle. Target = surface +
                                # margin along its outward normal.
                                new_pen[vi_int] = other_pos + other_n * _mmm
                                break
                    # Bone penetration push — same as bone-only branch.
                    if _bones and _bone_kdt is not None:
                        sv_pos = positions[_cv]
                        d_kd, _ = _bone_kdt.query(sv_pos)
                        near = d_kd < (_bm + 0.025)
                        if np.any(near):
                            near_sv = _cv[near]
                            near_pos = positions[near_sv]
                            for bm in _bones:
                                try:
                                    bmin = bm.bounds[0] - _bm
                                    bmax = bm.bounds[1] + _bm
                                    in_bbox = np.all((near_pos >= bmin) & (near_pos <= bmax), axis=1)
                                    if not np.any(in_bbox):
                                        continue
                                    bbox_pos = near_pos[in_bbox]
                                    bbox_sv = near_sv[in_bbox]
                                    inside = bm.contains(bbox_pos)
                                    if not np.any(inside):
                                        continue
                                    ipos = bbox_pos[inside]
                                    isv = bbox_sv[inside]
                                    cp, _, fid = trimesh.proximity.closest_point(bm, ipos)
                                    for k in range(len(isv)):
                                        vi = int(isv[k])
                                        if _fmask[vi]:
                                            continue
                                        away = cp[k] - ipos[k]
                                        norm = np.linalg.norm(away)
                                        if norm > 1e-12:
                                            away /= norm
                                        else:
                                            away = -np.asarray(bm.face_normals[fid[k]])
                                        new_pen[vi] = cp[k] + away * _bm
                                except Exception:
                                    continue
                    _state['pen_targets'] = new_pen
                for vi, tgt in _state['pen_targets'].items():
                    targets[vi] = tgt
                _state['iter'] += 1
                return targets

    is_first_frame = prev_solution is None
    solve_iters = max_iterations * 4 if is_first_frame else max_iterations
    if is_first_frame:
        print(f"  First frame: {solve_iters} iterations (4x cap) for convergence")

    start_time = time.time()
    solve_kwargs = dict(
        max_iterations=solve_iters, tolerance=tolerance,
        target_edges=target_edges, verbose=True,
    )
    if skin_prior_targets:
        solve_kwargs['skin_prior_targets'] = skin_prior_targets
    if collision_target_fn is not None:
        solve_kwargs['collision_target_fn'] = collision_target_fn
    if getattr(v, 'disable_plateau_exit', False):
        solve_kwargs['disable_plateau_exit'] = True
    volume_arap_alternations = max(
        1, int(getattr(v, 'volume_arap_alternations', 1)))
    if getattr(v, 'rigid_blend_only', False):
        iterations = 0
        max_disp = 0.0
        print("  SO(3) rigid-blend muscle warp (ARAP bypassed)")
    elif (volume_arap_alternations > 1 and global_tets is not None
            and len(global_tets)):
        total_iterations = 0
        cycle_iters = max(5, solve_iters // volume_arap_alternations)
        for cycle in range(volume_arap_alternations):
            cycle_kwargs = dict(solve_kwargs)
            cycle_kwargs['max_iterations'] = cycle_iters
            global_positions, iterations, max_disp = backend.solve(
                global_positions, global_rest_positions, neighbors,
                edge_weights, rest_edge_vectors, global_fixed_mask,
                fixed_targets_array, **cycle_kwargs)
            total_iterations += iterations
            if cycle + 1 < volume_arap_alternations:
                before_volume = global_positions.copy()
                global_positions, _, _ = _project_positive_tet_volumes(
                    global_positions, global_rest_positions, global_tets,
                    global_fixed_mask, sweeps=4, stiffness=0.75,
                    max_step=0.0005)
                # A volume pass is a local correction, not a second solver:
                # prevent accumulated projection blow-outs within one cycle.
                correction = global_positions - before_volume
                correction_len = np.linalg.norm(correction, axis=1)
                excessive = correction_len > 0.0015
                if np.any(excessive):
                    correction[excessive] *= (
                        0.0015 / correction_len[excessive])[:, None]
                    global_positions[excessive] = (
                        before_volume[excessive] + correction[excessive])
                global_positions[global_fixed_mask] = fixed_targets_array
        iterations = total_iterations
        print(f"  Alternated ARAP/volume: {volume_arap_alternations} cycles")
    else:
        global_positions, iterations, max_disp = backend.solve(
            global_positions, global_rest_positions, neighbors, edge_weights,
            rest_edge_vectors, global_fixed_mask, fixed_targets_array,
            **solve_kwargs)
    # Restore base rest so next frame starts clean
    if v.use_muscle_aware_arap and cache.get('csr_cross_mask') is not None:
        if hasattr(backend, 'update_rest_edges'):
            backend.update_rest_edges(cache['csr_rest_edges_base'])
    print(f"  ARAP solved in {time.time() - start_time:.3f}s ({iterations} iterations)")

    # ARAP/contact targets can satisfy surface constraints by crushing or
    # inverting tets. Alternate rest-volume recovery with contact correction;
    # hard attachment targets are restored after every pass.
    arap_positions = global_positions.copy()
    volume_sweeps = (0 if getattr(v, 'rigid_blend_only', False) else
                     int(getattr(v, 'volume_projection_sweeps', 24)))
    if global_tets is not None and len(global_tets) and volume_sweeps > 0:
        fixed_idx = np.where(global_fixed_mask)[0]
        max_volume_correction = float(
            getattr(v, 'volume_projection_max_correction', 0.0))

        def _bound_volume_correction():
            """Keep a local volume repair from ejecting a vertex from ARAP."""
            if max_volume_correction <= 0.0:
                return
            correction = global_positions - arap_positions
            correction_len = np.linalg.norm(correction, axis=1)
            excessive = (~global_fixed_mask) & (
                correction_len > max_volume_correction)
            if np.any(excessive):
                scale = max_volume_correction / correction_len[excessive]
                global_positions[excessive] = (
                    arap_positions[excessive]
                    + correction[excessive] * scale[:, None])

        contact_passes = int(getattr(v, 'volume_contact_passes', 4))
        for _ in range(max(contact_passes, 1)):
            global_positions, _, _ = _project_positive_tet_volumes(
                global_positions, global_rest_positions, global_tets,
                global_fixed_mask, sweeps=max(1, volume_sweeps // max(contact_passes, 1)),
                stiffness=0.85, max_step=0.001)
            _bound_volume_correction()
            global_positions[fixed_idx] = fixed_targets_array
            if collision_target_fn is not None:
                targets = collision_target_fn(global_positions)
                for vi, target in targets.items():
                    vi = int(vi)
                    if global_fixed_mask[vi]:
                        continue
                    delta = np.asarray(target) - global_positions[vi]
                    dn = np.linalg.norm(delta)
                    if dn > 0.002:
                        delta *= 0.002 / dn
                    global_positions[vi] += 0.5 * delta
        global_positions, n_inv, max_v_err = _project_positive_tet_volumes(
            global_positions, global_rest_positions, global_tets,
            global_fixed_mask, sweeps=volume_sweeps, stiffness=0.9,
            max_step=0.001)
        _bound_volume_correction()
        global_positions[fixed_idx] = fixed_targets_array
        print(f"  Volume projection: inverted={n_inv}, max |V/V0-1|={max_v_err:.3f}")

    # Build a vertex → muscle ID array so isolated/stuck fixes only borrow
    # displacement from same-muscle connected verts. Cross-muscle copies
    # produce visibly wrong attachments when adjacent muscles ride
    # different bones (e.g. soleus interior vs gastrocnemius).
    vert_muscle = np.empty(total_verts, dtype=np.int32)
    for mid, name in enumerate(muscle_names):
        off = global_offset[name]
        n = active_muscles[name].soft_body.num_vertices
        vert_muscle[off:off + n] = mid

    # Fix isolated vertices (0 neighbors) by copying displacement from the
    # nearest connected vertex IN THE SAME MUSCLE. Was global cKDTree,
    # which could copy displacement across muscle boundaries.
    nbr_lens = np.array([len(neighbors[i]) for i in range(total_verts)],
                        dtype=np.int32)
    iso_mask = (nbr_lens == 0) & (~global_fixed_mask)
    iso_indices = np.where(iso_mask)[0]
    if len(iso_indices) > 0:
        from scipy.spatial import cKDTree as _cKDT_iso
        connected_mask = nbr_lens > 0
        unmatched = []
        for mid in np.unique(vert_muscle[iso_indices]):
            iso_m = iso_indices[vert_muscle[iso_indices] == mid]
            conn_m = np.where(connected_mask & (vert_muscle == mid))[0]
            if len(conn_m) == 0:
                unmatched.extend(iso_m.tolist())
                continue
            tree = _cKDT_iso(global_rest_positions[conn_m])
            _, nearest_local = tree.query(global_rest_positions[iso_m])
            best_j = conn_m[nearest_local]
            disp = global_positions[best_j] - global_rest_positions[best_j]
            global_positions[iso_m] = global_rest_positions[iso_m] + disp
        if unmatched and not getattr(v, '_iso_unmatched_announced', False):
            print(f"  WARNING: {len(unmatched)} isolated verts have no connected same-muscle vert; left at rest")
            v._iso_unmatched_announced = True
        if not getattr(v, '_iso_fix_announced', False):
            print(f"  Fixed {len(iso_indices) - len(unmatched)} isolated vertices via same-muscle nearest connected (printed once)")
            v._iso_fix_announced = True

    # Stuck-vertex fix: free verts with neighbors but ~zero displacement.
    # Vectorize the prior Python per-vert loop and restrict to same-muscle
    # neighbors via average over neighbors[] (those are already global
    # indices; muscle membership comes from the same vert_muscle map).
    fixed_indices = np.where(global_fixed_mask)[0]
    fixed_disp = np.linalg.norm(global_positions[fixed_indices] - global_rest_positions[fixed_indices], axis=1)
    max_fixed_disp = np.max(fixed_disp) if len(fixed_disp) > 0 else 0.0
    if max_fixed_disp > 1e-6:
        total_disp_from_rest = np.linalg.norm(global_positions - global_rest_positions, axis=1)
        stuck_threshold = 1e-6
        stuck_mask = (~global_fixed_mask) & (total_disp_from_rest < stuck_threshold) & (nbr_lens > 0)
        stuck_indices = np.where(stuck_mask)[0]
        if len(stuck_indices) > 0:
            new_positions = global_positions[stuck_indices].copy()
            for k, i in enumerate(stuck_indices):
                same_mid = vert_muscle[i]
                same_neighbors = [j for j in neighbors[i] if vert_muscle[j] == same_mid]
                if not same_neighbors:
                    continue
                neighbor_avg = global_positions[np.asarray(same_neighbors, dtype=np.int64)].mean(axis=0)
                new_positions[k] = 0.3 * global_positions[i] + 0.7 * neighbor_avg
            global_positions[stuck_indices] = new_positions
            if not getattr(v, '_stuck_fix_announced', False):
                print(f"  Fixed {len(stuck_indices)} stuck vertices via same-muscle neighbor average (printed once)")
                v._stuck_fix_announced = True

    # Stash solution for warm-starting the next frame
    if v._unified_sim_cache is not None:
        v._unified_sim_cache['prev_solution'] = global_positions.copy()

    # Distribute results back to individual muscles
    total_change = 0.0
    for name, mobj in active_muscles.items():
        offset = global_offset[name]
        n = mobj.soft_body.num_vertices
        old_pos = mobj.tet_vertices.copy() if mobj.tet_vertices is not None else mobj.soft_body.rest_positions
        mobj.soft_body.positions = global_positions[offset:offset+n].copy()
        mobj.tet_vertices = mobj.soft_body.get_positions().astype(np.float32)
        if not getattr(mobj, '_baking_mode', False):
            mobj._prepare_tet_draw_arrays()

        # Update waypoints/fibers from deformed tetrahedra
        if not getattr(mobj, '_baking_mode', False) and getattr(mobj, 'waypoints_from_tet_sim', True):
            if hasattr(mobj, 'waypoints') and len(mobj.waypoints) > 0:
                if hasattr(mobj, '_update_waypoints_from_tet'):
                    mobj._update_waypoints_from_tet(v.env.skel)

        # Calculate change
        change = np.linalg.norm(mobj.tet_vertices - old_pos, axis=1).max()
        total_change = max(total_change, change)
        print(f"    {name}: max vertex change = {change:.4f}m")

    print(f"Unified volume sim complete (max change: {total_change:.4f}m)")


def _enforce_inter_muscle_constraints(v, active_muscles, stiffness=0.9):
    """
    Enforce inter-muscle distance constraints by adjusting vertex positions.
    Respects fixed vertices - only moves free vertices.

    Returns: average constraint error.

    Vectorized: groups constraints by (name1, name2) pair and applies all
    distance corrections at once. The previous Python loop over every
    constraint became O(seconds) per call once total constraints exceeded
    ~100k (e.g. dense original-mesh tets at 0.015 m threshold).
    """
    if not v.inter_muscle_constraints:
        return 0.0

    # Group constraints by (name1, name2) pair to amortize muscle-positions
    # lookup; each pair becomes a vectorized batch.
    if not hasattr(v, '_inter_muscle_grouped_cache') or \
            v._inter_muscle_grouped_cache_id != id(v.inter_muscle_constraints):
        groups = {}
        for c in v.inter_muscle_constraints:
            key = (c[0], c[3])
            groups.setdefault(key, []).append(c)
        compiled = {}
        for (n1, n2), lst in groups.items():
            v1_idx = np.array([c[1] for c in lst], dtype=np.int64)
            v1_fixed = np.array([c[2] for c in lst], dtype=bool)
            v2_idx = np.array([c[4] for c in lst], dtype=np.int64)
            v2_fixed = np.array([c[5] for c in lst], dtype=bool)
            rest = np.array([c[6] for c in lst], dtype=np.float64)
            both_fixed = v1_fixed & v2_fixed
            keep = ~both_fixed
            compiled[(n1, n2)] = (
                v1_idx[keep], v1_fixed[keep],
                v2_idx[keep], v2_fixed[keep],
                rest[keep],
            )
        v._inter_muscle_grouped_cache = compiled
        v._inter_muscle_grouped_cache_id = id(v.inter_muscle_constraints)

    total_error = 0.0
    total_count = 0
    skip_draw = any(getattr(m, "_baking_mode", False) for m in active_muscles.values())

    # Per-muscle accumulators so verts touched by N constraints get the AVERAGE
    # correction, not the sum (sum diverges on dense meshes with ~1.2M constraints).
    corr_acc = {name: np.zeros_like(m.soft_body.positions) for name, m in active_muscles.items()}
    count_acc = {name: np.zeros(len(m.soft_body.positions)) for name, m in active_muscles.items()}

    for (n1, n2), (i1, f1, i2, f2, rest) in v._inter_muscle_grouped_cache.items():
        if n1 not in active_muscles or n2 not in active_muscles:
            continue
        if i1.size == 0:
            continue
        m1 = active_muscles[n1]
        m2 = active_muscles[n2]
        p1 = m1.soft_body.positions[i1]
        p2 = m2.soft_body.positions[i2]
        diff = p2 - p1
        curr = np.linalg.norm(diff, axis=1)
        valid = curr > 1e-8
        if not valid.any():
            continue
        err = curr - rest
        total_error += float(np.sum(np.abs(err[valid])))
        total_count += int(valid.sum())
        # Correction weights per side: fixed verts don't move.
        w1 = np.where(f1, 0.0, np.where(f2, 1.0, 0.5))
        w2 = np.where(f2, 0.0, np.where(f1, 1.0, 0.5))
        unit = np.zeros_like(diff)
        unit[valid] = diff[valid] / curr[valid, None]
        corr = err[:, None] * stiffness * unit
        # Mask invalid rows out (no contribution)
        valid_mask = valid.astype(np.float64)[:, None]
        np.add.at(corr_acc[n1], i1, corr * w1[:, None] * valid_mask)
        np.add.at(count_acc[n1], i1, (w1 > 0) & valid)
        np.add.at(corr_acc[n2], i2, -corr * w2[:, None] * valid_mask)
        np.add.at(count_acc[n2], i2, (w2 > 0) & valid)

    # Apply averaged correction per muscle
    for name, mobj in active_muscles.items():
        cnt = count_acc[name]
        mask = cnt > 0
        if mask.any():
            mobj.soft_body.positions[mask] += corr_acc[name][mask] / cnt[mask, None]

    # Sync tet_vertices; skip draw-array rebuild when any muscle is baking.
    for name, mobj in active_muscles.items():
        mobj.tet_vertices = mobj.soft_body.get_positions().astype(np.float32)
        if not skip_draw:
            mobj._prepare_tet_draw_arrays()

    return total_error / max(1, total_count)


def draw_inter_muscle_constraint_lines(v):
    """Draw lines between inter-muscle constraint vertex pairs with strain visualization."""
    if not hasattr(v, 'inter_muscle_constraints') or len(v.inter_muscle_constraints) == 0:
        return

    glPushMatrix()
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(2.0)

    # Build vertex and color arrays
    verts = []
    colors = []
    for constraint in v.inter_muscle_constraints:
        name1, v1_idx, v1_fixed, name2, v2_idx, v2_fixed, rest_dist = constraint

        if name1 not in v.zygote_muscle_meshes or name2 not in v.zygote_muscle_meshes:
            continue

        mobj1 = v.zygote_muscle_meshes[name1]
        mobj2 = v.zygote_muscle_meshes[name2]

        if mobj1.tet_vertices is None or mobj2.tet_vertices is None:
            continue
        if v1_idx >= len(mobj1.tet_vertices) or v2_idx >= len(mobj2.tet_vertices):
            continue

        v1 = mobj1.tet_vertices[v1_idx]
        v2 = mobj2.tet_vertices[v2_idx]
        current_dist = np.linalg.norm(v2 - v1)

        # Calculate strain: positive = stretched, negative = compressed
        strain = (current_dist - rest_dist) / rest_dist if rest_dist > 0 else 0

        # Color based on strain with transparency
        if strain > 0.01:  # Stretched
            t = min(strain * 5, 1.0)
            c = [1.0, 1.0 - t * 0.7, 0.3, 0.5]  # Yellow -> Red
        elif strain < -0.01:  # Compressed
            t = min(-strain * 5, 1.0)
            c = [0.3, 1.0, 0.3 + t * 0.7, 0.5]  # Green -> Cyan
        else:  # Near rest length
            c = [0.9, 0.9, 0.9, 0.4]  # White/gray

        verts.append(v1)
        verts.append(v2)
        colors.append(c)
        colors.append(c)

    if len(verts) > 0:
        verts = np.array(verts, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, verts)
        glColorPointer(4, GL_FLOAT, 0, colors)
        glDrawArrays(GL_LINES, 0, len(verts))
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    glDisable(GL_BLEND)
    glEnable(GL_LIGHTING)
    glPopMatrix()


def drawTestMuscles(v, color=None):
    if color is not None:
        glColor4d(color[0], color[1], color[2], color[3])
    glLineWidth(v.line_width)
    glEnableClientState(GL_VERTEX_ARRAY)
    for m_wps in v.env.test_muscle_pos:
        verts = np.array(m_wps, dtype=np.float32)
        glVertexPointer(3, GL_FLOAT, 0, verts)
        glDrawArrays(GL_LINE_STRIP, 0, len(verts))
    glDisableClientState(GL_VERTEX_ARRAY)


def drawMuscles(v, color=None):
    if color is not None:
        glColor4d(color[0], color[1], color[2], color[3])

    if v.draw_line_muscle:
        glDisable(GL_LIGHTING)
        glLineWidth(v.line_width)
        glEnableClientState(GL_VERTEX_ARRAY)
        for idx, m_wps in enumerate(v.env.muscle_pos):
            # Bounds check for activation levels
            if idx < len(v.env.muscle_activation_levels):
                a = v.env.muscle_activation_levels[idx]
            else:
                a = 0.0  # Default activation if index out of bounds
            if color is None:
                glColor4d(1.0 * a,  0.2 * a, 0.2 * a, 0.2 + 0.6 * a)
            verts = np.array(m_wps, dtype=np.float32)
            glVertexPointer(3, GL_FLOAT, 0, verts)
            glDrawArrays(GL_LINE_STRIP, 0, len(verts))
        glDisableClientState(GL_VERTEX_ARRAY)
        glEnable(GL_LIGHTING)
    else:
        for idx, m_wps in enumerate(v.env.muscle_pos):
            # Bounds check for activation levels
            if idx < len(v.env.muscle_activation_levels):
                a = v.env.muscle_activation_levels[idx]
            else:
                a = 0.0  # Default activation if index out of bounds
            if color is None:
                glColor4d(1.0 * a,  0.2 * a, 0.2 * a, 0.2 + 0.6 * a)
            for i_wp in range(len(m_wps) - 1):
                t_parent = m_wps[i_wp]
                t_child = m_wps[i_wp + 1]

                glPushMatrix()
                m = (t_parent + t_child) / 2
                p2c = t_child - t_parent
                length = np.linalg.norm(p2c)
                p2c = p2c / length
                z = np.array([0, 0, 1])

                axis = np.cross(z, p2c)
                s = np.linalg.norm(axis)
                axis /= s
                c = np.dot(z, p2c)
                angle = np.rad2deg(np.arctan2(s, c))
                
                glTranslatef(m[0], m[1], m[2])
                glRotatef(angle, axis[0], axis[1], axis[2])
                mygl.draw_cube([0.01, 0.01, length])
                glPopMatrix()


def _build_combined_collision_mesh(v):
    """Build a combined collision mesh from all skeleton bones for batch operations."""
    import trimesh

    if not hasattr(v, 'zygote_skeleton_meshes') or v.zygote_skeleton_meshes is None:
        return None
    if v.env.skel is None:
        return None

    all_vertices = []
    all_faces = []
    vertex_offset = 0
    skeleton = v.env.skel

    for mesh_name, mesh_loader in v.zygote_skeleton_meshes.items():
        # Find the body node for this mesh
        body_node = skeleton.getBodyNode(mesh_name)
        if body_node is None:
            for i in range(skeleton.getNumBodyNodes()):
                bn = skeleton.getBodyNode(i)
                bn_name = bn.getName()
                if mesh_name.lower() in bn_name.lower() or bn_name.lower() in mesh_name.lower():
                    body_node = bn
                    break

        if body_node is None:
            continue

        # Get or create trimesh
        if not hasattr(mesh_loader, 'trimesh') or mesh_loader.trimesh is None:
            if hasattr(mesh_loader, 'vertices') and hasattr(mesh_loader, 'faces'):
                if mesh_loader.vertices is not None and mesh_loader.faces is not None and len(mesh_loader.vertices) > 0:
                    try:
                        mesh_loader.trimesh = trimesh.Trimesh(
                            vertices=np.array(mesh_loader.vertices),
                            faces=np.array(mesh_loader.faces),
                            process=False
                        )
                    except:
                        continue

        if mesh_loader.trimesh is None:
            continue

        try:
            original_verts = np.array(mesh_loader.trimesh.vertices)
            faces = np.array(mesh_loader.trimesh.faces)

            # Get current world transform
            world_transform = body_node.getWorldTransform()
            R_curr = world_transform.rotation()
            T_curr = world_transform.translation()

            # Get rest transforms if available (stored during soft body init)
            body_name = body_node.getName()
            # Check any muscle for skeleton_rest_transforms
            rest_transform = None
            for mobj in v.zygote_muscle_meshes.values():
                if hasattr(mobj, 'skeleton_rest_transforms') and body_name in mobj.skeleton_rest_transforms:
                    rest_transform = mobj.skeleton_rest_transforms[body_name]
                    break

            if rest_transform is not None:
                R_rest, T_rest = rest_transform
                delta_rotation = R_curr @ R_rest.T
                transformed_verts = (delta_rotation @ (original_verts - T_rest).T).T + T_curr
            else:
                # No rest transform, assume OBJ is already at rest
                transformed_verts = original_verts

            all_vertices.append(transformed_verts)
            all_faces.append(faces + vertex_offset)
            vertex_offset += len(transformed_verts)

        except Exception as e:
            continue

    if len(all_vertices) == 0:
        return None

    combined_vertices = np.vstack(all_vertices)
    combined_faces = np.vstack(all_faces)

    return trimesh.Trimesh(
        vertices=combined_vertices,
        faces=combined_faces,
        process=False
    )


def _apply_batched_collision(v, collision_mesh, margin=2.0, num_passes=2):
    """Apply collision to all muscles at once with a single batched query."""
    if collision_mesh is None:
        return

    # Collect all proximal vertices from all muscles
    position_arrays = []
    muscle_info = []  # (muscle_obj, proximal_indices, start_idx, end_idx)
    total_count = 0

    for mname, mobj in v.zygote_muscle_meshes.items():
        if mobj.soft_body is None:
            continue
        if not hasattr(mobj, 'bone_proximal_vertices') or len(mobj.bone_proximal_vertices) == 0:
            continue

        proximal_indices = np.array(list(mobj.bone_proximal_vertices.keys()))
        positions = mobj.soft_body.positions[proximal_indices].copy()

        start_idx = total_count
        total_count += len(positions)
        end_idx = total_count

        position_arrays.append(positions)
        muscle_info.append((mobj, proximal_indices, start_idx, end_idx))

    if len(position_arrays) == 0:
        print("No proximal vertices to check for collision")
        return

    all_positions = np.vstack(position_arrays)
    print(f"Batched collision: {len(all_positions)} vertices from {len(muscle_info)} muscles")

    # Do collision passes
    for pass_idx in range(num_passes):
        # Single batched query for all vertices
        closest_points, distances, face_ids = collision_mesh.nearest.on_surface(all_positions)

        need_push = distances < margin
        if not np.any(need_push):
            break

        push_indices = np.where(need_push)[0]
        face_normals = collision_mesh.face_normals[face_ids[push_indices]]
        all_positions[push_indices] = closest_points[push_indices] + face_normals * (margin + 1.0)

    # Distribute results back to muscles
    for mobj, proximal_indices, start_idx, end_idx in muscle_info:
        new_positions = all_positions[start_idx:end_idx]
        mobj.soft_body.positions[proximal_indices] = new_positions
        # Quick relax
        mobj.soft_body.step(5)
        # Update rendering
        mobj.tet_vertices = mobj.soft_body.get_positions().astype(np.float32)
        mobj._prepare_tet_draw_arrays()

    print(f"Batched collision done: {num_passes} passes")


def _scan_motion_files(v):
    """Scan data/motion/ for .bvh files."""
    v.motion_bvh_files = sorted(glob.glob('data/motion/*.bvh'))


def _detect_bvh_tframe(bvh_path):
    """Detect if a BVH file needs T_frame=0.

    Two triggers:
    1. Non-upright skeleton rest pose (legs not Y-dominant, e.g. LaFAN1 flat).
    2. T-pose arms (LeftArm/RightArm OFFSET X-dominant). Skel uses N-pose
       rest (arms hanging); subtracting frame 0 aligns BVH rest with skel.
    Returns 0 if any trigger fires, None otherwise.
    """
    import re
    with open(bvh_path, 'r') as f:
        content = f.read()
    # Check #1: leg upright
    for joint in ['LeftLeg', 'RightLeg']:
        pattern = rf'JOINT\s+{joint}\s*\{{[^}}]*?OFFSET\s+([\d.\-e]+)\s+([\d.\-e]+)\s+([\d.\-e]+)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            x, y, z = abs(float(match.group(1))), abs(float(match.group(2))), abs(float(match.group(3)))
            max_axis = max(x, y, z)
            if max_axis < 1e-6:
                continue
            if y / max_axis <= 0.8:
                print(f"[Motion] Non-upright rest pose detected (thigh offset: "
                      f"{match.group(1)}, {match.group(2)}, {match.group(3)}), using T_frame=0")
                return 0
            break  # leg upright; continue to arm T-pose check
    # T-pose arm trigger removed: was causing leg pose asymmetry when frame 0
    # actor stance is mid-stride. Use bake_arm_retarget_bvh.py instead to
    # subtract frame 0 from arm joints only.
    return None


def _load_motion_bvh(v, idx):
    """Load the BVH file at the given index."""
    if idx < 0 or idx >= len(v.motion_bvh_files):
        return
    bvh_path = v.motion_bvh_files[idx]
    v.motion_selected_idx = idx
    import time as _t
    _t_total = _t.time()
    try:
        # Reset skeleton to zero pose first so captured root rotation is
        # canonical (identity).  Without this reset the previous BVH's last
        # frame leaks into motion_root_rotation, leaving "Fix Rotation"
        # locked to a slightly rotated pelvis when the new BVH is loaded.
        root_jn = v.env.skel.getJoint(0)
        root_dofs = root_jn.getNumDofs()
        if root_dofs == 6:
            n_dofs = v.env.skel.getNumDofs()
            v.env.skel.setPositions(np.zeros(n_dofs))
            init_pos = v.env.skel.getPositions()
            v.motion_root_translation = init_pos[3:6].copy()
            v.motion_root_rotation = init_pos[0:3].copy()
        else:
            v.motion_root_translation = None
            v.motion_root_rotation = None
        print(f"[Motion] bvh_info: {v.env.bvh_info}")
        print(f"[Motion] skel DOFs: {v.env.skel.getNumDofs()}, joints: {v.env.skel.getNumJoints()}")
        # Auto-detect if BVH needs T-pose correction (non-upright rest pose)
        _t0 = _t.time()
        t_frame = _detect_bvh_tframe(bvh_path)
        print(f"[Motion] _detect_bvh_tframe: {_t.time() - _t0:.2f}s")
        _t0 = _t.time()
        v.motion_bvh = MyBVH(bvh_path, v.env.bvh_info, v.env.skel, T_frame=t_frame)
        # Sibling .npy override: skip MyBVH conversion entirely, use raw mocap_refs.
        npy_path = bvh_path[:-4] + ".npy" if bvh_path.lower().endswith(".bvh") else bvh_path + ".npy"
        if os.path.exists(npy_path):
            try:
                arr = np.load(npy_path)
                if arr.ndim == 2 and arr.shape[1] == v.motion_bvh.mocap_refs.shape[1]:
                    v.motion_bvh.mocap_refs = arr
                    v.motion_bvh.num_frames = arr.shape[0]
                    print(f"[Motion] mocap override from {os.path.basename(npy_path)}: shape {arr.shape}")
                else:
                    print(f"[Motion] WARN {npy_path} shape {arr.shape} mismatch — skip override")
            except Exception as e:
                print(f"[Motion] WARN failed loading {npy_path}: {e}")
        print(f"[Motion] MyBVH parse: {_t.time() - _t0:.2f}s, mocap_refs shape: {v.motion_bvh.mocap_refs.shape}, max abs: {np.abs(v.motion_bvh.mocap_refs).max():.6f}")
        v.motion_total_frames = v.motion_bvh.num_frames
        v.motion_current_frame = 0
        v.motion_is_playing = False
        v.motion_play_accumulator = 0.0
        # Enable OBJ skeleton rendering so posed skeleton is visible
        v.draw_obj = True
        # Load cache synchronously: numpy / zipfile in a daemon thread
        # races with PyOpenGL client-side draw arrays in the main loop and
        # SIGSEGV's inside zipfile._EndRecData while the renderer is
        # mid-glDrawArrays.  Loading inline is fast enough (~0.5s) and
        # avoids the thread-safety hazard entirely.
        v.motion_deform_cache = {}
        v.motion_cache_loading = True
        _t0 = _t.time()
        try:
            _motion_load_cache(v)
            _motion_load_tissue_cage_overlay(v)
        finally:
            v.motion_cache_loading = False
            print(f"[Motion] Cache load: {_t.time() - _t0:.2f}s, {len(v.motion_deform_cache)} muscles")
        _t0 = _t.time()
        _motion_load_nn_checkpoint(v)
        print(f"[Motion] NN ckpt: {_t.time() - _t0:.2f}s")
        v.motion_bake_end_frame = min(v.motion_bake_end_frame, v.motion_total_frames - 1)
        _t0 = _t.time()
        _motion_reset(v)
        print(f"[Motion] _motion_reset: {_t.time() - _t0:.2f}s")
        print(f"Loaded motion: {os.path.basename(bvh_path)} ({v.motion_total_frames} frames, {1.0/v.motion_bvh.frame_time:.0f} FPS) — total {_t.time() - _t_total:.2f}s")
    except Exception as e:
        print(f"Error loading BVH: {e}")
        traceback.print_exc()
        v.motion_bvh = None
        v.motion_total_frames = 0
        v.motion_current_frame = 0


def _motion_apply_pose(v, frame):
    """Set skeleton to the BVH pose at the given frame."""
    if v.motion_bvh is None or frame >= v.motion_total_frames:
        return
    v.motion_current_frame = frame
    pose = v.motion_bvh.mocap_refs[frame].copy()
    # Optionally fix root translation axes at rest position
    if hasattr(v, 'motion_root_translation') and v.motion_root_translation is not None:
        if v.motion_fix_x:
            pose[3] = v.motion_root_translation[0]
        if v.motion_fix_y:
            pose[4] = v.motion_root_translation[1]
        if v.motion_fix_z:
            pose[5] = v.motion_root_translation[2]
    if v.motion_fix_rotation and hasattr(v, 'motion_root_rotation') and v.motion_root_rotation is not None:
        pose[0:3] = v.motion_root_rotation
    v.env.skel.setPositions(pose)
    # Sync the joint angle slider state
    if hasattr(v, '_skel_dofs'):
        v._skel_dofs = pose.copy()


def _motion_step_forward(v, count=1, run_tet=False):
    """Advance count frames, applying pose+deformation only on the final frame.

    run_tet: if True, run tet sim for frames not in cache (used by Step+1 button and baking).
             When run_tet is True, we must step sequentially (each frame needs sim).
             Play mode always passes False — skip to final frame directly.
    """
    if v.motion_bvh is None:
        return
    if run_tet:
        # Sequential mode: must apply each frame for tet sim
        for _ in range(count):
            next_frame = v.motion_current_frame + 1
            if next_frame >= v.motion_total_frames:
                if v.motion_repeat:
                    next_frame = 0
                else:
                    v.motion_is_playing = False
                    return
            _motion_apply_pose(v, next_frame)
            if not _motion_apply_cached_deformation(v, next_frame):
                _motion_run_tet_settle(v)
            if getattr(v, 'reverse_lbs_enabled', False):
                _reverse_lbs_apply(v)
    else:
        # Skip mode: jump directly to final frame, apply pose+deformation once
        target_frame = v.motion_current_frame + count
        if target_frame >= v.motion_total_frames:
            if v.motion_repeat:
                target_frame = target_frame % v.motion_total_frames
            else:
                target_frame = v.motion_total_frames - 1
                v.motion_is_playing = False
        _motion_apply_pose(v, target_frame)
        if v.motion_use_nn and v.motion_nn_model is not None:
            _motion_apply_nn_deformation(v, target_frame)
            if v.motion_nn_error_heatmap:
                _motion_update_nn_error_heatmap(v, target_frame)
            else:
                _motion_clear_heatmap(v)
        else:
            _motion_clear_heatmap(v)
            _motion_apply_cached_deformation(v, target_frame)
        # Reverse-LBS overrides fiber waypoints last so it wins over cache/NN.
        if getattr(v, 'reverse_lbs_enabled', False):
            _reverse_lbs_apply(v)


def _motion_run_tet_settle(v):
    """Run coupled tet sim for all active soft bodies at current pose."""
    if getattr(v, 'use_fem_sim', False):
        from viewer.fem_sim import run_all_fem_sim
        run_all_fem_sim(v, max_iterations=10, tolerance=1e-4)
    else:
        run_all_tet_sim_with_constraints(v,
            max_iterations=v.motion_settle_iters,
            tolerance=1e-4
        )


def _motion_cache_dir(v):
    """Returns cache directory path for current BVH. Creates it if needed."""
    if v.motion_bvh is None:
        return None
    bvh_name = os.path.splitext(os.path.basename(v.motion_bvh_files[v.motion_selected_idx]))[0]
    cache_dir = f'data/motion_cache/{bvh_name}'
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _motion_cache_read_dirs(v):
    """Return cache roots in precedence order, including local offline bakes.

    ``data/motion_cache`` may be a read-only external mount.  The anatomical
    contact baker therefore writes to ``.bake_outputs/motion_cache``; treating
    it as a read overlay lets the viewer play those results without copying or
    mutating the external cache.
    """
    primary = _motion_cache_dir(v)
    if primary is None:
        return []
    bvh_name = os.path.basename(primary)
    local = os.path.join('.bake_outputs', 'motion_cache', bvh_name)
    dirs = [primary]
    if os.path.isdir(local) and os.path.realpath(local) != os.path.realpath(primary):
        dirs.append(local)
    return dirs


def _motion_load_tissue_cage_overlay(v):
    """Load the newest solved tissue-cage cache for viewport inspection."""
    candidates = []
    for root in _motion_cache_read_dirs(v):
        candidates.extend(glob.glob(os.path.join(
            root, "*", "__tissue_cage_chunk_*.npz")))
    if not candidates:
        v.tissue_cage_overlay = None
        print("[Tissue cage] No saved cage-state cache found")
        return
    candidates.sort(key=os.path.getmtime)
    # Establish topology from the newest bake. Older experiments may use a
    # different cage resolution; allowing the oldest file to choose topology
    # made a newly rebuilt cage silently invisible.
    rest = faces = None
    for path in reversed(candidates):
        try:
            newest = np.load(path)
            rest = np.asarray(newest["rest_positions"], dtype=np.float32)
            faces = np.asarray(newest["surface_faces"], dtype=np.int32)
            break
        except Exception:
            continue
    # Assemble compatible files old-to-new so later poses overlay frames.
    frame_positions = {}
    used = []
    for path in candidates:
        try:
            data = np.load(path)
            if (len(data["rest_positions"]) != len(rest)
                    or not np.array_equal(data["surface_faces"], faces)):
                continue
            for frame, positions in zip(data["frames"], data["positions"]):
                frame_positions[int(frame)] = np.asarray(
                    positions, dtype=np.float32)
            used.append(path)
        except Exception as exc:
            print(f"[Tissue cage] Skip {path}: {exc}")
    if rest is None:
        v.tissue_cage_overlay = None
        return
    edge_set = set()
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]),
                     (face[2], face[0])):
            edge_set.add(tuple(sorted((int(a), int(b)))))
    edges = np.asarray(sorted(edge_set), dtype=np.int32)
    v.tissue_cage_overlay = {
        "rest": rest, "faces": faces, "edges": edges,
        "frames": frame_positions, "sources": used,
    }
    print(f"[Tissue cage] Loaded {len(frame_positions)} posed frames, "
          f"{len(rest)} vertices, {len(edges)} surface edges")


def draw_tissue_cage_overlay(v):
    overlay = getattr(v, "tissue_cage_overlay", None)
    if not getattr(v, "draw_tissue_cage", False) or overlay is None:
        return
    use_rest = getattr(v, "draw_tissue_cage_rest", False)
    frame = int(getattr(v, "motion_current_frame", 0))
    positions = (overlay["rest"] if use_rest
                 else overlay["frames"].get(frame, overlay["rest"]))
    # Cage states are baked in the original BVH world frame. Apply exactly
    # the same root-axis/root-rotation correction used for cached muscles.
    if (not use_rest and frame in overlay["frames"]
            and getattr(v, "motion_bvh", None) is not None):
        fix_offset = np.zeros(3, dtype=np.float32)
        root_translation = getattr(v, "motion_root_translation", None)
        if root_translation is not None:
            baked_translation = v.motion_bvh.mocap_refs[frame, 3:6]
            if getattr(v, "motion_fix_x", False):
                fix_offset[0] = (
                    root_translation[0] - baked_translation[0])
            if getattr(v, "motion_fix_y", False):
                fix_offset[1] = (
                    root_translation[1] - baked_translation[1])
            if getattr(v, "motion_fix_z", False):
                fix_offset[2] = (
                    root_translation[2] - baked_translation[2])
        if (getattr(v, "motion_fix_rotation", False)
                and getattr(v, "motion_root_rotation", None) is not None):
            root_body = v.env.skel.getJoint(0).getChildBodyNode()
            current_transform = root_body.getWorldTransform().matrix()
            saved_pose = v.env.skel.getPositions().copy()
            v.env.skel.setPositions(v.motion_bvh.mocap_refs[frame])
            baked_transform = root_body.getWorldTransform().matrix()
            v.env.skel.setPositions(saved_pose)
            fix_rotation = (
                current_transform[:3, :3]
                @ baked_transform[:3, :3].T).astype(np.float32)
            positions = (
                (fix_rotation
                 @ (positions - baked_transform[:3, 3]).T).T
                + current_transform[:3, 3])
        else:
            positions = positions + fix_offset
    edges = overlay["edges"]
    line_vertices = np.ascontiguousarray(
        positions[edges].reshape(-1, 3), dtype=np.float32)
    v._tissue_cage_draw_keepalive = line_vertices
    glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_COLOR_BUFFER_BIT)
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(float(getattr(v, "tissue_cage_line_width", 1.0)))
    color = ((0.15, 0.95, 1.0, 0.55) if use_rest
             else (1.0, 0.75, 0.1, 0.7))
    glColor4f(*color)
    glEnableClientState(GL_VERTEX_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glVertexPointer(3, GL_FLOAT, 0, line_vertices)
    glDrawArrays(GL_LINES, 0, len(line_vertices))
    glDisableClientState(GL_VERTEX_ARRAY)
    glPopAttrib()


def _motion_load_nn_checkpoint(v):
    """Load NN checkpoint and rest positions (shared across all motions)."""
    v.motion_nn_model = None
    v.motion_nn_rest_positions = None
    v.motion_nn_checkpoint_path = None
    v._motion_nn_model_version = "v1"
    # Check for per-motion checkpoint first: V2 mirror > V1 mirror > regular > global
    ckpt_path = None
    if v.motion_bvh is not None and hasattr(v, 'motion_selected_idx'):
        bvh_stem = os.path.splitext(os.path.basename(v.motion_bvh_files[v.motion_selected_idx]))[0]
        per_motion_v1dec_alt = f'volume_distill/{bvh_stem}_v1dec_checkpoints/best_v1dec.pt'
        per_motion_v1dec = f'volume_distill/{bvh_stem}_mirror_checkpoints/best_v1dec.pt'
        per_motion_v1_pca = f'volume_distill/{bvh_stem}_mirror_checkpoints/best_v1_pca.pt'
        per_motion_mirror_v2 = f'volume_distill/{bvh_stem}_mirror_checkpoints/best_v2.pt'
        per_motion_mirror = f'volume_distill/{bvh_stem}_mirror_checkpoints/best.pt'
        per_motion = f'volume_distill/{bvh_stem}_checkpoints/best.pt'
        # Also try base motion name (e.g. "dance1_subject1" → "dance")
        import re
        base_stem = re.sub(r'\d+(_subject\d+)?$', '', bvh_stem)
        base_v1dec_alt = f'volume_distill/{base_stem}_v1dec_checkpoints/best_v1dec.pt' if base_stem != bvh_stem else None
        if os.path.exists(per_motion_v1dec_alt):
            ckpt_path = per_motion_v1dec_alt
        elif os.path.exists(per_motion_v1dec):
            ckpt_path = per_motion_v1dec
        elif os.path.exists(per_motion_v1_pca):
            ckpt_path = per_motion_v1_pca
        elif os.path.exists(per_motion_mirror_v2):
            ckpt_path = per_motion_mirror_v2
        elif os.path.exists(per_motion_mirror):
            ckpt_path = per_motion_mirror
        elif os.path.exists(per_motion):
            ckpt_path = per_motion
        elif base_v1dec_alt and os.path.exists(base_v1dec_alt):
            ckpt_path = base_v1dec_alt
    if ckpt_path is None:
        ckpt_path = 'volume_distill/dof_grid_checkpoints/best.pt'
    if not os.path.exists(ckpt_path):
        ckpt_path = 'volume_distill/checkpoints/best.pt'
    if not os.path.exists(ckpt_path):
        print(f"[Motion] No checkpoint found (last tried: {ckpt_path})")
        return
    print(f"[Motion] Loading NN checkpoint: {ckpt_path}")
    try:
        import torch
        from volume_distill.dance.evaluate import load_model
        import torch as _torch
        _nn_device = 'cuda' if _torch.cuda.is_available() else 'cpu'
        model, metadata = load_model(ckpt_path, device=_nn_device)
        rest_positions = metadata.get("rest_positions")
        if rest_positions is None:
            preproc_path = 'data/motion_cache/locomotion/preprocessed.pt'
            data = torch.load(preproc_path, map_location='cpu', weights_only=False)
            rest_positions = data["rest_positions"]
        # TODO: torch.compile causes issues with dict-output models, re-enable later
        v.motion_nn_model = model
        v.motion_nn_rest_positions = rest_positions
        v.motion_nn_r_rest_positions = metadata.get("r_rest_positions")
        v.motion_nn_checkpoint_path = ckpt_path
        v._motion_nn_epoch = metadata.get("epoch", "?")
        v._motion_nn_val_loss = metadata.get("val_loss")
        v._motion_nn_model_version = metadata.get("model_version", "v1")
        v._motion_nn_mirror_trained = metadata.get("mirror_trained", False)
        print(f"[Motion] Loaded NN checkpoint: {ckpt_path} "
              f"(epoch {v._motion_nn_epoch}, version={v._motion_nn_model_version}"
              f"{', mirror' if v._motion_nn_mirror_trained else ''})")
    except Exception as e:
        print(f"[Motion] Failed to load NN checkpoint: {e}")
        v.motion_nn_model = None


def _predict_frame_batched_gpu(model, l_dofs, r_dofs, rest_positions, r_rest_positions=None, device=None):
    """V1/V1Dec GPU-end-to-end variant of _predict_frame_batched.

    Returns (l_world_dict, r_world_dict) of torch tensors on GPU, in pelvis-local frame.
    Caller must apply world transform R/t.  Use for fast pipelines (RL/sim) that
    need GPU output for downstream operations (e.g. update_waypoints_fast_gpu).
    """
    import torch as _torch
    if device is None:
        device = next(model.parameters()).device
    if r_rest_positions is None:
        r_rest_positions = rest_positions
    x = _torch.tensor(np.stack([l_dofs, r_dofs]), dtype=_torch.float32).to(device)
    with _torch.no_grad():
        preds = model(x)
    # Cache rest tensors on device
    if not hasattr(model, '_rest_dev'):
        model._rest_dev = {}
        model._r_rest_dev = {}
        for n in rest_positions:
            r = rest_positions[n]
            model._rest_dev[n] = (r.to(device) if isinstance(r, _torch.Tensor)
                                  else _torch.tensor(r, dtype=_torch.float32, device=device))
            rr = r_rest_positions.get(n, r) if r_rest_positions else r
            model._r_rest_dev[n] = (rr.to(device) if isinstance(rr, _torch.Tensor)
                                    else _torch.tensor(rr, dtype=_torch.float32, device=device))
    l_local = {}
    r_local = {}
    for name, disp_flat in preds.items():
        l_local[name] = model._rest_dev[name] + disp_flat[0].reshape(-1, 3)
        r_local[name] = model._r_rest_dev[name] + disp_flat[1].reshape(-1, 3)
    return l_local, r_local


def _predict_frame_batched(model, l_dofs, r_dofs, rest_positions, r_rest_positions=None, device=None):
    """Batch L+R DOFs into single forward pass. Returns (l_preds, r_preds) dicts.
    Uses L rest for L output and R rest for R output (they differ due to body asymmetry).
    Supports V1 (direct displacement), V1PCA (batched decoders + PCA), and V2 (PCA reconstruction)."""
    import torch as _torch
    from volume_distill.model import DistillNetV2
    if device is None:
        device = next(model.parameters()).device
    if r_rest_positions is None:
        r_rest_positions = rest_positions
    x = _torch.tensor(
        np.stack([l_dofs, r_dofs]), dtype=_torch.float32
    ).to(device)

    is_v2 = isinstance(model, DistillNetV2)
    has_pca = hasattr(model, '_pca_components')

    with _torch.no_grad():
        preds = model(x)

    l_result = {}
    r_result = {}

    if is_v2:
        # V2: preds = {muscle_idx: (2, K)} — PCA reconstruction
        idx_to_name = {v: k for k, v in model._muscle_name_to_idx.items()}
        if not hasattr(model, '_pca_comps_dev'):
            model._pca_comps_dev = {n: model._pca_components[n].to(device) for n in model._pca_components}
            model._pca_means_dev = {n: model._pca_means[n].to(device) for n in model._pca_means}
            model._pca_stds_dev = {n: model._pca_stds[n].to(device) for n in model._pca_stds} if model._pca_stds else None
        for m_idx, coeffs in preds.items():
            name = idx_to_name[m_idx]
            l_rest = rest_positions[name]
            r_rest = r_rest_positions.get(name, l_rest) if r_rest_positions else l_rest
            if isinstance(l_rest, _torch.Tensor):
                l_rest = l_rest.cpu()
            if isinstance(r_rest, _torch.Tensor):
                r_rest = r_rest.cpu()
            # Denormalize PCA coefficients
            l_c = coeffs[0]  # (K,)
            r_c = coeffs[1]  # (K,)
            if model._pca_stds_dev is not None and name in model._pca_stds_dev:
                stds = model._pca_stds_dev[name]
                l_c = l_c * stds
                r_c = r_c * stds
            comps = model._pca_comps_dev[name]
            means = model._pca_means_dev[name]
            l_disp = (l_c @ comps + means).reshape(-1, 3).cpu()
            r_disp = (r_c @ comps + means).reshape(-1, 3).cpu()
            l_result[name] = (l_rest + l_disp).numpy()
            r_result[name] = (r_rest + r_disp).numpy()
    elif has_pca:
        # V1PCA: preds = {name: (2, K)} — PCA reconstruction
        if not hasattr(model, '_pca_comps_dev'):
            model._pca_comps_dev = {n: model._pca_components[n].to(device) for n in model._pca_components}
            model._pca_means_dev = {n: model._pca_means[n].to(device) for n in model._pca_means}
            model._pca_stds_dev = {n: model._pca_stds[n].to(device) for n in model._pca_stds} if model._pca_stds else None
        for name, coeffs in preds.items():
            l_rest = rest_positions[name]
            r_rest = r_rest_positions.get(name, l_rest) if r_rest_positions else l_rest
            if isinstance(l_rest, _torch.Tensor):
                l_rest = l_rest.cpu()
            if isinstance(r_rest, _torch.Tensor):
                r_rest = r_rest.cpu()
            l_c = coeffs[0]  # (K,)
            r_c = coeffs[1]  # (K,)
            if model._pca_stds_dev is not None and name in model._pca_stds_dev:
                stds = model._pca_stds_dev[name]
                l_c = l_c * stds
                r_c = r_c * stds
            comps = model._pca_comps_dev[name]
            means = model._pca_means_dev[name]
            l_disp = (l_c @ comps + means).reshape(-1, 3).cpu()
            r_disp = (r_c @ comps + means).reshape(-1, 3).cpu()
            l_result[name] = (l_rest + l_disp).numpy()
            r_result[name] = (r_rest + r_disp).numpy()
    else:
        # V1/V1Dec: preds = {name: (2, V*3)}
        # Cache rest positions on GPU for fast add
        if not hasattr(model, '_rest_dev'):
            model._rest_dev = {}
            model._r_rest_dev = {}
            for n in rest_positions:
                r = rest_positions[n]
                model._rest_dev[n] = (r.to(device) if isinstance(r, _torch.Tensor)
                                      else _torch.tensor(r, dtype=_torch.float32, device=device))
                rr = r_rest_positions.get(n, r) if r_rest_positions else r
                model._r_rest_dev[n] = (rr.to(device) if isinstance(rr, _torch.Tensor)
                                        else _torch.tensor(rr, dtype=_torch.float32, device=device))
        # Add rest + disp on GPU, then single bulk transfer
        for name, disp_flat in preds.items():
            l_pos = model._rest_dev[name] + disp_flat[0].reshape(-1, 3)
            r_pos = model._r_rest_dev[name] + disp_flat[1].reshape(-1, 3)
            l_result[name] = l_pos.cpu().numpy()
            r_result[name] = r_pos.cpu().numpy()
    return l_result, r_result


def _motion_apply_nn_deformation(v, frame):
    """Apply NN-predicted deformation for the given frame. Returns True if any muscle updated."""
    if v.motion_nn_model is None or v.motion_nn_rest_positions is None:
        return False
    if v.motion_bvh is None or frame >= v.motion_total_frames:
        return False
    try:
        from volume_distill.dance.evaluate import predict_frame

        v1_input_dim = getattr(v.motion_nn_model, '_input_dim', None)
        if v._motion_nn_model_version == "v3_dof":
            dof_indices = [6, 7, 8, 9, 10, 11, 12]
            dofs = v.motion_bvh.mocap_refs[frame, dof_indices].astype(np.float32)
        elif v._motion_nn_model_version in ("v2", "v1_pca", "v1dec"):
            if v1_input_dim is not None and v1_input_dim == 7:
                dof_indices = [6, 7, 8, 9, 10, 11, 12]
            else:
                dof_indices = [6, 7, 8, 9]
            dofs = v.motion_bvh.mocap_refs[frame, dof_indices].astype(np.float32)
        else:
            if v1_input_dim is not None and v1_input_dim == 7:
                dof_indices = [6, 7, 8, 9, 10, 11, 12]
                dofs = v.motion_bvh.mocap_refs[frame, dof_indices].astype(np.float32)
            elif v1_input_dim is not None and v1_input_dim == 4:
                dof_indices = [6, 7, 8, 9]
                dofs = v.motion_bvh.mocap_refs[frame, dof_indices].astype(np.float32)
            else:
                dof_indices = [6, 7, 8, 9]
                cur_dofs = v.motion_bvh.mocap_refs[frame, dof_indices].astype(np.float32)
                if frame > 0:
                    prev_dofs = v.motion_bvh.mocap_refs[frame - 1, dof_indices].astype(np.float32)
                else:
                    prev_dofs = cur_dofs
                dofs = np.concatenate([cur_dofs, cur_dofs - prev_dofs])

        # Batched L+R prediction in a single forward pass
        mirror_active = getattr(v, '_motion_nn_mirror_trained', False)
        use_gpu_path = (v._motion_nn_model_version in ("v1", "v1dec")
                        and mirror_active)
        if not mirror_active:
            predictions = predict_frame(v.motion_nn_model, dofs, v.motion_nn_rest_positions)
            r_preds = None
        elif use_gpu_path:
            if v1_input_dim is not None and v1_input_dim == 7:
                r_dof_indices = [18, 19, 20, 21, 22, 23, 24]
            else:
                r_dof_indices = [18, 19, 20, 21]
            r_dofs = v.motion_bvh.mocap_refs[frame, r_dof_indices].astype(np.float32)
            r_dofs[0] *= -1
            l_local_gpu, r_local_gpu = _predict_frame_batched_gpu(
                v.motion_nn_model, dofs, r_dofs, v.motion_nn_rest_positions,
                r_rest_positions=getattr(v, 'motion_nn_r_rest_positions', None))
            predictions = None
            r_preds = None
        else:
            if v1_input_dim is not None and v1_input_dim == 7:
                r_dof_indices = [18, 19, 20, 21, 22, 23, 24]
            else:
                r_dof_indices = [18, 19, 20, 21]
            r_dofs = v.motion_bvh.mocap_refs[frame, r_dof_indices].astype(np.float32)
            r_dofs[0] *= -1
            predictions, r_preds = _predict_frame_batched(
                v.motion_nn_model, dofs, r_dofs, v.motion_nn_rest_positions,
                r_rest_positions=getattr(v, 'motion_nn_r_rest_positions', None))

        # Get pelvis world transform
        T = v.env.skel.getBodyNode("Saccrum_Coccyx0").getWorldTransform().matrix()
        R = T[:3, :3].astype(np.float32)
        t = T[:3, 3].astype(np.float32)
        any_applied = False

        if use_gpu_path:
            import torch as _torch
            device = next(v.motion_nn_model.parameters()).device
            R_gpu = _torch.from_numpy(np.ascontiguousarray(R)).to(device)
            t_gpu = _torch.from_numpy(np.ascontiguousarray(t)).to(device)

            # Phase 1: pure GPU compute. Collect (mobj, world_gpu, wp_gpu_or_None).
            items = []
            for mname, local_pos_gpu in l_local_gpu.items():
                if mname not in v.zygote_muscle_meshes:
                    continue
                mobj = v.zygote_muscle_meshes[mname]
                if mobj.tet_vertices is None:
                    continue
                world_gpu = local_pos_gpu @ R_gpu.T + t_gpu
                wp_gpu = mobj.compute_waypoints_gpu(world_gpu) if hasattr(mobj, 'compute_waypoints_gpu') else None
                items.append((mobj, world_gpu, wp_gpu))
            for lname, local_pos_gpu in r_local_gpu.items():
                rname = "R_" + lname[2:] if lname.startswith("L_") else lname
                if rname not in v.zygote_muscle_meshes:
                    continue
                mobj = v.zygote_muscle_meshes[rname]
                if mobj.tet_vertices is None:
                    continue
                mirrored_gpu = local_pos_gpu.clone()
                mirrored_gpu[:, 0] = -mirrored_gpu[:, 0]
                world_gpu = mirrored_gpu @ R_gpu.T + t_gpu
                wp_gpu = mobj.compute_waypoints_gpu(world_gpu) if hasattr(mobj, 'compute_waypoints_gpu') else None
                items.append((mobj, world_gpu, wp_gpu))

            # Phase 2: TWO concatenated CPU syncs (tets + waypoints)
            tet_concat = _torch.cat([w for (_, w, _) in items], dim=0)
            wp_items = [wp for (_, _, wp) in items if wp is not None]
            wp_concat = _torch.cat(wp_items, dim=0) if wp_items else None
            tet_cpu = tet_concat.cpu().numpy()
            wp_cpu = wp_concat.cpu().numpy() if wp_concat is not None else None

            # Phase 3: scatter to numpy per-muscle, update tet draw
            tet_off = 0
            wp_off = 0
            for (mobj, world_gpu, wp_gpu) in items:
                n_v = world_gpu.shape[0]
                world_pos = tet_cpu[tet_off:tet_off + n_v]
                tet_off += n_v
                mobj.tet_vertices = world_pos
                mobj._update_tet_draw_positions(skip_normals=True)
                if wp_gpu is not None and wp_cpu is not None:
                    n_wp = wp_gpu.shape[0]
                    mobj.scatter_waypoints_numpy(wp_cpu[wp_off:wp_off + n_wp])
                    wp_off += n_wp
                any_applied = True
            return any_applied

        # Apply L predictions
        for mname, local_pos in predictions.items():
            if mname not in v.zygote_muscle_meshes:
                continue
            mobj = v.zygote_muscle_meshes[mname]
            if mobj.tet_vertices is None:
                continue
            world_pos = local_pos @ R.T + t
            if mobj.soft_body is not None:
                mobj.soft_body.positions = world_pos.astype(np.float64)
            mobj.tet_vertices = world_pos
            mobj._update_tet_draw_positions(skip_normals=True)
            if hasattr(mobj, 'update_waypoints_fast'):
                mobj.update_waypoints_fast()
            any_applied = True

        # Apply R predictions (mirrored)
        if r_preds is not None:
            for lname, local_pos in r_preds.items():
                rname = "R_" + lname[2:] if lname.startswith("L_") else lname
                if rname not in v.zygote_muscle_meshes:
                    continue
                mobj = v.zygote_muscle_meshes[rname]
                if mobj.tet_vertices is None:
                    continue
                mirrored_pos = local_pos.copy()
                mirrored_pos[:, 0] *= -1
                world_pos = mirrored_pos @ R.T + t
                if mobj.soft_body is not None:
                    mobj.soft_body.positions = world_pos.astype(np.float64)
                mobj.tet_vertices = world_pos
                mobj._update_tet_draw_positions(skip_normals=True)
                if hasattr(mobj, 'update_waypoints_fast'):
                    mobj.update_waypoints_fast()
                any_applied = True

        return any_applied
    except Exception as e:
        print(f"[Motion] NN inference error: {e}")
        return False


def _motion_clear_heatmap(v):
    """Remove heatmap colors from all muscles."""
    for mobj in v.zygote_muscle_meshes.values():
        if getattr(mobj, '_tet_surface_colors', None) is not None:
            mobj._tet_surface_colors = None


def _motion_update_nn_error_heatmap(v, frame):
    """Color muscles by per-vertex ||NN - GT|| error. Requires cache data for the frame."""
    if not v.motion_deform_cache:
        return
    # Compute fix_offset / fix_rot for cache positions (same logic as _motion_apply_cached_deformation)
    fix_offset = np.zeros(3, dtype=np.float32)
    if hasattr(v, 'motion_root_translation') and v.motion_root_translation is not None:
        bvh_trans = v.motion_bvh.mocap_refs[frame, 3:6]
        rest_trans = v.motion_root_translation
        if v.motion_fix_x:
            fix_offset[0] = rest_trans[0] - bvh_trans[0]
        if v.motion_fix_y:
            fix_offset[1] = rest_trans[1] - bvh_trans[1]
        if v.motion_fix_z:
            fix_offset[2] = rest_trans[2] - bvh_trans[2]
    fix_rot_mat = None
    fix_dest = None
    pivot = None
    if v.motion_fix_rotation and hasattr(v, 'motion_root_rotation') and v.motion_root_rotation is not None:
        root_bn = v.env.skel.getJoint(0).getChildBodyNode()
        # Current (fixed) root body transform
        T_now = root_bn.getWorldTransform().matrix()
        R_now = T_now[:3, :3]
        t_now = T_now[:3, 3]
        # Baked root body transform — temporarily set to original frame pose
        saved_pos = v.env.skel.getPositions().copy()
        v.env.skel.setPositions(v.motion_bvh.mocap_refs[frame])
        T_baked = root_bn.getWorldTransform().matrix()
        R_baked = T_baked[:3, :3]
        t_baked = T_baked[:3, 3]
        v.env.skel.setPositions(saved_pos)
        fix_rot_mat = (R_now @ R_baked.T).astype(np.float32)
        pivot = t_baked.astype(np.float32)
        fix_dest = t_now.astype(np.float32)

    HEATMAP_SCALE = 0.01  # 1 cm = full red
    for mname, mobj in v.zygote_muscle_meshes.items():
        if mobj.tet_vertices is None:
            continue
        if mname not in v.motion_deform_cache or frame not in v.motion_deform_cache[mname]:
            mobj._tet_surface_colors = None
            continue
        cached = v.motion_deform_cache[mname][frame]
        if fix_rot_mat is not None:
            gt_pos = (fix_rot_mat @ (cached['positions'] - pivot).T).T + fix_dest
        else:
            gt_pos = cached['positions'] + fix_offset
        gt_pos = gt_pos.astype(np.float32)
        nn_pos = mobj.tet_vertices
        if nn_pos.shape[0] != gt_pos.shape[0]:
            mobj._tet_surface_colors = None
            continue
        error = np.linalg.norm(nn_pos - gt_pos, axis=1)  # per vertex
        t = np.clip(error / HEATMAP_SCALE, 0.0, 1.0)
        alpha = mobj.contour_mesh_transparency
        n = len(t)
        colors = np.empty((n, 4), dtype=np.float32)
        colors[:, 0] = 1.0          # R (always 1)
        colors[:, 1] = 1.0 - t      # G (white→red)
        colors[:, 2] = 1.0 - t      # B (white→red)
        colors[:, 3] = alpha     # A
        # Map vertex colors to surface triangle vertices
        vidx = getattr(mobj, '_tet_surface_vidx', None)
        if vidx is not None:
            mobj._tet_surface_colors = colors[vidx]
        else:
            mobj._tet_surface_colors = None


def _motion_save_current_frame(v):
    """Save deformed tet positions + waypoints for all muscles with soft_body at the current frame."""
    cache_dir = _motion_cache_dir(v)
    if cache_dir is None:
        return
    frame = v.motion_current_frame
    saved_count = 0
    for mname, mobj in v.zygote_muscle_meshes.items():
        if mobj.soft_body is None:
            continue
        positions = mobj.soft_body.get_positions().astype(np.float32)
        # Flatten waypoints
        wp_flat = wp_shape = None
        if hasattr(mobj, 'waypoints') and len(mobj.waypoints) > 0:
            wp_flat, wp_shape = _flatten_waypoints(mobj.waypoints)

        filepath = os.path.join(cache_dir, f'{mname}.npz')
        if os.path.exists(filepath):
            data = np.load(filepath, allow_pickle=True)
            frames = list(data['frames'])
            all_pos = list(data['positions'])
            all_wp = list(data['waypoints_flat']) if 'waypoints_flat' in data else [None] * len(frames)
            if frame in frames:
                idx = frames.index(frame)
                all_pos[idx] = positions
                if wp_flat is not None:
                    all_wp[idx] = wp_flat
            else:
                frames.append(frame)
                all_pos.append(positions)
                all_wp.append(wp_flat)
                order = np.argsort(frames)
                frames = [frames[i] for i in order]
                all_pos = [all_pos[i] for i in order]
                all_wp = [all_wp[i] for i in order]
        else:
            frames = [frame]
            all_pos = [positions]
            all_wp = [wp_flat]

        save_dict = dict(
            frames=np.array(frames, dtype=np.int32),
            positions=np.array(all_pos, dtype=np.float32),
        )
        if wp_flat is not None and all(w is not None for w in all_wp):
            save_dict['waypoints_flat'] = np.stack(all_wp).astype(np.float32)
            save_dict['waypoints_shape'] = np.array([wp_shape.encode('utf-8')])
        np.savez_compressed(filepath, **save_dict)
        saved_count += 1
    _motion_load_cache(v)
    print(f"Saved deformation at frame {frame} for {saved_count} muscles")


def _motion_patch_waypoints(v):
    """Patch existing cache files by computing waypoints from cached tet positions.
    Much faster than re-baking since it skips tet sim — only does barycentric interpolation.
    Sets skeleton pose per frame so origin/insertion endpoints update correctly.
    Supports both legacy single-file and chunked cache formats."""
    cache_dir = _motion_cache_dir(v)
    if cache_dir is None:
        return
    # Collect muscles that need patching — find all cache files (legacy + chunks)
    to_patch = {}  # mname -> [(filepath, frames, positions), ...]
    for mname, mobj in v.zygote_muscle_meshes.items():
        if mobj.soft_body is None:
            continue
        if not (hasattr(mobj, 'waypoints') and len(mobj.waypoints) > 0):
            continue
        if not (hasattr(mobj, 'waypoint_bary_coords') and len(mobj.waypoint_bary_coords) > 0):
            continue
        file_list = []
        legacy = os.path.join(cache_dir, f'{mname}.npz')
        if os.path.exists(legacy):
            file_list.append(legacy)
        file_list.extend(sorted(glob.glob(os.path.join(cache_dir, f'{mname}_chunk_*.npz'))))
        if not file_list:
            continue
        entries = []
        for fp in file_list:
            data = np.load(fp, allow_pickle=True)
            entries.append((fp, data['frames'], data['positions']))
        to_patch[mname] = entries

    if not to_patch:
        print("All cache files already have waypoints")
        return

    # Build sorted list of all unique frame indices across muscles
    all_frames = sorted(set(
        int(f)
        for entries in to_patch.values()
        for _, frames, _ in entries
        for f in frames
    ))

    # Per-muscle: build frame->index mapping and accumulate results
    muscle_wp = {}  # mname -> {frame_idx: wp_flat}
    muscle_wp_shape = {}  # mname -> wp_shape_str
    for mname in to_patch:
        muscle_wp[mname] = {}

    # Build fast lookup: mname -> {frame_idx: (entry_idx, pos_idx)}
    muscle_frame_map = {}
    for mname, entries in to_patch.items():
        fmap = {}
        for entry_idx, (fp, frames, positions) in enumerate(entries):
            for pos_idx, f in enumerate(frames):
                fmap[int(f)] = (entry_idx, pos_idx)
        muscle_frame_map[mname] = fmap

    # Temporarily disable fix axes — cached positions were baked with original pose
    saved_fix_x = getattr(v, 'motion_fix_x', False)
    saved_fix_y = getattr(v, 'motion_fix_y', False)
    saved_fix_z = getattr(v, 'motion_fix_z', False)
    saved_fix_rot = getattr(v, 'motion_fix_rotation', False)
    v.motion_fix_x = False
    v.motion_fix_y = False
    v.motion_fix_z = False
    v.motion_fix_rotation = False

    # Iterate frames once, update all muscles per frame
    for frame_idx in all_frames:
        if frame_idx < v.motion_total_frames:
            _motion_apply_pose(v, frame_idx)
        for mname, entries in to_patch.items():
            fmap = muscle_frame_map[mname]
            if frame_idx not in fmap:
                continue
            entry_idx, pos_idx = fmap[frame_idx]
            positions = entries[entry_idx][2]
            mobj = v.zygote_muscle_meshes[mname]
            mobj.tet_vertices = positions[pos_idx].astype(np.float32).copy()
            mobj._update_waypoints_from_tet(v.env.skel, verbose=False)
            wp_flat, wp_shape_str = _flatten_waypoints(mobj.waypoints)
            muscle_wp[mname][frame_idx] = wp_flat
            muscle_wp_shape[mname] = wp_shape_str

    # Write patched files — one per original file
    patched = 0
    for mname, entries in to_patch.items():
        for filepath, frames, positions in entries:
            frame_list = [int(f) for f in frames]
            wp_flats = [muscle_wp[mname][f] for f in frame_list]
            save_dict = dict(
                frames=frames,
                positions=positions,
                waypoints_flat=np.stack(wp_flats).astype(np.float32),
                waypoints_shape=np.array([muscle_wp_shape[mname].encode('utf-8')]),
            )
            np.savez_compressed(filepath, **save_dict)
        patched += 1
        n_frames = sum(len(frames) for _, frames, _ in entries)
        print(f"  Patched {mname}: {n_frames} frames across {len(entries)} file(s)")

    # Restore fix axes
    v.motion_fix_x = saved_fix_x
    v.motion_fix_y = saved_fix_y
    v.motion_fix_z = saved_fix_z
    v.motion_fix_rotation = saved_fix_rot

    _motion_load_cache(v)
    print(f"Waypoint patch complete: {patched} muscles updated")


def _motion_load_cache(v, force=False, prefer_latest=False):
    """Load all cached deformation data for the current BVH into memory.
    Supports both legacy single-file ({mname}.npz) and chunked ({mname}_chunk_*.npz) formats.

    Filters cache entries by vertex-count match against current tet so a
    1216-vert original-mesh bake doesn't silently shadow a 512-vert contour
    bake (or vice versa) when subdirs from both modes coexist.

    Chunk files are read in parallel via a thread pool — ~10k chunk reads
    for a 7840-frame BVH × 25 muscles is pure I/O bound and scales well
    with concurrent reads (Linux page cache + NVMe).

    Incremental by default: muscles already in v.motion_deform_cache are
    skipped, and removed muscles are pruned.  Pass force=True to clear and
    rebuild from scratch (use when chunks on disk have changed).
    """
    from concurrent.futures import ThreadPoolExecutor
    import zipfile
    import time as _t
    _t_start = _t.time()

    if force or not hasattr(v, 'motion_deform_cache') or v.motion_deform_cache is None:
        v.motion_deform_cache = {}
    else:
        # Prune entries for muscles no longer loaded
        current_names = set(v.zygote_muscle_meshes.keys())
        for stale in [n for n in v.motion_deform_cache if n not in current_names]:
            del v.motion_deform_cache[stale]
    cache_dirs = _motion_cache_read_dirs(v)
    if not cache_dirs:
        return

    # ── 1. Collect (mname, npz_files_sorted, expected_n) per muscle ──
    muscle_files = []
    for mname in v.zygote_muscle_meshes:
        if mname in v.motion_deform_cache and not force:
            continue
        mobj = v.zygote_muscle_meshes[mname]
        # Skip muscles without tet — cached deformation can't apply, and
        # scanning disk for chunks adds significant load time over slow mounts.
        if mobj.tet_vertices is None:
            continue
        expected_n = mobj.tet_vertices.shape[0]
        npz_files = []
        for cache_dir in cache_dirs:
            for subdir in glob.glob(os.path.join(cache_dir, '*/')):
                # Smoke/test outputs may contain a syntactically valid frame
                # written before a bake later fails. They are diagnostics,
                # never viewer overlays.
                variant = os.path.basename(os.path.normpath(subdir)).lower()
                accepted_frame0 = variant.endswith('_anatomical_contact_test')
                if ('smoke' in variant or 'probe' in variant
                        or (variant.endswith('_test')
                                            and not accepted_frame0)
                        or variant.startswith('rejected')):
                    continue
                npz_files.extend(glob.glob(os.path.join(subdir, f'{mname}_chunk_*.npz')))
                sub_legacy = os.path.join(subdir, f'{mname}.npz')
                if os.path.exists(sub_legacy):
                    npz_files.append(sub_legacy)
            npz_files.extend(glob.glob(os.path.join(cache_dir, f'{mname}_chunk_*.npz')))
            legacy = os.path.join(cache_dir, f'{mname}.npz')
            if os.path.exists(legacy):
                npz_files.append(legacy)
        def cache_precedence(path):
            variant = os.path.basename(os.path.dirname(path)).lower()
            # Deterministic semantic precedence. The corrected full bake is
            # the baseline; a completed anatomical bake replaces it; the
            # medial-tibia repair is the final per-muscle overlay. Do not use
            # mtimes to decide anatomy.
            if variant == 'l_vastus_intermedius_subdivided_fast_smooth_arap_full76_v38':
                # Complete accelerated Smooth-ARAP bake. Frames 27-29 and
                # 72-75 are independently initialized to prevent continuation
                # drift while preserving a coherent formulation and topology.
                rank = 1170
            elif variant == 'l_vastus_intermedius_subdivided_tetwild_direct_remesh_frames0_5_v27':
                # Direct TetWild-remeshed VI: no preserved input topology, no
                # cage, zero inverted tets and zero active SDF residual 0-5.
                rank = 1120
            elif variant == 'l_vastus_intermedius_subdivided_joint_shape_contact_frames13_26_v31':
                # Joint corotational shape/contact/volume solve. Selectively
                # accepted through frame 26; frame 27 attachment failure omitted.
                rank = 1130
            elif variant == 'l_vastus_intermedius_subdivided_joint_shape_contact_frames13_27_full_v32':
                # Unfiltered diagnostic exposure, including frame 27's known
                # attachment failure, as explicitly requested for inspection.
                rank = 1140
            elif variant == 'l_vastus_intermedius_subdivided_smooth_arap_frames13_26_full_v34':
                # Smooth ARAP (Oehri et al.) volumetric graph adaptation,
                # fully exposed including frame 26's attachment failure.
                rank = 1150
            elif variant == 'l_vastus_intermedius_subdivided_smooth_arap_best_full76_v37':
                # Complete reviewed cache: Smooth ARAP through frame 26, then
                # validated direct-remesh fallback where continuation failed.
                rank = 1160
            elif variant == 'l_vastus_intermedius_subdivided_tetwild_direct_remesh_full76_v28':
                # Full diagnostic bake. Accepted v27 overrides its first six
                # frames; the remaining poses stay visible for failure review.
                rank = 1115
            elif variant == 'l_vastus_intermedius_subdivided_fast_strong_arap_quartermm_frames0_10_v26':
                # ~40% faster continuation with a 0.25 mm shell, comparable
                # visible-edge shape, and zero active SDF residual through 10.
                rank = 1110
            elif variant == 'l_vastus_intermedius_subdivided_strong_arap_halfmm_surface_frames0_10_v25':
                # Gradual direct-tet continuation through frame 10 with zero
                # active SDF residual and the accepted strong-ARAP settings.
                rank = 1100
            elif variant == 'l_vastus_intermedius_subdivided_strong_arap_halfmm_surface_frames0_5_v24':
                # Strong direct ARAP with a 0.5 mm zero-surface contact shell;
                # rest-inside samples remain exempt and final SDF residual is zero.
                rank = 1090
            elif variant == 'l_vastus_intermedius_subdivided_rest_inside_exempt_frames0_5_v21':
                # Direct-tet frames 0-5 with a fixed authored-rest mask:
                # vertices/edges originally inside femur receive no SDF force.
                rank = 1080
            elif variant == 'l_vastus_intermedius_subdivided_direct_arc_open_sdf_frames0_5_v20':
                # Validated direct-tet frames 0-5: original open anatomical
                # surface contact has only nanometre-scale SDF residuals.
                rank = 1070
            elif variant == 'l_vastus_intermedius_subdivided_direct_arc_open_sdf_frame25_v19':
                # Direct subdivided-tet frame 25. The original open surface
                # alone receives femur-SDF contact; the real patellar contour
                # reaches its target through a knee-centered arc homotopy.
                rank = 1060
            elif variant == 'l_vastus_intermedius_subdivided_direct_tet_arap_sdf_frame25_v16':
                # Exact subdivided surface is the simulation tet boundary:
                # direct ARAP DOFs, direct femur-SDF contact, no render cage.
                rank = 1050
            elif variant == 'l_vastus_intermedius_subdivided_cap_excluded12_wrap_frame25_v12':
                # Obsolete embedded-cage experiment retained only as history.
                rank = 900
            elif variant == 'l_vastus_intermedius_frame25_regime_full76_piecewise_v12':
                # Full independent GPU bake using the stable frame-25
                # compact-origin/coupled-contact formulation on every pose.
                # Piecewise cage transfer retains the refined SDF clearance
                # through the intercondylar frame instead of smoothing it away.
                rank = 1035
            elif variant == 'l_vastus_intermedius_cacheless_origin_shape_reference_v9':
                # The complete origin cap keeps its internal reference shape,
                # while all 32 femoral attachment positions remain uniformly
                # soft. This avoids the widened near-rest cap without turning
                # the origin into a hard positional constraint.
                rank = 1030
            elif variant == 'l_vastus_intermedius_cacheless_refined_origin_cohesion_v8':
                # Eight-frame origin-quality correction. Strong rest-edge
                # cohesion within the authored origin cap is preserved by
                # both ARAP and coupled contact/volume refinement, preventing
                # zero-flexion cap widening without pinning its position.
                rank = 1025
            elif variant == 'l_vastus_intermedius_cacheless_refined_origin_frame6shape_v6_clean':
                # Fast seven-frame update: zero-flexion poses use the
                # preferred frame-6-like compact origin stiffness. Frame 4
                # retains its higher-quality validated smoothfast result.
                rank = 1020
            elif variant == 'l_vastus_intermedius_cacheless_refined_origin_smoothfast_v5':
                # Fast selective rebake of the 14 near-rest transition poses.
                # The compact origin anchors stiffen continuously over 15
                # degrees, avoiding a binary attachment-mode shape jump while
                # retaining validated hard-patch deep-flexion frames.
                rank = 1015
            elif variant == 'l_vastus_intermedius_cacheless_refined_origin_compliant_v3':
                # Near rest, release the compact hard-origin patch and use a
                # uniform compliant 32-vertex femoral attachment. This lets
                # ARAP retain the natural proximal shape; the hard patch is
                # restored above one degree knee flexion.
                rank = 1010
            elif variant == 'l_vastus_intermedius_cacheless_refined_origin_gated_v2':
                # Validated VI refinement: the 22 non-hard origin-cap
                # vertices follow the femur only below 1 degree knee flexion,
                # preventing the near-rest attachment cap from folding
                # forward without changing the accepted flexed poses.
                rank = 1005
            elif variant == 'l_vastus_intermedius_cacheless_refined_coupled_v1':
                # Cache-free VI: independent refined-tet ARAP/SDF/volume
                # solves with compact femoral origin and full patellar
                # insertion, transferred by rest-pose tet embedding.
                rank = 1000
            elif variant == 'l_vastus_intermedius_new_corotated_arap_sdf_v2':
                # New cache-free generalized corotated-ARAP muscle solver
                # with dense differentiable femur-SDF contact and soft caps.
                rank = 995
            elif variant == 'l_vastus_intermedius_cacheless_rest_arap_sdf_v1':
                # Sole cache-free VI bake: rest tet + BVH attachments + femur
                # SDF, with state generated internally from frame zero.
                rank = 990
            elif variant == 'l_vastus_intermedius_nearperfect_regenerated_v4_exact':
                # Reconstructed historical v4 pipeline: 16 accepted key
                # poses, femur-local smoothstep resampling, legacy eight-guide
                # overwrite, and five recorded SDF-refined frame replacements.
                rank = 985
            elif variant == 'l_vastus_intermedius_fast_sdf_raw_v1':
                # Keep raw one-pose experiments inspectable without allowing
                # them to split a coherent full-frame VI bake at frame 25.
                rank = 950
            elif variant == 'l_vastus_intermedius_nearperfect_collision_resolved_v4':
                # Preserves the reviewed refined VI key poses exactly; smooth
                # femur-local continuation plus coupled SDF refinement clears
                # all authored-outside femur crossings on intermediate frames.
                rank = 980
            elif variant == 'quadriceps_contact_wrapped_deep_gpu_v1':
                # Deep-flexion VI/VM overlays use differentiable femur-SDF
                # contact coupled to signed-volume barriers. VL retains the
                # inversion-free lateral-routing result.
                rank = 970
            elif variant == 'quadriceps_rest_clearance_signed_volume_gpu_v1':
                # CUDA ARAP with sparse anatomical anchors, per-tet
                # signed-volume barriers, and one-sided authored-rest SDF
                # clearance. Validated across all 76 smooth-pose frames.
                rank = 960
            elif variant == 'quadriceps_compact_femur_following_full_v3':
                # Compact full bake: every free vertex preserves frame-0
                # femur proximity; only the true insertion cap follows patella.
                rank = 940
            elif variant == 'quadriceps_compact_manifold_full_gpu_v1':
                # Full compact-reference bake with strong volume, rest-SDF
                # manifold, exact caps, and a 10 mm shape trust region.
                rank = 930
            elif variant == 'quadriceps_vi_vl_vm_smooth_true_attached_gpu_v3':
                # Rest-bound skeletal caps with independently verified
                # attachment error below 3.4e-8 m across all 76 frames.
                rank = 920
            elif variant == 'quadriceps_vi_vl_vm_smooth_joint_gpu_v1':
                # Full 76-frame joint GPU bake on the eased pose motion.
                rank = 900
            elif variant == 'quadriceps_vi_vl_vm_reference_atlas_v1':
                # Every VM frame is rebuilt from its best collision-free
                # reference transfer; VI and VL remain unchanged.
                rank = 830
            elif variant == 'quadriceps_vi_vl_vm_reference_selected_v3':
                # Per-pose VM reference transfer selected by positive tet
                # quality and moving-femur SDF clearance.
                rank = 820
            elif variant == 'quadriceps_vi_vl_vm_bounded_stable_v1':
                # VM stabilization with an enforced 8 mm per-vertex trust
                # radius, preventing the optimizer from producing spikes.
                rank = 810
            elif variant == 'quadriceps_vi_vl_vm_stable_fast_v4':
                # VM-targeted GPU stabilization: VI/VL remain fixed contact
                # partners while a strong positive-volume barrier repairs VM.
                rank = 800
            elif variant == 'quadriceps_vi_vl_vm_relational_full_collisionfree_v3':
                # Full 16-frame relational VI/VL/VM bake with exact caps,
                # persistent inter-muscle cohesion, and validated femur clearance.
                rank = 790
            elif variant == 'quadriceps_vi_vl_vm_relational_frame5_v1':
                # Joint quadriceps frame with symmetric proximity contact plus
                # 343 persistent rest-neighbor cohesion links.
                rank = 780
            elif variant == 'quadriceps_vi_vl_vm_joint_frame5_final':
                # Joint three-muscle frame-5 solve: exact full caps, all three
                # pair contacts, and validated femur clearance for VI/VL/VM.
                rank = 770
            elif variant == 'quadriceps_vi_vl_joint_frame5_final2':
                # Joint VI/VL frame-5 solve with exact full caps, symmetric
                # pair separation, and independently validated femur SDF.
                rank = 760
            elif variant == 'quadriceps_vi_vl_attached_fixed_v1':
                # Accepted attached state: complete 32-vertex VL origin and
                # patellar caps, with no guided transition-ring peel.
                rank = 750
            elif variant == 'quadriceps_vi_vl_insertion_ring1_merged_v1':
                # Rejected as default: the strong first ring creates a sharp
                # stiffness boundary and moves the visible peel line upward.
                rank = 722
            elif variant == 'quadriceps_vi_vl_fast_sdf_merged_v1':
                # Accepted fast path: unchanged VI/full-cap poses with the
                # difficult VL frame replaced by the validated projective-SDF
                # continuation result.
                rank = 724
            elif variant == 'quadriceps_vi_vl_softtransition_merged_v1':
                # Rejected as the default: collision-free, but the guided
                # transition rings introduce visible longitudinal VL stretch.
                # Keep it inspectable below the accepted full-cap variant.
                rank = 723
            elif variant == 'quadriceps_vi_vl_frame5_fullcap_v1':
                # Full 32-vertex endpoint caps prevent VL's origin from
                # peeling laterally during the 120-degree continuation.
                rank = 730
            elif variant == 'quadriceps_vi_vl_frame5_continuation_v1':
                # VI + VL quasistatic pair. VL's 120-degree knee pose is
                # reached through hidden continuation frames, then mapped
                # back to the original 16-frame BVH indexing.
                rank = 720
            elif variant == 'l_vastus_intermedius_refined_transfer_v1':
                # Distal/bone-adjacent refined tet simulation transferred to
                # the original viewer topology, then SDF-validated.
                rank = 710
            elif variant == 'l_vastus_intermedius_snh_soft_attachment_diagnostic_v1':
                # Frame-4 Stable-NH experiment: inversion-free with compliant
                # attachments, intentionally exposed for visual diagnosis.
                rank = 675
            elif variant == 'l_vastus_intermedius_stiff_arap_femur_sdf_coupled_v1':
                # Joint edge/volume/contact solve. The infeasible 120-degree
                # stress frame uses its validated bounded-contact fallback.
                rank = 690
            elif variant == 'l_vastus_intermedius_stiff_arap_femur_sdf_bounded_v1':
                # Bounded final SDF projection: collision-free without the
                # high-weight least-squares vertex ejection failure.
                rank = 680
            elif variant == 'l_vastus_intermedius_stiff_arap_femur_sdf_soft_v1':
                rank = 660
            elif variant == 'l_vastus_intermedius_stiff_arap_femur_sdf_v1':
                # VI override using the femur-local voxel SDF transformed by
                # L_Femur0 at each BVH frame.
                rank = 650
            elif variant == 'l_vastus_intermedius_sdf_tune_ratio100000':
                # Rejected: removes penetration by ejecting vertices. Keep it
                # inspectable, but never let it override stable VI caches.
                rank = 640
            elif variant == 'l_vastus_intermedius_stiff_arap_gpu_v1':
                # Explicit single-muscle VI override.  Without a semantic
                # rank, the older shared-cage cache (rank 500) silently
                # replaces this newer GPU PBD result during cache assembly.
                rank = 600
            elif ('contour_shared_cage' in variant
                    or 'contour_tissue_cage' in variant):
                # One volumetric cage drives the complete upper-leg contour
                # group; it is a deliberate replacement for per-muscle
                # overlays on the frames it contains.
                rank = 500
            elif variant.endswith('_anatomical_contact_full_0_4'):
                # Coherent five-frame, all-muscle LBS/untangle/PN bake. It
                # supersedes the old single-muscle and medial post-fix layers.
                rank = 400
            elif 'medial_tibia_repair' in variant:
                rank = 300
            elif variant.endswith('_anatomical_contact'):
                rank = 200
            elif variant.endswith('_anatomical_contact_test'):
                # This is the accepted collision-resolved frame-0 bake. It
                # supplements the full baseline until the complete anatomical
                # bake has passed validation.
                rank = 150
            elif 'headless_full_corrected' in variant:
                rank = 100
            else:
                rank = 0
            return rank, os.path.getmtime(path), path
        if prefer_latest:
            # Explicit UI reload means exactly what it says: completed/newer
            # bake files override semantic defaults without restarting the
            # Python viewer to learn a newly authored variant name.
            npz_files.sort(key=lambda path: (os.path.getmtime(path), path))
        else:
            npz_files.sort(key=cache_precedence)
        if not npz_files:
            continue
        muscle_files.append((mname, npz_files, expected_n))

    if not muscle_files:
        return

    # ── 2. Build flat work list (mname, file_order_idx, path, expected_n) ──
    work = []
    for mname, files, exp_n in muscle_files:
        for order_idx, path in enumerate(files):
            work.append((mname, order_idx, path, exp_n))

    def _load_one(item):
        mname, order_idx, path, exp_n = item
        try:
            # A bake checkpoints directly into the cache tree.  If the viewer
            # scans while that file is being replaced (or a bake was killed
            # mid-write), the .npz may temporarily be an incomplete zip.
            # Treat that candidate exactly like a topology mismatch: skip it
            # and continue loading older/complete chunks for this muscle.
            with np.load(path, allow_pickle=True) as data:
                positions = np.asarray(data['positions'])
                if (positions.ndim != 3
                        or (exp_n is not None
                            and positions.shape[1] != exp_n)):
                    return mname, order_idx, None
                frames = np.asarray(data['frames'])
                if len(frames) != len(positions):
                    raise ValueError(
                        "frame/position count mismatch: "
                        f"{len(frames)} != {len(positions)}")
                has_wp = ('waypoints_flat' in data
                          and 'waypoints_shape' in data)
                wp_flats = (np.asarray(data['waypoints_flat'])
                            if has_wp else None)
                wp_shape_str = None
                if has_wp:
                    raw = data['waypoints_shape'][0]
                    wp_shape_str = (raw.decode('utf-8')
                                    if isinstance(raw, (bytes, np.bytes_))
                                    else str(raw))
                anim = (np.asarray(data['positions_anim'])
                        if 'positions_anim' in data.files else None)
            return mname, order_idx, (
                frames, positions, wp_flats, wp_shape_str, anim, has_wp)
        except (OSError, EOFError, ValueError, KeyError,
                zipfile.BadZipFile) as exc:
            print(f"[Motion] Skipping unreadable cache {path}: {exc}")
            return mname, order_idx, None

    # ── 3. Parallel load — disk I/O scales with threadpool, GIL released
    #       during np.load's blocking read ──
    max_workers = min(16, max(4, (os.cpu_count() or 4) * 2))
    chunk_results = {}  # mname -> list-of (order_idx, payload)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for mname, order_idx, payload in ex.map(_load_one, work):
            if payload is None:
                continue
            chunk_results.setdefault(mname, []).append((order_idx, payload))

    # ── 4. Assemble per-muscle frame dict in original (mtime) order ──
    for mname, items in chunk_results.items():
        items.sort(key=lambda x: x[0])
        cache = {}
        for _, (frames, positions, wp_flats, wp_shape_str, anim, has_wp) in items:
            for i, f in enumerate(frames):
                entry = {'positions': positions[i]}
                if has_wp:
                    entry['waypoints_flat'] = wp_flats[i]
                    entry['waypoints_shape'] = wp_shape_str
                if anim is not None:
                    entry['positions_anim'] = anim[i]
                cache[int(f)] = entry
        v.motion_deform_cache[mname] = cache

    n_chunks = len(work)
    print(f"[Motion cache] {len(v.motion_deform_cache)} muscles, "
          f"{n_chunks} chunks loaded via {max_workers}-thread pool "
          f"in {_t.time() - _t_start:.2f}s")

def _flatten_waypoints(waypoints):
    """Flatten nested waypoints list into a single float32 array + shape JSON string.
    waypoints[stream][contour] = array(num_fibers, 3)
    Returns (flat_array, shape_json_str).
    """
    import json
    parts = []
    shape_desc = []
    for stream in waypoints:
        contour_shapes = []
        for contour_arr in stream:
            arr = np.asarray(contour_arr, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, 3)
            parts.append(arr.ravel())
            contour_shapes.append(int(arr.shape[0]))
        shape_desc.append(contour_shapes)
    flat = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
    return flat, json.dumps(shape_desc)

def _unflatten_waypoints(flat, shape_json):
    """Reconstruct nested waypoints list from flat array + shape JSON string."""
    import json
    if isinstance(shape_json, (bytes, np.bytes_)):
        shape_json = shape_json.decode('utf-8')
    elif isinstance(shape_json, np.ndarray):
        shape_json = str(shape_json)
    shape_desc = json.loads(shape_json)
    waypoints = []
    offset = 0
    for contour_shapes in shape_desc:
        stream = []
        for nf in contour_shapes:
            n = nf * 3
            stream.append(flat[offset:offset + n].reshape(nf, 3).copy())
            offset += n
        waypoints.append(stream)
    return waypoints


def _import_reverse_lbs_into_dart(v):
    """Build env.muscles from reverse_lbs_results/*.npz (solver output)
    instead of zygote_muscle.xml arc-length scheme.

    For each fiber: groups records by (stream, fiber_idx), sorted by level,
    reconstructs world-rest position from solver's (local, weight) data,
    builds bn_names = [origin, mid?, insertion] union, and calls
    addMuscleWeight per fiber.  DART will re-derive its own per-bone locals
    from world_rest — solver's exact locals are not used (would require a
    dartpy binding change), but the bone choice + weights ARE used.
    """
    import glob as _glob
    import pickle as _pickle
    from collections import defaultdict
    import dartpy as dart

    if v.env is None or v.env.skel is None:
        print("[ReverseLBS->DART] skel not ready")
        return
    skel = v.env.skel
    saved_pos = skel.getPositions().copy()
    skel.resetPositions()

    body_T_rest = {}
    def get_T(name):
        if name not in body_T_rest:
            b = skel.getBodyNode(name)
            if b is None:
                body_T_rest[name] = None
            else:
                T = np.asarray(b.getWorldTransform().matrix())
                body_T_rest[name] = (T[:3, :3].copy(), T[:3, 3].copy())
        return body_T_rest[name]

    v.env.muscles = dart.dynamics.Muscles(skel)
    v.env.zygote_activation_indices = [0]

    default_props = [1000.0, 1.2, 0.2, 0.0, -0.1, 0.0]
    files = sorted(_glob.glob('reverse_lbs_results/*.npz'))
    total_fibers = 0

    for path in files:
        muscle_name = os.path.basename(path).replace('.npz', '')
        d = np.load(path, allow_pickle=True)
        records = list(d['records'])
        fiber_groups = defaultdict(list)
        for r in records:
            fiber_groups[(int(r['stream']), int(r['fiber']))].append(r)

        n_fibers_added = 0
        for (s_idx, f_idx), recs in sorted(fiber_groups.items()):
            recs.sort(key=lambda r: int(r['level']))
            origin_body = recs[0]['origin_body']
            insertion_body = recs[0]['insertion_body']
            mid_body = recs[0].get('mid_body')
            uses_mid = bool(mid_body) and any(float(r.get('w_m', 0.0)) > 0.0 for r in recs)
            if uses_mid:
                bn_names = [origin_body, mid_body, insertion_body]
            else:
                bn_names = [origin_body, insertion_body]
            # Validate bones exist
            if any(get_T(b) is None for b in bn_names):
                continue

            # Build per-waypoint explicit (bones, locals, weights) using
            # solver's exact data — addMuscleAnchorExplicit bypasses DART's
            # bone-inverse derivation, so off-rest motion matches the
            # viewer's reverse-LBS toggle exactly.
            bn_names_per_wp = []
            local_pos_per_wp = []
            weights_per_wp = []
            for r in recs:
                w_o = float(r.get('w_o', 0.0))
                w_m = float(r.get('w_m', 0.0)) if uses_mid else 0.0
                w_i = float(r.get('w_i', 0.0))
                bnodes = []
                locals_ = []
                wts = []
                if w_o > 0:
                    bnodes.append(origin_body)
                    locals_.append(np.asarray(r['local_o'], dtype=np.float64))
                    wts.append(w_o)
                if uses_mid and w_m > 0:
                    bnodes.append(mid_body)
                    locals_.append(np.asarray(r['local_m'], dtype=np.float64))
                    wts.append(w_m)
                if w_i > 0:
                    bnodes.append(insertion_body)
                    locals_.append(np.asarray(r['local_i'], dtype=np.float64))
                    wts.append(w_i)
                bn_names_per_wp.append(bnodes)
                local_pos_per_wp.append(locals_)
                weights_per_wp.append(wts)
            try:
                v.env.muscles.addMuscleAnchorExplicit(
                    f'{muscle_name}_{s_idx}_{f_idx}',
                    default_props, False,
                    bn_names_per_wp, local_pos_per_wp, weights_per_wp)
                n_fibers_added += 1
            except Exception as e:
                print(f"[ReverseLBS->DART] {muscle_name} s{s_idx} f{f_idx}: {e}")
                continue
        v.env.zygote_activation_indices.append(n_fibers_added)
        total_fibers += n_fibers_added

    for i in range(1, len(v.env.zygote_activation_indices)):
        v.env.zygote_activation_indices[i] += v.env.zygote_activation_indices[i - 1]
    v.env.muscle_activation_levels = np.zeros(v.env.muscles.getNumMuscles())
    v.env.zygote_activation_levels = np.zeros(len(files))
    skel.setPositions(saved_pos)
    print(f"[ReverseLBS->DART] Imported {total_fibers} fibers from {len(files)} muscle .npz files. "
          f"Total DART muscles: {v.env.muscles.getNumMuscles()}")


REVERSE_LBS_XML = 'data/zygote_muscle_25.xml'


def _reverse_lbs_load(v):
    """Lazy-load per-muscle K-bone LBS data from REVERSE_LBS_XML.

    Each <Waypoint> carries `lbs_bones` (CSV), `lbs_locals` (semicolon-
    separated "x y z" per bone) and `lbs_weights` (space-separated). Builds
    vectorised arrays for fast per-frame reconstruction.

    Stores on viewer as `v._reverse_lbs_cache[muscle_name]` = dict with:
        bodies: list[str] unique body names referenced by this muscle
        wp_bones: list[np.ndarray int32]  (n_wp,) of per-waypoint bone-index arrays
        wp_locals: list[np.ndarray float32 (k, 3)]
        wp_weights: list[np.ndarray float32 (k,)]
        wp_structure: list[(stream_idx, level_idx, n_fibers)] for scatter.
    """
    import xml.etree.ElementTree as ET
    if getattr(v, '_reverse_lbs_cache', None) is not None:
        return
    if not os.path.exists(REVERSE_LBS_XML):
        print(f'[ReverseLBS] XML not found: {REVERSE_LBS_XML}')
        v._reverse_lbs_cache = {}
        return
    tree = ET.parse(REVERSE_LBS_XML)
    cache = {}
    for unit in tree.getroot().findall('Unit'):
        muscle_name = unit.attrib.get('name')
        if muscle_name not in v.zygote_muscle_meshes:
            continue
        bodies = []
        body_to_idx = {}

        def _bidx(name):
            if name not in body_to_idx:
                body_to_idx[name] = len(bodies)
                bodies.append(name)
            return body_to_idx[name]

        raw = []  # (s, l, f, bone_idx_arr, locals, weights)
        for fib in unit.findall('Fiber'):
            s_idx = int(fib.attrib.get('stream', 0))
            f_idx = int(fib.attrib.get('fiber', 0))
            for l_idx, wp in enumerate(fib.findall('Waypoint')):
                bones = [b for b in wp.attrib['lbs_bones'].split(',') if b]
                locs = [np.fromstring(s, sep=' ', dtype=np.float32)
                        for s in wp.attrib['lbs_locals'].split(';')]
                ws = np.fromstring(wp.attrib['lbs_weights'], sep=' ', dtype=np.float32)
                raw.append((s_idx, l_idx, f_idx,
                            np.array([_bidx(b) for b in bones], dtype=np.int32),
                            np.stack(locs, axis=0), ws))
        if not raw:
            continue
        # Scatter expects (stream, level, fiber) order — sort.
        raw.sort(key=lambda r: (r[0], r[1], r[2]))
        wp_layout = [(r[0], r[1], r[2]) for r in raw]
        # Pad per-waypoint K to max K, vectorise across all waypoints.
        K_max = max(r[3].shape[0] for r in raw)
        n = len(raw)
        bones_pad = np.zeros((n, K_max), dtype=np.int32)
        locals_pad = np.zeros((n, K_max, 3), dtype=np.float32)
        weights_pad = np.zeros((n, K_max), dtype=np.float32)
        for i, r in enumerate(raw):
            k = r[3].shape[0]
            bones_pad[i, :k] = r[3]
            locals_pad[i, :k] = r[4]
            weights_pad[i, :k] = r[5]
        from collections import defaultdict
        stream_levels = defaultdict(lambda: defaultdict(int))
        for s, l, f in wp_layout:
            stream_levels[s][l] = max(stream_levels[s][l], f + 1)
        wp_structure = []
        for s in sorted(stream_levels.keys()):
            for l in sorted(stream_levels[s].keys()):
                wp_structure.append((s, l, stream_levels[s][l]))
        cache[muscle_name] = dict(
            bodies=bodies, bones_pad=bones_pad, locals_pad=locals_pad,
            weights_pad=weights_pad, wp_structure=wp_structure,
        )
    v._reverse_lbs_cache = cache
    print(f'[ReverseLBS] Loaded {len(cache)} muscles from {REVERSE_LBS_XML}')


def _reverse_lbs_compute(v, mobj_name, cache_entry, body_T=None):
    """Compute world waypoint positions for one muscle from per-waypoint
    K-bone LBS data + current skeleton pose. Returns flat (n_wp, 3) array.
    `body_T` may be a precomputed {name: (R, t)} dict shared across muscles."""
    skel = v.env.skel
    body_world_R = np.empty((len(cache_entry['bodies']), 3, 3), dtype=np.float32)
    body_world_t = np.empty((len(cache_entry['bodies']), 3), dtype=np.float32)
    for k, bn in enumerate(cache_entry['bodies']):
        if body_T is not None and bn in body_T:
            R, t = body_T[bn]
        else:
            b = skel.getBodyNode(bn)
            if b is None:
                R, t = np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
            else:
                T = np.asarray(b.getWorldTransform().matrix())
                R, t = T[:3, :3].astype(np.float32), T[:3, 3].astype(np.float32)
        body_world_R[k] = R; body_world_t[k] = t

    bones = cache_entry['bones_pad']      # (n, K)
    locals_ = cache_entry['locals_pad']    # (n, K, 3)
    weights = cache_entry['weights_pad']   # (n, K)
    R = body_world_R[bones]                # (n, K, 3, 3)
    t = body_world_t[bones]                # (n, K, 3)
    contrib = np.einsum('nkij,nkj->nki', R, locals_) + t  # (n, K, 3)
    out = (weights[..., None] * contrib).sum(axis=1)      # (n, 3)
    return out.astype(np.float32)


def _reverse_lbs_apply(v):
    """Apply (or revert) reverse-LBS waypoint override on all known muscles."""
    if not getattr(v, 'reverse_lbs_enabled', False):
        # User toggled off — leave current waypoints; they will be refreshed
        # next time motion advances or NN inference runs.
        return
    if getattr(v, '_reverse_lbs_cache', None) is None:
        _reverse_lbs_load(v)
    cache = getattr(v, '_reverse_lbs_cache', None)
    if not cache:
        return
    # Precompute body transforms shared across muscles (skip duplicate DART calls).
    skel = v.env.skel
    all_bodies = set()
    for entry in cache.values():
        all_bodies.update(entry['bodies'])
    body_T = {}
    for bn in all_bodies:
        b = skel.getBodyNode(bn)
        if b is None:
            continue
        T = np.asarray(b.getWorldTransform().matrix())
        body_T[bn] = (T[:3, :3].astype(np.float32), T[:3, 3].astype(np.float32))
    for mname, entry in cache.items():
        mobj = v.zygote_muscle_meshes.get(mname)
        if mobj is None:
            continue
        try:
            wp_flat = _reverse_lbs_compute(v, mname, entry, body_T=body_T)
            # Scatter into mobj.waypoints using stream/level structure stored
            # alongside solved data (must match the cache layout).
            if getattr(mobj, 'waypoints', None):
                offset = 0
                for (s_idx, l_idx, n_f) in entry['wp_structure']:
                    if n_f == 0:
                        offset += n_f; continue
                    if s_idx < len(mobj.waypoints) and l_idx < len(mobj.waypoints[s_idx]):
                        mobj.waypoints[s_idx][l_idx] = wp_flat[offset:offset + n_f].copy()
                    offset += n_f
                mobj._fiber_draw_dirty = True
        except Exception as e:
            print(f'[ReverseLBS] {mname}: apply failed: {e}')


def _motion_anim_steps(v, frame):
    """Return number of convergence snapshots cached for `frame` (0 if none)."""
    if not getattr(v, 'motion_deform_cache', None):
        return 0
    k = 0
    for cache in list(v.motion_deform_cache.values()):
        entry = cache.get(int(frame)) if cache else None
        if entry is not None and 'positions_anim' in entry:
            k = max(k, int(entry['positions_anim'].shape[0]))
    return k


def _motion_apply_cached_deformation(v, frame):
    """Try to apply cached deformation for the given frame. Returns True if cache was used."""
    if not v.motion_deform_cache:
        return False
    # Compute translation offset for fixed axes: cached positions were baked with
    # the skeleton moving freely, so subtract the root displacement on fixed axes.
    fix_offset = np.zeros(3, dtype=np.float32)
    if hasattr(v, 'motion_root_translation') and v.motion_root_translation is not None:
        bvh_trans = v.motion_bvh.mocap_refs[frame, 3:6]  # x, y, z root translation
        rest_trans = v.motion_root_translation
        if v.motion_fix_x:
            fix_offset[0] = rest_trans[0] - bvh_trans[0]
        if v.motion_fix_y:
            fix_offset[1] = rest_trans[1] - bvh_trans[1]
        if v.motion_fix_z:
            fix_offset[2] = rest_trans[2] - bvh_trans[2]
    # When fix_rotation is on, use DART's actual body transforms to compute
    # the rotation+translation change from baked pose to current (fixed) pose.
    # This guarantees the cached positions match the skeleton exactly.
    fix_rot_mat = None
    fix_dest = None  # destination translation (replaces pivot + fix_offset)
    if v.motion_fix_rotation and hasattr(v, 'motion_root_rotation') and v.motion_root_rotation is not None:
        root_bn = v.env.skel.getJoint(0).getChildBodyNode()
        # Current (fixed) root body transform — skeleton already at fixed pose
        T_now = root_bn.getWorldTransform().matrix()
        R_now = T_now[:3, :3]
        t_now = T_now[:3, 3]
        # Baked root body transform — temporarily set to original frame pose
        saved_pos = v.env.skel.getPositions().copy()
        v.env.skel.setPositions(v.motion_bvh.mocap_refs[frame])
        T_baked = root_bn.getWorldTransform().matrix()
        R_baked = T_baked[:3, :3]
        t_baked = T_baked[:3, 3]
        v.env.skel.setPositions(saved_pos)
        fix_rot_mat = (R_now @ R_baked.T).astype(np.float32)
        pivot = t_baked.astype(np.float32)
        fix_dest = t_now.astype(np.float32)
    any_applied = False
    for mname, mobj in v.zygote_muscle_meshes.items():
        if mobj.tet_vertices is None:
            continue
        if mname in v.motion_deform_cache and frame in v.motion_deform_cache[mname]:
            cached = v.motion_deform_cache[mname][frame]
            # Optional iron-man-style scrub: pick a per-iter snapshot if user
            # has selected one and this muscle has anim data for the frame.
            anim_idx = getattr(v, 'motion_anim_idx', None)
            base_pos = cached['positions']
            if anim_idx is not None and 'positions_anim' in cached:
                anim = cached['positions_anim']
                if 0 <= anim_idx < anim.shape[0]:
                    base_pos = anim[anim_idx]
            if base_pos.shape[0] != mobj.tet_vertices.shape[0]:
                continue  # skip: bake vertex count doesn't match current tet mesh
            if fix_rot_mat is not None:
                cached_pos = (fix_rot_mat @ (base_pos - pivot).T).T + fix_dest
            else:
                cached_pos = base_pos + fix_offset
            # Skip soft_body.positions update during cached playback — not needed
            # for rendering, and the internal C state can cause segfaults.
            mobj.tet_vertices = cached_pos.astype(np.float32).copy()
            mobj._update_tet_draw_positions()
            # Restore cached waypoints
            if 'waypoints_flat' in cached and 'waypoints_shape' in cached:
                if hasattr(mobj, 'waypoints') and len(mobj.waypoints) > 0:
                    try:
                        wp = _unflatten_waypoints(
                            cached['waypoints_flat'], cached['waypoints_shape'])
                        if fix_offset.any() or fix_rot_mat is not None:
                            for stream in wp:
                                for fi in range(len(stream)):
                                    if stream[fi] is None or not hasattr(stream[fi], 'shape'):
                                        continue
                                    if fix_rot_mat is not None:
                                        stream[fi] = (fix_rot_mat @ (stream[fi] - pivot).T).T + fix_dest
                                    else:
                                        stream[fi] = stream[fi] + fix_offset
                        mobj.waypoints = wp
                        mobj._fiber_draw_dirty = True
                    except Exception:
                        pass
            any_applied = True
    return any_applied


def _motion_start_bake(v):
    """Start baking: reset to frame 0, init accumulator, set baking flag."""
    if v.motion_baking:
        return
    # Auto-find inter-muscle constraints if none exist
    if len(v.inter_muscle_constraints) == 0:
        find_inter_muscle_constraints(v)
    _motion_reset(v)
    # Reset soft body positions to rest state so bake starts clean
    # (reset may have loaded stale cached positions from a previous bake)
    for mname, mobj in v.zygote_muscle_meshes.items():
        if mobj.soft_body is not None:
            mobj.soft_body.positions = mobj.soft_body.rest_positions.copy()
            mobj.tet_vertices = mobj.soft_body.rest_positions.astype(np.float32).copy()
    # Clear cached backend and topology so system is rebuilt fresh
    v._unified_arap_backend = None
    v._unified_sim_cache = None
    v.motion_baking = True
    v.motion_bake_current = 0
    v.motion_current_frame = -1  # so first _motion_bake_step processes frame 0
    v.motion_is_playing = False
    # Init per-muscle accumulator for baked results
    v._bake_data = {}
    v._bake_start_time = time.time()
    v._bake_flush_count = 0  # number of partial flushes done
    v._bake_flush_interval = 1000  # flush to disk every N frames
    cache_dir = _motion_cache_dir(v)
    v._bake_cache_dir = cache_dir
    # Remove old chunk files from previous bakes
    for old_chunk in glob.glob(os.path.join(cache_dir, '*_chunk_*.npz')):
        os.remove(old_chunk)
    # Clean up legacy temp dir if it exists
    legacy_temp = os.path.join(cache_dir, '_bake_temp')
    if os.path.isdir(legacy_temp):
        import shutil
        shutil.rmtree(legacy_temp, ignore_errors=True)
    for mname, mobj in v.zygote_muscle_meshes.items():
        if mobj.soft_body is not None:
            v._bake_data[mname] = {}
    print(f"Started bake: frames 0-{v.motion_bake_end_frame}")


def _motion_bake_flush(v):
    """Flush accumulated bake data to chunk files on disk, then clear memory and run GC."""
    if not v._bake_data:
        return
    import gc
    cache_dir = v._bake_cache_dir
    for mname, frame_data in v._bake_data.items():
        if len(frame_data) == 0:
            continue
        sorted_frames = sorted(frame_data.keys())
        filepath = os.path.join(cache_dir, f'{mname}_chunk_{v._bake_flush_count:04d}.npz')
        np.savez(filepath,
            frames=np.array(sorted_frames, dtype=np.int32),
            positions=np.array([frame_data[f]['positions'] for f in sorted_frames], dtype=np.float32),
        )
        frame_data.clear()
    v._bake_flush_count += 1
    gc.collect()


def _motion_bake_step(v):
    """Called once per render loop while baking. Simulates ONE frame, captures results.
    When done, writes all accumulated results to disk."""
    if not v.motion_baking:
        return

    end_frame = v.motion_bake_end_frame
    frame = v.motion_current_frame + 1

    if frame > end_frame:
        _motion_bake_finish(v)
        return

    try:
        # Apply pose
        _motion_apply_pose(v, frame)

        # Disable waypoint updates and draw array rebuilds during baking
        saved_flags = {}
        for mname, mobj in v.zygote_muscle_meshes.items():
            if mobj.soft_body is not None:
                saved_flags[mname] = {
                    'wp': getattr(mobj, 'waypoints_from_tet_sim', True),
                    'baking': getattr(mobj, '_baking_mode', False),
                }
                mobj.waypoints_from_tet_sim = False
                mobj._baking_mode = True

        run_all_tet_sim_with_constraints(v,
            max_iterations=v.motion_settle_iters,
            tolerance=1e-4
        )

        # Capture positions only
        for mname in v._bake_data:
            mobj = v.zygote_muscle_meshes[mname]
            v._bake_data[mname][frame] = {
                'positions': mobj.soft_body.get_positions().astype(np.float32)
            }

        # Restore flags
        for mname, flags in saved_flags.items():
            mobj = v.zygote_muscle_meshes[mname]
            mobj.waypoints_from_tet_sim = flags['wp']
            mobj._baking_mode = False

    except Exception as e:
        print(f"ERROR in bake step frame {frame}: {e}")
        import traceback
        traceback.print_exc()
        # Still try to capture whatever positions we have
        for mname in v._bake_data:
            mobj = v.zygote_muscle_meshes[mname]
            if mobj.soft_body is not None:
                v._bake_data[mname][frame] = {
                    'positions': mobj.soft_body.get_positions().astype(np.float32)
                }

    v.motion_bake_current = frame

    # Periodically flush to disk to prevent memory explosion
    n_accumulated = sum(len(fd) for fd in v._bake_data.values())
    if n_accumulated >= v._bake_flush_interval * len(v._bake_data):
        _motion_bake_flush(v)


def _motion_bake_finish(v):
    """Write accumulated bake results to disk and reload cache.
    Bake only saves positions. Use 'Recompute Waypoints' to patch waypoints after."""
    # Flush any remaining in-memory data
    _motion_bake_flush(v)

    cache_dir = _motion_cache_dir(v)

    # Remove legacy single-file caches — chunk files replace them
    muscle_names = list(v._bake_data.keys())
    for mname in muscle_names:
        legacy_file = os.path.join(cache_dir, f'{mname}.npz')
        if os.path.exists(legacy_file):
            os.remove(legacy_file)

    v.motion_baking = False
    v._bake_data = {}
    _motion_load_cache(v)
    n_simulated = v.motion_bake_end_frame + 1  # frames 0..end
    elapsed = time.time() - getattr(v, '_bake_start_time', time.time())
    avg_frame = elapsed / max(n_simulated, 1)
    cache = getattr(v, '_unified_sim_cache', None)
    n_muscles = len(cache['muscle_names']) if cache else 0
    total_verts = cache['total_verts'] if cache else 0
    avg_muscle = avg_frame / max(n_muscles, 1)
    print(f"Bake complete: {n_simulated} frames in {elapsed:.1f}s — "
          f"{n_muscles} muscles, {total_verts * 3} params ({total_verts} verts × 3), "
          f"{avg_frame:.2f}s/frame, {avg_muscle:.2f}s/muscle. Recomputing waypoints...")
    # Automatically recompute waypoints after baking
    _motion_patch_waypoints(v)


def _motion_reset(v):
    """Reset to frame 0, reset skeleton to rest pose, reset tet meshes."""
    v.motion_current_frame = 0
    v.motion_is_playing = False
    v.motion_play_accumulator = 0.0
    if v.motion_bvh is not None:
        _motion_apply_pose(v, 0)
    else:
        # Reset skeleton to zero pose
        v.env.skel.setPositions(np.zeros(v.env.skel.getNumDofs()))
        if hasattr(v, '_skel_dofs'):
            v._skel_dofs = np.zeros(v.env.skel.getNumDofs())
    # Reset soft bodies — use NN, cached deformation, or reset from skeleton
    _motion_clear_heatmap(v)
    nn_applied = False
    if v.motion_use_nn and v.motion_nn_model is not None:
        nn_applied = _motion_apply_nn_deformation(v, 0)
        if nn_applied and v.motion_nn_error_heatmap:
            _motion_update_nn_error_heatmap(v, 0)
    if not nn_applied and not _motion_apply_cached_deformation(v, 0):
        for mname, mobj in v.zygote_muscle_meshes.items():
            if mobj.soft_body is not None:
                mobj._update_tet_positions_from_skeleton(v.env.skel)
                mobj._update_fixed_targets_from_skeleton(v.zygote_skeleton_meshes, v.env.skel)


def reset(v, reset_time=None):
    v.env.reset(reset_time)
    v.reward_buffer = [v.env.get_reward()]
    # Reset soft body simulations + force fiber redraw
    for name, obj in v.zygote_muscle_meshes.items():
        if obj.soft_body is not None:
            obj.reset_soft_body()
        obj._fiber_draw_dirty = True
    if getattr(v, 'reverse_lbs_enabled', False):
        _reverse_lbs_apply(v)


def zero_reset(v):
    v.env.zero_reset()
    v.reward_buffer = [v.env.get_reward()]
    # Reset soft body simulations + force fiber redraw
    for name, obj in v.zygote_muscle_meshes.items():
        if obj.soft_body is not None:
            obj.reset_soft_body()
        obj._fiber_draw_dirty = True
    if getattr(v, 'reverse_lbs_enabled', False):
        _reverse_lbs_apply(v)
