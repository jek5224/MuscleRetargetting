#!/usr/bin/env python3
"""GPU corotational ARAP bake for one stiff tetrahedral muscle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional
import trimesh
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.bvhparser import MyBVH
from tools import bake_emu
from tools.bake_surface_fast import build_bones, load_tet, surface_faces
from tools.bake_stiff_tet_pbd import (
    build_surface_samples, project_bones, project_volumes, signed_volumes,
    signed_volumes_cuda, unique_edges)
import test_emu


def build_embedded_visible_surface_samples(data):
    """Collision samples on the exact anatomical skin, mapped to tet DOFs.

    Artificial faces used only to close an open surface are deliberately not
    included. Samples cover every visible vertex, every visible edge midpoint,
    and every original face centroid. Each is represented by at most twelve
    barycentric simulation-cage supports.
    """
    render_indices = np.asarray(data["render_vertex_indices"], dtype=np.int32)
    render_weights = np.asarray(data["render_vertex_weights"], dtype=np.float64)
    faces = np.asarray(data["source_obj_faces"], dtype=np.int32)
    vertex_ids = np.unique(faces).astype(np.int32)
    edges = np.unique(np.sort(np.vstack((
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1),
        axis=0)
    count = len(vertex_ids) + len(edges) + len(faces)
    sample_indices = np.zeros((count, 12), dtype=np.int32)
    sample_weights = np.zeros((count, 12), dtype=np.float64)
    row = 0
    n = len(vertex_ids)
    sample_indices[row:row + n, :4] = render_indices[vertex_ids]
    sample_weights[row:row + n, :4] = render_weights[vertex_ids]
    row += n
    n = len(edges)
    sample_indices[row:row + n, :8] = render_indices[edges].reshape(n, 8)
    sample_weights[row:row + n, :8] = (
        0.5 * render_weights[edges]).reshape(n, 8)
    row += n
    n = len(faces)
    sample_indices[row:row + n, :] = render_indices[faces].reshape(n, 12)
    sample_weights[row:row + n, :] = (
        render_weights[faces] / 3.0).reshape(n, 12)
    print(
        f"Embedded visible collision surface: {len(vertex_ids)} vertices + "
        f"{len(edges)} edge midpoints + {len(faces)} face centroids = "
        f"{len(sample_indices)} samples (closure caps omitted)")
    return sample_indices, sample_weights


def build_direct_visible_surface_samples(data):
    """Contact samples on an exact tet-boundary OBJ surface.

    The open anatomical OBJ is tetrahedralized by adding closure caps. Those
    artificial caps are volume-construction faces, not collision geometry.
    Original OBJ indices are direct tet DOFs for exact-boundary assets.
    """
    faces = np.asarray(
        data.get("collision_faces", data["source_obj_faces"]),
        dtype=np.int32)
    vertex_ids = np.unique(faces).astype(np.int32)
    edges = np.unique(np.sort(np.vstack((
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1),
        axis=0)
    count = len(vertex_ids) + len(edges) + len(faces)
    sample_indices = np.zeros((count, 3), dtype=np.int32)
    sample_weights = np.zeros((count, 3), dtype=np.float64)
    row = 0
    n = len(vertex_ids)
    sample_indices[row:row + n, 0] = vertex_ids
    sample_weights[row:row + n, 0] = 1.0
    row += n
    n = len(edges)
    sample_indices[row:row + n, :2] = edges
    sample_weights[row:row + n, :2] = 0.5
    row += n
    n = len(faces)
    sample_indices[row:row + n, :] = faces
    sample_weights[row:row + n, :] = 1.0 / 3.0
    print(
        f"Direct visible collision surface: {len(vertex_ids)} vertices + "
        f"{len(edges)} edge midpoints + {len(faces)} face centroids = "
        f"{len(sample_indices)} samples (closure caps omitted)")
    return sample_indices, sample_weights


def prepare_full_cap_group(data, name, tet_path, skel, trees):
    group = test_emu.prepare_group_data(
        data, name, skel, trees, source_path=tet_path, attachment_rings=0)
    levels = np.asarray(data["vertex_contour_level"], dtype=np.int32)
    origin = np.where(levels == levels.min())[0].astype(np.int32)
    insertion = np.where(levels == levels.max())[0].astype(np.int32)
    group["origin_fixed"] = origin.tolist()
    group["insertion_fixed"] = insertion.tolist()
    group["fixed_vertices"] = np.r_[origin, insertion].tolist()
    u = test_emu._harmonic_axis_coordinate(
        group["vertices"], group["tetrahedra"], origin, insertion)
    u[origin] = 0.0
    u[insertion] = 1.0
    group["axis_coordinate"] = u
    group["lbs_bindings"] = test_emu._make_lbs_bindings(
        group["vertices"], u,
        skel.getBodyNode(group["origin_body"]),
        skel.getBodyNode(group["insertion_body"]))
    print(f"Full hard caps: {len(origin)} + {len(insertion)} vertices")
    return group


def prepare_explicit_attachment_group(data, name, tet_path, skel, trees):
    """Use attachment support sets authored directly in a tet asset.

    Embedded anatomical skins do not share indices with their simulation
    cage, so endpoint inference from contour levels or nearest regions is not
    valid. Their saved support sets are authoritative.
    """
    group = test_emu.prepare_group_data(
        data, name, skel, trees, source_path=tet_path, attachment_rings=0)
    origin = np.asarray(
        data["origin_attachment_vertices"], dtype=np.int32)
    insertion = np.asarray(
        data["insertion_attachment_vertices"], dtype=np.int32)
    group["origin_fixed"] = origin.tolist()
    group["insertion_fixed"] = insertion.tolist()
    group["fixed_vertices"] = np.unique(
        np.r_[origin, insertion]).astype(np.int32).tolist()
    group["origin_body"] = "L_Femur0"
    group["insertion_body"] = "L_Patella0"
    u = test_emu._harmonic_axis_coordinate(
        group["vertices"], group["tetrahedra"], origin, insertion)
    u[origin] = 0.0
    u[insertion] = 1.0
    group["axis_coordinate"] = u
    group["lbs_bindings"] = test_emu._make_lbs_bindings(
        group["vertices"], u,
        skel.getBodyNode(group["origin_body"]),
        skel.getBodyNode(group["insertion_body"]))
    print(
        f"Explicit embedded-skin hard supports: "
        f"{len(origin)} femur + {len(insertion)} patella vertices")
    return group


def prepare_full_origin_group(data, name, tet_path, skel, trees):
    """Use the complete origin cap with sparse patellar insertion anchors."""
    group = test_emu.prepare_group_data(
        data, name, skel, trees, source_path=tet_path, attachment_rings=0)
    levels = np.asarray(data["vertex_contour_level"], dtype=np.int32)
    origin = np.where(levels == levels.min())[0].astype(np.int32)
    insertion = np.asarray(group["insertion_fixed"], dtype=np.int32)
    group["origin_fixed"] = origin.tolist()
    group["fixed_vertices"] = np.unique(
        np.r_[origin, insertion]).astype(np.int32).tolist()
    u = test_emu._harmonic_axis_coordinate(
        group["vertices"], group["tetrahedra"], origin, insertion)
    u[origin] = 0.0
    u[insertion] = 1.0
    group["axis_coordinate"] = u
    group["lbs_bindings"] = test_emu._make_lbs_bindings(
        group["vertices"], u,
        skel.getBodyNode(group["origin_body"]),
        skel.getBodyNode(group["insertion_body"]))
    print(
        f"Full origin / sparse insertion: {len(origin)} + "
        f"{len(insertion)} vertices")
    return group


def prepare_compact_origin_group(data, name, tet_path, skel, trees):
    """Use the authored insertion cap and a local origin attachment patch.

    Endpoint contours are stored as ordered closed rings.  Fixing the entire
    origin ring makes the broad proximal cross-section rigid; the original VI
    bake instead attached the five vertices on either side of the ring seam.
    This retains the anatomical attachment while allowing the rest of the
    cross-section to deform.
    """
    group = test_emu.prepare_group_data(
        data, name, skel, trees, source_path=tet_path, attachment_rings=0)
    full_origin = np.asarray(group["origin_fixed"], dtype=np.int32)
    insertion = np.asarray(group["insertion_fixed"], dtype=np.int32)
    if len(full_origin) < 10:
        origin = full_origin
    else:
        origin = np.r_[full_origin[:5], full_origin[-5:]]
    group["origin_fixed"] = origin.tolist()
    group["fixed_vertices"] = np.unique(
        np.r_[origin, insertion]).astype(np.int32).tolist()
    u = test_emu._harmonic_axis_coordinate(
        group["vertices"], group["tetrahedra"], origin, insertion)
    u[origin] = 0.0
    u[insertion] = 1.0
    group["axis_coordinate"] = u
    group["lbs_bindings"] = test_emu._make_lbs_bindings(
        group["vertices"], u,
        skel.getBodyNode(group["origin_body"]),
        skel.getBodyNode(group["insertion_body"]))
    print(
        f"Compact origin / full insertion: {len(origin)} + "
        f"{len(insertion)} vertices")
    return group


class CUDAARAP:
    def __init__(self, rest, edges, fixed, device="cuda",
                 edge_multiplier=None):
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.rest = torch.as_tensor(rest, dtype=self.dtype, device=self.device)
        undirected = torch.as_tensor(
            edges, dtype=torch.long, device=self.device)
        self.i = torch.cat((undirected[:, 0], undirected[:, 1]))
        self.j = torch.cat((undirected[:, 1], undirected[:, 0]))
        self.rest_edge = self.rest[self.i] - self.rest[self.j]
        length = torch.linalg.vector_norm(self.rest_edge, dim=1)
        self.weight = 1.0 / length.clamp_min(1e-5)
        self.shape_multiplier = torch.ones_like(self.weight)
        if edge_multiplier is not None:
            multiplier = torch.as_tensor(
                edge_multiplier, dtype=self.dtype, device=self.device)
            directed_multiplier = torch.cat((multiplier, multiplier))
            self.weight *= directed_multiplier
            self.shape_multiplier = directed_multiplier
        n = len(rest)
        L = torch.zeros((n, n), dtype=self.dtype, device=self.device)
        L.index_put_((self.i, self.i), self.weight, accumulate=True)
        L.index_put_((self.i, self.j), -self.weight, accumulate=True)
        fixed_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        fixed_mask[torch.as_tensor(fixed, device=self.device)] = True
        self.fixed = torch.where(fixed_mask)[0]
        self.free = torch.where(~fixed_mask)[0]
        self.L = L
        self.Lfc = L[self.free][:, self.fixed]
        Lff = L[self.free][:, self.free]
        self.chol = torch.linalg.cholesky(
            Lff + 1e-10 * torch.eye(
                len(self.free), dtype=self.dtype, device=self.device))
        self._soft_factor = None
        self._attachment_factor = None

    def _factor_attachment_system(self, attachment, arap_weight):
        indices, weights, _targets, penalty = attachment
        key = (indices.data_ptr(), weights.data_ptr(), float(penalty),
               float(arap_weight))
        if (self._attachment_factor is not None
                and self._attachment_factor[0] == key):
            return self._attachment_factor[1:]
        rows = torch.arange(
            len(indices), device=self.device).repeat_interleave(
                indices.shape[1])
        C = torch.zeros(
            (len(indices), len(self.rest)), dtype=self.dtype,
            device=self.device)
        C.index_put_((rows, indices.reshape(-1)), weights.reshape(-1),
                     accumulate=True)
        A = arap_weight * self.L + penalty * (C.T @ C)
        Aff = A[self.free][:, self.free]
        Afc = A[self.free][:, self.fixed]
        chol = torch.linalg.cholesky(
            Aff + 1e-10 * torch.eye(
                len(self.free), dtype=self.dtype, device=self.device))
        self._attachment_factor = (key, chol, Afc, C)
        return chol, Afc, C

    def _factor_soft_system(self, soft_ids, soft_weights, arap_weight):
        """Cache the constant ARAP + transition-ring system matrix."""
        key = (
            tuple(soft_ids.detach().cpu().tolist()),
            tuple(soft_weights.detach().cpu().tolist()),
            float(arap_weight))
        if self._soft_factor is not None and self._soft_factor[0] == key:
            return self._soft_factor[1:]
        A = arap_weight * self.L.clone()
        A[soft_ids, soft_ids] += soft_weights
        if len(self.free) == len(self.rest) and len(self.fixed) == 0:
            Aff = A
            Afc = A[:, :0]
        else:
            Aff = A[self.free][:, self.free]
            Afc = A[self.free][:, self.fixed]
        chol = torch.linalg.cholesky(
            Aff + 1e-10 * torch.eye(
                len(self.free), dtype=self.dtype, device=self.device))
        self._soft_factor = (key, chol, Afc)
        return chol, Afc

    def step(self, x, fixed_targets, collision=None, collision_weight=25.0,
             arap_weight=1.0, soft=None, attachment=None):
        # Local step: best-fit rotation at every vertex.
        current_edge = x[self.i] - x[self.j]
        outer = current_edge[:, :, None] * self.rest_edge[:, None, :]
        covariance = torch.zeros(
            (len(x), 3, 3), dtype=self.dtype, device=self.device)
        covariance.index_add_(0, self.i, self.weight[:, None, None] * outer)
        U, _, Vh = torch.linalg.svd(covariance)
        R = U @ Vh
        negative = torch.linalg.det(R) < 0
        if torch.any(negative):
            U = U.clone()
            U[negative, :, -1] *= -1
            R = U @ Vh

        # Global step: prefactorized Laplacian solve.
        rotated = 0.5 * (R[self.i] + R[self.j]) @ self.rest_edge[:, :, None]
        rhs = torch.zeros_like(x)
        rhs.index_add_(
            0, self.i, self.weight[:, None] * rotated[:, :, 0])
        no_collision = collision is None or not len(collision[0])
        if no_collision and attachment is not None:
            indices, weights, attachment_targets, penalty = attachment
            chol, Afc, C = self._factor_attachment_system(
                attachment, arap_weight)
            weighted_rhs = (
                arap_weight * rhs
                + penalty * (C.T @ attachment_targets))
            if soft is not None and len(soft[0]):
                raise RuntimeError(
                    "Embedded contour attachments cannot currently be "
                    "combined with per-vertex soft constraints")
            free_rhs = weighted_rhs[self.free] - Afc @ fixed_targets
            x[self.free] = torch.cholesky_solve(free_rhs, chol)
        elif no_collision and (soft is None or not len(soft[0])):
            free_rhs = rhs[self.free] - self.Lfc @ fixed_targets
            x[self.free] = torch.cholesky_solve(free_rhs, self.chol)
        elif no_collision:
            soft_ids, soft_targets, soft_weights = soft
            chol, Afc = self._factor_soft_system(
                soft_ids, soft_weights, arap_weight)
            weighted_rhs = arap_weight * rhs
            weighted_rhs[soft_ids] += (
                soft_weights[:, None] * soft_targets)
            free_rhs = weighted_rhs[self.free] - Afc @ fixed_targets
            x[self.free] = torch.cholesky_solve(free_rhs, chol)
        else:
            A = arap_weight * self.L
            rhs = arap_weight * rhs
            if collision is not None and len(collision[0]):
                sample_vertices, sample_weights, sample_targets = collision
                C = torch.zeros(
                    (len(sample_vertices), len(x)),
                    dtype=self.dtype, device=self.device)
                rows = torch.arange(
                    len(sample_vertices),
                    device=self.device).repeat_interleave(3)
                C.index_put_(
                    (rows, sample_vertices.reshape(-1)),
                    sample_weights.reshape(-1), accumulate=True)
                A = A + collision_weight * (C.T @ C)
                rhs = rhs + collision_weight * (C.T @ sample_targets)
            if soft is not None and len(soft[0]):
                soft_ids, soft_targets, soft_weights = soft
                A = A.clone()
                rhs = rhs.clone()
                A[soft_ids, soft_ids] += soft_weights
                rhs[soft_ids] += soft_weights[:, None] * soft_targets
            Aff = A[self.free][:, self.free]
            Afc = A[self.free][:, self.fixed]
            x[self.free] = torch.linalg.solve(
                Aff + 1e-10 * torch.eye(
                    len(self.free), dtype=self.dtype, device=self.device),
                rhs[self.free] - Afc @ fixed_targets)
        x[self.fixed] = fixed_targets
        return x


def collision_constraints(x, sample_vertices, sample_weights, bones,
                          fixed_mask, margin):
    """Build barycentric positional constraints for penetrating samples."""
    active_support = sample_weights > 0.0
    supported = np.any(
        fixed_mask[sample_vertices] & active_support, axis=1)
    points = np.einsum(
        "nij,ni->nj", x[sample_vertices], sample_weights)
    ids_out, targets_out = [], []
    claimed = np.zeros(len(points), dtype=bool)
    for bone in bones:
        ids = np.where(
            np.all((points >= bone.bounds[0] - margin)
                   & (points <= bone.bounds[1] + margin), axis=1)
            & ~supported & ~claimed)[0]
        if not len(ids):
            continue
        try:
            ids = ids[bone.contains(points[ids])]
        except Exception:
            continue
        if not len(ids):
            continue
        nearest, _, face_ids = trimesh.proximity.closest_point(
            bone, points[ids])
        direction = nearest - points[ids]
        length = np.linalg.norm(direction, axis=1)
        valid = length > 1e-12
        direction[valid] /= length[valid, None]
        direction[~valid] = bone.face_normals[face_ids[~valid]]
        ids_out.append(ids)
        targets_out.append(nearest + margin * direction)
        claimed[ids] = True
    if not ids_out:
        return None
    ids = np.concatenate(ids_out)
    return (
        sample_vertices[ids], sample_weights[ids],
        np.concatenate(targets_out))


class BoneSDF:
    def __init__(self, path):
        data = np.load(path)
        self.sdf = np.asarray(data["sdf"], dtype=np.float64)
        self.grad = np.asarray(data["gradients"], dtype=np.float64)
        self.origin = np.asarray(data["origin"], dtype=np.float64)
        self.pitch = float(data["pitch"])
        self.body = str(data["body"])
        self.rest_distance = None
        self._torch_loss_cache = {}

    def bind_rest_clearance(self, rest_x, sample_vertices, sample_weights,
                            skel, prevent_new_only=False, margin=0.0):
        """Remember each material sample's source clearance from the bone.

        The source VI mesh intentionally overlaps the femur in places.  A
        universal zero-SDF target therefore destroys its cross-section.
        Contact is instead a one-sided constraint: flexion may not make a
        sample more embedded than it was in the authored rest pose.
        """
        points = np.einsum(
            "nij,ni->nj", rest_x[sample_vertices], sample_weights)
        wt = skel.getBodyNode(self.body).getWorldTransform()
        local = (
            wt.rotation().T @ (points - wt.translation()).T).T
        coords = ((local - self.origin) / self.pitch).T
        self.rest_distance = map_coordinates(
            self.sdf, coords, order=1, mode="constant", cval=1.0)
        # Samples outside the finite SDF grid receive the 1 m sentinel from
        # map_coordinates; treating that as physical clearance pulls the
        # muscle explosively toward an unreachable target.
        valid = np.all(
            (coords >= 0.0)
            & (coords <= (np.asarray(self.sdf.shape) - 1)[:, None]),
            axis=0)
        self.rest_distance[~valid] = -np.inf
        if prevent_new_only:
            authored_inside = self.rest_distance < 0.0
            self.rest_distance[authored_inside] = -np.inf
            self.rest_distance[~authored_inside & valid] = margin
        self._torch_loss_cache.clear()

    def constraints(self, x, sample_vertices, sample_weights, fixed_mask,
                    skel, margin, max_step=None, rest_tolerance=0.0):
        # Attachment caps intentionally meet/enter their bones.  A primitive
        # touching any hard cap vertex cannot satisfy an exterior constraint
        # without peeling the first free surface ring away from that cap.
        supported = np.any(
            fixed_mask[sample_vertices] & (sample_weights > 0.0), axis=1)
        points = np.einsum("nij,ni->nj", x[sample_vertices], sample_weights)
        wt = skel.getBodyNode(self.body).getWorldTransform()
        R, t = wt.rotation(), wt.translation()
        local = (R.T @ (points - t).T).T
        coords = ((local - self.origin) / self.pitch).T
        distance = map_coordinates(
            self.sdf, coords, order=1, mode="constant", cval=1.0)
        if self.rest_distance is None:
            threshold = np.full_like(distance, margin)
        else:
            threshold = self.rest_distance - rest_tolerance
        ids = np.where((distance < threshold) & ~supported)[0]
        if not len(ids):
            return None
        gradient = np.column_stack([
            map_coordinates(self.grad[..., axis], coords[:, ids], order=1,
                            mode="constant", cval=0.0)
            for axis in range(3)])
        norm = np.linalg.norm(gradient, axis=1)
        valid = norm > 1e-8
        ids = ids[valid]
        gradient = gradient[valid] / norm[valid, None]
        if not len(ids):
            return None
        correction = (
            threshold[ids] - distance[ids])[:, None] * gradient
        if max_step is not None and max_step > 0.0:
            correction_length = np.linalg.norm(correction, axis=1)
            limited = correction_length > max_step
            correction[limited] *= (
                max_step / correction_length[limited])[:, None]
        target_local = local[ids] + correction
        targets = (R @ target_local.T).T + t
        return sample_vertices[ids], sample_weights[ids], targets

    def torch_penetration_loss(
            self, x, sample_vertices, sample_weights, fixed_mask, skel,
            tolerance):
        """Evaluate the one-sided SDF barrier differentiably on the GPU."""
        device, dtype = x.device, x.dtype
        key = (
            str(device), dtype, sample_vertices.ctypes.data,
            sample_weights.ctypes.data, fixed_mask.ctypes.data,
            float(tolerance))
        cached = self._torch_loss_cache.get(key)
        if cached is None:
            ids = torch.as_tensor(
                sample_vertices, dtype=torch.long, device=device)
            weights = torch.as_tensor(
                sample_weights, dtype=dtype, device=device)
            origin = torch.as_tensor(
                self.origin, dtype=dtype, device=device)
            shape = torch.as_tensor(
                self.sdf.shape, dtype=dtype, device=device)
            field = torch.as_tensor(
                self.sdf, dtype=dtype, device=device)[None, None]
            threshold = torch.as_tensor(
                self.rest_distance, dtype=dtype, device=device) - tolerance
            supported = np.any(
                fixed_mask[sample_vertices] & (sample_weights > 0.0), axis=1)
            active = torch.as_tensor(
                np.isfinite(self.rest_distance) & ~supported,
                dtype=torch.bool, device=device)
            # The authored-inside and cap-touching exclusions are immutable
            # for a bake. Compact them once rather than interpolating the SDF
            # for samples whose loss is guaranteed to be zero every iteration.
            ids = ids[active]
            weights = weights[active]
            threshold = threshold[active]
            cached = (ids, weights, origin, shape, field, threshold)
            self._torch_loss_cache[key] = cached
        ids, weights, origin, shape, field, threshold = cached
        if not len(ids):
            return x.new_zeros(())
        points = torch.sum(x[ids] * weights[:, :, None], dim=1)
        wt = skel.getBodyNode(self.body).getWorldTransform()
        rotation = torch.as_tensor(
            wt.rotation(), dtype=dtype, device=device)
        translation = torch.as_tensor(
            wt.translation(), dtype=dtype, device=device)
        local = (points - translation) @ rotation
        coords = (local - origin) / self.pitch
        normalized = 2.0 * coords / (shape - 1.0) - 1.0
        # grid_sample's coordinate order is W,H,D, while the stored field
        # axes are x,y,z => pass z,y,x.
        grid = normalized[:, [2, 1, 0]].reshape(1, -1, 1, 1, 3)
        distance = torch_functional.grid_sample(
            field, grid, mode="bilinear", padding_mode="border",
            align_corners=True).reshape(-1)
        penetration = torch.relu(threshold - distance)
        return torch.mean((penetration / self.pitch) ** 2)


def project_collision_constraints(x, constraints, fixed_mask, max_step):
    """Apply one bounded Jacobi projection of barycentric contact samples.

    Collision is an inequality, so it must finish with a projection rather
    than another unconstrained ARAP solve.  Distributing each sample's
    correction by its barycentric weights (and normalizing by squared weights)
    moves the represented point to its target when constraints do not overlap.
    Averaging overlapping proposals and bounding each vertex step prevents the
    high-weight least-squares "vertex fly-out" failure.
    """
    if constraints is None:
        return x, 0, 0.0
    sample_vertices, sample_weights, sample_targets = constraints
    points = np.einsum(
        "nij,ni->nj", x[sample_vertices], sample_weights)
    sample_delta = sample_targets - points
    movable = (~fixed_mask[sample_vertices]) & (sample_weights > 0.0)
    effective_weights = sample_weights * movable
    denominator = np.sum(effective_weights * effective_weights, axis=1)
    valid = denominator > 1e-18
    if not np.any(valid):
        return x, 0, 0.0

    vertex_delta = np.zeros_like(x)
    vertex_count = np.zeros(len(x), dtype=np.float64)
    for corner in range(sample_vertices.shape[1]):
        active = valid & movable[:, corner]
        if not np.any(active):
            continue
        ids = sample_vertices[active, corner]
        scale = (
            effective_weights[active, corner] / denominator[active])
        np.add.at(vertex_delta, ids, scale[:, None] * sample_delta[active])
        np.add.at(vertex_count, ids, 1.0)
    touched = vertex_count > 0.0
    vertex_delta[touched] /= vertex_count[touched, None]
    length = np.linalg.norm(vertex_delta, axis=1)
    if max_step is not None and max_step > 0.0:
        limited = touched & (length > max_step)
        vertex_delta[limited] *= (
            max_step / length[limited])[:, None]
    x[touched] += vertex_delta[touched]
    return x, int(np.sum(valid)), float(length[touched].max(initial=0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh", required=True, type=Path)
    ap.add_argument("--tet", default="tet/L_Vastus_Intermedius_tet.npz",
                    type=Path)
    ap.add_argument("--name", default="L_Vastus_Intermedius")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument(
        "--frames",
        help="Comma-separated BVH frame indices to solve. Omit for all "
             "frames; useful for fast selective rebakes of independent poses.")
    ap.add_argument("--iterations", type=int, default=120)
    ap.add_argument("--collision-every", type=int, default=5)
    ap.add_argument("--collision-margin", type=float, default=0.0015)
    ap.add_argument("--max-collision-step", type=float, default=0.001)
    ap.add_argument("--final-collision-passes", type=int, default=20)
    ap.add_argument("--quality-cycles", type=int, default=0,
                    help="Alternate bounded volume repair and SDF projection")
    ap.add_argument("--quality-volume-sweeps", type=int, default=5)
    ap.add_argument("--quality-volume-stiffness", type=float, default=0.15)
    ap.add_argument("--quality-max-step", type=float, default=0.00025)
    ap.add_argument("--coupled-quality-outer", type=int, default=0)
    ap.add_argument("--coupled-quality-inner", type=int, default=40)
    ap.add_argument("--coupled-quality-lr", type=float, default=0.00005)
    ap.add_argument("--coupled-volume-weight", type=float, default=2.0)
    ap.add_argument("--coupled-collision-weight", type=float, default=20.0)
    ap.add_argument(
        "--adaptive-coupled", action="store_true",
        help="Stop coupled refinement early once contact and signed-volume "
             "quality are already within tolerance.")
    ap.add_argument(
        "--adaptive-contact-loss", type=float, default=0.05,
        help="Early-stop threshold for normalized mean SDF penetration.")
    ap.add_argument(
        "--inversion-safe-line-search", action="store_true",
        help="Once all signed tet ratios are positive, backtrack every "
             "optimizer step that would cross the zero-volume boundary.")
    ap.add_argument(
        "--require-positive-tets", action="store_true",
        help="Abort instead of writing any frame containing an inverted tet.")
    ap.add_argument(
        "--pre-untangle-steps", type=int, default=0,
        help="Before coupled contact, restore a positive signed-volume state "
             "with current-frame anchors locked.")
    ap.add_argument("--collision-weight", type=float, default=25.0)
    ap.add_argument(
        "--vertex-contact-only", action="store_true",
        help="Use unique surface vertices for contact instead of redundant "
             "edge/face samples.")
    ap.add_argument(
        "--attachment-collision-exclusion-rings", type=int, default=0,
        help="Additional surface rings around hard caps that are exempt from "
             "bone contact but remain mechanically free unless hard-fixed.")
    ap.add_argument(
        "--projective-contact", action="store_true",
        help="Apply bounded SDF contact projections between prefactorized "
             "ARAP steps instead of rebuilding a dense contact system.")
    ap.add_argument(
        "--contact-in-coupled-only", action="store_true",
        help="Keep contact out of the ARAP iterations and apply it only in "
             "the signed-volume coupled refinement.")
    ap.add_argument(
        "--differentiable-sdf-contact", action="store_true",
        help="Evaluate trilinear SDF contact in every coupled optimizer step "
             "instead of using frozen closest-point targets.")
    ap.add_argument("--arap-weight", type=float, default=1.0,
                    help="Shape stiffness relative to collision constraints")
    ap.add_argument("--femur-sdf", type=Path)
    ap.add_argument(
        "--rest-sdf-clearance", action="store_true",
        help="Use each surface sample's authored rest SDF as a one-sided "
             "minimum clearance instead of forcing an impossible zero-SDF "
             "shape.")
    ap.add_argument("--rest-sdf-tolerance", type=float, default=0.001)
    ap.add_argument(
        "--prevent-new-penetration", action="store_true",
        help="Apply zero-SDF contact only to samples outside the femur in "
             "the authored pose; preserve pre-existing anatomical overlap.")
    ap.add_argument(
        "--anchor-caps", action="store_true",
        help="Use anatomical endpoint anchors only; required for refined "
             "meshes whose added vertices have interpolated contour metadata.")
    ap.add_argument(
        "--full-origin-cap", action="store_true",
        help="Hard-fix the full origin cap while retaining sparse anatomical "
             "insertion anchors.")
    ap.add_argument(
        "--compact-origin-cap", action="store_true",
        help="Hard-fix a local ten-vertex patch of the ordered origin cap and "
             "the complete insertion cap.")
    ap.add_argument("--insertion-soft-rings", type=int, default=0)
    ap.add_argument("--insertion-soft-weight", type=float, default=500.0)
    ap.add_argument(
        "--origin-cap-soft-weight", type=float, default=0.0,
        help="Softly follow the femur guide with origin-cap vertices that are "
             "not part of the compact hard attachment patch.")
    ap.add_argument(
        "--origin-cap-soft-fade-degrees", type=float, default=0.0,
        help="Fade the origin-cap spring from full strength at the rest knee "
             "angle to zero at this relative rotation; zero disables fading.")
    ap.add_argument(
        "--near-rest-compliant-origin", action="store_true",
        help="While the pose-gated origin spring is active, release the "
             "compact hard-origin patch and constrain the complete authored "
             "origin cap softly. The insertion remains hard.")
    ap.add_argument(
        "--always-compliant-origin", action="store_true",
        help="Never restore hard origin vertices after the near-rest fade. "
             "Keep the complete origin cap soft in every pose.")
    ap.add_argument(
        "--origin-flexed-soft-weight", type=float, default=0.0,
        help="Soft origin-reference weight after the flexion fade. With "
             "--always-compliant-origin, interpolate smoothly from "
             "--origin-cap-soft-weight near rest to this value.")
    ap.add_argument(
        "--origin-hard-transition-weight", type=float, default=0.0,
        help="For compliant-origin transition poses, continuously stiffen "
             "the compact hard-origin patch toward this soft weight as the "
             "near-rest compliance fades.")
    ap.add_argument(
        "--origin-compact-base-weight", type=float, default=0.0,
        help="Optional soft weight for the compact ten-vertex origin patch "
             "at zero flexion; the remaining authored cap uses "
             "--origin-cap-soft-weight.")
    ap.add_argument("--insertion-cohesion-rings", type=int, default=0)
    ap.add_argument("--insertion-cohesion-weight", type=float, default=8.0)
    ap.add_argument(
        "--origin-cap-cohesion-weight", type=float, default=1.0,
        help="ARAP edge multiplier for edges wholly inside the authored "
             "origin cap. Preserves cap shape without pinning its position.")
    ap.add_argument(
        "--origin-cap-complete-graph", action="store_true",
        help="Add all authored origin-cap vertex pairs to the ARAP shape "
             "reference instead of strengthening only existing tet edges.")
    ap.add_argument(
        "--embedded-contour-attachment-weight", type=float, default=0.0,
        help="Constrain embedded anatomical origin/insertion contour points "
             "barycentrically instead of hard-fixing their tet supports.")
    ap.add_argument(
        "--embedded-attachment-ramp-steps", type=int, default=0,
        help="Initialize the embedded muscle with the femur transform and "
             "ramp its insertion contour to the posed patella over this "
             "many ARAP iterations.")
    ap.add_argument(
        "--embedded-insertion-arc-ramp", action="store_true",
        help="Move the insertion around the current knee joint center on a "
             "spherical arc instead of linearly through the femur.")
    ap.add_argument(
        "--initialization-cache", type=Path,
        help="Optional collision-free cache used only to transfer an initial "
             "deformation field onto this tet mesh.")
    ap.add_argument(
        "--initialization-tet", type=Path,
        help="Rest tet corresponding to --initialization-cache.")
    ap.add_argument(
        "--independent-frames", action="store_true",
        help="Initialize every pose from its rigid guide (quasistatic sweep, "
             "with no history dependence).")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    skel, bvh_info, _ = bake_emu.load_skeleton()
    data = load_tet(args.tet)
    attachment_modes = sum((
        bool(args.anchor_caps), bool(args.full_origin_cap),
        bool(args.compact_origin_cap)))
    if attachment_modes > 1:
        ap.error("attachment cap modes are mutually exclusive")
    if args.compact_origin_cap:
        group = prepare_compact_origin_group(
            data, args.name, args.tet, skel, test_emu._load_bone_trees())
    elif args.full_origin_cap:
        group = prepare_full_origin_group(
            data, args.name, args.tet, skel, test_emu._load_bone_trees())
    elif args.anchor_caps:
        if ("origin_attachment_vertices" in data
                and "insertion_attachment_vertices" in data):
            group = prepare_explicit_attachment_group(
                data, args.name, args.tet, skel,
                test_emu._load_bone_trees())
        else:
            group = test_emu.prepare_group_data(
                data, args.name, skel, test_emu._load_bone_trees(),
                source_path=args.tet, attachment_rings=0)
            print(
                f"Anatomical hard anchors: {len(group['origin_fixed'])} + "
                f"{len(group['insertion_fixed'])} vertices")
    else:
        group = prepare_full_cap_group(
            data, args.name, args.tet, skel, test_emu._load_bone_trees())
    rest = np.asarray(group["vertices"], dtype=np.float64)
    tets = np.asarray(group["tetrahedra"], dtype=np.int32)
    initialization_by_frame = {}
    if args.initialization_cache is not None:
        if args.initialization_tet is None:
            ap.error("--initialization-cache requires --initialization-tet")
        source_data = load_tet(args.initialization_tet)
        source_rest = np.asarray(
            source_data["vertices"], dtype=np.float64)
        cache_path = args.initialization_cache
        if cache_path.is_dir():
            candidates = sorted(cache_path.glob("*_chunk_*.npz"))
            if not candidates:
                ap.error("initialization cache contains no chunks")
            cache_path = candidates[0]
        cache_data = np.load(cache_path)
        source_frames = np.asarray(cache_data["frames"], dtype=np.int32)
        source_positions = np.asarray(
            cache_data["positions"], dtype=np.float64)
        neighbor_count = min(96, len(source_rest))
        distance, neighbor = cKDTree(source_rest).query(
            rest, k=neighbor_count)
        if neighbor.ndim == 1:
            neighbor = neighbor[:, None]
            distance = distance[:, None]
        scale = np.maximum(distance[:, -1:], 1e-8)
        transfer_weight = np.exp(-6.0 * (distance / scale) ** 2)
        transfer_weight /= np.maximum(
            transfer_weight.sum(axis=1, keepdims=True), 1e-12)
        for row, frame_id in enumerate(source_frames):
            displacement = source_positions[row] - source_rest
            initialization_by_frame[int(frame_id)] = (
                rest + np.einsum(
                    "nk,nkj->nj", transfer_weight,
                    displacement[neighbor]))
        print(
            f"Transferred initialization available for "
            f"{len(initialization_by_frame)} frame(s) from {cache_path}")
    embedded_attachment_spec = None
    if args.embedded_contour_attachment_weight > 0.0:
        contour_path = data.get("attachment_contours_file")
        if not contour_path or not Path(str(contour_path)).exists():
            ap.error("embedded contour attachment metadata is missing")
        with open(str(contour_path)) as stream:
            contour_data = json.load(stream)
        origin_render = np.asarray(
            contour_data["origin"]["vertex_indices"], dtype=np.int32)
        insertion_render = np.asarray(
            contour_data["insertion"]["vertex_indices"], dtype=np.int32)
        if (data.get("tetrahedralization_method") ==
                "tetgen_exact_surface"):
            contour_vertices = np.r_[origin_render, insertion_render]
            render_indices = contour_vertices[:, None]
            render_weights = np.ones((len(contour_vertices), 1))
            render_rest = rest[contour_vertices]
            origin_rows = np.arange(len(origin_render), dtype=np.int32)
            insertion_rows = np.arange(
                len(origin_render), len(contour_vertices), dtype=np.int32)
        else:
            render_indices_all = np.asarray(
                data["render_vertex_indices"], dtype=np.int32)
            render_weights_all = np.asarray(
                data["render_vertex_weights"], dtype=np.float64)
            render_rest_all = np.asarray(
                data["render_vertices_rest"], dtype=np.float64)
            render_indices = np.vstack((
                render_indices_all[origin_render],
                render_indices_all[insertion_render]))
            render_weights = np.vstack((
                render_weights_all[origin_render],
                render_weights_all[insertion_render]))
            render_rest = np.vstack((
                render_rest_all[origin_render],
                render_rest_all[insertion_render]))
            origin_rows = np.arange(len(origin_render), dtype=np.int32)
            insertion_rows = np.arange(
                len(origin_render), len(origin_render) + len(insertion_render),
                dtype=np.int32)
        embedded_attachment_spec = {
            "indices": render_indices,
            "weights": render_weights,
            "rest_points": render_rest,
            "origin_count": len(origin_render),
            "direct_identity": bool(
                data.get("tetrahedralization_method") ==
                "tetgen_exact_surface"),
        }
        # Barycentric equations anchor the null space; support vertices must
        # remain free so the attachment does not become a thick rigid region.
        group["fixed_vertices"] = []
        print(
            f"Embedded hard-reference contours: {len(origin_render)} femur + "
            f"{len(insertion_render)} patella points; tet supports remain free")
    fixed = np.asarray(group["fixed_vertices"], dtype=np.int32)
    fixed_mask = np.zeros(len(rest), dtype=bool)
    fixed_mask[fixed] = True
    contact_exclusion_mask = fixed_mask.copy()
    if args.attachment_collision_exclusion_rings > 0:
        excluded = test_emu._expand_surface_vertex_rings(
            fixed.tolist(), group["surface_faces"],
            rings=args.attachment_collision_exclusion_rings)
        contact_exclusion_mask[np.asarray(excluded, dtype=np.int32)] = True
        print(
            f"Bone-contact attachment exclusion: "
            f"{int(contact_exclusion_mask.sum())} surface vertices")
    soft_weight_by_id = {}
    origin_soft_ids = np.empty(0, dtype=np.int32)
    if args.origin_cap_soft_weight > 0.0:
        levels = np.asarray(data["vertex_contour_level"], dtype=np.int32)
        authored_anchors = np.asarray(
            data.get("anchor_vertices", []), dtype=np.int32)
        anchor_levels = levels[authored_anchors]
        full_origin = authored_anchors[
            anchor_levels == anchor_levels.min()]
        free_origin = np.setdiff1d(full_origin, fixed)
        origin_soft_ids = (
            full_origin if args.near_rest_compliant_origin else free_origin)
        for vertex in origin_soft_ids:
            soft_weight_by_id[int(vertex)] = args.origin_cap_soft_weight
        print(
            f"Origin transition: {len(origin_soft_ids)} soft cap vertices, "
            f"weight={args.origin_cap_soft_weight:g}")
    if args.insertion_soft_rings > 0:
        insertion = set(map(int, group["insertion_fixed"]))
        previous_ring = insertion
        ids_out, weights_out = [], []
        for ring in range(1, args.insertion_soft_rings + 1):
            expanded = set(test_emu._expand_surface_vertex_rings(
                insertion, group["surface_faces"], rings=ring))
            current_ring = expanded - previous_ring - set(map(int, fixed))
            weight = args.insertion_soft_weight / (4.0 ** (ring - 1))
            ids_out.extend(sorted(current_ring))
            weights_out.extend([weight] * len(current_ring))
            previous_ring = expanded
        for vertex, weight in zip(ids_out, weights_out):
            soft_weight_by_id[int(vertex)] = max(
                soft_weight_by_id.get(int(vertex), 0.0), float(weight))
        print(
            f"Insertion transition: {len(ids_out)} soft vertices over "
            f"{args.insertion_soft_rings} ring(s)")
    soft_ids = np.asarray(
        sorted(soft_weight_by_id), dtype=np.int32)
    soft_weights = np.asarray(
        [soft_weight_by_id[int(vertex)] for vertex in soft_ids],
        dtype=np.float64)
    if "collision_faces" in data or (
            "source_obj_faces" in data
            and data.get("tetrahedralization_method") ==
            "tetgen_exact_surface"):
        samples_i, samples_w = build_direct_visible_surface_samples(data)
    elif embedded_attachment_spec is not None:
        samples_i, samples_w = build_embedded_visible_surface_samples(data)
    else:
        samples_i, samples_w = build_surface_samples(surface_faces(tets))
    if args.vertex_contact_only:
        surface_ids = np.unique(surface_faces(tets)).astype(np.int32)
        samples_i = np.repeat(surface_ids[:, None], 3, axis=1)
        samples_w = np.zeros((len(surface_ids), 3), dtype=np.float64)
        samples_w[:, 0] = 1.0
    femur_sdf = BoneSDF(args.femur_sdf) if args.femur_sdf else None
    if femur_sdf and args.rest_sdf_clearance:
        femur_sdf.bind_rest_clearance(
            rest, samples_i, samples_w, skel,
            prevent_new_only=args.prevent_new_penetration,
            margin=args.collision_margin)
    edges = unique_edges(tets)
    edge_multiplier = None
    if args.insertion_cohesion_rings > 0:
        insertion = set(map(int, group["insertion_fixed"]))
        vertex_multiplier = np.ones(len(rest), dtype=np.float64)
        previous = insertion
        vertex_multiplier[list(insertion)] = args.insertion_cohesion_weight
        for ring in range(1, args.insertion_cohesion_rings + 1):
            expanded = set(test_emu._expand_surface_vertex_rings(
                insertion, group["surface_faces"], rings=ring))
            current = expanded - previous
            strength = 1.0 + (
                (args.insertion_cohesion_weight - 1.0)
                * (args.insertion_cohesion_rings + 1 - ring)
                / (args.insertion_cohesion_rings + 1))
            vertex_multiplier[list(current)] = np.maximum(
                vertex_multiplier[list(current)], strength)
            previous = expanded
        # Strengthen an edge only when both endpoints belong to the graded
        # tendon zone; this avoids moving the peel line to its outer boundary.
        edge_multiplier = np.minimum(
            vertex_multiplier[edges[:, 0]],
            vertex_multiplier[edges[:, 1]])
        print(
            f"Insertion cohesion: {int(np.sum(vertex_multiplier > 1.0))} "
            f"vertices, {int(np.sum(edge_multiplier > 1.0))} edges")
    if args.origin_cap_cohesion_weight > 1.0:
        levels = np.asarray(data["vertex_contour_level"], dtype=np.int32)
        authored_anchors = np.asarray(
            data.get("anchor_vertices", []), dtype=np.int32)
        anchor_levels = levels[authored_anchors]
        full_origin = authored_anchors[
            anchor_levels == anchor_levels.min()]
        if args.origin_cap_complete_graph:
            existing = {
                (min(int(a), int(b)), max(int(a), int(b)))
                for a, b in edges}
            added = []
            for row, a in enumerate(full_origin):
                for b in full_origin[row + 1:]:
                    pair = (min(int(a), int(b)), max(int(a), int(b)))
                    if pair not in existing:
                        added.append(pair)
            if added:
                old_edge_count = len(edges)
                edges = np.vstack((
                    edges, np.asarray(added, dtype=np.int32)))
                if edge_multiplier is None:
                    edge_multiplier = np.ones(
                        old_edge_count, dtype=np.float64)
                edge_multiplier = np.r_[
                    edge_multiplier,
                    np.ones(len(added), dtype=np.float64)]
        origin_mask = np.zeros(len(rest), dtype=bool)
        origin_mask[full_origin] = True
        origin_edges = (
            origin_mask[edges[:, 0]] & origin_mask[edges[:, 1]])
        if edge_multiplier is None:
            edge_multiplier = np.ones(len(edges), dtype=np.float64)
        edge_multiplier[origin_edges] *= args.origin_cap_cohesion_weight
        print(
            f"Origin cap cohesion: {int(np.sum(origin_edges))} edges, "
            f"weight={args.origin_cap_cohesion_weight:g}")
    solver = CUDAARAP(
        rest, edges, fixed, args.device, edge_multiplier=edge_multiplier)
    if (embedded_attachment_spec is not None
            and embedded_attachment_spec.get("direct_identity", False)):
        # The direct-contour solve always uses its diagonal attachment
        # factor. Release the unanchored Laplacian Cholesky before building
        # that factor; retaining both exceeds an 8 GB GPU for this mesh.
        solver.chol = None
        torch.cuda.empty_cache()
    compliant_solver = None
    if args.near_rest_compliant_origin or args.always_compliant_origin:
        hard_origin = np.asarray(group["origin_fixed"], dtype=np.int32)
        compliant_fixed = np.setdiff1d(fixed, hard_origin)
        compliant_solver = CUDAARAP(
            rest, edges, compliant_fixed, args.device,
            edge_multiplier=edge_multiplier)
    motion = MyBVH(
        str(args.bvh), bvh_info, skel,
        T_frame=bake_emu._detect_bvh_tframe(str(args.bvh)))
    origin_body = skel.getBodyNode(group["origin_body"])
    insertion_body = skel.getBodyNode(group["insertion_body"])
    origin_rest_transform = origin_body.getWorldTransform()
    origin_rest_rotation = np.asarray(
        origin_rest_transform.rotation()).copy()
    origin_rest_translation = np.asarray(
        origin_rest_transform.translation()).copy()
    insertion_rest_transform = insertion_body.getWorldTransform()
    insertion_rest_rotation = np.asarray(
        insertion_rest_transform.rotation()).copy()
    insertion_rest_translation = np.asarray(
        insertion_rest_transform.translation()).copy()
    rest_relative_rotation = (
        np.asarray(origin_body.getWorldTransform().rotation()).T
        @ np.asarray(insertion_body.getWorldTransform().rotation()))

    output = []
    output_frames = []
    selected_frames = None
    if args.frames:
        selected_frames = {
            int(value.strip()) for value in args.frames.split(",")
            if value.strip()}
    previous = rest.copy()
    previous_guide = rest.copy()
    with torch.no_grad():
        for frame, pose in enumerate(motion.mocap_refs):
            if selected_frames is not None and frame not in selected_frames:
                continue
            projected = 0
            skel.setPositions(pose.copy())
            guide = bake_emu.compute_rigid_blend_positions(
                group["lbs_bindings"], skel, group["axis_coordinate"])
            origin_scale = 1.0
            if (len(origin_soft_ids)
                    and args.origin_cap_soft_fade_degrees > 0.0):
                relative_rotation = (
                    np.asarray(origin_body.getWorldTransform().rotation()).T
                    @ np.asarray(
                        insertion_body.getWorldTransform().rotation()))
                delta_rotation = (
                    relative_rotation @ rest_relative_rotation.T)
                angle = np.degrees(np.arccos(np.clip(
                    (np.trace(delta_rotation) - 1.0) * 0.5,
                    -1.0, 1.0)))
                origin_scale = max(
                    0.0, 1.0 - angle
                    / args.origin_cap_soft_fade_degrees)
            frame_contact_exclusion_mask = contact_exclusion_mask.copy()
            if embedded_attachment_spec is not None:
                embedded_supports = np.unique(
                    embedded_attachment_spec["indices"])
                if args.attachment_collision_exclusion_rings > 0:
                    embedded_supports = np.asarray(
                        test_emu._expand_surface_vertex_rings(
                            embedded_supports.tolist(),
                            group["surface_faces"],
                            rings=args.attachment_collision_exclusion_rings),
                        dtype=np.int32)
                frame_contact_exclusion_mask[embedded_supports] = True
            if (len(origin_soft_ids)
                    and (origin_scale > 0.0
                         or args.always_compliant_origin)):
                # These vertices terminate on the bone only while their
                # attachment spring is active.
                frame_contact_exclusion_mask[origin_soft_ids] = True
            use_compliant_origin = (
                compliant_solver is not None
                and (origin_scale > 0.0 or args.always_compliant_origin))
            active_solver = (
                compliant_solver if use_compliant_origin else solver)
            frame_fixed_mask = np.zeros(len(rest), dtype=bool)
            frame_fixed_mask[
                active_solver.fixed.detach().cpu().numpy()] = True
            initial = (guide if args.independent_frames
                       else previous + guide - previous_guide)
            if frame in initialization_by_frame:
                initial = initialization_by_frame[frame].copy()
            ramp_insertion_start = None
            ramp_knee_center = None
            if (embedded_attachment_spec is not None
                    and args.embedded_attachment_ramp_steps > 0):
                current_origin = origin_body.getWorldTransform()
                initial_rotation = (
                    np.asarray(current_origin.rotation())
                    @ origin_rest_rotation.T)
                initial_translation = (
                    np.asarray(current_origin.translation())
                    - initial_rotation @ origin_rest_translation)
                initial = rest @ initial_rotation.T + initial_translation
            x = torch.as_tensor(
                initial,
                dtype=torch.float64, device=args.device).clone()
            target = torch.as_tensor(
                guide[active_solver.fixed.detach().cpu().numpy()],
                dtype=torch.float64, device=args.device)
            soft = None
            if len(soft_ids):
                frame_soft_weights = soft_weights.copy()
                frame_soft_targets = guide[soft_ids].copy()
                if len(origin_soft_ids):
                    current_origin_transform = (
                        origin_body.getWorldTransform())
                    current_origin_rotation = np.asarray(
                        current_origin_transform.rotation())
                    rigid_rotation = (
                        current_origin_rotation @ origin_rest_rotation.T)
                    rigid_translation = (
                        np.asarray(current_origin_transform.translation())
                        - rigid_rotation @ origin_rest_translation)
                    rigid_origin_targets = (
                        rest[origin_soft_ids] @ rigid_rotation.T
                        + rigid_translation)
                    origin_row = {
                        int(vertex): row
                        for row, vertex in enumerate(soft_ids)}
                    for vertex, target_position in zip(
                            origin_soft_ids, rigid_origin_targets):
                        frame_soft_targets[
                            origin_row[int(vertex)]] = target_position
                if (len(origin_soft_ids)
                        and args.origin_cap_soft_fade_degrees > 0.0):
                    origin_mask = np.isin(soft_ids, origin_soft_ids)
                    if args.always_compliant_origin:
                        flexed_weight = (
                            args.origin_flexed_soft_weight
                            if args.origin_flexed_soft_weight > 0.0
                            else args.origin_cap_soft_weight)
                        frame_soft_weights[origin_mask] = (
                            args.origin_cap_soft_weight * origin_scale
                            + flexed_weight * (1.0 - origin_scale))
                    else:
                        frame_soft_weights[origin_mask] *= origin_scale
                    if (args.near_rest_compliant_origin
                            and not args.always_compliant_origin
                            and args.origin_hard_transition_weight > 0.0):
                        compact_origin = np.asarray(
                            group["origin_fixed"], dtype=np.int32)
                        compact_mask = np.isin(soft_ids, compact_origin)
                        compact_base_weight = (
                            args.origin_compact_base_weight
                            if args.origin_compact_base_weight > 0.0
                            else args.origin_cap_soft_weight)
                        frame_soft_weights[compact_mask] = (
                            compact_base_weight * origin_scale
                            + args.origin_hard_transition_weight
                            * (1.0 - origin_scale))
                soft = (
                    torch.as_tensor(
                        soft_ids, dtype=torch.long, device=args.device),
                    torch.as_tensor(
                        frame_soft_targets, dtype=torch.float64,
                        device=args.device),
                    torch.as_tensor(
                        frame_soft_weights, dtype=torch.float64,
                        device=args.device))
            attachment = None
            if embedded_attachment_spec is not None:
                current_origin = origin_body.getWorldTransform()
                origin_rotation = (
                    np.asarray(current_origin.rotation())
                    @ origin_rest_rotation.T)
                origin_translation = (
                    np.asarray(current_origin.translation())
                    - origin_rotation @ origin_rest_translation)
                current_insertion = insertion_body.getWorldTransform()
                insertion_rotation = (
                    np.asarray(current_insertion.rotation())
                    @ insertion_rest_rotation.T)
                insertion_translation = (
                    np.asarray(current_insertion.translation())
                    - insertion_rotation @ insertion_rest_translation)
                count = embedded_attachment_spec["origin_count"]
                rest_points = embedded_attachment_spec["rest_points"]
                attachment_targets = np.empty_like(rest_points)
                attachment_targets[:count] = (
                    rest_points[:count] @ origin_rotation.T
                    + origin_translation)
                attachment_targets[count:] = (
                    rest_points[count:] @ insertion_rotation.T
                    + insertion_translation)
                if args.embedded_attachment_ramp_steps > 0:
                    ramp_insertion_start = (
                        rest_points[count:] @ origin_rotation.T
                        + origin_translation)
                    if args.embedded_insertion_arc_ramp:
                        tibia = skel.getBodyNode("L_Tibia_Fibula0")
                        knee_joint = tibia.getParentJoint()
                        parent_body = knee_joint.getParentBodyNode()
                        parent_transform = parent_body.getWorldTransform()
                        joint_local = np.asarray(
                            knee_joint.getTransformFromParentBodyNode(
                            ).translation())
                        ramp_knee_center = (
                            np.asarray(parent_transform.rotation())
                            @ joint_local
                            + np.asarray(parent_transform.translation()))
                attachment = (
                    torch.as_tensor(
                        embedded_attachment_spec["indices"],
                        dtype=torch.long, device=args.device),
                    torch.as_tensor(
                        embedded_attachment_spec["weights"],
                        dtype=torch.float64, device=args.device),
                    torch.as_tensor(
                        attachment_targets, dtype=torch.float64,
                        device=args.device),
                    float(args.embedded_contour_attachment_weight))
            x[active_solver.fixed] = target
            bones = [] if femur_sdf else build_bones(skel)
            collision = None
            for iteration in range(args.iterations):
                iteration_attachment = attachment
                if (attachment is not None
                        and ramp_insertion_start is not None):
                    ai, aw, final_targets, apenalty = attachment
                    count = embedded_attachment_spec["origin_count"]
                    fraction = min(
                        1.0, (iteration + 1)
                        / max(1, args.embedded_attachment_ramp_steps))
                    ramp_targets = final_targets.clone()
                    ramp_start = torch.as_tensor(
                        ramp_insertion_start, dtype=torch.float64,
                        device=args.device)
                    if ramp_knee_center is None:
                        ramp_targets[count:] = (
                            (1.0 - fraction) * ramp_start
                            + fraction * final_targets[count:])
                    else:
                        center = torch.as_tensor(
                            ramp_knee_center, dtype=torch.float64,
                            device=args.device)
                        vector0 = ramp_start - center
                        vector1 = final_targets[count:] - center
                        radius0 = torch.linalg.vector_norm(
                            vector0, dim=1).clamp_min(1e-8)
                        radius1 = torch.linalg.vector_norm(
                            vector1, dim=1).clamp_min(1e-8)
                        unit0 = vector0 / radius0[:, None]
                        unit1 = vector1 / radius1[:, None]
                        omega = torch.acos(torch.clamp(
                            torch.sum(unit0 * unit1, dim=1),
                            -0.999999, 0.999999))
                        sine = torch.sin(omega).clamp_min(1e-6)
                        direction = (
                            torch.sin((1.0 - fraction) * omega)[:, None]
                            / sine[:, None] * unit0
                            + torch.sin(fraction * omega)[:, None]
                            / sine[:, None] * unit1)
                        direction = direction / torch.linalg.vector_norm(
                            direction, dim=1, keepdim=True).clamp_min(1e-8)
                        radius = (
                            (1.0 - fraction) * radius0
                            + fraction * radius1)
                        ramp_targets[count:] = (
                            center + radius[:, None] * direction)
                    iteration_attachment = (
                        ai, aw, ramp_targets, apenalty)
                iteration_soft = soft
                if (iteration_attachment is not None
                        and embedded_attachment_spec.get(
                            "direct_identity", False)):
                    ai, _aw, direct_targets, apenalty = iteration_attachment
                    direct_ids = ai[:, 0]
                    direct_weights = torch.full(
                        (len(direct_ids),), float(apenalty),
                        dtype=torch.float64, device=args.device)
                    iteration_soft = (
                        direct_ids, direct_targets, direct_weights)
                    iteration_attachment = None
                if (not args.contact_in_coupled_only
                        and iteration % args.collision_every == 0):
                    if femur_sdf:
                        raw = femur_sdf.constraints(
                            x.cpu().numpy(), samples_i, samples_w,
                            frame_contact_exclusion_mask,
                            skel, args.collision_margin,
                            args.max_collision_step,
                            args.rest_sdf_tolerance)
                    else:
                        raw = collision_constraints(
                            x.cpu().numpy(), samples_i, samples_w, bones,
                            frame_fixed_mask, args.collision_margin)
                    if args.projective_contact and raw is not None:
                        projected_x = x.cpu().numpy()
                        project_collision_constraints(
                            projected_x, raw, frame_fixed_mask,
                            args.max_collision_step)
                        x = torch.as_tensor(
                            projected_x, dtype=torch.float64,
                            device=args.device)
                        x[active_solver.fixed] = target
                        collision = None
                    else:
                        collision = None if raw is None else (
                            torch.as_tensor(
                                raw[0], dtype=torch.long,
                                device=args.device),
                            torch.as_tensor(
                                raw[1], dtype=torch.float64,
                                device=args.device),
                            torch.as_tensor(
                                raw[2], dtype=torch.float64,
                                device=args.device))
                x = active_solver.step(
                    x, target, collision, args.collision_weight,
                    args.arap_weight, iteration_soft, iteration_attachment)

            # Joint refinement: edge shape, signed volume, and active SDF
            # contact are differentiated in one objective.  This is intended
            # for deep intercondylar flexion where sequential volume/contact
            # projections fight each other.
            if femur_sdf and args.coupled_quality_outer > 0:
                torch.set_grad_enabled(True)
                tet_t = torch.as_tensor(
                    tets, dtype=torch.long, device=args.device)
                rest_v_t = torch.as_tensor(
                    signed_volumes(rest, tets), dtype=torch.float64,
                    device=args.device)
                rest_sign_t = torch.sign(rest_v_t)
                rest_abs_t = torch.abs(rest_v_t).clamp_min(1e-12)
                rest_length = torch.linalg.vector_norm(
                    active_solver.rest_edge, dim=1).clamp_min(1e-8)
                if args.pre_untangle_steps > 0:
                    initial_volume = signed_volumes_cuda(x, tet_t)
                    initial_ratio = (
                        initial_volume * rest_sign_t / rest_abs_t)
                    if float(initial_ratio.min()) <= 1e-5:
                        x.requires_grad_(True)
                        untangle_optimizer = torch.optim.Adam(
                            [x], lr=args.coupled_quality_lr)
                        for _ in range(args.pre_untangle_steps):
                            untangle_optimizer.zero_grad()
                            edge = x[active_solver.i] - x[active_solver.j]
                            edge_ratio = (
                                torch.linalg.vector_norm(edge, dim=1)
                                / rest_length)
                            volume = signed_volumes_cuda(x, tet_t)
                            ratio = (
                                volume * rest_sign_t / rest_abs_t)
                            loss = (
                                torch.sum(
                                    active_solver.shape_multiplier
                                    * (edge_ratio - 1.0) ** 2)
                                / torch.sum(
                                    active_solver.shape_multiplier)
                                + args.coupled_volume_weight
                                * torch.mean((ratio - 1.0) ** 2)
                                + 50.0 * args.coupled_volume_weight
                                * torch.mean(
                                    torch.relu(0.08 - ratio) ** 2))
                            if attachment is not None:
                                ai, aw, at, apenalty = attachment
                                attachment_points = torch.sum(
                                    x[ai] * aw[:, :, None], dim=1)
                                loss = loss + apenalty * torch.mean(
                                    torch.sum(
                                        (attachment_points - at) ** 2,
                                        dim=1))
                            loss.backward()
                            untangle_optimizer.step()
                            with torch.no_grad():
                                x[active_solver.fixed] = target
                            if float(ratio.min().detach()) > 0.03:
                                break
                        x = x.detach()
                coupled_converged = False
                for _ in range(args.coupled_quality_outer):
                    raw = femur_sdf.constraints(
                        x.detach().cpu().numpy(), samples_i, samples_w,
                        frame_contact_exclusion_mask, skel, args.collision_margin,
                        args.max_collision_step,
                        args.rest_sdf_tolerance)
                    if raw is None:
                        ci = cw = ct = None
                    else:
                        ci = torch.as_tensor(
                            raw[0], dtype=torch.long, device=args.device)
                        cw = torch.as_tensor(
                            raw[1], dtype=torch.float64, device=args.device)
                        ct = torch.as_tensor(
                            raw[2], dtype=torch.float64, device=args.device)
                    x.requires_grad_(True)
                    optimizer = torch.optim.Adam(
                        [x], lr=args.coupled_quality_lr)
                    for inner_step in range(args.coupled_quality_inner):
                        before_step = (
                            x.detach().clone()
                            if args.inversion_safe_line_search else None)
                        optimizer.zero_grad()
                        edge = x[active_solver.i] - x[active_solver.j]
                        edge_ratio = (
                            torch.linalg.vector_norm(edge, dim=1)
                            / rest_length)
                        shape_loss = (
                            torch.sum(
                                active_solver.shape_multiplier
                                * (edge_ratio - 1.0) ** 2)
                            / torch.sum(active_solver.shape_multiplier))
                        volume = signed_volumes_cuda(x, tet_t)
                        ratio = volume * rest_sign_t / rest_abs_t
                        volume_loss = torch.mean((ratio - 1.0) ** 2)
                        inversion_loss = torch.mean(
                            torch.relu(0.1 - ratio) ** 2)
                        if (args.differentiable_sdf_contact
                                and femur_sdf.rest_distance is not None):
                            contact_loss = femur_sdf.torch_penetration_loss(
                                x, samples_i, samples_w,
                                frame_contact_exclusion_mask, skel,
                                args.rest_sdf_tolerance)
                        elif ci is None:
                            contact_loss = x.new_zeros(())
                        else:
                            points = torch.sum(
                                x[ci] * cw[:, :, None], dim=1)
                            contact_loss = torch.mean(
                                torch.sum((points - ct) ** 2, dim=1)
                                / (args.collision_margin ** 2))
                        if soft is None:
                            transition_loss = x.new_zeros(())
                        else:
                            si, st, sw = soft
                            transition_loss = torch.mean(
                                sw * torch.sum((x[si] - st) ** 2, dim=1))
                        if attachment is None:
                            attachment_loss = x.new_zeros(())
                        else:
                            ai, aw, at, apenalty = attachment
                            attachment_points = torch.sum(
                                x[ai] * aw[:, :, None], dim=1)
                            attachment_loss = apenalty * torch.mean(
                                torch.sum(
                                    (attachment_points - at) ** 2,
                                    dim=1))
                        loss = (
                            shape_loss
                            + args.coupled_volume_weight * volume_loss
                            + 10.0 * args.coupled_volume_weight
                            * inversion_loss
                            + args.coupled_collision_weight * contact_loss
                            + transition_loss + attachment_loss)
                        loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                            x[active_solver.fixed] = target
                            if args.inversion_safe_line_search:
                                before_volume = signed_volumes_cuda(
                                    before_step, tet_t)
                                before_ratio = (
                                    before_volume * rest_sign_t / rest_abs_t)
                                candidate = x.detach().clone()
                                candidate_volume = signed_volumes_cuda(
                                    candidate, tet_t)
                                candidate_ratio = (
                                    candidate_volume * rest_sign_t
                                    / rest_abs_t)
                                if (float(before_ratio.min()) > 1e-5
                                        and float(candidate_ratio.min())
                                        <= 1e-5):
                                    accepted = False
                                    alpha = 0.5
                                    for _ in range(20):
                                        trial = (
                                            before_step
                                            + alpha
                                            * (candidate - before_step))
                                        trial[active_solver.fixed] = target
                                        trial_ratio = (
                                            signed_volumes_cuda(trial, tet_t)
                                            * rest_sign_t / rest_abs_t)
                                        if float(trial_ratio.min()) > 1e-5:
                                            x.copy_(trial)
                                            accepted = True
                                            break
                                        alpha *= 0.5
                                    if not accepted:
                                        x.copy_(before_step)
                                        x[active_solver.fixed] = target
                        if (args.adaptive_coupled
                                and inner_step % 20 == 19
                                and float(ratio.min().detach()) > 0.02
                                and float(contact_loss.detach())
                                <= args.adaptive_contact_loss):
                            coupled_converged = True
                            break
                    x = x.detach()
                    if coupled_converged:
                        break
                torch.set_grad_enabled(False)
            # A soft ARAP/contact compromise is not collision-free. End with
            # bounded inequality projections and refresh the SDF after each
            # pass; importantly, do not run ARAP again after this stage.
            if femur_sdf and args.final_collision_passes > 0:
                cpu = x.cpu().numpy()
                projected = 0

                inverse_mass = (~frame_fixed_mask).astype(np.float64)
                rest_volumes = signed_volumes(rest, tets)

                def repair_volume():
                    for _ in range(args.quality_volume_sweeps):
                        before = cpu.copy()
                        project_volumes(
                            cpu, tets, rest_volumes, inverse_mass,
                            args.quality_volume_stiffness)
                        delta = cpu - before
                        length = np.linalg.norm(delta, axis=1)
                        limited = length > args.quality_max_step
                        delta[limited] *= (
                            args.quality_max_step / length[limited])[:, None]
                        cpu[:] = before + delta
                        cpu[fixed] = guide[fixed]

                def project_contact(passes):
                    count_sum = 0
                    for _ in range(passes):
                        raw = femur_sdf.constraints(
                            cpu, samples_i, samples_w,
                            frame_contact_exclusion_mask, skel,
                            args.collision_margin, args.max_collision_step,
                            args.rest_sdf_tolerance)
                        _, count, _ = project_collision_constraints(
                            cpu, raw, frame_fixed_mask,
                            args.max_collision_step)
                        count_sum += count
                        if count == 0:
                            break
                    return count_sum

                projected += project_contact(args.final_collision_passes)
                for _ in range(args.quality_cycles):
                    repair_volume()
                    projected += project_contact(
                        args.final_collision_passes)
                # Collision is deliberately last: volume repair may push a
                # surface sample back into the intercondylar femur.
                projected += project_contact(args.final_collision_passes)
                x = torch.as_tensor(
                    cpu, dtype=torch.float64, device=args.device)
                x[active_solver.fixed] = target
            cpu = x.cpu().numpy()
            if not np.isfinite(cpu).all():
                raise FloatingPointError(f"non-finite frame {frame}")
            v = signed_volumes(cpu, tets) * np.sign(
                signed_volumes(rest, tets))
            if args.require_positive_tets and np.any(v <= 0.0):
                raise RuntimeError(
                    f"frame {frame} has {int(np.sum(v <= 0.0))} "
                    "inverted tets; refusing to write invalid simulation")
            origin_reference_text = ""
            if len(origin_soft_ids):
                origin_error = np.linalg.norm(
                    cpu[origin_soft_ids] - rigid_origin_targets, axis=1)
                origin_reference_text = (
                    f", origin_ref_mean/max="
                    f"{origin_error.mean() * 1000.0:.2f}/"
                    f"{origin_error.max(initial=0.0) * 1000.0:.2f}mm")
            print(
                f"frame {frame}: inverted={(v <= 0).sum()}/{len(tets)}, "
                f"volume_ratio={np.abs(v).sum()/np.abs(signed_volumes(rest, tets)).sum():.4f}, "
                f"contact_projections={projected if femur_sdf else 0}"
                f"{origin_reference_text}")
            output.append(cpu.astype(np.float32))
            output_frames.append(frame)
            previous, previous_guide = cpu, guide

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / f"{args.name}_chunk_0000.npz",
        frames=np.asarray(output_frames, dtype=np.int32),
        positions=np.stack(output))
    (args.output_dir / ".done").touch()
    print(f"Saved {len(output)} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
