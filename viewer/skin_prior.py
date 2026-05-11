"""Skinning-prior binder for ARAP.

For each tet vertex of every muscle, find the nearest triangle on the
nearest bone at REST pose.  Cache:

  - bone body name
  - barycentric coordinates on the triangle
  - signed offset along the triangle normal
  - weight w_i = exp(-rest_dist / sigma)

At simulation time the target world position is
``T_bone(world) @ (bary @ tri_rest_local + offset * normal_local)`` — a
linear function of the current bone pose.  The ARAP system gains an
extra ``w_i I`` on the diagonal of vertex i and ``w_i target_i`` on the
RHS, exactly mirroring the existing penalty-collision mechanism but
with per-vertex weights.

No barrier: this is a soft "stay where you started relative to your
nearest bone" prior, not a hard contact constraint.
"""

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None


_DEFAULT_SIGMA = 0.02      # 2cm decay length scale
_MIN_WEIGHT = 1e-3         # drop verts whose w_i falls below this
_MAX_BIND_DIST = 0.05      # 5cm: ignore verts farther than this at rest


class SkinPriorBinder:
    def __init__(self, mesh_scale=0.01, sigma=_DEFAULT_SIGMA,
                 max_bind_dist=_MAX_BIND_DIST, strength=1.0):
        self.mesh_scale = mesh_scale
        self.sigma = sigma
        self.max_bind_dist = max_bind_dist
        self.strength = float(strength)
        # bindings[muscle_name] = list of (global_vi, body_name, tri_idx,
        #                                  bary, normal_offset, weight)
        self.bindings = {}
        # Per-bone rest-pose triangle data, keyed by body name:
        #   { body_name: {'verts_local': (V,3), 'tris': (T,3),
        #                 'tri_normals_local': (T,3),
        #                 'rest_R': (3,3), 'rest_t': (3,)} }
        self.bone_rest = {}

    @staticmethod
    def _world_to_local(points_world, R, t):
        return (points_world - t) @ R  # = R^T @ (p - t)

    def _record_bone_rest(self, body_name, body_node, tri_mesh):
        """Capture rest-pose transform and triangle data for a bone."""
        if body_name in self.bone_rest:
            return self.bone_rest[body_name]
        wt = body_node.getWorldTransform()
        R_rest = np.array(wt.rotation())
        t_rest = np.array(wt.translation())
        verts_local = self._world_to_local(np.asarray(tri_mesh.vertices), R_rest, t_rest)
        tris = np.asarray(tri_mesh.faces, dtype=np.int32)
        tri_pts = verts_local[tris]
        e1 = tri_pts[:, 1] - tri_pts[:, 0]
        e2 = tri_pts[:, 2] - tri_pts[:, 0]
        n_local = np.cross(e1, e2)
        n_norm = np.linalg.norm(n_local, axis=1, keepdims=True)
        n_local = np.where(n_norm > 1e-12, n_local / n_norm, n_local)
        entry = {
            "verts_local": verts_local,
            "tris": tris,
            "tri_normals_local": n_local,
            "rest_R": R_rest,
            "rest_t": t_rest,
            "body_node": body_node,
        }
        self.bone_rest[body_name] = entry
        return entry

    @staticmethod
    def _project_to_triangle(p, a, b, c):
        """Project a 3D point onto the triangle (a,b,c).  Returns
        (closest_point, barycentric, signed_normal_offset)."""
        ab = b - a
        ac = c - a
        ap = p - a
        # Solve 2x2 for u,v where projection = a + u*ab + v*ac
        d00 = ab @ ab
        d01 = ab @ ac
        d11 = ac @ ac
        d20 = ap @ ab
        d21 = ap @ ac
        denom = d00 * d11 - d01 * d01
        if denom < 1e-20:
            return a.copy(), np.array([1.0, 0.0, 0.0]), 0.0
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        # Clamp to triangle
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        w = max(0.0, min(1.0, w))
        s = u + v + w
        if s > 0:
            u, v, w = u / s, v / s, w / s
        cp = u * a + v * b + w * c
        n = np.cross(ab, ac)
        n_len = np.linalg.norm(n)
        if n_len > 1e-12:
            n = n / n_len
        offset = (p - cp) @ n
        return cp, np.array([u, v, w]), float(offset)

    def precompute(self, muscle_meshes, skeleton_meshes, dart_skel,
                   skeleton_dir):
        """Build per-vert bindings using trimesh proximity for nearest
        triangle, then refine to triangle-local barycentric + normal
        offset.

        muscle_meshes: dict name -> MeshLoader (must have tet_vertices and
                       attach_skeleton_names so we can prefilter bones)
        skeleton_meshes: dict bone_name -> MeshLoader (rest-pose vertices
                         already in world)
        dart_skel: DART skeleton — used to retrieve current body world
                   transforms (treated as rest pose at this call).
        skeleton_dir: unused; kept for API symmetry with bone_sdf path.
        """
        if trimesh is None:
            print("  trimesh not available — skinning prior disabled")
            return False
        import os as _os
        # Load each bone mesh directly from disk so we get consistent
        # vertex/face arrays — MeshLoader's render-side vertex layout can
        # have duplicates that break trimesh.
        bone_tris = {}
        for bn in skeleton_meshes.keys():
            # Try the bone name as-is, then stripped of trailing digits
            # (DART body suffixes "0"/"1").
            tried = []
            for cand in (bn, bn.rstrip("0123456789")):
                if cand in tried:
                    continue
                tried.append(cand)
                p = _os.path.join(skeleton_dir, f"{cand}.obj")
                if _os.path.exists(p):
                    try:
                        tm = trimesh.load_mesh(p, process=False)
                        tm.vertices = tm.vertices * self.mesh_scale
                        bone_tris[bn] = tm
                        break
                    except Exception as _e:
                        print(f"    bone load fail {bn}: {_e}")
                        break

        # Resolve DART body node for each bone name (try suffix variants).
        def _body_node(name):
            for cand in (name, name + "0", name + "1"):
                bn = dart_skel.getBodyNode(cand)
                if bn is not None:
                    return bn, cand
            return None, name

        n_bound = 0
        n_skipped = 0
        for muscle_name, mobj in muscle_meshes.items():
            tet_v = getattr(mobj, 'tet_vertices', None)
            if tet_v is None:
                continue
            tet_v = np.asarray(tet_v, dtype=np.float64)
            # Prefilter bones: nearby ones via bbox proximity.
            mbb_min = tet_v.min(0) - 0.05
            mbb_max = tet_v.max(0) + 0.05
            cand_bones = []
            for bn, tm in bone_tris.items():
                bbb_min = tm.bounds[0]
                bbb_max = tm.bounds[1]
                if (bbb_max < mbb_min).any() or (bbb_min > mbb_max).any():
                    continue
                cand_bones.append(bn)
            if not cand_bones:
                n_skipped += len(tet_v)
                continue
            # For each tet vert, find nearest among candidate bones.
            best_d = np.full(len(tet_v), np.inf)
            best_bone = [None] * len(tet_v)
            best_tri = np.full(len(tet_v), -1, dtype=np.int32)
            best_cp = np.zeros((len(tet_v), 3))
            for bn in cand_bones:
                tm = bone_tris[bn]
                cp, dist, tri_idx = trimesh.proximity.closest_point(tm, tet_v)
                better = dist < best_d
                if not np.any(better):
                    continue
                best_d[better] = dist[better]
                best_cp[better] = cp[better]
                best_tri[better] = tri_idx[better]
                for i in np.where(better)[0]:
                    best_bone[i] = bn
            # Bind verts within max_bind_dist.
            entries = []
            for vi in range(len(tet_v)):
                bn = best_bone[vi]
                d = best_d[vi]
                if bn is None or d > self.max_bind_dist:
                    continue
                w = float(self.strength * np.exp(-d / self.sigma))
                if w < _MIN_WEIGHT:
                    continue
                # Capture this bone's rest transform (skips if already done).
                body_node, resolved_name = _body_node(bn)
                if body_node is None:
                    continue
                rest = self._record_bone_rest(resolved_name, body_node, bone_tris[bn])
                tri_idx = int(best_tri[vi])
                tri = rest["tris"][tri_idx]
                a = rest["verts_local"][tri[0]]
                b = rest["verts_local"][tri[1]]
                c = rest["verts_local"][tri[2]]
                # Project the muscle vert (in bone-local rest frame) onto
                # this triangle to extract barycentric + normal offset.
                p_local = self._world_to_local(
                    tet_v[vi], rest["rest_R"], rest["rest_t"])
                cp_local, bary, offset = self._project_to_triangle(p_local, a, b, c)
                entries.append({
                    "vi_local": vi,
                    "body": resolved_name,
                    "tri_idx": tri_idx,
                    "bary": bary,
                    "normal_offset": offset,
                    "weight": w,
                })
            self.bindings[muscle_name] = entries
            n_bound += len(entries)
        print(f"  Skin prior: {n_bound} verts bound across {len(self.bindings)} muscles")
        return n_bound > 0

    def compute_targets(self, muscle_name, vert_offset, current_positions=None):
        """Return (global_indices, weights, target_world_positions) for the
        given muscle's bound verts using CURRENT bone world transforms.

        vert_offset: integer added to local vi to get global system index.
        current_positions: optional np.ndarray (N_muscle_verts, 3) of the
            muscle's current world-space positions.  When provided, the
            skin-prior becomes ONE-SIDED: each vert is pulled toward
            ``surface + rest_offset * normal`` only if its current signed
            distance along the bone normal is BELOW its rest offset
            (i.e., the vert is trying to penetrate or sit closer to the
            bone than at rest).  Verts farther from the bone get a
            target equal to their current position, producing zero net
            spring force.  Without ``current_positions`` the legacy
            symmetric attractor is used.
        """
        entries = self.bindings.get(muscle_name)
        if not entries:
            return None
        n = len(entries)
        gi = np.empty(n, dtype=np.int64)
        ws = np.empty(n, dtype=np.float64)
        targets = np.empty((n, 3), dtype=np.float64)
        for k, e in enumerate(entries):
            rest = self.bone_rest[e["body"]]
            tri = rest["tris"][e["tri_idx"]]
            a = rest["verts_local"][tri[0]]
            b = rest["verts_local"][tri[1]]
            c = rest["verts_local"][tri[2]]
            n_local = rest["tri_normals_local"][e["tri_idx"]]
            cp_local = e["bary"][0] * a + e["bary"][1] * b + e["bary"][2] * c
            wt = rest["body_node"].getWorldTransform()
            R_now = np.array(wt.rotation())
            t_now = np.array(wt.translation())
            surface_world = R_now @ cp_local + t_now
            normal_world = R_now @ n_local
            rest_offset = e["normal_offset"]
            gi[k] = vert_offset + e["vi_local"]
            ws[k] = e["weight"]
            if current_positions is not None:
                vi_local = e["vi_local"]
                if vi_local < len(current_positions):
                    cur = current_positions[vi_local]
                    current_offset = float(np.dot(cur - surface_world, normal_world))
                    if current_offset < rest_offset:
                        # Penetration tendency — pull back to rest offset.
                        targets[k] = surface_world + rest_offset * normal_world
                    else:
                        # Outside rest distance — zero net pull (target =
                        # current pos, so spring force ≈ 0).
                        targets[k] = cur
                    continue
            # Legacy two-sided attractor (no current_positions supplied).
            targets[k] = surface_world + rest_offset * normal_world
        return gi, ws, targets
