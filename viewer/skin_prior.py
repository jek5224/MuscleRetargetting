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
_TOP_K_BONES = 2           # bones per vert for weighted multi-bone binding

# Muscles that should use DQS (dual-quaternion) blending instead of the
# legacy single-bone target.  These cross the knee/hip on the anterior side
# where candy-wrap is the dominant artefact and the inner-side pinch is
# minimal (no big posterior tendon mass).
DQS_MUSCLES = {
    'L_Vastus_Intermedius', 'L_Vastus_Lateralis', 'L_Vastus_Medialis',
    'L_Rectus_Femoris', 'L_Tensor_Fascia_Lata',
    'R_Vastus_Intermedius', 'R_Vastus_Lateralis', 'R_Vastus_Medialis',
    'R_Rectus_Femoris', 'R_Tensor_Fascia_Lata',
    # Pes anserinus: biarticular tendinous muscles wrapping medial knee.
    # DQS K=2 bone blend prevents the Z-kink from sharp femur→tibia
    # skinning-weight transition at the joint boundary.
    'L_Sartorius', 'L_Gracilis', 'L_Semitendinosus',
    'R_Sartorius', 'R_Gracilis', 'R_Semitendinosus',
}

# Muscles that span multiple joints (hip + knee) and wrap around the knee at
# flex.  Skin prior bone-glues them which fights cross-contour contraction
# (you get either rest-pose kinking or contracted explosions when both are
# active).  Disable skin prior here; rely on bone collision (--self-collision)
# for penetration barrier at extension and knee-angle-driven cross-contour
# rest contraction (in zygote_mesh_ui) for compact-wrap at flex.
KNEE_CROSSING_MUSCLES = {
    'L_Biceps_Femoris', 'L_Semitendinosus', 'L_Semimembranosus',
    'L_Gracilis', 'L_Sartorius',
    'R_Biceps_Femoris', 'R_Semitendinosus', 'R_Semimembranosus',
    'R_Gracilis', 'R_Sartorius',
}
NO_SKIN_PRIOR_MUSCLES = set()


# Pes anserinus bundle: 4-anchor anatomical path shared by the three medial
# knee-crossing tendons.  See PesAnserinusBundle below.
PES_ANSERINUS_BUNDLE = ('L_Sartorius', 'L_Gracilis', 'L_Semitendinosus')


def _world_to_local(p_world, R, t):
    """Transform world point(s) to bone-local frame: R^T (p - t)."""
    return (np.asarray(p_world) - t) @ R


def _local_to_world(p_local, R, t):
    """Inverse of _world_to_local."""
    return p_local @ R.T + t


def _project_to_polyline(point, anchors):
    """Project a 3D point onto a polyline through `anchors` (M, 3).
    Returns (s, offset, seg_idx, t_in_seg):
        s ∈ [0, 1]: normalized arc-length along the polyline.
        offset (3,): point - projected_point.
        seg_idx, t_in_seg: which segment + local param in [0, 1].
    """
    anchors = np.asarray(anchors, dtype=np.float64)
    n_seg = len(anchors) - 1
    # Compute segment lengths for arc-length param
    seg_vecs = anchors[1:] - anchors[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    cum = np.zeros(n_seg + 1)
    cum[1:] = np.cumsum(seg_lens)
    total_len = max(cum[-1], 1e-9)
    # Find best segment by projection distance
    best_dist = np.inf
    best = (0, 0.0, anchors[0])
    for i in range(n_seg):
        a = anchors[i]; b = anchors[i + 1]
        d = b - a
        dl = float(np.dot(d, d))
        if dl < 1e-12:
            t = 0.0
            p = a
        else:
            t = float(np.dot(point - a, d) / dl)
            t = max(0.0, min(1.0, t))
            p = a + t * d
        dist = np.linalg.norm(point - p)
        if dist < best_dist:
            best_dist = dist
            best = (i, t, p)
    i, t, p_on = best
    s = (cum[i] + t * seg_lens[i]) / total_len
    return float(s), np.asarray(point - p_on, dtype=np.float64), i, float(t)


def _polyline_at(s, anchors, cum_lens, total_len):
    """Evaluate polyline at normalized arc-length s ∈ [0, 1]."""
    arc = s * total_len
    n_seg = len(anchors) - 1
    # Locate segment
    i = 0
    for k in range(n_seg):
        if arc <= cum_lens[k + 1] + 1e-9:
            i = k
            break
    else:
        i = n_seg - 1
    seg_len = max(cum_lens[i + 1] - cum_lens[i], 1e-9)
    t = (arc - cum_lens[i]) / seg_len
    t = max(0.0, min(1.0, t))
    return anchors[i] + t * (anchors[i + 1] - anchors[i]), i, t


def _polyline_tangent(s, anchors, cum_lens, total_len):
    """Unit tangent of polyline at s."""
    _, i, _ = _polyline_at(s, anchors, cum_lens, total_len)
    d = anchors[i + 1] - anchors[i]
    dl = np.linalg.norm(d)
    return d / max(dl, 1e-9)


class PesAnserinusBundle:
    """Shared 4-anchor anatomical path for the pes anserinus muscle group.

    Anchors:
        0: origin centroid (pelvis-local)
        1: mid-thigh contour center at u≈0.35 (femur-local)
        2: knee wrap contour center at u≈0.75 (femur-local)
        3: insertion centroid (tibia-local)

    Per bundle vert: stored rest-pose (s, offset_local) where s is the arc-
    length parameter along the rest polyline, offset_local is the rest-frame
    perpendicular offset.  At runtime the polyline is reconstructed from the
    bones' current world transforms, and each vert's target =
    polyline_now(s) + R_local(s) × offset_local × shape_scale.

    R_local(s) is computed per-segment as a rotation aligning the rest
    segment direction with the current segment direction; offset is
    transformed by that rotation so transverse positions track the curve's
    deformation smoothly.
    """

    def __init__(self, pelvis_body_name='L_Os_Coxae0',
                 femur_body_name='L_Femur0',
                 tibia_body_name='L_Tibia_Fibula0'):
        self.pelvis_body_name = pelvis_body_name
        self.femur_body_name = femur_body_name
        self.tibia_body_name = tibia_body_name
        self.anchor_bone = ['pelvis', 'femur', 'femur', 'tibia']
        self.anchor_local = None       # (4, 3) bone-local positions
        self.rest_R = None             # dict bone -> rest world R
        self.rest_t = None             # dict bone -> rest world t
        self.rest_anchors = None       # (4, 3) rest world positions
        self.rest_cum_lens = None      # (5,) cumulative arc lengths
        self.rest_total_len = 1.0
        # Per-bundle-vert bindings.  Lists parallel-indexed.
        self.gi_list = []              # global vert idx
        self.s_list = []               # rest arc-length param
        self.seg_idx_list = []         # which segment vert binds to
        self.offset_local_list = []    # rest-frame transverse offset (3,)
        self.weight_list = []          # per-vert pull weight

    def _body(self, skel, name):
        return skel.getBodyNode(name)

    def _bone_transform(self, skel, bone_kind):
        if bone_kind == 'pelvis':
            b = self._body(skel, self.pelvis_body_name)
        elif bone_kind == 'femur':
            b = self._body(skel, self.femur_body_name)
        else:
            b = self._body(skel, self.tibia_body_name)
        wt = b.getWorldTransform()
        return np.array(wt.rotation()), np.array(wt.translation())

    def precompute(self, muscles, skel, weight=1.5, sigma=0.04):
        """Build the rest-pose polyline + per-vert bindings.

        muscles: dict {muscle_name: mobj} for the bundle muscles.
        skel: DART skeleton (in T-pose / rest pose for this call).
        Stores per-muscle local-vi lists; gi translation happens at
        compute_targets time when global_offset is known.
        """
        # Per-muscle binding storage (resolved to global later).
        self.per_muscle = {}  # mname -> dict(vi_local, s, seg, offset_local, w)
        # Capture rest bone transforms once
        self.rest_R = {}
        self.rest_t = {}
        for kind in ('pelvis', 'femur', 'tibia'):
            R, t = self._bone_transform(skel, kind)
            self.rest_R[kind] = R
            self.rest_t[kind] = t

        # Derive the 4 anchors from a "reference" muscle (Sartorius) at rest.
        # Anchors 1 and 2 come from contour-level centers at u≈0.35 / u≈0.75
        # of the reference; Anchors 0 and 3 come from union of fixed-vert
        # centroids across the bundle.
        ref_name = None
        for cand in ('L_Sartorius', 'L_Gracilis', 'L_Semitendinosus'):
            if cand in muscles:
                ref_name = cand
                break
        if ref_name is None:
            return False
        ref_mobj = muscles[ref_name]
        ref_rest = np.asarray(ref_mobj.soft_body.rest_positions, dtype=np.float64)
        ref_vcl = np.asarray(getattr(ref_mobj, 'vertex_contour_level',
                                     np.full(len(ref_rest), -1)),
                              dtype=np.int32)
        max_level = max(int(ref_vcl.max()), 1)
        u = np.clip(ref_vcl.astype(np.float64) / max_level, 0.0, 1.0)

        # Anchor 0 / 3: union of origin / insertion fixed cap centers across bundle
        origin_verts_world = []
        insertion_verts_world = []
        for mname, mobj in muscles.items():
            clv = getattr(mobj, 'contour_level_vertices', None)
            if not clv:
                continue
            rest = np.asarray(mobj.soft_body.rest_positions, dtype=np.float64)
            for vi, (s_idx, e) in clv.items():
                if e == 0:
                    origin_verts_world.append(rest[vi])
                elif e == 1:
                    insertion_verts_world.append(rest[vi])
        if not origin_verts_world or not insertion_verts_world:
            return False
        a0_world = np.mean(origin_verts_world, axis=0)
        a3_world = np.mean(insertion_verts_world, axis=0)

        # Anchor 1 / 2: ref-muscle contour centers at u close to 0.35 / 0.75
        def _ucenter(target_u):
            mask = np.abs(u - target_u) < 0.06
            if not np.any(mask):
                mask = np.abs(u - target_u) < 0.12
            if not np.any(mask):
                return None
            return ref_rest[mask].mean(axis=0)
        a1_world = _ucenter(0.35)
        a2_world = _ucenter(0.75)
        if a1_world is None or a2_world is None:
            return False

        # Convert anchors to their bone-local frames
        anchors_world = np.stack([a0_world, a1_world, a2_world, a3_world], axis=0)
        anchor_local = np.zeros((4, 3))
        for i, kind in enumerate(self.anchor_bone):
            R = self.rest_R[kind]
            t = self.rest_t[kind]
            anchor_local[i] = _world_to_local(anchors_world[i], R, t)
        self.anchor_local = anchor_local
        self.rest_anchors = anchors_world

        # Rest cum lens for arc-length param
        seg_vecs = anchors_world[1:] - anchors_world[:-1]
        seg_lens = np.linalg.norm(seg_vecs, axis=1)
        cum = np.zeros(len(anchors_world))
        cum[1:] = np.cumsum(seg_lens)
        self.rest_cum_lens = cum
        self.rest_total_len = max(cum[-1], 1e-9)

        # Per-vert bindings: bundle = mid-shaft non-fixed verts of all 3.
        for mname, mobj in muscles.items():
            rest = np.asarray(mobj.soft_body.rest_positions, dtype=np.float64)
            n = rest.shape[0]
            fixed_mask = np.asarray(mobj.soft_body.fixed_mask)
            vi_l, s_l, seg_l, off_l, w_l = [], [], [], [], []
            for vi in range(n):
                if fixed_mask[vi]:
                    continue
                s, off_world, seg_idx, _t = _project_to_polyline(rest[vi], anchors_world)
                d = float(np.linalg.norm(off_world))
                w = float(weight * np.exp(-d / sigma))
                if w < 1e-3:
                    continue
                local_off = self._world_offset_to_seg_frame(
                    off_world, anchors_world, seg_idx)
                vi_l.append(vi)
                s_l.append(s)
                seg_l.append(seg_idx)
                off_l.append(local_off)
                w_l.append(w)
            if not vi_l:
                continue
            self.per_muscle[mname] = dict(
                vi=np.array(vi_l, dtype=np.int64),
                s=np.array(s_l, dtype=np.float64),
                seg=np.array(seg_l, dtype=np.int32),
                offset=np.stack(off_l, axis=0),
                w=np.array(w_l, dtype=np.float64),
            )
        return bool(self.per_muscle)

    @staticmethod
    def _seg_frame(anchors, seg_idx):
        """Return (R, origin) for the local frame of segment `seg_idx`:
        +x = segment direction, +y/+z = stable perpendiculars.
        """
        a = anchors[seg_idx]
        b = anchors[seg_idx + 1]
        d = b - a
        dl = np.linalg.norm(d)
        if dl < 1e-9:
            ex = np.array([1.0, 0.0, 0.0])
        else:
            ex = d / dl
        # Pick up vector roughly world-up [0, 1, 0] unless near parallel
        up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(up, ex)) > 0.95:
            up = np.array([1.0, 0.0, 0.0])
        ez = np.cross(ex, up)
        ez /= max(np.linalg.norm(ez), 1e-9)
        ey = np.cross(ez, ex)
        # R columns = ex, ey, ez (frame basis vectors in world)
        return np.stack([ex, ey, ez], axis=1), a

    def _world_offset_to_seg_frame(self, off_world, anchors, seg_idx):
        R, _ = self._seg_frame(anchors, seg_idx)
        # local = R^T off_world
        return R.T @ off_world

    def _seg_frame_to_world(self, off_local, anchors, seg_idx):
        R, _ = self._seg_frame(anchors, seg_idx)
        return R @ off_local

    def compute_targets(self, skel, global_offset, shape_scale=1.0):
        """Per-frame: rebuild polyline in world, return (gi, w, target) arrays.

        global_offset: dict {muscle_name: int} mapping vi_local to global vert idx.
        """
        if self.anchor_local is None or not self.per_muscle:
            return None
        # Current bone transforms
        R_now = {}
        t_now = {}
        for kind in ('pelvis', 'femur', 'tibia'):
            R, t = self._bone_transform(skel, kind)
            R_now[kind] = R
            t_now[kind] = t
        # Current anchor world positions
        anchors_now = np.zeros((4, 3))
        for i, kind in enumerate(self.anchor_bone):
            anchors_now[i] = _local_to_world(self.anchor_local[i],
                                             R_now[kind], t_now[kind])
        seg_vecs = anchors_now[1:] - anchors_now[:-1]
        seg_lens = np.linalg.norm(seg_vecs, axis=1)
        cum = np.zeros(len(anchors_now))
        cum[1:] = np.cumsum(seg_lens)
        total = max(cum[-1], 1e-9)
        all_gi, all_w, all_t = [], [], []
        for mname, data in self.per_muscle.items():
            if mname not in global_offset:
                continue
            base = global_offset[mname]
            vi = data['vi']
            s = data['s']
            seg = data['seg']
            offsets = data['offset']
            ws = data['w']
            n = len(vi)
            for k in range(n):
                seg_idx = int(seg[k])
                arc = float(s[k]) * total
                seg_lo = cum[seg_idx]
                seg_hi = cum[seg_idx + 1]
                seg_len_now = max(seg_hi - seg_lo, 1e-9)
                t = float(np.clip((arc - seg_lo) / seg_len_now, 0.0, 1.0))
                p_on = anchors_now[seg_idx] + t * (anchors_now[seg_idx + 1]
                                                  - anchors_now[seg_idx])
                off_world = self._seg_frame_to_world(offsets[k],
                                                     anchors_now, seg_idx)
                all_gi.append(base + int(vi[k]))
                all_w.append(float(ws[k]))
                all_t.append(p_on + shape_scale * off_world)
        if not all_gi:
            return None
        return (np.array(all_gi, dtype=np.int64),
                np.array(all_w, dtype=np.float64),
                np.stack(all_t, axis=0))


class SkinPriorBinder:
    def __init__(self, mesh_scale=0.01, sigma=_DEFAULT_SIGMA,
                 max_bind_dist=_MAX_BIND_DIST, strength=0.35,
                 top_k=_TOP_K_BONES):
        self.mesh_scale = mesh_scale
        self.sigma = sigma
        self.max_bind_dist = max_bind_dist
        self.strength = float(strength)
        self.top_k = int(top_k)
        # bindings[muscle_name] = list of (global_vi, body_name, tri_idx,
        #                                  bary, normal_offset, weight)
        self.bindings = {}
        # Per-bone rest-pose triangle data, keyed by body name:
        #   { body_name: {'verts_local': (V,3), 'tris': (T,3),
        #                 'tri_normals_local': (T,3),
        #                 'rest_R': (3,3), 'rest_t': (3,)} }
        self.bone_rest = {}
        # Dynamic bindings for joint-crossing muscles.  Per-vert: keep
        # rest distance to nearest bone surface, and list of candidate
        # bone bodies to re-query per frame.  At runtime, the muscle vert
        # is pulled outward to maintain that rest distance from whichever
        # bone is closest now — accommodates muscles whose nearest bone
        # changes during joint flex.
        # dynamic_bindings[muscle_name] = list of dicts:
        #   {vi_local, rest_dist, weight, cand_bodies: [name,...]}
        self.dynamic_bindings = {}
        # Cache of bone trimesh (local-rest frame) for KDTree queries
        self.bone_trimesh_local = {}

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
            if muscle_name in NO_SKIN_PRIOR_MUSCLES:
                continue
            tet_v = getattr(mobj, 'tet_vertices', None)
            if tet_v is None:
                continue
            tet_v = np.asarray(tet_v, dtype=np.float64)
            # Exclude cap-attached / anchor verts — those are already pinned
            # to bone targets by the ARAP fixed-DOF mechanism; adding a skin
            # prior on top creates double-pull and makes cap regions fly
            # under transform blending.
            fixed_set = set(getattr(mobj, 'soft_body_fixed_vertices', []))
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
            # For each tet vert, collect per-bone (dist, cp, tri_idx) so we
            # can pick the top-K nearest bones for multi-bone weighted binding.
            per_bone_data = {}  # bn -> (dist[N], cp[N,3], tri_idx[N])
            for bn in cand_bones:
                tm = bone_tris[bn]
                cp, dist, tri_idx = trimesh.proximity.closest_point(tm, tet_v)
                per_bone_data[bn] = (dist, cp, tri_idx)
            # Bind verts to top-K nearest bones within max_bind_dist.
            # For non-DQS muscles only keep the single nearest bone (K=1)
            # — secondary bone contributes to bulge artefacts there.
            muscle_top_k = self.top_k if muscle_name in DQS_MUSCLES else 1
            entries = []
            for vi in range(len(tet_v)):
                if vi in fixed_set:
                    continue
                # Sort bones by distance for this vert
                cand_dist = sorted(
                    ((per_bone_data[bn][0][vi], bn) for bn in cand_bones),
                    key=lambda x: x[0],
                )
                used = 0
                for d, bn in cand_dist:
                    if used >= muscle_top_k:
                        break
                    if d > self.max_bind_dist:
                        break  # sorted, so no closer ones remain
                    w = float(self.strength * np.exp(-d / self.sigma))
                    if w < _MIN_WEIGHT:
                        continue
                    body_node, resolved_name = _body_node(bn)
                    if body_node is None:
                        continue
                    rest = self._record_bone_rest(resolved_name, body_node, bone_tris[bn])
                    tri_idx = int(per_bone_data[bn][2][vi])
                    tri = rest["tris"][tri_idx]
                    a = rest["verts_local"][tri[0]]
                    b = rest["verts_local"][tri[1]]
                    c = rest["verts_local"][tri[2]]
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
                        "rest_world": tet_v[vi].copy(),
                    })
                    used += 1
            self.bindings[muscle_name] = entries
            n_bound += len(entries)
        print(f"  Skin prior: {n_bound} verts bound across {len(self.bindings)} muscles")
        return n_bound > 0

    def compute_targets_dynamic(self, muscle_name, vert_offset, current_positions):
        """Dynamic per-frame nearest-bone target for joint-crossing muscles.

        Vectorized over all bound verts and bones — one batched trimesh
        closest_point query per bone, then numpy reduction to pick the
        closest bone per vert.
        """
        entries = self.dynamic_bindings.get(muscle_name)
        if not entries:
            return None
        n = len(entries)
        gi_arr = np.empty(n, dtype=np.int64)
        ws_arr = np.empty(n, dtype=np.float64)
        cur_world = np.empty((n, 3), dtype=np.float64)
        for k, e in enumerate(entries):
            gi_arr[k] = vert_offset + e["vi_local"]
            ws_arr[k] = e["weight"]
            cur_world[k] = current_positions[gi_arr[k]]
        # Gather union of candidate bones across all entries
        all_bones = set()
        for e in entries:
            all_bones.update(e["bone_rest_distances"].keys())
        all_bones = list(all_bones)
        # Per-bone batched closest-point + transform
        best_dist = np.full(n, np.inf)
        best_target = cur_world.copy()
        for bn in all_bones:
            rest = self.bone_rest[bn]
            wt = rest["body_node"].getWorldTransform()
            R_now = np.array(wt.rotation())
            t_now = np.array(wt.translation())
            # Transform all current world verts to this bone's local frame
            cur_local = (cur_world - t_now) @ R_now  # (n, 3)
            tm_local = self.bone_trimesh_local[bn]
            cp_local, dist, _ = trimesh.proximity.closest_point(tm_local, cur_local)
            cp_local = np.asarray(cp_local)
            dist = np.asarray(dist)
            # Per-vert rest distance for THIS bone (default large if missing)
            rest_dist_bn = np.array([
                e["bone_rest_distances"].get(bn, 1e9) for e in entries
            ])
            better = dist < best_dist
            if not np.any(better):
                continue
            best_dist[better] = dist[better]
            # Outward normal (per-vert).  Safe div.
            diff = cur_local[better] - cp_local[better]
            d_safe = np.where(dist[better] > 1e-9, dist[better], 1.0)
            n_local = diff / d_safe[:, None]
            target_local = cp_local[better] + rest_dist_bn[better, None] * n_local
            best_target[better] = target_local @ R_now.T + t_now
        return gi_arr, ws_arr, best_target

    @staticmethod
    def _rotmat_to_quat(R):
        """Convert 3x3 rotation matrix to unit quaternion (w, x, y, z)."""
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
            return np.array([0.25 / s,
                             (R[2, 1] - R[1, 2]) * s,
                             (R[0, 2] - R[2, 0]) * s,
                             (R[1, 0] - R[0, 1]) * s])
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                             (R[0, 1] + R[1, 0]) / s,
                             (R[0, 2] + R[2, 0]) / s])
        if R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            return np.array([(R[0, 2] - R[2, 0]) / s,
                             (R[0, 1] + R[1, 0]) / s, 0.25 * s,
                             (R[1, 2] + R[2, 1]) / s])
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return np.array([(R[1, 0] - R[0, 1]) / s,
                         (R[0, 2] + R[2, 0]) / s,
                         (R[1, 2] + R[2, 1]) / s, 0.25 * s])

    @staticmethod
    def _quat_to_rotmat(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    def compute_targets(self, muscle_name, vert_offset,
                        current_positions=None, outward_boost=1.0,
                        outward_max_extra=0.003):
        """Return (global_indices, weights, target_world_positions) for the
        given muscle's bound verts.  Uses CURRENT bone world transforms.

        Multi-bone bindings per vertex are aggregated via *transform
        blending*: each bone's RIGID motion (R_rel, t_rel = current bone
        transform composed with inverse rest transform) is blended with
        weights; rotations via NLERP on quaternions, translations linearly.
        This produces a smooth bend across joints — superior to position
        blending (LBS), which kinks when bone frames diverge at flexion.

        vert_offset: integer added to local vi to get global system index.
        current_positions: optional (N, 3) array of CURRENT global vertex
            positions.  If provided, enables outward-only asymmetric attractor:
            verts pushed closer to the bone than their rest normal-offset
            get a stronger target push outward (capped at outward_max_extra).
        outward_boost / outward_max_extra: shape and cap of the outward push.
        """
        entries = self.bindings.get(muscle_name)
        if not entries:
            return None

        # Group entries by vi_local.  For DQS muscles, multiple entries
        # per vi feed transform blending; for others precompute already
        # restricted bindings to K=1 so each vi has a single entry.
        by_vi = {}
        for e in entries:
            by_vi.setdefault(e["vi_local"], []).append(e)

        # Cache current bone (R_rel, t_rel, q_rel) per bone name this call.
        bone_motion = {}

        def _get_motion(body_name):
            m = bone_motion.get(body_name)
            if m is not None:
                return m
            rest = self.bone_rest[body_name]
            wt = rest["body_node"].getWorldTransform()
            R_now = np.array(wt.rotation())
            t_now = np.array(wt.translation())
            R_rest = rest["rest_R"]
            t_rest = rest["rest_t"]
            R_rel = R_now @ R_rest.T
            t_rel = t_now - R_rel @ t_rest
            q_rel = self._rotmat_to_quat(R_rel)
            m = (R_rel, t_rel, q_rel, R_now, t_now)
            bone_motion[body_name] = m
            return m

        n = len(by_vi)
        gi = np.empty(n, dtype=np.int64)
        ws = np.empty(n, dtype=np.float64)
        targets = np.empty((n, 3), dtype=np.float64)

        for k, (vi_local, evs) in enumerate(by_vi.items()):
            rest_world = evs[0]["rest_world"]
            # Hybrid LBS / DQS blend, chosen per-vert by the max pairwise
            # relative-rotation angle between bound bones:
            #   small angle (bones nearly aligned) → LBS (no bulge)
            #   large angle (joint flexed)         → DQS (no candy-wrap)
            # Inner-side-of-bend verts have ARAP collapse → bulge from DQS
            # makes pinch worse there.  LBS stays compact at small angles.
            w_total = 0.0
            # LBS accumulator
            lbs_target = np.zeros(3)
            # DQS accumulators
            qr_acc = np.zeros(4)
            qd_acc = np.zeros(4)
            q_ref = None
            per_bone_qr = []  # for angle measurement
            for e in evs:
                R_rel, t_rel, q_rel, R_now, t_now = _get_motion(e["body"])
                w_e = e["weight"]
                # LBS per-bone target
                lbs_target += w_e * (R_rel @ rest_world + t_rel)
                # DQS quat blend (hemisphere correct)
                if q_ref is None:
                    q_ref = q_rel
                    qr_e = q_rel
                else:
                    qr_e = q_rel if np.dot(q_ref, q_rel) >= 0 else -q_rel
                per_bone_qr.append(qr_e)
                tw, tx, ty, tz = 0.0, t_rel[0], t_rel[1], t_rel[2]
                rw, rx, ry, rz = qr_e
                qd_e = 0.5 * np.array([
                    tw * rw - tx * rx - ty * ry - tz * rz,
                    tw * rx + tx * rw + ty * rz - tz * ry,
                    tw * ry - tx * rz + ty * rw + tz * rx,
                    tw * rz + tx * ry - ty * rx + tz * rw,
                ])
                qr_acc += w_e * qr_e
                qd_acc += w_e * qd_e
                w_total += w_e
            lbs_target /= w_total
            # DQS extract
            qr_norm = np.linalg.norm(qr_acc)
            if qr_norm > 1e-12:
                qr_b = qr_acc / qr_norm
                qd_b = qd_acc / qr_norm
            else:
                qr_b = np.array([1.0, 0.0, 0.0, 0.0])
                qd_b = np.zeros(4)
            R_blend = self._quat_to_rotmat(qr_b)
            rw, rx, ry, rz = qr_b
            dw, dx, dy, dz = qd_b
            cw, cx, cy, cz = rw, -rx, -ry, -rz
            tx = 2.0 * (dw * cx + dx * cw + dy * cz - dz * cy)
            ty = 2.0 * (dw * cy - dx * cz + dy * cw + dz * cx)
            tz = 2.0 * (dw * cz + dx * cy - dy * cx + dz * cw)
            t_blend = np.array([tx, ty, tz])
            dqs_target = R_blend @ rest_world + t_blend
            # Choose blend factor by max pairwise inter-bone rotation angle.
            # Smoothstep over [LBS_angle_low, DQS_angle_high]: below low →
            # pure LBS (alpha=0); above high → pure DQS (alpha=1).
            if len(per_bone_qr) >= 2:
                max_cos = 1.0
                for i in range(len(per_bone_qr)):
                    for j in range(i + 1, len(per_bone_qr)):
                        c = abs(float(np.dot(per_bone_qr[i], per_bone_qr[j])))
                        c = min(1.0, max(-1.0, c))
                        if c < max_cos:
                            max_cos = c
                # Angle between unit quaternions: 2 * acos(|<q1,q2>|)
                angle = 2.0 * np.arccos(max_cos)
            else:
                angle = 0.0
            # Smoothstep blend: LBS up to 30°, DQS at 90°+.
            low = 30.0 * np.pi / 180.0
            high = 90.0 * np.pi / 180.0
            if angle <= low:
                alpha = 0.0
            elif angle >= high:
                alpha = 1.0
            else:
                t = (angle - low) / (high - low)
                alpha = t * t * (3.0 - 2.0 * t)
            target_world = (1.0 - alpha) * lbs_target + alpha * dqs_target
            # Optional outward-only target nudge using the blended frame:
            # if vert is currently CLOSER to nearest bone than its rest
            # normal offset, push target outward along the (rest-time)
            # nearest-bone normal.
            if current_positions is not None:
                gi_k = vert_offset + vi_local
                cur_world = current_positions[gi_k]
                e0 = evs[0]  # dominant bone (highest-weight at top of list? -- sorted by dist)
                rest = self.bone_rest[e0["body"]]
                if e0["normal_offset"] > 0.001:
                    tri = rest["tris"][e0["tri_idx"]]
                    a = rest["verts_local"][tri[0]]
                    b = rest["verts_local"][tri[1]]
                    c = rest["verts_local"][tri[2]]
                    n_local = rest["tri_normals_local"][e0["tri_idx"]]
                    cp_local = e0["bary"][0] * a + e0["bary"][1] * b + e0["bary"][2] * c
                    # cur in DOMINANT bone's current local frame
                    _, _, _, R_now0, t_now0 = _get_motion(e0["body"])
                    cur_local0 = (cur_world - t_now0) @ R_now0
                    cur_offset = float((cur_local0 - cp_local) @ n_local)
                    penetration = e0["normal_offset"] - cur_offset
                    if penetration > 0:
                        extra = min(penetration * outward_boost, outward_max_extra)
                        n_world_now = R_now0 @ n_local
                        target_world = target_world + extra * n_world_now
            gi[k] = vert_offset + vi_local
            ws[k] = w_total
            targets[k] = target_world
        return gi, ws, targets
