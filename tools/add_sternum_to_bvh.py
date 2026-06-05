"""Add a Sternum joint to a *_vert.bvh by:
1. Tracking 20 rib endpoints (L_Rib1..10, R_Rib1..10, vertex on each rib
   closest to sternum) — each endpoint is rigidly bound to its anatomical
   parent vertebra (Rib_i ↔ T_i).
2. Kabsch-fitting the sternum's matching contact points onto the per-frame
   rib endpoints → world rigid transform of sternum per frame.
3. Single-bone LBS solve over thoracic vertebrae (T1..T12) to pick the
   parent that best explains the sternum transform with a constant local
   offset. Per-frame local rotation extracted as ZXY Euler for BVH.
4. Inserting "JOINT Sternum" into the BVH hierarchy under the chosen
   parent and appending per-frame channels to MOTION.

Usage: python tools/add_sternum_to_bvh.py --in data/motion/run_vert.bvh \
                                          --out data/motion/run_vert_sternum.bvh
"""
import argparse
import os
import re
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH

SKEL_XML = "data/zygote_skel.xml"
ZYGOTE_DIR = "Zygote_Meshes_251229"
STERNUM_OBJ = os.path.join(ZYGOTE_DIR, "Skeleton/Thorax/Sternum.obj")
RIB_DIR = os.path.join(ZYGOTE_DIR, "Skeleton")
THORACIC_BODIES = [f"T{i}0" for i in range(1, 13)]  # T10..T120 in DART naming


def load_rib_endpoints():
    """For each of L/R_Rib1..10, find vertex closest to sternum mesh.
    Returns (endpoints_world (20,3), endpoint_parent_body (list of 20 strings),
    sternum_contacts_world (20,3), sternum_vertices (Nx3))."""
    if not os.path.exists(STERNUM_OBJ):
        sys.exit(f"Sternum OBJ not found: {STERNUM_OBJ}")
    sternum_tri = trimesh.load_mesh(STERNUM_OBJ)
    # OBJ vertices are in centimeters; DART skel is in meters. Scale to m.
    sternum_verts = np.asarray(sternum_tri.vertices) * 0.01
    print(f"Sternum: {len(sternum_verts)} vertices")

    sternum_tree = cKDTree(sternum_verts)

    endpoints = []
    parents = []
    contacts = []
    for side in ("L", "R"):
        for i in range(1, 11):  # ribs 1..10
            rib_path = os.path.join(RIB_DIR, f"{side}_Rib{i}.obj")
            if not os.path.exists(rib_path):
                print(f"  MISSING {rib_path}")
                continue
            rib_tri = trimesh.load_mesh(rib_path)
            rib_verts = np.asarray(rib_tri.vertices) * 0.01  # cm→m
            # Anatomical anterior tip = vertex with max +Z (Zygote: +Z anterior).
            # Nearest-to-sternum picks lateral midshaft points for ribs 5+ where
            # the rib's anterior end is far from sternum surface.
            best = int(np.argmax(rib_verts[:, 2]))
            endpoint_world = rib_verts[best]
            # Treat rib anterior tip as the "virtual contact" — i.e. sternum
            # has a rigid bind point at the rib endpoint position at REST.
            # Kabsch then finds rigid sternum transform that keeps these bind
            # points aligned with rib endpoints during motion. Rest residual
            # is exactly 0 (P=Q at rest). Anatomically: costal cartilage acts
            # as a rigid extension of the rib into the sternum.
            endpoints.append(endpoint_world)
            contacts.append(endpoint_world.copy())
            parents.append(f"T{i}0")  # rib i → T_i in DART naming
            print(f"  {side}_Rib{i}: tip=({endpoint_world[0]:.3f},"
                  f"{endpoint_world[1]:.3f},{endpoint_world[2]:.3f})")
    endpoints = np.asarray(endpoints, dtype=np.float64)
    contacts = np.asarray(contacts, dtype=np.float64)
    print(f"  {len(endpoints)} rib endpoints found")
    return endpoints, parents, contacts, sternum_verts


def rest_pose_vertebra_transforms(skel, body_names):
    """Reset skel + return dict body_name -> 4x4 world transform at rest."""
    skel.resetPositions()
    T = {}
    for bn in body_names:
        b = skel.getBodyNode(bn)
        if b is None:
            print(f"  WARN body not in skel: {bn}")
            continue
        T[bn] = np.asarray(b.getWorldTransform().matrix(), dtype=np.float64)
    return T


def transform_inverse(T):
    R_mat = T[:3, :3]
    t_vec = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R_mat.T
    out[:3, 3] = -R_mat.T @ t_vec
    return out


def kabsch(P, Q, weights=None):
    """Find R, t such that R @ P^T + t ≈ Q^T. P, Q are (N, 3).
    Optional per-point weights (N,). Higher weight = tighter fit."""
    if weights is None:
        weights = np.ones(len(P))
    w = np.asarray(weights, dtype=np.float64)
    w_sum = w.sum()
    cP = (w[:, None] * P).sum(axis=0) / w_sum
    cQ = (w[:, None] * Q).sum(axis=0) / w_sum
    Pc = P - cP
    Qc = Q - cQ
    H = (Pc * w[:, None]).T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R_mat = Vt.T @ D @ U.T
    t_vec = cQ - R_mat @ cP
    return R_mat, t_vec


def solve_endpoint_single_bone(target_world, R_b, t_b):
    """target_world (T,3), R_b (T,3,3) parent rotation, t_b (T,3) parent
    translation. Solve constant local = mean(R_b.T @ (target - t_b))."""
    diff = target_world - t_b
    local = np.einsum('tij,tj->i', np.transpose(R_b, (0, 2, 1)), diff) / R_b.shape[0]
    recon = np.einsum('tij,j->ti', R_b, local) + t_b
    resid = np.linalg.norm(target_world - recon, axis=1)
    return local, resid


def write_bvh_with_sternum(in_path, out_path, parent_body_name,
                           joint_offset, per_frame_euler_zxy,
                           src_to_bvh_name):
    """Insert 'JOINT Sternum' under the BVH joint matching parent_body_name's
    bvh attribute. Append 3 channels per frame for sternum rotation."""
    with open(in_path) as f:
        text = f.read()

    # Parent body's bvh joint name in the BVH file
    parent_bvh = src_to_bvh_name.get(parent_body_name)
    if parent_bvh is None:
        sys.exit(f"Parent body {parent_body_name} has no bvh mapping; can't insert Sternum")
    print(f"  Inserting Sternum as child of BVH joint '{parent_bvh}' "
          f"(skel body '{parent_body_name}')")

    # Split into hierarchy + motion
    motion_idx = text.find("\nMOTION\n")
    if motion_idx < 0:
        sys.exit("BVH MOTION section not found")
    hier_text = text[:motion_idx + 1]
    motion_text = text[motion_idx + 1:]  # starts with "MOTION\n"

    # Locate the line "JOINT parent_bvh" followed by its '{', then insert
    # a new JOINT block before the closing '}' of the parent.
    lines = hier_text.splitlines()
    parent_line_idx = None
    for i, ln in enumerate(lines):
        m = re.match(r'^(\s*)JOINT\s+' + re.escape(parent_bvh) + r'\s*$', ln)
        if m:
            parent_line_idx = i
            parent_indent = len(m.group(1))
            break
    if parent_line_idx is None:
        sys.exit(f"Could not find 'JOINT {parent_bvh}' line in BVH hierarchy")

    # Find matching closing '}' for this JOINT block
    depth = 0
    insert_idx = None
    for i in range(parent_line_idx + 1, len(lines)):
        s = lines[i].strip()
        if s == "{":
            depth += 1
        elif s == "}":
            depth -= 1
            if depth == 0:
                insert_idx = i  # insert before this line
                break
    if insert_idx is None:
        sys.exit(f"Unbalanced braces for {parent_bvh} block")

    inner_indent = "\t" * (parent_indent // 1 + 1)  # one tab deeper
    block = [
        inner_indent + "JOINT Sternum",
        inner_indent + "{",
        inner_indent + "\tOFFSET " + " ".join(f"{x:.6f}" for x in joint_offset),
        inner_indent + "\tCHANNELS 3 Zrotation Xrotation Yrotation",
        inner_indent + "\tEnd Site",
        inner_indent + "\t{",
        inner_indent + "\t\tOFFSET 0 0 0",
        inner_indent + "\t}",
        inner_indent + "}",
    ]
    new_lines = lines[:insert_idx] + block + lines[insert_idx:]
    new_hier = "\n".join(new_lines)

    # Compute column position to insert Sternum's 3 channels in motion rows.
    # BVH MOTION columns follow pre-order hierarchy traversal of CHANNELS.
    # Walk the NEW hierarchy line-by-line, sum channel counts of every joint
    # whose JOINT/ROOT block opens BEFORE Sternum's JOINT line.
    sternum_jline = None
    for k, ln in enumerate(new_lines):
        if re.match(r'^\s*JOINT\s+Sternum\s*$', ln):
            sternum_jline = k
            break
    if sternum_jline is None:
        sys.exit("Sternum JOINT line not found in new hierarchy")
    col_before = 0
    for k in range(sternum_jline):
        m_ch = re.match(r'^\s*CHANNELS\s+(\d+)\s', new_lines[k])
        if m_ch:
            col_before += int(m_ch.group(1))
    print(f"  Sternum channel insert column: {col_before}")

    # MOTION: insert 3 channels per frame at col_before.
    motion_lines = motion_text.splitlines()
    out_motion = []
    frame_i = 0
    for ln in motion_lines:
        s = ln.strip()
        if not s or s.startswith("MOTION") or s.startswith("Frames") or s.startswith("Frame Time"):
            out_motion.append(ln)
            continue
        vals = ln.split()
        ez, ex, ey = per_frame_euler_zxy[frame_i]
        new_vals = (vals[:col_before]
                    + [f"{ez:.6f}", f"{ex:.6f}", f"{ey:.6f}"]
                    + vals[col_before:])
        out_motion.append(" ".join(new_vals))
        frame_i += 1
    if frame_i != len(per_frame_euler_zxy):
        sys.exit(f"frame count mismatch: BVH {frame_i} vs euler {len(per_frame_euler_zxy)}")

    with open(out_path, "w") as f:
        f.write(new_hier + "\n" + "\n".join(out_motion) + "\n")
    print(f"  Wrote {out_path} (+{frame_i} frames × 3 sternum channels)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="bvh_in", required=True)
    ap.add_argument("--out", dest="bvh_out", required=True)
    ap.add_argument("--parent", default=None,
                    help="Force parent body (e.g. T50). Default: pick lowest residual.")
    ap.add_argument("--rib-weights", default=None,
                    help="10 comma-sep floats, weight for ribs 1..10 (both "
                         "sides share). Default: rib1 (manubrium) heavy, "
                         "lower ribs lighter — sternum pivots near manubrium "
                         "instead of swinging the full body.")
    args = ap.parse_args()
    if args.rib_weights:
        rib_w = np.array([float(x) for x in args.rib_weights.split(",")])
        assert rib_w.shape == (10,), "need 10 weights"
        rib_weights = np.concatenate([rib_w, rib_w])  # L + R
    else:
        # Manubrium-heavy default: weights decrease linearly from rib1 to
        # rib10 (10..1). Anchors sternum top tightly while lower ribs
        # maintain relative geometry with progressively less influence.
        rib_w = np.linspace(10.0, 1.0, 10)
        rib_weights = np.concatenate([rib_w, rib_w])
    print(f"Rib weights: {rib_weights[:10]} (L=R)")

    # 1. Build skel
    skel_info, root_name, bvh_info, _, mesh_info, _ = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)
    print(f"Skel: {skel.getNumBodyNodes()} bodies, {skel.getNumDofs()} DOFs")

    # 2. Rib endpoints + sternum contacts (rest pose, world coords)
    endpoints_world, parent_bodies, contacts_world, _ = load_rib_endpoints()

    # 3. Rest-pose vertebra transforms
    rest_T = rest_pose_vertebra_transforms(skel, THORACIC_BODIES)
    # Convert rib endpoints to local-in-parent-vertebra coords
    endpoints_local = []
    for ep_world, parent in zip(endpoints_world, parent_bodies):
        if parent not in rest_T:
            sys.exit(f"missing rest transform for {parent}")
        T_inv = transform_inverse(rest_T[parent])
        ep_h = np.append(ep_world, 1.0)
        ep_local = (T_inv @ ep_h)[:3]
        endpoints_local.append(ep_local)
    endpoints_local = np.asarray(endpoints_local)

    # Sternum contacts: keep world (rest); we'll Kabsch fit using these
    # as the source set against per-frame rib endpoint targets.
    contacts_rest = contacts_world

    # 4. Load BVH + drive skel per frame
    motion = MyBVH(args.bvh_in, bvh_info, skel)
    n_frames = motion.mocap_refs.shape[0]
    print(f"BVH frames: {n_frames}")

    # Cache per-frame vertebra world transforms (we'll need T1..T12)
    body_T = {bn: np.zeros((n_frames, 4, 4)) for bn in THORACIC_BODIES}
    saved_pos = skel.getPositions().copy()
    for f in range(n_frames):
        skel.setPositions(motion.mocap_refs[f])
        for bn in THORACIC_BODIES:
            b = skel.getBodyNode(bn)
            if b is not None:
                body_T[bn][f] = np.asarray(b.getWorldTransform().matrix())
    skel.setPositions(saved_pos)

    # 5. Per-frame: world rib endpoint = T_parent(f) @ endpoint_local
    # Track sternum reference point = MID-POINT between L_Rib1 + R_Rib1
    # anterior tips. Pivot at the sternal notch / manubrium superior border
    # so upper sternum stays close to rib 1 across motion; lower sternum
    # swings to absorb the rib 2-10 deviations.
    # endpoints_world index: 0..9 = L_Rib1..10, 10..19 = R_Rib1..10
    anchor_rest = (endpoints_world[0] + endpoints_world[10]) / 2.0
    print(f"Sternum anchor (rib1 midpoint, world rest): {anchor_rest}")
    sternum_R = np.zeros((n_frames, 3, 3))
    sternum_t = np.zeros((n_frames, 3))  # world pos of manubrium anchor
    centroid_rest = anchor_rest  # for residual diagnostic reuse
    for f in range(n_frames):
        eps_world = np.zeros((len(endpoints_local), 3))
        for k, parent in enumerate(parent_bodies):
            T = body_T[parent][f]
            ep_h = np.append(endpoints_local[k], 1.0)
            eps_world[k] = (T @ ep_h)[:3]
        R_fit, t_fit = kabsch(contacts_rest, eps_world, weights=rib_weights)
        sternum_R[f] = R_fit
        # Track manubrium anchor (rib1 midpoint) world position per frame.
        sternum_t[f] = R_fit @ anchor_rest + t_fit

    # 6. For each candidate parent in T1..T12, solve single-bone LBS for
    #    constant local offset; pick lowest residual.
    # Cache rest-pose parent rotations. With XML joint_r = identity and
    # sternum body_r = identity:
    # R_local(f) = R_parent_rest @ R_parent(f).T @ R_sternum_world(f)
    # (left-multiply by R_parent_rest absorbs the rest joint→parent offset).
    rest_R = {}
    saved = skel.getPositions().copy()
    skel.resetPositions()
    for cand in THORACIC_BODIES:
        rest_R[cand] = np.asarray(
            skel.getBodyNode(cand).getWorldTransform().matrix(),
            dtype=np.float64)[:3, :3]
    skel.setPositions(saved)

    best_parent = None
    best_local = None
    best_resid = float('inf')
    best_locrots = None
    print("\nParent search (T1..T12):")
    for cand in THORACIC_BODIES:
        R_b = body_T[cand][:, :3, :3]
        t_b = body_T[cand][:, :3, 3]
        local, resid = solve_endpoint_single_bone(sternum_t, R_b, t_b)
        # Per-frame local rotation in joint frame (joint_r = identity in XML,
        # so joint frame = world at rest; left-multiply by R_parent_rest to
        # absorb the static parent→joint offset).
        loc_rots = rest_R[cand] @ np.transpose(R_b, (0, 2, 1)) @ sternum_R
        max_r = float(np.max(resid)) * 1000
        p95 = float(np.percentile(resid, 95)) * 1000
        med = float(np.median(resid)) * 1000
        mark = ""
        if p95 < best_resid:
            best_resid = p95
            best_parent = cand
            best_local = local
            best_locrots = loc_rots
            mark = " *"
        print(f"  {cand}: pos resid med={med:.1f}mm p95={p95:.1f}mm max={max_r:.1f}mm{mark}")
    if args.parent is not None:
        # Override: recompute solution for forced parent
        R_b = body_T[args.parent][:, :3, :3]
        t_b = body_T[args.parent][:, :3, 3]
        best_local, resid = solve_endpoint_single_bone(sternum_t, R_b, t_b)
        best_locrots = rest_R[args.parent] @ np.transpose(R_b, (0, 2, 1)) @ sternum_R
        best_parent = args.parent
        print(f"\nForced parent: {best_parent} (p95={float(np.percentile(resid, 95))*1000:.1f}mm)")
        # Per-rib residual diagnostic: |Kabsch-predicted sternum_i - rib_endpoint_i| per frame.
        # Sternum_t[f] = R[f] @ anchor_rest + t_kabsch[f]
        # → R @ contacts_rest[k] + t_kabsch[f] = R @ (contacts_rest[k] - anchor_rest) + sternum_t[f]
        per_rib_max = []
        for k, parent in enumerate(parent_bodies):
            errs = []
            for f in range(n_frames):
                ep_world = (body_T[parent][f] @ np.append(endpoints_local[k], 1.0))[:3]
                rec_contact = sternum_R[f] @ (contacts_rest[k] - anchor_rest) + sternum_t[f]
                errs.append(np.linalg.norm(rec_contact - ep_world))
            per_rib_max.append(max(errs) * 1000)
        print("Per-rib max residual (mm):")
        for i in range(10):
            print(f"  Rib{i+1}: L={per_rib_max[i]:.1f}  R={per_rib_max[i+10]:.1f}")
    else:
        print(f"\nBest parent: {best_parent}")
    print(f"Local offset (m): {best_local}")

    # 7. Convert per-frame local rotation to ZXY Euler (deg)
    euler_zxy = R.from_matrix(best_locrots).as_euler("ZXY", degrees=True)

    # 8. Write new BVH with Sternum joint inserted
    write_bvh_with_sternum(
        args.bvh_in, args.bvh_out,
        parent_body_name=best_parent,
        joint_offset=best_local,
        per_frame_euler_zxy=euler_zxy,
        src_to_bvh_name={bn: skel_info[bn]['bvh']
                         for bn in skel_info
                         if 'bvh' in skel_info[bn]},
    )


if __name__ == "__main__":
    main()
