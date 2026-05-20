"""Per-pose moment-arm fiber clustering analysis.

For a chosen R-side muscle:
  1. Load cached waypoints over walk.bvh (132 frames).
  2. Build per-fiber length L_i(t) from waypoint-polyline segment sums.
  3. Pull leg joint angles q(t) from BVH.
  4. Per pose, compute local moment-arm m_i(t, j) via sliding-window
     least-squares: fit L_i ≈ m_i^T q + b inside a [t-w, t+w] window.
  5. Stack m_i over (frames × DOFs) -> feature vector per fiber.
  6. Hierarchical-cluster fibers; render dendrogram + per-cluster
     moment-arm curves + 3D rest-pose fiber paths colored by cluster.
  7. Save PNG.

Usage:
  python tools/fiber_moment_arm_analyze.py --muscle R_Rectus_Femoris
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# === DOF indices in BVH for R / L leg (hip 3 + knee 1 + ankle 3) ===
R_DOF_INDICES = [18, 19, 20, 21, 22, 23, 24]
L_DOF_INDICES = [6, 7, 8, 9, 10, 11, 12]
DOF_NAMES = ['hip_x', 'hip_y', 'hip_z', 'knee', 'ankle_x', 'ankle_y', 'ankle_z']


def load_waypoints(cache_dir, muscle):
    """Load per-frame waypoints from chunked NPZ.

    Returns (waypoints_per_frame_flat, shape_per_stream_level, n_frames)
      waypoints_per_frame_flat: (n_frames, n_wp, 3)
      shape: list[list[int]]  fiber count per (stream, level)
    """
    files = sorted(glob.glob(os.path.join(cache_dir, f'{muscle}_chunk_*.npz')))
    if not files:
        raise FileNotFoundError(f'no chunks for {muscle} in {cache_dir}')
    all_frames = []
    all_wp = []
    shape = None
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        frames = d['frames']
        wf = d['waypoints_flat']            # (n_in_chunk, n_wp*3)
        n_wp = wf.shape[1] // 3
        wp = wf.reshape(wf.shape[0], n_wp, 3)
        all_frames.append(np.asarray(frames, dtype=np.int64))
        all_wp.append(wp)
        if shape is None:
            shape_raw = d['waypoints_shape']
            shape_str = shape_raw.item().decode() if hasattr(shape_raw, 'item') else shape_raw[()]
            shape = json.loads(shape_str)
    all_frames = np.concatenate(all_frames)
    all_wp = np.concatenate(all_wp, axis=0)
    order = np.argsort(all_frames)
    return all_wp[order], shape, len(all_frames)


def fiber_lengths(wp_per_frame, shape):
    """Build L_i(t) per fiber.

    Args:
      wp_per_frame: (T, n_wp, 3)
      shape: list[list[int]] fiber counts per (stream, level)

    Returns:
      L: dict[stream_idx] -> (T, n_fibers_in_stream) length per frame
      fiber_pts_per_stream: dict[stream_idx] -> (T, n_fibers, n_levels, 3) for plotting
    """
    T = wp_per_frame.shape[0]
    L = {}
    fiber_pts = {}
    offset = 0
    for s_idx, levels in enumerate(shape):
        n_levels = len(levels)
        n_fibers = levels[0]  # assume same fiber count per level within stream
        # Sanity: all levels same count
        if not all(n == n_fibers for n in levels):
            print(f'WARN stream {s_idx}: mixed fiber counts {levels}, taking min')
            n_fibers = min(levels)
        # Gather (T, n_levels, n_fibers, 3)
        stream_pts = np.zeros((T, n_levels, n_fibers, 3), dtype=np.float32)
        for l_idx, n in enumerate(levels):
            stream_pts[:, l_idx] = wp_per_frame[:, offset:offset + n_fibers]
            offset += n
        # Transpose to (T, n_fibers, n_levels, 3)
        stream_pts = stream_pts.transpose(0, 2, 1, 3)
        fiber_pts[s_idx] = stream_pts
        # Segment vectors: between consecutive levels
        seg = stream_pts[:, :, 1:] - stream_pts[:, :, :-1]  # (T, F, L-1, 3)
        seg_len = np.linalg.norm(seg, axis=-1)  # (T, F, L-1)
        L_s = seg_len.sum(axis=-1)  # (T, F)
        L[s_idx] = L_s
    return L, fiber_pts


def load_bvh_q(bvh_path, side='R'):
    """Return q(t) of shape (T, 7) for one leg by parsing BVH via core.bvhparser."""
    from core.dartHelper import saveSkeletonInfo
    from core.bvhparser import MyBVH
    skel_info, root_name, bvh_info, _, _, _ = saveSkeletonInfo('data/zygote_skel.xml')
    # Need a skeleton instance just to parse BVH; build it.
    from core.dartHelper import buildFromInfo
    skel = buildFromInfo(skel_info, root_name)
    # Detect T-frame
    from viewer.zygote_mesh_ui import _detect_bvh_tframe
    t_frame = _detect_bvh_tframe(bvh_path)
    motion = MyBVH(bvh_path, bvh_info, skel, T_frame=t_frame)
    q_all = motion.mocap_refs  # (T, N_dofs)
    idx = R_DOF_INDICES if side == 'R' else L_DOF_INDICES
    return q_all[:, idx]


def local_moment_arm(L_t, q_t, window=10, ridge=1e-4):
    """For each frame t, fit L_t[t-w:t+w] ~ m^T q_t[t-w:t+w] + b via ridge LS.

    Boundary frames (where window is truncated) return NaN to be masked out.
    Ridge regularization stabilizes ill-conditioned windows.

    Args:
      L_t: (T,) fiber length time series
      q_t: (T, n_dof)
      window: half-window
      ridge: L2 penalty on slope (regularizes when q variation is tiny)

    Returns:
      M: (T, n_dof) moment arm per frame; NaN at boundaries.
    """
    T, n_dof = q_t.shape
    M = np.full((T, n_dof), np.nan, dtype=np.float32)
    eye = np.eye(n_dof) * ridge
    for t in range(window, T - window):
        lo = t - window
        hi = t + window + 1
        Y = L_t[lo:hi]
        X = q_t[lo:hi]
        Yc = Y - Y.mean()
        Xc = X - X.mean(axis=0, keepdims=True)
        # Ridge LS: (X^T X + lambda I) m = X^T Y
        try:
            beta = np.linalg.solve(Xc.T @ Xc + eye, Xc.T @ Yc)
            M[t] = beta.astype(np.float32)
        except np.linalg.LinAlgError:
            M[t] = 0.0
    return M


def cluster_fibers(features, n_clusters=8, method='ward'):
    """Hierarchical clustering on feature matrix (n_fibers, dim)."""
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(features, method=method)
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    return labels, Z


def visualize(muscle, fiber_pts, M_per_fiber, labels, Z, out_path):
    """Save dendrogram + per-cluster MA curves + 3D rest fibers as PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    n_fibers = M_per_fiber.shape[0]
    T = M_per_fiber.shape[1]
    n_dof = M_per_fiber.shape[2]
    n_clusters = labels.max()

    fig = plt.figure(figsize=(18, 12))

    # 1. Dendrogram (top-left)
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2)
    dendrogram(Z, no_labels=True, color_threshold=Z[-(n_clusters - 1), 2] if n_clusters >= 2 else None,
               ax=ax1)
    ax1.set_title(f'Fiber dendrogram ({muscle}, K={n_clusters})')
    ax1.set_xlabel('fiber index'); ax1.set_ylabel('linkage distance')

    # 2. Per-cluster mean moment-arm trajectories per DOF (top-right)
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2)
    cmap = plt.cm.tab10
    M_clean = np.nan_to_num(M_per_fiber, nan=0.0)
    for k in range(1, n_clusters + 1):
        mask = labels == k
        if not mask.any():
            continue
        cluster_M = M_clean[mask].mean(axis=0)  # (T, n_dof)
        dominant = np.argmax(np.abs(cluster_M).mean(axis=0))
        ax2.plot(cluster_M[:, dominant], color=cmap((k - 1) % 10),
                 label=f'C{k} ({mask.sum()}f, {DOF_NAMES[dominant]})')
    ax2.set_title('Per-cluster mean moment-arm (dominant DOF)')
    ax2.set_xlabel('frame'); ax2.set_ylabel('moment arm [m]')
    ax2.legend(loc='upper right', fontsize=7)
    ax2.axhline(0, color='gray', lw=0.5)

    # 3. Per-DOF mean moment arm across all clusters (mid row)
    ax3 = plt.subplot2grid((3, 4), (1, 0), colspan=4)
    avg_M_per_dof = np.nanmean(M_per_fiber, axis=0)  # (T, n_dof)
    for j in range(n_dof):
        ax3.plot(avg_M_per_dof[:, j], label=DOF_NAMES[j])
    ax3.set_title('Mean moment-arm across all fibers, per DOF')
    ax3.set_xlabel('frame'); ax3.set_ylabel('moment arm [m]'); ax3.legend(fontsize=8)
    ax3.axhline(0, color='gray', lw=0.5)

    # 4. 3D rest-pose fiber paths colored by cluster (Z is up).
    # `fiber_pts` here is a list of (L_s, 3) polylines (one per fiber).
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    poly_list = fiber_pts
    all_pts = np.concatenate(poly_list, axis=0)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    centers = (mins + maxs) / 2
    half = (maxs - mins).max() / 2 * 1.1

    # DART convention: world Y is up (verified via pelvis world transform).
    # Plot with Y as matplotlib z-axis so the vertical axis matches.
    def setup_axes(ax, elev, azim, title):
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(centers[0] - half, centers[0] + half)
        ax.set_ylim(centers[2] - half, centers[2] + half)
        ax.set_zlim(centers[1] - half, centers[1] + half)
        ax.set_xlabel('X (lateral)'); ax.set_ylabel('Z (anteroposterior)'); ax.set_zlabel('Y (up)')
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(title, fontsize=9)

    for v_idx, (elev, azim, title) in enumerate([
        (10, -60, 'oblique'),
        (0, -90, 'anterior view'),
        (0, 0, 'lateral view'),
    ]):
        ax = fig.add_subplot(3, 3, 7 + v_idx, projection='3d')
        for i, poly in enumerate(poly_list):
            c = cmap((labels[i] - 1) % 10)
            # Swap Y and Z so plot z-axis = anatomical up
            ax.plot(poly[:, 0], poly[:, 2], poly[:, 1],
                    color=c, lw=1.0, alpha=0.7)
        setup_axes(ax, elev, azim, title)

    plt.suptitle(f'Moment-arm clustering — {muscle}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--muscle', default='R_Rectus_Femoris')
    ap.add_argument('--cache-dir', default='data/motion_cache/walk/R_UpLeg')
    ap.add_argument('--bvh', default='data/motion/walk.bvh')
    ap.add_argument('--side', default='R', choices=['L', 'R'])
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--clusters', type=int, default=8)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    print(f'Loading waypoints for {args.muscle} ...')
    wp_t, shape, T = load_waypoints(args.cache_dir, args.muscle)
    print(f'  frames={T} streams={len(shape)} levels per stream={[len(s) for s in shape]}')

    L_per_stream, pts_per_stream = fiber_lengths(wp_t, shape)
    # Concatenate across streams for joint clustering across the muscle.
    L_all = np.concatenate([L_per_stream[s] for s in sorted(L_per_stream.keys())], axis=1)
    n_fibers = L_all.shape[1]
    print(f'  total fibers (across streams) = {n_fibers}')

    # Build per-fiber polylines at REST POSE from the tet file's stored
    # `waypoints` field (computed at T-pose during muscle build).
    import pickle
    tet_path = os.path.join('tet', f'{args.muscle}_tet.npz')
    with open(tet_path, 'rb') as f:
        tet_data = pickle.load(f)
    rest_wps = tet_data['waypoints']
    fiber_polylines = []
    for s_idx, stream in enumerate(rest_wps):
        n_levels = len(stream)
        if n_levels == 0:
            continue
        # Treat stream as (levels, n_fibers, 3); fiber i polyline = list of n_levels (3,) points
        n_f = min(np.asarray(c).shape[0] for c in stream if np.asarray(c).size > 0)
        if n_f == 0:
            continue
        stream_arrs = [np.asarray(c, dtype=np.float32)[:n_f] for c in stream]
        for i in range(n_f):
            poly = np.stack([a[i] for a in stream_arrs], axis=0)  # (n_levels, 3)
            fiber_polylines.append(poly)

    print(f'Loading BVH joint angles ({args.side} side) ...')
    q = load_bvh_q(args.bvh, side=args.side)
    print(f'  q shape: {q.shape}')

    print(f'Computing per-pose moment arms (window={args.window}) ...')
    M = np.zeros((n_fibers, T, q.shape[1]), dtype=np.float32)
    for i in range(n_fibers):
        M[i] = local_moment_arm(L_all[:, i], q, window=args.window)

    # Mask out boundary frames (NaN) for clustering feature
    valid = ~np.isnan(M).any(axis=(0, 2))  # frames where all fibers are valid
    print(f'  valid frames: {valid.sum()}/{T}')
    features = M[:, valid, :].reshape(n_fibers, -1)
    print(f'  feature dim per fiber: {features.shape[1]}')

    print(f'Clustering into K={args.clusters} ...')
    labels, Z = cluster_fibers(features, n_clusters=args.clusters)
    print(f'  cluster sizes: {np.bincount(labels)[1:]}')

    out = args.out or f'analysis_images/fiber_moment_arm_{args.muscle}.png'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    visualize(args.muscle, fiber_polylines, M, labels, Z, out)
    # Quick stats for batch summary
    M_clean = np.nan_to_num(M, nan=0.0)
    mean_M = M_clean.mean(axis=(0, 1))  # over fibers, frames -> (n_dof,)
    per_fiber_mean_M = M_clean.mean(axis=1)  # (n_fibers, n_dof)
    # sign-flip detection: any DOF where fibers span both + and - side
    sign_flip = []
    for j, name in enumerate(DOF_NAMES):
        v = per_fiber_mean_M[:, j]
        if v.max() > 0.005 and v.min() < -0.005:
            sign_flip.append(name)
    print(f'STATS muscle={args.muscle} mean_MA={dict(zip(DOF_NAMES, mean_M.round(4)))}')
    if sign_flip:
        print(f'STATS muscle={args.muscle} sign_flip_DOFs={sign_flip}  cluster_sizes={np.bincount(labels)[1:].tolist()}')


if __name__ == '__main__':
    main()
