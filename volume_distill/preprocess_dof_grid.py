"""Preprocess DOF grid bake results into training data for the DOF-based NN.

Loads per-region chunk files from data/motion_cache/dof_grid/{L_UpLeg,L_LowLeg}/,
transforms tet positions to pelvis-local frame, computes displacements from rest,
and saves a single preprocessed.pt with PCA targets.

Usage: python -m volume_distill.preprocess_dof_grid
"""
import os
import glob
import json
import numpy as np
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_BASE = os.path.join(PROJECT_ROOT, "data", "motion_cache", "dof_grid")
TET_DIR = os.path.join(PROJECT_ROOT, "tet")
SKEL_XML = os.path.join(PROJECT_ROOT, "data", "zygote_skel.xml")
OUTPUT_PATH = os.path.join(GRID_BASE, "preprocessed.pt")

# All 7 DOFs that were sampled
DOF_INDICES = [6, 7, 8, 9, 10, 11, 12]

# PCA components to keep per muscle
PCA_K = 64


def load_region_data(region_dir):
    """Load all chunk data for one region. Returns {mname: (N, V, 3)} positions + dof_values."""
    meta_path = os.path.join(region_dir, "meta.json")
    if not os.path.exists(meta_path):
        print(f"  No meta.json in {region_dir}, skipping")
        return {}, None, None

    with open(meta_path) as f:
        meta = json.load(f)

    muscle_names = meta["muscles"]

    # Load DOF values (from dofs_chunk files)
    dof_chunks = sorted(glob.glob(os.path.join(region_dir, "dofs_chunk_*.npz")))
    all_dof_values = []
    all_sample_indices = []
    for dc in dof_chunks:
        d = np.load(dc)
        all_sample_indices.append(d["sample_indices"])
        all_dof_values.append(d["dof_values"])

    if not all_dof_values:
        print(f"  No DOF chunks in {region_dir}")
        return {}, None, None

    dof_values = np.concatenate(all_dof_values, axis=0)  # (N, 7)
    sample_indices = np.concatenate(all_sample_indices, axis=0)

    # Load per-muscle positions
    muscle_data = {}
    for mname in muscle_names:
        chunks = sorted(glob.glob(os.path.join(region_dir, f"{mname}_chunk_*.npz")))
        if not chunks:
            print(f"  WARNING: No chunks for {mname}")
            continue
        positions_list = []
        frames_list = []
        for cp in chunks:
            d = np.load(cp)
            frames_list.append(d["frames"])
            positions_list.append(d["positions"])

        positions = np.concatenate(positions_list, axis=0)  # (N, V, 3)
        frames = np.concatenate(frames_list, axis=0)

        if len(positions) != len(dof_values):
            print(f"  WARNING: {mname} has {len(positions)} samples, expected {len(dof_values)}. Skipping.")
            continue

        muscle_data[mname] = positions

    return muscle_data, dof_values, sample_indices


def preprocess(pca_k=PCA_K, val_fraction=0.15):
    import sys
    sys.path.insert(0, PROJECT_ROOT)
    from core.dartHelper import saveSkeletonInfo, buildFromInfo

    # Load skeleton for pelvis transform
    print("Loading skeleton...")
    skel_info, root_name, bvh_info, _, mesh_info, _ = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)

    # Load both regions, find common sample indices
    regions = ["L_UpLeg", "L_LowLeg"]
    region_results = {}

    for region in regions:
        region_dir = os.path.join(GRID_BASE, region)
        if not os.path.exists(region_dir):
            print(f"Region {region} not found at {region_dir}")
            continue
        print(f"\nLoading {region}...")
        muscle_data, region_dofs, sample_idx = load_region_data(region_dir)
        if region_dofs is None:
            continue
        region_results[region] = (muscle_data, region_dofs, sample_idx)
        print(f"  Loaded {len(muscle_data)} muscles, {len(sample_idx)} samples "
              f"(indices {sample_idx[0]}..{sample_idx[-1]})")

    if not region_results:
        print("ERROR: No data loaded.")
        return

    # Find common sample indices across all regions
    common_indices = None
    for region, (_, _, sample_idx) in region_results.items():
        idx_set = set(sample_idx.tolist())
        common_indices = idx_set if common_indices is None else common_indices & idx_set
    common_indices = sorted(common_indices)
    print(f"\nCommon samples across regions: {len(common_indices)}")

    # Build aligned data using common indices
    all_muscle_data = {}
    dof_values = None

    for region, (muscle_data, region_dofs, sample_idx) in region_results.items():
        # Build index mapping: sample_idx value → position in region arrays
        idx_to_pos = {int(s): i for i, s in enumerate(sample_idx)}
        common_positions = [idx_to_pos[s] for s in common_indices]

        if dof_values is None:
            dof_values = region_dofs[common_positions]

        for mname, positions in muscle_data.items():
            all_muscle_data[mname] = positions[common_positions]

    if dof_values is None or len(all_muscle_data) == 0:
        print("ERROR: No data loaded.")
        return

    num_samples = len(dof_values)
    print(f"\nTotal: {len(all_muscle_data)} muscles, {num_samples} samples")

    # For each sample, compute pelvis transform so we can go to pelvis-local frame
    # Set skeleton to each DOF config and get pelvis world transform
    print("Computing pelvis transforms...")
    num_dofs = skel.getNumDofs()
    pelvis_R = np.zeros((num_samples, 3, 3), dtype=np.float64)
    pelvis_t = np.zeros((num_samples, 3), dtype=np.float64)

    for i in range(num_samples):
        pose = np.zeros(num_dofs)
        for j, dof_idx in enumerate(DOF_INDICES):
            pose[dof_idx] = dof_values[i, j]
        skel.setPositions(pose)
        T = skel.getBodyNode("Saccrum_Coccyx0").getWorldTransform().matrix()
        pelvis_R[i] = T[:3, :3]
        pelvis_t[i] = T[:3, 3]

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{num_samples}")

    # Get rest pose pelvis transform (DOFs = 0)
    skel.setPositions(np.zeros(num_dofs))
    T_rest = skel.getBodyNode("Saccrum_Coccyx0").getWorldTransform().matrix()
    rest_R = T_rest[:3, :3]
    rest_t = T_rest[:3, 3]

    # Transform positions to pelvis-local frame and compute displacements
    print("\nProcessing muscles...")
    muscle_names = sorted(all_muscle_data.keys())
    rest_positions = {}
    displacements = {}
    fixed_vertices = {}

    for mname in muscle_names:
        positions = all_muscle_data[mname]  # (N, V, 3) world frame
        num_verts = positions.shape[1]

        # Transform to pelvis-local: v_local = R^T @ (v_world - t)
        centered = positions - pelvis_t[:, None, :]  # (N, V, 3)
        R_inv = np.transpose(pelvis_R, (0, 2, 1))  # (N, 3, 3)
        local_pos = np.einsum("nij,nvj->nvi", R_inv, centered)  # (N, V, 3)

        # Load tet data for rest positions and fixed vertices
        tet_path = os.path.join(TET_DIR, f"{mname}_tet.npz")
        if os.path.exists(tet_path):
            import pickle
            with open(tet_path, 'rb') as f:
                tet_data = pickle.load(f)
            rest_world = tet_data['vertices'].astype(np.float64)  # (V, 3)
            rest_local = (rest_R.T @ (rest_world - rest_t).T).T  # (V, 3)
            rest = rest_local.astype(np.float32)

            # Compute all fixed vertices (matches muscle_mesh.py init_soft_body)
            fixed_set = set()

            # 1. Cap face vertices
            faces = tet_data['faces']
            for fi in tet_data.get('cap_face_indices', []):
                if fi < len(faces):
                    for vi in faces[fi]:
                        fixed_set.add(int(vi))

            # 2. Anchor vertices
            for vi in tet_data.get('anchor_vertices', []):
                fixed_set.add(int(vi))

            # 3. Contour origin/insertion vertices
            contours = tet_data.get('contours', [])
            contour_positions = []
            for stream_contours in contours:
                nl = len(stream_contours)
                if nl < 2:
                    continue
                origin_center = np.mean(stream_contours[0], axis=0)
                insertion_center = np.mean(stream_contours[-1], axis=0)
                for level_idx, contour in enumerate(stream_contours):
                    if level_idx == 0 or level_idx == nl - 1:
                        for pos in contour:
                            contour_positions.append(np.array(pos))
                    else:
                        cc = np.mean(contour, axis=0)
                        d_o = np.linalg.norm(cc - origin_center)
                        d_i = np.linalg.norm(cc - insertion_center)
                        if d_o < 0.003 and d_o <= d_i:
                            for pos in contour:
                                contour_positions.append(np.array(pos))
                        elif d_i < 0.003 and d_i < d_o:
                            for pos in contour:
                                contour_positions.append(np.array(pos))

            if contour_positions:
                contour_positions = np.array(contour_positions)
                for vi in range(len(rest_world)):
                    dists = np.linalg.norm(contour_positions - rest_world[vi], axis=1)
                    if dists.min() < 1e-4:
                        fixed_set.add(vi)

            fixed_list = sorted(fixed_set)
        else:
            rest = local_pos[0].astype(np.float32)
            fixed_list = []

        disp = (local_pos - rest[None, :, :]).astype(np.float32)  # (N, V, 3)

        rest_positions[mname] = torch.from_numpy(rest)
        displacements[mname] = torch.from_numpy(disp)
        fixed_vertices[mname] = torch.tensor(fixed_list, dtype=torch.long)

        disp_norm = np.linalg.norm(disp, axis=-1)
        print(f"  {mname}: {num_verts} verts, {len(fixed_list)} fixed, "
              f"disp [{disp_norm.min():.4f}, {disp_norm.max():.4f}]m")

    # Input DOFs — just the raw 7 DOF values (no derivatives for deterministic mapping)
    input_dofs = torch.from_numpy(dof_values.astype(np.float32))  # (N, 7)

    # PCA per muscle
    print(f"\nComputing PCA (k={pca_k})...")
    pca_components = {}
    pca_means = {}
    pca_stds = {}
    pca_targets = {}

    for mname in muscle_names:
        disp = displacements[mname].numpy()  # (N, V, 3)
        N, V, _ = disp.shape
        flat = disp.reshape(N, V * 3)

        k = min(pca_k, N, V * 3)
        pca = PCA(n_components=k)
        coeffs = pca.fit_transform(flat)  # (N, k)

        # Z-score normalization
        coeff_std = np.std(coeffs, axis=0)
        coeff_std[coeff_std < 1e-8] = 1.0
        coeffs_norm = coeffs / coeff_std

        explained = pca.explained_variance_ratio_.sum() * 100
        print(f"  {mname}: {k} components, {explained:.1f}% variance explained")

        pca_components[mname] = torch.from_numpy(pca.components_.astype(np.float32))  # (k, V*3)
        pca_means[mname] = torch.from_numpy(pca.mean_.astype(np.float32))  # (V*3,)
        pca_stds[mname] = torch.from_numpy(coeff_std.astype(np.float32))  # (k,)
        pca_targets[mname] = torch.from_numpy(coeffs_norm.astype(np.float32))  # (N, k)

    # Train/val split — random since samples are independent (no temporal structure)
    N = num_samples
    n_val = max(1, int(N * val_fraction))
    n_train = N - n_val
    perm = torch.randperm(N)
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    print(f"\nSplit: {n_train} train, {n_val} val")

    # Save
    save_data = {
        "model_version": "v3_dof",
        "input_dofs": input_dofs,  # (N, 7)
        "muscle_names": muscle_names,
        "rest_positions": rest_positions,
        "displacements": displacements,
        "fixed_vertices": fixed_vertices,
        "pca_components": pca_components,
        "pca_means": pca_means,
        "pca_stds": pca_stds,
        "pca_targets": pca_targets,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "dof_names": ["L_Femur_x", "L_Femur_y", "L_Femur_z", "L_Knee",
                       "L_Ankle_x", "L_Ankle_y", "L_Ankle_z"],
        "dof_indices": DOF_INDICES,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(save_data, OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"  Input: {input_dofs.shape}")
    print(f"  Muscles: {len(muscle_names)}")
    print(f"  PCA k: {pca_k}")


if __name__ == "__main__":
    preprocess()
