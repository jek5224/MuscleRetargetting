"""Preprocess dance motion cache into training-ready .pt file.

Transforms world-space vertex positions to femur-local frame and computes
displacements from rest pose. Using the femur (rather than pelvis) factors
out the hip rotation — the largest rigid-motion component — so the NN only
needs to learn knee-driven motion plus soft deformation.

Requires dartpy environment.

Usage: python -m volume_distill.dance.preprocess
"""
import os
import glob
import numpy as np
import torch

from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH


SKEL_XML = "data/zygote_skel.xml"
BVH_PATH = "data/motion/dance.bvh"
CACHE_DIR = "data/motion_cache/dance"
OUTPUT_PATH = "data/motion_cache/dance/preprocessed.pt"
# L_Femur0 is a ball joint at DOF indices 6,7,8; L_Tibia_Fibula0 is revolute at DOF 9
INPUT_DOF_INDICES = [6, 7, 8, 9]
VAL_FRACTION = 0.15


def main():
    # Build skeleton
    print("Loading skeleton...")
    skel_info, root_name, bvh_info, *_ = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)
    print(f"  DOFs: {skel.getNumDofs()}, Body nodes: {skel.getNumBodyNodes()}")

    # Load BVH
    print("Loading BVH...")
    motion_bvh = MyBVH(BVH_PATH, bvh_info, skel)
    mocap_refs = motion_bvh.mocap_refs  # (num_frames, num_dofs)
    num_frames = mocap_refs.shape[0]
    print(f"  Frames: {num_frames}, DOFs per frame: {mocap_refs.shape[1]}")

    # Compute femur transforms for every frame.
    # Using the femur (not pelvis) factors out hip rotation so the NN learns
    # smaller residual displacements instead of large rigid bone sweeps.
    REF_BODY = "L_Femur0"
    print(f"Computing {REF_BODY} transforms...")
    ref_R = np.zeros((num_frames, 3, 3), dtype=np.float64)
    ref_t = np.zeros((num_frames, 3), dtype=np.float64)
    ref_bn = skel.getBodyNode(REF_BODY)
    for i in range(num_frames):
        skel.setPositions(mocap_refs[i])
        T = ref_bn.getWorldTransform().matrix()  # 4x4
        ref_R[i] = T[:3, :3]
        ref_t[i] = T[:3, 3]
        if i % 1000 == 0:
            print(f"  Frame {i}/{num_frames}")

    # Discover muscle cache files
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "L_*.npz")))
    # Filter out preprocessed.pt if somehow named .npz
    npz_files = [f for f in npz_files if not f.endswith("preprocessed.npz")]
    print(f"Found {len(npz_files)} muscle cache files")

    muscle_names = []
    rest_positions = {}
    displacements = {}

    for npz_path in npz_files:
        mname = os.path.splitext(os.path.basename(npz_path))[0]
        data = np.load(npz_path)
        frames = data["frames"]  # (N,)
        positions = data["positions"]  # (N, V, 3)

        if len(frames) != num_frames:
            print(f"  WARNING: {mname} has {len(frames)} frames, expected {num_frames}. Skipping.")
            continue

        # Verify frames are 0..num_frames-1
        if not np.array_equal(frames, np.arange(num_frames)):
            print(f"  WARNING: {mname} frames not sequential. Skipping.")
            continue

        num_verts = positions.shape[1]

        # Transform to femur-local: v_local = R^T @ (v_world - t)
        # positions: (N, V, 3), ref_R: (N, 3, 3), ref_t: (N, 3)
        centered = positions - ref_t[:, None, :]  # (N, V, 3)
        # Batch matrix multiply: (N, 3, 3)^T @ (N, V, 3)^T -> transpose back
        R_inv = np.transpose(ref_R, (0, 2, 1))  # (N, 3, 3)
        local_pos = np.einsum("nij,nvj->nvi", R_inv, centered)  # (N, V, 3)

        # Rest = frame 0 in femur-local
        rest = local_pos[0].copy()  # (V, 3)
        disp = local_pos - rest[None, :, :]  # (N, V, 3)

        muscle_names.append(mname)
        rest_positions[mname] = torch.from_numpy(rest.astype(np.float32))
        displacements[mname] = torch.from_numpy(disp.astype(np.float32))

        print(f"  {mname}: {num_verts} verts, disp range [{disp.min():.4f}, {disp.max():.4f}]")

    # Extract input DOFs
    input_dofs = torch.from_numpy(mocap_refs[:, INPUT_DOF_INDICES].astype(np.float32))
    print(f"Input DOFs shape: {input_dofs.shape}")

    # Train/val split (random by frame)
    indices = torch.randperm(num_frames)
    n_val = int(num_frames * VAL_FRACTION)
    val_indices = indices[:n_val].sort().values
    train_indices = indices[n_val:].sort().values
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    # Save
    output = {
        "input_dofs": input_dofs,
        "muscle_names": muscle_names,
        "rest_positions": rest_positions,
        "displacements": displacements,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    torch.save(output, OUTPUT_PATH)
    print(f"Saved preprocessed data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
