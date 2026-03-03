"""Preprocess motion cache into training-ready .pt file.

Transforms world-space vertex positions to pelvis-local frame and computes
displacements from rest pose. Anchor vertex indices (origin/insertion caps)
are extracted from tet mesh files for loss weighting during training.

Requires dartpy environment.

Usage: python -m volume_distill.preprocess <motion_name>
  e.g. python -m volume_distill.preprocess locomotion
"""
import os
import sys
import glob
import numpy as np
import torch

from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH


SKEL_XML = "data/zygote_skel.xml"
TET_DIR = "tet"
INPUT_DOF_INDICES = [6, 7, 8, 9]
VAL_FRACTION = 0.15


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m volume_distill.preprocess <motion_name>")
        print("  e.g. python -m volume_distill.preprocess locomotion")
        sys.exit(1)

    motion_name = sys.argv[1]
    bvh_path = f"data/motion/{motion_name}.bvh"
    cache_dir = f"data/motion_cache/{motion_name}"
    output_path = f"data/motion_cache/{motion_name}/preprocessed.pt"

    if not os.path.exists(bvh_path):
        print(f"ERROR: BVH not found: {bvh_path}")
        sys.exit(1)
    if not os.path.isdir(cache_dir):
        print(f"ERROR: Cache directory not found: {cache_dir}")
        sys.exit(1)

    # Build skeleton
    print("Loading skeleton...")
    skel_info, root_name, bvh_info, *_ = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)
    print(f"  DOFs: {skel.getNumDofs()}, Body nodes: {skel.getNumBodyNodes()}")

    # Load BVH
    print(f"Loading BVH: {bvh_path}")
    motion_bvh = MyBVH(bvh_path, bvh_info, skel)
    mocap_refs = motion_bvh.mocap_refs  # (num_frames, num_dofs)
    num_frames = mocap_refs.shape[0]
    print(f"  Frames: {num_frames}, DOFs per frame: {mocap_refs.shape[1]}")

    # Compute pelvis transforms for every frame.
    REF_BODY = "Saccrum_Coccyx0"
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

    # Discover muscle cache files (exclude chunk files and preprocessed.pt)
    npz_files = sorted(glob.glob(os.path.join(cache_dir, "L_*.npz")))
    npz_files = [f for f in npz_files if "_chunk_" not in os.path.basename(f)]
    print(f"Found {len(npz_files)} muscle cache files")

    muscle_names = []
    rest_positions = {}
    displacements = {}
    anchor_vertices = {}

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

        # Transform to pelvis-local: v_local = R^T @ (v_world - t)
        centered = positions - ref_t[:, None, :]  # (N, V, 3)
        R_inv = np.transpose(ref_R, (0, 2, 1))  # (N, 3, 3)
        local_pos = np.einsum("nij,nvj->nvi", R_inv, centered)  # (N, V, 3)

        # Rest = frame 0 in pelvis-local
        rest = local_pos[0].copy()  # (V, 3)
        disp = local_pos - rest[None, :, :]  # (N, V, 3)

        # Load anchor vertex indices from tet mesh
        tet_path = os.path.join(TET_DIR, f"{mname}_tet.npz")
        anchors = []
        if os.path.exists(tet_path):
            tet_data = np.load(tet_path, allow_pickle=True)
            if 'anchor_vertices' in tet_data and len(tet_data['anchor_vertices']) > 0:
                anchors = list(tet_data['anchor_vertices'])

        muscle_names.append(mname)
        rest_positions[mname] = torch.from_numpy(rest.astype(np.float32))
        displacements[mname] = torch.from_numpy(disp.astype(np.float32))
        anchor_vertices[mname] = torch.tensor(anchors, dtype=torch.long)

        print(f"  {mname}: {num_verts} verts, {len(anchors)} anchors, disp range [{disp.min():.4f}, {disp.max():.4f}]")

    # Extract input DOFs with derivative features: [q_t, dq_t, ddq_t, q_{t-1}, q_{t-2}]
    raw_dofs = mocap_refs[:, INPUT_DOF_INDICES].astype(np.float32)  # (N, 4)
    D = raw_dofs.shape[1]  # 4

    # q_{t-1} and q_{t-2} with boundary padding (frame 0 repeats)
    q_prev1 = np.zeros_like(raw_dofs)
    q_prev1[1:] = raw_dofs[:-1]
    q_prev1[0] = raw_dofs[0]

    q_prev2 = np.zeros_like(raw_dofs)
    q_prev2[2:] = raw_dofs[:-2]
    q_prev2[:2] = raw_dofs[0]

    # Finite-difference derivatives
    dq = raw_dofs - q_prev1          # velocity
    ddq = raw_dofs - 2 * q_prev1 + q_prev2  # acceleration

    # [q_t(4), dq_t(4), ddq_t(4), q_{t-1}(4), q_{t-2}(4)] = 20D
    input_dofs = torch.from_numpy(
        np.concatenate([raw_dofs, dq, ddq, q_prev1, q_prev2], axis=1)
    )  # (N, 20)

    # Build prev_frame_idx: frame 0 points to itself, others point to frame-1
    prev_frame_idx = torch.arange(num_frames)
    prev_frame_idx[0] = 0  # boundary: frame 0 → self
    prev_frame_idx[1:] = torch.arange(num_frames - 1)

    print(f"Input DOFs shape: {input_dofs.shape} (derivative features: q, dq, ddq, q-1, q-2)")

    # Train/val split (temporal blocks of 120 frames)
    BLOCK_SIZE = 120
    num_blocks = (num_frames + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_perm = torch.randperm(num_blocks)
    n_val_blocks = max(1, int(num_blocks * VAL_FRACTION))
    val_blocks = set(block_perm[:n_val_blocks].tolist())

    train_list, val_list = [], []
    for b in range(num_blocks):
        start = b * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, num_frames)
        block_frames = torch.arange(start, end)
        if b in val_blocks:
            val_list.append(block_frames)
        else:
            train_list.append(block_frames)
    train_indices = torch.cat(train_list)
    val_indices = torch.cat(val_list)
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)} ({n_val_blocks}/{num_blocks} blocks)")

    # Save
    output = {
        "input_dofs": input_dofs,
        "prev_frame_idx": prev_frame_idx,
        "muscle_names": muscle_names,
        "rest_positions": rest_positions,
        "displacements": displacements,
        "anchor_vertices": anchor_vertices,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    torch.save(output, output_path)
    print(f"Saved preprocessed data to {output_path}")


if __name__ == "__main__":
    main()
