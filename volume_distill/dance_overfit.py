"""Preprocess dance.bvh cache + train V1 model to overfit.

7829 frames × 25 muscles (8 chunks). Input: 7 raw DOFs — left hip (3), knee (1),
ankle (3). Output: all left lower body muscles (upper + lower leg).
No constraint losses — in pelvis-local frame, fixed vertices move with bones
so penalizing their displacement is incorrect.

Usage: python -m volume_distill.dance_overfit
"""
import os
import glob
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH
from volume_distill.model import DistillNet


# === Paths ===
SKEL_XML = "data/zygote_skel.xml"
BVH_PATH = "data/motion/dance.bvh"
CACHE_DIR = "data/motion_cache/dance"
CHECKPOINT_DIR = "volume_distill/dance_checkpoints"
LOG_DIR = "volume_distill/dance_runs"

# === Training ===
EPOCHS = 10000
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
HIDDEN_DIM = 512
NUM_ENCODER_RES = 3
NUM_DECODER_RES = 2
INPUT_DOF_INDICES = [6, 7, 8, 9, 10, 11, 12]  # L hip(3) + knee(1) + ankle(3)


class SimpleDataset(Dataset):
    def __init__(self, input_dofs, displacements, muscle_names, indices):
        self.input_dofs = input_dofs
        self.displacements = displacements
        self.muscle_names = muscle_names
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.input_dofs[i]
        targets = {name: self.displacements[name][i].reshape(-1) for name in self.muscle_names}
        return x, targets


def collate_fn(batch):
    xs, tds = zip(*batch)
    x = torch.stack(xs)
    names = list(tds[0].keys())
    targets = {name: torch.stack([t[name] for t in tds]) for name in names}
    return x, targets


def preprocess():
    """Convert dance cache (multi-chunk) → input DOFs + pelvis-local displacements."""
    print("=== Preprocessing dance.bvh ===")
    skel_info, root_name, bvh_info, *_ = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)
    bvh = MyBVH(BVH_PATH, bvh_info, skel)
    mocap = bvh.mocap_refs
    N = mocap.shape[0]
    print(f"  {N} frames, {skel.getNumDofs()} DOFs")

    # Pelvis transforms
    ref_bn = skel.getBodyNode("Saccrum_Coccyx0")
    ref_R = np.zeros((N, 3, 3), dtype=np.float64)
    ref_t = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        skel.setPositions(mocap[i])
        T = ref_bn.getWorldTransform().matrix()
        ref_R[i] = T[:3, :3]
        ref_t[i] = T[:3, 3]

    # Discover muscles from chunk_0000 files
    chunk0_files = sorted(glob.glob(os.path.join(CACHE_DIR, "L_*_chunk_0000.npz")))
    muscle_names = []
    rest_positions = {}
    displacements = {}

    for npz_path in chunk0_files:
        basename = os.path.basename(npz_path)
        mname = basename.replace("_chunk_0000.npz", "")

        # Load all chunks for this muscle
        chunk_pattern = os.path.join(CACHE_DIR, f"{mname}_chunk_*.npz")
        chunk_files = sorted(glob.glob(chunk_pattern))

        all_frames = []
        all_positions = []
        for cf in chunk_files:
            data = np.load(cf)
            all_frames.append(data["frames"])
            all_positions.append(data["positions"])

        frames = np.concatenate(all_frames)
        positions = np.concatenate(all_positions)  # (N, V, 3)

        if len(frames) != N:
            print(f"  SKIP {mname}: {len(frames)} frames != {N}")
            continue

        # World → pelvis-local
        centered = positions - ref_t[:, None, :]
        R_inv = np.transpose(ref_R, (0, 2, 1))
        local_pos = np.einsum("nij,nvj->nvi", R_inv, centered)

        rest = local_pos[0].copy()
        disp = (local_pos - rest[None, :, :]).astype(np.float32)

        muscle_names.append(mname)
        rest_positions[mname] = torch.from_numpy(rest.astype(np.float32))
        displacements[mname] = torch.from_numpy(disp)
        print(f"  {mname}: {positions.shape[1]} verts, {len(chunk_files)} chunks, "
              f"disp range [{disp.min():.4f}, {disp.max():.4f}]")

    # Input DOFs: 4 raw DOFs only (deterministic mapping, no velocity)
    raw_dofs = mocap[:, INPUT_DOF_INDICES].astype(np.float32)
    input_dofs = torch.from_numpy(raw_dofs)  # (N, 4)
    print(f"  Input: {input_dofs.shape}")

    all_idx = torch.arange(N)
    return input_dofs, displacements, muscle_names, rest_positions, all_idx


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    input_dofs, displacements, muscle_names, rest_positions, indices = preprocess()

    ds = SimpleDataset(input_dofs, displacements, muscle_names, indices)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
                        num_workers=2, pin_memory=True)
    print(f"\nTrain: {len(ds)} frames, {len(muscle_names)} muscles")

    # Model
    muscle_vertex_counts = {name: rest_positions[name].shape[0] for name in muscle_names}
    input_dim = input_dofs.shape[1]
    model = DistillNet(
        muscle_vertex_counts, input_dim=input_dim,
        hidden_dim=HIDDEN_DIM, num_encoder_res=NUM_ENCODER_RES,
        num_decoder_res=NUM_DECODER_RES,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params (input_dim={input_dim})")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Resume from best.pt if it exists
    start_epoch = 0
    best_path = os.path.join(CHECKPOINT_DIR, "best.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {best_path} (epoch {start_epoch}, loss {ckpt.get('val_loss', '?'):.2e})")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=500, T_mult=2,
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    run_name = "dance_overfit_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(LOG_DIR, run_name))
    best_loss = float("inf")

    end_epoch = start_epoch + EPOCHS
    print(f"\n=== Training epochs {start_epoch+1} → {end_epoch} ===")
    for epoch in range(start_epoch + 1, end_epoch + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, targets in loader:
            x = x.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            preds = model(x)
            loss = sum(
                ((preds[name] - targets[name]) ** 2).mean()
                for name in muscle_names
            ) / len(muscle_names)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        writer.add_scalar("loss/train", avg_loss, epoch)
        writer.add_scalar("lr", lr, epoch)

        if epoch % 100 == 0 or epoch <= start_epoch + 5:
            print(f"Epoch {epoch:5d}/{end_epoch} | MSE: {avg_loss:.2e} | LR: {lr:.2e} | {elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_loss,
                "muscle_vertex_counts": muscle_vertex_counts,
                "input_dim": input_dim,
                "hidden_dim": HIDDEN_DIM,
                "num_encoder_res": NUM_ENCODER_RES,
                "num_decoder_res": NUM_DECODER_RES,
                "rest_positions": rest_positions,
                "model_version": "v1",
            }, os.path.join(CHECKPOINT_DIR, "best.pt"))

    writer.close()
    print(f"\nDone. Best MSE: {best_loss:.2e}")
    print(f"Checkpoint: {CHECKPOINT_DIR}/best.pt")


if __name__ == "__main__":
    train()
