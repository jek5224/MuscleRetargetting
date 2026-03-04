"""Train V3 DOF-based distillation network on systematic DOF grid data.

Maps 7 raw DOFs (hip 3 + knee 1 + ankle 3) → PCA coefficients for all
lower-body muscles (L_UpLeg + L_LowLeg, ~37 muscles). The mapping is
near-deterministic so we use a large network and train to overfit.

No temporal loss (samples are independent DOF configurations).
No derivative features (deterministic function of joint angles only).

Usage: python -m volume_distill.train_dof_grid
"""
import os
import math
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from volume_distill.model import DistillNetV2

DATA_PATH = "data/motion_cache/dof_grid/preprocessed.pt"
CHECKPOINT_DIR = "volume_distill/dof_grid_checkpoints"
LOG_DIR = "volume_distill/dof_grid_runs"

# Training hyperparameters — overfit-friendly
EPOCHS = 1000
BATCH_SIZE = 1024
LR = 5e-4
WEIGHT_DECAY = 1e-5
COSINE_T_MAX = 1000
PCA_K = 64
INPUT_NOISE_STD = 0.005  # small noise for slight regularization
DROPOUT = 0.0            # no dropout — we want to memorize

# Model — larger than V2 to handle 37 muscles
HIDDEN_DIM = 1024
NUM_ENCODER_RES = 6
NUM_DECODER_RES = 4
EMBED_DIM = 128
NUM_FREQS = 8


class DofGridDataset(Dataset):
    """Simple dataset: input DOFs → PCA targets per muscle."""

    def __init__(self, input_dofs, pca_targets, indices, muscle_names):
        self.input_dofs = input_dofs      # (N_total, 7)
        self.pca_targets = pca_targets    # {name: (N_total, K)}
        self.indices = indices            # subset indices
        self.muscle_names = muscle_names

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.input_dofs[i]
        targets = {name: self.pca_targets[name][i] for name in self.muscle_names}
        return x, targets


def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch])
    targets = {}
    for name in batch[0][1]:
        targets[name] = torch.stack([b[1][name] for b in batch])
    return xs, targets


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load preprocessed data
    print(f"Loading {DATA_PATH}...")
    data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)

    input_dofs = data["input_dofs"]        # (N, 7)
    muscle_names = data["muscle_names"]
    pca_targets = data["pca_targets"]      # {name: (N, K)}
    pca_components = data["pca_components"]
    pca_means = data["pca_means"]
    pca_stds = data["pca_stds"]
    rest_positions = data["rest_positions"]
    train_indices = data["train_indices"]
    val_indices = data["val_indices"]

    num_muscles = len(muscle_names)
    input_dim = input_dofs.shape[1]  # 7
    pca_k = pca_targets[muscle_names[0]].shape[1]

    print(f"Samples: {len(input_dofs)} ({len(train_indices)} train, {len(val_indices)} val)")
    print(f"Muscles: {num_muscles}, Input dim: {input_dim}, PCA k: {pca_k}")

    # Build datasets
    train_ds = DofGridDataset(input_dofs, pca_targets, train_indices, muscle_names)
    val_ds = DofGridDataset(input_dofs, pca_targets, val_indices, muscle_names)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    # Build muscle_name_to_idx (alphabetical)
    muscle_name_to_idx = {name: i for i, name in enumerate(muscle_names)}
    idx_to_name = {i: name for name, i in muscle_name_to_idx.items()}

    # Model
    model = DistillNetV2(
        num_muscles=num_muscles,
        muscle_name_to_idx=muscle_name_to_idx,
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_encoder_res=NUM_ENCODER_RES,
        num_decoder_res=NUM_DECODER_RES,
        embed_dim=EMBED_DIM,
        pca_k=pca_k,
        num_freqs=NUM_FREQS,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,} (hidden={HIDDEN_DIM}, enc_res={NUM_ENCODER_RES}, "
          f"dec_res={NUM_DECODER_RES}, embed={EMBED_DIM}, freqs={NUM_FREQS})")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=COSINE_T_MAX)

    muscle_indices = torch.arange(num_muscles, device=device)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(LOG_DIR, run_name))
    best_val_loss = float("inf")

    def compute_loss(preds, targets, B):
        """MSE loss across all muscles."""
        loss = 0.0
        for m_idx in range(num_muscles):
            name = idx_to_name[m_idx]
            pred = preds[m_idx]    # (B, K)
            gt = targets[name]     # (B, K)
            loss += ((pred - gt) ** 2).mean()
        return loss / num_muscles

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for x, targets in train_loader:
            B = x.shape[0]
            x = x.to(device)
            if INPUT_NOISE_STD > 0:
                x = x + INPUT_NOISE_STD * torch.randn_like(x)
            targets = {k: v.to(device) for k, v in targets.items()}

            preds = model(x, muscle_indices)
            loss = compute_loss(preds, targets, B)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        train_loss = train_loss_sum / train_batches

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        per_muscle_val = {name: 0.0 for name in muscle_names}
        with torch.no_grad():
            for x, targets in val_loader:
                B = x.shape[0]
                x = x.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                preds = model(x, muscle_indices)
                loss = compute_loss(preds, targets, B)
                val_loss_sum += loss.item()

                for m_idx in range(num_muscles):
                    name = idx_to_name[m_idx]
                    per_muscle_val[name] += ((preds[m_idx] - targets[name]) ** 2).mean().item()

                val_batches += 1

        val_loss = val_loss_sum / val_batches
        for name in muscle_names:
            per_muscle_val[name] /= val_batches

        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:4d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {lr:.2e} | {elapsed:.1f}s")

        # TensorBoard
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", lr, epoch)
        for name in muscle_names:
            writer.add_scalar(f"val_muscle/{name}", per_muscle_val[name], epoch)

        # Vertex-space RMSE every 20 epochs
        if epoch % 20 == 0:
            vertex_se = {name: 0.0 for name in muscle_names}
            vertex_count = {name: 0 for name in muscle_names}
            with torch.no_grad():
                for x, targets in val_loader:
                    B = x.shape[0]
                    x = x.to(device)
                    preds = model(x, muscle_indices)
                    targets_dev = {k: v.to(device) for k, v in targets.items()}
                    for m_idx in range(num_muscles):
                        name = idx_to_name[m_idx]
                        pred_norm = preds[m_idx]
                        gt_norm = targets_dev[name]
                        stds = pca_stds[name].to(device)
                        comps = pca_components[name].to(device)
                        pred_disp = (pred_norm * stds) @ comps
                        gt_disp = (gt_norm * stds) @ comps
                        vertex_se[name] += ((pred_disp - gt_disp) ** 2).sum().item()
                        vertex_count[name] += gt_disp.numel()

            print("  Per-muscle vertex RMSE (mm):")
            total_se, total_count = 0.0, 0
            for name in muscle_names:
                rmse_mm = math.sqrt(vertex_se[name] / max(vertex_count[name], 1)) * 1000
                total_se += vertex_se[name]
                total_count += vertex_count[name]
                writer.add_scalar(f"val_vertex_rmse_mm/{name}", rmse_mm, epoch)
                if rmse_mm > 1.0:  # only print muscles with >1mm error
                    print(f"    {name}: {rmse_mm:.2f} mm")
            avg_rmse_mm = math.sqrt(total_se / max(total_count, 1)) * 1000
            print(f"  Avg vertex RMSE: {avg_rmse_mm:.2f} mm")
            writer.add_scalar("val_vertex_rmse_mm/avg", avg_rmse_mm, epoch)

        # Checkpoint
        ckpt_base = {
            "model_version": "v3_dof",
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
            "epoch": epoch,
            "input_dim": input_dim,
            "hidden_dim": HIDDEN_DIM,
            "num_encoder_res": NUM_ENCODER_RES,
            "num_decoder_res": NUM_DECODER_RES,
            "embed_dim": EMBED_DIM,
            "dropout": DROPOUT,
            "pca_k": pca_k,
            "num_freqs": NUM_FREQS,
            "num_muscles": num_muscles,
            "muscle_name_to_idx": muscle_name_to_idx,
            "pca_components": pca_components,
            "pca_means": pca_means,
            "pca_stds": pca_stds,
            "rest_positions": rest_positions,
            "dof_names": data.get("dof_names"),
            "dof_indices": data.get("dof_indices"),
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_base, os.path.join(CHECKPOINT_DIR, "best.pt"))
            print(f"  -> Saved best (val_loss={val_loss:.6f})")

        if epoch % 50 == 0:
            ckpt_base["optimizer_state_dict"] = optimizer.state_dict()
            torch.save(ckpt_base, os.path.join(CHECKPOINT_DIR, "latest.pt"))

    writer.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()
