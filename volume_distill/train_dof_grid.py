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
EPOCHS = 2000
BATCH_SIZE = 256
LR = 3e-4
WEIGHT_DECAY = 0.0       # no regularization — we want to memorize
PCA_K = 64
INPUT_NOISE_STD = 0.0    # no noise — overfit to deterministic mapping
DROPOUT = 0.0            # no dropout — we want to memorize
GRAD_CLIP = 1.0          # gradient norm clipping

# Constraint loss weights
LAMBDA_FIXED = 10.0      # fixed vertex displacement penalty (hard constraint)
LAMBDA_INTER = 1.0       # inter-muscle distance penalty (soft constraint)

# Cosine warm restarts: LR resets every T_0 epochs, cycle doubles each time
COSINE_T0 = 300
COSINE_T_MULT = 2

# Model — larger than V2 to handle 37 muscles
HIDDEN_DIM = 1024
NUM_ENCODER_RES = 6
NUM_DECODER_RES = 4
EMBED_DIM = 128
NUM_FREQS = 6


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
    fixed_vertices_data = data.get("fixed_vertices", {})
    inter_muscle_constraints = data.get("inter_muscle_constraints", [])

    num_muscles = len(muscle_names)
    num_samples = len(input_dofs)
    input_dim = input_dofs.shape[1]  # 7
    pca_k = pca_targets[muscle_names[0]].shape[1]

    print(f"Samples: {num_samples} (all used for training — overfit mode)")
    print(f"Muscles: {num_muscles}, Input dim: {input_dim}, PCA k: {pca_k}")
    print(f"Fixed vertices: {sum(len(v) for v in fixed_vertices_data.values())} total")
    print(f"Inter-muscle constraints: {len(inter_muscle_constraints)}")

    # Use ALL samples for training (overfit to deterministic mapping)
    all_indices = torch.arange(num_samples)
    train_ds = DofGridDataset(input_dofs, pca_targets, all_indices, muscle_names)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=COSINE_T0, T_mult=COSINE_T_MULT,
    )

    muscle_indices = torch.arange(num_muscles, device=device)

    # Precompute constraint data on GPU
    # PCA reconstruction: disp_flat = (coeffs_norm * stds) @ components  (no mean — displacements)
    pca_comps_gpu = {name: pca_components[name].to(device) for name in muscle_names}  # (K, V*3)
    pca_stds_gpu = {name: pca_stds[name].to(device) for name in muscle_names}  # (K,)
    pca_means_gpu = {name: pca_means[name].to(device) for name in muscle_names}  # (V*3,)
    rest_pos_gpu = {name: rest_positions[name].to(device) for name in muscle_names}  # (V, 3)

    # Fixed vertex indices per muscle (for anchor constraint)
    fixed_verts_gpu = {}
    for name in muscle_names:
        fv = fixed_vertices_data.get(name, torch.tensor([], dtype=torch.long))
        if len(fv) > 0:
            fixed_verts_gpu[name] = fv.to(device)

    # Inter-muscle constraints: group by (muscle_a, muscle_b) pair for batched computation
    inter_constraint_pairs = {}  # (name_a, name_b) -> (verts_a[], verts_b[], rest_dists[])
    for name_a, va_idx, name_b, vb_idx, rest_dist in inter_muscle_constraints:
        key = (name_a, name_b)
        if key not in inter_constraint_pairs:
            inter_constraint_pairs[key] = ([], [], [])
        inter_constraint_pairs[key][0].append(va_idx)
        inter_constraint_pairs[key][1].append(vb_idx)
        inter_constraint_pairs[key][2].append(rest_dist)

    # Convert to tensors
    inter_constraints_gpu = {}
    for (name_a, name_b), (va_list, vb_list, dist_list) in inter_constraint_pairs.items():
        inter_constraints_gpu[(name_a, name_b)] = (
            torch.tensor(va_list, dtype=torch.long, device=device),
            torch.tensor(vb_list, dtype=torch.long, device=device),
            torch.tensor(dist_list, dtype=torch.float32, device=device),
        )
    print(f"Inter-muscle constraint pairs: {len(inter_constraints_gpu)} muscle pairs")

    # Muscles that need position reconstruction (involved in any constraint)
    muscles_needing_positions = set()
    for name in fixed_verts_gpu:
        muscles_needing_positions.add(name)
    for name_a, name_b in inter_constraints_gpu:
        muscles_needing_positions.add(name_a)
        muscles_needing_positions.add(name_b)
    print(f"Muscles needing position reconstruction: {len(muscles_needing_positions)}/{num_muscles}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(LOG_DIR, run_name))
    best_loss = float("inf")

    def reconstruct_positions(pred_norm, name):
        """PCA coefficients → pelvis-local positions (B, V, 3)."""
        disp_flat = (pred_norm * pca_stds_gpu[name]) @ pca_comps_gpu[name] + pca_means_gpu[name]  # (B, V*3)
        V = rest_pos_gpu[name].shape[0]
        disp = disp_flat.reshape(-1, V, 3)  # (B, V, 3)
        return rest_pos_gpu[name].unsqueeze(0) + disp  # (B, V, 3)

    def compute_loss(preds, targets):
        """MSE + fixed vertex constraint + inter-muscle constraint."""
        mse_loss = 0.0
        fixed_loss = 0.0
        positions = {}  # cache reconstructed positions for inter-muscle loss

        for m_idx in range(num_muscles):
            name = idx_to_name[m_idx]
            pred = preds[m_idx]    # (B, K)
            gt = targets[name]     # (B, K)
            mse_loss += ((pred - gt) ** 2).mean()

            # Reconstruct positions for constraint losses
            if name in muscles_needing_positions:
                pos = reconstruct_positions(pred, name)  # (B, V, 3)
                positions[name] = pos

                # Fixed vertex loss: displacement from rest should be zero
                if name in fixed_verts_gpu:
                    fv = fixed_verts_gpu[name]
                    rest = rest_pos_gpu[name]
                    fixed_disp = pos[:, fv, :] - rest[fv, :].unsqueeze(0)  # (B, F, 3)
                    fixed_loss += fixed_disp.pow(2).mean()

        mse_loss = mse_loss / num_muscles

        # Inter-muscle distance constraint
        inter_loss = 0.0
        n_inter_pairs = 0
        for (name_a, name_b), (va_idx, vb_idx, rest_dists) in inter_constraints_gpu.items():
            if name_a not in positions or name_b not in positions:
                continue
            pos_a = positions[name_a][:, va_idx, :]  # (B, C, 3)
            pos_b = positions[name_b][:, vb_idx, :]  # (B, C, 3)
            dists = (pos_a - pos_b).norm(dim=-1)     # (B, C)
            deviation = dists - rest_dists.unsqueeze(0)  # (B, C)
            inter_loss += deviation.pow(2).mean()
            n_inter_pairs += 1

        if n_inter_pairs > 0:
            inter_loss = inter_loss / n_inter_pairs

        total = mse_loss + LAMBDA_FIXED * fixed_loss + LAMBDA_INTER * inter_loss
        return total, mse_loss.item(), fixed_loss.item() if isinstance(fixed_loss, torch.Tensor) else 0.0, inter_loss.item() if isinstance(inter_loss, torch.Tensor) else 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Train on all data (overfit mode)
        model.train()
        train_loss_sum = 0.0
        mse_sum = 0.0
        fixed_sum = 0.0
        inter_sum = 0.0
        train_batches = 0
        for x, targets in train_loader:
            x = x.to(device)
            if INPUT_NOISE_STD > 0:
                x = x + INPUT_NOISE_STD * torch.randn_like(x)
            targets = {k: v.to(device) for k, v in targets.items()}

            preds = model(x, muscle_indices)
            loss, mse_val, fixed_val, inter_val = compute_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss_sum += loss.item()
            mse_sum += mse_val
            fixed_sum += fixed_val
            inter_sum += inter_val
            train_batches += 1

        train_loss = train_loss_sum / train_batches
        mse_avg = mse_sum / train_batches
        fixed_avg = fixed_sum / train_batches
        inter_avg = inter_sum / train_batches

        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:4d}/{EPOCHS} | Loss: {train_loss:.6f} "
              f"(mse={mse_avg:.6f} fix={fixed_avg:.6f} inter={inter_avg:.6f}) | "
              f"LR: {lr:.2e} | {elapsed:.1f}s")

        # TensorBoard
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/mse", mse_avg, epoch)
        writer.add_scalar("loss/fixed", fixed_avg, epoch)
        writer.add_scalar("loss/inter", inter_avg, epoch)
        writer.add_scalar("lr", lr, epoch)

        # Vertex-space RMSE every 20 epochs (evaluated on full training set)
        if epoch % 20 == 0:
            model.eval()
            vertex_se = {name: 0.0 for name in muscle_names}
            vertex_count = {name: 0 for name in muscle_names}
            per_muscle_loss = {name: 0.0 for name in muscle_names}
            eval_batches = 0
            with torch.no_grad():
                for x, targets in train_loader:
                    x = x.to(device)
                    preds = model(x, muscle_indices)
                    targets_dev = {k: v.to(device) for k, v in targets.items()}
                    for m_idx in range(num_muscles):
                        name = idx_to_name[m_idx]
                        pred_norm = preds[m_idx]
                        gt_norm = targets_dev[name]
                        per_muscle_loss[name] += ((pred_norm - gt_norm) ** 2).mean().item()
                        stds = pca_stds[name].to(device)
                        comps = pca_components[name].to(device)
                        pred_disp = (pred_norm * stds) @ comps
                        gt_disp = (gt_norm * stds) @ comps
                        vertex_se[name] += ((pred_disp - gt_disp) ** 2).sum().item()
                        vertex_count[name] += gt_disp.numel()
                    eval_batches += 1

            print("  Per-muscle vertex RMSE (mm):")
            total_se, total_count = 0.0, 0
            for name in muscle_names:
                rmse_mm = math.sqrt(vertex_se[name] / max(vertex_count[name], 1)) * 1000
                total_se += vertex_se[name]
                total_count += vertex_count[name]
                pml = per_muscle_loss[name] / eval_batches
                writer.add_scalar(f"muscle_rmse_mm/{name}", rmse_mm, epoch)
                writer.add_scalar(f"muscle_loss/{name}", pml, epoch)
                if rmse_mm > 1.0:
                    print(f"    {name}: {rmse_mm:.2f} mm")
            avg_rmse_mm = math.sqrt(total_se / max(total_count, 1)) * 1000
            print(f"  Avg vertex RMSE: {avg_rmse_mm:.2f} mm")
            writer.add_scalar("vertex_rmse_mm/avg", avg_rmse_mm, epoch)

        # Checkpoint
        ckpt_base = {
            "model_version": "v3_dof",
            "model_state_dict": model.state_dict(),
            "train_loss": train_loss,
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

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(ckpt_base, os.path.join(CHECKPOINT_DIR, "best.pt"))
            print(f"  -> Saved best (loss={train_loss:.6f})")

        if epoch % 50 == 0:
            ckpt_base["optimizer_state_dict"] = optimizer.state_dict()
            torch.save(ckpt_base, os.path.join(CHECKPOINT_DIR, "latest.pt"))

    writer.close()
    print(f"\nTraining complete. Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    train()
