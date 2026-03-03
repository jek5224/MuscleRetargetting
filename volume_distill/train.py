"""Train V2 distillation network on combined motion data (dance + locomotion).

Uses PCA output basis, temporal consistency loss, sliding window input,
muscle embedding + single decoder, and linear baseline + residual.

Usage: python -m volume_distill.train
"""
import os
import time
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from volume_distill.model import DistillNetV2
from volume_distill.dataset import load_train_val_v2, distill_collate_fn_v2


DATA_PATHS = [
    "data/motion_cache/dance/preprocessed.pt",
    "data/motion_cache/locomotion/preprocessed.pt",
]
CHECKPOINT_DIR = "volume_distill/checkpoints"
LOG_DIR = "volume_distill/runs"
EPOCHS = 600
BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-5
COSINE_T_MAX = 600
PCA_K = 64
TEMPORAL_LOSS_WEIGHT = 0.5
WINDOW_SIZE = 5


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load V2 datasets with PCA
    train_ds, val_ds, pca_components, pca_means, muscle_name_to_idx, rest_positions = \
        load_train_val_v2(DATA_PATHS, pca_k=PCA_K)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=distill_collate_fn_v2, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=distill_collate_fn_v2, num_workers=4, pin_memory=True,
    )
    print(f"Train: {len(train_ds)} frames, Val: {len(val_ds)} frames")

    muscle_names = train_ds.muscle_names
    num_muscles = len(muscle_names)
    input_dim = train_ds.input_dofs.shape[1]
    hidden_dim = 768
    num_encoder_res = 5
    num_decoder_res = 3
    embed_dim = 64

    model = DistillNetV2(
        num_muscles=num_muscles,
        muscle_name_to_idx=muscle_name_to_idx,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_encoder_res=num_encoder_res,
        num_decoder_res=num_decoder_res,
        embed_dim=embed_dim,
        pca_k=PCA_K,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,} (input_dim={input_dim})")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=COSINE_T_MAX,
    )

    # Pre-compute muscle index tensor
    muscle_indices = torch.arange(num_muscles, device=device)
    # Map name → index for loss computation
    idx_to_name = {v: k for k, v in muscle_name_to_idx.items()}

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(LOG_DIR, run_name))
    best_val_loss = float("inf")

    def compute_loss(preds, targets_t, targets_prev, B):
        """Compute reconstruction + temporal consistency loss.

        preds: dict {muscle_idx: (2B, K)} from forward on combined [x_t; x_prev]
        targets_t: {name: (B, K)}
        targets_prev: {name: (B, K)}
        """
        recon_loss = 0.0
        temporal_loss = 0.0
        for m_idx in range(num_muscles):
            name = idx_to_name[m_idx]
            pred_combined = preds[m_idx]    # (2B, K)
            pred_t = pred_combined[:B]       # (B, K)
            pred_prev = pred_combined[B:]    # (B, K)
            gt_t = targets_t[name]           # (B, K)
            gt_prev = targets_prev[name]     # (B, K)

            # Reconstruction loss on current frame
            recon_loss += ((pred_t - gt_t) ** 2).mean()

            # Temporal consistency: penalize delta mismatch
            pred_delta = pred_t - pred_prev
            gt_delta = gt_t - gt_prev
            temporal_loss += ((pred_delta - gt_delta) ** 2).mean()

        recon_loss /= num_muscles
        temporal_loss /= num_muscles
        return recon_loss, temporal_loss

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        train_recon_sum = 0.0
        train_temporal_sum = 0.0
        train_batches = 0
        for x_combined, targets_t, targets_prev in train_loader:
            B = targets_t[muscle_names[0]].shape[0]
            x_combined = x_combined.to(device)
            targets_t = {k: v.to(device) for k, v in targets_t.items()}
            targets_prev = {k: v.to(device) for k, v in targets_prev.items()}

            preds = model(x_combined, muscle_indices)
            recon_loss, temporal_loss = compute_loss(preds, targets_t, targets_prev, B)
            loss = recon_loss + TEMPORAL_LOSS_WEIGHT * temporal_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_recon_sum += recon_loss.item()
            train_temporal_sum += temporal_loss.item()
            train_batches += 1

        train_recon = train_recon_sum / train_batches
        train_temporal = train_temporal_sum / train_batches
        train_loss = train_recon + TEMPORAL_LOSS_WEIGHT * train_temporal

        # Validate
        model.eval()
        val_recon_sum = 0.0
        val_temporal_sum = 0.0
        val_batches = 0
        per_muscle_val = {name: 0.0 for name in muscle_names}
        with torch.no_grad():
            for x_combined, targets_t, targets_prev in val_loader:
                B = targets_t[muscle_names[0]].shape[0]
                x_combined = x_combined.to(device)
                targets_t = {k: v.to(device) for k, v in targets_t.items()}
                targets_prev = {k: v.to(device) for k, v in targets_prev.items()}

                preds = model(x_combined, muscle_indices)
                recon_loss, temporal_loss = compute_loss(preds, targets_t, targets_prev, B)
                val_recon_sum += recon_loss.item()
                val_temporal_sum += temporal_loss.item()

                # Per-muscle breakdown
                for m_idx in range(num_muscles):
                    name = idx_to_name[m_idx]
                    pred_t = preds[m_idx][:B]
                    gt_t = targets_t[name]
                    per_muscle_val[name] += ((pred_t - gt_t) ** 2).mean().item()

                val_batches += 1

        val_recon = val_recon_sum / val_batches
        val_temporal = val_temporal_sum / val_batches
        val_loss = val_recon + TEMPORAL_LOSS_WEIGHT * val_temporal
        for name in muscle_names:
            per_muscle_val[name] /= val_batches

        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{EPOCHS} | Recon: {val_recon:.6f} | Temporal: {val_temporal:.6f} | Total: {val_loss:.6f} | LR: {lr:.2e} | {elapsed:.1f}s")

        # TensorBoard
        writer.add_scalar("loss/train_total", train_loss, epoch)
        writer.add_scalar("loss/train_recon", train_recon, epoch)
        writer.add_scalar("loss/train_temporal", train_temporal, epoch)
        writer.add_scalar("loss/val_total", val_loss, epoch)
        writer.add_scalar("loss/val_recon", val_recon, epoch)
        writer.add_scalar("loss/val_temporal", val_temporal, epoch)
        writer.add_scalar("lr", lr, epoch)
        for name in muscle_names:
            writer.add_scalar(f"val_muscle/{name}", per_muscle_val[name], epoch)

        if epoch % 10 == 0:
            print("  Per-muscle val MSE:")
            for name in muscle_names:
                print(f"    {name}: {per_muscle_val[name]:.6f}")

        # Checkpoint data shared between best/latest
        ckpt_base = {
            "model_version": "v2",
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
            "epoch": epoch,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_encoder_res": num_encoder_res,
            "num_decoder_res": num_decoder_res,
            "embed_dim": embed_dim,
            "pca_k": PCA_K,
            "num_muscles": num_muscles,
            "muscle_name_to_idx": muscle_name_to_idx,
            "pca_components": pca_components,
            "pca_means": pca_means,
            "rest_positions": rest_positions,
            "window_size": WINDOW_SIZE,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_base, os.path.join(CHECKPOINT_DIR, "best.pt"))
            print(f"  -> Saved best model (val_loss={val_loss:.6f})")

        if epoch % 10 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
            ckpt_base["optimizer_state_dict"] = optimizer.state_dict()
            torch.save(ckpt_base, ckpt_path)

    writer.close()
    print(f"Training complete. Best val MSE: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()
