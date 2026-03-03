"""Train distillation network on combined motion data (dance + locomotion).

Usage: python -m volume_distill.train
"""
import os
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from volume_distill.model import DistillNet
from volume_distill.dataset import load_train_val, distill_collate_fn


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
COSINE_T0 = 150      # epochs per first restart cycle
COSINE_T_MULT = 2    # cycle length multiplier after each restart
ANCHOR_LOSS_WEIGHT = 10.0
DIST_LOSS_SCALE = 5.0  # weight scale for distance from pelvis


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load datasets (shared memory between train/val)
    train_ds, val_ds = load_train_val(DATA_PATHS)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=distill_collate_fn, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=distill_collate_fn, num_workers=4, pin_memory=True,
    )
    print(f"Train: {len(train_ds)} frames, Val: {len(val_ds)} frames")

    # Build model from dataset metadata (no extra file load)
    muscle_names = train_ds.muscle_names
    rest_positions = train_ds.rest_positions
    anchor_data = train_ds.anchor_vertices
    muscle_vertex_counts = {
        name: rest_positions[name].shape[0] for name in muscle_names
    }
    input_dim = train_ds.input_dofs.shape[1]
    model = DistillNet(
        muscle_vertex_counts, input_dim=input_dim,
        hidden_dim=768, num_encoder_res=5, num_decoder_res=3,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,} (input_dim={input_dim})")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=COSINE_T0, T_mult=COSINE_T_MULT,
    )

    # Build per-muscle vertex weight vectors
    vertex_weights = {}
    for name in muscle_names:
        n_verts = muscle_vertex_counts[name]
        rest = rest_positions[name]  # (V, 3)
        dist = rest.norm(dim=1)  # (V,)
        mean_dist = dist.mean()
        dist_w = 1.0 + DIST_LOSS_SCALE * (dist / mean_dist)
        dist_w = dist_w.unsqueeze(1).expand(-1, 3).reshape(-1)
        w = dist_w.to(device)
        n_anchors = 0
        if name in anchor_data and len(anchor_data[name]) > 0:
            for vi in anchor_data[name].tolist():
                w[vi * 3: vi * 3 + 3] = ANCHOR_LOSS_WEIGHT
            n_anchors = len(anchor_data[name])
        vertex_weights[name] = w
        print(f"  {name}: {n_verts} verts, {n_anchors} anchors, "
              f"dist weight range [{dist_w.min():.1f}, {dist_w.max():.1f}]")

    def weighted_mse(pred, target, weights):
        return (weights * (pred - target) ** 2).mean()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(LOG_DIR, run_name))
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for x, targets in train_loader:
            x = x.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            preds = model(x)
            loss = sum(weighted_mse(preds[name], targets[name], vertex_weights[name]) for name in muscle_names) / len(muscle_names)
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
                x = x.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                preds = model(x)
                batch_loss = 0.0
                for name in muscle_names:
                    ml = weighted_mse(preds[name], targets[name], vertex_weights[name]).item()
                    per_muscle_val[name] += ml
                    batch_loss += ml
                val_loss_sum += batch_loss / len(muscle_names)
                val_batches += 1
        val_loss = val_loss_sum / val_batches
        for name in muscle_names:
            per_muscle_val[name] /= val_batches

        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{EPOCHS} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | LR: {lr:.2e} | {elapsed:.1f}s")

        # TensorBoard scalars
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", lr, epoch)
        for name in muscle_names:
            writer.add_scalar(f"val_muscle/{name}", per_muscle_val[name], epoch)

        # Per-muscle breakdown every 10 epochs
        if epoch % 10 == 0:
            print("  Per-muscle val MSE:")
            for name in muscle_names:
                print(f"    {name}: {per_muscle_val[name]:.6f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "muscle_vertex_counts": muscle_vertex_counts,
                "input_dim": input_dim,
                "rest_positions": rest_positions,
            }, os.path.join(CHECKPOINT_DIR, "best.pt"))
            print(f"  -> Saved best model (val_loss={val_loss:.6f})")

        # Save periodic
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "muscle_vertex_counts": muscle_vertex_counts,
                "input_dim": input_dim,
                "rest_positions": rest_positions,
            }, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pt"))

    writer.close()
    print(f"Training complete. Best val MSE: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()
