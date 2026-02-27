"""Train distillation network on preprocessed dance motion data.

Usage: python -m volume_distill.dance.train
"""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from volume_distill.model import DistillNet
from volume_distill.dataset import MuscleDistillDataset, distill_collate_fn


DATA_PATH = "data/motion_cache/dance/preprocessed.pt"
CHECKPOINT_DIR = "volume_distill/dance/checkpoints"
EPOCHS = 200
BATCH_SIZE = 512
LR = 1e-4
WEIGHT_DECAY = 1e-5
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 10


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load datasets
    train_ds = MuscleDistillDataset(DATA_PATH, split="train")
    val_ds = MuscleDistillDataset(DATA_PATH, split="val")
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=distill_collate_fn, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=distill_collate_fn, num_workers=4, pin_memory=True,
    )
    print(f"Train: {len(train_ds)} frames, Val: {len(val_ds)} frames")

    # Build model
    data = torch.load(DATA_PATH, weights_only=False)
    muscle_vertex_counts = {
        name: data["rest_positions"][name].shape[0]
        for name in data["muscle_names"]
    }
    model = DistillNet(muscle_vertex_counts).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE,
    )
    criterion = nn.MSELoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    muscle_names = data["muscle_names"]

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
            loss = sum(criterion(preds[name], targets[name]) for name in muscle_names) / len(muscle_names)
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
                    ml = criterion(preds[name], targets[name]).item()
                    per_muscle_val[name] += ml
                    batch_loss += ml
                val_loss_sum += batch_loss / len(muscle_names)
                val_batches += 1
        val_loss = val_loss_sum / val_batches
        for name in muscle_names:
            per_muscle_val[name] /= val_batches

        scheduler.step(val_loss)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{EPOCHS} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | LR: {lr:.2e} | {elapsed:.1f}s")

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
            }, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pt"))

    print(f"Training complete. Best val MSE: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()
