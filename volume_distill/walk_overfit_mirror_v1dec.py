"""Train V1Dec DistillNetV1Dec on walk.bvh with L+R UpLeg data (mirrored to L-canonical).

V1Dec uses batched decoders (bmm) for parallel execution, same raw V×3 displacement
output as V1. Same 4-DOF input, same V1 encoder (512 hidden, 3 ResBlocks).

Usage: python -m volume_distill.walk_overfit_mirror_v1dec
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
from volume_distill.model import DistillNetV1Dec


# === Paths ===
SKEL_XML = "data/zygote_skel.xml"
BVH_PATH = "data/motion/walk.bvh"
L_CACHE_DIR = "data/motion_cache/walk/L_UpLeg"
R_CACHE_DIR = "data/motion_cache/walk/R_UpLeg"
CHECKPOINT_DIR = "volume_distill/walk_mirror_checkpoints"
LOG_DIR = "volume_distill/walk_mirror_runs"

# === Training ===
EPOCHS = 10000
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
HIDDEN_DIM = 512
NUM_ENCODER_RES = 3
NUM_DECODER_RES = 2

# L hip (3 DOFs) + L knee (1 DOF)
L_DOF_INDICES = [6, 7, 8, 9]
# R hip (3 DOFs) + R knee (1 DOF)
R_DOF_INDICES = [18, 19, 20, 21]


def mirror_dofs_r_to_l(dofs):
    mirrored = dofs.copy()
    mirrored[:, 0] *= -1
    return mirrored


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


def load_side(cache_dir, prefix, mocap, ref_R, ref_t, N):
    npz_files = sorted(glob.glob(os.path.join(cache_dir, f"{prefix}_*_chunk_0000.npz")))
    muscle_names = []
    rest_positions = {}
    displacements = {}

    for npz_path in npz_files:
        basename = os.path.basename(npz_path)
        mname = basename.replace("_chunk_0000.npz", "")
        canonical_name = "L_" + mname[2:] if mname.startswith("R_") else mname

        data = np.load(npz_path)
        frames = data["frames"]
        positions = data["positions"]

        if len(frames) != N:
            print(f"  SKIP {mname}: {len(frames)} frames != {N}")
            continue

        centered = positions - ref_t[:, None, :]
        R_inv = np.transpose(ref_R, (0, 2, 1))
        local_pos = np.einsum("nij,nvj->nvi", R_inv, centered)

        if prefix == "R":
            local_pos[:, :, 0] *= -1

        rest = local_pos[0].copy()
        disp = (local_pos - rest[None, :, :]).astype(np.float32)

        muscle_names.append(canonical_name)
        rest_positions[canonical_name] = torch.from_numpy(rest.astype(np.float32))
        displacements[canonical_name] = torch.from_numpy(disp)
        print(f"  {mname} → {canonical_name}: {positions.shape[1]} verts, "
              f"disp range [{disp.min():.4f}, {disp.max():.4f}]")

    return muscle_names, rest_positions, displacements


def preprocess():
    print("=== Preprocessing walk.bvh (L+R mirrored, V1Dec) ===")
    skel_info, root_name, bvh_info, *_ = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)
    bvh = MyBVH(BVH_PATH, bvh_info, skel)
    mocap = bvh.mocap_refs
    N = mocap.shape[0]
    print(f"  {N} frames, {skel.getNumDofs()} DOFs")

    ref_bn = skel.getBodyNode("Saccrum_Coccyx0")
    ref_R = np.zeros((N, 3, 3), dtype=np.float64)
    ref_t = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        skel.setPositions(mocap[i])
        T = ref_bn.getWorldTransform().matrix()
        ref_R[i] = T[:3, :3]
        ref_t[i] = T[:3, 3]

    print("\n--- L side ---")
    l_names, l_rest, l_disp = load_side(L_CACHE_DIR, "L", mocap, ref_R, ref_t, N)

    print("\n--- R side (mirrored to L-canonical) ---")
    r_names, r_rest, r_disp = load_side(R_CACHE_DIR, "R", mocap, ref_R, ref_t, N)

    muscle_names = l_names
    rest_positions = {}
    r_rest_positions = {}
    displacements = {}

    for mname in muscle_names:
        if mname in l_disp and mname in r_disp:
            rest_positions[mname] = l_rest[mname]
            r_rest_positions[mname] = r_rest[mname]
            displacements[mname] = torch.cat([l_disp[mname], r_disp[mname]], dim=0)
        elif mname in l_disp:
            rest_positions[mname] = l_rest[mname]
            r_rest_positions[mname] = l_rest[mname]
            displacements[mname] = l_disp[mname]

    l_dofs = mocap[:, L_DOF_INDICES].astype(np.float32)
    r_dofs = mocap[:, R_DOF_INDICES].astype(np.float32)
    r_dofs_mirrored = mirror_dofs_r_to_l(r_dofs)

    input_dofs = torch.from_numpy(np.concatenate([l_dofs, r_dofs_mirrored], axis=0))
    total = input_dofs.shape[0]
    print(f"\n  Combined: {total} samples ({N} L + {N} R), {len(muscle_names)} muscles")
    print(f"  Input: {input_dofs.shape}")

    all_idx = torch.arange(total)
    return input_dofs, displacements, muscle_names, rest_positions, r_rest_positions, all_idx


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    input_dofs, displacements, muscle_names, rest_positions, r_rest_positions, indices = preprocess()

    ds = SimpleDataset(input_dofs, displacements, muscle_names, indices)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    print(f"\nTrain: {len(ds)} samples, {len(muscle_names)} muscles")

    muscle_vertex_counts = {name: rest_positions[name].shape[0] for name in muscle_names}
    input_dim = input_dofs.shape[1]
    model = DistillNetV1Dec(
        muscle_vertex_counts, input_dim=input_dim,
        hidden_dim=HIDDEN_DIM, num_encoder_res=NUM_ENCODER_RES,
        num_decoder_res=NUM_DECODER_RES,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params (input_dim={input_dim})")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    start_epoch = 0
    best_loss = float("inf")
    run_name = None
    best_path = os.path.join(CHECKPOINT_DIR, "best_v1dec.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        if ckpt.get("model_version") == "v1dec":
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0)
            best_loss = ckpt.get("val_loss", float("inf"))
            run_name = ckpt.get("run_name")
            print(f"Resumed from {best_path} (epoch {start_epoch}, loss {best_loss:.2e})")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6,
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if not run_name:
        run_name = "walk_mirror_v1dec_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(LOG_DIR, run_name))

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
                "r_rest_positions": r_rest_positions,
                "model_version": "v1dec",
                "mirror_trained": True,
                "run_name": run_name,
            }, os.path.join(CHECKPOINT_DIR, "best_v1dec.pt"))

    writer.close()
    print(f"\nDone. Best MSE: {best_loss:.2e}")
    print(f"Checkpoint: {CHECKPOINT_DIR}/best_v1dec.pt")


if __name__ == "__main__":
    train()
