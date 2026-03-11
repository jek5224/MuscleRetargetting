"""Train V1Dec DistillNetV1Dec on dance.bvh with L+R UpLeg+LowLeg data (mirrored to L-canonical).

V1Dec uses batched decoders (bmm) for parallel execution, raw V×3 displacement output.
7-DOF input (hip 3 + knee 1 + ankle 3).

Usage: python -m volume_distill.dance_overfit_v1dec
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
BVH_PATH = "data/motion/dance.bvh"
L_CACHE_DIRS = [
    ("data/motion_cache/dance/L_UpLeg", "L"),
    ("data/motion_cache/dance/L_LowLeg", "L"),
]
R_CACHE_DIRS = [
    ("data/motion_cache/dance/R_UpLeg", "R"),
    ("data/motion_cache/dance/R_LowLeg", "R"),
]
CHECKPOINT_DIR = "volume_distill/dance_v1dec_checkpoints"
LOG_DIR = "volume_distill/dance_v1dec_runs"

# === Training ===
EPOCHS = 10000
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP = 0.1
HIDDEN_DIM = 768
NUM_ENCODER_RES = 3
NUM_DECODER_RES = 2

# L hip (3 DOFs) + L knee (1 DOF) + L ankle (3 DOFs)
L_DOF_INDICES = [6, 7, 8, 9, 10, 11, 12]
# R hip (3 DOFs) + R knee (1 DOF) + R ankle (3 DOFs)
R_DOF_INDICES = [18, 19, 20, 21, 22, 23, 24]


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


def load_side_chunked(cache_dir, prefix, mocap, ref_R, ref_t, N):
    """Load one side's cache from multiple chunks and compute pelvis-local displacements."""
    # Find unique muscle names from chunk_0000 files
    chunk0_files = sorted(glob.glob(os.path.join(cache_dir, f"{prefix}_*_chunk_0000.npz")))
    muscle_names = []
    rest_positions = {}
    displacements = {}

    for npz_path in chunk0_files:
        basename = os.path.basename(npz_path)
        mname = basename.replace("_chunk_0000.npz", "")
        canonical_name = "L_" + mname[2:] if mname.startswith("R_") else mname

        # Load all chunks for this muscle
        chunk_pattern = os.path.join(cache_dir, f"{mname}_chunk_*.npz")
        chunk_files = sorted(glob.glob(chunk_pattern))

        all_frames = []
        all_positions = []
        for cf in chunk_files:
            data = np.load(cf)
            all_frames.append(data["frames"])
            all_positions.append(data["positions"])

        frames = np.concatenate(all_frames)
        positions = np.concatenate(all_positions)  # (total_frames, V, 3)

        if len(frames) != N:
            print(f"  SKIP {mname}: {len(frames)} frames != {N}")
            continue

        # Sort by frame index (chunks should be ordered, but be safe)
        sort_idx = np.argsort(frames)
        positions = positions[sort_idx]

        # World → pelvis-local
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
    print("=== Preprocessing dance.bvh (L+R mirrored, UpLeg+LowLeg, V1Dec) ===")
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

    # Load all L-side regions
    l_rest_all = {}
    l_disp_all = {}
    l_names_all = []
    for cache_dir, prefix in L_CACHE_DIRS:
        if not os.path.isdir(cache_dir):
            print(f"\n--- SKIP {cache_dir} (not found) ---")
            continue
        print(f"\n--- L side: {cache_dir} ---")
        names, rest, disp = load_side_chunked(cache_dir, prefix, mocap, ref_R, ref_t, N)
        for n in names:
            if n not in l_rest_all:
                l_names_all.append(n)
                l_rest_all[n] = rest[n]
                l_disp_all[n] = disp[n]

    # Load all R-side regions
    r_rest_all = {}
    r_disp_all = {}
    for cache_dir, prefix in R_CACHE_DIRS:
        if not os.path.isdir(cache_dir):
            print(f"\n--- SKIP {cache_dir} (not found) ---")
            continue
        print(f"\n--- R side (mirrored to L-canonical): {cache_dir} ---")
        names, rest, disp = load_side_chunked(cache_dir, prefix, mocap, ref_R, ref_t, N)
        for n in names:
            if n not in r_rest_all:
                r_rest_all[n] = rest[n]
                r_disp_all[n] = disp[n]

    has_l = len(l_disp_all) > 0
    has_r = len(r_disp_all) > 0

    # Build canonical muscle name list from whichever side(s) are available
    if has_l:
        muscle_names = l_names_all
    elif has_r:
        muscle_names = list(r_disp_all.keys())
    else:
        raise RuntimeError("No L or R data found — nothing to train on")

    rest_positions = {}
    r_rest_positions = {}
    displacements = {}

    for mname in muscle_names:
        has_l_m = mname in l_disp_all
        has_r_m = mname in r_disp_all
        if has_l_m and has_r_m:
            rest_positions[mname] = l_rest_all[mname]
            r_rest_positions[mname] = r_rest_all[mname]
            displacements[mname] = torch.cat([l_disp_all[mname], r_disp_all[mname]], dim=0)
        elif has_l_m:
            rest_positions[mname] = l_rest_all[mname]
            r_rest_positions[mname] = l_rest_all[mname]
            displacements[mname] = l_disp_all[mname]
        elif has_r_m:
            rest_positions[mname] = r_rest_all[mname]
            r_rest_positions[mname] = r_rest_all[mname]
            displacements[mname] = r_disp_all[mname]

    l_dofs = mocap[:, L_DOF_INDICES].astype(np.float32)
    r_dofs = mocap[:, R_DOF_INDICES].astype(np.float32)
    r_dofs_mirrored = mirror_dofs_r_to_l(r_dofs)

    if has_l and has_r:
        input_dofs = torch.from_numpy(np.concatenate([l_dofs, r_dofs_mirrored], axis=0))
        total = input_dofs.shape[0]
        print(f"\n  Combined: {total} samples ({N} L + {N} R), {len(muscle_names)} muscles")
    elif has_r:
        input_dofs = torch.from_numpy(r_dofs_mirrored)
        total = input_dofs.shape[0]
        print(f"\n  R-only (mirrored): {total} samples, {len(muscle_names)} muscles (L not baked yet)")
    else:
        input_dofs = torch.from_numpy(l_dofs)
        total = input_dofs.shape[0]
        print(f"\n  L-only: {total} samples, {len(muscle_names)} muscles (R not baked yet)")
    print(f"  Input: {input_dofs.shape}")

    all_idx = torch.arange(total)
    return input_dofs, displacements, muscle_names, rest_positions, r_rest_positions, all_idx


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    input_dofs, displacements, muscle_names, rest_positions, r_rest_positions, indices = preprocess()

    ds = SimpleDataset(input_dofs, displacements, muscle_names, indices)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
                        num_workers=2, pin_memory=True)
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
        if (ckpt.get("model_version") == "v1dec"
                and ckpt.get("input_dim") == input_dim
                and set(ckpt.get("muscle_vertex_counts", {}).keys()) == set(muscle_names)):
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0)
            best_loss = ckpt.get("val_loss", float("inf"))
            run_name = ckpt.get("run_name")
            print(f"Resumed from {best_path} (epoch {start_epoch}, loss {best_loss:.2e})")
        else:
            print(f"Checkpoint incompatible, training from scratch")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6,
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if not run_name:
        run_name = "dance_v1dec_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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
