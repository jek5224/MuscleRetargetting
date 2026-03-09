"""Train V1PCA DistillNetV1PCA on walk.bvh with L+R UpLeg data (mirrored to L-canonical).

V1PCA uses PCA output basis with 25 batched decoders (bmm) for parallel execution.
Same 4-DOF input as V1, V1 encoder architecture (512 hidden, 3 ResBlocks).

L side: used directly. R side: DOFs mirrored (negate hip X), displacements relative to own rest.
This doubles the training data and produces a network in L-canonical space.

Usage: python -m volume_distill.walk_overfit_mirror_v2
"""
import os
import gc
import glob
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA

from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH
from volume_distill.model import DistillNetV1PCA


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
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
HIDDEN_DIM = 512
NUM_ENCODER_RES = 3
NUM_DECODER_RES = 2
PCA_K = 64

# L hip (3 DOFs) + L knee (1 DOF)
L_DOF_INDICES = [6, 7, 8, 9]
# R hip (3 DOFs) + R knee (1 DOF)
R_DOF_INDICES = [18, 19, 20, 21]


def mirror_dofs_r_to_l(dofs):
    """Mirror R hip+knee DOFs to L-canonical space.

    Hip is a ball joint (exponential map): negate X component (index 0).
    Knee is a revolute joint: same sign.
    Input: (N, 4) array of [hip_x, hip_y, hip_z, knee]
    """
    mirrored = dofs.copy()
    mirrored[:, 0] *= -1  # negate hip rotation X
    return mirrored


class SimpleDataset(Dataset):
    def __init__(self, input_dofs, pca_targets, muscle_names, indices):
        self.input_dofs = input_dofs
        self.pca_targets = pca_targets
        self.muscle_names = muscle_names
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.input_dofs[i]
        targets = {name: self.pca_targets[name][i] for name in self.muscle_names}
        return x, targets


def collate_fn(batch):
    xs, tds = zip(*batch)
    x = torch.stack(xs)
    names = list(tds[0].keys())
    targets = {name: torch.stack([t[name] for t in tds]) for name in names}
    return x, targets


def load_side(cache_dir, prefix, mocap, ref_R, ref_t, N):
    """Load one side's cache and compute pelvis-local displacements."""
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
        positions = data["positions"]  # (N, V, 3)

        if len(frames) != N:
            print(f"  SKIP {mname}: {len(frames)} frames != {N}")
            continue

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
    """Load L+R walk cache, mirror R to L-canonical, compute PCA + derivative features."""
    print("=== Preprocessing walk.bvh (L+R mirrored, V2) ===")
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

    # Load L side
    print("\n--- L side ---")
    l_names, l_rest, l_disp = load_side(L_CACHE_DIR, "L", mocap, ref_R, ref_t, N)

    # Load R side (mirrored to L-canonical)
    print("\n--- R side (mirrored to L-canonical) ---")
    r_names, r_rest, r_disp = load_side(R_CACHE_DIR, "R", mocap, ref_R, ref_t, N)

    # Merge: use L muscle names as canonical
    muscle_names = l_names
    rest_positions = {}
    r_rest_positions = {}
    all_disps = {}

    for mname in muscle_names:
        if mname in l_disp and mname in r_disp:
            rest_positions[mname] = l_rest[mname]
            r_rest_positions[mname] = r_rest[mname]
            all_disps[mname] = torch.cat([l_disp[mname], r_disp[mname]], dim=0)  # (2N, V, 3)
        elif mname in l_disp:
            rest_positions[mname] = l_rest[mname]
            r_rest_positions[mname] = l_rest[mname]
            all_disps[mname] = l_disp[mname]

    # Raw 4 DOFs: L direct + R mirrored
    l_dofs = mocap[:, L_DOF_INDICES].astype(np.float32)  # (N, 4)
    r_dofs = mocap[:, R_DOF_INDICES].astype(np.float32)  # (N, 4)
    r_dofs_mirrored = mirror_dofs_r_to_l(r_dofs)

    input_dofs = torch.from_numpy(
        np.concatenate([l_dofs, r_dofs_mirrored], axis=0)
    )  # (2N, 4)

    total = input_dofs.shape[0]

    # Compute PCA per muscle
    print("\n--- Computing PCA ---")
    pca_components = {}
    pca_means = {}
    pca_stds = {}
    pca_targets = {}
    muscle_name_to_idx = {name: i for i, name in enumerate(muscle_names)}

    for mname in muscle_names:
        disp = all_disps[mname]  # (2N, V, 3)
        Nt, V, _ = disp.shape
        flat = disp.reshape(Nt, V * 3).numpy()

        pca = PCA(n_components=PCA_K)
        coeffs = pca.fit_transform(flat)  # (2N, K)

        pca_components[mname] = torch.from_numpy(pca.components_.astype(np.float32))   # (K, V*3)
        pca_means[mname] = torch.from_numpy(pca.mean_.astype(np.float32))              # (V*3,)

        coeff_std = np.std(coeffs, axis=0)
        coeff_std[coeff_std < 1e-8] = 1.0
        coeffs_norm = coeffs / coeff_std

        pca_stds[mname] = torch.from_numpy(coeff_std.astype(np.float32))
        pca_targets[mname] = torch.from_numpy(coeffs_norm.astype(np.float32))

        explained = pca.explained_variance_ratio_.sum() * 100
        print(f"  {mname}: K={PCA_K}, explained={explained:.1f}%, "
              f"std range [{coeff_std.min():.4f}, {coeff_std.max():.4f}]")
        del pca, coeffs, coeffs_norm
        gc.collect()

    del all_disps
    gc.collect()

    print(f"\n  Combined: {total} samples ({N} L + {N} R), {len(muscle_names)} muscles")
    print(f"  Input: {input_dofs.shape}")

    all_idx = torch.arange(total)
    return (input_dofs, pca_targets, pca_components, pca_means, pca_stds,
            muscle_names, muscle_name_to_idx, rest_positions, r_rest_positions, all_idx)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    (input_dofs, pca_targets, pca_components, pca_means, pca_stds,
     muscle_names, muscle_name_to_idx, rest_positions, r_rest_positions, indices) = preprocess()

    num_muscles = len(muscle_names)
    input_dim = input_dofs.shape[1]

    ds = SimpleDataset(input_dofs, pca_targets, muscle_names, indices)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    print(f"\nTrain: {len(ds)} samples, {num_muscles} muscles, input_dim={input_dim}")

    model = DistillNetV1PCA(
        muscle_names=muscle_names,
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_encoder_res=NUM_ENCODER_RES,
        num_decoder_res=NUM_DECODER_RES,
        pca_k=PCA_K,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Resume from best.pt if it exists and is v1_pca
    start_epoch = 0
    best_path = os.path.join(CHECKPOINT_DIR, "best_v1_pca.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        if ckpt.get("model_version") == "v1_pca":
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0)
            print(f"Resumed from {best_path} (epoch {start_epoch}, loss {ckpt.get('val_loss', '?'):.2e})")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=500, T_mult=2,
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    run_name = "walk_mirror_v1pca_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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

            loss = 0.0
            for name in muscle_names:
                loss += ((preds[name] - targets[name]) ** 2).mean()
            loss /= num_muscles

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
                "model_version": "v1_pca",
                "mirror_trained": True,
                "input_dim": input_dim,
                "hidden_dim": HIDDEN_DIM,
                "num_encoder_res": NUM_ENCODER_RES,
                "num_decoder_res": NUM_DECODER_RES,
                "pca_k": PCA_K,
                "num_muscles": num_muscles,
                "muscle_names": muscle_names,
                "muscle_name_to_idx": muscle_name_to_idx,
                "pca_components": pca_components,
                "pca_means": pca_means,
                "pca_stds": pca_stds,
                "rest_positions": rest_positions,
                "r_rest_positions": r_rest_positions,
            }, os.path.join(CHECKPOINT_DIR, "best_v1_pca.pt"))

    writer.close()
    print(f"\nDone. Best total loss: {best_loss:.2e}")
    print(f"Checkpoint: {CHECKPOINT_DIR}/best_v1_pca.pt")


if __name__ == "__main__":
    train()
