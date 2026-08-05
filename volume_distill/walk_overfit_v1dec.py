"""Train V1Dec DistillNetV1Dec on walk-series BVHs (UpLeg L+R only, mirrored to L-canonical).

Multi-BVH: concatenates samples from all baked walk_* BVHs sharing one canonical
rest pose taken from the first BVH's frame 0. UpLeg-only: 4-DOF input (hip 3 + knee 1).
Cache layout: data/motion_cache/<bvh>/_L_UpLeg/, data/motion_cache/<bvh>/_R_UpLeg/.

Usage: python -m volume_distill.walk_overfit_v1dec
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
# 9 baked walk BVHs. walk1_subject1 included (already baked overnight); skip if absent.
WALK_BVHS = [
    "walk",
    "walk1_subject1",
    "walk1_subject2",
    "walk1_subject5",
    "walk2_subject1",
    "walk2_subject3",
    "walk2_subject4",
    "walk3_subject1",
    "walk3_subject2",
]
CHECKPOINT_DIR = "volume_distill/walk_v1dec_checkpoints"
LOG_DIR = "volume_distill/walk_v1dec_runs"

# === Training ===
EPOCHS = 10000
BATCH_SIZE = 512
LR = 3e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
EARLY_STOP_LOSS = 1e-9
HIDDEN_DIM = 768
NUM_ENCODER_RES = 3
NUM_DECODER_RES = 2

# UpLeg-only: hip 3 + knee 1 = 4 DOFs per side
L_DOF_INDICES = [6, 7, 8, 9]
R_DOF_INDICES = [18, 19, 20, 21]


def mirror_dofs_r_to_l(dofs):
    mirrored = dofs.copy()
    mirrored[:, 0] *= -1  # hip rotation X
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


def load_side_chunked(cache_dir, prefix, ref_R, ref_t, N, rest_lookup=None):
    """Load all chunks for muscles in cache_dir, return pelvis-local positions.

    If rest_lookup is None, this is the first BVH/side: each muscle's rest is set
    from frame 0 of its own pelvis-local trajectory and stored in returned dict.
    Otherwise rest_lookup[mname] is reused, and displacements are relative to it.
    """
    chunk0_files = sorted(glob.glob(os.path.join(cache_dir, f"{prefix}_*_chunk_0000.npz")))
    muscle_names = []
    rest_out = {}
    disp_out = {}

    for npz_path in chunk0_files:
        basename = os.path.basename(npz_path)
        mname = basename.replace("_chunk_0000.npz", "")
        canonical_name = "L_" + mname[2:] if mname.startswith("R_") else mname

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

        sort_idx = np.argsort(frames)
        positions = positions[sort_idx]

        # World → pelvis-local
        centered = positions - ref_t[:, None, :]
        R_inv = np.transpose(ref_R, (0, 2, 1))
        local_pos = np.einsum("nij,nvj->nvi", R_inv, centered)

        if prefix == "R":
            local_pos[:, :, 0] *= -1  # mirror X for L-canonical

        if rest_lookup is None:
            rest = local_pos[0].copy().astype(np.float32)
        else:
            rest = rest_lookup.get(canonical_name)
            if rest is None:
                # First time we see this muscle; fall back to local frame 0
                rest = local_pos[0].copy().astype(np.float32)

        disp = (local_pos - rest[None, :, :]).astype(np.float32)

        if canonical_name not in muscle_names:
            muscle_names.append(canonical_name)
        rest_out[canonical_name] = torch.from_numpy(rest)
        disp_out[canonical_name] = torch.from_numpy(disp)

    return muscle_names, rest_out, disp_out


def preprocess_one_bvh(skel_info, root_name, bvh_info, bvh_stem, rest_l=None, rest_r=None):
    """Preprocess a single BVH; return per-muscle disp tensors + dofs.

    rest_l / rest_r: optional dicts of per-muscle rest tensors to reuse for displacement
    (so multiple BVHs share one canonical rest). On first call pass None to record rests.
    Returns (l_disp, r_disp, l_dofs, r_dofs_mirrored, rest_l_out, rest_r_out, names).
    """
    skel = buildFromInfo(skel_info, root_name)
    bvh_path = f"data/motion/{bvh_stem}.bvh"
    if not os.path.exists(bvh_path):
        print(f"  SKIP BVH (file missing): {bvh_path}")
        return None
    bvh = MyBVH(bvh_path, bvh_info, skel)
    mocap = bvh.mocap_refs
    N = mocap.shape[0]
    print(f"\n[{bvh_stem}] frames={N} dofs={skel.getNumDofs()}")

    ref_bn = skel.getBodyNode("Saccrum_Coccyx0")
    ref_R = np.zeros((N, 3, 3), dtype=np.float64)
    ref_t = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        skel.setPositions(mocap[i])
        T = ref_bn.getWorldTransform().matrix()
        ref_R[i] = T[:3, :3]
        ref_t[i] = T[:3, 3]

    l_cache = f"data/motion_cache/{bvh_stem}/_L_UpLeg"
    r_cache = f"data/motion_cache/{bvh_stem}/_R_UpLeg"

    l_names, l_rest_new, l_disp = ([], {}, {})
    r_rest_new, r_disp = ({}, {})
    if os.path.isdir(l_cache):
        l_names, l_rest_new, l_disp = load_side_chunked(l_cache, "L", ref_R, ref_t, N,
                                                         rest_lookup=rest_l)
        print(f"  L: {len(l_names)} muscles loaded ({l_cache})")
    else:
        print(f"  L SKIP: {l_cache}")
    if os.path.isdir(r_cache):
        r_names, r_rest_new, r_disp = load_side_chunked(r_cache, "R", ref_R, ref_t, N,
                                                         rest_lookup=rest_r)
        print(f"  R: {len(r_names)} muscles loaded ({r_cache})")
    else:
        print(f"  R SKIP: {r_cache}")

    l_dofs = mocap[:, L_DOF_INDICES].astype(np.float32)
    r_dofs = mocap[:, R_DOF_INDICES].astype(np.float32)
    r_dofs_mirrored = mirror_dofs_r_to_l(r_dofs)

    return {
        "N": N,
        "l_disp": l_disp,
        "r_disp": r_disp,
        "l_dofs": l_dofs,
        "r_dofs_mirrored": r_dofs_mirrored,
        "rest_l": l_rest_new,
        "rest_r": r_rest_new,
        "muscle_names": l_names,
    }


def preprocess():
    print("=== Preprocessing walk series (UpLeg L+R, V1Dec) ===")
    skel_info, root_name, bvh_info, *_ = saveSkeletonInfo(SKEL_XML)

    rest_l = None
    rest_r = None
    canonical_names = None
    all_l_disp = {}
    all_r_disp = {}
    all_l_dofs = []
    all_r_dofs = []

    for bvh_stem in WALK_BVHS:
        result = preprocess_one_bvh(skel_info, root_name, bvh_info, bvh_stem,
                                    rest_l=rest_l, rest_r=rest_r)
        if result is None:
            continue
        # Lock rest from FIRST BVH that loaded each muscle
        if rest_l is None and result["rest_l"]:
            rest_l = result["rest_l"]
            print(f"  Captured L rest from {bvh_stem}: {len(rest_l)} muscles")
        else:
            for n, t in result["rest_l"].items():
                if n not in rest_l:
                    rest_l[n] = t
        if rest_r is None and result["rest_r"]:
            rest_r = result["rest_r"]
            print(f"  Captured R rest from {bvh_stem}: {len(rest_r)} muscles")
        else:
            for n, t in result["rest_r"].items():
                if n not in rest_r:
                    rest_r[n] = t
        if canonical_names is None:
            canonical_names = list(result["muscle_names"])

        for n, d in result["l_disp"].items():
            all_l_disp.setdefault(n, []).append(d)
        for n, d in result["r_disp"].items():
            all_r_disp.setdefault(n, []).append(d)
        all_l_dofs.append(result["l_dofs"])
        all_r_dofs.append(result["r_dofs_mirrored"])

    if not canonical_names:
        raise RuntimeError("No L muscles loaded — nothing to train on")

    # Concatenate over BVHs
    l_disp_cat = {n: torch.cat(v, dim=0) for n, v in all_l_disp.items()}
    r_disp_cat = {n: torch.cat(v, dim=0) for n, v in all_r_disp.items()}
    l_dofs_cat = np.concatenate(all_l_dofs, axis=0)  # (sum_N, 4)
    r_dofs_cat = np.concatenate(all_r_dofs, axis=0)  # (sum_N, 4)

    # Build per-muscle target by concatenating L (direct) and R (mirrored)
    rest_positions = {}
    r_rest_positions = {}
    displacements = {}
    for mname in canonical_names:
        has_l = mname in l_disp_cat
        has_r = mname in r_disp_cat
        if has_l and has_r:
            rest_positions[mname] = rest_l[mname]
            r_rest_positions[mname] = rest_r[mname]
            displacements[mname] = torch.cat([l_disp_cat[mname], r_disp_cat[mname]], dim=0)
        elif has_l:
            rest_positions[mname] = rest_l[mname]
            r_rest_positions[mname] = rest_l[mname]
            displacements[mname] = l_disp_cat[mname]
        elif has_r:
            rest_positions[mname] = rest_r[mname]
            r_rest_positions[mname] = rest_r[mname]
            displacements[mname] = r_disp_cat[mname]

    input_dofs = torch.from_numpy(np.concatenate([l_dofs_cat, r_dofs_cat], axis=0))
    total = input_dofs.shape[0]
    print(f"\n  Combined samples: {total} ({l_dofs_cat.shape[0]} L + {r_dofs_cat.shape[0]} R)")
    print(f"  Muscles: {len(canonical_names)}  Input shape: {input_dofs.shape}")

    indices = torch.arange(total)
    return input_dofs, displacements, canonical_names, rest_positions, r_rest_positions, indices


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
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0)
            best_loss = ckpt.get("val_loss", float("inf"))
            run_name = ckpt.get("run_name")
            print(f"Resumed from {best_path} (epoch {start_epoch}, loss {best_loss:.2e})")
        else:
            print("Checkpoint incompatible, training from scratch")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=200, min_lr=1e-6,
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if not run_name:
        run_name = "walk_v1dec_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(LOG_DIR, run_name))

    end_epoch = start_epoch + EPOCHS
    prev_loss = float("inf")
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

        avg_loss = epoch_loss / n_batches
        scheduler.step(avg_loss)
        elapsed = time.time() - t0

        if avg_loss > prev_loss * 10 and os.path.exists(best_path):
            old_lr = optimizer.param_groups[0]["lr"]
            new_lr = old_lr * 0.5
            print(f"  *** SPIKE at epoch {epoch}: {avg_loss:.2e} > 10x prev {prev_loss:.2e} ***")
            print(f"  *** Reverting to best, LR {old_lr:.2e} → {new_lr:.2e} ***")
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
            avg_loss = ckpt.get("val_loss", avg_loss)
            prev_loss = avg_loss
            continue

        prev_loss = avg_loss

        writer.add_scalar("loss/train", avg_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if epoch % 100 == 0 or epoch <= start_epoch + 5:
            print(f"Epoch {epoch:5d}/{end_epoch} | MSE: {avg_loss:.2e} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
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
                "bvh_stems": WALK_BVHS,
                "run_name": run_name,
            }, os.path.join(CHECKPOINT_DIR, "best_v1dec.pt"))

        if best_loss < EARLY_STOP_LOSS:
            print(f"\n  *** Early stop: loss {best_loss:.2e} < {EARLY_STOP_LOSS:.0e} at epoch {epoch} ***")
            break

    writer.close()
    print(f"\nDone. Best MSE: {best_loss:.2e}")
    print(f"Checkpoint: {CHECKPOINT_DIR}/best_v1dec.pt")


if __name__ == "__main__":
    train()
