import gc
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.decomposition import PCA


class MuscleDistillDataset(Dataset):
    def __init__(self, preprocessed_path, split="train"):
        """Load preprocessed muscle data from a single .pt file.

        For multiple files, use load_train_val() instead to avoid duplicating
        data in memory.
        """
        data = torch.load(preprocessed_path, weights_only=False)
        if split == "train":
            self.indices = data["train_indices"]
        elif split == "val":
            self.indices = data["val_indices"]
        else:
            raise ValueError(f"Unknown split: {split}")
        self.input_dofs = data["input_dofs"]
        self.displacements = data["displacements"]
        self.muscle_names = data["muscle_names"]
        self.rest_positions = data["rest_positions"]
        self.anchor_vertices = data.get("anchor_vertices", {})

    @classmethod
    def _from_shared(cls, input_dofs, displacements, muscle_names,
                     rest_positions, anchor_vertices, indices):
        """Create dataset from pre-loaded shared data (no copy)."""
        obj = cls.__new__(cls)
        obj.input_dofs = input_dofs
        obj.displacements = displacements
        obj.muscle_names = muscle_names
        obj.rest_positions = rest_positions
        obj.anchor_vertices = anchor_vertices
        obj.indices = indices
        return obj

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        frame_idx = self.indices[idx]
        x = self.input_dofs[frame_idx]
        targets = {
            name: self.displacements[name][frame_idx].reshape(-1)
            for name in self.muscle_names
        }
        return x, targets


def load_train_val(preprocessed_paths):
    """Load one or more preprocessed files and return (train_ds, val_ds).

    Both datasets share the same underlying tensors to avoid duplicating
    ~7GB of displacement data in memory. Files are loaded sequentially
    with cleanup to keep peak memory low.
    """
    if isinstance(preprocessed_paths, str):
        preprocessed_paths = [preprocessed_paths]

    # --- Load first file (canonical rest pose) ---
    print(f"Loading {preprocessed_paths[0]}...")
    d = torch.load(preprocessed_paths[0], weights_only=False)
    muscle_names = d["muscle_names"]
    rest_positions = d["rest_positions"]
    anchor_vertices = d.get("anchor_vertices", {})

    all_dofs = [d["input_dofs"]]
    all_disps = {name: [d["displacements"][name]] for name in muscle_names}
    train_idx = [d["train_indices"]]
    val_idx = [d["val_indices"]]
    frame_offset = d["input_dofs"].shape[0]
    del d
    gc.collect()

    # --- Load remaining files sequentially ---
    for p in preprocessed_paths[1:]:
        print(f"Loading {p}...")
        d = torch.load(p, weights_only=False)

        # Validate muscles match
        if set(d["muscle_names"]) != set(muscle_names):
            raise ValueError(
                f"Muscle mismatch: canonical has {muscle_names}, "
                f"{p} has {d['muscle_names']}"
            )

        all_dofs.append(d["input_dofs"])

        for name in muscle_names:
            disp = d["displacements"][name]  # (N, V, 3)
            # Adjust in-place so displacements are relative to canonical rest
            adjustment = d["rest_positions"][name] - rest_positions[name]
            disp += adjustment[None, :, :]
            all_disps[name].append(disp)

        train_idx.append(d["train_indices"] + frame_offset)
        val_idx.append(d["val_indices"] + frame_offset)
        frame_offset += d["input_dofs"].shape[0]
        del d
        gc.collect()

    # --- Concatenate (per-muscle to limit peak memory) ---
    print("Concatenating datasets...")
    input_dofs = torch.cat(all_dofs, dim=0)
    del all_dofs

    displacements = {}
    for name in muscle_names:
        displacements[name] = torch.cat(all_disps[name], dim=0)
        del all_disps[name]
    del all_disps
    gc.collect()

    train_indices = torch.cat(train_idx, dim=0)
    val_indices = torch.cat(val_idx, dim=0)
    print(f"Combined: {frame_offset} frames, train={len(train_indices)}, val={len(val_indices)}")

    # --- Build shared datasets ---
    train_ds = MuscleDistillDataset._from_shared(
        input_dofs, displacements, muscle_names,
        rest_positions, anchor_vertices, train_indices,
    )
    val_ds = MuscleDistillDataset._from_shared(
        input_dofs, displacements, muscle_names,
        rest_positions, anchor_vertices, val_indices,
    )
    return train_ds, val_ds


def distill_collate_fn(batch):
    xs, target_dicts = zip(*batch)
    x_batch = torch.stack(xs)
    muscle_names = list(target_dicts[0].keys())
    target_batch = {
        name: torch.stack([t[name] for t in target_dicts])
        for name in muscle_names
    }
    return x_batch, target_batch


# ---------- V2: PCA targets + temporal frame pairs ----------

class MuscleDistillDatasetV2(Dataset):
    """V2 dataset: returns frame pairs (t, prev) for temporal consistency loss."""

    def __init__(self, input_dofs, prev_frame_idx, pca_targets,
                 muscle_names, indices):
        self.input_dofs = input_dofs          # (N_total, 20)
        self.prev_frame_idx = prev_frame_idx  # (N_total,)
        self.pca_targets = pca_targets        # {name: (N_total, K)}
        self.muscle_names = muscle_names
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        frame_t = self.indices[idx]
        frame_prev = self.prev_frame_idx[frame_t]

        x_t = self.input_dofs[frame_t]
        x_prev = self.input_dofs[frame_prev]

        targets_t = {name: self.pca_targets[name][frame_t] for name in self.muscle_names}
        targets_prev = {name: self.pca_targets[name][frame_prev] for name in self.muscle_names}

        return x_t, x_prev, targets_t, targets_prev


def distill_collate_fn_v2(batch):
    """Collate V2 batch: stack frame pairs for single forward pass."""
    x_ts, x_prevs, tgt_ts, tgt_prevs = zip(*batch)

    x_t = torch.stack(x_ts)          # (B, 20)
    x_prev = torch.stack(x_prevs)    # (B, 20)
    x_combined = torch.cat([x_t, x_prev], dim=0)  # (2B, 20)

    muscle_names = list(tgt_ts[0].keys())
    targets_t = {name: torch.stack([t[name] for t in tgt_ts]) for name in muscle_names}
    targets_prev = {name: torch.stack([t[name] for t in tgt_prevs]) for name in muscle_names}

    return x_combined, targets_t, targets_prev


def load_train_val_v2(preprocessed_paths, pca_k=64):
    """Load preprocessed files, compute PCA, return V2 (train_ds, val_ds).

    PCA is computed on the combined adjusted displacement data to ensure
    the basis captures variance from all motions.

    Returns:
        train_ds, val_ds: MuscleDistillDatasetV2 instances
        pca_components: {name: Tensor (K, V*3)}
        pca_means: {name: Tensor (V*3,)}
        pca_stds: {name: Tensor (K,)} — per-component std for z-score denormalization
        muscle_name_to_idx: {name: int}
        rest_positions: {name: Tensor (V, 3)}
    """
    if isinstance(preprocessed_paths, str):
        preprocessed_paths = [preprocessed_paths]

    # --- Load first file (canonical rest pose) ---
    print(f"Loading {preprocessed_paths[0]}...")
    d = torch.load(preprocessed_paths[0], weights_only=False)
    muscle_names = d["muscle_names"]
    rest_positions = d["rest_positions"]

    all_dofs = [d["input_dofs"]]
    all_disps = {name: [d["displacements"][name]] for name in muscle_names}
    all_prev_idx = [d["prev_frame_idx"]]
    train_idx = [d["train_indices"]]
    val_idx = [d["val_indices"]]
    frame_offset = d["input_dofs"].shape[0]
    del d
    gc.collect()

    # --- Load remaining files sequentially ---
    for p in preprocessed_paths[1:]:
        print(f"Loading {p}...")
        d = torch.load(p, weights_only=False)

        if set(d["muscle_names"]) != set(muscle_names):
            raise ValueError(
                f"Muscle mismatch: canonical has {muscle_names}, "
                f"{p} has {d['muscle_names']}"
            )

        all_dofs.append(d["input_dofs"])

        for name in muscle_names:
            disp = d["displacements"][name]  # (N, V, 3)
            adjustment = d["rest_positions"][name] - rest_positions[name]
            disp += adjustment[None, :, :]
            all_disps[name].append(disp)

        # Adjust prev_frame_idx: add offset, but boundary frame (0) points to self
        prev_idx = d["prev_frame_idx"].clone()
        # Frame 0 of this motion → itself (at frame_offset)
        prev_idx[0] = 0
        prev_idx += frame_offset

        all_prev_idx.append(prev_idx)
        train_idx.append(d["train_indices"] + frame_offset)
        val_idx.append(d["val_indices"] + frame_offset)
        frame_offset += d["input_dofs"].shape[0]
        del d
        gc.collect()

    # --- Concatenate DOFs and prev_frame_idx ---
    print("Concatenating datasets...")
    input_dofs = torch.cat(all_dofs, dim=0)
    del all_dofs
    prev_frame_idx = torch.cat(all_prev_idx, dim=0)
    del all_prev_idx

    train_indices = torch.cat(train_idx, dim=0)
    val_indices = torch.cat(val_idx, dim=0)

    # --- Compute PCA per muscle, then free displacement memory ---
    pca_components = {}
    pca_means = {}
    pca_stds = {}
    pca_targets = {}
    muscle_name_to_idx = {name: i for i, name in enumerate(muscle_names)}

    for name in muscle_names:
        print(f"  PCA for {name}...")
        disp_cat = torch.cat(all_disps[name], dim=0)  # (N_total, V, 3)
        del all_disps[name]

        N, V, _ = disp_cat.shape
        flat = disp_cat.reshape(N, V * 3).numpy()  # (N, V*3)
        del disp_cat

        pca = PCA(n_components=pca_k)
        coeffs = pca.fit_transform(flat)  # (N, K)
        del flat

        pca_components[name] = torch.from_numpy(pca.components_.astype(np.float32))   # (K, V*3)
        pca_means[name] = torch.from_numpy(pca.mean_.astype(np.float32))              # (V*3,)

        # Z-score normalize coefficients so all components contribute equally to loss
        coeff_std = np.std(coeffs, axis=0)                                             # (K,)
        coeff_std[coeff_std < 1e-8] = 1.0
        coeffs_norm = coeffs / coeff_std                                               # (N, K)

        pca_stds[name] = torch.from_numpy(coeff_std.astype(np.float32))               # (K,)
        pca_targets[name] = torch.from_numpy(coeffs_norm.astype(np.float32))          # (N, K)

        explained = pca.explained_variance_ratio_.sum() * 100
        print(f"    {name}: K={pca_k}, explained variance={explained:.1f}%, "
              f"std range [{coeff_std.min():.4f}, {coeff_std.max():.4f}]")
        del pca, coeffs, coeffs_norm
        gc.collect()

    del all_disps
    gc.collect()

    print(f"Combined: {frame_offset} frames, train={len(train_indices)}, val={len(val_indices)}")

    # --- Build V2 datasets ---
    train_ds = MuscleDistillDatasetV2(
        input_dofs, prev_frame_idx, pca_targets, muscle_names, train_indices,
    )
    val_ds = MuscleDistillDatasetV2(
        input_dofs, prev_frame_idx, pca_targets, muscle_names, val_indices,
    )

    return train_ds, val_ds, pca_components, pca_means, pca_stds, muscle_name_to_idx, rest_positions
