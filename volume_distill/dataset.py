import gc
import torch
from torch.utils.data import Dataset


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
