import torch
from torch.utils.data import Dataset


class MuscleDistillDataset(Dataset):
    def __init__(self, preprocessed_path, split="train"):
        """Load preprocessed muscle data.

        Args:
            preprocessed_path: path to a single .pt file (str), or a list of
                paths to combine multiple motion sources.
            split: "train" or "val"
        """
        if isinstance(preprocessed_path, str):
            preprocessed_path = [preprocessed_path]

        if len(preprocessed_path) == 1:
            # Single source — original behavior
            data = torch.load(preprocessed_path[0], weights_only=False)
            if split == "train":
                self.indices = data["train_indices"]
            elif split == "val":
                self.indices = data["val_indices"]
            else:
                raise ValueError(f"Unknown split: {split}")
            self.input_dofs = data["input_dofs"]
            self.displacements = data["displacements"]
            self.muscle_names = data["muscle_names"]
        else:
            # Multiple sources — combine with rest-pose alignment
            self._load_combined(preprocessed_path, split)

    def _load_combined(self, paths, split):
        """Load and combine multiple preprocessed files."""
        datasets = [torch.load(p, weights_only=False) for p in paths]

        # Use first file as canonical reference
        canonical = datasets[0]
        self.muscle_names = canonical["muscle_names"]
        rest_canonical = canonical["rest_positions"]  # {name: (V, 3)}

        # Validate all files have the same muscles
        for i, d in enumerate(datasets[1:], 1):
            if set(d["muscle_names"]) != set(self.muscle_names):
                raise ValueError(
                    f"Muscle mismatch: file 0 has {self.muscle_names}, "
                    f"file {i} has {d['muscle_names']}"
                )

        # Combine input_dofs and displacements, aligning rest poses
        all_dofs = []
        all_disps = {name: [] for name in self.muscle_names}
        frame_offset = 0
        all_train_indices = []
        all_val_indices = []

        for d in datasets:
            all_dofs.append(d["input_dofs"])

            for name in self.muscle_names:
                disp = d["displacements"][name]  # (N, V, 3)
                rest_other = d["rest_positions"][name]  # (V, 3)
                # Adjust: disp_aligned = disp + (rest_other - rest_canonical)
                # so all displacements are relative to the canonical rest pose
                adjustment = rest_other - rest_canonical[name]  # (V, 3)
                disp_aligned = disp + adjustment[None, :, :]
                all_disps[name].append(disp_aligned)

            # Offset indices by cumulative frame count
            if split == "train":
                all_train_indices.append(d["train_indices"] + frame_offset)
            elif split == "val":
                all_val_indices.append(d["val_indices"] + frame_offset)
            else:
                raise ValueError(f"Unknown split: {split}")

            frame_offset += d["input_dofs"].shape[0]

        self.input_dofs = torch.cat(all_dofs, dim=0)
        self.displacements = {
            name: torch.cat(all_disps[name], dim=0)
            for name in self.muscle_names
        }

        if split == "train":
            self.indices = torch.cat(all_train_indices, dim=0)
        else:
            self.indices = torch.cat(all_val_indices, dim=0)

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


def distill_collate_fn(batch):
    xs, target_dicts = zip(*batch)
    x_batch = torch.stack(xs)
    muscle_names = list(target_dicts[0].keys())
    target_batch = {
        name: torch.stack([t[name] for t in target_dicts])
        for name in muscle_names
    }
    return x_batch, target_batch
