import torch
from torch.utils.data import Dataset


class MuscleDistillDataset(Dataset):
    def __init__(self, preprocessed_path, split="train"):
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
