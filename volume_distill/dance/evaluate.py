"""Inference utilities for the trained distillation model.

Usage:
    from volume_distill.dance.evaluate import load_model, predict_frame
"""
import numpy as np
import torch

from volume_distill.model import DistillNet


def load_model(checkpoint_path, device=None):
    """Load a trained DistillNet from a checkpoint.

    Returns (model, muscle_vertex_counts).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    muscle_vertex_counts = ckpt["muscle_vertex_counts"]
    model = DistillNet(muscle_vertex_counts).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, muscle_vertex_counts


def predict_frame(model, hip_dofs, knee_dof, rest_positions, device=None):
    """Predict vertex positions for a single frame.

    Args:
        model: trained DistillNet
        hip_dofs: array-like of 3 hip DOF values
        knee_dof: scalar knee DOF value
        rest_positions: dict of {muscle_name: Tensor[V, 3]} in pelvis-local frame

    Returns:
        dict of {muscle_name: ndarray[V, 3]} predicted positions in pelvis-local frame
    """
    if device is None:
        device = next(model.parameters()).device
    x = torch.tensor(
        [hip_dofs[0], hip_dofs[1], hip_dofs[2], knee_dof],
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(x)

    result = {}
    for name, disp_flat in preds.items():
        disp = disp_flat[0].cpu().reshape(-1, 3)
        rest = rest_positions[name]
        if isinstance(rest, torch.Tensor):
            rest = rest.cpu()
        result[name] = (rest + disp).numpy()
    return result


def per_muscle_rmse(model, data_path, device=None):
    """Compute per-muscle RMSE on the validation set.

    Returns dict of {muscle_name: rmse_in_meters}.
    """
    if device is None:
        device = next(model.parameters()).device
    data = torch.load(data_path, weights_only=False)
    val_indices = data["val_indices"]
    input_dofs = data["input_dofs"]
    displacements = data["displacements"]
    muscle_names = data["muscle_names"]

    model.eval()
    per_muscle_se = {name: 0.0 for name in muscle_names}
    per_muscle_count = {name: 0 for name in muscle_names}

    batch_size = 512
    for start in range(0, len(val_indices), batch_size):
        idx = val_indices[start : start + batch_size]
        x = input_dofs[idx].to(device)
        with torch.no_grad():
            preds = model(x)
        for name in muscle_names:
            gt = displacements[name][idx].reshape(len(idx), -1).to(device)
            se = ((preds[name] - gt) ** 2).sum().item()
            per_muscle_se[name] += se
            per_muscle_count[name] += gt.numel()

    rmse = {}
    for name in muscle_names:
        rmse[name] = np.sqrt(per_muscle_se[name] / per_muscle_count[name])
    return rmse
