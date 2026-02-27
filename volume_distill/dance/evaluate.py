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
    input_dim = ckpt.get("input_dim", 4)
    model = DistillNet(muscle_vertex_counts, input_dim=input_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, muscle_vertex_counts


def predict_frame(model, dofs, rest_positions, device=None):
    """Predict vertex positions for a single frame.

    Args:
        model: trained DistillNet
        dofs: array-like of input DOF values (length must match model input_dim)
        rest_positions: dict of {muscle_name: Tensor[V, 3]} in bone-local frame

    Returns:
        dict of {muscle_name: ndarray[V, 3]} predicted positions in bone-local frame
    """
    if device is None:
        device = next(model.parameters()).device
    x = torch.tensor(
        dofs, dtype=torch.float32,
    ).unsqueeze(0).to(device)

    # Ensure rest_positions tensors are on the model device
    rp_device = {name: t.to(device) if isinstance(t, torch.Tensor) else t
                 for name, t in rest_positions.items()}

    with torch.no_grad():
        preds = model(x, rp_device)

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
    rest_positions = {name: data["rest_positions"][name].to(device) for name in muscle_names}

    model.eval()
    per_muscle_se = {name: 0.0 for name in muscle_names}
    per_muscle_count = {name: 0 for name in muscle_names}

    batch_size = 512
    for start in range(0, len(val_indices), batch_size):
        idx = val_indices[start : start + batch_size]
        x = input_dofs[idx].to(device)
        with torch.no_grad():
            preds = model(x, rest_positions)
        for name in muscle_names:
            gt = displacements[name][idx].reshape(len(idx), -1).to(device)
            se = ((preds[name] - gt) ** 2).sum().item()
            per_muscle_se[name] += se
            per_muscle_count[name] += gt.numel()

    rmse = {}
    for name in muscle_names:
        rmse[name] = np.sqrt(per_muscle_se[name] / per_muscle_count[name])
    return rmse
