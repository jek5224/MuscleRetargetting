"""Inference utilities for the trained distillation model.

Supports both V1 (DistillNet) and V2 (DistillNetV2) checkpoints.

Usage:
    from volume_distill.dance.evaluate import load_model, predict_frame
"""
import numpy as np
import torch

from volume_distill.model import DistillNet, DistillNetV2


def load_model(checkpoint_path, device=None):
    """Load a trained model from a checkpoint.

    Detects model_version in checkpoint to choose V1 or V2.

    Returns (model, metadata_dict).
    For V1: metadata = {"muscle_vertex_counts": ...}
    For V2: metadata = {"muscle_name_to_idx": ..., "pca_components": ..., "pca_means": ..., "window_size": ...}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    version = ckpt.get("model_version", "v1")

    if version == "v2":
        model = DistillNetV2(
            num_muscles=ckpt["num_muscles"],
            muscle_name_to_idx=ckpt["muscle_name_to_idx"],
            input_dim=ckpt.get("input_dim", 20),
            hidden_dim=ckpt.get("hidden_dim", 768),
            num_encoder_res=ckpt.get("num_encoder_res", 5),
            num_decoder_res=ckpt.get("num_decoder_res", 3),
            embed_dim=ckpt.get("embed_dim", 64),
            pca_k=ckpt.get("pca_k", 64),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        # Attach PCA data for inference
        model._pca_components = ckpt["pca_components"]
        model._pca_means = ckpt["pca_means"]
        model._pca_stds = ckpt.get("pca_stds")  # z-score denorm (None for old ckpts)
        model._muscle_name_to_idx = ckpt["muscle_name_to_idx"]
        metadata = {
            "muscle_name_to_idx": ckpt["muscle_name_to_idx"],
            "pca_components": ckpt["pca_components"],
            "pca_means": ckpt["pca_means"],
            "pca_stds": ckpt.get("pca_stds"),
            "window_size": ckpt.get("window_size", 5),
            "model_version": "v2",
            "rest_positions": ckpt.get("rest_positions"),
            "epoch": ckpt.get("epoch", "?"),
            "val_loss": ckpt.get("val_loss"),
        }
        return model, metadata
    else:
        muscle_vertex_counts = ckpt["muscle_vertex_counts"]
        input_dim = ckpt.get("input_dim", 4)
        model = DistillNet(
            muscle_vertex_counts, input_dim=input_dim,
            hidden_dim=ckpt.get("hidden_dim", 512),
            num_encoder_res=ckpt.get("num_encoder_res", 3),
            num_decoder_res=ckpt.get("num_decoder_res", 2),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        metadata = {
            "muscle_vertex_counts": muscle_vertex_counts,
            "model_version": "v1",
            "rest_positions": ckpt.get("rest_positions"),
            "epoch": ckpt.get("epoch", "?"),
            "val_loss": ckpt.get("val_loss"),
        }
        return model, metadata


def predict_frame(model, dofs, rest_positions, device=None):
    """Predict vertex positions for a single frame.

    For V1: dofs is (8,) array [pos + vel]
    For V2: dofs is (20,) array [5-frame window], reconstruction from PCA handled internally.

    Returns:
        dict of {muscle_name: ndarray[V, 3]} predicted positions in bone-local frame
    """
    if device is None:
        device = next(model.parameters()).device
    x = torch.tensor(
        dofs, dtype=torch.float32,
    ).unsqueeze(0).to(device)

    is_v2 = isinstance(model, DistillNetV2)

    with torch.no_grad():
        preds = model(x)

    result = {}
    if is_v2:
        # preds: {muscle_idx: (1, K)} — reconstruct from PCA
        idx_to_name = {v: k for k, v in model._muscle_name_to_idx.items()}
        for m_idx, coeffs in preds.items():
            name = idx_to_name[m_idx]
            coeffs_np = coeffs[0].cpu().numpy()                       # (K,)
            # Denormalize z-scored coefficients if pca_stds available
            if model._pca_stds is not None and name in model._pca_stds:
                coeffs_np = coeffs_np * model._pca_stds[name].numpy() # (K,)
            components = model._pca_components[name].numpy()           # (K, V*3)
            mean = model._pca_means[name].numpy()                     # (V*3,)
            disp_flat = coeffs_np @ components + mean                  # (V*3,)
            disp = disp_flat.reshape(-1, 3)                            # (V, 3)
            rest = rest_positions[name]
            if isinstance(rest, torch.Tensor):
                rest = rest.cpu().numpy()
            result[name] = rest + disp
    else:
        # V1: preds = {name: (1, V*3)}
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
