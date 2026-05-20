"""Bench NN + waypoint compute pipeline (Option A: GPU end-to-end).

Times:
  1. NN forward (batched L+R)
  2. World transform on GPU
  3. Simulated waypoint bary interp on GPU (random tet idx + bary, matching realistic muscle sizes)
  4. CPU sync of final waypoint positions

Reports avg ms/frame over N iters. Indicates if real-time DART feed is feasible.
"""
import argparse
import time
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from volume_distill.model import DistillNetV1Dec


# Typical waypoint counts per muscle (rough order, based on streams×levels)
TYPICAL_WAYPOINTS_PER_MUSCLE = 200


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="volume_distill/walk_v1dec_checkpoints/best_v1dec.pt")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--wp-per-muscle", type=int, default=TYPICAL_WAYPOINTS_PER_MUSCLE)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    muscle_vertex_counts = ckpt["muscle_vertex_counts"]
    input_dim = ckpt["input_dim"]
    hidden_dim = ckpt["hidden_dim"]
    num_enc = ckpt["num_encoder_res"]
    num_dec = ckpt["num_decoder_res"]
    n_muscles = len(muscle_vertex_counts)
    total_verts = sum(muscle_vertex_counts.values())
    print(f"Muscles: {n_muscles}, total verts: {total_verts}, input_dim: {input_dim}")
    print(f"Model: hidden={hidden_dim}, enc_res={num_enc}, dec_res={num_dec}")

    model = DistillNetV1Dec(
        muscle_vertex_counts, input_dim=input_dim,
        hidden_dim=hidden_dim, num_encoder_res=num_enc, num_decoder_res=num_dec,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Rest positions (GPU)
    rest_dev = {n: torch.tensor(ckpt["rest_positions"][n], dtype=torch.float32, device=device)
                for n in muscle_vertex_counts}
    r_rest_dev = {n: torch.tensor(ckpt["r_rest_positions"][n], dtype=torch.float32, device=device)
                  for n in muscle_vertex_counts}

    # Pelvis-to-world transform (rotation+translation) — random unit
    R_world = torch.tensor(np.eye(3), dtype=torch.float32, device=device)
    t_world = torch.zeros(3, dtype=torch.float32, device=device)

    # Simulated waypoint embeddings per muscle: random tet_idx + bary
    # Use TET indices into available tets (estimate: V/4 tets per muscle)
    wp_tet_idx = {}
    wp_bary = {}
    # Simulated tetrahedra per muscle (random vertex 4-tuples)
    tet_indices = {}
    n_wp = args.wp_per_muscle
    for name, n_verts in muscle_vertex_counts.items():
        n_tets = max(1, n_verts // 4)
        # Random tet → 4 vertex idx (valid in [0, n_verts))
        tet_idx_per_tet = torch.randint(0, n_verts, (n_tets, 4), device=device)
        tet_indices[name] = tet_idx_per_tet
        # Each waypoint picks a tet
        wp_tet_idx[name] = torch.randint(0, n_tets, (n_wp,), device=device)
        bary = torch.rand(n_wp, 4, device=device)
        bary = bary / bary.sum(dim=-1, keepdim=True)
        wp_bary[name] = bary

    def one_step(dofs_lr):
        # 1) NN forward (batched L+R, dofs_lr is (2, input_dim))
        with torch.no_grad():
            preds = model(dofs_lr)
        # 2) Apply world transform + assemble per-muscle world_pos GPU tensors
        # 3) For each muscle, do waypoint bary on GPU
        wp_out = {}
        for name, disp_flat in preds.items():
            l_local = rest_dev[name] + disp_flat[0].reshape(-1, 3)  # (V, 3)
            l_world = l_local @ R_world.T + t_world                  # (V, 3)
            # Mirror for R: x flip
            r_local = r_rest_dev[name] + disp_flat[1].reshape(-1, 3)
            r_local = r_local.clone()
            r_local[:, 0] = -r_local[:, 0]
            r_world = r_local @ R_world.T + t_world

            # waypoint bary on L side (sample to test compute):
            #   tet_indices[name] : (T, 4) -> gather 4 verts per tet
            #   wp_tet_idx[name]  : (N,) -> select tet
            #   wp_bary[name]     : (N, 4)
            tet_v = l_world[tet_indices[name]]            # (T, 4, 3)
            tet_v_sel = tet_v[wp_tet_idx[name]]            # (N, 4, 3)
            wp_pos = torch.einsum('ni,nij->nj', wp_bary[name], tet_v_sel)  # (N, 3)
            wp_out[name] = wp_pos
        # 4) Single bulk CPU sync (concat all)
        flat = torch.cat([wp_out[n] for n in muscle_vertex_counts], dim=0)
        return flat.cpu().numpy()

    # Warmup
    print(f"Warmup: {args.warmup} iters")
    for _ in range(args.warmup):
        dofs = torch.randn(2, input_dim, device=device)
        one_step(dofs)
    torch.cuda.synchronize() if device.type == "cuda" else None

    # Measure
    print(f"Measuring: {args.iters} iters")
    # Phase A: end-to-end
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.iters):
        dofs = torch.randn(2, input_dim, device=device)
        one_step(dofs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total = time.time() - t0
    per_frame_ms = (total / args.iters) * 1000

    # Phase B: NN forward only (no waypoint)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.iters):
        dofs = torch.randn(2, input_dim, device=device)
        with torch.no_grad():
            _ = model(dofs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    nn_only = time.time() - t0
    nn_only_ms = (nn_only / args.iters) * 1000

    print(f"\n=== Results (per frame, batched L+R) ===")
    print(f"  NN forward only:           {nn_only_ms:.2f} ms")
    print(f"  Full GPU pipeline (A):     {per_frame_ms:.2f} ms")
    print(f"  Waypoint+transform+sync:   {(per_frame_ms - nn_only_ms):.2f} ms")
    print(f"\n  Real-time budget:")
    print(f"    60 Hz physics  (16.7 ms): {'OK ✓' if per_frame_ms < 16.7 else 'TIGHT' if per_frame_ms < 20 else 'TOO SLOW'}")
    print(f"   120 Hz physics  ( 8.3 ms): {'OK ✓' if per_frame_ms < 8.3 else 'TIGHT' if per_frame_ms < 10 else 'TOO SLOW'}")
    print(f"   240 Hz physics  ( 4.2 ms): {'OK ✓' if per_frame_ms < 4.2 else 'TIGHT' if per_frame_ms < 5 else 'TOO SLOW'}")


if __name__ == "__main__":
    main()
