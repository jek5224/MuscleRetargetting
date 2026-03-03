#!/usr/bin/env python3
"""Batch baking — spawns bake_headless.py subprocesses for multiple BVH files.

Usage:
    # Sequential (GPU/Taichi — default)
    python tools/bake_batch.py data/motion/*.bvh

    # Parallel with CPU backend (4 workers)
    python tools/bake_batch.py data/motion/*.bvh --workers 4 --backend cpu

    # Multi-GPU server: 4 workers across 4 GPUs
    python tools/bake_batch.py data/motion/*.bvh --workers 4 --backend taichi

    # Skip already-baked files
    python tools/bake_batch.py data/motion/*.bvh --skip-existing
"""
import argparse
import glob
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BAKE_SCRIPT = os.path.join(SCRIPT_DIR, "bake_headless.py")
CACHE_DIR = os.path.join("data", "motion_cache")


def has_cache(bvh_path):
    """Check if a BVH file already has cached chunk files."""
    stem = os.path.splitext(os.path.basename(bvh_path))[0]
    cache_path = os.path.join(CACHE_DIR, stem)
    if not os.path.isdir(cache_path):
        return False
    return len(glob.glob(os.path.join(cache_path, "*_chunk_*.npz"))) > 0


def main():
    parser = argparse.ArgumentParser(description="Batch bake BVH files in parallel")
    parser.add_argument("bvh_files", nargs="+", help="BVH file paths")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "taichi", "gpu", "cpu"],
        default="auto",
        help="ARAP backend passed to bake_headless.py (default: auto)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip BVH files that already have cached chunks",
    )
    # Pass-through args for bake_headless.py
    parser.add_argument("--muscles", default=None)
    parser.add_argument("--settle-iters", type=int, default=None)
    parser.add_argument("--constraint-threshold", type=float, default=None)
    args = parser.parse_args()

    # Filter files
    bvh_files = args.bvh_files
    if args.skip_existing:
        before = len(bvh_files)
        bvh_files = [f for f in bvh_files if not has_cache(f)]
        skipped = before - len(bvh_files)
        if skipped:
            print(f"Skipped {skipped} already-baked file(s)")

    if not bvh_files:
        print("Nothing to bake.")
        return

    num_workers = max(1, args.workers)
    print(f"Baking {len(bvh_files)} file(s) with {num_workers} worker(s), backend={args.backend}")
    print()

    # Detect available GPUs for CUDA_VISIBLE_DEVICES round-robin
    num_gpus = 0
    if args.backend in ("taichi", "auto") and num_workers > 1:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cvd:
            num_gpus = len(cvd.split(","))
        else:
            # Try nvidia-smi
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "-L"], stderr=subprocess.DEVNULL, text=True
                )
                num_gpus = sum(1 for line in out.strip().split("\n") if line.startswith("GPU"))
            except (FileNotFoundError, subprocess.CalledProcessError):
                num_gpus = 0

    # Build base command args
    base_cmd = [sys.executable, BAKE_SCRIPT, "--backend", args.backend]
    if args.muscles is not None:
        base_cmd += ["--muscles", args.muscles]
    if args.settle_iters is not None:
        base_cmd += ["--settle-iters", str(args.settle_iters)]
    if args.constraint_threshold is not None:
        base_cmd += ["--constraint-threshold", str(args.constraint_threshold)]

    # Track results
    results = {}  # bvh_path -> ("done"|"FAILED", elapsed)
    active = {}   # bvh_path -> (Popen, start_time, worker_idx)
    queue = list(bvh_files)
    batch_start = time.time()
    worker_idx_counter = 0

    def launch(bvh_path):
        nonlocal worker_idx_counter
        widx = worker_idx_counter
        worker_idx_counter += 1
        env = os.environ.copy()
        # Round-robin GPU assignment
        if num_gpus > 0 and args.backend in ("taichi", "auto"):
            env["CUDA_VISIBLE_DEVICES"] = str(widx % num_gpus)
        stem = os.path.splitext(os.path.basename(bvh_path))[0]
        log_path = os.path.join(CACHE_DIR, f"{stem}_bake.log")
        os.makedirs(CACHE_DIR, exist_ok=True)
        log_file = open(log_path, "w")
        cmd = base_cmd + ["--bvh", bvh_path]
        proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env
        )
        active[bvh_path] = (proc, time.time(), widx, log_file)
        print(f"  START  [{widx}] {bvh_path}  (log: {log_path})")

    def poll_active():
        done = []
        for bvh_path, (proc, t0, widx, log_file) in active.items():
            ret = proc.poll()
            if ret is not None:
                elapsed = time.time() - t0
                log_file.close()
                status = "done" if ret == 0 else "FAILED"
                results[bvh_path] = (status, elapsed, ret)
                stem = os.path.splitext(os.path.basename(bvh_path))[0]
                print(f"  {status:>6}  [{widx}] {stem}  ({elapsed:.1f}s, exit={ret})")
                done.append(bvh_path)
        for bvh_path in done:
            del active[bvh_path]

    # Main loop
    while queue or active:
        # Fill worker slots
        while queue and len(active) < num_workers:
            launch(queue.pop(0))

        # Wait a bit then check
        if active:
            time.sleep(1.0)
            poll_active()

    # Summary
    total_time = time.time() - batch_start
    n_ok = sum(1 for s, _, _ in results.values() if s == "done")
    n_fail = sum(1 for s, _, _ in results.values() if s == "FAILED")
    print(f"\n{'='*60}")
    print(f"Batch complete: {n_ok} succeeded, {n_fail} failed, {total_time:.1f}s total")
    if n_fail:
        print("\nFailed files:")
        for bvh_path, (status, elapsed, ret) in results.items():
            if status == "FAILED":
                stem = os.path.splitext(os.path.basename(bvh_path))[0]
                log_path = os.path.join(CACHE_DIR, f"{stem}_bake.log")
                print(f"  {bvh_path}  (exit={ret}, log: {log_path})")
        sys.exit(1)


if __name__ == "__main__":
    main()
