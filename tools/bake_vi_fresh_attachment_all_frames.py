#!/usr/bin/env python3
"""Bake all VI poses independently from the rest tet mesh."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    def bake(frame):
        output = args.work_dir / f"frame_{frame:04d}"
        command = [
            "pyMAC/bin/python", "tools/test_vi_attachment_ramp.py",
            "--soft-insertion", "--preconditioned",
            "--project-jacobian-tangent",
            "--frame", str(frame),
            "--ramp-steps", "30", "--iterations", "180",
            "--attachment-start", "0.01", "--attachment-end", "1000",
            "--minimum-jacobian", "0.01",
            "--low-jacobian-weight", "1000",
            "--contact-weight", "500",
            "--output-dir", str(output), "--device", "cuda",
        ]
        completed = subprocess.run(
            command, check=True, text=True, capture_output=True)
        result = output / (
            f"L_Vastus_Intermedius_frame_{frame:04d}.npz")
        data = np.load(result)
        metrics = (
            float(data["minimum_jacobian_ratio"]),
            float(np.max(data["insertion_error"])),
            int(data["remaining_contact_samples"]),
        )
        print(
            f"completed frame={frame:02d} minJ={metrics[0]:.6g} "
            f"attachment={metrics[1]:.6g} contact={metrics[2]}",
            flush=True)
        return frame, result, metrics

    results = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers) as pool:
        futures = [pool.submit(bake, frame) for frame in range(76)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort()

    frames, positions = [], []
    minimum_jacobian, attachment_error, contact_samples = [], [], []
    for frame, path, metrics in results:
        data = np.load(path)
        frames.append(frame)
        positions.append(np.asarray(data["positions"], dtype=np.float32))
        minimum_jacobian.append(metrics[0])
        attachment_error.append(metrics[1])
        contact_samples.append(metrics[2])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "L_Vastus_Intermedius_chunk_0000.npz",
        frames=np.asarray(frames, dtype=np.int32),
        positions=np.asarray(positions, dtype=np.float32),
        minimum_jacobian_ratio=np.asarray(minimum_jacobian),
        maximum_attachment_error=np.asarray(attachment_error),
        remaining_contact_samples=np.asarray(contact_samples))
    (args.output_dir / ".done").touch()
    print(
        f"RESULT frames={len(frames)} "
        f"minJ={min(minimum_jacobian):.6g} "
        f"max_attachment={max(attachment_error):.6g} "
        f"max_contact={max(contact_samples)}",
        flush=True)


if __name__ == "__main__":
    main()
