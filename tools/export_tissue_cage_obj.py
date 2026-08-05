#!/usr/bin/env python3
"""Export rest and posed tissue-cage surfaces from a cage debug cache."""
import argparse
import os

import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", help="__tissue_cage_chunk_0000.npz")
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--output-dir", default="cage/debug")
    args = ap.parse_args()
    data = np.load(args.cache)
    frames = np.asarray(data["frames"], dtype=np.int32)
    if args.frame is None:
        local = len(frames) - 1
    else:
        match = np.where(frames == args.frame)[0]
        if not len(match):
            raise ValueError(
                f"frame {args.frame} not present; available={frames.tolist()}")
        local = int(match[0])
    faces = np.asarray(data["surface_faces"], dtype=np.int32)
    os.makedirs(args.output_dir, exist_ok=True)
    rest_path = os.path.join(args.output_dir, "L_UpLeg_cage_rest.obj")
    pose_path = os.path.join(
        args.output_dir, f"L_UpLeg_cage_frame_{int(frames[local]):04d}.obj")
    trimesh.Trimesh(
        data["rest_positions"], faces, process=False).export(rest_path)
    trimesh.Trimesh(
        data["positions"][local], faces, process=False).export(pose_path)
    print(rest_path)
    print(pose_path)


if __name__ == "__main__":
    main()
