#!/usr/bin/env python3
"""Minimal LBS-initialized ARAP muscle baker.

Energy terms:
  1. ordinary per-muscle ARAP edges,
  2. hard skeleton attachment vertices,
  3. soft rest-length edges between nearby vertices of different muscles.

There is no collision, volume projection, axial activation, tendon
modification, skin prior, fascia model, or temporal warm start.
"""
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    defaults = [
        "--bvh", "data/motion/run.bvh",
        "--muscles", "tools/muscles_L_UpLeg.json",
        "--region-tag", "L_UpLeg_simple_arap",
        "--output-root", ".bake_outputs/motion_cache",
        "--tet-dir", "tet",
        "--backend", "taichi",
        "--start-frame", "0",
        "--end-frame", "4",
        "--settle-iters", "150",
        # Quasistatic solve: every frame starts from the pose-determined LBS
        # state. A previous-frame warm start introduces path dependence, so
        # equal poses can settle into visibly different ARAP minima.
        "--lbs-init-weight", "1.0",
        "--constraint-threshold", "0.01",
        "--inter-k", "3",
        "--inter-muscle-weight", "1.5",
        "--volume-projection-sweeps", "0",
        "--no-muscle-aware",
        "--no-tendon-elastic",
        "--skip-waypoints",
    ]
    # Appended command-line values override scalar defaults.
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    from tools import bake_headless
    bake_headless.main()


if __name__ == "__main__":
    main()
