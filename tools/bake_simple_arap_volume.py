#!/usr/bin/env python3
"""Simple LBS-guided ARAP with positive-volume recovery and preservation.

This is the matched volume-constrained variant of bake_simple_arap.py.
Twenty-four projection sweeps were selected because stronger projection
reduced more inversions but caused extreme local expansion artifacts.
"""
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    sys.argv = [
        sys.argv[0],
        "--region-tag", "L_UpLeg_simple_arap_volume",
        "--volume-projection-sweeps", "24",
        "--volume-projection-max-correction", "0.002",
        "--volume-arap-alternations", "6",
        *sys.argv[1:],
    ]
    from tools import bake_simple_arap
    bake_simple_arap.main()


if __name__ == "__main__":
    main()
