#!/usr/bin/env python3
"""Bake all saved tet muscles as one coupled volumetric group.

The unified ARAP system contains every tet body plus broad rest-pose
cross-muscle distance edges. Bone and muscle collision handling are disabled;
compactness comes from the group connectivity itself.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    defaults = [
        '--bvh', 'data/motion/run.bvh',
        '--muscles', '.muscles_L_UpLeg.json',
        '--region-tag', 'L_UpLeg_whole_group',
        '--output-root', '.bake_outputs/motion_cache',
        '--tet-dir', 'tet',
        '--backend', 'taichi',
        '--start-frame', '0',
        '--end-frame', '4',
        '--constraint-threshold', '0.006',
        '--inter-k', '3',
        '--inter-muscle-weight', '5.0',
        '--reciprocal-inter-muscle',
        '--anisotropic-contact',
        '--inter-muscle-contact-margin', '0.0005',
        '--fascia-constraints',
        '--fascia-constraint-threshold', '0.006',
        '--fascia-constraint-weight', '1.0',
        '--unified-bone-contact',
        '--unified-bone-contact-margin', '0.0005',
        '--unified-bone-contact-weight', '2.0',
        '--contact-recompute-every', '5',
        '--settle-iters', '150',
        '--volume-projection-sweeps', '600',
        '--volume-contact-passes', '1',
        '--no-muscle-aware',
        '--no-plateau-exit',
    ]
    # Scalar argparse options use the last occurrence, so callers may append
    # frame ranges or weights without editing this policy script.
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    from tools import bake_headless
    bake_headless.main()


if __name__ == '__main__':
    main()
