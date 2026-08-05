#!/usr/bin/env python3
"""Headless bonded-muscle bake with bone-muscle collision enabled.

This is intentionally a thin wrapper around ``bake_headless.py`` so it keeps
the same attachment setup, inter-muscle bonds, ARAP settings, plateau exit,
waypoint patching, and viewer cache format.  The only policy change is that
both existing bone-contact paths are forced on:

* unified-volume one-sided bone contact, and
* the per-muscle bone-contact push.

Example:
  python tools/bake_headless_bone_contact.py --bvh data/motion/run.bvh \
      --muscles .muscles_L_UpLeg.json --start-frame 0 --end-frame 4 \
      --region-tag L_UpLeg_bone_contact
"""
from __future__ import annotations

import sys

from tools import bake_headless


def main() -> None:
    # Append so these are the final values if a caller accidentally supplies
    # the disabling switches earlier on the command line.
    sys.argv.extend([
        "--unified-bone-contact",
        "--self-collision",
        # A fixed iteration budget avoids visible frame-to-frame changes in
        # convergence depth when the plateau detector trips at different
        # points during a motion.
        "--no-plateau-exit",
    ])
    bake_headless.main()


if __name__ == "__main__":
    main()
