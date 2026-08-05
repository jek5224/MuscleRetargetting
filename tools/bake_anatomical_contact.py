#!/usr/bin/env python3
"""Conservative offline bake for the left upper leg acceptance sequence.

This entry point deliberately chooses quality/stability settings instead of
exposing the older experimental collision modes:

* hard cap/anchor attachment constraints from the saved tet metadata;
* unified muscle solve (all muscles influence one equilibrium problem);
* one-sided muscle/bone and muscle/muscle separation;
* automatically discovered, reciprocal rest-pose anatomical interfaces;
* normal-gap interface cohesion, which permits tangential sliding;
* weak bone-relative material guidance to prevent whole-muscle migration;
* fixed iteration count for deterministic temporal behaviour.

The underlying project runtime (DART plus an ARAP backend) is required.
Default acceptance case:

    python tools/bake_anatomical_contact.py

This bakes frames 0..4 of run.bvh for .muscles_L_UpLeg.json.
"""
from __future__ import annotations

import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    # This policy is runnable on machines without a visible CUDA device while
    # retaining Taichi's collision-capable backend.
    os.environ.setdefault('MUSCLE_TAICHI_ARCH', 'cpu')
    os.environ.setdefault('TI_OFFLINE_CACHE', '0')
    # bake_fascia owns the complete loading/baking/cache pipeline.  Supplying
    # explicit final arguments here makes this script a reproducible policy,
    # rather than another collection of subtly incompatible presets.
    defaults = [
        '--bvh', 'data/motion/run.bvh',
        '--region', 'L_UpLeg',
        '--start-frame', '0',
        '--end-frame', '4',
        '--cache-tag', 'anatomical_contact',
        '--output-root', '.bake_outputs/motion_cache',
        '--tet-dir', 'tet',
        '--backend', 'taichi',
        '--fem',
        '--pn',
        '--settle-iters', '300',
        '--constraint-threshold', '0.006',
        '--inter-k', '0',
        '--inter-muscle-weight', '0.0',
        '--anisotropic-contact',
        '--inter-muscle-contact-margin', '0.001',
        '--fascia-constraints',
        '--fascia-constraint-threshold', '0.006',
        '--fascia-constraint-weight', '2.0',
        '--fascia-min-patch-vertices', '12',
        '--unified-bone-contact',
        '--unified-bone-contact-margin', '0.001',
        '--unified-bone-contact-weight', '8.0',
        '--skin-prior',
        '--skin-prior-sigma', '0.020',
        '--skin-prior-max-dist', '0.045',
        '--skin-prior-strength', '0.12',
        '--no-plateau-exit',
    ]

    # Advanced callers may append overrides. argparse uses the last value for
    # scalar options, so a command-line override wins without editing this file.
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    from tools import bake_fascia
    bake_fascia.main()

    # The medial tibial insertions need different boundary semantics from a
    # free collision surface. Run the shape-preserving untwist/embedded-cap
    # repair as a standard finalization stage on the cache just produced.
    # This replaces the former manually applied viewer overlay.
    def last_value(flag, fallback):
        value = fallback
        for i, arg in enumerate(sys.argv[:-1]):
            if arg == flag:
                value = sys.argv[i + 1]
        return value

    region = last_value('--region', 'L_UpLeg')
    # The direct LBS + signed-volume path already solves exact attachments and
    # joint contact. The legacy medial repair is a vertex post-process; applying
    # it afterward can reintroduce inversions and invalidate the coherent bake.
    if (region == 'L_UpLeg'
            and os.environ.get('MUSCLE_PN_DIRECT_INIT', '0') != '1'):
        bvh = last_value('--bvh', 'data/motion/run.bvh')
        output_root = last_value('--output-root', '.bake_outputs/motion_cache')
        cache_tag = last_value('--cache-tag', 'anatomical_contact')
        start = int(last_value('--start-frame', '0'))
        end = int(last_value('--end-frame', '4'))
        stem = os.path.splitext(os.path.basename(bvh))[0]
        cache_dir = os.path.join(output_root, stem,
                                 f'{region}_{cache_tag}')
        from tools.fix_medial_tibia_frame0 import repair_frames
        repair_frames(source=cache_dir, output=cache_dir,
                      motion_path=bvh, frames=range(start, end + 1))


if __name__ == '__main__':
    main()
