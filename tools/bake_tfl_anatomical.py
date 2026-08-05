#!/usr/bin/env python3
"""Bake, validate, and atomically publish TFL anatomical frame 0."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

from tools.validate_tfl_frame import NAME, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bvh', default='data/motion/run.bvh')
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument(
        '--neighbor-cache',
        default='.bake_outputs/motion_cache/run/L_UpLeg_anatomical_contact_test')
    parser.add_argument(
        '--publish-dir',
        default='.bake_outputs/motion_cache/run/L_UpLeg_anatomical_contact')
    args = parser.parse_args()

    staging_root = tempfile.mkdtemp(
        prefix='tfl_anatomical_', dir='.bake_outputs/staging')
    cache_tag = 'tfl_validation_candidate'
    try:
        command = [
            sys.executable, 'tools/bake_fascia.py',
            '--bvh', args.bvh, '--region', 'L_UpLeg',
            '--muscles-json', 'tools/muscles_tfl.json',
            '--start-frame', str(args.frame), '--end-frame', str(args.frame),
            '--cache-tag', cache_tag, '--output-root', staging_root,
            '--tet-dir', 'tet', '--backend', 'taichi', '--fem', '--pn',
            '--settle-iters', '300', '--fem-volume-penalty', '1000000',
            '--inter-k', '0', '--inter-muscle-weight', '0',
            '--no-fascia-constraints', '--unified-bone-contact',
            '--unified-bone-contact-margin', '0.001', '--no-plateau-exit']
        environment = os.environ.copy()
        environment.setdefault('MPLCONFIGDIR', '/tmp')
        environment.setdefault('MUSCLE_TAICHI_ARCH', 'cpu')
        environment.setdefault('TI_OFFLINE_CACHE', '0')
        subprocess.run(command, check=True, env=environment)

        stem = os.path.splitext(os.path.basename(args.bvh))[0]
        candidate_dir = os.path.join(
            staging_root, stem, f'L_UpLeg_{cache_tag}')
        report = validate(candidate_dir, args.neighbor_cache,
                          args.frame, args.bvh)

        os.makedirs(args.publish_dir, exist_ok=True)
        source = os.path.join(candidate_dir, f'{NAME}_chunk_0000.npz')
        destination = os.path.join(
            args.publish_dir, f'{NAME}_chunk_0000.npz')
        temporary = destination + '.validated.tmp'
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
        print(f'Published validated TFL frame {args.frame}: {destination}')
        print(report)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


if __name__ == '__main__':
    main()
