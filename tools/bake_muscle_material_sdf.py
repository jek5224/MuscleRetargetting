#!/usr/bin/env python3
"""Muscle-level tetrahedral bake with fibers, matrix, volume and contact.

This is a cache-free experimental entry point.  It reuses the mature BVH,
attachment, SDF, checkpoint and positive-J line-search infrastructure from
``bake_muscle_arap_sdf`` while selecting the muscle material objective.  Any
normal baker option can still be supplied on the command line; explicit user
arguments override these defaults.
"""
from __future__ import annotations

import sys

from tools import bake_muscle_arap_sdf


MUSCLE_DEFAULTS = {
    "--muscle-material": None,
    "--arap-weight": "0.0",
    "--matrix-weight": "0.20",
    "--fiber-weight": "0.20",
    "--fiber-target-from-attachments": None,
    "--fiber-bending-weight": "0.05",
    "--volume-weight": "1.0",
    "--inversion-barrier-weight": "0.25",
    "--inversion-barrier-start": "0.30",
    "--quality-weight": "30.0",
    "--minimum-jacobian": "0.01",
    "--localized-quality-barrier": None,
    "--local-jacobian-line-search": None,
    "--contact-weight": "300.0",
    "--origin-weight": "10000.0",
    "--insertion-weight": "15000.0",
    "--rotation-update-interval": "12",
}


def _has_option(arguments: list[str], option: str) -> bool:
    return option in arguments or any(
        argument.startswith(option + "=") for argument in arguments)


def main() -> None:
    arguments = sys.argv[1:]
    defaults: list[str] = []
    for option, value in MUSCLE_DEFAULTS.items():
        if _has_option(arguments, option):
            continue
        defaults.append(option)
        if value is not None:
            defaults.append(value)
    sys.argv = [sys.argv[0], *defaults, *arguments]
    bake_muscle_arap_sdf.main()


if __name__ == "__main__":
    main()
