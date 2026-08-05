#!/usr/bin/env python3
"""Minimal cache-free muscle bake: volume, attachments, and bone contact.

This experiment intentionally contains no ARAP, elastic matrix, fiber, or
bending energy.  Per-tet volume preservation is the only deformation energy;
the positive-J barrier is retained because volume squared alone cannot stop a
tetrahedron from crossing through zero volume and inverting.
"""
from __future__ import annotations

import sys

from tools import bake_muscle_arap_sdf


VOLUME_DEFAULTS = {
    "--muscle-material": None,
    "--arap-weight": "0.0",
    "--matrix-weight": "0.0",
    "--director-weight": "0.0",
    "--fiber-weight": "0.0",
    "--fiber-bending-weight": "0.0",
    "--smooth-arap-weight": "0.0",
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
}


def _has_option(arguments: list[str], option: str) -> bool:
    return option in arguments or any(
        argument.startswith(option + "=") for argument in arguments)


def main() -> None:
    arguments = sys.argv[1:]
    defaults: list[str] = []
    for option, value in VOLUME_DEFAULTS.items():
        if _has_option(arguments, option):
            continue
        defaults.append(option)
        if value is not None:
            defaults.append(value)
    sys.argv = [sys.argv[0], *defaults, *arguments]
    bake_muscle_arap_sdf.main()


if __name__ == "__main__":
    main()
