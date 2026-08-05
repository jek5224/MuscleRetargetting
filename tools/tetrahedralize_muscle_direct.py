#!/usr/bin/env python3
"""Default direct-remesh tetrahedralizer for open two-contour muscles.

This entry point intentionally uses the accepted TetWild boundary remesh:
the remeshed boundary is rendered and simulated directly, artificial closure
faces are omitted from collision, and no cage/skin embedding is created.
Pass --input and --output for subsequent muscles.
"""
from tools.tetrahedralize_subdivided_vi_remesh import main


if __name__ == "__main__":
    main()
