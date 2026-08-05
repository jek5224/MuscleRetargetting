---
name: Cut contour gap bug (active)
description: After BP Transform cut, contour pieces are visually separate — no shared boundary vertices between streams
type: project
---

**Problem**: After cutting a single contour into 2 streams (e.g., gastrocnemius level 46+), the two pieces are visually separate with a gap between them. The contour mesh has no faces connecting the pieces along the cut boundary.

**Root cause (not yet confirmed)**: The BP Transform cut method (`_cut_contour_bp_transform` at line 13427 in contour_mesh.py) uses vertex assignment — each vertex goes to one piece. At assignment boundaries, midpoint vertices are added to both pieces (lines 15930-15947). However, these shared boundary midpoints may not be enough, or the assignment path may not reach the voronoi cut code that adds them.

**What's been tried**:
- Increased merge epsilon in build_contour_mesh (broke non-boundary vertices)
- Added stitching faces (didn't work — shared vertex indices weren't found)
- Added intermediate vertices in Voronoi cut path (didn't fix BP Transform path)
- Added snap-after-resampling (wrong — gap exists before resampling)
- Many debug prints (user frustrated by token waste)

**Key finding**: The cut uses `[BP Transform] Mode: SEPARATE` for first diverging level, then `Mode: COMMON` for subsequent levels. The actual piece creation happens through vertex assignment, which assigns each vertex to nearest source. The boundary handling code at lines 15921-15947 adds midpoints. But this code may not be reached from the BP Transform path.

**What to check next**: Trace the EXACT code path from `_cut_contour_bp_transform` to piece creation. The BP Transform may use a different piece-building method that doesn't add shared boundary vertices at all. Search for where `_cut_contour_bp_transform` returns pieces and whether they share ANY vertices.

**Files**: viewer/contour_mesh.py (main), viewer/tetrahedron_mesh.py (tet gen), viewer/zygote_mesh_ui.py (UI)
