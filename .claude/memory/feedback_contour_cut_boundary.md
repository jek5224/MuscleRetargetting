---
name: Contour cut boundary vertex mismatch
description: Cut contour resampling produces different vertex counts on shared boundary between pieces, preventing mesh stitching
type: feedback
---

Cut contours have a shared boundary (straight line between cut points). After resampling, each piece independently calculates how many vertices go on the boundary vs surface based on its own surface/boundary length ratio. This produces DIFFERENT vertex counts on the shared edge between the two pieces, so they can't merge in build_contour_mesh.

**Why:** `_resample_cut_contour` computes `boundary_verts = distributable - surface_verts` independently per piece. Different surface lengths → different boundary_verts.

**How to apply:** When fixing cut boundary stitching, ensure both pieces of a cut use the SAME number of boundary vertices. This should be coordinated during the cut operation, not left to independent resampling.
