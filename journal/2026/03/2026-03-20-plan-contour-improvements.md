# Plan — Contour Selection & Refinement Improvements

## Overview

Two independent improvements to the contour pipeline:
1. **`select_levels`**: Replace centroid-only RDP with 3D inertia tensor RDP + absolute threshold
2. **`refine_contours`**: Fix divergence over-refinement by adding per-stream spacing tracking

---

## 1. `select_levels` — 3D Inertia Tensor RDP

### Problem
Current `select_levels` (contour_mesh.py:12449) uses centroid interpolation error only. This misses:
- Cross-section shape changes (round → flat)
- Cross-section size changes (belly → tendon taper)
- Plane orientation changes (muscle curving in 3D)

Threshold is relative (0.5% of muscle length), which over-selects contours for small muscles that don't need many.

### Changes

**A. Replace error metric**

Current (`compute_stream_error`, line 12523):
```python
error = ||actual_mean - interpolated_mean||
```

New:
```python
I_actual = inertia_tensor_3D(contour_points)   # 3x3 from raw 3D coords
I_prev = inertia_tensor_3D(prev_contour_points)
I_next = inertia_tensor_3D(next_contour_points)
I_interp = (1 - t) * I_prev + t * I_next

error = ||I_actual - I_interp||_F / ||I_actual||_F   # relative Frobenius
```

The 3D inertia tensor captures centroid position, area, aspect ratio, orientation, and plane tilt in one 3x3 symmetric matrix. Interpolation error detects all failure modes.

**B. Switch to absolute threshold**

Current: `error_threshold = 0.005 * muscle_length` (relative, per-muscle)
New: single absolute threshold, universal for all muscles (~5% relative Frobenius, tuned visually)

Small muscles naturally get fewer contours. Large complex muscles get more. No per-muscle scaling.

**C. Add helper function**

```python
def inertia_tensor_3D(points):
    """Compute 3x3 inertia tensor from 3D point cloud."""
    centered = points - points.mean(axis=0)
    return (centered.T @ centered) / len(points)
```

### Files to modify
- `viewer/contour_mesh.py`: `select_levels()` method (~line 12449)
  - Replace `compute_stream_error` helper
  - Change threshold logic
  - Add `inertia_tensor_3D` helper

### Validation
- Run on a few representative muscles:
  - Curved muscle (check curvature detection)
  - Tapering muscle (check area change detection)
  - Simple cylinder muscle (check it gets fewer contours than before)
- Compare selected contour counts: old vs new
- Visual check: do selected contours capture shape changes that centroid-only missed?

---

## 2. `refine_contours` — Per-Stream Spacing at Divergence

### Problem
`refine_contours` (contour_mesh.py:2888) fills gaps where contour spacing exceeds `max_spacing_threshold`. At divergence/merge points (different contour counts between levels), it uses centroid-to-centroid distance as a "more lenient" metric (line 2974-2980). For adductor magnus (fan-shaped, branches stay close for a long scalar range), this leniency isn't enough — it still inserts too many closely-spaced contours near the divergence.

Works fine for biceps/gastrocnemius (branches separate quickly).

### Current divergence handling (lines 2962-2980)
```python
if len(current_planes) == len(next_planes):
    # Same count: per-contour min distance to any in other level
    ...
else:
    # Diverging/merging: centroid-to-centroid (lenient)
    centroid_curr = np.mean(curr_means, axis=0)
    centroid_next = np.mean(next_means, axis=0)
    spacing = np.linalg.norm(centroid_next - centroid_curr)
```

### Changes

**A. Per-stream matching at diverging levels**

When contour counts differ, instead of averaging all centroids, match each contour to its closest counterpart in the adjacent level (same `find_closest_contour` pattern used in `smoothen_contours_z` and `smoothen_contours_x`):

```python
else:
    # Diverging/merging: per-stream matching by closest centroid
    # From the side with MORE contours, each finds its closest in the other level
    if len(current_planes) > len(next_planes):
        more_means, fewer_means = curr_means, next_means
    else:
        more_means, fewer_means = next_means, curr_means

    stream_distances = []
    for mean in more_means:
        dists = np.linalg.norm(fewer_means - mean, axis=1)
        stream_distances.append(np.min(dists))

    # Use min stream distance — tightest stream determines if gap exists
    spacing = min(stream_distances)
    has_gap = spacing > max_spacing_threshold
```

**B. Skip refinement when contour count changes within gap**

When `_find_contour_between` finds a contour with a different count than expected (the divergence point itself), accept it as a topological event but don't treat it as a gap to keep filling. Currently `expected_count` filtering partially handles this (line 3030-3032), but only when both adjacent levels have the same count.

### Files to modify
- `viewer/contour_mesh.py`: `refine_contours()` method (~line 2888)
  - Modify divergence spacing calculation (~lines 2974-2980)
  - Add per-stream matching for different-count levels

### Validation
- Test on adductor magnus: should produce fewer, better-spaced contours at divergence
- Test on biceps brachii: should behave same as before (already works)
- Test on gastrocnemius: should behave same as before
- Test on simple single-stream muscle: no change expected
- Compare total contour counts before/after

---

## Implementation Order

1. **`refine_contours` fix first** — it runs earlier in the pipeline (gap-filling after `find_contours`, before `select_levels`). If `refine_contours` produces cleaner input, `select_levels` improvement has better starting data.

2. **`select_levels` upgrade second** — builds on cleaner `refine_contours` output. Can be tested independently since the error metric change is self-contained.

## Dependencies
- Neither change affects downstream code (cut, stream smooth, fiber build). They only change WHICH contours are kept, not the data format.
- Both changes are backward-compatible: old muscles can be re-processed with new code.
