# Isolated fiber-informed muscle solver

## Repository mapping

- `tet/<name>_tet.npz` is a Python-pickled dictionary despite its extension.
  Positions are metres. `vertices` and `tetrahedra` define the simulation
  volume; `render_faces`/`sim_faces` define boundaries.
- `cap_face_indices`, `cap_attachments`, and `attach_skeleton_names` define
  multi-vertex origin/insertion regions. Cap indices refer to render faces.
- Explicit fibers are level-major:
  `waypoints[stream][level][fiber_index] -> xyz`. They are transposed into
  ordered curved polylines; no origin-insertion centerline is synthesized.
- `core/bvhparser.py::MyBVH` converts BVH channels to DART generalized
  coordinates and converts centimetre BVHs to metres.
- DART bodies provide rest and animated world transforms. Attachment vertices
  are stored in body-local coordinates and transformed exactly at every
  continuation substep.
- Existing `tools/bake_emu.py` supplies the nearest existing FEM architecture,
  but its contour-gradient/global-axis fiber field does not meet the explicit
  fiber requirement. The isolated implementation is therefore separate.
- Viewer motion caches contain `frames` and `(frame, vertex, xyz)` `positions`.

## Implemented

- Reported sliver-tet quality filtering using the existing EMU threshold.
- Tet-boundary extraction and connected cap-patch expansion.
- Exact bone-local moving attachment patches and free-DOF elimination.
- Explicit fiber orientation, containment, barycentric embedding, and sparse
  reconstruction matrices.
- Per-tet directions derived only from embedded fiber segments. Tets beyond
  the configured search radius remain explicitly isotropic.
- Compressible Neo-Hookean matrix and logarithmic volume terms.
- Smooth tension-dominant per-tet fiber response.
- Rotation-invariant weak turning-angle preservation for embedded curves.
- Analytic gradients, determinant rejection, L-BFGS plus an
  inversion-aware Newton-Krylov trust-region fallback, Armijo pre-relaxation,
  transform SLERP, adaptive pose subdivision, and previous-substep
  continuation.
- OBJ/NPZ debug geometry, viewer cache, embedding JSON, summary CSV, and
  iteration CSV.

## Commands

```bash
pyMAC/bin/python -m unittest tests.test_isolated_muscle -v

pyMAC/bin/python tools/bake_isolated_muscle.py \
  --muscle L_Gracilis \
  --bvh data/motion/left_thigh_quasistatic_5pose.bvh \
  --start-frame 0 --end-frame 0 \
  --configuration D \
  --output .bake_outputs/isolated_muscle_gracilis_rest
```

Configurations `A`, `B`, `C`, and `D` select the requested ablation terms.

## Validated results

- Seven unit/derivative tests pass.
- Combined and separated energy gradients pass centered finite differences.
- A real Rectus Femoris directional derivative has relative error
  `3.3e-10`.
- Rectus Femoris and Gracilis rigid/rest poses have exact attachments, no
  inversions, `J=1`, volume ratio `1`, and fiber length ratio `1`.
- Gracilis retains 1919/2010 tets after reported sliver filtering. Its 25
  fibers contribute 414 genuinely embedded internal samples; 20 supplied
  samples are reported as outside rather than assigned arbitrarily.

## Gracilis four-pose ablation

The real five-pose BVH was baked through frame 3 (deep flexion). Deep-pose
measurements are:

| Configuration | Result | min J | max J | volume ratio | max fiber turn |
|---|---:|---:|---:|---:|---:|
| A matrix only | failed at frame 1, pose fraction 0.890625 | -- | -- | -- | -- |
| B + volume | converged | 0.25253 | 8.40971 | 0.98497 | 0.96085 rad |
| C + fiber anisotropy | converged | 0.24954 | 4.01096 | 0.98623 | 0.94405 rad |
| D + weak fiber bending | converged | 0.27235 | 4.76037 | 0.98577 | 0.56063 rad |

All accepted B/C/D poses have zero reported attachment error and no inverted
tets. This does **not** establish anatomical correctness. Volume control is
necessary for this test. Fiber anisotropy cuts the extreme local expansion
seen in B; weak bending cuts the maximum reconstructed fiber turn by about
41% relative to C. Deep flexion still reaches low `J` and large local
expansion, so the distal shape must be inspected in the viewer and may require
bone contact or a tendon-wrapping model.

The frame debug NPZ files contain per-tet `J`, per-tet fiber stretch (NaN for
explicitly isotropic tets), reconstructed fiber points and turning angles, and
the relevant animated bone transforms. Per-fiber CSV files identify the
sample with the sharpest turn.

## Remaining limitations

- Bone meshes are not embedded in the OBJ export; animated attachment-bone
  transforms are exported in the frame NPZ.
- There is no bone contact, tendon wrapping, fascia, or multi-muscle contact.
- Active target scale is implemented behind
  `active_shortening_enabled`, but it has not been validated and remains off.
- The path term preserves turning angles rather than using polar-rotated rest
  curvature.
- SciPy's internal line-search alpha is unavailable in the iteration CSV.
- Rendering-mesh transfer is not implemented.
