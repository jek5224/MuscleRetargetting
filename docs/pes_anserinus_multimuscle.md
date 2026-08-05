# Pes-anserinus multi-muscle extension

## Architecture

The isolated constitutive solver is unchanged. `viewer/multimuscle.py` wraps
independent `MuscleData` bodies in a global vertex/DOF layout and assembles
their matrix, volume, explicit-fiber, and fiber-path energies into one
objective. Contact features are rebuilt at each objective evaluation.

`muscle_sim/run_multimuscle.py` loads the three requested bodies, preserves
their separate topology and attachments, transforms the Zygote bone collision
meshes with the same rest-local convention as the existing bakers, builds a
sectional fascia shell, performs intersection inspection, and drives the
coupled state with the existing BVH continuation convention.

The contact term is a symmetric, frictionless, quadratic point-triangle
penalty. Both point-to-triangle directions are evaluated. Buried vertices use
a signed inside/outside gap, so existing overlap is not mistaken for positive
clearance. Closest points and barycentric coordinates distribute equal and
opposite forces to the target triangle. This is a first penalty model, not
IPC.

## Commands

Inspect geometry without solving:

```bash
pyMAC/bin/python -m muscle_sim.run_multimuscle \
  --experiment fascia --inspect-only \
  --output .bake_outputs/pes_fascia_inspection
```

Run a preset:

```bash
pyMAC/bin/python -m muscle_sim.run_multimuscle \
  --config config/pes_anserinus_multimuscle.yaml \
  --bvh data/motion/left_thigh_quasistatic_5pose.bvh \
  --start-frame 0 --end-frame 3 \
  --experiment no_contact \
  --output .bake_outputs/pes_no_contact
```

## Verified data

All requested datasets exist; no muscle was substituted.

| Muscle | vertices | retained tets | attachment patches | fibers |
|---|---:|---:|---:|---:|
| Left Gracilis | 576 | 1919 | 2 x 32 vertices | 25 |
| Left Sartorius | 672 | 2378 | 2 x 32 vertices | 25 |
| Left Semitendinosus | 576 | 2193 | 2 x 32 vertices | 25 |

The contact derivative suite includes rigid point-triangle, deformable
point-triangle, a buried point in a closed deformable tetrahedron, symmetric
contact, fascia-side contact, tangential freedom, and normal-only cohesion.
All pass centered finite differences. All original isolated tests still pass.

The coupled no-contact baseline passes through frame 1 after restoring the
matrix-free Newton-Krylov fallback to the global solver. Frame 1 has exact
attachments, no inversion, `min J = 0.50443`, and free-gradient norm
`0.09913 N`. It required 20 accepted adaptive substeps. Frames 2-3 have not
yet been run in the coupled path.

## Measured initialization conflicts

The supplied rest surfaces already overlap:

| Pair | inside surface vertices | maximum exact surface depth |
|---|---:|---:|
| Left Gracilis / Left Sartorius | 5 | 0.096 mm |
| Left Gracilis / Left Semitendinosus | 34 | 1.239 mm |
| Left Sartorius / Left Semitendinosus | 14 | 0.479 mm |

Twenty of these vertices belong to hard attachment regions. Bone intersections
are also concentrated in attachment regions, which are excluded (plus one
surface ring) from bone contact as configured.

A five-stage contact-stiffness ramp does not yet produce an acceptable
depenetrated rest state. For muscle contact the final state has approximately
3.30 mm effective penalty penetration, `min J = 0.060`, and a large residual.
For bone-only contact, penetration is under the configured tolerance but the
state still has `min J = 0.063` and a large residual. Both are rejected and
exported; they are not counted as successful simulations.

The repaired sectional fascia is watertight and encloses every selected rest
surface vertex. It is deliberately not a global convex hull. It is currently
too loose: its rest geometric fill ratio is only 3.19%, so it is a collision
proxy rather than a validated compact fascia.

## Current limitations and next blocker

- Contact-only, fascia, cohesion, and wrapping pose experiments are not yet
  accepted because rest depenetration fails before pose continuation.
- The penalty closest-feature objective is piecewise smooth and L-BFGS stalls
  with a large residual. A contact-appropriate semismooth Newton,
  augmented-Lagrangian, or IPC backend is needed; loosening the residual or
  penetration criteria would hide the problem.
- Exact triangle-triangle intersection counts are unavailable because
  `python-fcl` is not installed. Inside-vertex counts and exact nearest-surface
  depths are reported instead.
- Cohesion has a verified energy kernel but is not enabled in the global
  experiment until contact passes.
- The configured knee capsule and distal-tendon classifier are not connected
  yet.
- GLB exports contain separate named muscle, bone, fascia, and contact debug
  geometry. Dedicated in-viewer toggles are not implemented yet.
- No anatomical-realism claim is made.
