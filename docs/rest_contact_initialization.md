# Rest-contact feasibility and augmented-Lagrangian status

## Exact feasibility result

The primitive-level report contains 221 initially penetrating directed
vertex-triangle constraints:

| Class | Count |
|---|---:|
| FREE_FREE | 23 |
| CONSTRAINED_FREE | 10 |
| FREE_CONSTRAINED | 40 |
| CONSTRAINED_CONSTRAINED_SAME_BONE | 148 |
| CONSTRAINED_CONSTRAINED_DIFFERENT_BONES | 0 |

There is no strict different-bone hard/hard infeasibility. Most same-bone
constraints are expected muscle/bone attachment intersections. Muscle-muscle
contact has 53 directed penetrations: 23 free/free, 10 constrained/free,
10 free/constrained, and 10 hard/hard on the same bone.

The maximum exact initial muscle-muscle depth is 1.2391285 mm. The initial
solver proxy is 2.2391285 mm because it is defined as
`thickness - signed_gap`, and thickness is 1 mm. The previous 3.30 mm proxy
was measured after the rejected penalty solve had deformed the geometry.

## Attachment inspection

Each supplied cap has 32 vertices. The cap seams are duplicated/disconnected
surface components, so they do not contain a healthy topological interior.
No patch was changed automatically.

The exported proposal ranks vertices geometrically and retains 19/32 (59%) as
a candidate hard core, followed by one transition ring and a free remainder.
This is a debug proposal, not an anatomical footprint determination.

Bone-distance outliers occur in:

- distal Sartorius: vertices 665–668;
- proximal Semitendinosus: vertices 1–9;
- distal Semitendinosus: vertices 548, 549, 574, and 575.

## Augmented-Lagrangian convention

Feasible normal gap is `g >= 0`; multiplier is `lambda >= 0`. For
`c=-g <= 0`, the local term is:

```text
(max(0, lambda-rho*g)^2-lambda^2)/(2*rho)
```

and the multiplier update is:

```text
lambda <- max(0, lambda-rho*g)
```

The implementation includes persistent IDs, activation/release hysteresis,
multiplier updates, Fischer-Burmeister residuals, and fully constrained
infeasibility detection. Twenty combined isolated, contact-gradient, and AL
tests pass.

## Gracilis-Semitendinosus Experiment F

Both required attachment variants were run:

| Variant | Result | max penetration | min J | mechanical residual |
|---|---|---:|---:|---:|
| full hard patches | rejected | 0.7920 mm | 0.2000 | 1.5967 N |
| 59% hard core + transition | rejected | 0.7924 mm | 0.2000 | 1.5991 N |

Neither variant passes the 0.1 mm penetration, 0.1 complementarity, mechanical
residual, and `J>=0.2` combined criteria. Shrinking the patch makes no material
difference, so attachment overconstraint is not the current cause.

The rejected trial moves no vertex more than 0.363 mm and keeps exact hard
attachments. It reaches the guard in Semitendinosus tet 337:

- rest volume: `2.2313e-11 m^3`;
- minimum rest dihedral: `0.967 degrees`;
- deformed singular values: `[5.97, 0.992, 0.0338]`;
- ten active contacts within 3 mm.

Semitendinosus tet 44 has edge aspect ratio 14.17 and minimum rest dihedral
0.454 degrees. These are local mesh-quality failures under concentrated
contact load. Local retetrahedralization is recommended before increasing
`rho` or weakening the `J` guard.

No corrected rest mesh was published to
`preprocessed/rest_contact_corrected/`, because neither run passed.
Experiments G/H and fascia remain gated.
