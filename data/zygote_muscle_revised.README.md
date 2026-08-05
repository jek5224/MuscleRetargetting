# `zygote_muscle_revised.xml` — usage guide

Muscle anatomy + per-waypoint LBS metadata for 78 leg muscle units,
10000 fibers, 100800 waypoints.

## Schema

```xml
<Muscle>
  <Unit name="L_Adductor_Brevis" f0="151.85" lm="1.2" lt="0.2"
        pen_angle="0.0" lmax="-0.1">
    <Fiber>
      <Waypoint body="L_Os_Coxae0"
                p="0.030073 0.919683 0.043564"
                lbs_bones="L_Os_Coxae0"
                lbs_locals="-0.019190 0.121575 0.072105"
                lbs_weights="1.000000"/>
      <Waypoint body="L_Femur0"
                p="0.052037 0.887136 0.035110"
                lbs_bones="L_Os_Coxae0,L_Femur0"
                lbs_locals="-0.016705 0.229990 0.173305; 0.169354 -0.015819 -0.015334"
                lbs_weights="0.279581 0.720419"/>
      ...
    </Fiber>
    <Fiber>...</Fiber>
  </Unit>
  <Unit name="L_Biceps_Femoris_Long" .../>      <!-- biarticular split -->
  <Unit name="L_Biceps_Femoris_Short" .../>
  ...
</Muscle>
```

## Unit attributes (Hill model parameters)

| Attr | Meaning |
|------|---------|
| `name` | unique muscle / fiber-group ID.  Split muscles use `_Long`, `_Short`, `_Medial`, `_Lateral` suffixes. |
| `f0` | max isometric force in newtons. |
| `lm`, `lt` | optimal muscle length / tendon length normalisation. |
| `pen_angle` | pennation angle (rad). |
| `lmax` | passive force exponent reference. |

## Waypoint attributes (LBS)

- `lbs_bones="A,B,C"`: K body names (K = 1 endpoint / 2 monoarticular /
  3 biarticular intermediate).
- `lbs_locals="x1 y1 z1; x2 y2 z2; ..."`: K **per-bone local** positions
  in each bone's body frame.  Semicolon-separated.
- `lbs_weights="w1 w2 w3"`: K LBS weights summing to 1.

(`body` and `p` are included for human readability — `body` matches
`lbs_bones[max_weight_index]`, and `p = Σ w_i · (R_rest_i · local_i +
t_rest_i)`.  Not required by the K-bone loader.)

Solved via least-squares over 132 walk.bvh poses
(`tools/reverse_lbs_solve.py`) so the LBS formula reproduces baked
muscle deformation to ≤15 mm max residual across all 100800 waypoints.

## LBS formula

At any skeleton pose, the waypoint's world position is

```
P_world(pose) = Σ_i  w_i · ( R_bone_i(pose) · local_i + t_bone_i(pose) )
```

where:
- `R_bone_i, t_bone_i` are body i's rotation and translation at the pose.
- `local_i` is `lbs_locals[i]`.
- `w_i` is `lbs_weights[i]`.

For K=1 (endpoint waypoint): single bone, weight 1, reduces to rigid
attachment.
