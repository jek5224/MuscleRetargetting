# Muscle Imitation Learning

**1000+ commits · 26 days · Jan–Mar 2026**

---

## Project Timeline

![[gantt.png]]

## Commit Activity

![[git_heatmap.png]]

## Lines Changed per Phase

![[code_stats.png]]

---

## Phase 1 · Contour Processing
**Jan 7–22**

Scalar fields → Contour detection → Gap filling → Transitions

M-to-N cutting · Bounding plane optimization · Stream building

Manual cutting window · Level Select GUI · Tetrahedralization

![[pipeline.png]]

---

## Phase 2 · Rendering
**Jan 29–30**

Two-pass transparency · Vertex array optimization

Bulk muscle UI · 1:1 edge case handling

---

## Phase 3 · Motion & FEM
**Feb 3–6**

BVH playback → Tet deformation baking → Waypoint caching

Refactored 6,680 lines into `zygote_mesh_ui.py`

---

## Phase 4 · Animations
**Feb 9–12**

10+ animated pipeline steps with replay infrastructure

Scalar spread · Contour reveal · Smooth slerp · Cut · Stream smooth · Fiber growth · Tet build

---

## Phase 5 · Solver Optimization
**Feb 20**

ARAP speedup: fused kernels · warm-starting · reduced system

scipy splu on free-DOF system

---

## Phase 6 · Neural Network
**Feb 26 – Mar 3**

V1: SIREN → LeakyReLU residual blocks

V2: Shared decoder · muscle embeddings · PCA targets · sliding window

Real-time NN inference in viewer

![[nn_v1.png]]

![[nn_v2.png]]

![[training_curves.png]]

---

## Phase 7 · Batch Baking
**Mar 3–4**

82 BVH × 4 regions → 5 GPUs via Slurm

Sync-and-delete workflow for 115 GB output

---

## Phase 8 · DOF Grid Distillation
**Mar 5–6**

Systematic 7-DOF sampling (hip 3 + knee 1 + ankle 3) via Latin Hypercube

Nearest-neighbor ordered traversal for FEM warm-starting

NN V3: 24M param network with fixed vertex + inter-muscle constraint losses

GPU inference at 365 FPS · Widened ROM from BVH exponential map analysis

4-GPU parallel baking on A6000 (30K samples × 2 regions)
