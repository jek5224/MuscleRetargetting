# Wholesome project recap: muscle imitation, viewer, anatomy, and baking

This is the single authoritative project history. It is intentionally
process-oriented: failed
approaches are preserved because they explain why the current pipeline has its
flags, manifests, diagnostics, and separate bake files.

## The story of the project

This project began with a simple visual ambition: take human motion and make a
digital body move in a way that still feels like a body. Not a collection of
rigid bones, not a painted skin that slides over a skeleton, and not a movie
that can only be watched from one prepared angle, but a body whose muscles have
shape, direction, attachments, neighbors, and a visible relationship to the
skeleton underneath.

At first, the problem looked like motion retargeting. A movement existed in a
BVH file, and a target skeleton existed in the project. The work was to make
one skeleton understand the other: axes had to be interpreted, rest poses had
to be aligned, bones had to be named, proportions had to be reconciled, and
left and right had to remain left and right. The early experiments with stretch
axes, skeleton loading, fitting boxes, PCA alignment, ground planting, ankle
mirroring, leg IK, spine direction, sternum fitting, and clavicle scale were
all versions of the same question: what does this motion mean on this body?

That question became visible in the viewer. The viewer grew from a renderer
into a place where the project could think. Clicking a muscle, selecting both
sides, expanding a tree, seeing a bone box, reloading a BVH, or watching a
fiber line move was not just interface polish. It allowed an abstract
transformation to become an anatomical observation. When something looked
wrong, the viewer helped distinguish a bad motion pose from a bad mesh, a bad
attachment from a bad solver, and a bad cache from a bad rendering state.

The next realization was that a muscle could not remain only a surface. A
surface can show where a muscle is, but it does not tell us how it should
shorten, how it should preserve its cross-section, where its origin and
insertion are, or how force and shape should travel through it. The project
therefore moved into tetrahedral volumes. Original and revised Zygote meshes
were examined, contour surfaces were built, TetGen pipelines were tested,
boundary loops were closed or deliberately left open, slivers were filtered,
orphan vertices were rescued or removed, and the surface was mapped into a
volume that could be simulated.

This was also the point where anatomy became data. Each tet file began to carry
more of the project’s understanding: cap faces, anchor vertices, attachment
bones, contour levels, stream endpoints, fibers, MVC weights, waypoint
barycentrics, and rendering metadata. A vertex was no longer just a row in an
array. It could be an origin anchor, an insertion anchor, a free belly vertex,
a contour sample, a fiber-supporting point, or a surface point that needed to
stay outside a bone.

The fiber work gave the muscles an internal language. Fibers were extracted
from contour streams and ordered from origin to insertion. Independent streams
had to be treated independently. Split or cut muscles could not lose their
stream identity. Contour levels had to be selected consistently, and stale
fiber state had to be cleared before a mesh was rebuilt. The viewer learned to
draw those fibers efficiently, fade them, animate them, and protect the draw
path from bad arrays and playback crashes. In the process, fibers became more
than lines on the screen: they became directions for deformation, clues for
cross-sectional shape, and a way to inspect whether the generated volume still
remembered the anatomy it came from.

The project then asked how the muscle should move between its attachments.
LBS was the first answer. It was fast, understandable, and useful as an
initial guess. But it could not by itself preserve a believable volume. ARAP
provided local shape preservation. Anisotropic edges allowed the muscle to
shorten more freely along one direction while retaining its cross-section.
Tendon zones received special treatment. Axial pose priors made knee-crossing
muscles contract with the pose, and transverse targets helped prevent a
muscle from becoming a collapsing ribbon.

At the same time, the body became a community rather than a set of isolated
muscles. Muscles that touch or run together should not drift apart without
reason. Inter-muscle constraints and fascia-like relationships were introduced
to keep neighboring structures connected. This created a new difficulty:
bonding is not the same as collision avoidance. Two muscles can be strongly
linked and still overlap. A muscle can remain attached to its neighbors and
still pass through a femur. Every improvement therefore revealed another
layer of the physical problem.

The viewer kept the work honest. A numerical solver could report convergence
while a muscle trembled in playback. A cache could contain all the expected
frames while the waypoints remained stale. A muscle could appear attached at
one endpoint while its fixed mask was empty. A bone-contact flag could be set
while imported face normals pushed a penetrated point in the wrong direction.
The project repeatedly returned to the viewer to ask not only “did the code
finish?” but “does the body still make sense when it moves?”

There were many paths toward contact. Surface projection was quick but could
introduce new penetrations through harmonic propagation. IPC promised stronger
point-triangle and edge-edge guarantees but required heavier dependencies and
more computation. Unified ARAP made it possible to place all active muscles in
one graph and add bone-contact targets, but contact targets needed correct
geometry and consistent temporal enforcement. The fast baker made iteration
quick enough to expose these failures; the headless bonded baker made the
useful baseline stable enough to produce full caches.

The learning work grew from the same need for trustworthy deformation. Reverse
LBS asked whether a deformed muscle could be expressed again as useful local
bone-space information. XML writers, DART bindings, multi-bone extensions, and
walk/run/combined outputs connected baked motion to interpretable targets.
Volume distillation and learned deformation could then be considered, not as a
replacement for anatomical reasoning, but as a way to make a reasoned result
faster.

Popliteus became a small but meaningful chapter in the story. It crossed the
knee, attached to the femur and tibia, and appeared in two region manifests.
When it drifted, the problem was not simply that the solver was weak. The
muscle had been placed in two different anatomical neighborhoods. Its
neighbors, collision candidates, and unified graph changed with the region.
Moving it into the lower-leg manifest was a reminder that bookkeeping is part
of anatomy: the definition of a group changes the forces a muscle feels.

The current result is therefore not one perfect algorithm. It is a layered
instrument built through questions, visual checks, failed experiments, and
careful corrections. The project can start with motion, expose the skeleton,
show the fibers, construct a volume, preserve attachments, couple neighboring
muscles, test contact, cache the result, and play it back. When something fails,
the history gives a place to look.

The wholesome part of the project is that every abstraction was brought back
to the body. Retargeting was judged by posture. Fiber extraction was judged by
direction. Tetrahedralization was judged by shape and attachment. ARAP was
judged by volume. Inter-muscle bonds were judged by relationships. Collision
was judged by whether a muscle respected bone. Caches were judged by whether
the motion remained stable when played. Learning was judged by whether it
could preserve the same understanding more efficiently.

That is the year-long arc: from moving a skeleton, to understanding a muscle,
to giving that muscle an internal architecture, to letting many muscles share
a body, to building a viewer where the entire chain can be questioned. The
current bake files are only the latest chapter of a much larger effort to make
motion, anatomy, geometry, mechanics, and visual understanding agree.

## Chronology from the repository history

### 2024: retargeting, skeleton loading, and geometric inspection

The first recorded commit is `a740a1c` (2024-08-25), “Working on
retargetting.” The early sequence tests multiple stretch axes and revises the
implementation when the assumptions fail. The central challenge is mapping
external motion onto a target skeleton with different rest pose, proportions,
axes, and naming.

October commits move into skeleton loading and sectioning. Bone fitting boxes
and PCA-aligned bounding boxes provide geometric diagnostics. November adds
clicking and symmetric muscle selection, allowing the skeleton and muscle
assets to be inspected interactively rather than treated as opaque arrays.

### 2025: fibers and structured muscle data

`e09e8c9` (2025-06-24) records multi-fiber extraction. Muscles become
structured volumes with fibers, contour streams, endpoint regions, MVC data,
bounding planes, and waypoints. This data later drives anisotropic ARAP,
contraction priors, tendon handling, fiber visualization, and reverse-LBS
experiments.

### January 2026: viewer restoration

The January history restores working MVC state, restores mesh-loader mixin
inheritance, auto-loads the previous muscle selection, auto-expands the
Zygote/muscle tree, repairs contour-finding signatures, and fixes pyimgui tree
node compatibility. These are experimental reproducibility fixes: a visual
comparison is not meaningful if the viewer silently loads a different muscle
set or fails to rebuild contour state.

### April–May 2026: many solver generations

The project branches into LBS barriers, original-mesh baking, layered baking,
contour/tet simulation, PolyFEM/quasi-static experiments, headless ARAP,
surface contact, IPC phase 2, PyUIPC, fascia, and cache repair. Local bake
scripts compare 50 and 150 iteration budgets, skin-prior/no-skin variants,
bone-contact settings, and regional output tags. Slurm scripts make the same
experiments portable to larger machines.

### June 2026: motion plausibility

Motion-side commits add or refine ground planting, two-link leg IK, ankle-rest
mirroring, continuous anatomical/BVH bend blending, anatomical-forward
fallbacks, humerus low-confidence behavior, clavicle scaling, and sternum
fitting. These changes reduce the chance that a later deformation solver is
compensating for an implausible skeleton trajectory.

### July 2026: fast bake, contact, stability, and regions

The fast surface baker is created and instrumented. It first moves whole
volumes by LBS target deltas after an earlier cap-only approach stretched the
muscles. Ten-frame tests expose large bone penetrations. Inter-muscle sweeps
are enabled for accuracy experiments, but their cost is measured explicitly.

The headless bonded baker becomes the speed/quality baseline. A dedicated
bone-contact wrapper forces unified contact and per-muscle contact. A bug in
the use of OBJ face normals is found and corrected using closest-point
geometry. Viewer tet-internal controls are hidden. Popliteus is removed from
the upper-leg manifest and added to the lower-leg manifest.

## Architecture in practical terms

The pipeline is:

```text
BVH → MyBVH/DART pose → retargeted body transforms
    → muscle manifest + tet metadata
    → anchors/fixed masks/fibers/contours/waypoints
    → LBS initialization and warm start
    → unified ARAP volume
       + internal tet edges
       + anisotropic/fiber/tendon targets
       + inter-muscle bond edges
       + optional fascia constraints
       + optional bone-contact targets
    → per-muscle position snapshots
    → chunked NPZ + waypoint patch
    → viewer cache loading and playback
```

The chain deliberately separates motion, geometry, mechanics, caching, and
visualization. Each boundary has its own diagnostics.

## Asset and data inventory

`Zygote_Meshes_251229/` is the active Zygote asset family. Its `Skeleton/`
directory supplies collision meshes such as femur, tibia/fibula, pelvis, and
patella. Its `Muscle/` directory supplies the source muscle surfaces.

`tet/` contains structured tet files. A tet file includes vertices,
tetrahedra, render/simulation faces, cap faces, anchor vertices,
`cap_attachments`, attachment body names, contours, fibers, MVC weights,
waypoints, stream endpoints, and contour levels. Missing or inconsistent
metadata can cause a visually plausible but mechanically unanchored result.

`tet_orig/`, `tet_orig_open/`, and `tet_sim/` preserve alternate preparation
stages. The revised, subdivided, and original Zygote directories preserve
geometry variants used during comparison.

`DTI/` contains subject-specific and built-mesh assets and analysis. `Papers/`,
`EMU.pdf`, `MUSIC.pdf`, `LowerBody_Dataset.pdf`, and related documents retain
the scientific context. `learning/`, `models/`, `volume_distill/`, and
`NeuralDeformationGradients/` contain learning and differentiable-deformation
experiments whose training targets depend on bake quality.

## Solver generations and what each taught

### LBS/barrier

`bake_lbs_barrier.py` is a control: it asks whether the skeleton and target
attachments are coherent without ARAP or inter-muscle springs. It is fast but
cannot provide the full volume and contact behavior.

### EMU and contour simulation

`bake_emu.py` and `bake_contour_sim.py` contain richer attachment, harmonic,
volume, contour, and ARAP primitives. They are useful reference implementations
for fixed targets, deformation initialization, and EMU-style behavior.

### Original and layered routes

`bake_original_mesh.py` and `bake_layered.py` preserve more original geometry
and layering. They provide high-fidelity comparisons but have heavier setup and
runtime costs.

### Headless bonded ARAP

`bake_headless.py` loads a muscle JSON, finds inter-muscle constraints, builds a
unified graph, solves with the TAICHI/GPU/CPU ARAP backend, writes chunked NPZ
files, and patches waypoints. Its key variables are `--settle-iters`,
`--inter-muscle-weight`, `--inter-k`, backend choice, skin prior, fiber/tendon
options, and plateau exit.

### Surface and IPC routes

`bake_headless_surfcoll.py`, `bake_surface_contact.py`, `bake_ipc_phase2.py`,
`bake_pyuipc.py`, and `sim_pyuipc_upleg.py` explore surface projection,
point-triangle/edge-edge contact, CCD, barrier energy, and IPC. These routes
are important research references, but `ipctk` availability and CPU cost make
them less convenient for routine full-motion iteration.

### Fast projection route

`bake_surface_fast.py` is a diagnostic/preview baker. It records frame timing,
supports region resolution, transports whole volumes by LBS deltas, and can
project bone/inter-muscle contacts. It revealed the difference between a
quickly plausible frame and a stable, collision-aware full cache.

## Detailed failure record

### Cap-only movement

Only moving fixed cap vertices left free vertices in their previous pose. The
muscle stretched between old geometry and new attachments. Whole-volume LBS
transport fixed the fundamental cause.

### Contact target direction

The unified contact code used face normals from imported bone meshes. OBJ
winding is not guaranteed consistently outward. A penetrated vertex could
therefore be pushed deeper into the bone. The corrected target uses the vector
from the current interior point back through its closest surface point and
adds the safety margin outward.

### Contact target cadence

Reusing penetration targets for several iterations amortized queries but could
leave stale targets as ARAP moved the volume. Corrected validation recomputes
targets every iteration.

### Plateau jitter

Plateau exit makes convergence depth differ by frame. The fixed-budget mode in
`bake_headless_bone_contact.py` is slower but useful when temporal consistency
matters more than peak speed.

### Inter-muscle bonds versus collision

Inter-muscle edges preserve relationship but do not guarantee non-intersection.
Bone contact keeps vertices outside bone surfaces but does not create fascia-like
bonding. Both mechanisms must be measured separately.

### Popliteus duplication

Popliteus attaches femur-to-tibia and crosses the knee. It was duplicated in
upper/lower region definitions. Its graph neighbors changed depending on the
region, so it drifted when included in the upper-leg unified solve. It now
belongs only to the lower-leg manifest.

## Current authoritative full caches

```text
data/motion_cache/run/L_UpLeg_headless_full_corrected/
data/motion_cache/run/L_LowLeg_headless_full_corrected/
```

Both use `bake_headless.py` and cover all 66 `run.bvh` frames. The upper cache
contains 24 muscles and excludes Popliteus. The lower cache contains 13
muscles and includes Popliteus. Waypoints were patched after baking.

The corrected five-frame bone-contact validation is:

```text
data/motion_cache/run/L_UpLeg_bone_contact_fix_5f/
```

The older `L_UpLeg_bone_contact_full` cache predates the contact-direction
correction and should not be used for accuracy comparisons.

## Command cookbook

Full upper-leg bonded bake:

```bash
source pyMAC/bin/activate
MPLCONFIGDIR=/tmp python tools/bake_headless.py \
  --bvh data/motion/run.bvh \
  --muscles .muscles_L_UpLeg.json \
  --region-tag L_UpLeg_headless_full_corrected \
  --backend auto
```

Full lower-leg bonded bake, including Popliteus:

```bash
source pyMAC/bin/activate
MPLCONFIGDIR=/tmp python tools/bake_headless.py \
  --bvh data/motion/run.bvh \
  --muscles .muscles_L_LowLeg.json \
  --region-tag L_LowLeg_headless_full_corrected \
  --backend auto
```

Five-frame forced bone-contact validation:

```bash
source pyMAC/bin/activate
MPLCONFIGDIR=/tmp python tools/bake_headless_bone_contact.py \
  --bvh data/motion/run.bvh \
  --muscles .muscles_L_UpLeg.json \
  --region-tag L_UpLeg_bone_contact_fix_5f \
  --start-frame 0 --end-frame 4 \
  --backend auto --settle-iters 50
```

Viewer:

```bash
source pyMAC/bin/activate
python viewer/viewer.py
```

Select `run.bvh`, select the cache tag, and reload from disk.

## Reproducibility checklist

For each experiment preserve: Git commit; uncommitted files; BVH and frame
count; detected `T_frame`; asset and tet directories; manifest and muscle
count; backend; iteration/tolerance/plateau policy; inter-muscle weight and
neighbor cap; bone-contact flags, margin, weight, and cadence; skin/fiber/
fascia options; cache tag; waypoint status; fixed-mask/target counts; fixed
anchor error; bone penetration depth; inter-muscle distances; tet inversion
count; frame-to-frame displacement; and a short playback review.

## Future validation priorities

The next quantitative tools should report fixed-anchor error per frame, bone
penetration count/depth, inter-muscle minimum distance, unintended
intersections, inverted tetrahedra, and RMS/max frame-to-frame displacement.
Only after those checks should finer patella/tibia geometry or learned
acceleration be evaluated. A denser mesh cannot compensate for a wrong target,
wrong manifest, missing attachment, or inconsistent temporal solve.

## Closing account

The project’s result is the accumulated method: motion retargeting, skeleton
inspection, structured muscle data, tet construction, attachment targets,
fiber-aware deformation, unified ARAP, bonded muscles, contact experiments,
viewer/cache infrastructure, and learning preparation. Every bake file records
a real attempt to isolate one failure mode. The repository is therefore not a
collection of redundant scripts; it is the experiment history that explains why
the current baseline is structured as it is.

## Muscle mesh construction: the story before baking

The muscles did not begin as ready-to-simulate tetrahedral volumes. A large
part of the project was turning anatomical surface information into a usable
volume while preserving the information needed later for attachments and
fibers.

### Source mesh families

The repository contains original, revised, subdivided, and Zygote mesh
families. The project repeatedly compared these because they have different
vertex counts, boundary quality, cap structure, and suitability for TetGen or
contour-based tetrahedralization. The choice affects both visual fidelity and
the number of unknowns in ARAP.

### Contour-driven tetrahedralization

The contour route uses extracted cross-sections and stream metadata to create
coarse, structured tet volumes. Contours provide an anatomical coordinate
system: origin-to-insertion levels, cross-sectional bands, and fiber-related
directions. The resulting tet vertices can be tagged with contour levels and
classified into cross-contour, intra-contour, neutral, and tendon-zone edges.

This structure is what later makes anisotropic ARAP possible. Instead of
treating every tet edge identically, the solver can allow more axial motion,
preserve cross-sectional shape, or apply tendon-specific slack behavior.

### Original-OBJ tetrahedralization

The original-mesh path was built because coarse contour tets can lose surface
detail. `tet_from_original_obj.py`, `convert_orig_tet.py`,
`build_contour_orig_mapping.py`, and `remesh_tet_for_collision.py` investigate
how to keep original boundary vertices while generating a usable volume.

The history includes several concrete repairs:

- use high-resolution Zygote OBJ assets as TetGen input;
- preserve boundary vertices during decimation;
- close every boundary loop to create watertight volumes;
- use TetGen quality settings and minimum-ratio filtering to reduce slivers;
- drop folded render-face pairs that create cap pinches;
- preserve open boundaries where attachments require them;
- map original vertices back to contour-simulated positions;
- save the original vertex count so the viewer accepts the cache.

The original route produced more detailed surfaces but introduced harder
attachment, topology, and solve-size problems. Keeping both original and
contour paths made it possible to compare geometry quality against speed.

### Cap and anchor assignment

Many of the hardest “solver” bugs were actually cap-assignment bugs.
The project progressively added per-loop anatomical bone assignment,
origin/insertion bipartite assignment, per-waypoint bone lookup, anchor-bone
maps, dominant-bone cap fans, and nearest-bone-surface classification.

`anchor_bone_map` became authoritative over a fragile initialization BFS. XML
attachment names were used as a fallback when contour files were incomplete.
Low-leg attachment filling and `fix_cap_attachments.py` address files that
had vertices but no usable cap attachment records.

These changes answer a practical anatomical question: which exact boundary
vertices are fixed to the origin, which belong to the insertion, and which
are free belly vertices? Without that answer, even perfect ARAP cannot produce
a stable muscle.

### Orphan, sliver, and isolated-vertex rescue

Tetrahedralization produced several classes of geometry defects: vertices in
the vertex array but in no tet, cap-ring vertices without a containing tet,
zero-weight vertices, sliver tetrahedra, and isolated graph vertices. The
history records star-fan rescue, Steiner-point insertion, orphan stripping,
nearest-tet search caps, no-orphan/no-zero-weight checks, and same-muscle
isolated/stuck-vertex fixes.

These fixes are why the current pipeline can report 8,023 or 8,971 unified
vertices while still preserving meaningful attachment masks and graph
connectivity.

## How fibers were extracted and made useful

Fiber extraction was not a single button. It is a sequence of geometric and
visual steps.

### Fiber source and coordinate system

The muscle representation stores one or more fiber sets per muscle. Each set
is associated with contour streams and endpoint regions. Origin and insertion
directions provide an anatomical axis, while cross-contour planes define the
local section in which fibers and contours are sampled.

The system must handle independent multi-stream muscles, cut or split streams,
stale stream-group metadata, and muscles with different contour counts. The
May 2026 history includes per-stream level selection, independent stream
spacing searches, stream-count handling, and cut-muscle linkage preservation.

### Contour selection and scalar levels

The viewer and preprocessing tools select contour levels along each stream.
Early selection used global thresholds; later changes made thresholds a
percentage of total contours and allowed independent stream counts. A hard cap
on contour-search scalar values prevents runaway searches.

This matters because a fiber is only meaningful if its contour sequence is
ordered consistently from origin to insertion. A bad level selection can make
the fiber kink, reverse direction, skip a belly, or attach to the wrong stream.

### Fiber architecture construction

`viewer/fiber_architecture.py`, `viewer/contour_mesh.py`, and the fiber-building
paths in `muscle_mesh.py` construct renderable fiber lines from the anatomical
metadata. Fiber state is reset at the start of building so stale cached fibers
do not survive a mesh reload. Lazy draw-array initialization avoids doing
expensive OpenGL preparation for muscles that are never displayed.

The fiber renderer went through several performance iterations: immediate-mode
draw paths, server-side VBOs, depth-fade lines, bad-array guards, and playback
segfault protection. The result is not only a visual overlay; it is a way to
inspect whether the anatomical coordinate system survived tetrahedralization
and deformation.

### Fiber-related mechanics

Fiber and contour information later became solver data. Cross-contour edges
can receive a lower or higher ARAP weight, intra-contour edges can preserve
cross-section, and tendon-zone edges can use slack-only or scalar spring
behavior. Axial pose priors shrink cross-contour rest lengths under knee flex
and expand selected transverse targets to avoid unrealistic volume collapse.

The project also added fiber transparency, contour highlights, belly-space
inspection, animation toggles, process-step controls, and global state
propagation across muscle trees. These UI features made it possible to compare
fiber direction with the actual deformed muscle instead of relying on a single
surface silhouette.

### Fiber quality checks

`check_fiber_orientation.py`, moment-arm analysis, reverse-LBS comparisons, and
visual overlays provide different checks. A fiber can be geometrically smooth
but anatomically reversed; it can point correctly at rest but kink under a
wrong retargeted pose; or it can look correct while its contour/tet mapping is
wrong. The project therefore keeps both numerical and visual checks.

## MVC, waypoints, and muscle metadata

MVC weights and waypoints connect the anatomical representation to viewer and
learning workflows. Waypoints can be stored as barycentric coordinates inside
tetrahedra, transformed with the deformed volume, and patched into cache files
after baking. MVC data supports activation-oriented views and later learning
experiments.

The project added loaders, XML writers, scatter-order fixes, reverse-LBS XML
injection, and cache waypoint restoration. A viewer toggle can show reverse-LBS
waypoints and restore baked waypoints when disabled. This is a separate data
path from tet rendering, so both must be validated.

## DTI and subject-specific anatomy

The `DTI/` subtree and `mri_dti_overlay.py` preserve the effort to relate the
generic Zygote/muscle representation to subject-specific diffusion and MRI
information. DTI assets include subject bone/muscle OBJ material, built meshes,
and visualization/overlay tools. This line of work asks whether fiber
architecture and deformation targets can be grounded in subject anatomy rather
than only a generic template.

The existence of DTI and MRI overlays also explains why the viewer emphasizes
layered inspection: skeleton, muscle surface, tet structure, fibers, contours,
and external anatomical data must be visible in compatible coordinate frames.

## Reverse-LBS and learning direction

The reverse-LBS effort asks the inverse question: given deformed or baked
muscle positions, can useful local bone-space targets and weights be recovered?
`reverse_lbs_solve.py`, `reverse_lbs_solve_multi.py`,
`write_reverse_lbs_xml.py`, `inject_lbs_into_xml.py`, and
`trim_reverse_lbs.py` form this pipeline.

The work includes three-bone extensions, DART binding, walk/run/combined XML
outputs, viewer XML loading, scatter-order fixes, and vectorized application.
Reverse-LBS is useful both as a visualization tool and as a bridge toward
learned deformation: it can provide compact, interpretable targets instead of
requiring a model to memorize every world-space vertex trajectory.

The learning and volume-distillation directories build on this foundation.
The order matters: reliable retargeting and anatomically coherent bake targets
must exist before a learned model can be evaluated fairly.

## Viewer as a scientific instrument

The viewer evolved from a renderer into a debugging instrument. It supports
selection, symmetric selection, contour and fiber inspection, tet rendering,
activation and transparency controls, cache reload, BVH list reload, reverse-
LBS display, skeleton/muscle overlays, and playback.

The cache reload path was important for long experiments. A bake can finish in
the background, then the viewer can rescan the BVH/cache directories and load
new chunks without restarting. Region tags make side-by-side comparisons
possible: headless bonded, fast projection, bone-contact, skin-prior, no-skin,
and original/contour variants can coexist.

The removal of tet-internal controls in July is therefore a cleanup of the
routine interface, not a deletion of the underlying tet data. Internal faces
remain part of development/debug paths; they are simply not needed during
normal fiber and surface inspection.

## Scientific questions the project has addressed

The implementation history can be read as a set of research questions:

1. Can external human motion be retargeted onto a detailed skeleton?
2. Can a muscle surface be converted into a volume without losing anatomical
   endpoints and fiber structure?
3. Can contours and fibers provide a stable local coordinate system?
4. Can ARAP preserve shape while allowing physiologically plausible shortening?
5. Can neighboring muscles remain attached without collapsing into one mesh?
6. Can bones act as true one-sided obstacles rather than visual references?
7. Can the result be cached in a format that a live viewer can inspect?
8. Can reverse-LBS and distillation compress the deformation into useful
   learned or interpretable targets?
9. Can all of the above remain reproducible across motion files and regions?

The project’s current bake baseline answers many of these questions
partially—not perfectly—and the retained alternative scripts document where
each answer is strong or weak.

## Code-level walkthrough of the viewer

The viewer is not a thin OpenGL shell. It is a stateful research application
whose UI, mesh objects, skeleton, solver caches, and motion cache all share
state. Understanding this code explains why a visual change can affect a bake,
and why a cache can be numerically valid but visually stale.

### Startup and global state

`viewer/viewer.py` constructs the environment, DART skeleton, OpenGL/GLFW
window, ImGui context, and the collections used by the Zygote UI. The UI
module stores per-viewer state for selected muscles, groups, loaded BVHs,
current motion frame, cache dictionaries, DTI/MRI overlays, draw flags,
activation, transparency, fiber display, and simulation settings.

The viewer persists the last muscle selection in `.last_loaded_muscles.json`
and can restore it at startup. This is why the tree and selected-muscle state
are part of reproducibility rather than merely convenience.

### Muscle loading

`add_muscle_mesh` and `_load_muscle_data` create a mesh object from a Zygote
OBJ, assign color/transparency/draw state, and attach the `MuscleMeshMixin`,
`ContourMeshMixin`, and `FiberArchitectureMixin` behavior. The object can then
own surface vertices, render faces, tet vertices, simulation tets, cap faces,
waypoints, contours, fiber samples, activation, and solver state.

`add_muscle_group` loads region/group definitions. Group processing keeps
belly/tendon counterparts linked, chooses an owner for connected components,
resamples contour streams, propagates draw/animate state, and prevents a
tendon mesh from being independently processed twice.

### Zygote processing pipeline

The central group pipeline performs several stages:

1. identify belly and tendon components;
2. establish counterpart links and side symmetry;
3. choose or rebuild contour sources;
4. establish corner correspondence and contour-level schedules;
5. extend or orient tendon streams relative to the belly;
6. generate or restore fibers and waypoints;
7. classify and build a contour/tet volume;
8. assign material/region labels and rendering arrays;
9. prepare the simulation and viewer state.

The code intentionally supports deferred processing. Expensive geometry can be
prepared after a group’s UI changes have settled, which prevents each checkbox
or slider event from rebuilding all tet and fiber arrays.

### Contour geometry

`viewer/contour_mesh.py` provides the geometry behind 2D inspection and
3D contour construction. It computes Newell and best-fit normals, projects
vertices into a local plane, finds minimum-area bounding boxes, aligns bases
continuously between levels, and reconstructs global positions from local
coordinates.

Continuous basis alignment is important: independently fitted planes can flip
or rotate from one contour to the next. The reference-basis logic keeps the
contour coordinate system coherent so fibers and corners do not suddenly
reverse direction.

The 2D inspection windows expose corner correspondence, contour levels,
stream/fiber edits, and belly-space highlights. These tools are where many
geometric assumptions can be observed before a bake consumes them.

### Fiber architecture implementation

`viewer/fiber_architecture.py` implements several related constructions. It
maps contour boundary points into angular unit-circle parameters, finds angle
brackets, computes radial weights, and creates boundary/interior triangulations.
The `UnitCircleTriangulation`, `DirectFiberTriangulation`, and
`GeodesicTriangulation` classes represent different parameterizations.

`FiberTriangleEmbedding` stores how sampled fibers sit inside the 2D
triangulation. Barycentric embedding lets a fiber waypoint be reconstructed
after the contour boundary deforms. Harmonic solves fill interior vertices
from fixed boundary positions. Signed-area checks detect triangle flips, and
area-preservation refinement reduces distortion while respecting boundary
targets.

The geodesic path uses a polygonal boundary rather than assuming a perfect
circle. Shared geodesic triangulation allows multiple streams to reuse a
consistent interior parameterization. This is important for muscles whose
cross-sections are irregular or whose fiber streams are not independent disks.

### Fiber editing and testing

The viewer contains operations to add, delete, test, and recompute fibers for
individual streams. Corner correspondence can be reset or solved across all
levels. Fiber samples are copied or resampled when a belly is extended with a
tendon, and linked counterpart meshes inherit the appropriate schedules.

Fiber draw arrays are generated lazily, reset on rebuild, guarded against bad
array lengths, and updated for playback. This implementation work prevents a
visualization optimization from becoming a segmentation fault during BVH
animation.

### Tetrahedralization in the viewer

`_tetrahedralize_zygote_group_surface` and its single-surface/original-surface
variants assemble a closed or open simulation surface, remove duplicate
vertices, identify boundary loops, cap required openings, and invoke the tet
pipeline. Component labels are propagated so the resulting tets retain
belly/tendon provenance.

The viewer computes volume and tet-quality statistics, identifies mixed-region
tets, and prepares render/simulation face arrays separately. A rendering face
can be visible while an internal simulation face remains hidden. This
separation is why the July UI change could hide internal tet drawing without
removing internal topology from the solver.

### Attachment initialization

`MuscleMeshMixin` and the initialization path construct soft bodies with rest
positions, fixed indices, fixed targets, cap attachments, anchor-bone maps,
skin weights, and local bone transforms. `_update_fixed_targets_from_skeleton`
updates endpoint targets from current DART transforms. The solver later builds a
global fixed mask and fixed-target array from each muscle’s local indices.

The code prints both counts because a fixed vertex without a target is not
actually constrained to the moving skeleton. This distinction was central to
diagnosing drifting attachments.

### Inter-muscle constraints in code

`find_inter_muscle_constraints` first identifies candidate surface vertices,
uses spatial searches to find nearby vertices on other muscles, caps the
number of neighbors per vertex with `inter-k`, and creates cross-muscle edges
with a configurable weight. The headless unified solver merges these edges
with internal tet edges into one global graph.

The threshold controls how far apart two surfaces may be before they can be
linked. The cap prevents dense areas from creating a quadratic explosion in
edges. This is why the same threshold behaves differently for coarse contour
tets and fine original meshes.

### Unified ARAP implementation

`_run_unified_volume_sim` concatenates all active muscle vertices into one
global indexing space. It stores per-muscle offsets, a global fixed mask, rest
positions, neighbor lists, edge weights, rest edge vectors, muscle IDs, and
CSR edge masks for cross/intra/tendon classes.

The solver begins by updating each muscle from the skeleton and obtaining a
global LBS guess. If a previous solution exists, the configured LBS/warm-start
blend is used; fixed vertices are immediately overwritten by current targets.
The backend then solves the unified ARAP system, optionally receiving skin
prior targets, axial target edges, fiber-spring targets, and bone-contact
targets.

After the solve, isolated vertices are repaired using the nearest connected
vertex from the same muscle, not a neighboring muscle. Nearly motionless stuck
vertices can receive a same-muscle neighbor average. The result is then split
back into each muscle object, and the previous global solution is stored for
the next frame.

### Bone contact implementation

The unified path constructs current-frame transformed bone meshes and DART
shape meshes. Surface candidate vertices are computed from tet topology and
fixed attachment vertices are excluded. The ARAP system receives a diagonal
collision weight for these candidates.

Every contact iteration checks candidate points against bone bounding boxes and
`trimesh.contains`. Penetrating points receive closest-surface targets outside
the bone margin. The corrected code uses closest-point displacement rather than
assuming face-normal winding. Contact recomputation cadence is configurable;
the validation wrapper uses every-iteration recomputation.

### Motion cache loading

`_motion_cache_dir` derives `data/motion_cache/<BVH stem>` from the currently
selected BVH. The loader scans both the root and tagged subdirectories, finds
legacy files and chunk files per muscle, sorts chunks, checks expected vertex
counts, and builds frame-indexed deformation data.

The loader can run in a thread pool because a full motion contains many muscle
chunks. It prunes stale entries when loaded muscles change and supports a
forced reload after a new bake finishes. This is the code path used when the
viewer’s “Reload cache” control is pressed.

### DTI and MRI viewer paths

The DTI/MRI UI loads OBJ overlays and tract lines, resamples tract polylines,
transforms patient coordinates into viewer coordinates, and can align a
selected tract to a mesh. MRI slices are converted to textures with adjustable
window/quantile and alpha controls. These paths let subject-specific anatomy
be compared to the generic Zygote skeleton and muscle/fiber model.

### Viewer controls and why they exist

The UI exposes group processing, contour inspection, fiber editing, muscle
color/transparency, tet surface visibility, fiber visibility/transparency,
activation, skeleton display, motion browsing, cache reload, DTI/MRI overlays,
reverse-LBS waypoints, and solver/bake controls. Several controls propagate
globally across linked components so a belly/tendon counterpart does not drift
into a contradictory display state.

Tet internal-face drawing and internal stride were removed from the normal UI
because they obscured the surface/fiber view. The internal data remains used by
tet topology, quality checks, and solver construction.

## Code-level walkthrough of the bake programs

### `bake_headless.py` startup

`load_skeleton` calls the DART skeleton-info loader, builds the skeleton, and
returns body metadata. `load_skeleton_meshes` loads OBJ meshes and scales them
into the DART world convention. `load_muscle_meshes` reads a JSON manifest and
creates the named muscle objects. `load_tet_meshes` attaches the corresponding
`<name>_tet.npz` data, including simulation topology and attachments.

Before simulation, `override_anatomical_skinning`, `override_uaxis_skinning`,
and `smooth_skinning_along_axis` can repair or regularize skinning weights.
These functions distinguish origin, intermediate, and insertion bones rather
than relying only on Euclidean distance. That distinction matters for muscles
that wrap or cross the knee.

### Context construction

`build_context` collects solver policy in a context object: iteration counts,
anisotropic settings, inter-muscle threshold and weight, bone-contact margin
and weight, fascia/contact switches, fiber-spring settings, skin priors, and
unified-volume mode. The headless bake later passes this context into
`run_all_tet_sim_with_constraints` from the viewer layer.

This division keeps command-line experiment configuration separate from the
actual simulation implementation. It also explains why a wrapper can inherit
the full bake while forcing only two policy flags.

### Waypoint patching

The bake stores deformed tet positions first. `patch_waypoints` then walks the
active muscles and reconstructs waypoints using their barycentric tet data and
the frame’s deformed positions. It writes the patched values into the cache so
fiber overlays and motion-dependent anatomical controls agree with the baked
surface.

### Chunk writing

`flush_bake_data` groups accumulated frame dictionaries by muscle and writes
compressed NPZ chunks. Chunks are flushed periodically to bound memory. The
viewer’s loader expects the muscle name and vertex count to remain stable across
chunks, which is why original/contour mesh swapping must invalidate cached
draw/index state.

### Region manifests

Region JSON files are part of the simulation graph. They define which muscles
are loaded, which inter-muscle neighbors can be found, which vertices are in
the unified system, and which cache files the viewer sees. The corrected left
manifests deliberately separate Popliteus into low leg even though its source
OBJ lives under the UpLeg asset path.

### Bone-contact wrapper

`bake_headless_bone_contact.py` does not duplicate the solver. It imports
`bake_headless.main`, appends `--unified-bone-contact`, `--self-collision`, and
the fixed-iteration policy, and lets the original parser and cache code run.
This keeps the attachment and inter-muscle implementation identical to the
speed baseline while making contact policy explicit.

## Code-level reconstruction of mesh-to-fiber data

The mesh-to-fiber story can be summarized as a concrete data transformation:

```text
surface OBJ / contour streams
        ↓
best-fit plane + continuous local basis
        ↓
ordered contour levels and boundary corners
        ↓
angular or geodesic 2D parameterization
        ↓
fiber samples embedded in triangles
        ↓
barycentric/harmonic interior waypoints
        ↓
3D fiber lines and tet-associated metadata
```

The unit-circle route is useful for regular sections. The direct route avoids
unnecessary remapping when fiber samples already correspond to boundary
segments. The geodesic route handles irregular boundaries and shared streams.
Area/sign checks prevent the parameterization from flipping triangles, while
harmonic solves produce smooth interior positions from boundary constraints.

The viewer’s fiber architecture mixin caches the expensive embedding and
invalidates it when contours, cuts, mesh topology, or stream correspondence
changes. This cache invalidation is essential: stale embeddings can look like
bad anatomical fibers even when the new mesh is correct.

## Code-level reconstruction of original-mesh conversion

`tet_from_original_obj.py` loads a high-resolution OBJ, finds boundary loops,
caps required openings, and invokes TetGen. It also loads bone surfaces and
uses nearest-bone queries to classify endpoint loops. The output preserves
anchor maps, cap faces, attachment names, and mappings back to contour data.

The follow-up remesh and conversion tools repair common failures: duplicated
vertices, folded cap pairs, open/closed boundary mismatch, sliver tets,
zero-weight vertices, and orphan vertices. The viewer can then swap between
original and contour modes while retaining compatible cache indexing.

This work explains the apparent tension between “fine mesh” and “fast mesh.”
Fine original meshes improve surface detail and collision sampling, but they
increase tet count, ARAP graph size, factorization time, and cache size. Coarse
contour tets make regional experimentation practical and preserve an explicit
anatomical coordinate system. The project keeps both because they serve
different stages of research.

## Code-level reconstruction of reverse-LBS

The reverse-LBS solver reads one or more baked caches, associates tet vertices
with candidate bone transforms, and solves for local targets/weights that can
reconstruct the observed deformations. Multi-source processing combines
walk/run/combined motion sources. The XML writer serializes the result for
viewer and DART loading, while the injection and trimming tools update existing
muscle metadata.

This is the bridge from expensive simulation to compact deformation control.
It also acts as an audit: if a baked trajectory cannot be expressed sensibly in
bone-local terms, the issue may be an attachment or retargeting error rather
than a model-capacity limitation.

## What “viewer detail” means in this project

The viewer is where the project’s representations meet. A loaded muscle can
simultaneously have:

- an original OBJ surface;
- a contour surface and contour levels;
- a tet volume and render/simulation faces;
- cap and endpoint attachment metadata;
- fibers and fiber samples;
- MVC activation values;
- waypoint barycentrics and patched world positions;
- solver positions and fixed targets;
- cache positions for the current motion frame;
- counterpart/tendon links;
- DTI/MRI context overlays.

The viewer therefore has to manage ownership, invalidation, draw flags,
animation state, counterpart synchronization, cache reload, and solver state at
the same time. A “display bug” can be a stale fiber embedding, an old tet draw
array, a cache with a different vertex count, an unsynchronized tendon
counterpart, or a genuinely wrong simulation. The code contains explicit
invalidation and synchronization functions because these cases occurred in
practice.

## Explicit fiber-extraction process from the implementation

The earlier description was too compressed. The code contains several fiber
extraction families, and they should not be conflated.

### A. Bounding-plane contour correspondence

`FiberArchitectureMixin.find_contour_match` receives a muscle contour and four
bounding-plane corners. It:

1. computes the bounding-plane center;
2. builds an orthonormal basis from two plane edges and their normal;
3. projects both contour and corners into 2D;
4. casts a ray from the plane center through each corner;
5. intersects the ray with every contour edge;
6. keeps the intersection nearest the corresponding corner;
7. falls back to closest-point-on-edge matching if a ray misses;
8. inserts a vertex when an intersection lies in an edge interior;
9. sorts intersection positions around the contour;
10. rolls the contour so corner zero is the start;
11. reverses winding when corner order indicates a flip;
12. returns contour points paired with interpolated bounding-plane positions.

This is the first real fiber prerequisite. It establishes consistent corners
and stream orientation across contour levels. Without it, a later fiber can
connect the wrong side of one section to another even if every section looks
reasonable alone.

### B. Standard fiber architecture sampling

The normal architecture path stores `fiber_architecture`, `waypoints`,
`waypoints_original`, and `waypoint_bary_coords` on the muscle object. The
sampling method can be grid, Sobol unit-square, or Sobol constrained by the
minimum contour. The cutting method can be bounding-plane, area-based,
Voronoi, angular, gradient, ratio, cumulative-area, or projected-area.

`sample_fibers_angular` converts a 2D unit-circle sample into an angular
boundary bracket and a radial interpolation parameter. `compute_radial_weights`
and `find_angle_bracket` determine which boundary vertices influence the
sample. This creates an initial cross-section fiber coordinate before it is
transported through the contour levels.

### C. Unit-circle triangulation

`create_unit_circle_triangulation` constructs a 2D parameter mesh from a
boundary ring and interior rings. `UnitCircleTriangulation` stores the
boundary angles, normalized positions, and triangulation. `embed_fibers_in_triangulation`
uses `FiberTriangleEmbedding` to locate each sampled fiber point in a 2D
triangle and stores:

- the containing triangle ID;
- barycentric coordinates;
- an external/clamped status for points outside the parameter polygon.

At deformation time, `compute_waypoints_triangulated` evaluates the same
barycentric coordinates against the deformed 3D triangle vertices. The fiber
therefore follows the deformed section without needing a new 3D search every
frame.

### D. Direct triangulation and harmonic interior solve

`DirectFiberTriangulation` takes a boundary and fiber samples directly. It uses
Delaunay-style interior connectivity and avoids unnecessary remapping when the
sample coordinates already correspond to the contour. `find_waypoints_harmonic_direct`
fixes boundary positions and solves the interior scalar coordinates using the
mesh Laplacian. The resulting 2D positions are lifted back through the
bounding-plane basis into 3D waypoints.

The harmonic path is useful when fibers should smoothly fill the interior of a
section rather than be independently snapped to the nearest surface sample.

### E. Geodesic triangulation

`find_geodesic_vertex_indices` selects geodesic boundary samples or uses
provided paths. `GeodesicTriangulation` builds a polygon whose boundary follows
the muscle section rather than an idealized circle. `create_shared_geodesic_triangulation`
allows several streams to share a compatible interior parameterization.

`map_fibers_to_geodesic_polygon` converts fiber samples into the geodesic
polygon. Boundary interpolation uses angular brackets and radial distance from
the section center. This route is intended for irregular, asymmetric, or
multi-stream cross-sections where unit-circle assumptions distort the anatomy.

### F. Laplace field and Epic shape-coordinate fibers

The `epic_*` methods in `muscle_mesh.py` are a separate volume-based family.
They first require an Epic tetrahedralization and solve a scalar Laplace field
between origin and insertion. The scalar field gives a continuous anatomical
coordinate `u` through the tet volume.

`epic_show_gradient_fibers` computes per-tet scalar gradients and draws them as
direction glyphs. The glyphs are colored from red near origin to blue near
insertion using reconstructed scalar values.

`epic_sample_shape_coordinate_fibers` then:

1. rejects degenerate tetrahedra by signed-volume magnitude;
2. samples target `u` levels from 0.02 to 0.98 to avoid collapsed caps;
3. extracts iso-section points by intersecting each tet edge with each level;
4. deduplicates coincident edge intersections by rounded spatial keys;
5. computes an iso-section center at every level;
6. derives a longitudinal axis from the endpoint centers;
7. derives a stable transverse basis by SVD/PCA projection;
8. samples candidate points inside valid tets with volume-weighted random seeds;
9. converts each seed to a section direction and normalized radius;
10. transports that direction/radius to every iso-section;
11. smooths the 2D transported coordinates twice along the levels;
12. lifts the smoothed coordinates back into 3D.

Seeds are ordered by farthest-point sampling so the final fibers distribute
through the belly instead of clustering. The result is stored as
`epic_fibers`, converted to per-level waypoint arrays, and marked for fiber
rendering.

### G. Laplace-gradient streamline fibers

`epic_sample_gradient_fibers` uses the same Laplace field but traces streamlines
through tetrahedra. The implementation finds origin-region tets from low `u`
values and origin cap ownership, samples a point inside each selected tet with
exponential barycentric weights, then traces backward and forward along the
piecewise-constant tet gradient.

Tracing handles tet-face crossings, degenerate gradients, maximum segment
length, and termination at the insertion side. It resamples the combined
backward/forward path to uniform `u` levels. Failed traces are skipped. This
path is closer to a paper-style streamline extraction, while shape-coordinate
fibers deliberately preserve a cross-sectional coordinate even when raw
streamlines would bunch or jump.

### H. Fiber waypoint storage

Regardless of extraction family, the object eventually stores fibers as a
stream/list of level arrays: each level contains one 3D point per fiber. The
same positions are used for rendering, animation, barycentric embedding, MVC
updates, reverse-LBS inspection, and cache waypoint patching.

`waypoints_original` preserves the pre-simulation state. If
`waypoints_from_tet_sim` is true, tet deformation updates waypoints using
barycentric coordinates or MVC. If false, imported/reverse-LBS waypoints are
left untouched. This switch is essential when comparing baked anatomy against
solver-generated fiber motion.

### I. Barycentric embedding into tets

During soft-body initialization, existing waypoint barycentric coordinates are
loaded when available. Otherwise the code searches for a containing tet,
computes barycentric coordinates, and saves them back into the tet file for
future loads. At each deformed frame, a waypoint is reconstructed as the
barycentric combination of its tet’s current vertices.

This is the bridge between extracted fibers and a volumetric bake: the fiber
does not float independently above the tet mesh; it is attached to the volume’s
deformation field.

### J. MVC waypoint update

The alternative MVC path recomputes waypoints from deformed contour geometry.
It uses mean-value coordinates when the contour mapping is available. This is
useful when a waypoint should follow a changing boundary/contour relationship
rather than remain tied to one fixed tet. The viewer chooses between MVC and
barycentric updates with `use_mvc_waypoint_update`.

### K. Fiber extension through tendons

The linked-component code copies and/or reverses stream levels when a belly is
extended by an origin or insertion tendon. It prepares tendon waypoints from
belly waypoints, orients tendon streams to the belly, trims seam levels, and
resamples the combined source. The belly/tendon counterpart mechanism prevents
the tendon from becoming a second unrelated fiber system.

The July “belly-space contour highlight after tendon extension” work is part of
this process: the project needed to see whether the extended contour/fiber
space was anatomically continuous before baking it.

### L. Fiber lengths, ratios, and hybrid ARAP

`SoftBodySimulation._compute_fiber_lengths` sums distances through each
stream’s level points. `_compute_vertex_fiber_membership` assigns nearby
vertices to fiber neighborhoods. `compute_fiber_ratios` compares current fiber
length to original length. `solve_arap_hybrid` uses those ratios to combine
ARAP deformation with fiber/waypoint targets.

This is the code-level reason fibers influence mechanics: they can provide a
measurement of contraction and a target for a hybrid solve, rather than merely
being drawn after the simulation.

### M. Rendering and playback

`_rebuild_fiber_draw_arrays` concatenates visible stream/level points, removes
non-finite data, builds line pairs between adjacent levels, and optionally
handles epic gradient direction lines and colors. VBO helpers lazily allocate
OpenGL buffers and upload contiguous float32 arrays. Fiber draw state is marked
dirty whenever samples, contours, mesh topology, or cache positions change.

Playback can instead use cached deformation and patched waypoints. The viewer
therefore distinguishes current editable fibers, tet-simulation waypoints,
reverse-LBS waypoints, and cache-driven animation data.

### N. Why the extraction process matters to the bake

The bake does not invent fibers after the fact. It inherits a chain of choices:

```text
OBJ/contours
  → plane/corner correspondence
  → ordered levels and streams
  → 2D parameterization
  → sampled fibers
  → barycentric/MVC waypoint embedding
  → tet/ARAP deformation
  → cached fiber/waypoint playback
```

A fiber can therefore fail because of a wrong contour corner, a reversed
winding, a bad geodesic boundary, an external sample clamped into a triangle,
a missing containing tet, stale waypoint barycentrics, a tendon seam mismatch,
or a solver that deforms the volume inconsistently. “Fiber extraction” is a
complete geometric pipeline, not one preprocessing step.

## Repository-wide inventory: work beyond baking

This section records the other code families that contributed to the project,
so the recap does not imply that the work began with `bake_headless.py`.

### Motion conversion and retargeting tools

The `tools/` directory contains several generations of motion preparation:

- `retarget_bvh.py` and `retarget_to_zygote.py` handle general retargeting;
- `bake_zygote_mocap.py` routes rig styles through the Zygote pose pipeline;
- `convert_bvh_axes.py` repairs coordinate conventions;
- `dynamic_retarget.py` and `compensate_chain.py` address changing chains;
- `pos_ik_retarget.py`, `ik_arm_retarget.py`, and `bake_arm_ik.py` solve arm
  targets with IK;
- `bake_arm_direct.py`, `bake_arm_body_relative.py`,
  `bake_arm_retarget_bvh.py`, and `bake_arm_world_retarget.py` compare direct,
  local, body-relative, and world-relative arm mappings;
- `bake_forearm_retarget_bvh.py` specializes forearm behavior;
- `walk1_full_retarget.py` produces a complete motion retarget;
- `make_run_vert_bvh.py` and `find_peak_frame.py` prepare or inspect motion;
- `verify_npy_mocap.py` checks converted motion arrays and leg directions.

The retargeting work establishes the pose sequence, root motion, rest
reference, joint-name mapping, and coordinate frame consumed by every later
muscle experiment.

### Skeleton and bone preparation

`add_sternum_to_skel.py`, `add_sternum_to_bvh.py`, and sternum-related paths add
and calibrate torso structures. `skel/`, `dart/data/skel`, and
`models/skel_models_v1.1` preserve alternate skeleton/OpenSim representations.
Bone meshes are used both as kinematic visualization and as geometric collision
obstacles.

### Muscle manifests and regions

The manifests define which named muscles enter a simulation graph:

- `.muscles_L_UpLeg.json` and `.muscles_R_UpLeg.json`;
- `.muscles_L_LowLeg.json` and `.muscles_R_LowLeg.json`;
- `.muscles_UpLeg.json`, `.muscles_LowLeg.json`, and `.muscles_all_LR.json`;
- `tools/muscles_LR_UpLeg.json` and `tools/muscles_L_Leg.json`.

An asset path is not necessarily its anatomical region. Popliteus’ OBJ lives
under the UpLeg asset directory while its corrected bake region is LowLeg.
Manifests are executable model topology, not only labels.

### Tet creation and repair tools

`cache_tets.py`, `cache_one_tet.py`, `tet_from_original_obj.py`,
`convert_orig_tet.py`, and `build_contour_orig_mapping.py` create and connect
tet representations. `save_tets_with_skel_attachments.py`,
`strip_tet_for_sim.py`, `strip_orphan_tet_verts.py`, `strip_fold_faces.py`,
`regen_lowleg_tets.py`, `subdivide_tet_near_bones.py`,
`remesh_tet_for_collision.py`, `fill_lowleg_attach_skeletons.py`,
`fix_cap_attachments.py`, and `mirror_tet_L_to_R.py` repair the practical
failure modes of volumes: orphan vertices, slivers, folded faces, missing cap
records, wrong bone assignments, and incompatible indexing.

### Contour, fiber, and moment-arm tools

`check_fiber_orientation.py`, `fiber_moment_arm_analyze.py`,
`debug_rectus_connected_contour.py`, `test_lowleg_contour_tet.py`,
`build_contour_orig_mapping.py`, and `patch_25fiber_waypoints.py` connect
geometry to fiber and waypoint output. The same representation is used by
preprocessing, the solver, the viewer, and cache postprocessing.

### DTI and MRI processing

`dicom_inventory.py` inventories inputs. `convert_dicom_npz.py` converts
volumes. `build_dti_vtp_meshes.py` prepares tract/mesh assets.
`mri_dti_overlay.py` and viewer DTI/MRI code transform, resample, texture, and
display them alongside the Zygote model, allowing generic fibers to be
compared against subject-specific anatomy.

### LBS, learning, and distillation

`LBS/` contains classic LBS, dual-quaternion skinning, sampling, fiber, tet,
triangulation, and trackball experiments. `learning/` contains Ray/RL model,
PPO, configuration, and Torch-policy code. `studying/` preserves an earlier
self-contained DART/OpenSim-style environment and training path.

`volume_distill/` contains datasets, preprocessing, model architectures,
training scripts, mirror/walk/dance/four-leg overfit experiments, logs, and
architecture images. The learned path depends on trustworthy deformation
targets; it is not independent of the geometry/bake work.

### Cache, test, and communication utilities

`diff_caches.py`, `decompress_cache.py`, `merge_converted_npz.py`,
`convert_pyuipc_to_cache.py`, `fix_muscle_collision.py`,
`fix_collisions_acap.py`, and `check_bone_penetration.py` make cache outputs
inspectable and repairable. `test_emu_pose_grid.py`,
`benchmark_emu_tet_settings.py`, `test_fem_gradient.py`, `test_xpbd.py`,
`test_ipc_phase2.py`, `test_pyuipc_muscle.py`, and `bench_nn_pipeline.py`
isolate numerical and learned behavior.

`journal/Project Timeline.md`, `research_journal_mcp.py`, journal assets,
screenshots, result images, and slide generators preserve the communication
side of the work: what changed, what it looked like, and how the results were
presented.

## The viewer’s complete role

The viewer is simultaneously a skeleton/motion browser, muscle and fiber
editor, tet preparation tool, DTI/MRI comparison surface, ARAP/FEM/XPBD host,
cache writer/loader, reverse-LBS inspection tool, playback quality test, and
visual communication tool. This explains the synchronization code in
`zygote_mesh_ui.py`: group ownership, counterpart meshes, stream levels, fiber
arrays, tet draw caches, solver positions, fixed targets, motion frames, and
cache entries must agree.

## What was inspected for this recap

This expanded account is grounded in implementation structure: all top-level
bake, conversion, retarget, collision, cache, and test tools; the viewer group,
contour, fiber, DTI/MRI, cache, and unified-simulation functions; muscle mesh,
fiber architecture, contour mesh, tet rendering, FEM, ARAP backend, and bone
collision modules; original-tet and reverse-LBS conversion; DTI, LBS, learning,
volume-distillation, journal, and studying trees.
