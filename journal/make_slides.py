#!/usr/bin/env python3
"""Generate project timeline slides as .pptx with visual assets."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pathlib import Path

ASSETS = Path("/home/jek/muscle_imitation_learning_study/journal/assets")

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Colors
BG_DARK = RGBColor(0x1A, 0x1A, 0x2E)
BG_CARD = RGBColor(0x16, 0x21, 0x3E)
ACCENT = RGBColor(0x53, 0x3C, 0xAB)
ACCENT_LIGHT = RGBColor(0x6C, 0x5C, 0xE7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY = RGBColor(0xB0, 0xB0, 0xC0)
HIGHLIGHT = RGBColor(0x00, 0xD2, 0xD3)
ORANGE = RGBColor(0xFD, 0xCB, 0x6E)
LIGHT_TEXT = RGBColor(0xE0, 0xE0, 0xF0)


def set_slide_bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_shape(slide, left, top, width, height, fill_color, corner_radius=None):
    if corner_radius:
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.adjustments[0] = corner_radius
    else:
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    return shape


def add_text(slide, left, top, width, height, text, font_size=18,
             color=WHITE, bold=False, alignment=PP_ALIGN.LEFT, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_bullet_list(slide, left, top, width, height, items, font_size=16,
                    color=WHITE, spacing=Pt(8)):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = "Calibri"
        p.space_after = spacing
    return txBox


def add_image(slide, path, left, top, width=None, height=None):
    if width and height:
        slide.shapes.add_picture(str(path), left, top, width, height)
    elif width:
        slide.shapes.add_picture(str(path), left, top, width=width)
    elif height:
        slide.shapes.add_picture(str(path), left, top, height=height)
    else:
        slide.shapes.add_picture(str(path), left, top)


def make_phase_slide(phase_num, title, period, items, accent_color=ACCENT_LIGHT,
                     image_path=None, image_pos=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_DARK)

    # Phase badge
    badge = add_shape(slide, Inches(0.8), Inches(0.5), Inches(1.2), Inches(0.45), accent_color, 0.3)
    p = badge.text_frame.paragraphs[0]
    p.text = f"PHASE {phase_num}"
    p.font.size = Pt(12)
    p.font.color.rgb = WHITE
    p.font.bold = True
    p.font.name = "Calibri"
    p.alignment = PP_ALIGN.CENTER
    badge.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE

    # Title
    add_text(slide, Inches(0.8), Inches(1.1), Inches(10), Inches(0.7),
             title, 36, WHITE, True)

    # Period
    add_text(slide, Inches(0.8), Inches(1.8), Inches(10), Inches(0.4),
             period, 16, GRAY)

    # Divider
    add_shape(slide, Inches(0.8), Inches(2.3), Inches(2.5), Inches(0.03), accent_color)

    if image_path:
        # Left: bullets, Right: image
        add_bullet_list(slide, Inches(0.8), Inches(2.6), Inches(5.5), Inches(4.5),
                        [f"→  {item}" for item in items], 17, LIGHT_TEXT, Pt(10))
        il, it, iw, ih = image_pos or (Inches(7), Inches(2.5), Inches(5.8), None)
        add_image(slide, image_path, il, it, iw, ih)
    else:
        add_bullet_list(slide, Inches(0.8), Inches(2.6), Inches(11), Inches(4.5),
                        [f"→  {item}" for item in items], 18, LIGHT_TEXT, Pt(12))

    return slide


# ══════════════════════════════════════════════
# SLIDE 1: Title
# ══════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_shape(slide, Inches(0), Inches(3.1), Inches(13.333), Inches(0.05), ACCENT_LIGHT)
add_text(slide, Inches(1), Inches(1.5), Inches(11), Inches(1.2),
         "Muscle Imitation Learning", 52, WHITE, True, PP_ALIGN.CENTER)
add_text(slide, Inches(1), Inches(3.4), Inches(11), Inches(0.7),
         "Project Timeline", 30, ACCENT_LIGHT, False, PP_ALIGN.CENTER)
add_text(slide, Inches(1), Inches(4.5), Inches(11), Inches(0.5),
         "989 commits  ·  24 working days  ·  January – March 2026", 18, GRAY, False, PP_ALIGN.CENTER)
# Gantt on title slide
add_image(slide, ASSETS / "gantt.png", Inches(1), Inches(5.3), Inches(11.3))

# ══════════════════════════════════════════════
# SLIDE 2: Phase 1 — Contour Processing
# ══════════════════════════════════════════════
make_phase_slide(1, "Contour Processing Pipeline", "January 7–22  ·  12 days", [
    "Scalar field contour detection",
    "Gap filling and transition detection",
    "M-to-N cutting for complex topologies",
    "Bounding plane optimization",
    "Cut → Select → Build Fibers pipeline",
    "Manual cutting window & Neck Viz",
    "Mesh stitching and tetrahedralization",
], image_path=ASSETS / "pipeline.png",
   image_pos=(Inches(6.5), Inches(2.8), Inches(6.3), None))

# ══════════════════════════════════════════════
# SLIDE 3: Phase 2 — Rendering
# ══════════════════════════════════════════════
make_phase_slide(2, "Rendering & UI Polish", "January 29–30  ·  2 days", [
    "Two-pass face culling for transparency",
    "Vertex array drawing optimization",
    "Bulk muscle UI by L/R pairs & body parts",
    "1:1 simple muscle edge case handling",
], HIGHLIGHT)

# ══════════════════════════════════════════════
# SLIDE 4: Phase 3 — Motion & FEM
# ══════════════════════════════════════════════
make_phase_slide(3, "Motion Playback & FEM Simulation", "February 3–6  ·  3 days", [
    "BVH motion browser with joint mapping",
    "Per-frame tet deformation baking",
    "Waypoint system for fiber tracking",
    "Refactored ~6,680 lines into zygote_mesh_ui.py",
    "Mesh quality: disconnected verts, gap closing",
])

# ══════════════════════════════════════════════
# SLIDE 5: Phase 4 — Animations
# ══════════════════════════════════════════════
make_phase_slide(4, "Pipeline Animation System", "February 9–12  ·  4 days", [
    "10+ animated visualization steps",
    "Compute-then-replay architecture",
    "Quaternion slerp for BP interpolation",
    "Play All with state management",
], ORANGE)

# ══════════════════════════════════════════════
# SLIDE 6: Phase 5 — Solver
# ══════════════════════════════════════════════
make_phase_slide(5, "ARAP Solver Optimization", "February 20  ·  1 day", [
    "Fused Taichi kernels & CSR gather",
    "Warm-starting from previous frame",
    "Reduced free-DOF system with scipy splu",
    "Topology caching eliminates per-frame rebuild",
], HIGHLIGHT)

# ══════════════════════════════════════════════
# SLIDE 7: Phase 6 — NN (V1)
# ══════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
badge = add_shape(slide, Inches(0.8), Inches(0.5), Inches(1.2), Inches(0.45), ACCENT_LIGHT, 0.3)
p = badge.text_frame.paragraphs[0]
p.text = "PHASE 6"
p.font.size = Pt(12)
p.font.color.rgb = WHITE
p.font.bold = True
p.font.name = "Calibri"
p.alignment = PP_ALIGN.CENTER
badge.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
add_text(slide, Inches(0.8), Inches(1.1), Inches(10), Inches(0.7),
         "Neural Network Distillation — V1", 36, WHITE, True)
add_text(slide, Inches(0.8), Inches(1.8), Inches(10), Inches(0.4),
         "February 26 – March 3  ·  Per-muscle SIREN → LeakyReLU residual blocks", 16, GRAY)
add_shape(slide, Inches(0.8), Inches(2.3), Inches(2.5), Inches(0.03), ACCENT_LIGHT)
add_bullet_list(slide, Inches(0.8), Inches(2.5), Inches(5), Inches(2),
                ["→  Separate decoder per muscle",
                 "→  4 DOF input (hip + knee)",
                 "→  Raw vertex coordinate output",
                 "→  SIREN diverged → LeakyReLU ResBlocks"],
                16, LIGHT_TEXT, Pt(8))
add_image(slide, ASSETS / "nn_v1.png", Inches(0.5), Inches(4.5), Inches(12))

# ══════════════════════════════════════════════
# SLIDE 8: Phase 6 — NN (V2)
# ══════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
badge = add_shape(slide, Inches(0.8), Inches(0.5), Inches(1.2), Inches(0.45), ACCENT_LIGHT, 0.3)
p = badge.text_frame.paragraphs[0]
p.text = "PHASE 6"
p.font.size = Pt(12)
p.font.color.rgb = WHITE
p.font.bold = True
p.font.name = "Calibri"
p.alignment = PP_ALIGN.CENTER
badge.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
add_text(slide, Inches(0.8), Inches(1.1), Inches(10), Inches(0.7),
         "Neural Network Distillation — V2", 36, WHITE, True)
add_text(slide, Inches(0.8), Inches(1.8), Inches(10), Inches(0.4),
         "Shared decoder + muscle embedding + PCA output", 16, GRAY)
add_shape(slide, Inches(0.8), Inches(2.3), Inches(2.5), Inches(0.03), ACCENT_LIGHT)
add_bullet_list(slide, Inches(0.8), Inches(2.5), Inches(5), Inches(2),
                ["→  Single shared decoder with muscle embeddings",
                 "→  20 DOF input with sliding window (W=5)",
                 "→  PCA coefficient output (k=64)",
                 "→  Linear baseline + residual learning",
                 "→  Temporal consistency loss"],
                16, LIGHT_TEXT, Pt(8))
add_image(slide, ASSETS / "nn_v2.png", Inches(0.3), Inches(4.5), Inches(12.5))

# ══════════════════════════════════════════════
# SLIDE 9: Training Curves
# ══════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_text(slide, Inches(0.8), Inches(0.5), Inches(10), Inches(0.7),
         "Training Results", 36, WHITE, True)
add_text(slide, Inches(0.8), Inches(1.2), Inches(10), Inches(0.4),
         "V2 combined dance + locomotion training", 16, GRAY)
add_shape(slide, Inches(0.8), Inches(1.7), Inches(2.5), Inches(0.03), ACCENT_LIGHT)
add_image(slide, ASSETS / "training_curves.png", Inches(0.5), Inches(2.0), Inches(12))
add_bullet_list(slide, Inches(0.8), Inches(5.2), Inches(11), Inches(2),
                ["→  Rapid convergence: reconstruction and temporal losses drop within first 100 steps",
                 "→  Cosine decay schedule, dropout 0.1, input noise augmentation (σ=0.02)",
                 "→  Real-time NN inference replaces expensive per-frame FEM simulation"],
                16, LIGHT_TEXT, Pt(8))

# ══════════════════════════════════════════════
# SLIDE 10: Phase 7 — Batch Baking
# ══════════════════════════════════════════════
make_phase_slide(7, "Batch Baking at Scale", "March 3–4  ·  2 days", [
    "82 BVH files × 4 muscle regions",
    "5 GPUs via Slurm (RTX 6000 Ada)",
    "Per-region parallel: L/R UpLeg, L/R LowLeg",
    "Sync-and-delete: 84 GB server → 115 GB output",
    "Disk space guards with auto-cancellation",
], ORANGE)

# ══════════════════════════════════════════════
# SLIDE 11: Activity & Stats
# ══════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_text(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         "Development Activity", 36, WHITE, True)
add_shape(slide, Inches(0.8), Inches(1.1), Inches(2.5), Inches(0.03), ACCENT_LIGHT)
add_image(slide, ASSETS / "git_heatmap.png", Inches(0.5), Inches(1.4), Inches(12))
add_image(slide, ASSETS / "code_stats.png", Inches(0.5), Inches(3.8), Inches(12))

# ══════════════════════════════════════════════
# SLIDE 12: Summary
# ══════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_text(slide, Inches(1), Inches(0.8), Inches(11), Inches(0.8),
         "Summary", 44, WHITE, True, PP_ALIGN.CENTER)
add_shape(slide, Inches(5.2), Inches(1.7), Inches(3), Inches(0.04), ACCENT_LIGHT)

# Stats
stats = [("989", "Commits"), ("24", "Working Days"), ("7", "Phases")]
for i, (num, label) in enumerate(stats):
    x = Inches(2.0 + i * 3.5)
    add_text(slide, x, Inches(2.2), Inches(2.5), Inches(0.9),
             num, 56, ACCENT_LIGHT, True, PP_ALIGN.CENTER)
    add_text(slide, x, Inches(3.1), Inches(2.5), Inches(0.5),
             label, 20, GRAY, False, PP_ALIGN.CENTER)

# Phase flow
flow_items = [
    "Contour Processing  →  Rendering  →  Motion & FEM Simulation",
    "Animation System  →  ARAP Solver Optimization",
    "Neural Network Distillation (V1 → V2)  →  Batch Baking at Scale",
]
add_bullet_list(slide, Inches(2), Inches(4.2), Inches(9), Inches(2),
                flow_items, 20, LIGHT_TEXT, Pt(14))

# Gantt at bottom
add_image(slide, ASSETS / "gantt.png", Inches(1), Inches(5.8), Inches(11.3))


# Save
out = "/home/jek/muscle_imitation_learning_study/journal/Project Timeline.pptx"
prs.save(out)
print(f"Saved to {out}")
