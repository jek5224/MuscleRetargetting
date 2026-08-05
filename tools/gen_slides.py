"""Generate presentation slides for the muscle distillation training pipeline."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from lxml import etree


def draw_network_architecture(path, dpi=200):
    """Network architecture diagram for DistillNetV2/V3 — clean horizontal flow."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Style
    MAIN_Y = 3.0        # main row center y
    BOT_Y = 1.0         # bottom row center y
    BH = 0.9            # box half-height
    arrow_kw = dict(arrowstyle='-|>', color='#444444', lw=2.0, mutation_scale=18,
                    shrinkA=2, shrinkB=2)
    skip_kw = dict(arrowstyle='-|>', color='#999999', lw=1.5, mutation_scale=14,
                   shrinkA=2, shrinkB=2, linestyle=(0, (5, 3)))

    def rbox(cx, cy, w, h, label, color, sub=None, fontsize=11, border=None):
        """Draw a rounded box centered at (cx, cy)."""
        ec = border or '#2a2a2a'
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                              boxstyle="round,pad=0.12", facecolor=color,
                              edgecolor=ec, linewidth=1.8)
        ax.add_patch(rect)
        ty = cy + 0.08 if sub else cy
        ax.text(cx, ty, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white', linespacing=1.3)
        if sub:
            ax.text(cx, cy - 0.3, sub, ha='center', va='center',
                    fontsize=8, color='#ffffffbb', style='italic')

    def dim_label(x, y, text):
        ax.text(x, y, text, ha='center', va='center', fontsize=8,
                color='#666666', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#cccccc', lw=0.8))

    def harrow(x1, x2, y, **kw):
        merged = {**arrow_kw, **kw}
        ax.annotate('', xy=(x2, y), xytext=(x1, y), arrowprops=merged)

    # ================================================================
    # INPUT (left, highlighted)
    # ================================================================
    inp_x = 1.0
    rbox(inp_x, MAIN_Y, 1.5, 1.6, '7 DOFs', '#3B7DD8', fontsize=13,
         sub='hip 3 · knee 1 · ankle 3', border='#2B5FA0')
    # "INPUT" label above
    ax.text(inp_x, MAIN_Y + 1.15, 'INPUT', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#3B7DD8')

    # ================================================================
    # Positional Encoding
    # ================================================================
    pe_x = 3.0
    rbox(pe_x, MAIN_Y, 1.5, 1.2, 'Positional\nEncoding', '#5B9BD5', sub='sin / cos · 6 freq')
    harrow(inp_x + 0.75, pe_x - 0.75, MAIN_Y)
    dim_label((inp_x + pe_x) / 2, MAIN_Y + 0.55, '7')

    # ================================================================
    # Shared Encoder
    # ================================================================
    enc_x = 5.3
    rbox(enc_x, MAIN_Y, 1.7, 1.2, 'Shared\nEncoder', '#4CAF81', sub='6 ResBlocks · 1024')
    harrow(pe_x + 0.75, enc_x - 0.85, MAIN_Y)
    dim_label((pe_x + enc_x) / 2, MAIN_Y + 0.55, '91')

    # ================================================================
    # Concatenate
    # ================================================================
    cat_x = 7.5
    rbox(cat_x, MAIN_Y, 0.9, 0.9, 'Cat', '#888888', fontsize=10)
    harrow(enc_x + 0.85, cat_x - 0.45, MAIN_Y)
    dim_label((enc_x + cat_x) / 2 + 0.1, MAIN_Y + 0.5, '1024')

    # PE skip connection (curved over the top)
    ax.annotate('', xy=(cat_x - 0.1, MAIN_Y + 0.45),
                xytext=(pe_x + 0.5, MAIN_Y + 0.6),
                arrowprops={**skip_kw, 'connectionstyle': 'arc3,rad=-0.35'})
    ax.text((pe_x + cat_x) / 2 + 0.3, MAIN_Y + 1.25, 'skip  (91)',
            ha='center', fontsize=8, color='#999999', style='italic')

    # ================================================================
    # Muscle Embedding (bottom row)
    # ================================================================
    emb_x = 5.3
    rbox(emb_x, BOT_Y, 1.7, 1.0, 'Muscle\nEmbedding', '#E05585', sub='37 muscles · 128-dim')
    # muscle index label
    ax.text(emb_x - 1.5, BOT_Y, 'muscle\nindex', ha='center', va='center',
            fontsize=9, color='#777777')
    harrow(emb_x - 1.1, emb_x - 0.85, BOT_Y)
    # embed → cat (vertical)
    ax.annotate('', xy=(cat_x, MAIN_Y - 0.45),
                xytext=(emb_x + 0.85, BOT_Y + 0.15),
                arrowprops={**arrow_kw, 'connectionstyle': 'arc3,rad=-0.25'})
    dim_label(cat_x - 0.6, (MAIN_Y + BOT_Y) / 2 - 0.15, '128')

    # ================================================================
    # Decoder
    # ================================================================
    dec_x = 9.3
    rbox(dec_x, MAIN_Y, 1.5, 1.2, 'Decoder', '#ED8A3B', sub='4 ResBlocks · 1024')
    harrow(cat_x + 0.45, dec_x - 0.75, MAIN_Y)
    dim_label((cat_x + dec_x) / 2, MAIN_Y + 0.55, '1243')

    # ================================================================
    # Linear Baseline (bottom row, right)
    # ================================================================
    lin_x = 9.3
    rbox(lin_x, BOT_Y, 1.5, 0.9, 'Linear\nBaseline', '#777777', sub='DOFs + embed', fontsize=10)

    # ================================================================
    # Add node (+)
    # ================================================================
    add_x = 11.0
    add_y = (MAIN_Y + BOT_Y) / 2
    circle = plt.Circle((add_x, add_y), 0.35, facecolor='#f0f0f0',
                        edgecolor='#444444', lw=2.0, zorder=5)
    ax.add_patch(circle)
    ax.text(add_x, add_y, '+', ha='center', va='center',
            fontsize=18, fontweight='bold', color='#444444', zorder=6)
    # decoder → +
    harrow(dec_x + 0.75, add_x - 0.15, MAIN_Y,
           connectionstyle='arc3,rad=0.2')
    # baseline → +
    harrow(lin_x + 0.75, add_x - 0.15, BOT_Y,
           connectionstyle='arc3,rad=-0.2')

    # ================================================================
    # OUTPUT (right, highlighted)
    # ================================================================
    out_x = 12.2
    rbox(out_x, add_y, 1.3, 1.4, 'PCA\nCoeffs', '#8E5CB8', fontsize=13,
         sub='K = 64', border='#6B3F96')
    harrow(add_x + 0.35, out_x - 0.65, add_y)
    dim_label((add_x + out_x) / 2 + 0.05, add_y + 0.45, '64')
    # "OUTPUT" label above
    ax.text(out_x, add_y + 1.05, 'OUTPUT', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#8E5CB8')

    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)


def _draw_pipeline(path, title, first_box_label, first_box_color, sub_labels, dpi=200):
    """Generic 4-box pipeline diagram."""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.0, 2.8)
    ax.set_aspect('equal')
    ax.axis('off')

    data_color = '#50B86C'
    nn_color = '#E8833A'

    boxes = [
        (0.0, 0.5, 1.8, 1.2, first_box_label, first_box_color),
        (2.5, 0.5, 1.8, 1.2, 'FEM\nSimulation', data_color),
        (5.0, 0.5, 1.8, 1.2, 'Training\nDataset', data_color),
        (7.5, 0.5, 1.8, 1.2, 'Neural\nNetwork', nn_color),
    ]

    for x, y, w, h, label, color in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='#333333', linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white', linespacing=1.4)

    for x, y, text, fs in sub_labels:
        ax.text(x, y, text, ha='center', va='top', fontsize=fs, color='#555555',
                linespacing=1.3)

    arrow_style = dict(arrowstyle='->', color='#333333', lw=2, mutation_scale=15)
    for i in range(3):
        x_start = boxes[i][0] + boxes[i][2]
        x_end = boxes[i+1][0]
        y_mid = boxes[i][1] + boxes[i][3] / 2
        ax.annotate('', xy=(x_end, y_mid), xytext=(x_start, y_mid),
                    arrowprops=arrow_style)

    ax.text(5.0, 2.5, title, ha='center', va='center',
            fontsize=14, fontweight='bold', color='#222222')

    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)


def draw_pipeline_v1(path, dpi=200):
    _draw_pipeline(path,
        title='Training Pipeline: Single BVH Motion',
        first_box_label='Single\nBVH',
        first_box_color='#7B68AE',
        sub_labels=[
            (0.9, 0.2, '1 dance clip\n(~3K frames)', 8),
            (3.4, 0.2, 'ARAP solver\nper-frame', 8),
            (5.9, 0.2, 'DOFs → vertex\ndisplacements', 8),
            (8.4, 0.2, 'per-muscle decoder\n(direct output)', 8),
        ], dpi=dpi)


def draw_pipeline_v2(path, dpi=200):
    _draw_pipeline(path,
        title='Training Pipeline: BVH Motion Dataset',
        first_box_label='BVH\nMotions',
        first_box_color='#4A90D9',
        sub_labels=[
            (0.9, 0.2, '82 diverse clips\n(walk, dance, run, ...)', 8),
            (3.4, 0.2, 'ARAP solver\nper-frame', 8),
            (5.9, 0.2, 'DOFs → vertex\ndisplacements', 8),
            (8.4, 0.2, 'shared encoder\n+ PCA decode', 8),
        ], dpi=dpi)


def draw_pipeline_v3(path, dpi=200):
    _draw_pipeline(path,
        title='Training Pipeline: DOF Grid Sampling Dataset',
        first_box_label='DOF Grid\nSampling',
        first_box_color='#D94A7A',
        sub_labels=[
            (0.9, 0.2, '7-DOF Latin Hypercube\n(hip 3, knee 1, ankle 3)', 8),
            (3.4, 0.2, 'ARAP solver\nper-sample', 8),
            (5.9, 0.2, 'DOFs → vertex\ndisplacements', 8),
            (8.4, 0.2, 'shared encoder\n+ PCA decode', 8),
        ], dpi=dpi)


def _add_slide(prs, title_text, img_path, bullets, slide_w):
    """Add a slide with title, pipeline diagram, and bullet points."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

    # Title
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.5), Inches(1.0))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    p.alignment = PP_ALIGN.LEFT

    # Pipeline diagram
    img_w = Inches(9.5)
    img_h = Inches(3.0)
    left = (slide_w - img_w) // 2
    slide.shapes.add_picture(img_path, left, Inches(1.8), img_w, img_h)

    # Bullet points
    txBox2 = slide.shapes.add_textbox(Inches(1.2), Inches(5.0), Inches(10.5), Inches(2.2))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    for i, text in enumerate(bullets):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.text = text
        p.font.size = Pt(16)
        p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.space_before = Pt(6)
        p.level = 0
        pPr = p._pPr
        if pPr is None:
            pPr = p._p.get_or_add_pPr()
        buChar = etree.SubElement(pPr, qn('a:buChar'))
        buChar.set('char', '\u2022')

    return slide


def build_pptx(out_path):
    """Build a 3-slide presentation."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide_w = prs.slide_width

    # Generate diagram images
    img_v1 = '/tmp/pipeline_v1.png'
    img_v2 = '/tmp/pipeline_v2.png'
    img_v3 = '/tmp/pipeline_v3.png'
    img_arch = '/tmp/network_arch.png'
    draw_pipeline_v1(img_v1)
    draw_pipeline_v2(img_v2)
    draw_pipeline_v3(img_v3)
    draw_network_architecture(img_arch)

    # Slide 1: V1
    _add_slide(prs, "V1: Training from Single BVH Motion", img_v1, [
        "Single dance BVH clip (~3K frames) as training data",
        "FEM (ARAP) simulation bakes per-frame vertex displacements",
        "Input: 4-dim DOFs + velocity (hip 3 + knee 1)",
        "Per-muscle decoder outputs vertex displacements directly (no PCA)",
    ], slide_w)

    # Slide 2: V2
    _add_slide(prs, "V2: Training from BVH Motion Dataset", img_v2, [
        "82 BVH clips (walk, run, dance, jump, ...) provide diverse motion coverage",
        "FEM (ARAP) simulation bakes per-frame vertex displacements for each muscle",
        "Input: 20-dim feature vector (joint angles + derivatives over sliding window)",
        "Shared encoder + muscle embeddings, PCA output basis, temporal consistency loss",
    ], slide_w)

    # Slide 3: Network Architecture
    slide_arch = prs.slides.add_slide(prs.slide_layouts[6])
    txBox = slide_arch.shapes.add_textbox(Inches(0.8), Inches(0.3), Inches(11.5), Inches(0.8))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Network Architecture"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    p.alignment = PP_ALIGN.LEFT
    img_w_arch = Inches(10.5)
    img_h_arch = Inches(4.8)
    left_arch = (slide_w - img_w_arch) // 2
    slide_arch.shapes.add_picture(img_arch, left_arch, Inches(1.2), img_w_arch, img_h_arch)
    # Specs below diagram
    txBox2 = slide_arch.shapes.add_textbox(Inches(1.2), Inches(6.1), Inches(10.5), Inches(1.2))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    arch_bullets = [
        "Shared encoder (6 ResBlocks, 1024-dim) + single decoder conditioned on muscle embedding",
        "Linear baseline + residual: stable training with skip connection from input",
        "~24M parameters, real-time inference at 365 FPS on GPU",
    ]
    for i, text in enumerate(arch_bullets):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.text = text
        p.font.size = Pt(16)
        p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.space_before = Pt(6)
        pPr = p._pPr
        if pPr is None:
            pPr = p._p.get_or_add_pPr()
        buChar = etree.SubElement(pPr, qn('a:buChar'))
        buChar.set('char', '\u2022')

    # Slide 4: V3
    _add_slide(prs, "V3: Training from DOF Grid Sampling Dataset", img_v3, [
        "Systematic 7-DOF sampling via Latin Hypercube (hip 3 + knee 1 + ankle 3)",
        "30K samples covering full ROM — no motion clip dependency",
        "Input: 7 raw joint angles (deterministic mapping, no temporal features)",
        "Constraint losses: fixed vertex penalty + inter-muscle distance consistency",
    ], slide_w)

    prs.save(out_path)
    print(f"Saved presentation to {out_path}")


if __name__ == '__main__':
    out = os.path.join(os.path.dirname(__file__), '..', 'slides', 'training_pipeline.pptx')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    build_pptx(out)
