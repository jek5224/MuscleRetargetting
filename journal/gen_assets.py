#!/usr/bin/env python3
"""Generate visual assets for the project timeline slides."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict
import subprocess
import json
from datetime import datetime, timedelta

OUT = Path("/home/jek/muscle_imitation_learning_study/journal/assets")
OUT.mkdir(exist_ok=True)

# Style
BG = '#1A1A2E'
CARD = '#16213E'
ACCENT = '#6C5CE7'
HIGHLIGHT = '#00D2D3'
ORANGE = '#FDCB6E'
WHITE = '#E0E0F0'
GRAY = '#B0B0C0'
GRID = '#2A2A4E'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor': BG,
    'text.color': WHITE,
    'axes.labelcolor': WHITE,
    'xtick.color': GRAY,
    'ytick.color': GRAY,
    'axes.edgecolor': GRID,
    'grid.color': GRID,
    'font.family': 'sans-serif',
    'font.size': 14,
})


# ── 1. Pipeline Flowchart ──
def make_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 2)
    ax.axis('off')

    steps = [
        ("Scalar\nField", ACCENT),
        ("Find\nContours", ACCENT),
        ("Fill\nGaps", ACCENT),
        ("Find\nTransitions", ACCENT),
        ("Smooth", ACCENT),
        ("Cut\nStreams", HIGHLIGHT),
        ("Stream\nSmooth", HIGHLIGHT),
        ("Level\nSelect", HIGHLIGHT),
        ("Build\nFibers", ORANGE),
        ("Build\nMesh", ORANGE),
        ("Tetra-\nhedralize", ORANGE),
    ]

    for i, (label, color) in enumerate(steps):
        rect = mpatches.FancyBboxPatch((i - 0.4, -0.4), 0.8, 1.2,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='none', alpha=0.85)
        ax.add_patch(rect)
        ax.text(i, 0.2, label, ha='center', va='center', fontsize=10,
                color='white', fontweight='bold')
        if i < len(steps) - 1:
            ax.annotate('', xy=(i + 0.55, 0.2), xytext=(i + 0.45, 0.2),
                        arrowprops=dict(arrowstyle='->', color=GRAY, lw=2))

    # Legend
    for i, (label, color) in enumerate([("Pre-cut", ACCENT), ("Post-cut", HIGHLIGHT), ("Mesh", ORANGE)]):
        rect = mpatches.FancyBboxPatch((3.5 + i * 2, -0.9), 0.3, 0.3,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='none')
        ax.add_patch(rect)
        ax.text(3.5 + i * 2 + 0.5, -0.75, label, ha='left', va='center',
                fontsize=11, color=GRAY)

    fig.tight_layout()
    fig.savefig(OUT / "pipeline.png", dpi=200, bbox_inches='tight',
                facecolor=BG, transparent=False)
    plt.close()
    print("  pipeline.png")


# ── 2. NN Architecture Diagrams ──
def make_nn_diagrams():
    # V1 diagram
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-2, 4)
    ax.axis('off')
    fig.suptitle("DistillNet V1 — Per-Muscle Decoders", fontsize=20, color=WHITE, fontweight='bold', y=0.95)

    blocks_v1 = [
        (0, 1.5, "DOFs\n(4)", GRAY, 1.2, 1.5),
        (2, 1.5, "Positional\nEncoding", ACCENT, 1.5, 1.5),
        (4.5, 1.5, "Shared\nEncoder\n(3 ResBlocks)", ACCENT, 1.8, 1.5),
        (7, 1.5, "Skip\nConcat", HIGHLIGHT, 1.2, 1.5),
    ]
    decoders = [
        (9.5, 3, "Decoder₁", ORANGE, 1.5, 0.8),
        (9.5, 1.8, "Decoder₂", ORANGE, 1.5, 0.8),
        (9.5, 0.6, "Decoder₃", ORANGE, 1.5, 0.8),
        (9.5, -0.6, "  ...  ", BG, 1.5, 0.8),
    ]
    outputs = [
        (11.8, 3, "Verts₁", GRAY, 1, 0.8),
        (11.8, 1.8, "Verts₂", GRAY, 1, 0.8),
        (11.8, 0.6, "Verts₃", GRAY, 1, 0.8),
    ]

    for x, y, label, color, w, h in blocks_v1 + decoders + outputs:
        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='none', alpha=0.85)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=10,
                color='white', fontweight='bold')

    # Arrows
    arrows = [(0.6, 1.5, 1.25, 1.5), (2.75, 1.5, 3.6, 1.5), (5.4, 1.5, 6.4, 1.5),
              (7.6, 1.5, 8.75, 3), (7.6, 1.5, 8.75, 1.8), (7.6, 1.5, 8.75, 0.6),
              (10.25, 3, 11.3, 3), (10.25, 1.8, 11.3, 1.8), (10.25, 0.6, 11.3, 0.6)]
    # PE skip arrow
    arrows.append((2.75, 0.9, 6.4, 0.9))
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))

    fig.tight_layout()
    fig.savefig(OUT / "nn_v1.png", dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()

    # V2 diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-2, 4)
    ax.axis('off')
    fig.suptitle("DistillNet V2 — Shared Decoder + Muscle Embedding + PCA", fontsize=20, color=WHITE, fontweight='bold', y=0.95)

    blocks_v2 = [
        (0, 2, "DOFs\n(20)", GRAY, 1.2, 1.2),
        (2.2, 2, "Positional\nEncoding", ACCENT, 1.5, 1.2),
        (4.5, 2, "Shared\nEncoder\n(5 ResBlocks)", ACCENT, 1.8, 1.5),
        (7.2, 2, "Concat", HIGHLIGHT, 1.2, 1.2),
        (9.5, 2, "Shared\nDecoder\n(3 ResBlocks)", ORANGE, 1.8, 1.5),
        (12, 2, "PCA\nCoeffs\n(k=64)", GRAY, 1.3, 1.2),
    ]
    embed = [
        (4.5, -0.5, "Muscle\nEmbedding", HIGHLIGHT, 1.5, 1),
    ]
    linear = [
        (2.2, -0.5, "Linear\nBaseline", '#555577', 1.5, 1),
        (12, -0.5, "+", HIGHLIGHT, 0.6, 0.6),
    ]

    for x, y, label, color, w, h in blocks_v2 + embed + linear:
        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='none', alpha=0.85)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=10,
                color='white', fontweight='bold')

    arrows_v2 = [
        (0.6, 2, 1.45, 2), (2.95, 2, 3.6, 2), (5.4, 2, 6.6, 2),
        (7.8, 2, 8.6, 2), (10.4, 2, 11.35, 2),
        (4.5, 0, 6.6, 1.5),  # embedding to concat
        (0.6, 1.5, 1.45, -0.5),  # DOFs to linear baseline
        (2.95, -0.5, 11.7, -0.5),  # linear to +
        (12, -0.2, 12, 1.4),  # + to PCA
    ]
    for x1, y1, x2, y2 in arrows_v2:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))

    fig.tight_layout()
    fig.savefig(OUT / "nn_v2.png", dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print("  nn_v1.png, nn_v2.png")


# ── 3. Training Loss Curves ──
def make_training_plots():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("  SKIP training plots (no tensorboard)")
        return

    # Find the latest/longest run
    run_dirs = sorted(Path("/home/jek/muscle_imitation_learning_study/volume_distill/runs").glob("*"))
    if not run_dirs:
        run_dirs = sorted(Path("/home/jek/muscle_imitation_learning_study/volume_distill/dance/runs").glob("*"))
    if not run_dirs:
        print("  SKIP training plots (no runs)")
        return

    # Try to read the run with the most data
    best_run = None
    best_steps = 0
    for rd in run_dirs:
        try:
            ea = EventAccumulator(str(rd))
            ea.Reload()
            tags = ea.Tags().get('scalars', [])
            if tags:
                steps = len(ea.Scalars(tags[0]))
                if steps > best_steps:
                    best_steps = steps
                    best_run = (rd, ea, tags)
        except:
            continue

    if not best_run:
        print("  SKIP training plots (no valid data)")
        return

    rd, ea, tags = best_run
    fig, axes = plt.subplots(1, min(len(tags), 3), figsize=(14, 4))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for i, tag in enumerate(tags[:3]):
        data = ea.Scalars(tag)
        steps = [d.step for d in data]
        values = [d.value for d in data]
        ax = axes[i]
        ax.plot(steps, values, color=ACCENT, lw=2, alpha=0.8)
        ax.set_title(tag.replace('/', '\n'), fontsize=12, color=WHITE)
        ax.set_xlabel('Step', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(f"Training Curves — {rd.name}", fontsize=16, color=WHITE, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print("  training_curves.png")


# ── 4. Git Activity Heatmap ──
def make_git_heatmap():
    result = subprocess.run(
        ['git', 'log', '--all', '--format=%ad', '--date=short'],
        capture_output=True, text=True,
        cwd='/home/jek/muscle_imitation_learning_study'
    )
    dates = result.stdout.strip().split('\n')
    counts = defaultdict(int)
    for d in dates:
        counts[d] += 1

    # Only Claude-era dates (2026)
    claude_dates = {k: v for k, v in counts.items() if k.startswith('2026')}
    if not claude_dates:
        print("  SKIP git heatmap (no 2026 data)")
        return

    # Build calendar grid from Jan 1 to Mar 15
    start = datetime(2026, 1, 1)
    end = datetime(2026, 3, 15)
    num_days = (end - start).days + 1

    # Week grid (columns = weeks, rows = weekdays)
    grid = np.zeros((7, (num_days // 7) + 2))
    day_labels = []

    for i in range(num_days):
        d = start + timedelta(days=i)
        ds = d.strftime('%Y-%m-%d')
        week = i // 7
        dow = d.weekday()  # Mon=0
        grid[dow, week] = counts.get(ds, 0)

    fig, ax = plt.subplots(figsize=(14, 3))

    # Custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = [BG, '#1e3a5f', ACCENT, '#a29bfe', '#ffffff']
    cmap = LinearSegmentedColormap.from_list('commits', colors_list)

    im = ax.imshow(grid, cmap=cmap, aspect='auto', vmin=0,
                   vmax=max(claude_dates.values()) * 0.7)

    ax.set_yticks(range(7))
    ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=10)

    # Month labels
    month_starts = []
    for i in range(num_days):
        d = start + timedelta(days=i)
        if d.day == 1:
            month_starts.append((i // 7, d.strftime('%b')))
    ax.set_xticks([w for w, _ in month_starts])
    ax.set_xticklabels([m for _, m in month_starts], fontsize=12)

    ax.set_title("Commit Activity (2026)", fontsize=16, color=WHITE, fontweight='bold', pad=10)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Commits', fontsize=10, color=GRAY)
    cbar.ax.tick_params(colors=GRAY)

    fig.tight_layout()
    fig.savefig(OUT / "git_heatmap.png", dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print("  git_heatmap.png")


# ── 5. Code Stats ──
def make_code_stats():
    # Get lines changed per phase date range
    phases = [
        ("Contour\nPipeline", "2026-01-07", "2026-01-22", ACCENT),
        ("Rendering", "2026-01-29", "2026-01-30", HIGHLIGHT),
        ("Motion\n& FEM", "2026-02-03", "2026-02-06", ACCENT),
        ("Animation", "2026-02-09", "2026-02-12", ORANGE),
        ("Solver\nOpt", "2026-02-20", "2026-02-20", HIGHLIGHT),
        ("Neural\nNet", "2026-02-26", "2026-03-03", ACCENT),
        ("Batch\nBake", "2026-03-03", "2026-03-04", ORANGE),
        ("DOF Grid\n& NN V3", "2026-03-05", "2026-03-06", ACCENT),
    ]

    additions = []
    deletions = []
    labels = []
    colors = []

    for label, start, end, color in phases:
        result = subprocess.run(
            ['git', 'log', f'--since={start}', f'--until={end} 23:59:59',
             '--shortstat', '--format='],
            capture_output=True, text=True,
            cwd='/home/jek/muscle_imitation_learning_study'
        )
        add = 0
        dele = 0
        for line in result.stdout.strip().split('\n'):
            if 'insertion' in line:
                parts = line.strip().split(',')
                for p in parts:
                    if 'insertion' in p:
                        add += int(p.strip().split()[0])
                    elif 'deletion' in p:
                        dele += int(p.strip().split()[0])
        additions.append(add)
        deletions.append(dele)
        labels.append(label)
        colors.append(color)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    w = 0.35

    bars1 = ax.bar(x - w/2, additions, w, color=colors, alpha=0.9, label='Additions')
    bars2 = ax.bar(x + w/2, deletions, w, color=colors, alpha=0.4, label='Deletions')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Lines', fontsize=12)
    ax.set_title('Lines Changed per Phase', fontsize=18, color=WHITE, fontweight='bold', pad=15)
    ax.legend(fontsize=11, facecolor=CARD, edgecolor='none', labelcolor=WHITE)
    ax.grid(True, axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Value labels on top
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 100, f'{int(h):,}',
                    ha='center', va='bottom', fontsize=9, color=GRAY)

    fig.tight_layout()
    fig.savefig(OUT / "code_stats.png", dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print("  code_stats.png")


# ── 6. Gantt Timeline ──
def make_gantt():
    phases = [
        ("Contour Processing", "2026-01-07", "2026-01-22", ACCENT),
        ("Rendering & UI", "2026-01-29", "2026-01-30", HIGHLIGHT),
        ("Motion & FEM", "2026-02-03", "2026-02-06", ACCENT),
        ("Animation System", "2026-02-09", "2026-02-12", ORANGE),
        ("Solver Optimization", "2026-02-20", "2026-02-20", HIGHLIGHT),
        ("Neural Network", "2026-02-26", "2026-03-03", ACCENT),
        ("Batch Baking", "2026-03-03", "2026-03-04", ORANGE),
        ("DOF Grid & NN V3", "2026-03-05", "2026-03-06", ACCENT),
    ]

    fig, ax = plt.subplots(figsize=(14, 4))

    base = datetime(2026, 1, 1)
    for i, (label, start, end, color) in enumerate(phases):
        s = (datetime.strptime(start, '%Y-%m-%d') - base).days
        e = (datetime.strptime(end, '%Y-%m-%d') - base).days
        duration = max(e - s + 1, 1)
        bar = ax.barh(i, duration, left=s, height=0.6, color=color, alpha=0.85,
                      edgecolor='none')
        ax.text(s + duration + 1, i, label, va='center', fontsize=12, color=WHITE)

    # X axis as dates
    month_days = [(0, 'Jan 1'), (6, 'Jan 7'), (21, 'Jan 22'),
                  (31, 'Feb 1'), (40, 'Feb 10'), (50, 'Feb 20'),
                  (59, 'Mar 1'), (65, 'Mar 6')]
    ax.set_xticks([d for d, _ in month_days])
    ax.set_xticklabels([l for _, l in month_days], fontsize=10)
    ax.set_yticks([])
    ax.set_title('Project Timeline', fontsize=18, color=WHITE, fontweight='bold', pad=15)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(True, axis='x', alpha=0.2)
    ax.set_xlim(-2, 90)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(OUT / "gantt.png", dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print("  gantt.png")


if __name__ == '__main__':
    print("Generating assets...")
    make_pipeline_diagram()
    make_nn_diagrams()
    make_training_plots()
    make_git_heatmap()
    make_code_stats()
    make_gantt()
    print("Done!")
