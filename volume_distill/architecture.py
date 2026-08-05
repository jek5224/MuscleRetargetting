"""Generate V1Dec architecture diagram."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_architecture(hidden_dim=768, title="DistillNetV1Dec", save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(14, 18))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 18)
    ax.axis('off')
    ax.set_aspect('equal')

    # Colors
    c_input = '#E3F2FD'
    c_pe = '#FFF3E0'
    c_encoder = '#E8F5E9'
    c_res = '#F3E5F5'
    c_decoder = '#FCE4EC'
    c_output = '#FFF9C4'
    c_skip = '#B0BEC5'

    def box(x, y, w, h, text, color, fontsize=10, bold=False):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                        facecolor=color, edgecolor='#333333', linewidth=1.5)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, wrap=True)

    def arrow(x1, y1, x2, y2, color='#333333', style='->', lw=1.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw))

    def brace_arrow(x1, y1, x2, y2, label=None):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=c_skip, lw=2, linestyle='dashed'))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx - 0.3, my, label, fontsize=8, color='#666666', ha='right', va='center')

    cx = 7  # center x
    bw = 5  # box width
    bh = 0.6

    # Title
    ax.text(cx, 17.5, title, ha='center', va='center', fontsize=16, fontweight='bold')

    # Input
    y = 16.5
    box(cx - bw/2, y, bw, bh, '7 DOFs  (hip 3 + knee 1 + ankle 3)', c_input, fontsize=10, bold=True)
    arrow(cx, y, cx, y - 0.4)

    # PE
    y = 15.4
    box(cx - bw/2, y, bw, bh, f'Positional Encoding (6 freqs)\n7 → 91 dims', c_pe, fontsize=9)
    pe_out_y = y

    arrow(cx, y, cx, y - 0.4)

    # Encoder header
    y = 14.2
    ax.text(cx, y + 0.35, '── Shared Encoder ──', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#2E7D32')

    # Input proj
    y = 13.6
    box(cx - bw/2, y, bw, bh, f'Linear(91 → {hidden_dim}) + LeakyReLU', c_encoder, fontsize=9)
    arrow(cx, y, cx, y - 0.4)

    # Res blocks
    for i in range(3):
        y = 12.6 - i * 0.9
        box(cx - bw/2, y, bw, bh, f'ResBlock {i+1}: Linear({hidden_dim}→{hidden_dim}) + (Linear → LeakyReLU) × 2 + skip', c_res, fontsize=8)
        if i < 2:
            arrow(cx, y, cx, y - 0.4)

    arrow(cx, y, cx, y - 0.4)

    # Latent proj
    y = 9.6
    box(cx - bw/2, y, bw, bh, f'Linear({hidden_dim} → {hidden_dim}) + LeakyReLU', c_encoder, fontsize=9)

    # Skip connection label
    brace_arrow(cx + bw/2 + 0.3, pe_out_y + 0.3, cx + bw/2 + 0.3, 8.85, 'PE skip\nconnection')

    arrow(cx, y, cx, y - 0.5)

    # Concat
    y = 8.5
    box(cx - bw/2, y, bw, bh, f'Concat [latent, PE] → {hidden_dim} + 91 = {hidden_dim + 91}', c_skip, fontsize=9, bold=True)

    arrow(cx, y, cx, y - 0.5)

    # Decoder header
    y = 7.5
    ax.text(cx, y + 0.3, '── 37 Parallel Decoders (bmm) ──', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#C62828')

    # Decoder input proj
    y = 6.8
    box(cx - bw/2, y, bw, bh, f'Linear({hidden_dim+91} → {hidden_dim}) + LeakyReLU', c_decoder, fontsize=9)
    arrow(cx, y, cx, y - 0.4)

    # Decoder res blocks
    for i in range(2):
        y = 5.9 - i * 0.9
        box(cx - bw/2, y, bw, bh, f'ResBlock {i+1}: Linear({hidden_dim}→{hidden_dim}) + (Linear → LeakyReLU) × 2 + skip', c_res, fontsize=8)
        if i < 1:
            arrow(cx, y, cx, y - 0.4)

    arrow(cx, y, cx, y - 0.4)

    # Output proj
    y = 3.8
    box(cx - bw/2, y, bw, bh, f'Linear({hidden_dim} → V×3)  (padded to max V×3)', c_decoder, fontsize=9)
    arrow(cx, y, cx, y - 0.5)

    # Output
    y = 2.7
    box(cx - bw/2, y, bw, bh, '37 × vertex displacements (pelvis-local)', c_output, fontsize=10, bold=True)

    # Decoder bracket
    ax.add_patch(mpatches.FancyBboxPatch((cx - bw/2 - 0.6, 2.5), bw + 1.2, 5.3,
                 boxstyle="round,pad=0.2", facecolor='none', edgecolor='#C62828',
                 linewidth=2, linestyle='--'))

    # Legend
    y = 1.5
    ax.text(cx, y, f'Walk: hidden={512}, ~143M params  |  Dance: hidden={768}, ~323M params',
            ha='center', va='center', fontsize=10, color='#555555',
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#CCCCCC'))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    draw_architecture(hidden_dim=768, save_path='volume_distill/v1dec_architecture.png')
