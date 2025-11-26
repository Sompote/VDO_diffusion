
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, w, h, text, color='#E0E0E0', edgecolor='black', fontsize=10, fontweight='normal'):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", 
                                  linewidth=1.5, edgecolor=edgecolor, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight=fontweight, wrap=True)
    return rect

def draw_arrow(ax, x1, y1, x2, y2, text=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))
    if text:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.02, text, ha='center', va='bottom', fontsize=9)

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    col_data = '#E3F2FD'  # Light Blue
    col_vae = '#FFF3E0'   # Light Orange
    col_diff = '#F3E5F5'  # Light Purple
    col_dit = '#E8F5E9'   # Light Green
    col_emb = '#FCE4EC'   # Light Pink

    # 1. Data Flow (Top Level)
    # Input Video
    draw_box(ax, 0.5, 8, 1.5, 1, "Input Video\n(Pixels)\n$x$", col_data)
    
    # VAE Encoder
    draw_box(ax, 2.5, 8, 1.5, 1, "VAE Encoder\n(3D CNN)", col_vae)
    draw_arrow(ax, 2.0, 8.5, 2.5, 8.5)
    
    # Latents
    draw_box(ax, 4.5, 8, 1.5, 1, "Latents\n$z$", col_data)
    draw_arrow(ax, 4.0, 8.5, 4.5, 8.5)
    
    # Forward Process
    draw_box(ax, 6.5, 8, 1.5, 1, "Forward\nDiffusion\n(Add Noise)", col_diff)
    draw_arrow(ax, 6.0, 8.5, 6.5, 8.5)
    
    # Noisy Latents
    draw_box(ax, 8.5, 8, 1.5, 1, "Noisy Latents\n$z_t$", col_data)
    draw_arrow(ax, 8.0, 8.5, 8.5, 8.5)

    # 2. DiT Architecture (Main Block)
    # Container for DiT
    dit_frame = patches.Rectangle((2, 1), 8, 6, linewidth=2, edgecolor='#2E7D32', facecolor='none', linestyle='--')
    ax.add_patch(dit_frame)
    ax.text(6, 7.2, "Latent Video DiT (Diffusion Transformer)", ha='center', fontsize=14, fontweight='bold', color='#2E7D32')

    # Inputs to DiT
    draw_arrow(ax, 9.25, 8.0, 9.25, 6.5) # z_t input
    draw_arrow(ax, 9.25, 6.5, 8.5, 6.0) # into patchify
    
    # Patchify
    draw_box(ax, 7.5, 5.5, 2.0, 0.8, "Patchify & Linear Proj\n(3D Patches)", col_dit)
    
    # Embeddings
    draw_box(ax, 2.5, 5.5, 2.0, 0.8, "Time/Class\nEmbeddings", col_emb)
    draw_box(ax, 5.0, 5.5, 2.0, 0.8, "Positional Embeddings\n(Row + Col + Time)", col_emb)
    
    # Combine
    draw_arrow(ax, 4.5, 5.9, 5.0, 5.9, "+")
    draw_arrow(ax, 7.0, 5.9, 7.5, 5.9, "+")
    
    # Transformer Blocks (Stacked)
    draw_box(ax, 4.0, 2.0, 4.0, 3.0, "", col_dit) # Block Container
    ax.text(6, 4.8, "DiT Block $\\times N$", ha='center', fontweight='bold')
    
    # Inside Block
    draw_box(ax, 4.5, 4.0, 3.0, 0.6, "AdaLN-Zero (Conditioning)", '#FFF9C4')
    draw_box(ax, 4.5, 3.2, 1.4, 0.6, "Spatial Attn\n(Intra-frame)", '#FFFFFF')
    draw_box(ax, 6.1, 3.2, 1.4, 0.6, "Temporal Attn\n(Inter-frame)", '#FFFFFF')
    draw_box(ax, 4.5, 2.2, 3.0, 0.6, "Feed Forward (MLP)", '#FFFFFF')
    
    # Arrows inside block
    draw_arrow(ax, 6.0, 4.0, 6.0, 3.8) # AdaLN -> Attn
    draw_arrow(ax, 5.2, 3.2, 5.2, 2.8) # Spatial -> FF (conceptual flow)
    draw_arrow(ax, 6.8, 3.2, 6.8, 2.8) # Temporal -> FF
    
    # Flow into Block
    draw_arrow(ax, 6.0, 5.5, 6.0, 5.0)
    
    # Output of DiT
    draw_arrow(ax, 6.0, 2.0, 6.0, 1.5)
    draw_box(ax, 4.5, 0.5, 3.0, 0.8, "Unpatchify & Output Proj\n(Predict Noise/Velocity)", col_dit)
    
    # 3. Reconstruction (Right Side)
    # Denoising Step
    draw_box(ax, 10.5, 3.5, 1.5, 1.0, "Sampler\n(DDIM/DPM)", col_diff)
    draw_arrow(ax, 7.5, 0.9, 11.25, 0.9) # Output -> Sampler path (long)
    ax.plot([7.5, 11.25, 11.25], [0.9, 0.9, 3.5], color='black', lw=1.5) # Path lines
    
    draw_arrow(ax, 11.25, 4.5, 11.25, 5.5)
    
    # Reconstructed Latents
    draw_box(ax, 10.5, 5.5, 1.5, 1.0, "Denoised Latents\n$z_0$", col_data)
    draw_arrow(ax, 11.25, 6.5, 11.25, 7.5)
    
    # VAE Decoder
    draw_box(ax, 10.5, 7.5, 1.5, 1.0, "VAE Decoder", col_vae)
    draw_arrow(ax, 11.25, 8.5, 11.25, 9.0)
    
    # Output Video
    draw_box(ax, 10.5, 9.0, 1.5, 1.0, "Generated Video\n(Pixels)", col_data, edgecolor='#1565C0', linewidth=2)

    # Titles
    plt.title("Advanced Latent Video Diffusion Transformer (DiT) Architecture", fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    plt.tight_layout()
    plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight')
    print("Successfully saved architecture_diagram.png")

if __name__ == "__main__":
    create_architecture_diagram()
