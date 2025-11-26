"""Diagnose where patches get scrambled"""
import torch
import sys
from pathlib import Path
import cv2
import numpy as np
from einops import rearrange

sys.path.append(str(Path(__file__).parent))

from models.advanced_diffusion import VideoVAE3D, LatentVideoDiT

def create_checkerboard_latent():
    """Create a checkerboard pattern in latent space"""
    B, C, T, H, W = 1, 4, 6, 32, 32

    # Create checkerboard in latent space
    z = torch.zeros(B, C, T, H, W)
    for i in range(H):
        for j in range(W):
            if (i // 2 + j // 2) % 2 == 0:
                z[:, :, :, i, j] = 1.0
            else:
                z[:, :, :, i, j] = -1.0

    return z

def test_dit_patchify_unpatchify():
    """Test if DiT's patchify/unpatchify preserves spatial structure"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create DiT
    dit = LatentVideoDiT(
        in_channels=4,
        out_channels=4,
        patch_size=(2, 2),
        num_frames=6,
        img_size=32,
        hidden_dim=768,
        depth=12,
        heads=12,
        dim_head=64,
        ff_mult=4,
        dropout=0.0,
        num_classes=None,
    ).to(device)

    # Create checkerboard input
    z = create_checkerboard_latent().to(device)
    print(f"Input shape: {z.shape}")
    print(f"Input checkerboard pattern (top-left 8x8, first frame):")
    print(z[0, 0, 0, :8, :8].cpu())

    # Step 1: Patchify (using DiT's to_patch_embedding)
    patches = dit.to_patch_embedding(z)
    print(f"\nPatches shape after patchify: {patches.shape}")

    # Step 2: Unpatchify
    reconstructed = dit.unpatchify(patches)
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Reconstructed checkerboard pattern (top-left 8x8):")
    print(reconstructed[0, 0, 0, :8, :8].cpu())

    # Check difference
    diff = torch.abs(z - reconstructed).max()
    print(f"\nMax difference: {diff.item()}")

    if diff < 1e-5:
        print("✅ DiT patchify/unpatchify preserves structure!")
    else:
        print("❌ BUG: DiT patchify/unpatchify scrambles structure!")

        # Save visualizations
        Path("./outputs").mkdir(exist_ok=True)

        # Original
        img_orig = ((z[0, 0, 0, :, :].cpu() + 1) / 2 * 255).numpy().astype(np.uint8)
        cv2.imwrite("./outputs/checkerboard_original.png", img_orig)

        # Reconstructed
        img_recon = ((reconstructed[0, 0, 0, :, :].cpu() + 1) / 2 * 255).numpy().astype(np.uint8)
        cv2.imwrite("./outputs/checkerboard_reconstructed.png", img_recon)

        print("Saved visualizations to ./outputs/checkerboard_*.png")

if __name__ == "__main__":
    test_dit_patchify_unpatchify()
