"""Test exact patch ordering with visual pattern"""
import torch
from einops import rearrange
import cv2
import numpy as np

def create_numbered_patches():
    """Create image where each patch has a unique number"""
    B, C, T, H, W = 1, 4, 1, 32, 32
    patch_size = (2, 2)

    # Create image where each 2x2 patch shows its patch number
    x = torch.zeros(B, C, T, H, W)

    p1, p2 = patch_size
    h_patches = H // p1  # 16
    w_patches = W // p2  # 16

    for i in range(h_patches):
        for j in range(w_patches):
            patch_num = i * w_patches + j  # Patch number in row-major order
            # Fill this 2x2 patch with a value representing its number
            x[:, :, :, i*p1:(i+1)*p1, j*p2:(j+1)*p2] = patch_num / 255.0

    print(f"Created {h_patches}x{w_patches} = {h_patches*w_patches} patches")
    print(f"Original image top-left corner (should show patch numbers 0-3):")
    print(x[0, 0, 0, :6, :6] * 255)

    # Patchify
    patches = rearrange(
        x,
        "b c t (h p1) (w p2) -> b t (h w) (p1 p2 c)",
        p1=p1,
        p2=p2,
    )

    print(f"\nPatches shape: {patches.shape}")
    print(f"First 4 patches (should be 0, 1, 2, 3):")
    for i in range(4):
        print(f"  Patch {i}: mean value = {patches[0, 0, i, :].mean().item() * 255:.1f}")

    # Unpatchify
    _, T_p, N, _ = patches.shape
    h = w = int(N**0.5)
    reconstructed = rearrange(
        patches,
        "b t (h w) (p1 p2 c) -> b c t (h p1) (w p2)",
        h=h,
        w=w,
        p1=p1,
        p2=p2,
        c=C,
    )

    print(f"\nReconstructed top-left corner:")
    print(reconstructed[0, 0, 0, :6, :6] * 255)

    # Check if equal
    diff = torch.abs(x - reconstructed).max()
    print(f"\nMax difference: {diff.item()}")

    if diff < 1e-6:
        print("✅ Patch ordering is correct!")
    else:
        print("❌ BUG: Patches are being reordered!")

    # Visualize as image
    img = x[0, 0, 0, :, :].numpy() * 255
    img = img.astype(np.uint8)
    img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite("./outputs/patch_order_original.png", img_color)

    img_recon = reconstructed[0, 0, 0, :, :].numpy() * 255
    img_recon = img_recon.astype(np.uint8)
    img_recon_color = cv2.applyColorMap(img_recon, cv2.COLORMAP_JET)
    cv2.imwrite("./outputs/patch_order_reconstructed.png", img_recon_color)

    print("\nSaved visualizations to ./outputs/patch_order_*.png")

if __name__ == "__main__":
    create_numbered_patches()
