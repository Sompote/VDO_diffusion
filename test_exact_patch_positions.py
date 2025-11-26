"""Test EXACT patch positions to find the bug"""
import torch
from einops import rearrange
import numpy as np

def test_patch_positions():
    """Create a grid where each PIXEL shows its position, then verify patch ordering"""

    # Create a small test case: 8x8 image, 2x2 patches = 4x4 = 16 patches
    B, C, T, H, W = 1, 1, 1, 8, 8
    p1, p2 = 2, 2

    # Create image where each pixel value = row*W + col (unique for each pixel)
    img = torch.zeros(B, C, T, H, W)
    for i in range(H):
        for j in range(W):
            img[0, 0, 0, i, j] = i * W + j

    print("Original image (each pixel shows its position):")
    print(img[0, 0, 0].numpy().astype(int))
    print()

    # Patchify
    patches = rearrange(
        img,
        "b c t (h p1) (w p2) -> b t (h w) (p1 p2 c)",
        p1=p1,
        p2=p2,
    )

    print(f"Patches shape: {patches.shape}")
    print(f"Should be: (1, 1, 16, 4)")
    print()

    # Check each patch
    h_patches = H // p1  # 4
    w_patches = W // p2  # 4

    print("Patch contents (should show 2x2 regions):")
    for patch_idx in range(min(16, patches.shape[2])):
        patch_data = patches[0, 0, patch_idx, :].view(p1, p2, C)[:,:,0]
        expected_row = patch_idx // w_patches
        expected_col = patch_idx % w_patches
        print(f"Patch {patch_idx} (expected pos: row={expected_row}, col={expected_col}):")
        print(patch_data.numpy().astype(int))
        print()

    # Unpatchify
    h = w = int(patches.shape[2]**0.5)
    reconstructed = rearrange(
        patches,
        "b t (h w) (p1 p2 c) -> b c t (h p1) (w p2)",
        h=h,
        w=w,
        p1=p1,
        p2=p2,
        c=C,
    )

    print("Reconstructed image:")
    print(reconstructed[0, 0, 0].numpy().astype(int))
    print()

    # Check if equal
    diff = torch.abs(img - reconstructed).max()
    print(f"Max difference: {diff.item()}")

    if diff < 1e-6:
        print("✅ Perfect reconstruction!")
    else:
        print("❌ BUG FOUND!")
        print("\nDifference map:")
        print((img[0,0,0] - reconstructed[0,0,0]).numpy().astype(int))

if __name__ == "__main__":
    test_patch_positions()
