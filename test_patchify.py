"""Test patchify/unpatchify to debug scrambled patches"""
import torch
from einops import rearrange

def test_patchify_unpatchify():
    """Test if patchify->unpatchify preserves image"""

    # Create a test image with clear pattern
    B, C, T, H, W = 1, 4, 6, 32, 32
    patch_size = (2, 2)

    # Create test tensor with gradient pattern
    x = torch.zeros(B, C, T, H, W)
    for i in range(H):
        for j in range(W):
            x[:, :, :, i, j] = (i * W + j) / (H * W)  # Gradient pattern

    print(f"Original shape: {x.shape}")
    print(f"Original values (first frame, first channel, corner): {x[0, 0, 0, :4, :4]}")

    # Patchify (same as in LatentVideoDiT)
    p1, p2 = patch_size
    patches = rearrange(
        x,
        "b c t (h p1) (w p2) -> b t (h w) (p1 p2 c)",
        p1=p1,
        p2=p2,
    )

    print(f"\nPatches shape: {patches.shape}")
    # patches should be: (B, T, num_patches, patch_dim)
    # num_patches = (H/p1) * (W/p2) = 16 * 16 = 256
    # patch_dim = p1 * p2 * C = 2 * 2 * 4 = 16

    # Unpatchify (same as in LatentVideoDiT)
    _, T_p, N, _ = patches.shape
    h = w = int(N**0.5)

    print(f"Computed h={h}, w={w}, N={N}")

    reconstructed = rearrange(
        patches,
        "b t (h w) (p1 p2 c) -> b c t (h p1) (w p2)",
        h=h,
        w=w,
        p1=p1,
        p2=p2,
        c=C,
    )

    print(f"\nReconstructed shape: {reconstructed.shape}")
    print(f"Reconstructed values (first frame, first channel, corner): {reconstructed[0, 0, 0, :4, :4]}")

    # Check if reconstruction matches original
    diff = torch.abs(x - reconstructed).max()
    print(f"\nMax difference: {diff.item()}")

    if diff < 1e-6:
        print("✅ Patchify/Unpatchify working correctly!")
    else:
        print("❌ Bug detected in patchify/unpatchify!")
        print(f"\nOriginal vs Reconstructed (first 8x8):")
        print("Original:")
        print(x[0, 0, 0, :8, :8])
        print("\nReconstructed:")
        print(reconstructed[0, 0, 0, :8, :8])

    return diff < 1e-6

if __name__ == "__main__":
    success = test_patchify_unpatchify()
    if not success:
        print("\n" + "="*70)
        print("ISSUE FOUND: Patch operations are scrambling the image!")
        print("="*70)
