"""Complete diagnostic of all model components"""
import torch
import sys
from pathlib import Path
import cv2
import numpy as np
from einops import rearrange

sys.path.append(str(Path(__file__).parent))

from models.advanced_diffusion import VideoVAE3D, LatentVideoDiT, AdvancedVideoDiffusion
from data.video_dataset import create_video_dataloader

def save_image(tensor, filename):
    """Save tensor as image"""
    if tensor.dim() == 4:
        tensor = tensor[0]  # Remove batch
    if tensor.shape[0] == 4:  # Latent channels
        tensor = tensor[:3]  # Use first 3 channels as RGB
    frame = (tensor + 1.0) / 2.0
    frame = torch.clamp(frame, 0, 1)
    frame_np = frame.permute(1, 2, 0).cpu().numpy()
    frame_np = (frame_np * 255).astype(np.uint8)
    if frame_np.shape[2] == 3:
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = frame_np
    cv2.imwrite(filename, frame_bgr)

def test_component_by_component():
    """Test each component separately"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*80)
    print("COMPLETE MODEL DIAGNOSTIC")
    print("="*80)

    # Load real training data
    train_loader = create_video_dataloader(
        video_dir="/workspace/data/train_videos",
        batch_size=1,
        num_frames=6,
        frame_size=(256, 256),
        frame_interval=1,
        mode="train",
        num_workers=0,
        augment=False,
        use_sliding_window=False,  # Get same data each time
    )

    # Get one sample
    for video in train_loader:
        break

    video = video.to(device)
    print(f"\n1. Input Video: {video.shape}")
    save_image(video[0, :, 0, :, :], "./outputs/diag_1_input.png")
    print("   Saved: diag_1_input.png")

    # Test VAE
    vae = VideoVAE3D(
        in_channels=3, latent_channels=4, base_channels=128,
        channel_mults=(1, 2, 4, 4),
        temporal_downsample=(False, False, False, False),
        spatial_downsample_factor=8, temporal_downsample_factor=1,
    ).to(device)

    # Load VAE weights
    checkpoint = torch.load("/workspace/VDO_diffusion/runs/advanced_experiment/best_model.pth", map_location=device)
    vae_state = {k[4:]: v for k, v in checkpoint['model_state_dict'].items() if k.startswith('vae.')}
    vae.load_state_dict(vae_state)
    vae.eval()

    with torch.no_grad():
        z, _, _ = vae.encode(video)
        print(f"\n2. VAE Encode: {z.shape}")
        save_image(z[0, :, 0, :, :], "./outputs/diag_2_latent.png")
        print("   Saved: diag_2_latent.png")

        vae_recon = vae.decode(z)
        print(f"\n3. VAE Decode: {vae_recon.shape}")
        save_image(vae_recon[0, :, 0, :, :], "./outputs/diag_3_vae_recon.png")
        print("   Saved: diag_3_vae_recon.png")

        vae_error = torch.abs(video - vae_recon).mean()
        print(f"   VAE reconstruction error: {vae_error.item():.6f}")
        if vae_error < 0.1:
            print("   ✅ VAE works correctly")
        else:
            print("   ❌ VAE has issues")

    # Test DiT patchify
    dit = LatentVideoDiT(
        in_channels=4, out_channels=4, patch_size=(2, 2),
        num_frames=6, img_size=32, hidden_dim=768,
        depth=12, heads=12, dim_head=64, ff_mult=4, num_classes=None,
    ).to(device)

    # Load DiT weights
    dit_state = {k[4:]: v for k, v in checkpoint['model_state_dict'].items() if k.startswith('dit.')}
    dit.load_state_dict(dit_state)
    dit.eval()

    with torch.no_grad():
        # Test patchify
        patches = dit.to_patch_embedding(z)
        print(f"\n4. DiT Patchify: {patches.shape}")
        print(f"   Expected: (1, 6, 256, 768)")

        # Test unpatchify manually
        # First need to project back from hidden_dim to patch_dim
        B, T, N, D = patches.shape
        patch_dim = 2 * 2 * 4  # p1 * p2 * c = 16

        # Create a dummy projection (just for testing)
        patches_projected = torch.randn(B, T, N, patch_dim, device=device)

        # Unpatchify
        reconstructed_latent = dit.unpatchify(patches_projected)
        print(f"\n5. DiT Unpatchify: {reconstructed_latent.shape}")
        print(f"   Expected: (1, 4, 6, 32, 32)")

        # Test full DiT forward
        t = torch.tensor([0], device=device)
        dit_output = dit(z, t, None)
        print(f"\n6. DiT Forward (t=0): {dit_output.shape}")
        save_image(dit_output[0, :, 0, :, :], "./outputs/diag_6_dit_output_latent.png")
        print("   Saved: diag_6_dit_output_latent.png")

        # Decode DiT output
        dit_decoded = vae.decode(dit_output)
        print(f"\n7. DiT Output Decoded: {dit_decoded.shape}")
        save_image(dit_decoded[0, :, 0, :, :], "./outputs/diag_7_dit_decoded.png")
        print("   Saved: diag_7_dit_decoded.png")

        dit_latent_error = torch.abs(z - dit_output).mean()
        print(f"   DiT latent error (t=0): {dit_latent_error.item():.6f}")

        dit_pixel_error = torch.abs(video - dit_decoded).mean()
        print(f"   DiT pixel error (t=0): {dit_pixel_error.item():.6f}")

        if dit_pixel_error < 0.2:
            print("   ✅ DiT reconstruction looks OK")
        else:
            print("   ❌ DiT has significant error")

    print("\n" + "="*80)
    print("ANALYSIS:")
    print("="*80)
    print("Check the saved images:")
    print("1. diag_1_input.png - Original (should be clear)")
    print("2. diag_2_latent.png - Latent (will look abstract)")
    print("3. diag_3_vae_recon.png - VAE reconstruction (should match input)")
    print("4. diag_7_dit_decoded.png - DiT output at t=0 (CRITICAL)")
    print("")
    print("If diag_7 is SCRAMBLED, the DiT itself has a bug!")
    print("If diag_7 is CLEAR, the bug is in the diffusion sampling!")
    print("="*80)

if __name__ == "__main__":
    test_component_by_component()
