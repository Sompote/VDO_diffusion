"""Test DiT output to see if it's producing scrambled latents"""
import torch
import sys
from pathlib import Path
import cv2
import numpy as np
import torchvision.transforms as transforms

sys.path.append(str(Path(__file__).parent))

from models.advanced_diffusion import VideoVAE3D, LatentVideoDiT, AdvancedVideoDiffusion

def load_and_test(checkpoint_path):
    """Test if DiT is producing coherent outputs"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    vae = VideoVAE3D(
        in_channels=3,
        latent_channels=4,
        base_channels=128,
        channel_mults=(1, 2, 4, 4),
        temporal_downsample=(False, False, False, False),
        spatial_downsample_factor=8,
        temporal_downsample_factor=1,
    )

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
    )

    model = AdvancedVideoDiffusion(
        vae=vae,
        dit=dit,
        num_timesteps=1000,
        beta_schedule="cosine",
        prediction_type="v",
        guidance_scale=1.0,
        p_uncond=0.1,
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    # Create random noise latent
    z = torch.randn(1, 4, 6, 32, 32, device=device)

    # Test DiT at t=500
    t = torch.tensor([500], device=device)

    with torch.no_grad():
        # Run DiT
        dit_output = model.dit(z, t, None)
        print(f"DiT output shape: {dit_output.shape}")
        print(f"DiT output stats: min={dit_output.min():.3f}, max={dit_output.max():.3f}, mean={dit_output.mean():.3f}, std={dit_output.std():.3f}")

        # Decode the DiT output directly (to see if DiT is producing scrambled latents)
        decoded = model.vae.decode(dit_output)
        print(f"Decoded shape: {decoded.shape}")

        # Save first frame
        Path("./outputs").mkdir(exist_ok=True)
        frame = decoded[0, :, 0, :, :]  # First frame (C, H, W)
        frame = (frame + 1.0) / 2.0
        frame = torch.clamp(frame, 0, 1)
        frame_np = frame.permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite("./outputs/dit_output_test.png", frame_bgr)
        print("Saved DiT output visualization to ./outputs/dit_output_test.png")

        # Also test with actual denoising from t=999 to t=0
        print("\n Testing full denoising...")
        z_noisy = torch.randn(1, 4, 6, 32, 32, device=device)

        # Simple single-step denoising test
        t_start = torch.tensor([999], device=device)
        pred = model.dit(z_noisy, t_start, None)

        # Convert v-prediction to x0
        if model.prediction_type == "v":
            alpha_t = model.sqrt_alphas_cumprod[t_start]
            sigma_t = model.sqrt_one_minus_alphas_cumprod[t_start]
            while len(alpha_t.shape) < len(z_noisy.shape):
                alpha_t = alpha_t[..., None]
                sigma_t = sigma_t[..., None]
            pred_x0 = alpha_t * z_noisy - sigma_t * pred
        else:
            pred_x0 = pred

        decoded_denoised = model.vae.decode(pred_x0)
        frame = decoded_denoised[0, :, 0, :, :]
        frame = (frame + 1.0) / 2.0
        frame = torch.clamp(frame, 0, 1)
        frame_np = frame.permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite("./outputs/dit_denoised_test.png", frame_bgr)
        print("Saved denoised output to ./outputs/dit_denoised_test.png")

if __name__ == "__main__":
    checkpoint_path = "/workspace/VDO_diffusion/runs/advanced_experiment/best_model.pth"
    load_and_test(checkpoint_path)
