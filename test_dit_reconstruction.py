"""Test if DiT can reconstruct clean latents (no diffusion, t=0)"""
import torch
import sys
from pathlib import Path
import cv2
import numpy as np
import torchvision.transforms as transforms

sys.path.append(str(Path(__file__).parent))

from models.advanced_diffusion import VideoVAE3D, LatentVideoDiT, AdvancedVideoDiffusion

def load_test_image():
    """Load a test image"""
    image_dir = Path("/workspace/data/train_videos/drive_0007")
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    frames = []
    for img_path in image_files[:6]:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(transform(img))

    return torch.stack(frames, dim=1).unsqueeze(0)

def save_frame(tensor, filename):
    """Save a single frame"""
    if tensor.dim() == 5:
        tensor = tensor.squeeze(0)
    frame = tensor[:, 0, :, :]  # First frame
    frame = (frame + 1.0) / 2.0
    frame = torch.clamp(frame, 0, 1)
    frame_np = frame.permute(1, 2, 0).cpu().numpy()
    frame_np = (frame_np * 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, frame_bgr)

def test_reconstruction():
    """Test DiT reconstruction at t=0"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    vae = VideoVAE3D(
        in_channels=3, latent_channels=4, base_channels=128,
        channel_mults=(1, 2, 4, 4),
        temporal_downsample=(False, False, False, False),
        spatial_downsample_factor=8, temporal_downsample_factor=1,
    )

    dit = LatentVideoDiT(
        in_channels=4, out_channels=4, patch_size=(2, 2),
        num_frames=6, img_size=32, hidden_dim=768,
        depth=12, heads=12, dim_head=64, ff_mult=4, num_classes=None,
    )

    model = AdvancedVideoDiffusion(
        vae=vae, dit=dit, num_timesteps=1000,
        beta_schedule="cosine", prediction_type="v",
        guidance_scale=1.0, p_uncond=0.1,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load("/workspace/VDO_diffusion/runs/advanced_experiment/checkpoint_latest.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test image
    video = load_test_image().to(device)
    print(f"Input video shape: {video.shape}")

    with torch.no_grad():
        # Encode to latent
        z, _, _ = model.vae.encode(video)
        print(f"Latent shape: {z.shape}")

        # Test 1: DiT at t=0 (should output z itself or very close)
        t_zero = torch.tensor([0], device=device)
        dit_output_t0 = model.dit(z, t_zero, None)
        print(f"DiT output at t=0 shape: {dit_output_t0.shape}")

        # Decode original latent
        recon_original = model.vae.decode(z)
        save_frame(recon_original, "./outputs/test_original.png")
        print("Saved: ./outputs/test_original.png")

        # Decode DiT output at t=0
        recon_dit_t0 = model.vae.decode(dit_output_t0)
        save_frame(recon_dit_t0, "./outputs/test_dit_t0.png")
        print("Saved: ./outputs/test_dit_t0.png")

        # Check difference
        latent_diff = torch.abs(z - dit_output_t0).mean()
        print(f"\nLatent difference (z vs DiT(z, t=0)): {latent_diff.item():.6f}")

        # Test 2: DiT at t=500 (middle of diffusion)
        t_mid = torch.tensor([500], device=device)
        dit_output_t500 = model.dit(z, t_mid, None)
        recon_dit_t500 = model.vae.decode(dit_output_t500)
        save_frame(recon_dit_t500, "./outputs/test_dit_t500.png")
        print("Saved: ./outputs/test_dit_t500.png")

        print("\n" + "="*70)
        print("CHECK THE SAVED IMAGES:")
        print("- test_original.png: Should be clean road image")
        print("- test_dit_t0.png: Should be SAME as original (if DiT works)")
        print("- test_dit_t500.png: DiT output at t=500")
        print("\nIf test_dit_t0.png is SCRAMBLED, the DiT itself has a bug!")
        print("="*70)

if __name__ == "__main__":
    test_reconstruction()
