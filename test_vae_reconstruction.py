"""Test VAE reconstruction quality"""
import torch
import sys
from pathlib import Path
import cv2
import numpy as np
import torchvision.transforms as transforms

sys.path.append(str(Path(__file__).parent))

from models.advanced_diffusion import VideoVAE3D

def load_image_as_video(image_dir, num_frames=6):
    """Load images as video tensor"""
    image_dir = Path(image_dir)
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    frames = []
    for img_path in image_files[:num_frames]:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(transform(img))

    # Stack: (C, T, H, W)
    video = torch.stack(frames, dim=1).unsqueeze(0)
    return video

def save_frames(tensor, prefix):
    """Save frames as images"""
    if tensor.dim() == 5:
        tensor = tensor.squeeze(0)

    # Denormalize
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)

    # (C, T, H, W) -> save each frame
    for t in range(tensor.shape[1]):
        frame = tensor[:, t, :, :]  # (C, H, W)
        frame_np = frame.permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"./outputs/test_{prefix}_frame_{t}.png", frame_bgr)

def test_vae(checkpoint_path):
    """Test VAE reconstruction"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create VAE
    vae = VideoVAE3D(
        in_channels=3,
        latent_channels=4,
        base_channels=128,
        channel_mults=(1, 2, 4, 4),
        temporal_downsample=(False, False, False, False),
        spatial_downsample_factor=8,
        temporal_downsample_factor=1,
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract VAE weights
    state_dict = checkpoint['model_state_dict']
    vae_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('vae.'):
            vae_state_dict[k[4:]] = v

    vae.load_state_dict(vae_state_dict)
    vae.eval()

    # Load test video
    print("Loading test video...")
    video = load_image_as_video("/workspace/data/train_videos/drive_0007", num_frames=6).to(device)
    print(f"Input shape: {video.shape}")

    # Test reconstruction
    with torch.no_grad():
        # Encode
        z, mean, logvar = vae.encode(video)
        print(f"Latent shape: {z.shape}")
        print(f"Latent stats: min={z.min().item():.3f}, max={z.max().item():.3f}, mean={z.mean().item():.3f}, std={z.std().item():.3f}")

        # Decode
        recon = vae.decode(z)
        print(f"Reconstruction shape: {recon.shape}")

    # Save frames
    Path("./outputs").mkdir(exist_ok=True)
    print("Saving frames...")
    save_frames(video, "original")
    save_frames(recon, "reconstructed")

    # Compute reconstruction error
    mse = torch.mean((video - recon) ** 2).item()
    print(f"\nMSE: {mse:.6f}")
    print(f"Saved frames to ./outputs/test_*.png")

if __name__ == "__main__":
    checkpoint_path = "/workspace/VDO_diffusion/runs/advanced_experiment/best_model.pth"
    test_vae(checkpoint_path)
