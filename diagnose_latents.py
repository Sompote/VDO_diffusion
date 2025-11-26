
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from models.advanced_diffusion import VideoVAE3D

def load_vae(checkpoint_path):
    print(f"Loading VAE from {checkpoint_path}")
    # We need to match the config used during training.
    # Assuming default config for now or inspecting the checkpoint if possible.
    # For safety, let's try to load the state dict and infer or use default args.
    
    vae = VideoVAE3D(
        in_channels=3,
        latent_channels=4,
        base_channels=128,
        channel_mults=(1, 2, 4, 4),
        temporal_downsample=(False, False, False, False),
        spatial_downsample_factor=8,
        temporal_downsample_factor=1,
    )
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return None

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle 'model_state_dict' key if present (from train_advanced.py save)
    # Or 'state_dict' or raw dict (from train_vae_only.py)
    if 'vae_state_dict' in checkpoint:
        vae_dict = checkpoint['vae_state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # If it was saved as a full model (AdvancedVideoDiffusion), the VAE keys might be prefixed with 'vae.'
        vae_dict = {k.replace('vae.', ''): v for k, v in state_dict.items() if k.startswith('vae.')}
        if not vae_dict:
             # Maybe it was just the VAE saved directly?
             vae_dict = state_dict
    elif 'state_dict' in checkpoint:
         state_dict = checkpoint['state_dict']
         vae_dict = {k.replace('vae.', ''): v for k, v in state_dict.items() if k.startswith('vae.')}
         if not vae_dict: vae_dict = state_dict
    else:
        # Assume raw state dict
        vae_dict = checkpoint

    # Clean up DDP prefix
    vae_dict = {k.replace('module.', ''): v for k, v in vae_dict.items()}
    
    try:
        vae.load_state_dict(vae_dict)
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Keys in checkpoint:", list(vae_dict.keys())[:5])
        return None
        
    return vae

def create_dummy_video():
    # Create a gradient video (C, T, H, W)
    T, H, W = 16, 256, 256
    video = torch.zeros(1, 3, T, H, W)
    for t in range(T):
        for h in range(H):
            for w in range(W):
                video[0, 0, t, h, w] = t / T
                video[0, 1, t, h, w] = h / H
                video[0, 2, t, h, w] = w / W
    
    # Normalize to [-1, 1]
    video = (video - 0.5) / 0.5
    return video

def main():
    checkpoint_path = "vae_pretrained.pth"
    vae = load_vae(checkpoint_path)
    if vae is None:
        print("Skipping diagnostic due to missing VAE.")
        return

    vae.eval()
    
    video = create_dummy_video()
    print(f"Input video stats: Mean={video.mean():.4f}, Std={video.std():.4f}, Min={video.min():.4f}, Max={video.max():.4f}")
    
    with torch.no_grad():
        z, mean, logvar = vae.encode(video)
        recon = vae.decode(z)
        
    print("\n=== Latent Space Statistics ===")
    print(f"Shape: {z.shape}")
    print(f"Mean:  {z.mean().item():.6f}")
    print(f"Std:   {z.std().item():.6f}")
    print(f"Min:   {z.min().item():.6f}")
    print(f"Max:   {z.max().item():.6f}")
    
    print("\n=== Reconstruction Statistics ===")
    mse = torch.nn.functional.mse_loss(recon, video).item()
    print(f"MSE Loss: {mse:.6f}")
    print(f"Recon Mean: {recon.mean().item():.4f}")
    
    # Save comparison image
    # Take first frame of first batch
    orig_frame = video[0, :, 0, :, :].cpu()
    recon_frame = recon[0, :, 0, :, :].detach().cpu()
    
    # Denormalize
    orig_frame = (orig_frame + 1) / 2
    recon_frame = (recon_frame + 1) / 2
    
    # Clamp
    orig_frame = torch.clamp(orig_frame, 0, 1)
    recon_frame = torch.clamp(recon_frame, 0, 1)
    
    # Make grid
    from torchvision.utils import save_image
    save_path = "vae_diagnostic.png"
    save_image([orig_frame, recon_frame], save_path)
    print(f"\nSaved diagnostic image to {save_path} (Left: Original, Right: Reconstruction)")

    std_z = z.std().item()
    if abs(std_z - 1.0) > 0.2:
        print("\n⚠️ WARNING: Latent standard deviation is far from 1.0!")
        scale_factor = 1.0 / std_z
        print(f"Recommendation: Set a scaling factor of approx {scale_factor:.4f} in your diffusion config.")
    else:
        print("\n✅ Latent statistics look good (close to standard normal).")

if __name__ == "__main__":
    main()
