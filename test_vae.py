import torch
import cv2
import numpy as np
import argparse
import yaml
from pathlib import Path
from types import SimpleNamespace
import torchvision.transforms as transforms
from models.advanced_diffusion import VideoVAE3D, AdvancedVideoDiffusion

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(cfg, device):
    # Create VAE
    vae = VideoVAE3D(
        in_channels=3,
        latent_channels=cfg['vae']['latent_channels'],
        base_channels=cfg['vae']['vae_base_channels'],
        channel_mults=cfg['vae']['vae_channel_mults'],
        temporal_downsample=cfg['vae']['vae_temporal_downsample']
    )
    
    # Create dummy DiT (not needed for this test but required for loading weights if stored together)
    # Actually we can just load the state dict and filter for VAE keys if needed, 
    # but loading the full model is safer to match the checkpoint structure.
    from models.advanced_diffusion import LatentVideoDiT
    dit = LatentVideoDiT(
        in_channels=cfg['vae']['latent_channels'],
        hidden_dim=cfg['dit']['hidden_dim'],
        patch_size=cfg['dit']['patch_size'],
        num_frames=cfg['video']['num_frames'] // cfg['vae']['temporal_downsample'],
        depth=cfg['dit']['depth'],
        heads=cfg['dit']['num_heads'],
        dim_head=cfg['dit']['dim_head'],
        ff_mult=cfg['dit']['ff_mult'],
        dropout=cfg['dit']['dropout']
    )

    model = AdvancedVideoDiffusion(
        vae=vae,
        dit=dit,
        num_timesteps=cfg['diffusion']['num_timesteps'],
        beta_schedule=cfg['diffusion']['beta_schedule'],
        prediction_type=cfg['diffusion']['prediction_type'],
        guidance_scale=cfg['diffusion']['guidance_scale'],
        p_uncond=cfg['diffusion']['p_uncond'],
        vae_loss_weight=cfg['diffusion'].get('vae_loss_weight', 1.0),
        kl_loss_weight=cfg['diffusion'].get('kl_loss_weight', 1e-6)
    )
    
    # Load checkpoint
    checkpoint_path = cfg['inference']['checkpoint']
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

def process_video(model, video_path, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Load one frame (or a few)
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for _ in range(6): # Load 6 frames
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(frame))
    cap.release()
    
    if not frames:
        print("Could not load frames")
        return

    # Stack: (1, C, T, H, W)
    x = torch.stack(frames, dim=1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 1. Encode
        z, _, _ = model.vae.encode(x)
        
        # 2. Decode immediately
        recon = model.vae.decode(z)
        
    # Denormalize
    def save_frame(tensor, name):
        img = (tensor[0, :, 0].permute(1, 2, 0).cpu().numpy() + 1) / 2
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(name, img)
        print(f"Saved {name}")

    save_frame(x, "test_original.png")
    save_frame(recon, "test_reconstruction.png")

if __name__ == "__main__":
    parser = argparse.argument_parser()
    parser.add_argument("--config", type=str, default="predict_advanced.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    device = torch.device(cfg['inference']['device'])
    
    model = load_model(cfg, device)
    
    # Use the input video from config
    video_path = cfg['inference']['input_video']
    process_video(model, video_path, device)
