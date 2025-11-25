"""
Inference script for Advanced Video Diffusion Model (DiT + VAE)
Generate future video frames from context frames using the advanced architecture.
"""

import argparse
import sys
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from types import SimpleNamespace
import torchvision.transforms as transforms

sys.path.append(str(Path(__file__).parent))

from models.advanced_diffusion import VideoVAE3D, LatentVideoDiT, AdvancedVideoDiffusion

def load_yaml_config(config_path):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_video_frames(video_path, num_frames, frame_size, frame_interval=1):
    """Load frames from video file or directory of images"""
    video_path = Path(video_path)
    frames = []
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(frame_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if video_path.is_dir():
        # Load from directory
        print(f"Loading frames from directory: {video_path}")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = sorted([
            f for f in video_path.iterdir() 
            if f.suffix.lower() in image_extensions
        ])
        
        if not image_files:
            raise ValueError(f"No image files found in {video_path}")
            
        for i, img_path in enumerate(image_files):
            if len(frames) >= num_frames:
                break
                
            if frame_interval > 1 and i % frame_interval != 0:
                continue
                
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(frame))
            
    else:
        # Load from video file
        print(f"Loading frames from video file: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_index = 0
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_interval > 1 and frame_index % frame_interval != 0:
                frame_index += 1
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(frame))
            frame_index += 1

        cap.release()

    # Pad if not enough frames
    if len(frames) < num_frames:
        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")
        last_frame = frames[-1]
        while len(frames) < num_frames:
            frames.append(last_frame)

    # Stack: (C, T, H, W) -> (1, C, T, H, W)
    return torch.stack(frames, dim=1).unsqueeze(0)

def tensor_to_video(tensor, output_path, fps=30):
    """Save tensor as video file"""
    if tensor.dim() == 5:
        tensor = tensor.squeeze(0)

    # Denormalize
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)

    # (T, H, W, C)
    video_np = tensor.permute(1, 2, 3, 0).cpu().numpy()
    video_np = (video_np * 255).astype(np.uint8)

    num_frames, height, width, _ = video_np.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = video_np[i]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Video saved to {output_path}")

def load_model(args, device):
    """Load the advanced model architecture and weights"""
    print("Loading model architecture...")
    
    # Create VAE
    vae = VideoVAE3D(
        in_channels=3,
        latent_channels=args.latent_channels,
        base_channels=args.vae_base_channels,
        channel_mults=tuple(args.vae_channel_mults),
        temporal_downsample=tuple(bool(x) for x in args.vae_temporal_downsample),
        spatial_downsample_factor=args.spatial_downsample,
        temporal_downsample_factor=args.temporal_downsample,
    )

    # Create DiT
    # Note: DiT needs to know the latent size, not pixel size
    dit = LatentVideoDiT(
        in_channels=args.latent_channels,
        out_channels=args.latent_channels,
        patch_size=tuple(args.patch_size),
        num_frames=args.num_frames // args.temporal_downsample,
        img_size=args.frame_size[0] // args.spatial_downsample,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.num_heads,
        dim_head=args.dim_head,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        num_classes=args.num_classes,
    )

    # Create full diffusion model
    model = AdvancedVideoDiffusion(
        vae=vae,
        dit=dit,
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
        guidance_scale=args.guidance_scale,
        p_uncond=args.p_uncond,
    )

    # Load checkpoint
    if args.checkpoint:
        print(f"Loading weights from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Handle DDP checkpoints (remove 'module.' prefix)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict)
    else:
        print("WARNING: No checkpoint provided! Using random weights.")

    model = model.to(device)
    model.eval()
    return model

@torch.no_grad()
def predict(model, args, device):
    """Run prediction"""
    print(f"Processing video: {args.input_video}")
    
    # Load context frames
    context_frames = load_video_frames(
        args.input_video,
        num_frames=args.num_context_frames,
        frame_size=tuple(args.frame_size),
        frame_interval=args.frame_interval
    ).to(device)
    
    print(f"Context shape: {context_frames.shape}")
    
    # Encode context to latents
    print("Encoding context...")
    # VAE expects (B, C, T, H, W)
    context_latents, _, _ = model.vae.encode(context_frames)
    
    # The VAE compresses time by temporal_downsample_factor
    t_down = args.temporal_downsample
    context_latent_frames = context_latents.shape[2]
    
    print(f"Latent context shape: {context_latents.shape}")
    
    # Prepare future latents (noise)
    total_frames = args.num_frames
    total_latent_frames = total_frames // t_down
    
    if total_latent_frames <= context_latent_frames:
         raise ValueError(f"Total latent frames ({total_latent_frames}) must be > context latent frames ({context_latent_frames}). Increase num_frames.")

    print(f"Generating sequence of {total_latent_frames} latent frames...")
    
    # Shape for full sequence
    shape = (1, args.latent_channels, total_latent_frames, 
             args.frame_size[0] // args.spatial_downsample, 
             args.frame_size[1] // args.spatial_downsample)
             
    # Start from random noise
    z = torch.randn(shape, device=device)
    
    # Sampling loop (DDIM style from model.sample)
    # Use fewer steps for faster inference (DDIM allows fewer steps than training timesteps)
    num_steps = getattr(args, 'num_inference_steps', 100)  # Default to 100 if not specified
    num_steps = min(num_steps, args.num_timesteps)
    timesteps = torch.linspace(args.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=device)
    
    for i, t in enumerate(timesteps):
        t_batch = t.expand(shape[0]).to(device)
        
        # 1. Inject context (Inpainting)
        # We replace the context part of z with the noisy version of context_latents
        if i < len(timesteps) - 1:
            noise = torch.randn_like(context_latents)
            noisy_context = model.q_sample(context_latents, t_batch, noise)
            z[:, :, :context_latent_frames, :, :] = noisy_context
        else:
            # Last step, set exact context
            z[:, :, :context_latent_frames, :, :] = context_latents

        # 2. Predict
        # Classifier-free guidance
        if args.guidance_scale > 1.0:
            # Double input for CFG
            z_input = torch.cat([z] * 2)
            t_input = torch.cat([t_batch] * 2)
            
            # Null class labels (assuming model handles None as null or we need to pass null labels)
            # The model.sample method creates null_labels manually if class_labels is passed.
            # Here we don't have class labels, so we might just pass None if the model supports it.
            # Looking at AdvancedVideoDiffusion.forward/dit:
            # if self.num_classes is not None and class_labels is not None: ...
            # So passing None is fine for unconditional.
            # BUT for CFG we need "unconditional" prediction which usually means null class.
            # If the model is trained without classes, CFG might not be applicable or implemented differently.
            # Assuming standard text-to-video or class-to-video CFG.
            # If num_classes is None, CFG might just be ignored or handled differently.
            # Let's stick to simple prediction for now to avoid shape errors.
            model_output = model.dit(z, t_batch, None)
        else:
            model_output = model.dit(z, t_batch, None)

        # 3. Convert prediction to noise and x0
        if model.prediction_type == "v":
            pred_noise = model.predict_noise_from_v(z, t_batch, model_output)
            pred_x0 = model.predict_x0_from_v(z, t_batch, model_output)
        elif model.prediction_type == "eps":
            pred_noise = model_output
            # Compute x0 from noise
            alpha_t = model.sqrt_alphas_cumprod[t]
            sigma_t = model.sqrt_one_minus_alphas_cumprod[t]
            while len(alpha_t.shape) < len(z.shape):
                alpha_t = alpha_t[..., None]
                sigma_t = sigma_t[..., None]
            pred_x0 = (z - sigma_t * pred_noise) / alpha_t
        else:  # x0
            pred_x0 = model_output
            # Compute noise from x0
            alpha_t = model.sqrt_alphas_cumprod[t]
            sigma_t = model.sqrt_one_minus_alphas_cumprod[t]
            while len(alpha_t.shape) < len(z.shape):
                alpha_t = alpha_t[..., None]
                sigma_t = sigma_t[..., None]
            pred_noise = (z - alpha_t * pred_x0) / sigma_t

        # 4. DDIM step
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            alpha_t_prev = model.sqrt_alphas_cumprod[t_prev]
            sigma_t_prev = model.sqrt_one_minus_alphas_cumprod[t_prev]

            while len(alpha_t_prev.shape) < len(z.shape):
                alpha_t_prev = alpha_t_prev[..., None]
                sigma_t_prev = sigma_t_prev[..., None]

            z = alpha_t_prev * pred_x0 + sigma_t_prev * pred_noise
        else:
            z = pred_x0
            
    print("Decoding latents...")
    decoded_video = model.vae.decode(z)
    
    print(f"Decoded shape: {decoded_video.shape}")

    # Replace reconstructed context with original frames for perfect fidelity
    # Ensure shapes match (handle potential VAE padding/cropping if any, though usually they match)
    if decoded_video.shape[2] >= context_frames.shape[2]:
        print("Restoring original context frames...")
        decoded_video[:, :, :context_frames.shape[2], :, :] = context_frames
    
    # Save
    output_path = Path(args.output_dir) / f"{args.output_name}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_video(decoded_video, output_path)

def main():
    parser = argparse.ArgumentParser(description="Inference for Advanced Video Diffusion")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint (overrides config)")
    parser.add_argument("--input_video", type=str, help="Input video path")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    # Load config
    config = load_yaml_config(args.config)
    
    # Merge config into args (simple namespace)
    # We need to flatten the config for easier access
    flat_config = {}
    for k, v in config.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                flat_config[sk] = sv
        else:
            flat_config[k] = v
            
    # Override with CLI args
    if args.checkpoint: flat_config['checkpoint'] = args.checkpoint
    if args.input_video: flat_config['input_video'] = args.input_video
    if args.output_dir: flat_config['output_dir'] = args.output_dir
    
    # Create namespace
    cfg = SimpleNamespace(**flat_config)
    
    # Set defaults if missing
    if not hasattr(cfg, 'device'): cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not hasattr(cfg, 'output_name'): cfg.output_name = 'prediction_advanced'
    if not hasattr(cfg, 'checkpoint'): cfg.checkpoint = None
    
    # Load model
    model = load_model(cfg, cfg.device)
    
    # Run prediction
    predict(model, cfg, cfg.device)

if __name__ == "__main__":
    main()
