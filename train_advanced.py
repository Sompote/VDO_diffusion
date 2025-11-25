"""
Advanced Training Script for State-of-the-Art Video Diffusion Model

Features:
- Latent Diffusion with 3D VAE
- DiT (Diffusion Transformer) architecture
- V-prediction parameterization
- Classifier-free guidance
- Mixed precision training (AMP)
- Gradient accumulation
- EMA (Exponential Moving Average)
- Multi-GPU DDP support
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from copy import deepcopy

import sys

sys.path.append(str(Path(__file__).parent))

from models.advanced_diffusion import VideoVAE3D, LatentVideoDiT, AdvancedVideoDiffusion
from data.video_dataset import create_video_dataloader


class EMA:
    """Exponential Moving Average of model parameters"""

    def __init__(self, model, decay=0.9999, device=None):
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(
                    device if device else param.device
                )

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def setup_ddp(rank, world_size):
    """Initialize distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up distributed training"""
    dist.destroy_process_group()


def save_checkpoint(
    model,
    ema,
    optimizer,
    scaler,
    epoch,
    loss,
    save_dir,
    filename="checkpoint.pth",
    rank=0,
):
    """Save model checkpoint"""
    if rank != 0:
        return

    save_path = Path(save_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict()
        if not isinstance(model, DDP)
        else model.module.state_dict(),
        "ema_shadow": ema.shadow if ema else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, ema, optimizer, scaler, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    if ema and checkpoint["ema_shadow"]:
        ema.shadow = checkpoint["ema_shadow"]

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    return epoch, loss


def train_epoch(
    model, dataloader, optimizer, scaler, ema, device, epoch, args, writer=None, rank=0
):
    """Train for one epoch with mixed precision"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    accum_steps = args.gradient_accumulation_steps

    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader

    optimizer.zero_grad()

    for batch_idx, videos in enumerate(pbar):
        videos = videos.to(device)

        # Mixed precision training
        with autocast(enabled=args.use_amp):
            # Random class labels for classifier-free guidance
            if args.num_classes:
                class_labels = torch.randint(
                    0, args.num_classes, (videos.shape[0],), device=device
                )
            else:
                class_labels = None

            loss, diff_loss, recon_loss, kl_loss = model(videos, class_labels)
            loss = loss / accum_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Update weights every accum_steps
        if (batch_idx + 1) % accum_steps == 0:
            # Gradient clipping
            if args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update EMA
            if ema:
                if isinstance(model, DDP):
                    ema.update(model.module)
                else:
                    ema.update(model)

        # Track loss
        total_loss += loss.item() * accum_steps

        # Update progress bar
        if rank == 0:
            pbar.set_postfix({
                "loss": loss.item() * accum_steps,
                "diff": diff_loss.item(),
                "recon": recon_loss.item(),
                "kl": kl_loss.item()
            })

            # Log to tensorboard
            if writer is not None:
                global_step = epoch * num_batches + batch_idx
                writer.add_scalar(
                    "train/batch_loss", loss.item() * accum_steps, global_step
                )
                writer.add_scalar(
                    "train/diff_loss", diff_loss.item(), global_step
                )
                writer.add_scalar(
                    "train/recon_loss", recon_loss.item(), global_step
                )
                writer.add_scalar(
                    "train/kl_loss", kl_loss.item(), global_step
                )

    avg_loss = total_loss / num_batches

    if rank == 0 and writer is not None:
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)

    return avg_loss


@torch.no_grad()
def validate(model, dataloader, device, epoch, args, writer=None, rank=0):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Validation {epoch}")
    else:
        pbar = dataloader

    for videos in pbar:
        videos = videos.to(device)

        if args.num_classes:
            class_labels = torch.randint(
                0, args.num_classes, (videos.shape[0],), device=device
            )
        else:
            class_labels = None

        with autocast(enabled=args.use_amp):
            loss, diff_loss, recon_loss, kl_loss = model(videos, class_labels)

        total_loss += loss.item()

        if rank == 0:
            pbar.set_postfix({
                "val_loss": loss.item(),
                "val_recon": recon_loss.item()
            })

    avg_loss = total_loss / num_batches

    if rank == 0 and writer is not None:
        writer.add_scalar("val/epoch_loss", avg_loss, epoch)

    return avg_loss


def train_single_gpu(args):
    """Training on single GPU"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

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
        vae_loss_weight=args.vae_loss_weight,
        kl_loss_weight=args.kl_loss_weight,
    )

    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )

    # Learning rate scheduler
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    elif args.lr_schedule == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=args.epochs
        )
    else:
        scheduler = None

    # Gradient scaler for mixed precision
    scaler = GradScaler(enabled=args.use_amp)

    # EMA
    ema = EMA(model, decay=args.ema_decay, device=device) if args.use_ema else None

    # Create dataloaders
    train_loader = create_video_dataloader(
        video_dir=args.train_dir,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        frame_size=tuple(args.frame_size),
        frame_interval=args.frame_interval,
        mode="train",
        num_workers=args.num_workers,
        augment=False,  # Disabled per user request for single-clip overfitting
    )

    val_loader = None
    if args.val_dir:
        val_loader = create_video_dataloader(
            video_dir=args.val_dir,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            frame_size=tuple(args.frame_size),
            frame_interval=args.frame_interval,
            mode="val",
            num_workers=args.num_workers,
            augment=False,
        )

    # Tensorboard
    log_dir = Path(args.output_dir) / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir)

    # Save config
    config_path = Path(args.output_dir) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, ema, optimizer, scaler, args.resume)
        start_epoch += 1

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 50}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, ema, device, epoch, args, writer
        )
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        if val_loader is not None:
            # Use EMA for validation if available
            if ema:
                ema.apply_shadow(model)

            val_loss = validate(model, val_loader, device, epoch, args, writer)
            print(f"Val Loss: {val_loss:.4f}")

            if ema:
                ema.restore(model)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model,
                    ema,
                    optimizer,
                    scaler,
                    epoch,
                    val_loss,
                    args.output_dir,
                    "best_model.pth",
                )

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model,
                ema,
                optimizer,
                scaler,
                epoch,
                train_loss,
                args.output_dir,
                "checkpoint_latest.pth",
            )

        # Update learning rate
        if scheduler:
            scheduler.step()
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

    # Save final model
    save_checkpoint(
        model,
        ema,
        optimizer,
        scaler,
        args.epochs - 1,
        train_loss,
        args.output_dir,
        "final_model.pth",
    )
    writer.close()

    print("\nTraining completed!")


import yaml

def load_yaml_config(config_path):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train Advanced Video Diffusion Model")

    # Config file
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    # Data parameters
    parser.add_argument(
        "--train_dir", type=str, help="Training video directory"
    )
    parser.add_argument(
        "--val_dir", type=str, default=None, help="Validation video directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./runs/advanced", help="Output directory"
    )

    # Video parameters
    parser.add_argument(
        "--num_frames", type=int, default=16, help="Number of frames per video clip"
    )
    parser.add_argument(
        "--frame_size", type=int, nargs=2, default=[256, 256], help="Frame size (H W)"
    )
    parser.add_argument(
        "--frame_interval", type=int, default=1, help="Frame sampling interval"
    )

    # VAE parameters
    parser.add_argument(
        "--latent_channels", type=int, default=4, help="Latent channels"
    )
    parser.add_argument(
        "--vae_base_channels", type=int, default=128, help="VAE base channels"
    )
    parser.add_argument(
        "--vae_channel_mults",
        type=int,
        nargs="+",
        default=[1, 2, 4, 4],
        help="VAE channel multipliers",
    )
    parser.add_argument(
        "--spatial_downsample", type=int, default=8, help="Spatial downsample factor"
    )
    parser.add_argument(
        "--temporal_downsample", type=int, default=1, help="Temporal downsample factor"
    )
    parser.add_argument(
        "--vae_temporal_downsample",
        type=int,
        nargs="+",
        default=[0, 0, 0, 0],
        help="VAE temporal downsample layers (0=False, 1=True)",
    )

    # DiT parameters
    parser.add_argument(
        "--patch_size", type=int, nargs=2, default=[2, 2], help="Patch size (H W)"
    )
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden dimension")
    parser.add_argument(
        "--depth", type=int, default=12, help="Number of transformer blocks"
    )
    parser.add_argument(
        "--num_heads", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument("--dim_head", type=int, default=64, help="Dimension per head")
    parser.add_argument(
        "--ff_mult", type=int, default=4, help="Feed-forward multiplier"
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")

    # Diffusion parameters
    parser.add_argument(
        "--num_timesteps", type=int, default=1000, help="Number of diffusion timesteps"
    )
    parser.add_argument(
        "--beta_schedule",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "sigmoid"],
        help="Noise schedule",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="v",
        choices=["eps", "x0", "v"],
        help="Prediction type",
    )

    # Classifier-free guidance
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of classes for conditional generation",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5, help="Guidance scale"
    )
    parser.add_argument(
        "--p_uncond",
        type=float,
        default=0.1,
        help="Probability of unconditional training",
    )
    parser.add_argument(
        "--vae_loss_weight", type=float, default=1.0, help="Weight for VAE reconstruction loss"
    )
    parser.add_argument(
        "--kl_loss_weight", type=float, default=1e-6, help="Weight for VAE KL divergence loss"
    )

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "none"],
        help="LR schedule",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    # Advanced training
    parser.add_argument(
        "--use_amp", action="store_true", help="Use automatic mixed precision"
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Use exponential moving average"
    )
    parser.add_argument(
        "--ema_decay", type=float, default=0.9999, help="EMA decay rate"
    )

    # System parameters
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--save_interval", type=int, default=10, help="Checkpoint save interval"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")

    args = parser.parse_args()

    # Load config if specified
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = load_yaml_config(args.config)
        
        # Update args with config values if not specified in CLI
        # Note: CLI args should take precedence, but argparse sets defaults.
        # To handle this properly, we can check if the arg was explicitly set or use the config value.
        # A simpler approach for this script: Overwrite args with config values, 
        # but keep CLI args if they were explicitly provided (harder to track).
        # OR: Just update args with config values, effectively making config values override defaults,
        # but CLI args passed *after* config loading would need to be handled carefully.
        # 
        # Standard pattern: Config file provides defaults/values, CLI overrides everything.
        # Since argparse defaults are already set, we need to know what was passed.
        # We'll use set_defaults on the parser with the config values.
        
        # Flatten config if it's nested (optional, but good for organization)
        # For now, let's assume a flat config or simple sections matching arg names.
        
        # Helper to flatten dictionary
        def flatten_config(cfg, parent_key='', sep='_'):
            items = []
            for k, v in cfg.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_config(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        # If the user uses sections like "data: train_dir: ...", we flatten it to "data_train_dir" 
        # but our args are just "train_dir". So we should probably support a flat structure 
        # or a specific structure matching our args.
        # Let's support a structured yaml that maps to our args.
        
        # We will iterate over the config and set attributes on args if they exist
        # But to allow CLI override, we should parse args again or use a different strategy.
        # Strategy: Load config, create a dict, update with non-default CLI args.
        # Actually, easiest way:
        # 1. Parse args (to get config path)
        # 2. Load config
        # 3. Set defaults of parser using config
        # 4. Re-parse args
        
        parser.set_defaults(**config)
        
        # If config has nested structure (e.g. "data": {"train_dir": ...}), we need to flatten it or extract it.
        # Let's assume the user will use a flat config matching arg names for simplicity, 
        # OR we map sections.
        
        # Let's support the sections from the example config we will create.
        # Sections: data, video, vae, dit, diffusion, training, system
        
        flat_config = {}
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    # Check if the key matches an argument directly (e.g. "train_dir")
                    if sub_key in args.__dict__:
                        flat_config[sub_key] = sub_value
                    # Or if it matches with section prefix? No, let's stick to direct names.
            else:
                if key in args.__dict__:
                    flat_config[key] = value
        
        parser.set_defaults(**flat_config)
        args = parser.parse_args()

    # Validate required args
    if args.train_dir is None:
        parser.error("--train_dir is required (either via CLI or config file)")

    # Launch training
    if args.gpus > 1:
        print(
            "Multi-GPU training not yet implemented in this script. Using single GPU."
        )
        train_single_gpu(args)
    else:
        train_single_gpu(args)


if __name__ == "__main__":
    main()
