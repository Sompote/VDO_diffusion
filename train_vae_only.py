"""
Stage 1: Pre-train VAE for video reconstruction
This creates a good latent space before training the diffusion model.
"""
import argparse
import sys
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from types import SimpleNamespace

sys.path.append(str(Path(__file__).parent))

from models.advanced_diffusion import VideoVAE3D
from data.video_dataset import create_video_dataloader


def train_vae(vae, dataloader, optimizer, device, epoch):
    """Train VAE for one epoch"""
    vae.train()
    total_recon_loss = 0.0
    total_kl_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for videos in pbar:
        videos = videos.to(device)
        optimizer.zero_grad()

        # Encode
        z, mean, logvar = vae.encode(videos)

        # Decode
        recon = vae.decode(z)

        # Reconstruction loss
        recon_loss = F.mse_loss(recon, videos)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss = kl_loss / (videos.shape[0] * videos.shape[1] * videos.shape[2] * videos.shape[3] * videos.shape[4])

        # Total loss
        loss = recon_loss + 1e-6 * kl_loss

        loss.backward()
        optimizer.step()

        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

        pbar.set_postfix({
            'recon': recon_loss.item(),
            'kl': kl_loss.item()
        })

    avg_recon = total_recon_loss / len(dataloader)
    avg_kl = total_kl_loss / len(dataloader)

    return avg_recon, avg_kl


@torch.no_grad()
def validate_vae(vae, dataloader, device):
    """Validate VAE"""
    vae.eval()
    total_loss = 0.0

    for videos in dataloader:
        videos = videos.to(device)

        # Encode and decode
        z, _, _ = vae.encode(videos)
        recon = vae.decode(z)

        # Reconstruction loss
        loss = F.mse_loss(recon, videos)
        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Pre-train VAE")
    parser.add_argument("--config", type=str, default="config_advanced.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train VAE")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output", type=str, default="vae_pretrained.pth", help="Output checkpoint name")

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Convert to namespace for easy access
    cfg = SimpleNamespace(**{
        **config.get('data', {}),
        **config.get('video', {}),
        **config.get('vae', {}),
        **config.get('training', {}),
    })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create VAE
    print("Creating VAE...")
    vae = VideoVAE3D(
        in_channels=3,
        latent_channels=cfg.latent_channels,
        base_channels=cfg.vae_base_channels,
        channel_mults=tuple(cfg.vae_channel_mults),
        temporal_downsample=tuple(bool(x) for x in cfg.vae_temporal_downsample),
        spatial_downsample_factor=cfg.spatial_downsample,
        temporal_downsample_factor=cfg.temporal_downsample,
    ).to(device)

    num_params = sum(p.numel() for p in vae.parameters())
    print(f"VAE parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # Create dataloaders
    print("Loading data...")
    train_loader = create_video_dataloader(
        video_dir=cfg.train_dir,
        batch_size=cfg.batch_size,
        num_frames=cfg.num_frames,
        frame_size=tuple(cfg.frame_size),
        frame_interval=cfg.frame_interval,
        mode="train",
        num_workers=getattr(config.get('system', {}), 'num_workers', 4),
        augment=True,
        use_sliding_window=True,
        window_stride=1,
    )

    val_loader = create_video_dataloader(
        video_dir=cfg.val_dir,
        batch_size=1,
        num_frames=cfg.num_frames,
        frame_size=tuple(cfg.frame_size),
        frame_interval=cfg.frame_interval,
        mode="val",
        num_workers=0,
        augment=False,
        use_sliding_window=False,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=0.01)

    # Training loop
    best_val_loss = float('inf')

    print(f"\n{'='*60}")
    print("STAGE 1: Pre-training VAE")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        # Train
        train_recon, train_kl = train_vae(vae, train_loader, optimizer, device, epoch)

        # Validate
        val_loss = validate_vae(vae, val_loader, device)

        print(f"Epoch {epoch}: train_recon={train_recon:.6f}, train_kl={train_kl:.6f}, val_loss={val_loss:.6f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'vae_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            torch.save(checkpoint, args.output)
            print(f"✓ Saved best VAE checkpoint (val_loss={val_loss:.6f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'vae_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            torch.save(checkpoint, f"vae_epoch_{epoch+1}.pth")

    print(f"\n{'='*60}")
    print("VAE pre-training complete!")
    print(f"Best VAE saved to: {args.output}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"{'='*60}")
    print("\nNext step: Train DiT with frozen VAE")
    print(f"  python train_advanced.py --config config_advanced.yaml --vae_checkpoint {args.output}")


if __name__ == "__main__":
    main()
