# Two-Stage Training Guide for Video Diffusion Model

## Overview

Latent diffusion models require **two-stage training**:

1. **Stage 1**: Pre-train VAE (learns good video compression)
2. **Stage 2**: Train DiT with frozen VAE (learns denoising in stable latent space)

Training them simultaneously causes a "moving target" problem that produces scrambled outputs.

## Stage 1: Pre-train the VAE

The VAE learns to compress videos into a latent space and reconstruct them.

### Run VAE Pre-training

```bash
python train_vae_only.py --epochs 100 --output vae_pretrained.pth
```

**Parameters:**
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--output`: Output checkpoint filename (default: vae_pretrained.pth)

**What to expect:**
- Training time: ~2-4 hours (depending on dataset size and GPU)
- Validation loss should decrease to < 0.02
- The VAE learns to reconstruct videos with minimal distortion

**Monitor:**
- `recon`: Reconstruction loss (should decrease steadily)
- `kl`: KL divergence (regularization, should be small)
- `val_loss`: Validation reconstruction loss

### Verify VAE Quality

Test the pre-trained VAE:

```bash
python -c "
import torch
from models.advanced_diffusion import VideoVAE3D
from data.video_dataset import create_video_dataloader

# Load VAE
vae = VideoVAE3D(
    in_channels=3, latent_channels=4, base_channels=128,
    channel_mults=(1, 2, 4, 4),
    temporal_downsample=(False, False, False, False),
    spatial_downsample_factor=8, temporal_downsample_factor=1,
).cuda()

checkpoint = torch.load('vae_pretrained.pth')
vae.load_state_dict(checkpoint['vae_state_dict'])
vae.eval()

# Test reconstruction
loader = create_video_dataloader(
    '/workspace/data/train_videos',
    batch_size=1, num_frames=6, frame_size=(256, 256),
    frame_interval=1, mode='train', num_workers=0
)

for video in loader:
    video = video.cuda()
    with torch.no_grad():
        z, _, _ = vae.encode(video)
        recon = vae.decode(z)
    error = torch.nn.functional.mse_loss(recon, video)
    print(f'VAE reconstruction error: {error.item():.6f}')
    print('✓ VAE is working!' if error < 0.05 else '⚠ VAE needs more training')
    break
"
```

**Good VAE:** Reconstruction error < 0.03

## Stage 2: Train DiT with Frozen VAE

Once you have a good VAE, train the diffusion model.

### Run DiT Training

```bash
python train_advanced.py \
    --config config_advanced.yaml \
    --vae_checkpoint vae_pretrained.pth
```

**Important:**
- The VAE will be automatically **frozen** (not updated)
- Only the DiT transformer will be trained
- This ensures a stable latent space throughout training

**What to expect:**
- Training time: ~10-20 hours for 600 epochs
- Diffusion loss should decrease from ~2.0 to < 0.5
- Generate sample videos every 50-100 epochs to monitor quality

### Monitor Training

Check training progress:

```bash
# View logs
tail -f runs/advanced_experiment/train.log

# Generate samples during training (every N epochs)
python predict_advanced.py \
    --config predict_advanced.yaml \
    --checkpoint runs/advanced_experiment/checkpoint_epoch_100.pth
```

## Complete Workflow

```bash
# Step 1: Pre-train VAE (Stage 1)
python train_vae_only.py --epochs 100 --output vae_pretrained.pth

# Step 2: Verify VAE quality
# (Run the verification script above)

# Step 3: Train DiT with frozen VAE (Stage 2)
python train_advanced.py \
    --config config_advanced.yaml \
    --vae_checkpoint vae_pretrained.pth

# Step 4: Generate videos
python predict_advanced.py \
    --config predict_advanced.yaml \
    --checkpoint runs/advanced_experiment/best_model.pth
```

## Why Two Stages?

### ❌ Single-Stage Training (Wrong)
```
VAE changes → Latent space shifts → DiT's learned patterns invalid → Scrambled output
```

### ✅ Two-Stage Training (Correct)
```
Stage 1: VAE learns stable compression
Stage 2: DiT learns denoising in FIXED latent space → Coherent output
```

## Troubleshooting

### VAE reconstruction is blurry
- Train longer (increase `--epochs`)
- Check if VAE loss is still decreasing
- Ensure dataset has good quality videos

### DiT produces scrambled outputs
- Verify VAE was loaded correctly (check logs for "✓ Pre-trained VAE loaded")
- Ensure VAE is frozen (check logs for "Freezing VAE parameters")
- Check if VAE reconstruction works well (error < 0.03)

### Training is slow
- Reduce batch size if running out of memory
- Use mixed precision (`use_amp: true` in config)
- Use fewer transformer blocks initially (`depth: 6` instead of 12)

## Configuration Tips

For faster experimentation during Stage 1:
- Reduce VAE size: `vae_base_channels: 64` (instead of 128)
- Train on smaller resolution: `frame_size: [128, 128]`
- Use fewer epochs: `--epochs 50`

For better quality:
- Increase VAE capacity: `vae_base_channels: 192`
- Longer training: `--epochs 200` for VAE, `epochs: 1000` for DiT
- More data augmentation (already enabled with sliding window)

## Next Steps

Once training is complete:
1. Generate videos with `predict_advanced.py`
2. Experiment with different guidance scales
3. Try longer sequences or higher resolutions
4. Fine-tune on specific video types

---

**Remember:** Always train VAE first, then freeze it for DiT training!
