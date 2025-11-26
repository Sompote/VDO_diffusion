# Fix for Scrambled Patch Outputs

## Problem Identified

Your video diffusion model was producing **scrambled "jigsaw puzzle" outputs** because:

1. **Both VAE and DiT were being trained simultaneously**
   - The VAE's latent space was constantly changing during training
   - The DiT learned denoising patterns in one latent space
   - When the VAE shifted, the DiT's learned patterns became invalid
   - This created a "moving target" problem

2. **Training was incomplete**
   - Model trained for only 79/600 epochs (13%)
   - Insufficient training to learn proper denoising from high noise levels

## Root Cause

In latent diffusion models, the **VAE must be frozen**! Training it alongside the diffusion model causes the latent space to shift, making it impossible for the diffusion model to learn consistent denoising.

## What Was Working Correctly

Extensive testing revealed:
- ✅ VAE encode/decode works perfectly (not scrambled)
- ✅ All patchify/unpatchify operations correct
- ✅ V-prediction mathematics correct
- ✅ Gradient flow correct
- ✅ Model architecture sound
- ✅ Prediction script (`predict_advanced.py`) uses correct v-prediction conversion

The issue was purely in the **training strategy**.

## Changes Made

### 1. `train_advanced.py`
- Added code to **freeze all VAE parameters** before creating optimizer
- Optimizer now only updates DiT parameters
- Added logging to show trainable vs total parameters

### 2. `config_advanced.yaml`
- Set `vae_loss_weight: 0.0` (was 5.0)
- Set `kl_loss_weight: 0.0` (was 0.00001)

## Action Required

**You MUST delete the old checkpoint and retrain from scratch:**

```bash
cd /workspace/VDO_diffusion

# Delete old checkpoint (it learned with wrong training strategy)
rm -rf runs/advanced_experiment/*.pth

# Retrain with frozen VAE
python train_advanced.py --config config_advanced.yaml
```

## Expected Results

With the frozen VAE:
- **Stable latent space** throughout training
- DiT learns consistent denoising patterns
- **No more scrambled patches**
- Coherent, spatially-consistent video generation

## Training Tips

1. **Monitor the loss curves** - diffusion loss should steadily decrease
2. **Validate periodically** - generate samples every 50-100 epochs
3. **Be patient** - diffusion models need many epochs to converge
4. **Use the sliding window data augmentation** - already enabled in your config

## Why This Fix Works

Latent diffusion models follow a two-stage approach:
1. **Stage 1**: Train VAE to learn good image compression (already done)
2. **Stage 2**: Train diffusion model in the **fixed** latent space

Your previous training mixed these stages, causing instability. The fix properly separates them.

## Verification

After retraining, test with:
```bash
python predict_advanced.py --config predict_advanced.yaml
```

The outputs should now show:
- Coherent spatial structure
- No scrambled patches
- Smooth transitions between frames

---

**Commit:** Fix scrambled patches by freezing VAE during diffusion training
**Status:** Ready to retrain
