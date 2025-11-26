#!/bin/bash

echo "=========================================="
echo "Two-Stage Video Diffusion Training"
echo "=========================================="
echo ""
echo "This will run:"
echo "  Stage 1: Pre-train VAE (100 epochs)"
echo "  Stage 2: Train DiT with frozen VAE (600 epochs)"
echo ""

# Stage 1: Pre-train VAE
echo "=========================================="
echo "STAGE 1: Pre-training VAE"
echo "=========================================="
echo ""

python train_vae_only.py --epochs 100 --output vae_pretrained.pth

if [ ! -f "vae_pretrained.pth" ]; then
    echo "❌ ERROR: VAE pre-training failed!"
    echo "vae_pretrained.pth not found"
    exit 1
fi

echo ""
echo "✓ Stage 1 complete: VAE pre-trained"
echo ""

# Verify VAE
echo "Verifying VAE quality..."
python -c "
import torch
checkpoint = torch.load('vae_pretrained.pth')
val_loss = checkpoint.get('val_loss', 'unknown')
print(f'VAE validation loss: {val_loss}')
if isinstance(val_loss, float) and val_loss < 0.05:
    print('✓ VAE quality is good!')
else:
    print('⚠ VAE validation loss is high. Consider training longer.')
"

echo ""
read -p "Continue to Stage 2? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Stopped after Stage 1"
    echo "VAE checkpoint saved: vae_pretrained.pth"
    echo ""
    echo "To continue later, run:"
    echo "  python train_advanced.py --config config_advanced.yaml --vae_checkpoint vae_pretrained.pth"
    exit 0
fi

# Stage 2: Train DiT
echo ""
echo "=========================================="
echo "STAGE 2: Training DiT with frozen VAE"
echo "=========================================="
echo ""

python train_advanced.py \
    --config config_advanced.yaml \
    --vae_checkpoint vae_pretrained.pth

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Model saved to: runs/advanced_experiment/best_model.pth"
echo ""
echo "To generate videos:"
echo "  python predict_advanced.py --config predict_advanced.yaml"
