#!/bin/bash

echo "=================================="
echo "Restarting Training with Fixed Configuration"
echo "=================================="
echo ""
echo "This script will:"
echo "1. Delete the old checkpoint (trained with wrong strategy)"
echo "2. Start fresh training with frozen VAE"
echo ""

read -p "Are you sure you want to delete old checkpoints? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cancelled. No changes made."
    exit 0
fi

echo ""
echo "Deleting old checkpoints..."
rm -rf runs/advanced_experiment/*.pth

echo "Old checkpoints deleted!"
echo ""
echo "Starting training with frozen VAE..."
echo "VAE will NOT be updated - only DiT will train"
echo ""

python train_advanced.py --config config_advanced.yaml
