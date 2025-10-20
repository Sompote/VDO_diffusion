#!/bin/bash

echo "=========================================="
echo "Video Diffusion Prediction Setup"
echo "=========================================="

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data/train
mkdir -p data/val
mkdir -p data/test
mkdir -p outputs
mkdir -p runs

echo "✓ Directories created:"
echo "  - data/train"
echo "  - data/val"
echo "  - data/test"
echo "  - outputs"
echo "  - runs"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "✓ Dependencies installed"

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    python -c "import torch; print(f'GPU device: {torch.cuda.get_device_name(0)}')"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Add your video files to data/train/ and data/val/"
echo "2. Run the example script: python example.py"
echo "3. Start training: python train.py --train_dir ./data/train --val_dir ./data/val"
echo ""
echo "For more information, see README.md"
echo ""
