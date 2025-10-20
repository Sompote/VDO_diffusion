# Video Diffusion Prediction Framework - Complete Index

## 📖 Documentation Guide

### Quick Start (Read First!)
1. **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
2. **[SUMMARY.md](SUMMARY.md)** - Complete project overview

### Implementation Documentation
3. **[README.md](README.md)** - Basic model (3D U-Net) documentation
4. **[ADVANCED_README.md](ADVANCED_README.md)** - Advanced model (DiT) documentation
5. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Visual architecture diagrams

### Configuration
6. **[config_example.yaml](config_example.yaml)** - Training configuration template
7. **[dataset_example.yaml](dataset_example.yaml)** - YOLO-style dataset definition
7. **[requirements.txt](requirements.txt)** - Python dependencies

## 🗂️ Code Structure

### Core Models (`models/`)
- **[models/diffusion.py](models/diffusion.py)** (520 lines)
  - `VideoDiffusionUNet` - 3D U-Net for pixel-space diffusion
  - `GaussianDiffusion` - Basic diffusion process
  - `ResidualBlock`, `AttentionBlock` - Building blocks

- **[models/advanced_diffusion.py](models/advanced_diffusion.py)** (890 lines)
  - `VideoVAE3D` - 3D Video VAE encoder/decoder
  - `LatentVideoDiT` - Diffusion Transformer (DiT)
  - `AdvancedVideoDiffusion` - Complete latent diffusion system
  - `SpatialAttention`, `TemporalAttention` - Factorized attention
  - `DiTBlock` - Transformer block with AdaLN-Zero

- **[models/__init__.py](models/__init__.py)**
  - Package exports

### Data Loading (`data/`)
- **[data/video_dataset.py](data/video_dataset.py)** (340 lines)
  - `VideoDataset` - General video dataset
  - `VideoPredictionDataset` - Context/future frame splitting
  - `create_video_dataloader` - DataLoader factory
  - `create_prediction_dataloader` - Prediction DataLoader factory

- **[data/__init__.py](data/__init__.py)**
  - Package exports

### Training Scripts
- **[train.py](train.py)** (430 lines)
  - Basic model training
  - Single/Multi-GPU support
  - Standard training loop

- **[train_advanced.py](train_advanced.py)** (500 lines)
  - Advanced model training
  - Mixed precision (AMP)
  - Exponential Moving Average (EMA)
  - Gradient accumulation
  - Classifier-free guidance

### Inference Scripts
- **[predict.py](predict.py)** (340 lines)
  - Video prediction from context
  - Unconditional generation
  - Model loading utilities
  - Video I/O handling

### Utilities
- **[example.py](example.py)** (280 lines)
  - Usage examples for all features
  - Model creation demos
  - DataLoader examples
  - Forward/backward pass demos

- **[compare_models.py](compare_models.py)** (350 lines)
  - Side-by-side comparison
  - Performance metrics
  - Architecture differences
  - Complexity analysis

- **[setup.sh](setup.sh)**
  - Automated installation
  - Directory creation
  - Dependency verification

## 📊 Statistics

```
Total Lines of Code:    ~3,900 lines
Total Documentation:    ~3,500 lines
Total Files:            16 files
Research Papers:        7+ papers implemented
Development Time:       Complete implementation
```

## 🎓 Learning Path

### Beginner Path
1. Read [QUICKSTART.md](QUICKSTART.md)
2. Run `python example.py`
3. Read [README.md](README.md) - Basic model
4. Train basic model: `python train.py`
5. Make predictions: `python predict.py`

### Advanced Path
6. Run `python compare_models.py`
7. Read [ADVANCED_README.md](ADVANCED_README.md)
8. Read [ARCHITECTURE.md](ARCHITECTURE.md)
9. Train advanced model: `python train_advanced.py`
10. Experiment with parameters

### Research Path
11. Study implemented papers (see ADVANCED_README.md)
12. Modify architectures in `models/advanced_diffusion.py`
13. Implement new features
14. Compare results

## 🚀 Use Cases by File

### Quick Prototyping
```bash
python example.py                    # Test installation
python compare_models.py             # Understand differences
```

### Basic Training
```bash
python train.py \
    --train_dir ./data/train \
    --batch_size 4
```

### Production Training
```bash
python train_advanced.py \
    --train_dir ./data/train \
    --use_amp --use_ema \
    --prediction_type v
```

### Inference
```bash
python predict.py \
    --checkpoint ./runs/best.pth \
    --mode predict \
    --input_video test.mp4
```

## 🔧 Key Components

### Basic Model Components
```python
from models.diffusion import (
    VideoDiffusionUNet,      # 3D U-Net architecture
    GaussianDiffusion,       # Diffusion process
)
```

### Advanced Model Components
```python
from models.advanced_diffusion import (
    VideoVAE3D,              # 3D VAE encoder/decoder
    LatentVideoDiT,          # Transformer backbone
    AdvancedVideoDiffusion,  # Complete system
    SpatialAttention,        # Spatial attention
    TemporalAttention,       # Temporal attention
    DiTBlock,                # Transformer block
)
```

### Data Components
```python
from data.video_dataset import (
    VideoDataset,            # Video dataset
    create_video_dataloader, # DataLoader factory
)
```

## 📈 Feature Matrix

| Feature | Basic Model | Advanced Model | File |
|---------|-------------|----------------|------|
| 3D U-Net | ✅ | - | `models/diffusion.py` |
| DiT Architecture | - | ✅ | `models/advanced_diffusion.py` |
| Pixel Space | ✅ | - | `train.py` |
| Latent Space | - | ✅ | `train_advanced.py` |
| Noise Prediction | ✅ | - | `models/diffusion.py` |
| V-Prediction | - | ✅ | `models/advanced_diffusion.py` |
| Simple Attention | ✅ | - | `models/diffusion.py` |
| Factorized Attention | - | ✅ | `models/advanced_diffusion.py` |
| Mixed Precision | - | ✅ | `train_advanced.py` |
| EMA | - | ✅ | `train_advanced.py` |
| CFG | - | ✅ | `models/advanced_diffusion.py` |
| Flash Attention | - | ✅ | `models/advanced_diffusion.py` |

## 🎯 By Task

### Task: Train Video Generation Model
**Files needed:**
- `train_advanced.py` - Training script
- `models/advanced_diffusion.py` - Model definition
- `data/video_dataset.py` - Data loading
- `config_example.yaml` - Training configuration reference
- `dataset_example.yaml` - Dataset split and sequence definition

### Task: Predict Future Frames
**Files needed:**
- `predict.py` - Inference script
- Trained checkpoint file
- Input video file

### Task: Understand Architecture
**Files needed:**
- `ARCHITECTURE.md` - Visual diagrams
- `compare_models.py` - Comparison tool
- `models/advanced_diffusion.py` - Implementation

### Task: Research New Methods
**Files needed:**
- `models/advanced_diffusion.py` - Modify here
- `train_advanced.py` - Training infrastructure
- `ADVANCED_README.md` - Research references

## 🔍 Finding Specific Features

### Attention Mechanisms
- **Spatial Attention**: `models/advanced_diffusion.py` line ~120
- **Temporal Attention**: `models/advanced_diffusion.py` line ~200
- **3D Attention**: `models/diffusion.py` line ~180

### Training Features
- **Mixed Precision**: `train_advanced.py` line ~150
- **EMA**: `train_advanced.py` line ~50
- **Gradient Accumulation**: `train_advanced.py` line ~180
- **Multi-GPU**: `train.py` line ~320

### Diffusion Process
- **Forward Process**: `models/diffusion.py` line ~280
- **Reverse Process**: `models/diffusion.py` line ~310
- **V-Prediction**: `models/advanced_diffusion.py` line ~680

### Data Processing
- **Video Loading**: `data/video_dataset.py` line ~80
- **Augmentation**: `data/video_dataset.py` line ~150
- **Batching**: `data/video_dataset.py` line ~250

## 📚 Documentation Cross-Reference

### For Installation
- [QUICKSTART.md](QUICKSTART.md) - Installation guide
- [setup.sh](setup.sh) - Automated setup
- [requirements.txt](requirements.txt) - Dependencies

### For Training
- [README.md](README.md) - Basic training
- [ADVANCED_README.md](ADVANCED_README.md) - Advanced training
- [config_example.yaml](config_example.yaml) - Configuration

### For Understanding
- [SUMMARY.md](SUMMARY.md) - Project overview
- [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture details
- [compare_models.py](compare_models.py) - Comparison

### For Research
- [ADVANCED_README.md](ADVANCED_README.md) - Paper references
- [models/advanced_diffusion.py](models/advanced_diffusion.py) - Implementation
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical details

## 🎓 Recommended Reading Order

### Quick Start (30 minutes)
1. INDEX.md (this file)
2. QUICKSTART.md
3. Run `python example.py`

### Basic Understanding (2 hours)
4. SUMMARY.md
5. README.md
6. Run `python train.py --help`

### Advanced Understanding (4 hours)
7. compare_models.py output
8. ADVANCED_README.md
9. ARCHITECTURE.md

### Implementation Details (8+ hours)
10. models/diffusion.py (basic)
11. models/advanced_diffusion.py (advanced)
12. train_advanced.py (training)

### Research & Development (ongoing)
13. Modify models/advanced_diffusion.py
14. Experiment with hyperparameters
15. Read referenced papers
16. Implement new features

## 💡 Tips

### For Learners
- Start with `example.py` to understand basics
- Read `compare_models.py` output to see differences
- Train basic model first, then advanced

### For Researchers
- Focus on `models/advanced_diffusion.py`
- Reference `ADVANCED_README.md` for papers
- Use `train_advanced.py` for experiments

### For Production Users
- Use advanced model only
- Enable all optimizations (AMP, EMA)
- Monitor with TensorBoard
- Use v-prediction and cosine schedule

## 📞 Support

### Issues
- Check QUICKSTART.md troubleshooting
- Review ADVANCED_README.md FAQ
- Run `python compare_models.py` for specs

### Contributing
- All code is modular and documented
- Add new models to `models/`
- Add new training features to `train_advanced.py`
- Update relevant documentation

## 🎯 Quick Command Reference

```bash
# Setup
bash setup.sh

# Basic training
python train.py --train_dir ./data/train

# Advanced training (recommended)
python train_advanced.py --train_dir ./data/train --use_amp --use_ema

# Prediction
python predict.py --checkpoint runs/model.pth --input_video test.mp4

# Compare models
python compare_models.py

# Examples
python example.py
```

---

**Total Project Size:**
- Code: ~3,900 lines
- Documentation: ~3,500 lines  
- Files: 16 files
- Complete, production-ready implementation

**Last Updated:** October 2024 (State-of-the-art 2024-2025)
