# Video Diffusion Prediction Framework

A PyTorch implementation of a diffusion model for video prediction. This framework offers two distinct architectures for different needs.

**Developed by AI Research Group, Department of Civil Engineering, King Mongkut's University of Technology Thonburi (KMUTT)**

## Features

- **Dual Architecture Support**:
  - **Advanced**: Latent Diffusion Transformer (DiT) with 3D VAE for high-resolution, state-of-the-art video generation.
  - **Basic**: 3D U-Net for pixel-space diffusion, suitable for learning and simple datasets.
- **Flexible Input**: Supports video files (`.mp4`, `.avi`) and image sequences.
- **Configuration**: YAML-based configuration.
- **Multi-GPU Training**: Distributed Data Parallel (DDP) support.

## ⚡ Architecture Comparison

| Feature | Basic (`train.py`) | Advanced (`train_advanced.py`) |
| :--- | :--- | :--- |
| **Architecture** | **3D U-Net** (Standard CNN) | **DiT** (Diffusion Transformer) |
| **Input Data** | Raw Pixels | **Latents** (Compressed via 3D VAE) |
| **Configuration** | `config.yaml` & `dataset.yaml` | **YAML Config** or CLI Flags |
| **Memory** | High (Stores full video) | **Efficient** (Stores compressed latents) |
| **Speed** | Slower | **Faster** (due to compression) |
| **Quality** | Good for simple motion | **State-of-the-Art** (High fidelity) |

## Installation

```bash
pip install -r requirements.txt
```

## Directory Structure

```
video_diffusion_prediction/
├── models/
│   ├── diffusion.py          # Basic U-Net model
│   └── advanced_diffusion.py # Advanced DiT + VAE model
├── data/
│   └── video_dataset.py      # Video dataset and dataloaders
├── train.py                  # Basic training script
├── train_advanced.py         # Advanced training script (DiT)
├── predict.py                # Basic inference script
├── predict_advanced.py       # Advanced inference script
├── config.yaml               # Basic training config
├── config_advanced.yaml      # Advanced training config
├── predict.yaml              # Basic inference config
├── predict_advanced.yaml     # Advanced inference config
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🚀 Advanced Model (Latent DiT) - Recommended

The advanced model implements a **Latent Diffusion Transformer (DiT)** architecture, similar to Sora or Stable Video Diffusion. It is designed for high-performance training on larger datasets.

### Model Architecture
![Advanced Architecture](assets/Gemini_Generated_Image_d3tzqd3tzqd3tzqd.png)

### Data Folder Structure

`train_advanced.py` expects a simple folder structure. You do **not** need a `dataset.yaml`.

**Option 1: Video Files (Easiest)**
Put all your video clips (`.mp4`, `.avi`, etc.) directly inside the train/val folders.
```text
data/
├── train_videos/
│   ├── clip_001.mp4
│   ├── clip_002.mp4
│   └── ...
└── val_videos/
    ├── clip_100.mp4
    └── ...
```

**Option 2: Image Sequences (Best for raw frames)**
Put each video clip in its own sub-folder containing the frames (`.jpg`, `.png`).
```text
data/
├── train_videos/
│   ├── drive_001/
│   │   ├── frame_0001.jpg
│   │   ├── frame_0002.jpg
│   │   └── ...
│   └── drive_002/
│       ├── frame_0001.jpg
│       └── ...
└── val_videos/
    └── ...
```

### How to Run

You can run the script using a YAML configuration file (Recommended) or by passing arguments directly.

**1. Using Config File (Recommended)**
Edit `config_advanced.yaml` to set your parameters, then run:
```bash
python train_advanced.py --config config_advanced.yaml
```

**2. Overriding via CLI**
You can override specific settings from the config file by passing them as arguments:
```bash
python train_advanced.py --config config_advanced.yaml --batch_size 2 --lr 5e-5
```

**3. Full Pipeline (Automated)**
To run the complete two-stage training process (Pre-train VAE -> Train DiT), use the provided shell script:
```bash
./train_full_pipeline.sh
```
This script handles:
1.  **Stage 1:** Pre-trains the 3D VAE (100 epochs) to learn efficient video compression.
2.  **Stage 2:** Trains the DiT model (600 epochs) using the pre-trained VAE.

### Key Parameters (Advanced)

**Data Settings**
*   `--train_dir`: Path to training videos folder.
*   `--val_dir`: Path to validation videos folder (optional).
*   `--num_frames`: Total frames per clip (default: 16).
*   `--frame_size`: Resolution as `Height Width` (default: 256 256).

**Model Architecture (DiT)**
*   `--patch_size`: Size of patches to tokenize (default: 2 2).
*   `--hidden_dim`: Width of the transformer (default: 768).
*   `--depth`: Number of transformer blocks (default: 12).
*   `--num_heads`: Attention heads (default: 12).

**VAE (Compression)**
*   `--latent_channels`: Channels in compressed latent space (default: 4).
*   `--spatial_downsample`: How much to shrink image size (default: 8x).
*   `--temporal_downsample`: How much to shrink frame count (default: 4x).

**Training Optimizations**
*   `--use_amp`: **Highly Recommended.** Uses Automatic Mixed Precision (FP16) to save memory and speed up training.
*   `--use_ema`: Maintains a "shadow" model with smoothed weights for better generation quality.
*   `--gradient_accumulation_steps`: Simulates larger batch sizes.

### Inference (Prediction)

After training, you can generate new videos using `predict_advanced.py`.

**1. Configure Prediction**
Edit `predict_advanced.yaml` to match your training settings (especially `video` and `vae` sections) and point to your checkpoint.

```yaml
inference:
  checkpoint: "./runs/advanced_experiment/final_model.pth"
  input_video: "./data/val_videos/clip_100.mp4"
  output_dir: "./outputs/advanced_predictions"

video:
  num_frames: 6               # Must match training
  num_context_frames: 5       # How many frames to condition on
  ...
```

**2. Run Prediction**
```bash
python predict_advanced.py --config predict_advanced.yaml
```

---

## 🎓 Basic Model (3D U-Net) - Research

The basic model uses a **3D U-Net**. Good for learning and simple datasets.

### How to Run

**1. Training**
Edit `config.yaml` and `dataset.yaml` (if needed).
```bash
python train.py --config config.yaml --data dataset.yaml
```

**2. Prediction**
Edit `predict.yaml` to set your checkpoint and input.
```bash
python predict.py --config predict.yaml
```

### Basic Model Parameters

**Data Parameters**
- `--config`: Path to training configuration YAML (optional)
- `--data`: Path to dataset YAML describing train/val/test splits
- `--train_dir`, `--val_dir`: Override dataset directories directly
- `--output_dir`: Override experiment output directory

**Model / Sequence Parameters**
- `--num_frames`: Total frames per training clip (optional with context/future split)
- `--context_frames`, `--future_frames`: Define input vs. prediction window
- `--frame_size`: Frame size (`H W`) or single square value
- `--frame_interval`: Sampling interval between frames
- `--base_channels`, `--channel_mults`, `--time_emb_dim`: Network width/depth controls

**Diffusion Parameters**
- `--num_timesteps`: Number of diffusion steps
- `--beta_start`, `--beta_end`: Beta schedule bounds
- `--schedule`: Noise schedule (`linear` or `cosine`)

**Training Parameters**
- `--batch_size`, `--epochs`, `--lr`, `--weight_decay`
- `--num_workers`: Data loader worker count
- `--save_interval`: Checkpoint frequency
- `--resume`: Resume from checkpoint path
- `--train_augment` / `--no-train-augment`: Toggle training augmentations
- `--val_augment` / `--no-val-augment`: Toggle validation augmentations
- `--gpus`: Number of GPUs (DDP spawns when >1)

---

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir ./runs/experiment1/logs
```

This will display:
- Training and validation loss curves
- Learning rate schedule
- Sample predictions (if implemented)

## Tips for Better Results

1. **Data Quality**: Use high-quality, diverse video data
2. **Frame Rate**: Adjust `frame_interval` based on video motion speed
3. **Model Size**: Increase `base_channels` or `channel_mults` for larger models
4. **Training Time**: Use cosine schedule and lower learning rate for longer training
5. **GPU Memory**: Reduce `batch_size` or `num_frames` if running out of memory
6. **Multi-GPU**: Use DDP for faster training on multiple GPUs

## Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM (NVIDIA GTX 1080 or better)
- RAM: 16GB
- Storage: Sufficient for video dataset

**Recommended:**
- GPU: 24GB VRAM (NVIDIA RTX 3090/4090 or A100)
- RAM: 32GB+
- Storage: SSD for faster data loading

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{video_diffusion_prediction_2025,
  title={Video Diffusion Prediction Framework},
  author={AI Research Group, Department of Civil Engineering, KMUTT},
  year={2025},
  institution={King Mongkut's University of Technology Thonburi},
  howpublished={\url{https://github.com/Sompote/VDO_diffusion}}
}
```

## License

MIT License

## Acknowledgments

This implementation is based on:
- Denoising Diffusion Probabilistic Models (DDPM)
- Video Diffusion Models
- U-Net architecture for diffusion models
