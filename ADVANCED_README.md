# Advanced Video Diffusion Model (State-of-the-Art 2024-2025)

This implementation incorporates the latest research in video diffusion models from 2024-2025, based on:

- **Latte** (ICLR 2024): Latent Diffusion Transformer for Video Generation
- **Sora** (OpenAI 2024): DiT architecture with spacetime patches
- **LTX-Video** (2025): Real-time Video Latent Diffusion
- **EDM** (Karras et al.): Elucidating Design Space of Diffusion Models
- **Classifier-Free Guidance**: Ho & Salimans (2022)
- **V-Prediction**: Progressive Distillation (Google)

## Key Innovations

### 1. Latent Diffusion with 3D VAE
- **High compression ratio**: 192x (32×32×8 spatial-temporal compression)
- **Temporal coherence**: 3D convolutions preserve motion continuity
- **Efficient training**: Operate in compact latent space

### 2. DiT (Diffusion Transformer) Architecture
- **Factorized Spatiotemporal Attention**: Separate spatial and temporal processing
- **AdaLN-Zero Conditioning**: Adaptive layer normalization with zero initialization
- **QK Normalization**: Improved training stability (from DALL-E 3)
- **Flash Attention**: Optimized attention computation when available

### 3. V-Prediction Parameterization
- **Better color stability**: Avoids color shift in video generation
- **Improved distillation**: Better for few-step sampling
- **Mathematical formulation**: v = α_t * noise - σ_t * x_0

### 4. Classifier-Free Guidance (CFG)
- **Conditional & unconditional training**: Single model, dual capability
- **Guidance scale control**: Trade-off between quality and diversity
- **Flexible conditioning**: Supports class labels, text (future), etc.

### 5. Advanced Training Techniques
- **Mixed Precision (AMP)**: Faster training, lower memory
- **Exponential Moving Average (EMA)**: Better sample quality
- **Gradient Accumulation**: Larger effective batch sizes
- **Cosine/Sigmoid schedules**: Improved noise scheduling

## Architecture Comparison

| Feature | Basic Model | Advanced Model |
|---------|-------------|----------------|
| Space | Pixel | Latent (192x compression) |
| Backbone | 3D U-Net | DiT (Transformer) |
| Attention | Simple 3D | Factorized Spatial-Temporal |
| Prediction | Noise (ε) | V-prediction |
| Conditioning | None | Classifier-free guidance |
| Training | Standard | AMP + EMA + Grad Accum |
| Parameters | ~50M | ~400M (configurable) |

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or with conda
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install einops timm opencv-python tensorboard tqdm
```

## Quick Start

### 1. Train the Advanced Model

**Basic Training (with VAE + DiT):**
```bash
python train_advanced.py \
    --train_dir ./data/train \
    --val_dir ./data/val \
    --output_dir ./runs/advanced_exp1 \
    --batch_size 4 \
    --epochs 100 \
    --num_frames 16 \
    --use_amp \
    --use_ema
```

**Advanced Training (Full Features):**
```bash
python train_advanced.py \
    --train_dir ./data/train \
    --val_dir ./data/val \
    --output_dir ./runs/sota_model \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --epochs 200 \
    --num_frames 16 \
    --frame_size 256 256 \
    --hidden_dim 1024 \
    --depth 24 \
    --num_heads 16 \
    --prediction_type v \
    --beta_schedule cosine \
    --use_amp \
    --use_ema \
    --ema_decay 0.9999 \
    --lr 1e-4 \
    --clip_grad_norm 1.0
```

**Conditional Generation (with class labels):**
```bash
python train_advanced.py \
    --train_dir ./data/train \
    --num_classes 10 \
    --p_uncond 0.1 \
    --guidance_scale 7.5 \
    --use_amp \
    --use_ema
```

### 2. Model Sizes

Configure model size based on your hardware:

**Small (8GB GPU):**
```bash
--hidden_dim 512 --depth 8 --num_heads 8 --batch_size 2
# ~100M parameters
```

**Medium (16GB GPU):**
```bash
--hidden_dim 768 --depth 12 --num_heads 12 --batch_size 4
# ~400M parameters (default)
```

**Large (24GB+ GPU):**
```bash
--hidden_dim 1024 --depth 24 --num_heads 16 --batch_size 4
# ~1B parameters
```

**XL (40GB+ GPU):**
```bash
--hidden_dim 1536 --depth 28 --num_heads 24 --batch_size 8
# ~3B parameters (Sora-like)
```

## Training Parameters

### VAE Parameters
- `--latent_channels`: Latent space channels (default: 4)
- `--vae_base_channels`: VAE base channels (default: 128)
- `--spatial_downsample`: Spatial compression factor (default: 8)
- `--temporal_downsample`: Temporal compression factor (default: 4)

### DiT Parameters
- `--patch_size`: Spatial patch size (default: 2 2)
- `--hidden_dim`: Transformer hidden dimension (default: 768)
- `--depth`: Number of transformer blocks (default: 12)
- `--num_heads`: Number of attention heads (default: 12)
- `--dim_head`: Dimension per head (default: 64)
- `--ff_mult`: Feed-forward multiplier (default: 4)

### Diffusion Parameters
- `--num_timesteps`: Diffusion steps (default: 1000)
- `--beta_schedule`: Noise schedule - linear/cosine/sigmoid (default: cosine)
- `--prediction_type`: Prediction target - eps/x0/v (default: v)

### Classifier-Free Guidance
- `--num_classes`: Number of classes (enables CFG)
- `--guidance_scale`: CFG scale (default: 7.5)
- `--p_uncond`: Unconditional training probability (default: 0.1)

### Advanced Training
- `--use_amp`: Enable mixed precision training
- `--use_ema`: Enable exponential moving average
- `--ema_decay`: EMA decay rate (default: 0.9999)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 1)
- `--clip_grad_norm`: Gradient clipping norm (default: 1.0)

## Architecture Details

### 1. Video VAE (3D Encoder-Decoder)

```
Input: (B, 3, 16, 256, 256)
  ↓ Encoder (3D Conv + Downsample)
Latent: (B, 4, 4, 32, 32)  # 192x compression
  ↓ Decoder (3D TransposeConv + Upsample)
Output: (B, 3, 16, 256, 256)
```

**Key Features:**
- Temporal downsampling: Preserves motion coherence
- Group normalization: Better than batch norm for small batches
- Residual connections: Stable training

### 2. DiT Block (Diffusion Transformer)

```
Input: (B, T, N, D)  # T=time, N=spatial patches, D=hidden_dim
  ↓ AdaLN Modulation (time conditioning)
  ↓ Spatial Attention (across patches)
  ↓ Temporal Attention (across frames)
  ↓ Feed-Forward Network
Output: (B, T, N, D)
```

**Key Features:**
- **Factorized Attention**: Spatial (H×W) then Temporal (T)
- **AdaLN-Zero**: Timestep conditioning with zero initialization
- **QK Normalization**: Stable training, better convergence
- **Flash Attention**: 2-4x faster when available

### 3. V-Prediction

Traditional diffusion models predict noise (ε) or original data (x₀).
V-prediction combines both:

```
v = √(ᾱ_t) * ε - √(1-ᾱ_t) * x₀
```

**Advantages:**
- Better for video: Avoids color shift
- Better for distillation: Enables few-step sampling
- Better stability: Smoother training dynamics

### 4. Classifier-Free Guidance

During training:
```python
# 90% conditional, 10% unconditional
if random() < 0.1:
    class_label = NULL_CLASS
model_output = DiT(x_noisy, t, class_label)
```

During sampling:
```python
# Predict with and without condition
out_cond = DiT(x_t, t, class_label)
out_uncond = DiT(x_t, t, NULL_CLASS)

# Combine with guidance scale
out = out_uncond + guidance_scale * (out_cond - out_uncond)
```

**Guidance Scale Effects:**
- `1.0`: Pure conditional (most diverse)
- `3.5-7.5`: Balanced (recommended)
- `>10.0`: High fidelity (less diverse)

## Performance Optimization

### Memory Optimization

**1. Gradient Checkpointing** (TODO):
```python
# Save memory at cost of speed
--gradient_checkpointing
```

**2. Gradient Accumulation**:
```bash
# Effective batch size = 2 * 4 = 8
--batch_size 2 --gradient_accumulation_steps 4
```

**3. Lower Precision**:
```bash
# Use AMP for 2x memory reduction
--use_amp
```

**4. Smaller Latent Space**:
```bash
# Increase compression
--spatial_downsample 16 --temporal_downsample 8
```

### Speed Optimization

**1. Flash Attention**:
- Automatically used if PyTorch ≥ 2.0
- 2-4x faster attention computation

**2. Compile Model** (PyTorch 2.0+):
```python
model = torch.compile(model)
```

**3. Efficient Data Loading**:
```bash
--num_workers 8  # More workers for I/O
```

**4. Mixed Precision**:
```bash
--use_amp  # ~2x faster training
```

## Comparison with Other Methods

### vs. GAN-based Methods
- ✅ Better mode coverage (more diverse outputs)
- ✅ More stable training (no mode collapse)
- ✅ Better quality at high resolution
- ❌ Slower inference (iterative process)

### vs. Autoregressive Models
- ✅ Parallel generation (faster)
- ✅ Better long-range coherence
- ❌ Requires more memory during training

### vs. Pixel-Space Diffusion
- ✅ 10-100x faster training (latent space)
- ✅ Can scale to higher resolutions
- ✅ Lower memory requirements
- ⚠️ Requires pre-training VAE

## Research Papers

### Core Architecture
1. **Latte** (ICLR 2024): https://arxiv.org/abs/2401.03048
   - Latent Diffusion Transformer for Video Generation

2. **DiT** (ICCV 2023): https://arxiv.org/abs/2212.09748
   - Scalable Diffusion Models with Transformers

3. **LTX-Video** (2025): https://arxiv.org/abs/2501.00103
   - Real-time Video Latent Diffusion

### Training Techniques
4. **EDM** (NeurIPS 2022): https://arxiv.org/abs/2206.00364
   - Elucidating the Design Space of Diffusion Models

5. **Classifier-Free Guidance** (2022): https://arxiv.org/abs/2207.12598
   - Improved conditional generation

6. **V-Prediction** (ICLR 2022): https://arxiv.org/abs/2202.00512
   - Progressive Distillation for Fast Sampling

### Attention Mechanisms
7. **Flash Attention** (NeurIPS 2022): https://arxiv.org/abs/2205.14135
   - Fast and Memory-Efficient Exact Attention

8. **Spatiotemporal Attention** (2024): Multiple papers on factorized attention

## Troubleshooting

### Out of Memory
```bash
# Solution 1: Reduce batch size
--batch_size 1

# Solution 2: Use gradient accumulation
--batch_size 2 --gradient_accumulation_steps 4

# Solution 3: Smaller model
--hidden_dim 512 --depth 8

# Solution 4: Increase compression
--spatial_downsample 16 --temporal_downsample 8

# Solution 5: Enable AMP
--use_amp
```

### Slow Training
```bash
# Enable all optimizations
--use_amp --num_workers 8

# Reduce diffusion steps (faster, slight quality loss)
--num_timesteps 500
```

### Poor Sample Quality
```bash
# Use EMA for better samples
--use_ema --ema_decay 0.9999

# Use v-prediction
--prediction_type v

# Use cosine schedule
--beta_schedule cosine

# Train longer
--epochs 200
```

### Color Shift in Videos
```bash
# Use v-prediction (recommended)
--prediction_type v

# Or adjust VAE regularization (TODO)
```

## Future Improvements

- [ ] Text-to-Video with CLIP/T5 conditioning
- [ ] Motion Buckets (Stable Video Diffusion)
- [ ] Camera control conditioning
- [ ] Sparse attention patterns optimization
- [ ] Multi-resolution training
- [ ] Video interpolation mode
- [ ] Real-time inference optimization

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{advanced_video_diffusion_2025,
  title={Advanced Video Diffusion Model: State-of-the-Art Implementation},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/video_diffusion_prediction}}
}
```

## License

MIT License

## Acknowledgments

This implementation builds upon:
- Latte (Shanghai AI Lab)
- DiT (Meta AI)
- LTX-Video (Lightricks)
- EDM (NVIDIA)
- Stable Diffusion (Stability AI)
- PyTorch Team (Flash Attention, AMP)
