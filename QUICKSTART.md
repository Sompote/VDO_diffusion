# Quick Start Guide

Get started with Video Diffusion Prediction in 5 minutes!

## 1. Installation

```bash
# Run the setup script
bash setup.sh

# Or manually install
pip install -r requirements.txt
```

## 2. Prepare Data

Create a YOLO-style dataset YAML (see `dataset_example.yaml`) that points to your train/val folders and defines how many frames to feed in vs. predict. Each folder can contain full videos **or** sub-directories of ordered frame images:

```yaml
train: ./data/train
val: ./data/val
context_frames: 8
future_frames: 1
frame_size: [256, 256]
frame_interval: 1
augment: true
```

For frame folders, keep images inside dedicated sub-directories (e.g. `train/drive_001/frame_0001.jpg ...`). Each sub-directory is treated as one clip and must contain at least `context_frames + future_frames` images after applying `frame_interval`.

Supported formats inside each folder: `.mp4`, `.avi`, `.mov`, `.mkv`

## 3. Test the Installation

Run the example script to verify everything works:

```bash
python example.py
```

This will:
- Create a model
- Run a forward pass
- Show model statistics

## 4. Train a Model

### Basic Training (Single GPU)

```bash
python train.py \
    --config config_example.yaml \
    --data dataset_example.yaml
```

### Advanced Training (Multi-GPU)

```bash
python train.py \
    --config config_example.yaml \
    --data dataset_example.yaml \
    --gpus 4
```

### Training on Small GPU (8GB VRAM)

```bash
python train.py \
    --config config_example.yaml \
    --data dataset_example.yaml \
    --batch_size 2 \
    --num_frames 8 \
    --frame_size 128 128 \
    --base_channels 32
```

You can still override directories directly (`--train_dir ./data/train`) if you prefer not to use YAML files.

## 5. Monitor Training

Open TensorBoard to view training progress:

```bash
tensorboard --logdir ./runs/my_experiment/logs
```

Then open http://localhost:6006 in your browser.

## 6. Make Predictions

### Predict Future Frames from Video

```bash
python predict.py \
    --checkpoint ./runs/my_experiment/best_model.pth \
    --config config_example.yaml \
    --data dataset_example.yaml \
    --mode predict \
    --input_video ./data/test/sample.mp4 \
    --num_context_frames 8 \
    --num_future_frames 8 \
    --output_dir ./outputs \
    --output_name prediction
```

This creates:
- `prediction.mp4` - Full video (context + prediction)
- `prediction_context.mp4` - Context frames only
- `prediction_prediction.mp4` - Predicted frames only

### Generate Videos from Noise

```bash
python predict.py \
    --checkpoint ./runs/my_experiment/best_model.pth \
    --config config_example.yaml \
    --data dataset_example.yaml \
    --mode generate \
    --num_frames 16 \
    --batch_size 4 \
    --output_dir ./outputs
```

## Common Issues

### Out of Memory

**Solution 1**: Reduce batch size
```bash
--batch_size 1
```

**Solution 2**: Reduce number of frames
```bash
--num_frames 8
```

**Solution 3**: Reduce resolution
```bash
--frame_size 128 128
```

**Solution 4**: Use smaller model
```bash
--base_channels 32 --channel_mults 1 2 4
```

### Videos Not Loading

**Check 1**: Verify video format is supported
```bash
# Convert to MP4 if needed
ffmpeg -i input.avi output.mp4
```

**Check 2**: Ensure videos are long enough
- Minimum frames needed = `num_frames * frame_interval`
- Example: 16 frames × 1 interval = 16 frames minimum

### Slow Training

**Solution 1**: Use multiple GPUs
```bash
--gpus 4
```

**Solution 2**: Increase batch size (if memory allows)
```bash
--batch_size 8
```

**Solution 3**: Use more data loading workers
```bash
--num_workers 8
```

**Solution 4**: Use fewer diffusion timesteps
```bash
--num_timesteps 500
```

## Tips for Best Results

1. **Data Quality**
   - Use high-quality, consistent videos
   - Similar resolution and frame rate
   - Diverse content

2. **Training Duration**
   - Start with 50-100 epochs
   - Use validation loss to monitor overfitting
   - Save checkpoints regularly

3. **Hyperparameters**
   - Learning rate: 1e-4 to 2e-4
   - Batch size: As large as GPU allows
   - Cosine schedule often works better than linear

4. **Hardware**
   - Use GPU if available (10-100× faster)
   - More VRAM = larger models/batches
   - SSD for faster data loading

## Example Workflow

```bash
# 1. Setup
bash setup.sh

# 2. Add your videos to data/train and data/val

# 3. Test installation
python example.py

# 4. Train model
python train.py \
    --train_dir ./data/train \
    --val_dir ./data/val \
    --output_dir ./runs/exp1 \
    --batch_size 4 \
    --epochs 100

# 5. Monitor training (in another terminal)
tensorboard --logdir ./runs/exp1/logs

# 6. Make predictions
python predict.py \
    --checkpoint ./runs/exp1/best_model.pth \
    --mode predict \
    --input_video ./data/test/video.mp4 \
    --output_dir ./outputs
```

## Need Help?

- Read the full documentation: `README.md`
- Check example code: `example.py`
- Review model architecture: `models/diffusion.py`
- Examine data loading: `data/video_dataset.py`

Happy video prediction! 🎥✨
