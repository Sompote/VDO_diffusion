# Sliding Window Data Augmentation - Implementation Summary

## Overview

The `VideoDataset` class has been updated to support **sliding window data augmentation**, which creates multiple overlapping clips from each image sequence or video. This significantly increases the training dataset size without requiring additional source data.

## What Changed

### 1. **New Parameters**

Added to `VideoDataset.__init__()` and `create_video_dataloader()`:

- **`use_sliding_window`** (bool, default=`True`):
  - If `True`: Generate multiple clips per sequence using sliding window
  - If `False`: Legacy mode - one random clip per sequence per epoch

- **`window_stride`** (int, default=`1`):
  - Controls overlap between clips
  - `stride=1`: Maximum overlap (most clips)
  - `stride=2`: 50% overlap
  - Higher values: Less overlap, fewer clips

### 2. **Core Functionality**

#### Previous Behavior (Legacy Mode)
```
Image sequence with 10 frames:
├── Dataset size: 1 sample
└── Each epoch: Random clip starting position
```

#### New Behavior (Sliding Window)
```
Image sequence with 10 frames, num_frames=6, stride=1:
├── Clip 0: frames [0, 1, 2, 3, 4, 5]
├── Clip 1: frames [1, 2, 3, 4, 5, 6]
├── Clip 2: frames [2, 3, 4, 5, 6, 7]
├── Clip 3: frames [3, 4, 5, 6, 7, 8]
└── Clip 4: frames [4, 5, 6, 7, 8, 9]
Result: Dataset size = 5 clips
```

### 3. **Modified Methods**

#### `VideoDataset.__init__()` (`video_dataset.py:21-117`)
- Added `use_sliding_window` and `window_stride` parameters
- Calls `_generate_clips()` to pre-compute all clips
- Stores clips instead of raw samples

#### `VideoDataset._generate_clips()` (`video_dataset.py:119-162`) **[NEW]**
- Pre-computes all valid clip starting positions
- For each image sequence:
  - Calculates `required_frames = (num_frames - 1) × frame_interval + 1`
  - Generates clips: `start_index = 0, stride, 2×stride, ...`
  - Skips sequences that are too short
- Returns list of clip dictionaries with `source` and `start_index`

#### `VideoDataset.__len__()` (`video_dataset.py:164-165`)
- Returns `len(self.clips)` instead of `len(self.samples)`
- Now reflects total number of clips, not sequences

#### `VideoDataset.__getitem__()` (`video_dataset.py:312-344`)
- Retrieves pre-computed clip by index
- Passes `start_index` to `load_image_sequence()`

#### `VideoDataset.load_image_sequence()` (`video_dataset.py:264-310`)
- Added optional `start_index` parameter
- If provided: Uses exact starting position (sliding window mode)
- If `None`: Random/fixed position (legacy mode)

#### `create_video_dataloader()` (`video_dataset.py:409-463`)
- Added `use_sliding_window` and `window_stride` parameters
- Passes them to `VideoDataset` constructor

## Usage Examples

### Example 1: Enable Sliding Window (Default)
```python
from data.video_dataset import create_video_dataloader

# Create dataloader with sliding window
train_loader = create_video_dataloader(
    video_dir="./data/train",
    num_frames=6,
    frame_interval=1,
    use_sliding_window=True,  # Enable sliding window
    window_stride=1,          # Maximum overlap
    mode="train"
)

# If you have 3 sequences with 10, 15, and 20 frames each:
# - Sequence 1 (10 frames): 5 clips
# - Sequence 2 (15 frames): 10 clips
# - Sequence 3 (20 frames): 15 clips
# Total dataset size: 30 clips (vs 3 in legacy mode)
```

### Example 2: Adjust Stride for Less Overlap
```python
# Use stride=2 for less overlap (faster training, less data)
train_loader = create_video_dataloader(
    video_dir="./data/train",
    num_frames=6,
    frame_interval=1,
    use_sliding_window=True,
    window_stride=2,  # 50% overlap
    mode="train"
)

# Same 3 sequences:
# - Sequence 1 (10 frames): 3 clips
# - Sequence 2 (15 frames): 5 clips
# - Sequence 3 (20 frames): 8 clips
# Total: 16 clips
```

### Example 3: Legacy Mode (Disable Sliding Window)
```python
# Use old behavior for comparison
train_loader = create_video_dataloader(
    video_dir="./data/train",
    num_frames=6,
    frame_interval=1,
    use_sliding_window=False,  # Legacy mode
    mode="train"
)

# Same 3 sequences:
# Total: 3 samples (random clip per epoch)
```

### Example 4: Integration with Training Script
```python
# In train_advanced.py, the dataloader is created like this:
train_loader = create_video_dataloader(
    video_dir=args.train_dir,
    batch_size=args.batch_size,
    num_frames=args.num_frames,
    frame_size=tuple(args.frame_size),
    frame_interval=args.frame_interval,
    mode="train",
    num_workers=args.num_workers,
    augment=False,
    debug_mode=args.debug_mode,
    use_sliding_window=True,   # Default: enabled
    window_stride=1,            # Default: maximum overlap
)
```

To modify behavior, you can:
1. Add CLI arguments to `train_advanced.py`
2. Add parameters to your YAML config file
3. Hardcode values in the dataloader creation

## Benefits

✅ **More Training Data**: 5-10x more clips from the same sequences
✅ **Better Temporal Learning**: Overlapping clips help learn smooth transitions
✅ **Configurable**: Adjust `window_stride` to balance data size vs diversity
✅ **Automatic Validation**: Skips sequences that are too short
✅ **Backward Compatible**: Set `use_sliding_window=False` for legacy behavior

## Testing

Run the test script to see sliding window in action:

```bash
cd /workspace/VDO_diffusion
python test_sliding_window.py
```

This will demonstrate:
- Sliding window with stride=1 (maximum overlap)
- Legacy mode (random sampling)
- Sliding window with stride=2 (less overlap)

## Technical Details

### Clip Generation Formula

For an image sequence with `N` frames:
```
required_frames = (num_frames - 1) × frame_interval + 1
max_start_index = N - required_frames

For stride S:
  valid_starts = [0, S, 2S, 3S, ..., max_start_index]
  num_clips = floor(max_start_index / S) + 1
```

### Example Calculation
```
Given:
  - N = 10 images
  - num_frames = 6
  - frame_interval = 1
  - stride = 1

Calculate:
  required_frames = (6-1)×1 + 1 = 6
  max_start_index = 10 - 6 = 4
  valid_starts = [0, 1, 2, 3, 4]
  num_clips = 5

Clips:
  [0,1,2,3,4,5], [1,2,3,4,5,6], [2,3,4,5,6,7], [3,4,5,6,7,8], [4,5,6,7,8,9]
```

## Files Modified

- `data/video_dataset.py`: Core implementation
- `test_sliding_window.py`: Test/demo script (new)
- `SLIDING_WINDOW_CHANGES.md`: This documentation (new)

## Next Steps

To use this in training:

1. **Default behavior** (sliding window enabled):
   ```bash
   python train_advanced.py --train_dir ./data/train --num_frames 6
   ```

2. **Disable for testing**:
   Modify `train_advanced.py` line 384:
   ```python
   train_loader = create_video_dataloader(
       # ...
       use_sliding_window=False,  # Disable sliding window
   )
   ```

3. **Add CLI arguments** (optional):
   Add to `train_advanced.py`:
   ```python
   parser.add_argument("--use_sliding_window", action="store_true", default=True)
   parser.add_argument("--window_stride", type=int, default=1)
   ```

---

**Author**: Claude Code
**Date**: 2025-11-25
**Version**: 1.0
