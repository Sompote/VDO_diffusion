# Sliding Window Visual Explanation

## How Frame Selection Works

### Scenario: 10 images in a folder, need 6 frames per clip

```
Available images: [0] [1] [2] [3] [4] [5] [6] [7] [8] [9]
                   img0.jpg, img1.jpg, ..., img9.jpg
```

---

## Old Method (Legacy Mode)
**`use_sliding_window=False`**

```
Dataset size: 1 sample

Each epoch randomly picks ONE clip:
Epoch 1: Maybe [3] [4] [5] [6] [7] [8]  ← random start
Epoch 2: Maybe [0] [1] [2] [3] [4] [5]  ← different random start
Epoch 3: Maybe [2] [3] [4] [5] [6] [7]  ← different random start
```

**Problem**: Only see 1 clip per sequence per epoch, wasting data!

---

## New Method (Sliding Window, Stride=1)
**`use_sliding_window=True, window_stride=1`**

```
Dataset size: 5 clips (pre-generated)

Clip 0: [0] [1] [2] [3] [4] [5]
           ↓
Clip 1:     [1] [2] [3] [4] [5] [6]
               ↓
Clip 2:         [2] [3] [4] [5] [6] [7]
                   ↓
Clip 3:             [3] [4] [5] [6] [7] [8]
                       ↓
Clip 4:                 [4] [5] [6] [7] [8] [9]

Can't start at [5]: Not enough frames left (only 5 frames: 5,6,7,8,9)
```

**Benefit**: 5x more training data from the same 10 images!

---

## Sliding Window with Stride=2
**`use_sliding_window=True, window_stride=2`**

```
Dataset size: 3 clips

Clip 0: [0] [1] [2] [3] [4] [5]
           ↓ (skip 1 frame)
Clip 1:         [2] [3] [4] [5] [6] [7]
                   ↓ (skip 1 frame)
Clip 2:                 [4] [5] [6] [7] [8] [9]

Can't start at [6]: Not enough frames left
```

**Benefit**: 3x more data, less overlap (faster training)

---

## Sliding Window with Stride=3
**`use_sliding_window=True, window_stride=3`**

```
Dataset size: 2 clips

Clip 0: [0] [1] [2] [3] [4] [5]
           ↓ (skip 2 frames)
Clip 1:             [3] [4] [5] [6] [7] [8]

Can't start at [6]: Not enough frames left
```

---

## Frame Interval Example
**`num_frames=6, frame_interval=2, window_stride=1`**

```
Available images: [0] [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12] [13]

Required frames = (6-1) × 2 + 1 = 11 frames needed

Clip 0: [0] - [2] - [4] - [6] - [8] - [10]  (every 2nd frame)
           ↓
Clip 1:     [1] - [3] - [5] - [7] - [9] - [11]
               ↓
Clip 2:         [2] - [4] - [6] - [8] - [10] - [12]
                   ↓
Clip 3:             [3] - [5] - [7] - [9] - [11] - [13]

Can't start at [4]: Need frames [4,6,8,10,12,14] but only have up to [13]
```

---

## Real-World Example

### Your Dataset
```
train/
  ├── video1/       (10 images)
  ├── video2/       (15 images)
  └── video3/       (8 images)   ← Too short! Need 6 frames
```

### Configuration
```python
num_frames = 6
frame_interval = 1
window_stride = 1
```

### Results

#### Legacy Mode (`use_sliding_window=False`)
```
Total dataset: 3 samples
  - video1: 1 sample (random each epoch)
  - video2: 1 sample (random each epoch)
  - video3: 1 sample (random each epoch)
```

#### Sliding Window (`use_sliding_window=True, stride=1`)
```
Total dataset: 15 clips
  - video1 (10 imgs): 5 clips  [0-5], [1-6], [2-7], [3-8], [4-9]
  - video2 (15 imgs): 10 clips [0-5], [1-6], ..., [9-14]
  - video3 (8 imgs):  SKIPPED! (only 8 images < 6 required)

If video3 had 10 images: Total would be 20 clips
```

#### Sliding Window (`use_sliding_window=True, stride=2`)
```
Total dataset: 8 clips
  - video1 (10 imgs): 3 clips  [0-5], [2-7], [4-9]
  - video2 (15 imgs): 5 clips  [0-5], [2-7], [4-9], [6-11], [8-13]
  - video3 (8 imgs):  SKIPPED!
```

---

## When to Use Each Mode

### Stride = 1 (Maximum Data)
✅ **Best for**: Small datasets, overfitting prevention
✅ **Pros**: Maximum training data, smooth temporal learning
⚠️ **Cons**: Slower training (more clips to process)

### Stride = 2-3 (Balanced)
✅ **Best for**: Medium datasets, faster training
✅ **Pros**: Good data augmentation, reasonable speed
⚠️ **Cons**: Less total data than stride=1

### Legacy Mode (No Sliding Window)
✅ **Best for**: Testing, debugging, very large datasets
✅ **Pros**: Fastest, simplest
⚠️ **Cons**: Wastes potential training data

---

## Summary Table

| Images | num_frames | stride | Clips | Multiplier |
|--------|------------|--------|-------|------------|
| 10     | 6          | 1      | 5     | 5x         |
| 10     | 6          | 2      | 3     | 3x         |
| 10     | 6          | 3      | 2     | 2x         |
| 15     | 6          | 1      | 10    | 10x        |
| 20     | 8          | 1      | 13    | 13x        |
| 20     | 8          | 2      | 7     | 7x         |

**Formula**: `clips = floor((total_frames - required_frames) / stride) + 1`

Where: `required_frames = (num_frames - 1) × frame_interval + 1`
