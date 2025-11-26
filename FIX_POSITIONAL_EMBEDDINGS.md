# Fix for Scrambled Patch Issue

## Root Cause
The DiT model uses 1D positional embeddings that don't explicitly encode 2D spatial structure (row, column positions). This causes the model to not learn proper spatial relationships between patches, resulting in scrambled output.

## Current Implementation (BROKEN)
```python
# Line 512-513 in models/advanced_diffusion.py
self.pos_embed_spatial = nn.Parameter(
    torch.randn(1, 1, self.num_patches_per_frame, hidden_dim) * 0.02
)
```
This creates a single embedding for each of the 256 patches, but doesn't encode WHERE each patch is in 2D space.

## Solution: Use 2D Positional Embeddings

### Option 1: Separate Row and Column Embeddings (Recommended)
```python
# Calculate patch grid dimensions
self.num_patch_rows = img_size // patch_size[0]  # 16
self.num_patch_cols = img_size // patch_size[1]  # 16

# Separate embeddings for rows and columns
self.pos_embed_row = nn.Parameter(
    torch.randn(1, 1, self.num_patch_rows, 1, hidden_dim) * 0.02
)
self.pos_embed_col = nn.Parameter(
    torch.randn(1, 1, 1, self.num_patch_cols, hidden_dim) * 0.02
)

# In forward():
# Reshape patches to 2D grid: (B, T, H_patches, W_patches, D)
B, T, N, D = x.shape
h = w = int(N**0.5)
x = x.view(B, T, h, w, D)

# Add 2D positional embeddings
x = x + self.pos_embed_row + self.pos_embed_col

# Flatten back: (B, T, N, D)
x = x.view(B, T, N, D)
```

### Option 2: 2D Sinusoidal Embeddings (More Standard)
```python
def get_2d_sincos_pos_embed(h, w, embed_dim):
    """Generate 2D sine-cosine positional embeddings"""
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)  # (2, H, W)

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = torch.cat([emb_h, emb_w], dim=-1)  # (H, W, embed_dim)
    return emb

# In __init__():
pos_embed = get_2d_sincos_pos_embed(
    self.num_patch_rows,
    self.num_patch_cols,
    hidden_dim
)
self.pos_embed_spatial = nn.Parameter(
    pos_embed.view(1, 1, -1, hidden_dim),
    requires_grad=False  # Fixed, not learned
)
```

## Implementation Steps

1. **Modify `models/advanced_diffusion.py`**:
   - Replace 1D positional embeddings with 2D embeddings
   - Update forward pass to add embeddings correctly

2. **Retrain the model**:
   - The old checkpoint won't work with new embeddings
   - Need to train from scratch
   - Should see coherent (non-scrambled) outputs

3. **Alternative Quick Fix** (if don't want to retrain):
   - Keep architecture but train longer
   - Add stronger spatial losses
   - Use data augmentation that preserves spatial structure

## Why This Fixes The Issue

With 2D positional embeddings:
- Each patch knows its (row, col) position explicitly
- The model learns "patch at position (5, 10) should have road content"
- The unpatchify operation correctly reconstructs spatial layout
- No more scrambled patches!

Without 2D embeddings:
- Patches only have arbitrary indices (0-255)
- Model doesn't know spatial relationships
- Can learn to output patches in any order
- Results in jigsaw-like scrambled output

## Testing

After implementing the fix:
```bash
# Test that embeddings work
python test_2d_pos_embeddings.py

# Train new model
python train_advanced.py --config config_advanced.yaml

# Generate predictions
python predict_advanced.py --config predict_advanced.yaml
```

Expected result: Clean, coherent video frames instead of scrambled patches.
