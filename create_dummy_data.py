import os
import cv2
import numpy as np
from pathlib import Path
import shutil

def create_dummy_image_sequence(path, frames=16, size=(256, 256)):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    
    for i in range(frames):
        # Create random frame
        frame = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        # Save as png
        cv2.imwrite(str(path / f"frame_{i:04d}.png"), frame)

# Create train and val directories
train_dir = Path("./data/train_videos")
val_dir = Path("./data/val_videos")

# Clean up existing data to ensure we only have images
if train_dir.exists():
    shutil.rmtree(train_dir)
if val_dir.exists():
    shutil.rmtree(val_dir)

train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

# Create dummy image sequences
print("Creating dummy image sequences...")
for i in range(4):
    create_dummy_image_sequence(train_dir / f"seq_{i}")
    
for i in range(2):
    create_dummy_image_sequence(val_dir / f"seq_{i}")

print("Done!")
