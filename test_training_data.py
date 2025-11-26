"""Check if training data is being loaded correctly"""
import torch
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent))

from data.video_dataset import create_video_dataloader

def check_training_data():
    """Load and visualize training data"""

    train_loader = create_video_dataloader(
        video_dir="/workspace/data/train_videos",
        batch_size=1,
        num_frames=6,
        frame_size=(256, 256),
        frame_interval=1,
        mode="train",
        num_workers=0,
        augment=False,
        use_sliding_window=True,
        window_stride=1,
    )

    print(f"Dataset size: {len(train_loader.dataset)}")

    # Get first batch
    for batch_idx, videos in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Shape: {videos.shape}")  # Should be (1, 3, 6, 256, 256)

        # Save first frame
        frame = videos[0, :, 0, :, :]  # (3, 256, 256)
        frame = (frame + 1.0) / 2.0
        frame = torch.clamp(frame, 0, 1)
        frame_np = frame.permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        Path("./outputs").mkdir(exist_ok=True)
        cv2.imwrite(f"./outputs/training_data_frame_{batch_idx}.png", frame_bgr)
        print(f"  Saved to: ./outputs/training_data_frame_{batch_idx}.png")

        if batch_idx >= 2:
            break

    print("\n✅ Check the saved frames - are they scrambled or normal?")

if __name__ == "__main__":
    check_training_data()
