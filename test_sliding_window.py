"""
Test script to demonstrate sliding window data augmentation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from data.video_dataset import VideoDataset

def test_sliding_window():
    """Test sliding window clip generation"""

    print("="*80)
    print("SLIDING WINDOW DATA AUGMENTATION TEST")
    print("="*80)

    # Example parameters
    video_dir = "./data/train"  # Replace with your actual data directory
    num_frames = 6
    frame_interval = 1
    window_stride = 1

    print(f"\nConfiguration:")
    print(f"  num_frames: {num_frames}")
    print(f"  frame_interval: {frame_interval}")
    print(f"  window_stride: {window_stride}")
    print(f"  video_dir: {video_dir}")

    # Test with sliding window ENABLED
    print("\n" + "-"*80)
    print("MODE 1: SLIDING WINDOW ENABLED (use_sliding_window=True)")
    print("-"*80)

    try:
        dataset_sliding = VideoDataset(
            video_dir=video_dir,
            num_frames=num_frames,
            frame_interval=frame_interval,
            use_sliding_window=True,
            window_stride=window_stride,
            mode="train"
        )

        print(f"\nDataset length: {len(dataset_sliding)} clips")
        print("\nExample: If you have 10 images in a sequence:")
        print("  With num_frames=6, frame_interval=1, stride=1:")
        print("    Clip 0: frames [0, 1, 2, 3, 4, 5]")
        print("    Clip 1: frames [1, 2, 3, 4, 5, 6]")
        print("    Clip 2: frames [2, 3, 4, 5, 6, 7]")
        print("    Clip 3: frames [3, 4, 5, 6, 7, 8]")
        print("    Clip 4: frames [4, 5, 6, 7, 8, 9]")
        print("    Total: 5 clips from 1 sequence")

    except Exception as e:
        print(f"Error: {e}")
        print("(This is expected if data directory doesn't exist)")

    # Test with sliding window DISABLED
    print("\n" + "-"*80)
    print("MODE 2: LEGACY MODE (use_sliding_window=False)")
    print("-"*80)

    try:
        dataset_legacy = VideoDataset(
            video_dir=video_dir,
            num_frames=num_frames,
            frame_interval=frame_interval,
            use_sliding_window=False,
            mode="train"
        )

        print(f"\nDataset length: {len(dataset_legacy)} sequences")
        print("\nExample: If you have 10 images in a sequence:")
        print("  Legacy mode: 1 sequence = 1 dataset sample")
        print("  Each epoch randomly picks a different starting position")

    except Exception as e:
        print(f"Error: {e}")
        print("(This is expected if data directory doesn't exist)")

    # Test with larger stride
    print("\n" + "-"*80)
    print("MODE 3: SLIDING WINDOW WITH STRIDE=2")
    print("-"*80)

    try:
        dataset_stride2 = VideoDataset(
            video_dir=video_dir,
            num_frames=num_frames,
            frame_interval=frame_interval,
            use_sliding_window=True,
            window_stride=2,  # Less overlap
            mode="train"
        )

        print(f"\nDataset length: {len(dataset_stride2)} clips")
        print("\nExample: If you have 10 images in a sequence:")
        print("  With num_frames=6, frame_interval=1, stride=2:")
        print("    Clip 0: frames [0, 1, 2, 3, 4, 5]")
        print("    Clip 1: frames [2, 3, 4, 5, 6, 7]")
        print("    Clip 2: frames [4, 5, 6, 7, 8, 9]")
        print("    Total: 3 clips from 1 sequence (less overlap)")

    except Exception as e:
        print(f"Error: {e}")
        print("(This is expected if data directory doesn't exist)")

    print("\n" + "="*80)
    print("KEY BENEFITS:")
    print("="*80)
    print("✓ More training data from same image sequences")
    print("✓ Better temporal learning (overlapping clips)")
    print("✓ Configurable stride to balance data size vs diversity")
    print("✓ Automatic skipping of sequences too short")
    print("✓ Works for both training and validation")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_sliding_window()
