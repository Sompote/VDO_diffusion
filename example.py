"""
Example script demonstrating how to use the Video Diffusion Model
"""

import torch
from pathlib import Path

from models.diffusion import VideoDiffusionUNet, GaussianDiffusion
from data.video_dataset import create_video_dataloader, create_prediction_dataloader


def example_model_creation():
    """Example: Create a video diffusion model"""
    print("=" * 60)
    print("Example 1: Creating a Video Diffusion Model")
    print("=" * 60)

    # Create U-Net model
    unet = VideoDiffusionUNet(
        in_channels=3,  # RGB videos
        out_channels=3,
        base_channels=64,  # Base number of channels
        channel_mults=(1, 2, 4, 8),  # Channel multipliers for each level
        time_emb_dim=256,  # Time embedding dimension
        num_frames=16,  # Number of frames
    )

    # Wrap in Gaussian Diffusion
    model = GaussianDiffusion(
        model=unet,
        num_timesteps=1000,  # Number of diffusion steps
        beta_start=1e-4,  # Noise schedule start
        beta_end=0.02,  # Noise schedule end
        schedule="linear",  # or 'cosine'
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    print(f"Model size: {num_params * 4 / 1024 / 1024:.2f} MB (float32)")

    return model


def example_forward_pass():
    """Example: Forward pass through the model"""
    print("\n" + "=" * 60)
    print("Example 2: Forward Pass (Training)")
    print("=" * 60)

    # Create model
    model = example_model_creation()
    model = model.cuda() if torch.cuda.is_available() else model

    # Create random video input (batch_size, channels, time, height, width)
    batch_size = 2
    num_frames = 16
    height, width = 128, 128

    video = torch.randn(batch_size, 3, num_frames, height, width)
    video = video.cuda() if torch.cuda.is_available() else video

    print(f"\nInput shape: {video.shape}")

    # Forward pass (computes loss)
    model.train()
    loss = model(video)

    print(f"Training loss: {loss.item():.4f}")
    print("Forward pass completed successfully!")


def example_sampling():
    """Example: Generate videos from noise"""
    print("\n" + "=" * 60)
    print("Example 3: Video Generation (Sampling)")
    print("=" * 60)

    # Create model
    model = example_model_creation()
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()

    # Sample videos
    batch_size = 2
    num_frames = 16
    height, width = 128, 128

    print(f"\nGenerating {batch_size} videos with {num_frames} frames...")

    with torch.no_grad():
        shape = (batch_size, 3, num_frames, height, width)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # This will take some time as it iterates through all timesteps
        generated_videos = model.sample(shape, device)

    print(f"Generated videos shape: {generated_videos.shape}")
    print("Sampling completed successfully!")

    return generated_videos


def example_video_prediction():
    """Example: Predict future frames"""
    print("\n" + "=" * 60)
    print("Example 4: Video Prediction")
    print("=" * 60)

    # Create model
    model = example_model_creation()
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()

    # Create context frames (past frames)
    batch_size = 1
    num_context_frames = 8
    num_future_frames = 8
    height, width = 128, 128

    context_frames = torch.randn(batch_size, 3, num_context_frames, height, width)
    context_frames = (
        context_frames.cuda() if torch.cuda.is_available() else context_frames
    )

    print(f"\nContext frames shape: {context_frames.shape}")
    print(f"Predicting {num_future_frames} future frames...")

    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        future_frames = model.predict_video(context_frames, num_future_frames, device)

    print(f"Predicted frames shape: {future_frames.shape}")
    print("Prediction completed successfully!")

    return future_frames


def example_dataloader():
    """Example: Create a video dataloader"""
    print("\n" + "=" * 60)
    print("Example 5: Creating Video Dataloader")
    print("=" * 60)

    # Note: This requires actual video files in the specified directory
    video_dir = "./data/videos"  # Change this to your video directory

    print(f"Video directory: {video_dir}")
    print("Note: Make sure you have videos in this directory!")

    # Check if directory exists
    if not Path(video_dir).exists():
        print(f"\nWarning: Directory {video_dir} does not exist!")
        print("Create the directory and add some video files to use this example.")
        return None

    try:
        # Create dataloader
        dataloader = create_video_dataloader(
            video_dir=video_dir,
            batch_size=4,
            num_frames=16,
            frame_size=(256, 256),
            frame_interval=1,
            mode="train",
            num_workers=2,
            augment=True,
        )

        print(f"\nDataloader created successfully!")
        print(f"Number of batches: {len(dataloader)}")

        # Get a batch
        for batch in dataloader:
            print(f"Batch shape: {batch.shape}")
            break

        return dataloader

    except Exception as e:
        print(f"\nError creating dataloader: {e}")
        print("Make sure you have video files in the specified directory.")
        return None


def example_prediction_dataloader():
    """Example: Create a prediction dataloader"""
    print("\n" + "=" * 60)
    print("Example 6: Creating Prediction Dataloader")
    print("=" * 60)

    video_dir = "./data/videos"

    print(f"Video directory: {video_dir}")

    if not Path(video_dir).exists():
        print(f"\nWarning: Directory {video_dir} does not exist!")
        return None

    try:
        # Create prediction dataloader (returns context and future frames)
        dataloader = create_prediction_dataloader(
            video_dir=video_dir,
            batch_size=2,
            context_frames=8,
            future_frames=8,
            frame_size=(256, 256),
            mode="train",
            num_workers=2,
        )

        print(f"\nPrediction dataloader created successfully!")
        print(f"Number of batches: {len(dataloader)}")

        # Get a batch
        for context, future in dataloader:
            print(f"Context frames shape: {context.shape}")
            print(f"Future frames shape: {future.shape}")
            break

        return dataloader

    except Exception as e:
        print(f"\nError creating dataloader: {e}")
        return None


def main():
    """Run all examples"""
    print("\n" + "#" * 60)
    print("# Video Diffusion Model - Usage Examples")
    print("#" * 60)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"\nCUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    else:
        print("\nCUDA is not available. Using CPU.")
        print("Note: Examples will run slower on CPU.")

    # Run examples
    try:
        # Basic examples (always runnable)
        example_model_creation()
        example_forward_pass()

        # These are slower - uncomment if you want to run them
        # example_sampling()  # Takes time to generate
        # example_video_prediction()  # Takes time to generate

        # Data loading examples (require video files)
        example_dataloader()
        example_prediction_dataloader()

    except Exception as e:
        print(f"\n{'!' * 60}")
        print(f"Error: {e}")
        print(f"{'!' * 60}")
        import traceback

        traceback.print_exc()

    print("\n" + "#" * 60)
    print("# Examples completed!")
    print("#" * 60)


if __name__ == "__main__":
    main()
