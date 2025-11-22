import torch
from models.diffusion import VideoDiffusionUNet

def test_flexible_frames():
    print("Testing flexible frame counts...")
    
    # Standard config: 4 downsample layers -> multiple of 16 required
    model = VideoDiffusionUNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=128
    )
    model.eval()
    
    # Test cases: (Batch, Channels, Frames, Height, Width)
    test_shapes = [
        (1, 3, 10, 64, 64),  # User's case: 8 context + 2 future = 10
        (1, 3, 9, 64, 64),   # Odd number
        (1, 3, 16, 64, 64),  # Perfect multiple (should still work)
        (1, 3, 1, 64, 64),   # Single frame
    ]
    
    for shape in test_shapes:
        print(f"\nTesting input shape: {shape}")
        x = torch.randn(shape)
        t = torch.randint(0, 1000, (shape[0],))
        
        try:
            output = model(x, t)
            print(f"Success! Output shape: {output.shape}")
            assert output.shape == shape, f"Output shape mismatch! Expected {shape}, got {output.shape}"
        except Exception as e:
            print(f"FAILED with error: {e}")
            raise e

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_flexible_frames()
