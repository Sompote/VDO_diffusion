"""
Inference script for Video Diffusion Model
Generate future video frames from context frames
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml

sys.path.append(str(Path(__file__).parent))

from models.diffusion import GaussianDiffusion, VideoDiffusionUNet


DEFAULT_MODEL_CFG: Dict[str, Any] = {
    "in_channels": 3,
    "out_channels": 3,
    "base_channels": 64,
    "channel_mults": [1, 2, 4, 8],
    "time_emb_dim": 256,
    "num_frames": 16,
}

DEFAULT_DIFFUSION_CFG: Dict[str, Any] = {
    "num_timesteps": 1000,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "schedule": "linear",
}

DEFAULT_RUNTIME_CFG: Dict[str, Any] = {
    "context_frames": 8,
    "future_frames": 8,
    "frame_size": (256, 256),
    "num_frames": 16,
    "batch_size": 1,
    "output_dir": "./outputs",
    "output_name": "prediction",
    "device": "cuda",
}


def load_mapping_file(path: Path) -> Dict[str, Any]:
    """Load a configuration mapping from JSON or YAML."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("r") as f:
            return yaml.safe_load(f) or {}
    if path.suffix.lower() == ".json":
        with path.open("r") as f:
            return json.load(f)

    raise ValueError(f"Unsupported configuration format: {path.suffix}")


def get_nested(config: Dict[str, Any], path: Sequence[str]) -> Any:
    """Safely fetch nested values from mapping configurations."""
    current: Any = config
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def parse_frame_size(value: Any, fallback: Sequence[int]) -> Tuple[int, int]:
    """Normalise frame size into (H, W)."""
    if value is None:
        return tuple(int(v) for v in fallback)  # type: ignore[return-value]

    if isinstance(value, (int, float)):
        size = int(value)
        return (size, size)

    if isinstance(value, str):
        parts = [part.strip() for part in value.replace("x", " ").split()]
        elems = [int(p) for p in parts if p]
    elif isinstance(value, Iterable):
        elems = [int(v) for v in value]
    else:
        raise ValueError(f"Unsupported frame size value: {value}")

    if len(elems) == 1:
        return (elems[0], elems[0])
    if len(elems) >= 2:
        return (elems[0], elems[1])

    return tuple(int(v) for v in fallback)  # type: ignore[return-value]


def parse_channel_mults(value: Any, fallback: Sequence[int]) -> Sequence[int]:
    """Convert channel multiplier configuration into integer list."""
    if value is None:
        return list(fallback)

    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(",", " ").split()]
        integers = [int(part) for part in parts if part]
        if not integers:
            raise ValueError("channel_mults string did not contain any integers.")
        return integers

    if isinstance(value, Iterable):
        integers = [int(part) for part in value]
        if not integers:
            raise ValueError("channel_mults iterable cannot be empty.")
        return integers

    raise ValueError(f"Unsupported channel_mults value: {value}")


def ensure_path(path_str: Optional[str], search_dirs: Sequence[Path]) -> Optional[Path]:
    """Resolve a file path trying a list of base directories."""
    if not path_str:
        return None

    candidate = Path(path_str)
    if candidate.exists():
        return candidate

    for base in search_dirs:
        if base is None:
            continue
        test_path = (base / path_str).expanduser()
        if test_path.exists():
            return test_path.resolve()

    return None


def resolve_inference_namespace(args: argparse.Namespace) -> SimpleNamespace:
    """Merge CLI overrides with saved configuration files and sensible defaults."""
    # Load training config if available
    config_candidate = args.config
    config_data: Dict[str, Any] = {}
    config_path: Optional[Path] = None

    if config_candidate is not None:
        config_path = ensure_path(config_candidate, [Path.cwd()])
        if config_path is None:
            raise FileNotFoundError(f"Could not locate config file: {config_candidate}")
        config_data = load_mapping_file(config_path)

    # Resolve checkpoint path
    checkpoint_str = args.checkpoint or config_data.get("inference", {}).get("checkpoint") or config_data.get("checkpoint")
    
    if checkpoint_str is None:
        raise ValueError("Checkpoint must be specified via --checkpoint or in config file.")
        
    checkpoint_path = Path(checkpoint_str).resolve()
    ckpt_parent = checkpoint_path.parent

    # If config was not explicitly provided, try to find one near the checkpoint
    if config_candidate is None:
        default_config = ckpt_parent / "config.json"
        if default_config.exists():
            config_candidate = str(default_config)
            config_path = default_config
            config_data = load_mapping_file(config_path)

    # Load dataset YAML if available
    data_candidate = args.data or config_data.get("data_config_path")
    data_mapping: Dict[str, Any] = {}
    data_path: Optional[Path] = None
    if data_candidate:
        data_path = ensure_path(
            data_candidate, [ckpt_parent, config_path.parent if config_path else None]
        )  # type: ignore[arg-type]
        if data_path:
            data_mapping = load_mapping_file(data_path)
        elif args.data:
            raise FileNotFoundError(f"Could not locate dataset file: {args.data}")

    def first_non_none(*values):
        for value in values:
            if value is not None:
                return value
        return None

    # Runtime parameters
    context_frames = first_non_none(
        args.num_context_frames,
        config_data.get("context_frames"),
        get_nested(config_data, ("prediction", "num_context_frames")),
        data_mapping.get("context_frames"),
    )
    future_frames = first_non_none(
        args.num_future_frames,
        config_data.get("future_frames"),
        get_nested(config_data, ("prediction", "num_future_frames")),
        data_mapping.get("future_frames"),
    )

    if (context_frames is None) ^ (future_frames is None):
        raise ValueError(
            "Both context_frames and future_frames must be specified together, either via CLI or configuration files."
        )

    num_frames_override = first_non_none(
        args.num_frames,
        config_data.get("num_frames"),
        get_nested(config_data, ("video", "num_frames")),
        data_mapping.get("num_frames"),
    )
    if (
        num_frames_override is None
        and context_frames is not None
        and future_frames is not None
    ):
        num_frames_override = int(context_frames) + int(future_frames)

    frame_size_value = first_non_none(
        args.frame_size,
        config_data.get("frame_size"),
        get_nested(config_data, ("video", "frame_size")),
        data_mapping.get("frame_size"),
    )
    frame_size = parse_frame_size(
        frame_size_value,
        DEFAULT_RUNTIME_CFG["frame_size"],  # type: ignore[arg-type]
    )

    runtime = {
        "checkpoint": checkpoint_path,
        "device": args.device
        or config_data.get("device")
        or DEFAULT_RUNTIME_CFG["device"],
        "mode": args.mode,
        "input_video": first_non_none(
            args.input_video,
            get_nested(config_data, ("inference", "input_video")),
        ),
        "context_frames": int(context_frames)
        if context_frames is not None
        else DEFAULT_RUNTIME_CFG["context_frames"],
        "future_frames": int(future_frames)
        if future_frames is not None
        else DEFAULT_RUNTIME_CFG["future_frames"],
        "frame_size": frame_size,
        "num_frames": int(num_frames_override or DEFAULT_RUNTIME_CFG["num_frames"]),
        "batch_size": int(
            first_non_none(
                args.batch_size,
                config_data.get("batch_size"),
                get_nested(config_data, ("training", "batch_size")),
                DEFAULT_RUNTIME_CFG["batch_size"],
            )
        ),
        "output_dir": Path(
            first_non_none(
                args.output_dir,
                config_data.get("output_dir"),
                get_nested(config_data, ("data", "output_dir")),
                DEFAULT_RUNTIME_CFG["output_dir"],
            )
        ),
        "output_name": first_non_none(
            args.output_name, DEFAULT_RUNTIME_CFG["output_name"]
        ),
        "frame_interval": first_non_none(
            args.frame_interval,
            config_data.get("frame_interval"),
            get_nested(config_data, ("video", "frame_interval")),
            data_mapping.get("frame_interval"),
            1,
        ),
        "config_path": config_path,
        "data_path": data_path,
    }

    # Model-specific configuration
    model_cfg = dict(DEFAULT_MODEL_CFG)
    model_cfg["in_channels"] = int(
        first_non_none(
            config_data.get("in_channels"),
            get_nested(config_data, ("model", "in_channels")),
            DEFAULT_MODEL_CFG["in_channels"],
        )
    )
    model_cfg["out_channels"] = int(
        first_non_none(
            config_data.get("out_channels"),
            get_nested(config_data, ("model", "out_channels")),
            DEFAULT_MODEL_CFG["out_channels"],
        )
    )
    model_cfg["base_channels"] = int(
        first_non_none(
            args.base_channels,
            config_data.get("base_channels"),
            get_nested(config_data, ("model", "base_channels")),
            DEFAULT_MODEL_CFG["base_channels"],
        )
    )
    model_cfg["channel_mults"] = parse_channel_mults(
        first_non_none(
            args.channel_mults,
            config_data.get("channel_mults"),
            get_nested(config_data, ("model", "channel_mults")),
        ),
        DEFAULT_MODEL_CFG["channel_mults"],
    )
    model_cfg["time_emb_dim"] = int(
        first_non_none(
            args.time_emb_dim,
            config_data.get("time_emb_dim"),
            get_nested(config_data, ("model", "time_emb_dim")),
            DEFAULT_MODEL_CFG["time_emb_dim"],
        )
    )
    model_cfg["num_frames"] = int(runtime["num_frames"])

    diffusion_cfg = {
        **DEFAULT_DIFFUSION_CFG,
        "num_timesteps": first_non_none(
            args.num_timesteps,
            config_data.get("num_timesteps"),
            get_nested(config_data, ("diffusion", "num_timesteps")),
            DEFAULT_DIFFUSION_CFG["num_timesteps"],
        ),
        "beta_start": float(
            first_non_none(
                args.beta_start,
                config_data.get("beta_start"),
                get_nested(config_data, ("diffusion", "beta_start")),
                DEFAULT_DIFFUSION_CFG["beta_start"],
            )
        ),
        "beta_end": float(
            first_non_none(
                args.beta_end,
                config_data.get("beta_end"),
                get_nested(config_data, ("diffusion", "beta_end")),
                DEFAULT_DIFFUSION_CFG["beta_end"],
            )
        ),
        "schedule": first_non_none(
            args.schedule,
            config_data.get("schedule"),
            get_nested(config_data, ("diffusion", "schedule")),
            DEFAULT_DIFFUSION_CFG["schedule"],
        ),
    }

    runtime["model_cfg"] = model_cfg
    runtime["diffusion_cfg"] = diffusion_cfg

    # Normalise device selection
    device = runtime["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        runtime["device"] = "cpu"

    return SimpleNamespace(**runtime)


def load_model(
    checkpoint_path: Path,
    device: str,
    model_cfg: Dict[str, Any],
    diffusion_cfg: Dict[str, Any],
):
    """Load trained model from checkpoint with matching architecture."""
    map_location = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    channel_mults = tuple(int(x) for x in model_cfg["channel_mults"])
    unet = VideoDiffusionUNet(
        in_channels=int(model_cfg.get("in_channels", 3)),
        out_channels=int(model_cfg.get("out_channels", 3)),
        base_channels=int(model_cfg["base_channels"]),
        channel_mults=channel_mults,
        time_emb_dim=int(model_cfg["time_emb_dim"]),
        num_frames=int(model_cfg["num_frames"]),
    )

    model = GaussianDiffusion(
        model=unet,
        num_timesteps=int(diffusion_cfg["num_timesteps"]),
        beta_start=float(diffusion_cfg["beta_start"]),
        beta_end=float(diffusion_cfg["beta_end"]),
        schedule=str(diffusion_cfg["schedule"]),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(
        f"Epoch: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', float('nan')):.4f}"
    )

    return model


def load_video_frames(
    video_path: str,
    num_frames: int = 8,
    frame_size: Tuple[int, int] = (256, 256),
    frame_interval: int = 1,
):
    """
    Load frames from video file or directory of images

    Returns:
        Tensor of shape (1, C, T, H, W)
    """
    video_path_obj = Path(video_path)
    frames = []
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    if video_path_obj.is_dir():
        # Load from directory
        print(f"Loading frames from directory: {video_path}")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = sorted([
            f for f in video_path_obj.iterdir() 
            if f.suffix.lower() in image_extensions
        ])
        
        if not image_files:
            raise ValueError(f"No image files found in {video_path}")
            
        for i, img_path in enumerate(image_files):
            if len(frames) >= num_frames:
                break
                
            if frame_interval > 1 and i % frame_interval != 0:
                continue
                
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(frame))

    else:
        # Load from video file
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_index = 0
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_interval > 1 and frame_index % frame_interval != 0:
                frame_index += 1
                continue

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Transform
            frame_tensor = transform(frame)
            frames.append(frame_tensor)
            frame_index += 1

        cap.release()

    if len(frames) < num_frames:
        if not frames:
            raise ValueError(
                f"Could not extract frames from video `{video_path}`; please check file integrity."
            )
        last_frame = frames[-1]
        while len(frames) < num_frames:
            frames.append(last_frame)

    # Stack: (C, T, H, W)
    video_tensor = torch.stack(frames, dim=1)

    # Add batch dimension: (1, C, T, H, W)
    video_tensor = video_tensor.unsqueeze(0)

    return video_tensor


def tensor_to_video(tensor, output_path, fps=30):
    """
    Save tensor as video file

    Args:
        tensor: Tensor of shape (C, T, H, W) or (1, C, T, H, W)
        output_path: Output video file path
        fps: Frames per second
    """
    # Remove batch dimension if present
    if tensor.dim() == 5:
        tensor = tensor.squeeze(0)

    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy: (T, H, W, C)
    video_np = tensor.permute(1, 2, 3, 0).cpu().numpy()
    video_np = (video_np * 255).astype(np.uint8)

    # Get video properties
    num_frames, height, width, channels = video_np.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Write frames
    for i in range(num_frames):
        frame = video_np[i]
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Video saved to {output_path}")


@torch.no_grad()
def predict_from_video(
    model,
    video_path,
    num_context_frames=8,
    num_future_frames=8,
    frame_size=(256, 256),
    frame_interval=1,
    output_path="prediction.mp4",
    device="cuda",
):
    """
    Predict future frames from a video file

    Args:
        model: Trained diffusion model
        video_path: Path to input video
        num_context_frames: Number of context frames to use
        num_future_frames: Number of future frames to predict
        frame_size: Frame size (H, W)
        frame_interval: Sampling interval between frames
        output_path: Path to save predicted video
        device: Device to run on
    """
    # Load context frames
    context_frames = load_video_frames(
        video_path,
        num_frames=num_context_frames,
        frame_size=frame_size,
        frame_interval=frame_interval,
    )
    context_frames = context_frames.to(device)

    print(f"Context frames shape: {context_frames.shape}")

    # Predict future frames
    print("Generating future frames...")
    future_frames = model.predict_video(context_frames, num_future_frames, device)

    print(f"Generated frames shape: {future_frames.shape}")

    # Combine context and future frames
    full_video = torch.cat([context_frames, future_frames], dim=2)

    # Save as video
    tensor_to_video(full_video, output_path)

    # Also save context and prediction separately
    output_dir = Path(output_path).parent
    output_name = Path(output_path).stem

    tensor_to_video(context_frames, output_dir / f"{output_name}_context.mp4")
    tensor_to_video(future_frames, output_dir / f"{output_name}_prediction.mp4")

    return full_video


@torch.no_grad()
def generate_unconditional(
    model,
    num_frames=16,
    frame_size=(256, 256),
    batch_size=1,
    output_dir="./outputs",
    device="cuda",
):
    """
    Generate videos unconditionally from pure noise

    Args:
        model: Trained diffusion model
        num_frames: Number of frames to generate
        frame_size: Frame size (H, W)
        batch_size: Number of videos to generate
        output_dir: Directory to save generated videos
        device: Device to run on
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {batch_size} videos with {num_frames} frames...")

    # Generate from noise
    shape = (batch_size, 3, num_frames, *frame_size)
    videos = model.sample(shape, device)

    # Save each video
    for i in range(batch_size):
        video = videos[i : i + 1]
        output_path = output_dir / f"generated_{i:04d}.mp4"
        tensor_to_video(video, output_path)
        print(f"Saved: {output_path}")

    return videos


def create_comparison_video(
    context_path, prediction_path, ground_truth_path, output_path
):
    """
    Create a side-by-side comparison video

    Args:
        context_path: Path to context frames video
        prediction_path: Path to predicted frames video
        ground_truth_path: Path to ground truth video (optional)
        output_path: Path to save comparison video
    """
    # Read videos
    cap_context = cv2.VideoCapture(str(context_path))
    cap_pred = cv2.VideoCapture(str(prediction_path))
    cap_gt = cv2.VideoCapture(str(ground_truth_path)) if ground_truth_path else None

    # Get properties
    fps = int(cap_context.get(cv2.CAP_PROP_FPS))
    width = int(cap_context.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_context.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output video
    num_videos = 3 if cap_gt else 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width * num_videos, height))

    while True:
        ret_context, frame_context = cap_context.read()
        ret_pred, frame_pred = cap_pred.read()

        if not ret_context or not ret_pred:
            break

        # Add text labels
        cv2.putText(
            frame_context,
            "Context",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame_pred,
            "Prediction",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        if cap_gt:
            ret_gt, frame_gt = cap_gt.read()
            if ret_gt:
                cv2.putText(
                    frame_gt,
                    "Ground Truth",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                combined = np.hstack([frame_context, frame_pred, frame_gt])
            else:
                combined = np.hstack([frame_context, frame_pred])
        else:
            combined = np.hstack([frame_context, frame_pred])

        out.write(combined)

    # Release resources
    cap_context.release()
    cap_pred.release()
    if cap_gt:
        cap_gt.release()
    out.release()

    print(f"Comparison video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Video Diffusion Model Inference")

    parser.add_argument(
        "--checkpoint", type=str, required=False, help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config (JSON or YAML)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset YAML describing context/future frames",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Computation device override",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="predict",
        choices=["predict", "generate"],
        help="Inference mode",
    )
    parser.add_argument(
        "--input_video",
        type=str,
        default=None,
        help="Input video path (prediction mode)",
    )
    parser.add_argument(
        "--num_context_frames",
        type=int,
        default=None,
        help="Override number of context frames",
    )
    parser.add_argument(
        "--num_future_frames",
        type=int,
        default=None,
        help="Override number of future frames to predict",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=None,
        help="Frame sampling interval when reading the input video",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Total frames per clip (for generation mode)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of samples to generate in unconditional mode",
    )
    parser.add_argument(
        "--frame_size",
        type=int,
        nargs="+",
        default=None,
        help="Output frame size (single value or H W)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for generated videos",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Base filename for prediction output",
    )

    # Model/Diffusion overrides
    parser.add_argument(
        "--base_channels",
        type=int,
        default=None,
        help="Override base channel count",
    )
    parser.add_argument(
        "--channel_mults",
        type=int,
        nargs="+",
        default=None,
        help="Override channel multipliers (space separated list)",
    )
    parser.add_argument(
        "--time_emb_dim",
        type=int,
        default=None,
        help="Override time embedding dimension",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=None,
        help="Override diffusion timestep count",
    )
    parser.add_argument(
        "--beta_start",
        type=float,
        default=None,
        help="Override diffusion beta start",
    )
    parser.add_argument(
        "--beta_end",
        type=float,
        default=None,
        help="Override diffusion beta end",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default=None,
        choices=["linear", "cosine"],
        help="Override diffusion schedule",
    )

    args = parser.parse_args()
    settings = resolve_inference_namespace(args)

    settings.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(
        checkpoint_path=settings.checkpoint,
        device=settings.device,
        model_cfg=settings.model_cfg,
        diffusion_cfg=settings.diffusion_cfg,
    )

    if settings.mode == "predict":
        if not settings.input_video:
            raise ValueError("Prediction mode requires --input_video to be specified.")

        output_path = settings.output_dir / f"{settings.output_name}.mp4"
        predict_from_video(
            model=model,
            video_path=settings.input_video,
            num_context_frames=settings.context_frames,
            num_future_frames=settings.future_frames,
            frame_size=settings.frame_size,
            frame_interval=int(settings.frame_interval),
            output_path=output_path,
            device=settings.device,
        )
    else:
        generate_unconditional(
            model=model,
            num_frames=settings.num_frames,
            frame_size=settings.frame_size,
            batch_size=settings.batch_size,
            output_dir=settings.output_dir,
            device=settings.device,
        )

    print("\nInference completed!")


if __name__ == "__main__":
    main()
