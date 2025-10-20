"""
Training script for Video Diffusion Model
Supports multi-GPU training with PyTorch DDP
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from models.diffusion import VideoDiffusionUNet, GaussianDiffusion
from data.video_dataset import VideoDataset, VideoPredictionDataset


DEFAULTS: Dict[str, Any] = {
    "output_dir": "./runs/train",
    "batch_size": 4,
    "epochs": 100,
    "num_frames": 16,
    "context_frames": None,
    "future_frames": None,
    "frame_size": (256, 256),
    "frame_interval": 1,
    "base_channels": 64,
    "channel_mults": [1, 2, 4, 8],
    "time_emb_dim": 256,
    "num_timesteps": 1000,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "schedule": "linear",
    "lr": 2e-4,
    "weight_decay": 0.01,
    "num_workers": 4,
    "save_interval": 10,
    "gpus": 1,
    "train_augment": True,
    "val_augment": False,
}

CONFIG_KEY_MAP: Dict[str, Sequence[Sequence[str]]] = {
    "train_dir": (("data", "train_dir"),),
    "val_dir": (("data", "val_dir"),),
    "output_dir": (
        ("data", "output_dir"),
        ("output", "dir"),
        ("output", "output_dir"),
    ),
    "batch_size": (("training", "batch_size"),),
    "epochs": (("training", "epochs"),),
    "lr": (("training", "lr"),),
    "weight_decay": (("training", "weight_decay"),),
    "num_workers": (("training", "num_workers"),),
    "save_interval": (("training", "save_interval"),),
    "base_channels": (("model", "base_channels"),),
    "channel_mults": (("model", "channel_mults"),),
    "time_emb_dim": (("model", "time_emb_dim"),),
    "num_frames": (("video", "num_frames"),),
    "frame_size": (("video", "frame_size"),),
    "frame_interval": (("video", "frame_interval"),),
    "num_timesteps": (("diffusion", "num_timesteps"),),
    "beta_start": (("diffusion", "beta_start"),),
    "beta_end": (("diffusion", "beta_end"),),
    "schedule": (("diffusion", "schedule"),),
    "gpus": (("gpu", "gpus"),),
    "context_frames": (("prediction", "num_context_frames"),),
    "future_frames": (("prediction", "num_future_frames"),),
}

DATA_KEY_MAP: Dict[str, Sequence[str]] = {
    "train_dir": ("train", "train_dir"),
    "val_dir": ("val", "val_dir"),
    "test_dir": ("test", "test_dir"),
    "context_frames": (
        "context_frames",
        "context",
        "input_frames",
        "input",
    ),
    "future_frames": (
        "future_frames",
        "future",
        "target_frames",
        "output_frames",
        "prediction_frames",
    ),
    "num_frames": ("num_frames", "sequence_length", "total_frames"),
    "frame_size": ("frame_size", "imgsz", "img_size"),
    "frame_interval": ("frame_interval", "interval", "stride"),
    "train_augment": ("augment", "train_augment"),
    "val_augment": ("val_augment",),
}


def load_yaml_file(path: Optional[str]) -> Dict[str, Any]:
    """Load YAML configuration as dictionary"""
    if path is None:
        return {}

    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with yaml_path.open("r") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of YAML file: {path}")

    return data


def get_nested(config: Dict[str, Any], path: Sequence[str]) -> Any:
    """Safely get nested config value"""
    current = config
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def first_non_none(*values):
    """Return the first non-None value"""
    for value in values:
        if value is not None:
            return value
    return None


def parse_frame_size(value: Any, fallback: Sequence[int]) -> Sequence[int]:
    """Normalize frame size configuration to (H, W) tuple"""
    if value is None:
        return tuple(int(v) for v in fallback)

    if isinstance(value, (int, float)):
        size = int(value)
        return (size, size)

    if isinstance(value, str):
        parts = [int(part.strip()) for part in value.replace("x", " ").split()]
    elif isinstance(value, Iterable):
        parts = [int(part) for part in value]
    else:
        raise ValueError(f"Unsupported frame_size value: {value}")

    if len(parts) == 1:
        return (parts[0], parts[0])
    if len(parts) >= 2:
        return (parts[0], parts[1])

    return tuple(int(v) for v in fallback)


def parse_channel_mults(value: Any, fallback: Sequence[int]) -> Sequence[int]:
    """Normalize channel multipliers list"""
    if value is None:
        return list(fallback)

    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(",", " ").split()]
        return [int(part) for part in parts if part]

    if isinstance(value, Iterable):
        return [int(part) for part in value]

    raise ValueError(f"Unsupported channel_mults value: {value}")


def resolve_data_path(
    value: Optional[Any], data_yaml_path: Optional[str]
) -> Optional[str]:
    """Resolve dataset path relative to YAML file location"""
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        raise ValueError("Dataset paths must be a single string, not a sequence.")

    raw_path = Path(str(value))
    if raw_path.is_absolute() or not data_yaml_path:
        return str(raw_path)

    base_dir = Path(data_yaml_path).parent
    return str((base_dir / raw_path).resolve())


def combine_video_batch(batch: Any) -> torch.Tensor:
    """
    Accept dataset batches that may be full videos or (context, future) tuples
    and return combined tensor shaped (B, C, T, H, W)
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) != 2:
            raise ValueError(
                "Expected dataset batch as (context, future) pair when using prediction dataset."
            )
        context, future = batch
        return torch.cat([context, future], dim=2)
    return batch


def build_training_namespace(parsed_args: argparse.Namespace) -> SimpleNamespace:
    """
    Merge CLI options, YAML configuration, and defaults into a namespace object
    used throughout training.
    """
    config_yaml = load_yaml_file(getattr(parsed_args, "config", None))
    data_yaml = load_yaml_file(getattr(parsed_args, "data", None))

    final: Dict[str, Any] = dict(DEFAULTS)
    cli_dict = vars(parsed_args)

    def cli_value(name: str) -> Any:
        value = cli_dict.get(name)
        if isinstance(value, list) and len(value) == 0:
            return None
        return value

    # Resolve data-related paths
    train_dir = first_non_none(
        cli_value("train_dir"),
        *(
            get_nested(config_yaml, path)
            for path in CONFIG_KEY_MAP.get("train_dir", ())
        ),
        *(data_yaml.get(key) for key in DATA_KEY_MAP.get("train_dir", ())),
    )
    val_dir = first_non_none(
        cli_value("val_dir"),
        *(get_nested(config_yaml, path) for path in CONFIG_KEY_MAP.get("val_dir", ())),
        *(data_yaml.get(key) for key in DATA_KEY_MAP.get("val_dir", ())),
    )
    test_dir = first_non_none(
        *(data_yaml.get(key) for key in DATA_KEY_MAP.get("test_dir", ()))
    )

    if train_dir is None:
        raise ValueError(
            "Training data directory must be provided either via --train_dir or the dataset YAML (train key)."
        )

    final["train_dir"] = resolve_data_path(train_dir, cli_dict.get("data"))
    final["val_dir"] = (
        resolve_data_path(val_dir, cli_dict.get("data"))
        if val_dir is not None
        else None
    )
    final["test_dir"] = (
        resolve_data_path(test_dir, cli_dict.get("data"))
        if test_dir is not None
        else None
    )

    # Resolve scalar hyperparameters
    for key in (
        "output_dir",
        "batch_size",
        "epochs",
        "lr",
        "weight_decay",
        "num_workers",
        "save_interval",
        "num_timesteps",
        "beta_start",
        "beta_end",
        "schedule",
        "gpus",
        "frame_interval",
    ):
        final[key] = first_non_none(
            cli_value(key),
            *(get_nested(config_yaml, path) for path in CONFIG_KEY_MAP.get(key, ())),
            *(data_yaml.get(dkey) for dkey in DATA_KEY_MAP.get(key, ())),
            DEFAULTS.get(key),
        )

    # Model architecture specifics
    for key in (
        "batch_size",
        "epochs",
        "num_workers",
        "save_interval",
        "num_timesteps",
        "gpus",
        "frame_interval",
    ):
        if key in final and final[key] is not None:
            final[key] = int(final[key])

    for key in ("lr", "weight_decay", "beta_start", "beta_end"):
        if key in final and final[key] is not None:
            final[key] = float(final[key])

    final["base_channels"] = first_non_none(
        cli_value("base_channels"),
        *(
            get_nested(config_yaml, path)
            for path in CONFIG_KEY_MAP.get("base_channels", ())
        ),
        DEFAULTS["base_channels"],
    )
    final["channel_mults"] = parse_channel_mults(
        first_non_none(
            cli_value("channel_mults"),
            *(
                get_nested(config_yaml, path)
                for path in CONFIG_KEY_MAP.get("channel_mults", ())
            ),
        ),
        DEFAULTS["channel_mults"],
    )
    if not final["channel_mults"]:
        raise ValueError("channel_mults configuration cannot be empty.")
    final["time_emb_dim"] = first_non_none(
        cli_value("time_emb_dim"),
        *(
            get_nested(config_yaml, path)
            for path in CONFIG_KEY_MAP.get("time_emb_dim", ())
        ),
        DEFAULTS["time_emb_dim"],
    )
    final["base_channels"] = int(final["base_channels"])
    final["time_emb_dim"] = int(final["time_emb_dim"])

    # Frame / temporal configuration
    frame_size_value = first_non_none(
        cli_value("frame_size"),
        *(
            get_nested(config_yaml, path)
            for path in CONFIG_KEY_MAP.get("frame_size", ())
        ),
        *(data_yaml.get(dkey) for dkey in DATA_KEY_MAP.get("frame_size", ())),
    )
    final["frame_size"] = parse_frame_size(frame_size_value, DEFAULTS["frame_size"])

    context_frames = first_non_none(
        cli_value("context_frames"),
        *(
            get_nested(config_yaml, path)
            for path in CONFIG_KEY_MAP.get("context_frames", ())
        ),
        *(data_yaml.get(dkey) for dkey in DATA_KEY_MAP.get("context_frames", ())),
    )
    future_frames = first_non_none(
        cli_value("future_frames"),
        *(
            get_nested(config_yaml, path)
            for path in CONFIG_KEY_MAP.get("future_frames", ())
        ),
        *(data_yaml.get(dkey) for dkey in DATA_KEY_MAP.get("future_frames", ())),
    )

    if context_frames is not None:
        final["context_frames"] = int(context_frames)
    if future_frames is not None:
        final["future_frames"] = int(future_frames)

    if (final.get("context_frames") is None) ^ (final.get("future_frames") is None):
        raise ValueError(
            "Both context_frames and future_frames must be provided together when specifying sequence splits."
        )

    # Determine clip length
    num_frames_value = first_non_none(
        cli_value("num_frames"),
        *(
            get_nested(config_yaml, path)
            for path in CONFIG_KEY_MAP.get("num_frames", ())
        ),
        *(data_yaml.get(dkey) for dkey in DATA_KEY_MAP.get("num_frames", ())),
        DEFAULTS["num_frames"],
    )
    if (
        final.get("context_frames") is not None
        and final.get("future_frames") is not None
    ):
        final["num_frames"] = int(final["context_frames"] + final["future_frames"])
    else:
        final["num_frames"] = int(num_frames_value)

    # Augmentation flags
    train_aug_value = first_non_none(
        cli_value("train_augment"),
        *(data_yaml.get(dkey) for dkey in DATA_KEY_MAP.get("train_augment", ())),
        DEFAULTS["train_augment"],
    )
    val_aug_value = first_non_none(
        cli_value("val_augment"),
        *(data_yaml.get(dkey) for dkey in DATA_KEY_MAP.get("val_augment", ())),
        DEFAULTS["val_augment"],
    )
    final["train_augment"] = bool(train_aug_value)
    final["val_augment"] = bool(val_aug_value)

    # Additional metadata
    final["data_config_path"] = cli_dict.get("data")
    final["config_path"] = cli_dict.get("config")
    final["data_names"] = data_yaml.get("names")
    final["data_num_classes"] = data_yaml.get("nc")

    # Handle optional project/name pattern
    project_dir = get_nested(config_yaml, ("output", "project"))
    run_name = first_non_none(
        get_nested(config_yaml, ("output", "name")),
        get_nested(config_yaml, ("output", "run_name")),
    )
    if project_dir or run_name:
        base = Path(project_dir or final["output_dir"])
        if run_name:
            base = base / str(run_name)
        final["output_dir"] = str(base)
    else:
        final["output_dir"] = str(final["output_dir"])

    final["resume"] = cli_value("resume")

    return SimpleNamespace(**final)


def make_dataset(args: SimpleNamespace, split: str) -> VideoDataset:
    """
    Build dataset instance for a specific split ('train' or 'val')
    respecting context/future settings when provided.
    """
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported dataset split: {split}")

    video_dir = args.train_dir if split == "train" else args.val_dir
    if video_dir is None:
        raise ValueError(f"No video directory configured for split '{split}'.")

    common_kwargs = {
        "video_dir": video_dir,
        "frame_size": tuple(args.frame_size),
        "frame_interval": args.frame_interval,
        "mode": split,
        "augment": args.train_augment if split == "train" else args.val_augment,
    }

    if args.context_frames is not None and args.future_frames is not None:
        return VideoPredictionDataset(
            context_frames=args.context_frames,
            future_frames=args.future_frames,
            **common_kwargs,
        )

    return VideoDataset(
        num_frames=args.num_frames,
        **common_kwargs,
    )


def make_dataloader(
    dataset: VideoDataset,
    args: SimpleNamespace,
    split: str,
    sampler: Optional[torch.utils.data.Sampler] = None,
) -> DataLoader:
    """Create DataLoader handling shuffle/sampler logic."""
    shuffle = split == "train" and sampler is None
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def setup_ddp(rank, world_size):
    """Initialize distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up distributed training"""
    dist.destroy_process_group()


def save_checkpoint(model, optimizer, epoch, loss, save_dir, filename="checkpoint.pth"):
    """Save model checkpoint"""
    save_path = Path(save_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    return epoch, loss


def train_epoch(model, dataloader, optimizer, device, epoch, writer=None, rank=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader

    for batch_idx, batch in enumerate(pbar):
        videos = combine_video_batch(batch).to(device)

        # Forward pass
        loss = model(videos)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track loss
        total_loss += loss.item()

        # Update progress bar
        if rank == 0:
            pbar.set_postfix({"loss": loss.item()})

            # Log to tensorboard
            if writer is not None:
                global_step = epoch * num_batches + batch_idx
                writer.add_scalar("train/batch_loss", loss.item(), global_step)

    avg_loss = total_loss / num_batches

    if rank == 0 and writer is not None:
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)

    return avg_loss


@torch.no_grad()
def validate(model, dataloader, device, epoch, writer=None, rank=0):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Validation {epoch}")
    else:
        pbar = dataloader

    for batch in pbar:
        videos = combine_video_batch(batch).to(device)
        loss = model(videos)
        total_loss += loss.item()

        if rank == 0:
            pbar.set_postfix({"val_loss": loss.item()})

    avg_loss = total_loss / num_batches

    if rank == 0 and writer is not None:
        writer.add_scalar("val/epoch_loss", avg_loss, epoch)

    return avg_loss


def train_single_gpu(args):
    """Training on single GPU"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Create model
    unet = VideoDiffusionUNet(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        channel_mults=tuple(args.channel_mults),
        time_emb_dim=args.time_emb_dim,
        num_frames=args.num_frames,
    )

    model = GaussianDiffusion(
        model=unet,
        num_timesteps=args.num_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        schedule=args.schedule,
    )

    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create dataloaders
    train_dataset = make_dataset(args, "train")
    train_loader = make_dataloader(train_dataset, args, "train")

    val_loader = None
    if args.val_dir:
        val_dataset = make_dataset(args, "val")
        val_loader = make_dataloader(val_dataset, args, "val")

    # Tensorboard
    log_dir = Path(args.output_dir) / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir)

    # Save config
    config_path = Path(args.output_dir) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume)
        start_epoch += 1

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 50}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, writer)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        if val_loader is not None:
            val_loss = validate(model, val_loader, device, epoch, writer)
            print(f"Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, val_loss, args.output_dir, "best_model.pth"
                )

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                args.output_dir,
                f"checkpoint_epoch_{epoch}.pth",
            )

        # Update learning rate
        scheduler.step()
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

    # Save final model
    save_checkpoint(
        model,
        optimizer,
        args.epochs - 1,
        train_loss,
        args.output_dir,
        "final_model.pth",
    )
    writer.close()

    print("\nTraining completed!")


def train_multi_gpu(rank, world_size, args):
    """Training on multiple GPUs with DDP"""
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Create model
    unet = VideoDiffusionUNet(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        channel_mults=tuple(args.channel_mults),
        time_emb_dim=args.time_emb_dim,
        num_frames=args.num_frames,
    )

    model = GaussianDiffusion(
        model=unet,
        num_timesteps=args.num_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        schedule=args.schedule,
    )

    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create dataloaders with distributed sampler
    train_dataset = make_dataset(args, "train")

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = make_dataloader(train_dataset, args, "train", sampler=train_sampler)

    val_loader = None
    if args.val_dir and rank == 0:
        val_dataset = make_dataset(args, "val")
        val_loader = make_dataloader(val_dataset, args, "val")

    # Tensorboard (only on rank 0)
    writer = None
    if rank == 0:
        log_dir = (
            Path(args.output_dir) / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        writer = SummaryWriter(log_dir)

        # Save config
        config_path = Path(args.output_dir) / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=4)

    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model.module, optimizer, args.resume)
        start_epoch += 1

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'=' * 50}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, writer, rank
        )

        if rank == 0:
            print(f"Train Loss: {train_loss:.4f}")

            # Validate
            if val_loader is not None:
                val_loss = validate(
                    model.module, val_loader, device, epoch, writer, rank
                )
                print(f"Val Loss: {val_loss:.4f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model.module,
                        optimizer,
                        epoch,
                        val_loss,
                        args.output_dir,
                        "best_model.pth",
                    )

            # Save checkpoint
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(
                    model.module,
                    optimizer,
                    epoch,
                    train_loss,
                    args.output_dir,
                    f"checkpoint_epoch_{epoch}.pth",
                )

        # Update learning rate
        scheduler.step()
        if rank == 0 and writer is not None:
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

    # Save final model
    if rank == 0:
        save_checkpoint(
            model.module,
            optimizer,
            args.epochs - 1,
            train_loss,
            args.output_dir,
            "final_model.pth",
        )
        writer.close()
        print("\nTraining completed!")

    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="Train Video Diffusion Model")

    # Configuration files
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML file with training/model settings",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to YOLO-style dataset YAML (defines train/val/test splits)",
    )

    # Optional direct overrides
    parser.add_argument(
        "--train_dir",
        type=str,
        default=None,
        help="Training video directory (overrides data YAML)",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default=None,
        help="Validation video directory (overrides data YAML)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs",
    )

    # Model & data shape overrides
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Total number of frames per clip (overrides context/future sum)",
    )
    parser.add_argument(
        "--context_frames",
        type=int,
        default=None,
        help="Number of input frames provided to the model",
    )
    parser.add_argument(
        "--future_frames",
        type=int,
        default=None,
        help="Number of frames to predict",
    )
    parser.add_argument(
        "--frame_size",
        type=int,
        nargs="+",
        default=None,
        help="Frame size (provide one value for square or two values H W)",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=None,
        help="Frame sampling interval",
    )
    parser.add_argument(
        "--base_channels",
        type=int,
        default=None,
        help="Base number of channels",
    )
    parser.add_argument(
        "--channel_mults",
        type=int,
        nargs="+",
        default=None,
        help="Channel multipliers (space separated list)",
    )
    parser.add_argument(
        "--time_emb_dim",
        type=int,
        default=None,
        help="Time embedding dimension",
    )

    # Diffusion parameters
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=None,
        help="Number of diffusion timesteps",
    )
    parser.add_argument(
        "--beta_start",
        type=float,
        default=None,
        help="Starting beta value",
    )
    parser.add_argument(
        "--beta_end",
        type=float,
        default=None,
        help="Ending beta value",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default=None,
        choices=["linear", "cosine"],
        help="Noise schedule override",
    )

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size per GPU"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=None, help="Weight decay value"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=None,
        help="Checkpoint save interval",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )

    parser.add_argument(
        "--train-augment",
        dest="train_augment",
        action="store_true",
        help="Force-enable data augmentation during training",
    )
    parser.add_argument(
        "--no-train-augment",
        dest="train_augment",
        action="store_false",
        help="Disable data augmentation during training",
    )
    parser.add_argument(
        "--val-augment",
        dest="val_augment",
        action="store_true",
        help="Enable validation augmentations (debug only)",
    )
    parser.add_argument(
        "--no-val-augment",
        dest="val_augment",
        action="store_false",
        help="Disable validation augmentations",
    )
    parser.set_defaults(train_augment=None, val_augment=None)

    # Multi-GPU parameters
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use")

    args = parser.parse_args()
    resolved_args = build_training_namespace(args)

    # Launch training
    if resolved_args.gpus > 1:
        world_size = resolved_args.gpus
        torch.multiprocessing.spawn(
            train_multi_gpu,
            args=(world_size, resolved_args),
            nprocs=world_size,
            join=True,
        )
    else:
        train_single_gpu(resolved_args)


if __name__ == "__main__":
    main()
