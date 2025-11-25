"""
Video Dataset for Diffusion Model Training
Handles video loading, preprocessing, and augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Union
import random


class VideoDataset(Dataset):
    """
    Dataset for loading video sequences for diffusion model training
    """

    def __init__(
        self,
        video_dir: Union[str, Path],
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (256, 256),
        frame_interval: int = 1,
        mode: str = "train",
        video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
        image_extensions: Tuple[str, ...] = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        ),
        augment: bool = True,
        debug_mode: bool = False,
        use_sliding_window: bool = True,
        window_stride: int = 1,
    ):
        """
        Args:
            video_dir: Directory containing video files
            num_frames: Number of frames to extract from each video
            frame_size: Target size (H, W) for frames
            frame_interval: Interval between frames (1 = consecutive frames)
            mode: 'train' or 'val'
            video_extensions: Tuple of valid video file extensions
            augment: Whether to apply data augmentation
            use_sliding_window: If True, create multiple clips per sequence using sliding window
            window_stride: Stride for sliding window (1 = maximum overlap, higher = less overlap)
        """
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frame_interval = frame_interval
        self.mode = mode
        self.image_extensions = image_extensions
        self.debug_mode = debug_mode
        self.augment = augment and mode == "train" and not debug_mode  # No augment in debug mode
        self.use_sliding_window = use_sliding_window
        self.window_stride = window_stride

        # Collect video files
        raw_samples = []
        for ext in video_extensions:
            for path in self.video_dir.rglob(f"*{ext}"):
                raw_samples.append({"type": "video", "path": path})

        # Collect image sequences grouped by parent directory
        image_groups = {}
        for ext in image_extensions:
            for img_path in self.video_dir.rglob(f"*{ext}"):
                parent = img_path.parent
                # Skip hidden directories and .ipynb_checkpoints
                if any(part.startswith('.') for part in parent.parts):
                    continue
                image_groups.setdefault(parent, []).append(img_path)

        for parent, files in image_groups.items():
            files = sorted(files)
            if not files:
                continue
            raw_samples.append({"type": "images", "path": parent, "frames": files})

        if len(raw_samples) == 0:
            raise ValueError(
                f"No video or image sequences found in {video_dir}."
            )

        video_count = sum(1 for s in raw_samples if s["type"] == "video")
        image_count = sum(1 for s in raw_samples if s["type"] == "images")
        print(
            f"Found {video_count} videos and {image_count} image sequences in {video_dir}"
        )

        # Generate clips using sliding window if enabled
        if self.use_sliding_window:
            self.clips = self._generate_clips(raw_samples)
            print(
                f"Generated {len(self.clips)} clips using sliding window (stride={window_stride})"
            )
        else:
            # Legacy mode: one random clip per sequence
            self.clips = [{"source": s, "start_index": None} for s in raw_samples]
            print(f"Using legacy mode: {len(self.clips)} sequences (random sampling per epoch)")

        # Define transforms
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(frame_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _generate_clips(self, raw_samples: List[dict]) -> List[dict]:
        """
        Generate all possible clips from sequences using sliding window.

        Args:
            raw_samples: List of raw video/image sequence samples

        Returns:
            List of clip dictionaries with source and start_index
        """
        clips = []

        for sample in raw_samples:
            if sample["type"] == "video":
                # For videos, we need to know total frame count
                # We'll compute this when loading
                clips.append({
                    "source": sample,
                    "start_index": None,  # Will be computed at load time for videos
                    "is_video": True
                })
            else:
                # For image sequences, pre-compute all valid starting positions
                frames = sorted(sample["frames"])
                total_frames = len(frames)

                # Calculate how many frames we need
                required_frames = (self.num_frames - 1) * self.frame_interval + 1

                if total_frames < required_frames:
                    print(f"Skipping {sample['path']}: only {total_frames} frames, need {required_frames}")
                    continue

                # Generate all valid starting positions
                max_start = total_frames - required_frames

                for start_idx in range(0, max_start + 1, self.window_stride):
                    clips.append({
                        "source": sample,
                        "start_index": start_idx,
                        "is_video": False
                    })

        return clips

    def __len__(self) -> int:
        return len(self.clips)

    def load_video(self, video_path: Path) -> Optional[torch.Tensor]:
        """
        Load video and extract frames

        Returns:
            Tensor of shape (C, T, H, W) or None if loading fails
        """
        try:
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                print(f"Failed to open video: {video_path}")
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate required frames including intervals
            required_frames = self.num_frames * self.frame_interval

            if total_frames < required_frames:
                print(
                    f"Video too short: {video_path} ({total_frames} < {required_frames})"
                )
                cap.release()
                return None

            # Random start position for training, fixed for validation
            # In debug mode, always use fixed position
            if self.mode == "train" and not self.debug_mode:
                start_frame = random.randint(0, total_frames - required_frames)
            else:
                start_frame = (total_frames - required_frames) // 2

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frames = []
            frame_count = 0

            while len(frames) < self.num_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.frame_interval == 0:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Apply transforms
                    frame_tensor = self.transform(frame)
                    frames.append(frame_tensor)

                frame_count += 1

            cap.release()

            if len(frames) < self.num_frames:
                print(f"Could not extract enough frames from: {video_path}")
                return None

            # Stack frames: (C, T, H, W)
            video_tensor = torch.stack(frames, dim=1)

            return video_tensor

        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None

    def augment_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to video

        Args:
            video: Tensor of shape (C, T, H, W)

        Returns:
            Augmented video tensor
        """
        # Random horizontal flip
        if random.random() > 0.5:
            video = torch.flip(video, dims=[3])

        # Random brightness adjustment
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            video = video * brightness_factor
            video = torch.clamp(video, -1.0, 1.0)

        # Random contrast adjustment
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = video.mean(dim=(2, 3), keepdim=True)
            video = (video - mean) * contrast_factor + mean
            video = torch.clamp(video, -1.0, 1.0)

        return video

    def load_image_sequence(self, image_paths, start_index: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Load a sequence of images located in the same directory.

        Args:
            image_paths: List of image file paths
            start_index: Starting frame index. If None, will be randomly chosen (legacy mode)
        """

        frames = sorted(image_paths)
        total_frames = len(frames)

        if total_frames == 0:
            return None

        max_offset = total_frames - ((self.num_frames - 1) * self.frame_interval + 1)
        if max_offset < 0:
            print(
                f"Image sequence too short: {frames[0].parent} ({total_frames} frames < required)"
            )
            return None

        # Use provided start_index or compute it (legacy mode)
        if start_index is None:
            # Legacy random/fixed selection
            if self.mode == "train" and not self.debug_mode:
                start_index = random.randint(0, max_offset)
            else:
                start_index = max_offset // 2 if max_offset > 0 else 0

        selected_indices = [
            start_index + i * self.frame_interval for i in range(self.num_frames)
        ]

        loaded_frames = []
        for idx in selected_indices:
            path = frames[idx]
            image = cv2.imread(str(path))
            if image is None:
                print(f"Failed to load image: {path}")
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame_tensor = self.transform(image)
            loaded_frames.append(frame_tensor)

        video_tensor = torch.stack(loaded_frames, dim=1)
        return video_tensor

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a video clip sample

        Returns:
            Video tensor of shape (C, T, H, W)
        """
        clip = self.clips[idx]
        source = clip["source"]
        start_index = clip["start_index"]

        # Try to load video, if fails, try another random clip
        max_attempts = 10
        for attempt in range(max_attempts):
            if source["type"] == "video":
                video = self.load_video(source["path"])
            else:
                video = self.load_image_sequence(source["frames"], start_index=start_index)

            if video is not None:
                if self.augment:
                    video = self.augment_video(video)
                return video

            # If loading fails, try a random different clip
            idx = random.randint(0, len(self.clips) - 1)
            clip = self.clips[idx]
            source = clip["source"]
            start_index = clip["start_index"]

        # If all attempts fail, return a tensor of zeros
        print(f"Failed to load video after {max_attempts} attempts, returning zeros")
        return torch.zeros(3, self.num_frames, *self.frame_size)


class VideoPredictionDataset(VideoDataset):
    """
    Dataset specifically for video prediction tasks
    Splits video into context frames and future frames
    """

    def __init__(
        self,
        video_dir: Union[str, Path],
        context_frames: int = 8,
        future_frames: int = 8,
        frame_size: Tuple[int, int] = (256, 256),
        frame_interval: int = 1,
        mode: str = "train",
        video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
        image_extensions: Tuple[str, ...] = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        ),
        augment: bool = True,
    ):
        """
        Args:
            context_frames: Number of past frames to use as context
            future_frames: Number of future frames to predict
        """
        self.context_frames = context_frames
        self.future_frames = future_frames
        total_frames = context_frames + future_frames

        super().__init__(
            video_dir=video_dir,
            num_frames=total_frames,
            frame_size=frame_size,
            frame_interval=frame_interval,
            mode=mode,
            video_extensions=video_extensions,
            image_extensions=image_extensions,
            augment=augment,
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a video sample split into context and future frames

        Returns:
            Tuple of (context_frames, future_frames)
            Each tensor has shape (C, T, H, W)
        """
        video = super().__getitem__(idx)

        # Split into context and future
        context = video[:, : self.context_frames, :, :]
        future = video[:, self.context_frames :, :, :]

        return context, future


def create_video_dataloader(
    video_dir: Union[str, Path],
    batch_size: int = 4,
    num_frames: int = 16,
    frame_size: Tuple[int, int] = (256, 256),
    frame_interval: int = 1,
    mode: str = "train",
    num_workers: int = 4,
    pin_memory: bool = True,
    augment: bool = True,
    debug_mode: bool = False,
    use_sliding_window: bool = True,
    window_stride: int = 1,
) -> DataLoader:
    """
    Create a DataLoader for video dataset

    Args:
        video_dir: Directory containing video files
        batch_size: Batch size
        num_frames: Number of frames per video clip
        frame_size: Target frame size (H, W)
        frame_interval: Interval between sampled frames
        mode: 'train' or 'val'
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        augment: Whether to apply augmentation
        use_sliding_window: If True, create multiple clips per sequence using sliding window
        window_stride: Stride for sliding window (1 = maximum overlap, higher = less overlap)

    Returns:
        DataLoader instance
    """
    dataset = VideoDataset(
        video_dir=video_dir,
        num_frames=num_frames,
        frame_size=frame_size,
        frame_interval=frame_interval,
        mode=mode,
        augment=augment,
        debug_mode=debug_mode,
        use_sliding_window=use_sliding_window,
        window_stride=window_stride,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train" and not debug_mode),  # No shuffle in debug mode
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    return dataloader


def create_prediction_dataloader(
    video_dir: Union[str, Path],
    batch_size: int = 4,
    context_frames: int = 8,
    future_frames: int = 8,
    frame_size: Tuple[int, int] = (256, 256),
    frame_interval: int = 1,
    mode: str = "train",
    num_workers: int = 4,
    pin_memory: bool = True,
    augment: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for video prediction dataset

    Returns:
        DataLoader that yields (context_frames, future_frames) tuples
    """
    dataset = VideoPredictionDataset(
        video_dir=video_dir,
        context_frames=context_frames,
        future_frames=future_frames,
        frame_size=frame_size,
        frame_interval=frame_interval,
        mode=mode,
        augment=augment,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    return dataloader
