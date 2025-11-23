"""
Video Diffusion Model for Video Prediction
Implements a diffusion-based approach for predicting future video frames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time steps"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time conditioning"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)

        # First conv block
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        time_emb = F.silu(time_emb)
        # Reshape time_emb to match spatial dimensions
        time_emb = time_emb[:, :, None, None, None]
        h = h + time_emb

        # Second conv block
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class AttentionBlock(nn.Module):
    """3D Spatial-Temporal Attention Block"""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x)

        # Reshape for multi-head attention
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, T * H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, THW, C//heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)

        # Apply attention to values
        h = attn @ v
        h = h.permute(0, 1, 3, 2).reshape(B, C, T, H, W)
        h = self.proj(h)

        return h + residual


class DownBlock(nn.Module):
    """Downsampling block"""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_emb_dim)
        self.downsample = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(
        self, x: torch.Tensor, time_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.res1(x, time_emb)
        x = self.res2(x, time_emb)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block"""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels, in_channels, kernel_size=4, stride=2, padding=1
        )
        self.res1 = ResidualBlock(in_channels * 2, out_channels, time_emb_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_emb_dim)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor
    ) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, time_emb)
        x = self.res2(x, time_emb)
        return x


class VideoDiffusionUNet(nn.Module):
    """
    3D U-Net for video diffusion model
    Takes noisy video frames and predicts the noise
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        time_emb_dim: int = 256,
        num_frames: int = 16,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Initial convolution
        self.init_conv = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsample blocks
        self.down_blocks = nn.ModuleList()
        channels = [base_channels * mult for mult in channel_mults]
        in_ch = base_channels

        for out_ch in channels:
            self.down_blocks.append(DownBlock(in_ch, out_ch, time_emb_dim))
            in_ch = out_ch

        # Middle blocks
        self.mid_block1 = ResidualBlock(channels[-1], channels[-1], time_emb_dim)
        self.mid_attn = AttentionBlock(channels[-1])
        self.mid_block2 = ResidualBlock(channels[-1], channels[-1], time_emb_dim)

        # Upsample blocks
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))

        for i in range(len(reversed_channels)):
            in_ch = reversed_channels[i]
            out_ch = (
                reversed_channels[i + 1]
                if i + 1 < len(reversed_channels)
                else base_channels
            )
            self.up_blocks.append(UpBlock(in_ch, out_ch, time_emb_dim))

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1),
        )

    def pad_to_multiple(self, x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int, int]:
        """
        Pad temporal dimension to be a multiple of the given value
        Returns: padded tensor, original size, padded size
        """
        T = x.shape[2]
        if T % multiple == 0:
            return x, T, T

        target_T = ((T // multiple) + 1) * multiple
        pad_size = target_T - T
        
        # Pad the last dimension (which is time in our case due to how F.pad works on last dims)
        # But F.pad works from last dim backwards. 
        # x shape is (B, C, T, H, W). 
        # We want to pad T.
        # F.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
        # Corresponds to W, H, T
        
        x_padded = F.pad(x, (0, 0, 0, 0, 0, pad_size), mode='replicate')
        return x_padded, T, target_T

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, T, H, W)
            timesteps: Timestep tensor of shape (B,)

        Returns:
            Predicted noise tensor of shape (B, C, T, H, W)
        """
        # Calculate required multiple based on number of downsampling layers
        # Each downblock reduces T by 2. We have len(self.down_blocks) layers.
        multiple = 2 ** len(self.down_blocks)
        
        # Pad input if necessary
        x_padded, orig_T, padded_T = self.pad_to_multiple(x, multiple)
        
        # Time embedding
        time_emb = self.time_mlp(timesteps)

        # Initial convolution
        h = self.init_conv(x_padded)

        # Downsample
        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, time_emb)
            skips.append(skip)

        # Middle
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)

        # Upsample
        for up_block in self.up_blocks:
            skip = skips.pop()
            h = up_block(h, skip, time_emb)

        # Final convolution
        h = self.final_conv(h)
        
        # Crop back to original size if padding was applied
        if orig_T != padded_T:
            h = h[:, :, :orig_T, :, :]

        return h


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Process for Video Prediction
    """

    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps

        # Create beta schedule
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        # Posterior variance
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward diffusion process: add noise to x_start at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[
            :, None, None, None, None
        ]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate loss for training
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t)

        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def p_losses_inpainting(
        self,
        context: torch.Tensor,
        future: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate loss for inpainting-style conditional prediction training
        
        This trains the model to predict future frames given clean context frames.
        Only the future portion is noised, while context remains clean.
        
        Args:
            context: Clean context frames (B, C, T_context, H, W)
            future: Clean future frames to predict (B, C, T_future, H, W)
            t: Timesteps (B,)
            noise: Optional noise to add to future frames
            
        Returns:
            MSE loss between predicted and actual noise (only for future portion)
        """
        if noise is None:
            noise = torch.randn_like(future)
        
        # Add noise only to future frames
        noisy_future = self.q_sample(future, t, noise)
        
        # Concatenate clean context + noisy future
        mixed_input = torch.cat([context, noisy_future], dim=2)
        
        # Predict noise for the full sequence
        predicted_noise = self.model(mixed_input, t)
        
        # Compute loss only on future portion
        T_context = context.shape[2]
        future_predicted_noise = predicted_noise[:, :, T_context:, :, :]
        
        loss = F.mse_loss(future_predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int, t_index: int) -> torch.Tensor:
        """
        Reverse diffusion process: denoise x at timestep t
        """
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]

        # Predict noise
        model_mean = sqrt_recip_alphas_t * (
            x
            - betas_t
            * self.model(
                x, torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            )
            / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...], device: str = "cuda") -> torch.Tensor:
        """
        Generate video by sampling from noise
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        for i in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, i, i)

        return x

    @torch.no_grad()
    def predict_video(
        self, context_frames: torch.Tensor, num_future_frames: int, device: str = "cuda"
    ) -> torch.Tensor:
        """
        Predict future video frames using inpainting-style denoising

        This method keeps context frames clean throughout the diffusion process
        while denoising future frames, maintaining the conditioning.

        Args:
            context_frames: Past frames of shape (B, C, T_context, H, W)
            num_future_frames: Number of future frames to predict
            device: Device to run inference on

        Returns:
            Predicted future frames of shape (B, C, num_future_frames, H, W)
        """
        B, C, T_context, H, W = context_frames.shape

        # Get model's expected number of frames
        model_num_frames = self.model.num_frames if hasattr(self.model, 'num_frames') else 16

        print(f"Generating {num_future_frames} frames with inpainting-style denoising...")
        print(f"Model expects {model_num_frames} frames, using {T_context} context frames")

        # Determine how many frames the model will process
        total_frames = T_context + num_future_frames

        # If total is less than model expects, pad with future frames
        if total_frames < model_num_frames:
            future_pad_count = model_num_frames - total_frames
            print(f"Padding with {future_pad_count} frames to match model's expected input")
        else:
            future_pad_count = num_future_frames

        # Initialize future frames with pure noise
        future_noise = torch.randn(B, C, future_pad_count, H, W, device=device)

        # Denoise using inpainting method (context stays clean, future denoises)
        full_denoised = self._inpaint_denoise(context_frames, future_noise, device)

        # Extract only the number of future frames requested
        predicted_frames = full_denoised[:, :, T_context:T_context+num_future_frames, :, :]

        print(f"✓ Generated {num_future_frames} frames")

        return predicted_frames
    
    @torch.no_grad()
    def _inpaint_denoise(
        self,
        context: torch.Tensor,
        future_noise: torch.Tensor,
        device: str
    ) -> torch.Tensor:
        """
        Inpainting-style denoising: denoise future frames while preserving context

        The model was trained with clean context + noisy future at each timestep.
        At each denoising step:
        1. Keep context COMPLETELY CLEAN (never noise it)
        2. Denoise the future portion

        This maintains consistency with how the model was trained.

        Args:
            context: Clean context frames (B, C, T_context, H, W)
            future_noise: Initial noise for future frames (B, C, T_future, H, W)
            device: Device to run on

        Returns:
            Denoised full sequence (B, C, T_context+T_future, H, W)
        """
        T_context = context.shape[2]

        # Concatenate context and future
        x = torch.cat([context, future_noise], dim=2)

        # Denoise from T to 0
        for i in reversed(range(self.num_timesteps)):
            # Denoise the full sequence
            x = self.p_sample(x, i, i)

            # CRITICAL: Keep context frames perfectly clean
            # Replace context portion with original clean context
            x[:, :, :T_context, :, :] = context

        return x
    
    @torch.no_grad()
    def _sample_from_partial(self, x: torch.Tensor, device: str) -> torch.Tensor:
        """
        Denoise a video sequence through the full diffusion process
        Helper method for simple denoising (deprecated, use _inpaint_denoise instead)
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
            device: Device to run on
            
        Returns:
            Denoised tensor of same shape
        """
        for i in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, i, i)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass
        """
        B = x.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=x.device).long()
        return self.p_losses(x, t)
