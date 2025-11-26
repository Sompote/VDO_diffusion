"""
State-of-the-Art Video Diffusion Model (2024-2025)

Based on latest research:
- Latte: Latent Diffusion Transformer for Video Generation (2024)
- Sora: DiT architecture with spacetime patches
- LTX-Video: Real-time Video Latent Diffusion (2025)
- EDM: Elucidating Design Space of Diffusion Models
- Spatiotemporal Attention with structured sparsity

Key Features:
1. Latent Diffusion with 3D VAE
2. DiT (Diffusion Transformer) architecture
3. Factorized spatiotemporal attention
4. V-prediction parameterization
5. Classifier-free guidance
6. EDM-style sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Union
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings"""

    def __init__(self, dim: int, theta: int = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    """Timestep embedding with MLP"""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4

        self.time_embed = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        return self.mlp(t_emb)


class FeedForward(nn.Module):
    """Feed-forward network with GEGLU activation"""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GEGLU(nn.Module):
    """Gated Linear Unit with GELU activation"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class SpatialAttention(nn.Module):
    """
    Spatial attention over H×W dimensions
    Uses multi-head self-attention with QK normalization
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_flash: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.scale = dim_head**-0.5
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # QK normalization (from DALL-E 3)
        self.q_norm = nn.LayerNorm(dim_head)
        self.k_norm = nn.LayerNorm(dim_head)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, H*W, C)
        """
        x = self.norm(x)

        # Project to Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b t n (h d) -> b t h n d", h=self.heads), qkv
        )

        # QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Attention
        if self.use_flash:
            # Use PyTorch's flash attention
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0 if not self.training else 0.1, is_causal=False
            )
        else:
            # Standard attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = attn @ v

        # Reshape and project
        out = rearrange(out, "b t h n d -> b t n (h d)")
        return self.to_out(out)


class TemporalAttention(nn.Module):
    """
    Temporal attention over T (time) dimension
    Supports both causal and non-causal variants
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        causal: bool = False,
        use_flash: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.causal = causal
        inner_dim = dim_head * heads
        self.scale = dim_head**-0.5
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # QK normalization
        self.q_norm = nn.LayerNorm(dim_head)
        self.k_norm = nn.LayerNorm(dim_head)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, C) where N = H*W
        """
        B, T, N, C = x.shape
        x = self.norm(x)

        # Rearrange to process temporal dimension
        x = rearrange(x, "b t n c -> (b n) t c")

        # Project to Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "bn t (h d) -> bn h t d", h=self.heads), qkv
        )

        # QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Attention
        if self.use_flash:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0 if not self.training else 0.1,
                is_causal=self.causal,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if self.causal:
                # Create causal mask
                mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(1)
                attn = attn.masked_fill(mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            out = attn @ v

        # Reshape
        out = rearrange(out, "bn h t d -> bn t (h d)")
        out = rearrange(out, "(b n) t c -> b t n c", b=B)

        return self.to_out(out)


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block with AdaLN-Zero
    Based on "Scalable Diffusion Models with Transformers" (DiT)
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()

        # Spatial attention
        self.spatial_attn = SpatialAttention(dim, heads, dim_head, dropout)

        # Temporal attention
        self.temporal_attn = TemporalAttention(dim, heads, dim_head, dropout, causal)

        # Feed-forward
        self.ff = FeedForward(dim, ff_mult, dropout)

        # AdaLN-Zero conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        # Initialize to zero (important for training stability)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, C)
        t_emb: (B, C)
        """
        # Get adaptive layer norm parameters
        shift_sa, scale_sa, gate_sa, shift_ta, scale_ta, gate_ta = (
            self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        )

        # Reshape for broadcasting
        shift_sa = shift_sa[:, None, None, :]
        scale_sa = scale_sa[:, None, None, :]
        gate_sa = gate_sa[:, None, None, :]
        shift_ta = shift_ta[:, None, None, :]
        scale_ta = scale_ta[:, None, None, :]
        gate_ta = gate_ta[:, None, None, :]

        # Spatial attention with AdaLN
        x_norm = F.layer_norm(x, (x.shape[-1],))
        x_norm = x_norm * (1 + scale_sa) + shift_sa
        x = x + gate_sa * self.spatial_attn(x_norm)

        # Temporal attention with AdaLN
        x_norm = F.layer_norm(x, (x.shape[-1],))
        x_norm = x_norm * (1 + scale_ta) + shift_ta
        x = x + gate_ta * self.temporal_attn(x_norm)

        # Feed-forward
        x = x + self.ff(F.layer_norm(x, (x.shape[-1],)))

        return x


class VideoVAE3D(nn.Module):
    """
    3D Video VAE for latent diffusion
    Compresses video to latent space with temporal coherence
    Based on LTX-Video and Stable Video Diffusion
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 4),
        temporal_downsample: Tuple[bool, ...] = (False, True, True, False),
        spatial_downsample_factor: int = 8,
        temporal_downsample_factor: int = 4,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.spatial_downsample_factor = spatial_downsample_factor
        self.temporal_downsample_factor = temporal_downsample_factor

        # Encoder
        self.encoder = nn.ModuleList()
        in_ch = in_channels

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult

            # Residual blocks
            self.encoder.append(
                nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.SiLU(),
                    nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.SiLU(),
                )
            )

            # Downsample
            if i < len(channel_mults) - 1:
                if temporal_downsample[i]:
                    # Spatial and temporal downsampling
                    self.encoder.append(
                        nn.Conv3d(
                            out_ch,
                            out_ch,
                            kernel_size=(3, 4, 4),
                            stride=(2, 2, 2),
                            padding=(1, 1, 1),
                        )
                    )
                else:
                    # Spatial only downsampling
                    self.encoder.append(
                        nn.Conv3d(
                            out_ch,
                            out_ch,
                            kernel_size=(1, 4, 4),
                            stride=(1, 2, 2),
                            padding=(0, 1, 1),
                        )
                    )

            in_ch = out_ch

        # Bottleneck to latent space
        final_ch = base_channels * channel_mults[-1]
        self.to_latent = nn.Conv3d(final_ch, latent_channels * 2, kernel_size=1)

        # Decoder
        self.from_latent = nn.Conv3d(latent_channels, final_ch, kernel_size=1)

        self.decoder = nn.ModuleList()
        reversed_mults = list(reversed(channel_mults))
        reversed_temporal = list(reversed(temporal_downsample))

        for i, mult in enumerate(reversed_mults):
            in_ch = base_channels * reversed_mults[i]
            out_ch = (
                base_channels * reversed_mults[i + 1]
                if i + 1 < len(reversed_mults)
                else base_channels
            )

            # Upsample
            if i > 0:
                if reversed_temporal[i - 1]:
                    # Spatial and temporal upsampling
                    # Use Upsample + Conv instead of ConvTranspose to avoid checkerboard artifacts
                    self.decoder.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
                            nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1)
                        )
                    )
                else:
                    # Spatial only upsampling
                    # Use Upsample + Conv instead of ConvTranspose to avoid checkerboard artifacts
                    self.decoder.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
                            nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1)
                        )
                    )

            # Residual blocks
            self.decoder.append(
                nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.SiLU(),
                    nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.SiLU(),
                )
            )

        # Output
        self.to_out = nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent distribution"""
        for layer in self.encoder:
            x = layer(x)

        # Get mean and logvar
        moments = self.to_latent(x)
        mean, logvar = moments.chunk(2, dim=1)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std

        return z, mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent"""
        x = self.from_latent(z)

        for layer in self.decoder:
            x = layer(x)

        return self.to_out(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full VAE forward pass"""
        z, mean, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, z, mean, logvar


class LatentVideoDiT(nn.Module):
    """
    Latent Video Diffusion Transformer
    State-of-the-art architecture combining:
    - 3D VAE for latent compression
    - DiT blocks with factorized spatiotemporal attention
    - AdaLN-Zero conditioning
    """

    def __init__(
        self,
        in_channels: int = 4,  # Latent channels
        out_channels: int = 4,
        patch_size: Tuple[int, int] = (2, 2),  # Spatial patch size
        num_frames: int = 16,
        img_size: int = 32,  # Latent spatial size
        hidden_dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False,
        num_classes: Optional[int] = None,  # For class conditioning
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Calculate number of patches
        self.num_patches_per_frame = (img_size // patch_size[0]) * (
            img_size // patch_size[1]
        )
        self.num_patches = num_frames * self.num_patches_per_frame

        # Calculate patch grid dimensions
        self.num_patch_rows = img_size // patch_size[0]
        self.num_patch_cols = img_size // patch_size[1]

        # Patchify: convert to tokens
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c t (h p1) (w p2) -> b t (h w) (p1 p2 c)",
                p1=patch_size[0],
                p2=patch_size[1],
            ),
            nn.LayerNorm(patch_size[0] * patch_size[1] * in_channels),
            nn.Linear(patch_size[0] * patch_size[1] * in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # 2D Positional embeddings (separate row and column)
        # This explicitly encodes spatial structure
        self.pos_embed_row = nn.Parameter(
            torch.randn(1, 1, self.num_patch_rows, 1, hidden_dim) * 0.02
        )
        self.pos_embed_col = nn.Parameter(
            torch.randn(1, 1, 1, self.num_patch_cols, hidden_dim) * 0.02
        )
        self.pos_embed_temporal = nn.Parameter(
            torch.randn(1, num_frames, 1, 1, hidden_dim) * 0.02
        )

        # Time embedding
        self.time_embed = TimestepEmbedding(hidden_dim)

        # Class embedding (for classifier-free guidance)
        if num_classes is not None:
            self.class_embed = nn.Embedding(
                num_classes + 1, hidden_dim
            )  # +1 for null class

        # DiT blocks
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_dim, heads, dim_head, ff_mult, dropout, causal)
                for _ in range(depth)
            ]
        )

        # Final layer with AdaLN
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, patch_size[0] * patch_size[1] * out_channels),
        )

        # AdaLN for final layer
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, patch_size^2 * C)
        return: (B, C, T, H, W)
        """
        B, T, N, _ = x.shape
        p1, p2 = self.patch_size
        h = w = int(N**0.5)

        x = rearrange(
            x,
            "b t (h w) (p1 p2 c) -> b c t (h p1) (w p2)",
            h=h,
            w=w,
            p1=p1,
            p2=p2,
            c=self.out_channels,
        )
        return x

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, C, T, H, W) - latent input
        timesteps: (B,)
        class_labels: (B,) - optional class labels
        """
        B = x.shape[0]

        # Convert to patches: (B, T, N, D)
        x = self.to_patch_embedding(x)

        # Reshape to 2D grid for adding 2D positional embeddings
        # IMPORTANT: Use einops to ensure consistent ordering with patchify!
        # (B, T, N, D) -> (B, T, H, W, D)
        x = rearrange(
            x,
            "b t (h w) d -> b t h w d",
            h=self.num_patch_rows,
            w=self.num_patch_cols,
        )

        # Add 2D positional embeddings
        x = x + self.pos_embed_row + self.pos_embed_col + self.pos_embed_temporal

        # Flatten back to sequence
        # (B, T, H, W, D) -> (B, T, N, D)
        x = rearrange(x, "b t h w d -> b t (h w) d")

        # Time embedding
        t_emb = self.time_embed(timesteps)

        # Class embedding (for classifier-free guidance)
        if self.num_classes is not None and class_labels is not None:
            c_emb = self.class_embed(class_labels)
            t_emb = t_emb + c_emb

        # Apply DiT blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # Final layer with AdaLN
        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=-1)
        shift = shift[:, None, None, :]
        scale = scale[:, None, None, :]

        x = F.layer_norm(x, (x.shape[-1],))
        x = x * (1 + scale) + shift
        x = self.final_layer(x)

        # Unpatchify
        x = self.unpatchify(x)

        return x


class AdvancedVideoDiffusion(nn.Module):
    """
    Advanced Video Diffusion Model with:
    - Latent diffusion (3D VAE)
    - V-prediction parameterization
    - EDM-style sampling
    - Classifier-free guidance
    """

    def __init__(
        self,
        vae: VideoVAE3D,
        dit: LatentVideoDiT,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        prediction_type: str = "v",  # 'eps', 'x0', or 'v'
        guidance_scale: float = 7.5,
        p_uncond: float = 0.1,  # Probability of unconditional training
        vae_loss_weight: float = 1.0,
        kl_loss_weight: float = 1e-6,
    ):
        super().__init__()

        self.vae = vae
        self.dit = dit
        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type
        self.guidance_scale = guidance_scale
        self.p_uncond = p_uncond
        self.vae_loss_weight = vae_loss_weight
        self.kl_loss_weight = kl_loss_weight

        # Create noise schedule
        if beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, num_timesteps)
        elif beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps)
        elif beta_schedule == "sigmoid":
            betas = self._sigmoid_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")

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

        # For v-prediction
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule (improved DDPM)"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def _sigmoid_beta_schedule(
        self, timesteps: int, start: float = -3, end: float = 3
    ) -> torch.Tensor:
        """Sigmoid schedule"""
        betas = torch.linspace(start, end, timesteps)
        return torch.sigmoid(betas) * (0.02 - 1e-4) + 1e-4

    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode video to latent space"""
        z, _, _ = self.vae.encode(x)
        return z

    @torch.no_grad()
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to video"""
        return self.vae.decode(z)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion: add noise to x_start"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[..., None]
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[..., None]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Convert v-prediction to noise prediction"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        while len(sqrt_alphas_cumprod_t.shape) < len(x_t.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[..., None]
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[..., None]

        return sqrt_alphas_cumprod_t * v + sqrt_one_minus_alphas_cumprod_t * x_t

    def predict_x0_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Convert v-prediction to x0 prediction"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        while len(sqrt_alphas_cumprod_t.shape) < len(x_t.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[..., None]
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[..., None]

        return sqrt_alphas_cumprod_t * x_t - sqrt_one_minus_alphas_cumprod_t * v

    def compute_v_target(
        self, x_start: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute v-prediction target"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[..., None]
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[..., None]

        return sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x_start

    def forward(
        self,
        x: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Training forward pass with classifier-free guidance

        x: (B, C, T, H, W) - pixel space video
        class_labels: (B,) - optional class labels
        """
        B = x.shape[0]
        device = x.device

        # Encode to latent (with gradients!)
        z, mean, logvar = self.vae.encode(x)

        # VAE Reconstruction Loss
        # We need to decode to compute reconstruction loss
        # To save memory, we can decode a subset or the whole batch depending on memory constraints
        # For now, let's decode the whole batch
        recon_x = self.vae.decode(z)
        recon_loss = F.mse_loss(recon_x, x)

        # VAE KL Divergence Loss
        # KL(N(mean, std) || N(0, 1)) = -0.5 * sum(1 + log(std^2) - mean^2 - std^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss = kl_loss / (B * x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4]) # Normalize

        # Sample timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()

        # Sample noise
        noise = torch.randn_like(z)

        # Forward diffusion
        z_t = self.q_sample(z, t, noise)

        # Classifier-free guidance: randomly drop conditioning
        if class_labels is not None and self.p_uncond > 0:
            mask = torch.rand(B, device=device) < self.p_uncond
            class_labels = class_labels.clone()
            class_labels[mask] = self.dit.num_classes  # Use null class

        # Predict
        model_output = self.dit(z_t, t, class_labels)

        # Compute loss based on prediction type
        if self.prediction_type == "eps":
            target = noise
        elif self.prediction_type == "x0":
            target = z
        elif self.prediction_type == "v":
            target = self.compute_v_target(z, noise, t)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        diff_loss = F.mse_loss(model_output, target)
        
        # Total loss
        total_loss = diff_loss + self.vae_loss_weight * recon_loss + self.kl_loss_weight * kl_loss

        return total_loss, diff_loss, recon_loss, kl_loss

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None,
        num_steps: Optional[int] = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Generate videos with classifier-free guidance

        shape: Latent shape (B, C, T, H, W)
        class_labels: (B,) - class labels for conditional generation
        guidance_scale: CFG scale (default: self.guidance_scale)
        num_steps: Number of denoising steps (default: self.num_timesteps)
        """
        guidance_scale = guidance_scale or self.guidance_scale
        num_steps = num_steps or self.num_timesteps

        # Start from random noise
        z = torch.randn(shape, device=device)

        # Timesteps for DDIM sampling
        timesteps = torch.linspace(
            num_steps - 1, 0, num_steps, dtype=torch.long, device=device
        )

        for i, t in enumerate(timesteps):
            t_batch = t.expand(shape[0]).to(device)

            # Predict with and without guidance
            if class_labels is not None and guidance_scale > 1.0:
                # Conditional prediction
                model_output_cond = self.dit(z, t_batch, class_labels)

                # Unconditional prediction (null class)
                null_labels = torch.full_like(class_labels, self.dit.num_classes)
                model_output_uncond = self.dit(z, t_batch, null_labels)

                # Classifier-free guidance
                model_output = model_output_uncond + guidance_scale * (
                    model_output_cond - model_output_uncond
                )
            else:
                model_output = self.dit(z, t_batch, class_labels)

            # Convert prediction to noise and x0
            if self.prediction_type == "v":
                pred_noise = self.predict_noise_from_v(z, t_batch, model_output)
                pred_x0 = self.predict_x0_from_v(z, t_batch, model_output)
            elif self.prediction_type == "eps":
                pred_noise = model_output
                # Compute x0 from noise
                alpha_t = self.sqrt_alphas_cumprod[t]
                sigma_t = self.sqrt_one_minus_alphas_cumprod[t]
                while len(alpha_t.shape) < len(z.shape):
                    alpha_t = alpha_t[..., None]
                    sigma_t = sigma_t[..., None]
                pred_x0 = (z - sigma_t * pred_noise) / alpha_t
            else:  # x0
                pred_x0 = model_output
                # Compute noise from x0
                alpha_t = self.sqrt_alphas_cumprod[t]
                sigma_t = self.sqrt_one_minus_alphas_cumprod[t]
                while len(alpha_t.shape) < len(z.shape):
                    alpha_t = alpha_t[..., None]
                    sigma_t = sigma_t[..., None]
                pred_noise = (z - alpha_t * pred_x0) / sigma_t

            # DDIM step
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_t_prev = self.sqrt_alphas_cumprod[t_prev]
                sigma_t_prev = self.sqrt_one_minus_alphas_cumprod[t_prev]

                while len(alpha_t_prev.shape) < len(z.shape):
                    alpha_t_prev = alpha_t_prev[..., None]
                    sigma_t_prev = sigma_t_prev[..., None]

                z = alpha_t_prev * pred_x0 + sigma_t_prev * pred_noise
            else:
                z = pred_x0

        # Decode from latent
        videos = self.decode_from_latent(z)

        return videos
