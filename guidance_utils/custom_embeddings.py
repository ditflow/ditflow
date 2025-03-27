import torch
from typing import Tuple

from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.models.embeddings import get_3d_rotary_pos_embed

def prepare_rotary_positional_embeddings(
        height: int,
        width: int,
        num_frames: int,
        vae_scale_factor_spatial: float,
        patch_size: int,
        attention_head_dim: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * patch_size)
        grid_width = width // (vae_scale_factor_spatial * patch_size)
        base_size_width = 720 // (vae_scale_factor_spatial * patch_size)
        base_size_height = 480 // (vae_scale_factor_spatial * patch_size)

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            use_real=True,
        )

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return torch.stack([freqs_cos, freqs_sin], dim=0)