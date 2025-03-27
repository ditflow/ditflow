from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from typing import Optional, Tuple, Dict, Any, Union
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version

from diffusers.models.embeddings import get_3d_sincos_pos_embed

class ControlledTransformer(CogVideoXTransformer3DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Extract the needed parameters from kwargs
        num_attention_heads = kwargs.get('num_attention_heads', 30)
        attention_head_dim = kwargs.get('attention_head_dim', 64)
        sample_height = kwargs.get('sample_height', 60)
        sample_width = kwargs.get('sample_width', 90)
        sample_frames = kwargs.get('sample_frames', 49)
        patch_size = kwargs.get('patch_size', 2)
        temporal_compression_ratio = kwargs.get('temporal_compression_ratio', 4)
        max_text_seq_length = kwargs.get('max_text_seq_length', 226)
        spatial_interpolation_scale = kwargs.get('spatial_interpolation_scale', 1.875)
        temporal_interpolation_scale = kwargs.get('temporal_interpolation_scale', 1.0)

        inner_dim = num_attention_heads * attention_head_dim

        post_patch_height = sample_height // patch_size
        post_patch_width = sample_width // patch_size
        post_time_compression_frames = (sample_frames - 1) // temporal_compression_ratio + 1
        self.num_patches = post_patch_height * post_patch_width * post_time_compression_frames

        # Trainable position embedding
        spatial_pos_embedding = get_3d_sincos_pos_embed(
            inner_dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            spatial_interpolation_scale,
            temporal_interpolation_scale,
        )
        spatial_pos_embedding = torch.from_numpy(spatial_pos_embedding).flatten(0, 1)
        pos_embedding = torch.zeros(1, max_text_seq_length + self.num_patches, inner_dim)
        pos_embedding.data[:, max_text_seq_length:].copy_(spatial_pos_embedding)
        self.init_pos_embedding = pos_embedding
        self.trainable_pos_embedding = None

        # Trainable RoPE
        self.init_rope = None
        self.trainable_rope = None
    
    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        pos_embedding: Optional[torch.Tensor] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)

        # 3. Position embedding
        text_seq_length = encoder_hidden_states.shape[1]
        if not self.config.use_rotary_positional_embeddings:
            seq_length = height * width * num_frames // (self.config.patch_size**2)

            if pos_embedding is not None:
                pos_embeds = torch.concat([self.init_pos_embedding[:, : text_seq_length], pos_embedding], dim=1)
            else:
                pos_embeds = self.init_pos_embedding[:, : text_seq_length + seq_length]
            
            hidden_states = hidden_states + pos_embeds
            hidden_states = self.embedding_dropout(hidden_states)

        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 4. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.init_rope is None:
                # 2B
                image_rotary_emb = None
            else:
                # 5B
                if rope is None:
                    q_rope, k_rope = self.init_rope.chunk(2), self.init_rope.chunk(2)
                else:
                    q_rope, k_rope = rope[0].chunk(2), rope[1].chunk(2)

                image_rotary_emb = q_rope, k_rope

            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 5. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 6. Unpatchify
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, channels, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)