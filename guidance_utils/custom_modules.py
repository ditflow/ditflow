import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Optional
from diffusers.models.attention_processor import Attention

class InjectionProcessor:
    r"""
    Modified CogVideoXAttnProcessor2_0 processor for guidance.
    """

    def __init__(self, block_name):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        self.block_name = block_name
        self.inject_kv = False
        self.copy_kv = False

        self.query = None
        self.key = None
        self.value = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        
        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            q_rope, k_rope = image_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], q_rope)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], k_rope)
        
        # Save QKV
        if self.inject_kv:
            key[-1:, :, text_seq_length:] = self.key[-1:, :, text_seq_length:]
            value[-1:, :, text_seq_length:] = self.value[-1:, :, text_seq_length:]
        elif self.copy_kv:
            self.key = key[-1:]
            self.query = query[-1:]
            self.value = value[-1:]
        
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        batch_size = query.shape[0]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * key.shape[-1])

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states

class ModuleWithGuidance(torch.nn.Module):
    def __init__(self, module, h, w, p, block_name, num_frames):
        """ self.num_frames must be registered separately. """
        super().__init__()
        self.module = module
        self.attn1 = module.attn1

        self.starting_shape = "(t h w) d"
        self.h = h
        self.w = w
        self.p = p
        self.block_name = block_name
        self.num_frames = num_frames

    def forward(self, *args, **kwargs):
        out, text_out = self.module(*args, **kwargs)
        p_h = self.h // self.p
        p_w = self.w // self.p

        self.saved_features = rearrange(
            out[-1], f"{self.starting_shape} -> t d h w", t=self.num_frames, h=p_h, w=p_w
        )

        return out, text_out