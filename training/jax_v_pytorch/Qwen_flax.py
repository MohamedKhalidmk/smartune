"""
Flax (JAX) implementation of the Qwen2 architecture.

transformers has no FlaxQwen2ForCausalLM, so this reimplements the pieces
needed to load Qwen/Qwen2.5-1.5B-Instruct's official PyTorch weights into
a JAX/Flax model with matching math. Config values (hidden size, n_layers,
n_heads, n_kv_heads, etc.) are read directly from the HF config so this
stays correct if you point it at a different Qwen2 checkpoint size.

Architecture notes (why each piece exists):
  - RMSNorm: cheaper norm than LayerNorm, no mean-subtraction/bias, used in
    place of LayerNorm throughout (pre-norm, applied before attention/MLP).
  - RoPE: rotary positional embeddings, injected into Q/K via a rotation
    rather than added to the embedding -- gives relative-position
    generalization to longer sequences.
  - GQA (Grouped Query Attention): num_key_value_heads < num_attention_heads.
    Multiple query heads share the same K/V head, cutting KV-cache size at
    inference. Qwen2 additionally uses QKV *bias* (Q/K/V projections have
    bias terms) unlike Llama, which is easy to miss if copying a Llama impl.
  - SwiGLU MLP: gated MLP using SiLU(gate) * up, then down-projected --
    same as Llama's MLP block.

USE: after building this module, run parity_check.py to confirm this
implementation's outputs match the official PyTorch model before trusting
any benchmark built on it.
"""

from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn


@dataclass
class Qwen2Config:
    vocab_size: int = 151936
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True

    @classmethod
    def from_hf_config(cls, hf_config):
        """Build from a transformers Qwen2Config (from AutoConfig.from_pretrained)."""
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            max_position_embeddings=hf_config.max_position_embeddings,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=hf_config.rope_theta,
            tie_word_embeddings=hf_config.tie_word_embeddings,
        )


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        weight = self.param("weight", nn.initializers.ones, (self.dim,))
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
        x_normed = x_f32 * jax.lax.rsqrt(variance + self.eps)
        return (weight * x_normed.astype(x.dtype))


def precompute_rope(head_dim: int, max_positions: int, theta: float):
    """Precompute cos/sin tables for RoPE, shape (max_positions, head_dim)."""
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    positions = jnp.arange(max_positions, dtype=jnp.float32)
    freqs = jnp.outer(positions, inv_freq)  # (max_positions, head_dim/2)
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # (max_positions, head_dim)
    return jnp.cos(emb), jnp.sin(emb)


def rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(q, k, cos, sin):
    # q, k: (batch, n_heads, seq_len, head_dim)
    # cos, sin: (seq_len, head_dim) -> broadcast to (1, 1, seq_len, head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class Qwen2Attention(nn.Module):
    config: Qwen2Config

    @nn.compact
    def __call__(self, x, cos, sin, attn_mask):
        cfg = self.config
        B, T, C = x.shape
        n_heads = cfg.num_attention_heads
        n_kv_heads = cfg.num_key_value_heads
        head_dim = cfg.hidden_size // n_heads
        n_rep = n_heads // n_kv_heads  # how many query heads share each KV head

        # Qwen2 uses bias on q/k/v projections (unlike Llama) -- easy detail to miss.
        q = nn.Dense(n_heads * head_dim, use_bias=True, name="q_proj")(x)
        k = nn.Dense(n_kv_heads * head_dim, use_bias=True, name="k_proj")(x)
        v = nn.Dense(n_kv_heads * head_dim, use_bias=True, name="v_proj")(x)

        q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        q, k = apply_rope(q, k, cos[:T], sin[:T])

        # expand KV heads to match n_heads (repeat_interleave along head axis)
        k = jnp.repeat(k, n_rep, axis=1)
        v = jnp.repeat(v, n_rep, axis=1)

        scale = 1.0 / jnp.sqrt(head_dim).astype(x.dtype)
        attn_weights = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale
        attn_weights = attn_weights + attn_mask  # additive causal mask (0 / -inf)
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(x.dtype)

        out = jnp.einsum("bhts,bhsd->bhtd", attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, n_heads * head_dim)

        out = nn.Dense(cfg.hidden_size, use_bias=False, name="o_proj")(out)
        return out


class Qwen2MLP(nn.Module):
    config: Qwen2Config

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        gate = nn.Dense(cfg.intermediate_size, use_bias=False, name="gate_proj")(x)
        up = nn.Dense(cfg.intermediate_size, use_bias=False, name="up_proj")(x)
        hidden = jax.nn.silu(gate) * up
        return nn.Dense(cfg.hidden_size, use_bias=False, name="down_proj")(hidden)


class Qwen2DecoderLayer(nn.Module):
    config: Qwen2Config

    @nn.compact
    def __call__(self, x, cos, sin, attn_mask):
        cfg = self.config
        residual = x
        x = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, name="input_layernorm")(x)
        x = Qwen2Attention(cfg, name="self_attn")(x, cos, sin, attn_mask)
        x = residual + x

        residual = x
        x = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, name="post_attention_layernorm")(x)
        x = Qwen2MLP(cfg, name="mlp")(x)
        x = residual + x
        return x


class Qwen2Model(nn.Module):
    config: Qwen2Config

    @nn.compact
    def __call__(self, input_ids, attention_mask):
        cfg = self.config
        B, T = input_ids.shape
        head_dim = cfg.hidden_size // cfg.num_attention_heads

        embed = nn.Embed(cfg.vocab_size, cfg.hidden_size, name="embed_tokens")
        x = embed(input_ids)

        cos, sin = precompute_rope(head_dim, cfg.max_position_embeddings, cfg.rope_theta)

        # causal mask combined with padding mask, additive form (0 keep / -inf mask)
        causal = jnp.tril(jnp.ones((T, T), dtype=bool))
        pad = attention_mask[:, None, None, :].astype(bool)  # (B,1,1,T) key-side padding
        combined = causal[None, None, :, :] & pad
        attn_mask = jnp.where(combined, 0.0, jnp.finfo(x.dtype).min).astype(x.dtype)

        for i in range(cfg.num_hidden_layers):
            x = Qwen2DecoderLayer(cfg, name=f"layers_{i}")(x, cos, sin, attn_mask)

        x = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, name="norm")(x)
        return x


class Qwen2ForCausalLM(nn.Module):
    config: Qwen2Config

    @nn.compact
    def __call__(self, input_ids, attention_mask):
        cfg = self.config
        hidden = Qwen2Model(cfg, name="model")(input_ids, attention_mask)

        # NOTE on weight tying: rather than reaching into `self.variables`
        # (fragile during `.init()`, since params don't exist yet on the
        # first trace), we always declare a separate `lm_head` Dense here.
        # If cfg.tie_word_embeddings is True, the weight-conversion script
        # is responsible for copying the embedding matrix's values into
        # lm_head's kernel after loading -- same net effect (same numbers
        # at inference time), simpler and safer Flax control flow.
        logits = nn.Dense(cfg.vocab_size, use_bias=False, name="lm_head")(hidden)
        return logits