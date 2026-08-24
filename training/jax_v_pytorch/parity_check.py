"""
Loads Qwen/Qwen2.5-1.5B-Instruct's official PyTorch weights, converts them
into the Flax params pytree defined in qwen2_flax.py, and verifies the two
implementations produce matching logits on the same input.

Run this BEFORE any training/benchmarking. If parity fails, the JAX
benchmark numbers are meaningless -- fix the port first.

Usage:
    python parity_check.py
"""

import numpy as np
import jax.numpy as jnp
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from qwen2_flax import Qwen2Config, Qwen2ForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def convert_pytorch_to_flax_params(pt_state_dict, cfg: Qwen2Config):
    """
    Maps HF PyTorch Qwen2 state_dict keys -> the nested param pytree Flax
    expects, matching the module names used in qwen2_flax.py.

    Key mapping notes:
      - PyTorch nn.Linear weight shape is (out_features, in_features).
        Flax nn.Dense kernel shape is (in_features, out_features).
        -> every Linear weight needs a transpose.
      - HF names layers as `model.layers.{i}.*`; qwen2_flax.py names them
        `layers_{i}` (underscore, not dot-index) because Flax module names
        can't contain raw integers the same way -- mapped explicitly below.
    """
    sd = {k: v.detach().cpu().numpy() for k, v in pt_state_dict.items()}
    params = {"model": {}, "lm_head": {}}

    # embedding: PyTorch (vocab, hidden) -> Flax nn.Embed expects same shape, no transpose
    params["model"]["embed_tokens"] = {"embedding": sd["model.embed_tokens.weight"]}

    params["model"]["norm"] = {"weight": sd["model.norm.weight"]}

    for i in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{i}."
        layer_params = {
            "input_layernorm": {"weight": sd[prefix + "input_layernorm.weight"]},
            "post_attention_layernorm": {"weight": sd[prefix + "post_attention_layernorm.weight"]},
            "self_attn": {
                "q_proj": {
                    "kernel": sd[prefix + "self_attn.q_proj.weight"].T,
                    "bias": sd[prefix + "self_attn.q_proj.bias"],
                },
                "k_proj": {
                    "kernel": sd[prefix + "self_attn.k_proj.weight"].T,
                    "bias": sd[prefix + "self_attn.k_proj.bias"],
                },
                "v_proj": {
                    "kernel": sd[prefix + "self_attn.v_proj.weight"].T,
                    "bias": sd[prefix + "self_attn.v_proj.bias"],
                },
                "o_proj": {"kernel": sd[prefix + "self_attn.o_proj.weight"].T},
            },
            "mlp": {
                "gate_proj": {"kernel": sd[prefix + "mlp.gate_proj.weight"].T},
                "up_proj": {"kernel": sd[prefix + "mlp.up_proj.weight"].T},
                "down_proj": {"kernel": sd[prefix + "mlp.down_proj.weight"].T},
            },
        }
        params["model"][f"layers_{i}"] = layer_params

    # lm_head: tied to embedding if cfg.tie_word_embeddings, else its own weight
    if cfg.tie_word_embeddings:
        lm_head_weight = sd["model.embed_tokens.weight"]  # (vocab, hidden)
    else:
        lm_head_weight = sd["lm_head.weight"]
    params["lm_head"] = {"kernel": lm_head_weight.T}  # Flax Dense kernel: (in, out) = (hidden, vocab)

    # convert every leaf to jnp array
    def to_jnp(tree):
        if isinstance(tree, dict):
            return {k: to_jnp(v) for k, v in tree.items()}
        return jnp.array(tree)

    return {"params": to_jnp(params)}


def main():
    print(f"Loading PyTorch model + config: {MODEL_NAME}")
    hf_config = AutoConfig.from_pretrained(MODEL_NAME)
    pt_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    pt_model.eval()

    cfg = Qwen2Config.from_hf_config(hf_config)
    print(f"Config: hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
          f"heads={cfg.num_attention_heads}, kv_heads={cfg.num_key_value_heads}")

    print("Converting weights to Flax pytree ...")
    flax_params = convert_pytorch_to_flax_params(pt_model.state_dict(), cfg)

    flax_model = Qwen2ForCausalLM(cfg)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_text = "The capital of France is"
    enc = tok(test_text, return_tensors="pt")
    input_ids_pt = enc["input_ids"]
    attention_mask_pt = enc["attention_mask"]

    print(f"Test input: {test_text!r} -> {input_ids_pt.shape[1]} tokens")

    with torch.no_grad():
        pt_out = pt_model(input_ids=input_ids_pt, attention_mask=attention_mask_pt)
        pt_logits = pt_out.logits.numpy().astype(np.float32)

    input_ids_jax = jnp.array(input_ids_pt.numpy())
    attention_mask_jax = jnp.array(attention_mask_pt.numpy())
    flax_logits = flax_model.apply(flax_params, input_ids_jax, attention_mask_jax)
    flax_logits = np.array(flax_logits).astype(np.float32)

    print(f"PyTorch logits shape: {pt_logits.shape}")
    print(f"Flax logits shape:    {flax_logits.shape}")

    abs_diff = np.abs(pt_logits - flax_logits)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    print(f"\nMax abs logit diff:  {max_diff:.6f}")
    print(f"Mean abs logit diff: {mean_diff:.6f}")

    # check top predicted token matches -- the practically meaningful check
    pt_next_token = pt_logits[0, -1].argmax()
    flax_next_token = flax_logits[0, -1].argmax()
    pt_word = tok.decode([pt_next_token])
    flax_word = tok.decode([flax_next_token])
    print(f"\nPyTorch predicted next token: {pt_word!r}")
    print(f"Flax predicted next token:    {flax_word!r}")

    TOLERANCE = 0.05  # fp32 logit tolerance; loosen slightly if using bf16 weights
    if max_diff < TOLERANCE and pt_next_token == flax_next_token:
        print(f"\n✅ PARITY CHECK PASSED (max_diff={max_diff:.6f} < {TOLERANCE}, "
              f"predicted tokens match)")
    else:
        print(f"\n❌ PARITY CHECK FAILED -- do NOT trust benchmark results from this port yet.")
        print("Likely causes: wrong transpose somewhere, RoPE base/formula mismatch, "
              "GQA head-repeat axis wrong, or tie_word_embeddings mismatch. "
              "Debug layer-by-layer (compare hidden states after layer 0) before re-running.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()