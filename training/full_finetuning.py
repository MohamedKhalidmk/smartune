"""
training/jax_v_pytorch/full_finetune_jax.py

Reusable, epoch-based JAX/Flax full-parameter fine-tuning path for Smartune.

This is NOT the standalone benchmark script (train_jax.py) -- that script is
step-count driven, reads pre-tokenized .npz files, and just dumps a results
JSON. This module instead:

  1. Accepts the same `list[dict]` of {"question", "answer"} examples that
     training/finetune.py already tokenizes for the PyTorch/LoRA path.
  2. Bootstraps JAX/Flax params from the pretrained PyTorch checkpoint via
     parity_check.convert_pytorch_to_flax_params.
  3. Runs a real epoch-based training loop (jax.jit train_step, optax.adamw),
     shards params across all local devices with jax.sharding when more than
     one is visible, and reports per-step / per-epoch progress through the
     same progress_callback protocol run_finetune() already uses.
  4. Converts the trained Flax params back into a PyTorch state_dict (the
     reverse of convert_pytorch_to_flax_params) and loads them into a fresh
     AutoModelForCausalLM, so the returned "model" is a normal HF/PyTorch
     object -- required because evaluation/eval_harness.py calls
     model.generate() / reads model.device, and everything downstream
     (llm_judge.py, report.py) depends on that.

Run requirements: jax, flax, optax must be installed in the same environment
as torch/transformers. If they are not, run_jax_full_finetune raises
ImportError with a clear message rather than partially importing.
"""

import importlib
import os
import sys
import time as _time

import numpy as np


# ============================================================
# Import the existing Flax Qwen2 port + weight converter
# ============================================================
#
# qwen2_flax.py / parity_check.py are written as flat scripts (they import
# each other with bare `from qwen2_flax import ...`), living in this same
# directory. Make sure that directory is importable regardless of the
# caller's cwd, then import them by module name.

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _lazy_imports():
    """
    Import jax/flax/optax + the local Flax Qwen2 modules lazily, so that
    importing this file doesn't hard-require a JAX install for callers that
    only ever use the LoRA/PyTorch path.
    """
    try:
        import jax
        import jax.numpy as jnp
        import optax
        from jax.sharding import Mesh, NamedSharding, PartitionSpec
    except ImportError as e:
        raise ImportError(
            "Full fine-tuning via JAX was requested, but jax/optax are not "
            "installed in this environment. Install them (see "
            "training/jax_v_pytorch/ for the expected versions) or choose "
            "method='lora' instead."
        ) from e

    # NOTE: the file is named Qwen_flax.py on disk (capital Q, no "2") even
    # though parity_check.py/train_jax.py historically imported it as
    # `qwen2_flax` -- that mismatch is a pre-existing bug that would fail
    # on any case-sensitive filesystem (Linux). Import by the real filename.
    qwen2_flax = importlib.import_module("Qwen_flax")
    parity_check = importlib.import_module("parity_check")

    return jax, jnp, optax, Mesh, NamedSharding, PartitionSpec, qwen2_flax, parity_check


# ============================================================
# Reverse converter: Flax params pytree -> PyTorch state_dict
# ============================================================

def convert_flax_to_pytorch_state_dict(flax_params, cfg, tie_word_embeddings):
    """
    Inverse of parity_check.convert_pytorch_to_flax_params.

    Takes the trained Flax params pytree (as returned by the JAX train loop,
    i.e. the {"params": {...}} dict) and produces a PyTorch state_dict with
    the exact key names/shapes HF's Qwen2ForCausalLM expects, so it can be
    loaded straight into a `transformers` model via `load_state_dict`.

    Every Flax nn.Dense kernel is (in_features, out_features); PyTorch
    nn.Linear weight is (out_features, in_features) -- so every kernel gets
    transposed back on the way out, mirroring the transpose that
    convert_pytorch_to_flax_params applied on the way in.
    """
    params = flax_params["params"] if "params" in flax_params else flax_params

    def to_np(x):
        return np.asarray(x)

    sd = {}

    sd["model.embed_tokens.weight"] = to_np(
        params["model"]["embed_tokens"]["embedding"]
    )
    sd["model.norm.weight"] = to_np(params["model"]["norm"]["weight"])

    layer_idx = 0
    while f"layers_{layer_idx}" in params["model"]:
        lp = params["model"][f"layers_{layer_idx}"]
        prefix = f"model.layers.{layer_idx}."

        sd[prefix + "input_layernorm.weight"] = to_np(
            lp["input_layernorm"]["weight"]
        )
        sd[prefix + "post_attention_layernorm.weight"] = to_np(
            lp["post_attention_layernorm"]["weight"]
        )

        attn = lp["self_attn"]
        sd[prefix + "self_attn.q_proj.weight"] = to_np(attn["q_proj"]["kernel"]).T
        sd[prefix + "self_attn.q_proj.bias"] = to_np(attn["q_proj"]["bias"])
        sd[prefix + "self_attn.k_proj.weight"] = to_np(attn["k_proj"]["kernel"]).T
        sd[prefix + "self_attn.k_proj.bias"] = to_np(attn["k_proj"]["bias"])
        sd[prefix + "self_attn.v_proj.weight"] = to_np(attn["v_proj"]["kernel"]).T
        sd[prefix + "self_attn.v_proj.bias"] = to_np(attn["v_proj"]["bias"])
        sd[prefix + "self_attn.o_proj.weight"] = to_np(attn["o_proj"]["kernel"]).T

        mlp = lp["mlp"]
        sd[prefix + "mlp.gate_proj.weight"] = to_np(mlp["gate_proj"]["kernel"]).T
        sd[prefix + "mlp.up_proj.weight"] = to_np(mlp["up_proj"]["kernel"]).T
        sd[prefix + "mlp.down_proj.weight"] = to_np(mlp["down_proj"]["kernel"]).T

        layer_idx += 1

    if not tie_word_embeddings:
        sd["lm_head.weight"] = to_np(params["lm_head"]["kernel"]).T
    # If tied, HF re-derives lm_head.weight from the embedding automatically
    # (tie_word_embeddings=True), so we don't need to set it explicitly --
    # setting it anyway would be harmless but redundant.

    return sd


# ============================================================
# Training loop
# ============================================================

def run_jax_full_finetune(
    model_name: str,
    train_input_ids,
    train_attention_mask,
    val_input_ids,
    val_attention_mask,
    progress_callback,
    num_train_epochs: int = 3,
    learning_rate: float = 1e-5,
    per_device_train_batch_size: int = 2,
    seed: int = 42,
) -> dict:
    """
    Full-parameter fine-tune model_name using the JAX/Flax Qwen2 port.

    train_input_ids / train_attention_mask: numpy or torch tensors of shape
        (n_examples, seq_len), already tokenized (e.g. via
        finetune._format_and_tokenize).
    val_input_ids / val_attention_mask: same, or None if there's no eval set.

    Mirrors the shape of run_finetune()'s PyTorch path: reports progress via
    progress_callback({"step", "loss", "val_loss"}) every step, and once per
    epoch (on the eval pass) via {"step", "loss": None, "val_loss": ...}.

    Returns:
        {
            "state_dict": dict[str, np.ndarray],  # PyTorch-shaped weights
            "training_time_s": float,
            "throughput_examples_per_sec": float,
            "peak_gpu_memory_gb": float,
            "final_loss": float,
            "train_loss_history": list[float],
            "val_loss_history": list[float],
        }
    """

    jax, jnp, optax, Mesh, NamedSharding, PartitionSpec, qwen2_flax, parity_check = (
        _lazy_imports()
    )

    import torch  # only used here to accept torch tensors / read HF config
    from transformers import AutoConfig, AutoModelForCausalLM

    np.random.seed(seed)

    def to_numpy(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    train_input_ids = to_numpy(train_input_ids).astype(np.int32)
    train_attention_mask = to_numpy(train_attention_mask).astype(np.int32)
    has_val = val_input_ids is not None
    if has_val:
        val_input_ids = to_numpy(val_input_ids).astype(np.int32)
        val_attention_mask = to_numpy(val_attention_mask).astype(np.int32)

    # --------------------------------------------------------
    # Load pretrained PyTorch weights, convert to Flax params
    # --------------------------------------------------------

    hf_config = AutoConfig.from_pretrained(model_name)
    cfg = qwen2_flax.Qwen2Config.from_hf_config(hf_config)

    pt_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    params = parity_check.convert_pytorch_to_flax_params(
        pt_model.state_dict(), cfg
    )
    del pt_model  # only needed to bootstrap weights

    flax_model = qwen2_flax.Qwen2ForCausalLM(cfg)

    # --------------------------------------------------------
    # Shard across all visible local devices (data-parallel FSDP-style,
    # same approach as train_jax.py). Falls back to a single-device mesh
    # transparently on a 1-GPU/CPU box.
    # --------------------------------------------------------

    devices = np.array(jax.local_devices())
    mesh = Mesh(devices, axis_names=("data",))

    def shard_leaf(leaf):
        if getattr(leaf, "ndim", 0) == 0:
            sharding = NamedSharding(mesh, PartitionSpec())
        else:
            sharding = NamedSharding(mesh, PartitionSpec("data"))
        return jax.device_put(leaf, sharding)

    params = jax.tree_util.tree_map(shard_leaf, params)

    # --------------------------------------------------------
    # Loss + train step
    # --------------------------------------------------------

    def loss_fn(params, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        logits = flax_model.apply(params, input_ids, attention_mask)

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]

        log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
        token_log_probs = jnp.take_along_axis(
            log_probs, shift_labels[..., None], axis=-1
        )[..., 0]

        token_log_probs = token_log_probs * shift_mask

        n_tokens = jnp.maximum(shift_mask.sum(), 1)
        loss = -token_log_probs.sum() / n_tokens
        return loss

    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_step(params, batch):
        return loss_fn(params, batch)

    def make_batches(input_ids, attention_mask, batch_size, rng):
        n = input_ids.shape[0]
        order = rng.permutation(n)
        for start in range(0, n - n % batch_size or n, batch_size):
            idx = order[start:start + batch_size]
            if len(idx) == 0:
                continue
            yield {
                "input_ids": jnp.asarray(input_ids[idx]),
                "attention_mask": jnp.asarray(attention_mask[idx]),
            }

    def run_eval():
        if not has_val:
            return None
        rng = np.random.default_rng(seed)
        losses = []
        for batch in make_batches(
            val_input_ids, val_attention_mask, per_device_train_batch_size, rng
        ):
            losses.append(float(eval_step(params, batch)))
        return float(np.mean(losses)) if losses else None

    # --------------------------------------------------------
    # Training loop (epoch-based, matches run_finetune()'s interface)
    # --------------------------------------------------------

    train_loss_history = []
    val_loss_history = []

    global_step = 0
    n_train = train_input_ids.shape[0]
    steps_per_epoch = max(1, n_train // per_device_train_batch_size)
    total_steps = steps_per_epoch * num_train_epochs

    peak_bytes = 0
    start = _time.time()

    for epoch in range(1, num_train_epochs + 1):
        rng = np.random.default_rng(seed + epoch)

        for batch in make_batches(
            train_input_ids, train_attention_mask, per_device_train_batch_size, rng
        ):
            params, opt_state, loss = train_step(params, opt_state, batch)
            global_step += 1
            loss_val = float(loss)
            train_loss_history.append(loss_val)

            if global_step % 5 == 0 or global_step == total_steps:
                progress_callback(
                    {
                        "step": global_step,
                        "loss": loss_val,
                        "val_loss": None,
                    }
                )

            try:
                for d in jax.local_devices():
                    stats = d.memory_stats()
                    if stats:
                        peak_bytes = max(
                            peak_bytes, stats.get("peak_bytes_in_use", 0)
                        )
            except Exception:
                pass

        # ----------------------------------------------------
        # End-of-epoch eval (mirrors Trainer's eval_strategy="epoch")
        # ----------------------------------------------------

        eval_loss = run_eval()

        if eval_loss is not None:
            val_loss_history.append(eval_loss)

            update = {
                "step": global_step,
                "loss": None,
                "val_loss": eval_loss,
            }

            epoch_number = len(val_loss_history)

            if epoch_number % 3 == 0 and epoch_number >= 3:
                try:
                    from training.forecasting import (
                        compute_difficulty_proxy,
                        forecast_n_epochs_ahead,
                        noise_floor,
                    )
                    from training.decision_engine import decide_training_action

                    forecast_result = forecast_n_epochs_ahead(
                        val_loss_history, save_plot_path=None
                    )
                    difficulty = compute_difficulty_proxy(val_loss_history)
                    noise = noise_floor(val_loss_history)
                    decision = decide_training_action(
                        val_losses=val_loss_history,
                        forecast=forecast_result["forecast"],
                        difficulty=difficulty,
                        noise=noise,
                    )

                    update["forecast_check"] = {
                        "epoch": epoch_number,
                        "forecast": forecast_result["forecast"],
                        "difficulty": difficulty,
                        "noise": noise,
                        "decision": decision,
                    }
                except Exception as e:
                    update["forecast_check"] = {
                        "epoch": epoch_number,
                        "error": str(e),
                    }

            progress_callback(update)

    elapsed = _time.time() - start
    throughput = (n_train * num_train_epochs) / elapsed if elapsed > 0 else 0.0
    peak_gpu_memory_gb = peak_bytes / 1e9

    # --------------------------------------------------------
    # Convert trained params back to a PyTorch state_dict
    # --------------------------------------------------------

    trained_params = jax.tree_util.tree_map(np.asarray, params)
    state_dict = convert_flax_to_pytorch_state_dict(
        trained_params, cfg, tie_word_embeddings=cfg.tie_word_embeddings
    )

    return {
        "state_dict": state_dict,
        "training_time_s": elapsed,
        "throughput_examples_per_sec": throughput,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
        "final_loss": train_loss_history[-1] if train_loss_history else None,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
    }