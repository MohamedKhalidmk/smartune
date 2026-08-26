"""
JAX full-fine-tuning benchmark for the Flax Qwen2 port (qwen2_flax.py) on
Alpaca, using FSDP-style sharding (params + optimizer state sharded across
devices via a device mesh) -- the closest JAX analog to PyTorch FSDP's
FULL_SHARD strategy.

IMPORTANT: run parity_check.py first and confirm it PASSES before trusting
any numbers from this script. This script does not re-verify correctness --
it assumes qwen2_flax.py + the weight conversion are already validated.

Launch (single process, multi-GPU visible to JAX automatically):
    python train_jax.py --data tokenized_alpaca.npz --steps 100

Same logging schema as train_pytorch_ddp.py / train_pytorch_fsdp.py:
    {step, timestamp, step_time_sec, tokens_per_sec, peak_memory_mb, loss}

Notes on what's JAX-specific here vs the PyTorch scripts:
  - No explicit backward() call -- jax.value_and_grad computes gradients
    functionally, returning them rather than mutating a .grad attribute.
  - No optimizer.step() mutating state in place -- optax returns new
    (params, opt_state), reassigned each step (functional update).
  - The first `jit`-compiled step will be slow (XLA compilation) -- this
    is logged separately as compile_time_sec and EXCLUDED from steady-state
    tokens_per_sec, matching how PyTorch's warmup steps are excluded (same
    intent -- isolate one-time cost from steady-state throughput -- 
    different underlying cause).
"""

import os
import json
import time
import argparse
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from transformers import AutoModelForCausalLM, AutoConfig
import torch

from Qwen_flax import Qwen2Config, Qwen2ForCausalLM
from parity_check import convert_pytorch_to_flax_params

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_sharded_params(cfg: Qwen2Config, mesh: Mesh):
    """
    Loads official PyTorch weights, converts to the Flax pytree, and shards
    every array across the mesh's 'data' axis (FSDP-style: each device holds
    a slice of every parameter, not a full replica).
    """
    print("Loading PyTorch weights for conversion ...")
    pt_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    params = convert_pytorch_to_flax_params(pt_model.state_dict(), cfg)
    del pt_model  # free CPU RAM once converted

    # Shard each leaf array along its first axis across the mesh. This is a
    # simple 1D FSDP-style sharding -- fine for a benchmark; production JAX
    # FSDP setups often use more careful per-tensor sharding specs, but this
    # gives the same core effect (no device holds a full param replica).
    sharding = NamedSharding(mesh, P("data"))

    def shard_leaf(x):
        if x.ndim == 0:
            return jax.device_put(x, NamedSharding(mesh, P()))  # scalars: replicate
        return jax.device_put(x, sharding)

    return jax.tree_util.tree_map(shard_leaf, params)


def make_causal_lm_loss(model: Qwen2ForCausalLM):
    def loss_fn(params, batch):
        logits = model.apply(params, batch["input_ids"], batch["attention_mask"])
        # shift for next-token prediction
        logits = logits[:, :-1, :]
        labels = batch["labels"][:, 1:]
        mask = labels != -100
        labels_safe = jnp.where(mask, labels, 0)
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        nll = -jnp.take_along_axis(log_probs, labels_safe[..., None], axis=-1)[..., 0]
        nll = jnp.where(mask, nll, 0.0)
        loss = nll.sum() / jnp.maximum(mask.sum(), 1)
        return loss
    return loss_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="tokenized_alpaca.npz")
    parser.add_argument("--batch-size", type=int, default=1, help="per-device micro batch size")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--out", type=str, default="results_jax.json")
    args = parser.parse_args()

    devices = jax.devices()
    n_devices = len(devices)
    print(f"JAX sees {n_devices} device(s): {devices}")

    mesh = Mesh(np.array(devices), axis_names=("data",))

    hf_config = AutoConfig.from_pretrained(MODEL_NAME)
    cfg = Qwen2Config.from_hf_config(hf_config)
    model = Qwen2ForCausalLM(cfg)

    with mesh:
        params = load_sharded_params(cfg, mesh)

        loss_fn = make_causal_lm_loss(model)
        grad_fn = jax.value_and_grad(loss_fn)

        optimizer = optax.adamw(args.lr)
        # optimizer state mirrors params' sharding -- optax state (m, v moments)
        # has the same pytree structure as params, so tree_map with the same
        # sharding function keeps it sharded too.
        opt_state = optimizer.init(params)

        data_sharding = NamedSharding(mesh, P("data"))

        @jax.jit
        def train_step(params, opt_state, batch):
            loss, grads = grad_fn(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        data = np.load(args.data)
        input_ids_all = data["input_ids"]
        attention_mask_all = data["attention_mask"]
        labels_all = data["labels"]
        n_examples = input_ids_all.shape[0]
        seq_len = input_ids_all.shape[1]

        global_batch_size = args.batch_size * n_devices
        tokens_per_step = global_batch_size * seq_len

        rng = np.random.default_rng(seed=42)
        logs = []
        compile_time_sec = None

        for step in range(1, args.steps + 1):
            idx = rng.choice(n_examples, size=global_batch_size, replace=False)
            batch = {
                "input_ids": jax.device_put(jnp.array(input_ids_all[idx]), data_sharding),
                "attention_mask": jax.device_put(jnp.array(attention_mask_all[idx]), data_sharding),
                "labels": jax.device_put(jnp.array(labels_all[idx]), data_sharding),
            }

            t0 = time.perf_counter()
            params, opt_state, loss = train_step(params, opt_state, batch)
            loss.block_until_ready()  # JAX is async -- must block to get real wall-clock time
            t1 = time.perf_counter()

            step_time = t1 - t0

            if step == 1:
                # first call includes jit compilation -- log separately, exclude from throughput stats
                compile_time_sec = step_time
                print(f"[step 1] compile+run time: {compile_time_sec:.2f}s (excluded from steady-state stats)")
                continue

            peak_mem_mb = max(
                d.memory_stats()["peak_bytes_in_use"] / (1024 ** 2)
                for d in devices if d.memory_stats() is not None
            )

            logs.append({
                "step": step,
                "timestamp": time.time(),
                "step_time_sec": step_time,
                "tokens_per_sec": tokens_per_step / step_time,
                "peak_memory_mb": peak_mem_mb,
                "loss": float(loss),
            })

            if step % 10 == 0:
                print(f"[step {step}] loss={float(loss):.4f} "
                      f"tok/s={tokens_per_step / step_time:.1f} "
                      f"peak_mem={peak_mem_mb:.0f}MB")

        with open(args.out, "w") as f:
            json.dump({
                "framework": "jax_fsdp_style",
                "n_devices": n_devices,
                "seq_len": seq_len,
                "per_device_batch_size": args.batch_size,
                "compile_time_sec": compile_time_sec,
                "logs": logs,
            }, f, indent=2)
        print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()