"""
PyTorch FSDP full-fine-tuning baseline for Qwen2.5-1.5B-Instruct on Alpaca.

Launch with:
    torchrun --nproc_per_node=<N_GPUS> train_pytorch_fsdp.py \
        --data tokenized_alpaca.npz --steps 100

Same logging schema as train_pytorch_ddp.py / train_jax.py so analysis.ipynb
can load all three results files without framework-specific parsing:
    {step, timestamp, step_time_sec, tokens_per_sec, peak_memory_mb, loss}

Key difference from DDP: instead of replicating the full model + optimizer
state on every GPU, FSDP shards params/gradients/optimizer state across
GPUs, gathering full weights per-layer only transiently during forward/
backward. This is what makes full fine-tuning of a 1.5B model feasible on
GPUs that couldn't hold the full optimizer state individually.
"""

import os
import json
import time
import argparse
import functools

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


class TokenizedDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.input_ids = torch.from_numpy(data["input_ids"]).long()
        self.attention_mask = torch.from_numpy(data["attention_mask"]).long()
        self.labels = torch.from_numpy(data["labels"]).long()

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="tokenized_alpaca.npz")
    parser.add_argument("--batch-size", type=int, default=1, help="per-GPU micro batch size")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=10, help="excluded from throughput stats")
    parser.add_argument("--out", type=str, default="results_fsdp.json")
    parser.add_argument(
        "--sharding-strategy",
        type=str,
        default="FULL_SHARD",
        choices=["FULL_SHARD", "SHARD_GRAD_OP", "HYBRID_SHARD"],
        help="FULL_SHARD = params+grads+optim all sharded (closest JAX-sharding analog). "
             "SHARD_GRAD_OP = only grads+optim sharded, params replicated (more like DDP+savings).",
    )
    args = parser.parse_args()

    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"World size: {world_size}, loading model {MODEL_NAME} in bf16 ...")
        print(f"Sharding strategy: {args.sharding_strategy}")

    # Load on CPU first / meta-ish to avoid every rank materializing a full
    # bf16 copy on GPU before sharding kicks in (matters more at larger scale;
    # kept simple here since 1.5B is small enough to load directly).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    # Wrap each Qwen2 decoder layer as its own FSDP unit. This is the standard
    # "auto wrap" pattern -- it determines the granularity at which params are
    # sharded/gathered. Wrapping at the transformer-block level (rather than
    # the whole model as one FSDP unit) is what allows the per-layer
    # gather-compute-discard cycle that keeps peak memory low.
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer},
    )

    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    sharding_strategy = getattr(ShardingStrategy, args.sharding_strategy)

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy,
        device_id=local_rank,
    )

    # NOTE: optimizer is constructed AFTER FSDP-wrapping the model, so the
    # optimizer only ever sees (and allocates state for) this rank's shard
    # of the parameters -- this is where FSDP's optimizer-state memory
    # savings actually come from.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    dataset = TokenizedDataset(args.data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True)

    seq_len = dataset.input_ids.shape[1]
    tokens_per_step = args.batch_size * world_size * seq_len

    logs = []
    step = 0
    data_iter = iter(loader)

    torch.cuda.reset_peak_memory_stats(device)
    model.train()

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            sampler.set_epoch(step)
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        step += 1
        step_time = t1 - t0

        if step > args.warmup_steps and rank == 0:
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            logs.append({
                "step": step,
                "timestamp": time.time(),
                "step_time_sec": step_time,
                "tokens_per_sec": tokens_per_step / step_time,
                "peak_memory_mb": peak_mem_mb,
                "loss": loss.item(),
            })
            if step % 10 == 0:
                print(f"[step {step}] loss={loss.item():.4f} "
                      f"tok/s={tokens_per_step / step_time:.1f} "
                      f"peak_mem={peak_mem_mb:.0f}MB")

    if rank == 0:
        with open(args.out, "w") as f:
            json.dump({
                "framework": "pytorch_fsdp",
                "sharding_strategy": args.sharding_strategy,
                "world_size": world_size,
                "seq_len": seq_len,
                "per_gpu_batch_size": args.batch_size,
                "warmup_steps_excluded": args.warmup_steps,
                "logs": logs,
            }, f, indent=2)
        print(f"Saved results to {args.out}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()