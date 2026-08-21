"""
Shared data prep for the DDP / FSDP / JAX benchmark.
Tokenizes a subsample of tatsu-lab/alpaca with the Qwen2.5-1.5B-Instruct tokenizer,
caps sequence length, and saves as a fixed-size tensor file so all three training
scripts train on IDENTICAL data (no framework-specific randomness in what's seen).
"""

import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

ALPACA_PROMPT = """### Instruction:
{instruction}
{input_block}
### Response:
{output}"""


def format_example(ex):
    input_block = f"\n### Input:\n{ex['input']}" if ex["input"].strip() else ""
    return ALPACA_PROMPT.format(
        instruction=ex["instruction"], input_block=input_block, output=ex["output"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-examples", type=int, default=3000,
                         help="how many Alpaca examples to use (benchmark, not full training)")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="tokenized_alpaca.npz")
    args = parser.parse_args()

    print(f"Loading tokenizer: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading tatsu-lab/alpaca ...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=args.seed).select(range(args.n_examples))

    texts = [format_example(ex) for ex in ds]

    print(f"Tokenizing {len(texts)} examples, max_length={args.max_length} ...")
    enc = tok(
        texts,
        max_length=args.max_length,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )

    input_ids = enc["input_ids"].astype(np.int32)
    attention_mask = enc["attention_mask"].astype(np.int32)

    # Causal LM labels = input_ids, with padding masked to -100
    labels = input_ids.copy()
    labels[attention_mask == 0] = -100

    np.savez(
        args.out,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    print(f"Saved {input_ids.shape[0]} examples of length {input_ids.shape[1]} to {args.out}")
    print(f"Shapes -> input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, labels: {labels.shape}")


if __name__ == "__main__":
    main()