# Smartune — Architecture & Feature Specification

Agentic fine-tuning pipeline: Claude Haiku curates and routes, LoRA/QLoRA
fine-tunes, Claude judges the result. Built to demonstrate model training,
evaluation, and pipeline engineering — not just AI system orchestration.

## Product Flow

1. **Welcome screen** — intro to what the tool does
2. **Upload dataset** — JSONL upload
3. **Preview dataset** — show sample rows, schema
   - 3.1 Remove specific samples manually before curation
4. **Define split** — what % of the dataset goes to train vs. validation
5. **Curation — yes/no** — user decides whether to run LLM curation at all
   - 6. If yes → choose threshold (Weak / Medium / Strong presets, or custom)
7. **Curation report** — per-example scores + reasons, kept/rejected counts
   - 7.1 Remove specific samples manually based on the report
8. **Fine-tune**
9. **Choose base model** — e.g. Qwen2.5-1.5B-Instruct, or others
10. **Choose method** — LoRA or Full Fine-Tuning; system automatically
    decides whether QLoRA (4-bit quantization) is needed based on model
    size vs. available GPU memory
11. **Live training curve** — loss plotted in real time
12. **Live training updates** — throughput, memory, step progress
13. **Recommendation to stop early** — LLM-based diagnostic flags
    overfitting/plateau from train vs. validation loss and suggests
    stopping or adjusting
14. **Review curve + problems after training** — full diagnostic pass
    once training completes
15. **Test + compare** — LLM-as-judge (Sonnet/Haiku) compares base vs.
    fine-tuned on held-out examples
16. **Final report** — training metrics, curation report, eval results,
    judge verdicts, all in one exportable document
17. **Reproducibility export** — save the entire run config (dataset
    sample, threshold, base model, method, hyperparameters, seed) so the
    exact run can be reproduced later

## Why this architecture

Static fine-tuning scripts are a solved, commoditized pattern. Smartune's
actual value is at the decision points most scripts hardcode:

- **Curation is a judgment call, not a fixed rule.** Rather than silently
  keeping or discarding data, Haiku scores each example against an
  explicit rubric, and the user sees and controls the threshold rather
  than trusting an opaque cutoff.
- **LoRA vs. full fine-tuning vs. QLoRA is an infrastructure decision,
  not a style preference.** The tool should reason about it (model size
  vs. available memory) rather than force the user to know the tradeoff
  in advance.
- **A loss curve going down does not mean the run is healthy.** Real
  diagnosis requires train vs. validation loss comparison, which is why
  step 4 (defining a validation split) and step 13 (stop recommendation)
  exist as first-class steps, not afterthoughts.
- **An LLM judge can be wrong or superficial** — this was confirmed
  directly during development (see Known Findings below). The tool
  should surface judge reasoning, not just a win/loss score, so a human
  can catch cases where the judge rewarded formatting over substance.

## Known Findings From Development

- `get_peft_model()` modifies the base model in place; generating a
  "baseline" after training without `model.disable_adapter()` silently
  reuses the fine-tuned weights. Any implementation must guard against
  this explicitly.
- Greedy decoding (`do_sample=False`) can mask real underlying weight
  changes from LoRA — identical top-1 tokens do not necessarily mean
  training had no effect. The tool should expose sampling parameters
  rather than hardcode greedy decoding.
- LLM judges (Haiku/Sonnet) can favor surface-level formatting matches
  over substantive answer quality. Manual inspection of raw outputs
  caught cases the automated judge scored incorrectly. This motivates
  always retaining raw outputs alongside judge verdicts in the final
  report, not just aggregate scores.

## Validated Results (reference run)

- Dataset: 50 examples sampled from Stanford Alpaca
- Curation: Haiku rubric-based scoring (clarity/correctness/value),
  threshold 6.7 → 35 kept, 15 rejected, with explicit reasoning per
  rejection
- Model: Qwen2.5-1.5B-Instruct, LoRA (r=8, q_proj/v_proj), 3 epochs
- Training: 170s, 4.73GB peak GPU memory, loss 2.45 → 1.93
- Evaluation: fine-tuned model won 5/10 held-out comparisons vs. base,
  4 ties, 1 base win (Haiku-as-judge); manual review confirmed the
  fine-tuned model avoided a degenerate repetition failure the base
  model exhibited on one example

## Extension: Full Fine-Tuning + JAX/Flax vs. PyTorch Benchmark

To support full fine-tuning (not just LoRA/QLoRA), Smartune was extended
with a from-scratch JAX/Flax reimplementation of a Transformer (Qwen2)
architecture, benchmarked against PyTorch DDP and FSDP across GPU and
Google Cloud TPU (`training/jax_v_pytorch/`). Before trusting any JAX
numbers, `parity_check.py` verifies the hand-written Flax port's logits
against the real PyTorch model — a first pass failed at a max diff of
0.13 against a 0.05 tolerance despite matching top-1 predictions, traced
to JAX defaulting to bfloat16-precision matmuls on TPU even for float32
arrays; forcing `jax_default_matmul_precision="float32"` dropped the diff
to 0.000034.

Results (Qwen2.5-1.5B fine-tuning on Alpaca, steady-state over 100 steps):

| | 2× T4 (GPU) | TPU v6e-4 |
|---|---|---|
| PyTorch DDP | not run | 3,418 tok/s · 26.8GB peak |
| PyTorch FSDP | 371 tok/s · 7.8GB peak | 3,219 tok/s · 6.2GB peak |
| JAX (this work) | 429 tok/s · 10.0GB peak | 27,006 tok/s · 9.9GB peak |

![JAX vs PyTorch throughput and memory, GPU vs TPU](results/jax_v_pytorch/barchart_1.png)
![Fine-tuning throughput comparison](results/jax_v_pytorch/barchart_throughput.png)
![Peak memory footprint comparison](results/jax_v_pytorch/barchart_memory.png)

- JAX matched PyTorch/XLA FSDP throughput on GPU, but ran ~8x faster on
  TPU (XLA-compiled) than PyTorch/XLA DDP (27,006 vs. 3,418 tok/s) — and
  the same JAX code is itself 63x faster on TPU than on the 2xT4 GPU pair.
- FSDP-style sharding cut peak memory ~4.4x versus DDP (26.8GB → 6.2GB
  per chip on a v6e-4 TPU).

This benchmark is a separate research artifact from the interactive
dashboard pipeline — it is not currently wired into `training/finetune.py`
as a callable, end-to-end training path (the dashboard's "Full
Fine-Tuning" method is marked "coming soon" for that reason; LoRA/QLoRA
is the only method that runs live through the UI today). See
`notebooks/JAX vs PyTorch Analysis.ipynb` and
`notebooks/JAX_v_PyTorch_T4.ipynb` for the full write-up, and
`results/jax_v_pytorch/` for the raw result files.

## Extension: Forecasting-Method Comparison

`training/forecasting.py` originally offered two forecasting arms: Arm A
(Domhan-style parametric curve extrapolation, Domhan et al. 2015) and Arm
B (LC-PFN, Adriaensen et al. 2023). A controlled comparison across 10 real
fine-tuning runs (`notebooks/Forecasting_Feature_Research.ipynb`,
`results/forcasting_results/`) tested both against a fixed-budget null
baseline at multiple epoch-cutoff sample sizes. An apparent LC-PFN
advantage at n=8 disappeared at n=160 — Arm A was more accurate on
average and in head-to-head win count at every cutoff tested except one
statistically indistinguishable case, and its advantage grew with more
observed epochs. The larger finding: how much forecast information was
available (epochs observed) mattered more than which method did the
extrapolating. Arm A is now the sole forecasting method in production
(`arm_a_forecast_with_uncertainty`); LC-PFN was dropped as a runtime
dependency entirely, since it also required bypassing its own PyPI
version pins and patching for modern PyTorch to install at all.

![Forecast MAPE by cutoff, 10-epoch vs 15-epoch runs](results/forcasting_results/barchart.png)
![Per-run forecast accuracy comparison](results/forcasting_results/dumbbell_plot.png)

Same 10 runs, more observed epochs, opposite statistical conclusion: with
only 10 epochs of signal, Arm A wins 8/10 runs but the gap isn't
significant (Wilcoxon p=0.11); with 15 epochs of signal, Arm A wins
10/10, now significantly (p=0.002). Observation window, not method
choice, is what moved the result from noise to signal.

## Tech Stack

- **Curation / Judging:** Claude Haiku (fast, cheap, rubric scoring),
  Claude Sonnet (deeper judge comparisons where needed)
- **Fine-tuning:** Hugging Face `transformers`, `peft` (LoRA/QLoRA),
  `accelerate`, `bitsandbytes` (quantization)
- **Full fine-tuning / benchmark:** JAX, Flax, Optax (`jax_v_pytorch/`),
  PyTorch DDP/FSDP via `torch_xla` for the TPU comparison
- **Dashboard:** Streamlit; `dashboard/app.py` is the user-facing
  pipeline UI wired directly to the curation/training/forecasting/
  evaluation modules below (not a mockup); `dashboard/dev_dashboard.py`
  is a developer harness that exercises each backend function in
  isolation with synthetic data where a GPU or API key isn't available

## Implemented (originally scoped as future work)

- **Dataset diversity check before curation** — `curation/curator.py`'s
  `find_duplicate_candidates` / `verify_duplicates_with_haiku` catch
  near-duplicate examples via embedding similarity clustering before
  per-example scoring, since curation alone catches per-example
  *quality*, not dataset-wide *redundancy*.
- **Rejected-sample review and override** — the curation report step
  shows every rejected example with its reason; `apply_manual_overrides`
  lets the user accept/restore any of them back into the training set,
  keyed by each example's stable `_id`.
- **Human-override log** — `log_user_override` records every manual
  accept/reject/restore decision, distinct from the model's own
  judgment (`results/user_override_log.jsonl`).
- **Dataset schema auto-detection** — `curation/preprocess.py` normalizes
  raw uploads (Alpaca instruction/input/output, prompt/completion,
  chat-message formats) to the pipeline's `question`/`answer` schema via
  heuristics, falling back to asking Claude to map fields only when the
  schema is unrecognized.

## Repo Structure (actual)

```
smartune/
├── curation/
│   ├── curator.py          # Haiku scoring, duplicate detection, overrides, reports
│   └── preprocess.py       # schema auto-detection for raw uploads
├── training/
│   ├── finetune.py         # LoRA/QLoRA fine-tuning, auto-QLoRA decision
│   ├── forecasting.py      # Arm A parametric curve extrapolation
│   ├── decision_engine.py  # stop/continue/flag decision from forecast
│   ├── check_dataset.py    # pre-training dataset warnings
│   ├── report.py           # training report assembly
│   ├── reproducibility.py  # run_config export
│   ├── run_log.py          # outcome logging
│   └── jax_v_pytorch/      # JAX/Flax vs. PyTorch DDP/FSDP benchmark scripts
│                           # (research artifact — not yet wired into
│                           # finetune.py as a callable full fine-tuning path)
├── evaluation/
│   ├── eval_harness.py     # generation (base vs. fine-tuned)
│   ├── llm_judge.py        # LLM-as-judge comparison
│   └── report_final.py     # eval summary + cross-check vs. training curve
├── dashboard/
│   ├── app.py              # Streamlit UI — the user-facing pipeline
│   └── dev_dashboard.py    # developer/test harness
├── notebooks/               # exploratory research and comparisons
├── results/                 # exported reports, run configs, benchmark results
├── requirements.txt
└── README.md
```
