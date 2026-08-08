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

## Tech Stack

- **Curation / Judging:** Claude Haiku (fast, cheap, rubric scoring),
  Claude Sonnet (deeper judge comparisons where needed)
- **Fine-tuning:** Hugging Face `transformers`, `peft` (LoRA/QLoRA),
  `accelerate` (multi-GPU), `bitsandbytes` (quantization)
- **Dashboard (planned):** Streamlit for the interactive control panel;
  core pipeline logic ported from the validated notebook into standalone
  modules (curation, training, evaluation) so the UI is a thin wrapper
  around tested logic, not new untested code

## Future Additions

- **Dataset diversity check before curation** — before running per-example
  LLM scoring, check for near-duplicate examples via embedding similarity
  clustering. Curation catches per-example *quality*, not dataset-wide
  *redundancy* — a large set of near-identical examples could all score
  well individually while adding little combined training value.
- **Rejected-sample review and override** — the curation report (step 7)
  shows every rejected example with its reason, and the user can manually
  accept/restore any of them back into the training set. Haiku's
  judgment is a strong default, not a final word.
- **Human-override log** — every time a user manually removes or restores
  a sample (steps 3.1, 7.1, or the rejected-sample override above), log
  the action and, optionally, the user's stated reason. Over time this
  becomes a record of human curation decisions distinct from the model's
  own — potentially useful later for auditing or improving the curator.

## Repo Structure (target)

```
smartune/
├── data/
├── curation/
│   └── haiku_curator.py
├── training/
│   └── finetune.py
├── evaluation/
│   ├── eval_harness.py
│   └── llm_judge.py
├── diagnostics/
│   └── training_diagnostics.py    # overfitting/underfitting detection
├── dashboard/
│   └── app.py                     # Streamlit UI
├── results/
├── main.py
├── requirements.txt
└── README.md
```
