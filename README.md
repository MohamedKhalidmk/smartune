# Smartune

Agentic fine-tuning pipeline: Claude Haiku curates and routes, LoRA/QLoRA
fine-tunes, Claude judges the result.

See `ARCHITECTURE.md` for the full product flow, design rationale, and
known findings from development (including two real bugs hit and fixed
during the build — see "Known Findings").

## Status

Core pipeline validated end-to-end in a Kaggle notebook (Qwen2.5-1.5B,
LoRA, 50-example Alpaca sample). Currently being ported into the modular
structure below for the interactive dashboard version.

## Structure

- `curation/` — Haiku-based dataset scoring and thresholding
- `training/` — LoRA/full fine-tuning, auto-QLoRA decision
- `evaluation/` — generation + LLM-as-judge comparison
- `diagnostics/` — training curve health diagnosis (overfitting/underfitting)
- `dashboard/` — Streamlit UI wiring the above together
- `results/` — exported reports and run configs

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add your ANTHROPIC_API_KEY
streamlit run dashboard/app.py
```

## Validated Reference Run

- 50 examples (Stanford Alpaca), Haiku curation at threshold 6.7 → 35 kept
- Qwen2.5-1.5B-Instruct, LoRA (r=8, q_proj/v_proj), 3 epochs, 170s, 4.73GB peak memory
- Fine-tuned model won 5/10 held-out comparisons vs. base (Haiku-as-judge), 4 ties, 1 loss

Full details in `ARCHITECTURE.md`.
