"""
training/reproducibility.py

Exports everything needed to exactly reproduce a run: which dataset
was used and how it was curated, the training config including the
now-genuinely-pinned seed (see finetune.run_finetune()'s seed
parameter — this was NOT pinned anywhere before, meaning "reproduce
this run" wasn't actually possible until that fix), and the resulting
metrics for reference.

This was originally planned back in ARCHITECTURE.md (step 17) but
never actually built while the project's focus was on the forecasting
work — this closes that gap.
"""

from __future__ import annotations


import json
import os
from datetime import datetime, timezone


def export_run_config(
    dataset_source: dict,
    curation_config: dict,
    training_config: dict,
    training_result: dict,
    export_path: str | None = None,
) -> dict:
    """
    dataset_source: e.g. {"source": "alpaca", "sample_seed": 100,
        "train_size": 40, "val_size": 10} — however the dataset was
        actually obtained, so the same source can be reconstructed.
    curation_config: e.g. {"threshold": 6.7, "mode": "normal"} — from
        curation/curator.py's classify_dataset() call.
    training_config: e.g. {"model_name": ..., "method": "lora",
        "lora_r": 8, "num_train_epochs": ..., "learning_rate": ...} —
        the actual arguments passed to finetune.run_finetune().
    training_result: the dict returned by run_finetune() — used here
        specifically to pull out "seed", so the exported config
        captures the REAL seed that was actually used, not just
        whatever was intended (they should match, but this is honest
        about pulling the value that was actually applied).

    Returns the full config dict. If export_path is given, also writes
    it to that path as JSON (not append-only like the other logs in
    this project — each run's config is a standalone, complete
    snapshot meant to be read on its own, not queried as a growing
    history the way llm_call_trace.jsonl or finetune_outcomes_log.jsonl
    are).
    """
    config = {
        "dataset_source": dataset_source,
        "curation_config": curation_config,
        "training_config": training_config,
        "actual_seed_used": training_result.get("seed"),
        "final_loss": training_result.get("final_loss"),
        "val_loss_history": training_result.get("val_loss_history"),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }

    if export_path:
        os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
        with open(export_path, "w") as f:
            json.dump(config, f, indent=2)

    return config


def load_run_config(export_path: str) -> dict:
    """
    Read back an exported config — this is what you'd feed back into
    the pipeline (same dataset_source to re-sample, same curation_config,
    same training_config including actual_seed_used) to reproduce the
    run exactly.
    """
    with open(export_path) as f:
        return json.load(f)