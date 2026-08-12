"""
training/run_log.py

Outcome logging: capture real fine-tuning results so future decisions
get better, rather than relying on guesses forever. Mirrors the same
idea as curation/curator.py's log_user_override().

log_finetune_outcome() records dataset stats + curation stats + the
actual final training result for every real, COMPLETED run — call it
once, right after finetune.run_finetune() returns, not from inside the
per-epoch callback (an outcome isn't meaningful until the run is
actually done, and calling it every epoch would just write repeated
near-duplicate rows of the same curation_stats/training_config).

This is training-data-shaped on purpose: once enough real runs
accumulate, this log is exactly the historical data you'd need to
eventually train or calibrate a REAL pre-training outcome predictor —
see check_dataset.py's assess_dataset_before_finetuning(), which is
the heuristic stand-in until that history exists.
"""

import json
import os
from datetime import datetime, timezone


def log_finetune_outcome(
    curation_stats: dict,
    training_config: dict,
    final_result: dict,
    log_path: str = "results/finetune_outcomes_log.jsonl",
) -> None:
    """
    Append-only log of a completed fine-tuning run's full context and
    outcome, mirroring curation's log_user_override() pattern.

    curation_stats: the classification dict from curation/curator.py's
        classify_dataset() — {"kept": [...], "rejected": [...],
        "failed": [...]} (lists of examples, same real shape
        check_dataset.assess_dataset_before_finetuning() expects —
        logged as-is here for a full record, not reduced to counts).
    training_config: e.g. {"model_name": ..., "method": "lora",
        "lora_r": 8, "num_epochs": ...} — from finetune.run_finetune()'s
        inputs.
    final_result: e.g. {"final_loss": ..., "best_val_loss": ...,
        "val_loss_history": [...], "decision_engine_actions": [...]}
        — from finetune.run_finetune()'s output plus any
        decision_engine.decide_training_action() calls made during
        the run.
    """
    record = {
        "curation_stats": curation_stats,
        "training_config": training_config,
        "final_result": final_result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_finetune_outcomes(log_path: str = "results/finetune_outcomes_log.jsonl") -> list[dict]:
    """
    Read back everything logged so far — this is what a future
    trained/calibrated pre-training predictor would train on, once
    there's enough history. Returns an empty list if nothing's been
    logged yet, rather than raising.
    """
    if not os.path.exists(log_path):
        return []
    with open(log_path) as f:
        return [json.loads(line) for line in f if line.strip()]