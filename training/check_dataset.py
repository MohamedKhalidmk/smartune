"""
training/check_dataset.py

Pre-training dataset checks that run BEFORE fine-tuning starts.

These checks use only the curated dataset because no training curve
exists yet.

1. assess_dataset_before_finetuning()
   A heuristic, rule-based sanity check. This is explicitly NOT a
   trained predictive model and is not equivalent to the curve-based
   forecaster (forecasting.py), which requires an observed partial
   training curve.

2. decide_dataset_warning()
   Claude decides whether the heuristic findings are significant
   enough to warn the user before spending time and compute on
   fine-tuning.

The heuristic can eventually be replaced or calibrated using historical
dataset-statistics-to-training-outcome data collected by
run_log.log_finetune_outcome().
"""

import json

import anthropic

from evaluation.llm_trace import traced_claude_call


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Conservative generic rules of thumb. These are not calibrated against
# historical training outcomes.
MIN_RECOMMENDED_EXAMPLES = 20
MIN_RECOMMENDED_KEEP_RATE = 0.3


# ---------------------------------------------------------------------------
# Pre-training heuristic
# ---------------------------------------------------------------------------

def assess_dataset_before_finetuning(classification: dict) -> dict:
    """
    Run a heuristic pre-training dataset sanity check.

    This is NOT a trained predictive model and is NOT equivalent to
    Arm B's forecasting. Forecasting requires an observed partial
    training curve, while this function runs before training begins.

    Args:
        classification:
            Output of curation.curator.classify_dataset() or
            apply_manual_overrides():

            {
                "kept": [...],
                "rejected": [...],
                "failed": [...]
            }

    Returns:
        {
            "warning": bool,
            "reasons": [str, ...],
            "note": str
        }

    The function uses the actual classification lists and derives
    counts with len(). Passing integer counts instead of lists is not
    supported.
    """
    kept = len(classification.get("kept", []))
    rejected = len(classification.get("rejected", []))
    failed = len(classification.get("failed", []))
    total = kept + rejected + failed

    reasons = []

    # Check whether the final curated dataset is very small.
    if kept < MIN_RECOMMENDED_EXAMPLES:
        reasons.append(
            f"Only {kept} examples kept after curation "
            f"(recommended minimum: {MIN_RECOMMENDED_EXAMPLES}) — "
            "very small training sets often produce noisy, "
            "hard-to-interpret loss curves."
        )

    # Check whether an unusually small fraction of the source dataset
    # survived curation.
    if total > 0 and (kept / total) < MIN_RECOMMENDED_KEEP_RATE:
        reasons.append(
            f"Only {kept}/{total} ({100 * kept / total:.0f}%) of raw "
            "examples passed curation — a low keep-rate can mean the "
            "source data itself is poor quality for this task, not "
            "just that curation was strict."
        )

    return {
        "warning": len(reasons) > 0,
        "reasons": reasons,
        "note": (
            "This is a heuristic sanity check based on generic "
            "dataset-size/quality rules of thumb — it is NOT a "
            "trained predictive model, and it cannot forecast "
            "training success the way Arm B does once a real curve "
            "exists. Log enough real outcomes via "
            "run_log.log_finetune_outcome() to eventually calibrate "
            "this against actual historical results."
        ),
    }


# ---------------------------------------------------------------------------
# Claude warning decision
# ---------------------------------------------------------------------------

def decide_dataset_warning(heuristic_result: dict) -> dict:
    """
    Ask Claude whether the heuristic findings are significant enough
    to interrupt the user with a warning.

    Claude does NOT inspect the dataset itself. It only reasons over
    the structured findings produced by
    assess_dataset_before_finetuning().

    If the heuristic found no issues, Claude is not called.

    Returns:
        {
            "warn_user": bool,
            "message": str
        }
    """
    if not heuristic_result["warning"]:
        return {
            "warn_user": False,
            "message": "",
        }

    prompt = f"""
A heuristic pre-training check flagged the following issue(s) with a
dataset about to be used for fine-tuning:

{chr(10).join("- " + reason for reason in heuristic_result["reasons"])}

Decide whether this is genuinely worth warning the user about before
they spend time/compute on this run.

If it is worth warning them, write a short 1-2 sentence,
plain-language warning.

Note: this heuristic is based on generic rules of thumb, not a
calibrated model. Factor that uncertainty into your decision. A
borderline case might not warrant interrupting the user.

Respond ONLY with JSON:
{{"warn_user": <true or false>, "message": "<1-2 sentences, or empty string if warn_user is false>"}}
"""

    client = anthropic.Anthropic()

    response = traced_claude_call(
        client,
        "training.check_dataset",
        "decide_dataset_warning",
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Handle ```json ... ``` responses.
    if text.startswith("```"):
        text = text.split("```")[1]

        if text.startswith("json"):
            text = text[4:]

        text = text.strip()

    try:
        result = json.loads(text)

        # Fail toward warning rather than silently hiding a detected issue.
        result.setdefault("warn_user", True)
        result.setdefault(
            "message",
            "; ".join(heuristic_result["reasons"]),
        )

        return result

    except json.JSONDecodeError:
        # If Claude's response cannot be parsed, preserve the original
        # deterministic warning rather than silently dropping it.
        return {
            "warn_user": True,
            "message": "; ".join(heuristic_result["reasons"]),
        }