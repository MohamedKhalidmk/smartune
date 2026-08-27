"""
training/decision_engine.py

Claude decides what to do about a training run using pre-computed,
structured signals from forecasting.py.

The numerical analysis is deterministic. Claude's role is only to
reason over the structured signals and choose a constrained action.

Valid actions:
    STOP
    CONTINUE
    CONTINUE_MORE_EPOCHS
    UNLIKELY_TO_SUCCEED
"""
from __future__ import annotations

import json

import anthropic

from evaluation.llm_trace import traced_claude_call


# ============================================================
# Configuration
# ============================================================

client = anthropic.Anthropic()

VALID_ACTIONS = {
    "STOP",
    "CONTINUE",
    "CONTINUE_MORE_EPOCHS",
    "UNLIKELY_TO_SUCCEED",
}


# ============================================================
# Claude prompt
# ============================================================

DECISION_PROMPT = """You are monitoring a fine-tuning run and deciding what action to recommend, based ONLY on the structured signals below — not on any other assumption.

Observed validation losses so far: {val_losses}

Forecast (curve extrapolation, from the current point forward):
  median: {median}
  5th percentile: {lower_5}
  95th percentile: {upper_95}

Difficulty signals (derived from this curve's own early dynamics, DA-LCE-style):
  rate of progress: {prog:.4f}
  non-linearity: {nonlin:.4f}
  volatility: {vol:.4f}

Noise floor (typical epoch-to-epoch jitter for this specific run): {noise:.4f}

Choose exactly one action from this fixed set:

  "STOP" — the forecast shows validation loss has plateaued or is
  trending worse; further training is unlikely to help.

  "CONTINUE" — the forecast shows genuine, meaningful improvement
  ahead; keep training as planned.

  "CONTINUE_MORE_EPOCHS" — early signal is positive but the curve
  hasn't converged; recommend extending beyond the currently planned
  epoch count specifically.

  "UNLIKELY_TO_SUCCEED" — the difficulty signals (very high
  non-linearity and/or volatility relative to the noise floor) suggest
  this specific dataset/configuration is unlikely to converge well at
  all, regardless of how many more epochs are run. This is a signal
  about the DATA/SETUP, not just this point in training.

Then decide notify_user: true only if this action is genuinely
actionable right now (STOP, UNLIKELY_TO_SUCCEED, or a clear
CONTINUE_MORE_EPOCHS recommendation). Use false for a routine
CONTINUE where nothing needs the user's attention yet.

Respond ONLY with JSON:

{{
    "action": "<one of the four above>",
    "notify_user": <true or false>,
    "reason": "<2-3 sentences, referencing the actual numbers above>"
}}
"""


# ============================================================
# Main decision function
# ============================================================

def decide_training_action(
    val_losses: list[float],
    forecast: dict,
    difficulty: dict,
    noise: float,
) -> dict:
    """
    Decide what action to take for the current fine-tuning run.

    Parameters
    ----------
    val_losses:
        Validation-loss history collected during training.

    forecast:
        Output of forecasting.arm_a_forecast_with_uncertainty().

    difficulty:
        Output of forecasting.compute_difficulty_proxy().

    noise:
        Output of forecasting.noise_floor().

    Returns
    -------
    dict
        {
            "action": one of VALID_ACTIONS,
            "notify_user": bool,
            "reason": str
        }

    If Claude returns malformed JSON or an invalid action, the function
    falls back to CONTINUE without notifying the user. The failure is
    explicitly included in the reason instead of silently producing an
    invalid decision.
    """

    # --------------------------------------------------------
    # Build prompt from deterministic signals
    # --------------------------------------------------------

    prompt = DECISION_PROMPT.format(
        val_losses=val_losses,
        median=forecast["median"],
        lower_5=forecast["lower_5"],
        upper_95=forecast["upper_95"],
        prog=difficulty["prog"],
        nonlin=difficulty["nonlin"],
        vol=difficulty["vol"],
        noise=noise,
    )

    # --------------------------------------------------------
    # Call Claude
    # --------------------------------------------------------

    response = traced_claude_call(
        client,
        "training.decision_engine",
        "decide_training_action",
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # --------------------------------------------------------
    # Remove optional Markdown code fences
    # --------------------------------------------------------

    if text.startswith("```"):
        text = text.split("```")[1]

        if text.startswith("json"):
            text = text[4:]

        text = text.strip()

    # --------------------------------------------------------
    # Parse and validate Claude's response
    # --------------------------------------------------------

    try:
        result = json.loads(text)

        action = result.get("action")

        if action not in VALID_ACTIONS:
            raise ValueError(
                f"Action {action!r} not in {VALID_ACTIONS}"
            )

        # Fail toward notifying rather than silently skipping a
        # potentially important recommendation.
        result.setdefault("notify_user", True)

        return result

    # --------------------------------------------------------
    # Safe fallback
    # --------------------------------------------------------

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return {
            "action": "CONTINUE",
            "notify_user": False,
            "reason": (
                "Decision engine failed to produce a valid response "
                f"({e}) — defaulting to CONTINUE without notifying, "
                "pending manual review."
            ),
        }