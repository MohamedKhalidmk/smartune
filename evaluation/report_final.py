"""
evaluation/report.py

Two reports from judge_outputs()'s results:

1. Summary report — win/tie/loss counts, average quality margin
   (magnitude of improvement, not just a count), a regression flag
   (examples where fine-tuned scored meaningfully WORSE than base —
   a good win-rate can hide this), and a cross-check against the
   actual training curve (did loss go down AND did judged quality
   actually improve, or do the two signals disagree).

2. Detailed report — every example, both outputs, both score sets,
   the judge's reasoning — same one-report-per-run pattern as
   training/report.py, for consistency across the pipeline.
"""

REGRESSION_THRESHOLD = 1.5  # avg score drop (0-10 scale) considered a
                             # genuine regression, not just noise/a tie


def _avg_score(scores: dict | None) -> float | None:
    if not scores:
        return None
    return sum(scores.values()) / len(scores)


def compute_summary(judged_results: list[dict]) -> dict:
    """
    Computes the three-part score discussed and agreed on:
    win/tie/loss counts, average quality margin, and a regression list
    — deliberately not collapsed into one single number, since a
    single number would hide exactly the nuance (won by a lot vs.
    barely; consistent vs. occasionally catastrophic) that matters for
    judging whether a fine-tune is actually good.
    """
    valid = [r for r in judged_results if r["winner"] != "JUDGE_FAILED"]
    failed_count = len(judged_results) - len(valid)

    wins = sum(1 for r in valid if r["winner"] == "B")
    losses = sum(1 for r in valid if r["winner"] == "A")
    ties = sum(1 for r in valid if r["winner"] == "tie")

    margins = []
    regressions = []
    for r in valid:
        avg_a = _avg_score(r["score_a"])
        avg_b = _avg_score(r["score_b"])
        if avg_a is None or avg_b is None:
            continue
        margin = avg_b - avg_a
        margins.append(margin)
        if margin <= -REGRESSION_THRESHOLD:
            regressions.append({
                "question": r["question"],
                "margin": margin,
                "reason": r["reason"],
            })

    avg_quality_margin = sum(margins) / len(margins) if margins else None

    return {
        "total_examples": len(judged_results),
        "judge_failures": failed_count,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "avg_quality_margin": avg_quality_margin,
        "regressions": regressions,
    }


def cross_check_with_training_curve(val_loss_history: list[float], avg_quality_margin: float | None) -> dict:
    """
    Checks whether the two independent signals AGREE: did validation
    loss actually go down during training (training/finetune.py's
    val_loss_history), and did judged output quality actually improve
    (avg_quality_margin > 0)? A fine-tune can show lower loss with no
    real quality gain, or vice versa — this flags that mismatch rather
    than silently trusting either signal alone.

    Returns {"loss_improved": bool | None, "quality_improved": bool | None,
             "signals_agree": bool | None, "note": str}.
    """
    if len(val_loss_history) < 2:
        return {
            "loss_improved": None, "quality_improved": None, "signals_agree": None,
            "note": "Not enough validation loss history to compare (need at least 2 points).",
        }
    if avg_quality_margin is None:
        return {
            "loss_improved": None, "quality_improved": None, "signals_agree": None,
            "note": "No valid judged quality margin available to compare (all judge calls may have failed).",
        }

    loss_improved = val_loss_history[-1] < val_loss_history[0]
    quality_improved = avg_quality_margin > 0
    signals_agree = loss_improved == quality_improved

    if signals_agree:
        note = (
            "Loss and judged quality agree: "
            + ("both improved." if loss_improved else "neither improved.")
        )
    else:
        note = (
            "MISMATCH: "
            f"validation loss {'improved' if loss_improved else 'did not improve'}, "
            f"but judged output quality {'improved' if quality_improved else 'did not improve'}. "
            "Lower loss doesn't always mean better generated output quality, and vice versa — "
            "worth reviewing the detailed report before trusting either signal alone."
        )

    return {
        "loss_improved": loss_improved,
        "quality_improved": quality_improved,
        "signals_agree": signals_agree,
        "note": note,
    }


def generate_summary_report(summary: dict, cross_check: dict) -> str:
    lines = ["# Fine-Tuning Evaluation — Summary Report", ""]

    lines.append("## Win / Tie / Loss")
    lines.append("")
    lines.append(f"- Fine-tuned wins: {summary['wins']}")
    lines.append(f"- Base wins: {summary['losses']}")
    lines.append(f"- Ties: {summary['ties']}")
    if summary["judge_failures"]:
        lines.append(f"- Judge failures (excluded from counts above): {summary['judge_failures']}")
    lines.append("")

    lines.append("## Quality Margin")
    lines.append("")
    margin = summary["avg_quality_margin"]
    if margin is not None:
        direction = "improvement" if margin > 0 else ("regression" if margin < 0 else "no change")
        lines.append(f"- Average score margin (fine-tuned − base): {margin:+.2f} ({direction})")
    else:
        lines.append("- No valid margin available (all judge calls may have failed).")
    lines.append("")

    lines.append("## Regressions")
    lines.append("")
    if summary["regressions"]:
        lines.append(f"**{len(summary['regressions'])} example(s) where fine-tuned scored meaningfully worse than base:**")
        lines.append("")
        for reg in summary["regressions"]:
            lines.append(f"- \"{reg['question']}\" — margin {reg['margin']:+.2f} — {reg['reason']}")
    else:
        lines.append("No regressions detected (no example dropped by more than the regression threshold).")
    lines.append("")

    lines.append("## Training Curve Cross-Check")
    lines.append("")
    lines.append(f"- {cross_check['note']}")

    return "\n".join(lines)


def generate_detailed_report(judged_results: list[dict]) -> str:
    lines = ["# Fine-Tuning Evaluation — Detailed Report", ""]

    for i, r in enumerate(judged_results, start=1):
        lines.append(f"## Example {i}")
        lines.append("")
        lines.append(f"**Q:** {r['question']}")
        lines.append("")
        lines.append(f"**Reference:** {r['reference']}")
        lines.append("")
        lines.append(f"**Base output:** {r['base_output']}")
        lines.append("")
        lines.append(f"**Fine-tuned output:** {r['ft_output']}")
        lines.append("")
        if r["winner"] == "JUDGE_FAILED":
            lines.append(f"**Judge failed:** {r['reason']}")
        else:
            lines.append(f"**Base scores:** {r['score_a']}")
            lines.append(f"**Fine-tuned scores:** {r['score_b']}")
            lines.append(f"**Winner:** {r['winner']} — {r['reason']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)