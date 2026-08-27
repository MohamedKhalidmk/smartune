"""
training/report.py

Generates a human-readable report of a completed (or in-progress)
fine-tuning run.

The report includes:
- training configuration and final metrics
- validation loss history
- every forecast_check event collected during training

finetune.py does not store forecast_check events itself. They flow
through progress_callback as training runs. The caller is responsible
for collecting them into a list.

This module only formats the collected data into a Markdown report.
"""
from __future__ import annotations


def generate_training_report(
    training_result: dict,
    forecast_checks: list[dict],
) -> str:
    """
    Generate a Markdown-formatted report for a fine-tuning run.

    Args:
        training_result:
            The dictionary returned by finetune.run_finetune().

        forecast_checks:
            The forecast_check dictionaries collected from
            progress_callback during training.

    Returns:
        A Markdown-formatted training report.
    """
    lines = [
        "# Fine-Tuning Run Report",
        "",
    ]

    # ------------------------------------------------------------
    # Training summary
    # ------------------------------------------------------------
    lines.extend([
        "## Training Summary",
        "",
        f"- Final loss: {training_result.get('final_loss'):.4f}",
        (
            f"- Training time: "
            f"{training_result.get('training_time_s'):.1f}s"
        ),
        (
            f"- Throughput: "
            f"{training_result.get('throughput_examples_per_sec'):.2f} "
            f"examples/sec"
        ),
        (
            f"- Peak GPU memory: "
            f"{training_result.get('peak_gpu_memory_gb'):.2f} GB"
        ),
        f"- Used QLoRA: {training_result.get('used_qlora')}",
        "",
    ])

    # ------------------------------------------------------------
    # Validation loss history
    # ------------------------------------------------------------
    val_history = training_result.get(
        "val_loss_history",
        [],
    )

    lines.extend([
        "## Validation Loss by Epoch",
        "",
    ])

    if val_history:
        for epoch, loss in enumerate(
            val_history,
            start=1,
        ):
            lines.append(
                f"- Epoch {epoch}: {loss:.4f}"
            )
    else:
        lines.append(
            "_No validation history recorded "
            "(no val_dataset provided)._"
        )

    lines.append("")

    # ------------------------------------------------------------
    # Forecast checks
    # ------------------------------------------------------------
    lines.extend([
        "## Forecast Checks During Training",
        "",
    ])

    if not forecast_checks:
        lines.append(
            "_No forecast checks were recorded. Either the run was "
            "too short (fewer than 3 epochs) to trigger one, or the "
            "caller did not collect them from progress_callback's "
            "`forecast_check` updates as they occurred._"
        )
    else:
        for check in forecast_checks:
            epoch = check.get("epoch")

            lines.extend([
                f"### Epoch {epoch}",
                "",
            ])

            if "error" in check:
                lines.append(
                    f"- Forecast check failed: {check['error']}"
                )
                lines.append("")
                continue

            decision = check.get(
                "decision",
                {},
            )

            difficulty = check.get(
                "difficulty",
                {},
            )

            forecast = check.get(
                "forecast",
                {},
            )

            lines.append(
                f"- **Decision:** {decision.get('action')} "
                f"(notify user: {decision.get('notify_user')})"
            )

            lines.append(
                f"- **Reason:** {decision.get('reason')}"
            )

            lines.append(
                "- Forecast (median, next epochs): "
                f"{forecast.get('median')}"
            )

            lines.append(
                "- Uncertainty band: "
                f"[{forecast.get('lower_5')}, "
                f"{forecast.get('upper_95')}]"
            )

            lines.append(
                "- Difficulty signals: "
                f"progress={difficulty.get('prog'):.4f}, "
                f"non-linearity={difficulty.get('nonlin'):.4f}, "
                f"volatility={difficulty.get('vol'):.4f}"
            )

            lines.append(
                f"- Noise floor: {check.get('noise'):.4f}"
            )

            lines.append("")

    # ------------------------------------------------------------
    # Overall verdict
    # ------------------------------------------------------------
    lines.extend([
        "## Overall Verdict",
        "",
    ])

    notified_checks = [
        check
        for check in forecast_checks
        if check.get("decision", {}).get("notify_user")
    ]

    if notified_checks:
        last = notified_checks[-1]
        decision = last["decision"]

        lines.append(
            f"Most recent actionable recommendation "
            f"(epoch {last['epoch']}): "
            f"**{decision['action']}** — "
            f"{decision['reason']}"
        )
    else:
        lines.append(
            "No actionable recommendations were raised during this "
            "run (all forecast checks, if any, resulted in a routine "
            "CONTINUE or were not flagged for the user's attention)."
        )

    return "\n".join(lines)