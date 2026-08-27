"""
evaluation/llm_judge.py

Sonnet compares base vs. fine-tuned model outputs on each held-out
example, scoring both numerically on quality, coherence, and task fit.

This follows the same rubric-based philosophy as curation/curator.py,
so the fine-tuned model's evaluation can be measured as a real quality
margin rather than only counting wins.

Known limitation:
The judge can sometimes favor surface-level formatting matches,
such as exact phrasing or capitalization, over genuine answer
substance.

Raw outputs are therefore always preserved in judge_outputs() so a
human can inspect the actual responses alongside the judge's verdict.
"""
from __future__ import annotations

import json
import time

import anthropic

from evaluation.llm_trace import traced_claude_call


client = anthropic.Anthropic()


JUDGE_PROMPT = """You are comparing two AI model responses to the same
question, against a reference answer.

Question: {question}
Reference answer: {reference}

Response A (base model): {base_output}
Response B (fine-tuned model): {ft_output}

Score EACH response from 0-10 on:

- quality: overall correctness and usefulness
- coherence: clarity and logical structure
- task_fit: how well it actually answers what was asked

Then say which is better overall: "A", "B", or "tie".

Respond ONLY with JSON, no other text:
{{"score_a": {{"quality": <0-10>, "coherence": <0-10>, "task_fit": <0-10>}}, "score_b": {{"quality": <0-10>, "coherence": <0-10>, "task_fit": <0-10>}}, "winner": "<A, B, or tie>", "reason": "<1-2 sentences>"}}"""


def _judge_one(
    question: str,
    reference: str,
    base_output: str,
    ft_output: str,
    model: str,
) -> dict:
    """
    Judge one base-vs-fine-tuned output pair using Claude.

    Returns the parsed judge result.

    If the response cannot be parsed or contains an invalid winner,
    returns an explicit JUDGE_FAILED result instead of fabricating
    scores. This prevents a failed LLM call from silently affecting
    evaluation statistics.
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        base_output=base_output,
        ft_output=ft_output,
    )

    response = traced_claude_call(
        client,
        "evaluation.llm_judge",
        "_judge_one",
        model=model,
        max_tokens=300,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Handle responses wrapped in ```json ... ```
    if text.startswith("```"):
        text = text.split("```")[1]

        if text.startswith("json"):
            text = text[4:]

        text = text.strip()

    try:
        result = json.loads(text)

        if result.get("winner") not in {"A", "B", "tie"}:
            raise ValueError(
                f"winner {result.get('winner')!r} not in "
                "{'A', 'B', 'tie'}"
            )

        return result

    except (json.JSONDecodeError, ValueError, KeyError) as error:
        return {
            "score_a": None,
            "score_b": None,
            "winner": "JUDGE_FAILED",
            "reason": f"Judge response could not be parsed: {error}",
        }


def judge_outputs(
    base_outputs: list[dict],
    ft_outputs: list[dict],
    model: str = "claude-sonnet-4-5",
) -> list[dict]:
    """
    Compare base-model and fine-tuned-model outputs across the
    evaluation set.

    base_outputs:
        Outputs generated with use_adapter=False.

    ft_outputs:
        Outputs generated with use_adapter=True.

    Both lists should contain the same examples in the same order.

    Returns one dictionary per example containing:

        {
            "question": ...,
            "reference": ...,
            "base_output": ...,
            "ft_output": ...,
            "score_a": ...,
            "score_b": ...,
            "winner": "A" | "B" | "tie" | "JUDGE_FAILED",
            "reason": ...
        }

    Raw model outputs are preserved so the final report can show the
    actual responses alongside Sonnet's judgment.
    """
    if len(base_outputs) != len(ft_outputs):
        raise ValueError(
            "base_outputs and ft_outputs must have the same length"
        )

    results = []

    for base, ft in zip(base_outputs, ft_outputs):
        judged = _judge_one(
            question=base["question"],
            reference=base["reference"],
            base_output=base["output"],
            ft_output=ft["output"],
            model=model,
        )

        results.append(
            {
                "question": base["question"],
                "reference": base["reference"],
                "base_output": base["output"],
                "ft_output": ft["output"],
                **judged,
            }
        )

        # Small delay between judge requests.
        time.sleep(0.3)

    return results