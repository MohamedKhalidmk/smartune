"""
curation/curator.py

Claude Haiku curates and classifies a fine-tuning dataset: scores each
example on clarity/correctness/value, flags near-duplicates via FAISS
ANN search, and routes genuine curation failures to manual review
instead of guessing.

Manual overrides update each example's final_status directly, keyed by
a stable _id rather than question text, so the report always reflects
the current resolved state of the dataset.
"""

from __future__ import annotations


# ============================================================
# STANDARD LIBRARY
# ============================================================

import concurrent.futures
import csv
import io
import json
import os
import uuid
from datetime import datetime, timezone


# ============================================================
# THIRD-PARTY
# ============================================================

import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT IMPORTS
# ============================================================

from evaluation.llm_trace import traced_claude_call


# ============================================================
# CLIENT
# ============================================================

client = anthropic.Anthropic()


# ============================================================
# CURATION PROMPT
# ============================================================

CURATION_PROMPT = """You are reviewing a candidate training example for
fine-tuning a language model.

Question: {question}
Answer: {answer}

First, check: does the answer actually do what the question asks?
(e.g. if asked for similar-SOUNDING words, opposite-MEANING words would
be wrong even if grammatically fine). Note this explicitly in your
reason if there's a mismatch.

Score this example from 0-10 on each dimension, using these anchors:

CLARITY:
- 0-3: question or answer is ambiguous, contradictory, or confusing
- 4-6: understandable but imprecise or could be misread
- 7-10: fully clear and unambiguous

CORRECTNESS:
- 0-3: factually or logically wrong, OR fails to actually do what the
      task asked
- 4-6: mostly correct but with a notable gap, oversimplification, or
      minor error
- 7-10: fully accurate, sound, and correctly fulfills the task

VALUE:
- 0-3: trivial, boilerplate, or teaches nothing useful
- 4-6: reasonable but generic, common knowledge with little depth
- 7-10: substantive, specific, and genuinely useful for training

You MUST always include a "reason" field with a specific, non-empty
explanation — never omit it.

Also include your own holistic "keep" recommendation (true/false) —
this is used only as a tiebreaker for borderline scores, not as the
primary decision.

Respond ONLY with a JSON object like this, no other text:
{{"clarity": <0-10>, "correctness": <0-10>, "value": <0-10>,
"keep": <true or false>, "reason": "<specific sentence explaining
the scores, including any task-mismatch found>"}}
"""


# ============================================================
# SCORING
# ============================================================

def _score_example(example: dict, retries: int = 1) -> dict:
    """
    Score a single example via Haiku.

    Retries once with slightly higher temperature if the response is
    malformed or missing a required field.

    On failure after retries, returns an explicit failure marker rather
    than fabricating scores. The example can then be routed to manual
    review.
    """
    prompt = CURATION_PROMPT.format(
        question=example["question"],
        answer=example["answer"],
    )

    for attempt in range(retries + 1):
        response = traced_claude_call(
            client,
            "curation.curator",
            "_score_example",
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            temperature=0 if attempt == 0 else 0.3,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()

        # Handle Claude wrapping JSON in a markdown code block.
        if text.startswith("```"):
            text = text.split("```")[1]

            if text.startswith("json"):
                text = text[4:]

            text = text.strip()

        try:
            result = json.loads(text)

            required_fields = ["clarity", "correctness", "value"]

            if result.get("reason") and all(
                field in result for field in required_fields
            ):
                result["avg_score"] = (
                    result["clarity"]
                    + result["correctness"]
                    + result["value"]
                ) / 3

                result["curation_failed"] = False

                return result

        except json.JSONDecodeError:
            pass

    return {
        "clarity": None,
        "correctness": None,
        "value": None,
        "avg_score": None,
        "keep": None,
        "reason": (
            "Curation failed after retry — response was malformed or "
            "incomplete. Needs manual review."
        ),
        "curation_failed": True,
    }


# ============================================================
# DATASET CURATION
# ============================================================

def curate_dataset(
    raw_examples: list[dict],
    max_workers: int = 10,
) -> list[dict]:
    """
    Score every example in raw_examples via Haiku concurrently.

    Each returned example receives:
        - a stable "_id"
        - a "_curation" dictionary

    avg_score is calculated locally from clarity/correctness/value
    rather than trusting the model's keep recommendation.
    """
    scored = [None] * len(raw_examples)

    def _worker(idx_ex):
        idx, ex = idx_ex

        ex = dict(ex)
        ex["_id"] = str(uuid.uuid4())
        ex["_curation"] = _score_example(ex)

        return idx, ex

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        for idx, ex in executor.map(
            _worker,
            enumerate(raw_examples),
        ):
            scored[idx] = ex

    return scored


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def find_duplicate_candidates(
    scored: list[dict],
    similarity_threshold: float = 0.92,
    use_ann: bool = True,
    n_neighbors: int = 5,
) -> list[dict]:
    """
    Find near-duplicate pairs via embedding similarity.

    For datasets larger than 200 examples and use_ann=True, FAISS HNSW
    approximate nearest-neighbor search is used.

    For smaller datasets or when use_ann=False, exact pairwise cosine
    similarity is used.

    Returns candidate pairs only. Claude verification is handled
    separately by verify_duplicates_with_haiku().
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [
        f"{ex['question']} {ex['answer']}"
        for ex in scored
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=False,
    ).astype("float32")

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    normalized = embeddings / np.clip(
        norms,
        1e-8,
        None,
    )

    n = len(scored)
    candidates = []
    seen_pairs = set()

    if use_ann and n > 200:
        import faiss

        dim = normalized.shape[1]

        index = faiss.IndexHNSWFlat(
            dim,
            32,
            faiss.METRIC_INNER_PRODUCT,
        )

        index.hnsw.efConstruction = 40
        index.add(normalized)

        k = min(n_neighbors + 1, n)

        similarities, indices = index.search(
            normalized,
            k,
        )

        for i in range(n):
            for sim, j in zip(
                similarities[i],
                indices[i],
            ):
                if i == j or j == -1:
                    continue

                pair_key = tuple(
                    sorted((i, int(j)))
                )

                if (
                    sim >= similarity_threshold
                    and pair_key not in seen_pairs
                ):
                    seen_pairs.add(pair_key)

                    candidates.append({
                        "index_a": pair_key[0],
                        "index_b": pair_key[1],
                        "similarity": float(sim),
                        "question_a": scored[pair_key[0]]["question"],
                        "question_b": scored[pair_key[1]]["question"],
                    })

    else:
        similarity_matrix = normalized @ normalized.T

        for i in range(n):
            for j in range(i + 1, n):
                sim = float(similarity_matrix[i, j])

                if sim >= similarity_threshold:
                    candidates.append({
                        "index_a": i,
                        "index_b": j,
                        "similarity": sim,
                        "question_a": scored[i]["question"],
                        "question_b": scored[j]["question"],
                    })

    return candidates


# ============================================================
# DUPLICATE VERIFICATION
# ============================================================

def verify_duplicates_with_haiku(
    candidates: list[dict],
    max_workers: int = 10,
) -> list[dict]:
    """
    Send likely duplicate pairs to Haiku for confirmation, 10 at a time.
 
    Embedding similarity only identifies candidates. Claude makes the
    final duplicate judgment. Runs concurrently (like curate_dataset)
    instead of one call at a time, since this can otherwise be the
    slowest step for datasets with many candidate pairs.
    """
    prompt_template = """Are these two training examples genuinely
redundant (would keeping both add no real training value over keeping
just one), or are they different enough to both be worth keeping?
 
Example A:
Q: {q_a}
 
Example B:
Q: {q_b}
 
Respond ONLY with JSON:
{{"confirmed_duplicate": <true or false>, "reason": "<one sentence>"}}
"""
 
    def _verify_one(candidate: dict) -> dict:
        response = traced_claude_call(
            client,
            "curation.curator",
            "verify_duplicates_with_haiku",
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt_template.format(
                        q_a=candidate["question_a"],
                        q_b=candidate["question_b"],
                    ),
                }
            ],
        )
 
        text = response.content[0].text.strip()
 
        if text.startswith("```"):
            text = text.split("```")[1]
 
            if text.startswith("json"):
                text = text[4:]
 
            text = text.strip()
 
        try:
            result = json.loads(text)
 
        except json.JSONDecodeError:
            result = {
                "confirmed_duplicate": False,
                "reason": "verification_parse_failed",
            }
 
        return {
            **candidate,
            **result,
        }
 
    verified = [None] * len(candidates)
 
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        futures = {
            executor.submit(_verify_one, candidate): idx
            for idx, candidate in enumerate(candidates)
        }
 
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            verified[idx] = future.result()
 
    return verified


# ============================================================
# CLASSIFICATION
# ============================================================

def compute_auto_threshold(
    scored: list[dict],
    method: str = "otsu",
    percentile: float = 40.0,
) -> float:
    """
    Suggest a quality threshold from the score distribution instead of
    hardcoding one (e.g. the 6.7 used in the README's reference run).

    method="otsu": finds the score that best separates the distribution
    into a low-quality and high-quality cluster (Otsu's method, applied
    to the 1-D score histogram). Works well when scores are roughly
    bimodal, which is the common case for Haiku-scored examples (most
    are clearly good or clearly bad, with a thinner middle band).

    method="percentile": simple fallback — threshold = the given
    percentile of the score distribution (default: 40th percentile,
    i.e. keep the top 60%).

    curation_failed examples are excluded since they have no avg_score.
    Raises ValueError if there's nothing to compute from.
    """
    scores = np.array(
        [
            ex["_curation"]["avg_score"]
            for ex in scored
            if not ex["_curation"].get("curation_failed")
        ],
        dtype=float,
    )

    if len(scores) == 0:
        raise ValueError(
            "No valid (non-failed) scores to compute a threshold from."
        )

    if method == "percentile":
        return float(np.percentile(scores, percentile))

    if method == "otsu":
        return _otsu_threshold(scores)

    raise ValueError(
        f"Unknown method: {method!r}. Use 'otsu' or 'percentile'."
    )


def _otsu_threshold(scores: np.ndarray, bins: int = 50) -> float:
    """Otsu's method over a 0-10 score histogram: pick the split point
    that maximizes between-cluster variance (low-score vs high-score)."""
    hist, bin_edges = np.histogram(scores, bins=bins, range=(0.0, 10.0))
    hist = hist.astype(float)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    if total == 0:
        return float(np.median(scores))

    sum_total = np.dot(hist, bin_centers)

    weight_bg, sum_bg = 0.0, 0.0
    best_threshold = float(bin_centers[0])
    best_variance = 0.0

    for i in range(bins):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue

        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += bin_centers[i] * hist[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        between_variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between_variance > best_variance:
            best_variance = between_variance
            best_threshold = float(bin_centers[i])

    return best_threshold


def classify_dataset(
    scored: list[dict],
    threshold: float | None = None,
    mode: str = "normal",
    borderline_width: float = 1.0,
    duplicate_pairs: list[dict] | None = None,
    threshold_method: str = "otsu",
) -> dict:
    """
    Classify each example as kept, rejected, or failed.

    threshold: pass a number to use it as-is (manual override). Leave
    as None to auto-compute one from the score distribution via
    compute_auto_threshold(scored, method=threshold_method).

    Possible final statuses:
        kept
        rejected_quality
        rejected_duplicate
        failed
        manually_accepted
        manually_rejected
    """
    threshold_was_auto = threshold is None
    if threshold_was_auto:
        threshold = compute_auto_threshold(scored, method=threshold_method)

    duplicate_indices = set()

    if duplicate_pairs:
        for pair in duplicate_pairs:
            if pair.get("confirmed_duplicate", True):
                duplicate_indices.add(pair["index_b"])

    kept = []
    rejected = []
    failed = []

    for i, ex in enumerate(scored):
        curation = ex["_curation"]

        if curation.get("curation_failed"):
            curation["final_status"] = "failed"
            failed.append(ex)
            continue

        if i in duplicate_indices:
            curation["final_status"] = "rejected_duplicate"
            rejected.append(ex)
            continue

        score = curation["avg_score"]

        if mode == "normal":
            passes = score >= threshold

        elif mode == "hyper":
            lower = threshold - borderline_width / 2
            upper = threshold + borderline_width / 2

            if score >= upper:
                passes = True
            elif score < lower:
                passes = False
            else:
                passes = curation.get("keep", True)

        else:
            raise ValueError(
                f"Unknown mode: {mode!r}. "
                "Use 'normal' or 'hyper'."
            )

        if passes:
            curation["final_status"] = "kept"
            kept.append(ex)
        else:
            curation["final_status"] = "rejected_quality"
            rejected.append(ex)

    return {
        "kept": kept,
        "rejected": rejected,
        "failed": failed,
        "threshold_used": threshold,
        "threshold_auto": threshold_was_auto,
    }


# ============================================================
# MANUAL OVERRIDES
# ============================================================

def apply_manual_overrides(
    classification: dict,
    overrides: dict[str, str],
) -> dict:
    """
    Resolve examples with human decisions.

    overrides:
        {
            example_id: "accept" | "reject"
        }

    Uses the stable example _id rather than question text.
    """
    all_examples = (
        classification["kept"]
        + classification["rejected"]
        + classification["failed"]
    )

    kept = []
    rejected = []
    still_failed = []

    for ex in all_examples:
        decision = overrides.get(ex["_id"])
        previous_status = ex["_curation"]["final_status"]

        if decision == "accept":
            ex["_curation"]["automatic_status"] = previous_status
            ex["_curation"]["final_status"] = "manually_accepted"

            kept.append(ex)

            log_user_override(
                ex,
                "accept",
                previous_status=previous_status,
            )

        elif decision == "reject":
            ex["_curation"]["automatic_status"] = previous_status
            ex["_curation"]["final_status"] = "manually_rejected"

            rejected.append(ex)

            log_user_override(
                ex,
                "reject",
                previous_status=previous_status,
            )

        else:
            if previous_status == "kept":
                kept.append(ex)

            elif previous_status == "failed":
                still_failed.append(ex)

            else:
                rejected.append(ex)

    return {
        "kept": kept,
        "rejected": rejected,
        "failed": still_failed,
    }


# ============================================================
# REPORTING
# ============================================================

def generate_curation_report(
    classification: dict,
    threshold: float,
    mode: str = "normal",
    sample_size: int = 5,
) -> str:
    """
    Build a markdown report reflecting each example's current
    final_status.
    """
    all_examples = (
        classification["kept"]
        + classification["rejected"]
        + classification["failed"]
    )

    total = len(all_examples)

    by_status = {}

    for ex in all_examples:
        status = ex["_curation"]["final_status"]
        by_status.setdefault(status, []).append(ex)

    manually_accepted = by_status.get(
        "manually_accepted",
        [],
    )

    manually_rejected = by_status.get(
        "manually_rejected",
        [],
    )

    accepted_from_rejection = [
        ex
        for ex in manually_accepted
        if ex["_curation"].get("automatic_status")
        in ("rejected_quality", "rejected_duplicate")
    ]

    accepted_from_failed = [
        ex
        for ex in manually_accepted
        if ex["_curation"].get("automatic_status") == "failed"
    ]

    rejected_from_kept = [
        ex
        for ex in manually_rejected
        if ex["_curation"].get("automatic_status") == "kept"
    ]

    rejected_from_failed = [
        ex
        for ex in manually_rejected
        if ex["_curation"].get("automatic_status") == "failed"
    ]

    lines = [
        "# Curation Report",
        "",
        f"- Raw examples: {total}",
        f"- Threshold: {threshold}",
        f"- Mode: {mode}",
        f"- Kept: {len(by_status.get('kept', []))}",
        f"- Quality rejected: {len(by_status.get('rejected_quality', []))}",
        f"- Duplicates: {len(by_status.get('rejected_duplicate', []))}",
        f"- Manually accepted: {len(manually_accepted)}",
        f"- Manually rejected: {len(manually_rejected)}",
        f"- Failed (unresolved): {len(by_status.get('failed', []))}",
        "",
    ]

    sections = [
        (
            "rejected_quality",
            "Sample quality rejections",
            by_status.get("rejected_quality", []),
        ),
        (
            "rejected_duplicate",
            "Sample duplicate rejections",
            by_status.get("rejected_duplicate", []),
        ),
        (
            "accepted_from_rejection",
            "Manually accepted (overrode automatic rejection)",
            accepted_from_rejection,
        ),
        (
            "accepted_from_failed",
            "Manually accepted (previously unresolved)",
            accepted_from_failed,
        ),
        (
            "rejected_from_kept",
            "Manually rejected (overrode automatic decision)",
            rejected_from_kept,
        ),
        (
            "rejected_from_failed",
            "Manually rejected (previously unresolved)",
            rejected_from_failed,
        ),
        (
            "failed",
            "Unresolved curation failures — status: failed",
            by_status.get("failed", []),
        ),
    ]

    for _, title, examples in sections:
        if not examples:
            continue

        lines.append(f"## {title}")
        lines.append("")

        for ex in examples[:sample_size]:
            curation = ex["_curation"]

            lines.append(f"**Q:** {ex['question']}")
            lines.append("")

            lines.append(f"**A:** {ex['answer']}")
            lines.append("")

            lines.append(
                "**Automatic status:** "
                f"{curation.get('automatic_status', curation['final_status'])}"
            )

            lines.append(
                f"**Final status:** {curation['final_status']}"
            )

            lines.append("")

            reason = curation.get("reason", "")

            if reason:
                lines.append(f"**Reason:** {reason}")
                lines.append("")

            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def generate_decisions_csv(
    classification: dict,
) -> str:
    """
    Export every example with its final decision as CSV.
    """
    all_examples = (
        classification["kept"]
        + classification["rejected"]
        + classification["failed"]
    )

    output = io.StringIO()

    writer = csv.writer(output)

    writer.writerow([
        "id",
        "question",
        "answer",
        "clarity",
        "correctness",
        "value",
        "avg_score",
        "automatic_status",
        "final_status",
        "reason",
    ])

    for ex in all_examples:
        curation = ex["_curation"]

        writer.writerow([
            ex.get("_id", ""),
            ex["question"],
            ex["answer"],
            curation.get("clarity"),
            curation.get("correctness"),
            curation.get("value"),
            curation.get("avg_score"),
            curation.get(
                "automatic_status",
                curation.get("final_status"),
            ),
            curation.get("final_status"),
            curation.get("reason", ""),
        ])

    return output.getvalue()


# ============================================================
# USER OVERRIDE LOGGING
# ============================================================

def log_user_override(
    example: dict,
    user_decision: str,
    previous_status: str,
    user_reason: str | None = None,
    log_path: str = "results/user_override_log.jsonl",
) -> None:
    """
    Record cases where a human resolved or overrode an automatic
    curation decision.

    The log is append-only and stores the automatic status, human
    decision, scores, and timestamp.
    """
    record = {
        "id": example.get("_id"),
        "question": example["question"],
        "answer": example["answer"],
        "haiku_score": example.get("_curation", {}),
        "automatic_status": previous_status,
        "user_decision": user_decision,
        "user_reason": user_reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(
        os.path.dirname(log_path),
        exist_ok=True,
    )

    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")