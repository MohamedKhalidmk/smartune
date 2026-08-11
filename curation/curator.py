"""
curation/haiku_curator.py

Claude Haiku curates and classifies a fine-tuning dataset: scores each
example on clarity/correctness/value, flags near-duplicates via FAISS
ANN search, and routes genuine curation failures to manual review
instead of guessing. Manual overrides update each example's
final_status directly (keyed by a stable _id, not question text), so
the report always reflects the current resolved state of the dataset.
"""

import anthropic
import json

client = anthropic.Anthropic()

CURATION_PROMPT = '''You are reviewing a candidate training example for fine-tuning a language model.

Question: {question}
Answer: {answer}

First, check: does the answer actually do what the question asks? (e.g. if asked for similar-SOUNDING words, opposite-MEANING words would be wrong even if grammatically fine). Note this explicitly in your reason if there's a mismatch.

Score this example from 0-10 on each dimension, using these anchors:

CLARITY:
- 0-3: question or answer is ambiguous, contradictory, or confusing
- 4-6: understandable but imprecise or could be misread
- 7-10: fully clear and unambiguous

CORRECTNESS:
- 0-3: factually or logically wrong, OR fails to actually do what the task asked
- 4-6: mostly correct but with a notable gap, oversimplification, or minor error
- 7-10: fully accurate, sound, and correctly fulfills the task

VALUE:
- 0-3: trivial, boilerplate, or teaches nothing useful
- 4-6: reasonable but generic, common knowledge with little depth
- 7-10: substantive, specific, and genuinely useful for training

You MUST always include a "reason" field with a specific, non-empty explanation — never omit it.
Also include your own holistic "keep" recommendation (true/false) — this is used only as a tiebreaker for borderline scores, not as the primary decision.

Respond ONLY with a JSON object like this, no other text:
{"clarity": <0-10>, "correctness": <0-10>, "value": <0-10>, "keep": <true or false>, "reason": "<specific sentence explaining the scores, including any task-mismatch found>"}'''

def curate_dataset(raw_examples: list[dict], max_workers: int = 10) -> list[dict]:
    """
    Score every example in raw_examples via Haiku, concurrently.
    Returns each example with an added "_curation" key and a stable
    "_id" (UUID) assigned here, at the point examples enter the
    pipeline.

    The ID exists so downstream steps (apply_manual_overrides, the
    dashboard's per-row accept/reject controls, log_user_override) never
    have to key off question text, which can collide (duplicate or
    near-duplicate questions are exactly what find_duplicate_candidates
    is looking for) or be brittle to whitespace/encoding differences.
    """
    import concurrent.futures
    import uuid

    scored = [None] * len(raw_examples)

    def _worker(idx_ex):
        idx, ex = idx_ex
        ex = dict(ex)  # don't mutate caller's dict
        ex["_id"] = str(uuid.uuid4())
        ex["_curation"] = _score_example(ex)
        return idx, ex

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, ex in executor.map(_worker, enumerate(raw_examples)):
            scored[idx] = ex

    return scored


def _score_example(example: dict, retries: int = 1) -> dict:
    """
    Score a single example via Haiku. Retries once (with slightly
    higher temperature) if the response is malformed or missing a
    required field.

    IMPORTANT: on failure after retries, this does NOT fabricate a
    plausible-looking score (e.g. 5/5/5) and let it silently flow into
    keep/reject decisions. It returns an explicit failure marker
    (curation_failed=True, all scores None) so the caller routes it to
    manual review instead of risking an automatic decision on a
    genuinely unscored example.
    """
    prompt = CURATION_PROMPT.format(question=example["question"], answer=example["answer"])

    for attempt in range(retries + 1):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            temperature=0 if attempt == 0 else 0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()

        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            result = json.loads(text)
            if result.get("reason") and all(k in result for k in ["clarity", "correctness", "value"]):
                result["avg_score"] = (result["clarity"] + result["correctness"] + result["value"]) / 3
                result["curation_failed"] = False
                return result
        except json.JSONDecodeError:
            pass

    return {
        "clarity": None, "correctness": None, "value": None, "avg_score": None,
        "keep": None,
        "reason": "Curation failed after retry — response was malformed or incomplete. Needs manual review.",
        "curation_failed": True,
    }

    
def find_duplicate_candidates(
    scored: list[dict],
    similarity_threshold: float = 0.92,
    use_ann: bool = True,
    n_neighbors: int = 5,
) -> list[dict]:
    """
    Finds near-duplicate pairs via embedding similarity.

    use_ann=True (default): uses FAISS's HNSW index — a genuine
    approximate nearest-neighbor structure (graph-based, sub-linear
    query time), not an exact brute-force search wearing an "ANN" label.

    use_ann=False: exact brute-force full pairwise comparison. Fine for
    small datasets, guaranteed exact — useful for sanity-checking ANN.

    Uses sentence-transformers (all-MiniLM-L6-v2) — local, free, no API
    cost, since this runs on every example and an API call per
    comparison would be far too slow/expensive at scale.

    Returns candidate pairs only — does NOT call Claude. Pass the result
    to verify_duplicates_with_haiku() for LLM confirmation before
    actually rejecting anything as a duplicate.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [f"{ex['question']} {ex['answer']}" for ex in scored]
    embeddings = model.encode(texts, show_progress_bar=False).astype("float32")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.clip(norms, 1e-8, None)

    n = len(scored)
    candidates = []
    seen_pairs = set()

    if use_ann and n > 200:
        import faiss

        dim = normalized.shape[1]
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 40
        index.add(normalized)

        k = min(n_neighbors + 1, n)
        similarities, indices = index.search(normalized, k)

        for i in range(n):
            for sim, j in zip(similarities[i], indices[i]):
                if i == j or j == -1:
                    continue
                pair_key = tuple(sorted((i, int(j))))
                if sim >= similarity_threshold and pair_key not in seen_pairs:
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


def verify_duplicates_with_haiku(candidates: list[dict]) -> list[dict]:
    """
    Optional second pass: embedding similarity finds likely duplicate
    CANDIDATES, but similarity alone can be wrong. Send only these
    flagged candidate pairs to Haiku for a real judgment.

    Adds "confirmed_duplicate": bool and "reason": str to each candidate.
    """
    prompt_template = '''Are these two training examples genuinely
redundant (would keeping both add no real training value over keeping
just one), or are they different enough to both be worth keeping?

Example A:
Q: {q_a}

Example B:
Q: {q_b}

Respond ONLY with JSON: {{"confirmed_duplicate": <true or false>, "reason": "<one sentence>"}}'''

    verified = []
    for c in candidates:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            temperature=0,
            messages=[{"role": "user", "content": prompt_template.format(
                q_a=c["question_a"], q_b=c["question_b"])}]
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
            result = {"confirmed_duplicate": False, "reason": "verification_parse_failed"}
        verified.append({**c, **result})

    return verified


def classify_dataset(
    scored: list[dict],
    threshold: float,
    mode: str = "normal",
    borderline_width: float = 1.0,
    duplicate_pairs: list[dict] | None = None,
) -> dict:
    """
    Single classification pass combining quality thresholding, duplicate
    rejection, and curation-failure handling into one decision per
    example. Every example gets curation["final_status"] set to one of:

        "kept"                — passed quality threshold
        "rejected_quality"     — scored below threshold
        "rejected_duplicate"   — matched a confirmed duplicate pair
        "failed"               — Haiku scoring failed after retry;
                                  NOT auto-kept or auto-rejected

    "failed" is not a permanent bucket — call apply_manual_overrides()
    afterward to resolve it into "manually_accepted" or
    "manually_rejected" once a human reviews it.

    mode="normal": plain cutoff on avg_score.
    mode="hyper": hybrid — clearly above/below threshold decides
    directly; only the borderline band defers to Haiku's own "keep"
    opinion from the same scoring call.

    Returns {"kept": [...], "rejected": [...], "failed": [...]}.
    """
    duplicate_indices = set()
    if duplicate_pairs:
        for pair in duplicate_pairs:
            if pair.get("confirmed_duplicate", True):
                duplicate_indices.add(pair["index_b"])

    kept, rejected, failed = [], [], []

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
            raise ValueError(f"Unknown mode: {mode!r}. Use 'normal' or 'hyper'.")

        if passes:
            curation["final_status"] = "kept"
            kept.append(ex)
        else:
            curation["final_status"] = "rejected_quality"
            rejected.append(ex)

    return {"kept": kept, "rejected": rejected, "failed": failed}


def apply_manual_overrides(classification: dict, overrides: dict[str, str]) -> dict:
    """
    Resolve examples with a human decision, moving them between buckets
    and updating final_status accordingly. Works on ANY example, not
    just failed ones.

    overrides: {example_id: "accept" | "reject"}, keyed by each
    example's stable "_id" (assigned in curate_dataset), NOT question
    text — question text can collide (duplicates are exactly what
    find_duplicate_candidates looks for) or differ by trivial
    whitespace, so it's not a safe key for this.

    Captures each example's final_status BEFORE mutating it, stores
    that as "automatic_status" on the curation dict (so the report can
    later distinguish "overrode a real rejection" from "resolved a
    previously-unscored failure"), then sets final_status to
    "manually_accepted" or "manually_rejected" and logs the override.

    Returns a new classification dict in the same {"kept", "rejected",
    "failed"} shape.
    """
    all_examples = classification["kept"] + classification["rejected"] + classification["failed"]

    kept, rejected, still_failed = [], [], []

    for ex in all_examples:
        decision = overrides.get(ex["_id"])
        previous_status = ex["_curation"]["final_status"]

        if decision == "accept":
            ex["_curation"]["automatic_status"] = previous_status
            ex["_curation"]["final_status"] = "manually_accepted"
            kept.append(ex)
            log_user_override(ex, "accept", previous_status=previous_status)
        elif decision == "reject":
            ex["_curation"]["automatic_status"] = previous_status
            ex["_curation"]["final_status"] = "manually_rejected"
            rejected.append(ex)
            log_user_override(ex, "reject", previous_status=previous_status)
        else:
            # no override — keep wherever it already was
            if previous_status == "kept":
                kept.append(ex)
            elif previous_status == "failed":
                still_failed.append(ex)
            else:
                rejected.append(ex)

    return {"kept": kept, "rejected": rejected, "failed": still_failed}


def generate_curation_report(classification: dict, threshold: float, mode: str = "normal", sample_size: int = 5) -> str:
    """
    Build a markdown report reflecting each example's current
    final_status: "kept", "rejected_quality", "rejected_duplicate",
    "failed", "manually_accepted", "manually_rejected".

    Call this AFTER apply_manual_overrides() if you want the report to
    show resolved outcomes for failed/borderline examples instead of an
    unresolved "failed" bucket — the report just reflects whatever
    final_status is currently set, so the same classification produces
    a different report before vs. after manual review, with no separate
    gate in between.

    Each example's automatic_status (what Haiku originally decided) and
    final_status (what it currently is, after any manual override) are
    both shown explicitly, so a reviewer never has to infer whether or
    how a decision changed.
    """
    all_examples = classification["kept"] + classification["rejected"] + classification["failed"]
    total = len(all_examples)

    by_status = {}
    for ex in all_examples:
        status = ex["_curation"]["final_status"]
        by_status.setdefault(status, []).append(ex)

    # Split manual overrides by what they were BEFORE the override,
    # since "overrode automatic decision" only makes sense if there was
    # an actual automatic decision (a rejection) to override — resolving
    # a previously-unscored "failed" example isn't overriding anything.
    manually_accepted = by_status.get("manually_accepted", [])
    manually_rejected = by_status.get("manually_rejected", [])

    accepted_from_rejection = [e for e in manually_accepted if e["_curation"].get("automatic_status") in ("rejected_quality", "rejected_duplicate")]
    accepted_from_failed = [e for e in manually_accepted if e["_curation"].get("automatic_status") == "failed"]
    rejected_from_kept = [e for e in manually_rejected if e["_curation"].get("automatic_status") == "kept"]
    rejected_from_failed = [e for e in manually_rejected if e["_curation"].get("automatic_status") == "failed"]

    lines = ["# Curation Report", ""]
    lines.append(f"- Raw examples: {total}")
    lines.append(f"- Threshold: {threshold}")
    lines.append(f"- Mode: {mode}")
    lines.append(f"- Kept: {len(by_status.get('kept', []))}")
    lines.append(f"- Quality rejected: {len(by_status.get('rejected_quality', []))}")
    lines.append(f"- Duplicates: {len(by_status.get('rejected_duplicate', []))}")
    lines.append(f"- Manually accepted: {len(manually_accepted)}")
    lines.append(f"- Manually rejected: {len(manually_rejected)}")
    lines.append(f"- Failed (unresolved): {len(by_status.get('failed', []))}")
    lines.append("")

    sections = [
        ("rejected_quality", "Sample quality rejections", by_status.get("rejected_quality", [])),
        ("rejected_duplicate", "Sample duplicate rejections", by_status.get("rejected_duplicate", [])),
        ("accepted_from_rejection", "Manually accepted (overrode automatic rejection)", accepted_from_rejection),
        ("accepted_from_failed", "Manually accepted (previously unresolved)", accepted_from_failed),
        ("rejected_from_kept", "Manually rejected (overrode automatic decision)", rejected_from_kept),
        ("rejected_from_failed", "Manually rejected (previously unresolved)", rejected_from_failed),
        ("failed", "Unresolved curation failures — status: failed", by_status.get("failed", [])),
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
            lines.append(f"**Automatic status:** {curation.get('automatic_status', curation['final_status'])}")
            lines.append(f"**Final status:** {curation['final_status']}")
            lines.append("")
            reason = curation.get("reason", "")
            if reason:
                lines.append(f"**Reason:** {reason}")
                lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def generate_decisions_csv(classification: dict) -> str:
    """
    Export every example with its final decision as CSV — id, question,
    answer, scores, automatic_status, final_status, and reason for the
    whole dataset. automatic_status is included alongside final_status
    so a reviewer auditing the CSV in a spreadsheet can see what Haiku
    originally decided, not just where the example ended up — the same
    distinction the markdown report makes.
    """
    import csv
    import io

    all_examples = classification["kept"] + classification["rejected"] + classification["failed"]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "question", "answer", "clarity", "correctness", "value",
        "avg_score", "automatic_status", "final_status", "reason",
    ])

    for ex in all_examples:
        c = ex["_curation"]
        writer.writerow([
            ex.get("_id", ""),
            ex["question"],
            ex["answer"],
            c.get("clarity"),
            c.get("correctness"),
            c.get("value"),
            c.get("avg_score"),
            c.get("automatic_status", c.get("final_status")),
            c.get("final_status"),
            c.get("reason", ""),
        ])

    return output.getvalue()


def log_user_override(
    example: dict,
    user_decision: str,
    previous_status: str,
    user_reason: str | None = None,
    log_path: str = "results/user_override_log.jsonl",
) -> None:
    """
    Record cases where a human resolved or overrode an automatic
    curation decision. Append-only log, one JSON object per line.

    previous_status is the example's final_status BEFORE this override
    (e.g. "rejected_quality", "failed", "kept") — the caller must
    capture this before mutating final_status, since by the time this
    is called the example's own final_status has already been
    overwritten and no longer reflects what Haiku originally decided.
    Without this, the log can't distinguish "human overrode a
    rejection" from "human resolved an unscored failure" — exactly the
    distinction needed to later analyze where Haiku is actually making
    mistakes vs. where it just couldn't score something.

    This is training-data-shaped on purpose: {id, question, answer,
    haiku_score, automatic_status, user_decision, user_reason,
    timestamp}.

    user_decision must be "accept" or "reject" — the human's FINAL
    decision, whether or not it agrees with Haiku's verdict.
    """
    import os
    from datetime import datetime, timezone

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

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")