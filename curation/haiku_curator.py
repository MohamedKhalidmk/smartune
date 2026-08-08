"""
curation/haiku_curator.py

YOUR MODULE — port the validated notebook logic here.

Source material: your Kaggle notebook cells for Step 6 (Haiku curation
with rubric anchors, temperature=0, code-fence stripping, retry-on-missing-
field logic) already work — this is a port, not a rewrite.

Contract expected by dashboard/app.py:

    scored = curate_dataset(raw_examples: list[dict]) -> list[dict]

    Each returned dict = original example + "_curation" key containing:
        {"clarity": float, "correctness": float, "value": float,
         "avg_score": float, "reason": str}

    def apply_threshold(scored: list[dict], threshold: float) -> tuple[list[dict], list[dict]]
        returns (kept, rejected)

Known-good pieces from the notebook to bring over as-is:
- CURATION_PROMPT with rubric anchors (clarity/correctness/value 0-10
  scales with explicit anchor descriptions)
- temperature=0 for determinism
- code-fence stripping before json.loads
- retry-once-with-temperature-0.3 on parse/missing-field failure
- explicit avg_score computed by us, not trusted from the model's own
  "keep" field (see ARCHITECTURE.md — "curation is a judgment call")

TODO(ML): also wire in the dataset diversity check (embedding similarity
clustering) mentioned in ARCHITECTURE.md's Future Additions, if you get
to it.
"""

import anthropic
import json
import time

client = anthropic.Anthropic()


def curate_dataset(raw_examples: list[dict]) -> list[dict]:
    """
    Port your validated notebook curation loop here.
    """
    raise NotImplementedError("Port from notebook Step 6")


def apply_threshold(scored: list[dict], threshold: float) -> tuple[list[dict], list[dict]]:
    """
    Split scored examples into (kept, rejected) based on avg_score.
    """
    kept = [e for e in scored if e["_curation"]["avg_score"] >= threshold]
    rejected = [e for e in scored if e["_curation"]["avg_score"] < threshold]
    return kept, rejected
