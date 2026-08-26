"""
curation/preprocess.py

Auto-detect an uploaded dataset's schema (Alpaca instruction/input/output,
prompt/completion, chat "messages" turns, or already question/answer) and
normalize every example to Smartune's expected {"question", "answer"} shape
before it reaches curate_dataset().

Cheap heuristics run first and cover the common formats without any API
call. Claude Haiku is only used as a fallback when the heuristics can't
confidently identify the question/answer fields — e.g. an unfamiliar or
inconsistently-keyed schema — so normal Alpaca/JSONL uploads cost nothing.
"""

import json

import anthropic

client = anthropic.Anthropic()

HAIKU_MODEL = "claude-haiku-4-5"

# Known field-name pairs, checked in order. First match wins.
_KNOWN_SCHEMAS = [
    # already-normalized
    (("question", "answer"), None),
    # Alpaca-style
    (("instruction", "output"), "alpaca"),
    # prompt/completion
    (("prompt", "completion"), "prompt_completion"),
    # generic instruction/response
    (("instruction", "response"), "instruction_response"),
]


def _detect_heuristic(example: dict) -> str | None:
    """Return a schema name if the example matches a known shape, else None."""
    keys = set(example.keys())

    if {"question", "answer"} <= keys:
        return "qa"

    if {"instruction", "output"} <= keys:
        return "alpaca"

    if {"prompt", "completion"} <= keys:
        return "prompt_completion"

    if {"instruction", "response"} <= keys:
        return "instruction_response"

    if "messages" in keys and isinstance(example.get("messages"), list):
        return "chat"

    return None


def _apply_schema(example: dict, schema: str) -> dict:
    if schema == "qa":
        return {"question": example["question"], "answer": example["answer"]}

    if schema == "alpaca":
        instruction = str(example.get("instruction", "")).strip()
        extra_input = str(example.get("input", "")).strip()
        question = f"{instruction}\n\n{extra_input}" if extra_input else instruction
        return {"question": question, "answer": str(example.get("output", "")).strip()}

    if schema == "prompt_completion":
        return {
            "question": str(example.get("prompt", "")).strip(),
            "answer": str(example.get("completion", "")).strip(),
        }

    if schema == "instruction_response":
        return {
            "question": str(example.get("instruction", "")).strip(),
            "answer": str(example.get("response", "")).strip(),
        }

    if schema == "chat":
        turns = example.get("messages", [])
        user_turns = [t.get("content", "") for t in turns if t.get("role") == "user"]
        assistant_turns = [
            t.get("content", "") for t in turns if t.get("role") == "assistant"
        ]
        return {
            "question": user_turns[-1] if user_turns else "",
            "answer": assistant_turns[-1] if assistant_turns else "",
        }

    raise ValueError(f"Unknown schema: {schema}")


def _ask_claude_for_mapping(sample_example: dict) -> dict:
    """
    Fallback for unrecognized schemas: ask Claude which field holds the
    question/instruction and which holds the answer/output.

    Returns {"question_field": ..., "answer_field": ...}.
    """
    prompt = (
        "You are given a single JSON training example from a dataset "
        "intended for LLM fine-tuning. Identify which top-level field "
        "contains the question/instruction/prompt, and which field "
        "contains the answer/response/output. If the question is split "
        "across multiple fields (e.g. an instruction plus separate "
        "input/context), list all of them in order.\n\n"
        f"Example:\n{json.dumps(sample_example, indent=2)[:3000]}\n\n"
        'Respond with ONLY JSON: {"question_fields": [...], "answer_field": "..."}'
    )

    response = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Be lenient about stray markdown fences.
    text = text.strip("`").removeprefix("json").strip()
    return json.loads(text)


def detect_schema(examples: list[dict]) -> dict:
    """
    Inspect a sample of the dataset and return a detection report:
    {"schema": "alpaca" | ... | "llm_mapping", "confidence": "heuristic" | "llm",
     "mapping": {...} or None, "sample_fields": [...]}
    """
    if not examples:
        return {"schema": None, "confidence": None, "mapping": None, "sample_fields": []}

    sample = examples[0]
    schema = _detect_heuristic(sample)

    if schema is not None:
        return {
            "schema": schema,
            "confidence": "heuristic",
            "mapping": None,
            "sample_fields": sorted(sample.keys()),
        }

    # Heuristics found nothing recognizable — fall back to Claude.
    mapping = _ask_claude_for_mapping(sample)
    return {
        "schema": "llm_mapping",
        "confidence": "llm",
        "mapping": mapping,
        "sample_fields": sorted(sample.keys()),
    }


def normalize_dataset(examples: list[dict], detection: dict) -> list[dict]:
    """Apply a detection report (from detect_schema) to every example."""
    schema = detection["schema"]

    if schema == "llm_mapping":
        mapping = detection["mapping"]
        q_fields = mapping.get("question_fields", [])
        a_field = mapping.get("answer_field")

        normalized = []
        for ex in examples:
            question = "\n\n".join(
                str(ex.get(f, "")).strip() for f in q_fields if ex.get(f)
            )
            normalized.append(
                {"question": question, "answer": str(ex.get(a_field, "")).strip()}
            )
        return normalized

    return [_apply_schema(ex, schema) for ex in examples]