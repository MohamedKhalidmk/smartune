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

from __future__ import annotations


import json
import zipfile
from pathlib import PurePosixPath

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


_DATA_EXTENSIONS = (".jsonl", ".json")
_TABULAR_EXTENSIONS = (".csv", ".tsv")
_SKIP_DIR_MARKERS = ("__MACOSX", ".git", ".DS_Store")


def _parse_examples_text(text: str) -> list[dict]:
    """Parse a .json or .jsonl blob into a list of example dicts."""
    text = text.strip()
    if not text:
        return []

    # Try whole-file JSON first (Alpaca ships as a single JSON array).
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [ex for ex in parsed if isinstance(ex, dict)]
        if isinstance(parsed, dict):
            # Some exports wrap the list under a top-level key.
            for value in parsed.values():
                if isinstance(value, list):
                    return [ex for ex in value if isinstance(ex, dict)]
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL — one JSON object per line.
    examples = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            examples.append(obj)
    return examples


def _parse_csv_text(text: str, delimiter: str = ",") -> list[dict]:
    import csv
    import io

    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    return [dict(row) for row in reader]


def load_examples_from_zip(zip_bytes, _depth: int = 0) -> list[dict]:
    """
    Extract every .json/.jsonl (and, as a fallback, .csv/.tsv) file inside
    an uploaded zip archive and concatenate their parsed examples into one
    list. Recurses one level into any zip-within-a-zip.

    zip_bytes: a file-like object (e.g. Streamlit's UploadedFile) opened
    in binary mode, or anything zipfile.ZipFile accepts directly.

    Raises ValueError (listing what the archive actually contains) if no
    readable dataset file is found.
    """
    examples: list[dict] = []
    found_any_data_file = False
    all_names = []

    with zipfile.ZipFile(zip_bytes) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            name = PurePosixPath(info.filename)
            all_names.append(str(name))

            if any(marker in name.parts for marker in _SKIP_DIR_MARKERS):
                continue

            suffix = name.suffix.lower()

            if suffix in _DATA_EXTENSIONS:
                found_any_data_file = True
                with zf.open(info) as f:
                    text = f.read().decode("utf-8", errors="replace")
                examples.extend(_parse_examples_text(text))

            elif suffix == ".zip" and _depth < 1:
                # One level of nested-zip support (e.g. a GitHub repo
                # zip that itself contains a data.zip).
                with zf.open(info) as f:
                    import io

                    nested_bytes = io.BytesIO(f.read())
                try:
                    nested = load_examples_from_zip(nested_bytes, _depth=_depth + 1)
                    examples.extend(nested)
                    found_any_data_file = True
                except ValueError:
                    pass

            elif suffix in _TABULAR_EXTENSIONS:
                found_any_data_file = True
                with zf.open(info) as f:
                    text = f.read().decode("utf-8", errors="replace")
                examples.extend(
                    _parse_csv_text(text, delimiter="\t" if suffix == ".tsv" else ",")
                )

    if not found_any_data_file:
        preview = ", ".join(all_names[:20])
        if len(all_names) > 20:
            preview += f", ... ({len(all_names)} files total)"
        raise ValueError(
            "No .json/.jsonl (or .csv/.tsv) files found inside the zip "
            f"archive. Contents were: {preview or '(empty archive)'}"
        )

    return examples


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