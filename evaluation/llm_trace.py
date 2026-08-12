"""
common/llm_trace.py

A single shared wrapper every module that calls Claude routes through
(curation/curator.py, training/decision_engine.py,
training/check_dataset.py, evaluation/llm_judge.py), so there's one
unified trace of every LLM call made anywhere in the pipeline —
module, function, prompt, raw response, success/failure, latency,
timestamp — instead of each module logging (or not logging) its own
calls independently. This is what makes it possible to look at a full
run afterward and see exactly where in the flow something went wrong,
not just the final downstream symptom of it.

Logs to results/llm_call_trace.jsonl, append-only, same pattern as
every other log in this project (curation's log_user_override,
training/run_log's log_finetune_outcome).
"""

import json
import os
import time
from datetime import datetime, timezone

TRACE_LOG_PATH = "results/llm_call_trace.jsonl"


def traced_claude_call(client, module: str, function: str, **create_kwargs):
    """
    Thin wrapper around client.messages.create(**create_kwargs) — same
    signature, same return value, so existing call sites only need to
    swap `client.messages.create(...)` for
    `traced_claude_call(client, "module_name", "function_name", ...)`,
    nothing else about their existing parsing logic changes.

    Logs the call regardless of success or failure (via try/finally),
    and re-raises the original exception unchanged on failure — this
    preserves existing behavior at call sites exactly, it just adds a
    trace record as a side effect.

    module/function: caller-provided labels (e.g. "curation.curator",
    "_score_example") so the trace log can be filtered/read per
    pipeline stage afterward.
    """
    start = time.time()
    prompt_content = create_kwargs.get("messages", [{}])[-1].get("content", "")

    record = {
        "module": module,
        "function": function,
        "model": create_kwargs.get("model"),
        "prompt": prompt_content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        response = client.messages.create(**create_kwargs)
        record["response_text"] = response.content[0].text
        record["success"] = True
        record["error"] = None
        return response
    except Exception as e:
        record["response_text"] = None
        record["success"] = False
        record["error"] = str(e)
        raise
    finally:
        record["latency_s"] = time.time() - start
        os.makedirs(os.path.dirname(TRACE_LOG_PATH), exist_ok=True)
        with open(TRACE_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")


def load_llm_trace(log_path: str = TRACE_LOG_PATH) -> list[dict]:
    """
    Read back the full trace so far — for reviewing an entire run's
    LLM call flow after the fact, or filtering to just the calls from
    one module/function to isolate where something went wrong.
    """
    if not os.path.exists(log_path):
        return []
    with open(log_path) as f:
        return [json.loads(line) for line in f if line.strip()]