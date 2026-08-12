"""
evaluation/eval_harness.py

Generates base vs. fine-tuned model outputs on a held-out eval set, so
evaluation/llm_judge.py has something real to compare.

This is about judging the FINE-TUNED MODEL's output quality. It is
different from training/forecasting.py, which forecasts the training
curve rather than the model's actual generation quality.
"""

import time

import torch

from curation.curator import curate_dataset, classify_dataset


def generate(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 300,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    """
    Generate a response for a single question.

    Two important implementation details:

    1. apply_chat_template(..., return_dict=True) is used because some
       transformers versions return a BatchEncoding rather than a plain
       tensor. The resulting inputs can be safely passed with
       model.generate(**inputs).

    2. Greedy decoding can sometimes hide changes caused by LoRA.
       do_sample and temperature are exposed so callers can optionally
       use sampling when checking whether fine-tuning changed behavior.
    """
    messages = [{"role": "user", "content": question}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        generate_kwargs["temperature"] = temperature

    with torch.no_grad():
        output = model.generate(
            **inputs,
            **generate_kwargs,
        )

    input_len = inputs["input_ids"].shape[1]

    return tokenizer.decode(
        output[0][input_len:],
        skip_special_tokens=True,
    )


def prepare_eval_dataset(
    uploaded_examples: list[dict] | None,
    validation_examples: list[dict],
    curation_threshold: float = 6.7,
    curation_mode: str = "normal",
) -> list[dict]:
    """
    Decide which examples should be used for evaluation.

    If the user does not provide an evaluation dataset, the existing
    validation set is used because it has already gone through the
    project's curation pipeline.

    If the user uploads their own evaluation examples, they are first
    passed through the same curation pipeline instead of being trusted
    as-is.

    Returns:
        A list of {"question", "answer"} examples.
    """
    if uploaded_examples is None:
        return validation_examples

    scored = curate_dataset(uploaded_examples)

    classification = classify_dataset(
        scored,
        threshold=curation_threshold,
        mode=curation_mode,
    )

    return [
        {
            "question": ex["question"],
            "answer": ex["answer"],
        }
        for ex in classification["kept"]
    ]


def generate_outputs(
    model,
    tokenizer,
    eval_examples: list[dict],
    use_adapter: bool,
    max_new_tokens: int = 300,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> list[dict]:
    """
    Generate outputs for every example in the evaluation set.

    use_adapter=True:
        Generate normally with the LoRA adapter enabled.

    use_adapter=False:
        Temporarily disable the adapter so the same model can be used
        to produce a genuine base-model baseline.

    This is necessary because get_peft_model() modifies the base model
    in place. Without disable_adapter(), a supposed "base model"
    evaluation would actually still use the fine-tuned adapter.

    Returns:
        A list containing:

        {
            "question": ...,
            "reference": ...,
            "output": ...,
            "latency_s": ...
        }
    """
    results = []

    def _run():
        for example in eval_examples:
            start = time.time()

            output = generate(
                model,
                tokenizer,
                example["question"],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )

            elapsed = time.time() - start

            results.append(
                {
                    "question": example["question"],
                    "reference": example["answer"],
                    "output": output,
                    "latency_s": elapsed,
                }
            )

    if use_adapter:
        _run()
    else:
        with model.disable_adapter():
            _run()

    return results