"""
training/finetune.py

LoRA / full fine-tuning, with an automatic QLoRA decision based on
estimated model size vs. available GPU memory.
"""

import random
import re
import time as _time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


# ============================================================
# Configuration
# ============================================================

BYTES_PER_PARAM = {
    "fp32": 4,
    "bf16": 2,
    "int4": 0.5,  # 4-bit quantization (QLoRA)
}

LORA_OVERHEAD_FACTOR = 1.3

# Optimizer states (Adam: 2x) + gradients (1x) + activations
FULL_FINETUNE_OVERHEAD_FACTOR = 4.0

FORECAST_EVERY_N_EPOCHS = 3

# The curve-fit forecaster can technically run on fewer points, but a
# forecast from 1-2 points is close to meaningless (not enough data
# to distinguish curve shapes).
MIN_POINTS_FOR_FORECAST = 3


# ============================================================
# Model-size estimation
# ============================================================

def _estimate_param_count_billions(model_name: str) -> float:
    """
    Estimate model size in billions of parameters.

    First tries to parse the parameter count directly from the model
    name, e.g. "Qwen2.5-1.5B-Instruct" -> 1.5 or "Llama-3-8B" -> 8.

    If the model name does not contain a parseable size, falls back to
    loading only the model configuration and estimating parameter count
    from its architecture dimensions.
    """

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*[bB](?:illion)?",
        model_name,
    )

    if match:
        return float(match.group(1))

    config = AutoConfig.from_pretrained(model_name)

    hidden_size = getattr(config, "hidden_size", None)
    num_layers = getattr(config, "num_hidden_layers", None)
    vocab_size = getattr(config, "vocab_size", None)

    if not all([hidden_size, num_layers, vocab_size]):
        raise ValueError(
            f"Could not parse a size from model name {model_name!r} "
            "and its config doesn't expose the fields needed to "
            "estimate one. Pass the size explicitly if you hit this."
        )

    approx_params = (
        12 * num_layers * hidden_size**2
        + 2 * vocab_size * hidden_size
    )

    return approx_params / 1e9


# ============================================================
# QLoRA decision
# ============================================================

def decide_qlora(
    model_name: str,
    num_gpus: int,
    gpu_memory_gb: float,
    safety_margin: float = 0.8,
) -> tuple[bool, str]:
    """
    Decide whether QLoRA is needed instead of standard LoRA.

    Returns:
        (should_use_qlora, reason)
    """

    params_billion = _estimate_param_count_billions(model_name)

    bf16_size_gb = (
        params_billion
        * BYTES_PER_PARAM["bf16"]
        * LORA_OVERHEAD_FACTOR
    )

    int4_size_gb = (
        params_billion
        * BYTES_PER_PARAM["int4"]
        * LORA_OVERHEAD_FACTOR
    )

    available_gb = (
        num_gpus
        * gpu_memory_gb
        * safety_margin
    )

    if bf16_size_gb <= available_gb:
        return False, (
            f"~{params_billion:.1f}B parameter model fits comfortably "
            f"in bf16 (~{bf16_size_gb:.1f}GB estimated) within your "
            f"{available_gb:.1f}GB available across {num_gpus} GPU(s). "
            "Using standard LoRA."
        )

    if int4_size_gb <= available_gb:
        return True, (
            f"~{params_billion:.1f}B parameter model does not fit in "
            f"bf16 (~{bf16_size_gb:.1f}GB estimated) within your "
            f"{available_gb:.1f}GB available, but does fit 4-bit "
            f"quantized (~{int4_size_gb:.1f}GB). Switching to QLoRA."
        )

    return True, (
        f"~{params_billion:.1f}B parameter model does not comfortably "
        f"fit even 4-bit quantized (~{int4_size_gb:.1f}GB estimated) "
        f"within your {available_gb:.1f}GB available. Recommending "
        "QLoRA anyway as the best available option, but this run may "
        "still hit an out-of-memory error — consider a smaller model "
        "or more/larger GPUs."
    )


# ============================================================
# Dataset preparation helpers
# ============================================================

def _format_and_tokenize(
    examples: list[dict],
    tokenizer,
):
    """
    Convert question/answer examples into chat-formatted tokenized text.
    """

    texts = []

    for ex in examples:
        messages = [
            {
                "role": "user",
                "content": ex["question"],
            },
            {
                "role": "assistant",
                "content": ex["answer"],
            },
        ]

        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
        )

    return tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )


def _to_hf_dataset(
    examples: list[dict],
    tokenizer,
):
    """
    Convert a list of question/answer dictionaries into a Hugging Face
    Dataset.
    """

    tokenized = _format_and_tokenize(examples, tokenizer)

    return Dataset.from_dict(
        {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
    )


# ============================================================
# Fine-tuning
# ============================================================

def run_finetune(
    model_name: str,
    method: str,
    train_dataset: list[dict],
    val_dataset: list[dict] | None,
    progress_callback,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-4,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
    lora_r: int = 8,
    lora_target_modules: list[str] | None = None,
    num_gpus: int = 1,
    gpu_memory_gb: float = 16.0,
    seed: int = 42,
) -> dict:
    """
    Fine-tune a causal language model using LoRA or full fine-tuning.

    method:
        "lora" -> automatically chooses standard LoRA or QLoRA.
        "full" -> trains every parameter without PEFT/quantization.

    progress_callback is called during training with available metrics.

    Returns:
        {
            "model": model,
            "training_time_s": float,
            "throughput_examples_per_sec": float,
            "peak_gpu_memory_gb": float,
            "final_loss": float,
            "train_loss_history": list,
            "val_loss_history": list,
            "used_qlora": bool,
            "seed": int,
        }
    """

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --------------------------------------------------------
    # Default LoRA target modules
    # --------------------------------------------------------

    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj"]

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --------------------------------------------------------
    # QLoRA decision
    # --------------------------------------------------------

    quantization_config = None

    if method == "lora":
        use_qlora, qlora_reason = decide_qlora(
            model_name,
            num_gpus,
            gpu_memory_gb,
        )

        progress_callback(
            {
                "step": 0,
                "loss": None,
                "val_loss": None,
                "qlora_decision": qlora_reason,
            }
        )

        if use_qlora:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

    # --------------------------------------------------------
    # Load base model
    # --------------------------------------------------------

    has_cuda = torch.cuda.is_available()

    model_dtype = (
        torch.bfloat16
        if has_cuda
        else torch.float32
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=model_dtype,
        device_map="auto" if has_cuda else None,
        quantization_config=quantization_config,
    )

    # --------------------------------------------------------
    # Configure training method
    # --------------------------------------------------------

    if method == "lora":
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # get_peft_model() modifies base_model in place.
        model = get_peft_model(
            base_model,
            lora_config,
        )

    elif method == "full":
        model = base_model

    else:
        raise ValueError(
            f"Unknown method: {method!r}. Use 'lora' or 'full'."
        )

    # --------------------------------------------------------
    # Dataset preparation
    # --------------------------------------------------------

    train_hf_dataset = _to_hf_dataset(
        train_dataset,
        tokenizer,
    )

    eval_hf_dataset = (
        _to_hf_dataset(val_dataset, tokenizer)
        if val_dataset
        else None
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    train_loss_history = []
    val_loss_history = []

    # --------------------------------------------------------
    # Training callback
    # --------------------------------------------------------

    class ProgressCallback(TrainerCallback):
        def on_log(
            self,
            args,
            state,
            control,
            logs=None,
            **kwargs,
        ):
            if logs and "loss" in logs:
                train_loss_history.append(logs["loss"])

                progress_callback(
                    {
                        "step": state.global_step,
                        "loss": logs["loss"],
                        "val_loss": None,
                    }
                )

        def on_evaluate(
            self,
            args,
            state,
            control,
            metrics=None,
            **kwargs,
        ):
            if not metrics or "eval_loss" not in metrics:
                return

            val_loss_history.append(
                metrics["eval_loss"]
            )

            update = {
                "step": state.global_step,
                "loss": None,
                "val_loss": metrics["eval_loss"],
            }

            epoch_number = len(val_loss_history)

            # ------------------------------------------------
            # Forecasting / decision engine
            # ------------------------------------------------

            if (
                epoch_number % FORECAST_EVERY_N_EPOCHS == 0
                and epoch_number >= MIN_POINTS_FOR_FORECAST
            ):
                try:
                    from training.forecasting import (
                        compute_difficulty_proxy,
                        forecast_n_epochs_ahead,
                        noise_floor,
                    )
                    from training.decision_engine import (
                        decide_training_action,
                    )

                    forecast_result = forecast_n_epochs_ahead(
                        val_loss_history,
                        save_plot_path=None,
                    )

                    difficulty = compute_difficulty_proxy(
                        val_loss_history
                    )

                    noise = noise_floor(
                        val_loss_history
                    )

                    decision = decide_training_action(
                        val_losses=val_loss_history,
                        forecast=forecast_result["forecast"],
                        difficulty=difficulty,
                        noise=noise,
                    )

                    update["forecast_check"] = {
                        "epoch": epoch_number,
                        "forecast": forecast_result["forecast"],
                        "difficulty": difficulty,
                        "noise": noise,
                        "decision": decision,
                    }

                except Exception as e:
                    update["forecast_check"] = {
                        "epoch": epoch_number,
                        "error": str(e),
                    }

            progress_callback(update)

    # --------------------------------------------------------
    # Training arguments
    # --------------------------------------------------------

    training_args = TrainingArguments(
        output_dir="./smartune-checkpoints",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=5,
        eval_strategy=(
            "epoch"
            if eval_hf_dataset is not None
            else "no"
        ),
        save_strategy="epoch",
        load_best_model_at_end=(
            eval_hf_dataset is not None
        ),
        metric_for_best_model=(
            "eval_loss"
            if eval_hf_dataset is not None
            else None
        ),
        greater_is_better=False,
        bf16=has_cuda,
        use_cpu=not has_cuda,
        report_to="none",
        seed=seed,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf_dataset,
        eval_dataset=eval_hf_dataset,
        data_collator=data_collator,
        callbacks=[ProgressCallback()],
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    start = _time.time()

    train_result = trainer.train()

    elapsed = _time.time() - start

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    throughput = (
        len(train_hf_dataset) * num_train_epochs
    ) / elapsed

    peak_mem_gb = (
        torch.cuda.max_memory_allocated() / 1e9
        if torch.cuda.is_available()
        else 0.0
    )

    return {
        "model": model,
        "training_time_s": elapsed,
        "throughput_examples_per_sec": throughput,
        "peak_gpu_memory_gb": peak_mem_gb,
        "final_loss": train_result.training_loss,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "used_qlora": quantization_config is not None,
        "seed": seed,
    }