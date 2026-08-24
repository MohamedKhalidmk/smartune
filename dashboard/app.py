"""
dashboard/app.py

Streamlit shell for Smartune.

This file owns navigation, layout, and session state. It does NOT
contain curation, training, forecasting, or evaluation logic itself.
Every step calls into the corresponding module.

Run with:
    streamlit run dashboard/app.py
"""

import json

import streamlit as st

# ============================================================
# Backend imports
# ============================================================

from curation.curator import (
    apply_manual_overrides,
    classify_dataset,
    curate_dataset,
    generate_curation_report,
    generate_decisions_csv,
)
from training.finetune import decide_qlora, run_finetune
from training.forecasting import (
    compute_difficulty_proxy,
    forecast_n_epochs_ahead,
    noise_floor,
)
from training.decision_engine import decide_training_action
from training.run_log import (
    assess_dataset_before_finetuning,
    decide_dataset_warning,
    log_finetune_outcome,
)
from evaluation.eval_harness import generate_outputs
from evaluation.llm_judge import judge_outputs

# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="Smartune",
    layout="wide",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"], .stApp, .stMarkdown, .stButton button,
    .stTextInput input, .stTextArea textarea, .stSelectbox, .stRadio,
    .stTabs, table, th, td, code, pre {
        font-family: "Times New Roman", Times, serif !important;
    }
    .stApp {
        background-color: #F5F5F5;
        color: #303841;
    }
    [data-testid="stSidebar"], [data-testid="stExpander"] {
        background-color: #76ABAE;
    }
    .stButton > button, .stDownloadButton > button {
        background-color: #FF5722;
        color: #F5F5F5;
        border: none;
        font-family: "Times New Roman", Times, serif !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #e64a19;
        color: #F5F5F5;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #303841;
        font-family: "Times New Roman", Times, serif !important;
    }
    a { color: #FF5722; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Session State
# ============================================================

if "step" not in st.session_state:
    st.session_state.step = "welcome"

if "raw_dataset" not in st.session_state:
    st.session_state.raw_dataset = None

if "scored_dataset" not in st.session_state:
    st.session_state.scored_dataset = None

if "curated_dataset" not in st.session_state:
    st.session_state.curated_dataset = None

if "run_config" not in st.session_state:
    st.session_state.run_config = {}


# ============================================================
# Navigation
# ============================================================

def go_to(step_name: str) -> None:
    """Navigate to a different dashboard step."""
    st.session_state.step = step_name


# ============================================================
# STEP 1 — Welcome
# ============================================================

def render_welcome() -> None:
    st.title("Smartune")

    st.markdown(
        "Agentic fine-tuning pipeline: Claude Haiku curates and routes, "
        "LoRA/QLoRA fine-tunes, Claude judges the result."
    )

    if st.button("Get started"):
        go_to("upload")


# ============================================================
# STEP 2/3 — Upload + Preview Dataset
# ============================================================

def render_upload() -> None:
    st.header("1. Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload training dataset (JSONL)",
        type=["jsonl"],
    )

    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")

        examples = [
            json.loads(line)
            for line in content.strip().split("\n")
        ]

        st.session_state.raw_dataset = examples

        st.success(f"Loaded {len(examples)} examples")

        go_to("preview")


def render_preview() -> None:
    st.header("2. Preview Dataset")

    examples = st.session_state.raw_dataset

    if examples is None:
        st.warning("No dataset has been uploaded yet.")
        if st.button("Back to upload"):
            go_to("upload")
        return

    st.write(f"Total examples: {len(examples)}")

    st.dataframe(examples[:20])

    st.subheader("Remove specific samples (optional)")

    # TODO(ML): expose per-row removal.
    # Likely use st.data_editor with a "keep" checkbox column,
    # then filter raw_dataset by selection.

    if st.button("Continue"):
        go_to("split")


# ============================================================
# STEP 4 — Train/Validation Split
# ============================================================

def render_split() -> None:
    st.header("3. Define Train/Validation Split")

    pct = st.slider(
        "% of dataset used for training",
        50,
        100,
        90,
    )

    st.session_state.run_config["train_pct"] = pct

    # TODO(ML): actually perform the split here or pass through
    # to the curation/training pipeline.

    if st.button("Continue"):
        go_to("curation_choice")


# ============================================================
# STEP 5/6 — Curation Choice + Threshold
# ============================================================

def render_curation_choice() -> None:
    st.header("4. Curation")

    run_curation = st.radio(
        "Run LLM-based curation?",
        ["Yes", "No"],
    )

    if run_curation == "Yes":
        threshold_mode = st.selectbox(
            "Threshold",
            ["Weak", "Medium", "Strong", "Custom"],
        )

        if threshold_mode == "Custom":
            custom_threshold = st.slider(
                "Custom threshold",
                0.0,
                10.0,
                6.7,
            )

            st.session_state.run_config["threshold"] = custom_threshold

        else:
            preset_map = {
                "Weak": 5.0,
                "Medium": 6.0,
                "Strong": 6.7,
            }

            st.session_state.run_config["threshold"] = (
                preset_map[threshold_mode]
            )

        if st.button("Run curation"):
            # TODO(ML): call:
            # curate_dataset(raw_dataset)
            #
            # Then:
            # classify_dataset(scored, threshold=...)

            st.info(
                "Wire up curate_dataset() from curation/curator.py"
            )

            go_to("curation_report")

    else:
        st.session_state.curated_dataset = (
            st.session_state.raw_dataset
        )

        if st.button("Skip to fine-tuning"):
            go_to("finetune_setup")


# ============================================================
# STEP 7 — Curation Report + Rejected-Sample Review
# ============================================================

def render_curation_report() -> None:
    st.header("5. Curation Report")

    # TODO(ML):
    # Render the real scored_dataset:
    # - kept/rejected/failed counts
    # - per-example scores
    # - per-example reasons
    # - rejected samples
    # - accept/reject manual override controls
    #
    # Manual decisions should use the example's stable "_id",
    # not question text.

    st.info(
        "Wire up scored_dataset display + "
        "rejected-sample override here"
    )

    if st.button("Continue to fine-tuning"):
        go_to("finetune_setup")


# ============================================================
# STEP 8/9/10 — Fine-Tune Setup
# ============================================================

def render_finetune_setup() -> None:
    st.header("6. Fine-Tune Setup")

    model_name = st.selectbox(
        "Base model",
        [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Other...",
        ],
    )

    method = st.radio(
        "Method",
        [
            "LoRA",
            "Full Fine-Tuning",
        ],
    )

    st.session_state.run_config["model_name"] = model_name
    st.session_state.run_config["method"] = method

    # TODO(ML):
    # Call training.finetune.decide_qlora(model_name, gpu_info)
    # and display the automatic QLoRA decision and reasoning.

    st.info(
        "Wire up decide_qlora() here to show the "
        "auto-QLoRA decision"
    )

    if st.button("Start fine-tuning"):
        go_to("training_monitor")


# ============================================================
# STEP 11/12/13 — Live Training Monitor
# ============================================================

def render_training_monitor() -> None:
    st.header("7. Training Monitor")

    # TODO(ML):
    #
    # Call training.finetune.run_finetune() with a
    # progress_callback that updates:
    #
    # - live loss chart
    # - throughput
    # - memory usage
    # - current step/epoch
    #
    # run_finetune() already calls forecasting.py and
    # decision_engine.py internally every 3 epochs through
    # ProgressCallback.on_evaluate.
    #
    # The callback update dict contains "forecast_check" on
    # those epochs:
    #
    # {
    #     "forecast": ...,
    #     "difficulty": ...,
    #     "noise": ...,
    #     "decision": ...
    # }
    #
    # Only show the recommendation banner when:
    #
    # forecast_check["decision"]["notify_user"] is True

    st.info(
        "Wire up run_finetune() with live progress "
        "callbacks here"
    )

    if st.button("Training complete — review results"):
        go_to("training_review")


# ============================================================
# STEP 14 — Post-Training Review
# ============================================================

def render_training_review() -> None:
    st.header("8. Training Review")

    # TODO(ML):
    #
    # Render the FULL history of forecast_check entries collected
    # during training_monitor, not just the final decision.
    #
    # Also provide a user-controlled forecast horizon and call:
    #
    # forecast_n_epochs_ahead(
    #     val_loss_history,
    #     n_epochs_ahead=user_selected_horizon,
    # )
    #
    # This can be called again whenever the user changes the
    # forecast horizon.

    st.info("Wire up full diagnostic report here")

    if st.button("Continue to evaluation"):
        go_to("evaluation")


# ============================================================
# STEP 15 — Test + Compare
# ============================================================

def render_evaluation() -> None:
    st.header("9. Evaluation")

    # TODO(ML):
    #
    # 1. Prepare the evaluation dataset.
    #
    # 2. Generate base-model outputs:
    #       generate_outputs(..., use_adapter=False)
    #
    #    This must use model.disable_adapter() because get_peft_model()
    #    modifies the base model in place.
    #
    # 3. Generate fine-tuned outputs:
    #       generate_outputs(..., use_adapter=True)
    #
    # 4. Judge the outputs:
    #       judge_outputs(...)
    #
    # 5. Render:
    #       question / reference / base output /
    #       fine-tuned output / judge verdict

    st.info(
        "Wire up generate_outputs() + judge_outputs() here"
    )

    if st.button("Generate final report"):
        go_to("final_report")


# ============================================================
# STEP 16/17 — Final Report + Reproducibility Export
# ============================================================

def render_final_report() -> None:
    st.header("10. Final Report")

    # TODO(ML):
    #
    # Assemble:
    # - curation report
    # - training metrics
    # - forecasting/decision history
    # - evaluation results
    # - LLM trace information
    #
    # into one downloadable Markdown/JSON report.

    st.info("Wire up full report assembly here")

    st.subheader("Reproducibility")

    if st.button("Export run config"):
        st.download_button(
            "Download run_config.json",
            data=json.dumps(
                st.session_state.run_config,
                indent=2,
            ),
            file_name="run_config.json",
            mime="application/json",
        )


# ============================================================
# Router
# ============================================================

STEP_RENDERERS = {
    "welcome": render_welcome,
    "upload": render_upload,
    "preview": render_preview,
    "split": render_split,
    "curation_choice": render_curation_choice,
    "curation_report": render_curation_report,
    "finetune_setup": render_finetune_setup,
    "training_monitor": render_training_monitor,
    "training_review": render_training_review,
    "evaluation": render_evaluation,
    "final_report": render_final_report,
}


STEP_RENDERERS[st.session_state.step]()