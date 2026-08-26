"""
dashboard/app.py

Streamlit shell for Smartune.

This file owns navigation, layout, and session state. It does NOT
contain curation, training, forecasting, or evaluation logic itself.
Every step calls into the corresponding module.

Run with:
    streamlit run dashboard/app.py

Sandbox note: several backend calls need a GPU and/or ANTHROPIC_API_KEY
that may not be available wherever this is run. Every real backend call
below is wrapped in try/except with a clearly-labeled synthetic/demo
fallback (mirroring dashboard/dev_dashboard.py's fake data) so the whole
flow stays clickable end-to-end even without them.
"""

import json
import random
import time
import traceback

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
from curation.preprocess import detect_schema, normalize_dataset
from training.finetune import decide_qlora, run_finetune
from training.forecasting import (
    compute_difficulty_proxy,
    forecast_n_epochs_ahead,
    noise_floor,
)
from training.decision_engine import decide_training_action
from training.check_dataset import (
    assess_dataset_before_finetuning,
    decide_dataset_warning,
)
from training.run_log import log_finetune_outcome
from training.report import generate_training_report
from training.reproducibility import export_run_config
from evaluation.eval_harness import generate_outputs
from evaluation.llm_judge import judge_outputs
from evaluation.report_final import (
    compute_summary,
    cross_check_with_training_curve,
    generate_summary_report,
    generate_detailed_report,
)

# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="Smartune",
    layout="wide",
)

SMARTUNE_FONT = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, '
    '"Helvetica Neue", Arial, sans-serif'
)

st.markdown(
    f"""
    <style>
    html, body, [class*="css"], .stApp, .stMarkdown, .stButton button,
    .stTextInput input, .stTextArea textarea, .stSelectbox, .stRadio,
    .stTabs, table, th, td, code, pre, [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {{
        font-family: {SMARTUNE_FONT} !important;
    }}

    .stApp {{
        background-color: #F5F5F5;
        color: #303841;
    }}

    /* ---- Top bar (Cohere-style) ---- */
    .smartune-topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 10px;
        padding: 14px 24px;
        margin-bottom: 20px;
    }}
    .smartune-topbar .brand {{
        font-size: 1.15rem;
        font-weight: 700;
        color: #303841;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .smartune-topbar .brand .dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #FF5722;
        display: inline-block;
    }}
    .smartune-topbar .steps {{
        font-size: 0.8rem;
        letter-spacing: 0.04em;
        color: #76ABAE;
        text-transform: uppercase;
        font-weight: 600;
    }}

    /* ---- Sidebar (Cohere-style: white, thin divider, orange active dot) ---- */
    [data-testid="stSidebar"] {{
        background-color: #FFFFFF;
        border-right: 1px solid #E5E5E5;
    }}
    [data-testid="stSidebar"] .block-container {{
        padding-top: 1.5rem;
    }}
    .smartune-nav-section {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #9AA1A9;
        margin: 18px 0 6px 4px;
        text-transform: uppercase;
    }}
    .smartune-nav-item {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 4px;
        border-radius: 6px;
        color: #303841;
        font-size: 0.92rem;
    }}
    .smartune-nav-item .dot {{
        width: 7px;
        height: 7px;
        min-width: 7px;
        border-radius: 50%;
        background: #D9D9D9;
    }}
    .smartune-nav-item.active {{
        font-weight: 700;
        background: rgba(118, 171, 174, 0.14);
    }}
    .smartune-nav-item.active .dot {{
        background: #FF5722;
    }}
    .smartune-nav-item.done .dot {{
        background: #76ABAE;
    }}

    /* ---- Cards ---- */
    .smartune-card {{
        background-color: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 18px;
    }}
    .smartune-card.accent-teal {{
        background-color: rgba(118, 171, 174, 0.14);
        border-color: rgba(118, 171, 174, 0.35);
    }}
    .smartune-card.accent-orange {{
        background-color: rgba(255, 87, 34, 0.08);
        border-color: rgba(255, 87, 34, 0.3);
    }}
    .smartune-badge {{
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: #FF5722;
        border: 1px solid #FF5722;
        border-radius: 5px;
        padding: 2px 8px;
        margin-bottom: 8px;
        text-transform: uppercase;
    }}

    /* ---- Buttons ---- */
    .stButton > button, .stDownloadButton > button {{
        background-color: #FF5722;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.1rem;
        font-family: {SMARTUNE_FONT} !important;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: #E64A19;
        color: #FFFFFF;
    }}
    .stButton > button:focus:not(:active) {{
        color: #FFFFFF;
    }}

    /* ---- Misc ---- */
    h1, h2, h3, h4, h5, h6 {{
        color: #303841;
        font-family: {SMARTUNE_FONT} !important;
    }}
    a {{ color: #FF5722; }}
    [data-testid="stExpander"] {{
        background-color: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 10px;
    }}
    [data-testid="stMetric"] {{
        background-color: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 10px;
        padding: 12px 16px;
    }}
    .stProgress > div > div > div > div {{
        background-color: #FF5722;
    }}
    hr {{ border-color: #E5E5E5; }}
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

# classify_dataset() output: {"kept": [...], "rejected": [...], "failed": [...]}
if "classification" not in st.session_state:
    st.session_state.classification = None

if "manual_overrides" not in st.session_state:
    st.session_state.manual_overrides = {}

if "final_classification" not in st.session_state:
    st.session_state.final_classification = None

if "curation_demo" not in st.session_state:
    st.session_state.curation_demo = False

if "qlora_decision" not in st.session_state:
    st.session_state.qlora_decision = None

if "training_result" not in st.session_state:
    st.session_state.training_result = None

if "forecast_checks" not in st.session_state:
    st.session_state.forecast_checks = []

if "training_progress" not in st.session_state:
    st.session_state.training_progress = []

if "training_demo" not in st.session_state:
    st.session_state.training_demo = False

if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

if "eval_demo" not in st.session_state:
    st.session_state.eval_demo = False


# ============================================================
# Navigation
# ============================================================

def go_to(step_name: str) -> None:
    """Navigate to a different dashboard step."""
    st.session_state.step = step_name


PIPELINE_STEPS = [
    ("welcome", "Welcome"),
    ("upload", "Upload Dataset"),
    ("preview", "Preview Dataset"),
    ("split", "Train/Val Split"),
    ("curation_choice", "Curation"),
    ("curation_report", "Curation Report"),
    ("finetune_setup", "Fine-Tune Setup"),
    ("training_monitor", "Training Monitor"),
    ("training_review", "Training Review"),
    ("evaluation", "Evaluation"),
    ("final_report", "Final Report"),
]


def render_topbar() -> None:
    current_label = dict(PIPELINE_STEPS).get(st.session_state.step, "")
    st.markdown(
        f"""
        <div class="smartune-topbar">
            <div class="brand"><span class="dot"></span>Smartune</div>
            <div class="steps">{current_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_nav() -> None:
    current_step = st.session_state.step
    current_index = next(
        (i for i, (key, _) in enumerate(PIPELINE_STEPS) if key == current_step),
        0,
    )

    with st.sidebar:
        st.markdown(
            '<div class="smartune-nav-section">Pipeline</div>',
            unsafe_allow_html=True,
        )
        for i, (key, label) in enumerate(PIPELINE_STEPS):
            css_class = "smartune-nav-item"
            if key == current_step:
                css_class += " active"
            elif i < current_index:
                css_class += " done"
            st.markdown(
                f'<div class="{css_class}"><span class="dot"></span>{label}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="smartune-nav-section">Session</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Examples loaded: "
            f"{len(st.session_state.raw_dataset) if st.session_state.raw_dataset else 0}"
        )


# ============================================================
# Demo/fallback fixtures (mirrors dev_dashboard.py's fake data)
# ============================================================

FAKE_VAL_LOSSES = [2.45, 2.21, 1.87, 1.72, 1.64, 1.59, 1.57, 1.56]


def _fake_scored_dataset(raw_examples):
    """Build a demo-scored dataset shaped exactly like curate_dataset()'s
    real output, so downstream code (classify_dataset, overrides, report
    generation) exercises the real contract even without an API key."""
    scored = []
    for i, ex in enumerate(raw_examples or [{"question": "Demo question?", "answer": "Demo answer."}] * 8):
        score = round(random.uniform(3.0, 9.5), 2)
        scored.append({
            **ex,
            "_id": f"demo-{i}",
            "_curation": {
                "clarity": round(score),
                "correctness": round(score),
                "value": round(score),
                "avg_score": score,
                "keep": score >= 6.7,
                "reason": "Synthetic demo score (no ANTHROPIC_API_KEY / curation call failed).",
                "curation_failed": False,
            },
        })
    return scored


# ============================================================
# STEP 1 — Welcome
# ============================================================

def render_welcome() -> None:
    st.markdown(
        """
        <div class="smartune-card">
            <span class="smartune-badge">NEW</span>
            <h1 style="margin:0 0 6px 0;">Welcome to Smartune</h1>
            <p style="font-size:1.05rem; color:#303841; max-width:640px;">
                Agentic fine-tuning pipeline: an LLM curates and routes your
                dataset, LoRA/QLoRA fine-tunes the model, and a forecasting
                engine watches the run mid-training and recommends stopping,
                continuing, or flagging it as unlikely to succeed.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="smartune-card accent-teal">
                <h4 style="margin-top:0;">Curate &amp; Fine-Tune</h4>
                <p style="margin-bottom:0;">
                    Score examples with an LLM rubric, filter duplicates,
                    and launch a LoRA or full fine-tune with automatic
                    QLoRA decisions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="smartune-card accent-orange">
                <h4 style="margin-top:0;">Forecast &amp; Evaluate</h4>
                <p style="margin-bottom:0;">
                    Watch a live forecast of your training curve, get a
                    stop/continue recommendation, then judge fine-tuned vs.
                    base outputs side by side.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
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

        raw_examples = [
            json.loads(line)
            for line in content.strip().split("\n")
        ]

        st.success(f"Loaded {len(raw_examples)} raw examples")

        # ---- Auto-detect schema and normalize to {"question","answer"} ----
        # Cheap heuristics cover Alpaca (instruction/input/output),
        # prompt/completion, instruction/response, and chat "messages"
        # formats with no API call. Only an unrecognized schema falls
        # back to asking Claude which fields hold the question/answer.
        try:
            detection = detect_schema(raw_examples)
        except Exception as exc:
            st.error(
                "Could not auto-detect the dataset schema, and the "
                f"Claude fallback failed too: {exc}"
            )
            detection = None

        if detection is not None:
            if detection["confidence"] == "heuristic":
                st.info(
                    f"Detected schema **{detection['schema']}** from fields "
                    f"{detection['sample_fields']} (no API call needed)."
                )
            elif detection["confidence"] == "llm":
                st.info(
                    "Fields didn't match a known schema "
                    f"({detection['sample_fields']}) — asked Claude to map "
                    f"them: {detection['mapping']}"
                )

            examples = normalize_dataset(raw_examples, detection)
            st.session_state.raw_dataset = examples
            go_to("preview")
        else:
            st.warning(
                "Falling back to raw examples as-is — curation may fail "
                "if they don't already have \"question\"/\"answer\" keys."
            )
            st.session_state.raw_dataset = raw_examples
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

        mode = st.radio("Classification mode", ["normal", "hyper"], horizontal=True)
        st.session_state.run_config["curation_mode"] = mode

        if st.button("Run curation"):
            raw_dataset = st.session_state.raw_dataset or []
            threshold = st.session_state.run_config["threshold"]

            demo = False
            try:
                with st.spinner("Scoring dataset with Claude Haiku..."):
                    scored = curate_dataset(raw_dataset)
            except Exception as e:
                demo = True
                st.warning(
                    f"curate_dataset() failed ({type(e).__name__}: {e}) — "
                    "falling back to synthetic demo scores. This usually "
                    "means ANTHROPIC_API_KEY is not set in this environment."
                )
                scored = _fake_scored_dataset(raw_dataset)

            classification = classify_dataset(
                scored,
                threshold=threshold,
                mode=mode,
            )

            st.session_state.scored_dataset = scored
            st.session_state.classification = classification
            st.session_state.manual_overrides = {}
            st.session_state.final_classification = classification
            st.session_state.curation_demo = demo

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

    classification = st.session_state.classification

    if classification is None:
        st.warning("No curation has been run yet.")
        if st.button("Back to curation"):
            go_to("curation_choice")
        return

    if st.session_state.curation_demo:
        st.info(
            "DEMO DATA: curate_dataset() could not be reached (no "
            "ANTHROPIC_API_KEY?), so scores below are synthetic."
        )

    kept = classification["kept"]
    rejected = classification["rejected"]
    failed = classification["failed"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Kept", len(kept))
    c2.metric("Rejected", len(rejected))
    c3.metric("Failed", len(failed))

    st.caption(
        f"Threshold used: {classification.get('threshold_used')} "
        f"({'auto-computed' if classification.get('threshold_auto') else 'manual'})"
    )

    st.subheader("Per-example scores")
    all_examples = kept + rejected + failed
    st.dataframe([
        {
            "id": ex["_id"],
            "question": ex.get("question", "")[:80],
            "avg_score": ex["_curation"].get("avg_score"),
            "status": ex["_curation"].get("final_status"),
            "reason": ex["_curation"].get("reason", ""),
        }
        for ex in all_examples
    ])

    st.subheader("Rejected / failed samples — manual override")
    st.caption("Overrides are keyed by each example's stable _id, not question text.")

    overrides = dict(st.session_state.manual_overrides)

    for ex in rejected + failed:
        ex_id = ex["_id"]
        with st.expander(f"{ex.get('question', '')[:80]}  (id={ex_id})"):
            st.write(f"**Answer:** {ex.get('answer', '')}")
            st.write(f"**Score:** {ex['_curation'].get('avg_score')}")
            st.write(f"**Reason:** {ex['_curation'].get('reason')}")

            current = overrides.get(ex_id, "none")
            choice = st.radio(
                "Decision",
                ["none", "accept", "reject"],
                index=["none", "accept", "reject"].index(current),
                key=f"override_{ex_id}",
                horizontal=True,
            )
            if choice != "none":
                overrides[ex_id] = choice
            elif ex_id in overrides:
                del overrides[ex_id]

    st.session_state.manual_overrides = overrides

    if st.button("Apply overrides"):
        final_classification = apply_manual_overrides(classification, overrides)
        st.session_state.final_classification = final_classification
        st.session_state.curated_dataset = final_classification["kept"]
        st.success(
            f"Applied. Final: {len(final_classification['kept'])} kept, "
            f"{len(final_classification['rejected'])} rejected, "
            f"{len(final_classification['failed'])} still failed."
        )

    if st.session_state.curated_dataset is None:
        st.session_state.curated_dataset = kept

    st.subheader("Downloads")
    try:
        report_md = generate_curation_report(
            st.session_state.final_classification or classification,
            threshold=classification.get("threshold_used"),
            mode=st.session_state.run_config.get("curation_mode", "normal"),
        )
        st.download_button(
            "Download curation report (Markdown)",
            data=report_md,
            file_name="curation_report.md",
            mime="text/markdown",
        )
    except Exception as e:
        st.caption(f"Could not build curation report: {e}")

    try:
        csv_data = generate_decisions_csv(st.session_state.final_classification or classification)
        st.download_button(
            "Download decisions (CSV)",
            data=csv_data,
            file_name="curation_decisions.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.caption(f"Could not build decisions CSV: {e}")

    if st.button("Continue to fine-tuning"):
        go_to("finetune_setup")


# ============================================================
# STEP 8/9/10 — Fine-Tune Setup
# ============================================================

def _detect_gpu_info():
    """Best-effort GPU detection, falling back to a reasonable assumed
    single-GPU config (mirrors dev_dashboard.py's approach of using
    plausible fake specs when there's no real GPU to query)."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return torch.cuda.device_count(), round(props.total_memory / 1e9, 1), False
    except Exception:
        pass
    # No GPU available in this sandbox — assume a single 16GB GPU so the
    # decision logic can still be demonstrated end-to-end.
    return 1, 16.0, True


def render_finetune_setup() -> None:
    st.header("6. Fine-Tune Setup")

    model_name = st.selectbox(
        "Base model",
        [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Other...",
        ],
    )
    if model_name == "Other...":
        model_name = st.text_input("Model name", "Qwen/Qwen2.5-1.5B-Instruct")

    method = st.radio(
        "Method",
        [
            "LoRA",
            "Full Fine-Tuning",
        ],
    )

    st.session_state.run_config["model_name"] = model_name
    st.session_state.run_config["method"] = "lora" if method == "LoRA" else "full"

    num_gpus, gpu_memory_gb, gpu_is_fake = _detect_gpu_info()
    if gpu_is_fake:
        st.caption(
            f"No GPU detected in this environment — assuming {num_gpus} x "
            f"{gpu_memory_gb}GB GPU for the QLoRA decision below (demo)."
        )
    else:
        st.caption(f"Detected {num_gpus} GPU(s), {gpu_memory_gb}GB each.")

    if method == "LoRA":
        try:
            use_qlora, reason = decide_qlora(model_name, num_gpus, gpu_memory_gb)
            st.session_state.qlora_decision = {"use_qlora": use_qlora, "reason": reason}
            if use_qlora:
                st.warning(f"Auto-QLoRA decision: **use QLoRA**. {reason}")
            else:
                st.success(f"Auto-QLoRA decision: **standard LoRA is sufficient**. {reason}")
        except Exception as e:
            st.error(f"decide_qlora() failed: {e}")
            st.session_state.qlora_decision = None

    num_epochs = st.slider("Epochs", 1, 12, 6)
    st.session_state.run_config["num_train_epochs"] = num_epochs
    st.session_state.run_config["num_gpus"] = num_gpus
    st.session_state.run_config["gpu_memory_gb"] = gpu_memory_gb

    demo_mode = st.checkbox(
        "Demo mode (simulate training — for when no GPU is available)",
        value=False,
        help="Skips the real run_finetune() call (which needs a GPU and a "
        "model download) and instead simulates a plausible training run "
        "with the same forecasting/decision-engine pipeline wired in. "
        "Off by default — the real run_finetune() still falls back to "
        "demo automatically if it raises (e.g. an OOM or missing model).",
    )
    st.session_state.training_demo = demo_mode

    if st.button("Start fine-tuning"):
        go_to("training_monitor")


# ============================================================
# STEP 11/12/13 — Live Training Monitor
# ============================================================

def _run_demo_training(progress_callback, num_epochs):
    """Simulate a training run with a plausible loss curve, calling the
    REAL forecasting + decision_engine pipeline every 3 epochs — same
    contract run_finetune() uses — so this exercises real logic even
    though the loss values themselves are synthetic."""
    val_loss_history = []
    train_loss_history = []
    start = time.time()

    base = 2.5
    for epoch in range(1, num_epochs + 1):
        loss = max(0.3, base * (0.82 ** epoch) + random.uniform(-0.03, 0.03))
        val_loss_history.append(loss)
        train_loss_history.append(loss + random.uniform(0.05, 0.15))

        update = {
            "step": epoch * 10,
            "loss": train_loss_history[-1],
            "val_loss": loss,
        }

        if epoch % 3 == 0 and epoch >= 3:
            try:
                forecast_result = forecast_n_epochs_ahead(val_loss_history, save_plot_path=None)
                difficulty = compute_difficulty_proxy(val_loss_history)
                noise = noise_floor(val_loss_history)
                try:
                    decision = decide_training_action(
                        val_losses=val_loss_history,
                        forecast=forecast_result["forecast"],
                        difficulty=difficulty,
                        noise=noise,
                    )
                except Exception as e:
                    decision = {
                        "action": "CONTINUE",
                        "notify_user": False,
                        "reason": f"decide_training_action() failed ({e}); no API key? Defaulting to CONTINUE.",
                    }
                update["forecast_check"] = {
                    "epoch": epoch,
                    "forecast": forecast_result["forecast"],
                    "difficulty": difficulty,
                    "noise": noise,
                    "decision": decision,
                }
            except Exception as e:
                update["forecast_check"] = {"epoch": epoch, "error": str(e)}

        progress_callback(update)
        time.sleep(0.05)

    elapsed = time.time() - start
    return {
        "training_time_s": elapsed,
        "throughput_examples_per_sec": (len(val_loss_history) * 8) / max(elapsed, 1e-6),
        "peak_gpu_memory_gb": round(random.uniform(4.0, 7.5), 2),
        "final_loss": train_loss_history[-1],
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "used_qlora": bool(st.session_state.qlora_decision and st.session_state.qlora_decision["use_qlora"]),
        "used_jax_full_finetune": False,
        "seed": 42,
    }


def render_training_monitor() -> None:
    st.header("7. Training Monitor")

    if st.session_state.training_result is not None:
        st.success("Training already completed for this run.")
        if st.button("Training complete — review results"):
            go_to("training_review")
        return

    num_epochs = st.session_state.run_config.get("num_train_epochs", 6)
    demo_mode = st.session_state.get("training_demo", True)

    loss_chart = st.empty()
    metrics_placeholder = st.empty()
    banner_placeholder = st.container()

    progress_log = []
    forecast_checks = []

    def progress_callback(update):
        progress_log.append(update)
        losses = [u["val_loss"] for u in progress_log if u.get("val_loss") is not None]
        train_losses = [u["loss"] for u in progress_log if u.get("loss") is not None]
        if losses or train_losses:
            loss_chart.line_chart({
                "train_loss": train_losses,
                "val_loss": losses,
            } if losses and train_losses else {"loss": train_losses or losses})

        metrics_placeholder.write(
            f"Step: {update.get('step')} | loss: {update.get('loss')} | "
            f"val_loss: {update.get('val_loss')}"
        )

        fc = update.get("forecast_check")
        if fc:
            forecast_checks.append(fc)
            decision = fc.get("decision")
            if decision and decision.get("notify_user"):
                with banner_placeholder:
                    st.warning(
                        f"**Recommendation (epoch {fc['epoch']}): "
                        f"{decision['action']}** — {decision['reason']}"
                    )

    if st.button("Run training"):
        result = None
        used_demo = demo_mode

        if not demo_mode:
            try:
                train_dataset = st.session_state.curated_dataset or [
                    {"question": "Placeholder question", "answer": "Placeholder answer"}
                ]
                with st.spinner("Running run_finetune() — this needs a GPU/model download..."):
                    result = run_finetune(
                        model_name=st.session_state.run_config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
                        method=st.session_state.run_config.get("method", "lora"),
                        train_dataset=train_dataset,
                        val_dataset=train_dataset,
                        progress_callback=progress_callback,
                        num_train_epochs=num_epochs,
                        num_gpus=st.session_state.run_config.get("num_gpus", 1),
                        gpu_memory_gb=st.session_state.run_config.get("gpu_memory_gb", 16.0),
                    )
            except Exception as e:
                used_demo = True
                st.warning(
                    f"run_finetune() failed ({type(e).__name__}: {e}) — falling "
                    "back to a simulated training run. This is expected without "
                    "a GPU / local model weights."
                )
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())

        if result is None:
            result = _run_demo_training(progress_callback, num_epochs)

        st.session_state.training_result = result
        st.session_state.forecast_checks = forecast_checks
        st.session_state.training_progress = progress_log
        st.session_state.training_demo = used_demo

        try:
            log_finetune_outcome(
                curation_stats=st.session_state.final_classification
                or {"kept": [], "rejected": [], "failed": []},
                training_config=dict(st.session_state.run_config),
                final_result={k: v for k, v in result.items() if k != "model"},
            )
        except Exception:
            pass

        st.rerun()

    if st.session_state.training_demo:
        st.caption(
            "DEMO MODE: simulating training (no GPU / model download in this "
            "sandbox). The forecasting + decision engine calls above are real."
        )

    if st.button("Training complete — review results", disabled=st.session_state.training_result is None):
        go_to("training_review")


# ============================================================
# STEP 14 — Post-Training Review
# ============================================================

def render_training_review() -> None:
    st.header("8. Training Review")

    result = st.session_state.training_result
    if result is None:
        st.warning("No training result yet.")
        if st.button("Back to training"):
            go_to("training_monitor")
        return

    if st.session_state.training_demo:
        st.info("DEMO DATA: training result below is simulated, not from a real GPU run.")

    val_losses = result.get("val_loss_history", [])

    c1, c2, c3 = st.columns(3)
    c1.metric("Final loss", f"{result.get('final_loss', 0):.4f}")
    c2.metric("Training time (s)", f"{result.get('training_time_s', 0):.1f}")
    c3.metric("Peak GPU mem (GB)", f"{result.get('peak_gpu_memory_gb', 0):.2f}")

    st.line_chart({"val_loss": val_losses})

    st.subheader("Forecast / decision history")
    checks = st.session_state.forecast_checks
    if not checks:
        st.write("No forecast checkpoints were recorded (run may be shorter than 3 epochs).")
    for fc in checks:
        if "error" in fc:
            st.error(f"Epoch {fc['epoch']}: forecast_check error — {fc['error']}")
            continue
        decision = fc["decision"]
        label = f"Epoch {fc['epoch']}: {decision['action']}" + (" ⚠️" if decision.get("notify_user") else "")
        with st.expander(label):
            st.json(fc["forecast"])
            st.write("Difficulty:", fc["difficulty"])
            st.write("Noise floor:", fc["noise"])
            st.write("Reason:", decision["reason"])

    st.subheader("Interactive forecast horizon")
    horizon = st.slider("Epochs ahead", 1, 10, 3, key="review_horizon")
    if len(val_losses) >= 3:
        try:
            live_forecast = forecast_n_epochs_ahead(
                val_losses, n_epochs_ahead=horizon, save_plot_path=None
            )
            st.json(live_forecast["forecast"])
        except Exception as e:
            st.error(f"forecast_n_epochs_ahead() failed: {e}")
    else:
        st.caption("Need at least 3 validation-loss points to forecast.")

    try:
        st.download_button(
            "Download training report (Markdown)",
            data=generate_training_report(result, checks),
            file_name="training_report.md",
            mime="text/markdown",
        )
    except Exception as e:
        st.caption(f"Could not build training report: {e}")

    if st.button("Continue to evaluation"):
        go_to("evaluation")


# ============================================================
# STEP 15 — Test + Compare
# ============================================================

def _fake_eval_results(n=5):
    questions = [
        "What are the benefits of unit testing?",
        "Explain the CAP theorem in one paragraph.",
        "What is gradient descent?",
        "Summarize the plot of a two-sentence story.",
        "Why use version control?",
    ]
    results = []
    for i in range(n):
        q = questions[i % len(questions)]
        results.append({
            "question": q,
            "reference": f"Reference answer for: {q}",
            "base_output": f"[DEMO base model output for] {q}",
            "ft_output": f"[DEMO fine-tuned output for] {q}",
            "score_a": {"quality": 6, "coherence": 6, "task_fit": 6},
            "score_b": {"quality": 7, "coherence": 8, "task_fit": 7},
            "winner": "B",
            "reason": "Synthetic demo verdict (no GPU/API key available).",
        })
    return results


def render_evaluation() -> None:
    st.header("9. Evaluation")

    result = st.session_state.training_result
    if result is None:
        st.warning("Run training first.")
        if st.button("Back to training"):
            go_to("training_monitor")
        return

    if st.button("Run evaluation"):
        eval_examples = (st.session_state.curated_dataset or [
            {"question": "What is the capital of France?", "answer": "Paris."}
        ])[:5]

        demo = False
        model = result.get("model")
        tokenizer = None

        try:
            if model is None:
                raise RuntimeError("No real model object in memory (demo training run).")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                st.session_state.run_config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
            )
            with st.spinner("Generating base-model outputs..."):
                base_outputs = generate_outputs(model, tokenizer, eval_examples, use_adapter=False)
            with st.spinner("Generating fine-tuned outputs..."):
                ft_outputs = generate_outputs(model, tokenizer, eval_examples, use_adapter=True)
            with st.spinner("Judging outputs with Claude..."):
                judged = judge_outputs(base_outputs, ft_outputs)
        except Exception as e:
            demo = True
            st.warning(
                f"generate_outputs()/judge_outputs() failed ({type(e).__name__}: {e}) "
                "— falling back to synthetic demo outputs/verdicts. This is expected "
                "without a GPU + loaded model and an ANTHROPIC_API_KEY."
            )
            judged = _fake_eval_results(len(eval_examples))

        st.session_state.eval_results = judged
        st.session_state.eval_demo = demo
        st.rerun()

    judged = st.session_state.eval_results
    if judged is None:
        st.caption("No evaluation run yet.")
    else:
        if st.session_state.eval_demo:
            st.info("DEMO DATA: outputs and judge verdicts below are synthetic.")

        summary = compute_summary(judged)
        cross = cross_check_with_training_curve(
            st.session_state.training_result.get("val_loss_history", []),
            summary["avg_quality_margin"],
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Fine-tuned wins", summary["wins"])
        c2.metric("Base wins", summary["losses"])
        c3.metric("Ties", summary["ties"])
        st.write(cross["note"])

        for r in judged:
            with st.expander(r["question"]):
                st.write(f"**Reference:** {r['reference']}")
                st.write(f"**Base output:** {r['base_output']}")
                st.write(f"**Fine-tuned output:** {r['ft_output']}")
                if r["winner"] == "JUDGE_FAILED":
                    st.error(f"Judge failed: {r['reason']}")
                else:
                    st.write(f"**Base scores:** {r['score_a']}")
                    st.write(f"**Fine-tuned scores:** {r['score_b']}")
                    st.write(f"**Winner:** {r['winner']} — {r['reason']}")

        st.download_button(
            "Download summary report (Markdown)",
            data=generate_summary_report(summary, cross),
            file_name="eval_summary_report.md",
            mime="text/markdown",
        )
        st.download_button(
            "Download detailed report (Markdown)",
            data=generate_detailed_report(judged),
            file_name="eval_detailed_report.md",
            mime="text/markdown",
        )

    if st.button("Generate final report"):
        go_to("final_report")


# ============================================================
# STEP 16/17 — Final Report + Reproducibility Export
# ============================================================

def render_final_report() -> None:
    st.header("10. Final Report")

    classification = st.session_state.final_classification or st.session_state.classification
    training_result = st.session_state.training_result
    judged = st.session_state.eval_results

    sections = ["# Smartune — Final Run Report", ""]

    if classification:
        sections.append("## Curation")
        sections.append(
            f"- Kept: {len(classification['kept'])}, Rejected: "
            f"{len(classification['rejected'])}, Failed: {len(classification['failed'])}"
        )
        if st.session_state.curation_demo:
            sections.append("- (DEMO curation scores — no API key was available.)")
        sections.append("")

    if training_result:
        sections.append("## Training")
        sections.append(generate_training_report(training_result, st.session_state.forecast_checks))
        if st.session_state.training_demo:
            sections.append("\n_(DEMO training run — simulated, not from a real GPU.)_")
        sections.append("")

    if judged:
        summary = compute_summary(judged)
        cross = cross_check_with_training_curve(
            training_result.get("val_loss_history", []) if training_result else [],
            summary["avg_quality_margin"],
        )
        sections.append("## Evaluation")
        sections.append(generate_summary_report(summary, cross))
        if st.session_state.eval_demo:
            sections.append("\n_(DEMO evaluation — synthetic outputs/verdicts.)_")
        sections.append("")

    try:
        from evaluation.llm_trace import load_llm_trace
        trace = load_llm_trace()
        sections.append("## LLM Call Trace")
        sections.append(f"- {len(trace)} Claude call(s) logged this session.")
        sections.append("")
    except Exception:
        pass

    full_report = "\n".join(sections)
    st.markdown(full_report)

    st.download_button(
        "Download full report (Markdown)",
        data=full_report,
        file_name="smartune_final_report.md",
        mime="text/markdown",
    )

    st.subheader("Reproducibility")

    if st.button("Export run config"):
        try:
            exported = export_run_config(
                dataset_source={
                    "n_raw_examples": len(st.session_state.raw_dataset or []),
                    "train_pct": st.session_state.run_config.get("train_pct"),
                },
                curation_config={
                    "threshold": st.session_state.run_config.get("threshold"),
                    "mode": st.session_state.run_config.get("curation_mode"),
                },
                training_config=dict(st.session_state.run_config),
                training_result=training_result or {},
            )
        except Exception:
            exported = dict(st.session_state.run_config)

        st.download_button(
            "Download run_config.json",
            data=json.dumps(exported, indent=2, default=str),
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


render_sidebar_nav()
render_topbar()
STEP_RENDERERS[st.session_state.step]()