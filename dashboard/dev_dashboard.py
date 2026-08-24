"""
dashboard/dev_dashboard.py

DEVELOPER / TEST dashboard — exercises every module and feature
individually, with fake-but-realistic data where a real GPU or API key
isn't available, so each piece can be verified in isolation rather than
only as part of a full pipeline run.

This is NOT the user-facing app — see dashboard/app.py for that.
Purpose here is debugging and verification: "does this specific
function actually work, and what exactly does it return?"

Run with: streamlit run dashboard/dev_dashboard.py
"""

import sys
import os
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(page_title="Smartune — Dev/Test", layout="wide")

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

st.title("Smartune — Developer Test Dashboard")

# Realistic fake data reused across sections — shaped exactly like what
# the real pipeline produces, so these tests exercise the real contracts.
FAKE_VAL_LOSSES = [2.45, 2.21, 1.87, 1.72, 1.64, 1.59, 1.57, 1.56]
FAKE_SCORED_EXAMPLE = {
    "question": "What are the advantages of using a Scrum Agile methodology?",
    "answer": "Improved communication, increased accountability, faster delivery.",
    "_id": "fake-id-001",
    "_curation": {
        "clarity": 8, "correctness": 8, "value": 6, "avg_score": 7.33,
        "keep": True, "reason": "Clear and accurate but somewhat generic.",
        "curation_failed": False,
    },
}


def show_result(label: str, fn):
    """Run fn(), show its output or the real traceback if it fails."""
    try:
        result = fn()
        st.success(f"{label} — OK")
        st.json(result if isinstance(result, (dict, list)) else {"result": str(result)})
    except Exception as e:
        st.error(f"{label} — failed: {type(e).__name__}: {e}")
        with st.expander("Full traceback"):
            st.code(traceback.format_exc())


tabs = st.tabs([
    "Curation", "Dataset Check", "Forecasting",
    "Decision Engine", "Evaluation", "Reports", "Logs & Repro",
])


# ============================================================
# CURATION
# ============================================================
with tabs[0]:
    st.header("curation/curator.py")

    st.subheader("classify_dataset() — no API needed")

    auto_threshold = st.checkbox("Auto-compute threshold", value=True, key="cur_auto")
    threshold_method = st.radio(
        "Auto method", ["otsu", "percentile"], key="cur_thresh_method",
        horizontal=True, disabled=not auto_threshold,
    )
    manual_threshold = st.slider(
        "Threshold (used when auto is off)", 0.0, 10.0, 6.7,
        key="cur_thresh", disabled=auto_threshold,
    )
    mode = st.radio("Mode", ["normal", "hyper"], key="cur_mode", horizontal=True)

    if st.button("Run classify_dataset on fake scored data"):
        from curation.curator import classify_dataset
        fake_scored = [
            dict(FAKE_SCORED_EXAMPLE, _id="a", _curation=dict(FAKE_SCORED_EXAMPLE["_curation"], avg_score=8.5)),
            dict(FAKE_SCORED_EXAMPLE, _id="b", _curation=dict(FAKE_SCORED_EXAMPLE["_curation"], avg_score=3.0)),
            dict(FAKE_SCORED_EXAMPLE, _id="c", _curation=dict(FAKE_SCORED_EXAMPLE["_curation"], avg_score=6.8)),
            dict(FAKE_SCORED_EXAMPLE, _id="d", _curation={
                "clarity": None, "correctness": None, "value": None, "avg_score": None,
                "keep": None, "reason": "failed", "curation_failed": True}),
        ]

        def run():
            result = classify_dataset(
                fake_scored,
                threshold=None if auto_threshold else manual_threshold,
                mode=mode,
                threshold_method=threshold_method,
            )
            return {
                "kept": [e["_id"] for e in result["kept"]],
                "rejected": [e["_id"] for e in result["rejected"]],
                "failed": [e["_id"] for e in result["failed"]],
                "threshold_used": result["threshold_used"],
                "threshold_auto": result["threshold_auto"],
            }

        show_result("classify_dataset", run)

    st.subheader("apply_manual_overrides() — no API needed")
    if st.button("Test override (accept the failed example 'd')"):
        from curation.curator import classify_dataset, apply_manual_overrides
        fake_scored = [
            dict(FAKE_SCORED_EXAMPLE, _id="d", _curation={
                "clarity": None, "correctness": None, "value": None, "avg_score": None,
                "keep": None, "reason": "failed", "curation_failed": True}),
        ]
        classification = classify_dataset(fake_scored, threshold=6.7)
        show_result("apply_manual_overrides", lambda: {
            k: [{"id": e["_id"], "final_status": e["_curation"]["final_status"],
                 "automatic_status": e["_curation"].get("automatic_status")} for e in v]
            for k, v in apply_manual_overrides(classification, {"d": "accept"}).items()
        })

    st.subheader("curate_dataset() — NEEDS ANTHROPIC_API_KEY")
    if st.button("Run real curation on 2 examples"):
        from curation.curator import curate_dataset
        show_result("curate_dataset", lambda: curate_dataset([
            {"question": "What is the capital of France?", "answer": "Paris."},
            {"question": "Generate two similar SOUNDING but semantically different words for: Light",
             "answer": "Bright and Dim."},  # known task-mismatch case
        ]))


# ============================================================
# DATASET CHECK
# ============================================================
with tabs[1]:
    st.header("training/check_dataset.py")

    st.subheader("assess_dataset_before_finetuning() — no API needed")
    n_kept = st.number_input("kept", 0, 1000, 8, key="dc_kept")
    n_rejected = st.number_input("rejected", 0, 1000, 2, key="dc_rej")
    n_failed = st.number_input("failed", 0, 1000, 0, key="dc_fail")

    if st.button("Run heuristic check"):
        from training.check_dataset import assess_dataset_before_finetuning
        fake_classification = {
            "kept": [{}] * int(n_kept),
            "rejected": [{}] * int(n_rejected),
            "failed": [{}] * int(n_failed),
        }
        show_result("assess_dataset_before_finetuning",
                    lambda: assess_dataset_before_finetuning(fake_classification))

    st.subheader("decide_dataset_warning() — NEEDS ANTHROPIC_API_KEY")
    if st.button("Ask Claude whether to warn"):
        from training.check_dataset import assess_dataset_before_finetuning, decide_dataset_warning
        fake_classification = {"kept": [{}] * int(n_kept), "rejected": [{}] * int(n_rejected), "failed": []}
        heuristic = assess_dataset_before_finetuning(fake_classification)
        show_result("decide_dataset_warning", lambda: decide_dataset_warning(heuristic))


# ============================================================
# FORECASTING
# ============================================================
with tabs[2]:
    st.header("training/forecasting.py")

    st.subheader("compute_difficulty_proxy() + noise_floor() — no API/GPU needed")
    curve_text = st.text_area(
        "Validation losses (comma-separated)",
        ", ".join(str(v) for v in FAKE_VAL_LOSSES),
        key="fc_curve",
    )
    try:
        curve = [float(x.strip()) for x in curve_text.split(",") if x.strip()]
    except ValueError:
        curve = FAKE_VAL_LOSSES
        st.warning("Could not parse input, using default curve.")

    if st.button("Compute difficulty + noise"):
        from training.forecasting import compute_difficulty_proxy, noise_floor
        show_result("difficulty + noise", lambda: {
            "difficulty": compute_difficulty_proxy(curve),
            "noise_floor": noise_floor(curve),
        })

    st.subheader("forecast_n_epochs_ahead() — no special install needed")
    st.caption(
        "This is the user-controllable horizon feature. Now runs Arm A "
        "(parametric curve-fit extrapolation via scipy) instead of the "
        "old LC-PFN implementation — no GPU, no lcpfn install/patches, "
        "just numpy/scipy."
    )
    n_ahead = st.slider("Epochs ahead", 1, 10, 3, key="fc_ahead")
    if st.button("Run curve-fit forecast"):
        from training.forecasting import forecast_n_epochs_ahead
        result = None
        try:
            result = forecast_n_epochs_ahead(curve, n_epochs_ahead=n_ahead, save_plot_path="/tmp/dev_forecast.png")
            st.success("Forecast OK")
            st.json(result["forecast"])
            if result.get("plot_path") and os.path.exists(result["plot_path"]):
                st.image(result["plot_path"])
        except Exception as e:
            st.error(f"Forecast failed: {type(e).__name__}: {e}")
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())


# ============================================================
# DECISION ENGINE
# ============================================================
with tabs[3]:
    st.header("training/decision_engine.py")
    st.caption("NEEDS ANTHROPIC_API_KEY. Uses a fake forecast so lcpfn isn't required.")

    fake_forecast = {
        "median": [1.55, 1.54, 1.54],
        "lower_5": [1.45, 1.43, 1.42],
        "upper_95": [1.66, 1.67, 1.68],
    }
    st.write("Fake forecast being passed in:")
    st.json(fake_forecast)

    if st.button("Ask Claude for a training action"):
        from training.decision_engine import decide_training_action
        from training.forecasting import compute_difficulty_proxy, noise_floor
        show_result("decide_training_action", lambda: decide_training_action(
            val_losses=curve if 'curve' in dir() else FAKE_VAL_LOSSES,
            forecast=fake_forecast,
            difficulty=compute_difficulty_proxy(FAKE_VAL_LOSSES),
            noise=noise_floor(FAKE_VAL_LOSSES),
        ))


# ============================================================
# EVALUATION
# ============================================================
with tabs[4]:
    st.header("evaluation/")

    st.subheader("compute_summary() + cross_check — no API needed")
    st.caption("Uses fake judged results including a deliberate regression case.")

    if st.button("Run summary + cross-check on fake judged data"):
        from evaluation.report_final import compute_summary, cross_check_with_training_curve
        fake_judged = [
            {"question": "Q1", "reference": "R1", "base_output": "b1", "ft_output": "f1",
             "score_a": {"quality": 6, "coherence": 6, "task_fit": 6},
             "score_b": {"quality": 8, "coherence": 8, "task_fit": 8},
             "winner": "B", "reason": "Fine-tuned more focused."},
            {"question": "Q2", "reference": "R2", "base_output": "b2", "ft_output": "f2",
             "score_a": {"quality": 8, "coherence": 8, "task_fit": 8},
             "score_b": {"quality": 4, "coherence": 4, "task_fit": 4},
             "winner": "A", "reason": "Real regression — lost key details."},
            {"question": "Q3", "reference": "R3", "base_output": "b3", "ft_output": "f3",
             "score_a": None, "score_b": None,
             "winner": "JUDGE_FAILED", "reason": "Malformed judge response."},
        ]
        summary = compute_summary(fake_judged)
        st.json(summary)
        st.write("Cross-check against a falling loss curve:")
        st.json(cross_check_with_training_curve(FAKE_VAL_LOSSES, summary["avg_quality_margin"]))

    st.subheader("generate_outputs() / judge_outputs() — NEED GPU + API KEY")
    st.info(
        "These require an actual fine-tuned model object in memory (from "
        "run_finetune()) and a real API key, so they can't be exercised "
        "standalone here — they're covered by the full user-facing pipeline "
        "in dashboard/app.py instead."
    )


# ============================================================
# REPORTS
# ============================================================
with tabs[5]:
    st.header("Report generators — no API/GPU needed")

    if st.button("Generate training report (fake data)"):
        from training.report import generate_training_report
        fake_result = {
            "final_loss": 1.56, "training_time_s": 170.1,
            "throughput_examples_per_sec": 0.63, "peak_gpu_memory_gb": 4.73,
            "used_qlora": False, "val_loss_history": FAKE_VAL_LOSSES,
        }
        fake_checks = [{
            "epoch": 3,
            "forecast": {"median": [1.7, 1.65, 1.6], "lower_5": [1.6, 1.55, 1.5], "upper_95": [1.8, 1.78, 1.75]},
            "difficulty": {"prog": -0.19, "nonlin": 0.02, "vol": 0.15},
            "noise": 0.08,
            "decision": {"action": "CONTINUE", "notify_user": False, "reason": "Still improving."},
        }, {
            "epoch": 6,
            "forecast": {"median": [1.57, 1.57, 1.56], "lower_5": [1.5, 1.49, 1.48], "upper_95": [1.65, 1.66, 1.66]},
            "difficulty": {"prog": -0.15, "nonlin": 0.03, "vol": 0.12},
            "noise": 0.04,
            "decision": {"action": "STOP", "notify_user": True, "reason": "Curve has plateaued."},
        }]
        st.markdown(generate_training_report(fake_result, fake_checks))

    if st.button("Generate evaluation reports (fake data)"):
        from evaluation.report_final import compute_summary, cross_check_with_training_curve, generate_summary_report, generate_detailed_report
        fake_judged = [
            {"question": "Q1", "reference": "R1", "base_output": "base text 1", "ft_output": "ft text 1",
             "score_a": {"quality": 6, "coherence": 6, "task_fit": 6},
             "score_b": {"quality": 8, "coherence": 8, "task_fit": 8},
             "winner": "B", "reason": "Fine-tuned more focused."},
            {"question": "Q2", "reference": "R2", "base_output": "base text 2", "ft_output": "ft text 2",
             "score_a": {"quality": 8, "coherence": 8, "task_fit": 8},
             "score_b": {"quality": 4, "coherence": 4, "task_fit": 4},
             "winner": "A", "reason": "Real regression here."},
        ]
        summary = compute_summary(fake_judged)
        cross = cross_check_with_training_curve(FAKE_VAL_LOSSES, summary["avg_quality_margin"])
        st.markdown(generate_summary_report(summary, cross))
        with st.expander("Detailed report"):
            st.markdown(generate_detailed_report(fake_judged))


# ============================================================
# LOGS & REPRODUCIBILITY
# ============================================================
with tabs[6]:
    st.header("Logs & reproducibility — no API/GPU needed")

    st.subheader("common/llm_trace.py — every Claude call in the pipeline")
    if st.button("Load LLM call trace"):
        from evaluation.llm_trace import load_llm_trace
        trace = load_llm_trace()
        if not trace:
            st.info("No trace entries yet — run something that calls Claude (any tab above) first.")
        else:
            st.write(f"{len(trace)} call(s) logged.")
            st.dataframe([
                {"module": t["module"], "function": t["function"], "model": t.get("model"),
                 "success": t["success"], "latency_s": round(t.get("latency_s", 0), 2),
                 "error": t.get("error")}
                for t in trace
            ])
            with st.expander("Raw entries (full prompts + responses)"):
                st.json(trace)

    st.subheader("training/run_log.py — completed run outcomes")
    if st.button("Load finetune outcomes log"):
        from training.run_log import load_finetune_outcomes
        outcomes = load_finetune_outcomes()
        st.json(outcomes if outcomes else {"note": "No outcomes logged yet."})

    st.subheader("training/reproducibility.py — export/reload a run config")
    if st.button("Test export + reload round-trip"):
        from training.reproducibility import export_run_config, load_run_config
        exported = export_run_config(
            dataset_source={"source": "alpaca", "sample_seed": 100, "train_size": 40, "val_size": 10},
            curation_config={"threshold": 6.7, "mode": "normal"},
            training_config={"model_name": "Qwen/Qwen2.5-1.5B-Instruct", "method": "lora",
                             "lora_r": 8, "num_train_epochs": 6},
            training_result={"seed": 42, "final_loss": 1.56, "val_loss_history": FAKE_VAL_LOSSES},
            export_path="/tmp/dev_run_config.json",
        )
        reloaded = load_run_config("/tmp/dev_run_config.json")
        st.json(exported)
        st.success(f"Round-trip identical: {exported == reloaded}")