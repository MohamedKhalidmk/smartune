"""
Quick standalone test for curation/curator.py.

Run this directly, without Streamlit, to confirm the module works
on its own before wiring it into the dashboard.

Usage:
    export ANTHROPIC_API_KEY=your-key-here
    python -m curation.test_curator
"""

from curation.curator import classify_dataset, curate_dataset


SAMPLE_EXAMPLES = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris.",
    },
    {
        "question": (
            "Generate two similar sounding but semantically different "
            "words to contrast this word. Light"
        ),
        "answer": "Bright and Dim.",
    },  # Known task-mismatch case from development.
]


def main() -> None:
    scored = curate_dataset(SAMPLE_EXAMPLES)

    for example in scored:
        print(example["question"])
        print(example["_curation"])
        print("---")

    classification = classify_dataset(
        scored,
        threshold=6.7,
    )

    print(
        f"Kept: {len(classification['kept'])}, "
        f"Rejected: {len(classification['rejected'])}, "
        f"Failed: {len(classification['failed'])}"
    )


if __name__ == "__main__":
    main()
