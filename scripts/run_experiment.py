"""Run the full TextRank evaluation on the cached CNN/DailyMail slice.

Produces:
    results/per_document.csv        one row per (document, method, k)
    results/metrics_by_method_k.csv mean ROUGE per method and length
    results/results_table_k3.csv    publication-style table
    results/paired_tests.json       paired comparisons with bootstrap CI + Wilcoxon
    results/run_metadata.json       dataset stats, config, timings, package versions
    results/examples.json           qualitative best/worst/median examples
    results/figures/*.png           all report figures
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from src import evaluate, visualize  # noqa: E402
from src.baselines import lead_k, random_k  # noqa: E402
from src.dataset import get_documents, load_sample  # noqa: E402
from src.preprocess import count_words, preprocess, split_sentences  # noqa: E402
from src.textrank import summarize  # noqa: E402

METHODS = ["TextRank", "Lead-k", "Random-k", "TextRank + MMR"]
PRIMARY_K = 3
SUMMARY_LENGTHS = [2, 3, 4, 5, 6, 8]
REDUNDANCY_THRESHOLD = 0.30
RANDOM_SEED = 42
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "nltk",
    "networkx",
    "rouge-score",
    "matplotlib",
    "streamlit",
    "scipy",
]


def _decorate(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename generic baseline methods to their concrete k for display."""
    frame = frame.copy()
    frame["method"] = frame["method"].replace(
        {"Lead-k": f"Lead-{PRIMARY_K}", "Random-k": f"Random-{PRIMARY_K}"}
    )
    return frame


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def decorate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return _decorate(summary)


def run(limit: int | None = None, lengths: list[int] | None = None) -> dict:
    lengths = lengths or SUMMARY_LENGTHS
    documents = get_documents()
    if limit:
        documents = documents[:limit]

    predictions: list[dict] = []
    dataset_stats: list[dict] = []
    examples: list[dict] = []
    started = time.perf_counter()

    for number, document in enumerate(documents, start=1):
        article = document["article"]
        reference = document["reference_summary"]
        doc = preprocess(article)
        source_sentences = doc.num_sentences
        source_words = count_words(article)
        reference_sentences = len(split_sentences(reference, min_tokens=1))
        dataset_stats.append(
            {
                "doc_id": document["id"],
                "source_sentences": source_sentences,
                "source_words": source_words,
                "reference_sentences": reference_sentences,
                "reference_words": count_words(reference),
            }
        )

        for k in lengths:
            # ---- TextRank (default configuration) -------------------------
            start = time.perf_counter()
            tr = summarize(article, k=k, alpha=0.85, doc=doc)
            tr_ms = (time.perf_counter() - start) * 1000
            predictions.append(
                {
                    "doc_id": document["id"],
                    "method": "TextRank",
                    "k": k,
                    "summary": tr.summary,
                    "reference": reference,
                    "runtime_ms": tr_ms,
                    "source_sentences": source_sentences,
                    "source_words": source_words,
                    "selected_indices": tr.selected_indices,
                }
            )

            # ---- TextRank + redundancy filter (ablation) ------------------
            start = time.perf_counter()
            tr_mmr = summarize(
                article,
                k=k,
                alpha=0.85,
                redundancy_threshold=REDUNDANCY_THRESHOLD,
                doc=doc,
            )
            mmr_ms = (time.perf_counter() - start) * 1000
            predictions.append(
                {
                    "doc_id": document["id"],
                    "method": "TextRank + MMR",
                    "k": k,
                    "summary": tr_mmr.summary,
                    "reference": reference,
                    "runtime_ms": mmr_ms,
                    "source_sentences": source_sentences,
                    "source_words": source_words,
                    "selected_indices": tr_mmr.selected_indices,
                }
            )

            # ---- Lead-k baseline ------------------------------------------
            start = time.perf_counter()
            lead = lead_k(article, k=k, doc=doc)
            lead_ms = (time.perf_counter() - start) * 1000
            predictions.append(
                {
                    "doc_id": document["id"],
                    "method": "Lead-k",
                    "k": k,
                    "summary": lead["summary"],
                    "reference": reference,
                    "runtime_ms": lead_ms,
                    "source_sentences": source_sentences,
                    "source_words": source_words,
                    "selected_indices": lead["selected_indices"],
                }
            )

            # ---- Random-k baseline ----------------------------------------
            start = time.perf_counter()
            rand = random_k(article, k=k, seed=RANDOM_SEED, doc=doc)
            rand_ms = (time.perf_counter() - start) * 1000
            predictions.append(
                {
                    "doc_id": document["id"],
                    "method": "Random-k",
                    "k": k,
                    "summary": rand["summary"],
                    "reference": reference,
                    "runtime_ms": rand_ms,
                    "source_sentences": source_sentences,
                    "source_words": source_words,
                    "selected_indices": rand["selected_indices"],
                }
            )

            # Keep one qualitative record per document at the primary length.
            if k == PRIMARY_K:
                examples.append(
                    {
                        "doc_id": document["id"],
                        "article": article,
                        "reference": reference,
                        "textrank": tr.summary,
                        "lead": lead["summary"],
                        "mmr": tr_mmr.summary,
                        "textrank_sentences": [tr.sentences[i] for i in tr.selected_indices],
                        "selected_indices": tr.selected_indices,
                        "source_sentences": source_sentences,
                    }
                )

        if number % 25 == 0 or number == len(documents):
            print(f"processed {number}/{len(documents)} documents", flush=True)

    elapsed = time.perf_counter() - started
    per_document = evaluate.evaluate_predictions(predictions)
    per_document.to_csv(RESULTS / "per_document.csv", index=False)

    metrics_by_method_k = (
        per_document.groupby(["method", "k"], sort=False)[evaluate.METRIC_COLUMNS + ["runtime_ms"]]
        .mean()
        .round(4)
        .reset_index()
    )
    metrics_by_method_k.to_csv(RESULTS / "metrics_by_method_k.csv", index=False)

    summary_k3 = evaluate.aggregate_scores(per_document, k=PRIMARY_K)
    table_k3 = evaluate.results_table(per_document, k=PRIMARY_K, methods=METHODS)
    table_k3.to_csv(RESULTS / "results_table_k3.csv", index=False)
    _decorate(summary_k3).to_csv(RESULTS / "metrics_by_method_k3.csv", index=False)

    # ---- Paired statistical comparisons (primary length) ------------------
    comparisons = []
    for other in ("Lead-k", "Random-k", "TextRank + MMR"):
        for metric in ("rouge1_f1", "rouge2_f1", "rougeL_f1"):
            comparisons.append(
                evaluate.paired_comparison(per_document, "TextRank", other, metric=metric, k=PRIMARY_K)
            )
    (RESULTS / "paired_tests.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")

    # ---- Statistics for the report ---------------------------------------
    stats = pd.DataFrame(dataset_stats)
    dataset_summary = {
        "documents": int(len(documents)),
        "source_sentences_mean": round(float(stats["source_sentences"].mean()), 2),
        "source_sentences_median": float(stats["source_sentences"].median()),
        "source_sentences_min": int(stats["source_sentences"].min()),
        "source_sentences_max": int(stats["source_sentences"].max()),
        "source_words_mean": round(float(stats["source_words"].mean()), 2),
        "reference_words_mean": round(float(stats["reference_words"].mean()), 2),
        "reference_sentences_mean": round(float(stats["reference_sentences"].mean()), 2),
    }
    dataset_summary.update(
        {
            "name": "CNN/DailyMail",
            "config": "3.0.0",
            "split": "test",
            "note": "First N rows of the test split, downloaded via the HuggingFace datasets-server.",
        }
    )

    # ---- Figures ----------------------------------------------------------
    FIGURES.mkdir(parents=True, exist_ok=True)
    visualize.plot_method_comparison(
        _decorate(summary_k3), FIGURES / "rouge_by_method.png"
    )
    visualize.plot_metric_by_length(metrics_by_method_k, FIGURES / "rouge_by_length.png")
    visualize.plot_delta_histogram(per_document, FIGURES / "delta_vs_lead3.png")
    visualize.plot_ablation(
        metrics_by_method_k[
            (metrics_by_method_k["k"] == PRIMARY_K)
            & (metrics_by_method_k["method"].isin(["TextRank", "TextRank + MMR"]))
        ],
        FIGURES / "ablation_mmr.png",
    )

    # Example-specific figures (graph, scores, similarity and a comparison table).
    sample = documents[0]
    sample_doc = preprocess(sample["article"])
    sample_result = summarize(sample["article"], k=PRIMARY_K, doc=sample_doc)
    visualize.plot_graph(
        sample_result.graph,
        sample_result.scores,
        sample_result.selected_indices,
        FIGURES / "sentence_graph.png",
    )
    visualize.plot_sentence_scores(
        sample_result.sentences,
        sample_result.scores,
        sample_result.selected_indices,
        FIGURES / "sentence_scores.png",
    )
    visualize.plot_similarity_heatmap(sample_result.similarity, FIGURES / "similarity_heatmap.png")

    # ---- Qualitative examples (best / median / worst vs Lead-3) -----------
    primary = per_document[per_document["k"] == PRIMARY_K]
    pivot = primary.pivot_table(index="doc_id", columns="method", values="rouge1_f1")
    pivot["delta"] = pivot["TextRank"] - pivot["Lead-k"]
    by_id = {example["doc_id"]: example for example in examples}
    selection = {}
    for label, index in (
        ("best_vs_lead", pivot["delta"].idxmax()),
        ("median", pivot["delta"].sort_values().index[len(pivot) // 2]),
        ("worst_vs_lead", pivot["delta"].idxmin()),
    ):
        example = dict(by_id[index])
        scores = primary[
            (primary["doc_id"] == index) & (primary["method"].isin(["TextRank", "Lead-k"]))
        ]
        example["rouge"] = scores.set_index("method")[
            ["rouge1_f1", "rouge2_f1", "rougeL_f1"]
        ].round(4).to_dict(orient="index")
        selection[label] = example
    (RESULTS / "examples.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    # ---- Run metadata -----------------------------------------------------
    metadata_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset": dataset_summary,
        "configuration": {
            "damping_factor_alpha": 0.85,
            "max_iterations": 200,
            "tolerance": 1e-8,
            "similarity": "TF-IDF cosine (sublinear tf, L2 normalised)",
            "summary_lengths": lengths,
            "primary_length": PRIMARY_K,
            "redundancy_threshold": REDUNDANCY_THRESHOLD,
            "random_seed": RANDOM_SEED,
            "rouge_implementation": "rouge-score 0.1.x, use_stemmer=True",
        },
        "evaluation": {
            "predictions": len(predictions),
            "documents": len(documents),
            "total_runtime_seconds": round(elapsed, 2),
            "mean_textrank_runtime_ms": round(
                float(per_document[per_document["method"] == "TextRank"]["runtime_ms"].mean()), 3
            ),
        },
        "packages": {package: _version(package) for package in PACKAGES},
        "paired_tests": comparisons,
        "results_table_k3": table_k3.to_dict(orient="records"),
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    print("\n=== ROUGE F1 at k = 3 (mean [95% bootstrap CI]) ===")
    print(table_k3.to_string(index=False))
    print(f"\nSaved {len(predictions)} predictions in {elapsed:.1f}s -> {RESULTS}")
    return {
        "per_document": per_document,
        "metrics_by_method_k": metrics_by_method_k,
        "summary_k3": summary_k3,
        "table_k3": table_k3,
        "comparisons": comparisons,
        "dataset_summary": dataset_summary,
        "metadata": metadata_payload,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the TextRank + ROUGE experiment.")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N documents")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=SUMMARY_LENGTHS,
        help="summary lengths (number of sentences) to evaluate",
    )
    args = parser.parse_args()
    run(limit=args.limit, lengths=args.lengths)
