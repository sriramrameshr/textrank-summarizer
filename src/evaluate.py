"""ROUGE evaluation utilities (ROUGE-1, ROUGE-2, ROUGE-L)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

ROUGE_KEYS = ("rouge1", "rouge2", "rougeL")
METRIC_COLUMNS = [
    f"{key}_{stat}" for key in ROUGE_KEYS for stat in ("precision", "recall", "f1")
]
_SCORER = rouge_scorer.RougeScorer(list(ROUGE_KEYS), use_stemmer=True)


def score_pair(candidate: str, reference: str) -> dict[str, float]:
    """Score one candidate summary against one reference summary.

    Uses the standard ``rouge-score`` package with Porter stemming and returns
    precision, recall and F1 for ROUGE-1, ROUGE-2 and ROUGE-L.
    """
    result = {column: 0.0 for column in METRIC_COLUMNS}
    if not (candidate or "").strip() or not (reference or "").strip():
        return result

    scores = _SCORER.score(reference, candidate)
    for key in ROUGE_KEYS:
        score = scores[key]
        result[f"{key}_precision"] = float(score.precision)
        result[f"{key}_recall"] = float(score.recall)
        result[f"{key}_f1"] = float(score.fmeasure)
    return result


def evaluate_predictions(predictions: list[dict]) -> pd.DataFrame:
    """Build a per-document ROUGE dataframe from prediction records.

    Each record must contain: ``doc_id``, ``method``, ``k``, ``summary``,
    ``reference``, ``runtime_ms``, ``source_sentences`` and ``source_words``.
    """
    rows = []
    for record in predictions:
        metrics = score_pair(record["summary"], record["reference"])
        rows.append(
            {
                "doc_id": record["doc_id"],
                "method": record["method"],
                "k": int(record["k"]),
                "summary_sentences": len(record.get("selected_indices", [])),
                "summary_words": len(record["summary"].split()),
                "source_sentences": int(record.get("source_sentences", 0)),
                "source_words": int(record.get("source_words", 0)),
                "reference_words": len(record["reference"].split()),
                "compression_ratio": round(
                    len(record["summary"].split()) / max(1, int(record.get("source_words", 1))), 4
                ),
                "runtime_ms": round(float(record.get("runtime_ms", 0.0)), 3),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def aggregate_scores(per_document: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
    """Mean metric values per method (optionally restricted to one summary length)."""
    frame = per_document if k is None else per_document[per_document["k"] == k]
    numeric_columns = METRIC_COLUMNS + [
        "runtime_ms",
        "summary_sentences",
        "summary_words",
        "compression_ratio",
    ]
    grouped = frame.groupby("method", sort=False)
    summary = grouped[numeric_columns].mean().round(4)
    summary.insert(0, "n_documents", grouped.size())
    return summary.reset_index()


def bootstrap_ci(
    values: np.ndarray | list[float],
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lower = float(np.quantile(means, (1 - confidence) / 2))
    upper = float(np.quantile(means, 1 - (1 - confidence) / 2))
    return lower, upper


def paired_comparison(
    per_document: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str = "rouge1_f1",
    k: int = 3,
) -> dict:
    """Compare two methods on the same documents (difference, CI and Wilcoxon p)."""
    frame = per_document[per_document["k"] == k]
    pivot = frame.pivot_table(index="doc_id", columns="method", values=metric)
    if method_a not in pivot or method_b not in pivot:
        raise KeyError(f"Methods {method_a!r} and {method_b!r} must both be present at k={k}")

    paired = pivot[[method_a, method_b]].dropna()
    differences = (paired[method_a] - paired[method_b]).to_numpy(dtype=float)
    low, high = bootstrap_ci(differences)
    result = {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "k": k,
        "n_pairs": int(differences.size),
        "mean_a": float(paired[method_a].mean()),
        "mean_b": float(paired[method_b].mean()),
        "mean_difference": float(differences.mean()),
        "ci_low": low,
        "ci_high": high,
        "a_wins": int((differences > 0).sum()),
        "b_wins": int((differences < 0).sum()),
        "ties": int((differences == 0).sum()),
        "p_value": float("nan"),
    }

    try:
        from scipy.stats import wilcoxon

        if differences.size >= 10 and np.any(differences != 0):
            result["p_value"] = float(
                wilcoxon(differences, alternative="two-sided", zero_method="wilcox").pvalue
            )
    except Exception:  # pragma: no cover - scipy is optional for the write-up
        pass
    return result


def format_mean_ci(values: pd.Series, digits: int = 2) -> str:
    """Format a metric column as ``mean (95% CI low-high)`` in percent."""
    mean = values.mean() * 100
    low, high = bootstrap_ci(values.to_numpy(dtype=float))
    return f"{mean:.{digits}f} [{low * 100:.{digits}f}-{high * 100:.{digits}f}]"


def results_table(
    per_document: pd.DataFrame,
    k: int = 3,
    methods: list[str] | None = None,
    digits: int = 2,
) -> pd.DataFrame:
    """Publication-style table: mean F1 with bootstrap CI per method."""
    frame = per_document[per_document["k"] == k]
    if methods:
        frame = frame[frame["method"].isin(methods)]
    rows = []
    for method, group in frame.groupby("method", sort=False):
        row = {"Method": method, "Documents": len(group)}
        for key, label in (("rouge1_f1", "ROUGE-1"), ("rouge2_f1", "ROUGE-2"), ("rougeL_f1", "ROUGE-L")):
            mean = group[key].mean() * 100
            low, high = bootstrap_ci(group[key].to_numpy(dtype=float))
            row[label] = f"{mean:.{digits}f} [{low * 100:.{digits}f}-{high * 100:.{digits}f}]"
        row["Avg. words"] = f"{group['summary_words'].mean():.0f}"
        row["Compression"] = f"{group['compression_ratio'].mean() * 100:.1f}%"
        rows.append(row)
    return pd.DataFrame(rows)
