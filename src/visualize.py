"""Matplotlib visualisations for the notebook, report and Streamlit app."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

PALETTE = {
    "TextRank": "#1f4e79",
    "TextRank + MMR": "#2e8b57",
    "Lead-3": "#c55a11",
    "Random-3": "#7f7f7f",
    "Lead-k": "#c55a11",
    "Random-k": "#7f7f7f",
}
METRICS = [("rouge1_f1", "ROUGE-1"), ("rouge2_f1", "ROUGE-2"), ("rougeL_f1", "ROUGE-L")]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "figure.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, path: Path | str | None) -> plt.Figure:
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor="white")
    return fig


def plot_method_comparison(
    summary: pd.DataFrame,
    path: Path | str | None = None,
    title: str = "ROUGE F1 by summarisation method (k = 3)",
) -> plt.Figure:
    """Grouped bar chart comparing methods on ROUGE-1/2/L F1."""
    _style()
    methods = list(summary["method"])
    x = np.arange(len(METRICS))
    width = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, method in enumerate(methods):
        row = summary[summary["method"] == method].iloc[0]
        values = [row[column] * 100 for column, _ in METRICS]
        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=method,
            color=PALETTE.get(method, f"C{i}"),
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.1f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#333333",
            )

    ax.set_xticks(x, [label for _, label in METRICS])
    ax.set_ylabel("F1 score (%)")
    ax.set_title(title)
    ax.set_ylim(0, max(0.1, summary[[c for c, _ in METRICS]].to_numpy().max() * 100 * 1.18))
    ax.legend(ncols=len(methods), loc="upper right")
    return _save(fig, path)


def plot_dataset_overview(stats: pd.DataFrame, path: Path | str | None = None) -> plt.Figure:
    """Two-panel overview of article and reference-summary length distributions."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))

    axes[0].hist(stats["source_sentences"], bins=20, color=PALETTE["TextRank"], alpha=0.85, edgecolor="white")
    axes[0].axvline(
        stats["source_sentences"].mean(),
        color=PALETTE["Lead-3"],
        linestyle="--",
        linewidth=1.6,
        label=f"mean = {stats['source_sentences'].mean():.1f}",
    )
    axes[0].set_xlabel("Sentences per article")
    axes[0].set_ylabel("Number of articles")
    axes[0].set_title("Source article length")
    axes[0].legend()

    axes[1].hist(
        stats["reference_words"], bins=20, color=PALETTE["TextRank + MMR"], alpha=0.85, edgecolor="white"
    )
    axes[1].axvline(
        stats["reference_words"].mean(),
        color=PALETTE["Lead-3"],
        linestyle="--",
        linewidth=1.6,
        label=f"mean = {stats['reference_words'].mean():.1f}",
    )
    axes[1].set_xlabel("Words per reference summary")
    axes[1].set_ylabel("Number of articles")
    axes[1].set_title("Human reference summary length")
    axes[1].legend()

    fig.suptitle("CNN/DailyMail evaluation slice — descriptive statistics", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_metric_by_length(
    sweep: pd.DataFrame,
    path: Path | str | None = None,
    metric: str = "rouge1_f1",
    ylabel: str = "ROUGE-1 F1 (%)",
    title: str = "Effect of summary length on ROUGE-1 F1",
) -> plt.Figure:
    """Line chart (mean +/- 95% CI) of a metric as the number of sentences varies."""
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for method, group in sweep.groupby("method", sort=False):
        group = group.sort_values("k")
        mean = group[metric] * 100
        ci = 1.96 * group[metric].sem() * 100
        color = PALETTE.get(method, None)
        ax.plot(group["k"], mean, marker="o", label=method, color=color, linewidth=2)
        ax.fill_between(group["k"], mean - ci, mean + ci, color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("Number of extracted sentences (k)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted(sweep["k"].unique()))
    ax.legend()
    return _save(fig, path)


def plot_sentence_scores(
    sentences: list[str],
    scores: np.ndarray,
    selected: list[int],
    path: Path | str | None = None,
    top_n: int = 14,
    title: str = "TextRank PageRank score per sentence",
) -> plt.Figure:
    """Horizontal bars for the strongest sentences, highlighting the extract."""
    _style()
    order = np.argsort(scores)[::-1][:top_n][::-1]
    labels = [f"S{i + 1}" for i in order]
    values = [float(scores[i]) * 100 for i in order]
    colors = [PALETTE["TextRank"] if i in selected else "#c9d6e3" for i in order]

    fig, ax = plt.subplots(figsize=(7.4, 0.34 * len(order) + 1.5))
    ax.barh(labels, values, color=colors, edgecolor="white")
    for y, (index, value) in enumerate(zip(order, values)):
        ax.text(value, y, f"  {value:.2f}", va="center", fontsize=7.5, color="#333333")
    ax.set_xlabel("Normalised PageRank score (x100)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(values) * 1.18 if values else 1)

    if selected:
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=PALETTE["TextRank"]),
            plt.Rectangle((0, 0), 1, 1, color="#c9d6e3"),
        ]
        ax.legend(handles, ["Selected in summary", "Not selected"], loc="lower right", fontsize=8)
    return _save(fig, path)


def plot_similarity_heatmap(
    similarity: np.ndarray,
    path: Path | str | None = None,
    max_sentences: int = 40,
    title: str = "Sentence-pair cosine similarity matrix",
) -> plt.Figure:
    """Heatmap of the TF-IDF cosine similarity matrix (subsampled if large)."""
    _style()
    matrix = similarity[:max_sentences, :max_sentences]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(0.01, float(matrix.max())))
    ax.set_xlabel("Sentence index")
    ax.set_ylabel("Sentence index")
    ax.set_title(title)
    ax.grid(False)
    ticks = list(range(0, matrix.shape[0], max(1, matrix.shape[0] // 8)))
    ax.set_xticks(ticks, [str(t + 1) for t in ticks])
    ax.set_yticks(ticks, [str(t + 1) for t in ticks])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="cosine similarity")
    return _save(fig, path)


def plot_graph(
    graph: nx.Graph,
    scores: np.ndarray,
    selected: list[int],
    path: Path | str | None = None,
    max_nodes: int = 45,
    title: str = "TextRank sentence graph (node size and colour = PageRank score)",
) -> plt.Figure:
    """Spring-layout visualisation of the sentence similarity graph."""
    _style()
    if graph.number_of_nodes() > max_nodes:
        keep = sorted(np.argsort(scores)[::-1][:max_nodes])
        graph = graph.subgraph(keep).copy()
        mapping = {old: new for new, old in enumerate(keep)}
        graph = nx.relabel_nodes(graph, mapping)
        scores = np.array([scores[old] for old in keep])
        selected = [mapping[i] for i in selected if i in mapping]

    position = nx.spring_layout(graph, seed=42, k=0.55, iterations=120, weight="weight")
    node_scores = np.asarray(scores, dtype=float)
    norm = Normalize(vmin=float(node_scores.min()), vmax=float(node_scores.max() + 1e-12))
    colors = [plt.cm.YlGnBu(0.25 + 0.7 * norm(float(s))) for s in node_scores]
    sizes = 120 + 1500 * (node_scores - node_scores.min()) / (np.ptp(node_scores) + 1e-12)

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    widths = [0.2 + 2.2 * graph[u][v].get("weight", 0) for u, v in graph.edges()]
    nx.draw_networkx_edges(graph, position, ax=ax, width=widths, edge_color="#9fb3c8", alpha=0.65)
    nx.draw_networkx_nodes(
        graph, position, ax=ax, node_color=colors, node_size=sizes, edgecolors="#1f4e79", linewidths=0.7
    )

    important = set(np.argsort(node_scores)[::-1][:6]) | set(selected)
    labels = {i: f"S{i + 1}" for i in important}
    nx.draw_networkx_labels(graph, position, labels=labels, font_size=7, font_color="#12263a", ax=ax)

    ax.set_title(title)
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap="YlGnBu", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, label="PageRank score")
    return _save(fig, path)


def plot_delta_histogram(
    per_document: pd.DataFrame,
    path: Path | str | None = None,
    metric: str = "rouge1_f1",
    k: int = 3,
) -> plt.Figure:
    """Distribution of per-document TextRank minus Lead-3 ROUGE-1 F1."""
    _style()
    frame = per_document[per_document["k"] == k]
    pivot = frame.pivot_table(index="doc_id", columns="method", values=metric)
    baseline = "Lead-k" if "Lead-k" in pivot.columns else f"Lead-{k}"
    delta = (pivot["TextRank"] - pivot[baseline]).dropna() * 100

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.hist(delta, bins=24, color=PALETTE["TextRank"], alpha=0.85, edgecolor="white")
    ax.axvline(0, color="#333333", linewidth=1.2, linestyle="--", label="No difference")
    ax.axvline(delta.mean(), color="#c55a11", linewidth=1.8, label=f"Mean = {delta.mean():+.2f}")
    ax.set_xlabel(f"ROUGE-1 F1 difference (TextRank - {baseline.replace('Lead-k', f'Lead-{k}')}), percentage points")
    ax.set_ylabel("Number of documents")
    ax.set_title(f"Per-document comparison against the Lead-{k} baseline")
    ax.legend()
    return _save(fig, path)


def plot_ablation(
    summary: pd.DataFrame,
    path: Path | str | None = None,
    title: str = "Effect of redundancy filtering (MMR-style) on ROUGE F1",
) -> plt.Figure:
    """Grouped bars comparing plain TextRank with the redundancy-filtered variant."""
    _style()
    methods = list(summary["method"])
    x = np.arange(len(METRICS))
    width = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, method in enumerate(methods):
        row = summary[summary["method"] == method].iloc[0]
        values = [row[column] * 100 for column, _ in METRICS]
        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=method, color=PALETTE.get(method, f"C{i}"))
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.1f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x, [label for _, label in METRICS])
    ax.set_ylabel("F1 score (%)")
    ax.set_title(title)
    ax.set_ylim(0, max(0.1, summary[[c for c, _ in METRICS]].to_numpy().max() * 100 * 1.2))
    ax.legend()
    return _save(fig, path)
