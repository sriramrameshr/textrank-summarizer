"""Generate the pipeline diagram used in the report and README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures" / "pipeline.png"
NAVY = "#16324f"
BLUE = "#1f4e79"
LIGHT = "#eaf2f9"
GREEN = "#e9f5ee"
ORANGE = "#fbeee3"


def box(ax, x, y, w, h, text, facecolor=LIGHT, edgecolor=BLUE, fontsize=9, bold=False):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.25,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=NAVY,
        fontweight="bold" if bold else "normal",
        zorder=3,
        linespacing=1.35,
    )


def arrow(ax, start, end, color=BLUE, style="-|>", lw=1.4, linestyle="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=13,
            linewidth=lw,
            color=color,
            linestyle=linestyle,
            zorder=1,
            shrinkA=2,
            shrinkB=2,
        )
    )


def build() -> Path:
    fig, ax = plt.subplots(figsize=(9.6, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.975,
        "TextRank extractive summarisation pipeline",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
    )

    pipeline = [
        ("Input news article\n(raw text)", LIGHT),
        ("Sentence segmentation\nNLTK Punkt  •  deduplication", LIGHT),
        ("Tokenisation\nlowercase  •  stopwords  •  Porter stemming", LIGHT),
        ("TF-IDF sentence vectors\nsublinear tf  •  L2 normalised", LIGHT),
        ("Pairwise cosine similarity\nS × S similarity matrix", LIGHT),
        ("Sentence graph\nnodes = sentences, edges = similarity", LIGHT),
        ("PageRank\npower iteration, damping d = 0.85", LIGHT),
        ("Top-k selection\n+ chronological re-ordering", GREEN),
        ("Extractive summary\n(k sentences, original wording)", GREEN),
    ]

    x = 0.30
    top, bottom = 0.905, 0.075
    step = (top - bottom) / (len(pipeline) - 1)
    positions = []
    for i, (label, color) in enumerate(pipeline):
        y = top - i * step
        height = 0.062 if "\n" not in label else 0.074
        box(ax, x, y, 0.46, height, label, facecolor=color, fontsize=8.6)
        positions.append((x, y, height))

    for i in range(len(positions) - 1):
        _, y1, h1 = positions[i]
        _, y2, h2 = positions[i + 1]
        arrow(ax, (x, y1 - h1 / 2 - 0.004), (x, y2 + h2 / 2 + 0.004))

    # ---------------- Evaluation branch ----------------
    eval_x = 0.79
    box(ax, eval_x, 0.79, 0.34, 0.08, "Human reference summary\n(CNN/DailyMail highlights)", facecolor=ORANGE, edgecolor="#c55a11", fontsize=8.6)
    box(ax, eval_x, 0.60, 0.34, 0.10, "ROUGE evaluation\nROUGE-1  •  ROUGE-2  •  ROUGE-L", facecolor=ORANGE, edgecolor="#c55a11", fontsize=8.8, bold=True)
    box(ax, eval_x, 0.41, 0.34, 0.085, "Precision  •  Recall  •  F1\n95% bootstrap confidence intervals", facecolor="#fdf6ec", edgecolor="#c55a11", fontsize=8.2)
    box(ax, eval_x, 0.20, 0.34, 0.085, "Baselines\nLead-3  •  Random-3", facecolor="#f1f1f1", edgecolor="#7f7f7f", fontsize=8.6)

    arrow(ax, (eval_x, 0.748), (eval_x, 0.652), color="#c55a11")
    arrow(ax, (eval_x, 0.548), (eval_x, 0.455), color="#c55a11")
    arrow(ax, (eval_x, 0.158), (eval_x, 0.545), color="#7f7f7f", linestyle="--")
    arrow(ax, (eval_x, 0.365), (eval_x, 0.49), color="#c55a11", style="-")

    # Summary -> ROUGE
    summary_y = positions[-1][1]
    arrow(ax, (x + 0.235, summary_y), (eval_x - 0.175, 0.60), color="#2e8b57", lw=1.8)
    ax.text(0.55, 0.315, "generated\nsummary", fontsize=7.8, color="#2e8b57", ha="center", va="center", style="italic")

    ax.text(
        0.02,
        0.02,
        "Unsupervised • no training data • deterministic with a fixed random seed",
        fontsize=7.8,
        color="#5b6b7b",
        style="italic",
    )

    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}")
    return OUT


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    build()
