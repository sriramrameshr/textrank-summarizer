"""Generate the term-project report (Markdown -> HTML -> PDF).

    python scripts/build_report.py

The Markdown source is written to report/report.md so it can be edited and
re-rendered. The PDF is produced by printing the styled HTML with headless
Microsoft Edge, so no LaTeX installation is required.
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
FIGURES = ROOT / "results" / "figures"
DELIVERABLES = ROOT / "deliverables"
EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

STUDENT = json.loads((REPORT_DIR / "student.json").read_text(encoding="utf-8"))
GUIDE = STUDENT.get("guide", "").strip()
GUIDE_ROW = (
    f'<tr><td>Project guide</td><td>{html.escape(GUIDE)}</td></tr>' if GUIDE else ""
)
META = json.loads((ROOT / "results" / "run_metadata.json").read_text(encoding="utf-8"))
PER_DOCUMENT = pd.read_csv(ROOT / "results" / "per_document.csv")
TABLE_MAIN = pd.read_csv(ROOT / "results" / "results_table_k3.csv")
COMPARISONS = pd.DataFrame(json.loads((ROOT / "results" / "paired_tests.json").read_text(encoding="utf-8")))
EXAMPLES = json.loads((ROOT / "results" / "examples.json").read_text(encoding="utf-8"))

DATASET = META["dataset"]
CONFIG = META["configuration"]
PACKAGES = META["packages"]
METHOD_ORDER = ["TextRank", "TextRank + MMR", "Lead-k", "Random-k"]
DISPLAY = {"TextRank": "TextRank", "TextRank + MMR": "TextRank + MMR", "Lead-k": "Lead-3", "Random-k": "Random-3"}

K3 = PER_DOCUMENT[PER_DOCUMENT["k"] == CONFIG["primary_length"]]
AGG = K3.groupby("method", sort=False)[
    [
        "rouge1_precision",
        "rouge1_recall",
        "rouge1_f1",
        "rouge2_precision",
        "rouge2_recall",
        "rouge2_f1",
        "rougeL_precision",
        "rougeL_recall",
        "rougeL_f1",
        "summary_words",
        "compression_ratio",
        "runtime_ms",
    ]
].mean()

TR = AGG.loc["TextRank"]
LEAD = AGG.loc["Lead-k"]
RANDOM = AGG.loc["Random-k"]
MMR = AGG.loc["TextRank + MMR"]

SWEEP = PER_DOCUMENT.groupby(["method", "k"])["rouge1_f1"].mean().unstack("method").sort_index()
SWEEP_L = PER_DOCUMENT.groupby(["method", "k"])["rougeL_f1"].mean().unstack("method").sort_index()


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}"


def comparison(method_b: str, metric: str = "rouge1_f1") -> dict:
    rows = COMPARISONS[
        (COMPARISONS["method_a"] == "TextRank")
        & (COMPARISONS["method_b"] == method_b)
        & (COMPARISONS["metric"] == metric)
    ]
    return rows.iloc[0].to_dict()


CMP_LEAD = comparison("Lead-k")
CMP_RANDOM = comparison("Random-k")
CMP_MMR = comparison("TextRank + MMR")


def figure(filename: str, caption: str, width: str = "100%") -> str:
    return (
        f'<figure><img src="../results/figures/{filename}" style="width:{width}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def equation(filename: str, alt: str, width: str = "auto") -> str:
    return f'<p class="eq"><img src="../results/figures/{filename}" alt="{alt}" style="width:{width}"></p>'


def table(frame: pd.DataFrame, caption: str) -> str:
    return f"**{caption}**\n\n" + frame.to_markdown(index=False) + "\n"


def method_table() -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        source = TABLE_MAIN[TABLE_MAIN["Method"] == method].iloc[0]
        rows.append(
            {
                "Method": DISPLAY[method],
                "ROUGE-1 F1": source["ROUGE-1"],
                "ROUGE-2 F1": source["ROUGE-2"],
                "ROUGE-L F1": source["ROUGE-L"],
                "Avg. words": source["Avg. words"],
                "Compression": source["Compression"],
            }
        )
    return pd.DataFrame(rows)


def detail_table() -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        row = AGG.loc[method]
        rows.append(
            {
                "Method": DISPLAY[method],
                "R-1 P": pct(row["rouge1_precision"]),
                "R-1 R": pct(row["rouge1_recall"]),
                "R-1 F1": pct(row["rouge1_f1"]),
                "R-2 P": pct(row["rouge2_precision"]),
                "R-2 R": pct(row["rouge2_recall"]),
                "R-2 F1": pct(row["rouge2_f1"]),
                "R-L P": pct(row["rougeL_precision"]),
                "R-L R": pct(row["rougeL_recall"]),
                "R-L F1": pct(row["rougeL_f1"]),
            }
        )
    return pd.DataFrame(rows)


def significance_table() -> pd.DataFrame:
    rows = []
    for method in ["Lead-k", "Random-k", "TextRank + MMR"]:
        record = comparison(method)
        rows.append(
            {
                "Comparison": f"TextRank vs {DISPLAY[method]}",
                "Mean difference (pp)": f"{record['mean_difference'] * 100:+.2f}",
                "95% CI (pp)": f"[{record['ci_low'] * 100:+.2f}, {record['ci_high'] * 100:+.2f}]",
                "TextRank wins": f"{record['a_wins']}/{record['n_pairs']}",
                "Wilcoxon p": f"{record['p_value']:.4f}",
            }
        )
    return pd.DataFrame(rows)


def length_table() -> pd.DataFrame:
    frame = (SWEEP * 100).round(2)
    frame.index.name = "k"
    frame = frame.reset_index()
    frame.columns = ["k"] + [DISPLAY.get(c, c) for c in frame.columns[1:]]
    return frame[["k", "TextRank", "TextRank + MMR", "Lead-3", "Random-3"]]


def ablation_table() -> pd.DataFrame:
    rows = []
    for k in sorted(SWEEP.index):
        rows.append(
            {
                "k": k,
                "TextRank R-1": pct(SWEEP.loc[k, "TextRank"]),
                "TextRank + MMR R-1": pct(SWEEP.loc[k, "TextRank + MMR"]),
                "Delta (pp)": f"{(SWEEP.loc[k, 'TextRank + MMR'] - SWEEP.loc[k, 'TextRank']) * 100:+.2f}",
            }
        )
    return pd.DataFrame(rows)


def dataset_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Corpus", "CNN/DailyMail (abisee/cnn_dailymail), version 3.0.0"],
            ["Split", f"test, first {DATASET['documents']} articles"],
            ["Reference summaries", "Human-written highlights (multi-sentence)"],
            ["Articles evaluated", str(DATASET["documents"])],
            ["Mean sentences per article", f"{DATASET['source_sentences_mean']:.1f}"],
            ["Median sentences per article", f"{DATASET['source_sentences_median']:.0f}"],
            ["Mean words per article", f"{DATASET['source_words_mean']:.0f}"],
            ["Mean words per reference", f"{DATASET['reference_words_mean']:.1f}"],
            ["Mean sentences per reference", f"{DATASET['reference_sentences_mean']:.2f}"],
            ["Cache file", "data/cnn_dailymail_sample.json"],
        ],
        columns=["Property", "Value"],
    )


def config_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Summary length (primary)", f"k = {CONFIG['primary_length']} sentences"],
            ["Length sweep", ", ".join(str(k) for k in CONFIG["summary_lengths"])],
            ["Similarity", "TF-IDF cosine (sublinear tf, L2 normalised)"],
            ["Stopwords", "scikit-learn English list + news boilerplate"],
            ["Stemming", "Porter stemmer"],
            ["Minimum sentence length", "4 tokens"],
            ["Graph", "Undirected, complete weighted graph, zero-weight edges removed"],
            ["PageRank damping factor", str(CONFIG["damping_factor_alpha"])],
            ["PageRank max iterations", str(CONFIG["max_iterations"])],
            ["PageRank tolerance", str(CONFIG["tolerance"])],
            ["Redundancy threshold (ablation)", str(CONFIG["redundancy_threshold"])],
            ["Random seed", str(CONFIG["random_seed"])],
            ["ROUGE implementation", "rouge-score 0.1.2, use_stemmer=True"],
            ["Confidence intervals", "Percentile bootstrap, 2,000 resamples, 95%"],
            ["Significance test", "Two-sided Wilcoxon signed-rank, paired by document"],
        ],
        columns=["Parameter", "Value"],
    )


def environment_table() -> pd.DataFrame:
    return pd.DataFrame(
        [["Python", META["python"].split()[0]], *[[name, PACKAGES[name]] for name in PACKAGES]],
        columns=["Component", "Version"],
    )


def example_card(label: str, heading: str, record: dict) -> str:
    textrank_rouge = record["rouge"].get("TextRank", {}).get("rouge1_f1", 0) * 100
    lead_rouge = record["rouge"].get("Lead-k", {}).get("rouge1_f1", 0) * 100

    def clip(text: str, limit: int = 480) -> str:
        text = " ".join(text.split())
        return html.escape(text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …")

    return f"""
<div class="example-card">
<div class="example-head"><strong>{heading}</strong> &nbsp;·&nbsp; document <code>{html.escape(record["doc_id"][:40])}</code></div>
<div class="example-rouge">ROUGE-1 F1 &nbsp; TextRank <strong>{textrank_rouge:.1f}%</strong> &nbsp;|&nbsp; Lead-3 <strong>{lead_rouge:.1f}%</strong></div>
<p><span class="tag tag-blue">TextRank</span> {clip(record["textrank"])}</p>
<p><span class="tag tag-orange">Lead-3</span> {clip(record["lead"])}</p>
<p><span class="tag tag-grey">Reference</span> {clip(record["reference"])}</p>
</div>
"""


def build_markdown() -> str:
    best = EXAMPLES["best_vs_lead"]
    median = EXAMPLES["median"]
    worst = EXAMPLES["worst_vs_lead"]

    main_table = method_table()
    detail = detail_table()
    significance = significance_table()
    lengths = length_table()
    ablation = ablation_table()

    best_gain = (best["rouge"]["TextRank"]["rouge1_f1"] - best["rouge"]["Lead-k"]["rouge1_f1"]) * 100
    worst_gap = (worst["rouge"]["TextRank"]["rouge1_f1"] - worst["rouge"]["Lead-k"]["rouge1_f1"]) * 100

    markdown = f"""
<div class="cover">
  <div class="cover-institution">{html.escape(STUDENT["institution"])}<br><span>{html.escape(STUDENT["department"])}</span></div>
  <div class="cover-rule"></div>
  <div class="cover-kicker">MINI PROJECT REPORT</div>
  <h1 class="cover-title">Extractive Text Summarisation with TextRank</h1>
  <div class="cover-subtitle">A TF-IDF sentence-graph approach evaluated with ROUGE-1, ROUGE-2 and ROUGE-L</div>
  <table class="cover-table">
    <tr><td>Submitted by</td><td><strong>{html.escape(STUDENT["student_name"])}</strong></td></tr>
    <tr><td>Roll number</td><td>{html.escape(STUDENT["roll_number"])}</td></tr>
    <tr><td>Course</td><td>{html.escape(STUDENT["course"])} · {html.escape(STUDENT["semester"])}</td></tr>
    {GUIDE_ROW}
  </table>
  <div class="cover-date">{html.escape(STUDENT["submission_date"])}</div>
</div>

<div class="page-break"></div>

## Abstract

Automatic text summarisation reduces a long document to its most important content. This project implements an **extractive** summariser based on **TextRank**, an unsupervised graph-ranking algorithm. Sentences are represented as TF-IDF vectors, pairwise cosine similarity defines a weighted graph, and PageRank (damping factor *d* = 0.85) ranks sentences by recursive centrality. The top-*k* sentences are extracted and restored to their original order so that the summary reads naturally.

The system is evaluated on a fixed slice of 150 articles from the **CNN/DailyMail** test split using **ROUGE-1, ROUGE-2 and ROUGE-L** against human-written highlights. TextRank achieves **{pct(TR['rouge1_f1'])} ROUGE-1 F1**, **{pct(TR['rouge2_f1'])} ROUGE-2 F1** and **{pct(TR['rougeL_f1'])} ROUGE-L F1** at three sentences per summary. It significantly outperforms a random extractor (+{abs(CMP_RANDOM['mean_difference']) * 100:.2f} ROUGE-1 points, *p* = {CMP_RANDOM['p_value']:.3f}) but trails the Lead-3 baseline ({pct(LEAD['rouge1_f1'])} ROUGE-1, *p* = {CMP_LEAD['p_value']:.3f}), a result explained by the inverted-pyramid structure of news and the lead bias of the CNN/DailyMail references. A redundancy-filtered variant, a summary-length sweep, paired bootstrap confidence intervals and a Wilcoxon signed-rank test are reported to separate genuine algorithmic value from dataset artefacts. The complete pipeline is also deployed as an interactive Streamlit application.

**Keywords:** extractive summarisation, TextRank, PageRank, TF-IDF, ROUGE, CNN/DailyMail, unsupervised NLP

<div class="page-break"></div>

## Contents

1. Introduction
2. Background and related work
3. Methodology
4. Experimental setup
5. Results
6. Discussion
7. Conclusion and future work
8. References
9. Appendix A — Repository structure
10. Appendix B — Reproduction guide

### List of Figures

1. End-to-end TextRank summarisation and evaluation pipeline
2. Distribution of article and reference-summary lengths
3. ROUGE-1/2/L F1 for all methods at k = 3
4. Per-document ROUGE-1 difference between TextRank and Lead-3
5. ROUGE-1 F1 versus summary length
6. Redundancy-filter ablation
7. Sentence similarity graph for one article
8. Streamlit demonstration with live ROUGE metrics

### List of Tables

1. Dataset statistics for the evaluation slice
2. Experimental configuration
3. Software environment
4. Main ROUGE results at k = 3
5. Precision/recall/F1 breakdown
6. Paired significance tests
7. ROUGE-1 F1 by summary length
8. Redundancy-filter ablation

<div class="page-break"></div>

## 1. Introduction

### 1.1 Motivation

The volume of text produced every day — news wires, reports, transcripts, research articles — far exceeds the time available to read it. A summariser condenses a document while preserving its essential information. **Extractive** summarisation does this by selecting whole sentences from the source; **abstractive** summarisation rewrites them, potentially producing more fluent text but also risking factual drift and hallucination.

Extractive methods remain attractive in practice because they are faithful by construction: every word in the output was written by the original author. They are also cheap, deterministic, interpretable and do not require a labelled training set. The central research question is therefore *which sentences to select* — a ranking problem that graph-based algorithms such as TextRank address directly.

### 1.2 Problem statement

Given a document consisting of sentences *S* = (S₁, …, Sₙ), produce a summary of at most *k* sentences that maximises overlap with a human reference while remaining non-redundant and coherent. Naive strategies are weak: a random extract ignores content entirely, while taking the first *k* sentences exploits the layout of news but fails on documents whose key information appears later.

### 1.3 Objectives

1. Implement the complete TextRank pipeline from first principles: sentence segmentation, preprocessing, TF-IDF representation, cosine-similarity graph construction, PageRank scoring and ranked extraction.
2. Evaluate the summariser quantitatively with ROUGE-1, ROUGE-2 and ROUGE-L against human reference summaries on a standard benchmark.
3. Compare TextRank with **Lead-3** and **Random-3** baselines to measure genuine algorithmic value rather than the lead bias of news.
4. Study the effect of summary length and of a redundancy filter, and test the statistical significance of the differences.
5. Package the work as a reproducible notebook, an interactive Streamlit demo, and a complete public repository.

### 1.4 Contributions

* A clean, dependency-light implementation of TextRank that runs in about {META['evaluation']['mean_textrank_runtime_ms']:.1f} ms per article on a CPU.
* A reproducible ROUGE evaluation on a cached CNN/DailyMail slice, with bootstrap confidence intervals and paired significance tests.
* An honest analysis of the Lead-3 effect, including per-document wins and losses.
* A working demonstration application with live ROUGE scoring, and the full replication package (notebook, scripts, figures, report).

### 1.5 Organisation

Section 2 reviews graph-based summarisation and ROUGE. Section 3 presents the methodology and equations. Section 4 describes the dataset, hyperparameters and protocol. Section 5 reports results; Section 6 discusses them and states limitations; Section 7 concludes.

## 2. Background and related work

### 2.1 Graph-based ranking

PageRank [2] assigns importance to nodes of a directed graph by simulating a random walk: a node is important if important nodes link to it. TextRank [1] transfers this idea to natural language by treating linguistic units — words, collocations or sentences — as nodes and similarity relations as weighted edges. For sentence extraction, an edge weight is the lexical similarity between two sentences, and the PageRank score of a sentence measures how central it is to the document's discourse. LexRank [6] uses the same intuition with IDF-modified cosine similarity and degree normalisation.

Unlike centroid-based ranking, which scores each sentence independently, graph ranking is *recursive*: a sentence gains importance by being similar to sentences that are themselves important. This captures the fact that central concepts are mentioned in several places, in different words.

### 2.2 Neural and other approaches

Recent abstractive systems based on sequence-to-sequence models, BART and Pegasus produce more fluent summaries, but they require large training corpora and GPUs and can hallucinate content. Extractive neural models (e.g. BERTSum) frame sentence selection as classification and need document-summary labels. TextRank remains a strong, transparent baseline and a useful teaching vehicle because every step is inspectable.

### 2.3 The Lead-3 baseline

News articles follow the **inverted pyramid**: the most newsworthy facts appear first. CNN/DailyMail highlights were written by editors under the same convention, so the first three sentences are an unusually strong baseline [4]. Any evaluation that omits it risks attributing the dataset's layout bias to the algorithm. We therefore report Lead-3 alongside Random-3 and discuss the gap explicitly.

### 2.4 ROUGE

ROUGE [3] measures n-gram overlap between a candidate and one or more references. ROUGE-1 counts unigram matches, ROUGE-2 counts bigrams (capturing local phrasing), and ROUGE-L uses the longest common subsequence, which rewards in-order sentence structure. All three are reported as precision, recall and F1. ROUGE is lexical: a semantically correct paraphrase can score poorly, which is why qualitative examples and per-document analysis accompany the aggregate numbers.

## 3. Methodology

### 3.1 System overview

{figure('pipeline.png', 'Figure 1. End-to-end TextRank summarisation and evaluation pipeline.', '92%')}

The pipeline has two branches. The **summarisation** branch converts the article into a sentence graph, ranks sentences with PageRank and extracts the top-*k*. The **evaluation** branch scores the generated summary against the human reference with ROUGE, and repeats the process for the baselines.

### 3.2 Preprocessing

The article is segmented with NLTK's Punkt tokeniser, using a local copy of the model so the notebook runs offline. Sentences shorter than four tokens are discarded (they are usually datelines, captions or fragments), and exact duplicates are removed to prevent redundancy from inflating similarity.

Each sentence is lowercased, tokenised with a regular expression that preserves hyphenated compounds and numbers, stripped of stopwords, and Porter-stemmed. The stopword list combines the scikit-learn English list with a small set of news-boilerplate words (*said*, *also*, *would*, …) that would otherwise create spurious similarity. A sentence with no remaining content tokens is dropped because it cannot be ranked meaningfully.

### 3.3 Sentence representation

Each sentence is represented as a sparse TF-IDF vector over the document vocabulary:

{equation('eq_tfidf.png', 'TF-IDF weighting equation')}

where *tf(t, d)* is the term frequency of term *t* in sentence *d*, *df(t)* is the number of sentences containing *t*, and *N* is the number of sentences. Sublinear term frequency prevents a repeated word from dominating; the logarithm down-weights terms that occur throughout the article. Vectors are L2-normalised so that the dot product equals cosine similarity.

### 3.4 Similarity graph

Pairwise similarity is computed for every sentence pair:

{equation('eq_cosine.png', 'Cosine similarity between two sentence vectors')}

The similarity matrix is symmetric with a zero diagonal. It is converted into an undirected weighted graph: nodes are sentences, edge weights are similarities, and zero-weight edges are removed. In practice the matrix is sparse and low-valued — most sentence pairs share few content words — which is exactly why recursive centrality is more informative than raw term overlap.

### 3.5 PageRank

The PageRank score of sentence *Vᵢ* is:

{equation('eq_pagerank.png', 'PageRank recurrence used for sentence scoring')}

The damping factor *d* = 0.85 is the standard value: with probability 0.85 the walk follows an edge proportionally to its weight, and with probability 0.15 it jumps to a random sentence. The denominator normalises the contribution of an incoming neighbour by its total outgoing weight, so a sentence connected to many weakly related sentences does not accumulate undue importance. Scores are computed by power iteration with a tolerance of 10⁻⁸ and a cap of 200 iterations.

### 3.6 Extraction and ordering

Sentences are sorted by descending PageRank score and the top-*k* are selected. Selected sentences are then **re-sorted by their original position** before concatenation. Without this step the summary would be a bag of fluent but disjointed sentences; with it, the extract follows the narrative order of the source. The number of sentences *k* is the only length control, making compression directly comparable across methods.

As an ablation, *TextRank + MMR* greedily skips a candidate whose cosine similarity to an already selected sentence exceeds 0.30, then fills the remaining budget with the next-best candidates. In TF-IDF space 0.30 is already in the extreme tail of the similarity distribution, so the filter targets near-duplicates rather than paraphrases.

### 3.7 Baselines

* **Lead-3** returns the first three sentences. It is a strong, widely reported news baseline.
* **Random-3** samples three sentences uniformly without replacement using seed 42. Averaged over documents it estimates the score of a content-blind summariser.

### 3.8 Evaluation metrics

For a candidate *C* and reference *R*, ROUGE-1 and ROUGE-2 compute clipped n-gram overlap and report precision, recall and F1. ROUGE-L uses the length of the longest common subsequence *LCS*:

{equation('eq_rougel.png', 'ROUGE-L F-measure')}

where *m* and *n* are the lengths of *R* and *C*, *β* controls the precision/recall trade-off (set to 1 for F1), and *P_lcs* and *R_lcs* are the LCS-based precision and recall. Scores are computed with `rouge-score` using Porter stemming, the standard convention in summarisation research.

### 3.9 Statistical protocol

Aggregate means alone can hide variance across documents. We therefore report **95% percentile bootstrap confidence intervals** (2,000 resamples) for every mean, and compare methods with a **two-sided Wilcoxon signed-rank test** on paired per-document ROUGE differences. The test is non-parametric and appropriate because the same 150 documents are used for every method.

## 4. Experimental setup

### 4.1 Dataset

{table(dataset_table(), 'Table 1. Dataset statistics for the evaluation slice.')}

CNN/DailyMail is the standard benchmark for summarisation. Each example pairs a news article with multi-sentence highlights written by editors. We use the first 150 articles of the test split, cached locally after a one-time download from the HuggingFace datasets server. The slice is large enough for stable means and small enough that every document can be inspected.

{figure('dataset_overview.png', 'Figure 2. Distribution of article length and reference-summary length in the evaluation slice.', '92%')}

### 4.2 Hyperparameters

{table(config_table(), 'Table 2. Experimental configuration.')}

### 4.3 Implementation and environment

{table(environment_table(), 'Table 3. Software environment used for the reported results.')}

The implementation is organised as importable modules (`src/`) shared by the notebook, the experiment runner and the web application. The full evaluation — 150 documents × 6 summary lengths × 4 methods = 3,600 summaries — runs in {META['evaluation']['total_runtime_seconds']:.1f} seconds on a laptop CPU, excluding ROUGE scoring.

### 4.4 Protocol

Every method extracts exactly *k* = {CONFIG['primary_length']} sentences for the primary comparison, matching the average reference length of {DATASET['reference_sentences_mean']:.1f} sentences. The length sweep repeats the evaluation for *k* ∈ {{{", ".join(str(k) for k in CONFIG["summary_lengths"])}}}. All methods are evaluated on the same documents and against the same references; ROUGE uses Porter stemming.

## 5. Results

### 5.1 Main comparison

{table(main_table, 'Table 4. ROUGE F1 at k = 3 with 95% bootstrap confidence intervals, mean summary length and compression ratio (n = 150 documents).')}

{figure('rouge_by_method.png', 'Figure 3. ROUGE-1, ROUGE-2 and ROUGE-L F1 for all methods at k = 3.', '88%')}

TextRank achieves **{pct(TR['rouge1_f1'])} ROUGE-1 F1**, **{pct(TR['rouge2_f1'])} ROUGE-2 F1** and **{pct(TR['rougeL_f1'])} ROUGE-L F1**. Three findings stand out:

1. **TextRank beats the random baseline on every metric.** The ROUGE-1 gain is +{abs(CMP_RANDOM['mean_difference']) * 100:.2f} points and is statistically significant (*p* = {CMP_RANDOM['p_value']:.3f}). This isolates genuine algorithmic value: the graph ranking is learning something about sentence importance.
2. **Lead-3 is {abs(CMP_LEAD['mean_difference']) * 100:.2f} ROUGE-1 points stronger** ({pct(LEAD['rouge1_f1'])} vs {pct(TR['rouge1_f1'])}; *p* = {CMP_LEAD['p_value']:.3f}). The difference is small but consistent, and is discussed in Section 6.
3. **Redundancy filtering gives a negligible gain** ({pct(MMR['rouge1_f1'])} vs {pct(TR['rouge1_f1'])}; *p* = {CMP_MMR['p_value']:.3f}). Three-sentence summaries from short news articles rarely contain near-duplicates, so there is little for the filter to remove.

TextRank selects longer sentences on average ({TR['summary_words']:.0f} words vs {LEAD['summary_words']:.0f} for Lead-3) while covering a similar fraction of the source ({pct(TR['compression_ratio'], 1)}% vs {pct(LEAD['compression_ratio'], 1)}%). Because ROUGE precision penalises extra words, this difference is a mild disadvantage for TextRank; it is nonetheless the honest output of the algorithm.

### 5.2 Precision and recall

{table(detail, 'Table 5. Full precision/recall/F1 breakdown at k = 3 (%).')}

All methods have higher precision than recall, which is expected: a three-sentence extract is shorter than most references, so much of the reference content is never covered. Lead-3's precision advantage ({(LEAD['rouge1_precision'] - TR['rouge1_precision']) * 100:+.2f} points of ROUGE-1 precision) reflects the fact that its sentences concentrate on the article's main event, while TextRank sometimes selects a central background sentence that the reference omits.

### 5.3 Statistical significance

{table(significance, 'Table 6. Paired comparisons for ROUGE-1 F1 at k = 3 (positive difference favours TextRank).')}

The paired analysis is more informative than the means. TextRank wins on {CMP_LEAD['a_wins']} of {CMP_LEAD['n_pairs']} documents against Lead-3, loses on {CMP_LEAD['b_wins']}, and ties on {CMP_LEAD['ties']}: the baseline's advantage comes from a subset of documents with a very strong lede, not from uniform dominance. Against Random-3, TextRank wins on {CMP_RANDOM['a_wins']} documents and loses on {CMP_RANDOM['b_wins']}. The MMR comparison is statistically indistinguishable from zero, confirming that the ablation changes little.

{figure('delta_vs_lead3.png', 'Figure 4. Distribution of per-document ROUGE-1 F1 differences between TextRank and Lead-3.', '82%')}

### 5.4 Effect of summary length

{table(lengths, 'Table 7. ROUGE-1 F1 (%) as the number of extracted sentences varies.')}

{figure('rouge_by_length.png', 'Figure 5. ROUGE-1 F1 versus summary length, with 95% confidence bands.', '86%')}

All methods peak at a short length and then decline, because precision falls faster than recall rises. TextRank trails Lead-3 by {abs(SWEEP.loc[2, 'TextRank'] - SWEEP.loc[2, 'Lead-k']) * 100:.2f} points at *k* = 2; the gap peaks at {abs(SWEEP.loc[4, 'TextRank'] - SWEEP.loc[4, 'Lead-k']) * 100:.2f} points at *k* = 4 and narrows to {abs(SWEEP.loc[8, 'TextRank'] - SWEEP.loc[8, 'Lead-k']) * 100:.2f} points at *k* = 8. TextRank also beats Random-3 at every length, so the graph ranking is consistently more informative than content-blind selection; it does not, however, overcome the positional prior of the news lede.

### 5.5 Redundancy-filter ablation

{table(ablation, 'Table 8. ROUGE-1 F1 with and without the MMR-style redundancy filter.')}

{figure('ablation_mmr.png', 'Figure 6. Ablation at k = 3: plain TextRank versus the redundancy-filtered variant.', '80%')}

The filter is neutral on this dataset. The likely reason is document length: with roughly {DATASET['source_sentences_mean']:.0f} sentences per article and a three-sentence budget, a top-3 set rarely contains two sentences with cosine similarity above 0.30. On longer documents or larger *k*, the filter should matter more; that is a natural extension.

### 5.6 Qualitative analysis

{figure('sentence_graph.png', 'Figure 7. Sentence similarity graph for one article. Node colour and size encode PageRank score; labelled nodes are highly ranked sentences.', '82%')}

The graph above makes the algorithm's behaviour visible. Highly scored nodes sit at the centre of the similarity network: they are not necessarily the first sentences, but they are connected to many other sentences. Isolated nodes — often quotations or tangential details — receive low scores and are not selected.

{example_card('best', 'Largest gain over Lead-3', best)}
{example_card('median', 'Median document', median)}
{example_card('worst', 'Largest loss to Lead-3', worst)}

The best case (+{best_gain:.1f} ROUGE-1 points) is an article whose lede is scene-setting and whose key facts appear later: the graph finds them, while Lead-3 cannot. The worst case ({worst_gap:+.1f} points) shows the opposite pattern — a tightly written inverted-pyramid article in which the first three sentences already contain almost every reference fact. TextRank spends part of its budget on a central but background sentence, and precision falls.

### 5.7 Efficiency

At *k* = 3, TextRank takes **{TR['runtime_ms']:.2f} ms** per article on average, including TF-IDF construction and PageRank. Lead-3 takes {LEAD['runtime_ms']:.2f} ms and Random-3 {RANDOM['runtime_ms']:.2f} ms. The full 3,600-summary evaluation takes {META['evaluation']['total_runtime_seconds']:.1f} s, with ROUGE scoring accounting for most of the wall-clock time. The method is therefore suitable for interactive use, which is why the demo can score summaries live.

### 5.8 Demonstration application

{figure('app_overview.png', 'Figure 8. Streamlit demonstration: generated summary, reference summary and live ROUGE metrics.', '100%')}

The Streamlit application exposes the same pipeline with controls for method, summary length, damping factor and similarity threshold. It computes ROUGE against a pasted reference and shows the sentence ranking, similarity heatmap and graph. Automated screenshots are collected in `deliverables/output_screenshots.pdf`.

## 6. Discussion

### 6.1 Why Lead-3 is hard to beat

The result is a property of the benchmark, not a defect in the implementation. CNN/DailyMail articles are news stories written in inverted-pyramid style, and the highlights were written by editors who read the same article and followed the same news values. The first two or three sentences therefore contain a disproportionate share of reference content. Lead-3 benefits from a positional prior that exactly matches the data-generating process, while TextRank is position-agnostic and must infer salience from lexical structure alone.

TextRank nevertheless does what it is designed to do. It beats a random extractor significantly at every summary length, and its wins cluster on articles whose important content is not at the top. The per-document analysis (Figure 4) shows a distribution centred slightly below zero but with a long right tail: on a meaningful minority of documents, the graph finds a better summary than the lede.

### 6.2 Implications for design

If the goal is the best possible ROUGE on news, a position prior is the obvious next step: initialise PageRank with a decaying positional vector, or blend the PageRank score with a lead score. That hybrid would retain the graph's ability to find later content while exploiting the strong prior. If the goal is domain independence, pure TextRank is the safer choice — it has no built-in assumption about where information lives.

The redundancy filter is worth keeping as an option but is not a priority for short news summaries. A proper MMR objective, which jointly optimises relevance and novelty rather than applying a hard threshold, would be a cleaner formulation.

### 6.3 Threats to validity

* **Sample size.** The evaluation uses 150 test articles. Bootstrap intervals quantify the resulting uncertainty, but they do not eliminate it.
* **Single reference.** CNN/DailyMail has one reference per article; a valid extract that paraphrases it can score low. ROUGE is a lower bound on quality.
* **Preprocessing sensitivity.** Sentence segmentation and stopword choice affect the graph. The configuration is fixed and reported, but different choices would shift scores by a few points.
* **Threshold selection.** The 0.30 redundancy threshold was chosen from the similarity distribution, not tuned on the test set. It is reported as an ablation, not as a tuned result.
* **Implementation variants.** TextRank has many published variants (directed graphs, windowed co-occurrence, degree normalisation). We implement the sentence-similarity formulation of Mihalcea and Tarau and state the choices explicitly.

### 6.4 Ethical considerations

Extractive summarisation cannot fabricate facts, which is a meaningful safety advantage over abstractive models. It can, however, amplify selection bias: if the graph systematically favours certain perspectives, the summary inherits that bias. It can also produce misleading summaries by removing necessary context. Any deployment should preserve a link to the source and make clear that the output is an extract.

## 7. Conclusion and future work

This project implemented TextRank from first principles and evaluated it rigorously against human summaries. The pipeline — sentence segmentation, TF-IDF representation, cosine-similarity graph, PageRank with *d* = 0.85, ranked extraction and chronological re-ordering — is compact, interpretable and fast ({TR['runtime_ms']:.1f} ms per article). On 150 CNN/DailyMail test articles it reaches {pct(TR['rouge1_f1'])} ROUGE-1 F1, {pct(TR['rouge2_f1'])} ROUGE-2 F1 and {pct(TR['rougeL_f1'])} ROUGE-L F1, significantly beating a random baseline while trailing the strong Lead-3 baseline. The analysis explains both outcomes and identifies where the graph adds value.

Future work will pursue four directions: (i) a position-aware PageRank initialisation to combine the graph with the news prior; (ii) embedding-based sentence representations (Sentence-BERT) to capture paraphrase; (iii) a proper MMR or submodular selection objective for longer summaries; and (iv) evaluation with BERTScore and human judgements to complement lexical ROUGE.

## 8. References

1. Mihalcea, R., & Tarau, P. (2004). TextRank: Bringing Order into Texts. *Proceedings of EMNLP 2004*, 404–411.
2. Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank Citation Ranking: Bringing Order to the Web. *Stanford InfoLab Technical Report*.
3. Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *ACL Workshop on Text Summarization Branches Out*.
4. Nallapati, R., Zhou, B., Gulcehre, C., & Xiang, B. (2016). Abstractive Text Summarization using Sequence-to-Sequence RNNs and Beyond. *CoNLL 2016*.
5. Hermann, K. M., Kočiský, T., Grefenstette, E., Espeholt, L., Kay, W., Suleyman, M., & Blunsom, P. (2015). Teaching Machines to Read and Comprehend. *NeurIPS 2015*.
6. Erkan, G., & Radev, D. R. (2004). LexRank: Graph-based Lexical Centrality as Salience in Text Summarization. *Journal of Artificial Intelligence Research, 22*, 457–479.
7. Lewis, M., et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. *ACL 2020*.
8. Zhang, J., Zhao, Y., Saleh, M., & Liu, P. (2020). PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization. *ICML 2020*.
9. scikit-learn developers. TfidfVectorizer documentation. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
10. NetworkX developers. PageRank documentation. https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html

<div class="page-break"></div>

## 9. Appendix A — Repository structure

```text
textrank-summarizer/
├── README.md
├── requirements.txt
├── data/
│   ├── cnn_dailymail_sample.json     150 articles + reference summaries
│   └── nltk_data/                    bundled Punkt model (offline)
├── notebooks/
│   └── textrank_evaluation.ipynb     executed notebook with all outputs
├── src/
│   ├── dataset.py                    download / load the cached slice
│   ├── preprocess.py                 segmentation, tokenisation, stemming
│   ├── textrank.py                   TF-IDF graph + PageRank + extraction
│   ├── baselines.py                  Lead-k and Random-k
│   ├── evaluate.py                   ROUGE, bootstrap CI, Wilcoxon tests
│   ├── visualize.py                  all report and app figures
│   └── app.py                        Streamlit demonstration
├── scripts/
│   ├── fetch_data.py                 re-download the dataset slice
│   ├── run_experiment.py             full evaluation -> results/
│   ├── build_notebook.py             build + execute the notebook
│   ├── capture_app.py                Playwright app screenshots
│   ├── capture_notebook.py           Playwright notebook screenshots
│   ├── build_report.py               Markdown -> HTML -> PDF report
│   ├── build_screenshots_pdf.py      screenshots compilation PDF
│   └── build_demo_video.py           narrated MP4 demo generator
├── results/
│   ├── per_document.csv              one row per (document, method, k)
│   ├── results_table_k3.csv          main results table
│   ├── paired_tests.json             significance tests
│   ├── examples.json                 best / median / worst cases
│   └── figures/                      all figures and app screenshots
├── report/
│   ├── report.md                     Markdown source (regenerated by this script)
│   └── student.json                  student details for the cover
└── deliverables/
    ├── project_report.pdf
    ├── output_screenshots.pdf
    └── demo_video.mp4                narrated walkthrough (GitHub release)
```

## 10. Appendix B — Reproduction guide

```bash
# 1. Environment
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt

# 2. Dataset (already cached; this re-downloads it)
python scripts/fetch_data.py

# 3. Evaluation: writes results/ and results/figures/
python scripts/run_experiment.py

# 4. Notebook (builds and executes every cell)
python scripts/build_notebook.py
python scripts/capture_notebook.py

# 5. Interactive demo
streamlit run src/app.py

# 6. Report and screenshots PDF
python scripts/build_report.py
python scripts/build_screenshots_pdf.py

# 7. Narrated demo video (Windows SAPI)
python scripts/build_demo_video.py
```

All random operations are seeded (seed {CONFIG['random_seed']}); rerunning the commands above reproduces the reported numbers exactly, up to library-version differences.

<div class="page-break"></div>

<div class="declaration">
<h2>Declaration</h2>
<p>I declare that this mini project report is my own work. The implementation, experiments, figures and analysis were produced by me, and all external sources, libraries and datasets are acknowledged in the references and in the repository README.</p>
<table class="signature">
<tr><td>Name: {html.escape(STUDENT["student_name"])}</td><td>Signature: ______________________</td></tr>
<tr><td>Roll number: {html.escape(STUDENT["roll_number"])}</td><td>Date: {html.escape(STUDENT["submission_date"])}</td></tr>
</table>
</div>
"""
    return markdown


CSS = """
@page { size: A4; margin: 19mm 17mm 16mm 17mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10pt; line-height: 1.52; color: #1d2b3a; margin: 0;
}
h1.cover-title { font-size: 27pt; line-height: 1.18; color: #16324f; margin: 14px 0 10px; border: none; }
h2 { font-size: 14.5pt; color: #16324f; border-bottom: 2px solid #1f4e79; padding-bottom: 4px; margin: 22px 0 10px; page-break-after: avoid; }
h3 { font-size: 11.2pt; color: #1f4e79; margin: 16px 0 6px; page-break-after: avoid; }
p { margin: 7px 0; text-align: justify; }
ul, ol { margin: 7px 0 9px 0; padding-left: 20px; }
li { margin: 3px 0; }
strong { color: #12263a; }
a { color: #1f4e79; text-decoration: none; }
table { width: 100%; border-collapse: collapse; font-size: 8.1pt; margin: 8px 0 14px; page-break-inside: avoid; }
th { background: #16324f; color: #ffffff; padding: 5px 6px; text-align: left; font-weight: 600; border: 1px solid #16324f; }
td { padding: 4px 6px; border: 1px solid #d7e1eb; vertical-align: top; }
tr:nth-child(even) td { background: #f6f9fc; }
figure { margin: 12px 0 16px; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 205mm; border: 1px solid #e3eaf2; border-radius: 4px; }
figcaption { font-size: 8.6pt; color: #4a5568; font-style: italic; margin-top: 5px; text-align: center; }
.eq { text-align: center; margin: 12px 0; }
.eq img { max-height: 120px; max-width: 92%; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.4pt; background: #f2f5f8; padding: 1px 3px; border-radius: 3px; }
pre { font-family: Consolas, "Courier New", monospace; font-size: 7.9pt; line-height: 1.4; background: #f7f9fb; border-left: 3px solid #1f4e79; padding: 8px 10px; page-break-inside: avoid; white-space: pre-wrap; }
blockquote { border-left: 3px solid #c55a11; margin: 10px 0; padding: 6px 12px; background: #fdf8f2; color: #3b2a1a; }
.page-break { page-break-before: always; }
.cover { min-height: 245mm; display: flex; flex-direction: column; justify-content: center; text-align: center; }
.cover-institution { font-size: 14pt; font-weight: 700; letter-spacing: .4px; color: #16324f; line-height: 1.4; }
.cover-institution span { font-size: 10.5pt; font-weight: 400; color: #4a5568; letter-spacing: .2px; }
.cover-rule { width: 55%; height: 3px; background: #1f4e79; margin: 16px auto 26px; }
.cover-kicker { font-size: 10.5pt; letter-spacing: 3px; color: #c55a11; font-weight: 700; }
.cover-subtitle { font-size: 12.5pt; color: #4a5568; margin: 4px auto 34px; max-width: 150mm; line-height: 1.45; }
.cover-table { width: 125mm; margin: 0 auto; font-size: 10.5pt; border: none; }
.cover-table td { border: none; border-bottom: 1px solid #e3eaf2; padding: 7px 6px; background: transparent !important; text-align: left; }
.cover-table td:first-child { color: #5b6b7b; width: 32%; }
.cover-date { margin-top: 34px; font-size: 11pt; color: #16324f; font-weight: 600; }
.example-card { border: 1px solid #d7e1eb; border-radius: 7px; padding: 9px 12px; margin: 11px 0; page-break-inside: avoid; background: #fbfdff; }
.example-head { font-size: 9pt; color: #16324f; margin-bottom: 3px; }
.example-rouge { font-size: 8.5pt; color: #4a5568; margin-bottom: 6px; }
.example-card p { font-size: 8.9pt; margin: 4px 0; text-align: left; }
.tag { display: inline-block; font-size: 7.4pt; font-weight: 700; color: #fff; padding: 1px 6px; border-radius: 3px; margin-right: 4px; letter-spacing: .3px; }
.tag-blue { background: #1f4e79; }
.tag-orange { background: #c55a11; }
.tag-grey { background: #6b7c8f; }
.declaration { margin-top: 30px; }
.signature { width: 100%; margin-top: 26px; font-size: 9.5pt; }
.signature td { border: none; padding: 12px 4px; background: transparent !important; }
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Extractive Text Summarisation with TextRank — Mini Project Report</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    edge = next((path for path in EDGE_CANDIDATES if path.exists()), None)
    if edge is None:
        raise RuntimeError("Microsoft Edge was not found; open report.html manually and print to PDF.")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(edge),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--print-to-pdf-no-header",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=15000",
        "--allow-file-access-from-files",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
    if not pdf_path.exists() or pdf_path.stat().st_size < 20_000:
        raise RuntimeError(f"PDF generation failed or produced an empty file: {pdf_path}")


def main() -> None:
    import markdown

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DELIVERABLES.mkdir(parents=True, exist_ok=True)

    markdown_text = build_markdown()
    (REPORT_DIR / "report.md").write_text(markdown_text, encoding="utf-8")

    body = markdown.markdown(
        markdown_text,
        extensions=["tables", "fenced_code", "sane_lists", "md_in_html"],
        output_format="html5",
    )
    html_document = HTML_TEMPLATE.format(css=CSS, body=body)
    html_path = REPORT_DIR / "report.html"
    html_path.write_text(html_document, encoding="utf-8")

    pdf_path = DELIVERABLES / "project_report.pdf"
    render_pdf(html_path, pdf_path)
    print(f"wrote {REPORT_DIR / 'report.md'}")
    print(f"wrote {html_path}")
    print(f"wrote {pdf_path} ({pdf_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
