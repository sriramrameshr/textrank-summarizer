"""Streamlit demo for the TextRank extractive summariser.

Run locally:
    streamlit run src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import visualize  # noqa: E402
from src.baselines import lead_k  # noqa: E402
from src.dataset import get_documents  # noqa: E402
from src.evaluate import score_pair  # noqa: E402
from src.textrank import summarize  # noqa: E402

st.set_page_config(
    page_title="TextRank Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    header[data-testid="stHeader"] {display: none;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stDecoration"] {display: none;}
    [data-testid="stStatusWidget"] {display: none;}
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1280px;}
    h1, h2, h3 {color: #16324f;}
    .hero {
        background: linear-gradient(120deg, #16324f 0%, #1f4e79 60%, #2e6f95 100%);
        padding: 1.15rem 1.4rem; border-radius: 12px; color: #ffffff; margin-bottom: 1rem;
    }
    .hero h1 {color: #ffffff; margin: 0 0 .25rem 0; font-size: 1.75rem;}
    .hero p {margin: 0; color: #d7e6f2; font-size: .95rem;}
    .summary-box {
        background: #f4f8fc; border-left: 5px solid #1f4e79; border-radius: 8px;
        padding: 1rem 1.1rem; font-size: .95rem; line-height: 1.55; color: #14293d;
    }
    .reference-box {
        background: #fbf7f0; border-left: 5px solid #c55a11; border-radius: 8px;
        padding: 1rem 1.1rem; font-size: .95rem; line-height: 1.55; color: #3b2a1a;
    }
    div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: .6rem .8rem;
    }
    div[data-testid="stMetricLabel"] p {font-size: .8rem; color: #4a5568;}
    .small-note {color: #5b6b7b; font-size: .82rem;}
    .pill {
        display: inline-block; background: #e8f0f7; color: #1f4e79; border-radius: 999px;
        padding: .12rem .6rem; font-size: .78rem; margin-right: .35rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _documents() -> list[dict]:
    return get_documents()


@st.cache_data(show_spinner=False)
def _benchmark_table() -> pd.DataFrame | None:
    path = ROOT / "results" / "results_table_k3.csv"
    return pd.read_csv(path) if path.exists() else None


def _load_example() -> None:
    documents = _documents()
    document = documents[st.session_state.example_index]
    st.session_state.article_text = document["article"]
    st.session_state.reference_text = document["reference_summary"]


def _on_upload() -> None:
    uploaded = st.session_state.get("uploaded_file")
    if uploaded is None:
        return
    content = uploaded.read().decode("utf-8", errors="replace").strip()
    if content:
        st.session_state.article_text = content


def _init_state() -> None:
    documents = _documents()
    if "example_index" not in st.session_state:
        st.session_state.example_index = 0
    if "article_text" not in st.session_state:
        st.session_state.article_text = documents[0]["article"]
    if "reference_text" not in st.session_state:
        st.session_state.reference_text = documents[0]["reference_summary"]
    if "run_snapshot" not in st.session_state:
        st.session_state.run_snapshot = None


_init_state()
documents = _documents()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.selectbox(
        "Example article (CNN/DailyMail)",
        options=list(range(len(documents))),
        format_func=lambda i: f"Article {i + 1} — {documents[i]['article'][:42]}…",
        key="example_index",
        on_change=_load_example,
    )
    method = st.radio(
        "Summarisation method",
        ["TextRank", "TextRank + redundancy filter"],
        help="The second option adds a Maximal Marginal Relevance style duplicate check.",
    )
    k = st.slider("Summary length (sentences)", 1, 10, 3)
    alpha = st.slider("Damping factor α", 0.50, 0.95, 0.85, 0.05)
    threshold = st.slider(
        "Similarity threshold",
        0.0,
        0.5,
        0.0,
        0.05,
        help="Edges below this cosine similarity are removed. 0 keeps the complete graph.",
    )

    st.divider()
    st.markdown("### 📊 Benchmark results")
    table = _benchmark_table()
    if table is not None:
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("150 CNN/DailyMail test articles, k = 3, mean F1 with 95% bootstrap CI.")
    else:
        st.info("Run `python scripts/run_experiment.py` to generate the benchmark table.")

    with st.expander("About this project"):
        st.markdown(
            "Extractive summarisation with **TextRank**: sentences are nodes in a "
            "TF-IDF cosine-similarity graph, and PageRank scores decide which "
            "sentences enter the summary. Output is evaluated with ROUGE-1/2/L."
        )

# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
      <h1>📄 TextRank Extractive Summarizer</h1>
      <p>TF-IDF sentence graph &nbsp;•&nbsp; PageRank ranking &nbsp;•&nbsp; ROUGE-1 / ROUGE-2 / ROUGE-L evaluation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.35, 1], gap="large")
with left:
    st.markdown("#### Source article")
    st.file_uploader(
        "Upload a .txt file (optional)",
        type=["txt", "md"],
        key="uploaded_file",
        on_change=_on_upload,
    )
    st.text_area(
        "Article text",
        key="article_text",
        height=300,
        label_visibility="collapsed",
        placeholder="Paste a news article here…",
    )

with right:
    st.markdown("#### Reference summary (optional)")
    st.text_area(
        "Reference summary",
        key="reference_text",
        height=300,
        label_visibility="collapsed",
        placeholder="Paste the human-written summary to compute ROUGE…",
    )
    st.caption("ROUGE is computed against this reference. Leave it empty to only extract a summary.")

article = st.session_state.article_text.strip()
reference = st.session_state.reference_text.strip()

generate = st.button("🚀 Generate summary", type="primary", width="stretch")
if generate:
    st.session_state.run_snapshot = {"article": article, "reference": reference, "k": k, "alpha": alpha,
                                     "threshold": threshold, "method": method}

# ``?demo=1`` pre-runs the first example, which makes automated screenshots and
# the recorded demo deterministic (no clicking required).
if str(st.query_params.get("demo", "0")) == "1" and st.session_state.run_snapshot is None:
    st.session_state.run_snapshot = {
        "article": article,
        "reference": reference,
        "k": 3,
        "alpha": 0.85,
        "threshold": 0.0,
        "method": "TextRank",
    }

snapshot = st.session_state.run_snapshot
if not snapshot or not snapshot["article"]:
    st.info("Click **Generate summary** to run TextRank on the article.")
    st.stop()

# --------------------------------------------------------------------------- #
# Run the selected method
# --------------------------------------------------------------------------- #
result = summarize(
    snapshot["article"],
    k=snapshot["k"],
    alpha=snapshot["alpha"],
    threshold=snapshot["threshold"],
    redundancy_threshold=0.30 if snapshot["method"].endswith("filter") else None,
)
reference = snapshot["reference"]
source_words = len(snapshot["article"].split())
summary_words = len(result.summary.split())
compression = 100 * summary_words / max(1, source_words)

scores = score_pair(result.summary, reference) if reference else {}
lead_indices = lead_k(snapshot["article"], k=snapshot["k"])["selected_indices"]
lead_summary = " ".join(result.sentences[i] for i in lead_indices)
lead_scores = score_pair(lead_summary, reference) if reference else {}

st.markdown(
    f"<span class='pill'>{snapshot['method']}</span>"
    f"<span class='pill'>k = {snapshot['k']}</span>"
    f"<span class='pill'>α = {snapshot['alpha']:.2f}</span>"
    f"<span class='pill'>{result.num_sentences} sentences scored</span>"
    f"<span class='pill'>{compression:.1f}% compression</span>",
    unsafe_allow_html=True,
)
st.write("")

if reference:
    metric_columns = st.columns(5)
    metric_columns[0].metric("ROUGE-1 F1", f"{scores['rouge1_f1'] * 100:.1f}%",
                             delta=f"{(scores['rouge1_f1'] - lead_scores['rouge1_f1']) * 100:+.1f} vs Lead-{snapshot['k']}")
    metric_columns[1].metric("ROUGE-2 F1", f"{scores['rouge2_f1'] * 100:.1f}%",
                             delta=f"{(scores['rouge2_f1'] - lead_scores['rouge2_f1']) * 100:+.1f} vs Lead-{snapshot['k']}")
    metric_columns[2].metric("ROUGE-L F1", f"{scores['rougeL_f1'] * 100:.1f}%",
                             delta=f"{(scores['rougeL_f1'] - lead_scores['rougeL_f1']) * 100:+.1f} vs Lead-{snapshot['k']}")
    metric_columns[3].metric("Summary words", f"{summary_words}", delta=f"{summary_words - len(reference.split()):+d} vs reference")
    metric_columns[4].metric("Compression", f"{compression:.1f}%")
else:
    metric_columns = st.columns(3)
    metric_columns[0].metric("Sentences extracted", f"{len(result.selected_indices)}")
    metric_columns[1].metric("Summary words", f"{summary_words}")
    metric_columns[2].metric("Compression", f"{compression:.1f}%")

st.write("")
summary_column, reference_column = st.columns(2, gap="large")
with summary_column:
    st.markdown("#### ✂️ Generated summary")
    st.markdown(f"<div class='summary-box'>{result.summary}</div>", unsafe_allow_html=True)
    st.download_button(
        "⬇️ Download summary (.txt)",
        data=result.summary,
        file_name="textrank_summary.txt",
        mime="text/plain",
    )
with reference_column:
    st.markdown("#### 🧑‍⚖️ Reference summary")
    if reference:
        st.markdown(f"<div class='reference-box'>{reference}</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='reference-box'>No reference summary provided — add one above to compute ROUGE.</div>",
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------- #
# Explainability tabs
# --------------------------------------------------------------------------- #
st.write("")
tab_scores, tab_graph, tab_similarity, tab_sentences = st.tabs(
    ["📊 Sentence ranking", "🕸️ Sentence graph", "🔥 Similarity matrix", "📝 Selected sentences"]
)

with tab_scores:
    figure = visualize.plot_sentence_scores(
        result.sentences,
        result.scores,
        result.selected_indices,
        path=None,
        title="PageRank score per sentence (top 14 shown)",
    )
    st.pyplot(figure, width="stretch")
    plt.close(figure)

with tab_graph:
    figure = visualize.plot_graph(
        result.graph,
        result.scores,
        result.selected_indices,
        path=None,
        title="Sentence similarity graph — central nodes are the most 'central' sentences",
    )
    st.pyplot(figure, width="stretch")
    plt.close(figure)

with tab_similarity:
    figure = visualize.plot_similarity_heatmap(
        result.similarity, path=None, title="TF-IDF cosine similarity between sentences"
    )
    st.pyplot(figure, width="stretch")
    plt.close(figure)

with tab_sentences:
    rows = [
        {
            "rank": rank + 1,
            "position": index + 1,
            "sentence": result.sentences[index],
            "score": round(float(result.scores[index]), 5),
        }
        for rank, index in enumerate(sorted(result.selected_indices, key=lambda i: -result.scores[i]))
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.divider()
st.markdown(
    "<p class='small-note'>TextRank (Mihalcea &amp; Tarau, 2004) • TF-IDF cosine graph • "
    "PageRank α = 0.85 • ROUGE via <code>rouge-score</code> with Porter stemming.</p>",
    unsafe_allow_html=True,
)
