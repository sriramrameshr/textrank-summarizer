"""TextRank: TF-IDF sentence similarity graph + PageRank sentence ranking."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocess import PreprocessedDocument, preprocess


@dataclass
class TextRankResult:
    """Everything needed to display, evaluate and explain one summary."""

    summary: str
    selected_indices: list[int]
    scores: np.ndarray
    similarity: np.ndarray
    graph: nx.Graph
    sentences: list[str] = field(default_factory=list)
    method: str = "TextRank"

    @property
    def num_sentences(self) -> int:
        return len(self.sentences)


def tfidf_similarity(tokens: tuple[tuple[str, ...], ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return (L2-normalised TF-IDF matrix, cosine-similarity matrix).

    The TF-IDF matrix uses sublinear term frequency and L2 normalisation, so the
    dot product of two rows is exactly the cosine similarity of the sentences.
    """
    n = len(tokens)
    if n == 0:
        empty = np.zeros((0, 0), dtype=float)
        return empty, empty

    vectorizer = TfidfVectorizer(
        analyzer=lambda token_list: list(token_list),
        lowercase=False,
        sublinear_tf=True,
        norm="l2",
    )
    try:
        matrix = vectorizer.fit_transform(tokens)
    except ValueError:  # empty vocabulary: every sentence was all stopwords
        matrix = None

    if matrix is None or matrix.shape[1] == 0:
        return np.zeros((n, 0), dtype=float), np.zeros((n, n), dtype=float)

    similarity = (matrix @ matrix.T).toarray().astype(float)
    similarity = np.clip(similarity, 0.0, 1.0)
    np.fill_diagonal(similarity, 0.0)
    return matrix.toarray(), similarity


def build_graph(similarity: np.ndarray, threshold: float = 0.0) -> nx.Graph:
    """Convert a similarity matrix into an undirected weighted graph."""
    graph = nx.from_numpy_array(similarity)
    graph.remove_edges_from(
        [(u, v) for u, v, weight in graph.edges(data="weight") if weight <= threshold]
    )
    return graph


def pagerank_scores(
    graph: nx.Graph,
    alpha: float = 0.85,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> np.ndarray:
    """Run PageRank and return scores in sentence order (uniform if no edges)."""
    n = graph.number_of_nodes()
    if n == 0:
        return np.zeros(0, dtype=float)
    if graph.number_of_edges() == 0:
        return np.full(n, 1.0 / n, dtype=float)

    raw = nx.pagerank(graph, alpha=alpha, max_iter=max_iter, tol=tol, weight="weight")
    return np.array([raw[i] for i in range(n)], dtype=float)


def rank_sentences(scores: np.ndarray) -> list[int]:
    """Rank sentence indices by descending score, breaking ties by position."""
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


def select_sentences(
    scores: np.ndarray,
    k: int,
    similarity: np.ndarray | None = None,
    redundancy_threshold: float | None = None,
) -> list[int]:
    """Pick the top-k sentences, optionally suppressing near-duplicates.

    ``redundancy_threshold`` enables a Maximal Marginal Relevance style filter: a
    candidate is skipped when its cosine similarity to an already selected
    sentence exceeds the threshold. This is used only as an ablation; the default
    TextRank configuration is the faithful, unfiltered algorithm.
    """
    ranked = rank_sentences(scores)
    k = max(0, min(k, len(ranked)))
    if not k:
        return []

    if redundancy_threshold is None or similarity is None or similarity.size == 0:
        return sorted(ranked[:k])

    selected: list[int] = []
    for index in ranked:
        if len(selected) == k:
            break
        if selected and max(similarity[index, chosen] for chosen in selected) > redundancy_threshold:
            continue
        selected.append(index)

    # Never return fewer than k sentences because of the filter.
    if len(selected) < k:
        for index in ranked:
            if index not in selected:
                selected.append(index)
            if len(selected) == k:
                break
    return sorted(selected)


def summarize(
    text: str,
    k: int = 3,
    alpha: float = 0.85,
    threshold: float = 0.0,
    redundancy_threshold: float | None = None,
    doc: PreprocessedDocument | None = None,
) -> TextRankResult:
    """Summarise raw ``text`` with TextRank and return the top-``k`` sentences."""
    document = doc or preprocess(text)
    _, similarity = tfidf_similarity(document.tokens)
    graph = build_graph(similarity, threshold=threshold)
    scores = pagerank_scores(graph, alpha=alpha)
    selected = select_sentences(scores, k, similarity, redundancy_threshold)
    summary = " ".join(document.sentences[i] for i in selected)
    return TextRankResult(
        summary=summary,
        selected_indices=selected,
        scores=scores,
        similarity=similarity,
        graph=graph,
        sentences=list(document.sentences),
    )


def sentence_score_table(result: TextRankResult) -> list[dict]:
    """Convenience rows for a notebook dataframe: sentence, score, selected."""
    rows = []
    for i, sentence in enumerate(result.sentences):
        rows.append(
            {
                "sentence_id": i + 1,
                "sentence": sentence,
                "pagerank_score": round(float(result.scores[i]), 6),
                "selected": i in result.selected_indices,
            }
        )
    return rows
