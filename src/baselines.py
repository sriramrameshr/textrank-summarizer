"""Extractive baselines used to contextualise TextRank's ROUGE scores."""

from __future__ import annotations

import random

from .preprocess import PreprocessedDocument, preprocess


def lead_k(text: str, k: int = 3, doc: PreprocessedDocument | None = None) -> dict:
    """Lead-k baseline: the first k sentences of the document."""
    document = doc or preprocess(text)
    indices = list(range(min(k, document.num_sentences)))
    return {
        "summary": " ".join(document.sentences[i] for i in indices),
        "selected_indices": indices,
        "sentences": list(document.sentences),
        "method": f"Lead-{k}",
    }


def random_k(text: str, k: int = 3, seed: int = 42, doc: PreprocessedDocument | None = None) -> dict:
    """Random-k baseline: k sentences sampled uniformly without replacement."""
    document = doc or preprocess(text)
    k = min(k, document.num_sentences)
    indices = sorted(random.Random(seed).sample(range(document.num_sentences), k))
    return {
        "summary": " ".join(document.sentences[i] for i in indices),
        "selected_indices": indices,
        "sentences": list(document.sentences),
        "method": f"Random-{k}",
    }
