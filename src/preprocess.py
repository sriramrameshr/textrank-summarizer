"""Text preprocessing: sentence segmentation, tokenisation, stopwords, stemming."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

BASE_DIR = Path(__file__).resolve().parent.parent
NLTK_DATA_DIR = BASE_DIR / "data" / "nltk_data"
if str(NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))

# Standard English stopword list, extended with a few news-boilerplate words that
# carry no discriminative signal for sentence similarity.
STOPWORDS: frozenset[str] = frozenset(ENGLISH_STOP_WORDS) | {
    "also",
    "said",
    "says",
    "would",
    "could",
    "one",
    "two",
    "three",
    "new",
    "like",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:[.,]\d+)*")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_FALLBACK_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_stemmer = PorterStemmer()


@dataclass(frozen=True)
class PreprocessedDocument:
    """A document represented as sentences plus their content tokens."""

    sentences: tuple[str, ...]
    tokens: tuple[tuple[str, ...], ...]

    @property
    def num_sentences(self) -> int:
        return len(self.sentences)


def _normalise_sentence(sentence: str) -> str:
    sentence = _WHITESPACE_RE.sub(" ", sentence).strip()
    # Normalise curly quotes so ROUGE comparisons are not penalised by encoding.
    return (
        sentence.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


@lru_cache(maxsize=1)
def _use_nltk_tokenizer() -> bool:
    try:
        nltk.data.find("tokenizers/punkt_tab/english")
        return True
    except LookupError:
        return False


def split_sentences(
    text: str,
    min_tokens: int = 4,
    dedupe: bool = True,
    max_sentences: int | None = None,
) -> list[str]:
    """Split ``text`` into clean, order-preserving sentences.

    Uses NLTK's Punkt tokenizer when the bundled model is available and falls back
    to a regular-expression splitter otherwise, so the pipeline never crashes on a
    fresh machine without network access.
    """
    text = _WHITESPACE_RE.sub(" ", text or "").strip()
    if not text:
        return []

    if _use_nltk_tokenizer():
        raw_sentences = nltk.sent_tokenize(text)
    else:  # pragma: no cover - only on machines without the bundled model
        raw_sentences = _SENTENCE_FALLBACK_RE.split(text)

    sentences: list[str] = []
    seen: set[str] = set()
    for raw in raw_sentences:
        sentence = _normalise_sentence(raw)
        if not sentence:
            continue
        if len(TOKEN_RE.findall(sentence)) < min_tokens:
            continue
        key = sentence.lower()
        if dedupe and key in seen:
            continue
        seen.add(key)
        sentences.append(sentence)
        if max_sentences is not None and len(sentences) >= max_sentences:
            break
    return sentences


def tokenize(text: str, stem: bool = True) -> list[str]:
    """Lowercase, filter stopwords/short tokens and (optionally) Porter-stem."""
    tokens: list[str] = []
    for match in TOKEN_RE.findall(text):
        token = match.lower().strip("'-")
        if len(token) < 2 or token in STOPWORDS:
            continue
        tokens.append(_stemmer.stem(token) if stem else token)
    return tokens


def preprocess(text: str, min_tokens: int = 4) -> PreprocessedDocument:
    """Full pipeline: sentence split -> per-sentence content tokens."""
    sentences = split_sentences(text, min_tokens=min_tokens)
    kept_sentences: list[str] = []
    token_lists: list[tuple[str, ...]] = []
    for sentence in sentences:
        tokens = tuple(tokenize(sentence))
        if not tokens:  # a sentence made only of stopwords cannot be ranked
            continue
        kept_sentences.append(sentence)
        token_lists.append(tokens)
    return PreprocessedDocument(tuple(kept_sentences), tuple(token_lists))


def count_words(text: str) -> int:
    """Whitespace word count used for compression-ratio statistics."""
    return len((text or "").split())
