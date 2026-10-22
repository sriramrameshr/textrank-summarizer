"""Dataset loading utilities for the TextRank summarizer.

The project uses a fixed, reproducible slice of the CNN/DailyMail test split
(config ``3.0.0``), fetched from the HuggingFace *datasets-server* REST API.
The slice is cached as JSON so that the notebook, the experiments and the
Streamlit app all run fully offline afterwards.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

DATASET_NAME = "abisee/cnn_dailymail"
DATASET_CONFIG = "3.0.0"
DATASET_SPLIT = "test"
API_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "cnn_dailymail_sample.json"

_WHITESPACE_RE = re.compile(r"\s+")
_LEADING_TAG_RE = re.compile(r"^\s*\((?:CNN|cnn)\)\s*")
_CNN_TAG_RE = re.compile(r"\(CNN\)\s*")


def clean_article(text: str) -> str:
    """Normalise an article: strip wire-service tags and collapse whitespace."""
    text = _LEADING_TAG_RE.sub("", text or "")
    text = _CNN_TAG_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_summary(text: str) -> str:
    """Normalise a reference summary (one sentence per line in CNN/DailyMail)."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return _WHITESPACE_RE.sub(" ", " ".join(lines)).strip()


def _fetch_rows(offset: int, length: int, retries: int = 5) -> list[dict]:
    params = {
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "offset": offset,
        "length": length,
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                payload = json.load(response)
            return payload["rows"]
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            import time

            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Could not download rows at offset {offset}: {last_error}")


def fetch_sample(n_documents: int = 150, out_path: Path | str = DEFAULT_PATH) -> dict:
    """Download ``n_documents`` articles + highlight summaries and cache them."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    documents: list[dict] = []
    batch = 100
    for offset in range(0, n_documents, batch):
        length = min(batch, n_documents - offset)
        rows = _fetch_rows(offset, length)
        for item in rows:
            row = item["row"]
            article = clean_article(row.get("article", ""))
            summary = clean_summary(row.get("highlights", ""))
            if not article or not summary:
                continue
            documents.append(
                {
                    "id": str(row.get("id", f"doc-{offset + len(documents):04d}")),
                    "article": article,
                    "reference_summary": summary,
                    "source": "cnn_dailymail/3.0.0/test",
                }
            )
        print(f"downloaded {len(documents)} documents")

    payload = {
        "name": "CNN/DailyMail",
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "license": "Apache-2.0 (dataset card: abisee/cnn_dailymail)",
        "source": f"https://huggingface.co/datasets/{DATASET_NAME}",
        "retrieved_via": API_URL,
        "requested": n_documents,
        "count": len(documents),
        "documents": documents,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {len(documents)} documents -> {out_path}")
    return payload


def load_sample(path: Path | str = DEFAULT_PATH) -> dict:
    """Load the cached dataset, with a helpful error if it is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python scripts/fetch_data.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def get_documents(path: Path | str = DEFAULT_PATH) -> list[dict]:
    """Return the list of documents from the cached dataset."""
    return load_sample(path)["documents"]


if __name__ == "__main__":
    fetch_sample()
