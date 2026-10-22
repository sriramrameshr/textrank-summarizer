"""Download (or refresh) the cached CNN/DailyMail evaluation slice.

    python scripts/fetch_data.py [--n 150]

The notebook, experiments and Streamlit app read the cached JSON, so this only
needs to be run once (or when you want a different sample size).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import DEFAULT_PATH, fetch_sample  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the CNN/DailyMail evaluation slice.")
    parser.add_argument("--n", type=int, default=150, help="number of test articles (default: 150)")
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH, help="output JSON path")
    args = parser.parse_args()
    fetch_sample(n_documents=args.n, out_path=args.out)


if __name__ == "__main__":
    main()
