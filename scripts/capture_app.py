"""Capture high-resolution screenshots of the running Streamlit demo.

Usage:
    streamlit run src/app.py --server.headless true --server.port 8501
    python scripts/capture_app.py

Requires Playwright's Chromium: ``python -m playwright install chromium``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "figures"

EXPAND_JS = """() => {
  for (const selector of ['[data-testid="stMain"]', '[data-testid="stAppViewContainer"]', '[data-testid="stApp"]']) {
    const element = document.querySelector(selector);
    if (element) {
      element.style.height = 'auto';
      element.style.maxHeight = 'none';
      element.style.overflow = 'visible';
    }
  }
  document.body.style.overflow = 'visible';
}"""


def _expand_page(page: Page) -> None:
    page.evaluate(EXPAND_JS)
    page.wait_for_timeout(600)


def capture(url: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1440, "height": 1000},
            device_scale_factor=1.5,
        )
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_selector("text=Generated summary", timeout=180_000)
        page.wait_for_timeout(3_000)

        # 1. Results view: metrics, generated summary and reference side by side.
        page.locator('[data-testid="stMain"]').evaluate("element => element.scrollTo(0, 640)")
        page.wait_for_timeout(800)
        overview = out_dir / "app_overview.png"
        page.screenshot(path=str(overview), full_page=False)
        saved.append(overview)
        print(f"saved {overview}")

        # 2. Full page: every section, used in the screenshots compilation.
        _expand_page(page)
        full = out_dir / "app_full_page.png"
        page.screenshot(path=str(full), full_page=True)
        saved.append(full)
        print(f"saved {full}")

        # 3. Explainability tabs.
        for tab_label, filename in (
            ("📊 Sentence ranking", "app_sentence_ranking.png"),
            ("🕸️ Sentence graph", "app_sentence_graph.png"),
            ("🔥 Similarity matrix", "app_similarity_matrix.png"),
            ("📝 Selected sentences", "app_selected_sentences.png"),
        ):
            try:
                page.get_by_role("tab", name=tab_label).click(timeout=15_000)
                page.wait_for_timeout(1_200)
                target = out_dir / filename
                page.screenshot(path=str(target), full_page=True)
                saved.append(target)
                print(f"saved {target}")
            except Exception as exc:  # pragma: no cover - UI wording may change
                print(f"skipped {tab_label}: {exc}", file=sys.stderr)

        browser.close()
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture Streamlit screenshots with Playwright.")
    parser.add_argument("--url", default="http://localhost:8501/?demo=1")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    capture(args.url, args.out)
