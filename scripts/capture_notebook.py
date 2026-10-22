"""Capture the notebook screenshots used in ``output_screenshots.pdf``.

    python scripts/capture_notebook.py

Pipeline:
1. Export ``notebooks/textrank_evaluation.ipynb`` to a self-contained HTML file
   with nbconvert (images are embedded, so the page works offline).
2. Open it in Playwright Chromium at a 1008x805 viewport with a 2x device scale,
   which produces 2016x1610 screenshots.
3. Scroll to the title and to the three analysis sections and save the viewport
   as ``results/figures/nb_*.png``.

Requires Playwright's Chromium build (install once):
    python -m playwright install chromium
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "textrank_evaluation.ipynb"
FIGURES = ROOT / "results" / "figures"

# (output filename, heading text or None for the top of the page)
SHOTS: list[tuple[str, str | None]] = [
    ("nb_title.png", None),
    ("nb_preprocessing.png", "Text preprocessing"),
    ("nb_results.png", "Corpus-level ROUGE results"),
    ("nb_length.png", "Length sweep, ablation and error analysis"),
]

VIEWPORT = {"width": 1008, "height": 805}
SCALE = 2


def export_html(destination: Path) -> Path:
    exporter = HTMLExporter(template_name="lab")
    body, _ = exporter.from_filename(str(NOTEBOOK))
    destination.write_text(body, encoding="utf-8")
    return destination


def capture(html_path: Path) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    uri = html_path.as_uri()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=SCALE)
        page.goto(uri, wait_until="load")
        page.wait_for_timeout(1500)  # fonts, MathJax and embedded images

        for filename, heading in SHOTS:
            if heading is None:
                page.evaluate("window.scrollTo(0, 0)")
            else:
                locator = page.locator("h1, h2, h3", has_text=heading).first
                locator.wait_for(state="attached", timeout=30_000)
                locator.evaluate(
                    "el => el.scrollIntoView({block: 'start', inline: 'nearest'})"
                )
                page.evaluate("window.scrollBy(0, -30)")
            page.wait_for_timeout(400)

            out = FIGURES / filename
            page.screenshot(path=str(out))
            print(f"wrote {out}")

        browser.close()


def main() -> int:
    if not NOTEBOOK.exists():
        print(f"Notebook not found: {NOTEBOOK}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="textrank-nb-") as tmp:
        html_path = export_html(Path(tmp) / "textrank_evaluation.html")
        capture(html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
