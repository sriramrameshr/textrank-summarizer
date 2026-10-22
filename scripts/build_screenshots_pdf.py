"""Compile the output screenshots into a single PDF (deliverable 3).

    python scripts/build_screenshots_pdf.py

The PDF is assembled as styled HTML and printed with headless Microsoft Edge.
"""

from __future__ import annotations

import html
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "results" / "figures"
DELIVERABLES = ROOT / "deliverables"
STUDENT = json.loads((ROOT / "report" / "student.json").read_text(encoding="utf-8"))
GUIDE = STUDENT.get("guide", "").strip()
GUIDE_ROW = f'<tr><td>Guide</td><td>{html.escape(GUIDE)}</td></tr>' if GUIDE else ""
RESULTS = pd.read_csv(ROOT / "results" / "results_table_k3.csv")
LOG_TEXT = (ROOT / "results" / "experiment_log.txt").read_text(encoding="utf-8", errors="replace")
EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

DISPLAY = {"TextRank": "TextRank", "TextRank + MMR": "TextRank + MMR", "Lead-k": "Lead-3", "Random-k": "Random-3"}


def split_image(path: Path, parts: int, out_dir: Path) -> list[Path]:
    """Split a tall screenshot into readable page-sized slices."""
    image = Image.open(path)
    width, height = image.size
    if height <= 2200:
        return [path]
    slice_height = height // parts
    outputs = []
    for index in range(parts):
        top = index * slice_height
        bottom = height if index == parts - 1 else (index + 1) * slice_height
        target = out_dir / f"{path.stem}_part{index + 1}.png"
        image.crop((0, top, width, bottom)).save(target)
        outputs.append(target)
    return outputs


def results_html() -> str:
    rows = []
    for _, row in RESULTS.iterrows():
        method = DISPLAY.get(row["Method"], row["Method"])
        rows.append(
            "<tr>"
            f"<td>{html.escape(method)}</td>"
            f"<td>{html.escape(str(row['Method']))}</td>"
            f"<td>{row['Documents']}</td>"
            f"<td>{html.escape(str(row['ROUGE-1']))}</td>"
            f"<td>{html.escape(str(row['ROUGE-2']))}</td>"
            f"<td>{html.escape(str(row['ROUGE-L']))}</td>"
            f"<td>{html.escape(str(row['Avg. words']))}</td>"
            f"<td>{html.escape(str(row['Compression']))}</td>"
            "</tr>"
        )
    return (
        "<table class='results'><thead><tr>"
        "<th>Method</th><th>Run label</th><th>N</th><th>ROUGE-1 F1</th><th>ROUGE-2 F1</th>"
        "<th>ROUGE-L F1</th><th>Words</th><th>Compression</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def shot(title: str, image_path: Path, caption: str, section: str, tall: bool = False) -> str:
    css_class = "shot tall" if tall else "shot"
    return f"""
<section class="{css_class}">
  <div class="section-label">{html.escape(section)}</div>
  <h2>{html.escape(title)}</h2>
  <div class="image-frame"><img src="{image_path.as_uri()}"></div>
  <p class="caption">{caption}</p>
</section>
"""


def build(temp_dir: Path) -> str:
    pages: list[str] = []

    pages.append(
        f"""
<section class="cover">
  <div class="cover-institution">{html.escape(STUDENT["institution"])}</div>
  <div class="cover-rule"></div>
  <div class="cover-kicker">MINI PROJECT · DELIVERABLE 3</div>
  <h1>Output Screenshots</h1>
  <div class="cover-subtitle">Extractive Text Summarisation with TextRank<br>TF-IDF sentence graph · PageRank · ROUGE-1/2/L evaluation</div>
  <table class="cover-table">
    <tr><td>Student</td><td><strong>{html.escape(STUDENT["student_name"])}</strong></td></tr>
    <tr><td>Roll number</td><td>{html.escape(STUDENT["roll_number"])}</td></tr>
    <tr><td>Course</td><td>{html.escape(STUDENT["course"])} · {html.escape(STUDENT["semester"])}</td></tr>
    {GUIDE_ROW}
  </table>
  <div class="cover-contents">
    <strong>Contents</strong>
    <ol>
      <li>Streamlit application — summary, ROUGE metrics and explainability views</li>
      <li>Jupyter Notebook — preprocessing, outputs and result tables</li>
      <li>Evaluation charts and ROUGE results table</li>
      <li>Terminal output of the full experiment</li>
    </ol>
  </div>
  <div class="cover-date">{html.escape(STUDENT["submission_date"])}</div>
</section>
"""
    )

    sections = [
        (
            "1. Interactive demo (Streamlit)",
            [
                ("Generated summary and live ROUGE", "app_overview.png",
                 "Figure 1. The Streamlit application after summarising a CNN/DailyMail article: generated summary on the left, human reference on the right, and ROUGE-1, ROUGE-2 and ROUGE-L F1 cards with the delta against the Lead-3 baseline.", False),
                ("Full application page", "app_full_page.png",
                 "Figure 2. Complete application page, including the input panels, summary outputs and explainability tabs.", True),
                ("PageRank sentence ranking", "app_sentence_ranking.png",
                 "Figure 3. PageRank score for each sentence. Bars in dark blue are selected in the summary; the top-14 sentences are shown.", True),
                ("Sentence similarity graph", "app_sentence_graph.png",
                 "Figure 4. Spring-layout visualisation of the sentence graph. Node size and colour encode PageRank score; hub sentences are the ones TextRank selects.", True),
                ("Similarity matrix", "app_similarity_matrix.png",
                 "Figure 5. TF-IDF cosine similarity between sentence pairs for the selected article.", True),
                ("Selected sentences table", "app_selected_sentences.png",
                 "Figure 6. Selected sentences with their document position, PageRank score and extraction rank.", True),
            ],
        ),
        (
            "2. Notebook outputs",
            [
                ("Notebook title and objective", "nb_title.png",
                 "Figure 7. The executed notebook opens with the project title, student details and objective.", False),
                ("Preprocessing output", "nb_preprocessing.png",
                 "Figure 8. Sentence segmentation and tokenisation output for the first article, including the number of sentences and content tokens.", False),
                ("Corpus-level ROUGE results", "nb_results.png",
                 "Figure 9. The notebook computes ROUGE for every method on all 150 documents and renders the comparison table inline.", False),
                ("Length sweep table and figure", "nb_length.png",
                 "Figure 10. ROUGE-1 F1 as the summary length varies, produced during notebook execution.", False),
            ],
        ),
    ]

    for section_title, entries in sections:
        for title, filename, caption, tall in entries:
            path = FIGURES / filename
            if not path.exists():
                continue
            if tall:
                parts = split_image(path, 2, temp_dir)
                for index, part in enumerate(parts):
                    suffix = f" (part {index + 1} of {len(parts)})" if len(parts) > 1 else ""
                    part_caption = caption if index == 0 else f"{caption} (continued)"
                    pages.append(shot(title + suffix, part, part_caption, section_title, tall=True))
            else:
                pages.append(shot(title, path, caption, section_title))

    pages.append(
        f"""
<section class="shot">
  <div class="section-label">3. Evaluation results</div>
  <h2>ROUGE results table (k = 3, n = 150)</h2>
  {results_html()}
  <p class="caption">Table 1. Mean ROUGE F1 with 95% bootstrap confidence intervals, saved by the experiment runner as <code>results/results_table_k3.csv</code>. The table shows the canonical run labels (<code>Lead-k</code>, <code>Random-k</code>); at k = 3 these correspond to Lead-3 and Random-3.</p>
</section>
"""
    )

    chart_entries = [
        ("ROUGE F1 by method", "rouge_by_method.png",
         "Figure 11. ROUGE-1, ROUGE-2 and ROUGE-L F1 for TextRank, the redundancy-filtered variant, Lead-3 and Random-3.", "3. Evaluation results"),
        ("Per-document comparison against Lead-3", "delta_vs_lead3.png",
         "Figure 12. Distribution of per-document ROUGE-1 F1 differences between TextRank and Lead-3, with the mean difference marked.", "3. Evaluation results"),
        ("Effect of summary length", "rouge_by_length.png",
         "Figure 13. ROUGE-1 F1 versus summary length with 95% confidence bands for all four methods.", "3. Evaluation results"),
        ("Redundancy-filter ablation", "ablation_mmr.png",
         "Figure 14. Ablation at k = 3: plain TextRank versus TextRank with the MMR-style redundancy filter.", "3. Evaluation results"),
        ("Sentence graph of an example article", "sentence_graph.png",
         "Figure 15. Static rendering of the TextRank sentence graph used in the report; node colour and size encode PageRank score.", "3. Evaluation results"),
    ]
    for title, filename, caption, section in chart_entries:
        path = FIGURES / filename
        if path.exists():
            pages.append(shot(title, path, caption, section))

    log_lines = "\n".join(LOG_TEXT.splitlines()[-24:])
    pages.append(
        f"""
<section class="shot terminal-page">
  <div class="section-label">4. Reproducibility</div>
  <h2>Terminal output of the full evaluation</h2>
  <pre class="terminal">$ python scripts/run_experiment.py
{html.escape(log_lines)}</pre>
  <p class="caption">Figure 16. Console output of the experiment runner: 3,600 summaries (150 documents × 6 lengths × 4 methods) evaluated in a few seconds, followed by the ROUGE table that is written to <code>results/</code>.</p>
  <div class="file-tree">
    <div class="section-label">Repository snapshot</div>
    <pre class="tree">textrank-summarizer/
├── data/cnn_dailymail_sample.json     150 CNN/DailyMail articles + references
├── notebooks/textrank_evaluation.ipynb  executed notebook (27 cells)
├── src/                                pipeline, baselines, ROUGE, app
├── scripts/                            data, experiments, report, screenshots
├── results/                            CSVs, metrics, figures, metadata
├── report/                             report.md, report.html, student.json
└── deliverables/                       project_report.pdf, this PDF</pre>
  </div>
</section>
"""
    )

    css = """
@page { size: A4; margin: 16mm 15mm 14mm 15mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; color: #1d2b3a; margin: 0; font-size: 10pt; }
h1 { font-size: 30pt; color: #16324f; margin: 12px 0; }
h2 { font-size: 13.5pt; color: #16324f; margin: 4px 0 10px; page-break-after: avoid; }
.cover { min-height: 250mm; display: flex; flex-direction: column; justify-content: center; text-align: center; page-break-after: always; }
.cover-institution { font-size: 15pt; font-weight: 700; color: #16324f; }
.cover-rule { width: 55%; height: 3px; background: #1f4e79; margin: 14px auto 24px; }
.cover-kicker { font-size: 10pt; letter-spacing: 3px; color: #c55a11; font-weight: 700; }
.cover-subtitle { font-size: 12pt; color: #4a5568; line-height: 1.5; margin: 6px auto 26px; }
.cover-table { width: 110mm; margin: 0 auto; border-collapse: collapse; font-size: 10.5pt; }
.cover-table td { border: none; border-bottom: 1px solid #e3eaf2; padding: 7px 5px; text-align: left; }
.cover-table td:first-child { color: #5b6b7b; width: 38%; }
.cover-contents { width: 128mm; margin: 24px auto 0; text-align: left; font-size: 9.6pt; color: #33475b; }
.cover-contents ol { margin: 6px 0 0 18px; padding: 0; }
.cover-contents li { margin: 3px 0; }
.cover-date { margin-top: 22px; color: #16324f; font-weight: 600; }
.shot { page-break-after: always; }
.shot:last-of-type { page-break-after: auto; }
.section-label { display: inline-block; font-size: 7.6pt; letter-spacing: 1.4px; text-transform: uppercase; color: #ffffff; background: #1f4e79; padding: 2px 8px; border-radius: 3px; margin-bottom: 8px; }
.image-frame { text-align: center; }
.image-frame img { max-width: 100%; max-height: 215mm; border: 1px solid #d7e1eb; border-radius: 5px; }
.tall .image-frame img { max-height: 215mm; }
.caption { font-size: 8.5pt; font-style: italic; color: #4a5568; margin-top: 7px; text-align: left; }
table.results { width: 100%; border-collapse: collapse; font-size: 9pt; margin: 10px 0 8px; }
table.results th { background: #16324f; color: #fff; padding: 6px 6px; text-align: left; border: 1px solid #16324f; }
table.results td { padding: 5px 6px; border: 1px solid #d7e1eb; }
table.results tr:nth-child(even) td { background: #f6f9fc; }
.terminal { background: #0f1b26; color: #d8e4ef; font-family: Consolas, "Courier New", monospace; font-size: 8.3pt; line-height: 1.45; padding: 12px 14px; border-radius: 6px; white-space: pre-wrap; page-break-inside: avoid; }
.tree { background: #f5f8fb; border-left: 3px solid #1f4e79; font-family: Consolas, "Courier New", monospace; font-size: 8.3pt; line-height: 1.5; padding: 10px 12px; white-space: pre-wrap; }
.file-tree { margin-top: 16px; }
code { font-family: Consolas, monospace; font-size: 8.6pt; background: #f1f4f7; padding: 1px 3px; border-radius: 3px; }
"""

    body = "".join(pages)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Output Screenshots — TextRank Summarizer</title>
<style>{css}</style></head>
<body>{body}</body></html>
"""


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    edge = next((path for path in EDGE_CANDIDATES if path.exists()), None)
    if edge is None:
        raise RuntimeError("Microsoft Edge was not found; open the HTML file and print to PDF.")
    command = [
        str(edge),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--print-to-pdf-no-header",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=20000",
        "--allow-file-access-from-files",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=240)
    if not pdf_path.exists() or pdf_path.stat().st_size < 20_000:
        raise RuntimeError(f"PDF generation failed: {pdf_path}")


def main() -> None:
    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)
        document = build(temp_dir)
        html_path = temp_dir / "screenshots.html"
        html_path.write_text(document, encoding="utf-8")
        pdf_path = DELIVERABLES / "output_screenshots.pdf"
        render_pdf(html_path, pdf_path)
        print(f"wrote {pdf_path} ({pdf_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
