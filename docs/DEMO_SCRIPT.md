# Demo video — 2½ minute recording plan

**Deliverable:** `deliverables/demo_video.mp4` · 1920×1080 · 30 fps · 2:20–2:40 long · clear voice-over or burned-in captions.

This plan is a single continuous screen recording. Rehearse once, then record in one take; small pauses are fine and can be trimmed.

---

## Before recording

- [ ] Fill `report/student.json` and rebuild the PDFs if the cover pages will appear on camera.
- [ ] Close email, chat and notifications; hide bookmarks; use a clean browser profile at 100% zoom.
- [ ] Set the display to 1920×1080 and disable night-light/colour shifting.
- [ ] Increase terminal and browser font size so text is readable at 1080p.
- [ ] Start the app in a terminal and keep that terminal visible for the first shot:

  ```powershell
  cd C:\Users\srira\textrank-summarizer
  .\.venv\Scripts\Activate.ps1
  streamlit run src/app.py
  ```

- [ ] Open `http://localhost:8501/?demo=1` in the browser. The first article is pre-run, so the summary, ROUGE cards and tabs are visible immediately.
- [ ] Open the report PDF at the results table and the notebook in a second tab.
- [ ] Do a 15-second test recording and check microphone level and text sharpness.

---

## Shot list and narration

### 0:00–0:12 — Title

**On screen:** The app hero panel (`TextRank Extractive Summarizer`).

**Say:**

> "Hi, I'm [Name]. This is my NLP mini project: an extractive text summariser based on TextRank, evaluated with ROUGE on CNN/DailyMail."

### 0:12–0:30 — Problem and approach

**On screen:** Scroll slightly to the input article, then back to the top.

**Say:**

> "The goal is to shorten a news article while keeping its most important sentences. TextRank treats every sentence as a node in a similarity graph, then uses PageRank to find the sentences that are most central to the document. No training and no GPU are required."

### 0:30–1:00 — Generate a summary

**On screen:** With `?demo=1`, the first article is already summarised.

1. Point at the article on the left.
2. Point at the generated summary on the right.
3. Point at the five metric cards: ROUGE-1, ROUGE-2, ROUGE-L, summary words, compression.

**Say:**

> "Here is a CNN/DailyMail article and its human-written reference. The app has extracted three sentences — about eighteen percent of the source — and scored them live against the reference. The cards show ROUGE-1, ROUGE-2 and ROUGE-L F1, plus the delta against the Lead-3 baseline."

**Action:** Change **Summary length** to 5, then back to 3. Optionally move **Damping factor α** from 0.85 to 0.80.

**Say:**

> "The sidebar exposes the method, summary length, damping factor and similarity threshold, so the behaviour can be explored interactively."

### 1:00–1:25 — Explainability tabs

**On screen:** Click each tab in turn.

1. **📊 Sentence ranking** — PageRank score per sentence.
2. **🕸️ Sentence graph** — nodes sized/coloured by score.
3. **🔥 Similarity matrix** — pairwise TF-IDF cosine similarity.

**Say:**

> "The ranking tab shows the PageRank score of every sentence. The graph tab makes the algorithm visible: highly scored sentences sit at the centre of the similarity network, not necessarily at the top of the article. The heatmap shows the underlying pairwise cosine similarities."

### 1:25–1:55 — Benchmark results

**On screen:** Switch to the report PDF, Results section (`Table 4`).

**Say:**

> "I evaluated on 150 CNN/DailyMail test articles at three sentences. TextRank reaches 28.7 ROUGE-1, 10.1 ROUGE-2 and 19.8 ROUGE-L. It significantly beats a random extractor — plus 1.7 ROUGE-1 points, p equals 0.015 — but it trails Lead-3, which scores 30.6. That is expected on news: the first sentences already contain the main facts, and the human highlights follow the same inverted-pyramid convention. The report analyses this honestly, including per-document wins and losses."

**Action:** Scroll to the length-sweep figure.

**Say:**

> "A length sweep from two to eight sentences and a redundancy-filter ablation are also reported."

### 1:55–2:15 — Notebook and repository

**On screen:** Switch to the executed notebook, scroll through the ROUGE table and one figure, then briefly show the repository tree in the terminal or GitHub.

**Say:**

> "The repository contains the executed notebook, the reusable modules, the experiment runner, the report source and this Streamlit app. One command reruns the full evaluation and regenerates every table and figure."

### 2:15–2:30 — Close

**On screen:** Back to the app summary.

**Say:**

> "All results are reproducible offline from the cached dataset. Thank you for watching."

---

## Optional on-screen captions

If recording without a microphone, burn these captions into the edit:

1. "TextRank: sentences = nodes, similarity = edges"
2. "PageRank finds the most central sentences"
3. "Extractive: every output sentence comes from the source"
4. "150 CNN/DailyMail articles · ROUGE-1/2/L"
5. "Beats random (p = 0.015) · trails Lead-3 (p = 0.026)"
6. "Executed notebook + report + app included"

---

## Recording and export

**Windows Game Bar (simplest):** `Win + G` → Capture → Start recording. Records the active window; stop with `Win + Alt + R`.

**OBS Studio (more control):** Display Capture → 1920×1080, 30 fps → record to MP4. Use a noise-suppression filter on the microphone.

**After recording:**

- [ ] Trim dead time at the start/end; keep the final length between 2:00 and 3:00.
- [ ] Check that no notification, password or personal file is visible.
- [ ] Verify audio is audible and synced; normalise peaks to about −3 dB.
- [ ] Export H.264 MP4, 1080p, and name it `demo_video.mp4`.
- [ ] Place it in `deliverables/demo_video.mp4` (ignored by Git) and upload it to the submission portal or as an unlisted YouTube link.
- [ ] Add the video link to the README or submission form if required.

## Quick reference — commands used on camera

```powershell
cd C:\Users\srira\textrank-summarizer
.\.venv\Scripts\Activate.ps1
streamlit run src/app.py
# browser: http://localhost:8501/?demo=1

python scripts/run_experiment.py       # ~10 s, rewrites results/ and figures
python scripts/build_report.py         # rebuild the report PDF
```
