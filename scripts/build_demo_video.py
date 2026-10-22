"""Build the narrated demo video from the project's own screenshots and charts.

    python scripts/build_demo_video.py

The script is fully offline after the dependencies are installed:

1. renders 1920x1080 slides with Pillow;
2. synthesises the narration per scene with Windows SAPI (Zira);
3. encodes each slide with ffmpeg (via imageio-ffmpeg) and concatenates them;
4. muxes the narration and writes ``deliverables/demo_video.mp4``.

Windows only (System.Speech). If you prefer your own voice, record the script in
``docs/DEMO_SCRIPT.md`` and replace the output file.
"""

from __future__ import annotations

import math
import json
import random
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "results" / "figures"
DELIVERABLES = ROOT / "deliverables"
BUILD = ROOT / "build" / "demo_video"
OUTPUT = DELIVERABLES / "demo_video.mp4"
STUDENT = json.loads((ROOT / "report" / "student.json").read_text(encoding="utf-8"))

W, H = 1920, 1080
FPS = 30
FFMPEG = Path(imageio_ffmpeg.get_ffmpeg_exe())

NAVY = (12, 34, 56)
NAVY_2 = (24, 64, 96)
BLUE = (31, 78, 121)
ACCENT = (64, 156, 206)
LIGHT = (244, 248, 252)
WHITE = (255, 255, 255)
INK = (22, 50, 79)
MUTED = (174, 196, 214)
GREEN = (64, 145, 108)
ORANGE = (197, 90, 17)
GRAY = (120, 138, 154)


def _font_path(bold: bool = False, semibold: bool = False) -> str:
    names = (
        ["seguisb.ttf", "segoeuib.ttf", "arialbd.ttf"]
        if semibold and not bold
        else ["segoeuib.ttf", "arialbd.ttf"]
        if bold
        else ["segoeui.ttf", "arial.ttf"]
    )
    for name in names:
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            return str(path)
    return str(Path(r"C:\Windows\Fonts\arial.ttf"))


def font(size: int, bold: bool = False, semibold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_font_path(bold=bold, semibold=semibold), size)


def mono_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["consolab.ttf", "consola.ttf"] if bold else ["consola.ttf", "cour.ttf"]
    for name in names:
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.truetype(str(Path(r"C:\Windows\Fonts\cour.ttf")), size)


def gradient_background() -> Image.Image:
    image = Image.new("RGB", (W, H), NAVY)
    draw = ImageDraw.Draw(image)
    for y in range(H):
        t = y / (H - 1)
        color = tuple(round(NAVY[i] + (NAVY_2[i] - NAVY[i]) * t) for i in range(3))
        draw.line([(0, y), (W, y)], fill=color)

    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((1180, -320, 2180, 680), fill=(64, 156, 206, 38))
    glow = glow.filter(ImageFilter.GaussianBlur(180))
    image = Image.alpha_composite(image.convert("RGBA"), glow).convert("RGB")

    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, W, 7), fill=ACCENT)
    return image


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join(current + [word])
        if draw.textlength(trial, font=fnt) <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def draw_paragraph(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    fnt: ImageFont.FreeTypeFont,
    max_width: int,
    fill: tuple[int, int, int] = WHITE,
    line_height: float = 1.28,
) -> int:
    x, y = xy
    for line in wrap(draw, text, fnt, max_width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += int(fnt.size * line_height)
    return y


def header(draw: ImageDraw.ImageDraw, index: int, total: int, section: str) -> None:
    draw.text((96, 42), section.upper(), font=font(19, semibold=True), fill=(150, 190, 220))
    label = f"{index:02d} / {total:02d}"
    draw.text((W - 96 - draw.textlength(label, font=font(19, semibold=True)), 42), label,
              font=font(19, semibold=True), fill=(150, 190, 220))
    draw.line((96, 84, W - 96, 84), fill=(52, 96, 128), width=1)


def footer(draw: ImageDraw.ImageDraw) -> None:
    draw.line((96, H - 66, W - 96, H - 66), fill=(52, 96, 128), width=1)
    draw.text((96, H - 48), "github.com/sriramrameshr/textrank-summarizer",
              font=font(17), fill=(145, 177, 200))
    right = "Extractive summarisation · ROUGE-1/2/L"
    draw.text((W - 96 - draw.textlength(right, font=font(17)), H - 48), right,
              font=font(17), fill=(145, 177, 200))


def node_motif(draw: ImageDraw.ImageDraw, cx: int, cy: int, radius: int) -> None:
    rng = random.Random(7)
    nodes: list[tuple[float, float, int]] = []
    for i in range(16):
        angle = 2 * math.pi * i / 16 + rng.uniform(-0.12, 0.12)
        r = radius * rng.uniform(0.45, 1.0)
        nodes.append((cx + r * math.cos(angle), cy + r * math.sin(angle), rng.randint(7, 17)))
    for i, (x1, y1, _) in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            x2, y2, _ = nodes[j]
            if math.hypot(x1 - x2, y1 - y2) < radius * 0.72 and rng.random() < 0.55:
                draw.line((x1, y1, x2, y2), fill=(90, 160, 205, 90), width=2)
    for x, y, r in nodes:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(86, 170, 220, 210))


def card(image: Image.Image, box: tuple[int, int, int, int], radius: int = 22) -> None:
    x1, y1, x2, y2 = box
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle((x1 + 10, y1 + 14, x2 + 10, y2 + 14), radius=radius, fill=(0, 0, 0, 95))
    shadow = shadow.filter(ImageFilter.GaussianBlur(16))
    image.alpha_composite(shadow)
    ImageDraw.Draw(image).rounded_rectangle(box, radius=radius, fill=(255, 255, 255, 255))


def place_image(
    image: Image.Image,
    box: tuple[int, int, int, int],
    path: Path,
    crop: tuple[int, int, int, int] | None = None,
    pad: int = 18,
) -> None:
    picture = Image.open(path).convert("RGB")
    if crop:
        picture = picture.crop(crop)
    max_w = box[2] - box[0] - 2 * pad
    max_h = box[3] - box[1] - 2 * pad
    picture.thumbnail((max_w, max_h), Image.LANCZOS)
    x = box[0] + (box[2] - box[0] - picture.width) // 2
    y = box[1] + (box[3] - box[1] - picture.height) // 2
    image.paste(picture, (x, y))


def pill(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, fg=WHITE, bg=(38, 86, 122)) -> int:
    fnt = font(20, semibold=True)
    width = int(draw.textlength(text, font=fnt)) + 34
    draw.rounded_rectangle((x, y, x + width, y + 44), radius=22, fill=bg)
    draw.text((x + 17, y + 10), text, font=fnt, fill=fg)
    return x + width + 12


@dataclass
class Scene:
    key: str
    kicker: str
    title: str
    bullets: list[str]
    narration: str
    image: str | None = None
    crop: tuple[int, int, int, int] | None = None
    layout: str = "content"


SCENES: list[Scene] = [
    Scene(
        "title",
        "NLP MINI PROJECT",
        "Extractive Text\nSummarisation with\nTextRank",
        [],
        "This is a walkthrough of my NLP mini project: an extractive text summariser built with "
        "TextRank and evaluated with ROUGE on CNN Daily Mail. The full pipeline, notebook, report "
        "and demo are included.",
        layout="title",
    ),
    Scene(
        "pipeline",
        "METHOD",
        "Five stages, no training",
        [
            "Sentence segmentation + Porter stemming",
            "Sublinear TF-IDF sentence vectors",
            "Cosine-similarity weighted graph",
            "PageRank ranking, damping alpha 0.85",
            "Top-k sentences, restored to source order",
        ],
        "The method has five stages. First, the article is split into sentences, and each sentence "
        "is reduced to content words with stop-word removal and Porter stemming. Second, sentences "
        "become TF I D F vectors. Third, pairwise cosine similarity defines a weighted graph. "
        "Fourth, PageRank ranks sentences by recursive centrality. Finally, the top k sentences are "
        "restored to their original order. No training and no GPU are needed.",
        image="pipeline.png",
    ),
    Scene(
        "app",
        "LIVE DEMO",
        "Summary and live ROUGE",
        [
            "Three-sentence extract, 12.9% compression",
            "ROUGE-1, ROUGE-2 and ROUGE-L vs the reference",
            "Delta against the Lead-3 baseline",
            "Controls for method, k, alpha and threshold",
        ],
        "The Streamlit app runs the same pipeline. Here, a CNN Daily Mail article has been summarised "
        "to three sentences, about thirteen percent of the source. On the right is the human reference. "
        "The cards show ROUGE-1, ROUGE-2 and ROUGE-L F1, plus the delta against Lead-3. The sidebar "
        "controls the method, length, damping factor and similarity threshold.",
        image="app_overview.png",
    ),
    Scene(
        "ranking",
        "EXPLAINABILITY 1 / 3",
        "Why each sentence?",
        [
            "PageRank score for every sentence",
            "Selected sentences highlighted",
            "Top-14 sentences shown",
        ],
        "The ranking tab shows the PageRank score of every sentence. Dark bars are the selected "
        "sentences, so the decision is fully inspectable.",
        image="app_sentence_ranking.png",
        crop=(0, 0, 2160, 1500),
    ),
    Scene(
        "graph",
        "EXPLAINABILITY 2 / 3",
        "The similarity network",
        [
            "Nodes are sentences",
            "Size and colour encode PageRank",
            "Hub sentences are selected",
        ],
        "The graph tab makes the algorithm visible. Node size and colour encode PageRank score. "
        "Highly scored sentences sit at the centre of the similarity network, not necessarily at the "
        "top of the article.",
        image="app_sentence_graph.png",
        crop=(0, 0, 2160, 1500),
    ),
    Scene(
        "similarity",
        "EXPLAINABILITY 3 / 3",
        "Pairwise similarity",
        [
            "TF-IDF cosine heatmap",
            "Weak edges removed before ranking",
        ],
        "The similarity matrix shows the pairwise TF I D F cosine similarities behind the graph edges.",
        image="app_similarity_matrix.png",
        crop=(0, 0, 2160, 1500),
    ),
    Scene(
        "results",
        "BENCHMARK",
        "TextRank beats random, trails Lead-3",
        [],
        "On one hundred and fifty CNN Daily Mail articles, TextRank reaches twenty eight point seven "
        "ROUGE-1, ten point one ROUGE-2 and nineteen point eight ROUGE-L. It significantly beats random "
        "selection by one point seven ROUGE-1, with p equals zero point zero one five. It trails Lead-3 "
        "at thirty point six, which is expected on news: the first sentences already contain the main "
        "facts, and the human highlights follow the same convention. The report analyses this honestly, "
        "including per-document wins and losses.",
        image="rouge_by_method.png",
        layout="results",
    ),
    Scene(
        "length",
        "ANALYSIS",
        "Length sweep and ablation",
        [
            "All methods decline as k grows",
            "TextRank beats random at every length",
            "Redundancy filter adds little at k = 3",
            "Best TextRank score at k = 2: 29.74",
        ],
        "A length sweep from two to eight sentences shows all methods decline as summaries grow, "
        "because precision falls faster than recall rises. TextRank still beats random selection at "
        "every length. A redundancy filter adds only a negligible gain at three sentences, because "
        "short news extracts rarely contain near duplicates.",
        image="rouge_by_length.png",
    ),
    Scene(
        "notebook",
        "REPRODUCIBILITY",
        "Executed notebook, one command",
        [
            "27 executed cells, zero errors",
            "Reusable modules and experiment runner",
            "Offline dataset and Punkt model",
            "Report, screenshots and demo included",
        ],
        "The repository includes the executed notebook, reusable modules, the experiment runner, the "
        "report source and the demo. One command reruns the evaluation and regenerates every table and "
        "figure. The dataset and Punkt model are cached, so everything works offline.",
        layout="terminal",
    ),
    Scene(
        "closing",
        "THANK YOU",
        "Reproduce every number",
        [],
        "The code, report and screenshots are on GitHub, at the link on screen. Thank you for watching.",
        layout="closing",
    ),
]


def title_slide(scene: Scene, index: int, total: int) -> Image.Image:
    image = gradient_background().convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    node_motif(ImageDraw.Draw(overlay), 1500, 520, 380)
    image.alpha_composite(overlay)
    draw = ImageDraw.Draw(image)

    draw.text((96, 150), scene.kicker, font=font(23, semibold=True), fill=(132, 190, 226))
    y = 210
    for line in scene.title.splitlines():
        draw.text((96, y), line, font=font(92, bold=True), fill=WHITE)
        y += 104
    draw.text(
        (100, y + 20),
        "A TF-IDF sentence-graph approach evaluated with ROUGE-1/2/L",
        font=font(31),
        fill=(205, 224, 238),
    )
    draw.text(
        (100, y + 84),
        f"{STUDENT['student_name']}  ·  {STUDENT['roll_number']}",
        font=font(28, semibold=True),
        fill=WHITE,
    )
    draw.text(
        (100, y + 126),
        f"{STUDENT['institution']}  ·  {STUDENT['department']}",
        font=font(19),
        fill=MUTED,
    )

    x = 100
    for label, bg in (
        ("150 CNN/DailyMail articles", (38, 86, 122)),
        ("28.70 ROUGE-1", (38, 86, 122)),
        ("~3.2 ms / article", (38, 86, 122)),
        ("27-cell notebook", (38, 86, 122)),
    ):
        x = pill(draw, x, 760, label, bg=bg)

    draw.text((100, 860), "Python 3.10+   ·   MIT licence   ·   no GPU",
              font=font(21), fill=MUTED)
    draw.text((100, H - 60), "Mini project walkthrough",
              font=font(19), fill=(140, 174, 198))
    right = f"{index:02d} / {total:02d}"
    draw.text((W - 96 - draw.textlength(right, font=font(19)), H - 60), right,
              font=font(19), fill=(140, 174, 198))
    return image.convert("RGB")


def content_slide(scene: Scene, index: int, total: int) -> Image.Image:
    image = gradient_background().convert("RGBA")
    draw = ImageDraw.Draw(image)
    header(draw, index, total, scene.kicker)
    footer(draw)

    draw.text((96, 148), scene.kicker, font=font(21, semibold=True), fill=(125, 185, 222))
    y = draw_paragraph(draw, scene.title, (96, 190), font(50, bold=True), 560, WHITE, 1.12)
    y += 28
    for bullet in scene.bullets:
        draw.ellipse((101, y + 12, 111, y + 22), fill=ACCENT)
        y = draw_paragraph(draw, bullet, (132, y), font(26), 505, (218, 233, 244), 1.28) + 16

    if scene.image:
        box = (720, 154, 1824, 970)
        card(image, box)
        place_image(image, box, FIGURES / scene.image, scene.crop, pad=22)
        draw = ImageDraw.Draw(image)
        label = scene.image.replace("_", " ").replace(".png", "").upper()
        draw.text((744, 916), label, font=font(16, semibold=True), fill=GRAY)

    draw.rectangle((0, H - 8, int(W * index / total), H), fill=ACCENT)
    return image.convert("RGB")


def results_slide(scene: Scene, index: int, total: int) -> Image.Image:
    image = gradient_background().convert("RGBA")
    draw = ImageDraw.Draw(image)
    header(draw, index, total, scene.kicker)
    footer(draw)

    draw.text((96, 142), scene.kicker, font=font(21, semibold=True), fill=(125, 185, 222))
    draw_paragraph(draw, scene.title, (96, 184), font(47, bold=True), 760, WHITE, 1.1)

    rows = [
        ("TextRank", "28.70", "10.13", "19.75"),
        ("Lead-3", "30.58", "12.53", "21.19"),
        ("Random-3", "27.01", "8.29", "18.26"),
    ]
    x0, y0 = 96, 360
    widths = [270, 165, 165, 165]
    headers = ["Method", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
    x = x0
    for width, label in zip(widths, headers):
        draw.text((x, y0), label, font=font(19, semibold=True), fill=(158, 194, 220))
        x += width
    y = y0 + 42
    for i, row in enumerate(rows):
        fill = (38, 86, 122, 220) if i == 0 else (20, 52, 80, 190)
        draw.rounded_rectangle((x0 - 10, y - 8, x0 + sum(widths) - 18, y + 54), radius=10, fill=fill)
        x = x0
        for j, value in enumerate(row):
            color = WHITE if j == 0 else (219, 234, 244)
            draw.text((x, y + 6), value, font=font(27, bold=(i == 0)), fill=color)
            x += widths[j]
        y += 70

    draw.text((96, 590), "Mean F1 at k = 3  ·  n = 150  ·  95% bootstrap CI",
              font=font(19), fill=MUTED)
    pill(draw, 96, 642, "+1.69 ROUGE-1 vs random (p = 0.015)", bg=(35, 104, 76))
    pill(draw, 96, 704, "-1.88 vs Lead-3 (p = 0.026)", bg=(143, 67, 18))
    draw.text((96, 786), "Wins: 85/150 vs random  ·  60/150 vs Lead-3",
              font=font(20), fill=(196, 216, 231))

    box = (930, 170, 1824, 940)
    card(image, box)
    place_image(image, box, FIGURES / scene.image, scene.crop, pad=26)
    draw = ImageDraw.Draw(image)
    draw.text((952, 888), "MEAN ROUGE F1 BY METHOD", font=font(16, semibold=True), fill=GRAY)

    draw.rectangle((0, H - 8, int(W * index / total), H), fill=ACCENT)
    return image.convert("RGB")


def terminal_slide(scene: Scene, index: int, total: int) -> Image.Image:
    image = gradient_background().convert("RGBA")
    draw = ImageDraw.Draw(image)
    header(draw, index, total, scene.kicker)
    footer(draw)

    draw.text((96, 148), scene.kicker, font=font(21, semibold=True), fill=(125, 185, 222))
    y = draw_paragraph(draw, scene.title, (96, 190), font(50, bold=True), 560, WHITE, 1.12)
    y += 28
    for bullet in scene.bullets:
        draw.ellipse((101, y + 12, 111, y + 22), fill=ACCENT)
        y = draw_paragraph(draw, bullet, (132, y), font(26), 505, (218, 233, 244), 1.28) + 16

    box = (720, 154, 1824, 970)
    draw.rounded_rectangle(box, radius=22, fill=(8, 20, 32), outline=(76, 118, 148), width=2)
    draw.rounded_rectangle((box[0], box[1], box[2], box[1] + 62), radius=22, fill=(20, 42, 62))
    draw.rectangle((box[0], box[1] + 40, box[2], box[1] + 62), fill=(20, 42, 62))
    for i, color in enumerate(((238, 106, 94), (238, 190, 92), (110, 200, 130))):
        cx = box[0] + 30 + i * 26
        draw.ellipse((cx, box[1] + 24, cx + 12, box[1] + 36), fill=color)
    draw.text((box[0] + 120, box[1] + 18), "PowerShell  —  project root",
              font=mono_font(19), fill=(150, 180, 202))

    lines = [
        ("$", "python scripts/run_experiment.py", WHITE),
        ("", "processed 150/150 documents", (164, 192, 212)),
        ("", "Saved 3600 predictions in 7.0s", (164, 192, 212)),
        ("", "", WHITE),
        ("$", "python scripts/build_notebook.py", WHITE),
        ("", "executed and saved  27 cells", (164, 192, 212)),
        ("", "10 figures  ·  12 tables  ·  0 errors", (118, 208, 168)),
        ("", "", WHITE),
        ("$", "streamlit run src/app.py", WHITE),
        ("", "Local URL: http://localhost:8501", (164, 192, 212)),
    ]
    x = box[0] + 38
    y = box[1] + 100
    for prompt, text, color in lines:
        if not prompt and not text:
            y += 22
            continue
        if prompt:
            draw.text((x, y), prompt, font=mono_font(24, bold=True), fill=ACCENT)
        draw.text((x + 34, y), text, font=mono_font(24), fill=color)
        y += 43

    draw.text((box[0] + 38, 900), "Reproducible offline  ·  seeded with 42  ·  no GPU",
              font=font(19), fill=(120, 156, 184))
    draw.text((box[2] - 142, 900), "TERMINAL", font=font(16, semibold=True), fill=(86, 124, 154))

    draw.rectangle((0, H - 8, int(W * index / total), H), fill=ACCENT)
    return image.convert("RGB")


def closing_slide(scene: Scene, index: int, total: int) -> Image.Image:
    image = gradient_background().convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    node_motif(ImageDraw.Draw(overlay), 1550, 780, 300)
    image.alpha_composite(overlay)
    draw = ImageDraw.Draw(image)

    draw.text((96, 150), scene.kicker, font=font(22, semibold=True), fill=(132, 190, 226))
    draw.text((96, 205), scene.title, font=font(84, bold=True), fill=WHITE)
    draw.text((100, 330), "Code, report, screenshots and notebook",
              font=font(32), fill=(205, 224, 238))

    box = (96, 430, 1250, 640)
    card(image, box)
    draw = ImageDraw.Draw(image)
    draw.text((130, 468), "PUBLIC GITHUB REPOSITORY", font=font(18, semibold=True), fill=GRAY)
    draw.text((130, 510), "github.com/sriramrameshr/", font=font(38, bold=True), fill=INK)
    draw.text((130, 562), "textrank-summarizer", font=font(38, bold=True), fill=BLUE)

    x = 100
    for label in ("27-cell notebook", "17-page report", "23-page screenshots", "2,000-resample CIs"):
        x = pill(draw, x, 720, label, bg=(38, 86, 122))

    draw.text((100, 810), "Dataset and Punkt model cached   ·   seeded evaluation   ·   MIT licence",
              font=font(22), fill=(196, 216, 231))
    draw.text((100, 870), "Thank you for watching.", font=font(28, semibold=True), fill=WHITE)
    draw.text(
        (100, 930),
        f"{STUDENT['student_name']}  ·  {STUDENT['roll_number']}  ·  {STUDENT['institution']}",
        font=font(21),
        fill=(196, 216, 231),
    )
    draw.rectangle((0, H - 8, int(W * index / total), H), fill=ACCENT)
    return image.convert("RGB")


def render_slide(scene: Scene, index: int, total: int) -> Image.Image:
    if scene.layout == "title":
        return title_slide(scene, index, total)
    if scene.layout == "results":
        return results_slide(scene, index, total)
    if scene.layout == "terminal":
        return terminal_slide(scene, index, total)
    if scene.layout == "closing":
        return closing_slide(scene, index, total)
    return content_slide(scene, index, total)


PS_SCRIPT = r"""
param([string]$TextFile, [string]$OutFile, [int]$Rate)
Add-Type -AssemblyName System.Speech
$text = [System.IO.File]::ReadAllText($TextFile, [System.Text.Encoding]::UTF8)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice('Microsoft Zira Desktop')
$synth.Rate = $Rate
$synth.Volume = 100
$synth.SetOutputToWaveFile($OutFile)
$synth.Speak($text)
$synth.Dispose()
"""


def synthesise(text: str, destination: Path, rate: int) -> None:
    text_file = destination.with_suffix(".txt")
    text_file.write_text(text, encoding="utf-8")
    script = BUILD / "synth.ps1"
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-TextFile",
            str(text_file),
            "-OutFile",
            str(destination),
            "-Rate",
            str(rate),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not destination.exists():
        raise RuntimeError(f"SAPI synthesis failed: {result.stderr or result.stdout}")


def pad_wav(source: Path, destination: Path, pre: float = 0.35, post: float = 0.55) -> float:
    with wave.open(str(source), "rb") as wav:
        params = wav.getparams()
        frames = wav.readframes(wav.getnframes())
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        pre_frames = int(rate * pre)
        post_frames = int(rate * post)
        silence = b"\x00" * (width * channels)
        with wave.open(str(destination), "wb") as out:
            out.setparams(params)
            out.writeframes(silence * pre_frames + frames + silence * post_frames)
        return (pre_frames + wav.getnframes() + post_frames) / rate


def concat_audio(paths: list[Path], destination: Path) -> None:
    first = wave.open(str(paths[0]), "rb")
    params = first.getparams()
    first.close()
    with wave.open(str(destination), "wb") as out:
        out.setparams(params)
        for path in paths:
            with wave.open(str(path), "rb") as wav:
                out.writeframes(wav.readframes(wav.getnframes()))


def encode_scene(slide: Path, duration: float, destination: Path) -> None:
    fade = 0.35
    video_filter = (
        f"fade=t=in:st=0:d={fade},"
        f"fade=t=out:st={max(0.0, duration - fade):.3f}:d={fade},format=yuv420p"
    )
    command = [
        str(FFMPEG),
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(FPS),
        "-i",
        str(slide),
        "-t",
        f"{duration:.3f}",
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-tune",
        "stillimage",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True)


def concat_video(paths: list[Path], destination: Path) -> None:
    listing = BUILD / "concat.txt"
    listing.write_text(
        "\n".join(f"file '{path.as_posix()}'" for path in paths), encoding="utf-8"
    )
    subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            str(destination),
        ],
        check=True,
        capture_output=True,
    )


def main() -> int:
    if not sys.platform.startswith("win"):
        print("This script uses Windows SAPI for narration.", file=sys.stderr)
        return 1

    BUILD.mkdir(parents=True, exist_ok=True)
    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    (BUILD / "synth.ps1").write_text(PS_SCRIPT, encoding="utf-8-sig")

    total = len(SCENES)
    rate = 1
    slides: list[Path] = []
    raw_audio: list[Path] = []
    padded_audio: list[Path] = []

    print(f"rendering {total} slides ...")
    for index, scene in enumerate(SCENES, start=1):
        slide = BUILD / f"slide_{index:02d}.png"
        render_slide(scene, index, total).save(slide)
        slides.append(slide)

        wav = BUILD / f"narration_{index:02d}.wav"
        synthesise(scene.narration, wav, rate)
        raw_audio.append(wav)
        print(f"  {index:02d} {scene.key}: slide + narration ready")

    total_speech = sum(
        (lambda w: w.getnframes() / w.getframerate())(wave.open(str(path), "rb"))
        for path in raw_audio
    )
    if total_speech > 175:
        rate = 2
        print(f"speech is {total_speech:.1f}s; re-synthesising at rate {rate} ...")
        for index, scene in enumerate(SCENES, start=1):
            synthesise(scene.narration, raw_audio[index - 1], rate)

    durations: list[float] = []
    for index, wav in enumerate(raw_audio, start=1):
        padded = BUILD / f"narration_{index:02d}_padded.wav"
        durations.append(pad_wav(wav, padded))
        padded_audio.append(padded)

    narration = BUILD / "narration.wav"
    concat_audio(padded_audio, narration)

    print("encoding scenes ...")
    scene_videos: list[Path] = []
    for index, (slide, duration) in enumerate(zip(slides, durations), start=1):
        destination = BUILD / f"scene_{index:02d}.mp4"
        encode_scene(slide, duration, destination)
        scene_videos.append(destination)
        print(f"  {index:02d}: {duration:.2f}s")

    silent = BUILD / "silent.mp4"
    concat_video(scene_videos, silent)

    print("muxing narration ...")
    subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-i",
            str(silent),
            "-i",
            str(narration),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-movflags",
            "+faststart",
            "-shortest",
            "-metadata",
            "title=Extractive Text Summarisation with TextRank - Demo",
            str(OUTPUT),
        ],
        check=True,
        capture_output=True,
    )

    duration = sum(durations)
    print(f"wrote {OUTPUT} ({duration:.1f}s, {OUTPUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
