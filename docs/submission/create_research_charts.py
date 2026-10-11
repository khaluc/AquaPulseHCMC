"""Render the report's detector figure directly from saved evaluation JSON.

Run from any working directory with ``python docs/submission/create_research_charts.py``.
Only the clean YOLO11n and YOLO11s runs are compared: both reports contain
167 test images and 316 annotated objects. Their confidence thresholds were
selected separately on validation data, not optimized on the test results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "submission" / "report-assets"
SOURCES = {
    "YOLO11n": ROOT / "ml" / "runs" / "n_clean" / "eval" / "report.json",
    "YOLO11s": ROOT / "ml" / "runs" / "s_clean" / "eval" / "report.json",
}
CLASS_NAMES = {
    "used_tire": "Used tire",
    "plastic_container": "Plastic container",
    "plant_pot": "Plant pot",
    "coconut_shell": "Coconut shell",
    "drain_inlet": "Drain inlet",
}
TEAL = "#397B7D"
NAVY = "#29485F"
ORANGE = "#A96D37"
INK = "#24343D"
GRID = "#E2E7E9"


def read_reports() -> dict:
    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in SOURCES.items()
    }
    for name, report in reports.items():
        if (report["test_images"], report["test_box_overall"]["gt"]) != (167, 316):
            raise ValueError(f"{name}: expected the common 167-image, 316-object test set")
        if sum(row["gt"] for row in report["test_box_per_class"].values()) != 316:
            raise ValueError(f"{name}: per-class object counts do not total 316")
    for name, expected_threshold in (("YOLO11n", 0.35), ("YOLO11s", 0.25)):
        if reports[name]["threshold"] != expected_threshold:
            raise ValueError(f"{name}: saved validation-selected threshold has changed")
    return reports


def style_axis(axis, label: str) -> None:
    axis.set_xlim(0, 100)
    axis.xaxis.set_major_locator(MultipleLocator(20))
    axis.set_xlabel(label, labelpad=7)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color=GRID, linewidth=0.65)
    axis.tick_params(axis="x", length=3, color="#A7B3B8", pad=4)
    axis.tick_params(axis="y", length=0, pad=8)
    for edge in ("top", "right", "left"):
        axis.spines[edge].set_visible(False)
    axis.spines["bottom"].set_color("#B8C1C5")
    axis.spines["bottom"].set_linewidth(0.7)


def main() -> None:
    reports = read_reports()
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })
    # The PNG is exactly 1700 x 850; the PDF retains vector bars and text.
    figure = plt.figure(figsize=(8.5, 4.25), dpi=200)
    overall = figure.add_axes((0.112, 0.300, 0.320, 0.510))
    classes = figure.add_axes((0.665, 0.300, 0.305, 0.510))

    metric_keys = ("precision", "recall", "f1")
    metric_labels = ("Precision", "Recall", "F1")
    for name, color, offset in (("YOLO11n", TEAL, -0.17), ("YOLO11s", NAVY, 0.17)):
        values = [100 * reports[name]["test_box_overall"][key] for key in metric_keys]
        positions = [index + offset for index in range(len(metric_keys))]
        overall.barh(positions, values, height=0.30, color=color, linewidth=0)
        for y, value in zip(positions, values):
            overall.text(value - 1.8, y, f"{value:.2f}%", ha="right", va="center",
                         color="white", fontsize=10.5)
    overall.set_yticks(range(len(metric_keys)), metric_labels)
    overall.set_ylim(2.55, -0.55)
    style_axis(overall, "Score (%)")

    class_rows = reports["YOLO11n"]["test_box_per_class"]
    for index, (key, label) in enumerate(CLASS_NAMES.items()):
        row = class_rows[key]
        value = 100 * row["recall"]
        color = ORANGE if key == "used_tire" else TEAL
        classes.barh(index, value, height=0.57, color=color, linewidth=0)
        classes.text(value - 1.8, index, f"{value:.2f}%", ha="right", va="center",
                     color="white", fontsize=10.5)
    classes.set_yticks(range(len(CLASS_NAMES)), [
        f"{label} ({class_rows[key]['gt']})" for key, label in CLASS_NAMES.items()
    ])
    classes.set_ylim(4.6, -0.6)
    style_axis(classes, "Recall (%)")

    figure.text(0.112, 0.920, "a  Overall detection metrics", fontsize=12, weight="bold")
    figure.text(0.112, 0.865, "167 held-out images; 316 objects", fontsize=9.7, color="#53636B")
    figure.text(0.665, 0.920, "b  YOLO11n class recall", fontsize=12, weight="bold")
    figure.text(0.665, 0.865, "Object counts in parentheses", fontsize=9.7, color="#53636B")
    figure.legend(handles=[
        Patch(facecolor=TEAL, label="YOLO11n (threshold 0.35)"),
        Patch(facecolor=NAVY, label="YOLO11s (threshold 0.25)"),
    ], loc="lower left", bbox_to_anchor=(0.108, 0.115), frameon=False, ncol=2,
        fontsize=9.7, handlelength=1.1, handleheight=0.9, columnspacing=2.0,
        handletextpad=0.5, borderaxespad=0)
    figure.text(0.090, 0.068,
                "Thresholds selected separately on validation data to reach 80% recall.",
                fontsize=9.1, color="#53636B")
    figure.text(0.090, 0.026,
                "Used-tire recall is based on only 3 test objects (2 detected).",
                fontsize=9.1, color="#53636B")

    caption = (
        "Detector performance on the common held-out test set (167 images, 316 annotated objects). "
        "(a) Clean YOLO11n and YOLO11s precision, recall, and F1 at their respective "
        "validation-selected confidence thresholds of 0.35 and 0.25. "
        "Thresholds were selected separately to reach 80% validation recall. "
        "(b) Clean YOLO11n per-class recall, with annotated object counts in parentheses. "
        "Used-tire recall (66.67%) is based on only three objects, of which two were detected."
    )
    source_text = "; ".join(str(path.relative_to(ROOT)) for path in SOURCES.values())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT / "05-detector-performance.png"
    pdf_path = OUTPUT / "05-detector-performance.pdf"
    figure.savefig(png_path, dpi=200, metadata={
        "Title": "Detector performance",
        "Description": caption,
        "Source": source_text,
    })
    figure.savefig(pdf_path, metadata={
        "Title": "Detector performance",
        "Subject": caption,
        "Keywords": source_text,
        "Creator": "AquaPulse create_research_charts.py / Matplotlib",
    })
    plt.close(figure)
    print(json.dumps({
        "outputs": [str(png_path.relative_to(ROOT)), str(pdf_path.relative_to(ROOT))],
        "test_images": 167,
        "test_objects": 316,
        "overall_percent": {
            name: {key: 100 * report["test_box_overall"][key] for key in metric_keys}
            for name, report in reports.items()
        },
        "n_clean_class_recall_percent": {
            key: {"recall": 100 * row["recall"], "objects": row["gt"]}
            for key, row in class_rows.items()
        },
        "caption": caption,
    }, indent=2))


if __name__ == "__main__":
    main()
