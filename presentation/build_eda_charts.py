"""Renders EDA charts (PNG) from dataset_prep/ground_truth.csv for the slide deck.

Usage:
    python build_eda_charts.py
"""

from pathlib import Path

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GT_CSV = PROJECT_ROOT / "dataset_prep" / "ground_truth.csv"
OUT_DIR = Path(__file__).resolve().parent / "charts"

# Reference palette (dataviz skill, light mode)
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

STATUS_COLOR = {"ok": "#0ca30c", "no_date": "#fab219", "no_title": "#ec835a", "unreadable": "#d03b3b"}
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eb6834"]  # slots 1, 3, 2 (fixed order)
SEQ_BLUE = "#2a78d6"

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style_axes(ax, y_grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="both", length=0)
    if y_grid:
        ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
        ax.set_axisbelow(True)


def chart_status_breakdown(df):
    counts = df["status"].value_counts()
    order = [s for s in ["ok", "no_date", "no_title", "unreadable"] if s in counts.index]
    values = [counts[s] for s in order]
    colors = [STATUS_COLOR[s] for s in order]
    labels = {"ok": "OK", "no_date": "No date", "no_title": "No title", "unreadable": "Unreadable"}

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar([labels[s] for s in order], values, color=colors, width=0.55, zorder=3)
    style_axes(ax)
    ax.set_title("Labeling status — 42 plans", fontsize=15, color=INK, pad=14, loc="left", fontweight="bold")
    ax.set_ylim(0, max(values) * 1.2)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.03, str(v),
                 ha="center", va="bottom", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "status_breakdown.png", dpi=200)
    plt.close(fig)


def chart_dates_by_decade(df):
    years = df.loc[df["date_iso"] != "", "date_iso"].str.slice(0, 4).astype(int)
    decades = (years // 10 * 10).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x_labels = [f"{d}s" for d in decades.index]
    bars = ax.bar(x_labels, decades.values, color=SEQ_BLUE, width=0.6, zorder=3)
    style_axes(ax)
    ax.set_title(f"Plan dates by decade — {len(years)} parsed ({years.min()}–{years.max()})",
                 fontsize=15, color=INK, pad=14, loc="left", fontweight="bold")
    for b, v in zip(bars, decades.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, str(v), ha="center", va="bottom",
                 fontsize=11, color=INK, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dates_by_decade.png", dpi=200)
    plt.close(fig)


def chart_split_examples(train_n, val_n, test_n, train_base, val_base, test_base):
    splits = ["train", "val", "test"]
    examples = [train_n, val_n, test_n]
    bases = [train_base, val_base, test_base]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(splits, examples, color=CATEGORICAL, width=0.5, zorder=3)
    style_axes(ax)
    ax.set_title("Training examples per split (after crop + augmentation)",
                 fontsize=15, color=INK, pad=14, loc="left", fontweight="bold")
    for b, v, base in zip(bars, examples, bases):
        ax.text(b.get_x() + b.get_width() / 2, v + max(examples) * 0.03, f"{v}\n({base} plans)",
                 ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylim(0, max(examples) * 1.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "split_examples.png", dpi=200)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(GT_CSV, dtype=str, keep_default_na=False)

    chart_status_breakdown(df)
    chart_dates_by_decade(df)

    # Pull split sizes straight from the prepared dataset if it exists, else recompute.
    data_dir = PROJECT_ROOT / "finetune" / "data"
    import json
    counts = {}
    bases = {}
    for split in ("train", "val", "test"):
        jsonl = data_dir / f"{split}.jsonl"
        if jsonl.exists():
            with jsonl.open() as f:
                rows = [json.loads(line) for line in f]
            counts[split] = len(rows)
            base_names = {
                re.sub(r"_(full|crop)(_aug\d+)?$", "", Path(r["images"][0]).stem)
                for r in rows
            }
            bases[split] = len(base_names)
        else:
            counts[split] = 0
            bases[split] = 0

    chart_split_examples(counts["train"], counts["val"], counts["test"],
                          bases["train"], bases["val"], bases["test"])

    print(f"Wrote charts to {OUT_DIR}")


if __name__ == "__main__":
    main()
