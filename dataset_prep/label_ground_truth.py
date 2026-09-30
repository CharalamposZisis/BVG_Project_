"""Interactive ground-truth labeling tool for title/date extraction.

For every scanned plan in plans/, shows a downsampled preview, lets the user
drag a box around the title block ("Schriftfeld"), then prompts in the
terminal for the title text, the date as written, and a status flag. Writes
one row per file to dataset_prep/ground_truth.csv.

Resumable: files already present in ground_truth.csv are skipped on the next
run. Use --relabel <filename> to redo a single file, or --limit N to only
label the next N unlabeled files in a session.

Usage:
    python label_ground_truth.py
    python label_ground_truth.py --limit 5
    python label_ground_truth.py --relabel S_714_006.tif
"""

import argparse
import csv
import re
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from plan_io import load_downsampled as _load_downsampled

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLANS_DIR = PROJECT_ROOT / "plans"
OUT_CSV = Path(__file__).resolve().parent / "ground_truth.csv"

FIELDNAMES = [
    "filename", "title", "date_raw", "date_iso",
    "title_bbox_x0", "title_bbox_y0", "title_bbox_x1", "title_bbox_y1",
    "status", "notes",
]

MAX_DISPLAY_DIM = 2000

GERMAN_MONTHS = {
    "januar": 1, "jan": 1, "februar": 2, "feb": 2, "märz": 3, "maerz": 3, "mrz": 3,
    "april": 4, "apr": 4, "mai": 5, "juni": 6, "jun": 6, "juli": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "oktober": 10, "okt": 10,
    "november": 11, "nov": 11, "dezember": 12, "dez": 12,
}


def load_downsampled(path: Path, max_dim: int = MAX_DISPLAY_DIM) -> np.ndarray:
    """Load a plan file (tif/tiff/pdf) as a downsampled RGB array."""
    return np.asarray(_load_downsampled(path, max_dim))


def parse_date_raw(date_raw: str):
    """Best-effort normalization of a hand-entered date string to ISO 8601."""
    s = date_raw.strip()
    if not s:
        return ""
    # Normalize "14. 6. 79" / "14 . 6 . 79" to "14.6.79" before matching --
    # the patterns below don't otherwise tolerate stray whitespace.
    s = re.sub(r"\s*([.\-/])\s*", r"\1", s)

    m = re.match(r"^(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})$", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"

    m = re.match(r"^(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2})$", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"19{y:02d}-{mo:02d}-{d:02d}"  # archive is historical; assume 19xx

    m = re.match(r"^(\d{1,2})[.\-/](\d{4})$", s)
    if m:
        mo, y = int(m.group(1)), int(m.group(2))
        return f"{y:04d}-{mo:02d}-01"

    m = re.match(r"^(\d{4})$", s)
    if m:
        return f"{int(m.group(1)):04d}-01-01"

    m = re.match(r"^(\d{1,2})\.?\s*([A-Za-zÄÖÜäöü]+)\.?\s*(\d{4})$", s)
    if m:
        d, month_name, y = m.group(1), m.group(2).lower(), int(m.group(3))
        mo = GERMAN_MONTHS.get(month_name)
        if mo:
            return f"{y:04d}-{mo:02d}-{int(d):02d}"

    # Month name + year, no day (e.g. "März 1970", "Jan. 1985")
    m = re.match(r"^([A-Za-zÄÖÜäöü]+)\.?\s*(\d{4})$", s)
    if m:
        month_name, y = m.group(1).lower(), int(m.group(2))
        mo = GERMAN_MONTHS.get(month_name)
        if mo:
            return f"{y:04d}-{mo:02d}-01"

    # MM/YY, 2-digit year (e.g. "06/90")
    m = re.match(r"^(\d{1,2})[.\-/](\d{2})$", s)
    if m:
        mo, y = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            return f"19{y:02d}-{mo:02d}-01"

    return ""  # ambiguous / unparsed — leave for manual review in the CSV


def read_existing(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def write_all(csv_path: Path, rows: dict):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for filename in sorted(rows):
            writer.writerow({k: rows[filename].get(k, "") for k in FIELDNAMES})


def label_one(path: Path) -> dict:
    print("\n" + "=" * 60)
    print(path.name)
    print("=" * 60)

    try:
        arr = load_downsampled(path)
    except Exception as exc:  # noqa: BLE001
        print(f"  Failed to load: {exc}")
        return {
            "filename": path.name, "title": "", "date_raw": "", "date_iso": "",
            "title_bbox_x0": "", "title_bbox_y0": "", "title_bbox_x1": "", "title_bbox_y1": "",
            "status": "unreadable", "notes": str(exc),
        }

    height, width = arr.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(arr)
    ax.set_title(f"{path.name}\nDrag a box around the title block, then close this window.")

    # Start zoomed into the bottom-right quadrant, the usual Schriftfeld location.
    # Press 'f' to see the full sheet, 'r' to return to this quadrant.
    def show_quadrant():
        ax.set_xlim(width * 0.55, width)
        ax.set_ylim(height, height * 0.6)  # inverted y (image origin top-left)

    def show_full():
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

    show_quadrant()

    def on_key(event):
        if event.key == "f":
            show_full()
            fig.canvas.draw_idle()
        elif event.key == "r":
            show_quadrant()
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    selector = RectangleSelector(
        ax, lambda eclick, erelease: None,
        useblit=True, button=[1], interactive=True, minspanx=5, minspany=5,
    )

    plt.show()

    bbox = ("", "", "", "")
    extents = getattr(selector, "extents", None)
    if extents and (extents[1] - extents[0] > 1) and (extents[3] - extents[2] > 1):
        x0, x1, y0, y1 = extents
        bbox = (
            round(max(0, x0) / width, 4), round(max(0, y0) / height, 4),
            round(min(width, x1) / width, 4), round(min(height, y1) / height, 4),
        )
    else:
        print("  (no bounding box drawn)")

    title = input("  Title (as written in the title block): ").strip()
    date_raw = input("  Date (exactly as written, e.g. 12.05.1927): ").strip()
    status = input("  Status [ok/no_date/no_title/unreadable] (default ok): ").strip() or "ok"
    notes = input("  Notes (optional): ").strip()

    plt.close(fig)

    return {
        "filename": path.name,
        "title": title,
        "date_raw": date_raw,
        "date_iso": parse_date_raw(date_raw),
        "title_bbox_x0": bbox[0], "title_bbox_y0": bbox[1],
        "title_bbox_x1": bbox[2], "title_bbox_y1": bbox[3],
        "status": status,
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relabel", help="Filename to relabel, overwriting its existing row.")
    parser.add_argument("--limit", type=int, help="Only label this many files this session.")
    parser.add_argument("--plans-dir", default=str(PLANS_DIR))
    args = parser.parse_args()

    plans_dir = Path(args.plans_dir)
    rows = read_existing(OUT_CSV)

    files = sorted(
        p for p in plans_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff", ".pdf")
    )

    if args.relabel:
        files = [p for p in files if p.name == args.relabel]
        if not files:
            print(f"No file named {args.relabel} found in {plans_dir}")
            return
    else:
        files = [p for p in files if p.name not in rows]

    if not files:
        print(f"Nothing to label. {len(rows)}/{len(list(plans_dir.iterdir()))} already in {OUT_CSV.name}.")
        return

    if args.limit:
        files = files[: args.limit]

    print(f"Labeling {len(files)} file(s). Ctrl+C at any prompt to stop and save progress so far.")

    try:
        for path in files:
            row = label_one(path)
            rows[row["filename"]] = row
            write_all(OUT_CSV, rows)  # save after every file, not just at the end
    except KeyboardInterrupt:
        print("\nStopped early. Progress saved.")

    print(f"\n{len(rows)} file(s) now in {OUT_CSV}")


if __name__ == "__main__":
    main()
