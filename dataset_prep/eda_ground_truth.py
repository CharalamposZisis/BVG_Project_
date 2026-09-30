"""Exploratory data analysis over dataset_prep/ground_truth.csv.

Merges the title/date labels with plans_metadata.csv and reports labeling
coverage, date-format variety, title stats, and title-block bbox position —
the signals that matter for preparing the fine-tuning dataset in finetune/.
"""

import re
from pathlib import Path

import pandas as pd

GT_CSV = Path(__file__).resolve().parent / "ground_truth.csv"
META_CSV = Path(__file__).resolve().parent / "plans_metadata.csv"

DATE_FORMAT_PATTERNS = [
    ("DD.MM.YYYY", r"^\d{1,2}[.\-/]\d{1,2}[.\-/]\d{4}$"),
    ("DD.MM.YY", r"^\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2}$"),
    ("MM.YYYY", r"^\d{1,2}[.\-/]\d{4}$"),
    ("YYYY", r"^\d{4}$"),
    ("DD Month YYYY (text)", r"^\d{1,2}\.?\s*[A-Za-zÄÖÜäöü]+\.?\s*\d{4}$"),
]


def section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def classify_date_format(s: str) -> str:
    s = s.strip()
    if not s:
        return "(empty)"
    for name, pattern in DATE_FORMAT_PATTERNS:
        if re.match(pattern, s):
            return name
    return "other/unrecognized"


def main():
    if not GT_CSV.exists():
        print(f"{GT_CSV} does not exist yet — run label_ground_truth.py first.")
        return

    gt = pd.read_csv(GT_CSV, dtype=str, keep_default_na=False)
    meta = pd.read_csv(META_CSV, dtype=str, keep_default_na=False) if META_CSV.exists() else None

    all_plan_names = None
    if meta is not None:
        df = gt.merge(meta, on="filename", how="outer", indicator=True)
        all_plan_names = set(meta["filename"])
    else:
        df = gt.copy()
        df["_merge"] = "left_only"

    section("Labeling coverage")
    n_total = len(all_plan_names) if all_plan_names is not None else len(gt)
    n_labeled = len(gt)
    print(f"Labeled: {n_labeled} / {n_total}")
    if meta is not None:
        unlabeled = sorted(all_plan_names - set(gt["filename"]))
        if unlabeled:
            print(f"\nNot yet labeled ({len(unlabeled)}):")
            for name in unlabeled:
                print(f"  - {name}")

    if n_labeled == 0:
        return

    section("Status breakdown")
    print(gt["status"].value_counts(dropna=False))

    section("Date formats (as written)")
    formats = gt["date_raw"].apply(classify_date_format)
    print(formats.value_counts())

    section("Date parsing (date_iso)")
    n_parsed = (gt["date_iso"] != "").sum()
    print(f"Parsed to ISO: {n_parsed} / {n_labeled}")
    years = gt.loc[gt["date_iso"] != "", "date_iso"].str.slice(0, 4).astype(int)
    if len(years):
        print(f"Year range: {years.min()} - {years.max()}")
        print(years.value_counts().sort_index())

    unparsed = gt[(gt["date_raw"] != "") & (gt["date_iso"] == "")]
    if len(unparsed):
        print(f"\nWritten but NOT parsed ({len(unparsed)}) — needs a manual date_iso or a parser fix:")
        print(unparsed[["filename", "date_raw"]].to_string(index=False))

    section("Title stats")
    titles = gt.loc[gt["title"] != "", "title"]
    print(f"Non-empty titles: {len(titles)} / {n_labeled}")
    if len(titles):
        lengths = titles.str.len()
        print(f"Length (chars): min={lengths.min()} mean={lengths.mean():.1f} max={lengths.max()}")
        dupes = titles[titles.duplicated(keep=False)]
        if len(dupes):
            print(f"\nDuplicate titles ({dupes.nunique()} distinct, {len(dupes)} rows):")
            print(dupes.value_counts())

    if meta is not None and "status_flags" in df.columns:
        section("Title text vs filename-derived status flags")
        both = df[(df["_merge"] == "both") & (df["title"] != "")]
        flagged = both[both["status_flags"].fillna("") != ""]
        for _, row in flagged.iterrows():
            title_lower = row["title"].lower()
            flags = row["status_flags"].split(";")
            hinted = any(
                kw in title_lower
                for kw in ("ungültig", "ungueltig", "nicht realisiert", "überholt", "ueberholt", "nie gebaut")
            )
            marker = "OK" if hinted else "title doesn't mention it"
            print(f"  {row['filename']}: flags={flags} -> {marker}")

    section("Title-block bbox position (fraction of sheet)")
    has_bbox = gt[gt["title_bbox_x0"] != ""]
    if len(has_bbox):
        for col in ("title_bbox_x0", "title_bbox_y0", "title_bbox_x1", "title_bbox_y1"):
            has_bbox = has_bbox.copy()
            has_bbox[col] = has_bbox[col].astype(float)
        print(has_bbox[["title_bbox_x0", "title_bbox_y0", "title_bbox_x1", "title_bbox_y1"]].describe().round(3))

        if meta is not None and "din_class" in df.columns:
            merged_bbox = df[(df["_merge"] == "both") & (df["title_bbox_x0"] != "")]
            if len(merged_bbox):
                print("\nBy DIN class (mean position):")
                merged_bbox = merged_bbox.copy()
                for col in ("title_bbox_x0", "title_bbox_y0", "title_bbox_x1", "title_bbox_y1"):
                    merged_bbox[col] = merged_bbox[col].astype(float)
                print(
                    merged_bbox.groupby("din_class")[
                        ["title_bbox_x0", "title_bbox_y0", "title_bbox_x1", "title_bbox_y1"]
                    ].mean().round(3)
                )
    else:
        print("No bounding boxes recorded yet.")

    section("Rows still needing attention")
    needs_attention = gt[
        (gt["status"] != "ok")
        | (gt["title"] == "")
        | (gt["date_raw"] == "")
        | ((gt["date_raw"] != "") & (gt["date_iso"] == ""))
    ]
    print(f"{len(needs_attention)} / {n_labeled} rows")
    if len(needs_attention):
        print(needs_attention[["filename", "status", "title", "date_raw", "date_iso"]].to_string(index=False))


if __name__ == "__main__":
    main()
