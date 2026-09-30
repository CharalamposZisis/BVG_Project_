"""Cross-references evaluate.py's failures with the original labeling confidence
notes, to test the hypothesis: "the model fails mainly on plans that were already
hard to read" (faint stamps, low-confidence transcription) rather than failing
randomly.

Usage (after running evaluate.py --skip-finetuned):
    python analyze_failures.py
"""

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_CSV = Path(__file__).resolve().parent / "eval_zero_shot.csv"
GT_CSV = PROJECT_ROOT / "dataset_prep" / "ground_truth.csv"

TITLE_PASS_THRESHOLD = 0.6
LOW_CONF_KEYWORDS = ["low confidence", "best-effort", "please verify", "not confidently", "faint", "faded"]


def base_filename(eval_filename: str) -> str:
    """'00000102_crop.png' -> '00000102' (strip _full/_crop and _augN suffixes)."""
    stem = Path(eval_filename).stem
    return re.sub(r"_(full|crop)(_aug\d+)?$", "", stem)


def match_ground_truth_filename(base: str, gt_filenames: list) -> str:
    """The eval base name has no extension; find the matching ground_truth.csv row."""
    for name in gt_filenames:
        if Path(name).stem == base:
            return name
    return ""


def is_low_confidence(notes: str) -> bool:
    n = notes.lower()
    has_low = any(k in n for k in LOW_CONF_KEYWORDS)
    has_high = "high confidence" in n or "good confidence" in n
    return has_low and not has_high


def main():
    if not EVAL_CSV.exists():
        print(f"{EVAL_CSV} not found — run evaluate.py --skip-finetuned first.")
        return

    eval_df = pd.read_csv(EVAL_CSV, dtype=str, keep_default_na=False)
    gt_df = pd.read_csv(GT_CSV, dtype=str, keep_default_na=False)
    gt_filenames = gt_df["filename"].tolist()

    rows = []
    for _, r in eval_df.iterrows():
        base = base_filename(r["filename"])
        gt_name = match_ground_truth_filename(base, gt_filenames)
        gt_row = gt_df[gt_df["filename"] == gt_name].iloc[0] if gt_name else None

        title_ok = float(r["title_similarity"]) >= TITLE_PASS_THRESHOLD
        date_ok = r["date_presence_class"] == "TN" or r["date_result"] in ("exact", "same_date_different_format")
        passed = title_ok and date_ok

        rows.append({
            "eval_filename": r["filename"],
            "plan": gt_name or base,
            "passed": passed,
            "title_ok": title_ok,
            "date_ok": date_ok,
            "was_flagged_low_confidence_at_labeling": is_low_confidence(gt_row["notes"]) if gt_row is not None else None,
            "labeling_status": gt_row["status"] if gt_row is not None else None,
        })

    df = pd.DataFrame(rows)

    print("=" * 70)
    print("FAILURES (for the presentation — \"model needs clean scans\")")
    print("=" * 70)
    failures = df[~df["passed"]]
    if len(failures):
        print(failures[["eval_filename", "plan", "title_ok", "date_ok",
                         "was_flagged_low_confidence_at_labeling", "labeling_status"]].to_string(index=False))
    else:
        print("No failures.")

    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST: does low labeling-confidence predict model failure?")
    print("=" * 70)
    known = df[df["was_flagged_low_confidence_at_labeling"].notna()]
    if len(known):
        cross = pd.crosstab(known["was_flagged_low_confidence_at_labeling"], known["passed"])
        cross.index = cross.index.map({True: "flagged low-confidence at labeling", False: "flagged high-confidence"})
        cross.columns = cross.columns.map({True: "model passed", False: "model failed"})
        print(cross)
        fail_rate_low = (~known[known["was_flagged_low_confidence_at_labeling"]]["passed"]).mean() if known["was_flagged_low_confidence_at_labeling"].any() else float("nan")
        fail_rate_high = (~known[~known["was_flagged_low_confidence_at_labeling"]]["passed"]).mean() if (~known["was_flagged_low_confidence_at_labeling"]).any() else float("nan")
        print(f"\nFailure rate on low-confidence-labeled plans:  {fail_rate_low:.0%}")
        print(f"Failure rate on high-confidence-labeled plans: {fail_rate_high:.0%}")
    else:
        print("No ground-truth confidence info matched.")


if __name__ == "__main__":
    main()
