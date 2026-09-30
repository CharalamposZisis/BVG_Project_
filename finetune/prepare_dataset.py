"""Build the LoRA fine-tuning dataset from dataset_prep/ground_truth.csv.

For each labeled plan (status == "ok", non-empty title and date_raw):
  - a full-page variant (downscaled, matches real inference input)
  - a title-block crop variant (from the labeled bbox, matches a pipeline
    that crops before calling the model)
  - a couple of light augmentations (small rotation, brightness/contrast
    jitter) of each, for the TRAIN split only

Splits at the base-image level first (so augmented siblings of one plan
never land in two different splits), then writes ShareGPT-style JSONL in the
format LLaMA-Factory expects for multimodal data:

    {"messages": [{"role": "user", "content": "<image>..."},
                   {"role": "assistant", "content": "{...}"}],
     "images": ["images/....png"]}

Val/test splits get exactly one (clean, non-augmented, full-page) example
per plan, since that's the realistic inference-time input.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --val-frac 0.1 --test-frac 0.1 --seed 42
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageEnhance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "dataset_prep"))
from plan_io import load_full, downsample  # noqa: E402

GT_CSV = PROJECT_ROOT / "dataset_prep" / "ground_truth.csv"
PLANS_DIR = PROJECT_ROOT / "plans"
DATA_DIR = Path(__file__).resolve().parent / "data"
IMAGES_DIR = DATA_DIR / "images"

FULL_MAX_DIM = 1280
CROP_MAX_DIM = 896
N_TRAIN_AUGS = 2  # extra augmented copies per variant, train split only

PROMPTS = [
    'Read the title block of this construction plan. Respond with JSON: {"title": ..., "date": ...}.',
    'Look at the title block (Schriftfeld) on this drawing and extract the title and the date exactly '
    'as written. Respond with JSON: {"title": ..., "date": ...}.',
]


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    out = img.rotate(rng.uniform(-3, 3), resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.85, 1.15))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.85, 1.15))
    return out


def make_example(image_rel_path: str, title: str, date_raw: str, rng: random.Random) -> dict:
    answer = json.dumps({"title": title, "date": date_raw}, ensure_ascii=False)
    prompt = rng.choice(PROMPTS)
    return {
        "messages": [
            {"role": "user", "content": f"<image>{prompt}"},
            {"role": "assistant", "content": answer},
        ],
        "images": [image_rel_path],
    }


def crop_box(bbox, width, height, rng: random.Random):
    x0, y0, x1, y1 = bbox
    pad_x = (x1 - x0) * rng.uniform(0.02, 0.12)
    pad_y = (y1 - y0) * rng.uniform(0.02, 0.12)
    px0 = max(0, int((x0 - pad_x) * width))
    py0 = max(0, int((y0 - pad_y) * height))
    px1 = min(width, int((x1 + pad_x) * width))
    py1 = min(height, int((y1 + pad_y) * height))
    return px0, py0, px1, py1


def build_split(rows: list, split: str, rng: random.Random) -> list:
    examples = []
    for row in rows:
        filename = row["filename"]
        stem = Path(filename).stem
        title, date_raw = row["title"], row["date_raw"]
        path = PLANS_DIR / filename

        try:
            full_img = load_full(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  skipping {filename}: {exc}")
            continue
        width, height = full_img.size

        has_bbox = row.get("title_bbox_x0", "") != ""
        bbox = None
        if has_bbox:
            bbox = tuple(float(row[f"title_bbox_{k}"]) for k in ("x0", "y0", "x1", "y1"))

        variants = []  # list of (tag, PIL.Image)
        full_variant = downsample(full_img, FULL_MAX_DIM)
        variants.append(("full", full_variant))

        if bbox:
            px0, py0, px1, py1 = crop_box(bbox, width, height, rng)
            if px1 > px0 and py1 > py0:
                crop_img = downsample(full_img.crop((px0, py0, px1, py1)), CROP_MAX_DIM)
                variants.append(("crop", crop_img))

        if split == "train":
            aug_variants = []
            for tag, base in variants:
                for i in range(N_TRAIN_AUGS):
                    aug_variants.append((f"{tag}_aug{i+1}", augment(base, rng)))
            variants.extend(aug_variants)

        for tag, img in variants:
            out_name = f"{stem}_{tag}.png"
            img.save(IMAGES_DIR / out_name)
            examples.append(make_example(f"images/{out_name}", title, date_raw, rng))

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not GT_CSV.exists():
        print(f"{GT_CSV} not found — run dataset_prep/label_ground_truth.py first.")
        return

    gt = pd.read_csv(GT_CSV, dtype=str, keep_default_na=False)
    labeled = gt[(gt["status"] == "ok") & (gt["title"] != "") & (gt["date_raw"] != "")]
    skipped = len(gt) - len(labeled)
    if skipped:
        print(f"Skipping {skipped} row(s) without status=ok + title + date_raw.")

    rows = labeled.to_dict("records")
    if len(rows) < 6:
        print(f"Only {len(rows)} usable labeled rows — finish labeling before building the training set.")
        return

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n = len(rows)
    n_val = max(1, round(n * args.val_frac))
    n_test = max(1, round(n * args.test_frac))
    n_train = n - n_val - n_test
    if n_train < 1:
        print(f"Not enough rows ({n}) to carve out train/val/test with these fractions.")
        return

    splits = {
        "train": rows[:n_train],
        "val": rows[n_train:n_train + n_val],
        "test": rows[n_train + n_val:],
    }

    # Add a few genuinely date-less plans to the TEST split only (never train/val) so
    # the date-presence confusion matrix (TP/FN/FP/TN) in evaluate.py has real negative
    # examples — otherwise every test plan has a true date and FP/TN are always 0.
    no_date = gt[(gt["status"] == "no_date") & (gt["title"] != "")]
    if len(no_date):
        no_date_rows = no_date.to_dict("records")
        splits["test"] = splits["test"] + no_date_rows
        print(f"\nAdded {len(no_date_rows)} no-date plan(s) to test (for the date-presence "
              f"confusion matrix): {[r['filename'] for r in no_date_rows]}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for split, split_rows in splits.items():
        print(f"\n{split}: {len(split_rows)} base image(s) -> {[r['filename'] for r in split_rows]}")
        examples = build_split(split_rows, split, rng)
        out_path = DATA_DIR / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  wrote {len(examples)} example(s) to {out_path}")


if __name__ == "__main__":
    main()
