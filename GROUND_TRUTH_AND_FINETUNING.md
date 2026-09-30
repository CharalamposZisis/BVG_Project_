# Ground truth → EDA → LoRA fine-tuning: what was built and why

This documents the pipeline added to this project to train a multimodal LLM
to read the **title** and **date** out of each scanned BVG construction plan
in `plans/`.

## Starting point

- `plans/`: 42 scanned sheets (39 `.tif`, 1 `.tiff`, 2 `.pdf`), spanning
  decades — some filenames encode dates back to 1927.
- `dataset_prep/build_metadata.py` + `eda.py`: already existed, capture
  image-level metadata (dimensions, DPI, DIN sheet size, filename-derived
  status flags like "ungültig") — but nothing about the actual title-block
  content.
- `OCR/`: a generic PaddleOCR example pipeline (not specific to this
  dataset).
- `BVG_Project/test_htw_infrastructure/`: confirms access to an
  OpenAI-compatible vision model at HTW Berlin
  (`f2ki-h100-1.f2.htw-berlin.de`) — **inference only**, it cannot run a
  training job.
- No GPU, no `torch`/`transformers`/`peft` locally. Training runs on the HTW
  GPU cluster via direct SSH/Jupyter access.

There was no ground truth for title/date yet, and 42 images is too small to
fine-tune a multimodal model from scratch without labeling + augmentation.
So the work was split into three phases.

## Phase 1 — Ground truth labeling

**Files:** `dataset_prep/plan_io.py`, `dataset_prep/label_ground_truth.py`

`plan_io.py` is a shared loader that opens any plan file — `.tif`, `.tiff`,
or `.pdf` — into a Pillow image at full or downsampled resolution. This was
needed because the existing `plans/_preview` PNGs (generated once with
ImageMagick) only cover 37 of the 42 files: ImageMagick ran out of cache on
the 5 largest TIFFs, and the 2 PDFs were never rendered. `plan_io.py` reads
the TIFFs directly with `tifffile`/Pillow and renders PDFs with `pymupdf`,
so every file loads the same way regardless of size or format.

`label_ground_truth.py` is the interactive labeling tool:

1. Opens a matplotlib window per plan, pre-zoomed to the bottom-right
   quadrant (the usual location of a German title block / "Schriftfeld"),
   with `f`/`r` to toggle full view.
2. You drag a box around the title block.
3. It then asks, in the terminal, for the title text, the date exactly as
   written, and a status (`ok` / `no_date` / `no_title` / `unreadable`).
4. Everything is written to `dataset_prep/ground_truth.csv` — one row per
   file, including the bounding box as a fraction of the sheet (so it's
   resolution-independent) and a best-effort ISO-normalized date.
5. It's resumable: rerunning skips files already labeled, so labeling can be
   done in several short sessions instead of one sitting.

## Phase 2 — EDA on the labels

**File:** `dataset_prep/eda_ground_truth.py`

Once some rows exist in `ground_truth.csv`, this merges them with
`plans_metadata.csv` and reports:

- labeling coverage (how many of 42 are done, and which are left)
- which date formats show up and whether they parsed to ISO
  (`DD.MM.YYYY`, `DD.MM.YY`, German month names, etc.)
- title length/duplicate stats, and whether titles mentioning "ungültig"
  etc. line up with the filename-derived status flags
- where the title-block bounding box tends to sit, grouped by DIN sheet
  size — tells us whether a single default crop region is reasonable
- which rows still need attention before moving to fine-tuning

This is the checkpoint before spending any GPU time: it flags bad/missing
labels early.

## Phase 3 — Fine-tuning pipeline

**Files:** `finetune/prepare_dataset.py`, `finetune/dataset_info.json`,
`finetune/qwen2vl_lora.yaml`, `finetune/evaluate.py`, `finetune/README.md`

`prepare_dataset.py` turns the ground truth into a LoRA training set:

- Splits the labeled plans into train/val/test **at the base-image level
  first** (so augmented copies of the same plan never end up split across
  train and test).
- For each **train** plan, generates: the full page (downscaled, matching
  what a real inference call would see), a crop of just the title block
  (from the labeled bounding box), and 2 light augmentations (small
  rotation, brightness/contrast jitter) of each — about 6 examples per
  labeled plan, to stretch ~35 base images into roughly 200 training
  examples.
- **Val/test** get exactly one clean, full-page example per plan — the
  realistic inference-time input, no augmentation, so evaluation numbers
  reflect real usage.
- Writes ShareGPT-style JSONL that LLaMA-Factory expects for multimodal
  fine-tuning.

The model target is **Qwen2-VL-7B-Instruct**, fine-tuned with **LoRA**
(`qwen2vl_lora.yaml`, run via `llamafactory-cli train`) — an open,
document-capable vision-language model that's small enough to train on a
single H100 without quantization, and already has strong OCR/document
priors to build on rather than learning to read from nothing.

`evaluate.py` runs the **test** split two ways and prints a comparison:
zero-shot (calling the existing HTW inference API, same client pattern as
`test_htw_vision.py`) versus the fine-tuned LoRA checkpoint, scoring title
similarity and date match. This exists specifically so the gain from
fine-tuning is *measured*, not assumed — with only ~35 base training
images, that gain might be modest, especially on the open-vocabulary title
field.

`README.md` in `finetune/` has the exact command sequence from labeling
through cluster training to evaluation.

## What was verified

- The `plan_io` loader was tested against a normal TIFF, one of the 5 TIFFs
  that failed the old ImageMagick preview, and a PDF — all load correctly.
- The date parser was unit-tested against `12.05.1927`, `5.1.27`,
  `03.1965`, `1948`, and `12. Mai 1927` — all normalize correctly.
- `eda_ground_truth.py` and `prepare_dataset.py` were dry-run against
  synthetic labels to confirm the merge, split, crop/augmentation, and
  JSONL output all work end-to-end; the synthetic data was removed
  afterward so it doesn't get mistaken for real labels.
- Actual LoRA training and the fine-tuned half of `evaluate.py` need the
  HTW GPU cluster and haven't been run yet — that's the next step, after
  labeling is complete.

## What's left to do

1. Run `python dataset_prep/label_ground_truth.py` and label all 42 plans.
2. Run `python dataset_prep/eda_ground_truth.py` to sanity-check the labels.
3. Copy `finetune/` to the HTW cluster and follow `finetune/README.md`.
