# BVG Plans: Title and Date Extraction

Pipeline for extracting the title and date from the title block ("Schriftfeld")
of scanned historical construction plans, developed for the BIP project with
BVG (Berliner Verkehrsbetriebe) at HTW Berlin.

The underlying scan collection is confidential per the course terms and is not
included in this repository. This repository contains only the pipeline code.

## Data placement

Nothing under this heading is included in the repository; it must be added
locally before running anything, in exactly these locations (relative to the
project root):

| What | Where it goes | Needed for |
|---|---|---|
| Raw plan scans (`.tif`/`.tiff`/`.pdf`) | `plans/` | Labeling (`dataset_prep/label_ground_truth.py`) |
| `ground_truth.csv` (if supplied directly, rather than produced by labeling) | `dataset_prep/ground_truth.csv` | Everything in `finetune/` |
| Prepared train/val/test images (if supplied directly, rather than produced by `prepare_dataset.py`) | `finetune/data/images/` | `evaluate.py`, `streamlit_app.py`, `analyze_failures.py` |

If you have the raw scans, run `dataset_prep/label_ground_truth.py` and
`finetune/prepare_dataset.py` yourself to generate everything else (see
Usage below). If you were instead handed an already-prepared
`ground_truth.csv` and/or `finetune/data/` folder, just place them at the
paths above directly — no need to regenerate them.

## Approach

The prototype extracts title and date by prompting a multimodal LLM (the HTW
Berlin inference API) with each plan image. No model weights are trained or
modified; this is zero-shot prompting, not fine-tuning. A LoRA fine-tuning
pipeline for the same task is included but unused, since running it requires
GPU hardware that was not available on the project laptops. See
`finetune/README.md` for details on both.

## Repository structure

```
dataset_prep/
  plan_io.py              Shared loader for .tif/.tiff/.pdf plan scans
  label_ground_truth.py   Interactive tool to record title/date/bbox per plan
  eda_ground_truth.py     Coverage, date-format, and quality report over labels
  build_metadata.py       Image-level metadata (size, DPI, sheet format)
  eda.py                  Report over that metadata

finetune/
  prepare_dataset.py      Builds a train/val/test split with crop/augmentation
  evaluate.py              Zero-shot (and optional fine-tuned) evaluation,
                            including a date-presence confusion matrix
  analyze_failures.py     Cross-references evaluation failures with labeling
                            confidence notes
  streamlit_app.py        Demo UI: single-image lookup and a test-set results
                            gallery
  quick_test.py            Minimal single-image API smoke test
  qwen2vl_lora.yaml         LoRA fine-tuning config (LLaMA-Factory), unused
  dataset_info.json         Dataset registration for the above
  README.md                 Step-by-step usage for everything in this folder

presentation/
  build_slides.py           Generates the project status slide deck
  build_eda_charts.py       Generates the supporting charts
```

## Setup

```
pip install -r requirements.txt
```

This installs everything needed except LoRA training, which requires an
NVIDIA GPU; see the comment block at the end of `requirements.txt` and
`finetune/README.md` before installing those packages.

## Usage

1. Label ground truth: `python dataset_prep/label_ground_truth.py`
2. Check label quality: `python dataset_prep/eda_ground_truth.py`
3. Build the dataset split: `python finetune/prepare_dataset.py`
4. Evaluate the zero-shot prototype: `python finetune/evaluate.py --skip-finetuned`
5. Inspect failures: `python finetune/analyze_failures.py`
6. Run the demo: `streamlit run finetune/streamlit_app.py`

Full detail on each step, including required network access, is in
`finetune/README.md`.
