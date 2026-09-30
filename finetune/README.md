# Fine-tuning: title/date extraction on BVG plans

Full pipeline: `dataset_prep/label_ground_truth.py` (labels) →
`dataset_prep/eda_ground_truth.py` (sanity check) → `prepare_dataset.py` (this
folder, builds the training set) → LoRA training on the HTW GPU cluster →
`evaluate.py` (zero-shot vs fine-tuned).

## Setup (once, from the project root)

```
pip install -r requirements.txt
```

Covers everything below except actual training (step 2), which needs a real
GPU — see `requirements-training.txt` and step 2.

## 0. Label the plans (local machine, no GPU needed)

```
cd dataset_prep
python label_ground_truth.py
```

Draw a box around the title block, then close the plot window and answer the
title/date/status prompts in the terminal. Re-run any time — it skips files
already in `ground_truth.csv`.

Check progress and data quality with:

```
python eda_ground_truth.py
```

Aim for all 42 rows with `status=ok` and a parsed `date_iso` before moving on.

## 1. Build the training set (local machine)

```
cd finetune
python prepare_dataset.py
```

Writes `finetune/data/{train,val,test}.jsonl` and the crop/augmented PNGs
under `finetune/data/images/`. Pure PIL/numpy — no GPU needed, safe to
re-run and spot-check locally. Copy (or `rsync`) the whole `finetune/`
folder, including `data/`, to the HTW cluster.

## 2. Train (HTW GPU cluster — needs actual GPU access, not the inference API)

```
python -m venv ~/venvs/llamafactory
source ~/venvs/llamafactory/bin/activate
pip install -r requirements-training.txt

cd finetune
llamafactory-cli train qwen2vl_lora.yaml
```

Check GPU/CUDA is actually available *before* installing anything:
`python -c "import torch; print(torch.cuda.is_available())"` — if that's
`False` on Windows, you likely got the CPU-only torch build; reinstall from
https://pytorch.org/get-started/locally/ with the right CUDA version.

This downloads `Qwen/Qwen2-VL-7B-Instruct` from the Hugging Face Hub on first
run. The adapter is written to `finetune/output/qwen2vl_lora`. Watch the
`eval_loss` in the logs — with ~35 base training images this will overfit
fast; if val loss stops improving well before `num_train_epochs` finishes,
lower `num_train_epochs` in `qwen2vl_lora.yaml` and re-run.

## 3. Evaluate

Zero-shot only (runs anywhere with `pip install openai`, no GPU):

```
python evaluate.py --skip-finetuned
```

Full zero-shot vs fine-tuned comparison (run on the cluster, after step 2):

```
python evaluate.py --adapter-dir finetune/output/qwen2vl_lora
```

Prints per-example title similarity and date match, plus aggregate scores
for both models. This is the number that tells you whether fine-tuning
actually helped over just prompting the base model — with this little data,
treat any single run's numbers as noisy; look at the qualitative per-example
output too, not just the aggregate.

## 4. Demo UI (Streamlit)

Run this on whichever machine holds the fine-tuned checkpoint (the GPU box) —
it loads the model once and serves requests to anyone on the network:

```
streamlit run finetune/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Then open `http://<that machine's IP>:8501` from any other laptop on the same
network. Two tabs: "Try an image" (upload one, tick "Zero-shot" and/or
"Fine-tuned" in the sidebar, hit Run) and "Test set results" (a gallery of the
whole test split with pass/fail badges, once `eval_zero_shot.csv` exists —
see step 3). The zero-shot panel only needs `openai` + network access to the
HTW API; the fine-tuned panel needs `requirements-training.txt` plus the
adapter dir from step 2.

## 5. Failure analysis

```
python analyze_failures.py
```

Cross-references which test-set images the model got wrong against the
confidence notes recorded during labeling (`dataset_prep/ground_truth.csv`) —
answers "does the model mainly fail on scans that were already hard to read?"
with actual numbers, not just impressions.

## Notes / known constraints

- The `f2ki-h100-1` endpoint used for zero-shot (`BVG_Project/test_htw_infrastructure/test_htw_vision.py`)
  is an inference-only Ollama API — it cannot run training, hence step 2 needs direct cluster access.
- 42 base images is a small fine-tuning set. `prepare_dataset.py` augments
  (crop + full-page variants, light rotation/brightness jitter) to get to
  roughly 200 training examples, but this mainly adapts the model to this
  specific title-block layout and date style — it is not a substitute for
  more labeled data if the accuracy gain over zero-shot turns out to be small.
