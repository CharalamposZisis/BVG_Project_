# Fine-tuning: title/date extraction on BVG plans

Full pipeline: `dataset_prep/label_ground_truth.py` (labels) →
`dataset_prep/eda_ground_truth.py` (sanity check) → `prepare_dataset.py` (this
folder, builds the training set) → LoRA training on the HTW GPU cluster →
`evaluate.py` (zero-shot vs fine-tuned).

## 0. Label the plans (local machine, no GPU needed)

```
cd dataset_prep
python label_ground_truth.py
```

Draw a box around the title block, then close the plot window and answer the
title/date/status prompts in the terminal. Re-run any time — it skips files
already in `ground_truth.csv`. Labeling a `.pdf` plan needs `pip install
pymupdf` first.

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
pip install torch transformers accelerate peft "llamafactory[torch,metrics]" qwen-vl-utils

cd finetune
llamafactory-cli train qwen2vl_lora.yaml
```

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
pip install streamlit
streamlit run finetune/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Then open `http://<that machine's IP>:8501` from any other laptop on the same
network. Upload a plan image, tick "Zero-shot" and/or "Fine-tuned" in the
sidebar, hit Run. The zero-shot panel only needs `openai` + network access to
the HTW API; the fine-tuned panel needs the same `torch`/`transformers`/`peft`/
`qwen-vl-utils` stack as training, plus the adapter dir from step 2.

## Notes / known constraints

- The `f2ki-h100-1` endpoint used for zero-shot (`BVG_Project/test_htw_infrastructure/test_htw_vision.py`)
  is an inference-only Ollama API — it cannot run training, hence step 2 needs direct cluster access.
- 42 base images is a small fine-tuning set. `prepare_dataset.py` augments
  (crop + full-page variants, light rotation/brightness jitter) to get to
  roughly 200 training examples, but this mainly adapts the model to this
  specific title-block layout and date style — it is not a substitute for
  more labeled data if the accuracy gain over zero-shot turns out to be small.
