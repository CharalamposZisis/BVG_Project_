"""Compare zero-shot vs fine-tuned title/date extraction on the held-out test split.

Zero-shot: calls the HTW inference API (same OpenAI-compatible client pattern as
BVG_Project/test_htw_infrastructure/test_htw_vision.py) with the base vision model.

Fine-tuned: loads the LoRA adapter produced by `llamafactory-cli train
finetune/qwen2vl_lora.yaml` on top of the base Qwen2-VL model and runs local
generation. Requires torch/transformers/peft/qwen-vl-utils — only available on
the training machine (see finetune/README.md), so this half is optional
(--skip-finetuned) and importable-only-when-used.

Usage (zero-shot only, runs anywhere with the `openai` package):
    python evaluate.py --skip-finetuned

Usage (full comparison, run on the HTW cluster after training):
    python evaluate.py --adapter-dir finetune/output/qwen2vl_lora
"""

import argparse
import base64
import difflib
import json
import mimetypes
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "dataset_prep"))
from label_ground_truth import parse_date_raw  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"

HTW_API_KEY = os.environ.get("HTW_API_KEY", "sk-LVatN8KX-mbfvLbCWLsNzg")
HTW_BASE_URL = os.environ.get("HTW_BASE_URL", "https://f2ki-h100-1.f2.htw-berlin.de:11435/v1")
HTW_MODEL = os.environ.get("HTW_MODEL", "qwen3.8-27b")

# Used for zero-shot calls instead of whatever prompt is baked into the JSONL
# (that one is phrased for training consistency, not necessarily the best
# zero-shot instruction). More specific = less guessing.
ZERO_SHOT_PROMPT = (
    "This is a scanned German construction plan (Berlin U-Bahn / BVG), dating "
    "from 1900-2013. Find the title block stamp. Transcribe the title EXACTLY "
    "as written, in German — do not translate it. Transcribe the date EXACTLY "
    "as written. Only report a date if you can clearly see it printed or "
    "handwritten as a date in the title block itself — never infer one from a "
    "scale, a drawing number, a reference code, or any other unrelated number "
    "on the sheet. If you are unsure whether a number is actually a date, or "
    "you cannot find a date at all, return an empty string for date — an "
    'empty answer is correct far more often than a guess. Respond with JSON: '
    '{"title": ..., "date": ...}.'
)


def load_test_set(path: Path) -> list:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            user_msg, assistant_msg = row["messages"][0], row["messages"][1]
            prompt = user_msg["content"].replace("<image>", "").strip()
            true_answer = json.loads(assistant_msg["content"])
            image_path = DATA_DIR / row["images"][0]
            examples.append({"image_path": image_path, "prompt": prompt, "true": true_answer})
    return examples


def parse_json_answer(text) -> dict:
    if not text:
        return {"title": "", "date": ""}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"title": "", "date": ""}


def title_similarity(pred: str, true: str) -> float:
    return difflib.SequenceMatcher(None, pred.strip().lower(), true.strip().lower()).ratio()


def date_presence_class(pred: str, true: str) -> str:
    has_true, has_pred = bool(true.strip()), bool(pred.strip())
    if has_true and has_pred:
        return "TP"
    if has_true and not has_pred:
        return "FN"
    if not has_true and has_pred:
        return "FP"
    return "TN"


def date_match(pred: str, true: str) -> str:
    pred, true = pred.strip(), true.strip()
    if pred == true and pred:
        return "exact"
    if parse_date_raw(pred) and parse_date_raw(pred) == parse_date_raw(true):
        return "same_date_different_format"
    return "no_match"


def run_zero_shot(examples: list) -> list:
    from openai import OpenAI

    client = OpenAI(api_key=HTW_API_KEY, base_url=HTW_BASE_URL, timeout=120.0)
    predictions = []
    n = len(examples)
    for i, ex in enumerate(examples, 1):
        name = ex["image_path"].name
        print(f"  [{i}/{n}] {name} ...", end=" ", flush=True)
        try:
            mime_type, _ = mimetypes.guess_type(str(ex["image_path"]))
            mime_type = mime_type or "image/png"
            with open(ex["image_path"], "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
            response = client.chat.completions.create(
                model=HTW_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ZERO_SHOT_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                    ],
                }],
                temperature=0.0,
            )
            pred = parse_json_answer(response.choices[0].message.content)
            print("ok" if (pred.get("title") or pred.get("date")) else "empty response")
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED ({exc})")
            pred = {"title": "", "date": ""}
        predictions.append(pred)
    return predictions


def run_finetuned(examples: list, adapter_dir: str, base_model: str) -> list:
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    predictions = []
    for ex in examples:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(ex["image_path"])},
                {"type": "text", "text": ex["prompt"]},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=256)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        output_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        predictions.append(parse_json_answer(output_text))
    return predictions


def score(examples: list, predictions: list, label: str, csv_path: Path = None):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    title_scores, date_results, presence_classes = [], [], []
    rows = []
    for ex, pred in zip(examples, predictions):
        pred_title, pred_date = pred.get("title", ""), pred.get("date", "")
        t_sim = title_similarity(pred_title, ex["true"]["title"])
        d_res = date_match(pred_date, ex["true"]["date"])
        d_class = date_presence_class(pred_date, ex["true"]["date"])
        title_scores.append(t_sim)
        date_results.append(d_res)
        presence_classes.append(d_class)
        print(f"  {ex['image_path'].name}")
        print(f"    title  true={ex['true']['title']!r} pred={pred_title!r}  sim={t_sim:.2f}")
        print(f"    date   true={ex['true']['date']!r} pred={pred_date!r}  -> {d_res} ({d_class})")
        rows.append({
            "filename": ex["image_path"].name,
            "true_title": ex["true"]["title"], "pred_title": pred_title, "title_similarity": round(t_sim, 3),
            "true_date": ex["true"]["date"], "pred_date": pred_date, "date_result": d_res,
            "date_presence_class": d_class,
        })

    n = len(examples)
    exact = sum(r == "exact" for r in date_results)
    close = sum(r == "same_date_different_format" for r in date_results)
    tp = presence_classes.count("TP")
    fn = presence_classes.count("FN")
    fp = presence_classes.count("FP")
    tn = presence_classes.count("TN")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) and precision == precision and recall == recall and (precision + recall) > 0 else float("nan")

    print(f"\n  mean title similarity: {sum(title_scores) / n:.3f}")
    print(f"  date exact match:      {exact}/{n}")
    print(f"  date same-but-reformatted: {close}/{n}")
    print(f"\n  date-presence confusion matrix: TP={tp} FN={fn} FP={fp} TN={tn}")
    print(f"  date-presence precision={precision:.2f} recall={recall:.2f} F1={f1:.2f}")
    if fp == 0 and tn == 0:
        print("  (no true-negative examples in this test set — it can't measure hallucination "
              "on genuinely date-less plans; add some 'no_date' plans to test.jsonl for that)")

    if csv_path:
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  wrote {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-jsonl", default=str(DATA_DIR / "test.jsonl"))
    parser.add_argument("--adapter-dir", default="finetune/output/qwen2vl_lora")
    parser.add_argument("--base-model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--skip-zero-shot", action="store_true")
    parser.add_argument("--skip-finetuned", action="store_true")
    parser.add_argument("--csv-dir", default=str(DATA_DIR.parent), help="Where to write eval_*.csv")
    parser.add_argument("--filter", help="Only test images whose filename contains this substring "
                                          "(e.g. --filter S_112_002 to re-test just one plan)")
    args = parser.parse_args()

    test_path = Path(args.test_jsonl)
    if not test_path.exists():
        print(f"{test_path} not found — run finetune/prepare_dataset.py first.")
        return
    Path(args.csv_dir).mkdir(parents=True, exist_ok=True)
    examples = load_test_set(test_path)
    if args.filter:
        examples = [ex for ex in examples if args.filter in ex["image_path"].name]
    print(f"Loaded {len(examples)} test example(s) from {test_path}"
          + (f" (filtered by {args.filter!r})" if args.filter else ""))
    if not examples:
        print("No examples matched the filter.")
        return

    if not args.skip_zero_shot:
        score(examples, run_zero_shot(examples), f"Zero-shot ({HTW_MODEL} via HTW API)",
              csv_path=Path(args.csv_dir) / "eval_zero_shot.csv")

    if not args.skip_finetuned:
        if not Path(args.adapter_dir).exists():
            print(f"\nAdapter dir {args.adapter_dir} not found — skipping fine-tuned eval. "
                  f"Train first with: llamafactory-cli train finetune/qwen2vl_lora.yaml")
        else:
            score(examples, run_finetuned(examples, args.adapter_dir, args.base_model),
                  f"Fine-tuned ({args.base_model} + LoRA {args.adapter_dir})",
                  csv_path=Path(args.csv_dir) / "eval_finetuned.csv")


if __name__ == "__main__":
    main()
