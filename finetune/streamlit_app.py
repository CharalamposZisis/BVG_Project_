"""Demo UI: upload a plan image, get title + date back from the model(s).

Meant to run on whichever machine holds the fine-tuned checkpoint (the GPU
box), so it can load the model once and serve requests. Other people reach it
over the network at http://<that machine's IP>:8501.

Two independent panels, either can be used alone:
  - Fine-tuned (local): loads the LoRA checkpoint with transformers/peft.
    Needs torch/transformers/peft/qwen-vl-utils installed (see finetune/README.md)
    and enough GPU memory for Qwen2-VL-7B. Only imported if you tick the box,
    so the app still runs (zero-shot only) on a machine without those installed.
  - Zero-shot (HTW API): calls the existing inference endpoint, works with
    just `pip install openai streamlit`, no GPU needed.

Run:
    streamlit run finetune/streamlit_app.py
    streamlit run finetune/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
"""

import base64
import csv
import mimetypes
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

TITLE_PASS_THRESHOLD = 0.6

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "dataset_prep"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import parse_json_answer, HTW_API_KEY, HTW_BASE_URL, HTW_MODEL, ZERO_SHOT_PROMPT  # noqa: E402
from analyze_failures import base_filename, match_ground_truth_filename, is_low_confidence  # noqa: E402

PROMPT = ZERO_SHOT_PROMPT

st.set_page_config(page_title="BVG Plan — Title & Date Extraction", page_icon="📐", layout="wide")
st.title("📐 BVG Plan — Title & Date Extraction")
st.caption("Upload a scanned plan (or a crop of its title block) and see what each model reads.")

with st.sidebar:
    st.header("Settings")
    use_zero_shot = st.checkbox("Zero-shot (HTW API)", value=True)
    use_finetuned = st.checkbox("Fine-tuned (local LoRA checkpoint)", value=False)
    adapter_dir = st.text_input("Adapter dir", value="finetune/output/qwen2vl_lora")
    base_model = st.text_input("Base model", value="Qwen/Qwen2-VL-7B-Instruct")
    st.caption("Fine-tuned inference needs torch/transformers/peft/qwen-vl-utils "
               "and a GPU — see finetune/README.md.")


@st.cache_resource(show_spinner="Loading fine-tuned model (first request only)...")
def load_finetuned(base_model_name: str, adapter_path: str):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return processor, model


def run_finetuned(image_path: str, base_model_name: str, adapter_path: str) -> dict:
    import torch
    from qwen_vl_utils import process_vision_info

    processor, model = load_finetuned(base_model_name, adapter_path)
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image_path}, {"type": "text", "text": PROMPT}],
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
    return parse_json_answer(output_text)


def run_zero_shot(image_path: str) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=HTW_API_KEY, base_url=HTW_BASE_URL)
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/png"
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    response = client.chat.completions.create(
        model=HTW_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
            ],
        }],
        temperature=0.0,
    )
    return parse_json_answer(response.choices[0].message.content)


def load_eval_rows(csv_path: Path) -> list:
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_verdict(row: dict) -> dict:
    title_ok = float(row["title_similarity"]) >= TITLE_PASS_THRESHOLD
    date_ok = row["date_presence_class"] == "TN" or row["date_result"] in ("exact", "same_date_different_format")
    return {"title_ok": title_ok, "date_ok": date_ok, "overall_ok": title_ok and date_ok}


@st.cache_data
def load_ground_truth():
    gt_path = PROJECT_ROOT / "dataset_prep" / "ground_truth.csv"
    if not gt_path.exists():
        return None
    import pandas as pd
    return pd.read_csv(gt_path, dtype=str, keep_default_na=False)


def get_image_quality(filename: str, gt_df) -> str:
    if gt_df is None:
        return "Unknown"
    base = base_filename(filename)
    gt_name = match_ground_truth_filename(base, gt_df["filename"].tolist())
    if not gt_name:
        return "Unknown"
    row = gt_df[gt_df["filename"] == gt_name].iloc[0]
    if row["status"] != "ok" or is_low_confidence(row["notes"]):
        return "Bad"
    return "Ok"


tab_try, tab_results = st.tabs(["🔍 Try an image", "📊 Test set results"])

with tab_try:
    uploaded = st.file_uploader("Plan image", type=["png", "jpg", "jpeg", "tif", "tiff"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        col_img, col_results = st.columns([1, 1])

        with col_img:
            st.image(image, caption=uploaded.name, use_container_width=True)

        tmp_path = PROJECT_ROOT / "finetune" / "_streamlit_tmp.png"
        image.save(tmp_path)

        if st.button("Run", type="primary"):
            with col_results:
                if use_zero_shot:
                    st.subheader("Zero-shot (HTW API)")
                    try:
                        with st.spinner("Calling HTW API..."):
                            result = run_zero_shot(str(tmp_path))
                        st.json(result)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Zero-shot call failed: {exc}")

                if use_finetuned:
                    st.subheader("Fine-tuned (LoRA)")
                    if not Path(adapter_dir).exists():
                        st.warning(f"Adapter dir not found: {adapter_dir}")
                    else:
                        try:
                            with st.spinner("Running fine-tuned model..."):
                                result = run_finetuned(str(tmp_path), base_model, adapter_dir)
                            st.json(result)
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Fine-tuned inference failed: {exc}")

                if not use_zero_shot and not use_finetuned:
                    st.info("Tick at least one model in the sidebar.")
    else:
        st.info("Upload an image to get started.")

with tab_results:
    csv_path = PROJECT_ROOT / "finetune" / "eval_zero_shot.csv"
    rows = load_eval_rows(csv_path)

    if not rows:
        st.info(
            "No results yet. Run this first, then reload this page:\n\n"
            "```\ncd finetune\npython evaluate.py --skip-finetuned\n```"
        )
    else:
        verdicts = [row_verdict(r) for r in rows]
        n = len(rows)
        n_title_ok = sum(v["title_ok"] for v in verdicts)
        n_date_ok = sum(v["date_ok"] for v in verdicts)
        n_both_ok = sum(v["overall_ok"] for v in verdicts)

        c1, c2, c3 = st.columns(3)
        c1.metric("Title recognized", f"{n_title_ok}/{n}")
        c2.metric("Date correct", f"{n_date_ok}/{n}")
        c3.metric("Both correct", f"{n_both_ok}/{n}")

        show_only_failures = st.checkbox("Show only failures")

        gt_df = load_ground_truth()
        images_dir = PROJECT_ROOT / "finetune" / "data" / "images"
        cols = st.columns(4)
        i = 0
        for row, verdict in zip(rows, verdicts):
            if show_only_failures and verdict["overall_ok"]:
                continue

            col = cols[i % 4]
            i += 1
            with col:
                img_path = images_dir / row["filename"]
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)

                badge = "✅" if verdict["overall_ok"] else "❌"
                st.markdown(f"**{badge} {row['filename']}**")

                title_mark = "✅" if verdict["title_ok"] else "❌"
                st.markdown(f"{title_mark} **Title** (similarity = {float(row['title_similarity']):.0%})")
                st.caption(f"True: {row['true_title'][:60]}")
                st.caption(f"Predicted: {row['pred_title'][:60]}")

                date_mark = "✅" if verdict["date_ok"] else "❌"
                st.markdown(f"{date_mark} **Date**")
                st.caption(f"True: {row['true_date'] or '(none)'}")
                st.caption(f"Predicted: {row['pred_date'] or '(none)'}")

                quality = get_image_quality(row["filename"], gt_df)
                st.markdown(f"**Image Quality:** {quality}")
                st.divider()

        if i == 0:
            st.success("No failures — everything in the test set passed both checks.")
