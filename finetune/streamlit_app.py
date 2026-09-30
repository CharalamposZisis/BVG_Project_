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
import mimetypes
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "dataset_prep"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import parse_json_answer, HTW_API_KEY, HTW_BASE_URL, HTW_MODEL, ZERO_SHOT_PROMPT  # noqa: E402

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
