"""Builds presentation/BVG_plans_finetuning.pptx summarizing the
ground-truth -> EDA -> LoRA fine-tuning pipeline for title/date extraction
on the BVG construction plans in plans/.

Usage:
    python build_slides.py
"""

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

OUT_PATH = Path(__file__).resolve().parent / "BVG_plans_finetuning.pptx"
CHARTS_DIR = Path(__file__).resolve().parent / "charts"

ACCENT = RGBColor(0x1F, 0x4E, 0x79)      # dark blue
ACCENT_LIGHT = RGBColor(0x5B, 0x9B, 0xD5)  # lighter blue
TEXT_DARK = RGBColor(0x22, 0x22, 0x22)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def set_background(slide, color):
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = color


def add_title_bar(slide, text, subtitle=None):
    set_background(slide, WHITE)

    bar = slide.shapes.add_shape(1, Emu(0), Emu(0), SLIDE_W, Inches(1.3))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    bar.shadow.inherit = False

    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.18), SLIDE_W - Inches(1.0), Inches(0.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(30)
    run.font.bold = True
    run.font.color.rgb = WHITE

    if subtitle:
        p2 = tf.add_paragraph()
        run2 = p2.add_run()
        run2.text = subtitle
        run2.font.size = Pt(14)
        run2.font.color.rgb = RGBColor(0xD9, 0xE6, 0xF2)


def add_bullets(slide, items, top=Inches(1.6), left=Inches(0.7), width=None, font_size=18, bold_first_level=False):
    width = width or (SLIDE_W - Inches(1.4))
    tb = slide.shapes.add_textbox(left, top, width, SLIDE_H - top - Inches(0.4))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        if isinstance(item, tuple):
            text, level = item
        else:
            text, level = item, 0
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        run = p.add_run()
        run.text = ("• " if level == 0 else "‒ ") + text
        run.font.size = Pt(font_size - level * 2)
        run.font.color.rgb = TEXT_DARK
        run.font.bold = bold_first_level and level == 0
        p.space_after = Pt(10)
    return tb


def add_footer(slide, text):
    tb = slide.shapes.add_textbox(Inches(0.5), SLIDE_H - Inches(0.4), SLIDE_W - Inches(1.0), Inches(0.3))
    p = tb.text_frame.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)


def title_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_background(slide, ACCENT)

    tb = slide.shapes.add_textbox(Inches(1.0), Inches(2.6), Inches(11.3), Inches(1.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = "Title & Date Extraction from BVG Construction Plans"
    run.font.size = Pt(40)
    run.font.bold = True
    run.font.color.rgb = WHITE

    tb2 = slide.shapes.add_textbox(Inches(1.0), Inches(4.1), Inches(11.3), Inches(1.0))
    p2 = tb2.text_frame.paragraphs[0]
    run2 = p2.add_run()
    run2.text = "Ground truth labeling → EDA → LoRA fine-tuning of a multimodal LLM (Qwen2-VL)"
    run2.font.size = Pt(20)
    run2.font.color.rgb = RGBColor(0xD9, 0xE6, 0xF2)

    add_footer(slide, "HTW Berlin · F2KI")


def content_slide(prs, title, bullets, subtitle=None, footer=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, title, subtitle)
    add_bullets(slide, bullets)
    if footer:
        add_footer(slide, footer)
    return slide


def flow_slide(prs, title, steps):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, title)

    n = len(steps)
    margin = Inches(0.6)
    gap = Inches(0.25)
    box_w = (SLIDE_W - 2 * margin - gap * (n - 1)) / n
    box_h = Inches(1.5)
    top = Inches(2.8)

    for i, step in enumerate(steps):
        left = margin + i * (box_w + gap)
        box = slide.shapes.add_shape(1, left, top, box_w, box_h)
        box.fill.solid()
        box.fill.fore_color.rgb = ACCENT if i % 2 == 0 else ACCENT_LIGHT
        box.line.fill.background()
        box.shadow.inherit = False
        tf = box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = step
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.color.rgb = WHITE

        if i < n - 1:
            arrow = slide.shapes.add_textbox(left + box_w, top + box_h / 2 - Inches(0.25), gap, Inches(0.5))
            ap = arrow.text_frame.paragraphs[0]
            ap.alignment = PP_ALIGN.CENTER
            ar = ap.add_run()
            ar.text = "→"
            ar.font.size = Pt(20)
            ar.font.bold = True
            ar.font.color.rgb = ACCENT

    add_bullets(
        slide,
        ["Each stage is a standalone script under dataset_prep/ or finetune/ — see GROUND_TRUTH_AND_FINETUNING.md"],
        top=Inches(4.8), font_size=16,
    )


def image_slide(prs, title, image_path, caption=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, title)

    from PIL import Image
    img = Image.open(image_path)
    aspect = img.height / img.width

    max_w = SLIDE_W - Inches(1.4)
    max_h = SLIDE_H - Inches(2.0)
    w = max_w
    h = w * aspect
    if h > max_h:
        h = max_h
        w = h / aspect

    left = (SLIDE_W - w) / 2
    top = Inches(1.6)
    slide.shapes.add_picture(str(image_path), left, top, width=w, height=h)

    if caption:
        add_footer(slide, caption)
    return slide


def build():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    title_slide(prs)

    content_slide(
        prs, "The Problem",
        [
            "42 scanned construction-plan sheets in plans/, spanning decades (some back to 1927)",
            "Each sheet has a title block (“Schriftfeld”) with a title and a date — but only as pixels, not structured data",
            "No OCR/metadata pipeline currently reads title + date specifically",
            "Goal: a model that looks at a scan and returns {title, date}",
        ],
    )

    content_slide(
        prs, "The Dataset",
        [
            "42 files: 39 .tif, 1 .tiff, 2 .pdf",
            "Multiple DIN sheet sizes: A0, A1, A2, A3, and several “extended” banner formats",
            "Large range of scan quality, resolution, and age",
            "Zero existing title/date labels — first task is building ground truth",
        ],
    )

    content_slide(
        prs, "Prototype Approach: Zero-Shot Prompting",
        [
            "Prototype = prompting the multimodal LLM provided by HTW (Ollama server, f2ki-h100-1)",
            "No training involved: the model's weights never change, we only ask it questions",
            "Prompt is domain-specific: tells the model this is a German BVG construction plan, forbids translation and date-guessing",
            "This satisfies the task brief directly — it explicitly allows “multimodal LLMs OR OCR”, not only fine-tuning",
        ],
    )

    flow_slide(
        prs, "Pipeline Overview",
        ["1. Label\nground truth", "2. EDA on\nlabels", "3. Prepare +\naugment dataset", "4. Zero-shot\nprototype", "5. Evaluate\nsystematically"],
    )

    content_slide(
        prs, "Phase 1 — Ground Truth Labeling",
        [
            "dataset_prep/label_ground_truth.py: interactive labeling tool",
            "Loads every file uniformly — .tif, .tiff, .pdf, including 5 large TIFFs the old preview generator failed on",
            "Shows the sheet zoomed to the bottom-right quadrant (usual title-block location)",
            "You draw a box around the title block, then type title / date / status in the terminal",
            "Resumable: saves after every file, skips already-labeled files on rerun",
            "Output: dataset_prep/ground_truth.csv (title, date_raw, date_iso, bbox, status)",
            "All 42 plans labeled — transcription done by a vision-language model reading each scan directly, then reviewed",
        ],
    )

    content_slide(
        prs, "Phase 2 — EDA on Ground Truth",
        [
            "dataset_prep/eda_ground_truth.py merges labels with the existing plans_metadata.csv",
            "Reports: labeling coverage, date-format variety and parse success, title stats",
            "Cross-checks title text against filename-derived status flags (e.g. “ungültig”)",
            "Shows title-block bbox position by DIN sheet size",
            "Flags rows still needing attention — checkpoint before spending GPU time",
        ],
    )

    image_slide(prs, "EDA — Labeling Status", CHARTS_DIR / "status_breakdown.png",
                "34/42 confidently labeled (status=ok); the rest are excluded automatically from training")

    image_slide(prs, "EDA — Historical Span", CHARTS_DIR / "dates_by_decade.png",
                "Dates parsed to ISO format; sheets range from 1900 to 2013")

    image_slide(prs, "EDA — Prepared Training Set", CHARTS_DIR / "split_examples.png",
                "finetune/prepare_dataset.py: base plans -> full-page + title-crop + augmented variants")

    content_slide(
        prs, "Phase 3 — Dataset Preparation",
        [
            "finetune/prepare_dataset.py builds the LoRA training set",
            "Splits train/val/test at the base-image level first (no leakage between augmented siblings)",
            "Train: full page + title-block crop, each with 2 light augmentations (rotation, brightness/contrast)",
            ("~35 labeled images → roughly 200 training examples", 1),
            "Val/test: one clean full-page example per plan — matches real inference input",
            "Output: ShareGPT-style JSONL + images, ready for LLaMA-Factory",
        ],
    )

    content_slide(
        prs, "LoRA Fine-Tuning — Attempted, Blocked by Hardware",
        [
            "Full pipeline built and ready: finetune/prepare_dataset.py, finetune/qwen2vl_lora.yaml (Qwen2-VL-7B, LoRA rank 16), LLaMA-Factory config",
            "Checked available hardware: neither team laptop has a discrete GPU (Intel/no NVIDIA) — bf16/CUDA training is not possible on them",
            "The HTW “Ollama server” endpoint is inference-only — it cannot run a training job",
            "Kept as optional bonus if real GPU access becomes available; not required for the prototype",
        ],
    )

    content_slide(
        prs, "Evaluation",
        [
            "finetune/evaluate.py --skip-finetuned runs the zero-shot prototype on the held-out test split",
            "Per-example: predicted vs true title (similarity score) and date (exact / reformatted / no match)",
            "Exports finetune/eval_zero_shot.csv — one row per plan, ready for reporting/spreadsheet review",
            "Same script also supports a fine-tuned column if/when a LoRA checkpoint exists",
        ],
    )

    if (CHARTS_DIR / "eval_confusion_matrix.png").exists():
        image_slide(prs, "Evaluation — Date Presence (TP/TN/FN/FP)",
                    CHARTS_DIR / "eval_confusion_matrix.png",
                    "TP/TN = correct behavior; FN = missed a real date; FP = hallucinated one that isn't there")

    if (CHARTS_DIR / "eval_pass_fail.png").exists():
        image_slide(prs, "Evaluation — Zero-Shot Accuracy",
                    CHARTS_DIR / "eval_pass_fail.png",
                    "Title similarity threshold 0.6; date counts exact + reformatted matches and correct abstentions")

    content_slide(
        prs, "Status — What's Done",
        [
            "All 42 plans labeled: 34 ok, 6 no_date, 1 no_title, 1 unreadable — low-confidence rows flagged in notes for review",
            "Date parser handles German formats (DD.MM.YYYY, month-name-only, MM/YY, etc.) — 27/42 dates parsed to ISO",
            "Training set built: 20 plans -> 120 train examples, 3 plans -> 6 val, 3 plans -> 6 test (ready if GPU access appears)",
            "Working prototype: zero-shot extraction via HTW API + Streamlit demo UI (finetune/streamlit_app.py)",
            "Code, ground truth, and prepared data all pushed to GitHub",
        ],
    )

    content_slide(
        prs, "Next Steps",
        [
            "Run finetune/evaluate.py --skip-finetuned on Eduroam, review finetune/eval_zero_shot.csv",
            "Demo the prototype: streamlit run finetune/streamlit_app.py",
            "Optional: spot-check flagged low-confidence rows in dataset_prep/ground_truth.csv",
            "Optional bonus, only if GPU access materializes: llamafactory-cli train finetune/qwen2vl_lora.yaml",
        ],
    )

    thanks = prs.slides.add_slide(prs.slide_layouts[6])
    set_background(thanks, ACCENT)
    tb = thanks.shapes.add_textbox(Inches(1.0), Inches(3.2), Inches(11.3), Inches(1.2))
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = "Thank you — Questions?"
    run.font.size = Pt(36)
    run.font.bold = True
    run.font.color.rgb = WHITE

    prs.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
