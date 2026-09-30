"""Simplest possible smoke test: send ONE plan image to the HTW API, print the result.

No project imports, no JSONL, no matplotlib — only needs `pip install openai`.
If this doesn't work, the problem is the API/network, not the rest of the pipeline.

Usage:
    python quick_test.py "../plans/GLW-SM 01.tif"
    python quick_test.py path\to\some_plan.png
"""

import base64
import mimetypes
import sys

API_KEY = "sk-LVatN8KX-mbfvLbCWLsNzg"
BASE_URL = "https://f2ki-h100-1.f2.htw-berlin.de:11435/v1"
MODEL = "qwen3.8-27b"

PROMPT = (
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_test.py <path-to-image>")
        return

    image_path = sys.argv[1]
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/png"

    print(f"Reading {image_path} ...")
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    print("Connecting to HTW API ...")
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120.0)

    print("Sending request (this can take up to ~1 min if the server is busy) ...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
            ],
        }],
        temperature=0.0,
    )

    print("\n=== RAW MODEL OUTPUT ===")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
