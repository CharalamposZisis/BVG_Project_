"""Shared image loading for the plans/ scans (tif/tiff/pdf).

Used by both label_ground_truth.py (downsampled preview for labeling) and
finetune/prepare_dataset.py (full-resolution load for cropping/augmenting).
"""

from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # legitimate large scans, not decompression bombs


def load_full(path: Path) -> Image.Image:
    """Load a plan file at (near-)native resolution as an RGB PIL Image."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import pymupdf as fitz
        except ImportError as exc:
            raise RuntimeError(
                "Reading a .pdf plan requires pymupdf: pip install pymupdf"
            ) from exc
        doc = fitz.open(str(path))
        page = doc[0]
        zoom = 3.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.frombytes(
            "RGB" if pix.n < 4 else "RGBA", [pix.width, pix.height], pix.samples
        ).convert("RGB")
        doc.close()
        return img

    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def downsample(img: Image.Image, max_dim: int) -> Image.Image:
    width, height = img.size
    scale = min(1.0, max_dim / max(width, height))
    if scale < 1.0:
        img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)
    return img


def load_downsampled(path: Path, max_dim: int) -> Image.Image:
    return downsample(load_full(path), max_dim)
