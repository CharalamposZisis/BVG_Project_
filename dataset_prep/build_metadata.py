"""Build a metadata table for the scanned construction plans in plans/.

Extracts per-file image properties (dimensions, dpi, bit depth, physical
sheet size) plus filename-derived signals (status keywords like
"ungueltig"/"nicht realisiert") that matter for a title-block recognition
dataset. Writes plans_metadata.csv next to this script.
"""

import csv
import re
import warnings
from pathlib import Path

import tifffile

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLANS_DIR = PROJECT_ROOT / "plans"
OUT_CSV = Path(__file__).resolve().parent / "plans_metadata.csv"

# DIN 476 series (mm), long edge x short edge
DIN_A = {
    "A0": (1189, 841),
    "A1": (841, 594),
    "A2": (594, 420),
    "A3": (420, 297),
    "A4": (297, 210),
}

STATUS_PATTERNS = {
    "ungueltig": r"ung[uü]ltig",
    "nicht_realisiert": r"nicht realisiert",
    "ueberholt": r"[uü]berholt",
    "nie_gebaut": r"nie gebaut",
    "neu": r"(?<![a-zA-Z])neu(?![a-zA-Z])",
}


def classify_din(width_mm: float, height_mm: float, tol: float = 0.06):
    long_edge, short_edge = max(width_mm, height_mm), min(width_mm, height_mm)
    for name, (dl, ds) in DIN_A.items():
        if abs(long_edge - dl) / dl <= tol and abs(short_edge - ds) / ds <= tol:
            return name
    # Check elongated/banner formats (common for long plan strips): multiples of a DIN short edge
    for name, (dl, ds) in DIN_A.items():
        if abs(short_edge - ds) / ds <= tol and long_edge > dl:
            ratio = long_edge / dl
            return f"{name}-extended(x{ratio:.1f})"
    return "non-standard"


def detect_status_flags(filename: str):
    flags = []
    for key, pattern in STATUS_PATTERNS.items():
        if re.search(pattern, filename, flags=re.IGNORECASE):
            flags.append(key)
    return flags


def read_tiff_meta(path: Path):
    with tifffile.TiffFile(str(path)) as tf:
        page = tf.pages[0]
        height, width = page.shape[:2]
        tags = page.tags
        xres = tags["XResolution"].value if "XResolution" in tags else None
        yres = tags["YResolution"].value if "YResolution" in tags else None
        dpi_x = (xres[0] / xres[1]) if xres and xres[1] else None
        dpi_y = (yres[0] / yres[1]) if yres and yres[1] else None
        bits = page.bitspersample
        photometric = str(page.photometric).split(".")[-1]
        compression = str(page.compression).split(".")[-1]
        n_pages = len(tf.pages)
    return dict(
        width_px=width,
        height_px=height,
        dpi_x=dpi_x,
        dpi_y=dpi_y,
        bits_per_sample=bits,
        photometric=photometric,
        compression=compression,
        n_pages=n_pages,
    )


def main():
    rows = []
    for path in sorted(PLANS_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".tif", ".tiff"):
            continue

        filesize = path.stat().st_size
        row = dict(
            filename=path.name,
            ext=path.suffix.lower(),
            filesize_bytes=filesize,
            filesize_mb=round(filesize / 1e6, 2),
        )

        try:
            meta = read_tiff_meta(path)
            row.update(meta)
            if meta["dpi_x"] and meta["dpi_y"]:
                width_mm = meta["width_px"] / meta["dpi_x"] * 25.4
                height_mm = meta["height_px"] / meta["dpi_y"] * 25.4
                row["width_mm"] = round(width_mm, 1)
                row["height_mm"] = round(height_mm, 1)
                row["din_class"] = classify_din(width_mm, height_mm)
                row["aspect_ratio"] = round(
                    max(meta["width_px"], meta["height_px"])
                    / min(meta["width_px"], meta["height_px"]),
                    2,
                )
            row["read_error"] = ""
        except Exception as exc:  # noqa: BLE001
            row["read_error"] = str(exc)

        row["status_flags"] = ";".join(detect_status_flags(path.name))
        rows.append(row)

    fieldnames = [
        "filename", "ext", "filesize_bytes", "filesize_mb",
        "width_px", "height_px", "aspect_ratio",
        "dpi_x", "dpi_y", "width_mm", "height_mm", "din_class",
        "bits_per_sample", "photometric", "compression", "n_pages",
        "status_flags", "read_error",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()