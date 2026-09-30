"""Exploratory data analysis over plans_metadata.csv.

Prints summary stats relevant to preparing a title-block recognition
dataset: format/bit-depth mix, DPI spread, sheet-size (DIN) distribution,
aspect ratios, file-size spread, and filename status-flag counts.
"""

from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parent / "plans_metadata.csv"


def section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    df = pd.read_csv(CSV_PATH)

    section("Overview")
    print(f"Total files: {len(df)}")
    print(f"Read errors: {df['read_error'].notna().sum() - (df['read_error'] == '').sum() * -1 if False else (df['read_error'].fillna('') != '').sum()}")

    section("File size (MB)")
    print(df["filesize_mb"].describe().round(2))

    section("Bit depth (bits_per_sample)")
    print(df["bits_per_sample"].value_counts(dropna=False))

    section("Photometric interpretation")
    print(df["photometric"].value_counts(dropna=False))

    section("Compression")
    print(df["compression"].value_counts(dropna=False))

    section("DPI (x)")
    print(df["dpi_x"].value_counts(dropna=False).sort_index())

    section("DIN sheet-size classification")
    print(df["din_class"].value_counts(dropna=False))

    section("Aspect ratio (long/short edge)")
    print(df["aspect_ratio"].describe().round(2))
    print("\nMost elongated 5:")
    print(
        df.sort_values("aspect_ratio", ascending=False)
        .head(5)[["filename", "width_px", "height_px", "aspect_ratio"]]
        .to_string(index=False)
    )

    section("Pixel dimensions extremes")
    df["megapixels"] = (df["width_px"] * df["height_px"]) / 1e6
    print(df[["filename", "megapixels"]].sort_values("megapixels", ascending=False).head(5).to_string(index=False))
    print("...")
    print(df[["filename", "megapixels"]].sort_values("megapixels").head(5).to_string(index=False))

    section("Filename status-flags (revision/validity markers)")
    flags = df["status_flags"].fillna("").str.split(";").explode()
    flags = flags[flags != ""]
    print(flags.value_counts())
    print(f"\nFiles with no status flag: {(df['status_flags'].fillna('') == '').sum()} / {len(df)}")

    section("Multi-page TIFFs")
    print(df[df["n_pages"] > 1][["filename", "n_pages"]].to_string(index=False) or "None")


if __name__ == "__main__":
    main()