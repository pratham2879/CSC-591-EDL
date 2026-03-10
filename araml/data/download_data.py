"""
download_data.py — Download XNLI and SIB-200 datasets from Hugging Face.

XNLI:    Cross-lingual NLI, 15 languages, 3 classes (entailment / neutral / contradiction)
SIB-200: Topic classification, 200+ languages, 7 classes
"""
import os
import argparse
from datasets import load_dataset


# ── XNLI ─────────────────────────────────────────────────────────────────────
XNLI_LANGUAGES = [
    "en", "fr", "es", "de", "el", "bg", "ru",
    "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur"
]


def download_xnli(save_dir: str = "data/raw/xnli"):
    """Download XNLI for all 15 languages (single config, language is a column)."""
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "xnli_all")
    if os.path.exists(out_path):
        print("[XNLI] Already downloaded, skipping.")
        return

    print("Downloading XNLI ...")
    ds = load_dataset("xnli", "all_languages")
    ds.save_to_disk(out_path)
    total = sum(len(ds[s]) for s in ds)
    print(f"  Saved {total:,} records -> {out_path}")


# ── SIB-200 ──────────────────────────────────────────────────────────────────
SIB200_LANGUAGES = [
    "eng_Latn",   # English
    "fra_Latn",   # French
    "deu_Latn",   # German
    "spa_Latn",   # Spanish
    "arb_Arab",   # Arabic
    "hin_Deva",   # Hindi
    "swa_Latn",   # Swahili
    "urd_Arab",   # Urdu
    "tam_Taml",   # Tamil
    "tha_Thai",   # Thai
    "vie_Latn",   # Vietnamese
    "zho_Hans",   # Chinese (Simplified)
    "yor_Latn",   # Yoruba
    "amh_Ethi",   # Amharic
]


def download_sib200(save_dir: str = "data/raw/sib200"):
    """Download SIB-200 for selected languages."""
    os.makedirs(save_dir, exist_ok=True)

    for lang in SIB200_LANGUAGES:
        out_path = os.path.join(save_dir, f"sib200_{lang}")
        if os.path.exists(out_path):
            print(f"[SIB-200 / {lang}] Already downloaded, skipping.")
            continue

        print(f"Downloading SIB-200 [{lang}] ...")
        try:
            ds = load_dataset("Davlan/sib200", lang)
            ds.save_to_disk(out_path)
            total = sum(len(ds[s]) for s in ds)
            print(f"  Saved {total:,} records -> {out_path}")
        except Exception as e:
            print(f"  WARNING: Failed [{lang}]: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for ARAML")
    parser.add_argument(
        "--dataset", choices=["xnli", "sib200", "all"], default="all",
        help="Which dataset(s) to download."
    )
    args = parser.parse_args()

    if args.dataset in ("xnli", "all"):
        download_xnli()
    if args.dataset in ("sib200", "all"):
        download_sib200()

    print("\nDone.")
