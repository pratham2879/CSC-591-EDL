"""
download_data.py — Download mteb/amazon_reviews_multi from Hugging Face.

Dataset: mteb/amazon_reviews_multi
  - JSONL format per language, one file per split (train/test/validation)
  - Same task/domain/schema across 6 languages — ideal for cross-lingual transfer
  - Columns: review_body (text), stars (1-5), language, review_title, product_category
  - Size: ~62 MB per language (373 MB total compressed)
  - Per language: 200k train / 5k val / 5k test rows

Label mapping (applied in preprocess.py):
  stars 1-2  -> label 0 (negative)
  stars 3    -> DROP (neutral)
  stars 4-5  -> label 1 (positive)

Language tiers:
  HIGH_RESOURCE (en, de, es, fr): Full train split downloaded -> FAISS retrieval index
  LOW_RESOURCE  (ja, zh):         Full download, but training pool capped at 500 in preprocess.py

NOTE on datasets v4.x:
  mteb/amazon_reviews_multi contains an old loading script that datasets v4+ refuses
  to run.  We bypass it by loading the JSONL files directly via the 'json' loader,
  which avoids the RuntimeError entirely.
"""
import os
from datasets import load_dataset, DatasetDict

DATASET_NAME = "mteb/amazon_reviews_multi"
HF_BASE      = f"hf://datasets/{DATASET_NAME}"

# Language tiers — determines retrieval vs. episode roles (see preprocess.py)
HIGH_RESOURCE = ["en", "de", "es", "fr"]
LOW_RESOURCE  = ["ja", "zh"]
LANGUAGES     = HIGH_RESOURCE + LOW_RESOURCE


def _load_lang_jsonl(lang: str) -> DatasetDict:
    """
    Load a language's JSONL splits directly, bypassing the dataset loading
    script (which datasets v4+ refuses to execute).
    """
    data_files = {
        split: f"{HF_BASE}/{lang}/{split}.jsonl"
        for split in ("train", "validation", "test")
    }
    ds = load_dataset("json", data_files=data_files)
    return DatasetDict(dict(ds))


def download_amazon_reviews(save_dir: str = "data/raw"):
    """
    Download the full dataset for all languages.

    High-resource languages are downloaded WITHOUT any cap because the full
    train split (~200k per language) is required for the FAISS retrieval index.
    Low-resource languages are also downloaded in full; the 500-example
    training pool cap is applied later in preprocess.py.
    """
    os.makedirs(save_dir, exist_ok=True)

    for lang in LANGUAGES:
        out_path = os.path.join(save_dir, f"amazon_{lang}")
        if os.path.exists(out_path):
            print(f"[{lang}] Already downloaded, skipping.")
            continue

        tier = "HIGH-RESOURCE" if lang in HIGH_RESOURCE else "LOW-RESOURCE"
        print(f"Downloading {DATASET_NAME} [{lang}] ({tier}) via JSONL loader...")
        try:
            ds = _load_lang_jsonl(lang)
            ds.save_to_disk(out_path)
            total      = sum(len(ds[s]) for s in ds)
            split_info = {s: len(ds[s]) for s in ds}
            print(f"  [{lang}] Saved {total} records {split_info} -> {out_path}")
        except Exception as e:
            print(f"  [{lang}] Failed: {e}")


if __name__ == "__main__":
    download_amazon_reviews()
    print("\nDownload complete.")
