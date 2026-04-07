"""
download_data.py — Download mteb/amazon_reviews_multi from Hugging Face.

Dataset: mteb/amazon_reviews_multi
  - Pure Parquet format, works with datasets v4.x (no script issues)
  - Same task/domain/schema across 6 languages — ideal for cross-lingual transfer
  - Columns: review_body (text), stars (1-5), language, review_title, product_category
  - Size: ~62 MB per language (373 MB total compressed)
  - Per language: 200k train / 5k val / 5k test rows

Label mapping (applied in preprocess.py):
  stars 1-2  → label 0 (negative)   [i.e. 0-indexed labels 0,1]
  stars 3    → DROP (neutral)        [i.e. 0-indexed label 2]
  stars 4-5  → label 1 (positive)   [i.e. 0-indexed labels 3,4]

Language tiers:
  HIGH_RESOURCE (en, de, es, fr): Full train split downloaded → FAISS retrieval index
  LOW_RESOURCE  (ja, zh):         Full download, but training pool capped at 500 in preprocess.py

Download time estimate (without HF token, unauthenticated):
  ~62 MB/language x 6 = ~373 MB compressed
  At ~2-5 MB/s (unauthenticated HF rate): ~75-180 seconds per language
  Total: ~8-18 minutes for all 6 languages
  With HF_TOKEN (authenticated):  ~3-5 minutes total

NOTE: If you see "Dataset scripts are no longer supported" error, delete the
stale cache entry and retry:
  Windows: rmdir /s /q %USERPROFILE%\.cache\huggingface\hub\datasets--mteb--amazon_reviews_multi
  Linux:   rm -rf ~/.cache/huggingface/hub/datasets--mteb--amazon_reviews_multi
"""
import os
from datasets import load_dataset, DatasetDict

DATASET_NAME = "mteb/amazon_reviews_multi"

# Language tiers — determines retrieval vs. episode roles (see preprocess.py)
HIGH_RESOURCE = ["en", "de", "es", "fr"]
LOW_RESOURCE = ["ja", "zh"]
LANGUAGES = HIGH_RESOURCE + LOW_RESOURCE


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
        print(f"Downloading {DATASET_NAME} [{lang}] ({tier}) — full dataset, no cap...")
        try:
            ds = load_dataset(DATASET_NAME, lang)
            DatasetDict(dict(ds)).save_to_disk(out_path)
            total = sum(len(ds[s]) for s in ds)
            split_info = {s: len(ds[s]) for s in ds}
            print(f"  [{lang}] Saved {total} records {split_info} -> {out_path}")

        except RuntimeError as e:
            if "scripts are no longer supported" in str(e):
                print(f"  [{lang}] Stale cache detected. Delete cache and retry:")
                print(f"    Windows: rmdir /s /q %USERPROFILE%\\.cache\\huggingface\\hub\\datasets--mteb--amazon_reviews_multi")
                print(f"    Linux:   rm -rf ~/.cache/huggingface/hub/datasets--mteb--amazon_reviews_multi")
            else:
                print(f"  [{lang}] Failed: {e}")
        except Exception as e:
            print(f"  [{lang}] Failed: {e}")


if __name__ == "__main__":
    download_amazon_reviews()
    print("\nDownload complete.")
