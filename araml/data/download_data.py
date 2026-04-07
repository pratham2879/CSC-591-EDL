"""
download_data.py — Download mteb/amazon_reviews_multi from Hugging Face.

Dataset: mteb/amazon_reviews_multi
  - Pure Parquet format, works with datasets v4.x (no script issues)
  - Same task/domain/schema across 6 languages — ideal for cross-lingual transfer
  - Columns: review_body (text), stars (1-5), language, review_title, product_category
  - Size: ~62 MB per language (373 MB total compressed)
  - Per language: 200k train / 5k val / 5k test rows

Label mapping (applied in preprocess.py):
  stars 1-2  -> label 0 (negative)
  stars 3    -> skip (neutral, ambiguous)
  stars 4-5  -> label 1 (positive)

Source languages (high-resource, used for retrieval index + meta-training):
  en, de, fr, es

Target languages (low-resource, evaluation only):
  ja, zh

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
LANGUAGES = ["en", "de", "fr", "es", "ja", "zh"]

# How many examples to keep per split (balanced across train/test)
MAX_PER_SPLIT = 5000


def download_amazon_reviews(save_dir: str = "data/raw"):
    os.makedirs(save_dir, exist_ok=True)

    for lang in LANGUAGES:
        out_path = os.path.join(save_dir, f"amazon_{lang}")
        if os.path.exists(out_path):
            print(f"[{lang}] Already downloaded, skipping.")
            continue

        print(f"Downloading {DATASET_NAME} [{lang}]...")
        try:
            ds = load_dataset(DATASET_NAME, lang)

            # Sample up to MAX_PER_SPLIT rows per split to keep disk usage reasonable
            sampled = {}
            for split_name in ds.keys():
                n = min(MAX_PER_SPLIT, len(ds[split_name]))
                sampled[split_name] = ds[split_name].select(range(n))

            DatasetDict(sampled).save_to_disk(out_path)
            total = sum(len(sampled[s]) for s in sampled)
            print(f"  [{lang}] Saved {total} records -> {out_path}")

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
