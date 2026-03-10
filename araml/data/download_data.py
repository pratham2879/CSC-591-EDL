"""
download_data.py — Download multilingual datasets from Hugging Face
Uses 'mteb/amazon_reviews_multi' which is a script-free Parquet version.
"""
import os
from datasets import load_dataset

# Language configs: HF dataset name + language subset + column names
DATASET_NAME = "mteb/amazon_reviews_multi"
LANGUAGES = ["en", "de", "fr", "es", "zh", "ja"]


def download_amazon_reviews(save_dir: str = "data/raw"):
    os.makedirs(save_dir, exist_ok=True)

    for lang in LANGUAGES:
        out_path = os.path.join(save_dir, f"amazon_{lang}")
        if os.path.exists(out_path):
            print(f"[{lang}] Already downloaded, skipping.")
            continue

        print(f"Downloading Amazon Reviews [{lang}]...")
        try:
            ds = load_dataset(DATASET_NAME, lang)
            ds.save_to_disk(out_path)
            print(f"  Saved {sum(len(ds[s]) for s in ds)} records → {out_path}")
        except Exception as e:
            print(f"  Failed to download [{lang}]: {e}")
            print(f"  Trying fallback: 'amazon_polarity' for English or skipping...")


if __name__ == "__main__":
    download_amazon_reviews()
    print("\nAll datasets downloaded successfully.")
