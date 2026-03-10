"""
preprocess.py — Tokenize and prepare data for episode sampling
Compatible with mteb/amazon_reviews_multi column schema.
"""
import os
import json
from datasets import load_from_disk

TEXT_COL = "text"
LABEL_COL = "label"


def preprocess_amazon(raw_dir: str = "data/raw", out_dir: str = "data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    languages = ["en", "de", "fr", "es", "zh", "ja"]

    for lang in languages:
        path = os.path.join(raw_dir, f"amazon_{lang}")
        if not os.path.exists(path):
            print(f"Skipping {lang} — not found at {path}")
            continue

        ds = load_from_disk(path)
        records = []

        for split in ds.keys():
            for item in ds[split]:
                text = item.get(TEXT_COL) or item.get("review_body", "")
                raw_label = item.get(LABEL_COL)
                if raw_label is None:
                    raw_label = item.get("stars", 1) - 1
                label = int(raw_label)
                if not text:
                    continue
                records.append({
                    "text": text,
                    "label": label,
                    "language": lang,
                    "split": split
                })

        out_path = os.path.join(out_dir, f"amazon_{lang}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[{lang}] Saved {len(records)} records → {out_path}")


if __name__ == "__main__":
    preprocess_amazon()
    print("\nPreprocessing complete.")
