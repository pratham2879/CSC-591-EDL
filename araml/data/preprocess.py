"""
preprocess.py — Prepare multilingual sentiment data for episode sampling.

Label conventions per dataset:
  imdb (en):     label 0=neg, 1=pos  — already binary
  allocine (fr): label 0=neg, 1=pos  — already binary
  germeval (de): label 0=neg, 1=pos  — already binary
  amazon_*:      stars 1-2=neg(0), 4-5=pos(1), skip 3 — needs mapping

Tier 3 fix: properly skip neutral examples instead of clamping all >1 to 1.
"""
import os
import json
from datasets import load_from_disk

LANGUAGES = ["en", "de", "fr", "es", "ja", "zh"]

# Per-dataset column mappings: (text_col, label_col, label_type)
# label_type: "binary" = already 0/1, "stars" = 1-5 needs mapping
COLUMN_MAPS = {
    "en": ("text",   "label", "binary"),
    "fr": ("review", "label", "binary"),
    "de": ("text",   "label", "binary"),
    "es": ("text",   "label", "binary"),
    "ja": ("text",   "label", "binary"),
    "zh": ("text",   "label", "binary"),
}


def get_binary_label(item, label_col, label_type):
    """Return 0, 1, or None (skip) for a dataset item."""
    raw = item.get(label_col)
    if raw is None:
        # Try stars field for amazon-style datasets
        stars = item.get("stars")
        if stars is None:
            return None
        label_type = "stars"
        raw = int(stars)

    if label_type == "binary":
        label = int(raw)
        return label if label in (0, 1) else None

    if label_type == "stars":
        raw = int(raw)
        if raw <= 2:
            return 0
        elif raw >= 4:
            return 1
        return None  # skip neutral 3-star

    return None


def get_text(item, text_col):
    """Try multiple column names for text.
    mteb/amazon_reviews_multi uses 'review_body' as the main text column.
    """
    for col in [text_col, "review_body", "text", "review", "sentence"]:
        val = item.get(col)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return None


def preprocess_amazon(raw_dir: str = "data/raw", out_dir: str = "data/processed"):
    os.makedirs(out_dir, exist_ok=True)

    for lang in LANGUAGES:
        path = os.path.join(raw_dir, f"amazon_{lang}")
        if not os.path.exists(path):
            print(f"Skipping {lang} -- not found at {path}")
            continue

        ds = load_from_disk(path)
        splits = dict(ds) if hasattr(ds, 'keys') else {"train": ds}

        # mteb/amazon_reviews_multi: text in review_body, rating in stars (1-5)
        # Detect schema from first item
        first_split = next(iter(splits.values()))
        first_item = first_split[0]
        if "stars" in first_item:
            label_type = "stars"
            label_col = "stars"
        else:
            label_type = "binary"
            label_col = "label"

        text_col, _, _ = COLUMN_MAPS.get(lang, ("text", "label", "binary"))

        records = []
        skipped = 0

        for split_name, split_data in splits.items():
            for item in split_data:
                text = get_text(item, text_col)
                if not text:
                    skipped += 1
                    continue

                label = get_binary_label(item, label_col, label_type)
                if label is None:
                    skipped += 1
                    continue

                records.append({
                    "text": text,
                    "label": label,
                    "language": lang,
                    "split": split_name
                })

        out_path = os.path.join(out_dir, f"amazon_{lang}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        pos = sum(1 for r in records if r["label"] == 1)
        neg = sum(1 for r in records if r["label"] == 0)
        print(f"[{lang}] {len(records)} records (pos={pos}, neg={neg}, skipped={skipped}) -> {out_path}")


if __name__ == "__main__":
    preprocess_amazon()
    print("\nPreprocessing complete.")
