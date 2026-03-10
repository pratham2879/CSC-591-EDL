"""
preprocess.py — Preprocess XNLI and SIB-200 into uniform JSON records.

Output format per record:
    {"text": str, "label": int, "language": str, "split": str}
"""
import os
import json
import argparse
from datasets import load_from_disk

# ── Label maps ────────────────────────────────────────────────────────────────
XNLI_LABEL_MAP = {0: 0, 1: 1, 2: 2}          # entailment / neutral / contradiction
SIB200_CATEGORY_TO_ID: dict = {}               # built dynamically


# ── XNLI ─────────────────────────────────────────────────────────────────────
XNLI_LANGUAGES = [
    "en", "fr", "es", "de", "el", "bg", "ru",
    "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur"
]


def preprocess_xnli(raw_dir: str = "data/raw/xnli",
                     out_dir: str = "data/processed/xnli"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(raw_dir, "xnli_all")
    if not os.path.exists(path):
        print("XNLI raw data not found. Run download_data.py first.")
        return

    ds = load_from_disk(path)

    # XNLI stores premise/hypothesis per language inside each row
    for lang in XNLI_LANGUAGES:
        records = []
        for split_name in ds.keys():
            for item in ds[split_name]:
                premise = item["premise"].get(lang)
                hypothesis = item["hypothesis"]["translation"]
                # hypothesis is a list aligned with language list
                lang_list = item["hypothesis"]["language"]
                try:
                    lang_idx = lang_list.index(lang)
                    hyp = hypothesis[lang_idx]
                except (ValueError, IndexError):
                    continue

                if not premise or not hyp:
                    continue
                text = f"{premise} [SEP] {hyp}"
                label = int(item["label"])
                records.append({
                    "text": text,
                    "label": label,
                    "language": lang,
                    "split": split_name
                })

        out_path = os.path.join(out_dir, f"xnli_{lang}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[XNLI / {lang}] {len(records):,} records -> {out_path}")


# ── SIB-200 ──────────────────────────────────────────────────────────────────
SIB200_LANGUAGES = [
    "eng_Latn", "fra_Latn", "deu_Latn", "spa_Latn", "arb_Arab",
    "hin_Deva", "swa_Latn", "urd_Arab", "tam_Taml", "tha_Thai",
    "vie_Latn", "zho_Hans", "yor_Latn", "amh_Ethi",
]


def preprocess_sib200(raw_dir: str = "data/raw/sib200",
                      out_dir: str = "data/processed/sib200"):
    os.makedirs(out_dir, exist_ok=True)

    # Build global category -> id map from the first language we find
    global SIB200_CATEGORY_TO_ID
    if not SIB200_CATEGORY_TO_ID:
        for lang in SIB200_LANGUAGES:
            p = os.path.join(raw_dir, f"sib200_{lang}")
            if not os.path.exists(p):
                continue
            ds = load_from_disk(p)
            cats = set()
            for split in ds:
                for item in ds[split]:
                    cats.add(item["category"])
            SIB200_CATEGORY_TO_ID = {c: i for i, c in enumerate(sorted(cats))}
            break
        if not SIB200_CATEGORY_TO_ID:
            print("SIB-200 raw data not found. Run download_data.py first.")
            return

    # Save category map
    cat_map_path = os.path.join(out_dir, "category_map.json")
    with open(cat_map_path, "w") as f:
        json.dump(SIB200_CATEGORY_TO_ID, f, indent=2)

    for lang in SIB200_LANGUAGES:
        p = os.path.join(raw_dir, f"sib200_{lang}")
        if not os.path.exists(p):
            print(f"Skipping SIB-200 [{lang}] — not found.")
            continue

        ds = load_from_disk(p)
        records = []
        for split_name in ds:
            for item in ds[split_name]:
                text = item.get("text", "")
                if not text:
                    continue
                cat = item["category"]
                label = SIB200_CATEGORY_TO_ID.get(cat, -1)
                if label < 0:
                    continue
                records.append({
                    "text": text,
                    "label": label,
                    "language": lang,
                    "split": split_name
                })

        out_path = os.path.join(out_dir, f"sib200_{lang}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[SIB-200 / {lang}] {len(records):,} records -> {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["xnli", "sib200", "all"], default="all")
    args = parser.parse_args()

    if args.dataset in ("xnli", "all"):
        preprocess_xnli()
    if args.dataset in ("sib200", "all"):
        preprocess_sib200()

    print("\nPreprocessing complete.")
