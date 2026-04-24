"""
preprocess.py — Prepare multilingual sentiment data for episode sampling (REGRESSION).

FIX 1 — Label normalization (regression):
  The mteb/amazon_reviews_multi dataset stores ratings as `stars` (1-5).
  We convert to normalized regression targets in range [0, 1]:
    y = (stars - 1) / 4
  This preserves the full range of star ratings and does NOT drop neutral (3-star) samples.
  Labels are stored as float32 tensors.

FIX 2 — Language tier separation:
  HIGH_RESOURCE (en, de, es, fr):
    - Full processed train split → FAISS retrieval index candidates
    - Never used as query/test language in meta-learning episodes
  LOW_RESOURCE (ja, zh):
    - Training pool hard-capped at 2000 examples (seed=42, saved to disk)
    - All meta-learning episodes constructed from ja/zh support/query sets
    - Validation and test use full splits (no cap)

Output files:
  data/processed/amazon_{lang}.json      — full processed data per language,
                                           keyed by split {train, validation, test}
  data/lowresource_pool_ja.json          — 2000-example ja training pool
  data/lowresource_pool_zh.json          — 2000-example zh training pool

Regression output head: 1 output neuron predicting continuous sentiment ∈ [0, 1].
"""
import os
import json
import random
from collections import defaultdict, Counter

from datasets import load_from_disk

# ---------------------------------------------------------------------------
# Language tiers
# ---------------------------------------------------------------------------
HIGH_RESOURCE = ["en", "de", "es", "fr"]
LOW_RESOURCE  = ["ja", "zh"]
LANGUAGES     = HIGH_RESOURCE + LOW_RESOURCE

# Low-resource training pool parameters (FIX 2)
LOW_RESOURCE_TRAIN_CAP      = 2000  # total pool size target
LOW_RESOURCE_SEED           = 42
# Stratified sampling floor: guarantee at least this many examples per category.
# For regression, we still want good representation across all star ratings
# in each category, but no longer need the class-balance constraint.
MIN_PER_CAT                 = 30    # configurable
MIN_VIABLE_EXAMPLES         = 5     # warn if a category has fewer than this

# Output paths
LOWRESOURCE_POOL_DIR = "data"  # saves to data/lowresource_pool_{lang}.json


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def get_raw_label_0indexed(item: dict) -> int | None:
    """
    Extract the 0-indexed label (0-4) from a dataset item.

    mteb/amazon_reviews_multi stores ratings as `stars` (int, 1-5).
    We convert:  stars → stars - 1  →  0-indexed label (0-4).
    If a `label` field is present instead (already 0-4), use it directly.
    Returns None if the field is missing.
    """
    if "stars" in item:
        return int(item["stars"]) - 1   # 1-5 → 0-4
    if "label" in item:
        return int(item["label"])       # already 0-4
    return None


def normalize_to_regression(label_0_4: int) -> float:
    """
    Convert 0-indexed label (0-4) to normalized regression target ∈ [0, 1].
    
    Formula: y = (stars - 1) / 4  where stars = label_0_4 + 1
      label 0 (1-star) → (0) / 4 = 0.00   (negative)
      label 1 (2-star) → (1) / 4 = 0.25   (negative)
      label 2 (3-star) → (2) / 4 = 0.50   (neutral — kept, not dropped)
      label 3 (4-star) → (3) / 4 = 0.75   (positive)
      label 4 (5-star) → (4) / 4 = 1.00   (positive)
    
    Returns float in [0, 1].
    """
    return label_0_4 / 4.0


def get_text(item: dict) -> str | None:
    """Extract review text, trying multiple column names."""
    for col in ("review_body", "text", "review", "sentence"):
        val = item.get(col)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return None


# ---------------------------------------------------------------------------
# Dataset summary printer
# ---------------------------------------------------------------------------

def print_summary(lang: str, splits: dict[str, list], stage: str) -> None:
    """
    Print per-split label and category distributions.

    `splits` is {split_name: [{"raw_label": int, "product_category": str, ...}]}.
    Called before and after label-2 drop so we can verify the remapping.
    """
    print(f"\n{'='*64}")
    print(f"  [{lang}] Dataset summary — {stage}")
    print(f"{'='*64}")
    for split_name, records in splits.items():
        label_dist = Counter(r.get("raw_label", r.get("label", "?")) for r in records)
        cat_dist   = Counter(r.get("product_category", "unknown")     for r in records)
        top_cats   = cat_dist.most_common(5)
        print(f"\n  split={split_name}  n={len(records)}")
        print(f"    label distribution : {dict(sorted(label_dist.items()))}")
        print(f"    categories (total) : {len(cat_dist)}")
        print(f"    top-5 categories   : {top_cats}")


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess_amazon(
    raw_dir: str = "data/raw",
    out_dir: str = "data/processed",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(LOWRESOURCE_POOL_DIR, exist_ok=True)

    for lang in LANGUAGES:
        raw_path = os.path.join(raw_dir, f"amazon_{lang}")
        if not os.path.exists(raw_path):
            print(f"[{lang}] Skipping — not found at {raw_path}")
            continue

        ds     = load_from_disk(raw_path)
        splits = dict(ds) if hasattr(ds, "keys") else {"train": ds}

        # ----------------------------------------------------------------
        # STEP 1: Extract raw records (preserve product_category + raw label)
        # ----------------------------------------------------------------
        raw_splits: dict[str, list] = {}
        for split_name, split_data in splits.items():
            records = []
            for item in split_data:
                item  = dict(item)
                text  = get_text(item)
                if not text:
                    continue
                raw_label = get_raw_label_0indexed(item)
                if raw_label is None:
                    continue
                records.append({
                    "text":             text,
                    "raw_label":        raw_label,
                    "product_category": item.get("product_category", "unknown"),
                })
            raw_splits[split_name] = records

        # Print BEFORE-drop summary
        print_summary(lang, raw_splits, stage="BEFORE label-2 drop")

        # ----------------------------------------------------------------
        # STEP 2: Apply label normalization (regression, keep all ratings)
        # ----------------------------------------------------------------
        processed_splits: dict[str, list] = {}
        total_processed = 0

        for split_name, records in raw_splits.items():
            processed = []
            for r in records:
                normalized_label = normalize_to_regression(r["raw_label"])
                processed.append({
                    "text":             r["text"],
                    "label":            normalized_label,  # float in [0, 1]
                    "language":         lang,
                    "product_category": r["product_category"],
                    "split":            split_name,
                    "raw_stars":        r["raw_label"] + 1,  # 1-5 for reference
                })
            processed_splits[split_name] = processed
            total_processed += len(processed)
            
            # Print label distribution (continuous)
            min_label = min(x["label"] for x in processed) if processed else 0
            max_label = max(x["label"] for x in processed) if processed else 0
            mean_label = sum(x["label"] for x in processed) / len(processed) if processed else 0
            print(f"[{lang}] {split_name:12s}: n={len(processed):6d}  "
                  f"label_range=[{min_label:.2f}, {max_label:.2f}]  mean={mean_label:.2f}")

        print(f"[{lang}] TOTAL: processed={total_processed} (all examples kept for regression)")

        # Print normalized label summary
        all_processed = [r for recs in processed_splits.values() for r in recs]
        if all_processed:
            star_dist = Counter(r["raw_stars"] for r in all_processed)
            print(f"[{lang}] Star distribution (1-5): {dict(sorted(star_dist.items()))}")

        # ----------------------------------------------------------------
        # STEP 3: Save full processed dataset (all splits, all examples)
        # ----------------------------------------------------------------
        out_path = os.path.join(out_dir, f"amazon_{lang}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(processed_splits, f, ensure_ascii=False, indent=2)
        print(f"[{lang}] Saved processed data -> {out_path}")

        # ----------------------------------------------------------------
        # STEP 4 (FIX 2): For low-resource languages, build stratified pool
        # ----------------------------------------------------------------
        if lang in LOW_RESOURCE:
            train_records = processed_splits.get("train", [])
            pool = _build_stratified_pool(lang, train_records)

            pool_path = os.path.join(LOWRESOURCE_POOL_DIR, f"lowresource_pool_{lang}.json")
            with open(pool_path, "w", encoding="utf-8") as f:
                json.dump(pool, f, ensure_ascii=False, indent=2)

            pool_pos  = sum(1 for r in pool if r["label"] == 1)
            pool_neg  = sum(1 for r in pool if r["label"] == 0)
            pool_cats = len(set(r["product_category"] for r in pool))
            print(f"[{lang}] Low-resource training pool: n={len(pool)}  "
                  f"(neg={pool_neg}, pos={pool_pos})  categories={pool_cats}  "
                  f"seed={LOW_RESOURCE_SEED} -> {pool_path}")
            _report_category_viability(lang, pool)

    # ----------------------------------------------------------------
    # STEP 5: Cross-language leakage assertions
    # ----------------------------------------------------------------
    _assert_no_faiss_leakage(out_dir)


# ---------------------------------------------------------------------------
# Stratified pool construction (Option B)
# ---------------------------------------------------------------------------

def _build_stratified_pool(lang: str, train_records: list) -> list:
    """
    Build a low-resource training pool that guarantees MIN_PER_CAT examples
    per category, then fills remaining slots randomly.

    Motivation: For regression tasks, we want good representation of each
    product category in the training pool. Random sampling alone might leave
    some categories with very few examples, making episode formation impossible.

    Algorithm:
      Phase 1 — Stratified floor:
        For each category, sample up to MIN_PER_CAT examples.
        This is the guaranteed floor regardless of natural distribution.
      Phase 2 — Random fill:
        If phase-1 total < LOW_RESOURCE_TRAIN_CAP, fill remaining slots by
        sampling uniformly from examples NOT already selected in phase 1.
      Phase 3 — Warn:
        Log any category that had fewer than MIN_VIABLE_EXAMPLES examples
        available even before the floor cap — those categories will be
        silently excluded by the episode sampler.
    """
    rng = random.Random(LOW_RESOURCE_SEED)

    # Index: cat -> [records]
    cat_records: dict[str, list] = defaultdict(list)
    for r in train_records:
        cat_records[r["product_category"]].append(r)

    selected_ids: set[int] = set()   # track by id() to avoid duplication
    pool: list = []

    # Phase 1: stratified floor
    for cat, records in cat_records.items():
        if len(records) < MIN_VIABLE_EXAMPLES:
            print(f"  WARNING [{lang}] cat='{cat}': only {len(records)} examples "
                  f"available (< MIN_VIABLE_EXAMPLES={MIN_VIABLE_EXAMPLES}) -- "
                  f"this category may be excluded from episodes")
        take = min(len(records), MIN_PER_CAT)
        sampled = rng.sample(records, take) if take > 0 else []
        for r in sampled:
            if id(r) not in selected_ids:
                selected_ids.add(id(r))
                pool.append(r)

    # Phase 2: random fill to reach LOW_RESOURCE_TRAIN_CAP
    remaining_cap = LOW_RESOURCE_TRAIN_CAP - len(pool)
    if remaining_cap > 0:
        leftover = [r for r in train_records if id(r) not in selected_ids]
        fill = rng.sample(leftover, min(remaining_cap, len(leftover)))
        pool.extend(fill)
        for r in fill:
            selected_ids.add(id(r))

    rng.shuffle(pool)   # randomise order so phase-1 examples aren't first

    return pool
            selected_ids.add(id(r))

    rng.shuffle(pool)   # randomise order so phase-1 examples aren't first

    print(f"  [{lang}] Stratified pool: phase1={len(pool) - (LOW_RESOURCE_TRAIN_CAP - remaining_cap if remaining_cap > 0 else 0)} "
          f"phase2_fill={max(0, remaining_cap - max(0, remaining_cap - len([r for r in train_records if id(r) not in selected_ids])))} "
          f"total={len(pool)}")

    return pool


def _report_category_viability(lang: str, pool: list, n_shot: int = 5) -> None:
    """
    After pool construction, report per-category example counts.
    For regression, we just want to ensure sufficient examples per category
    to form n_shot episodes (no longer need class balance).
    """
    cat_dist: dict[str, int] = defaultdict(int)
    for r in pool:
        cat_dist[r["product_category"]] += 1

    viable = 0
    print(f"\n  [{lang}] Per-category example counts (need >={n_shot} for episodes):")
    for cat in sorted(cat_dist):
        count = cat_dist[cat]
        status = "OK  " if count >= n_shot else "SKIP"
        if status == "OK  ":
            viable += 1
        print(f"    {status}  {cat}: {count} examples")
    print(f"  [{lang}] Viable categories: {viable}/{len(cat_dist)}")

    if viable < 5:
        print(f"  WARNING [{lang}] only {viable} viable categories -- "
              f"consider increasing MIN_PER_CAT or LOW_RESOURCE_TRAIN_CAP")


def _assert_no_faiss_leakage(out_dir: str) -> None:
    """
    Assertions that must hold before the FAISS index is built:

      1. All FAISS index candidates come from HIGH_RESOURCE languages only.
      2. No low-resource training pool example appears in the high-resource
         training data (prevents the model from retrieving its own support set).
    """
    print(f"\n{'='*64}")
    print("  FAISS index integrity checks")
    print(f"{'='*64}")

    # Collect low-resource training pool texts
    lr_pool_texts: set[str] = set()
    for lang in LOW_RESOURCE:
        pool_path = os.path.join(LOWRESOURCE_POOL_DIR, f"lowresource_pool_{lang}.json")
        if not os.path.exists(pool_path):
            print(f"  WARNING: low-resource pool for {lang} not found, skipping leakage check.")
            continue
        with open(pool_path, encoding="utf-8") as f:
            pool = json.load(f)
        lr_pool_texts.update(r["text"] for r in pool)
    print(f"  Low-resource pool texts collected: {len(lr_pool_texts)}")

    # Check 1: high-resource processed files contain ONLY their own language
    for lang in HIGH_RESOURCE:
        hr_path = os.path.join(out_dir, f"amazon_{lang}.json")
        if not os.path.exists(hr_path):
            continue
        with open(hr_path, encoding="utf-8") as f:
            hr_data = json.load(f)
        for split_name, records in hr_data.items():
            bad = [r for r in records if r.get("language") != lang]
            assert len(bad) == 0, (
                f"ASSERTION FAILED: {lang}/{split_name} contains "
                f"{len(bad)} records with wrong language field."
            )
    print("  ASSERTION PASSED: All FAISS index candidates are high-resource language examples.")

    # Check 2: no low-resource training text appears in high-resource training data
    leakage_count = 0
    for lang in HIGH_RESOURCE:
        hr_path = os.path.join(out_dir, f"amazon_{lang}.json")
        if not os.path.exists(hr_path):
            continue
        with open(hr_path, encoding="utf-8") as f:
            hr_data = json.load(f)
        for r in hr_data.get("train", []):
            if r["text"] in lr_pool_texts:
                leakage_count += 1

    if leakage_count > 0:
        print(f"  WARNING: {leakage_count} low-resource training examples "
              f"found in high-resource data — potential FAISS leakage!")
    else:
        print("  ASSERTION PASSED: No low-resource training examples in FAISS index candidates.")

    print(f"{'='*64}\n")


# ---------------------------------------------------------------------------
# Runtime FAISS integrity check (call from training code after index is built)
# ---------------------------------------------------------------------------

def assert_faiss_index_integrity(
    index_texts: list[str],
    high_resource_train_records: list[dict],
    low_resource_pools: dict[str, list[dict]],
) -> None:
    """
    Call this after building the FAISS index to verify:
      1. Every text in the index is from a HIGH_RESOURCE language.
      2. No low-resource training pool text is in the index (no leakage).

    Args:
        index_texts:                Ordered list of texts stored in the FAISS index.
        high_resource_train_records: All records used to build the index
                                    (must have a 'language' field).
        low_resource_pools:         {lang: [records]} for ja and zh training pools.
    """
    index_text_set = set(index_texts)

    # Verify all index records are high-resource
    hr_texts = set()
    for r in high_resource_train_records:
        assert r.get("language") in HIGH_RESOURCE, (
            f"FAISS index contains a non-high-resource example (language={r.get('language')})"
        )
        hr_texts.add(r["text"])

    non_hr = index_text_set - hr_texts
    assert len(non_hr) == 0, (
        f"FAISS index contains {len(non_hr)} texts not in high-resource training data."
    )

    # Verify no low-resource training example leaked into the index
    for lang, pool in low_resource_pools.items():
        for r in pool:
            assert r["text"] not in index_text_set, (
                f"DATA LEAKAGE: {lang} training example found in FAISS index: "
                f"'{r['text'][:80]}...'"
            )

    print("assert_faiss_index_integrity: all checks passed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    preprocess_amazon()
    print("\nPreprocessing complete.")
    print("Output head must use output_dim=1 (regression with normalized labels in [0, 1]).")
