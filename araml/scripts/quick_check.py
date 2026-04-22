"""
quick_check.py — Validate real mteb/amazon_reviews_multi data before training.

CHECK 9: Validates per-language episode pools:
  - Enough unique labels (n_way=2)
  - Neg/pos split not worse than 80/20
  - Enough examples per class for k_shot + query_size
"""
import os
import json
import sys
from collections import Counter

LANGUAGES = ["en", "de", "fr", "es", "ja", "zh"]
DATA_DIR = "data/processed"
N_WAY = 2
K_SHOT = 5
QUERY_SIZE = 15
MIN_PER_CLASS = K_SHOT + QUERY_SIZE  # 20 minimum per class


def check_pool(lang):
    path = os.path.join(DATA_DIR, f"amazon_{lang}.json")
    if not os.path.exists(path):
        print(f"  [{lang}] SKIP — file not found")
        return None

    with open(path) as f:
        data = json.load(f)

    # Handle both flat list and split-keyed dict formats
    if isinstance(data, dict):
        train = data.get("train", [])
    else:
        train = [r for r in data if r["split"] == "train"]
    if not train:
        print(f"  [{lang}] FAIL — no training records")
        return False

    label_counts = Counter(r["label"] for r in train)
    total = len(train)
    neg = label_counts.get(0, 0)
    pos = label_counts.get(1, 0)

    # Check 1: both classes present
    if len(label_counts) < N_WAY:
        print(f"  [{lang}] FAIL — only {len(label_counts)} class(es), need {N_WAY}")
        return False

    # Check 2: neg/pos balance not worse than 80/20
    minority = min(neg, pos)
    majority = max(neg, pos)
    ratio = minority / total if total > 0 else 0
    balance_ok = ratio >= 0.20
    balance_str = f"{neg}/{pos} ({100*neg//total}%/{100*pos//total}%)"

    # Check 3: enough examples per class
    enough = neg >= MIN_PER_CLASS and pos >= MIN_PER_CLASS

    status = "PASS" if (balance_ok and enough) else "FAIL"
    print(f"  [{lang}] {status} — train={total}, neg/pos={balance_str}, "
          f"min_needed={MIN_PER_CLASS} {'✓' if enough else '✗'}, "
          f"balance {'✓' if balance_ok else '✗ (>80/20)'}")
    return status == "PASS"


def main():
    print("=" * 60)
    print("CHECK 9 — Episode pool validation on real data")
    print("=" * 60)

    results = {}
    for lang in LANGUAGES:
        results[lang] = check_pool(lang)

    print("\nSummary:")
    passed = [l for l, r in results.items() if r is True]
    failed = [l for l, r in results.items() if r is False]
    skipped = [l for l, r in results.items() if r is None]

    print(f"  PASS:  {passed}")
    print(f"  FAIL:  {failed}")
    print(f"  SKIP:  {skipped}")

    if failed:
        print("\n✗ CHECK 9 FAILED — fix issues above before training.")
        sys.exit(1)
    elif len(passed) < 2:
        print("\n✗ Need at least 2 languages passing to train.")
        sys.exit(1)
    else:
        print(f"\n✓ CHECK 9 PASSED — {len(passed)} language(s) ready for training.")


if __name__ == "__main__":
    main()
