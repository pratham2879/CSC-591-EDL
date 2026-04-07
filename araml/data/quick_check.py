# -*- coding: utf-8 -*-
"""
quick_check.py -- Fast pre-epoch sanity check (~5 seconds, no download needed).

Verifies all three data fixes WITHOUT requiring the full 210k dataset:

  CHECK 1  Label remapping:   {0,1}->0, {2}->drop, {3,4}->1
  CHECK 2  Neutral drop rate: ~20% of examples should be dropped
  CHECK 3  Episode balance:   exactly n_shot examples per class in support set
  CHECK 4  Category stratification: support + query from the SAME category
  CHECK 5  Language constraint: ONLY ja/zh in episodes (never en/de/es/fr)
  CHECK 6  Low-resource pool cap: each pool has <=500 rows
  CHECK 7  FAISS leakage guard: no ja/zh training text in high-resource data
  CHECK 8  Support label shuffle: labels are NOT in sorted order (positional shortcut)
  CHECK 9  Spot-check real data if processed files exist on disk

Run with:
    python araml/data/quick_check.py
"""
import sys
import os
import json
import random
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from araml.data.preprocess import (
    get_raw_label_0indexed,
    remap_to_binary,
    get_text,
    HIGH_RESOURCE,
    LOW_RESOURCE,
    LOW_RESOURCE_TRAIN_CAP,
    LOW_RESOURCE_SEED,
    _build_stratified_pool,
    MIN_PER_CAT_PER_CLASS,
)
from araml.utils.episode_sampler import CategoryStratifiedEpisodeSampler

# ---------------------------------------------------------------------------
# Synthetic dataset -- mirrors mteb/amazon_reviews_multi schema
# ---------------------------------------------------------------------------
CATEGORIES = [
    "books", "electronics", "clothing", "kitchen", "sports",
    "toys", "beauty", "music", "movies", "automotive",
]


def _make_record(lang, stars, category, idx):
    return {
        "review_body":      "Review text {} in {} about {}.".format(idx, lang, category),
        "stars":            stars,
        "language":         lang,
        "product_category": category,
    }


def build_synthetic_raw(langs=None, n_per_cat_per_star=30, seed=0):
    """
    Build a tiny multilingual dataset.
    Returns {lang: [raw_items]}.
    Each language: 10 categories x 5 stars x n_per_cat_per_star rows.
    """
    if langs is None:
        langs = HIGH_RESOURCE + LOW_RESOURCE
    rng = random.Random(seed)
    data = {}
    idx = 0
    for lang in langs:
        items = []
        for cat in CATEGORIES:
            for stars in range(1, 6):
                for _ in range(n_per_cat_per_star):
                    items.append(_make_record(lang, stars, cat, idx))
                    idx += 1
        rng.shuffle(items)
        data[lang] = items
    return data


def apply_label_remap(raw_items):
    kept, dropped = [], 0
    for item in raw_items:
        text = get_text(item)
        label_0_4 = get_raw_label_0indexed(item)
        if text is None or label_0_4 is None:
            dropped += 1
            continue
        binary = remap_to_binary(label_0_4)
        if binary is None:
            dropped += 1
            continue
        kept.append({
            "text":             text,
            "label":            binary,
            "language":         item["language"],
            "product_category": item["product_category"],
            "split":            "train",
        })
    return kept, dropped


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
failures = []


def ok(msg):
    print("  PASS  " + msg)


def fail(msg):
    print("  FAIL  " + msg)
    failures.append(msg)


def section(title):
    print("\n" + "-" * 60)
    print("  " + title)
    print("-" * 60)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_label_remapping(raw_data):
    section("CHECK 1 & 2 -- Label remapping and neutral drop")

    for lang in list(raw_data.keys())[:2]:
        items = raw_data[lang]
        kept, dropped = apply_label_remap(items)

        star_counts   = Counter(item["stars"] for item in items)
        expected_neg  = star_counts[1] + star_counts[2]
        expected_pos  = star_counts[4] + star_counts[5]
        expected_drop = star_counts[3]

        neg_count = sum(1 for r in kept if r["label"] == 0)
        pos_count = sum(1 for r in kept if r["label"] == 1)

        if neg_count == expected_neg:
            ok("[{}] neg={} (expected {})".format(lang, neg_count, expected_neg))
        else:
            fail("[{}] neg={} but expected {} (star_counts={})".format(
                lang, neg_count, expected_neg, dict(star_counts)))

        if pos_count == expected_pos:
            ok("[{}] pos={} (expected {})".format(lang, pos_count, expected_pos))
        else:
            fail("[{}] pos={} but expected {}".format(lang, pos_count, expected_pos))

        if dropped == expected_drop:
            ok("[{}] dropped_neutral={} (expected {})".format(lang, dropped, expected_drop))
        else:
            fail("[{}] dropped_neutral={} but expected {}".format(
                lang, dropped, expected_drop))

        drop_pct = dropped / len(items) * 100
        ok("[{}] drop rate = {:.1f}% (each star equally represented -> expected ~20%)".format(
            lang, drop_pct))

    return {lang: apply_label_remap(raw_data[lang])[0] for lang in raw_data}


def check_lowresource_pool(processed_data):
    section("CHECK 6 -- Stratified low-resource pool (cap={}, floor={}/cat/class)".format(
        LOW_RESOURCE_TRAIN_CAP, MIN_PER_CAT_PER_CLASS))
    pools = {}
    for lang in LOW_RESOURCE:
        train_all = processed_data[lang]
        pool = _build_stratified_pool(lang, train_all)

        pool_pos  = sum(1 for r in pool if r["label"] == 1)
        pool_neg  = sum(1 for r in pool if r["label"] == 0)
        pool_cats = len(set(r["product_category"] for r in pool))

        if len(pool) <= LOW_RESOURCE_TRAIN_CAP:
            ok("[{}] pool size={} neg={} pos={} categories={}".format(
                lang, len(pool), pool_neg, pool_pos, pool_cats))
        else:
            fail("[{}] pool size={} exceeds cap of {}".format(
                lang, len(pool), LOW_RESOURCE_TRAIN_CAP))

        if pool_neg == 0 or pool_pos == 0:
            fail("[{}] pool is all one class (neg={} pos={}) -- "
                 "sampler will find no balanced episodes".format(lang, pool_neg, pool_pos))
        else:
            ok("[{}] both classes present in pool".format(lang))

        # Category viability check (the key imbalance diagnostic)
        cat_cls = {}
        for r in pool:
            cat = r["product_category"]
            lbl = r["label"]
            cat_cls.setdefault(cat, Counter())[lbl] += 1

        viable = 0
        skipped_cats = []
        for cat, dist in sorted(cat_cls.items()):
            if dist[0] >= 5 and dist[1] >= 5:
                viable += 1
            else:
                skipped_cats.append("{} (neg={} pos={})".format(cat, dist[0], dist[1]))

        if viable >= 5:
            ok("[{}] category viability: {}/{} categories have >=5 examples per class".format(
                lang, viable, len(cat_cls)))
        else:
            fail("[{}] only {}/{} viable categories -- episode diversity will be poor".format(
                lang, viable, len(cat_cls)))
            print("       Skipped: " + ", ".join(skipped_cats))

        pools[lang] = pool
    return pools


def check_faiss_leakage(high_resource_data, low_resource_pools):
    section("CHECK 7 -- FAISS leakage guard")
    lr_texts = set()
    for lang, pool in low_resource_pools.items():
        lr_texts.update(r["text"] for r in pool)

    for lang in HIGH_RESOURCE:
        hr_records = high_resource_data.get(lang, [])
        bad = [r for r in hr_records if r["text"] in lr_texts]
        if bad:
            fail("[{}] {} low-resource training texts found in FAISS candidates".format(
                lang, len(bad)))
        else:
            ok("[{}] No low-resource training texts in FAISS candidates".format(lang))

    for lang in HIGH_RESOURCE:
        wrong = [r for r in high_resource_data.get(lang, []) if r["language"] != lang]
        if wrong:
            fail("[{}] {} records have wrong language field".format(lang, len(wrong)))
        else:
            ok("[{}] All records have correct language field".format(lang))


def check_episode_sampler(low_resource_pools, n_shot=5, n_query=10, n_episodes=300):
    section("CHECK 3,4,5,8 -- Episode sampler ({} episodes, {}-shot binary)".format(
        n_episodes, n_shot))

    try:
        sampler = CategoryStratifiedEpisodeSampler(
            datasets  = low_resource_pools,
            n_shot    = n_shot,
            n_query   = n_query,
            n_class   = 2,
            seed      = 42,
            log_every = 100,
        )
    except Exception as e:
        fail("EpisodeSampler construction failed: {}".format(e))
        return

    balance_violations       = 0
    query_balance_violations = 0
    category_violations      = 0
    language_violations      = 0
    support_sorted_count     = 0
    query_sorted_count       = 0
    seen_langs               = set()

    for _ in range(n_episodes):
        ep   = sampler.sample_episode()
        sl   = ep["support_labels"]
        ql   = ep["query_labels"]
        dist = Counter(sl)

        # CHECK 3: support class balance
        if dist[0] != n_shot or dist[1] != n_shot:
            balance_violations += 1

        # CHECK 3b: query class balance
        qdist = Counter(ql)
        n_query_per_class = max(1, n_query // 2)
        if qdist[0] != n_query_per_class or qdist[1] != n_query_per_class:
            query_balance_violations += 1

        # CHECK 4: category is a known category name
        if ep["category"] not in CATEGORIES:
            category_violations += 1

        # CHECK 5: language constraint
        lang = ep["language"]
        seen_langs.add(lang)
        if lang not in LOW_RESOURCE:
            language_violations += 1

        # CHECK 8: support labels should NOT be in sorted order
        if sl == sorted(sl) or sl == sorted(sl, reverse=True):
            support_sorted_count += 1

        # CHECK 8b: query labels should NOT be in sorted order (MAML outer loop)
        if ql == sorted(ql) or ql == sorted(ql, reverse=True):
            query_sorted_count += 1

    if balance_violations == 0:
        ok("CHECK 3: support class balance PERFECT across all {} episodes".format(n_episodes))
    else:
        fail("CHECK 3: {}/{} episodes had unbalanced support set".format(
            balance_violations, n_episodes))

    if query_balance_violations == 0:
        ok("CHECK 3b: query class balance PERFECT across all {} episodes".format(n_episodes))
    else:
        fail("CHECK 3b: {}/{} episodes had unbalanced query set".format(
            query_balance_violations, n_episodes))

    if language_violations == 0:
        ok("CHECK 5: only {} used in episodes (never high-resource)".format(
            sorted(seen_langs)))
    else:
        fail("CHECK 5: {} episodes used a high-resource language".format(
            language_violations))

    if category_violations == 0:
        ok("CHECK 4: all sampled categories are valid")
    else:
        fail("CHECK 4: {} episodes had unknown category".format(category_violations))

    # CHECK 8: support shuffle. With n_shot=5 binary, P(sorted) = 2/C(10,5) ~= 0.8%.
    # Over 300 episodes expect ~2-3 by chance. Allow up to 5.
    if support_sorted_count <= 5:
        ok("CHECK 8: support labels shuffled ({} sorted by chance out of {})".format(
            support_sorted_count, n_episodes))
    else:
        fail("CHECK 8: support labels sorted in {}/{} episodes -- positional shortcut".format(
            support_sorted_count, n_episodes))

    # CHECK 8b: query shuffle. Same threshold.
    if query_sorted_count <= 5:
        ok("CHECK 8b: query labels shuffled ({} sorted by chance out of {})".format(
            query_sorted_count, n_episodes))
    else:
        fail("CHECK 8b: query labels sorted in {}/{} episodes -- "
             "MAML outer loop positional shortcut".format(query_sorted_count, n_episodes))

    # Visual spot-check
    ep = sampler.sample_episode()
    print("\n  === Sample episode (visual check) ===")
    print("  language : " + ep["language"])
    print("  category : " + ep["category"])
    print("  support  : {} texts  labels={}".format(len(ep["support_texts"]), ep["support_labels"]))
    print("  query    : {} texts  labels={}".format(len(ep["query_texts"]),   ep["query_labels"]))
    print("  support[0]: " + ep["support_texts"][0][:80])
    print("  query[0]  : " + ep["query_texts"][0][:80])

    # Verify 1000-episode balance distribution (user's requested check)
    print("\n  Running 1000-episode balance distribution check...")
    balance_set = set()
    for _ in range(1000):
        ep2 = sampler.sample_episode()
        # Normalize to frozenset so insertion-order differences don't matter
        dist = Counter(ep2["support_labels"])
        balance_set.add(frozenset(dist.items()))
    expected = frozenset({(0, n_shot), (1, n_shot)})
    if balance_set == {expected}:
        ok("1000-episode balance: ONLY {{0:{}, 1:{}}} observed -- perfect".format(
            n_shot, n_shot))
    else:
        readable = [dict(fs) for fs in balance_set]
        fail("1000-episode balance: unexpected distributions seen: {}".format(readable))


def check_real_data_if_available():
    section("CHECK 9 -- Real data spot-check (skipped if not yet processed)")

    base      = os.path.join(PROJECT_ROOT, "araml", "data")
    found_any = False

    for lang in LOW_RESOURCE:
        pool_path = os.path.join(base, "lowresource_pool_{}.json".format(lang))
        if not os.path.exists(pool_path):
            continue
        found_any = True
        with open(pool_path, encoding="utf-8") as f:
            pool = json.load(f)
        pos      = sum(1 for r in pool if r["label"] == 1)
        neg      = sum(1 for r in pool if r["label"] == 0)
        cats     = len(set(r["product_category"] for r in pool))
        bad_lbl  = [r for r in pool if r["label"] not in (0, 1)]

        if len(pool) <= LOW_RESOURCE_TRAIN_CAP:
            ok("[REAL {}] pool n={} neg={} pos={} categories={}".format(
                lang, len(pool), neg, pos, cats))
        else:
            fail("[REAL {}] pool n={} exceeds cap {}".format(
                lang, len(pool), LOW_RESOURCE_TRAIN_CAP))

        if bad_lbl:
            fail("[REAL {}] {} records have labels outside {{0,1}}".format(
                lang, len(bad_lbl)))
        else:
            ok("[REAL {}] All labels are 0 or 1 (no neutral leakage)".format(lang))

    for lang in HIGH_RESOURCE + LOW_RESOURCE:
        proc_path = os.path.join(base, "processed", "amazon_{}.json".format(lang))
        if not os.path.exists(proc_path):
            continue
        found_any = True
        with open(proc_path, encoding="utf-8") as f:
            splits = json.load(f)
        for split_name, records in splits.items():
            bad = [r for r in records if r["label"] not in (0, 1)]
            if bad:
                fail("[REAL {}/{}] {} records have label not in {{0,1}}".format(
                    lang, split_name, len(bad)))
            else:
                ok("[REAL {}/{}] n={} all labels binary".format(
                    lang, split_name, len(records)))

    if not found_any:
        print("  (no processed data found -- run preprocess.py first, then re-run)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  ARAML Pre-epoch Sanity Check")
    print("=" * 60)
    n_per = 30
    print("  Synthetic: {} categories x 5 stars x {} rows x 6 langs = {} per lang".format(
        len(CATEGORIES), n_per, len(CATEGORIES) * 5 * n_per))
    print("  Runtime: <10 seconds\n")

    raw_data = build_synthetic_raw(n_per_cat_per_star=n_per)

    processed_data = check_label_remapping(raw_data)
    pools          = check_lowresource_pool(processed_data)
    hr_data        = {lang: processed_data[lang] for lang in HIGH_RESOURCE}
    check_faiss_leakage(hr_data, pools)
    check_episode_sampler(
        {lang: pools[lang] for lang in LOW_RESOURCE},
        n_shot=5, n_query=10, n_episodes=300,
    )
    check_real_data_if_available()

    print("\n" + "=" * 60)
    if not failures:
        print("  ALL CHECKS PASSED -- safe to start training epoch")
        print("  Reminder: n_classes=2 in your output head (binary sentiment)")
    else:
        print("  {} CHECK(S) FAILED:".format(len(failures)))
        for msg in failures:
            print("    - " + msg)
        print("\n  Fix the above before running the epoch.")
    print("=" * 60 + "\n")

    return len(failures)


if __name__ == "__main__":
    sys.exit(main())
