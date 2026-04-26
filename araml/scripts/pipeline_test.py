"""
pipeline_test.py — Regression-oriented pipeline validation.

This is a lightweight preflight check for the regression branch. It validates:
  1. Regression config shape
  2. Processed data / pool file availability
  3. Label format for regression data
  4. Episode sampler behavior for regression episodes
  5. Retrieval index availability
  6. Encoder / model initialization when local model assets are present

Checks that depend on optional local artifacts are reported as skips instead of
failing immediately, so the script is useful in clean or offline workspaces too.

Run from inside araml/:
    PYTHONPATH=. python scripts/pipeline_test.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS} {name}")
    else:
        msg = f"  {FAIL} {name}"
        if detail:
            msg += f": {detail}"
        print(msg)
        failures.append(name)


def skip(name, detail=""):
    msg = f"  {SKIP} {name}"
    if detail:
        msg += f": {detail}"
    print(msg)


print("\n[1] Config validation")
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

check("output_dim = 1 (regression)", config["model"].get("output_dim") == 1,
      f"got {config['model'].get('output_dim')}")
check("inner_lr > 0", config["meta_learning"]["inner_lr"] > 0,
      f"got {config['meta_learning']['inner_lr']}")
check("inner_steps >= 1", config["meta_learning"]["inner_steps"] >= 1,
      f"got {config['meta_learning']['inner_steps']}")
check("query_size >= 1", config["meta_learning"]["query_size"] >= 1,
      f"got {config['meta_learning']['query_size']}")


print("\n[2] Data file existence")
processed_path = "data/processed/amazon_en.json"
ja_pool_path = "data/lowresource_pool_ja.json"
zh_pool_path = "data/lowresource_pool_zh.json"

has_processed = os.path.exists(processed_path)
has_ja_pool = os.path.exists(ja_pool_path)
has_zh_pool = os.path.exists(zh_pool_path)

if has_processed:
    check("amazon_en.json exists", True)
else:
    skip("amazon_en.json exists", "run: python data/preprocess.py")

if has_ja_pool or has_zh_pool:
    check("at least one low-resource pool exists", True)
else:
    skip("at least one low-resource pool exists", "run: python data/preprocess.py")


print("\n[3] Regression label validation")
if has_processed:
    with open(processed_path, encoding="utf-8") as f:
        en_splits = json.load(f)

    en_train = en_splits.get("train", [])
    labels = [r["label"] for r in en_train]
    in_range = all(isinstance(l, (int, float)) and 0.0 <= float(l) <= 1.0 for l in labels)
    check("EN train labels are floats in [0,1]", in_range,
          "processed labels should be normalized regression targets")

    raw_stars = [r.get("raw_stars") for r in en_train if "raw_stars" in r]
    check("raw_stars field is present", bool(raw_stars),
          "rerun data/preprocess.py to regenerate regression artifacts")
    if raw_stars:
        check("raw_stars are in [1,5]", all(1 <= int(s) <= 5 for s in raw_stars))
else:
    skip("EN regression label validation", "processed dataset unavailable")


print("\n[4] Episode sampler")
sampler = None
if has_ja_pool or has_zh_pool:
    from utils.episode_sampler import CategoryStratifiedEpisodeSampler

    languages = tuple(
        lang for lang, exists in (("ja", has_ja_pool), ("zh", has_zh_pool)) if exists
    )
    try:
        sampler = CategoryStratifiedEpisodeSampler.from_pool_files(
            pool_dir="data",
            languages=languages,
            n_shot=config["meta_learning"]["k_shot"],
            n_query=config["meta_learning"]["query_size"],
            n_class=config["meta_learning"]["n_way"],
        )
        ep = sampler.sample_episode()
        check("Support size = k_shot",
              len(ep["support_texts"]) == config["meta_learning"]["k_shot"],
              f"got {len(ep['support_texts'])}")
        check("Query size = query_size",
              len(ep["query_texts"]) == config["meta_learning"]["query_size"],
              f"got {len(ep['query_texts'])}")
        check("Support labels are in [0,1]",
              all(0.0 <= float(v) <= 1.0 for v in ep["support_labels"]))
        check("Query labels are in [0,1]",
              all(0.0 <= float(v) <= 1.0 for v in ep["query_labels"]))
        check("Episode language is low-resource",
              ep["language"] in languages,
              f"got {ep['language']}")
        check("Episode category is non-empty", bool(ep["category"]))
    except Exception as exc:
        check("EpisodeSampler construction", False, str(exc))
else:
    skip("Episode sampler regression checks", "low-resource pools unavailable")


print("\n[5] Retrieval index")
from models.retrieval_index import CrossLingualRetrievalIndex

index_path = "results/retrieval_index.faiss"
has_index = os.path.exists(index_path)
if has_index:
    index = CrossLingualRetrievalIndex(
        embedding_dim=config["model"]["hidden_dim"],
        similarity=config["retrieval"]["similarity"],
    )
    index.load("results/retrieval_index")
    check("Index loaded (non-empty)", len(index) > 0, f"size={len(index)}")

    dummy_query = np.random.randn(1, config["model"]["hidden_dim"]).astype(np.float32)
    retrieved = index.retrieve(dummy_query, k=5)
    check("Retrieval returns 5 results", len(retrieved["texts"]) == 5)
else:
    skip("Retrieval index checks", "run: python scripts/build_index.py")


print("\n[6] Encoder / model init")
from models.araml import ARAML
from models.encoder import TextEncoder

device = torch.device("cpu")
encoder_ready = False
try:
    encoder = TextEncoder(model_name=config["model"]["encoder"]).to(device)
    encoder.eval()
    encoder_ready = True
    check("TextEncoder created", True)
except Exception as exc:
    skip("TextEncoder created", str(exc))

try:
    model = ARAML(config).to(device)
    check("ARAML model created", True)
    check("Meta-learner output dim = 1",
          model.meta_learner.regressor.out_features == 1,
          f"got {model.meta_learner.regressor.out_features}")
except Exception as exc:
    if encoder_ready:
        check("ARAML model created", False, str(exc))
    else:
        skip("ARAML model created", str(exc))


print("\n" + "=" * 50)
if failures:
    print(f"FAILED: {len(failures)} check(s):")
    for name in failures:
        print(f"  - {name}")
    sys.exit(1)
else:
    print("All required regression pipeline checks passed.")
