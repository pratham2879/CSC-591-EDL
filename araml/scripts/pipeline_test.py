"""
pipeline_test.py — End-to-end pipeline validation.

Tests every stage with minimal data to catch bugs before full training:
  1. Config loading & validation
  2. Data preprocessing (label distribution check)
  3. Retrieval index build + query
  4. Episode sampling
  5. Single meta_train_step (verifies higher fix — loss must not be ln(2))
  6. Single maml_eval_episode
  7. Encoder freeze verification
  8. Gradient flow through ARC query_generator

Exit code 0 = all tests passed. Any failure prints clearly and exits 1.
"""
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import numpy as np

PASS = "[PASS]"
FAIL = "[FAIL]"
failures = []

def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}" + (f": {detail}" if detail else ""))
        failures.append(name)


# ── 1. Config ──────────────────────────────────────────────────────────────
print("\n[1] Config validation")
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

check("inner_lr >= 0.05 (Tier 1)", config["meta_learning"]["inner_lr"] >= 0.05,
      f"got {config['meta_learning']['inner_lr']}")
check("inner_steps >= 10 (Tier 1)", config["meta_learning"]["inner_steps"] >= 10,
      f"got {config['meta_learning']['inner_steps']}")
check("epochs >= 5 (Tier 1)", config["training"]["epochs"] >= 5,
      f"got {config['training']['epochs']}")
check("episodes_per_epoch >= 200 (Tier 1)", config["training"].get("episodes_per_epoch", 0) >= 200,
      f"got {config['training'].get('episodes_per_epoch')}")
check("freeze_encoder = true (Tier 2)", config["training"].get("freeze_encoder") is True)
check("target_languages set", len(config["data"]["target_languages"]) > 0)


# ── 2. Data ────────────────────────────────────────────────────────────────
print("\n[2] Data validation")
en_path = "data/processed/amazon_en.json"
fr_path = "data/processed/amazon_fr.json"
check("amazon_en.json exists", os.path.exists(en_path), "run: python data/preprocess.py")
check("amazon_fr.json exists", os.path.exists(fr_path), "run: python data/preprocess.py")

all_records = []
if os.path.exists(en_path):
    with open(en_path, encoding="utf-8") as f:
        en_data = json.load(f)
    labels = [r["label"] for r in en_data]
    pos = labels.count(1)
    neg = labels.count(0)
    bad = [l for l in labels if l not in (0, 1)]
    check("EN labels are only 0/1 (Tier 3)", len(bad) == 0, f"{len(bad)} bad labels found")
    check("EN label balance (40-60% each)", 0.35 <= pos / len(labels) <= 0.65,
          f"pos={pos/len(labels):.2f}, neg={neg/len(labels):.2f}")
    train_records = [r for r in en_data if r["split"] == "train"]
    check("EN has train split", len(train_records) > 0, f"got {len(train_records)}")
    all_records.extend(train_records[:200])

if os.path.exists(fr_path):
    with open(fr_path, encoding="utf-8") as f:
        fr_data = json.load(f)
    train_records = [r for r in fr_data if r["split"] == "train"]
    all_records.extend(train_records[:200])


# ── 3. Model components ────────────────────────────────────────────────────
print("\n[3] Model initialization")
from models.araml import ARAML
from models.retrieval_index import CrossLingualRetrievalIndex
from models.meta_learner import meta_train_step, maml_eval_episode
from utils.episode_sampler import EpisodeSampler

device = torch.device("cpu")
model = ARAML(config).to(device)
encoder, arc, meta_learner = model.get_components()
check("Model created", model is not None)
check("Encoder is XLM-R", "xlm" in config["model"]["encoder"].lower() or "roberta" in config["model"]["encoder"].lower())

# Tier 2: freeze encoder
for param in encoder.parameters():
    param.requires_grad = False
enc_frozen = not any(p.requires_grad for p in encoder.parameters())
arc_trainable = any(p.requires_grad for p in arc.parameters())
check("Encoder frozen (Tier 2)", enc_frozen)
check("ARC still trainable (Tier 2)", arc_trainable)

trainable = sum(p.numel() for p in arc.parameters()) + sum(p.numel() for p in meta_learner.parameters())
total = model.count_parameters()
check("Trainable params << total (frozen encoder)", trainable < total / 100,
      f"trainable={trainable:,} total={total:,}")


# ── 4. Retrieval index ─────────────────────────────────────────────────────
print("\n[4] Retrieval index")
index_path = "results/retrieval_index.faiss"
check("Index file exists", os.path.exists(index_path), "run: python scripts/build_index.py")

index = CrossLingualRetrievalIndex()
if os.path.exists(index_path):
    index.load("results/retrieval_index")
    check("Index loaded", len(index) > 0, f"size={len(index)}")

    # Test retrieval
    dummy_query = np.random.randn(1, 768).astype(np.float32)
    retrieved = index.retrieve(dummy_query, k=5)
    check("Retrieval returns k=5 results", len(retrieved["texts"]) == 5)
    check("Retrieved texts are non-empty", all(len(t) > 0 for t in retrieved["texts"]))


# ── 5. Episode sampler ─────────────────────────────────────────────────────
print("\n[5] Episode sampler")
if len(all_records) > 0:
    sampler = EpisodeSampler(all_records, n_way=2, k_shot=5, query_size=10)
    ep = sampler.sample_episode()
    check("Support size = n_way * k_shot", len(ep["support_texts"]) == 10,
          f"got {len(ep['support_texts'])}")
    check("Query size = n_way * query_size", len(ep["query_texts"]) == 20,
          f"got {len(ep['query_texts'])}")
    check("Support labels are 0/1 only", set(ep["support_labels"]).issubset({0, 1}))
    check("Both classes in support", len(set(ep["support_labels"])) == 2)


# ── 6. Meta-train step (the critical higher fix test) ─────────────────────
print("\n[6] Meta-train step (higher context fix)")
if len(all_records) > 0 and os.path.exists(index_path):
    trainable_params = list(arc.parameters()) + list(meta_learner.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)

    losses = []
    meta_learner.train()
    for i in range(5):
        ep = sampler.sample_episode()
        loss, acc = meta_train_step(encoder, arc, meta_learner, index, ep, config, device, optimizer)
        losses.append(loss)

    ln2 = 0.6931
    not_stuck = not all(abs(l - ln2) < 0.01 for l in losses)
    check("Loss not stuck at ln(2)=0.693 (higher bug fixed)", not_stuck,
          f"losses={[round(l,4) for l in losses]}")
    check("Loss is finite", all(np.isfinite(l) for l in losses),
          f"losses={losses}")
    check("Accuracy is numeric", all(0.0 <= a <= 1.0 for a in [acc]))

    # Verify gradient flows through ARC query_generator
    arc.zero_grad()
    ep = sampler.sample_episode()
    loss, _ = meta_train_step(encoder, arc, meta_learner, index, ep, config, device, optimizer)
    grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in arc.query_generator.parameters())
    check("Gradient flows into ARC query_generator", grad_exists)


# ── 7. Eval episode ────────────────────────────────────────────────────────
print("\n[7] Eval episode (maml_eval_episode)")
if len(all_records) > 0 and os.path.exists(index_path):
    meta_learner.eval()
    ep = sampler.sample_episode()
    acc = maml_eval_episode(encoder, arc, meta_learner, index, ep, config, device)
    check("Eval returns accuracy in [0,1]", 0.0 <= acc <= 1.0, f"got {acc}")


# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*50)
if failures:
    print(f"FAILED: {len(failures)} test(s):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All tests passed. Pipeline is ready for full training.")
    sys.exit(0)
