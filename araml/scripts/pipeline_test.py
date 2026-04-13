"""
pipeline_test.py — End-to-end pipeline validation.

Tests every stage with minimal data to catch bugs before full training:
  1. Config loading & validation
  2. Data file existence check
  3. Model initialization + encoder partial-unfreeze (FIX 2)
  4. Retrieval index load + query
  5. Episode sampling from low-resource pools (ja/zh)
  6. Single meta_train_step — loss must move from ln(2) and ARC must get grads
  7. Single maml_eval_episode

Exit code 0 = all tests passed. Any failure prints clearly and exits 1.

Run from inside araml/:
    PYTHONPATH=. python scripts/pipeline_test.py
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
        msg = f"  {FAIL} {name}"
        if detail:
            msg += f": {detail}"
        print(msg)
        failures.append(name)


# ── 1. Config ──────────────────────────────────────────────────────────────
print("\n[1] Config validation")
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

check("inner_lr >= 0.05",     config["meta_learning"]["inner_lr"] >= 0.05,
      f"got {config['meta_learning']['inner_lr']}")
check("inner_steps >= 5",     config["meta_learning"]["inner_steps"] >= 5,
      f"got {config['meta_learning']['inner_steps']}")
check("epochs >= 5",          config["training"]["epochs"] >= 5,
      f"got {config['training']['epochs']}")
check("episodes_per_epoch >= 100",
      config["training"].get("episodes_per_epoch", 0) >= 100,
      f"got {config['training'].get('episodes_per_epoch')}")
check("num_classes = 2 (binary)",
      config["model"]["num_classes"] == 2,
      f"got {config['model']['num_classes']}")


# ── 2. Data files ──────────────────────────────────────────────────────────
print("\n[2] Data file existence")
en_path      = "data/processed/amazon_en.json"
fr_path      = "data/processed/amazon_fr.json"
ja_pool_path = "data/lowresource_pool_ja.json"
zh_pool_path = "data/lowresource_pool_zh.json"

check("amazon_en.json exists", os.path.exists(en_path),
      "run: python data/preprocess.py")
check("amazon_fr.json exists", os.path.exists(fr_path),
      "run: python data/preprocess.py")
check("lowresource_pool_ja.json exists", os.path.exists(ja_pool_path),
      "run: python data/preprocess.py (requires ja raw data)")
check("lowresource_pool_zh.json exists", os.path.exists(zh_pool_path),
      "run: python data/preprocess.py (requires zh raw data)")

if os.path.exists(en_path):
    with open(en_path, encoding="utf-8") as f:
        en_splits = json.load(f)
    # processed file is keyed by split name
    en_train = en_splits.get("train", [])
    labels = [r["label"] for r in en_train]
    bad    = [l for l in labels if l not in (0, 1)]
    check("EN labels are only 0/1", len(bad) == 0,
          f"{len(bad)} unexpected labels found")
    if labels:
        pos_ratio = sum(l == 1 for l in labels) / len(labels)
        check("EN label balance (35-65%)", 0.35 <= pos_ratio <= 0.65,
              f"pos_ratio={pos_ratio:.2f}")


# ── 3. Model + partial encoder unfreeze (FIX 2) ───────────────────────────
print("\n[3] Model initialization")
from models.araml        import ARAML
from models.retrieval_index import CrossLingualRetrievalIndex
from models.meta_learner import meta_train_step, maml_eval_episode

device = torch.device("cpu")
model  = ARAML(config).to(device)
encoder, arc, meta_learner = model.get_components()
check("Model created", model is not None)

# FIX 2: unfreeze layers 9-11 (same as scripts/train.py)
for param in encoder.parameters():
    param.requires_grad = False
base = encoder.encoder
for i, layer in enumerate(base.encoder.layer):
    if i >= 9:
        for param in layer.parameters():
            param.requires_grad = True
if hasattr(base, "pooler"):
    for param in base.pooler.parameters():
        param.requires_grad = True

enc_unfrozen  = sum(1 for p in encoder.parameters() if p.requires_grad)
arc_trainable = any(p.requires_grad for p in arc.parameters())
check("Encoder partially unfrozen (layers 9-11)", enc_unfrozen > 0,
      f"unfrozen tensors: {enc_unfrozen}")
check("ARC trainable", arc_trainable)

total_trainable = (sum(p.numel() for p in arc.parameters()) +
                   sum(p.numel() for p in meta_learner.parameters()) +
                   sum(p.numel() for p in encoder.parameters() if p.requires_grad))
check("Trainable params > 0", total_trainable > 0,
      f"trainable={total_trainable:,}")
print(f"  Trainable parameters: {total_trainable:,}")


# ── 4. Retrieval index ─────────────────────────────────────────────────────
print("\n[4] Retrieval index")
index_path = "results/retrieval_index.faiss"
index = CrossLingualRetrievalIndex()
has_index = os.path.exists(index_path)
check("Index file exists", has_index,
      "run: python scripts/build_index.py")

if has_index:
    index.load("results/retrieval_index")
    check("Index loaded (non-empty)", len(index) > 0,
          f"size={len(index)}")

    dummy_query = np.random.randn(1, 768).astype(np.float32)
    retrieved   = index.retrieve(dummy_query, k=5)
    check("Retrieval returns 5 results", len(retrieved["texts"]) == 5)
    check("Retrieved texts non-empty",
          all(len(t) > 0 for t in retrieved["texts"]))


# ── 5. Episode sampler (low-resource pools) ────────────────────────────────
print("\n[5] Episode sampler")
from utils.episode_sampler import CategoryStratifiedEpisodeSampler

pools_exist = os.path.exists(ja_pool_path) or os.path.exists(zh_pool_path)
check("At least one low-resource pool exists", pools_exist,
      "run: python data/preprocess.py")

sampler = None
if pools_exist:
    try:
        sampler = CategoryStratifiedEpisodeSampler.from_pool_files(
            pool_dir="data",
            n_shot=5,
            n_query=10,
            n_class=2,
        )
        ep = sampler.sample_episode()
        n_support_expected = 5 * 2   # n_shot * n_class
        n_query_expected   = (10 // 2) * 2  # (n_query // n_class) * n_class

        check("Support size = n_shot * n_class",
              len(ep["support_texts"]) == n_support_expected,
              f"expected {n_support_expected}, got {len(ep['support_texts'])}")
        check("Query size correct",
              len(ep["query_texts"]) == n_query_expected,
              f"expected {n_query_expected}, got {len(ep['query_texts'])}")
        check("Support labels are 0/1 only",
              set(ep["support_labels"]).issubset({0, 1}))
        check("Both classes in support",
              len(set(ep["support_labels"])) == 2)
        check("Episode language is low-resource",
              ep["language"] in ("ja", "zh"),
              f"got {ep['language']}")
    except Exception as e:
        check("EpisodeSampler construction", False, str(e))


# ── 6. Meta-train step ─────────────────────────────────────────────────────
print("\n[6] meta_train_step (gradient-flow check)")
if sampler is not None and has_index:
    trainable_params = (
        list(arc.parameters()) +
        list(meta_learner.parameters()) +
        [p for p in encoder.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4)

    # Run 5 steps — check loss moves from ln(2)
    losses, accs = [], []
    encoder.train(); arc.train(); meta_learner.train()
    for _ in range(5):
        ep   = sampler.sample_episode()
        loss, acc, grad_norm = meta_train_step(
            encoder, arc, meta_learner, index, ep, config, device, optimizer
        )
        losses.append(loss)
        accs.append(acc)

    ln2 = 0.6931
    not_stuck = not all(abs(l - ln2) < 0.005 for l in losses)
    check("Loss not stuck at ln(2)=0.6931 (create_graph fix)",
          not_stuck,
          f"losses={[round(l,4) for l in losses]}")
    check("Loss is finite",
          all(np.isfinite(l) for l in losses))
    check("Accuracy in [0,1]",
          all(0.0 <= a <= 1.0 for a in accs))

    # Gradient flow into ARC — run one more step and inspect grads
    # (meta_train_step zeros grads at start; grads remain populated after step)
    ep = sampler.sample_episode()
    loss, acc, grad_norm = meta_train_step(
        encoder, arc, meta_learner, index, ep, config, device, optimizer
    )
    arc_grad_ok = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in arc.query_generator.parameters()
    )
    check("Gradient flows into ARC query_generator", arc_grad_ok)
    check("Gradient norm > 0", grad_norm > 0,
          f"grad_norm={grad_norm:.6f}")
    print(f"  grad_norm (after clipping at 1.0): {grad_norm:.6f}")
else:
    print("  Skipped — requires pool files and retrieval index.")


# ── 7. Eval episode ────────────────────────────────────────────────────────
print("\n[7] maml_eval_episode")
if sampler is not None and has_index:
    encoder.eval(); arc.eval()
    ep  = sampler.sample_episode()
    acc = maml_eval_episode(encoder, arc, meta_learner, index, ep, config, device)
    check("Eval accuracy in [0,1]", 0.0 <= acc <= 1.0,
          f"got {acc}")
else:
    print("  Skipped — requires pool files and retrieval index.")


# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
if failures:
    print(f"FAILED: {len(failures)} test(s):")
    for name in failures:
        print(f"  - {name}")
    sys.exit(1)
else:
    print("All tests passed. Pipeline is ready for full training.")
    sys.exit(0)
