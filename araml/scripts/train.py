"""
train.py — ARAML meta-training script (canonical entry point).

Run from inside araml/ with:
    PYTHONPATH=. python scripts/train.py [--epochs N]

Fixes applied (see models/meta_learner.py for implementation details):
  FIX 1  create_graph=True in inner loop        (meta_learner.py)
  FIX 2  Unfreeze XLM-R layers 9-11 + pooler   (this file)
  FIX 3  outer_lr = 0.0003, AdamW, grad clip    (this file + meta_learner.py)
  FIX 5  diagnose_gradient_flow() before epoch 1 (this file)

Episode sampling:
  Only LOW_RESOURCE languages (ja, zh) appear in support/query sets.
  Pools are loaded from data/lowresource_pool_{ja,zh}.json (built by
  data/preprocess.py).  HIGH_RESOURCE (en, de, es, fr) training data
  is indexed in the FAISS retrieval store and used for cross-lingual
  augmentation only.
"""
import os
import sys
import json
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from models.araml         import ARAML
from models.retrieval_index import CrossLingualRetrievalIndex
from models.meta_learner  import meta_train_step, maml_eval_episode, diagnose_gradient_flow
from utils.episode_sampler import CategoryStratifiedEpisodeSampler


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# FIX 2 — partial encoder unfreeze
# ---------------------------------------------------------------------------

def setup_encoder_freezing(encoder, unfreeze_from_layer: int = 9) -> list:
    """
    Freeze XLM-R embeddings + transformer layers 0..(unfreeze_from_layer-1).
    Unfreeze layers unfreeze_from_layer..11 and the pooler.

    Returns the list of trainable encoder parameter tensors.

    Why: a fully frozen encoder means support/query embeddings are detached
    constants.  The inner loop cannot propagate gradients back to ARC
    through the embedding path — the second-order MAML signal vanishes.
    Unfreezing the last 3 layers restores that path without destabilising
    the lower-level representations.
    """
    base = encoder.encoder          # AutoModel (XLM-R)

    for param in base.parameters():
        param.requires_grad = False

    unfrozen = []
    for i, layer in enumerate(base.encoder.layer):
        if i >= unfreeze_from_layer:
            for param in layer.parameters():
                param.requires_grad = True
            unfrozen.append(i)

    if hasattr(base, "pooler"):
        for param in base.pooler.parameters():
            param.requires_grad = True

    trainable = [p for p in encoder.parameters() if p.requires_grad]
    frozen    = sum(1 for p in base.parameters() if not p.requires_grad)
    print(f"Encoder: layers {unfrozen} + pooler unfrozen | "
          f"frozen={frozen} params  trainable={len(trainable)} param tensors")
    return trainable


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # -- Load config (for meta-learning hyperparameters) ---------------------
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    meta_cfg = config["meta_learning"]

    # -- Model ---------------------------------------------------------------
    model  = ARAML(config).to(device)
    encoder, arc, meta_learner = model.get_components()

    # FIX 2: partial encoder unfreeze
    enc_trainable   = setup_encoder_freezing(encoder, unfreeze_from_layer=9)
    trainable_params = (
        list(arc.parameters())
        + list(meta_learner.parameters())
        + enc_trainable
    )
    total = sum(p.numel() for p in trainable_params)
    print(f"Total trainable parameters: {total:,}")

    # -- Retrieval index (CrossLingualRetrievalIndex on disk) ----------------
    index = CrossLingualRetrievalIndex()
    index.load(args.index_path)
    print(f"Loaded retrieval index with {len(index)} entries.")

    # -- Episode sampler: LOW_RESOURCE pools (ja, zh) ------------------------
    # CategoryStratifiedEpisodeSampler enforces that ONLY ja/zh appear in
    # support/query sets.  High-resource languages are in the retrieval index.
    sampler = CategoryStratifiedEpisodeSampler.from_pool_files(
        pool_dir=args.pool_dir,
        n_shot=meta_cfg["k_shot"],
        n_query=meta_cfg["query_size"],
        n_class=meta_cfg["n_way"],
        seed=args.seed,
    )

    # -- Optimiser: FIX 3 outer_lr = 0.0003, AdamW --------------------------
    outer_optimizer = torch.optim.AdamW(
        trainable_params, lr=args.outer_lr, weight_decay=1e-4
    )
    print(f"Optimiser: AdamW  lr={args.outer_lr}  weight_decay=1e-4")

    # -- Training loop -------------------------------------------------------
    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    GRAD_LOG_BATCHES = {25, 100}

    for epoch in range(args.epochs):
        encoder.train(); arc.train(); meta_learner.train()
        epoch_losses, epoch_accs, epoch_gnorms = [], [], []
        lang_preds:  dict[str, list] = {"ja": [], "zh": []}
        lang_labels: dict[str, list] = {"ja": [], "zh": []}

        for ep_idx in tqdm(range(args.episodes_per_epoch), desc=f"Epoch {epoch+1}",
                           disable=True):
            episode = sampler.sample_episode()
            lang    = episode["language"]
            loss, acc, grad_norm, ep_preds, ep_labels = meta_train_step(
                encoder, arc, meta_learner, index,
                episode, config, device, outer_optimizer,
                max_grad_norm=args.max_grad_norm,
                scaler=scaler,
            )
            epoch_losses.append(loss)
            epoch_accs.append(acc)
            epoch_gnorms.append(grad_norm)
            lang_preds[lang].extend(ep_preds)
            lang_labels[lang].extend(ep_labels)

            if (ep_idx + 1) in GRAD_LOG_BATCHES:
                print(f"  [batch {ep_idx+1:3d}] grad_norm={grad_norm:.4f}  loss={loss:.4f}  acc={acc:.3f}")

        mean_loss  = np.mean(epoch_losses)
        mean_acc   = np.mean(epoch_accs)
        mean_gnorm = np.mean(epoch_gnorms)

        all_preds  = lang_preds["ja"]  + lang_preds["zh"]
        all_labels = lang_labels["ja"] + lang_labels["zh"]
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )

        print(f"\nEpoch {epoch+1:3d} | Loss: {mean_loss:.4f} | GradNorm: {mean_gnorm:.4f}")
        print(f"  Overall  | Acc: {mean_acc:.4f} | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f} | N={len(all_preds)}")
        for lang in ("ja", "zh"):
            lp, ll = lang_preds[lang], lang_labels[lang]
            if lp:
                lprec, lrec, lf1, _ = precision_recall_fscore_support(
                    ll, lp, average="macro", zero_division=0
                )
                lacc = sum(p == l for p, l in zip(lp, ll)) / len(lp)
                print(f"  [{lang}]     | Acc: {lacc:.4f} | P: {lprec:.4f} | R: {lrec:.4f} | F1: {lf1:.4f} | N={len(lp)}")
        print()

        if mean_acc > best_acc:
            best_acc = mean_acc
            ckpt_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARAML training")
    parser.add_argument("--config",            default="configs/config.yaml")
    parser.add_argument("--pool_dir",          default="data",
                        help="Directory containing lowresource_pool_{ja,zh}.json")
    parser.add_argument("--index_path",        default="results/retrieval_index",
                        help="Base path for CrossLingualRetrievalIndex files "
                             "(.faiss + _meta.npy)")
    parser.add_argument("--save_dir",          default="results/")
    parser.add_argument("--epochs",            type=int,   default=20)
    parser.add_argument("--episodes_per_epoch",type=int,   default=1000)
    parser.add_argument("--outer_lr",          type=float, default=0.0003)
    parser.add_argument("--max_grad_norm",     type=float, default=1.0)
    parser.add_argument("--seed",              type=int,   default=42)
    args = parser.parse_args()

    train(args)
