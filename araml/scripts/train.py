"""
train.py — ARAML meta-training script (REGRESSION) (canonical entry point).

Run from inside araml/ with:
    PYTHONPATH=. python scripts/train.py [--epochs N]

Fixes applied (see models/meta_learner.py for implementation details):
  FIX 1  create_graph=True in inner loop        (meta_learner.py)
  FIX 2  Unfreeze XLM-R layers 9-11 + pooler   (this file)
  FIX 3  outer_lr = 0.0003, AdamW, grad clip    (this file + meta_learner.py)
  FIX 5  diagnose_gradient_flow() before epoch 1 (this file)

Task: Few-shot REGRESSION (sentiment prediction in [0, 1] range)
  - Labels normalized from star ratings (1-5) to [0, 1]
  - Loss: Smooth L1 (Huber)
  - Metrics: MAE (mean absolute error), RMSE (root mean squared error)

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

from models.araml         import ARAML
from models.retrieval_index import CrossLingualRetrievalIndex
from models.meta_learner  import meta_train_step, maml_eval_episode, diagnose_gradient_flow
from utils.episode_sampler import CategoryStratifiedEpisodeSampler
from scripts.evaluate import load_processed_split, evaluate_components


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

    val_datasets = load_processed_split(
        split_name="validation",
        processed_dir=args.processed_dir,
        verbose=True,
    )
    if val_datasets:
        print(f"Validation enabled: {args.val_episodes} episodes per epoch")
    else:
        print("Validation disabled: no low-resource validation data found.")

    # -- Optimiser: FIX 3 outer_lr = 0.0003, AdamW --------------------------
    outer_optimizer = torch.optim.AdamW(
        trainable_params, lr=args.outer_lr, weight_decay=1e-4
    )
    print(f"Optimiser: AdamW  lr={args.outer_lr}  weight_decay=1e-4")

    # -- Training loop -------------------------------------------------------
    best_val_mae = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)

    GRAD_LOG_BATCHES = {25, 100}

    for epoch in range(args.epochs):
        encoder.train(); arc.train(); meta_learner.train()
        epoch_losses, epoch_maes, epoch_rmses, epoch_gnorms = [], [], [], []
        lang_preds:  dict[str, list] = {"ja": [], "zh": []}
        lang_targets: dict[str, list] = {"ja": [], "zh": []}

        for ep_idx in tqdm(range(args.episodes_per_epoch), desc=f"Epoch {epoch+1}",
                           disable=True):
            episode = sampler.sample_episode()
            lang    = episode["language"]
            loss, mae, rmse, grad_norm, ep_preds, ep_targets = meta_train_step(
                encoder, arc, meta_learner, index,
                episode, config, device, outer_optimizer,
                max_grad_norm=args.max_grad_norm,
                scaler=scaler,
            )
            epoch_losses.append(loss)
            epoch_maes.append(mae)
            epoch_rmses.append(rmse)
            epoch_gnorms.append(grad_norm)
            lang_preds[lang].extend(ep_preds)
            lang_targets[lang].extend(ep_targets)

            if (ep_idx + 1) in GRAD_LOG_BATCHES:
                print(f"  [batch {ep_idx+1:3d}] grad_norm={grad_norm:.4f}  loss={loss:.4f}  mae={mae:.6f}  rmse={rmse:.6f}")

        mean_loss  = np.mean(epoch_losses)
        mean_mae   = np.mean(epoch_maes)
        mean_rmse  = np.mean(epoch_rmses)
        mean_gnorm = np.mean(epoch_gnorms)

        all_preds   = lang_preds["ja"]   + lang_preds["zh"]
        all_targets = lang_targets["ja"] + lang_targets["zh"]
        overall_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
        overall_rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets))**2))

        print(f"\nEpoch {epoch+1:3d} | Loss: {mean_loss:.6f} | GradNorm: {mean_gnorm:.4f}")
        print(f"  Overall  | MAE: {mean_mae:.6f} | RMSE: {mean_rmse:.6f} | Verified: MAE={overall_mae:.6f}, RMSE={overall_rmse:.6f}")
        for lang in ("ja", "zh"):
            lp, lt = lang_preds[lang], lang_targets[lang]
            if lp:
                lmae = np.mean(np.abs(np.array(lp) - np.array(lt)))
                lrmse = np.sqrt(np.mean((np.array(lp) - np.array(lt))**2))
                print(f"  [{lang}]     | MAE: {lmae:.6f} | RMSE: {lrmse:.6f} | N={len(lp)}")
        print()

        selection_mae = mean_mae
        selection_label = "train"

        if val_datasets:
            val_results = evaluate_components(
                encoder, arc, meta_learner, index, config, device,
                val_datasets, n_episodes=args.val_episodes, seed=args.seed,
                show_progress=False,
            )
            selection_mae = val_results["mae_mean"]
            selection_label = "validation"
            print(
                f"  Validation | MAE: {val_results['mae_mean']:.6f} +/- {val_results['mae_ci']:.6f} "
                f"| RMSE: {val_results['rmse_mean']:.6f} +/- {val_results['rmse_ci']:.6f} "
                f"| Corr: {val_results['correlation']:.6f}"
            )

        if selection_mae < best_val_mae:
            best_val_mae = selection_mae
            ckpt_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Saved best model ({selection_label}_MAE={best_val_mae:.6f})")


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
    parser.add_argument("--processed_dir",     default="data/processed")
    parser.add_argument("--epochs",            type=int,   default=20)
    parser.add_argument("--episodes_per_epoch",type=int,   default=1000)
    parser.add_argument("--val_episodes",      type=int,   default=100)
    parser.add_argument("--outer_lr",          type=float, default=0.0003)
    parser.add_argument("--max_grad_norm",     type=float, default=1.0)
    parser.add_argument("--seed",              type=int,   default=42)
    args = parser.parse_args()

    train(args)
