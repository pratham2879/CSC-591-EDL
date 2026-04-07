"""
train.py -- ARAML meta-learning training loop.

Root-cause of loss=0.6931 (ln2, i.e. random chance):
  The MAML inner loop was breaking the computational graph, so no gradient
  ever reached the encoder or ARC.  Five targeted fixes restore gradient flow:

  FIX 1  create_graph=True in inner loop  (THE critical fix)
  FIX 2  Unfreeze XLM-R layers 9-11 + pooler
  FIX 3  outer_lr = 0.0003  (second-order grads are larger -- needs lower lr)
  FIX 4  Gradient clipping  max_norm=1.0  (second-order grad spikes)
  FIX 5  Smoke test before epoch 1 -- aborts if ARC/encoder grads are zero

Architecture recap:
  encoder     : XLM-R base -> CLS embedding (768-d)
  arc         : AdaptiveRetrievalController -- generates retrieval query,
                attention-weights retrieved high-resource embeddings
  classifier  : Linear(768, 2) -- binary sentiment head (2 classes, NOT 5)
  faiss       : FaissRetriever -- offline index of high-resource embeddings

Per-episode forward pass:
  1. Encode support + query texts  (differentiable, encoder unfrozen at layers 9-11)
  2. Compute task_emb = mean(support_embs)
  3. ARC: arc_query = query_generator(task_emb)       <- in gradient graph
           retrieved  = faiss.retrieve(arc_query.detach(), k)  <- detached search
           attn_weights = attention_scorer(task_emb, retrieved) <- in gradient graph
           weighted_ret = sum(attn * retrieved)               <- in gradient graph
  4. augment = arc_query.squeeze(0) + weighted_ret    (both ARC sub-nets in graph)
  5. support_embs_aug = support_embs + augment        (broadcast)
  6. Inner loop (FIX 1): adapt classifier on support_embs_aug with create_graph=True
  7. Outer loss: F.cross_entropy(adapted_classifier(query_embs_aug), query_labels)
  8. outer_loss.backward()  -- gradients reach encoder, ARC, classifier
"""
import os
import sys
import json
import random
import logging
import argparse
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from araml.models.encoder import TextEncoder
from araml.models.arc     import AdaptiveRetrievalController
from araml.utils.episode_sampler import CategoryStratifiedEpisodeSampler
from araml.utils.faiss_index     import FaissRetriever
from araml.utils.metrics         import evaluate_few_shot, aggregate_episode_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Paths
    pool_dir:        str = "araml/data"
    processed_dir:   str = "araml/data/processed"
    faiss_dir:       str = "araml/results/faiss"
    checkpoint_dir:  str = "araml/results"

    # Episode sampling
    n_shot:              int = 5
    n_query:             int = 10
    n_class:             int = 2
    episodes_per_epoch:  int = 100
    val_episodes:        int = 50

    # MAML (FIX 1, FIX 3)
    inner_lr:    float = 0.01
    inner_steps: int   = 5
    outer_lr:    float = 0.0003   # FIX 3: reduced from typical 1e-3 for second-order grads

    # Optimiser (FIX 4)
    max_grad_norm: float = 1.0

    # Encoder (FIX 2)
    unfreeze_from_layer: int = 9   # unfreeze layers 9, 10, 11

    # Model
    model_name:  str = "xlm-roberta-base"
    hidden_dim:  int = 768
    n_retrieve:  int = 10          # k for FAISS retrieval per episode
    arc_max_k:   int = 10

    # Training
    n_epochs:    int   = 10
    seed:        int   = 42
    device:      str   = "cuda" if torch.cuda.is_available() else "cpu"
    smoke_episodes: int = 3        # FIX 5: episodes in pre-training smoke test


# ---------------------------------------------------------------------------
# FIX 2 -- Encoder layer freezing
# ---------------------------------------------------------------------------

def setup_encoder_freezing(encoder: TextEncoder, unfreeze_from_layer: int = 9) -> list:
    """
    Freeze XLM-R embeddings and layers 0..(unfreeze_from_layer-1).
    Unfreeze layers unfreeze_from_layer..11 and the pooler.

    Returns the list of trainable parameter tensors (for the optimiser).
    """
    base = encoder.encoder   # the AutoModel (XLM-R)

    # Freeze everything first
    for param in base.parameters():
        param.requires_grad = False

    # Unfreeze last (12 - unfreeze_from_layer) transformer layers
    unfrozen_layers = []
    for i, layer in enumerate(base.encoder.layer):
        if i >= unfreeze_from_layer:
            for param in layer.parameters():
                param.requires_grad = True
            unfrozen_layers.append(i)

    # Always unfreeze pooler (used for CLS representation)
    if hasattr(base, "pooler"):
        for param in base.pooler.parameters():
            param.requires_grad = True

    frozen_count   = sum(1 for p in base.parameters() if not p.requires_grad)
    unfrozen_count = sum(1 for p in base.parameters() if p.requires_grad)
    logger.info(
        "Encoder: layers %s unfrozen + pooler | frozen=%d params, trainable=%d params",
        unfrozen_layers, frozen_count, unfrozen_count,
    )

    trainable = [p for p in encoder.parameters() if p.requires_grad]
    return trainable


# ---------------------------------------------------------------------------
# FIX 1 -- Functional MAML inner loop with create_graph=True
# ---------------------------------------------------------------------------

def functional_forward(params: dict, x: torch.Tensor) -> torch.Tensor:
    """
    Apply a Linear(in, 2) classifier functionally.
    params must have 'weight' (2, D) and 'bias' (2,).
    x : (N, D)
    Returns logits: (N, 2)
    """
    return F.linear(x, params["weight"], params["bias"])


def inner_loop(
    classifier: nn.Linear,
    support_embs: torch.Tensor,
    support_labels: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
) -> dict:
    """
    FIX 1: Functional MAML inner loop with create_graph=True.

    Without create_graph=True the gradient through the inner update is
    treated as a constant, so outer_loss.backward() cannot reach the
    encoder or ARC -- producing zero gradients everywhere and loss stuck
    at ln(2) = 0.6931.

    Args:
        classifier    : nn.Linear(hidden_dim, 2) -- provides initial fast weights
        support_embs  : (n_support, D) float tensor, requires_grad may be True
        support_labels: (n_support,) long tensor
        inner_lr      : step size for inner updates
        inner_steps   : number of gradient steps

    Returns:
        adapted_params: dict {"weight": tensor, "bias": tensor}
                        These tensors carry the full computational graph back
                        through every inner-loop gradient computation, enabling
                        the outer loop to differentiate through the inner loop.
    """
    # Clone initial weights; the clone op keeps the graph connection
    # to the original classifier parameters (outer meta-update path).
    params = {
        "weight": classifier.weight.clone(),
        "bias":   classifier.bias.clone(),
    }

    for _ in range(inner_steps):
        logits = functional_forward(params, support_embs)          # (N, 2)
        loss   = F.cross_entropy(logits, support_labels)

        grads = torch.autograd.grad(
            loss,
            list(params.values()),
            create_graph=True,   # FIX 1: retain graph for second-order gradients
            allow_unused=True,
        )

        params = {
            k: v - inner_lr * (g if g is not None else torch.zeros_like(v))
            for (k, v), g in zip(params.items(), grads)
        }

    return params


# ---------------------------------------------------------------------------
# Per-episode forward pass
# ---------------------------------------------------------------------------

def run_episode(
    encoder:    TextEncoder,
    classifier: nn.Linear,
    arc:        AdaptiveRetrievalController,
    retriever:  FaissRetriever,
    episode:    dict,
    cfg:        Config,
    device:     torch.device,
) -> tuple:
    """
    Run one meta-learning episode.

    Returns
    -------
    outer_loss : scalar tensor (with full gradient graph)
    acc        : float (query accuracy with adapted classifier)
    """
    support_texts  = episode["support_texts"]
    support_labels = torch.tensor(episode["support_labels"], dtype=torch.long,  device=device)
    query_texts    = episode["query_texts"]
    query_labels   = torch.tensor(episode["query_labels"],   dtype=torch.long,  device=device)

    # 1. Encode support and query sets (differentiable; encoder layers 9-11 unfrozen)
    support_embs = encoder.encode_text(support_texts, device)   # (n_s, 768)
    query_embs   = encoder.encode_text(query_texts,   device)   # (n_q, 768)

    # 2. Task embedding = mean of support embeddings
    task_emb = support_embs.mean(dim=0, keepdim=True)           # (1, 768)

    # 3. ARC retrieval augmentation
    #    arc_query: (1, 768) -- through query_generator, stays in graph
    #    retrieve uses .detach() -- FAISS search is non-differentiable
    #    attention weights: through attention_scorer, stays in graph
    arc_query = arc.generate_query(task_emb)                    # (1, 768)  grad: query_generator
    k_retrieve = arc.predict_budget(task_emb)                   # int, discrete -- detached
    k_retrieve = max(1, min(k_retrieve, cfg.n_retrieve))

    retrieved_embs = retriever.retrieve(
        arc_query, k=k_retrieve, device=device                  # detached inside retrieve()
    )                                                            # (K, 768), leaf tensor

    attn_weights = arc.compute_attention_weights(
        task_emb, retrieved_embs                                 # grad: attention_scorer
    )                                                            # (K,)
    weighted_ret = (attn_weights.unsqueeze(-1) * retrieved_embs).sum(0)  # (768,)

    # 4. Augment embeddings
    #    arc_query.squeeze(0): keeps query_generator in gradient graph
    #    weighted_ret:         keeps attention_scorer in gradient graph
    augment      = arc_query.squeeze(0) + weighted_ret          # (768,)
    support_aug  = support_embs + augment.unsqueeze(0)          # (n_s, 768)
    query_aug    = query_embs   + augment.unsqueeze(0)          # (n_q, 768)

    # 5. Inner loop -- FIX 1: create_graph=True
    adapted_params = inner_loop(
        classifier, support_aug, support_labels,
        cfg.inner_lr, cfg.inner_steps,
    )

    # 6. Outer loss on query set (with adapted classifier)
    query_logits = functional_forward(adapted_params, query_aug) # (n_q, 2)
    outer_loss   = F.cross_entropy(query_logits, query_labels)

    # 7. Query accuracy (for logging)
    with torch.no_grad():
        preds = query_logits.argmax(dim=-1).cpu().tolist()
        acc   = sum(p == t for p, t in zip(preds, episode["query_labels"])) / len(preds)

    return outer_loss, acc


# ---------------------------------------------------------------------------
# FIX 5 -- Smoke test: verify gradient flow before epoch 1
# ---------------------------------------------------------------------------

def smoke_test(
    encoder:    TextEncoder,
    classifier: nn.Linear,
    arc:        AdaptiveRetrievalController,
    retriever:  FaissRetriever,
    optimizer:  torch.optim.Optimizer,
    sampler:    CategoryStratifiedEpisodeSampler,
    cfg:        Config,
    device:     torch.device,
) -> None:
    """
    FIX 5: Run cfg.smoke_episodes episodes, backprop, and assert:
      1. ARC parameters have non-zero gradients
         (verifies create_graph=True and that ARC is in the gradient path)
      2. At least one unfrozen encoder layer has non-zero gradients
         (verifies encoder unfreezing + gradient flow through inner loop)
      3. Classifier parameters have non-zero gradients
         (basic sanity: outer loss is computing something)

    Aborts with AssertionError if any check fails -- do not start full
    training with a broken gradient graph.
    """
    print("\n" + "=" * 64)
    print("  FIX 5: Smoke test -- verifying gradient flow")
    print("=" * 64)

    encoder.train()
    classifier.train()
    arc.train()

    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=device)

    for i in range(cfg.smoke_episodes):
        episode   = sampler.sample_episode()
        loss, acc = run_episode(encoder, classifier, arc, retriever, episode, cfg, device)
        total_loss = total_loss + loss
        print(f"  smoke episode {i+1}/{cfg.smoke_episodes}: loss={loss.item():.4f}  acc={acc:.3f}")

    (total_loss / cfg.smoke_episodes).backward()

    print("\n  Gradient report (non-zero only):")

    # -- Classifier --
    clf_grads = {
        n: p.grad.abs().max().item()
        for n, p in classifier.named_parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().max() > 0
    }
    for n, g in clf_grads.items():
        print(f"    GRAD OK  classifier.{n:<20s}  max_grad={g:.6f}")

    # -- ARC --
    arc_grads = {
        n: p.grad.abs().max().item()
        for n, p in arc.named_parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().max() > 0
    }
    for n, g in arc_grads.items():
        print(f"    GRAD OK  arc.{n:<28s}  max_grad={g:.6f}")

    # -- Encoder (unfrozen layers only) --
    enc_grads = {
        n: p.grad.abs().max().item()
        for n, p in encoder.named_parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().max() > 0
    }
    for n, g in list(enc_grads.items())[:10]:   # print first 10 to avoid spam
        print(f"    GRAD OK  encoder.{n:<28s}  max_grad={g:.6f}")
    if len(enc_grads) > 10:
        print(f"    ... and {len(enc_grads)-10} more encoder params with gradients")

    # -- Assertions --
    assert len(clf_grads) > 0, (
        "CRITICAL: Classifier parameters have ZERO gradient.\n"
        "The outer cross-entropy loss is not flowing back at all."
    )

    assert len(arc_grads) > 0, (
        "CRITICAL: ARC parameters have ZERO gradient.\n"
        "The inner loop is NOT using create_graph=True, or ARC output\n"
        "is not connected to the loss. Check run_episode() augment path."
    )

    assert len(enc_grads) > 0, (
        "CRITICAL: No unfrozen encoder parameters have non-zero gradient.\n"
        "Either encoder freezing is wrong, or gradients are not flowing\n"
        "from the inner loop back to the encoder embeddings."
    )

    print("\n  SMOKE TEST PASSED -- gradient flow verified for:")
    print(f"    classifier: {len(clf_grads)} params with gradients")
    print(f"    ARC:        {len(arc_grads)} params with gradients")
    print(f"    encoder:    {len(enc_grads)} params with gradients (unfrozen layers)")
    print("=" * 64 + "\n")

    # Clear gradients before real training
    optimizer.zero_grad()


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(
    encoder:    TextEncoder,
    classifier: nn.Linear,
    arc:        AdaptiveRetrievalController,
    retriever:  FaissRetriever,
    optimizer:  torch.optim.Optimizer,
    sampler:    CategoryStratifiedEpisodeSampler,
    cfg:        Config,
    device:     torch.device,
    epoch:      int,
) -> dict:
    encoder.train()
    classifier.train()
    arc.train()

    losses, accs = [], []

    for ep_idx in range(cfg.episodes_per_epoch):
        optimizer.zero_grad()

        episode   = sampler.sample_episode()
        loss, acc = run_episode(encoder, classifier, arc, retriever, episode, cfg, device)

        loss.backward()

        # FIX 4: gradient clipping -- second-order gradients can spike
        all_trainable = (
            list(classifier.parameters()) +
            list(arc.parameters()) +
            [p for p in encoder.parameters() if p.requires_grad]
        )
        torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=cfg.max_grad_norm)

        optimizer.step()

        losses.append(loss.item())
        accs.append(acc)

        if (ep_idx + 1) % 20 == 0:
            logger.info(
                "Epoch %d  ep %d/%d  loss=%.4f  acc=%.3f",
                epoch, ep_idx + 1, cfg.episodes_per_epoch,
                np.mean(losses[-20:]), np.mean(accs[-20:]),
            )

    return {
        "train_loss": float(np.mean(losses)),
        "train_acc":  float(np.mean(accs)),
    }


@torch.no_grad()
def evaluate(
    encoder:    TextEncoder,
    classifier: nn.Linear,
    arc:        AdaptiveRetrievalController,
    retriever:  FaissRetriever,
    sampler:    CategoryStratifiedEpisodeSampler,
    cfg:        Config,
    device:     torch.device,
) -> dict:
    """
    Evaluation uses the full inner loop (test-time adaptation) but no outer backprop.
    We need torch.enable_grad() for the inner loop even during eval.
    """
    encoder.eval()
    arc.eval()

    accs = []
    for _ in range(cfg.val_episodes):
        episode = sampler.sample_episode()
        with torch.enable_grad():
            # Create fresh classifier copy for eval adaptation
            eval_clf = nn.Linear(cfg.hidden_dim, cfg.n_class).to(device)
            eval_clf.weight.data.copy_(classifier.weight.data)
            eval_clf.bias.data.copy_(classifier.bias.data)
            _, acc = run_episode(encoder, eval_clf, arc, retriever, episode, cfg, device)
        accs.append(acc)

    return aggregate_episode_results(accs)


# ---------------------------------------------------------------------------
# FAISS index builder (call once after preprocess.py)
# ---------------------------------------------------------------------------

def build_faiss_index(cfg: Config, device: torch.device) -> FaissRetriever:
    """
    Build and save the FAISS index from high-resource processed training data.
    Skips building if the index already exists on disk.
    """
    if os.path.exists(os.path.join(cfg.faiss_dir, "metadata.json")):
        logger.info("FAISS index already exists at %s -- loading.", cfg.faiss_dir)
        retriever = FaissRetriever(hidden_dim=cfg.hidden_dim)
        retriever.load(cfg.faiss_dir, device=device)
        retriever.assert_high_resource_only()
        return retriever

    from araml.data.preprocess import HIGH_RESOURCE, LOW_RESOURCE

    logger.info("Building FAISS index from high-resource training data...")
    encoder = TextEncoder(model_name=cfg.model_name, hidden_dim=cfg.hidden_dim).to(device)

    all_records = []
    for lang in HIGH_RESOURCE:
        path = os.path.join(cfg.processed_dir, f"amazon_{lang}.json")
        if not os.path.exists(path):
            logger.warning("  %s not found, skipping for FAISS index.", path)
            continue
        with open(path, encoding="utf-8") as f:
            splits = json.load(f)
        train_records = splits.get("train", [])
        all_records.extend(train_records)
        logger.info("  Loaded %d %s train records", len(train_records), lang)

    if not all_records:
        logger.warning(
            "No high-resource records found. FAISS will run in dummy mode.\n"
            "Run data/download_data.py + data/preprocess.py first."
        )
        return FaissRetriever(hidden_dim=cfg.hidden_dim)

    # Leakage guard: collect low-resource pool texts
    lr_pool_texts = set()
    for lang in LOW_RESOURCE:
        pool_path = os.path.join(cfg.pool_dir, f"lowresource_pool_{lang}.json")
        if os.path.exists(pool_path):
            with open(pool_path, encoding="utf-8") as f:
                pool = json.load(f)
            lr_pool_texts.update(r["text"] for r in pool)

    retriever = FaissRetriever(hidden_dim=cfg.hidden_dim)
    retriever.build(all_records, encoder, device, batch_size=64)
    retriever.assert_high_resource_only()
    retriever.assert_no_lowresource_leakage(lr_pool_texts)
    retriever.save(cfg.faiss_dir)

    return retriever


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(cfg: Config) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = torch.device(cfg.device)

    # -- Models --
    encoder    = TextEncoder(model_name=cfg.model_name, hidden_dim=cfg.hidden_dim).to(device)
    arc        = AdaptiveRetrievalController(
        input_dim=cfg.hidden_dim, max_k=cfg.arc_max_k
    ).to(device)
    classifier = nn.Linear(cfg.hidden_dim, cfg.n_class).to(device)   # FIX: 2 classes

    # -- FIX 2: Freeze encoder, unfreeze layers 9-11 --
    enc_trainable = setup_encoder_freezing(encoder, cfg.unfreeze_from_layer)

    all_trainable = enc_trainable + list(arc.parameters()) + list(classifier.parameters())
    trainable_count = sum(p.numel() for p in all_trainable)
    logger.info("Total trainable parameters: %d (~%.1fM)", trainable_count, trainable_count / 1e6)

    # -- FIX 3: outer_lr = 0.0003 --
    optimizer = torch.optim.AdamW(all_trainable, lr=cfg.outer_lr, weight_decay=1e-4)

    # -- FAISS retrieval index --
    retriever = build_faiss_index(cfg, device)

    # -- Episode sampler (low-resource ja/zh only) --
    sampler = CategoryStratifiedEpisodeSampler.from_pool_files(
        pool_dir=cfg.pool_dir,
        n_shot=cfg.n_shot,
        n_query=cfg.n_query,
        n_class=cfg.n_class,
        seed=cfg.seed,
    )

    # -- FIX 5: Smoke test --
    smoke_test(encoder, classifier, arc, retriever, optimizer, sampler, cfg, device)

    # -- Training --
    best_val_acc = 0.0
    for epoch in range(1, cfg.n_epochs + 1):
        train_metrics = train_epoch(
            encoder, classifier, arc, retriever, optimizer, sampler, cfg, device, epoch
        )
        val_metrics = evaluate(
            encoder, classifier, arc, retriever, sampler, cfg, device
        )

        logger.info(
            "Epoch %2d | train_loss=%.4f  train_acc=%.3f | "
            "val_acc=%.3f +/-%.3f",
            epoch,
            train_metrics["train_loss"], train_metrics["train_acc"],
            val_metrics["mean_accuracy"], val_metrics["95ci"],
        )

        # Save best checkpoint
        if val_metrics["mean_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["mean_accuracy"]
            ckpt = {
                "epoch":      epoch,
                "encoder":    encoder.state_dict(),
                "arc":        arc.state_dict(),
                "classifier": classifier.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_acc":    best_val_acc,
            }
            torch.save(ckpt, os.path.join(cfg.checkpoint_dir, "best_model.pt"))
            logger.info("  Saved best checkpoint (val_acc=%.3f)", best_val_acc)

    logger.info("Training complete. Best val_acc=%.3f", best_val_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARAML Training")
    parser.add_argument("--pool_dir",       default="araml/data")
    parser.add_argument("--processed_dir",  default="araml/data/processed")
    parser.add_argument("--faiss_dir",      default="araml/results/faiss")
    parser.add_argument("--checkpoint_dir", default="araml/results")
    parser.add_argument("--n_epochs",       type=int,   default=10)
    parser.add_argument("--inner_lr",       type=float, default=0.01)
    parser.add_argument("--inner_steps",    type=int,   default=5)
    parser.add_argument("--outer_lr",       type=float, default=0.0003)
    parser.add_argument("--n_shot",         type=int,   default=5)
    parser.add_argument("--n_query",        type=int,   default=10)
    parser.add_argument("--episodes_per_epoch", type=int, default=100)
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = Config(**{k: v for k, v in vars(args).items()})
    main(cfg)
