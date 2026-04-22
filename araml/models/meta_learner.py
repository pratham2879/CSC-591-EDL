"""
meta_learner.py — MAML-based Meta-Learner

Root cause of loss = ln(2) = 0.6931 (random-chance binary):
  The `higher`-based inner loop did NOT set track_higher_grads / create_graph,
  so adapted_params were disconnected from the original weights.
  outer_loss.backward() produced zero gradients in ARC and the encoder.

Fixes applied here:
  FIX 1  inner_loop uses torch.autograd.grad(create_graph=True)
         — this is THE critical change; everything else is secondary.
  FIX 3  meta_train_step clips gradients before optimizer.step()
         (second-order grads spike; clipping at 1.0 stabilises training).

Fixes applied in the training script (scripts/train.py):
  FIX 2  Encoder layers 9-11 + pooler unfrozen (so support/query embeddings
         carry gradients back through the inner loop to both encoder and ARC).
  FIX 3  outer_lr = 0.0003  (second-order gradients are larger in magnitude
         than first-order; lower lr prevents overshooting).

Architecture change:
  MetaLearner is now a single nn.Linear (was Sequential with ReLU + Dropout).
  A single Linear lets inner_loop use F.linear directly with 'weight'/'bias'
  keys from named_parameters().  Augmentation changes from concatenation
  (→ 1536-d) to elementwise addition (→ 768-d) to match the head dimension.
  This also removes Dropout which made the inner loss non-deterministic and
  slowed adaptation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MetaLearner(nn.Module):
    """
    Binary sentiment classifier for MAML few-shot adaptation.

    Single Linear layer so that the functional inner loop can operate
    on {'weight', 'bias'} directly via F.linear without a custom
    functional_sequential_forward implementation.

    input_dim  : encoder hidden dim (768 for XLM-R base).
    num_classes: 2  (negative / positive, after label remapping in preprocess.py).
    """
    def __init__(self, input_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.1)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# ---------------------------------------------------------------------------
# FIX 1 — functional MAML inner loop with create_graph=True
# ---------------------------------------------------------------------------

def inner_loop(
    classifier: nn.Linear,
    support_embs: torch.Tensor,
    support_labels: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
) -> dict:
    """
    Differentiable MAML inner loop.

    create_graph=True keeps every gradient computation in the forward
    graph so that outer_loss.backward() can differentiate through the
    entire chain of inner updates.  Without this flag, adapted_params
    are leaf constants — the second-order path vanishes and ARC / encoder
    gradients are identically zero.

    Args:
        classifier    : nn.Linear providing the initial fast weights.
                        Pass meta_learner.classifier (not MetaLearner itself)
                        so named_parameters() yields 'weight' and 'bias'.
        support_embs  : (n_support, D) float tensor — may carry grad if
                        encoder layers 9-11 are unfrozen (FIX 2).
        support_labels: (n_support,) long tensor.
        inner_lr      : inner-loop step size (config: meta_learning.inner_lr).
        inner_steps   : number of gradient steps (config: meta_learning.inner_steps).

    Returns:
        adapted_params: {'weight': tensor (n_cls, D), 'bias': tensor (n_cls,)}
                        These tensors carry the full computation graph back
                        through every inner-loop gradient step.
    """
    params = {k: v.clone() for k, v in classifier.named_parameters()}

    for _ in range(inner_steps):
        logits = F.linear(support_embs, params["weight"], params["bias"])
        loss   = F.cross_entropy(logits, support_labels, label_smoothing=0.1)
        grads  = torch.autograd.grad(
            loss,
            list(params.values()),
            create_graph=True,   # THE CRITICAL LINE — retains graph for second-order grads
            allow_unused=True,
        )
        params = {
            k: v - inner_lr * (g if g is not None else torch.zeros_like(v))
            for (k, v), g in zip(params.items(), grads)
        }

    return params


# ---------------------------------------------------------------------------
# Shared episode forward pass (used by training, eval, and diagnostic)
# ---------------------------------------------------------------------------

def _episode_forward(
    encoder,
    arc,
    meta_learner: MetaLearner,
    retrieval_index,
    episode: dict,
    config: dict,
    device: torch.device,
) -> tuple:
    """
    One episode: encode → retrieve → augment → inner loop → outer loss.

    Returns (outer_loss tensor, accuracy float).
    The returned loss still has its computation graph attached;
    caller decides when to call .backward().
    """
    meta_cfg = config["meta_learning"]
    use_amp  = device.type == "cuda"

    support_texts  = episode["support_texts"]
    support_labels = torch.tensor(
        episode["support_labels"], dtype=torch.long, device=device
    )
    query_texts  = episode["query_texts"]
    query_labels = torch.tensor(
        episode["query_labels"], dtype=torch.long, device=device
    )

    # 1. Encode — differentiable once encoder layers 9-11 are unfrozen (FIX 2).
    #    Encode support+query in a single forward pass to maximise GPU utilisation.
    all_texts = support_texts + query_texts
    n_support = len(support_texts)
    with autocast('cuda', enabled=use_amp):
        all_embs     = encoder.encode_text(all_texts, device)   # (N*K+Q, D)
    support_embs = all_embs[:n_support]                         # (N*K, D)
    query_embs   = all_embs[n_support:]                         # (Q,   D)

    # 2. Task embedding: mean of support representations
    task_emb = support_embs.mean(0, keepdim=True)               # (1, D)

    with autocast('cuda', enabled=use_amp):
        # 3. ARC: generate retrieval query
        arc_query = arc.generate_query(task_emb)                    # (1, D)
        k         = max(1, arc.predict_budget(task_emb))

        # FAISS search is non-differentiable → detach before numpy conversion
        query_vec       = arc_query.detach().float().cpu().numpy()
        retrieved       = retrieval_index.retrieve(query_vec, k=k)
        retrieved_texts = retrieved["texts"]

        with torch.no_grad():
            ret_embs = encoder.encode_text(retrieved_texts, device) # (k, D)

        # 4. Attention — grad path: arc_query → query_generator params
        attn_weights = arc.compute_attention_weights(arc_query, ret_embs)  # (k,)
        weighted_ret = (attn_weights.unsqueeze(-1) * ret_embs).sum(0)      # (D,)

        # 5. Augment: L2-normalize before elementwise add
        support_embs_n = F.normalize(support_embs, p=2, dim=-1)
        query_embs_n   = F.normalize(query_embs,   p=2, dim=-1)
        weighted_ret_n = F.normalize(weighted_ret,  p=2, dim=-1)
        support_aug = support_embs_n + weighted_ret_n.unsqueeze(0)    # (N*K, D)
        query_aug   = query_embs_n   + weighted_ret_n.unsqueeze(0)    # (Q,   D)

    # 6. Inner loop with create_graph=True (FIX 1).
    #    Run in float32 — autograd.grad(create_graph=True) is more numerically
    #    stable outside autocast; cast augmented embeddings back to fp32 first.
    support_aug = support_aug.float()
    query_aug   = query_aug.float()

    adapted = inner_loop(
        meta_learner.classifier,
        support_aug, support_labels,
        meta_cfg["inner_lr"], meta_cfg["inner_steps"],
    )

    # 7. Outer loss on query set with adapted weights
    query_logits = F.linear(query_aug, adapted["weight"], adapted["bias"])  # (Q, n_cls)
    outer_loss   = F.cross_entropy(query_logits, query_labels)

    with torch.no_grad():
        acc = (query_logits.argmax(-1) == query_labels).float().mean().item()
        predictions = query_logits.argmax(-1).cpu().numpy()
        targets = query_labels.cpu().numpy()

    return outer_loss, acc, predictions, targets


# ---------------------------------------------------------------------------
# Diagnostic — call BEFORE epoch 1 to confirm ARC gradient flow
# ---------------------------------------------------------------------------

def diagnose_gradient_flow(
    encoder,
    arc,
    meta_learner: MetaLearner,
    retrieval_index,
    episode: dict,
    config: dict,
    device: torch.device,
) -> None:
    """
    Run one episode, call backward(), and report which ARC parameters have
    non-zero gradients.

    Expected result after FIX 1:
      ARC params with nonzero grad: [query_generator.0.weight,
                                     query_generator.0.bias,
                                     query_generator.2.weight,
                                     query_generator.2.bias,
                                     attention_scorer.0.weight,
                                     attention_scorer.0.bias,
                                     attention_scorer.2.weight,
                                     attention_scorer.2.bias]

    If the list is empty, the inner loop is still severing the graph —
    double-check that inner_loop() passes create_graph=True.
    """
    # Zero any stale gradients
    for module in [arc, meta_learner]:
        for p in module.parameters():
            if p.grad is not None:
                p.grad.zero_()

    outer_loss, acc, _, _ = _episode_forward(
        encoder, arc, meta_learner, retrieval_index, episode, config, device
    )
    outer_loss.backward()

    arc_grads = [
        name for name, p in arc.named_parameters()
        if p.grad is not None and p.grad.abs().max().item() > 1e-9
    ]

    print("\n" + "=" * 64)
    print("  GRADIENT DIAGNOSTIC — run before epoch 1")
    print("=" * 64)
    print(f"  outer_loss : {outer_loss.item():.4f}  (ln2 = 0.6931)")
    print(f"  acc        : {acc:.3f}")
    print()
    print(f"  ARC params with nonzero grad ({len(arc_grads)} / "
          f"{sum(1 for _ in arc.named_parameters())} total):")
    if arc_grads:
        for name, p in arc.named_parameters():
            status = "OK  " if (p.grad is not None and p.grad.abs().max().item() > 1e-9) else "ZERO"
            mag    = p.grad.abs().max().item() if p.grad is not None else 0.0
            print(f"    [{status}] {name:<42s}  max={mag:.6f}")
        print()
        print("  GRADIENT FLOW OK — ARC is in the computation graph.")
    else:
        print("    (none)")
        print()
        print("  CRITICAL: ARC grads are ALL ZERO.")
        print("  The inner loop is severing the computation graph.")
        print("  Verify inner_loop() has create_graph=True in autograd.grad().")
    print("=" * 64 + "\n")

    # Clear grads so real training starts clean
    for module in [arc, meta_learner]:
        for p in module.parameters():
            if p.grad is not None:
                p.grad.zero_()
    for p in encoder.parameters():
        if p.grad is not None:
            p.grad.zero_()


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def meta_train_step(
    encoder,
    arc,
    meta_learner: MetaLearner,
    retrieval_index,
    episode: dict,
    config: dict,
    device: torch.device,
    outer_optimizer: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
    scaler=None,
) -> tuple:
    """
    One MAML meta-training step: forward → backward → clip → step.

    Pass a torch.cuda.amp.GradScaler as `scaler` to enable mixed-precision
    training (RTX 4000 / any CUDA device with FP16 support).

    Returns
    -------
    loss      : float  — scalar outer cross-entropy loss
    acc       : float  — query-set accuracy with adapted classifier
    grad_norm : float  — total gradient norm BEFORE clipping
    """
    outer_optimizer.zero_grad()

    outer_loss, acc, predictions, targets = _episode_forward(
        encoder, arc, meta_learner, retrieval_index, episode, config, device
    )

    all_trainable = (
        list(meta_learner.parameters())
        + list(arc.parameters())
        + [p for p in encoder.parameters() if p.requires_grad]
    )

    if scaler is not None:
        scaler.scale(outer_loss).backward()
        scaler.unscale_(outer_optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=max_grad_norm)
        scaler.step(outer_optimizer)
        scaler.update()
    else:
        outer_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=max_grad_norm)
        outer_optimizer.step()

    return outer_loss.item(), acc, float(grad_norm), predictions, targets


# ---------------------------------------------------------------------------
# Evaluation step
# ---------------------------------------------------------------------------

def maml_eval_episode(
    encoder,
    arc,
    meta_learner: MetaLearner,
    retrieval_index,
    episode: dict,
    config: dict,
    device: torch.device,
) -> dict:
    """
    Evaluate one episode: adapt on support, predict on query.
    Uses a fresh classifier copy so eval never modifies training weights.
    Needs enable_grad() for the inner loop even during encoder.eval().
    
    Returns:
        dict with 'accuracy' and 'kappa' keys
    """
    from sklearn.metrics import cohen_kappa_score
    
    encoder.eval()
    arc.eval()

    with torch.enable_grad():
        eval_clf = MetaLearner(
            input_dim=meta_learner.classifier.in_features,
            num_classes=meta_learner.classifier.out_features,
        ).to(device)
        eval_clf.classifier.weight.data.copy_(meta_learner.classifier.weight.data)
        eval_clf.classifier.bias.data.copy_(meta_learner.classifier.bias.data)

        _, acc, predictions, targets = _episode_forward(
            encoder, arc, eval_clf, retrieval_index, episode, config, device
        )

        kappa = cohen_kappa_score(targets, predictions)

    return {"accuracy": acc, "kappa": kappa, "predictions": predictions, "targets": targets}
