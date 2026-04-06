"""
meta_learner.py — MAML-based Meta-Learner
Uses the `higher` library for differentiable inner-loop optimization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import higher


class MetaLearner(nn.Module):
    def __init__(self, input_dim: int = 768, num_classes: int = 5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, 256),  # *2 because input = [task_emb || retrieval_emb]
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


def meta_train_step(
    encoder,
    arc,
    meta_learner: MetaLearner,
    retrieval_index,
    episode: dict,
    config: dict,
    device: torch.device,
    outer_optimizer: torch.optim.Optimizer
):
    """
    One full MAML meta-training step over an episode.
    Query evaluation happens inside the higher context so adapted weights are valid.
    """
    outer_optimizer.zero_grad()

    meta_cfg = config["meta_learning"]
    support_texts = episode["support_texts"]
    support_labels = torch.tensor(episode["support_labels"]).to(device)
    query_texts = episode["query_texts"]
    query_labels = torch.tensor(episode["query_labels"]).to(device)

    # Encode support and query — no_grad since encoder is updated by outer loop only
    with torch.no_grad():
        support_embs = encoder.encode_text(support_texts, device)   # (N*K, D)
        query_embs = encoder.encode_text(query_texts, device)        # (Q, D)

    task_emb = support_embs.mean(0, keepdim=True)                    # (1, D)

    # Adaptive retrieval
    k = arc.predict_budget(task_emb)
    query_tensor = arc.generate_query(task_emb)                      # (1, D) — grad preserved
    query_vec = query_tensor.detach().cpu().numpy()                   # FAISS needs numpy

    retrieved = retrieval_index.retrieve(query_vec, k=k)
    retrieved_texts = retrieved["texts"]

    with torch.no_grad():
        ret_embs = encoder.encode_text(retrieved_texts, device)      # (k, D)

    # Attention-weighted retrieval — gradient flows through query_tensor -> query_generator
    weights = arc.compute_attention_weights(query_tensor, ret_embs)  # (K,)
    weighted_ret_emb = (weights.unsqueeze(-1) * ret_embs).sum(0)     # (D,)

    # Augmented support and query: [text_emb || retrieval_emb]
    aug_support = torch.cat(
        [support_embs, weighted_ret_emb.unsqueeze(0).expand(support_embs.size(0), -1)], dim=-1
    )
    aug_query = torch.cat(
        [query_embs, weighted_ret_emb.unsqueeze(0).expand(query_embs.size(0), -1)], dim=-1
    )

    # MAML: inner loop + query eval must both happen inside the higher context
    inner_opt = torch.optim.SGD(meta_learner.parameters(), lr=meta_cfg["inner_lr"])

    with higher.innerloop_ctx(meta_learner, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
        # Inner loop adaptation on support set
        for _ in range(meta_cfg["inner_steps"]):
            support_logits = fmodel(aug_support)
            inner_loss = F.cross_entropy(support_logits, support_labels)
            diffopt.step(inner_loss)

        # Query evaluation with adapted weights — still inside context
        query_logits = fmodel(aug_query)
        outer_loss = F.cross_entropy(query_logits, query_labels)

    outer_loss.backward()
    outer_optimizer.step()

    acc = (query_logits.argmax(-1) == query_labels).float().mean().item()
    return outer_loss.item(), acc


def maml_eval_episode(
    encoder,
    arc,
    meta_learner: MetaLearner,
    retrieval_index,
    episode: dict,
    config: dict,
    device: torch.device
) -> float:
    """
    Evaluate a single episode: adapt on support, predict on query.
    Used by evaluate.py. No outer gradient computation.
    """
    meta_cfg = config["meta_learning"]

    with torch.no_grad():
        support_embs = encoder.encode_text(episode["support_texts"], device)
        query_embs = encoder.encode_text(episode["query_texts"], device)

    support_labels = torch.tensor(episode["support_labels"]).to(device)
    query_labels = torch.tensor(episode["query_labels"]).to(device)

    task_emb = support_embs.mean(0, keepdim=True)
    k = arc.predict_budget(task_emb)
    query_vec = arc.generate_query(task_emb).detach().cpu().numpy()
    retrieved = retrieval_index.retrieve(query_vec, k=k)

    with torch.no_grad():
        ret_embs = encoder.encode_text(retrieved["texts"], device)

    weights = arc.compute_attention_weights(task_emb, ret_embs)
    weighted_ret_emb = (weights.unsqueeze(-1) * ret_embs).sum(0)

    aug_support = torch.cat(
        [support_embs, weighted_ret_emb.unsqueeze(0).expand(support_embs.size(0), -1)], dim=-1
    )
    aug_query = torch.cat(
        [query_embs, weighted_ret_emb.unsqueeze(0).expand(query_embs.size(0), -1)], dim=-1
    )

    inner_opt = torch.optim.SGD(meta_learner.parameters(), lr=meta_cfg["inner_lr"])

    with higher.innerloop_ctx(meta_learner, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
        for _ in range(meta_cfg["inner_steps"]):
            support_logits = fmodel(aug_support)
            inner_loss = F.cross_entropy(support_logits, support_labels)
            diffopt.step(inner_loss)

        with torch.no_grad():
            query_logits = fmodel(aug_query)

    acc = (query_logits.argmax(-1) == query_labels).float().mean().item()
    return acc
