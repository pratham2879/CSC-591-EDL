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


def maml_inner_loop(
    model: MetaLearner,
    support_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
    device: torch.device
):
    """
    Perform MAML inner loop adaptation on support set.
    Returns adapted model (via `higher` functional model).
    """
    inner_opt = torch.optim.SGD(model.parameters(), lr=inner_lr)

    with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
        for _ in range(inner_steps):
            support_logits = fmodel(support_embeddings)
            inner_loss = F.cross_entropy(support_logits, support_labels)
            diffopt.step(inner_loss)
        return fmodel


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
    """
    outer_optimizer.zero_grad()

    support_texts = episode["support_texts"]
    support_labels = torch.tensor(episode["support_labels"]).to(device)
    query_texts = episode["query_texts"]
    query_labels = torch.tensor(episode["query_labels"]).to(device)

    # Encode support set
    support_embs = encoder.encode_text(support_texts, device)  # (N*K, D)
    task_emb = support_embs.mean(0, keepdim=True)              # (1, D)

    # Adaptive retrieval
    k = arc.predict_budget(task_emb)
    query_vec = arc.generate_query(task_emb).detach().cpu().numpy()
    retrieved = retrieval_index.retrieve(query_vec, k=k)

    retrieved_texts = retrieved["texts"]
    ret_embs = encoder.encode_text(retrieved_texts, device)    # (k, D)

    # Attention-weighted retrieval embedding
    _, _, weighted_ret_emb, _ = arc(task_emb, ret_embs)       # (D,)
    weighted_ret_emb = weighted_ret_emb.unsqueeze(0).expand(support_embs.size(0), -1)

    # Augmented support: [support_emb || retrieval_emb]
    aug_support = torch.cat([support_embs, weighted_ret_emb], dim=-1)

    # MAML inner loop
    adapted_model = maml_inner_loop(
        meta_learner, aug_support, support_labels,
        config["meta_learning"]["inner_lr"],
        config["meta_learning"]["inner_steps"],
        device
    )

    # Query set evaluation
    query_embs = encoder.encode_text(query_texts, device)
    weighted_ret_emb_q = weighted_ret_emb[:1].expand(query_embs.size(0), -1)
    aug_query = torch.cat([query_embs, weighted_ret_emb_q], dim=-1)

    query_logits = adapted_model(aug_query)
    outer_loss = F.cross_entropy(query_logits, query_labels)
    outer_loss.backward()
    outer_optimizer.step()

    acc = (query_logits.argmax(-1) == query_labels).float().mean().item()
    return outer_loss.item(), acc
