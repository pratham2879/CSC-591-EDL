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
    # here the model is the meta-learner, and support_embeddings are the augmented support set embeddings
    # here tensor dimensions are (N*K, D) where N=number of classes, K=shots per class, D=hidden_dim*2 (after concatenation)
    # here tensor means the support_labels are the labels for the support set, with dimensions (N*K,) where each label is in [0, N-1] for N-way classification
):
    """
    Perform MAML inner loop adaptation on support set.
    Returns adapted model (via `higher` functional model).
    inner loop and outer loop are fully differentiable, allowing gradients to flow back to the meta-learner parameters that
    means the meta-learner is updated based on how well it adapts to the support set and performs on the query set after adaptation.
    """
    inner_opt = torch.optim.SGD(model.parameters(), lr=inner_lr)
# inner loop ctx means that we are creating a differentiable copy of the model for the inner loop optimization, allowing us to compute gradients through the adaptation process. 
# The copy_initial_weights=False argument means that the inner loop starts with the same initial weights as the original model, which is important for MAML to learn good initializations.
    with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
        for _ in range(inner_steps):
            support_logits = fmodel(support_embeddings)
            inner_loss = F.cross_entropy(support_logits, support_labels)
            diffopt.step(inner_loss)
        return fmodel
# the inner loop and outer loop are fully differentiable, allowing gradients to flow back to the meta-learner parameters that influence how well the model adapts to the support set and performs on the query set after adaptation.
#the inner loop performs several steps of gradient descent on the support set, and the outer loop updates the meta-learner parameters based on the performance of the adapted model on the query set.
#the inner loop performs on the support set, and the outer loop evaluates on the query set, allowing the meta-learner to learn how to adapt quickly to new tasks based on limited support examples.
#inner loop retrieves the support set embeddings, computes the loss, and updates the model parameters using the differentiable optimizer. The adapted model is then returned for evaluation on the query set in the outer loop.
#the support set embeddings are augmented with retrieval-based information, allowing the meta-learner to leverage relevant examples from the retrieval index during adaptation. 
# The inner loop optimization allows the model to quickly adapt to the specific task defined by the support set, while the outer loop updates the meta-learner parameters based on how well it performs on the query set after adaptation.
# what are support set embeddings? - they are the encoded representations of the support set examples, which are used for the inner loop adaptation. They are typically obtained by encoding the support set texts using the text encoder and then augmenting them with retrieval-based information from the retrieval index. The resulting augmented embeddings are then used as input to the meta-learner during the inner loop optimization process.
def meta_train_step(
    encoder,
    arc,
    meta_learner: MetaLearner,
    retrieval_index,
    episode: dict,
    config: dict,
    device: torch.device,
    outer_optimizer: torch.optim.Optimizer,
    step: bool = True
):
    """
    One full MAML meta-training step over an episode.
    If step=False, only accumulates gradients (for batched episodes).
    """
    if step:
        outer_optimizer.zero_grad()

    meta_cfg = config["meta_learning"]
    support_texts = episode["support_texts"]
    support_labels = torch.tensor(episode["support_labels"]).to(device)
    query_texts = episode["query_texts"]
    query_labels = torch.tensor(episode["query_labels"]).to(device)
#above information is the support set and query set for the current episode, where support_texts and support_labels are the texts and labels for the support set, and query_texts and query_labels are the texts and labels for the query set. 
# The support set is used for inner loop adaptation, while the query set is used for evaluating the adapted model in the outer loop.
    
    # Encode support set
    support_embs = encoder.encode_text(support_texts, device)  # (N*K, D)
    task_emb = support_embs.mean(0, keepdim=True)              # (1, D)

    # Adaptive retrieval
    k = arc.predict_budget(task_emb)
    query_vec = arc.generate_query(task_emb).detach().cpu().numpy() #this line generates a query vector based on the task embedding, which is then used to retrieve relevant examples from the retrieval index. 
    #The detach() method is used to prevent gradients from flowing back through the ARC during the retrieval process, and the resulting query vector is converted to a NumPy array for compatibility with the retrieval index.
    retrieved = retrieval_index.retrieve(query_vec, k=k)
#retrieval index is formed in the quick_test.py file, where we encode a set of texts and add their embeddings to the index along with their labels and language information. The retrieve method of the retrieval index takes a query vector and retrieves the top-k most similar examples from the index based on the specified similarity metric (e.g., cosine similarity). The retrieved examples are then used to augment the support set embeddings for the inner loop adaptation in the meta-training step.
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

    if step:
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
