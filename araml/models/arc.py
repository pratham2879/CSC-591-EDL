"""
arc.py — Adaptive Retrieval Controller (ARC)
Decides: what to retrieve (query generation) and how many examples to retrieve.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveRetrievalController(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, max_k: int = 10):
        """
        Args:
            input_dim: Dimension of task embedding from encoder.
            hidden_dim: ARC hidden layer size.
            max_k: Maximum number of examples to retrieve.
        """
        super().__init__()
        self.max_k = max_k

        # Query generation: maps task embedding → retrieval query vector
        self.query_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Budget predictor: maps task embedding → number of examples to retrieve
        self.budget_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()   # Output in [0, 1], scaled to [1, max_k]
        )

        # Attention weights for retrieved examples
        self.attention_scorer = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def generate_query(self, task_embedding: torch.Tensor) -> torch.Tensor:
        """Generate a task-specific retrieval query."""
        return self.query_generator(task_embedding)

    def predict_budget(self, task_embedding: torch.Tensor) -> int:
        """Predict how many examples to retrieve based on task difficulty."""
        budget_ratio = self.budget_predictor(task_embedding)  # (batch, 1)
        k = max(1, int(budget_ratio.mean().item() * self.max_k))
        return k

    def compute_attention_weights(
        self,
        task_embedding: torch.Tensor,
        retrieved_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention weights for retrieved examples.
        Args:
            task_embedding: (1, D)
            retrieved_embeddings: (K, D)
        Returns:
            weights: (K,) softmax attention weights
        """
        k = retrieved_embeddings.size(0)
        task_exp = task_embedding.expand(k, -1)           # (K, D)
        combined = torch.cat([task_exp, retrieved_embeddings], dim=-1)  # (K, 2D)
        scores = self.attention_scorer(combined).squeeze(-1)  # (K,)
        return F.softmax(scores, dim=0)

    def forward(self, task_embedding: torch.Tensor, retrieved_embeddings: torch.Tensor):
        """
        Returns retrieval query, budget, and attention-weighted retrieved embedding.
        """
        query = self.generate_query(task_embedding)
        budget = self.predict_budget(task_embedding)
        weights = self.compute_attention_weights(task_embedding, retrieved_embeddings)
        weighted_retrieval = (weights.unsqueeze(-1) * retrieved_embeddings).sum(0)  # (D,)
        return query, budget, weighted_retrieval, weights
