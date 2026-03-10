"""
araml.py — Full ARAML model integrating all components
"""
import torch
import torch.nn as nn
from models.encoder import TextEncoder
from models.arc import AdaptiveRetrievalController
from models.meta_learner import MetaLearner


class ARAML(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        arc_cfg = config["retrieval"]
        meta_cfg = config["meta_learning"]

        self.encoder = TextEncoder(
            model_name=model_cfg["encoder"],
            hidden_dim=model_cfg["hidden_dim"]
        )
        self.arc = AdaptiveRetrievalController(
            input_dim=model_cfg["hidden_dim"],
            hidden_dim=arc_cfg["arc_hidden_dim"],
            max_k=arc_cfg["max_retrieved"]
        )
        self.meta_learner = MetaLearner(
            input_dim=model_cfg["hidden_dim"],
            num_classes=model_cfg["num_classes"]
        )

    def get_components(self):
        return self.encoder, self.arc, self.meta_learner

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
