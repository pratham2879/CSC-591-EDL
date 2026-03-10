"""
araml.py — Full ARAML model integrating all components.
"""
import torch
import torch.nn as nn
from models.encoder import TextEncoder
from models.arc import AdaptiveRetrievalController
from models.meta_learner import MetaLearner


def _resolve_num_classes(config: dict) -> int:
    """Pull num_classes from the active dataset sub-config."""
    ds_name = config["dataset"]["name"]
    return config["dataset"][ds_name]["num_classes"]


class ARAML(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        arc_cfg = config["retrieval"]
        num_classes = _resolve_num_classes(config)

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
            num_classes=num_classes
        )

    def get_components(self):
        return self.encoder, self.arc, self.meta_learner

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
