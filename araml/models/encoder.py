"""
encoder.py — Multilingual text encoder (mBERT / XLM-R)
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "xlm-roberta-base", hidden_dim: int = 768):
        super().__init__()
        model_source = self._resolve_model_source(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.encoder = AutoModel.from_pretrained(model_source)
        self.hidden_dim = hidden_dim

    @staticmethod
    def _resolve_model_source(model_name: str) -> str:
        """
        Prefer a cached Hugging Face snapshot when available so local smoke tests
        do not depend on network access.
        """
        if os.path.exists(model_name):
            return model_name

        hub_root = Path.home() / ".cache" / "huggingface" / "hub"
        snapshot_root = hub_root / f"models--{model_name.replace('/', '--')}" / "snapshots"
        if snapshot_root.exists():
            snapshots = sorted(p for p in snapshot_root.iterdir() if p.is_dir())
            if snapshots:
                return str(snapshots[-1])

        return model_name

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        return cls_output  # (batch, hidden_dim)

    def encode_text(self, texts: list, device, max_length: int = 128):
        """Tokenize and encode a list of strings."""
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        return self.forward(input_ids, attention_mask)
