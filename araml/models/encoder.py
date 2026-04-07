"""
encoder.py — Multilingual text encoder (mBERT / XLM-R)
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "xlm-roberta-base", hidden_dim: int = 768):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation -- classification token
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
