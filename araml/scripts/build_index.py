"""
build_index.py — Build the cross-lingual FAISS retrieval index from HRL data
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import yaml
import torch
import numpy as np
import argparse
from models.encoder import TextEncoder
from models.retrieval_index import CrossLingualRetrievalIndex

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_LANGUAGES = ["en", "de", "es", "fr"]
MAX_PER_LANG = 10_000
BATCH_SIZE = 32


def build_index(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder = TextEncoder(model_name=config["model"]["encoder"]).to(device)
    encoder.eval()

    index = CrossLingualRetrievalIndex(
        embedding_dim=config["model"]["hidden_dim"],
        similarity=config["retrieval"]["similarity"],
    )

    data_dir = os.path.join(_ROOT, "data", "processed")

    for lang in SOURCE_LANGUAGES:
        data_path = os.path.join(data_dir, f"amazon_{lang}.json")
        if not os.path.exists(data_path):
            print(f"Skipping {lang} — data not found at {data_path}")
            continue

        with open(data_path, encoding="utf-8") as f:
            records = json.load(f)

        train_records = records["train"][:MAX_PER_LANG]
        texts  = [r["text"]  for r in train_records]
        labels = [r["label"] for r in train_records]

        n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n[{lang}] {len(texts)} examples — {n_batches} batches")
        all_embs = []

        with torch.no_grad():
            for b, i in enumerate(range(0, len(texts), BATCH_SIZE)):
                batch = texts[i:i + BATCH_SIZE]
                embs  = encoder.encode_text(batch, device)
                all_embs.append(embs.cpu().numpy())
                print(f"  Encoding [{lang}] batch {b+1}/{n_batches}...", flush=True)

        all_embs = np.vstack(all_embs)
        index.add(all_embs, texts, labels, lang)
        print(f"  Done [{lang}]. Index size so far: {len(index)}")

    save_path = os.path.join(_ROOT, "results", "retrieval_index")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    index.save(save_path)
    print(f"\nRetrieval index saved to {save_path}")
    print(f"Total vectors: {index.index.ntotal}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(_ROOT, "configs", "config.yaml"))
    args = parser.parse_args()
    build_index(args.config)
