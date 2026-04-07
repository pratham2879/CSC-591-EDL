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
from tqdm import tqdm
from models.encoder import TextEncoder
from models.retrieval_index import CrossLingualRetrievalIndex


def build_index(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    encoder = TextEncoder(model_name=config["model"]["encoder"]).to(device)
    encoder.eval()

    index = CrossLingualRetrievalIndex(
        embedding_dim=config["model"]["hidden_dim"],
        similarity=config["retrieval"]["similarity"]
    )

    source_langs = config["data"]["source_languages"]
    data_dir = "data/processed"
    batch_size = 32

    for lang in source_langs:
        data_path = os.path.join(data_dir, f"amazon_{lang}.json")
        if not os.path.exists(data_path):
            print(f"Skipping {lang} — data not found.")
            continue

        with open(data_path, encoding="utf-8") as f:
            records = json.load(f)

        # Only use training split for the index
        train_records = [r for r in records if r["split"] == "train"]
        texts = [r["text"] for r in train_records]
        labels = [r["label"] for r in train_records]

        print(f"Encoding {len(texts)} examples from [{lang}]...")
        all_embs = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i:i + batch_size]
                embs = encoder.encode_text(batch, device)
                all_embs.append(embs.cpu().numpy())

        all_embs = np.vstack(all_embs)
        index.add(all_embs, texts, labels, lang)
        print(f"  Added {len(texts)} embeddings. Index size: {len(index)}")

    os.makedirs("results", exist_ok=True)
    index.save("results/retrieval_index")
    print("\nRetrieval index saved to results/retrieval_index")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    build_index(args.config)
