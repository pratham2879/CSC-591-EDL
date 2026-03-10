"""
build_index.py — Build the cross-lingual FAISS retrieval index from HRL data.
"""
import os
import sys
import yaml
import torch
import numpy as np
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.encoder import TextEncoder
from models.retrieval_index import CrossLingualRetrievalIndex
from utils.config_utils import get_dataset_config, load_language_data


def build_index(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ds_cfg = get_dataset_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset:       {ds_cfg['dataset_name']}")

    encoder = TextEncoder(model_name=config["model"]["encoder"]).to(device)
    encoder.eval()

    index = CrossLingualRetrievalIndex(
        embedding_dim=config["model"]["hidden_dim"],
        similarity=config["retrieval"]["similarity"]
    )

    batch_size = 32
    for lang in ds_cfg["source_languages"]:
        records = load_language_data(ds_cfg, lang, split="train")
        if not records:
            print(f"Skipping {lang} — no training data.")
            continue

        texts = [r["text"] for r in records]
        labels = [r["label"] for r in records]

        print(f"Encoding {len(texts):,} examples from [{lang}] ...")
        all_embs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i:i + batch_size]
                embs = encoder.encode_text(batch, device)
                all_embs.append(embs.cpu().numpy())

        all_embs = np.vstack(all_embs)
        index.add(all_embs, texts, labels, lang)
        print(f"  Added {len(texts):,} embeddings. Index size: {len(index):,}")

    os.makedirs("results", exist_ok=True)
    index.save("results/retrieval_index")
    print("\nRetrieval index saved to results/retrieval_index")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    build_index(args.config)
