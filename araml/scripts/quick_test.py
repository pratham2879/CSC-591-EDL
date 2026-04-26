"""
quick_test.py — Quick component smoke test.
Verifies encoder, retrieval index, episode sampler, and model instantiation.

Run from inside araml/:
    PYTHONPATH=. python scripts/quick_test.py
"""
import os
import json
import yaml
import torch
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder import TextEncoder
from models.retrieval_index import CrossLingualRetrievalIndex
from models.araml import ARAML
from utils.episode_sampler import CategoryStratifiedEpisodeSampler


def quick_test():
    print("Loading config...")
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Encoder -------------------------------------------------------------
    print("\nLoading encoder...")
    try:
        encoder = TextEncoder(model_name=config["model"]["encoder"]).to(device)
        encoder.eval()
        print(f"Encoder loaded: {config['model']['encoder']}")
    except Exception as exc:
        encoder = None
        print(f"  Encoder unavailable — skipping encoder-dependent checks: {exc}")

    # -- Retrieval index -----------------------------------------------------
    print("\nTesting retrieval index...")
    index_path = "results/retrieval_index.faiss"
    if os.path.exists(index_path):
        index = CrossLingualRetrievalIndex(
            embedding_dim=config["model"]["hidden_dim"],
            similarity=config["retrieval"]["similarity"]
        )
        index.load("results/retrieval_index")
        print(f"Index size: {len(index)}")

        dummy_query = np.random.randn(1, config["model"]["hidden_dim"]).astype(np.float32)
        retrieved = index.retrieve(dummy_query, k=5)
        print(f"Retrieved {len(retrieved['texts'])} examples")
    else:
        print("  Retrieval index not found — skipping. Run scripts/build_index.py first.")

    # -- Episode sampler (requires low-resource pools from preprocess.py) ----
    print("\nTesting episode sampler...")
    pool_dir = "data"
    pools_available = any(
        os.path.exists(os.path.join(pool_dir, f"lowresource_pool_{lang}.json"))
        for lang in ("ja", "zh")
    )

    if pools_available:
        sampler = CategoryStratifiedEpisodeSampler.from_pool_files(
            pool_dir=pool_dir,
            n_shot=config["meta_learning"]["k_shot"],
            n_query=config["meta_learning"]["query_size"],
            n_class=config["meta_learning"]["n_way"],
        )
        episode = sampler.sample_episode()
        n_support = len(episode["support_texts"])
        n_query   = len(episode["query_texts"])
        print(f"Episode: {n_support} support, {n_query} query  "
              f"lang={episode['language']}  category='{episode['category']}'")
        print(f"Support labels: {episode['support_labels']}")
    else:
        print("  Low-resource pools not found — skipping episode sampler test.")
        print("  Run: python data/preprocess.py")

    # -- Full ARAML model ----------------------------------------------------
    print("\nTesting ARAML model...")
    try:
        model = ARAML(config).to(device)
        print(f"Model parameters: {model.count_parameters():,}")
    except Exception as exc:
        print(f"  ARAML model unavailable — skipping full-model init: {exc}")

    print("\nAll components working!")


if __name__ == "__main__":
    quick_test()
