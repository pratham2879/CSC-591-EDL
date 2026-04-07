"""
quick_test.py — Quick test with minimal data
"""
import os
import json
import yaml
import torch
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder import TextEncoder
from models.retrieval_index import CrossLingualRetrievalIndex
from models.araml import ARAML
from utils.episode_sampler import EpisodeSampler

def quick_test():
    print("Loading config...")
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    
    # Load small subset of data
    print("\nLoading data...")
    with open("data/processed/amazon_en.json", encoding="utf-8") as f:
        all_data = json.load(f)
    
    # Get balanced subset with both labels
    en_data = []
    for label in [0, 1]:
        label_data = [r for r in all_data if r["label"] == label][:50]
        en_data.extend(label_data)
    
    print(f"Loaded {len(en_data)} English samples")
    
    # Build small index
    print("\nBuilding retrieval index...")
    encoder = TextEncoder(model_name=config["model"]["encoder"]).to(device)
    encoder.eval()
    
    index = CrossLingualRetrievalIndex(
        embedding_dim=config["model"]["hidden_dim"],
        similarity=config["retrieval"]["similarity"]
    )
    
    texts = [r["text"] for r in en_data]
    labels = [r["label"] for r in en_data]
    
    with torch.no_grad():
        embs = encoder.encode_text(texts, device)
        index.add(embs.cpu().numpy(), texts, labels, "en")
    
    print(f"Index size: {len(index)}")
    
    # Test retrieval
    print("\nTesting retrieval...")
    query_emb = embs[0:1].cpu().numpy()
    retrieved = index.retrieve(query_emb, k=5)
    print(f"Retrieved {len(retrieved['texts'])} examples")
    
    # Test episode sampling
    print("\nTesting episode sampler...")
    sampler = EpisodeSampler(en_data, n_way=2, k_shot=5, query_size=10)
    episode = sampler.sample_episode()
    print(f"Episode: {len(episode['support_texts'])} support, {len(episode['query_texts'])} query")
    
    # Test model
    print("\nTesting ARAML model...")
    model = ARAML(config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    print("\nAll components working!")

if __name__ == "__main__":
    quick_test()
