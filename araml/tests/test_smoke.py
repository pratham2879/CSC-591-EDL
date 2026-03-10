"""
test_smoke.py — Quick smoke test: model instantiation, forward pass, episode sampling.
Uses synthetic data (no downloads required).
"""
import os
import sys
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.encoder import TextEncoder
from models.arc import AdaptiveRetrievalController
from models.meta_learner import MetaLearner, maml_inner_loop
from models.araml import ARAML
from models.retrieval_index import CrossLingualRetrievalIndex
from utils.episode_sampler import EpisodeSampler
from utils.metrics import evaluate_few_shot, aggregate_episode_results


def test_encoder():
    print("1. Testing TextEncoder ...")
    enc = TextEncoder("xlm-roberta-base", 768)
    texts = ["Hello world", "Bonjour le monde", "Hallo Welt"]
    device = torch.device("cpu")
    embs = enc.encode_text(texts, device)
    assert embs.shape == (3, 768), f"Expected (3,768), got {embs.shape}"
    print(f"   OK — output shape: {embs.shape}")
    return enc


def test_arc():
    print("2. Testing AdaptiveRetrievalController ...")
    arc = AdaptiveRetrievalController(input_dim=768, hidden_dim=256, max_k=10)
    task_emb = torch.randn(1, 768)
    retrieved_embs = torch.randn(5, 768)

    query = arc.generate_query(task_emb)
    budget = arc.predict_budget(task_emb)
    weights = arc.compute_attention_weights(task_emb, retrieved_embs)
    _, _, weighted_ret, _ = arc(task_emb, retrieved_embs)

    assert query.shape == (1, 768)
    assert 1 <= budget <= 10
    assert weights.shape == (5,)
    assert abs(weights.sum().item() - 1.0) < 1e-5
    assert weighted_ret.shape == (768,)
    print(f"   OK — query:{query.shape}, budget:{budget}, weights:{weights.shape}")
    return arc


def test_meta_learner():
    print("3. Testing MetaLearner ...")
    ml = MetaLearner(input_dim=768, num_classes=3)
    x = torch.randn(15, 768 * 2)  # [emb || retrieval_emb]
    logits = ml(x)
    assert logits.shape == (15, 3)
    print(f"   OK — logits shape: {logits.shape}")
    return ml


def test_retrieval_index():
    print("4. Testing CrossLingualRetrievalIndex ...")
    idx = CrossLingualRetrievalIndex(embedding_dim=768, similarity="cosine")
    fake_embs = np.random.randn(100, 768).astype(np.float32)
    fake_texts = [f"text_{i}" for i in range(100)]
    fake_labels = [i % 3 for i in range(100)]
    idx.add(fake_embs, fake_texts, fake_labels, "en")

    query = np.random.randn(1, 768).astype(np.float32)
    result = idx.retrieve(query, k=5)
    assert len(result["texts"]) == 5
    assert len(result["labels"]) == 5
    print(f"   OK — retrieved {len(result['texts'])} items, index size: {len(idx)}")
    return idx


def test_episode_sampler():
    print("5. Testing EpisodeSampler ...")
    fake_data = [
        {"text": f"text_{i}", "label": i % 3, "language": "en"}
        for i in range(300)
    ]
    sampler = EpisodeSampler(fake_data, n_way=3, k_shot=5, query_size=10)
    ep = sampler.sample_episode()
    assert len(ep["support_texts"]) == 15   # 3 * 5
    assert len(ep["query_texts"]) == 30     # 3 * 10
    assert set(ep["support_labels"]) == {0, 1, 2}
    print(f"   OK — support:{len(ep['support_texts'])}, query:{len(ep['query_texts'])}")
    return sampler


def test_araml_model():
    print("6. Testing full ARAML model instantiation ...")
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = ARAML(config)
    enc, arc, ml = model.get_components()
    n_params = model.count_parameters()
    print(f"   OK — total trainable parameters: {n_params:,}")
    return model


def test_maml_inner_loop():
    print("7. Testing MAML inner loop ...")
    ml = MetaLearner(input_dim=768, num_classes=3)
    support = torch.randn(15, 768 * 2)
    labels = torch.tensor([0]*5 + [1]*5 + [2]*5)
    adapted = maml_inner_loop(ml, support, labels, inner_lr=0.01, inner_steps=2, device=torch.device("cpu"))
    logits = adapted(support)
    assert logits.shape == (15, 3)
    print(f"   OK — adapted model logits: {logits.shape}")


def test_metrics():
    print("8. Testing metrics ...")
    preds = [0, 1, 2, 0, 1]
    targets = [0, 1, 1, 0, 2]
    result = evaluate_few_shot(preds, targets)
    assert "accuracy" in result
    assert "f1_macro" in result

    accs = [0.6, 0.7, 0.65, 0.72, 0.68, 0.71, 0.66, 0.69, 0.73, 0.67]
    agg = aggregate_episode_results(accs)
    assert "mean_accuracy" in agg
    assert "95ci" in agg
    print(f"   OK — accuracy={result['accuracy']:.2f}, "
          f"episode mean={agg['mean_accuracy']:.4f} +/- {agg['95ci']:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("ARAML Smoke Test")
    print("=" * 60)

    test_encoder()
    test_arc()
    test_meta_learner()
    test_retrieval_index()
    test_episode_sampler()
    test_araml_model()
    test_maml_inner_loop()
    test_metrics()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
