"""
test_custom_data.py — Test the trained regression model on custom data

This script tests the model on user-provided sentiments and shows predictions.
"""
import sys
import os
import json
import torch
import yaml
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.araml import ARAML
from models.meta_learner import inner_loop
from models.retrieval_index import CrossLingualRetrievalIndex
import torch.nn.functional as F


def normalize_label(value: float) -> float:
    """Accept normalized labels [0, 1] or star ratings [1, 5]."""
    value = float(value)
    if 0.0 <= value <= 1.0:
        return value
    if 1.0 <= value <= 5.0:
        return (value - 1.0) / 4.0
    raise ValueError(f"Unsupported label value: {value}")


def predicted_stars(score: float) -> float:
    """Convert normalized score [0, 1] to star rating [1, 5]."""
    return 1.0 + 4.0 * score


def load_model_and_index(config_path, checkpoint_path, index_path, device):
    """Load model, index, and config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model = ARAML(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    index = CrossLingualRetrievalIndex(
        embedding_dim=config["model"]["hidden_dim"],
        similarity=config["retrieval"]["similarity"],
    )
    index.load(index_path)
    
    return model, index, config, device


def predict_sentiment(model, index, config, device, support_texts, support_labels, query_text):
    """Predict sentiment for a query given support examples."""
    encoder, arc, meta_learner = (
        model.encoder,
        model.arc,
        model.meta_learner,
    )
    
    support_labels_tensor = torch.tensor(
        support_labels, dtype=torch.float32, device=device
    )
    
    # Encode support
    with torch.no_grad():
        support_embs = encoder.encode_text(support_texts, device)
    
    task_emb = support_embs.mean(0, keepdim=True)
    
    # ARC retrieval
    with torch.no_grad():
        arc_query = arc.generate_query(task_emb)
        budget_ratio = arc.budget_predictor(task_emb).item()
        k_predicted = max(1, int(budget_ratio * config["retrieval"]["max_retrieved"]))
    
    retrieved = index.retrieve(
        arc_query.detach().float().cpu().numpy(), k=k_predicted
    )
    
    # Compute attention weights
    with torch.no_grad():
        ret_embs = encoder.encode_text(retrieved["texts"], device)
        attn_weights = arc.compute_attention_weights(arc_query, ret_embs)
        weighted_ret = (attn_weights.unsqueeze(-1) * ret_embs).sum(0)
    
    # Augment support
    support_aug = (
        F.normalize(support_embs, p=2, dim=-1)
        + F.normalize(weighted_ret, p=2, dim=-1).unsqueeze(0)
    ).float()
    
    # Inner loop adaptation
    adapted = inner_loop(
        meta_learner.regressor,
        support_aug,
        support_labels_tensor,
        inner_lr=config["meta_learning"]["inner_lr"],
        inner_steps=config["meta_learning"]["inner_steps"],
    )
    
    # Query prediction
    with torch.no_grad():
        query_emb = encoder.encode_text([query_text], device)
        query_aug = (
            F.normalize(query_emb, p=2, dim=-1)
            + F.normalize(weighted_ret, p=2, dim=-1).unsqueeze(0)
        ).float()
        pred = F.linear(query_aug, adapted["weight"], adapted["bias"]).squeeze().item()
        support_fit = F.linear(support_aug, adapted["weight"], adapted["bias"]).squeeze(-1)
        support_mae = torch.mean(torch.abs(support_fit - support_labels_tensor)).item()
    
    pred_clamped = max(0.0, min(1.0, pred))
    
    return {
        "query": query_text,
        "prediction": pred_clamped,
        "predicted_stars": predicted_stars(pred_clamped),
        "support_mae": support_mae,
        "budget_ratio": budget_ratio,
        "k_predicted": k_predicted,
        "top_retrieved": retrieved["texts"][:3],
    }


def main():
    parser = argparse.ArgumentParser(description="Test regression model on custom data")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="results/best_model.pt")
    parser.add_argument("--index_path", default="results/retrieval_index")
    parser.add_argument("--language", choices=("ja", "zh", "en"), default="ja")
    parser.add_argument("--support_file", default=None, help="JSON file with support examples")
    parser.add_argument("--test_file", default=None, help="JSON file with queries to test")
    args = parser.parse_args()
    
    device = torch.device("cpu")
    print(f"Device: {device}\n")
    
    # Load model
    print("Loading model...")
    model, index, config, device = load_model_and_index(
        args.config, args.checkpoint, args.index_path, device
    )
    print(f"✅ Model loaded from {args.checkpoint}")
    print(f"✅ Retrieval index: {len(index)} examples\n")
    
    # Load support examples
    if args.support_file:
        with open(args.support_file) as f:
            support_data = json.load(f)
        support_texts = [item["text"] for item in support_data]
        support_labels = [normalize_label(item["label"]) for item in support_data]
        print(f"✅ Loaded {len(support_texts)} support examples from {args.support_file}\n")
    else:
        # Default support
        if args.language == "ja":
            support_data = [
                ("ひどい品質で、すぐ壊れました。", 0.00),
                ("期待以下で、あまり満足できません。", 0.25),
                ("普通です。悪くも良くもありません。", 0.50),
                ("使いやすくて満足しています。", 0.75),
                ("最高です。買って本当に良かったです。", 1.00),
            ]
        elif args.language == "zh":
            support_data = [
                ("质量很差，很快就坏了。", 0.00),
                ("没有达到预期，但还能用。", 0.25),
                ("一般般，没有特别好也没有特别差。", 0.50),
                ("挺好用的，我比较满意。", 0.75),
                ("非常优秀，完全超出预期。", 1.00),
            ]
        else:  # en
            support_data = [
                ("Terrible quality, broke immediately.", 0.00),
                ("Below expectations, not satisfied.", 0.25),
                ("Average product, nothing special.", 0.50),
                ("Very satisfied with this product.", 0.75),
                ("Excellent! Exceeded my expectations.", 1.00),
            ]
        
        support_texts = [text for text, _ in support_data]
        support_labels = [label for _, label in support_data]
        print(f"✅ Using default {args.language} support set\n")
    
    # Test on queries
    if args.test_file:
        print(f"Testing on queries from {args.test_file}...\n")
        with open(args.test_file) as f:
            test_data = json.load(f)
        
        if isinstance(test_data, dict):
            queries = test_data.get("queries", [])
            if not queries:
                queries = [test_data.get("query", "")]
        else:
            queries = test_data
        
        results = []
        for query in queries:
            result = predict_sentiment(
                model, index, config, device,
                support_texts, support_labels, query
            )
            results.append(result)
            
            print("=" * 80)
            print(f"Query: {result['query']}")
            print(f"Prediction: {result['prediction']:.4f} ({result['predicted_stars']:.2f}★ / 5)")
            print(f"Support MAE: {result['support_mae']:.4f}")
            print(f"Budget: {result['k_predicted']} examples (ratio={result['budget_ratio']:.4f})")
            print(f"Top retrieved examples:")
            for i, text in enumerate(result['top_retrieved'], 1):
                print(f"  [{i}] {text[:70]}")
            print()
        
        # Save results
        output_file = f"test_results_{args.language}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results saved to {output_file}")
        
    else:
        # Interactive mode
        print("Enter queries to test (type 'exit' to quit):\n")
        while True:
            query = input("Query> ").strip()
            if query.lower() == "exit" or not query:
                break
            
            result = predict_sentiment(
                model, index, config, device,
                support_texts, support_labels, query
            )
            
            print("\n" + "=" * 80)
            print(f"Prediction: {result['prediction']:.4f} ({result['predicted_stars']:.2f}★ / 5)")
            print(f"Support MAE: {result['support_mae']:.4f}")
            print(f"Budget: {result['k_predicted']} examples (ratio={result['budget_ratio']:.4f})")
            print(f"Top retrieved examples:")
            for i, text in enumerate(result['top_retrieved'], 1):
                print(f"  [{i}] {text[:70]}")
            print()


if __name__ == "__main__":
    main()
