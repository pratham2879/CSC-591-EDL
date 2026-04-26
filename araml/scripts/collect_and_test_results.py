"""
collect_and_test_results.py — Collect training results and test on custom data

This script:
1. Parses training log for metrics
2. Tests model on custom queries
3. Generates a comprehensive report
"""
import sys
import os
import json
import re
import torch
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.araml import ARAML
from models.meta_learner import inner_loop
from models.retrieval_index import CrossLingualRetrievalIndex
import torch.nn.functional as F


def parse_training_log(log_file):
    """Parse training log to extract metrics."""
    if not os.path.exists(log_file):
        return {"status": "Log file not found", "epochs": []}
    
    with open(log_file) as f:
        content = f.read()
    
    epochs = []
    
    # Extract epoch lines
    epoch_pattern = r'Epoch\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+GradNorm:\s+([\d.]+)'
    for match in re.finditer(epoch_pattern, content):
        epoch_num = int(match.group(1))
        loss = float(match.group(2))
        grad_norm = float(match.group(3))
        epochs.append({"epoch": epoch_num, "loss": loss, "grad_norm": grad_norm})
    
    # Extract validation metrics
    val_pattern = r'Validation\s+\|\s+MAE:\s+([\d.]+)\s+\+/-\s+([\d.]+)\s+\|\s+RMSE:\s+([\d.]+)\s+\+/-\s+([\d.]+)\s+\|\s+Corr:\s+([\d.\-]+)'
    val_metrics = []
    for match in re.finditer(val_pattern, content):
        val_metrics.append({
            "mae": float(match.group(1)),
            "mae_ci": float(match.group(2)),
            "rmse": float(match.group(3)),
            "rmse_ci": float(match.group(4)),
            "correlation": float(match.group(5)),
        })
    
    return {
        "num_epochs": len(epochs),
        "epochs": epochs,
        "validation_metrics": val_metrics,
        "training_complete": "Saved best model" in content,
    }


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


def test_on_custom_data(model, index, config, device, language="ja"):
    """Test model on predefined custom data."""
    
    # Define test cases
    test_cases = {
        "ja": [
            ("この商品は素晴らしいです。本当に満足しています。", "positive"),
            ("ひどい品質で、すぐに壊れてしまいました。", "negative"),
            ("普通の商品ですね。特に文句はありませんが、特に良くもない。", "neutral"),
            ("期待以上でした。とても満足です。", "positive"),
            ("値段の割に質が悪い。がっかりしました。", "negative"),
        ],
        "zh": [
            ("这个产品太好了！我非常满意。", "positive"),
            ("质量很差，立即坏了。", "negative"),
            ("一般般的产品。没有什么特别的。", "neutral"),
            ("超出预期，很满意。", "positive"),
            ("太失望了，质量很差。", "negative"),
        ],
        "en": [
            ("This product is amazing! Very satisfied.", "positive"),
            ("Terrible quality, broke immediately.", "negative"),
            ("Average product, nothing special.", "neutral"),
            ("Exceeded expectations, very happy.", "positive"),
            ("Very disappointed, poor quality.", "negative"),
        ]
    }
    
    # Default support set
    support_sets = {
        "ja": [
            ("ひどい品質で、すぐ壊れました。", 0.00),
            ("期待以下で、あまり満足できません。", 0.25),
            ("普通です。悪くも良くもありません。", 0.50),
            ("使いやすくて満足しています。", 0.75),
            ("最高です。買って本当に良かったです。", 1.00),
        ],
        "zh": [
            ("质量很差，很快就坏了。", 0.00),
            ("没有达到预期，但还能用。", 0.25),
            ("一般般，没有特别好也没有特别差。", 0.50),
            ("挺好用的，我比较满意。", 0.75),
            ("非常优秀，完全超出预期。", 1.00),
        ],
        "en": [
            ("Terrible quality, broke immediately.", 0.00),
            ("Below expectations, not satisfied.", 0.25),
            ("Average product, nothing special.", 0.50),
            ("Very satisfied with this product.", 0.75),
            ("Excellent! Exceeded my expectations.", 1.00),
        ]
    }
    
    encoder, arc, meta_learner = model.encoder, model.arc, model.meta_learner
    support_data = support_sets[language]
    support_texts = [text for text, _ in support_data]
    support_labels = torch.tensor(
        [label for _, label in support_data], dtype=torch.float32, device=device
    )
    
    results = []
    
    for query_text, expected_sentiment in test_cases[language]:
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
        
        # Attention weights
        with torch.no_grad():
            ret_embs = encoder.encode_text(retrieved["texts"], device)
            attn_weights = arc.compute_attention_weights(arc_query, ret_embs)
            weighted_ret = (attn_weights.unsqueeze(-1) * ret_embs).sum(0)
        
        # Augment support
        support_aug = (
            F.normalize(support_embs, p=2, dim=-1)
            + F.normalize(weighted_ret, p=2, dim=-1).unsqueeze(0)
        ).float()
        
        # Inner loop
        adapted = inner_loop(
            meta_learner.regressor,
            support_aug,
            support_labels,
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
            support_mae = torch.mean(torch.abs(support_fit - support_labels)).item()
        
        pred_clamped = max(0.0, min(1.0, pred))
        
        results.append({
            "query": query_text,
            "expected": expected_sentiment,
            "prediction_score": pred_clamped,
            "prediction_stars": predicted_stars(pred_clamped),
            "support_mae": support_mae,
        })
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect results and test model")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="results/best_model.pt")
    parser.add_argument("--index_path", default="results/retrieval_index")
    parser.add_argument("--language", choices=("ja", "zh", "en"), default="ja")
    args = parser.parse_args()
    
    print("=" * 80)
    print("ARAML REGRESSION MODEL — Results Collection & Testing")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Parse training log
    print("Parsing training log...")
    log_stats = parse_training_log("training_log.txt")
    print(f"✅ Epochs completed: {log_stats['num_epochs']}")
    print(f"   Training complete: {log_stats['training_complete']}\n")
    
    if log_stats["num_epochs"] > 0:
        print("Training Metrics Summary:")
        print("-" * 80)
        last_epoch = log_stats["epochs"][-1]
        print(f"Last Epoch: {last_epoch['epoch']}")
        print(f"  Loss: {last_epoch['loss']:.6f}")
        print(f"  Grad Norm: {last_epoch['grad_norm']:.6f}")
        
        if log_stats["validation_metrics"]:
            last_val = log_stats["validation_metrics"][-1]
            print(f"  Val MAE: {last_val['mae']:.6f} ± {last_val['mae_ci']:.6f}")
            print(f"  Val RMSE: {last_val['rmse']:.6f} ± {last_val['rmse_ci']:.6f}")
            print(f"  Correlation: {last_val['correlation']:.6f}")
        print()
    
    # Load model
    device = torch.device("cpu")
    print("Loading model and index...")
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    model = ARAML(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    index = CrossLingualRetrievalIndex(
        embedding_dim=config["model"]["hidden_dim"],
        similarity=config["retrieval"]["similarity"],
    )
    index.load(args.index_path)
    print(f"✅ Model loaded from {args.checkpoint}")
    print(f"✅ Index loaded: {len(index)} examples\n")
    
    # Test on custom data
    print("Testing on custom queries...")
    print("-" * 80)
    test_results = test_on_custom_data(model, index, config, device, args.language)
    
    for i, result in enumerate(test_results, 1):
        print(f"\n[Test {i}] {result['expected'].upper()}")
        print(f"  Query: {result['query']}")
        print(f"  Prediction: {result['prediction_score']:.4f} ({result['prediction_stars']:.2f}★)")
        print(f"  Support MAE: {result['support_mae']:.6f}")
    
    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "language": args.language,
        "training": log_stats,
        "model_path": args.checkpoint,
        "index_size": len(index),
        "test_results": test_results,
        "summary": {
            "avg_prediction": sum(r["prediction_score"] for r in test_results) / len(test_results),
            "avg_stars": sum(r["prediction_stars"] for r in test_results) / len(test_results),
            "avg_support_mae": sum(r["support_mae"] for r in test_results) / len(test_results),
            "test_count": len(test_results),
        }
    }
    
    # Save report
    report_file = f"results/test_report_{args.language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("results", exist_ok=True)
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Average Prediction: {report['summary']['avg_prediction']:.4f}")
    print(f"Average Stars: {report['summary']['avg_stars']:.2f}★")
    print(f"Average Support MAE: {report['summary']['avg_support_mae']:.6f}")
    print(f"\n✅ Report saved to: {report_file}\n")


if __name__ == "__main__":
    main()
