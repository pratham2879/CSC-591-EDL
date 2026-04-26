"""
regression_cli.py — Interactive terminal CLI for ARAML regression inference.

Usage:
    PYTHONPATH=. python scripts/regression_cli.py
    PYTHONPATH=. python scripts/regression_cli.py --language zh
    PYTHONPATH=. python scripts/regression_cli.py --query "この商品はかなり良いです。"
    PYTHONPATH=. python scripts/regression_cli.py --support-file my_support.json

Support file format:
[
  {"text": "example 1", "label": 0.0},
  {"text": "example 2", "label": 0.25},
  {"text": "example 3", "label": 0.5},
  {"text": "example 4", "label": 0.75},
  {"text": "example 5", "label": 1.0}
]
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.araml import ARAML
from models.meta_learner import inner_loop
from models.retrieval_index import CrossLingualRetrievalIndex


DEFAULT_SUPPORT = {
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
}


def normalize_label(value: float) -> float:
    """
    Accept either normalized labels in [0, 1] or star ratings in [1, 5].
    """
    value = float(value)
    if 0.0 <= value <= 1.0:
        return value
    if 1.0 <= value <= 5.0:
        return (value - 1.0) / 4.0
    raise ValueError(f"Unsupported label value: {value}")


def predicted_stars(score: float) -> float:
    return 1.0 + 4.0 * score


def load_support_examples(language: str, support_file: str | None) -> list[tuple[str, float]]:
    if support_file is None:
        return DEFAULT_SUPPORT[language]

    with open(support_file, encoding="utf-8") as f:
        raw = json.load(f)

    examples = []
    for item in raw:
        text = item["text"].strip()
        label = normalize_label(item["label"])
        examples.append((text, label))

    if len(examples) < 2:
        raise ValueError("Support file must contain at least 2 examples.")
    return examples


def load_runtime(config_path: str, checkpoint: str, index_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ARAML(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}")

    index = CrossLingualRetrievalIndex(
        embedding_dim=config["model"]["hidden_dim"],
        similarity=config["retrieval"]["similarity"],
    )
    index.load(index_path)
    print(f"Loaded retrieval index: {len(index)} entries")
    return model, index, config, device


def run_prediction(model, retrieval_index, config, device, support_examples, query_text: str) -> dict:
    encoder, arc, meta_learner = model.get_components()
    support_texts = [text for text, _ in support_examples]
    support_labels = torch.tensor(
        [label for _, label in support_examples], dtype=torch.float32, device=device
    )

    with torch.no_grad():
        support_embs = encoder.encode_text(support_texts, device)
    task_emb = support_embs.mean(0, keepdim=True)

    with torch.no_grad():
        arc_query = arc.generate_query(task_emb)
        budget_ratio = arc.budget_predictor(task_emb).item()
        k_predicted = max(1, int(budget_ratio * config["retrieval"]["max_retrieved"]))

    retrieved = retrieval_index.retrieve(
        arc_query.detach().float().cpu().numpy(), k=k_predicted
    )

    with torch.no_grad():
        ret_embs = encoder.encode_text(retrieved["texts"], device)
        attn_weights = arc.compute_attention_weights(arc_query, ret_embs)
        weighted_ret = (attn_weights.unsqueeze(-1) * ret_embs).sum(0)

    support_aug = (
        F.normalize(support_embs, p=2, dim=-1)
        + F.normalize(weighted_ret, p=2, dim=-1).unsqueeze(0)
    ).float()

    adapted = inner_loop(
        meta_learner.regressor,
        support_aug,
        support_labels,
        inner_lr=config["meta_learning"]["inner_lr"],
        inner_steps=config["meta_learning"]["inner_steps"],
    )

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
    return {
        "prediction": pred_clamped,
        "predicted_stars": predicted_stars(pred_clamped),
        "raw_prediction": pred,
        "support_mae": support_mae,
        "budget_ratio": budget_ratio,
        "k_predicted": k_predicted,
        "retrieved": retrieved,
        "attention_weights": attn_weights.tolist(),
    }


def print_support_examples(support_examples) -> None:
    print("\nSupport set:")
    for idx, (text, label) in enumerate(support_examples, start=1):
        print(f"  [{idx}] label={label:.2f} (~{predicted_stars(label):.1f} stars)  {text}")


def print_prediction(result: dict) -> None:
    print("\nPrediction")
    print(f"  normalized score : {result['prediction']:.4f}")
    print(f"  approx stars     : {result['predicted_stars']:.2f} / 5")
    print(f"  raw score        : {result['raw_prediction']:.4f}")
    print(f"  support MAE      : {result['support_mae']:.4f}")
    print(f"  retrieval budget : ratio={result['budget_ratio']:.4f}  k={result['k_predicted']}")
    print("\nTop retrieved examples:")
    for i, text in enumerate(result["retrieved"]["texts"][:5]):
        lang = result["retrieved"]["languages"][i]
        label = result["retrieved"]["labels"][i]
        weight = result["attention_weights"][i]
        print(f"  [{i+1}] lang={lang} label={label:.2f} attn={weight:.4f}")
        print(f"      {text[:120]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ARAML regression CLI")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="results/best_model.pt")
    parser.add_argument("--index_path", default="results/retrieval_index")
    parser.add_argument("--language", choices=("ja", "zh"), default="ja")
    parser.add_argument("--support_file", default=None)
    parser.add_argument("--query", default=None, help="Run a single query and exit.")
    args = parser.parse_args()

    support_examples = load_support_examples(args.language, args.support_file)
    model, retrieval_index, config, device = load_runtime(
        args.config, args.checkpoint, args.index_path
    )
    print_support_examples(support_examples)

    if args.query:
        result = run_prediction(model, retrieval_index, config, device, support_examples, args.query)
        print(f"\nQuery: {args.query}")
        print_prediction(result)
        return

    print("\nEnter text to score. Type 'exit' or press Enter on a blank line to quit.")
    while True:
        query = input("\nQuery> ").strip()
        if not query or query.lower() == "exit":
            break
        result = run_prediction(model, retrieval_index, config, device, support_examples, query)
        print_prediction(result)


if __name__ == "__main__":
    main()
