"""
interactive_cli.py — Enhanced interactive CLI for ARAML regression with testing and retraining.

Usage:
    PYTHONPATH=. python scripts/interactive_cli.py
    PYTHONPATH=. python scripts/interactive_cli.py --language zh
    PYTHONPATH=. python scripts/interactive_cli.py --checkpoint results/best_model.pt

Features:
    - Interactive query testing
    - Model statistics and performance metrics
    - Fine-tuning on custom support examples
    - Batch inference on multiple queries
    - Model state persistence
    - Retraining with updated data

Commands:
    test [query]         - Test model with a single query (or interactive mode)
    retrain [steps]      - Fine-tune model on current support set for N steps
    metrics              - Show current model performance stats
    support [file.json]  - Load custom support examples
    batch [file.json]    - Run batch inference on queries from file
    info                 - Show model and config information
    save [checkpoint]    - Save current model state
    load [checkpoint]    - Load model checkpoint
    help                 - Show this help message
    exit                 - Exit the CLI
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.araml import ARAML
from models.meta_learner import inner_loop
from models.retrieval_index import CrossLingualRetrievalIndex


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PredictionResult:
    """Container for a single prediction result."""
    prediction: float
    predicted_stars: float
    raw_prediction: float
    support_mae: float
    budget_ratio: float
    k_predicted: int
    retrieved_texts: List[str]
    retrieved_labels: List[float]
    retrieved_langs: List[str]
    attention_weights: List[float]


# ============================================================================
# Support Examples Management
# ============================================================================

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


def load_support_examples(
    language: str, support_file: Optional[str] = None
) -> List[Tuple[str, float]]:
    """Load support examples from default set or custom file."""
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


# ============================================================================
# Model Management
# ============================================================================

class ARMLInteractiveCLI:
    """Enhanced interactive CLI for ARAML regression model."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        index_path: str,
        language: str = "ja",
        support_file: Optional[str] = None,
    ):
        """Initialize CLI with model and runtime."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        print(f"Config loaded from: {config_path}")

        # Load model
        self.model = ARAML(self.config).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        print(f"Checkpoint loaded: {checkpoint_path}")

        # Load retrieval index
        self.index = CrossLingualRetrievalIndex(
            embedding_dim=self.config["model"]["hidden_dim"],
            similarity=self.config["retrieval"]["similarity"],
        )
        self.index.load(index_path)
        print(f"Retrieval index loaded: {len(self.index)} entries")

        # Load support examples
        self.language = language
        self.support_examples = load_support_examples(language, support_file)
        print(f"Language: {language}")
        print(f"Support examples: {len(self.support_examples)}")

        # Stats tracking
        self.prediction_history: List[Dict] = []
        self.eval_metrics = {}

    def get_encoder(self):
        """Get text encoder component."""
        return self.model.encoder

    def get_arc(self):
        """Get ARC component."""
        return self.model.arc

    def get_meta_learner(self):
        """Get meta-learner component."""
        return self.model.meta_learner

    def predict(self, query_text: str) -> PredictionResult:
        """
        Generate prediction for a single query text.
        Uses current support examples and retrieval index.
        """
        encoder, arc, meta_learner = (
            self.get_encoder(),
            self.get_arc(),
            self.get_meta_learner(),
        )

        # Extract support texts and labels
        support_texts = [text for text, _ in self.support_examples]
        support_labels = torch.tensor(
            [label for _, label in self.support_examples],
            dtype=torch.float32,
            device=self.device,
        )

        # Encode support examples
        with torch.no_grad():
            support_embs = encoder.encode_text(support_texts, self.device)

        # Generate task embedding
        task_emb = support_embs.mean(0, keepdim=True)

        # ARC: Retrieve and weight examples
        with torch.no_grad():
            arc_query = arc.generate_query(task_emb)
            budget_ratio = arc.budget_predictor(task_emb).item()
            k_predicted = max(1, int(budget_ratio * self.config["retrieval"]["max_retrieved"]))

        # Retrieve examples from index
        retrieved = self.index.retrieve(
            arc_query.detach().float().cpu().numpy(), k=k_predicted
        )

        # Compute attention weights
        with torch.no_grad():
            ret_embs = encoder.encode_text(retrieved["texts"], self.device)
            attn_weights = arc.compute_attention_weights(arc_query, ret_embs)
            weighted_ret = (attn_weights.unsqueeze(-1) * ret_embs).sum(0)

        # Augment support examples with retrieval
        support_aug = (
            F.normalize(support_embs, p=2, dim=-1)
            + F.normalize(weighted_ret, p=2, dim=-1).unsqueeze(0)
        ).float()

        # Inner loop: Adapt meta-learner to support set
        adapted = inner_loop(
            meta_learner.regressor,
            support_aug,
            support_labels,
            inner_lr=self.config["meta_learning"]["inner_lr"],
            inner_steps=self.config["meta_learning"]["inner_steps"],
        )

        # Query prediction
        with torch.no_grad():
            query_emb = encoder.encode_text([query_text], self.device)
            query_aug = (
                F.normalize(query_emb, p=2, dim=-1)
                + F.normalize(weighted_ret, p=2, dim=-1).unsqueeze(0)
            ).float()
            pred = F.linear(query_aug, adapted["weight"], adapted["bias"]).squeeze().item()
            support_fit = F.linear(support_aug, adapted["weight"], adapted["bias"]).squeeze(-1)
            support_mae = torch.mean(torch.abs(support_fit - support_labels)).item()

        pred_clamped = max(0.0, min(1.0, pred))

        return PredictionResult(
            prediction=pred_clamped,
            predicted_stars=predicted_stars(pred_clamped),
            raw_prediction=pred,
            support_mae=support_mae,
            budget_ratio=budget_ratio,
            k_predicted=k_predicted,
            retrieved_texts=retrieved["texts"],
            retrieved_labels=retrieved["labels"],
            retrieved_langs=retrieved["languages"],
            attention_weights=attn_weights.tolist(),
        )

    def print_support_examples(self) -> None:
        """Display current support examples."""
        print("\nSupport Set:")
        print("-" * 80)
        for idx, (text, label) in enumerate(self.support_examples, start=1):
            stars = predicted_stars(label)
            print(f"  [{idx}] label={label:.2f} ({stars:.1f}★)  {text[:60]}")
        print()

    def print_prediction(self, result: PredictionResult, query_text: str) -> None:
        """Display prediction result with detailed info."""
        print("\n" + "=" * 80)
        print(f"Query: {query_text}")
        print("=" * 80)
        print(f"Prediction:       {result.prediction:.4f} ({result.predicted_stars:.2f}★ / 5)")
        print(f"Raw Score:        {result.raw_prediction:.4f}")
        print(f"Support MAE:      {result.support_mae:.4f}")
        print(f"Budget:           {result.k_predicted} examples (ratio={result.budget_ratio:.4f})")
        print("\nTop 5 Retrieved Examples:")
        print("-" * 80)
        for i, (text, label, lang, weight) in enumerate(
            zip(
                result.retrieved_texts[:5],
                result.retrieved_labels[:5],
                result.retrieved_langs[:5],
                result.attention_weights[:5],
            ),
            start=1,
        ):
            stars = predicted_stars(label)
            print(f"  [{i}] {lang} label={label:.2f} ({stars:.1f}★) attn={weight:.4f}")
            print(f"      {text[:70]}")
        print()

    def finetune_support_set(self, steps: int = 3, lr: float = 0.01) -> Dict:
        """
        Fine-tune meta-learner on current support set for a few steps.
        Returns adaptation metrics.
        """
        encoder, arc, meta_learner = (
            self.get_encoder(),
            self.get_arc(),
            self.get_meta_learner(),
        )

        support_texts = [text for text, _ in self.support_examples]
        support_labels = torch.tensor(
            [label for _, label in self.support_examples],
            dtype=torch.float32,
            device=self.device,
        )

        # Encode support
        with torch.no_grad():
            support_embs = encoder.encode_text(support_texts, self.device)

        task_emb = support_embs.mean(0, keepdim=True)

        # Retrieve augmentation
        with torch.no_grad():
            arc_query = arc.generate_query(task_emb)
            budget_ratio = arc.budget_predictor(task_emb).item()
            k_predicted = max(1, int(budget_ratio * self.config["retrieval"]["max_retrieved"]))

        retrieved = self.index.retrieve(
            arc_query.detach().float().cpu().numpy(), k=k_predicted
        )

        with torch.no_grad():
            ret_embs = encoder.encode_text(retrieved["texts"], self.device)
            attn_weights = arc.compute_attention_weights(arc_query, ret_embs)
            weighted_ret = (attn_weights.unsqueeze(-1) * ret_embs).sum(0)

        support_aug = (
            F.normalize(support_embs, p=2, dim=-1)
            + F.normalize(weighted_ret, p=2, dim=-1).unsqueeze(0)
        ).float()

        # Optimize regressor on support set
        regressor_copy = torch.nn.Linear(
            meta_learner.regressor.in_features, meta_learner.regressor.out_features
        ).to(self.device)
        regressor_copy.load_state_dict(meta_learner.regressor.state_dict())

        optimizer = optim.SGD(regressor_copy.parameters(), lr=lr)
        losses = []

        for step in range(steps):
            optimizer.zero_grad()
            preds = F.linear(support_aug, regressor_copy.weight, regressor_copy.bias).squeeze(-1)
            loss = F.smooth_l1_loss(preds, support_labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        mae = torch.mean(torch.abs(preds - support_labels)).item()

        print(f"Fine-tuned for {steps} steps:")
        print(f"  Initial loss: {losses[0]:.6f}")
        print(f"  Final loss:   {losses[-1]:.6f}")
        print(f"  Final MAE:    {mae:.6f}")

        return {"initial_loss": losses[0], "final_loss": losses[-1], "final_mae": mae}

    def batch_predict(self, queries_file: str) -> List[PredictionResult]:
        """Run batch inference on multiple queries from a JSON file."""
        with open(queries_file, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "queries" in data:
            queries = data["queries"]
        elif isinstance(data, list):
            queries = data
        else:
            raise ValueError("Invalid format: expected list of queries or {'queries': [...]}")

        results = []
        print(f"\nRunning batch inference on {len(queries)} queries...")
        for query_text in tqdm(queries, desc="Predicting"):
            result = self.predict(query_text)
            results.append(result)

        # Print summary
        preds = [r.prediction for r in results]
        print(f"\nBatch Results Summary:")
        print(f"  Avg prediction:   {sum(preds) / len(preds):.4f}")
        print(f"  Min prediction:   {min(preds):.4f}")
        print(f"  Max prediction:   {max(preds):.4f}")
        print(f"  Avg stars:        {predicted_stars(sum(preds) / len(preds)):.2f}★")

        return results

    def save_checkpoint(self, path: str) -> None:
        """Save current model checkpoint."""
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Checkpoint loaded: {path}")

    def print_model_info(self) -> None:
        """Display model configuration and statistics."""
        encoder, arc, meta_learner = (
            self.get_encoder(),
            self.get_arc(),
            self.get_meta_learner(),
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("\n" + "=" * 80)
        print("MODEL INFORMATION")
        print("=" * 80)
        print(f"Device:                  {self.device}")
        print(f"Total parameters:        {total_params:,}")
        print(f"Trainable parameters:    {trainable_params:,}")
        print(f"\nEncoder:                 {self.config['model']['encoder']}")
        print(f"  Hidden dim:            {self.config['model']['hidden_dim']}")
        print(f"ARC Components:")
        print(f"  Max retrieved:         {self.config['retrieval']['max_retrieved']}")
        print(f"  Similarity metric:     {self.config['retrieval']['similarity']}")
        print(f"Meta-Learner (MAML):")
        print(f"  Inner LR:              {self.config['meta_learning']['inner_lr']}")
        print(f"  Inner steps:           {self.config['meta_learning']['inner_steps']}")
        print(f"  K-shot:                {self.config['meta_learning']['k_shot']}")
        print(f"  Query set size:        {self.config['meta_learning']['query_size']}")
        print(f"\nRetrieval Index:")
        print(f"  Size:                  {len(self.index):,} examples")
        print()

    def run_interactive_mode(self) -> None:
        """Run main interactive CLI loop."""
        self.print_support_examples()
        self.print_model_info()

        print("Type 'help' for available commands or 'exit' to quit.\n")

        while True:
            try:
                user_input = input(">>> ").strip()
                if not user_input:
                    continue

                self.process_command(user_input)

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

    def process_command(self, user_input: str) -> None:
        """Parse and execute user command."""
        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command == "test":
            if args:
                result = self.predict(args)
                self.print_prediction(result, args)
            else:
                print("Enter query (or 'done' to finish):")
                while True:
                    query = input("  > ").strip()
                    if query.lower() == "done":
                        break
                    if query:
                        result = self.predict(query)
                        self.print_prediction(result, query)

        elif command == "retrain":
            steps = int(args) if args else 3
            self.finetune_support_set(steps=steps)

        elif command == "metrics":
            self.print_model_info()

        elif command == "support":
            if args:
                self.support_examples = load_support_examples(self.language, args)
                print(f"Support examples loaded from: {args}")
            self.print_support_examples()

        elif command == "batch":
            if not args:
                print("Usage: batch <file.json>")
            else:
                results = self.batch_predict(args)
                for i, (result, _) in enumerate(
                    zip(results, open(args).read().count('queries'))
                ):
                    print(
                        f"{i+1}. Score: {result.prediction:.4f} ({result.predicted_stars:.2f}★)"
                    )

        elif command == "info":
            self.print_model_info()

        elif command == "save":
            path = args or "results/checkpoint_custom.pt"
            self.save_checkpoint(path)

        elif command == "load":
            if not args:
                print("Usage: load <checkpoint.pt>")
            else:
                self.load_checkpoint(args)

        elif command == "help":
            print(__doc__)

        elif command == "exit":
            raise KeyboardInterrupt

        else:
            print(f"Unknown command: {command}. Type 'help' for available commands.")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced interactive CLI for ARAML regression model"
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="results/best_model.pt")
    parser.add_argument("--index_path", default="results/retrieval_index")
    parser.add_argument("--language", choices=("ja", "zh"), default="ja")
    parser.add_argument("--support_file", default=None, help="Custom support examples JSON")
    parser.add_argument(
        "--query",
        default=None,
        help="Run single query and exit (non-interactive mode)",
    )
    args = parser.parse_args()

    cli = ARMLInteractiveCLI(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        index_path=args.index_path,
        language=args.language,
        support_file=args.support_file,
    )

    if args.query:
        result = cli.predict(args.query)
        cli.print_prediction(result, args.query)
    else:
        cli.run_interactive_mode()


if __name__ == "__main__":
    main()
