"""
baseline_maml.py — MAML-only baseline (no retrieval augmentation).
Uses the same meta-learning loop as ARAML but without retrieval.
"""
import os
import sys
import yaml
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import higher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.encoder import TextEncoder
from utils.config_utils import get_dataset_config, load_multi_language_data, load_language_data
from utils.episode_sampler import EpisodeSampler
from utils.metrics import aggregate_episode_results


class MAMLClassifier(nn.Module):
    """Simple classifier head for MAML (no retrieval concatenation)."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_maml(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ds_cfg = get_dataset_config(config)
    num_classes = ds_cfg["num_classes"]
    meta_cfg = config["meta_learning"]
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Baseline: MAML] device={device}, dataset={ds_cfg['dataset_name']}")

    encoder = TextEncoder(model_name=config["model"]["encoder"]).to(device)
    classifier = MAMLClassifier(config["model"]["hidden_dim"], num_classes).to(device)
    print(f"Encoder params (frozen last layers): {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Classifier params: {sum(p.numel() for p in classifier.parameters() if p.requires_grad):,}")

    all_records = load_multi_language_data(ds_cfg, ds_cfg["source_languages"], split="train")
    print(f"Training records: {len(all_records):,}")

    sampler = EpisodeSampler(all_records, meta_cfg["n_way"], meta_cfg["k_shot"], meta_cfg["query_size"])
    # Only optimize classifier + encoder jointly
    outer_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=meta_cfg["outer_lr"]
    )

    best_acc = 0.0
    episode_iter = iter(sampler)
    episodes_per_epoch = config["training"].get("episodes_per_epoch", 100)

    for epoch in range(config["training"]["epochs"]):
        epoch_losses, epoch_accs = [], []

        for _ in tqdm(range(episodes_per_epoch), desc=f"MAML Epoch {epoch+1}"):
            episode = next(episode_iter)
            outer_optimizer.zero_grad()

            support_texts = episode["support_texts"]
            support_labels = torch.tensor(episode["support_labels"]).to(device)
            query_texts = episode["query_texts"]
            query_labels = torch.tensor(episode["query_labels"]).to(device)

            # Encode
            support_embs = encoder.encode_text(support_texts, device)
            query_embs = encoder.encode_text(query_texts, device)

            # MAML inner loop (on classifier only)
            inner_opt = torch.optim.SGD(classifier.parameters(), lr=meta_cfg["inner_lr"])
            with higher.innerloop_ctx(classifier, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                for _ in range(meta_cfg["inner_steps"]):
                    s_logits = fmodel(support_embs)
                    inner_loss = F.cross_entropy(s_logits, support_labels)
                    diffopt.step(inner_loss)

                # Outer loss on query
                q_logits = fmodel(query_embs)
                outer_loss = F.cross_entropy(q_logits, query_labels)
                outer_loss.backward()

            outer_optimizer.step()
            acc = (q_logits.argmax(-1) == query_labels).float().mean().item()
            epoch_losses.append(outer_loss.item())
            epoch_accs.append(acc)

        mean_loss = np.mean(epoch_losses)
        mean_acc = np.mean(epoch_accs)
        print(f"Epoch {epoch+1:3d} | Loss: {mean_loss:.4f} | Acc: {mean_acc:.4f}")

        if mean_acc > best_acc:
            best_acc = mean_acc
            os.makedirs(config["training"]["save_dir"], exist_ok=True)
            torch.save({
                "encoder": encoder.state_dict(),
                "classifier": classifier.state_dict()
            }, os.path.join(config["training"]["save_dir"], "baseline_maml.pt"))
            print(f"  Saved best (acc={best_acc:.4f})")

    # Evaluate on target languages
    encoder.eval()
    classifier.eval()
    for lang in ds_cfg["target_languages"]:
        test_records = load_language_data(ds_cfg, lang, split="test")
        if not test_records:
            test_records = load_language_data(ds_cfg, lang, split="validation")
        if not test_records:
            print(f"Skipping {lang}")
            continue

        eval_sampler = EpisodeSampler(test_records, meta_cfg["n_way"], meta_cfg["k_shot"], meta_cfg["query_size"])
        ep_iter = iter(eval_sampler)
        accs = []
        for _ in range(200):
            ep = next(ep_iter)
            support_embs = encoder.encode_text(ep["support_texts"], device)
            query_embs = encoder.encode_text(ep["query_texts"], device)
            support_labels = torch.tensor(ep["support_labels"]).to(device)
            query_labels = torch.tensor(ep["query_labels"]).to(device)

            inner_opt = torch.optim.SGD(classifier.parameters(), lr=meta_cfg["inner_lr"])
            with higher.innerloop_ctx(classifier, inner_opt, copy_initial_weights=True) as (fmodel, diffopt):
                for _ in range(meta_cfg["inner_steps"]):
                    s_logits = fmodel(support_embs)
                    diffopt.step(F.cross_entropy(s_logits, support_labels))

                with torch.no_grad():
                    q_logits = fmodel(query_embs)
                    acc = (q_logits.argmax(-1) == query_labels).float().mean().item()
                    accs.append(acc)

        results = aggregate_episode_results(accs)
        print(f"[{lang}] MAML {meta_cfg['k_shot']}-shot | "
              f"Acc: {results['mean_accuracy']:.4f} +/- {results['95ci']:.4f}")

    print(f"\nDone. Best train acc: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train_maml(args.config)
