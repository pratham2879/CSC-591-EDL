"""
baseline_protonet.py — Prototypical Networks baseline for few-shot classification.
No retrieval, no MAML — just prototype-based nearest-centroid classification.
"""
import os
import sys
import yaml
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.encoder import TextEncoder
from utils.config_utils import get_dataset_config, load_multi_language_data, load_language_data
from utils.episode_sampler import EpisodeSampler
from utils.metrics import aggregate_episode_results


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prototypical_loss(support_embs, support_labels, query_embs, query_labels, n_way):
    """Compute prototypical network loss and accuracy."""
    prototypes = []
    for c in range(n_way):
        mask = support_labels == c
        prototypes.append(support_embs[mask].mean(0))
    prototypes = torch.stack(prototypes)  # (n_way, D)

    # Euclidean distance
    dists = torch.cdist(query_embs, prototypes)  # (Q, n_way)
    logits = -dists  # negative distance -> higher is closer
    loss = F.cross_entropy(logits, query_labels)
    acc = (logits.argmax(-1) == query_labels).float().mean().item()
    return loss, acc


def train_protonet(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ds_cfg = get_dataset_config(config)
    meta_cfg = config["meta_learning"]
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Baseline: ProtoNet] device={device}, dataset={ds_cfg['dataset_name']}")

    encoder = TextEncoder(model_name=config["model"]["encoder"]).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=meta_cfg["outer_lr"])

    all_records = load_multi_language_data(ds_cfg, ds_cfg["source_languages"], split="train")
    print(f"Training records: {len(all_records):,}")

    sampler = EpisodeSampler(all_records, meta_cfg["n_way"], meta_cfg["k_shot"], meta_cfg["query_size"])
    episode_iter = iter(sampler)
    episodes_per_epoch = config["training"].get("episodes_per_epoch", 100)
    best_acc = 0.0

    for epoch in range(config["training"]["epochs"]):
        epoch_losses, epoch_accs = [], []

        for _ in tqdm(range(episodes_per_epoch), desc=f"Proto Epoch {epoch+1}"):
            episode = next(episode_iter)
            optimizer.zero_grad()

            support_embs = encoder.encode_text(episode["support_texts"], device)
            query_embs = encoder.encode_text(episode["query_texts"], device)
            support_labels = torch.tensor(episode["support_labels"]).to(device)
            query_labels = torch.tensor(episode["query_labels"]).to(device)

            loss, acc = prototypical_loss(
                support_embs, support_labels,
                query_embs, query_labels, meta_cfg["n_way"]
            )
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_accs.append(acc)

        mean_loss = np.mean(epoch_losses)
        mean_acc = np.mean(epoch_accs)
        print(f"Epoch {epoch+1:3d} | Loss: {mean_loss:.4f} | Acc: {mean_acc:.4f}")

        if mean_acc > best_acc:
            best_acc = mean_acc
            os.makedirs(config["training"]["save_dir"], exist_ok=True)
            torch.save(encoder.state_dict(),
                       os.path.join(config["training"]["save_dir"], "baseline_protonet.pt"))
            print(f"  Saved best (acc={best_acc:.4f})")

    # Evaluate on target languages
    encoder.eval()
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
            with torch.no_grad():
                support_embs = encoder.encode_text(ep["support_texts"], device)
                query_embs = encoder.encode_text(ep["query_texts"], device)
                support_labels = torch.tensor(ep["support_labels"]).to(device)
                query_labels = torch.tensor(ep["query_labels"]).to(device)
                _, acc = prototypical_loss(
                    support_embs, support_labels,
                    query_embs, query_labels, meta_cfg["n_way"]
                )
                accs.append(acc)

        results = aggregate_episode_results(accs)
        print(f"[{lang}] ProtoNet {meta_cfg['k_shot']}-shot | "
              f"Acc: {results['mean_accuracy']:.4f} +/- {results['95ci']:.4f}")

    print(f"\nDone. Best train acc: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train_protonet(args.config)
