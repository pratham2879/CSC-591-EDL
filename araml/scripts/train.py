"""
train.py — ARAML meta-training script
"""
import os
import json
import yaml
import torch
import random
import argparse
import numpy as np
import tqdm

from models.araml import ARAML
from models.retrieval_index import CrossLingualRetrievalIndex
from models.meta_learner import meta_train_step
from utils.episode_sampler import EpisodeSampler


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["seed"])
    device = get_device()
    print(f"Training on: {device}")

    model = ARAML(config).to(device)
    encoder, arc, meta_learner = model.get_components()

<<<<<<< HEAD
=======
    # Tier 2: Freeze XLM-R encoder — only ARC + classifier are meta-learned
    if config["training"].get("freeze_encoder", True):
        for param in encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen. Trainable: ARC + MetaLearner only.")
        trainable_params = list(arc.parameters()) + list(meta_learner.parameters())
    else:
        trainable_params = list(model.parameters())

    total = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {total:,}")

>>>>>>> 62a7004391bd75afdca4a3784a3e05f5fe7dde7c
    index = CrossLingualRetrievalIndex()
    index.load("results/retrieval_index")
    print(f"Loaded retrieval index with {len(index)} entries.")

    source_langs = config["data"]["source_languages"]
    all_records = []
    for lang in source_langs:
        path = f"data/processed/amazon_{lang}.json"
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                records = json.load(f)
            train_records = [r for r in records if r["split"] == "train"]
            all_records.extend(train_records)
            print(f"  [{lang}] {len(train_records)} train records")

    print(f"Total training records: {len(all_records)}")

    meta_cfg = config["meta_learning"]
    batch_size = meta_cfg.get("meta_batch_size", 4)
    episodes_per_epoch = 100
    num_batches = episodes_per_epoch // batch_size

    sampler = EpisodeSampler(
        all_records,
        n_way=meta_cfg["n_way"],
        k_shot=meta_cfg["k_shot"],
        query_size=meta_cfg["query_size"]
    )

<<<<<<< HEAD
    outer_optimizer = torch.optim.Adam(model.parameters(), lr=meta_cfg["outer_lr"])
=======
    outer_optimizer = torch.optim.Adam(trainable_params, lr=meta_cfg["outer_lr"])
>>>>>>> 62a7004391bd75afdca4a3784a3e05f5fe7dde7c

    episodes_per_epoch = config["training"].get("episodes_per_epoch", 500)
    best_acc = 0.0
    episode_iter = iter(sampler)

    print(f"\nTraining: {config['training']['epochs']} epochs × {episodes_per_epoch} episodes "
          f"(batch_size={batch_size}, {num_batches} batches/epoch)\n")

    for epoch in range(config["training"]["epochs"]):
        epoch_losses, epoch_accs = [], []
        meta_learner.train()

<<<<<<< HEAD
        for _ in tqdm.tqdm(range(num_batches), desc=f"Epoch {epoch+1:2d}/{config['training']['epochs']}"):
            outer_optimizer.zero_grad()
            batch_losses, batch_accs = [], []

            for ep_idx in range(batch_size):
                episode = next(episode_iter)
                is_last = (ep_idx == batch_size - 1)
                loss, acc = meta_train_step(
                    encoder, arc, meta_learner, index, episode,
                    config, device, outer_optimizer,
                    step=is_last
                )
                batch_losses.append(loss)
                batch_accs.append(acc)

            epoch_losses.append(np.mean(batch_losses))
            epoch_accs.append(np.mean(batch_accs))
=======
        for _ in tqdm.tqdm(range(episodes_per_epoch), desc=f"Epoch {epoch+1}"):
            episode = next(episode_iter)
            loss, acc = meta_train_step(
                encoder, arc, meta_learner, index, episode,
                config, device, outer_optimizer
            )
            epoch_losses.append(loss)
            epoch_accs.append(acc)
>>>>>>> 62a7004391bd75afdca4a3784a3e05f5fe7dde7c

        mean_loss = np.mean(epoch_losses)
        mean_acc = np.mean(epoch_accs)
        print(f"Epoch {epoch+1:2d}/{config['training']['epochs']} | Loss: {mean_loss:.4f} | Acc: {mean_acc:.4f}")

        if mean_acc > best_acc:
            best_acc = mean_acc
            os.makedirs(config["training"]["save_dir"], exist_ok=True)
            torch.save(model.state_dict(), config["training"]["save_dir"] + "best_model.pt")
            print(f"  → Saved new best model (acc={best_acc:.4f})")

    print(f"\nTraining complete. Best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
