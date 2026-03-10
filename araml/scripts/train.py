"""
train.py — ARAML meta-training script.
"""
import os
import sys
import yaml
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.araml import ARAML
from models.retrieval_index import CrossLingualRetrievalIndex
from models.meta_learner import meta_train_step
from utils.episode_sampler import EpisodeSampler
from utils.config_utils import get_dataset_config, load_multi_language_data


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ds_cfg = get_dataset_config(config)
    num_classes = ds_cfg["num_classes"]
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Dataset:     {ds_cfg['dataset_name']}  ({num_classes} classes)")

    # Load model
    model = ARAML(config).to(device)
    encoder, arc, meta_learner = model.get_components()
    print(f"Total parameters: {model.count_parameters():,}")

    # Load retrieval index
    index = CrossLingualRetrievalIndex()
    index.load("results/retrieval_index")
    print(f"Loaded retrieval index with {len(index):,} entries.")

    # Load source-language training data
    print("Loading source-language training data ...")
    all_records = load_multi_language_data(
        ds_cfg, ds_cfg["source_languages"], split="train"
    )
    print(f"Total training records: {len(all_records):,}")

    meta_cfg = config["meta_learning"]
    sampler = EpisodeSampler(
        all_records,
        n_way=meta_cfg["n_way"],
        k_shot=meta_cfg["k_shot"],
        query_size=meta_cfg["query_size"]
    )

    outer_optimizer = torch.optim.Adam(model.parameters(), lr=meta_cfg["outer_lr"])
    best_acc = 0.0
    episode_iter = iter(sampler)
    episodes_per_epoch = config["training"].get("episodes_per_epoch", 100)

    for epoch in range(config["training"]["epochs"]):
        epoch_losses, epoch_accs = [], []

        for _ in tqdm(range(episodes_per_epoch), desc=f"Epoch {epoch+1}"):
            episode = next(episode_iter)
            loss, acc = meta_train_step(
                encoder, arc, meta_learner, index, episode,
                config, device, outer_optimizer
            )
            epoch_losses.append(loss)
            epoch_accs.append(acc)

        mean_loss = np.mean(epoch_losses)
        mean_acc = np.mean(epoch_accs)
        print(f"Epoch {epoch+1:3d} | Loss: {mean_loss:.4f} | Acc: {mean_acc:.4f}")

        if mean_acc > best_acc:
            best_acc = mean_acc
            os.makedirs(config["training"]["save_dir"], exist_ok=True)
            save_path = os.path.join(config["training"]["save_dir"], "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  Saved new best model (acc={best_acc:.4f})")

    print(f"\nTraining complete. Best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
