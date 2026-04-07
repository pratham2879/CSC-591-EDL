"""
evaluate.py — Evaluate ARAML on low-resource target languages
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import yaml
import torch
import argparse
from tqdm import tqdm

from models.araml import ARAML
from models.retrieval_index import CrossLingualRetrievalIndex
from models.meta_learner import maml_eval_episode
from utils.episode_sampler import EpisodeSampler
from utils.metrics import aggregate_episode_results


def evaluate(config_path: str, checkpoint: str, n_episodes: int = 600):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    model = ARAML(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    encoder, arc, meta_learner = model.get_components()

    index = CrossLingualRetrievalIndex()
    index.load("results/retrieval_index")

    target_langs = config["data"]["target_languages"]
    meta_cfg = config["meta_learning"]

    for lang in target_langs:
        path = f"data/processed/amazon_{lang}.json"
        if not os.path.exists(path):
            print(f"Skipping {lang} — data not found.")
            continue

        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        test_records = [r for r in records if r["split"] == "test"]

        sampler = EpisodeSampler(test_records, meta_cfg["n_way"], meta_cfg["k_shot"], meta_cfg["query_size"])
        episode_iter = iter(sampler)

        accs = []
        for _ in tqdm(range(n_episodes), desc=f"Evaluating [{lang}]"):
            ep = next(episode_iter)
            acc = maml_eval_episode(encoder, arc, meta_learner, index, ep, config, device)
            accs.append(acc)

        results = aggregate_episode_results(accs)
        print(f"\n[{lang}] {meta_cfg['k_shot']}-shot | "
              f"Acc: {results['mean_accuracy']:.4f} +/- {results['95ci']:.4f} | "
              f"Std: {results['std']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="results/best_model.pt")
    parser.add_argument("--n_episodes", type=int, default=600)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.n_episodes)
