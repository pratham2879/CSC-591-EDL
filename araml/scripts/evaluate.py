"""
evaluate.py — Evaluate ARAML on low-resource target languages.
"""
import os
import sys
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.araml import ARAML
from models.retrieval_index import CrossLingualRetrievalIndex
from models.meta_learner import maml_inner_loop
from utils.episode_sampler import EpisodeSampler
from utils.metrics import aggregate_episode_results
from utils.config_utils import get_dataset_config, load_language_data


def evaluate(config_path: str, checkpoint: str, n_episodes: int = 600):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ds_cfg = get_dataset_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ARAML(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    encoder, arc, meta_learner = model.get_components()

    index = CrossLingualRetrievalIndex()
    index.load("results/retrieval_index")

    meta_cfg = config["meta_learning"]

    for lang in ds_cfg["target_languages"]:
        test_records = load_language_data(ds_cfg, lang, split="test")
        if not test_records:
            test_records = load_language_data(ds_cfg, lang, split="validation")
        if not test_records:
            print(f"Skipping {lang} — no test/validation data.")
            continue

        sampler = EpisodeSampler(
            test_records, meta_cfg["n_way"],
            meta_cfg["k_shot"], meta_cfg["query_size"]
        )
        episode_iter = iter(sampler)

        accs = []
        for _ in tqdm(range(n_episodes), desc=f"Evaluating [{lang}]"):
            ep = next(episode_iter)
            support_embs = encoder.encode_text(ep["support_texts"], device)
            query_embs = encoder.encode_text(ep["query_texts"], device)
            support_labels = torch.tensor(ep["support_labels"]).to(device)
            query_labels = torch.tensor(ep["query_labels"]).to(device)

            task_emb = support_embs.mean(0, keepdim=True)
            k = arc.predict_budget(task_emb)
            query_vec = arc.generate_query(task_emb).detach().cpu().numpy()
            retrieved = index.retrieve(query_vec, k=k)
            ret_embs = encoder.encode_text(retrieved["texts"], device)

            _, _, weighted_ret_emb, _ = arc(task_emb, ret_embs)
            weighted_ret_emb_s = weighted_ret_emb.unsqueeze(0).expand(support_embs.size(0), -1)
            aug_support = torch.cat([support_embs, weighted_ret_emb_s], dim=-1)

            adapted = maml_inner_loop(
                meta_learner, aug_support, support_labels,
                meta_cfg["inner_lr"], meta_cfg["inner_steps"], device
            )

            weighted_ret_emb_q = weighted_ret_emb.unsqueeze(0).expand(query_embs.size(0), -1)
            aug_query = torch.cat([query_embs, weighted_ret_emb_q], dim=-1)

            with torch.no_grad():
                logits = adapted(aug_query)
                preds = logits.argmax(-1)
                acc = (preds == query_labels).float().mean().item()
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
