"""
baseline_raml.py — Static Retrieval-Augmented Meta-Learning (RAML) baseline.
Uses MAML + fixed-K retrieval with uniform weighting (no adaptive controller).
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
import higher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.encoder import TextEncoder
from models.meta_learner import MetaLearner
from models.retrieval_index import CrossLingualRetrievalIndex
from utils.config_utils import get_dataset_config, load_multi_language_data, load_language_data
from utils.episode_sampler import EpisodeSampler
from utils.metrics import aggregate_episode_results


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_raml(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ds_cfg = get_dataset_config(config)
    num_classes = ds_cfg["num_classes"]
    meta_cfg = config["meta_learning"]
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Baseline: RAML] device={device}, dataset={ds_cfg['dataset_name']}")

    encoder = TextEncoder(model_name=config["model"]["encoder"]).to(device)
    # Same classifier architecture as ARAML (input = emb || retrieval_emb)
    meta_learner = MetaLearner(
        input_dim=config["model"]["hidden_dim"],
        num_classes=num_classes
    ).to(device)

    # Load retrieval index
    index = CrossLingualRetrievalIndex()
    index.load("results/retrieval_index")
    print(f"Retrieval index: {len(index):,} entries")

    all_records = load_multi_language_data(ds_cfg, ds_cfg["source_languages"], split="train")
    print(f"Training records: {len(all_records):,}")

    sampler = EpisodeSampler(all_records, meta_cfg["n_way"], meta_cfg["k_shot"], meta_cfg["query_size"])
    outer_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(meta_learner.parameters()),
        lr=meta_cfg["outer_lr"]
    )

    FIXED_K = 5  # Static retrieval budget
    best_acc = 0.0
    episode_iter = iter(sampler)
    episodes_per_epoch = config["training"].get("episodes_per_epoch", 100)

    for epoch in range(config["training"]["epochs"]):
        epoch_losses, epoch_accs = [], []

        for _ in tqdm(range(episodes_per_epoch), desc=f"RAML Epoch {epoch+1}"):
            outer_optimizer.zero_grad()
            episode = next(episode_iter)

            support_texts = episode["support_texts"]
            support_labels = torch.tensor(episode["support_labels"]).to(device)
            query_texts = episode["query_texts"]
            query_labels = torch.tensor(episode["query_labels"]).to(device)

            # Encode support
            support_embs = encoder.encode_text(support_texts, device)
            task_emb = support_embs.mean(0, keepdim=True)

            # Static retrieval: use mean support embedding as query, fixed K
            query_vec = task_emb.detach().cpu().numpy()
            retrieved = index.retrieve(query_vec, k=FIXED_K)
            ret_embs = encoder.encode_text(retrieved["texts"], device)

            # Uniform weighting (no attention)
            uniform_ret = ret_embs.mean(0, keepdim=True)  # (1, D)
            uniform_ret_s = uniform_ret.expand(support_embs.size(0), -1)

            aug_support = torch.cat([support_embs, uniform_ret_s], dim=-1)

            # MAML inner loop
            inner_opt = torch.optim.SGD(meta_learner.parameters(), lr=meta_cfg["inner_lr"])
            with higher.innerloop_ctx(meta_learner, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                for _ in range(meta_cfg["inner_steps"]):
                    s_logits = fmodel(aug_support)
                    inner_loss = F.cross_entropy(s_logits, support_labels)
                    diffopt.step(inner_loss)

                # Query evaluation
                query_embs = encoder.encode_text(query_texts, device)
                uniform_ret_q = uniform_ret.expand(query_embs.size(0), -1)
                aug_query = torch.cat([query_embs, uniform_ret_q], dim=-1)

                q_logits = fmodel(aug_query)
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
                "meta_learner": meta_learner.state_dict()
            }, os.path.join(config["training"]["save_dir"], "baseline_raml.pt"))
            print(f"  Saved best (acc={best_acc:.4f})")

    # Evaluate on target languages
    encoder.eval()
    meta_learner.eval()
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

            task_emb = support_embs.mean(0, keepdim=True)
            qvec = task_emb.detach().cpu().numpy()
            ret = index.retrieve(qvec, k=FIXED_K)
            ret_embs = encoder.encode_text(ret["texts"], device)
            uniform_ret = ret_embs.mean(0, keepdim=True)

            uniform_ret_s = uniform_ret.expand(support_embs.size(0), -1)
            aug_support = torch.cat([support_embs, uniform_ret_s], dim=-1)

            inner_opt = torch.optim.SGD(meta_learner.parameters(), lr=meta_cfg["inner_lr"])
            with higher.innerloop_ctx(meta_learner, inner_opt, copy_initial_weights=True) as (fmodel, diffopt):
                for _ in range(meta_cfg["inner_steps"]):
                    s_logits = fmodel(aug_support)
                    diffopt.step(F.cross_entropy(s_logits, support_labels))

                uniform_ret_q = uniform_ret.expand(query_embs.size(0), -1)
                aug_query = torch.cat([query_embs, uniform_ret_q], dim=-1)
                with torch.no_grad():
                    q_logits = fmodel(aug_query)
                    acc = (q_logits.argmax(-1) == query_labels).float().mean().item()
                    accs.append(acc)

        results = aggregate_episode_results(accs)
        print(f"[{lang}] RAML {meta_cfg['k_shot']}-shot | "
              f"Acc: {results['mean_accuracy']:.4f} +/- {results['95ci']:.4f}")

    print(f"\nDone. Best train acc: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train_raml(args.config)
