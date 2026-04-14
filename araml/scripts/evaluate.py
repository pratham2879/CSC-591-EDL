"""
evaluate.py — Evaluate ARAML on low-resource target languages (ja, zh).

Evaluation follows the same few-shot protocol as training:
  - Episodes sampled from the TEST split of low-resource languages.
  - Support set: adapt the classifier in k inner-loop steps.
  - Query set: measure accuracy with adapted classifier.
  - Report mean accuracy ± 95% CI over n_episodes episodes.

Run from inside araml/:
    PYTHONPATH=. python scripts/evaluate.py \
        --checkpoint results/best_model.pt \
        --n_episodes 600
"""
import os
import sys
import json
import yaml
import torch
import argparse
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.araml          import ARAML
from models.retrieval_index import CrossLingualRetrievalIndex
from models.meta_learner   import maml_eval_episode
from utils.episode_sampler import CategoryStratifiedEpisodeSampler
from utils.metrics         import aggregate_episode_results
from sklearn.metrics       import precision_recall_fscore_support

# Languages that may appear as support/query in episodes.
# Only LOW_RESOURCE languages are valid per CategoryStratifiedEpisodeSampler.
EVAL_LANGUAGES = ("ja", "zh")


def evaluate(config_path: str, checkpoint: str, n_episodes: int = 600,
             index_path: str = "results/retrieval_index") -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # -- Load model ----------------------------------------------------------
    model = ARAML(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    encoder, arc, meta_learner = model.get_components()
    print(f"Loaded checkpoint: {checkpoint}")

    # -- Retrieval index -----------------------------------------------------
    index = CrossLingualRetrievalIndex()
    index.load(index_path)
    print(f"Loaded retrieval index: {len(index)} entries")

    # -- Build episode sampler from TEST splits of low-resource languages ----
    meta_cfg     = config["meta_learning"]
    processed_dir = "data/processed"

    test_datasets: dict[str, list] = {}
    for lang in EVAL_LANGUAGES:
        path = os.path.join(processed_dir, f"amazon_{lang}.json")
        if not os.path.exists(path):
            print(f"  [{lang}] processed file not found — skipping.")
            continue
        with open(path, encoding="utf-8") as f:
            splits = json.load(f)
        test_records = splits.get("test", [])
        if not test_records:
            print(f"  [{lang}] test split is empty — skipping.")
            continue
        test_datasets[lang] = test_records
        print(f"  [{lang}] {len(test_records)} test records")

    if not test_datasets:
        print("\nNo test data found for any low-resource language.")
        print("Run data/download_data.py + data/preprocess.py first.")
        return

    sampler = CategoryStratifiedEpisodeSampler(
        datasets=test_datasets,
        n_shot=meta_cfg["k_shot"],
        n_query=meta_cfg["query_size"],
        n_class=meta_cfg["n_way"],
        seed=42,
    )

    # -- Evaluate over n_episodes --------------------------------------------
    accs = []
    lang_preds:  dict[str, list] = {lg: [] for lg in EVAL_LANGUAGES}
    lang_labels: dict[str, list] = {lg: [] for lg in EVAL_LANGUAGES}

    for _ in tqdm(range(n_episodes), desc="Evaluating"):
        ep              = sampler.sample_episode()
        lang            = ep["language"]
        acc, preds, labels = maml_eval_episode(
            encoder, arc, meta_learner, index, ep, config, device
        )
        accs.append(acc)
        lang_preds[lang].extend(preds)
        lang_labels[lang].extend(labels)

    results = aggregate_episode_results(accs)

    all_preds  = lang_preds["ja"]  + lang_preds["zh"]
    all_labels = lang_labels["ja"] + lang_labels["zh"]
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    langs = list(test_datasets.keys())
    print(f"\n{'='*55}")
    print(f"  ARAML Evaluation — {meta_cfg['k_shot']}-shot binary sentiment")
    print(f"  Languages : {'/'.join(langs)}")
    print(f"  Episodes  : {n_episodes}")
    print(f"{'='*55}")
    print(f"  Overall  | Acc: {results['mean_accuracy']:.4f} ± {results['95ci']:.4f} (95% CI)"
          f" | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}")
    print(f"  Std      : {results['std']:.4f}")
    for lang in EVAL_LANGUAGES:
        lp, ll = lang_preds[lang], lang_labels[lang]
        if lp:
            lprec, lrec, lf1, _ = precision_recall_fscore_support(
                ll, lp, average="macro", zero_division=0
            )
            lacc = sum(p == l for p, l in zip(lp, ll)) / len(lp)
            print(f"  [{lang}]     | Acc: {lacc:.4f} | P: {lprec:.4f}"
                  f" | R: {lrec:.4f} | F1: {lf1:.4f} | N={len(lp)}")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARAML evaluation")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--checkpoint",  default="results/best_model.pt")
    parser.add_argument("--index_path",  default="results/retrieval_index")
    parser.add_argument("--n_episodes",  type=int, default=600)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.n_episodes, args.index_path)
