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
import numpy as np
import argparse
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.araml          import ARAML
from models.retrieval_index import CrossLingualRetrievalIndex
from models.meta_learner   import maml_eval_episode
from utils.episode_sampler import CategoryStratifiedEpisodeSampler
from utils.metrics         import aggregate_episode_results
from sklearn.metrics       import (precision_recall_fscore_support,
                                   confusion_matrix,
                                   matthews_corrcoef)

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
    accs   = []
    kappas = []
    lang_preds:  dict[str, list] = {lg: [] for lg in EVAL_LANGUAGES}
    lang_labels: dict[str, list] = {lg: [] for lg in EVAL_LANGUAGES}

    for _ in tqdm(range(n_episodes), desc="Evaluating"):
        ep      = sampler.sample_episode()
        lang    = ep["language"]
        metrics = maml_eval_episode(encoder, arc, meta_learner, index, ep, config, device)
        accs.append(metrics["accuracy"])
        kappas.append(metrics["kappa"])
        lang_preds[lang].extend(metrics["predictions"])
        lang_labels[lang].extend(metrics["targets"])

    results = aggregate_episode_results(accs, kappas)

    all_preds  = lang_preds["ja"]  + lang_preds["zh"]
    all_labels = lang_labels["ja"] + lang_labels["zh"]
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    mcc = matthews_corrcoef(all_labels, all_preds)

    ep_accs = np.array(accs)
    pct_above_80 = (ep_accs >= 0.80).mean() * 100
    pct_above_90 = (ep_accs >= 0.90).mean() * 100

    langs = list(test_datasets.keys())
    W = 62
    print(f"\n{'='*W}")
    print(f"  ARAML Evaluation -- {meta_cfg['k_shot']}-shot binary sentiment")
    print(f"  Languages : {'/'.join(langs)}")
    print(f"  Episodes  : {n_episodes}")
    print(f"{'='*W}")

    # ---- Overall metrics ---------------------------------------------------
    print(f"\n  OVERALL METRICS")
    print(f"  {'Accuracy':<18}: {results['mean_accuracy']:.4f} +/- {results['95ci']:.4f}  (95% CI)")
    print(f"  {'Std Dev':<18}: {results['std']:.4f}")
    if results.get("mean_kappa") is not None:
        print(f"  {'Kappa':<18}: {results['mean_kappa']:.4f} +/- {results['kappa_95ci']:.4f}  (Cohen's)")
    print(f"  {'Macro Precision':<18}: {prec:.4f}")
    print(f"  {'Macro Recall':<18}: {rec:.4f}")
    print(f"  {'Macro F1':<18}: {f1:.4f}")
    print(f"  {'MCC':<18}: {mcc:.4f}  (0=random, 1=perfect)")

    # ---- Episode accuracy distribution ------------------------------------
    print(f"\n  EPISODE ACCURACY DISTRIBUTION")
    print(f"  {'Min':<18}: {ep_accs.min():.4f}")
    print(f"  {'Median':<18}: {np.median(ep_accs):.4f}")
    print(f"  {'Max':<18}: {ep_accs.max():.4f}")
    print(f"  {'>= 80% accuracy':<18}: {pct_above_80:.1f}% of episodes")
    print(f"  {'>= 90% accuracy':<18}: {pct_above_90:.1f}% of episodes")

    # ---- Per-class breakdown (neg / pos) -----------------------------------
    print(f"\n  PER-CLASS BREAKDOWN  (negative=0, positive=1)")
    _, _, per_cls_f1, per_cls_sup = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1], zero_division=0
    )
    cls_prec, cls_rec, _, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1], zero_division=0
    )
    for cls_idx, cls_name in [(0, "negative"), (1, "positive")]:
        print(f"  {cls_name:<18}: P={cls_prec[cls_idx]:.4f}  R={cls_rec[cls_idx]:.4f}"
              f"  F1={per_cls_f1[cls_idx]:.4f}  support={per_cls_sup[cls_idx]}")

    # ---- Confusion matrix --------------------------------------------------
    print(f"\n  CONFUSION MATRIX  (rows=actual, cols=predicted)")
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    print(f"                pred_neg  pred_pos")
    print(f"  actual_neg :  {cm[0,0]:>7}   {cm[0,1]:>7}")
    print(f"  actual_pos :  {cm[1,0]:>7}   {cm[1,1]:>7}")
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Negatives : {tn}   False Positives : {fp}")
    print(f"  False Negatives: {fn}   True Positives  : {tp}")

    # ---- Per-language breakdown --------------------------------------------
    print(f"\n  PER-LANGUAGE BREAKDOWN")
    for lang in EVAL_LANGUAGES:
        lp, ll = lang_preds[lang], lang_labels[lang]
        if lp:
            lprec, lrec, lf1, _ = precision_recall_fscore_support(
                ll, lp, average="macro", zero_division=0
            )
            lmcc  = matthews_corrcoef(ll, lp)
            lacc  = sum(p == l for p, l in zip(lp, ll)) / len(lp)
            lcm   = confusion_matrix(ll, lp, labels=[0, 1])
            print(f"\n  [{lang.upper()}]  N={len(lp)} predictions  ({len(lp) // meta_cfg['query_size']} episodes)")
            print(f"    Acc={lacc:.4f}  P={lprec:.4f}  R={lrec:.4f}  F1={lf1:.4f}  MCC={lmcc:.4f}")
            print(f"    Confusion:  TN={lcm[0,0]}  FP={lcm[0,1]}  FN={lcm[1,0]}  TP={lcm[1,1]}")

    print(f"\n{'='*W}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARAML evaluation")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--checkpoint",  default="results/best_model.pt")
    parser.add_argument("--index_path",  default="results/retrieval_index")
    parser.add_argument("--n_episodes",  type=int, default=600)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.n_episodes, args.index_path)
