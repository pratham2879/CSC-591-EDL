"""
evaluate.py — Evaluate ARAML on low-resource target languages (ja, zh) (REGRESSION).

Evaluation follows the same few-shot protocol as training:
  - Episodes sampled from the TEST split of low-resource languages.
  - Support set: adapt the regressor in k inner-loop steps.
  - Query set: measure MAE and RMSE with adapted regressor.
  - Report mean MAE ± 95% CI over n_episodes episodes.

Task: Few-shot REGRESSION (sentiment prediction in [0, 1] range)

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

    # -- Evaluate over n_episodes (REGRESSION) ----------------------------------
    maes   = []
    rmses  = []
    lang_preds:  dict[str, list] = {lg: [] for lg in EVAL_LANGUAGES}
    lang_targets: dict[str, list] = {lg: [] for lg in EVAL_LANGUAGES}

    for _ in tqdm(range(n_episodes), desc="Evaluating"):
        ep      = sampler.sample_episode()
        lang    = ep["language"]
        metrics = maml_eval_episode(encoder, arc, meta_learner, index, ep, config, device)
        maes.append(metrics["mae"])
        rmses.append(metrics["rmse"])
        lang_preds[lang].extend(metrics["predictions"])
        lang_targets[lang].extend(metrics["targets"])

    all_preds   = lang_preds["ja"]   + lang_preds["zh"]
    all_targets = lang_targets["ja"] + lang_targets["zh"]
    
    # Compute overall regression metrics
    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    overall_mae = np.mean(np.abs(all_preds_arr - all_targets_arr))
    overall_rmse = np.sqrt(np.mean((all_preds_arr - all_targets_arr)**2))
    
    # Compute correlationcoefficient
    correlation = np.corrcoef(all_preds_arr, all_targets_arr)[0, 1]

    mae_arr = np.array(maes)
    rmse_arr = np.array(rmses)
    
    # Compute 95% CI for MAE and RMSE
    mae_mean = mae_arr.mean()
    mae_std = mae_arr.std()
    mae_ci = 1.96 * mae_std / np.sqrt(len(mae_arr))
    
    rmse_mean = rmse_arr.mean()
    rmse_std = rmse_arr.std()
    rmse_ci = 1.96 * rmse_std / np.sqrt(len(rmse_arr))

    langs = list(test_datasets.keys())
    W = 62
    print(f"\n{'='*W}")
    print(f"  ARAML Evaluation (REGRESSION) -- {meta_cfg['k_shot']}-shot sentiment")
    print(f"  Languages : {'/'.join(langs)}")
    print(f"  Episodes  : {n_episodes}")
    print(f"{'='*W}")

    # ---- Overall metrics (REGRESSION) ------------------------------------------
    print(f"\n  OVERALL METRICS (across all episodes)")
    print(f"  {'MAE (Mean Absolute Error)':<30}: {mae_mean:.6f} +/- {mae_ci:.6f}  (95% CI)")
    print(f"  {'RMSE (Root Mean Squared Error)':<30}: {rmse_mean:.6f} +/- {rmse_ci:.6f}  (95% CI)")
    print(f"  {'Overall MAE (on all predictions)':<30}: {overall_mae:.6f}")
    print(f"  {'Overall RMSE (on all predictions)':<30}: {overall_rmse:.6f}")
    print(f"  {'Pearson Correlation':<30}: {correlation:.6f}  (1=perfect, 0=none)")
    print(f"  {'Mean Std Dev (MAE across eps)':<30}: {mae_std:.6f}")
    print(f"  {'Mean Std Dev (RMSE across eps)':<30}: {rmse_std:.6f}")

    # ---- Episode MAE distribution -----------------------------------------
    print(f"\n  EPISODE MAE DISTRIBUTION")
    print(f"  {'Min':<30}: {mae_arr.min():.6f}")
    print(f"  {'Median':<30}: {np.median(mae_arr):.6f}")
    print(f"  {'Max':<30}: {mae_arr.max():.6f}")
    
    # Percentiles for MAE
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        val = np.percentile(mae_arr, p)
        print(f"  {f'{p}th percentile':<30}: {val:.6f}")

    # ---- Episode RMSE distribution -----------------------------------------
    print(f"\n  EPISODE RMSE DISTRIBUTION")
    print(f"  {'Min':<30}: {rmse_arr.min():.6f}")
    print(f"  {'Median':<30}: {np.median(rmse_arr):.6f}")
    print(f"  {'Max':<30}: {rmse_arr.max():.6f}")

    # ---- Per-language breakdown (REGRESSION) ----------------------------------
    print(f"\n  PER-LANGUAGE BREAKDOWN")
    for lang in EVAL_LANGUAGES:
        lp, lt = lang_preds[lang], lang_targets[lang]
        if lp:
            lp_arr = np.array(lp)
            lt_arr = np.array(lt)
            lmae = np.mean(np.abs(lp_arr - lt_arr))
            lrmse = np.sqrt(np.mean((lp_arr - lt_arr)**2))
            lcorr = np.corrcoef(lp_arr, lt_arr)[0, 1]
            print(f"\n  [{lang.upper()}]  N={len(lp)} predictions  ({len(lp) // meta_cfg['query_size']} episodes)")
            print(f"    MAE={lmae:.6f}  RMSE={lrmse:.6f}  Correlation={lcorr:.6f}")
            print(f"    Pred range: [{lp_arr.min():.3f}, {lp_arr.max():.3f}]")
            print(f"    True range: [{lt_arr.min():.3f}, {lt_arr.max():.3f}]")

    print(f"\n{'='*W}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARAML evaluation")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--checkpoint",  default="results/best_model.pt")
    parser.add_argument("--index_path",  default="results/retrieval_index")
    parser.add_argument("--n_episodes",  type=int, default=600)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.n_episodes, args.index_path)
