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


def load_processed_split(
    split_name: str,
    processed_dir: str = "data/processed",
    languages: tuple[str, ...] = EVAL_LANGUAGES,
    verbose: bool = True,
) -> dict[str, list]:
    datasets: dict[str, list] = {}
    for lang in languages:
        path = os.path.join(processed_dir, f"amazon_{lang}.json")
        if not os.path.exists(path):
            if verbose:
                print(f"  [{lang}] processed file not found — skipping.")
            continue
        with open(path, encoding="utf-8") as f:
            splits = json.load(f)
        records = splits.get(split_name, [])
        if not records:
            if verbose:
                print(f"  [{lang}] {split_name} split is empty — skipping.")
            continue
        datasets[lang] = records
        if verbose:
            print(f"  [{lang}] {len(records)} {split_name} records")
    return datasets


def evaluate_components(
    encoder,
    arc,
    meta_learner,
    retrieval_index,
    config: dict,
    device: torch.device,
    datasets: dict[str, list],
    n_episodes: int = 600,
    seed: int = 42,
    show_progress: bool = True,
) -> dict:
    meta_cfg = config["meta_learning"]

    sampler = CategoryStratifiedEpisodeSampler(
        datasets=datasets,
        n_shot=meta_cfg["k_shot"],
        n_query=meta_cfg["query_size"],
        n_class=meta_cfg["n_way"],
        seed=seed,
    )

    maes = []
    rmses = []
    lang_preds: dict[str, list] = {lg: [] for lg in EVAL_LANGUAGES}
    lang_targets: dict[str, list] = {lg: [] for lg in EVAL_LANGUAGES}

    iterator = tqdm(range(n_episodes), desc="Evaluating", disable=not show_progress)
    for _ in iterator:
        ep = sampler.sample_episode()
        lang = ep["language"]
        metrics = maml_eval_episode(encoder, arc, meta_learner, retrieval_index, ep, config, device)
        maes.append(metrics["mae"])
        rmses.append(metrics["rmse"])
        lang_preds[lang].extend(metrics["predictions"])
        lang_targets[lang].extend(metrics["targets"])

    all_preds = lang_preds["ja"] + lang_preds["zh"]
    all_targets = lang_targets["ja"] + lang_targets["zh"]
    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    overall_mae = np.mean(np.abs(all_preds_arr - all_targets_arr))
    overall_rmse = np.sqrt(np.mean((all_preds_arr - all_targets_arr) ** 2))
    correlation = np.corrcoef(all_preds_arr, all_targets_arr)[0, 1]

    mae_arr = np.array(maes)
    rmse_arr = np.array(rmses)
    mae_mean = mae_arr.mean()
    mae_std = mae_arr.std()
    mae_ci = 1.96 * mae_std / np.sqrt(len(mae_arr))
    rmse_mean = rmse_arr.mean()
    rmse_std = rmse_arr.std()
    rmse_ci = 1.96 * rmse_std / np.sqrt(len(rmse_arr))

    per_language = {}
    for lang in EVAL_LANGUAGES:
        lp, lt = lang_preds[lang], lang_targets[lang]
        if lp:
            lp_arr = np.array(lp)
            lt_arr = np.array(lt)
            per_language[lang] = {
                "mae": float(np.mean(np.abs(lp_arr - lt_arr))),
                "rmse": float(np.sqrt(np.mean((lp_arr - lt_arr) ** 2))),
                "correlation": float(np.corrcoef(lp_arr, lt_arr)[0, 1]),
                "pred_min": float(lp_arr.min()),
                "pred_max": float(lp_arr.max()),
                "true_min": float(lt_arr.min()),
                "true_max": float(lt_arr.max()),
                "num_predictions": len(lp),
                "num_episodes": len(lp) // meta_cfg["query_size"],
            }

    return {
        "mae_mean": float(mae_mean),
        "mae_std": float(mae_std),
        "mae_ci": float(mae_ci),
        "rmse_mean": float(rmse_mean),
        "rmse_std": float(rmse_std),
        "rmse_ci": float(rmse_ci),
        "overall_mae": float(overall_mae),
        "overall_rmse": float(overall_rmse),
        "correlation": float(correlation),
        "mae_distribution": mae_arr,
        "rmse_distribution": rmse_arr,
        "per_language": per_language,
        "languages": list(datasets.keys()),
        "n_episodes": n_episodes,
    }


def print_evaluation_report(results: dict, k_shot: int) -> None:
    langs = results["languages"]
    mae_arr = results["mae_distribution"]
    rmse_arr = results["rmse_distribution"]
    W = 62
    print(f"\n{'='*W}")
    print(f"  ARAML Evaluation (REGRESSION) -- {k_shot}-shot sentiment")
    print(f"  Languages : {'/'.join(langs)}")
    print(f"  Episodes  : {results['n_episodes']}")
    print(f"{'='*W}")

    print(f"\n  OVERALL METRICS (across all episodes)")
    print(f"  {'MAE (Mean Absolute Error)':<30}: {results['mae_mean']:.6f} +/- {results['mae_ci']:.6f}  (95% CI)")
    print(f"  {'RMSE (Root Mean Squared Error)':<30}: {results['rmse_mean']:.6f} +/- {results['rmse_ci']:.6f}  (95% CI)")
    print(f"  {'Overall MAE (on all predictions)':<30}: {results['overall_mae']:.6f}")
    print(f"  {'Overall RMSE (on all predictions)':<30}: {results['overall_rmse']:.6f}")
    print(f"  {'Pearson Correlation':<30}: {results['correlation']:.6f}  (1=perfect, 0=none)")
    print(f"  {'Mean Std Dev (MAE across eps)':<30}: {results['mae_std']:.6f}")
    print(f"  {'Mean Std Dev (RMSE across eps)':<30}: {results['rmse_std']:.6f}")

    print(f"\n  EPISODE MAE DISTRIBUTION")
    print(f"  {'Min':<30}: {mae_arr.min():.6f}")
    print(f"  {'Median':<30}: {np.median(mae_arr):.6f}")
    print(f"  {'Max':<30}: {mae_arr.max():.6f}")
    for p in [10, 25, 50, 75, 90]:
        print(f"  {f'{p}th percentile':<30}: {np.percentile(mae_arr, p):.6f}")

    print(f"\n  EPISODE RMSE DISTRIBUTION")
    print(f"  {'Min':<30}: {rmse_arr.min():.6f}")
    print(f"  {'Median':<30}: {np.median(rmse_arr):.6f}")
    print(f"  {'Max':<30}: {rmse_arr.max():.6f}")

    print(f"\n  PER-LANGUAGE BREAKDOWN")
    for lang in EVAL_LANGUAGES:
        stats = results["per_language"].get(lang)
        if stats:
            print(f"\n  [{lang.upper()}]  N={stats['num_predictions']} predictions  ({stats['num_episodes']} episodes)")
            print(f"    MAE={stats['mae']:.6f}  RMSE={stats['rmse']:.6f}  Correlation={stats['correlation']:.6f}")
            print(f"    Pred range: [{stats['pred_min']:.3f}, {stats['pred_max']:.3f}]")
            print(f"    True range: [{stats['true_min']:.3f}, {stats['true_max']:.3f}]")

    print(f"\n{'='*W}")


def evaluate(config_path: str, checkpoint: str, n_episodes: int = 600,
             index_path: str = "results/retrieval_index") -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    model = ARAML(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    encoder, arc, meta_learner = model.get_components()
    print(f"Loaded checkpoint: {checkpoint}")

    index = CrossLingualRetrievalIndex()
    index.load(index_path)
    print(f"Loaded retrieval index: {len(index)} entries")

    test_datasets = load_processed_split("test")

    if not test_datasets:
        print("\nNo test data found for any low-resource language.")
        print("Run data/download_data.py + data/preprocess.py first.")
        return

    results = evaluate_components(
        encoder, arc, meta_learner, index, config, device, test_datasets,
        n_episodes=n_episodes, seed=42, show_progress=True
    )
    print_evaluation_report(results, config["meta_learning"]["k_shot"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARAML evaluation")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--checkpoint",  default="results/best_model.pt")
    parser.add_argument("--index_path",  default="results/retrieval_index")
    parser.add_argument("--n_episodes",  type=int, default=600)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.n_episodes, args.index_path)
