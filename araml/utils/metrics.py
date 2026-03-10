"""
metrics.py — Evaluation metrics for few-shot classification
"""
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from scipy import stats


def compute_confidence_interval(scores: list, confidence: float = 0.95):
    """Compute mean ± 95% confidence interval."""
    n = len(scores)
    mean = np.mean(scores)
    se = stats.sem(scores)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, h


def evaluate_few_shot(predictions: list, targets: list):
    acc = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    return {"accuracy": acc, "f1_macro": f1}


def aggregate_episode_results(episode_accs: list):
    mean, ci = compute_confidence_interval(episode_accs)
    return {
        "mean_accuracy": mean,
        "95ci": ci,
        "std": np.std(episode_accs)
    }
