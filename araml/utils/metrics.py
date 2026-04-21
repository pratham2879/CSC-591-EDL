"""
metrics.py — Evaluation metrics for few-shot classification
"""
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, cohen_kappa_score
from scipy import stats


def compute_confidence_interval(scores: list, confidence: float = 0.95):
    """Compute mean ± 95% confidence interval."""
    n = len(scores)
    mean = np.mean(scores)
    se = stats.sem(scores)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, h


def evaluate_few_shot(predictions: list, targets: list):
    """
    Evaluate few-shot predictions with multiple metrics.
    
    Returns:
        dict with accuracy, f1_macro, kappa, confusion matrix, and per-class metrics
    """
    acc = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    kappa = cohen_kappa_score(targets, predictions)
    
    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(targets, predictions, labels=[0, 1])
    
    # Per-class precision, recall, f1
    precision_0 = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0
    recall_0 = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
    
    precision_1 = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    recall_1 = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "cohen_kappa": kappa,
        "confusion_matrix": cm,
        "precision_0": precision_0,  # Negative class
        "recall_0": recall_0,
        "f1_0": f1_0,
        "precision_1": precision_1,  # Positive class
        "recall_1": recall_1,
        "f1_1": f1_1,
    }


def aggregate_episode_results(episode_accs: list, episode_kappas: list = None):
    """
    Aggregate results across episodes.
    
    Args:
        episode_accs: list of accuracies per episode
        episode_kappas: list of Cohen's Kappas per episode (optional)
    """
    mean, ci = compute_confidence_interval(episode_accs)
    result = {
        "mean_accuracy": mean,
        "95ci": ci,
        "std": np.std(episode_accs)
    }
    
    if episode_kappas is not None:
        kappa_mean, kappa_ci = compute_confidence_interval(episode_kappas)
        result["mean_kappa"] = kappa_mean
        result["kappa_95ci"] = kappa_ci
        result["kappa_std"] = np.std(episode_kappas)
    
    return result
