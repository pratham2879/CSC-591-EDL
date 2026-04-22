"""
profile_inference.py — Profile ARAML inference: latency, GPU memory

Tracks per-episode:
  - Inference latency (total time for one episode)
  - GPU memory allocation (peak usage)
  - Cohen's Kappa and accuracy metrics

Generates plots: latency distribution, memory usage, accuracy/kappa relationships

Run from inside araml/:
    PYTHONPATH=. python scripts/profile_inference.py \
        --checkpoint results/best_model.pt \
        --n_episodes 100
"""
import os
import sys
import json
import yaml
import torch
import argparse
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.araml          import ARAML
from models.retrieval_index import CrossLingualRetrievalIndex
from models.meta_learner   import maml_eval_episode
from utils.episode_sampler import CategoryStratifiedEpisodeSampler


# ============================================================================
# GPU Memory Tracking
# ============================================================================

class MemoryTracker:
    """Track GPU memory usage during inference."""
    def __init__(self, device):
        self.device = device
        self.peak_memory = 0
        self.is_cuda = device.type == "cuda"
    
    def reset(self):
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    
    def get_peak_memory_mb(self):
        if self.is_cuda:
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            return psutil.Process().memory_info().rss / 1024 / 1024


# ============================================================================
# Profiled Episode Evaluation
# ============================================================================

def profile_episode(encoder, arc, meta_learner, index, episode, config, device,
                   memory_tracker, timers):
    """
    Evaluate a single episode with detailed profiling.
    
    Returns:
        dict with accuracy, kappa, and timing breakdown
    """
    memory_tracker.reset()
    
    t_retrieval_start = time.time()
    
    # For now, we estimate retrieval time by the encoding + retrieval operations
    # Since maml_eval_episode is a black box, we measure overall latency
    t_start = time.time()
    metrics = maml_eval_episode(encoder, arc, meta_learner, index, episode, config, device)
    t_end = time.time()
    
    total_episode_time = t_end - t_start
    timers["total"].append(total_episode_time)
    
    peak_memory = memory_tracker.get_peak_memory_mb()
    accuracy = metrics["accuracy"]
    kappa = metrics["kappa"]
    
    return {
        "accuracy": accuracy,
        "kappa": kappa,
        "total_time": total_episode_time,
        "peak_memory_mb": peak_memory,
    }


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_profiling_results(results, output_dir="results"):
    """Generate profiling plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    latencies = np.array([r["total_time"] for r in results])
    memory_usage = np.array([r["peak_memory_mb"] for r in results])
    accuracies = np.array([r["accuracy"] for r in results])
    kappas = np.array([r["kappa"] for r in results])
    
    # --- Figure 1: Latency Distribution ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(latencies * 1000, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0, 0].set_xlabel("Total Latency (ms)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Episode Inference Latency Distribution")
    axes[0, 0].axvline(np.mean(latencies) * 1000, color="red", linestyle="--", label=f"Mean: {np.mean(latencies)*1000:.1f}ms")
    axes[0, 0].legend()
    
    # --- Figure 2: GPU Memory Usage ---
    axes[0, 1].plot(memory_usage, marker="o", linestyle="-", alpha=0.7, color="darkgreen")
    axes[0, 1].axhline(np.mean(memory_usage), color="red", linestyle="--", label=f"Mean: {np.mean(memory_usage):.1f}MB")
    axes[0, 1].set_xlabel("Episode #")
    axes[0, 1].set_ylabel("Peak Memory (MB)")
    axes[0, 1].set_title("GPU Memory Footprint per Episode")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # --- Figure 3: Accuracy vs Latency ---
    axes[1, 0].scatter(latencies * 1000, accuracies, alpha=0.6, s=50, color="purple")
    axes[1, 0].set_xlabel("Latency (ms)")
    axes[1, 0].set_ylabel("Episode Accuracy")
    axes[1, 0].set_title("Accuracy vs Inference Latency")
    axes[1, 0].grid(True, alpha=0.3)
    
    # --- Figure 4: Cohen's Kappa vs Accuracy ---
    axes[1, 1].scatter(accuracies, kappas, alpha=0.6, s=50, color="orange")
    axes[1, 1].set_xlabel("Episode Accuracy")
    axes[1, 1].set_ylabel("Cohen's Kappa")
    axes[1, 1].set_title("Cohen's Kappa vs Accuracy")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "profiling_latency_memory.png"), dpi=150)
    print(f"✓ Saved: {os.path.join(output_dir, 'profiling_latency_memory.png')}")
    plt.close()
    
    # --- Figure 5: Latency Timeline ---
    fig, ax = plt.subplots(figsize=(14, 6))
    
    episodes = np.arange(len(latencies))
    ax.plot(episodes, latencies * 1000, marker="o", linestyle="-", alpha=0.7, color="steelblue")
    ax.axhline(np.mean(latencies) * 1000, color="red", linestyle="--", label=f"Mean: {np.mean(latencies)*1000:.1f}ms")
    
    ax.set_xlabel("Episode #")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency per Episode")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "profiling_latency_timeline.png"), dpi=150)
    print(f"✓ Saved: {os.path.join(output_dir, 'profiling_latency_timeline.png')}")
    plt.close()


# ============================================================================
# Main Profiling Loop
# ============================================================================

def profile_inference(config_path: str, checkpoint: str, n_episodes: int = 100,
                     index_path: str = "results/retrieval_index") -> None:
    """Run profiling benchmark."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on: {device}")

    # -- Load model --
    model = ARAML(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    encoder, arc, meta_learner = model.get_components()
    print(f"Loaded checkpoint: {checkpoint}")

    # -- Retrieval index --
    index = CrossLingualRetrievalIndex()
    index.load(index_path)
    print(f"Loaded retrieval index: {len(index)} entries")

    # -- Build episode sampler --
    meta_cfg = config["meta_learning"]
    processed_dir = "data/processed"
    
    EVAL_LANGUAGES = ("ja", "zh")
    test_datasets = {}
    for lang in EVAL_LANGUAGES:
        path = os.path.join(processed_dir, f"amazon_{lang}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            splits = json.load(f)
        test_records = splits.get("test", [])
        if test_records:
            test_datasets[lang] = test_records
            print(f"  [{lang}] {len(test_records)} test records")

    if not test_datasets:
        print("No test data found.")
        return

    sampler = CategoryStratifiedEpisodeSampler(
        datasets=test_datasets,
        n_shot=meta_cfg["k_shot"],
        n_query=meta_cfg["query_size"],
        n_class=meta_cfg["n_way"],
        seed=42,
    )

    # -- Profiling loop --
    memory_tracker = MemoryTracker(device)
    
    timers = {
        "total": []
    }
    
    results = []
    
    print(f"\nProfiling {n_episodes} episodes...")
    for _ in tqdm(range(n_episodes), desc="Profiling"):
        ep = sampler.sample_episode()
        profile_data = profile_episode(encoder, arc, meta_learner, index, ep,
                                      config, device, memory_tracker, timers)
        results.append(profile_data)

    # -- Summary statistics --
    latencies = np.array([r["total_time"] for r in results])
    memory_usage = np.array([r["peak_memory_mb"] for r in results])
    accuracies = np.array([r["accuracy"] for r in results])
    kappas = np.array([r["kappa"] for r in results])
    
    print("\n" + "="*60)
    print("PROFILING RESULTS")
    print("="*60)
    
    print("\n[LATENCY]")
    print(f"  Total latency        : {np.mean(latencies)*1000:.2f} ± {np.std(latencies)*1000:.2f} ms")
    print(f"  Min                  : {np.min(latencies)*1000:.2f} ms")
    print(f"  Max                  : {np.max(latencies)*1000:.2f} ms")
    print(f"  P50 (median)         : {np.median(latencies)*1000:.2f} ms")
    print(f"  P95                  : {np.percentile(latencies, 95)*1000:.2f} ms")
    
    print("\n[GPU MEMORY]")
    print(f"  Peak memory          : {np.mean(memory_usage):.2f} ± {np.std(memory_usage):.2f} MB")
    print(f"  Min                  : {np.min(memory_usage):.2f} MB")
    print(f"  Max                  : {np.max(memory_usage):.2f} MB")
    
    print("\n[ACCURACY & KAPPA]")
    print(f"  Mean accuracy        : {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"  Mean Cohen's Kappa   : {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")
    print(f"  Accuracy range       : [{np.min(accuracies):.4f}, {np.max(accuracies):.4f}]")
    print(f"  Kappa range          : [{np.min(kappas):.4f}, {np.max(kappas):.4f}]")
    
    # -- Generate plots --
    plot_profiling_results(results)
    
    # -- Save detailed results --
    output_file = "results/profiling_results.json"
    os.makedirs("results", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "summary": {
                "mean_latency_ms": float(np.mean(latencies) * 1000),
                "std_latency_ms": float(np.std(latencies) * 1000),
                "p95_latency_ms": float(np.percentile(latencies, 95) * 1000),
                "mean_memory_mb": float(np.mean(memory_usage)),
                "std_memory_mb": float(np.std(memory_usage)),
                "mean_accuracy": float(np.mean(accuracies)),
                "std_accuracy": float(np.std(accuracies)),
                "mean_kappa": float(np.mean(kappas)),
                "std_kappa": float(np.std(kappas)),
            },
            "per_episode": results
        }, f, indent=2)
    print(f"\n✓ Detailed results saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile ARAML inference")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="results/best_model.pt")
    parser.add_argument("--index_path", default="results/retrieval_index")
    parser.add_argument("--n_episodes", type=int, default=100)
    args = parser.parse_args()
    
    profile_inference(args.config, args.checkpoint, args.n_episodes, args.index_path)
