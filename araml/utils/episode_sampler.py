"""
episode_sampler.py — N-way K-shot episode sampler for meta-learning
"""
import random
from typing import List, Dict


class EpisodeSampler:
    def __init__(self, dataset: list, n_way: int = 5, k_shot: int = 5, query_size: int = 15):
        """
        Args:
            dataset: List of {"text": str, "label": int, "language": str}
            n_way: Number of classes per episode
            k_shot: Support examples per class
            query_size: Query examples per class
        """
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size

        # Group by label
        self.label_to_samples = {}
        for item in dataset:
            label = item["label"]
            if label not in self.label_to_samples:
                self.label_to_samples[label] = []
            self.label_to_samples[label].append(item)

        self.all_labels = list(self.label_to_samples.keys())

    def sample_episode(self) -> Dict:
        """Sample one N-way K-shot episode."""
        selected_labels = random.sample(self.all_labels, self.n_way)
        label_map = {orig: idx for idx, orig in enumerate(selected_labels)}

        support_texts, support_labels = [], []
        query_texts, query_labels = [], []

        for label in selected_labels:
            samples = self.label_to_samples[label]
            needed = self.k_shot + self.query_size
            if len(samples) < needed:
                samples = random.choices(samples, k=needed)
            else:
                samples = random.sample(samples, needed)

            for item in samples[:self.k_shot]:
                support_texts.append(item["text"])
                support_labels.append(label_map[label])

            for item in samples[self.k_shot:]:
                query_texts.append(item["text"])
                query_labels.append(label_map[label])

        return {
            "support_texts": support_texts,
            "support_labels": support_labels,
            "query_texts": query_texts,
            "query_labels": query_labels,
            "label_map": label_map
        }

    def __iter__(self):
        while True:
            yield self.sample_episode()
