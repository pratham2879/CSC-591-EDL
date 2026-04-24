"""
episode_sampler.py — Category-stratified episode sampler for meta-learning (REGRESSION).

FIX 3 — Category-stratified episode construction:
  Sampling order per episode:
    1. Sample a language  (ja or zh, uniformly at random)
    2. Sample a category  (uniformly at random from categories that have enough
                           examples)
    3. Sample support and query sets ONLY from that (language, category) pool.
    
  For regression, we don't need to balance by class, but we still need enough
  examples per category to form valid episodes.

FIX 2 enforcement:
  Only LOW_RESOURCE languages (ja, zh) appear in the episode support/query sets.
  High-resource languages are NEVER passed to this sampler.

Logging:
  Every 100 episodes: logs sampled language, category, and support/query sizes
  to stdout so bad sampling is caught early.
"""
import random
import logging
from collections import defaultdict, Counter
from typing import Iterator

logger = logging.getLogger(__name__)

LOW_RESOURCE = ("ja", "zh")


class CategoryStratifiedEpisodeSampler:
    """
    Few-shot episode sampler with category stratification (REGRESSION).

    Parameters
    ----------
    datasets : dict[str, list[dict]]
        {lang: records} where each record must contain:
          "text"             : str
          "label"            : float  (normalized to [0, 1])
          "product_category" : str
        Only LOW_RESOURCE languages (ja, zh) should be passed here.
    n_shot   : int   Support examples (default 5).
    n_query  : int   Total query examples per episode.
    n_class  : int   Historical parameter (not used for regression, kept for compatibility).
    seed     : int   RNG seed for reproducibility.
    log_every: int   Log episode stats every N episodes.
    """

    def __init__(
        self,
        datasets: dict,
        n_shot:    int = 5,
        n_query:   int = 10,
        n_class:   int = 2,
        seed:      int = 42,
        log_every: int = 100,
    ):
        # Note: n_class is kept for API compatibility but not used in regression
        for lang in datasets:
            assert lang in LOW_RESOURCE, (
                f"Language '{lang}' is not a low-resource language. "
                f"Only {LOW_RESOURCE} may be used as query/test languages."
            )

        self.n_shot    = n_shot
        self.n_query   = n_query
        self.n_class   = n_class  # Kept for API compatibility
        self.log_every = log_every
        self.rng       = random.Random(seed)
        self._episode_count = 0

        # Minimum examples needed per category to form valid episodes
        self._min_per_category = n_shot + n_query

        # Build index: {lang: {category: [records]}}
        # For regression, we don't separate by class; we just track all records per category.
        self._index: dict[str, dict[str, list]] = {}
        self._valid_lang_cats: dict[str, list[str]] = {}

        for lang, records in datasets.items():
            cat_records: dict[str, list] = defaultdict(list)
            for r in records:
                cat_records[r["product_category"]].append(r)

            valid_cats = []
            for cat, examples in cat_records.items():
                if len(examples) >= self._min_per_category:
                    valid_cats.append(cat)

            self._index[lang]           = dict(cat_records)
            self._valid_lang_cats[lang] = valid_cats

            total_cats = len(cat_records)
            print(
                f"[EpisodeSampler] {lang}: {len(valid_cats)}/{total_cats} categories valid "
                f"for {n_shot}-shot regression episodes "
                f"(need >={self._min_per_category} examples per category)"
            )

        self._languages = [lg for lg in datasets if self._valid_lang_cats.get(lg)]
        if not self._languages:
            raise ValueError(
                "No valid (language, category) combinations found. "
                "Check that the low-resource pools contain enough per-category examples."
            )

    # ------------------------------------------------------------------
    # Core sampling
    # ------------------------------------------------------------------

    def sample_episode(self) -> dict:
        """
        Sample one regression few-shot episode, category-stratified.

        Returns
        -------
        dict with keys:
          support_texts   : list[str]   length = n_shot
          support_labels  : list[float] normalized to [0, 1]
          query_texts     : list[str]   length = n_query
          query_labels    : list[float] normalized to [0, 1]
          language        : str         ja or zh
          category        : str         sampled product category
        """
        max_retries = 200
        for attempt in range(max_retries):
            # 1. Sample language
            lang = self.rng.choice(self._languages)
            valid_cats = self._valid_lang_cats[lang]
            if not valid_cats:
                continue

            # 2. Sample category
            cat = self.rng.choice(valid_cats)
            examples = self._index[lang].get(cat, [])

            # Double-check availability (pool may shrink in future extensions)
            if len(examples) < self._min_per_category:
                continue

            # 3. Sample support + query without replacement
            needed = self.n_shot + self.n_query
            sampled = self.rng.sample(examples, needed)
            
            support_texts  = [sampled[i]["text"] for i in range(self.n_shot)]
            support_labels = [sampled[i]["label"] for i in range(self.n_shot)]
            
            query_texts    = [sampled[i]["text"] for i in range(self.n_shot, needed)]
            query_labels   = [sampled[i]["label"] for i in range(self.n_shot, needed)]

            self._episode_count += 1

            # 4. Shuffle BOTH support and query sets independently so the model
            #    doesn't learn positional shortcuts.
            combined_support = list(zip(support_texts, support_labels))
            self.rng.shuffle(combined_support)
            support_texts, support_labels = zip(*combined_support)
            support_texts  = list(support_texts)
            support_labels = list(support_labels)

            combined_query = list(zip(query_texts, query_labels))
            self.rng.shuffle(combined_query)
            query_texts, query_labels = zip(*combined_query)
            query_texts  = list(query_texts)
            query_labels = list(query_labels)

            # 5. Log every `log_every` episodes
            if self._episode_count % self.log_every == 0:
                support_mean = sum(support_labels) / len(support_labels)
                query_mean = sum(query_labels) / len(query_labels)
                logger.info(
                    "[Episode %d] lang=%s  category='%s'  "
                    "support_mean_label=%.3f  query_mean_label=%.3f",
                    self._episode_count, lang, cat, support_mean, query_mean
                )

            return {
                "support_texts":  support_texts,
                "support_labels": support_labels,
                "query_texts":    query_texts,
                "query_labels":   query_labels,
                "language":       lang,
                "category":       cat,
            }

        raise RuntimeError(
            f"Failed to sample valid episode after {max_retries} attempts. "
            "Check pool structure."
        )

    def __iter__(self) -> Iterator[dict]:
        while True:
            yield self.sample_episode()

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_pool_files(
        cls,
        pool_dir:  str = "data",
        languages: tuple = ("ja", "zh"),
        **kwargs,
    ) -> "CategoryStratifiedEpisodeSampler":
        """
        Load low-resource training pools from disk and build the sampler.

        Expects files: {pool_dir}/lowresource_pool_{lang}.json
        """
        import json, os
        datasets = {}
        for lang in languages:
            path = os.path.join(pool_dir, f"lowresource_pool_{lang}.json")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Low-resource pool not found: {path}. "
                    f"Run data/preprocess.py first."
                )
            with open(path, encoding="utf-8") as f:
                datasets[lang] = json.load(f)
            print(f"[EpisodeSampler] Loaded {len(datasets[lang])} {lang} examples from {path}")
        return cls(datasets, **kwargs)
