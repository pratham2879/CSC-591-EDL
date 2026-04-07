"""
episode_sampler.py — Category-stratified binary episode sampler for meta-learning.

FIX 3 — Category-stratified episode construction:
  Sampling order per episode:
    1. Sample a language  (ja or zh, uniformly at random)
    2. Sample a category  (uniformly at random from categories that have enough
                           examples for BOTH classes)
    3. Sample support and query sets ONLY from that (language, category) pool,
       ensuring exact class balance: n_shot examples per class for support,
       n_query // n_class examples per class for query.

  If a category has fewer than (n_shot * n_class + n_query) examples in the
  low-resource pool, it is excluded during index construction (not retried
  at sample time, which would waste GPU time).

FIX 2 enforcement:
  Only LOW_RESOURCE languages (ja, zh) appear in the episode support/query sets.
  High-resource languages are NEVER passed to this sampler.

Logging:
  Every 100 episodes: logs sampled language, category, and support class
  distribution to stdout so bad sampling is caught early.
"""
import random
import logging
from collections import defaultdict, Counter
from typing import Iterator

logger = logging.getLogger(__name__)

LOW_RESOURCE = ("ja", "zh")


class CategoryStratifiedEpisodeSampler:
    """
    Binary few-shot episode sampler with category stratification.

    Parameters
    ----------
    datasets : dict[str, list[dict]]
        {lang: records} where each record must contain:
          "text"             : str
          "label"            : int  (0 or 1, after label-2 drop)
          "product_category" : str
        Only LOW_RESOURCE languages (ja, zh) should be passed here.
    n_shot   : int   Support examples per class (default 5 → 5-shot).
    n_query  : int   Total query examples per episode (split equally across classes).
    n_class  : int   Fixed at 2 for binary sentiment; kept as a parameter for clarity.
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
        assert n_class == 2, "This sampler is hard-coded for binary classification (n_class=2)."
        for lang in datasets:
            assert lang in LOW_RESOURCE, (
                f"Language '{lang}' is not a low-resource language. "
                f"Only {LOW_RESOURCE} may be used as query/test languages."
            )

        self.n_shot    = n_shot
        self.n_query   = n_query
        self.n_class   = n_class
        self.log_every = log_every
        self.rng       = random.Random(seed)
        self._episode_count = 0

        # Per class in a balanced episode
        self._n_support_per_class = n_shot
        self._n_query_per_class   = max(1, n_query // n_class)
        self._min_per_class       = self._n_support_per_class + self._n_query_per_class

        # Build index: {lang: {category: {label: [records]}}}
        # Only keep categories where BOTH labels have >= _min_per_class examples.
        self._index: dict[str, dict[str, dict[int, list]]] = {}
        self._valid_lang_cats: dict[str, list[str]] = {}

        for lang, records in datasets.items():
            cat_cls: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
            for r in records:
                cat_cls[r["product_category"]][r["label"]].append(r)

            valid_cats = []
            for cat, cls_map in cat_cls.items():
                # Both labels must be present with enough examples
                if len(cls_map) < n_class:
                    continue
                if all(len(cls_map[lbl]) >= self._min_per_class for lbl in (0, 1)):
                    valid_cats.append(cat)

            self._index[lang]           = {c: dict(cm) for c, cm in cat_cls.items()}
            self._valid_lang_cats[lang] = valid_cats

            total_cats = len(cat_cls)
            print(
                f"[EpisodeSampler] {lang}: {len(valid_cats)}/{total_cats} categories valid "
                f"for {n_shot}-shot binary episodes "
                f"(need >={self._min_per_class} examples per class per category)"
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
        Sample one binary few-shot episode, category-stratified.

        Returns
        -------
        dict with keys:
          support_texts   : list[str]   length = n_shot * n_class
          support_labels  : list[int]   0 or 1, class-balanced
          query_texts     : list[str]   length = n_query_per_class * n_class
          query_labels    : list[int]   0 or 1, class-balanced
          language        : str         ja or zh
          category        : str         sampled product category
          n_class         : int         always 2
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
            cls_map = self._index[lang].get(cat, {})

            # Double-check class availability (pool may shrink in future extensions)
            if not all(
                len(cls_map.get(lbl, [])) >= self._min_per_class
                for lbl in (0, 1)
            ):
                continue

            # 3. Sample support + query per class, no overlap
            support_texts, support_labels = [], []
            query_texts,   query_labels   = [], []

            ok = True
            for lbl in (0, 1):
                examples = cls_map[lbl]
                needed   = self._n_support_per_class + self._n_query_per_class
                if len(examples) < needed:
                    ok = False
                    break
                sampled  = self.rng.sample(examples, needed)
                for item in sampled[:self._n_support_per_class]:
                    support_texts.append(item["text"])
                    support_labels.append(lbl)
                for item in sampled[self._n_support_per_class:]:
                    query_texts.append(item["text"])
                    query_labels.append(lbl)

            if not ok:
                continue

            self._episode_count += 1

            # 4. Shuffle BOTH support and query sets independently so labels are
            #    never in sorted order [0,0,...,1,1,...].
            #    The MAML inner loop trains on support; the outer loop trains on query.
            #    A positional shortcut is equally learnable from either — both must be shuffled.
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
                support_dist = dict(Counter(support_labels))
                logger.info(
                    "[Episode %d] lang=%s  category='%s'  "
                    "support_class_dist=%s  "
                    "(neg=%d, pos=%d)",
                    self._episode_count, lang, cat,
                    support_dist,
                    support_dist.get(0, 0),
                    support_dist.get(1, 0),
                )
                print(
                    f"[Episode {self._episode_count}] lang={lang}  "
                    f"category='{cat}'  support_class_dist={support_dist}"
                )

            return {
                "support_texts":  support_texts,
                "support_labels": support_labels,
                "query_texts":    query_texts,
                "query_labels":   query_labels,
                "language":       lang,
                "category":       cat,
                "n_class":        self.n_class,
            }

        raise RuntimeError(
            f"Could not sample a valid episode after {max_retries} attempts. "
            f"Verify that the low-resource pools have enough per-category, "
            f"per-class examples (need >= {self._min_per_class} each)."
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
