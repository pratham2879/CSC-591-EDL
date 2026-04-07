"""
faiss_index.py — FAISS retrieval index for high-resource cross-lingual examples.

Stores pre-computed XLM-R embeddings from HIGH_RESOURCE training splits
(en, de, es, fr).  Used by ARC during meta-learning to retrieve semantically
similar examples for the query language (ja / zh).

Design constraints:
  - Index contains ONLY high-resource language examples (asserted at build time).
  - No ja/zh training pool examples appear in the index (leakage guard).
  - Query is detached before FAISS search (search is non-differentiable);
    the returned embedding tensors are new leaf tensors placed on the
    training device so ARC's attention scorer can compute gradients through them.

Disk layout (all under one directory):
  faiss.index        — the FAISS flat L2 index
  embeddings.npy     — (N, D) float32 matrix of all stored embeddings
  metadata.json      — [{text, language, label, product_category}, ...]
"""
import os
import json
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

HIGH_RESOURCE = ("en", "de", "es", "fr")

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning(
        "faiss-cpu not installed. FaissRetriever will use a random-fallback "
        "dummy retriever. Install with: pip install faiss-cpu"
    )


class FaissRetriever:
    """
    Build, persist, and query a FAISS flat-L2 index over high-resource embeddings.

    If faiss is not installed the retriever operates in dummy mode: retrieve()
    returns random unit-norm vectors.  This allows smoke tests and gradient
    flow verification to run without the full data pipeline.
    """

    def __init__(self, hidden_dim: int = 768):
        self.hidden_dim   = hidden_dim
        self._embeddings  = None   # np.ndarray (N, D)
        self._metadata    = []     # list of dicts
        self._index       = None   # faiss index
        self._available   = _FAISS_AVAILABLE
        self._dummy_mode  = not _FAISS_AVAILABLE

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build(
        self,
        records: list,
        encoder,
        device: torch.device,
        batch_size: int = 64,
    ) -> None:
        """
        Encode all records and build the FAISS index.

        Args:
            records   : list of dicts with 'text' and 'language' fields.
                        MUST be from HIGH_RESOURCE languages only.
            encoder   : TextEncoder (araml.models.encoder)
            device    : torch device
            batch_size: encoding batch size
        """
        # Safety assertion — no low-resource examples in the index
        bad = [r for r in records if r.get("language") not in HIGH_RESOURCE]
        assert len(bad) == 0, (
            f"ASSERTION FAILED: {len(bad)} non-high-resource records passed to "
            f"FaissRetriever.build(). Languages found: "
            f"{set(r['language'] for r in bad)}"
        )

        encoder.eval()
        all_embs = []

        texts = [r["text"] for r in records]
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embs  = encoder.encode_text(batch, device)   # (B, D)
            all_embs.append(embs.cpu().numpy().astype(np.float32))
            if (i // batch_size) % 50 == 0:
                logger.info("  FAISS build: %d / %d encoded", i + len(batch), len(texts))

        self._embeddings = np.concatenate(all_embs, axis=0)   # (N, D)
        self._metadata   = [
            {
                "text":             r["text"],
                "language":         r["language"],
                "label":            r["label"],
                "product_category": r.get("product_category", "unknown"),
            }
            for r in records
        ]

        if self._available:
            self._index = faiss.IndexFlatL2(self.hidden_dim)
            self._index.add(self._embeddings)
            logger.info(
                "FAISS index built: %d vectors (dim=%d)", self._index.ntotal, self.hidden_dim
            )
        else:
            logger.info(
                "Dummy FAISS: stored %d embeddings (faiss not available)", len(self._embeddings)
            )
            self._dummy_mode = False   # real embeddings available, just no faiss search

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_emb: torch.Tensor,
        k: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Retrieve the k nearest embeddings to query_emb.

        The query is detached before search (FAISS is non-differentiable).
        The returned embedding tensor IS on `device` and is a fresh leaf
        tensor — ARC's attention scorer will compute gradients through it.

        Args:
            query_emb : (1, D) tensor (will be detached for search)
            k         : number of neighbors to retrieve
            device    : device to place retrieved embeddings on

        Returns:
            retrieved : (k, D) float tensor on `device`
        """
        if self._dummy_mode or self._embeddings is None:
            # No real data — return random unit-norm vectors (for smoke tests)
            rand = torch.randn(k, self.hidden_dim, device=device)
            return rand / (rand.norm(dim=-1, keepdim=True) + 1e-8)

        q = query_emb.detach().cpu().numpy().astype(np.float32)   # (1, D)

        if self._available and self._index is not None:
            # True FAISS nearest-neighbour search
            k_actual = min(k, self._index.ntotal)
            _, indices = self._index.search(q, k_actual)           # (1, k)
            idx = indices[0]
        else:
            # Fallback: brute-force cosine (when embeddings stored but no faiss)
            q_t    = torch.from_numpy(q)                           # (1, D)
            emb_t  = torch.from_numpy(self._embeddings)            # (N, D)
            dists  = ((emb_t - q_t) ** 2).sum(-1)                 # (N,)
            k_actual = min(k, len(self._embeddings))
            idx    = dists.topk(k_actual, largest=False).indices.numpy()

        retrieved = self._embeddings[idx]                           # (k, D)
        return torch.from_numpy(retrieved).float().to(device)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)

        np.save(os.path.join(directory, "embeddings.npy"), self._embeddings)
        with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False)

        if self._available and self._index is not None:
            faiss.write_index(self._index, os.path.join(directory, "faiss.index"))

        logger.info("FaissRetriever saved to %s (%d vectors)", directory, len(self._metadata))

    def load(self, directory: str, device: torch.device = None) -> None:
        emb_path  = os.path.join(directory, "embeddings.npy")
        meta_path = os.path.join(directory, "metadata.json")
        idx_path  = os.path.join(directory, "faiss.index")

        self._embeddings = np.load(emb_path).astype(np.float32)
        with open(meta_path, encoding="utf-8") as f:
            self._metadata = json.load(f)

        if self._available and os.path.exists(idx_path):
            self._index = faiss.read_index(idx_path)
        else:
            self._dummy_mode = False   # embeddings loaded, brute-force fallback

        logger.info(
            "FaissRetriever loaded from %s (%d vectors)", directory, len(self._metadata)
        )

    # ------------------------------------------------------------------
    # Integrity assertions (call before training)
    # ------------------------------------------------------------------

    def assert_high_resource_only(self) -> None:
        """Assert every stored record is from a HIGH_RESOURCE language."""
        if not self._metadata:
            logger.warning("assert_high_resource_only: index is empty, nothing to check.")
            return
        bad = [m for m in self._metadata if m.get("language") not in HIGH_RESOURCE]
        assert len(bad) == 0, (
            f"FAISS index contains {len(bad)} non-high-resource records: "
            f"{set(m['language'] for m in bad)}"
        )
        print(f"ASSERT PASSED: FAISS index contains only high-resource examples "
              f"({len(self._metadata)} total).")

    def assert_no_lowresource_leakage(self, lowresource_pool_texts: set) -> None:
        """Assert no low-resource training text appears in the FAISS index."""
        if not self._metadata:
            return
        leaked = [m for m in self._metadata if m["text"] in lowresource_pool_texts]
        assert len(leaked) == 0, (
            f"DATA LEAKAGE: {len(leaked)} low-resource training examples found "
            f"in FAISS index."
        )
        print(f"ASSERT PASSED: No low-resource training texts in FAISS index.")

    @property
    def size(self) -> int:
        return len(self._metadata) if self._metadata else 0
