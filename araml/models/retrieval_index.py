"""
retrieval_index.py — Cross-Lingual Retrieval Index using FAISS
Stores high-resource language (HRL) embeddings for retrieval.
"""
import torch
import numpy as np
import faiss


class CrossLingualRetrievalIndex:
    def __init__(self, embedding_dim: int = 768, similarity: str = "cosine"):
        self.embedding_dim = embedding_dim
        self.similarity = similarity

        if similarity == "cosine":
            self.index = faiss.IndexFlatIP(embedding_dim)   # Inner product (for normalized = cosine)
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)

        self.texts = []       # Stores raw text
        self.labels = []      # Stores labels
        self.languages = []   # Stores source language tags

    def add(self, embeddings: np.ndarray, texts: list, labels: list, language: str):
        """Add HRL embeddings to the index."""
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        if self.similarity == "cosine":
            faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.labels.extend(labels)
        self.languages.extend([language] * len(texts))

    def retrieve(self, query_embedding: np.ndarray, k: int = 5):
        """
        Retrieve top-k similar examples.
        Returns: distances, retrieved texts, retrieved labels
        """
        query_embedding = np.ascontiguousarray(query_embedding.astype(np.float32))
        if self.similarity == "cosine":
            faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, k)

        retrieved = {
            "distances": distances[0],
            "texts": [self.texts[i] for i in indices[0]],
            "labels": [self.labels[i] for i in indices[0]],
            "languages": [self.languages[i] for i in indices[0]],
            "indices": indices[0]
        }
        return retrieved

    def save(self, path: str):
        faiss.write_index(self.index, path + ".faiss")
        np.save(path + "_meta.npy", {
            "texts": self.texts,
            "labels": self.labels,
            "languages": self.languages
        })

    def load(self, path: str):
        self.index = faiss.read_index(path + ".faiss")
        meta = np.load(path + "_meta.npy", allow_pickle=True).item()
        self.texts = meta["texts"]
        self.labels = meta["labels"]
        self.languages = meta["languages"]

    def __len__(self):
        return self.index.ntotal