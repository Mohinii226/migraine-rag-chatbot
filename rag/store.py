"""Step 4 of RAG: store embeddings in a vector index and search them.

Search modes
  dense   - meaning-based search with embeddings (FAISS, or NumPy if FAISS is missing)
  keyword - word-based search with TF-IDF (good at exact terms and numbers)
  hybrid  - runs both and merges the two rankings with Reciprocal Rank Fusion (RRF)
"""
import json
from pathlib import Path

import numpy as np

from . import config

try:
    import faiss
except ImportError:  # FAISS missing: the NumPy fallback still works
    faiss = None

RRF_K = 60  # standard constant in Reciprocal Rank Fusion


class VectorStore:
    def __init__(self, embedder, chunks, vectors, model_name=""):
        self.embedder = embedder
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype="float32")
        self.model_name = model_name
        self.index = None
        self._tfidf = None
        self._tfidf_matrix = None
        if faiss is not None:
            # Inner product on normalized vectors == cosine similarity.
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(np.ascontiguousarray(self.vectors))

    @property
    def backend(self):
        return "faiss" if self.index is not None else "numpy"

    @classmethod
    def build(cls, embedder, chunks, show_progress=False):
        if not chunks:
            raise ValueError("No chunks to index.")
        vectors = embedder.encode([c["text"] for c in chunks], show_progress=show_progress)
        return cls(embedder, chunks, vectors, getattr(embedder, "model_name", ""))

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=1), encoding="utf-8")
        (directory / "meta.json").write_text(json.dumps({
            "model": self.model_name, "num_chunks": len(self.chunks),
            "dim": int(self.vectors.shape[1]), "backend": self.backend}), encoding="utf-8")
        if self.index is not None:
            faiss.write_index(self.index, str(directory / "index.faiss"))

    @classmethod
    def load(cls, directory, embedder):
        directory = Path(directory)
        if not (directory / "vectors.npy").exists():
            raise FileNotFoundError(
                f"No index found in {directory}. Run `python ingest.py` first.")
        vectors = np.load(directory / "vectors.npy")
        chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        return cls(embedder, chunks, vectors, meta.get("model", ""))

    # ---- the two rankers -------------------------------------------------
    def _dense_ranked(self, query, n):
        n = min(n, len(self.chunks))
        query_vec = np.ascontiguousarray(self.embedder.encode([query]), dtype="float32")
        if self.index is not None:
            scores, ids = self.index.search(query_vec, n)
            return [(int(i), float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]
        sims = self.vectors @ query_vec[0]
        return [(int(i), float(sims[i])) for i in np.argsort(-sims)[:n]]

    def _keyword_ranked(self, query, n):
        if self._tfidf is None:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                          sublinear_tf=True, stop_words="english")
            self._tfidf_matrix = self._tfidf.fit_transform([c["text"] for c in self.chunks])
        sims = (self._tfidf_matrix @ self._tfidf.transform([query]).T).toarray().ravel()
        order = np.argsort(-sims)[:n]
        return [(int(i), float(sims[i])) for i in order if sims[i] > 0]

    # ---- public search ---------------------------------------------------
    def search(self, query, k=4, mode=None):
        """Return the k best chunks, each with a `score`. `mode` defaults to config.SEARCH_MODE."""
        mode = mode or config.SEARCH_MODE
        k = min(k, len(self.chunks))
        if mode == "dense":
            pairs = self._dense_ranked(query, k)
        elif mode == "keyword":
            pairs = self._keyword_ranked(query, k)
            if len(pairs) < k:  # never return fewer passages than asked for: back-fill with embeddings
                seen = {i for i, _ in pairs}
                for i, score in self._dense_ranked(query, len(self.chunks)):
                    if i not in seen:
                        pairs.append((i, score))
                        seen.add(i)
                    if len(pairs) == k:
                        break
        elif mode == "hybrid":
            n = max(k * 5, 20)
            fused = {}
            for ranked in (self._dense_ranked(query, n), self._keyword_ranked(query, n)):
                for rank, (i, _) in enumerate(ranked):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
            pairs = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        else:
            raise ValueError(f"Unknown search mode {mode!r}. Use dense, keyword or hybrid.")
        return [{"score": score, **self.chunks[i]} for i, score in pairs]
