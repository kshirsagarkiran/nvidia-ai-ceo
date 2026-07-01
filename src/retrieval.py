"""Hybrid retrieval: fuse keyword (BM25) and semantic (dense) search.

This is the retriever the C-suite agents use. It mirrors the course MiniHackathon
formula:

    score_hybrid = alpha * dense + (1 - alpha) * sparse

Keyword search nails exact terms (tickers, "H100", "Blackwell"); semantic search
catches paraphrase ("GPU shortage" ~ "supply constraints"). Each returned chunk
carries its source metadata so the answer can cite it as evidence.
"""
from __future__ import annotations

from typing import List, Dict, Optional
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import config


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


class HybridRetriever:
    def __init__(self, alpha: float = None):
        self.alpha = config.HYBRID_ALPHA if alpha is None else alpha

        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        col = client.get_collection(config.COLLECTION)
        got = col.get(include=["documents", "embeddings", "metadatas"])

        self.ids = got["ids"]
        self.docs = got["documents"]
        self.metas = got["metadatas"]
        emb = np.asarray(got["embeddings"], dtype="float32")
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)   # guard against bad rows
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        self.emb = emb / np.where(norms > 0, norms, 1.0)

        self.bm25 = BM25Okapi([d.lower().split() for d in self.docs])
        # device="cpu": avoids NaN embeddings from Apple Silicon's MPS backend.
        self.embedder = SentenceTransformer(config.EMBED_MODEL, device="cpu")

    def query(self, text: str, k: int = 6, alpha: Optional[float] = None,
              source_type: Optional[str] = None) -> List[Dict]:
        alpha = self.alpha if alpha is None else alpha

        # dense scores (query gets the bge instruction prefix)
        qv = self.embedder.encode([config.BGE_QUERY_PREFIX + text],
                                  normalize_embeddings=True)[0]
        qv = np.nan_to_num(np.asarray(qv, dtype="float32"), nan=0.0, posinf=0.0, neginf=0.0)
        # NOTE: use an explicit element-wise multiply + sum instead of `self.emb @ qv`.
        # On Apple Silicon, NumPy's BLAS (Accelerate) matmul can emit spurious
        # divide-by-zero/overflow/invalid warnings and return wrong values; this path avoids it.
        dense = np.nan_to_num((self.emb * qv).sum(axis=1))

        # sparse scores
        sparse = np.array(self.bm25.get_scores(text.lower().split()))

        fused = alpha * _minmax(dense) + (1 - alpha) * _minmax(sparse)

        # optional metadata filter (e.g. only financial filings for the CFO agent)
        order = np.argsort(fused)[::-1]
        results = []
        for i in order:
            if source_type and self.metas[i].get("source_type") != source_type:
                continue
            results.append({
                "chunk_id": self.ids[i],
                "text": self.docs[i],
                "score": float(fused[i]),
                "dense": float(_minmax(dense)[i]),
                "sparse": float(_minmax(sparse)[i]),
                "source": self.metas[i].get("source"),
                "source_type": self.metas[i].get("source_type"),
                "title": self.metas[i].get("title"),
                "url": self.metas[i].get("url"),
                "doc_id": self.metas[i].get("doc_id"),
            })
            if len(results) >= k:
                break
        return results
