"""Deduplication.

Two stages exist in the full pipeline:
  1. EXACT dedup (here): drop documents with identical content hashes. Cheap,
     runs at collection time, removes the same article collected twice or
     syndicated under the same URL.
  2. SEMANTIC near-duplicate dedup (added in the indexing stage): drop docs
     whose embeddings are above config.NEAR_DUPLICATE_THRESHOLD cosine
     similarity, catching the same story reworded across outlets.

This file implements stage 1.
"""
from __future__ import annotations

from typing import List

from src.schema import Document


def dedup_exact(docs: List[Document]) -> List[Document]:
    seen: set[str] = set()
    unique: List[Document] = []
    for d in docs:
        if d.doc_id in seen:
            continue
        seen.add(d.doc_id)
        unique.append(d)
    return unique
