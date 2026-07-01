"""Indexing: turn raw documents into a searchable knowledge repository.

Pipeline: load raw JSONL -> split each document into overlapping word chunks ->
embed every chunk with bge-base-en-v1.5 -> store chunks + embeddings + metadata in
a persistent Chroma collection.

Chunking matters because (a) long SEC filings must be broken up to fit the embedding
model's context and to retrieve the *relevant passage* rather than a whole 50-page
filing, and (b) each chunk keeps a pointer back to its source document so every
retrieved piece of evidence can be cited.
"""
from __future__ import annotations

from typing import List, Dict
import chromadb

from src.storage_raw import load_jsonl
import config


def chunk_text(text: str, size: int = None, overlap: int = None) -> List[str]:
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    words = text.split()
    if len(words) <= size:
        return [text] if text.strip() else []
    chunks, i = [], 0
    step = max(1, size - overlap)
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += step
    return chunks


def build_chunks(docs: List[dict]) -> Dict[str, list]:
    """Return parallel lists of chunk ids / texts / metadatas ready for Chroma."""
    ids, texts, metas = [], [], []
    for d in docs:
        for j, chunk in enumerate(chunk_text(d["text"])):
            ids.append(f"{d['doc_id']}_{j}")
            texts.append(chunk)
            metas.append({
                "doc_id": d["doc_id"],
                "source": d["source"],
                "source_type": d["source_type"],
                "title": d.get("title", ""),
                "url": d.get("url", ""),
                "published_at": d.get("published_at") or "",
            })
    return {"ids": ids, "texts": texts, "metas": metas}


def build_index() -> int:
    """Build (or rebuild) the Chroma knowledge base from the raw JSONL store."""
    docs = load_jsonl()
    chunks = build_chunks(docs)
    print(f"{len(docs)} documents -> {len(chunks['ids'])} chunks")

    from sentence_transformers import SentenceTransformer  # lazy import: keeps chunking torch-free
    # device="cpu": the bge model can emit NaN embeddings on Apple Silicon's MPS/Metal
    # backend; CPU is numerically safe and plenty fast for this corpus.
    embedder = SentenceTransformer(config.EMBED_MODEL, device="cpu")
    # passages are embedded WITHOUT the query prefix; embeddings are normalized for cosine.
    embeddings = embedder.encode(chunks["texts"], normalize_embeddings=True,
                                 show_progress_bar=True, batch_size=32).tolist()

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    # start clean so re-runs don't accumulate duplicates
    try:
        client.delete_collection(config.COLLECTION)
    except Exception:
        pass
    col = client.create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(ids=chunks["ids"], documents=chunks["texts"],
            embeddings=embeddings, metadatas=chunks["metas"])
    print(f"indexed {col.count()} chunks into Chroma at {config.CHROMA_DIR}")
    return col.count()
