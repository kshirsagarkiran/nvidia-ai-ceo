"""Quick check that hybrid retrieval works end to end.

Run AFTER build_index.py, from the project root:
    python scripts/smoke_test.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import HybridRetriever

QUERIES = [
    "NVIDIA data center revenue growth",
    "competition from AMD and custom AI chips",
    "supply chain and export restriction risks",
]


def main() -> None:
    r = HybridRetriever()
    for q in QUERIES:
        print(f"\n=== {q} ===")
        for h in r.query(q, k=3):
            title = (h.get("title") or "")[:42]
            print(f"[{h['score']:.2f}] {h['source']:<12} {title}")
            print(f"        {h['text'][:90]}")


if __name__ == "__main__":
    main()
