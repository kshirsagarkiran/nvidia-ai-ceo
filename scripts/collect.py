"""Entry point: run every registered collector, deduplicate, and save.

Usage (from the project root):
    python scripts/collect.py

As we add sources, we simply append them to the COLLECTORS list below; nothing
else in this script changes, because every collector returns the same Document
type. This is the payoff of the BaseCollector contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collectors.google_news import GoogleNewsCollector
from src.collectors.sec_edgar import SECEdgarCollector
from src.collectors.hacker_news import HackerNewsCollector
from src.dedup import dedup_exact
from src.storage_raw import save_jsonl
import config


# Register collectors here. Three INDEPENDENT source types now satisfy the rubric:
#   news (Google News) · financial filings (SEC EDGAR) · community (Hacker News).
COLLECTORS = [
    # --- News ---
    GoogleNewsCollector(query=config.COMPANY_NAME),
    GoogleNewsCollector(query=f"{config.COMPANY_NAME} AI chips"),
    GoogleNewsCollector(query=f"{config.COMPANY_NAME} competitors AMD Intel"),
    # --- Financial filings (highest-quality evidence) ---
    SECEdgarCollector(max_filings=25),
    # --- Community / technology sentiment ---
    HackerNewsCollector(query=config.COMPANY_NAME, max_results=50),
    HackerNewsCollector(query=f"{config.COMPANY_NAME} GPU", max_results=30),
]


def main() -> None:
    all_docs = []
    for collector in COLLECTORS:
        try:
            docs = collector.collect()
            print(f"[{collector.source}] collected {len(docs)} documents")
            all_docs.extend(docs)
        except Exception as exc:  # noqa: BLE001 - we want collection to be resilient
            print(f"[{collector.source}] FAILED: {exc}")

    before = len(all_docs)
    all_docs = dedup_exact(all_docs)
    print(f"deduplicated: {before} -> {len(all_docs)} unique documents")

    by_source: dict[str, int] = {}
    for d in all_docs:
        by_source[d.source] = by_source.get(d.source, 0) + 1
    print("by source:", by_source)

    path = save_jsonl(all_docs)
    print(f"saved {len(all_docs)} documents to {path}")


if __name__ == "__main__":
    main()
