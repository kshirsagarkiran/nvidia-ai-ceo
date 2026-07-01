"""The normalized document schema.

This is the single most important design decision in the data layer: every
collector, regardless of source (news, SEC filing, Reddit, etc.), MUST emit
objects in this exact shape. That one shared contract is what makes
deduplication, embedding, indexing, retrieval and (crucially) evidence
citation uniform across the whole system. If a source can't be mapped into
this schema, it doesn't enter the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
from typing import Optional


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Document:
    source: str                          # specific source, e.g. "google_news"
    source_type: str                     # category, e.g. "news" / "filing" / "community"
    title: str
    text: str                            # the body we embed and later cite as evidence
    url: str
    published_at: Optional[str] = None   # ISO-8601 timestamp if known
    author: Optional[str] = None
    collected_at: str = field(default_factory=_utcnow_iso)
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""                     # stable content hash (filled automatically)

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = self.content_hash()

    def content_hash(self) -> str:
        """Stable ID derived from normalized title + url.

        Used for exact deduplication: the same article re-collected on a later
        run produces the same id, so we never store it twice.
        """
        basis = f"{self.title.strip().lower()}|{self.url.strip().lower()}"
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)
