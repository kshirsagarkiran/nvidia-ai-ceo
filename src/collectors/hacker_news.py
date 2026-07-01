"""Collector for Hacker News (no API key required).

Hacker News is a strong, reliable "community / technology" source for a company like
NVIDIA: stories and discussion from a technical audience. We use the public Algolia
search API (no auth, no rate-key needed for modest use).
"""
from __future__ import annotations

from typing import List
from datetime import datetime, timezone

import requests

from src.schema import Document
from src.collectors.base import BaseCollector
import config


class HackerNewsCollector(BaseCollector):
    source = "hacker_news"
    source_type = "community"

    SEARCH = "https://hn.algolia.com/api/v1/search_by_date"

    def __init__(self, query: str | None = None, max_results: int = 50, min_points: int = 0):
        self.query = query or config.COMPANY_NAME
        self.max_results = max_results
        self.min_points = min_points

    def collect(self) -> List[Document]:
        params = {"query": self.query, "tags": "story", "hitsPerPage": self.max_results}
        resp = requests.get(self.SEARCH, params=params,
                            headers={"User-Agent": config.USER_AGENT},
                            timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        hits = resp.json().get("hits", [])

        docs: List[Document] = []
        for h in hits:
            title = (h.get("title") or "").strip()
            if not title:
                continue
            if (h.get("points") or 0) < self.min_points:
                continue

            object_id = h.get("objectID", "")
            url = h.get("url") or f"https://news.ycombinator.com/item?id={object_id}"
            body = (h.get("story_text") or "").strip()
            text = f"{title}. {body}".strip()

            published = None
            ts = h.get("created_at_i")
            if ts:
                published = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

            docs.append(Document(
                source=self.source,
                source_type=self.source_type,
                title=title,
                text=text,
                url=url,
                published_at=published,
                author=h.get("author"),
                metadata={"points": h.get("points"), "num_comments": h.get("num_comments"),
                          "hn_id": object_id},
            ))
        return docs
