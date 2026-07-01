"""Collector for Google News RSS.

Why this source first: it needs no API key, returns up to ~100 recent items
for a query, and is a reliable way to satisfy the volume requirement quickly.

What the RSS feed gives us: title, link, publish date, the outlet name, and a
short HTML summary. We strip the HTML and use title + summary as the document
text. NOTE: this is deliberately a first pass. Full-article text extraction
(following the link and pulling the body with trafilatura) is added later as a
separate enrichment step, because Google News links are redirect URLs and need
care to resolve.
"""
from __future__ import annotations

from typing import List, Optional

import feedparser
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

from src.schema import Document
from src.collectors.base import BaseCollector
import config


class GoogleNewsCollector(BaseCollector):
    source = "google_news"
    source_type = "news"

    BASE = "https://news.google.com/rss/search"

    def __init__(self, query: Optional[str] = None,
                 language: str = "en-US", country: str = "US"):
        self.query = query or config.COMPANY_NAME
        self.language = language
        self.country = country

    def _feed_url(self) -> str:
        q = self.query.replace(" ", "+")
        lang_short = self.language.split("-")[0]
        return (f"{self.BASE}?q={q}&hl={self.language}"
                f"&gl={self.country}&ceid={self.country}:{lang_short}")

    @staticmethod
    def _clean(html: str) -> str:
        return BeautifulSoup(html or "", "html.parser").get_text(" ", strip=True)

    def collect(self) -> List[Document]:
        feed = feedparser.parse(self._feed_url())
        docs: List[Document] = []

        for entry in feed.entries:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            if not title or not link:
                continue

            summary = self._clean(getattr(entry, "summary", ""))

            published = None
            raw_published = getattr(entry, "published", None)
            if raw_published:
                try:
                    published = dateparser.parse(raw_published).isoformat()
                except (ValueError, TypeError, OverflowError):
                    published = None

            outlet = ""
            entry_source = getattr(entry, "source", None)
            if entry_source is not None:
                outlet = getattr(entry_source, "title", "") or ""

            text = f"{title}. {summary}".strip()

            docs.append(Document(
                source=self.source,
                source_type=self.source_type,
                title=title,
                text=text,
                url=link,
                published_at=published,
                author=outlet or None,
                metadata={"outlet": outlet, "query": self.query},
            ))

        return docs
