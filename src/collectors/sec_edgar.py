"""Collector for SEC EDGAR filings (no API key required).

This is the highest-quality evidence source in the corpus: NVIDIA's own regulatory
filings. We use the EDGAR submissions API to list recent filings, then fetch the
primary document of each filing of interest (8-K = current events, 10-Q = quarterly,
10-K = annual) and extract its text.

SEC requires a descriptive User-Agent with a contact address (set in config.USER_AGENT)
and asks clients to stay under 10 requests/second.
"""
from __future__ import annotations

from typing import List
import time

import requests
from bs4 import BeautifulSoup

from src.schema import Document
from src.collectors.base import BaseCollector
import config


class SECEdgarCollector(BaseCollector):
    source = "sec_edgar"
    source_type = "filing"

    SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
    ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{doc}"

    def __init__(self, cik: str | None = None,
                 forms=("8-K", "10-Q", "10-K"), max_filings: int = 25):
        self.cik = (cik or config.COMPANY_CIK).zfill(10)   # EDGAR wants 10 digits
        self.cik_int = int(self.cik)                       # archive paths use the un-padded int
        self.forms = set(forms)
        self.max_filings = max_filings
        self.headers = {"User-Agent": config.USER_AGENT}

    def _get(self, url: str):
        resp = requests.get(url, headers=self.headers, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp

    def collect(self) -> List[Document]:
        data = self._get(self.SUBMISSIONS.format(cik=self.cik)).json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        docs: List[Document] = []
        for form, date, accession, primary in zip(forms, dates, accessions, primary_docs):
            if form not in self.forms or not primary:
                continue
            if len(docs) >= self.max_filings:
                break

            accession_nodash = accession.replace("-", "")
            url = self.ARCHIVE.format(cik_int=self.cik_int, accession=accession_nodash, doc=primary)
            try:
                html = self._get(url).text
            except requests.RequestException:
                continue
            time.sleep(0.2)   # stay polite with SEC's rate limit

            text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
            text = text[:8000]   # filings can be huge; keep a leading slice (chunked later at indexing)
            if not text:
                continue

            docs.append(Document(
                source=self.source,
                source_type=self.source_type,
                title=f"{config.COMPANY_NAME} {form} ({date})",
                text=text,
                url=url,
                published_at=date,
                author="SEC EDGAR",
                metadata={"form": form, "accession": accession},
            ))
        return docs
