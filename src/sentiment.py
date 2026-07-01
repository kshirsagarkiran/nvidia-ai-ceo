"""Sentiment analysis over the collected documents.

Uses VADER (a lexicon-based analyzer) rather than a Transformer: it needs no model
download, is fast, and is tuned for short news/social text — a sensible choice for a
fanless laptop. We report sentiment per source type (news vs community vs filing) and
over time, which feeds the dashboard's Sentiment section.
"""
from __future__ import annotations

from typing import List, Dict
from collections import defaultdict


def compute_sentiment(docs: List[dict]) -> Dict:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    by_type = defaultdict(list)
    timeline = defaultdict(list)
    for d in docs:
        score = sia.polarity_scores(d.get("text", "")[:500])["compound"]
        by_type[d.get("source_type", "unknown")].append(score)
        date = (d.get("published_at") or "")[:10]
        if date:
            timeline[date].append(score)

    def avg(xs):
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    return {
        "overall": avg([s for v in by_type.values() for s in v]),
        "by_source_type": {k: {"mean": avg(v), "count": len(v)} for k, v in by_type.items()},
        "timeline": sorted(
            [{"date": k, "mean": avg(v), "count": len(v)} for k, v in timeline.items()],
            key=lambda x: x["date"],
        ),
    }
