"""Orchestrate the C-suite analysis and write the results store.

This is the heavy batch job: it runs every specialist analyst, then the CEO
synthesizer, and writes a single JSON the dashboard reads. Running it as a batch
job (rather than live in the dashboard) is what keeps the fanless laptop cool and
the dashboard instant.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.retrieval import HybridRetriever
from src.agents import run_analyst, run_ceo
from src.storage_raw import load_jsonl
from src.sentiment import compute_sentiment
from collections import Counter
import config


# role, focus, default finding type, queries, source_type filter
AGENTS = [
    ("CTO", "technology opportunities and emerging trends", "opportunity",
     ["NVIDIA new products and technology roadmap", "AI accelerator innovation and partnerships"], None),
    ("CFO", "financial risks and opportunities", "risk",
     ["NVIDIA revenue growth and guidance", "NVIDIA risk factors and financial outlook"], "filing"),
    ("CMO", "market sentiment and competitive positioning", "trend",
     ["NVIDIA market sentiment and customer demand", "NVIDIA brand reception and adoption"], None),
    ("Risk Officer", "competitive threats, regulatory and supply-chain risks", "risk",
     ["NVIDIA export restrictions and regulation", "competition from AMD Intel and custom AI chips",
      "NVIDIA supply chain risks"], None),
]


def run_analysis() -> dict:
    retriever = HybridRetriever()

    findings = []
    for role, focus, ftype, queries, stype in AGENTS:
        print(f"[{role}] analysing...")
        part = run_analyst(role, focus, ftype, queries, retriever, source_type=stype)
        print(f"   {len(part)} grounded findings")
        findings.extend(part)

    # --- entailment verification: does the evidence actually support each claim? ---
    if config.USE_VERIFIER and findings:
        print("verifying findings with BART-MNLI (entailment)...")
        from src.verifier import Verifier, blend_confidence
        v = Verifier()
        for f in findings:
            ev_texts = f.pop("_evidence_texts", [])
            claim = f.get("summary") or f.get("title")
            ent = v.verify(claim, ev_texts)
            distinct_docs = len({s.get("doc_id") for s in f["sources"]})
            f["entailment"] = ent["entailment"]
            f["confidence"] = blend_confidence(ent["entailment"], distinct_docs)
    else:
        for f in findings:
            f.pop("_evidence_texts", None)

    print(f"synthesising {len(findings)} findings into recommendations...")
    ceo = run_ceo(findings)

    # --- dashboard support data (overview, market intelligence, sentiment) ---
    docs = load_jsonl()
    news = [d for d in docs if d.get("source_type") == "news"]
    news.sort(key=lambda d: d.get("published_at") or "", reverse=True)
    recent_news = [{"title": d.get("title", ""), "url": d.get("url", ""),
                    "source": d.get("source", ""), "published_at": d.get("published_at")}
                   for d in news[:12]]

    results = {
        "company": config.COMPANY_NAME,
        "industry": "Semiconductors / AI computing",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_documents": len(docs),
        "sources": dict(Counter(d.get("source") for d in docs)),
        "num_findings": len(findings),
        "opportunities": [f for f in findings if f["type"] == "opportunity"],
        "risks": [f for f in findings if f["type"] == "risk"],
        "trends": [f for f in findings if f["type"] == "trend"],
        "recommendations": ceo.get("recommendations", []),
        "briefing": ceo.get("briefing", {}),
        "recent_news": recent_news,
        "sentiment": compute_sentiment(docs),
    }

    out = Path(config.RESULTS_DIR) / "analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\nwrote {out}")
    print(f"  opportunities: {len(results['opportunities'])}  "
          f"risks: {len(results['risks'])}  trends: {len(results['trends'])}  "
          f"recommendations: {len(results['recommendations'])}")
    return results
