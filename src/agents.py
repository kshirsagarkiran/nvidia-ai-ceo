"""The C-suite reasoning layer.

Each specialist analyst (CTO / CFO / CMO / Risk Officer) retrieves evidence with the
hybrid retriever, then asks the local LLM to extract findings (opportunities, risks,
trends) that MUST cite the retrieved chunks by `src-#`. Citations that don't match the
retrieved evidence are dropped, so every surviving finding is grounded. The CEO
synthesizer turns the findings into prioritised recommendations and the briefing.
"""
from __future__ import annotations

from typing import List, Dict, Optional

from src.llm import chat_json
import config


# --- evidence formatting -------------------------------------------------
def format_evidence(hits: List[Dict]):
    """Label retrieved chunks src-1..src-k and return (text_block, reference_map)."""
    lines, refmap = [], {}
    for i, h in enumerate(hits, 1):
        tag = f"src-{i}"
        lines.append(f"[{tag}] ({h['source']}) {h['text'][:300]}")
        refmap[tag] = {"source": h["source"], "title": h.get("title", ""),
                       "url": h.get("url", ""), "doc_id": h.get("doc_id", ""),
                       "text": h.get("text", "")}
    return "\n".join(lines), refmap


def _confidence(sources: List[Dict]) -> float:
    """Transparent heuristic: more citations and more *distinct* sources -> higher
    confidence. (A BART-MNLI entailment verifier refines this in the next stage.)"""
    n = len(sources)
    distinct = len({s["source"] for s in sources})
    return round(min(1.0, 0.45 + 0.12 * n + 0.10 * distinct), 2)


# --- specialist analyst --------------------------------------------------
ANALYST_SYSTEM = (
    "You are the {role} advising the CEO of {company}. From the evidence only, identify "
    "{focus}. Respond ONLY as JSON of the form "
    '{{"findings":[{{"type":"opportunity|risk|trend","title":"short title",'
    '"summary":"one or two sentences","impact":"High|Medium|Low","evidence":["src-1"]}}]}}. '
    "Every finding MUST cite at least one src-# that appears in the evidence. "
    "Do not invent sources, numbers, or facts that are not in the evidence."
)


def run_analyst(role: str, focus: str, default_type: str, queries: List[str],
                retriever, source_type: Optional[str] = None, k: int = 6) -> List[Dict]:
    findings: List[Dict] = []
    system = ANALYST_SYSTEM.format(role=role, company=config.COMPANY_NAME, focus=focus)

    for q in queries:
        hits = retriever.query(q, k=k, source_type=source_type)
        if not hits:
            continue
        evidence_text, refmap = format_evidence(hits)
        out = chat_json(f"Evidence:\n{evidence_text}\n\nIdentify {focus}.", system=system)

        for f in out.get("findings", []):
            cited = [c for c in f.get("evidence", []) if c in refmap]   # grounding filter
            if not cited:
                continue                                                # drop ungrounded claims
            # public source metadata (goes to JSON) vs. raw evidence text (for the verifier only)
            sources = [{k: v for k, v in refmap[c].items() if k != "text"} for c in cited]
            evidence_texts = [refmap[c]["text"] for c in cited]
            findings.append({
                "type": f.get("type", default_type),
                "title": f.get("title", "").strip(),
                "summary": f.get("summary", "").strip(),
                "impact": f.get("impact", "Medium"),
                "analyst": role,
                "sources": sources,
                "confidence": _confidence(sources),
                "_evidence_texts": evidence_texts,
                "query": q,
            })
    return findings


# --- CEO synthesizer -----------------------------------------------------
CEO_SYSTEM = (
    "You are the CEO synthesizer for {company}. Given NUMBERED findings from your executive "
    "team, produce prioritised strategic recommendations and an executive briefing. Each "
    "recommendation MUST cite the finding ids it is based on (e.g. \"F1\",\"F3\"), using ONLY "
    "ids that appear in the list. Do not invent recommendations unsupported by the findings. "
    "Respond ONLY as JSON: "
    '{{"recommendations":[{{"recommendation":"...","priority":"High|Medium|Low",'
    '"rationale":"...","expected_impact":["..."],"risk_level":"High|Medium|Low",'
    '"supporting_findings":["F1","F3"]}}],'
    '"briefing":{{"what_happened":"...","why_it_matters":"...","what_to_do_next":"..."}}}}'
)


def run_ceo(findings: List[Dict]) -> Dict:
    top = sorted(findings, key=lambda f: f["confidence"], reverse=True)[:20]
    id2f = {f"F{i}": f for i, f in enumerate(top, 1)}                 # stable ids for citing
    lines = [f"{fid}. [{f['type']}|{f['impact']}] {f['title']}: {f['summary']}"
             for fid, f in id2f.items()]
    prompt = ("Executive team findings:\n" + "\n".join(lines) +
              "\n\nProduce recommendations and a briefing. Each recommendation must cite the "
              "supporting finding ids.")
    out = chat_json(prompt, system=CEO_SYSTEM.format(company=config.COMPANY_NAME))

    # resolve each recommendation's cited ids back to the actual findings (for validation)
    for rec in out.get("recommendations", []) if isinstance(out, dict) else []:
        cited = rec.get("supporting_findings", []) or []
        if isinstance(cited, str):
            cited = [cited]
        support = [id2f[c] for c in cited if c in id2f]
        rec["_support"] = support                                    # internal: full findings
        rec["supporting_findings"] = [f["title"] for f in support]   # public: titles
    return out
