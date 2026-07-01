"""Recommendation validator -- 'validation of recommendations before presenting them'.

The CEO synthesizer drafts strategic recommendations, each citing the finding ids it
rests on. This validator is the gate between drafting and display: for every
recommendation it inspects the findings it cited and decides a verdict:

    validated   -- rests on at least one finding at/above the confidence threshold
    weak        -- rests only on thin findings (kept, but flagged for the reader)
    unsupported -- cites no resolvable findings at all -> DROPPED, never shown

Every verdict is written to the decision trace, and each kept recommendation gets a
`validation` block carrying its verdict, support score, the supporting finding titles,
and the underlying evidence sources (which also satisfies the brief's Task 6
"supporting evidence" requirement on the dashboard).

The score reuses the confidence the verifier already computed for findings
(0.6*entailment + 0.4*corroboration), so validation is consistent with how findings
themselves are judged -- no second, conflicting notion of "good evidence".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.state import AgentState
import config

VALIDATED = "validated"
WEAK = "weak"
UNSUPPORTED = "unsupported"


def _evidence_sources(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Union of the public source records across the supporting findings (deduped)."""
    seen, out = set(), []
    for f in findings:
        for s in f.get("sources", []):
            key = s.get("url") or s.get("title")
            if key and key not in seen:
                seen.add(key)
                out.append(s)
    return out


def _support_score(findings: List[Dict[str, Any]]) -> float:
    """Strength of support = the best (max) confidence among the cited findings."""
    if not findings:
        return 0.0
    return round(max(float(f.get("confidence", 0.0)) for f in findings), 2)


def _semantic_link(rec: Dict[str, Any], findings: List[Dict[str, Any]],
                   embedder, k: int = 3, thr: float = 0.45) -> List[Dict[str, Any]]:
    """Fallback when a recommendation cited no usable ids: link by meaning instead.

    Embeds the recommendation and every finding with the SAME bge model retrieval uses,
    and links the top-k findings above a similarity floor. Uses an element-wise dot
    (not `@`) to stay clear of the Apple-Silicon BLAS matmul issue.
    """
    import numpy as np
    if not findings:
        return []
    text = f"{rec.get('recommendation','')}. {rec.get('rationale','')}"
    fv = np.asarray(embedder.encode(
        [f"{f.get('title','')}. {f.get('summary','')}" for f in findings],
        normalize_embeddings=True))
    qv = np.asarray(embedder.encode([text], normalize_embeddings=True))[0]
    sims = np.nan_to_num((fv * qv).sum(axis=1))
    order = np.argsort(-sims)[:k]
    return [findings[i] for i in order if sims[i] >= thr]


def validate(state: AgentState, threshold: Optional[float] = None,
             embedder=None) -> List[Dict[str, Any]]:
    """Validate every recommendation on the state; drop unsupported ones; log decisions.

    Returns (and stores back on state) only the recommendations cleared for display.
    """
    if threshold is None:
        threshold = getattr(config, "VALIDATION_THRESHOLD", 0.5)

    kept: List[Dict[str, Any]] = []
    dropped = 0
    for rec in state.recommendations:
        support = rec.pop("_support", []) or []
        if not support and embedder is not None:                    # citation failed -> fall back
            support = _semantic_link(rec, state.findings, embedder)

        title = (rec.get("recommendation", "") or "")[:70]
        score = _support_score(support)

        if not support:
            dropped += 1
            state.log("validate", "DROP — unsupported",
                      f"no resolvable supporting findings for '{title}'")
            continue

        verdict = VALIDATED if score >= threshold else WEAK
        rec["validation"] = {
            "verdict": verdict,
            "support_score": score,
            "supporting_findings": [f.get("title", "") for f in support],
            "evidence": _evidence_sources(support),
        }
        state.log("validate", f"{verdict} (score {score:.2f})",
                  f"'{title}' rests on {len(support)} finding(s); "
                  f"best confidence {score:.2f} vs threshold {threshold:.2f}",
                  verdict=verdict, score=score, n_support=len(support))
        kept.append(rec)

    state.recommendations = kept
    state.log("validate", f"{len(kept)} recommendation(s) cleared for display",
              f"dropped {dropped} unsupported; {len(kept)} passed validation")
    return kept
