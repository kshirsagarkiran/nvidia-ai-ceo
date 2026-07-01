"""The agent orchestrator -- the explicit control loop the brief asks for:

    Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend -> Validate

It threads a single AgentState through every stage, logs each autonomous decision to a
visible trace, and conditions planning on episodic memory of past runs. Every component
that already worked is reused unchanged -- the hybrid retriever, the analysts, the
BART-MNLI verifier, finding-level dedup, the CEO synthesizer, sentiment. What is new is
the *explicit agency around them*: a planner decides what to investigate, a decision
gate records why findings are kept or dropped, and a validator gates recommendations
before they are ever shown.

It writes the same analysis.json the dashboard already reads, plus new fields
(goal, plan, decisions, validation) that Phase 5 surfaces as an "Agent Reasoning" panel.

Run it with:  python scripts/run_agent.py   (Ollama up, after build_index.py)
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.retrieval import HybridRetriever
from src.agents import run_analyst, run_ceo
from src.storage_raw import load_jsonl
from src.sentiment import compute_sentiment
from src.state import AgentState
from src.memory import EpisodicMemory
from src.planner import make_plan
from src.validator import validate
import config


GOAL = (f"If you were the CEO of {config.COMPANY_NAME} today, what would you do next and "
        f"why? Identify the key opportunities, risks and trends and produce prioritised, "
        f"evidence-backed strategic recommendations.")


# --- finding-level dedup (self-contained; Apple-Silicon-safe dot product) ----
def _dedup_findings(findings, embedder, threshold):
    """Collapse near-duplicate findings of the same type, keeping the best-scored one.

    Overlapping analyst queries can surface the same source twice, so two near-identical
    findings appear. We embed each (title + summary) with the SAME bge model retrieval
    uses, greedily cluster by cosine similarity within each finding type, and keep the
    highest-confidence representative. Uses an element-wise dot (not `@`) to avoid the
    Apple-Silicon Accelerate matmul issue.
    """
    import numpy as np

    by_type = {}
    for f in findings:
        by_type.setdefault(f.get("type"), []).append(f)

    kept = []
    for group in by_type.values():
        group = sorted(group, key=lambda f: f.get("confidence", 0), reverse=True)
        vecs = np.asarray(embedder.encode(
            [f"{f.get('title','')}. {f.get('summary','')}" for f in group],
            normalize_embeddings=True))
        reps = []
        for i, f in enumerate(group):
            match = next((r for r in reps
                          if float((vecs[i] * vecs[r]).sum()) >= threshold), None)
            if match is None:
                reps.append(i)
                kept.append(f)
            else:
                print(f"   deduped '{f.get('title')}' -> '{group[match].get('title')}'")
    return kept


# --- DECIDE: verify findings, then filter/dedup, logging why ----------------
def _decide(state: AgentState, findings, retriever):
    """The explicit decision gate: verify evidence, drop weak/duplicate findings, log it."""
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
        state.log("decide", f"verified {len(findings)} findings",
                  "scored each claim against its cited evidence with BART-MNLI entailment")
    else:
        for f in findings:
            f.pop("_evidence_texts", None)

    if getattr(config, "USE_FINDING_DEDUP", True) and findings:
        before = len(findings)
        findings = _dedup_findings(findings, retriever.embedder,
                                   getattr(config, "DEDUP_THRESHOLD", 0.82))
        if before != len(findings):
            state.log("decide", f"dropped {before - len(findings)} duplicate findings",
                      "clustered near-identical findings of the same type; kept best-scored")

    by_type = Counter(f.get("type") for f in findings)
    state.log("decide", f"kept {len(findings)} findings for synthesis",
              f"opportunities {by_type.get('opportunity',0)}, risks {by_type.get('risk',0)}, "
              f"trends {by_type.get('trend',0)}")
    return findings


# --- the full loop ----------------------------------------------------------
def run_agent() -> dict:
    retriever = HybridRetriever()
    mem = EpisodicMemory()
    state = AgentState(goal=GOAL, company=config.COMPANY_NAME)
    max_iters = getattr(config, "MAX_AGENT_ITERATIONS", 2)

    # GOAL
    state.log("goal", "set strategic goal", GOAL)

    briefing = mem.briefing_for_planner(n=1)
    if briefing:
        state.log("plan", "loaded episodic memory", "conditioning the plan on the previous run")

    while True:
        state.iterations += 1
        retry = state.iterations > 1
        hint = ("\nNOTE: the previous attempt produced no validated recommendations. "
                "Broaden the focus and use more specific, evidence-rich queries.") if retry else ""

        # PLAN
        print(f"\n=== planning (pass {state.iterations}) ===")
        plan = make_plan(state, memory_briefing=briefing + hint)

        # RETRIEVE + ANALYZE
        findings = []
        for step in plan:
            print(f"[{step.analyst}] {step.finding_type} :: {step.focus}")
            part = run_analyst(step.analyst, step.focus, step.finding_type,
                               step.queries, retriever, source_type=step.source_type)
            print(f"   {len(part)} grounded findings")
            if not part:
                state.log("analyze", f"no evidence for '{step.focus}'",
                          "analyst returned no grounded findings; dropping this line of inquiry")
            findings.extend(part)
        state.findings = []                                  # reset (fresh each pass)

        # DECIDE
        findings = _decide(state, findings, retriever)
        state.add_findings(findings)

        # RECOMMEND
        print(f"synthesising {len(findings)} findings into recommendations...")
        ceo = run_ceo(findings)
        state.recommendations = ceo.get("recommendations", []) if isinstance(ceo, dict) else []
        state.briefing = ceo.get("briefing", {}) if isinstance(ceo, dict) else {}
        state.log("recommend", f"drafted {len(state.recommendations)} recommendations",
                  "CEO synthesizer produced prioritised recommendations citing finding ids")

        # VALIDATE
        print("validating recommendations against cited evidence...")
        validate(state, embedder=retriever.embedder)

        if state.recommendations or state.iterations >= max_iters:
            break
        state.log("validate", "re-planning", "no recommendation passed validation; trying again")

    # MEMORY: record this run so the next one can reason about what changed
    mem.record(state)
    state.log("memory", "recorded episode", "saved goal, plan, top findings and recommendations")

    # --- assemble analysis.json (existing keys + new agent-trace fields) ---
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
        "num_findings": len(state.findings),
        "opportunities": [f for f in state.findings if f["type"] == "opportunity"],
        "risks": [f for f in state.findings if f["type"] == "risk"],
        "trends": [f for f in state.findings if f["type"] == "trend"],
        "recommendations": state.recommendations,
        "briefing": state.briefing,
        "recent_news": recent_news,
        "sentiment": compute_sentiment(docs),
        # --- new: the agent's explicit reasoning, for the dashboard panel ---
        "goal": state.goal,
        "plan": [asdict(p) for p in state.plan],
        "decisions": state.trace(),
        "iterations": state.iterations,
        "memory_used": bool(briefing),
    }

    out = Path(config.RESULTS_DIR) / "analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\nwrote {out}")
    print(f"  plan tasks: {len(results['plan'])}  decisions logged: {len(results['decisions'])}  "
          f"passes: {results['iterations']}")
    print(f"  opportunities: {len(results['opportunities'])}  risks: {len(results['risks'])}  "
          f"trends: {len(results['trends'])}  recommendations: {len(results['recommendations'])}")
    return results
