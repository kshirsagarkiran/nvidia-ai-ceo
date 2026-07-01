"""The Planner agent -- 'Planning before execution'.

This is the step that turns the system from a fixed pipeline into an agent. Instead of
a hardcoded list of analysts and queries, the planner is given the GOAL and the agent's
MEMORY of previous runs, and it *decides*:

    - which specialists to deploy,
    - what each should investigate,
    - which retrieval queries to run,
    - and why (a rationale that is logged to the decision trace).

The output is a list of PlanStep objects, which the existing `run_analyst` consumes
unchanged. If the LLM is unavailable or returns malformed output, we fall back to a
sensible default plan so a run never crashes -- but the normal path is a real, dynamic
plan produced by the model and conditioned on memory.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.llm import chat_json
from src.state import AgentState, PlanStep
import config


# Roster the planner may draw on. The planner picks and tailors; it need not use all.
ROSTER = ["CTO", "CFO", "CMO", "Risk Officer"]
VALID_TYPES = {"opportunity", "risk", "trend"}
VALID_SOURCE_TYPES = {"news", "filing", "community"}     # None = search all sources
MAX_STEPS = 6
MAX_QUERIES_PER_STEP = 3


PLANNER_SYSTEM = (
    "You are the planning agent for a strategic-intelligence system advising the CEO of "
    "{company} (competitors: {competitors}). Your job is NOT to answer the question yet. "
    "Your job is to PLAN the investigation: decide which specialist analysts to deploy and "
    "what each should retrieve and examine, so the team can later identify opportunities, "
    "risks and trends and make evidence-backed recommendations.\n"
    "Available analysts: {roster}. Each analyst task targets one finding type "
    "(opportunity, risk, or trend) and may filter the source type (news, filing, community) "
    "or leave it null to search everything.\n"
    "Respond ONLY as JSON of the form: "
    '{{"strategy":"one sentence on your overall approach",'
    '"plan":[{{"analyst":"CTO","focus":"what to look for","finding_type":"opportunity",'
    '"source_type":"news|filing|community|null","queries":["query 1","query 2"],'
    '"rationale":"why this task matters for the goal"}}]}}. '
    "Produce between 3 and 6 tasks covering opportunities, risks AND trends. Keep queries "
    "short and specific (good for semantic + keyword retrieval). If memory of previous runs "
    "is provided, prioritise checking whether prior risks persist and what has CHANGED."
)


def _coerce_step(raw: Dict[str, Any]) -> Optional[PlanStep]:
    """Validate and normalise one raw plan task from the LLM; return None if unusable."""
    analyst = str(raw.get("analyst", "")).strip()
    if analyst not in ROSTER:
        # tolerate close matches / casing; otherwise assign a generic analyst
        match = next((r for r in ROSTER if r.lower() == analyst.lower()), None)
        analyst = match or "Risk Officer"

    queries = raw.get("queries") or []
    if isinstance(queries, str):
        queries = [queries]
    queries = [str(q).strip() for q in queries if str(q).strip()][:MAX_QUERIES_PER_STEP]
    if not queries:
        return None                                   # a task with no queries is useless

    ftype = str(raw.get("finding_type", "")).strip().lower()
    if ftype not in VALID_TYPES:
        ftype = "opportunity"

    stype = raw.get("source_type")
    stype = str(stype).strip().lower() if stype not in (None, "", "null") else None
    if stype not in VALID_SOURCE_TYPES:
        stype = None

    return PlanStep(
        analyst=analyst,
        focus=str(raw.get("focus", "")).strip() or f"{ftype}s relevant to the goal",
        queries=queries,
        finding_type=ftype,
        source_type=stype,
        rationale=str(raw.get("rationale", "")).strip(),
    )


def _fallback_plan() -> List[PlanStep]:
    """Deterministic default if the LLM fails -- mirrors the original static analyst set."""
    return [
        PlanStep("CTO", "technology opportunities and emerging trends",
                 ["NVIDIA new products and technology roadmap",
                  "AI accelerator innovation and partnerships"], "opportunity", None,
                 "Cover technology opportunities (fallback plan)."),
        PlanStep("CFO", "financial risks and outlook",
                 ["NVIDIA revenue growth and guidance",
                  "NVIDIA risk factors and financial outlook"], "risk", "filing",
                 "Cover financial risks from filings (fallback plan)."),
        PlanStep("CMO", "market sentiment and competitive positioning",
                 ["NVIDIA market sentiment and customer demand",
                  "NVIDIA brand reception and adoption"], "trend", None,
                 "Cover market trends (fallback plan)."),
        PlanStep("Risk Officer", "competitive, regulatory and supply-chain risks",
                 ["NVIDIA export restrictions and regulation",
                  "competition from AMD Intel and custom AI chips",
                  "NVIDIA supply chain risks"], "risk", None,
                 "Cover regulatory and supply-chain risks (fallback plan)."),
    ]


def make_plan(state: AgentState, memory_briefing: str = "") -> List[PlanStep]:
    """Ask the planning agent to decide the investigation plan for `state.goal`.

    Logs its decisions into the AgentState's trace and returns the PlanStep list
    (also stored on state.plan).
    """
    system = PLANNER_SYSTEM.format(
        company=state.company,
        competitors=", ".join(getattr(config, "COMPETITORS", [])) or "various",
        roster=", ".join(ROSTER),
    )
    user = f"GOAL: {state.goal}\n"
    if memory_briefing:
        user += f"\n{memory_briefing}\n"
    user += "\nProduce the investigation plan now."

    out = chat_json(user, system=system)
    raw_steps = out.get("plan", []) if isinstance(out, dict) else []

    steps: List[PlanStep] = []
    for raw in raw_steps[:MAX_STEPS]:
        step = _coerce_step(raw)
        if step:
            steps.append(step)

    if not steps:                                     # model failed -> safe default
        steps = _fallback_plan()
        state.log("plan", "used fallback plan",
                  "planner LLM returned no usable tasks; deployed default analyst set",
                  num_tasks=len(steps))
    else:
        strategy = str(out.get("strategy", "")).strip()
        used_memory = bool(memory_briefing)
        analysts = sorted({s.analyst for s in steps})
        types = sorted({s.finding_type for s in steps})
        state.log("plan",
                  f"planned {len(steps)} tasks across {len(analysts)} analysts",
                  strategy or "planner decided the investigation tasks from the goal",
                  analysts=analysts, finding_types=types, used_memory=used_memory)

    state.set_plan(steps)
    return steps
