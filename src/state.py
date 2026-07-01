"""Working memory for one agent run.

This is the shared state object that threads through the whole agent loop:

    Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend -> Validate

Every stage reads from and writes to a single AgentState. That is what makes the
system a *stateful agent* rather than a one-shot pipeline: the planner writes the
plan, the analysts append findings, the decision gate records which findings cleared
the evidence bar and *why*, and the validator records a verdict on each recommendation.

The `decisions` list is a human-readable reasoning trace. The dashboard renders it so
the agent's autonomous decision-making is *visible* instead of buried in code -- which
is exactly the gap the brief asks us to close.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Decision:
    """One autonomous decision the agent made, recorded with its reason."""
    stage: str                                   # plan|retrieve|analyze|decide|recommend|validate
    decision: str                                # what the agent chose to do
    reason: str                                  # why (grounded in counts / scores / evidence)
    detail: Dict[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=_now)


@dataclass
class PlanStep:
    """One investigative task the planner decided to pursue *before* any retrieval."""
    analyst: str                                 # which specialist handles it
    focus: str                                   # what to look for
    queries: List[str]                           # retrieval queries the planner chose
    finding_type: str = "opportunity"            # opportunity|risk|trend (the analyst's default)
    source_type: Optional[str] = None            # optional retrieval filter (e.g. "filing")
    rationale: str = ""                          # why the planner chose this task


@dataclass
class AgentState:
    """The agent's working memory for a single run."""
    goal: str
    company: str
    created_at: str = field(default_factory=_now)

    plan: List[PlanStep] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    briefing: Dict[str, Any] = field(default_factory=dict)

    decisions: List[Decision] = field(default_factory=list)
    iterations: int = 0                          # how many plan->validate passes ran

    # -- decision logging -------------------------------------------------
    def log(self, stage: str, decision: str, reason: str, **detail) -> "AgentState":
        """Record one autonomous decision and echo it to the console trace."""
        self.decisions.append(Decision(stage=stage, decision=decision,
                                        reason=reason, detail=detail))
        print(f"   [decide:{stage}] {decision} -- {reason}")
        return self

    # -- convenience ------------------------------------------------------
    def set_plan(self, steps: List[PlanStep]) -> "AgentState":
        self.plan = steps
        return self

    def add_findings(self, items: List[Dict[str, Any]]) -> "AgentState":
        self.findings.extend(items)
        return self

    def findings_by_type(self, ftype: str) -> List[Dict[str, Any]]:
        return [f for f in self.findings if f.get("type") == ftype]

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Plain dict for JSON (dashboard + episodic memory)."""
        return asdict(self)

    def trace(self) -> List[Dict[str, Any]]:
        """Just the decision trace, for the dashboard's Agent Reasoning panel."""
        return [asdict(d) for d in self.decisions]
