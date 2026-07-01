"""Episodic memory -- what the agent remembers from previous runs.

The system actually has three kinds of memory, and it helps to name them:

  * semantic / long-term memory  = the Chroma + BM25 knowledge repository (the indexed
                                   corpus the analysts retrieve from)
  * working memory               = the AgentState within a single run (state.py)
  * episodic memory              = THIS module: a compact record of each completed run

Each finished run appends one "episode" (goal, plan headlines, top findings, the
recommendations, a timestamp) to a JSONL log. At the start of the next run the planner
reads the most recent episode, so it can reason about *change over time*:

    "Last run, export controls were the top risk. Is it still active, resolved, or
     escalated? What is new since then?"

That is genuine cross-run memory use, and it powers a 'what's new since last run'
view on the dashboard. Episodic memory is deliberately small and human-readable -- it
is a journal of decisions, not a second copy of the corpus.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    import config
    _DEFAULT_DIR = Path(getattr(config, "DATA_DIR", "data")) / "memory"
except Exception:                                    # config not importable in isolation
    _DEFAULT_DIR = Path("data") / "memory"

_EPISODE_FILE = "episodes.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpisodicMemory:
    """A tiny append-only journal of past runs."""

    def __init__(self, directory: Path | str | None = None):
        self.dir = Path(directory) if directory else _DEFAULT_DIR
        self.path = self.dir / _EPISODE_FILE

    # -- write ------------------------------------------------------------
    def record(self, state) -> Dict[str, Any]:
        """Append a compact summary of a completed run. Accepts an AgentState or dict."""
        s = state.to_dict() if hasattr(state, "to_dict") else dict(state)

        def _top(items, n=5):
            ranked = sorted(items, key=lambda f: f.get("confidence", 0), reverse=True)
            return [{"type": f.get("type"), "title": f.get("title"),
                     "confidence": f.get("confidence")} for f in ranked[:n]]

        episode = {
            "at": _now(),
            "goal": s.get("goal", ""),
            "company": s.get("company", ""),
            "plan": [{"analyst": p.get("analyst"), "focus": p.get("focus")}
                     for p in s.get("plan", [])],
            "top_findings": _top(s.get("findings", [])),
            "recommendations": [r.get("recommendation") for r in s.get("recommendations", [])],
            "num_findings": len(s.get("findings", [])),
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(episode, ensure_ascii=False) + "\n")
        return episode

    # -- read -------------------------------------------------------------
    def recent(self, n: int = 3) -> List[Dict[str, Any]]:
        """Return the most recent n episodes (newest last)."""
        if not self.path.exists():
            return []
        lines = [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        episodes = []
        for ln in lines[-n:]:
            try:
                episodes.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return episodes

    def last(self) -> Dict[str, Any] | None:
        eps = self.recent(1)
        return eps[-1] if eps else None

    # -- summarize for the planner ---------------------------------------
    def briefing_for_planner(self, n: int = 1) -> str:
        """A short natural-language recap the planner can condition its plan on.

        Returns '' on the very first run, so the planner simply plans from scratch.
        """
        eps = self.recent(n)
        if not eps:
            return ""
        lines = []
        for e in eps:
            when = e.get("at", "")[:10]
            risks = [f["title"] for f in e.get("top_findings", []) if f.get("type") == "risk"]
            opps = [f["title"] for f in e.get("top_findings", []) if f.get("type") == "opportunity"]
            lines.append(f"Run {when}: top risks were {risks or 'none recorded'}; "
                         f"top opportunities were {opps or 'none recorded'}.")
        return ("Memory of previous runs (use this to focus on what may have CHANGED, "
                "to confirm whether prior risks persist, and to avoid simply repeating "
                "the same analysis):\n" + "\n".join(lines))
