"""Run the full strategic-intelligence AGENT (the explicit agent loop).

Run AFTER build_index.py, with Ollama running and qwen3:8b pulled:
    python scripts/run_agent.py

This replaces the old run_analysis.py flow with the explicit loop:
    Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend -> Validate
It writes the same data/results/analysis.json the dashboard reads, plus the agent's
goal, plan, decision trace and per-recommendation validation.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project root on path

from src.orchestrator import run_agent

if __name__ == "__main__":
    run_agent()
