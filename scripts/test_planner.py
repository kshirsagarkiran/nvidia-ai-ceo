"""Quick local check of the Planner agent against the real LLM (Ollama + Qwen).

Run with Ollama up:   python scripts/test_planner.py
It builds the CEO goal, loads any episodic memory from previous runs, asks the planner
to decide an investigation plan, and prints the plan + the decision trace. Nothing is
written or indexed -- this is just to see the planner think.
"""
from src.state import AgentState
from src.memory import EpisodicMemory
from src.planner import make_plan
import config

GOAL = (f"If you were the CEO of {config.COMPANY_NAME} today, what would you do next "
        f"and why? Identify the key opportunities, risks and trends and prepare "
        f"evidence-backed strategic recommendations.")


def main():
    state = AgentState(goal=GOAL, company=config.COMPANY_NAME)

    mem = EpisodicMemory()
    briefing = mem.briefing_for_planner(n=1)
    if briefing:
        print(">> memory found from a previous run; planner will condition on it\n")
    else:
        print(">> no episodic memory yet (first run) — planning from scratch\n")

    steps = make_plan(state, memory_briefing=briefing)

    print("\n================ PLAN ================")
    for i, s in enumerate(steps, 1):
        src = s.source_type or "all sources"
        print(f"\n[{i}] {s.analyst}  ->  {s.finding_type.upper()}   (sources: {src})")
        print(f"    focus    : {s.focus}")
        print(f"    queries  : {s.queries}")
        if s.rationale:
            print(f"    rationale: {s.rationale}")

    print("\n============ DECISION TRACE ============")
    for d in state.trace():
        print(f"- [{d['stage']}] {d['decision']} :: {d['reason']}")


if __name__ == "__main__":
    main()
