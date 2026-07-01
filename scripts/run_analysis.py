"""Run the full C-suite analysis (the batch job).

Run AFTER build_index.py, with Ollama running and qwen3:8b pulled:
    python scripts/run_analysis.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import run_analysis


if __name__ == "__main__":
    run_analysis()
