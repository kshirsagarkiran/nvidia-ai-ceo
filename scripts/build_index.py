"""Build the knowledge repository from collected documents.

Run AFTER scripts/collect.py, from the project root:
    python scripts/build_index.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indexing import build_index


if __name__ == "__main__":
    build_index()
