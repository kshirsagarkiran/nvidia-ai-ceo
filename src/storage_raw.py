"""Raw document storage as JSONL (one JSON object per line).

JSONL is chosen for the raw layer because it is append-friendly,
human-inspectable (you can open it and read what was collected), and trivial
to reload. The embedded/indexed representation lives elsewhere (the vector
store, added next); this is the durable source of truth for what we collected.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from src.schema import Document
import config


def save_jsonl(docs: List[Document], path: Optional[Path] = None) -> Path:
    path = path or (config.RAW_DIR / "documents.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: Optional[Path] = None) -> List[dict]:
    path = path or (config.RAW_DIR / "documents.jsonl")
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
