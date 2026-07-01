"""Local LLM client (Ollama + Qwen 3).

`chat_json` forces the model into JSON mode and returns a parsed dict, so the
reasoning agents always get structured output they can rely on. Qwen 3's thinking
mode is disabled and any stray `<think>` block is stripped before parsing.

`chat_text` returns a plain-text response (no JSON mode), used by the interactive
CEO chatbox on the dashboard.
"""
from __future__ import annotations

import json
import re
from typing import Optional

import config


def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def _safe_json(text: str) -> dict:
    """Parse a JSON object from the model's reply, tolerating minor noise."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)   # grab the first {...} block
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def chat_json(prompt: str, system: Optional[str] = None) -> dict:
    import ollama   # lazy import so the module loads without a running Ollama

    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    kwargs = dict(model=config.LLM_MODEL, messages=messages, format="json",
                  options={"temperature": 0.2, "num_predict": 1200})
    try:
        resp = ollama.chat(think=False, **kwargs)
    except TypeError:                                # older ollama-python: no think= kwarg
        resp = ollama.chat(**kwargs)
    return _safe_json(_strip_think(resp["message"]["content"]))


def chat_text(prompt: str, system: Optional[str] = None) -> str:
    """Plain-text LLM call (no JSON mode). Used by the interactive CEO chatbox."""
    import ollama

    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    kwargs = dict(model=config.LLM_MODEL, messages=messages,
                  options={"temperature": 0.3, "num_predict": 800})
    try:
        resp = ollama.chat(think=False, **kwargs)
    except TypeError:
        resp = ollama.chat(**kwargs)
    return _strip_think(resp["message"]["content"])
