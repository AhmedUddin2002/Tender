from __future__ import annotations

import hashlib
import json
import re
import math
from pathlib import Path
from typing import Dict, Tuple

import requests

from config import AI_CACHE_PATH, LMSTUDIO_MODEL, LMSTUDIO_URL


CACHE_VERSION = "v1"


def _ensure_cache_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps({"version": CACHE_VERSION, "entries": {}}))


def _load_cache(path: Path = AI_CACHE_PATH) -> Dict[str, Dict[str, str]]:
    _ensure_cache_file(path)
    try:
        payload = json.loads(path.read_text())
        if payload.get("version") != CACHE_VERSION:
            return {}
        return payload.get("entries", {})
    except json.JSONDecodeError:
        return {}


def _save_cache(entries: Dict[str, Dict[str, str]], path: Path = AI_CACHE_PATH) -> None:
    _ensure_cache_file(path)
    path.write_text(json.dumps({"version": CACHE_VERSION, "entries": entries}, indent=2))


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _cache_key(title: str, description: str, keywords: str) -> str:
    key_material = "||".join(
        (_normalize_text(title), _normalize_text(description), _normalize_text(keywords))
    )
    return hashlib.md5(key_material.encode("utf-8")).hexdigest()


def _build_prompt(title: str, description: str, keywords: str) -> str:
    title_text = _normalize_text(title)
    description_text = _normalize_text(description)
    keyword_text = _normalize_text(keywords)
    return f"""
You are a tender evaluator.

Tender Title:
{title_text}

Tender Description:
{description_text}

User Keywords:
{keyword_text}

Please respond in the following strict format:
SCORE: <integer between 0 and 100>
SUMMARY: <2-3 sentences in plain English>
"""


def _parse_response(text: str) -> Tuple[int, str]:
    score_match = re.search(r"SCORE\s*:\s*(\d{1,3})", text, re.IGNORECASE)
    summary_match = re.search(r"SUMMARY\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

    score = int(score_match.group(1)) if score_match else 0
    score = max(0, min(100, score))
    summary = summary_match.group(1).strip() if summary_match else "AI summary unavailable."
    summary = summary.replace("\n", " ").strip()
    return score, summary


def _call_model(prompt: str, timeout: int = 60) -> Tuple[int, str]:
    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": [{"role": "user", "content": prompt.strip()}],
        "temperature": 0.2,
        "max_tokens": 200,
    }
    response = requests.post(LMSTUDIO_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    text = data["choices"][0]["message"]["content"]
    return _parse_response(text)


def get_ai_score_and_summary(title: str, description: str, keywords: str) -> Tuple[int, str]:
    cache = _load_cache()
    key = _cache_key(title, description, keywords)
    if key in cache:
        entry = cache[key]
        return entry["score"], entry["summary"]

    prompt = _build_prompt(title, description, keywords)
    try:
        score, summary = _call_model(prompt)
    except Exception as exc:
        score, summary = 0, f"Local AI error: {exc}"

    cache[key] = {"score": score, "summary": summary}
    _save_cache(cache)
    return score, summary


def clear_ai_cache() -> None:
    _save_cache({})
