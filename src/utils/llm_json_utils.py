"""
Robust JSON extraction from LLM outputs: bracket matching, optional json-repair, fallbacks.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

try:
    from json_repair import repair_json as _json_repair
except ImportError:
    _json_repair = None


def repair_json_string(raw: str) -> str:
    """Try to fix common LLM JSON issues (trailing commas, unclosed brackets)."""
    if not raw or not raw.strip():
        return raw
    if _json_repair is not None:
        try:
            return _json_repair(raw)
        except Exception:
            pass
    return raw


def _skip_string(s: str, i: int) -> int:
    """Advance index past a JSON string starting at s[i]=='\"'."""
    n = len(s)
    i += 1
    while i < n:
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            return i + 1
        i += 1
    return n


def extract_first_json_object(text: str) -> Optional[str]:
    """Extract first top-level {...} by brace depth, respecting strings."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"':
            i = _skip_string(text, i - 1)
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    return None


def parse_llm_json(
    text: str,
    default: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Parse a JSON object from arbitrary LLM text.
    Returns (dict_or_none, used_repair).
    """
    default = default or {}
    if not text or not str(text).strip():
        return None, False

    candidates = []
    blob = str(text).strip()
    # Strip markdown code fence
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", blob, re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())

    obj = extract_first_json_object(blob)
    if obj:
        candidates.append(obj)

    greedy = re.search(r"\{[\s\S]*\}", blob)
    if greedy:
        candidates.append(greedy.group(0))

    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        for variant in (cand, repair_json_string(cand)):
            try:
                data = json.loads(variant)
                if isinstance(data, dict):
                    return data, variant != cand
            except json.JSONDecodeError:
                continue
    return None, False


def parse_llm_json_with_default(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    parsed, _ = parse_llm_json(text, default)
    return parsed if parsed is not None else default
