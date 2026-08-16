"""Heuristic fast-path rules for pre-classifier evaluation."""
from __future__ import annotations

import re
from typing import Any, Optional

from app.schemas.router import Level, ClassificationResult, ClassificationSource


def evaluate_heuristics(
    digest: str,
    *,
    has_code: bool = False,
    code_fences: int = 0,
    json_mode: bool = False,
    prompt_tokens: int = 0,
    huge_context_tokens: int = 32000,
    rules: list[dict] | None = None,
    measure: str = "task_payload",
) -> Optional[tuple[Level, bool, str]]:
    """Evaluate heuristic rules. Returns (level, stop, rule_name) or None.

    If stop=True, the classifier is skipped entirely.
    If stop=False, the level is a floor the classifier cannot go below.
    """
    # Default rules
    if rules is None:
        rules = [
            {"name": "tiny_prompt", "when": "len(digest) < 40 and not has_code", "level": "L1", "stop": True},
            {"name": "json_reshape", "when": "json_mode and len(digest) < 600", "level": "L1", "stop": True},
            {"name": "huge_context", "when": f"prompt_tokens > {huge_context_tokens}", "level": "L4", "stop": True},
            {"name": "code_heavy", "when": "code_fences >= 3", "level": "L3", "stop": False},
        ]

    # Deep keywords (always evaluated if rules exist)
    deep_kw_pattern = re.compile(
        r"\b(architect|design a system|prove|derive|refactor the|threat model|optimize the algorithm)\b",
        re.IGNORECASE,
    )

    # Evaluate explicit rules
    floor_level: Optional[Level] = None
    for rule in rules:
        name = rule.get("name", "")
        when = rule.get("when", "")
        level_str = rule.get("level", "L1")
        stop = rule.get("stop", False)

        # Simple expression evaluation
        matched = _eval_condition(when, digest=digest, has_code=has_code,
                                  code_fences=code_fences, json_mode=json_mode,
                                  prompt_tokens=prompt_tokens)

        if matched:
            try:
                level = Level.from_str(level_str)
            except ValueError:
                continue

            if stop:
                return (level, True, name)
            else:
                # Floor: track the highest floor
                if floor_level is None or level > floor_level:
                    floor_level = level

    # Deep keywords as a floor
    if deep_kw_pattern.search(digest):
        if floor_level is None or Level.L4 > floor_level:
            floor_level = Level.L4

    if floor_level is not None:
        return (floor_level, False, "floor")

    return None


def _eval_condition(expr: str, **kwargs) -> bool:
    """Safely evaluate a simple condition expression."""
    try:
        # Build a safe namespace
        ns = {
            "len": len,
            "digest": kwargs.get("digest", ""),
            "has_code": kwargs.get("has_code", False),
            "code_fences": kwargs.get("code_fences", 0),
            "json_mode": kwargs.get("json_mode", False),
            "prompt_tokens": kwargs.get("prompt_tokens", 0),
        }
        return bool(eval(expr, {"__builtins__": {}}, ns))
    except Exception:
        return False
