"""Heuristic fast-path rules for pre-classifier evaluation."""
from __future__ import annotations

import ast
import re
from functools import lru_cache
from types import CodeType

from app.schemas.router import Level

_ALLOWED_NAMES = frozenset({
    "len", "digest", "has_code", "code_fences",
    "json_mode", "prompt_tokens", "task_chars",
    "True", "False", "None",
})

_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BoolOp, ast.And, ast.Or,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    ast.Call,
    ast.Name, ast.Load,
    ast.Constant,
)


@lru_cache(maxsize=128)
def _compile_condition(expr: str) -> CodeType | None:
    """Validate AST safety and compile a heuristic expression once."""
    if not expr or not expr.strip():
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            return None
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            return None
        if isinstance(node, ast.Call) and not (
            isinstance(node.func, ast.Name) and node.func.id == "len"
        ):
            return None
    return compile(tree, "<heuristic>", "eval")


def evaluate_heuristics(
    digest: str,
    *,
    has_code: bool = False,
    code_fences: int = 0,
    json_mode: bool = False,
    prompt_tokens: int = 0,
    task_chars: int | None = None,
    huge_context_tokens: int = 32000,
    rules: list | None = None,
    measure: str = "task_payload",
) -> tuple[Level, bool, str] | None:
    """Evaluate heuristic rules. Returns (level, stop, rule_name) or None.

    If stop=True, the classifier is skipped entirely.
    If stop=False, the level is a floor the classifier cannot go below.

    Note: rule expressions should prefer ``task_chars`` over ``len(digest)``.
    ``digest`` includes the untrusted-input wrapper and the context-summary
    line (~95 chars of boilerplate), so ``len(digest)`` can never be small
    even for a two-word prompt. ``task_chars`` is the raw user payload.
    When ``task_chars`` is not supplied it falls back to ``len(digest)`` so
    callers passing an unwrapped digest keep the old behaviour.
    """
    if task_chars is None:
        task_chars = len(digest)
    # Default rules
    if rules is None:
        rules = [
            {"name": "tiny_prompt", "when": "task_chars < 40 and not has_code", "level": "L1", "stop": True},
            {"name": "json_reshape", "when": "json_mode and task_chars < 600", "level": "L1", "stop": True},
            {"name": "huge_context", "when": f"prompt_tokens > {huge_context_tokens}", "level": "L4", "stop": True},
            {"name": "code_heavy", "when": "code_fences >= 3", "level": "L3", "stop": False},
        ]

    stop_result, floor_level = _evaluate_rules(
        rules, digest, has_code, code_fences, json_mode, prompt_tokens, task_chars
    )
    if stop_result:
        return stop_result

    if _DEEP_KW_PATTERN.search(digest) and (floor_level is None or floor_level < Level.L4):
        floor_level = Level.L4

    if floor_level is not None:
        return (floor_level, False, "floor")

    return None


_DEEP_KW_PATTERN = re.compile(
    r"\b(architect|design a system|prove|derive|refactor the|threat model|optimize the algorithm)\b",
    re.IGNORECASE,
)


def _rule_fields(rule) -> tuple[str, str, str, bool]:
    """Extract (name, when, level, stop) from dict or Pydantic HeuristicRule without model_dump."""
    if isinstance(rule, dict):
        return (
            rule.get("name", ""),
            rule.get("when", ""),
            rule.get("level", "L1"),
            bool(rule.get("stop", False)),
        )
    if hasattr(rule, "when"):
        return (
            getattr(rule, "name", ""),
            getattr(rule, "when", ""),
            getattr(rule, "level", "L1"),
            bool(getattr(rule, "stop", False)),
        )
    return str(rule), "", "L1", False


def _evaluate_rules(rules, digest, has_code, code_fences, json_mode, prompt_tokens, task_chars):
    """Evaluate explicit rules. Returns (stop_result, floor_level)."""
    floor_level: Level | None = None
    ns = {
        "len": len,
        "digest": digest,
        "has_code": has_code,
        "code_fences": code_fences,
        "json_mode": json_mode,
        "prompt_tokens": prompt_tokens,
        "task_chars": task_chars,
    }
    for rule in rules:
        name, when, level_str, stop = _rule_fields(rule)
        matched = _eval_condition(when, _ns=ns)
        if matched:
            try:
                level = Level.from_str(level_str)
            except ValueError:
                continue
            if stop:
                return (level, True, name), None
            if floor_level is None or level > floor_level:
                floor_level = level
    return None, floor_level


def _eval_condition(expr: str, _ns: dict | None = None, **kwargs) -> bool:
    """Safely evaluate a simple condition expression via AST-validated compiled code."""
    code = _compile_condition(expr)
    if code is None:
        return False
    try:
        ns = _ns if _ns is not None else {
            "len": len,
            "digest": kwargs.get("digest", ""),
            "has_code": kwargs.get("has_code", False),
            "code_fences": kwargs.get("code_fences", 0),
            "json_mode": kwargs.get("json_mode", False),
            "prompt_tokens": kwargs.get("prompt_tokens", 0),
            "task_chars": kwargs.get("task_chars", 0),
        }
        return bool(eval(code, {"__builtins__": {}}, ns))
    except Exception:
        return False
