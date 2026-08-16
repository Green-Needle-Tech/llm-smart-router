"""Scaffolding/task separation: strip persistent agent context from task payload."""
from __future__ import annotations

import re
from typing import Optional


class CommonPrefixLearner:
    """Learns the longest common prefix across system messages."""

    def __init__(self, max_samples: int = 20, min_prefix_chars: int = 200):
        self.max_samples = max_samples
        self.min_prefix_chars = min_prefix_chars
        self._samples: list[str] = []
        self._cached_prefix: str = ""

    def add_sample(self, system_message: str) -> None:
        """Add a system message sample and recompute the common prefix."""
        if not system_message or len(system_message) < 20:
            return
        if system_message not in self._samples:
            self._samples.append(system_message)
            if len(self._samples) > self.max_samples:
                self._samples.pop(0)
        self._recompute_prefix()

    def _recompute_prefix(self) -> None:
        if len(self._samples) < 2:
            self._cached_prefix = ""
            return
        self._cached_prefix = _longest_common_prefix(self._samples)

    @property
    def prefix(self) -> str:
        return self._cached_prefix

    def strip(self, system_message: str) -> tuple[str, int]:
        """Strip the learned common prefix. Returns (stripped_text, chars_removed)."""
        if self._cached_prefix and system_message.startswith(self._cached_prefix):
            stripped = system_message[len(self._cached_prefix):]
            return stripped, len(self._cached_prefix)
        return system_message, 0


def _longest_common_prefix(strings: list[str]) -> str:
    """Compute the longest common prefix of a list of strings."""
    if not strings:
        return ""
    strings = sorted(strings)
    first, last = strings[0], strings[-1]
    i = 0
    while i < len(first) and i < len(last) and first[i] == last[i]:
        i += 1
    prefix = first[:i]
    # Only return if it meets minimum length
    if len(prefix) >= 20:
        return prefix
    return ""


def strip_sections(text: str, patterns: list[str]) -> tuple[str, list[str]]:
    """Remove delimited blocks matching configured regex patterns.

    Returns (cleaned_text, list_of_section_names_removed).
    """
    removed = []
    for pattern in patterns:
        try:
            new_text, count = re.subn(pattern, "", text, flags=re.MULTILINE | re.DOTALL)
            if count > 0:
                removed.append(f"section:{pattern[:30]}")
                text = new_text
        except re.error:
            pass
    return text, removed


def split_scaffolding(
    system_message: str,
    user_messages: list[str],
    *,
    strip_enabled: bool = True,
    learn_prefix: bool = True,
    prefix_learner: Optional[CommonPrefixLearner] = None,
    strip_patterns: Optional[list[str]] = None,
    task_text: Optional[str] = None,
    ignore_system: bool = False,
) -> dict:
    """Split a request into scaffolding and task payload.

    Returns dict with:
      - task_system: stripped system text (task-relevant)
      - task_user: the task user message(s)
      - scaffolding_chars: total chars stripped
      - stripped_by: list of mechanism names
      - task_text: explicit task text if provided
    """
    if task_text:
        return {
            "task_system": "",
            "task_user": task_text,
            "scaffolding_chars": len(system_message),
            "stripped_by": ["task_text"],
            "task_text": task_text,
        }

    if ignore_system:
        return {
            "task_system": "",
            "task_user": " ".join(user_messages),
            "scaffolding_chars": len(system_message),
            "stripped_by": ["ignore_system"],
            "task_text": None,
        }

    if not strip_enabled:
        return {
            "task_system": system_message[:500],
            "task_user": " ".join(user_messages),
            "scaffolding_chars": 0,
            "stripped_by": [],
            "task_text": None,
        }

    original_len = len(system_message)
    stripped_by = []
    current = system_message

    # 1. Learn common prefix
    if learn_prefix and prefix_learner:
        prefix_learner.add_sample(system_message)
        current, prefix_removed = prefix_learner.strip(current)
        if prefix_removed > 0:
            stripped_by.append("learned_prefix")

    # 2. Section stripping
    if strip_patterns:
        current, sections_removed = strip_sections(current, strip_patterns)
        stripped_by.extend(sections_removed)

    scaffolding_chars = original_len - len(current)

    return {
        "task_system": current[:500],
        "task_user": " ".join(user_messages),
        "scaffolding_chars": scaffolding_chars,
        "stripped_by": stripped_by,
        "task_text": None,
    }
