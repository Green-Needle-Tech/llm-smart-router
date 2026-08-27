"""System prompt leak detection validator.

Detects when an LLM response leaks system prompt content by comparing
the response against known system prompt fragments using fuzzy matching.

Inspired by Guardrails AI's detect_system_prompt_leakage validator,
which uses rapidfuzz for fuzzy string matching. This implementation
uses Python's built-in difflib.SequenceMatcher to avoid external deps.

Detection methods:
1. Direct substring match — response contains a verbatim chunk of the
   system prompt (case-insensitive, normalized whitespace).
2. Fuzzy ratio match — response contains text with high similarity
   (> threshold) to a system prompt fragment, indicating paraphrased
   leakage.

Config:
  - system_prompt_fragments: list[str] — fragments of the system prompt
    to check against (set via config, hot-reloadable).
  - fuzzy_threshold: float (0.0–1.0) — minimum similarity ratio to flag
    (default 0.85; higher = stricter, fewer false positives).
  - min_fragment_len: int — minimum fragment length to check (default 20;
    shorter fragments produce too many false positives).
  - fragment_overlap: int — sliding window overlap for chunking long
    fragments (default 40 chars).

Actions:
  - "log" (default) — monitor only; findings logged + counted
  - "mask" — replace matched spans with [REDACTED-SYSTEM-PROMPT]
  - "block" — would block the response (not implemented for proxy;
    use "mask" instead)
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from app.guardrails.base import BaseValidator, GuardrailFinding

# Default config values
DEFAULT_FUZZY_THRESHOLD = 0.85
DEFAULT_MIN_FRAGMENT_LEN = 20
DEFAULT_FRAGMENT_OVERLAP = 40
DEFAULT_MASK = "[REDACTED-SYSTEM-PROMPT]"


def _normalize(text: str) -> str:
    """Normalize whitespace and case for comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _fuzzy_ratio(a: str, b: str) -> float:
    """Compute similarity ratio between two strings (0.0–1.0)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _chunk_fragment(fragment: str, overlap: int = DEFAULT_FRAGMENT_OVERLAP) -> list[str]:
    """Split a long fragment into overlapping chunks for sliding-window matching.

    Short fragments (<= 2 * overlap) are returned as-is.
    """
    if len(fragment) <= overlap * 2:
        return [fragment]
    chunks = []
    step = overlap
    for i in range(0, len(fragment) - overlap + 1, step):
        chunk = fragment[i:i + overlap * 2]
        if len(chunk) >= overlap:
            chunks.append(chunk)
    return chunks


class SystemPromptLeakValidator(BaseValidator):
    """Output validator: detects system prompt content leaking in responses.

    Compares response text against configured system prompt fragments using
    direct substring matching and fuzzy similarity matching.
    """

    rule_id = "output-system-prompt-leak"
    severity = "HIGH"
    direction = "output"

    def __init__(
        self,
        fragments: list[str] | None = None,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        min_fragment_len: int = DEFAULT_MIN_FRAGMENT_LEN,
        fragment_overlap: int = DEFAULT_FRAGMENT_OVERLAP,
        mask_str: str = DEFAULT_MASK,
    ):
        self._fragments = [f for f in (fragments or []) if len(f) >= min_fragment_len]
        self._fuzzy_threshold = fuzzy_threshold
        self._min_fragment_len = min_fragment_len
        self._fragment_overlap = fragment_overlap
        self._mask_str = mask_str

    def update_fragments(self, fragments: list[str]) -> None:
        """Hot-reload fragment list (called when config changes)."""
        self._fragments = [f for f in fragments if len(f) >= self._min_fragment_len]

    def scan(self, text: str) -> list[GuardrailFinding]:
        if not text or not self._fragments:
            return []
        findings: list[GuardrailFinding] = []
        norm_text = _normalize(text)

        for fragment in self._fragments:
            norm_frag = _normalize(fragment)
            if len(norm_frag) < self._min_fragment_len:
                continue

            # Method 1: Direct substring match (verbatim leak)
            idx = norm_text.find(norm_frag)
            if idx != -1:
                # Map normalized index back to original text — approximate.
                # We search for the original fragment (case-insensitive) to
                # get precise spans. Fall back to normalized position.
                orig_idx = self._find_in_original(text, fragment, idx)
                findings.append(GuardrailFinding(
                    rule_id=self.rule_id,
                    severity=self.severity,
                    snippet=fragment[:60] + "…" if len(fragment) > 60 else fragment,
                    start=orig_idx,
                    end=orig_idx + len(fragment) if orig_idx >= 0 else -1,
                    direction="output",
                    metadata={"method": "exact", "fragment_len": len(fragment)},
                ))
                continue  # Don't double-report with fuzzy

            # Method 2: Fuzzy match via sliding window
            chunks = _chunk_fragment(norm_frag, self._fragment_overlap)
            for chunk in chunks:
                # Slide the chunk across the normalized text
                chunk_len = len(chunk)
                if chunk_len < self._min_fragment_len:
                    continue
                # Check at word boundaries in the normalized text
                best_ratio = 0.0
                best_pos = -1
                # Sample at word boundaries to reduce comparisons
                words = norm_text.split()
                if len(words) < 2:
                    continue
                # Build positions map
                pos = 0
                word_starts = []
                for w in words:
                    word_starts.append(pos)
                    pos += len(w) + 1  # +1 for space
                for ws in word_starts:
                    candidate = norm_text[ws:ws + chunk_len]
                    if len(candidate) < chunk_len // 2:
                        break
                    ratio = _fuzzy_ratio(chunk, candidate)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_pos = ws
                    if ratio >= 0.98:
                        break  # Near-perfect match, no need to continue

                if best_ratio >= self._fuzzy_threshold and best_pos >= 0:
                    # Approximate span in original text
                    orig_idx = self._find_in_original(text, chunk, best_pos)
                    findings.append(GuardrailFinding(
                        rule_id=self.rule_id,
                        severity=self.severity,
                        snippet=chunk[:60] + "…" if len(chunk) > 60 else chunk,
                        start=orig_idx,
                        end=orig_idx + len(chunk) if orig_idx >= 0 else -1,
                        direction="output",
                        metadata={
                            "method": "fuzzy",
                            "similarity": round(best_ratio, 3),
                            "fragment_len": len(chunk),
                        },
                    ))
                    break  # One finding per fragment is enough

        return findings

    def _find_in_original(self, original: str, fragment: str, norm_pos: int) -> int:
        """Try to find the fragment in the original text (case-insensitive).

        Falls back to a position estimate if exact search fails (whitespace
        normalization can shift positions).
        """
        # Try case-insensitive search for the first 20+ chars of the fragment
        search = fragment[:min(len(fragment), 40)].strip()
        if len(search) >= 10:
            idx = original.lower().find(search.lower())
            if idx != -1:
                return idx
        # Fallback: approximate position (count non-space chars up to norm_pos)
        count = 0
        for i, ch in enumerate(original):
            if not ch.isspace():
                if count >= norm_pos:
                    return i
                count += 1
        return -1

    def mask_value(self) -> str:
        return self._mask_str

    def mask(self, text: str) -> tuple[str, list[GuardrailFinding]]:
        """Mask detected system prompt leaks in text."""
        findings = self.scan(text)
        if not findings:
            return text, []
        for f in sorted(findings, key=lambda x: x.start, reverse=True):
            if f.start >= 0 and f.end > f.start:
                text = text[:f.start] + self._mask_str + text[f.end:]
        return text, findings
