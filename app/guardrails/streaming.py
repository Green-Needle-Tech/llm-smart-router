"""Streaming carry-buffer support for guardrail secret masking.

Secrets frequently arrive split across multiple SSE chunks (tokenizers
emit long alphanumeric strings in pieces). Per-chunk ``mask_secrets``
cannot see a complete key in any single chunk, so the stream handler
holds back a trailing "plausible partial secret" in a carry buffer
until it either completes (and gets masked) or proves benign.
"""
from __future__ import annotations

import re

# (marker, body char-class, min body length for the full secret regex to match)
# A tail starting with `marker` is held when the body after it consists only
# of body-class characters and is shorter than min_body + MARGIN — at that
# length the full secret regex would already have matched and masked it, so
# an unmasked tail this short may still be growing into a real secret.
_MARGIN = 8

SECRET_CARRY_MARKERS: list[tuple[str, str, int]] = [
    ("sk-or-v1-", r"[A-Za-z0-9]", 16),
    ("sk-ant-", r"[A-Za-z0-9_-]", 16),  # + optional apiNN- prefix -> margin covers it
    ("sk-proj-", r"[A-Za-z0-9_-]", 20),
    ("sk-", r"[A-Za-z0-9]", 32),
    ("ghp_", r"[A-Za-z0-9]", 20),
    ("gho_", r"[A-Za-z0-9]", 20),
    ("ghu_", r"[A-Za-z0-9]", 20),
    ("ghs_", r"[A-Za-z0-9]", 20),
    ("ghr_", r"[A-Za-z0-9]", 20),
    ("github_pat_", r"[A-Za-z0-9_]", 20),
    ("AKIA", r"[0-9A-Z]", 16),
    ("AIza", r"[0-9A-Za-z_-]", 30),
    ("xoxa-", r"[A-Za-z0-9-]", 10),
    ("xoxb-", r"[A-Za-z0-9-]", 10),
    ("xoxp-", r"[A-Za-z0-9-]", 10),
    ("xoxr-", r"[A-Za-z0-9-]", 10),
    ("xoxs-", r"[A-Za-z0-9-]", 10),
    ("glpat-", r"[A-Za-z0-9_-]", 20),
    ("sk_live_", r"[A-Za-z0-9]", 20),
    ("rk_live_", r"[A-Za-z0-9]", 20),
    ("-----BEGIN ", r"[A-Z -]", 30),
]

_COMPILED_MARKERS = [
    (marker, re.compile(body_cls + r"*\Z"), threshold)
    for marker, body_cls, threshold in SECRET_CARRY_MARKERS
]

# Telegram bot token: 8-10 digits, ':AA', then 30+ body chars
_TG_PARTIAL_RE = re.compile(r"\d{8,10}:AA[A-Za-z0-9_-]{0,29}\Z")
_TG_DIGITS_RE = re.compile(r"\d{8,10}\Z")


def secret_carry_split(text: str) -> int:
    """Return the index at which the streaming carry buffer should start.

    ``len(text)`` means nothing needs to be held back. The caller keeps
    ``text[idx:]`` in the carry buffer and flushes ``text[:idx]``.
    """
    if not text:
        return 0
    best = len(text)

    # (a) A full marker is present but the secret body after it is still
    # short enough to be incomplete (a complete one would already be masked).
    for marker, body_re, threshold in _COMPILED_MARKERS:
        idx = text.rfind(marker)
        if idx == -1:
            continue
        body = text[idx + len(marker):]
        if len(body) < threshold + _MARGIN and body_re.match(body):
            best = min(best, idx)

    # (b) Telegram bot tokens (digit-run marker, not a fixed string).
    m = _TG_PARTIAL_RE.search(text)
    if m:
        best = min(best, m.start())
    else:
        m = _TG_DIGITS_RE.search(text)
        if m:
            best = min(best, m.start())

    # (c) The text ends with a partial prefix of a marker (e.g. "sk-or"
    # of "sk-or-v1-") — the rest of the marker may arrive next chunk.
    for keep in range(min(len(text), 16), 0, -1):
        tail = text[-keep:]
        for marker, _, _ in SECRET_CARRY_MARKERS:
            if len(tail) < len(marker) and marker.startswith(tail):
                best = min(best, len(text) - keep)
                break
        if best < len(text) and len(text) - best == keep:
            break  # longest matching prefix found

    return best
