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

# Character classes for secret body matching (extracted to avoid duplication)
_BODY_ALNUM = r"[A-Za-z0-9]"
_BODY_ALNUM_DASH = r"[A-Za-z0-9_-]"
_BODY_ALNUM_HYPHEN = r"[A-Za-z0-9-]"

SECRET_CARRY_MARKERS: list[tuple[str, str, int]] = [
    ("sk-or-v1-", _BODY_ALNUM, 16),
    ("sk-ant-", _BODY_ALNUM_DASH, 16),  # + optional apiNN- prefix -> margin covers it
    ("sk-proj-", _BODY_ALNUM_DASH, 20),
    ("sk-", _BODY_ALNUM, 32),
    ("ghp_", _BODY_ALNUM, 20),
    ("gho_", _BODY_ALNUM, 20),
    ("ghu_", _BODY_ALNUM, 20),
    ("ghs_", _BODY_ALNUM, 20),
    ("ghr_", _BODY_ALNUM, 20),
    ("github_pat_", r"[A-Za-z0-9_]", 20),
    ("AKIA", r"[0-9A-Z]", 16),
    ("AIza", r"[0-9A-Za-z_-]", 30),
    ("xoxa-", _BODY_ALNUM_HYPHEN, 10),
    ("xoxb-", _BODY_ALNUM_HYPHEN, 10),
    ("xoxp-", _BODY_ALNUM_HYPHEN, 10),
    ("xoxr-", _BODY_ALNUM_HYPHEN, 10),
    ("xoxs-", _BODY_ALNUM_HYPHEN, 10),
    ("glpat-", _BODY_ALNUM_DASH, 20),
    ("sk_live_", _BODY_ALNUM, 20),
    ("rk_live_", _BODY_ALNUM, 20),
    ("-----BEGIN ", r"[A-Z -]", 30),
]

_COMPILED_MARKERS = [
    (marker, re.compile(body_cls + r"*\Z"), threshold)
    for marker, body_cls, threshold in SECRET_CARRY_MARKERS
]

# Telegram bot token: 8-10 digits, ':AA', then 30+ body chars
_TG_PARTIAL_RE = re.compile(r"\d{8,10}:AA[A-Za-z0-9_-]{0,29}\Z")
# A trailing digit run of 4+ chars may still grow into an 8-10 digit Telegram
# bot ID. Tokenizers frequently split the ID before ':AA' arrives (e.g. "123456"
# then "789:AA..."), and once the leading digits flush the full-secret regex can
# never reassemble (it needs the complete \d{8,10} run). Holding from the start
# of a 4+ digit run bridges that gap; benign numbers release on the next chunk
# once a non-digit (or a run that cannot grow into a token) proves them harmless.
_TG_DIGITS_RE = re.compile(r"\d{4,10}\Z")
# Tokenizers also split right AFTER the ':AA' separator (e.g. "123456789:" then
# "AAb1B2..."). Once the digits+colon flush, the next chunk starts with 'AA...'
# and no hold triggers — the telegram regex needs the leading \d{8,10} run and
# can never reassemble. Hold digits+colon and a partial ':AA' continuation too,
# so the carry bridges across the separator. Benign digit+colon tails (times
# like "12:3", ratios) release on the next chunk once the continuation proves
# harmless (not 'AA' or a partial 'AA' prefix).
_TG_DIGITS_COLON_RE = re.compile(r"\d{4,10}:(?:AA[A-Za-z0-9_-]{0,29}|A?)\Z")
# Char-by-char streaming: a single trailing digit may be the first char of an
# 8-10 digit Telegram bot ID. Hold any trailing digit run (1+), not just 4+.
# Benign single digits flush on the next chunk when a non-digit arrives.
_TG_DIGIT_RUN_RE = re.compile(r"\d{1,10}\Z")


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

    # (a2) Tail-leak guard: a marker whose body is >= threshold+MARGIN but
    # STILL growing (all body-class chars to the very end, no terminator) may
    # be a long secret (real keys exceed the regex minimum). If the strict
    # regex masked at minimum length mid-growth, the marker would be destroyed
    # and the remaining tail would flush as plaintext. Hold the whole run
    # until a non-body char arrives to terminate it.
    for marker, body_re, threshold in _COMPILED_MARKERS:
        idx = text.rfind(marker)
        if idx == -1:
            continue
        body = text[idx + len(marker):]
        if len(body) >= threshold + _MARGIN and body and body_re.fullmatch(body):
            best = min(best, idx)

    # (b) Telegram bot tokens (digit-run marker, not a fixed string).
    m = _TG_PARTIAL_RE.search(text)
    if m:
        best = min(best, m.start())
    else:
        m = _TG_DIGITS_COLON_RE.search(text)
        if m:
            best = min(best, m.start())
        else:
            m = _TG_DIGIT_RUN_RE.search(text)
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

    # (d) Whitespace-interleaved partial secrets: the collapsed tail (all
    # whitespace removed) may end with a partial marker or a growing body
    # ("s\nk\n-\no\nr" collapses to "sk-or"). Without this hold, per-character
    # emission defeats the carry entirely — each fragment flushes before the
    # next arrives and no regex ever sees a complete secret.
    best = min(best, _collapsed_tail_hold(text))

    return best


# Window for collapsed-tail checks, in collapsed (non-whitespace) chars.
# Must exceed the longest plausible secret: marker (<=11) + body (up to
# ~150 for the longest real key formats).
_MAX_COLLAPSED_WINDOW = 256


def _collapsed_tail_hold(text: str) -> int:
    """Hold index for whitespace-interleaved partial secrets (check (d))."""
    pairs: list[tuple[int, str]] = []
    for i in range(len(text) - 1, -1, -1):
        if not text[i].isspace():
            pairs.append((i, text[i]))
            if len(pairs) >= _MAX_COLLAPSED_WINDOW:
                break
    if not pairs:
        return len(text)
    pairs.reverse()
    collapsed = "".join(ch for _, ch in pairs)
    # Fast path: no whitespace inside the window -> raw checks cover it.
    if not re.search(r"\s", text[pairs[0][0]:]):
        return len(text)
    best = len(text)
    # (d1) collapsed ends with a proper prefix of a marker (marker arriving)
    for marker, _, _ in SECRET_CARRY_MARKERS:
        for k in range(len(marker) - 1, 0, -1):
            if collapsed.endswith(marker[:k]):
                best = min(best, pairs[len(pairs) - k][0])
                break
    # (d2) marker present in collapsed; body empty or all body-class chars
    # (still growing). A terminated body releases so the flush masks it.
    for marker, body_re, _threshold in _COMPILED_MARKERS:
        cidx = collapsed.rfind(marker)
        if cidx == -1:
            continue
        body = collapsed[cidx + len(marker):]
        if body_re.fullmatch(body):
            best = min(best, pairs[cidx][0])
    # (d3) telegram patterns in collapsed space
    m = _TG_PARTIAL_RE.search(collapsed)
    if m:
        best = min(best, pairs[m.start()][0])
    else:
        m = _TG_DIGITS_COLON_RE.search(collapsed)
        if m:
            best = min(best, pairs[m.start()][0])
        else:
            m = _TG_DIGIT_RUN_RE.search(collapsed)
            if m:
                best = min(best, pairs[m.start()][0])
    return best
