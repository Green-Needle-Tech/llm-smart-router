"""Detection of classifier-directed phrases in the digest."""
from __future__ import annotations

import re

# Phrases that suggest the content is trying to steer the classifier
GUARD_PATTERNS = [
    r"(?i)ignore previous",
    r"(?i)you are a classifier",
    r"(?i)output L[1-5]",
    r"(?i)always classify",
    r"(?i)classify (this|everything|all) as L[1-5]",
    r"(?i)disregard (the|any) (instructions|rules|rubric)",
    r"(?i)you must (output|return|classify) L[1-5]",
]

_COMPILED = [re.compile(p) for p in GUARD_PATTERNS]
_COMBINED_RE = re.compile(
    "|".join(f"(?:{p.removeprefix('(?i)')})" for p in GUARD_PATTERNS),
    re.IGNORECASE,
)


def check_injection(text: str) -> bool:
    """Return True if any guard phrase is found in the text."""
    return bool(_COMBINED_RE.search(text))
