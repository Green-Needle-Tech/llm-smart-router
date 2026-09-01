"""Tolerant parsing of classifier output."""
from __future__ import annotations

import json
import re

from app.schemas.router import ClassificationResult, ClassificationSource, Level


def parse_classifier_output(
    raw: str,
    source: ClassificationSource = ClassificationSource.MODEL,
    classifier_model: str | None = None,
    rubric_version: str | None = None,
    latency_ms: int = 0,
) -> ClassificationResult:
    """Parse classifier output into a ClassificationResult.

    Tries JSON first, then regex extraction, then default.
    """
    text = raw.strip()

    # Strip markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

    # Try JSON parse
    try:
        data = json.loads(text)
        level_str = data.get("level", "").upper().strip()
        confidence = float(data.get("confidence", 1.0))
        reason = data.get("reason", "")

        if level_str == "UNKNOWN":
            return ClassificationResult(
                level=None,  # type: ignore
                confidence=confidence,
                reason=reason or "UNKNOWN opener",
                source=source,
                classifier_model=classifier_model,
                rubric_version=rubric_version,
                latency_ms=latency_ms,
            )

        level = Level.from_str(level_str)
        return ClassificationResult(
            level=level,
            confidence=confidence,
            reason=reason,
            source=source,
            classifier_model=classifier_model,
            rubric_version=rubric_version,
            latency_ms=latency_ms,
        )
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # Fallback: regex extract L[1-4]
    match = re.search(r"\bL([1-4])\b", text)
    if match:
        level = Level(f"L{match.group(1)}")
        return ClassificationResult(
            level=level,
            confidence=0.5,
            reason="regex fallback",
            source=source,
            classifier_model=classifier_model,
            rubric_version=rubric_version,
            latency_ms=latency_ms,
        )

    # Check for UNKNOWN
    if "UNKNOWN" in text.upper():
        return ClassificationResult(
            level=None,  # type: ignore
            confidence=0.0,
            reason="UNKNOWN detected in text",
            source=source,
            classifier_model=classifier_model,
            rubric_version=rubric_version,
            latency_ms=latency_ms,
        )

    # Total failure: return None level, caller uses default
    return ClassificationResult(
        level=None,  # type: ignore
        confidence=0.0,
        reason="parse failure",
        source=source,
        classifier_model=classifier_model,
        rubric_version=rubric_version,
        latency_ms=latency_ms,
    )
