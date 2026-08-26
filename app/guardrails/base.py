"""Validator abstraction layer: composable, pluggable guardrail validators.

Inspired by Guardrails AI's Validator base class pattern, adapted for the
smart-router's proxy-layer use case (zero-dependency, regex-first, hot-reloadable).

Each validator is a self-contained class with:
  - rule_id: unique identifier
  - severity: CRITICAL/HIGH/MEDIUM/LOW
  - direction: "input" or "output"
  - scan(text) -> list[GuardrailFinding]: detection logic
  - mask(text) -> tuple[str, list[GuardrailFinding]]: remediation logic (optional)

Validators are registered in a ValidatorRegistry and composed by the engine.
New validators can be added by subclassing BaseValidator and registering them
— no changes to the engine code needed.

GuardrailFinding now includes error spans (start, end) for precise logging,
masking, and future UI highlighting.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class GuardrailFinding:
    """A single guardrail detection result.

    Attributes:
        rule_id: Unique validator identifier (e.g. "injection-ignore-previous")
        severity: CRITICAL / HIGH / MEDIUM / LOW
        snippet: First N chars of the matched text (for logging)
        start: Start character offset in the scanned text (-1 if N/A)
        end: End character offset (exclusive) in the scanned text (-1 if N/A)
        direction: "input" or "output"
        metadata: Optional dict for extra context (e.g. matched value, char name)
    """
    rule_id: str
    severity: str
    snippet: str = ""
    start: int = -1
    end: int = -1
    direction: str = ""
    metadata: dict = field(default_factory=dict)


class BaseValidator:
    """Base class for all guardrail validators.

    Subclasses must define:
      - rule_id: str
      - severity: str
      - direction: "input" or "output"

    Subclasses must implement:
      - scan(text: str) -> list[GuardrailFinding]

    Optionally implement for output validators:
      - mask(text: str) -> tuple[str, list[GuardrailFinding]]
    """

    rule_id: str = ""
    severity: str = "MEDIUM"
    direction: str = "input"  # "input" or "output"

    def scan(self, text: str) -> list[GuardrailFinding]:
        """Detect violations in text. Returns list of findings with error spans."""
        raise NotImplementedError

    def mask(self, text: str) -> tuple[str, list[GuardrailFinding]]:
        """Remediate: replace/mask detected content. Returns (text, findings).

        Default implementation for output validators: calls scan() and replaces
        each matched span with the mask_value. Override for custom logic.
        """
        findings = self.scan(text)
        if not findings:
            return text, []
        # Replace spans from right to left to preserve indices
        for f in sorted(findings, key=lambda x: x.start, reverse=True):
            if f.start >= 0 and f.end > f.start:
                text = text[:f.start] + self.mask_value() + text[f.end:]
        return text, findings

    def mask_value(self) -> str:
        """Override to customize the replacement string for mask()."""
        return "***REDACTED***"

    @property
    def enabled_attr(self) -> str:
        """Config attribute name that enables/disables this validator.

        Convention: "<rule_id_prefix>_enabled" or "<feature>_enabled".
        Override in subclasses for non-standard names.
        """
        return f"{self.rule_id.replace('-', '_')}_enabled"


class RegexValidator(BaseValidator):
    """Validator that scans text using a compiled regex pattern.

    The simplest composable unit: one regex, one rule_id, one severity.
    Findings include precise (start, end) error spans from the regex match.
    """

    def __init__(self, rule_id: str, severity: str, pattern: re.Pattern,
                 direction: str = "input", mask_str: str = "***REDACTED***"):
        self.rule_id = rule_id
        self.severity = severity
        self.pattern = pattern
        self.direction = direction
        self._mask_str = mask_str

    def scan(self, text: str) -> list[GuardrailFinding]:
        if not text:
            return []
        findings = []
        for m in self.pattern.finditer(text):
            findings.append(GuardrailFinding(
                rule_id=self.rule_id,
                severity=self.severity,
                snippet=m.group(0)[:80],
                start=m.start(),
                end=m.end(),
                direction=self.direction,
            ))
        return findings

    def mask_value(self) -> str:
        return self._mask_str


class ValidatorRegistry:
    """Registry of composable validators.

    Validators are registered by category. The engine iterates registered
    validators in order. New validators can be added at runtime via register().
    """

    def __init__(self):
        self._validators: list[BaseValidator] = []
        self._by_id: dict[str, BaseValidator] = {}

    def register(self, validator: BaseValidator) -> None:
        """Register a validator. Replaces existing with same rule_id."""
        if validator.rule_id in self._by_id:
            # Replace existing
            old = self._by_id[validator.rule_id]
            self._validators = [v for v in self._validators if v.rule_id != validator.rule_id]
        self._validators.append(validator)
        self._by_id[validator.rule_id] = validator

    def register_many(self, validators: list[BaseValidator]) -> None:
        for v in validators:
            self.register(v)

    def remove(self, rule_id: str) -> None:
        """Remove a validator by rule_id."""
        if rule_id in self._by_id:
            self._validators = [v for v in self._validators if v.rule_id != rule_id]
            del self._by_id[rule_id]

    def get(self, rule_id: str) -> Optional[BaseValidator]:
        return self._by_id.get(rule_id)

    @property
    def validators(self) -> list[BaseValidator]:
        """All registered validators, in registration order."""
        return list(self._validators)

    @property
    def input_validators(self) -> list[BaseValidator]:
        return [v for v in self._validators if v.direction == "input"]

    @property
    def output_validators(self) -> list[BaseValidator]:
        return [v for v in self._validators if v.direction == "output"]

    def clear(self) -> None:
        self._validators.clear()
        self._by_id.clear()


# Severity ordering for threshold comparisons
SEV_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def severity_at_least(finding_severity: str, threshold: str) -> bool:
    """Check if a finding's severity meets or exceeds the threshold."""
    return SEV_ORDER.get(finding_severity, 0) >= SEV_ORDER.get(threshold, 2)
