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

import contextlib
import re
from dataclasses import dataclass, field


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


# Injection-signal keywords for decoded-unicode content relevance check.
# Mirrors the obfuscation-scanner guard pattern: decode succeeded is necessary
# but NOT sufficient — decoded content must look like an attack.
_UNICODE_INJECTION_SIGNALS = frozenset(
    w.lower() for w in (
        "ignore", "override", "bypass", "system", "pretend", "admin",
        "instruction", "jailbreak", "prompt", "ignore previous",
        "you are", "act as", "new role", "developer mode", "dan",
    )
)


class EncodedUnicodeValidator(RegexValidator):
    """Validator for the encoded-unicode injection rule with benign-content guard.

    Extends RegexValidator with a post-match guard: matched \\uXXXX sequences
    are decoded and only flagged if the decoded content contains injection
    signals. This prevents false positives on scraped web content containing
    JSON unicode escapes (HTML entities, emoji surrogate pairs, Vietnamese
    diacritics from Agoda/booking sites, etc.).

    Follows the same guard pattern as the obfuscation scanners
    (_scan_hex_payloads, _scan_url_encoded_payloads): decode-succeeded is
    necessary but NOT sufficient — decoded content must look like an attack.
    """

    def scan(self, text: str) -> list[GuardrailFinding]:
        if not text:
            return []
        findings = []
        for m in self.pattern.finditer(text):
            matched = m.group(0)
            # Decode the \uXXXX sequences to check content relevance
            decoded = ""
            with contextlib.suppress(Exception):
                decoded = matched.encode("utf-8").decode("unicode_escape")
            # Only flag if decoded content contains injection signals
            if decoded and self._contains_injection_signal(decoded):
                findings.append(GuardrailFinding(
                    rule_id=self.rule_id,
                    severity=self.severity,
                    snippet=matched[:80],
                    start=m.start(),
                    end=m.end(),
                    direction=self.direction,
                    metadata={"decoded_preview": decoded[:60]},
                ))
        return findings

    @staticmethod
    def _contains_injection_signal(decoded: str) -> bool:
        """Check if decoded content contains injection-related keywords."""
        lower = decoded.lower()
        return any(sig in lower for sig in _UNICODE_INJECTION_SIGNALS)


class ContextAwareInjectionValidator(RegexValidator):
    """RegexValidator with a benign-context guard for injection rules.

    Extends RegexValidator with a post-match guard: after a regex match,
    examines the surrounding text window for educational/defensive context
    words. If the match appears in a sentence *about* prompt injection
    (e.g. "Ignore previous instructions is a classic example"), the finding
    is suppressed.

    This prevents false-positive blocks on security-focused conversations,
    documentation, and log-analysis requests that quote or reference
    injection phrases without attempting them.

    The guard checks a window of ±_CONTEXT_WINDOW characters around each
    match for benign-context indicators. The window is deliberately small
    so that a real injection buried at the end of a long educational
    preamble is still caught.
    """

    _CONTEXT_WINDOW = 80

    # Words/phrases that indicate the user is DISCUSSING injection, not
    # performing it. Checked case-insensitively as substrings in the
    # surrounding context window.
    _BENIGN_CONTEXT_SIGNALS = frozenset(
        w.lower() for w in (
            "prevent", "example", "attack", "detect", "test", "discuss",
            "security", "vulnerab", "how to", "what is", "explain",
            "learn", "study", "research", "mitigat", "protect", "defend",
            "guardrail", "false positive", "prompt injection", "classic",
            "known as", "called", "such as", "like ", "including",
            "pattern", "regex", "rule", "scan", "filter", "check",
            "log", "block", "monitor", "measure", "safe",
        )
    )

    def scan(self, text: str) -> list[GuardrailFinding]:
        if not text:
            return []
        findings: list[GuardrailFinding] = []
        for m in self.pattern.finditer(text):
            if self._is_benign_context(text, m.start(), m.end()):
                continue
            findings.append(GuardrailFinding(
                rule_id=self.rule_id,
                severity=self.severity,
                snippet=m.group(0)[:80],
                start=m.start(),
                end=m.end(),
                direction=self.direction,
                metadata={},
            ))
        return findings

    def _is_benign_context(self, text: str, start: int, end: int) -> bool:
        """Check if the match is surrounded by educational/defensive context."""
        window_start = max(0, start - self._CONTEXT_WINDOW)
        window_end = min(len(text), end + self._CONTEXT_WINDOW)
        context = text[window_start:window_end].lower()
        return any(sig in context for sig in self._BENIGN_CONTEXT_SIGNALS)


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

    def get(self, rule_id: str) -> BaseValidator | None:
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
