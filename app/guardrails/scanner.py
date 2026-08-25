"""Guardrail engine: scans messages in / responses out, applies actions."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.guardrails.rules import (
    COMPILED_INJECTION, SECRET_RULES, SECRET_MASK,
)

_CODE_BLOCK_RE = re.compile(r"```")


@dataclass
class GuardrailFinding:
    rule_id: str
    severity: str
    snippet: str = ""


@dataclass
class GuardrailConfig:
    # Input guardrail
    input_enabled: bool = True
    input_action: str = "log"       # "log" | "block" | "tag"
    block_on_severity: str = "HIGH" # block findings at/above this severity
    # Output guardrail
    output_enabled: bool = True
    output_action: str = "mask"     # "mask" | "log" | "block"


@dataclass
class InputScanResult:
    findings: list[GuardrailFinding] = field(default_factory=list)
    blocked: bool = False

    @property
    def triggered(self) -> bool:
        return bool(self.findings)


_SEV_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def _in_code_block_heavy_text(text: str) -> bool:
    """Heuristic: message dominated by code blocks → likely educational."""
    fences = _CODE_BLOCK_RE.findall(text)
    return len(fences) >= 2 and text.count("```") / max(len(text), 1) > 0.0005 and len(fences) % 2 == 0 and len(text) > 400


class GuardrailEngine:
    """Scans text on the request and response paths."""

    def __init__(self, config: GuardrailConfig):
        self.config = config

    # --- Input path ----------------------------------------------------------

    def scan_text(self, text: str) -> list[GuardrailFinding]:
        """Run injection rules over a single text. Skips code-block-heavy text."""
        if not text:
            return []
        if _in_code_block_heavy_text(text):
            return []
        findings = []
        for pid, sev, pattern in COMPILED_INJECTION:
            m = pattern.search(text)
            if m:
                findings.append(GuardrailFinding(
                    rule_id=pid, severity=sev,
                    snippet=m.group(0)[:80],
                ))
        return findings

    def scan_messages(self, messages: list) -> InputScanResult:
        """Scan all message contents; decides block per config."""
        if not self.config.input_enabled:
            return InputScanResult()
        all_findings: list[GuardrailFinding] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                all_findings.extend(self.scan_text(content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        all_findings.extend(self.scan_text(block["text"]))
        blocked = False
        if self.config.input_action == "block" and all_findings:
            threshold = _SEV_ORDER.get(self.config.block_on_severity, 2)
            blocked = any(_SEV_ORDER.get(f.severity, 0) >= threshold for f in all_findings)
        return InputScanResult(findings=all_findings, blocked=blocked)

    # --- Output path -----------------------------------------------------------

    def scan_output_secrets(self, text: str) -> list[GuardrailFinding]:
        if not text:
            return []
        findings = []
        for pid, pattern in SECRET_RULES:
            m = pattern.search(text)
            if m:
                findings.append(GuardrailFinding(
                    rule_id=pid, severity="CRITICAL", snippet=m.group(0)[:12] + "…",
                ))
        return findings

    def mask_secrets(self, text: str) -> tuple[str, list[GuardrailFinding]]:
        """Replace every secret match with ***REDACTED***. Returns (text, findings)."""
        findings = []
        for pid, pattern in SECRET_RULES:
            def _sub(m: re.Match, _pid=pid) -> str:
                findings.append(GuardrailFinding(
                    rule_id=_pid, severity="CRITICAL", snippet=m.group(0)[:12] + "…",
                ))
                return SECRET_MASK
            text = pattern.sub(_sub, text)
        return text, findings

    def process_response_content(self, content):
        """Mask secrets in a message content value (str or block list).

        Returns (content, findings).
        """
        if not self.config.output_enabled:
            return content, []
        findings: list[GuardrailFinding] = []
        if isinstance(content, str):
            if self.config.output_action == "mask":
                content, findings = self.mask_secrets(content)
            else:
                findings = self.scan_output_secrets(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    if self.config.output_action == "mask":
                        block["text"], fs = self.mask_secrets(block["text"])
                    else:
                        fs = self.scan_output_secrets(block["text"])
                    findings.extend(fs)
        return content, findings
