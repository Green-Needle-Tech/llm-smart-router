"""Guardrail engine: scans messages in / responses out, applies actions."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.guardrails.rules import (
    COMPILED_INJECTION, SECRET_RULES, SECRET_MASK, find_interleaved_secrets,
    INVISIBLE_CHARS_RE, detect_invisible_text, strip_invisible_text,
    PII_RULES, PII_MASK, _is_likely_credit_card,
    MALICIOUS_URL_RE, REFUSAL_RULES,
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
    # Invisible text detection (input) — strip zero-width/format chars
    invisible_text_detection: bool = True
    # PII masking (output) — mask email, phone, SSN, credit card
    pii_masking_enabled: bool = True
    # Banned substrings (input) — configurable list, blocks or logs
    banned_substrings: list[str] = field(default_factory=list)
    # Refusal detection (output) — log-only monitoring
    refusal_detection: bool = True
    # Malicious URL detection (output) — log or mask exfil domains
    malicious_url_detection: bool = True


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

    # --- Invisible text detection (input) ------------------------------------

    def scan_invisible_text(self, text: str) -> list[GuardrailFinding]:
        """Detect invisible/zero-width characters in text.

        Returns findings with MEDIUM severity. Does not modify the text —
        the caller (chat.py) is responsible for stripping if desired.
        """
        if not text or not self.config.invisible_text_detection:
            return []
        detected = detect_invisible_text(text)
        if not detected:
            return []
        # Deduplicate by char name
        names = set(name for name, _ in detected)
        return [
            GuardrailFinding(
                rule_id=f"invisible-text-{name.lower()}",
                severity="MEDIUM",
                snippet=f"{name} at pos {pos}",
            )
            for name, pos in detected[:10]  # cap to avoid flooding
        ]

    # --- Banned substrings (input) -------------------------------------------

    def scan_banned_substrings(self, text: str) -> list[GuardrailFinding]:
        """Check text against configurable banned substrings list.

        Case-insensitive substring match. Returns HIGH severity findings.
        """
        if not text or not self.config.banned_substrings:
            return []
        findings = []
        text_lower = text.lower()
        for banned in self.config.banned_substrings:
            if not banned:
                continue
            if banned.lower() in text_lower:
                findings.append(GuardrailFinding(
                    rule_id="banned-substring",
                    severity="HIGH",
                    snippet=banned[:40],
                ))
        return findings

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
        """Replace every secret match with ***REDACTED***. Returns (text, findings).

        Two passes:
        1. Strict contiguous regexes (SECRET_RULES).
        2. Whitespace-interleaved secrets (evasion via "s\\nk\\n-\\no\\nr...").
           Detected by find_interleaved_secrets() and masked whole.
        """
        findings: list[GuardrailFinding] = []
        for pid, pattern in SECRET_RULES:
            def _sub(m: re.Match, _pid=pid) -> str:
                findings.append(GuardrailFinding(
                    rule_id=_pid, severity="CRITICAL", snippet=m.group(0)[:12] + "…",
                ))
                return SECRET_MASK
            text = pattern.sub(_sub, text)
        # Interleaved-evasion pass: mask whole spans (marker + interleaved body).
        # Runs after the strict pass so contiguous secrets are already gone.
        interleaved = find_interleaved_secrets(text)
        if interleaved:
            # Merge overlapping spans (union) so replacement indices stay valid.
            merged: list[tuple[str, int, int]] = []
            for rid, start, end in sorted(interleaved, key=lambda s: (s[1], s[2])):
                if merged and start <= merged[-1][2]:
                    merged[-1] = (merged[-1][0], merged[-1][1], max(merged[-1][2], end))
                else:
                    merged.append((rid, start, end))
            for rid, start, end in sorted(merged, key=lambda s: s[1], reverse=True):
                findings.append(GuardrailFinding(
                    rule_id=rid, severity="CRITICAL",
                    snippet=text[start:start + 12].replace("\n", " ") + "…",
                ))
                text = text[:start] + SECRET_MASK + text[end:]
        return text, findings

    # --- PII masking (output) -------------------------------------------------

    def mask_pii(self, text: str) -> tuple[str, list[GuardrailFinding]]:
        """Replace PII matches (email, phone, SSN, credit card) with [REDACTED-PII].

        Returns (text, findings). Credit card matches are filtered through
        a false-positive guard (_is_likely_credit_card) to reduce noise.
        """
        if not text or not self.config.pii_masking_enabled:
            return text, []
        findings: list[GuardrailFinding] = []
        for pid, pattern in PII_RULES:
            def _sub(m: re.Match, _pid=pid) -> str:
                match_text = m.group(0)
                # Credit card false-positive guard
                if _pid == "pii-credit-card" and not _is_likely_credit_card(match_text):
                    return match_text  # don't mask non-CC digit runs
                findings.append(GuardrailFinding(
                    rule_id=_pid, severity="HIGH", snippet=match_text[:20] + "…",
                ))
                return PII_MASK
            text = pattern.sub(_sub, text)
        return text, findings

    # --- Malicious URL detection (output) -------------------------------------

    def scan_malicious_urls(self, text: str) -> list[GuardrailFinding]:
        """Detect known exfiltration/malicious URLs in output text.

        Returns HIGH severity findings. Does not mask — the caller decides
        based on output_action.
        """
        if not text or not self.config.malicious_url_detection:
            return []
        findings = []
        for m in MALICIOUS_URL_RE.finditer(text):
            findings.append(GuardrailFinding(
                rule_id="malicious-url",
                severity="HIGH",
                snippet=m.group(0)[:60] + "…",
            ))
        return findings

    def mask_malicious_urls(self, text: str) -> tuple[str, list[GuardrailFinding]]:
        """Replace malicious URLs with [REDACTED-URL]."""
        if not text or not self.config.malicious_url_detection:
            return text, []
        findings: list[GuardrailFinding] = []

        def _sub(m: re.Match) -> str:
            findings.append(GuardrailFinding(
                rule_id="malicious-url",
                severity="HIGH",
                snippet=m.group(0)[:60] + "…",
            ))
            return "[REDACTED-URL]"
        text = MALICIOUS_URL_RE.sub(_sub, text)
        return text, findings

    # --- Refusal detection (output, log-only) ---------------------------------

    def scan_refusal(self, text: str) -> list[GuardrailFinding]:
        """Detect LLM refusal patterns for monitoring (log-only, never blocks).

        Returns LOW severity findings — refusals are legitimate safety behavior,
        not violations. Useful for observability and quality metrics.
        """
        if not text or not self.config.refusal_detection:
            return []
        findings = []
        for pid, pattern in REFUSAL_RULES:
            if pattern.search(text):
                findings.append(GuardrailFinding(
                    rule_id=pid, severity="LOW",
                    snippet="refusal detected",
                ))
        return findings

    def process_response_content(self, message):
        """Mask secrets, PII, and malicious URLs in a message dict's content, in place.

        Also scans for refusal patterns (log-only, never modifies content).
        Accepts the message dict ({role, content}) and mutates
        message["content"] when masking applies. Returns findings.
        """
        if not self.config.output_enabled:
            return []
        content = message.get("content") if isinstance(message, dict) else message
        findings: list[GuardrailFinding] = []
        if isinstance(content, str):
            if self.config.output_action == "mask":
                masked, fs = self.mask_secrets(content)
                # PII masking
                if self.config.pii_masking_enabled:
                    masked, pii_fs = self.mask_pii(masked)
                    fs.extend(pii_fs)
                # Malicious URL masking
                if self.config.malicious_url_detection:
                    masked, url_fs = self.mask_malicious_urls(masked)
                    fs.extend(url_fs)
                if masked != content and isinstance(message, dict):
                    message["content"] = masked
            else:
                fs = self.scan_output_secrets(content)
                if self.config.malicious_url_detection:
                    fs.extend(self.scan_malicious_urls(content))
            # Refusal detection (always log-only, regardless of output_action)
            if self.config.refusal_detection:
                fs.extend(self.scan_refusal(content))
            findings.extend(fs)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    if self.config.output_action == "mask":
                        masked, fs_block = self.mask_secrets(block["text"])
                        if self.config.pii_masking_enabled:
                            masked, pii_fs = self.mask_pii(masked)
                            fs_block.extend(pii_fs)
                        if self.config.malicious_url_detection:
                            masked, url_fs = self.mask_malicious_urls(masked)
                            fs_block.extend(url_fs)
                        if masked != block["text"]:
                            block["text"] = masked
                    else:
                        fs_block = self.scan_output_secrets(block["text"])
                        if self.config.malicious_url_detection:
                            fs_block.extend(self.scan_malicious_urls(block["text"]))
                    # Refusal detection
                    if self.config.refusal_detection:
                        fs_block.extend(self.scan_refusal(block["text"]))
                    findings.extend(fs_block)
        return findings
