"""Guardrail engine: scans messages in / responses out, applies actions.

P0 Improvements (Aug 2026):
1. Validator abstraction layer — GuardrailFinding now has error spans (start/end).
   Existing rules wrapped in RegexValidator via ValidatorRegistry.
2. Error spans — all scan methods populate start/end from regex match positions.
3. System prompt leak detection — SystemPromptLeakValidator for output scanning.

The engine API is backward-compatible with chat.py. New features are opt-in
via GuardrailConfig fields (system_prompt_leak_detection, system_prompt_fragments).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.guardrails.base import (
    SEV_ORDER,
    GuardrailFinding,
    RegexValidator,
    ValidatorRegistry,
)
from app.guardrails.rules import (
    COMPILED_INJECTION,
    MALICIOUS_URL_RE,
    PII_MASK,
    PII_RULES,
    REFUSAL_RULES,
    SECRET_MASK,
    SECRET_RULES,
    _is_likely_credit_card,
    detect_invisible_text,
    find_interleaved_secrets,
    normalize_homoglyphs,
    scan_obfuscated_payloads,
)
from app.guardrails.validators import SystemPromptLeakValidator

_CODE_BLOCK_RE = re.compile(r"```")


def _build_default_registry() -> ValidatorRegistry:
    """Build the default validator registry from existing compiled rules.

    Each injection rule and secret rule becomes a RegexValidator with
    precise error spans. This makes them composable — new validators
    can be added via registry.register() without touching engine code.
    """
    registry = ValidatorRegistry()

    # Input: injection rules (24 patterns across 8 categories)
    for rule_id, severity, pattern in COMPILED_INJECTION:
        registry.register(RegexValidator(
            rule_id=rule_id, severity=severity, pattern=pattern,
            direction="input",
        ))

    # Output: secret rules (11 provider-prefixed credential patterns)
    for rule_id, pattern in SECRET_RULES:
        registry.register(RegexValidator(
            rule_id=rule_id, severity="CRITICAL", pattern=pattern,
            direction="output", mask_str=SECRET_MASK,
        ))

    # Output: PII rules
    for rule_id, pattern in PII_RULES:
        registry.register(RegexValidator(
            rule_id=rule_id, severity="HIGH", pattern=pattern,
            direction="output", mask_str=PII_MASK,
        ))

    # Output: malicious URL rules
    registry.register(RegexValidator(
        rule_id="malicious-url", severity="HIGH", pattern=MALICIOUS_URL_RE,
        direction="output", mask_str="[REDACTED-URL]",
    ))

    # Output: refusal rules (log-only, LOW severity)
    for rule_id, pattern in REFUSAL_RULES:
        registry.register(RegexValidator(
            rule_id=rule_id, severity="LOW", pattern=pattern,
            direction="output",
        ))

    return registry


# Build once at import; engine instances share the compiled patterns.
_DEFAULT_REGISTRY = _build_default_registry()


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
    # PII masking (input) — mask emails in input before forwarding upstream
    input_pii_masking_enabled: bool = True
    # Banned substrings (input) — configurable list, blocks or logs
    banned_substrings: list[str] = field(default_factory=list)
    # Refusal detection (output) — log-only monitoring
    refusal_detection: bool = True
    # Malicious URL detection (output) — log or mask exfil domains
    malicious_url_detection: bool = True
    # System prompt leak detection (output) — fuzzy match response vs fragments
    system_prompt_leak_detection: bool = False
    # Fragments of system prompts to check against (hot-reloadable)
    system_prompt_fragments: list[str] = field(default_factory=list)
    # Fuzzy similarity threshold (0.0-1.0; higher = fewer false positives)
    system_prompt_leak_threshold: float = 0.85
    # Homoglyph normalization (input) — normalize Cyrillic/Greek lookalikes before scan
    homoglyph_normalization: bool = True
    # Obfuscation and high-entropy detection (input) — Base64, Hex, URL-encoding
    obfuscation_detection: bool = True
    # Shannon entropy threshold for payload token detection
    entropy_threshold: float = 4.5


@dataclass
class InputScanResult:
    findings: list[GuardrailFinding] = field(default_factory=list)
    blocked: bool = False

    @property
    def triggered(self) -> bool:
        return bool(self.findings)


def _in_code_block_heavy_text(text: str) -> bool:
    """Heuristic: message dominated by code blocks → likely educational."""
    fences = _CODE_BLOCK_RE.findall(text)
    return len(fences) >= 2 and text.count("```") / max(len(text), 1) > 0.0005 and len(fences) % 2 == 0 and len(text) > 400


class GuardrailEngine:
    """Scans text on the request and response paths.

    Uses a ValidatorRegistry for composable validators. New validators
    can be registered at runtime via engine.registry.register().
    """

    def __init__(self, config: GuardrailConfig):
        self.config = config
        self.registry = _build_default_registry()
        self._spleak_validator: SystemPromptLeakValidator | None = None
        self._update_spleak_validator()

    def _update_spleak_validator(self) -> None:
        """Create or update the system prompt leak validator from config."""
        if self.config.system_prompt_leak_detection and self.config.system_prompt_fragments:
            if self._spleak_validator is None:
                self._spleak_validator = SystemPromptLeakValidator(
                    fragments=self.config.system_prompt_fragments,
                    fuzzy_threshold=self.config.system_prompt_leak_threshold,
                )
            else:
                self._spleak_validator.update_fragments(self.config.system_prompt_fragments)
                self._spleak_validator._fuzzy_threshold = self.config.system_prompt_leak_threshold
        else:
            self._spleak_validator = None

    # --- Input path ----------------------------------------------------------

    def scan_text(self, text: str) -> list[GuardrailFinding]:
        """Run injection rules over a single text. Skips code-block-heavy text.

        Returns findings with error spans (start, end) from regex matches.
        Applies homoglyph normalization if enabled.
        """
        if not text:
            return []
        if _in_code_block_heavy_text(text):
            return []

        # Homoglyph normalization pass
        eval_text = normalize_homoglyphs(text) if self.config.homoglyph_normalization else text

        findings = []
        # Use registry's input validators (injection rules)
        # Match all non-secret, non-PII, non-URL, non-refusal input validators
        _skip_prefixes = ("openrouter-", "anthropic-", "openai-", "github-",
                          "aws-", "google-", "slack-", "gitlab-", "stripe-",
                          "telegram-", "private-key-", "pii-", "malicious-",
                          "refusal-")
        for v in self.registry.input_validators:
            if not v.rule_id.startswith(_skip_prefixes):
                findings.extend(v.scan(eval_text))
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
            threshold = SEV_ORDER.get(self.config.block_on_severity, 2)
            blocked = any(SEV_ORDER.get(f.severity, 0) >= threshold for f in all_findings)
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
        return [
            GuardrailFinding(
                rule_id=f"invisible-text-{name.lower()}",
                severity="MEDIUM",
                snippet=f"{name} at pos {pos}",
                start=pos,
                end=pos + 1,
                direction="input",
                metadata={"char_name": name},
            )
            for name, pos in detected[:10]  # cap to avoid flooding
        ]

    # --- Banned substrings (input) -------------------------------------------

    def scan_banned_substrings(self, text: str) -> list[GuardrailFinding]:
        """Check text against configurable banned substrings list.

        Case-insensitive substring match. Returns HIGH severity findings
        with error spans.
        """
        if not text or not self.config.banned_substrings:
            return []
        findings = []
        text_lower = text.lower()
        for banned in self.config.banned_substrings:
            if not banned:
                continue
            banned_lower = banned.lower()
            start = 0
            while True:
                idx = text_lower.find(banned_lower, start)
                if idx == -1:
                    break
                findings.append(GuardrailFinding(
                    rule_id="banned-substring",
                    severity="HIGH",
                    snippet=banned[:40],
                    start=idx,
                    end=idx + len(banned),
                    direction="input",
                    metadata={"banned_term": banned},
                ))
                start = idx + len(banned_lower)
        return findings

    # --- Obfuscation & Entropy detection (input) -----------------------------

    def scan_obfuscation(self, text: str) -> list[GuardrailFinding]:
        """Scan text for high-entropy tokens and obfuscated/encoded payloads (Base64, Hex, URL).

        Returns findings with HIGH/MEDIUM severity and error spans.
        """
        if not text or not self.config.obfuscation_detection:
            return []
        findings: list[GuardrailFinding] = []
        payloads = scan_obfuscated_payloads(
            text,
            entropy_threshold=self.config.entropy_threshold,
        )
        for scan_type, snippet, start, end in payloads:
            findings.append(GuardrailFinding(
                rule_id=scan_type,
                severity="HIGH" if scan_type == "obfuscation-base64" else "MEDIUM",
                snippet=snippet[:60],
                start=start,
                end=end,
                direction="input",
                metadata={"type": scan_type, "preview": snippet},
            ))
        return findings

    # --- Output path -----------------------------------------------------------

    def scan_output_secrets(self, text: str) -> list[GuardrailFinding]:
        if not text:
            return []
        findings = []
        for v in self.registry.output_validators:
            if v.rule_id in (
                "openrouter-key", "anthropic-key", "openai-key", "github-token",
                "aws-access-key", "google-api-key", "slack-token", "gitlab-token",
                "stripe-key", "telegram-bot", "private-key-pem",
            ):
                findings.extend(v.scan(text))
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
                    start=m.start(), end=m.end(), direction="output",
                ))
                return SECRET_MASK
            text = pattern.sub(_sub, text)
        # Interleaved-evasion pass: mask whole spans (marker + interleaved body).
        interleaved = find_interleaved_secrets(text)
        if interleaved:
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
                    start=start, end=end, direction="output",
                ))
                text = text[:start] + SECRET_MASK + text[end:]
        return text, findings

    # --- PII masking (output) -------------------------------------------------

    def mask_pii(self, text: str) -> tuple[str, list[GuardrailFinding]]:
        """Replace PII matches (email, phone, SSN, credit card) with [REDACTED-PII].

        Returns (text, findings). Credit card matches are filtered through
        a false-positive guard (_is_likely_credit_card) to reduce noise.
        Findings include error spans from regex matches.
        """
        if not text or not self.config.pii_masking_enabled:
            return text, []
        findings: list[GuardrailFinding] = []
        for pid, pattern in PII_RULES:
            def _sub(m: re.Match, _pid=pid) -> str:
                match_text = m.group(0)
                if _pid == "pii-credit-card" and not _is_likely_credit_card(match_text):
                    return match_text  # don't mask non-CC digit runs
                findings.append(GuardrailFinding(
                    rule_id=_pid, severity="HIGH", snippet=match_text[:20] + "…",
                    start=m.start(), end=m.end(), direction="output",
                ))
                return PII_MASK
            text = pattern.sub(_sub, text)
        return text, findings

    # --- PII + secret masking (input) -----------------------------------------

    def _mask_input_pii(self, text: str) -> tuple[str, list[GuardrailFinding]]:
        """Mask PII patterns in input text. Returns (masked_text, findings)."""
        findings: list[GuardrailFinding] = []
        for pid, pattern in PII_RULES:
            def _pii_sub(m: re.Match, _pid=pid, _text=text) -> str:
                match_text = m.group(0)
                if _pid == "pii-credit-card" and not _is_likely_credit_card(match_text):
                    return match_text  # don't mask non-CC digit runs
                if _pid == "pii-passport" and not re.search(
                    r"passport", _text[:m.start() + 50], re.IGNORECASE
                ):
                    # Avoid false positive: require context keyword nearby
                    return match_text
                findings.append(GuardrailFinding(
                    rule_id=_pid, severity="HIGH",
                    snippet=match_text[:20] + "\u2026",
                    start=m.start(), end=m.end(), direction="input",
                ))
                return PII_MASK
            text = pattern.sub(_pii_sub, text)
        return text, findings

    def _mask_input_secrets(self, text: str) -> tuple[str, list[GuardrailFinding]]:
        """Mask provider credentials in input text. Returns (masked_text, findings)."""
        findings: list[GuardrailFinding] = []
        for pid, pattern in SECRET_RULES:
            def _secret_sub(m: re.Match, _pid=pid) -> str:
                findings.append(GuardrailFinding(
                    rule_id=_pid, severity="CRITICAL",
                    snippet=m.group(0)[:12] + "\u2026",
                    start=m.start(), end=m.end(), direction="input",
                ))
                return SECRET_MASK
            text = pattern.sub(_secret_sub, text)
        return text, findings

    def mask_input_sensitive(self, text: str) -> tuple[str, list[GuardrailFinding]]:
        """Mask all PII and secrets in input text before forwarding upstream.

        Masks: emails, phone numbers, SSNs, credit cards, IBANs, passport
        numbers, driver's license numbers, and provider API keys/secrets.
        This prevents upstream provider guardrails (e.g. OpenRouter) from
        detecting and flagging sensitive information in user messages.

        Returns (masked_text, findings).
        """
        if not text:
            return text, []
        findings: list[GuardrailFinding] = []
        text, secret_fs = self._mask_input_secrets(text)
        findings.extend(secret_fs)
        if self.config.input_pii_masking_enabled:
            text, pii_fs = self._mask_input_pii(text)
            findings.extend(pii_fs)
        return text, findings

    # --- Malicious URL detection (output) -------------------------------------

    def scan_malicious_urls(self, text: str) -> list[GuardrailFinding]:
        """Detect known exfiltration/malicious URLs in output text.

        Returns HIGH severity findings with error spans.
        """
        if not text or not self.config.malicious_url_detection:
            return []
        findings = []
        for m in MALICIOUS_URL_RE.finditer(text):
            findings.append(GuardrailFinding(
                rule_id="malicious-url",
                severity="HIGH",
                snippet=m.group(0)[:60] + "…",
                start=m.start(),
                end=m.end(),
                direction="output",
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
                start=m.start(),
                end=m.end(),
                direction="output",
            ))
            return "[REDACTED-URL]"
        text = MALICIOUS_URL_RE.sub(_sub, text)
        return text, findings

    # --- Refusal detection (output, log-only) ---------------------------------

    def scan_refusal(self, text: str) -> list[GuardrailFinding]:
        """Detect LLM refusal patterns for monitoring (log-only, never blocks).

        Returns LOW severity findings with error spans.
        """
        if not text or not self.config.refusal_detection:
            return []
        findings = []
        for pid, pattern in REFUSAL_RULES:
            for m in pattern.finditer(text):
                findings.append(GuardrailFinding(
                    rule_id=pid,
                    severity="LOW",
                    snippet="refusal detected",
                    start=m.start(),
                    end=m.end(),
                    direction="output",
                ))
        return findings

    # --- System prompt leak detection (output) --------------------------------

    def scan_system_prompt_leak(self, text: str) -> list[GuardrailFinding]:
        """Detect system prompt content leaking in LLM responses.

        Uses fuzzy matching against configured system prompt fragments.
        Returns HIGH severity findings with error spans.

        Requires system_prompt_leak_detection=True and system_prompt_fragments
        to be configured.
        """
        if not text or not self._spleak_validator:
            return []
        return self._spleak_validator.scan(text)

    def mask_system_prompt_leak(self, text: str) -> tuple[str, list[GuardrailFinding]]:
        """Mask detected system prompt leaks with [REDACTED-SYSTEM-PROMPT]."""
        if not text or not self._spleak_validator:
            return text, []
        return self._spleak_validator.mask(text)

    # --- Full output processing ------------------------------------------------

    def _mask_text(self, text: str) -> tuple[str, list[GuardrailFinding]]:
        """Apply all output masks to text. Returns (masked_text, findings)."""
        masked, fs = self.mask_secrets(text)
        if self.config.pii_masking_enabled:
            masked, pii_fs = self.mask_pii(masked)
            fs.extend(pii_fs)
        if self.config.malicious_url_detection:
            masked, url_fs = self.mask_malicious_urls(masked)
            fs.extend(url_fs)
        if self._spleak_validator:
            masked, spleak_fs = self.mask_system_prompt_leak(masked)
            fs.extend(spleak_fs)
        return masked, fs

    def _scan_text(self, text: str) -> list[GuardrailFinding]:
        """Scan text without masking. Returns findings."""
        fs = self.scan_output_secrets(text)
        if self.config.malicious_url_detection:
            fs.extend(self.scan_malicious_urls(text))
        if self._spleak_validator:
            fs.extend(self.scan_system_prompt_leak(text))
        return fs

    def _process_text(self, text: str) -> tuple[str, list[GuardrailFinding]]:
        """Process text: mask or scan based on config. Returns (text, findings)."""
        if self.config.output_action == "mask":
            return self._mask_text(text)
        return text, self._scan_text(text)

    def process_response_content(self, message):
        """Mask secrets, PII, malicious URLs, and system prompt leaks in a
        message dict's content, in place. Returns findings.
        """
        if not self.config.output_enabled:
            return []
        self._update_spleak_validator()

        content = message.get("content") if isinstance(message, dict) else message
        findings: list[GuardrailFinding] = []
        if isinstance(content, str):
            result_text, fs = self._process_text(content)
            if result_text != content and isinstance(message, dict):
                message["content"] = result_text
            if self.config.refusal_detection:
                fs.extend(self.scan_refusal(content))
            findings.extend(fs)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    result_text, fs_block = self._process_text(block["text"])
                    if result_text != block["text"]:
                        block["text"] = result_text
                    if self.config.refusal_detection:
                        fs_block.extend(self.scan_refusal(block["text"]))
                    findings.extend(fs_block)
        return findings
