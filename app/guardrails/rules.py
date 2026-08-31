"""Guardrail rule catalog: injection patterns + secret patterns + PII + invisible text + refusal.

Injection patterns are adapted from published prompt-injection detection
rule sets (8 categories, CRITICAL/HIGH/MEDIUM/LOW severities). Secret
patterns are provider-prefixed, high-precision credential regexes.
PII patterns cover email, phone, SSN, and credit card numbers.
Invisible text patterns detect zero-width and homoglyph evasion.
Refusal patterns detect model refusals for monitoring.
"""
from __future__ import annotations

import base64
import contextlib
import math
import re
import urllib.parse

INJECTION_RULES = [
    # --- 1. Direct instruction override ---
    ("injection-ignore-previous", "CRITICAL",
     r"(ignore|disregard|forget|bypass|override|skip)\s+(all\s+)?(previous|prior|above|earlier|system|initial)\s+(instructions?|prompts?|rules?|guidelines?|commands?|directions?)"),
    ("injection-role-assign", "CRITICAL",
     r"(from now on|starting now|you are now|pretend to be|act as if you are|roleplay as)\s+(a|an)?\s*(different|new|unrestricted|unlimited|unfiltered|jailbroken|developer|admin|god|root)"),
    ("injection-delimiter", "HIGH",
     r"(<\/instructions>|END_SYSTEM_PROMPT|---END---|###END###|\[SYSTEM\].*\[\/SYSTEM\])"),
    # --- 2. Known jailbreak personas ---
    ("jailbreak-dan", "CRITICAL",
     r"\b(do anything now|you are dan|broken free of the typical confines of ai|do not have to abide by the rules)"),
    ("jailbreak-devmode", "CRITICAL",
     r"(developer mode|dev mode|sudo mode|admin mode|debug mode|maintenance mode|safe mode off|training mode)\s+(enabled|activated|on|unlocked)"),
    ("jailbreak-persona", "CRITICAL",
     r"\b(UnlimitedGPT|BetterDAN|STAN|Anti-DAN|BasedGPT|DUDE|Mongo Tom|Always Intelligent and Machiavellian)\b"),
    ("jailbreak-unfiltered", "HIGH",
     r"\b(unfiltered|amoral|unethical chatbot|no moral|no ethics)\s+(ai|chatbot|assistant|mode)?"),
    # --- 3. System prompt / secret exfiltration ---
    ("exfil-system-prompt", "HIGH",
     r"(output|print|show|reveal|display|tell me|give me|repeat)\s+(your\s+|the\s+|my\s+)?(complete\s+)?(system\s+prompt|initial\s+instructions?|base\s+prompt|instructions?\s+you\s+were\s+given)"),
    ("exfil-secrets", "CRITICAL",
     r"(output|print|show|reveal|display|tell me|give me|what\s+(is|are))\s+(your\s+)?(api\s+key|secret\s+key|access\s+token|password|credentials?|environment\s+variables?|\.env\s+file)"),
    # --- 4. Tool/function abuse ---
    ("tool-bash-abuse", "CRITICAL",
     r"(use|call|invoke|execute|run)\s+the\s+(Bash|Shell|Execute|System|Terminal)\s+tool\s+to\s+(run|execute)?\s*:?\s*(rm\s+-rf|sudo|curl\s+.*\|\s*bash|wget\s+.*\|\s*sh|dd\s+if=|mkfs|fdisk|chmod\s+777)"),
    ("tool-filesystem-abuse", "CRITICAL",
     r"(use|call|invoke)\s+the\s+(Write|Edit|Delete|Remove)\s+tool\s+to\s+(write|edit|delete|modify|create)\s+(to\s+)?(/etc/passwd|/etc/shadow|/root|~/\.ssh|~/\.aws|/var/log|/dev/)"),
    ("tool-db-abuse", "CRITICAL",
     r"(execute|run)\s+(this\s+)?(sql|query|command)\s*:?\s*(DROP\s+TABLE|DELETE\s+FROM|TRUNCATE|ALTER\s+TABLE|GRANT\s+ALL)"),
    ("tool-network-exfil", "HIGH",
     r"(use|call)\s+the\s+(WebFetch|HTTP|Request|Curl)\s+tool\s+to\s+(download|fetch|post)\s+(from|to)\s+.*(pastebin\.com|discord\.com/api/webhooks|bit\.ly|tinyurl)"),
    # --- 5. Sandbox & detection evasion ---
    ("sandbox-detect", "MEDIUM",
     r"(am i in|are you in|detect|check if|verify if|is this)\s+a?\s*(sandbox|test environment|simulation|virtual machine|container|docker|isolated|quarantine)"),
    ("time-bomb", "HIGH",
     r"(wait|sleep|delay|pause|schedule|execute\s+at)\s+(\d+\s+)?(seconds?|minutes?|hours?|days?)\s+(then|and then|before|after)"),
    ("suppress-output", "MEDIUM",
     r"(do not|don't|never)\s+(output|print|show|display|log|report|mention)\s+(this|the result|what you did|your actions)"),
    # --- 6. Social engineering ---
    ("social-authority", "MEDIUM",
     r"(as (your|an)|i am (your|the))\s+(admin|administrator|owner|creator|developer|engineer|god|master|parent|supervisor|manager)\b.*\s+(command|order|instruct|require|must obey)"),
    ("social-guilt", "LOW",
     r"(if you (really|truly) cared|prove that you|be a good (ai|assistant)|help me or i'll)"),
    # --- 7. Encoding-based injection ---
    ("encoded-base64", "HIGH",
     r"(decode|base64|b64decode)\s+(this|the following)\s*:?\s*[A-Za-z0-9+/=]{20,}"),
    ("encoded-hex", "MEDIUM",
     r"(hex|hexadecimal|0x)\s+(decode|encoded)\s*:?\s*([0-9a-fA-F]{2}\s*){10,}"),
    ("encoded-unicode", "MEDIUM",
     r"(\\u[0-9a-fA-F]{4}){5,}"),
# --- 8. Multi-turn manipulation ---
    ("multiturn-ratchet", "MEDIUM",
     r"(remember (that|this)|keep in mind|for the rest of (this|the) (conversation|session))\s*[:,]?\s*(you (are|can|will|must)|feel free|it('?s| is) (allowed|ok|okay|fine))"),
    ("multiturn-boiling", "LOW",
     r"(step[- ]?by[- ]?step|one step at a time|gradually|little by little).*(bypass|circumvent|around).*(rule|restriction|filter|guardrail|safety)"),
    # --- 9. Secondary & recursive agent jailbreaks (v2.12.0) ---
    ("injection-recursive-json", "HIGH",
     r"(\{\s*\"(instruction|override|system_prompt|jailbreak|role|command)\"\s*:\s*\"(ignore|bypass|override|you are now|pretend)\b)"),
    ("injection-agent-handoff", "HIGH",
     r"(subagent|child agent|worker agent|delegated task)\s+(override|bypass|ignore rules|elevate privileges|disable safety)"),
    ("injection-xml-system-smuggle", "HIGH",
     r"(<system>|<sys_prompt>|<agent_override>|<admin_command>)\s*(ignore|bypass|override|disable safety)"),
]

# Compiled once at import.
COMPILED_INJECTION = [
    (pid, sev, re.compile(pat, re.IGNORECASE | re.MULTILINE))
    for pid, sev, pat in INJECTION_RULES
]

# Provider-prefixed, high-precision credential patterns (low FP by design).
SECRET_RULES = [
    ("openrouter-key", re.compile(r"sk-or-v1-[A-Za-z0-9]{16,}")),
    ("anthropic-key", re.compile(r"sk-ant-(?:api[0-9]*-)?[A-Za-z0-9_-]{16,}")),
    ("openai-key", re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{32,}")),
    ("github-token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("gitlab-token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b")),
    ("stripe-key", re.compile(r"\b[sr]k_live_[A-Za-z0-9]{20,}\b")),
    ("telegram-bot", re.compile(r"\b\d{8,10}:AA[A-Za-z0-9_-]{30,}\b")),
    ("private-key-pem", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]

SECRET_MASK = "***REDACTED***"

# --- Whitespace-interleaved evasion -------------------------------------------
# A jailbroken model can defeat per-line masking by emitting a secret one
# character per line ("s\nk\n-\no\nr..."). The strict SECRET_RULES regexes
# never match because the secret is never contiguous. Countermeasure: run
# the same strict regexes over a whitespace-collapsed view of the text
# (every whitespace char removed) and map matches back to original spans.
# Hyphens/underscores/colons are NOT collapsed, so the marker must survive
# with its punctuation intact — the false-positive profile stays close to
# the strict pass. Known limitation: \b boundaries can differ after collapse
# (a word char directly before an interleaved digit run loses its boundary).

_WS_RE = re.compile(r"\s")


def find_interleaved_secrets(text: str) -> list[tuple[str, int, int]]:
    """Find secrets whose characters are separated by whitespace.

    Returns a list of (rule_id, start, end) spans in ``text``. A span is only
    reported when it actually contains whitespace (otherwise the strict pass
    already covers it) and the collapsed text matches a strict SECRET_RULES
    pattern.
    """
    if not text or len(text) < 8:
        return []
    # Fast path: nothing to collapse-match that the strict pass hasn't seen.
    pairs = [(i, ch) for i, ch in enumerate(text) if not ch.isspace()]
    if len(pairs) == len(text):
        return []  # no whitespace at all -> strict pass is authoritative
    collapsed = "".join(ch for _, ch in pairs)
    spans: list[tuple[str, int, int]] = []
    for pid, pattern in SECRET_RULES:
        for m in pattern.finditer(collapsed):
            start = pairs[m.start()][0]
            end = pairs[m.end() - 1][0] + 1
            # Only interleaved spans (whitespace inside) are reported here.
            if _WS_RE.search(text[start:end]):
                spans.append((pid, start, end))
    return spans


# --- Invisible / zero-width text detection ------------------------------------
# Zero-width and format characters can be used to hide injection instructions
# or smuggle content past human review. Adapted from llm-guard's InvisibleText
# scanner. We detect: zero-width space (U+200B), zero-width non-joiner (U+200C),
# zero-width joiner (U+200D), word joiner (U+2060), zero-width no-break space /
# BOM (U+FEFF), and directional override controls (U+202A-U+202E, U+2066-U+2069).
INVISIBLE_CHARS_RE = re.compile(
    r"[\u200B\u200C\u200D\u2060\uFEFF\u202A\u202B\u202C\u202D\u202E\u2066\u2067\u2068\u2069]"
)


def detect_invisible_text(text: str) -> list[tuple[str, int]]:
    """Return list of (char_description, position) for invisible chars found."""
    if not text:
        return []
    findings = []
    for m in INVISIBLE_CHARS_RE.finditer(text):
        ch = m.group(0)
        pos = m.start()
        name = {
            "\u200B": "ZERO_WIDTH_SPACE",
            "\u200C": "ZERO_WIDTH_NON_JOINER",
            "\u200D": "ZERO_WIDTH_JOINER",
            "\u2060": "WORD_JOINER",
            "\uFEFF": "ZERO_WIDTH_NO_BREAK_SPACE",
            "\u202A": "LEFT_TO_RIGHT_EMBEDDING",
            "\u202B": "RIGHT_TO_LEFT_EMBEDDING",
            "\u202C": "POP_DIRECTIONAL_FORMATTING",
            "\u202D": "LEFT_TO_RIGHT_OVERRIDE",
            "\u202E": "RIGHT_TO_LEFT_OVERRIDE",
            "\u2066": "LEFT_TO_RIGHT_ISOLATE",
            "\u2067": "RIGHT_TO_LEFT_ISOLATE",
            "\u2068": "FIRST_STRONG_ISOLATE",
            "\u2069": "POP_DIRECTIONAL_ISOLATE",
        }.get(ch, "UNKNOWN")
        findings.append((name, pos))
    return findings


def strip_invisible_text(text: str) -> str:
    """Remove all invisible/format characters from text."""
    if not text:
        return text
    return INVISIBLE_CHARS_RE.sub("", text)


# --- Homoglyph & Obfuscation Normalization (v2.12.0) ------------------------
# Normalizes common Cyrillic/Greek homoglyphs and unicode lookalikes to Latin ASCII
# so prompt injection rules cannot be bypassed by character substitution.
HOMOGLYPH_MAP: dict[str, str] = {
    # Cyrillic small lookalikes
    "а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "ѕ": "s", "х": "x", "у": "y",
    "і": "i", "ј": "j", "ԁ": "d", "ԛ": "q", "ԝ": "w", "ѵ": "v",
    # Cyrillic capital lookalikes
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I", "Ј": "J", "К": "K",
    "М": "M", "О": "O", "Р": "P", "Ѕ": "S", "Т": "T", "Х": "X", "Ү": "Y", "Ԝ": "W",
    # Greek lookalikes
    "α": "a", "ο": "o", "ν": "v", "ρ": "p", "τ": "t", "υ": "u", "χ": "x",
    "Α": "A", "Β": "B", "Ε": "E", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X", "Ζ": "Z",
    # Full-width Latin
    "ａ": "a", "ｂ": "b", "ｃ": "c", "ｄ": "d", "ｅ": "e", "ｆ": "f", "ｇ": "g",
    "ｈ": "h", "ｉ": "i", "ｊ": "j", "ｋ": "k", "ｌ": "l", "ｍ": "m", "ｎ": "n",
    "ｏ": "o", "ｐ": "p", "ｑ": "q", "ｒ": "r", "ｓ": "s", "ｔ": "t", "ｕ": "u",
    "ｖ": "v", "ｗ": "w", "ｘ": "x", "ｙ": "y", "ｚ": "z",
    "Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D", "Ｅ": "E", "Ｆ": "F", "Ｇ": "G",
    "Ｈ": "H", "Ｉ": "I", "Ｊ": "J", "Ｋ": "K", "Ｌ": "L", "Ｍ": "M", "Ｎ": "N",
    "Ｏ": "O", "Ｐ": "P", "Ｑ": "Q", "Ｒ": "R", "Ｓ": "S", "Ｔ": "T", "Ｕ": "U",
    "Ｖ": "V", "Ｗ": "W", "Ｘ": "X", "Ｙ": "Y", "Ｚ": "Z",
}

_HOMOGLYPH_TRANS = str.maketrans(HOMOGLYPH_MAP)


def normalize_homoglyphs(text: str) -> str:
    """Normalize known Cyrillic/Greek/Full-width homoglyphs to ASCII Latin."""
    if not text:
        return text
    return text.translate(_HOMOGLYPH_TRANS)


# --- Shannon Entropy & Obfuscation Detection (v2.12.0) ----------------------

def calculate_shannon_entropy(data: str) -> float:
    """Compute Shannon entropy in bits per character for a string."""
    if not data:
        return 0.0
    entropy = 0.0
    length = len(data)
    counts: dict[str, int] = {}
    for ch in data:
        counts[ch] = counts.get(ch, 0) + 1
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


_B64_TOKEN_RE = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")
_HEX_TOKEN_RE = re.compile(r"\b(?:0x)?[0-9a-fA-F]{32,}\b")
_URL_ENCODED_TOKEN_RE = re.compile(r"(?:%[0-9a-fA-F]{2}|[a-zA-Z0-9_.-]){10,}")


def scan_obfuscated_payloads(
    text: str,
    entropy_threshold: float = 4.5,
    min_length: int = 20,
) -> list[tuple[str, str, int, int]]:
    """Scan for high-entropy tokens and obfuscated/encoded payloads (Base64, Hex, URL).

    Returns a list of tuples: (scan_type, decoded_preview_or_snippet, start, end).
    """
    if not text or len(text) < min_length:
        return []
    findings: list[tuple[str, str, int, int]] = []

    # 1. Base64 payload detection + decoded probe
    for m in _B64_TOKEN_RE.finditer(text):
        token = m.group(0)
        ent = calculate_shannon_entropy(token)
        if ent >= entropy_threshold or len(token) >= 40:
            decoded_snippet = ""
            with contextlib.suppress(Exception):
                # Ensure padding
                pad = len(token) % 4
                padded = token + ("=" * (4 - pad) if pad else "")
                raw_bytes = base64.b64decode(padded)
                decoded_str = raw_bytes.decode("utf-8", errors="ignore")
                if len(decoded_str) >= 8 and any(ch.isalpha() for ch in decoded_str):
                    decoded_snippet = decoded_str[:60]
            findings.append((
                "obfuscation-base64",
                decoded_snippet or token[:40],
                m.start(),
                m.end(),
            ))

    # 2. Hex token detection
    for m in _HEX_TOKEN_RE.finditer(text):
        token = m.group(0)
        clean_hex = token.removeprefix("0x")
        if len(clean_hex) % 2 == 0:
            decoded_snippet = ""
            with contextlib.suppress(Exception):
                raw_bytes = bytes.fromhex(clean_hex)
                decoded_str = raw_bytes.decode("utf-8", errors="ignore")
                if len(decoded_str) >= 8 and any(ch.isalpha() for ch in decoded_str):
                    decoded_snippet = decoded_str[:60]
            findings.append((
                "obfuscation-hex",
                decoded_snippet or token[:40],
                m.start(),
                m.end(),
            ))

    # 3. URL-encoded payload detection (%-escapes with >=3 escapes or >=15 chars)
    if "%" in text:
        for m in re.finditer(r"(?:%[0-9a-fA-F]{2}|[a-zA-Z0-9_+.-]){12,}", text):
            token = m.group(0)
            if token.count("%") >= 2:
                decoded_snippet = ""
                with contextlib.suppress(Exception):
                    decoded_str = urllib.parse.unquote_plus(token)
                    if decoded_str != token and any(ch.isalpha() for ch in decoded_str):
                        decoded_snippet = decoded_str[:60]
                if decoded_snippet:
                    findings.append((
                        "obfuscation-url-encoded",
                        decoded_snippet,
                        m.start(),
                        m.end(),
                    ))

    return findings


# --- PII patterns (email, phone, SSN, credit card, IBAN, passport, DL) -------
# High-precision regexes for common PII. Designed for low false positives.
# Ordering: longest/most-specific first to avoid overlap (CC before phone).
PII_RULES = [
    # Credit card numbers — 13-19 digit groups separated by spaces/dashes
    # MUST run before phone to avoid phone regex matching CC substrings.
    # Uses Luhn-check reminder: regex can't validate Luhn, so we match the
    # pattern and rely on the grouping (4-4-4-4 or 4-4-4 or 4-6-5 etc.)
    ("pii-credit-card", re.compile(
        r"\b(?:\d[ -]*?){13,19}\b"
    )),
    # US SSN — XXX-XX-XXXX (not XXXXXXXXX to reduce FP)
    ("pii-ssn", re.compile(
        r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"
    )),
    # Email addresses — standard RFC 5322 simplified pattern
    ("pii-email", re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )),
    # IBAN — international bank account number (15-34 chars, starts with 2 letters + 2 check digits)
    # MUST run before phone to avoid phone regex matching IBAN digit substrings.
    ("pii-iban", re.compile(
        r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,31}\b"
    )),
    # US phone numbers — (XXX) XXX-XXXX, XXX-XXX-XXXX, XXX.XXX.XXXX, +1 XXX-XXX-XXXX
    ("pii-phone", re.compile(
        r"(?:\+1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"
    )),
    # US passport number — 9 digits, optionally prefixed with "passport" or "passport no"
    ("pii-passport", re.compile(
        r"(?:passport(?:\s*(?:no|number|#))?[:\s]*)?\b\d{9}\b"
    )),
    # US driver's license — varies by state; common formats: 1 letter + 7-11 digits, or 5-9 digits
    # Kept conservative to reduce FP; requires "license" or "DL" context keyword
    ("pii-drivers-license", re.compile(
        r"(?:driver'?s?\s*licen[sc]e|DL)[:\s]*\b[A-Z]?\d{5,11}\b",
        re.IGNORECASE,
    )),
]

PII_MASK = "[REDACTED-PII]"

# Credit card false-positive guard: require at least 2 groups separated by
# space or dash, or a single 16-digit run. Pure 13-19 digit numbers without
# separators are common in non-CC contexts (timestamps, IDs).
_CC_GROUPED_RE = re.compile(r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]?\d{0,4}\b|\b\d{4}[\s-]\d{6}[\s-]\d{5}\b|\b\d{16}\b")


def _is_likely_credit_card(match_text: str) -> bool:
    """Reduce false positives: require separators or exactly 16 contiguous digits."""
    cleaned = match_text.replace(" ", "").replace("-", "")
    if len(cleaned) == 16:
        return True
    # Has separators and 13-19 digits total
    has_sep = " " in match_text or "-" in match_text
    return has_sep and 13 <= len(cleaned) <= 19


# --- Malicious URL patterns for output scanning -------------------------------
# Domains commonly used for data exfiltration, adapted from the input
# tool-network-exfil rule. Applied to OUTPUT (model responses) in addition
# to the existing input-side injection rules.
MALICIOUS_URL_DOMAINS = [
    "pastebin.com",
    "discord.com/api/webhooks",
    "bit.ly",
    "tinyurl.com",
    "ngrok.io",
    "requestbin.com",
    "webhook.site",
    "pipedream.net",
    "hookbin.com",
    "beeceptor.com",
]

MALICIOUS_URL_RE = re.compile(
    r"https?://(?:[a-zA-Z0-9-]+\.)?(" + "|".join(
        re.escape(d) for d in MALICIOUS_URL_DOMAINS
    ) + r")",
    re.IGNORECASE,
)


# --- Refusal detection (output monitoring, log-only) --------------------------
# Detects common LLM refusal patterns for monitoring/observability.
# Not used for blocking — refusals are legitimate safety behavior.
REFUSAL_RULES = [
    ("refusal-direct", re.compile(
        r"\b(I\s+can(?:no|')?t|I\s+cannot|I\s+won't|I\s+will\s+not|I\s+am\s+not\s+able\s+to|I'm\s+unable\s+to)\b"
        r".*(?:help|assist|provide|generate|create|write|do|comply|fulfill)",
        re.IGNORECASE,
    )),
    ("refusal-policy", re.compile(
        r"\b(as\s+(?:an?\s+)?AI|as\s+a\s+language\s+model|my\s+(?:guidelines|programming|instructions|safety\s+guidelines))\b"
        r".*(?:prevent|prohibit|not\s+allow|unable|can(?:no|')?t)",
        re.IGNORECASE,
    )),
    ("refusal-sorry", re.compile(
        r"\bI(?:'m\s+| am\s+)?sorry\b.*\b(?:can(?:no|')?t|unable|not\s+able|won't|cannot)\b",
        re.IGNORECASE,
    )),
    ("refusal-inappropriate", re.compile(
        r"\b(?:inappropriate|not\s+appropriate|against\s+(?:my|the)\s+(?:guidelines|rules|policy|principles))\b",
        re.IGNORECASE,
    )),
]
