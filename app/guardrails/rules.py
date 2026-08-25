"""Guardrail rule catalog: injection patterns + secret patterns.

Injection patterns are adapted from published prompt-injection detection
rule sets (8 categories, CRITICAL/HIGH/MEDIUM/LOW severities). Secret
patterns are provider-prefixed, high-precision credential regexes.
"""
from __future__ import annotations

import re

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
]

# Compiled once at import.
COMPILED_INJECTION = [
    (pid, sev, re.compile(pat, re.IGNORECASE | re.MULTILINE))
    for pid, sev, pat in INJECTION_RULES
]

# Provider-prefixed, high-precision credential patterns (low FP by design).
SECRET_RULES = [
    ("openrouter-key", re.compile(r"sk-or-v1-[A-Za-z0-9]{16,}")),
    ("anthropic-key", re.compile(r"sk-ant-(api[0-9]*-)?[A-Za-z0-9_-]{16,}")),
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
