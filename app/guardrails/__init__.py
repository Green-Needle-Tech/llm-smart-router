"""LLM Guardrails: input injection/jailbreak detection + output secret masking.

Two independent guardrails that run at the router layer (below the AI
agent's own guardrails and the upstream LLM API's safety filters):

1. INPUT — prompt injection / jailbreak detection
   A curated regex catalog (categories: instruction override, jailbreak
   personas, system-prompt/secret exfiltration, tool abuse, sandbox
   evasion, social engineering, encoded payloads, multi-turn manipulation)
   compiled from published detection rule sets. Configurable action:
   "log" (monitor) | "block" (reject with 400) | "tag" (annotate + forward).

2. OUTPUT — secret / credential leak masking + system prompt leak detection
   Scans LLM responses for provider-prefixed API keys (sk-, sk-or-, sk-ant-,
   ghp_, github_pat_, AKIA…, AIza…, xox…, glpat-…) and PEM blocks, masking
   them before the response reaches the caller. Also detects system prompt
   leakage via fuzzy matching against configured fragments.
   Action: "mask" | "log" | "block".

P0 Improvements (Aug 2026):
- Validator abstraction layer (base.py): composable BaseValidator + ValidatorRegistry
- Error spans (start, end) on all GuardrailFinding results
- System prompt leak detection (validators.py): fuzzy match output vs fragments

False-positive reduction: matches inside fenced code blocks are skipped
for injection detection (educational/discussion contexts), per common
practice in published rule sets.
"""
from app.guardrails.base import (
    SEV_ORDER,
    BaseValidator,
    GuardrailFinding,
    RegexValidator,
    ValidatorRegistry,
    severity_at_least,
)
from app.guardrails.scanner import GuardrailConfig, GuardrailEngine
from app.guardrails.validators import SystemPromptLeakValidator

__all__ = [
    "SEV_ORDER",
    "BaseValidator",
    "GuardrailConfig",
    "GuardrailEngine",
    "GuardrailFinding",
    "RegexValidator",
    "SystemPromptLeakValidator",
    "ValidatorRegistry",
    "severity_at_least",
]
