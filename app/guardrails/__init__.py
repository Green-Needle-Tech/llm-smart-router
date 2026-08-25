"""LLM Guardrails: input injection/jailbreak detection + output secret masking.

Two independent guardrails that run at the router layer (below the AI
agent's own guardrails and the upstream LLM API's safety filters):

1. INPUT — prompt injection / jailbreak detection
   A curated regex catalog (categories: instruction override, jailbreak
   personas, system-prompt/secret exfiltration, tool abuse, sandbox
   evasion, social engineering, encoded payloads, multi-turn manipulation)
   compiled from published detection rule sets. Configurable action:
   "log" (monitor) | "block" (reject with 400) | "tag" (annotate + forward).

2. OUTPUT — secret / credential leak masking
   Scans LLM responses for provider-prefixed API keys (sk-, sk-or-, sk-ant-,
   ghp_, github_pat_, AKIA…, AIza…, xox…, glpat-…) and PEM blocks, masking
   them before the response reaches the caller. Action: "mask" | "log" | "block".

False-positive reduction: matches inside fenced code blocks are skipped
for injection detection (educational/discussion contexts), per common
practice in published rule sets.
"""
from app.guardrails.scanner import GuardrailEngine, GuardrailConfig
