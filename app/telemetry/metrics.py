"""Prometheus metrics definitions."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

# Request metrics
router_requests_total = Counter(
    "router_requests_total",
    "Total requests processed",
    ["level", "model", "source", "status"],
)

router_active_requests = Gauge(
    "router_active_requests",
    "Currently in-flight requests",
)

router_stream_errors_total = Counter(
    "router_stream_errors_total",
    "Upstream stream failures after response headers were sent",
    ["level", "model", "kind"],
)

# Session metrics
router_sessions_active = Gauge(
    "router_sessions_active",
    "Active session pins",
    ["level"],
)

router_sessions_created_total = Counter(
    "router_sessions_created_total",
    "Total sessions created",
    ["level", "id_source"],
)

router_sessions_expired_total = Counter(
    "router_sessions_expired_total",
    "Total sessions expired",
    ["reason"],
)

router_session_lookups_total = Counter(
    "router_session_lookups_total",
    "Session store lookups",
    ["result"],
)

router_session_turns = Histogram(
    "router_session_turns",
    "Turns per session at expiry",
    ["level"],
)

router_session_lock_waits_total = Counter(
    "router_session_lock_waits_total",
    "Session lock waits",
    ["outcome"],
)

# Classification metrics
router_classifier_calls_total = Counter(
    "router_classifier_calls_total",
    "Classifier calls",
    ["result"],
)

router_classification_duration_seconds = Histogram(
    "router_classification_duration_seconds",
    "Classification duration",
    ["source"],
)

router_classifier_failures_total = Counter(
    "router_classifier_failures_total",
    "Classifier failures",
    ["reason"],
)

# Reclassification and escalation
router_reclassifications_total = Counter(
    "router_reclassifications_total",
    "Reclassifications",
    ["trigger"],
)

router_escalations_total = Counter(
    "router_escalations_total",
    "Session escalations",
    ["from_level", "to_level", "trigger", "layer"],
)

router_escalation_signals_total = Counter(
    "router_escalation_signals_total",
    "Escalation signals fired",
    ["signal"],
)

router_escalations_capped_total = Counter(
    "router_escalations_capped_total",
    "Escalations blocked by max_level",
    ["level"],
)

router_escalation_turn = Histogram(
    "router_escalation_turn",
    "Turn number at escalation",
)

router_retry_on_failure_total = Counter(
    "router_retry_on_failure_total",
    "Retry-on-failure attempts",
    ["from_level", "outcome"],
)

# Upstream metrics
router_upstream_duration_seconds = Histogram(
    "router_upstream_duration_seconds",
    "Upstream call duration",
    ["level", "model"],
)

router_fallbacks_total = Counter(
    "router_fallbacks_total",
    "Fallback model used",
    ["level", "from_model", "to_model", "reason"],
)

# Token and cost metrics
router_tokens_total = Counter(
    "router_tokens_total",
    "Tokens consumed",
    ["level", "model", "kind"],
)

router_cost_usd_total = Counter(
    "router_cost_usd_total",
    "Cost in USD",
    ["level", "model"],
)

# Digest/scaffolding metrics
router_digest_scaffolding_ratio = Histogram(
    "router_digest_scaffolding_ratio",
    "Scaffolding stripped ratio",
)

router_digest_task_tokens = Histogram(
    "router_digest_task_tokens",
    "Task payload tokens after stripping",
)

router_scaffolding_strip_source_total = Counter(
    "router_scaffolding_strip_source_total",
    "Scaffolding strip mechanism used",
    ["source"],
)

router_injection_suspected_total = Counter(
    "router_injection_suspected_total",
    "Injection suspected in digest",
)

# Cache metrics
router_cache_events_total = Counter(
    "router_cache_events_total",
    "Classification cache events",
    ["result"],
)

# Upstream prompt (KV) cache metrics
router_prompt_cached_tokens_total = Counter(
    "router_prompt_cached_tokens_total",
    "Prompt tokens served from provider cache (cache reads)",
    ["level", "model"],
)

router_prompt_cache_writes_total = Counter(
    "router_prompt_cache_writes_total",
    "Prompt tokens written to provider cache (cache writes)",
    ["level", "model"],
)

router_prompt_cache_hit_ratio = Gauge(
    "router_prompt_cache_hit_ratio",
    "Rolling provider prompt-cache hit ratio (cached_tokens / prompt_tokens)",
    ["level", "model"],
)

# Privacy middleware metrics
router_privacy_redactions_total = Counter(
    "router_privacy_redactions_total",
    "Requests that passed through IP redaction",
)

# Tier-prefix pinning metrics
router_tier_prefix_pins_total = Counter(
    "router_tier_prefix_pins_total",
    "Sessions pinned via tier-prefix detection (bypasses classifier)",
    ["level"],
)

# Guardrail metrics
router_guardrail_findings_total = Counter(
    "router_guardrail_findings_total",
    "Guardrail findings by rule and direction",
    ["rule_id", "severity", "direction"],
)

router_guardrail_blocks_total = Counter(
    "router_guardrail_blocks_total",
    "Requests blocked by the input guardrail",
    ["rule_id", "severity"],
)

router_guardrail_secret_masks_total = Counter(
    "router_guardrail_secret_masks_total",
    "Secrets masked in LLM output by provider pattern",
    ["rule_id"],
)

# Info
router_info = Info(
    "router",
    "Router build information",
)
