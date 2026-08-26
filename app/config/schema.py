"""Pydantic models mirroring settings.json structure."""
from __future__ import annotations

from typing import Any, Optional, Union
from pydantic import BaseModel, Field, model_validator


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    request_timeout_seconds: int = 600
    max_body_bytes: int = 10485760


class AuthConfig(BaseModel):
    enabled: bool = True
    header: str = "Authorization"
    allow_anonymous_health: bool = True


class PromptCachingConfig(BaseModel):
    """Provider-side prompt (KV) cache optimization via OpenRouter."""
    enabled: bool = True
    # Forward the router session_id upstream so OpenRouter provider
    # sticky routing keeps the conversation on one warm provider cache.
    forward_session_id: bool = True
    # Auto-inject cache_control breakpoints for Anthropic/Qwen routes.
    inject_cache_control: bool = True
    # Anthropic cache breakpoint TTL: "5m" (default) or "1h" (2x write cost).
    anthropic_ttl: str = "5m"
    # Approximate minimum stable-prefix tokens worth anchoring.
    min_tokens: int = 1024


class GuardrailsConfig(BaseModel):
    """LLM guardrails: input injection detection + output secret/PII masking."""
    input_enabled: bool = True
    # "log" — monitor only | "block" — reject request | "tag" — log + annotate
    input_action: str = "log"
    # Block requests with findings at/above this severity ("HIGH" or "CRITICAL")
    block_on_severity: str = "HIGH"
    output_enabled: bool = True
    # "mask" — replace secrets with ***REDACTED*** | "log" | "block"
    output_action: str = "mask"
    # Invisible text detection (input) — detect zero-width/format chars
    invisible_text_detection: bool = True
    # PII masking (output) — mask email, phone, SSN, credit card
    pii_masking_enabled: bool = True
    # Banned substrings (input) — configurable list, case-insensitive
    banned_substrings: list[str] = Field(default_factory=list)
    # Refusal detection (output) — log-only monitoring, never blocks
    refusal_detection: bool = True
    # Malicious URL detection (output) — detect/mask exfil domains
    malicious_url_detection: bool = True


class PrivacyConfig(BaseModel):
    """IP redaction & re-hydration privacy middleware."""
    enabled: bool = True
    # SQLite database file for session-scoped IP↔placeholder mappings.
    db_path: str = "/data/ip_redaction.db"
    # Purge mapping records older than this many hours (background job).
    retention_hours: float = 24.0
    # How often the purge job runs, in seconds.
    purge_interval_seconds: float = 3600.0


class ProviderConfig(BaseModel):
    name: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: int = 120
    connect_timeout_seconds: int = 10
    max_retries: int = 2
    retry_backoff_seconds: float = 1.5
    retry_on_status: list[int] = Field(default_factory=lambda: [429, 500, 502, 503, 504])
    headers: dict[str, str] = Field(default_factory=dict)
    pricing_refresh_seconds: int = 21600
    prompt_caching: PromptCachingConfig = Field(default_factory=PromptCachingConfig)


class DigestConfig(BaseModel):
    system_chars: int = 500
    tail_chars: int = 2000
    include_tool_names: bool = True
    include_context_summary: bool = True
    strip_scaffolding: bool = True
    learn_common_prefix: bool = True
    prefix_samples: int = 20
    min_prefix_chars: int = 200
    strip_sections_enabled: bool = True
    strip_sections: list[str] = Field(default_factory=list)
    keep_sections: list[str] = Field(default_factory=list)
    delimit_untrusted: bool = True
    injection_guard: bool = True


class ClassificationCacheConfig(BaseModel):
    enabled: bool = True
    ttl_seconds: int = 3600
    max_entries: int = 10000


class ClassificationConfig(BaseModel):
    enabled: bool = True
    model: str = "mistralai/mistral-small-3.2-24b-instruct"
    temperature: float = 0
    max_tokens: int = 60
    timeout_seconds: int = 8
    default_level: str = "L3"
    unknown_level: str = "L1"
    min_confidence: float = 0.5
    low_confidence_action: str = "escalate"
    prompt_file: str = "/app/config/prompts/classifier.txt"
    rubric_version: str = "v1"
    # Custom provider for classifier (optional — overrides global provider)
    base_url: Optional[str] = None
    # Environment variable name holding the API key for the classifier.
    api_key_env: Optional[str] = None
    digest: DigestConfig = Field(default_factory=DigestConfig)
    cache: ClassificationCacheConfig = Field(default_factory=ClassificationCacheConfig)


class EscalationConfig(BaseModel):
    enabled: bool = True
    threshold: int = 3
    decay_per_turn: int = 1
    cooldown_turns: int = 3
    max_escalations_per_session: int = 2
    never_downgrade: bool = True
    respect_max_level: bool = True
    explicit_signals_enabled: bool = True
    free_signals_enabled: bool = True
    signal_weights: dict[str, int] = Field(default_factory=lambda: {
        "repair_language": 3,
        "tool_error_loop": 3,
        "deep_keywords": 2,
        "context_growth": 2,
        "truncation": 2,
        "degenerate_response": 2,
        "turn_depth": 1,
        "code_volume_growth": 1,
    })
    escalate_after_turns: int = 12
    escalate_on_context_growth_tokens: int = 24000
    huge_context_hard_override: bool = True
    shadow_classify_every_n_turns: Optional[int] = None
    reclassify_every_n_turns: Optional[int] = None
    retry_on_failure: bool = False
    retry_on_failure_max_per_session: int = 2


class SessionConfig(BaseModel):
    enabled: bool = True
    backend: str = "memory"
    id_header: str = "X-Session-Id"
    use_user_field: bool = False
    fingerprint_fallback: bool = True
    fingerprint_salt: str = ""
    fingerprint_strip_patterns: list[str] = Field(default_factory=list)
    on_unidentifiable: str = "classify"
    idle_ttl_seconds: int = 7200
    max_ttl_seconds: int = 86400
    max_turns: Optional[int] = None
    max_sessions: int = 50000
    max_provisional_turns: int = 3
    lock_wait_ms: int = 5000
    lock_reservation_seconds: int = 30
    # Behavior when tier→model mapping changes in settings.json:
    #   "keep_level" — keep the session's pinned LEVEL, but re-resolve the
    #                  MODEL from the live config on every turn (default;
    #                  tier model changes apply to existing sessions).
    #   "keep_pin"   — keep both level and model exactly as pinned (frozen;
    #                  tier model changes apply to new sessions only).
    on_config_change: str = "keep_level"
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)


class HeuristicRule(BaseModel):
    name: str
    when: str
    level: str
    stop: bool = False


class HeuristicsConfig(BaseModel):
    enabled: bool = True
    measure: str = "task_payload"
    huge_context_tokens: int = 32000
    rules: list[HeuristicRule] = Field(default_factory=list)


class TierConfig(BaseModel):
    label: str
    model: str
    fallbacks: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    max_cost_per_request_usd: float = 1.0
    # "auto" = detect from OpenRouter /v1/models API (max_completion_tokens)
    # int  = use this fixed cap
    # unset/None = inherit from params or provider default
    max_tokens: Union[int, str] = "auto"
    # Custom provider configuration (optional — overrides global provider)
    # When set, the tier uses this base URL instead of settings.provider.base_url
    base_url: Optional[str] = None
    # Environment variable name holding the API key for this tier.
    # When set, the actual key is read from os.environ at request time.
    api_key_env: Optional[str] = None


class RoutingConfig(BaseModel):
    allow_passthrough: bool = False
    expose_upstream_models: bool = True
    global_max_level: str = "L4"
    global_min_level: str = "L1"
    L1: TierConfig = Field(default_factory=lambda: TierConfig(label="trivial", model=""))
    L2: TierConfig = Field(default_factory=lambda: TierConfig(label="easy", model=""))
    L3: TierConfig = Field(default_factory=lambda: TierConfig(label="medium", model=""))
    L4: TierConfig = Field(default_factory=lambda: TierConfig(label="hard", model=""))
    L5: TierConfig = Field(default_factory=lambda: TierConfig(label="extreme", model=""))

    def get_tier(self, level: str) -> TierConfig:
        return getattr(self, level)

    def get_model(self, level: str) -> str:
        return self.get_tier(level).model

    def get_fallbacks(self, level: str) -> list[str]:
        return self.get_tier(level).fallbacks

    def get_params(self, level: str) -> dict[str, Any]:
        return self.get_tier(level).params

    def get_max_tokens(self, level: str) -> Union[int, str]:
        """Return the tier's max_tokens setting ('auto' or int)."""
        return self.get_tier(level).max_tokens


class BudgetConfig(BaseModel):
    enabled: bool = False
    daily_limit_usd: float = 25.0
    on_exceeded: str = "downgrade"
    downgrade_to: str = "L2"


class TemporalAwarenessConfig(BaseModel):
    enabled: bool = False
    default_timezone: str = "UTC"  # IANA Time Zone Database format, e.g., "America/New_York"
    strategy: str = "replace"  # "replace" or "context_block"


class TelemetryConfig(BaseModel):
    log_level: str = "INFO"
    log_format: str = "json"
    log_prompts: bool = False
    log_prompt_hash: bool = True
    include_metadata_in_body: bool = False
    metrics_enabled: bool = True
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    temporal_awareness: TemporalAwarenessConfig = Field(default_factory=TemporalAwarenessConfig)


class Settings(BaseModel):
    version: int = 1
    server: ServerConfig = Field(default_factory=ServerConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    heuristics: HeuristicsConfig = Field(default_factory=HeuristicsConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    @model_validator(mode="after")
    def validate_no_secrets(self) -> "Settings":
        """Reject any value matching sk-* or sk-or-* patterns."""
        import json
        raw = json.dumps(self.model_dump())
        import re
        if re.search(r'"[^"]*sk-or-[A-Za-z0-9_-]+"', raw):
            raise ValueError("settings.json must not contain API keys (sk-or-* pattern detected)")
        if re.search(r'"[^"]*sk-[A-Za-z0-9]{20,}"', raw):
            raise ValueError("settings.json must not contain API keys (sk-* pattern detected)")
        return self

    @model_validator(mode="after")
    def validate_memory_workers(self) -> "Settings":
        """Memory backend with WORKERS > 1 is rejected at startup in loader."""
        return self
