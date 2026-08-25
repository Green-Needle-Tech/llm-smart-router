# LLM Smart Router

<p align="center">
  <img src="assets/icon.png" width="200" alt="LLM Smart Router logo">
</p>

A self-hosted Docker application that exposes an OpenAI-compatible API, classifies the **first prompt of each chat session** by task complexity (L1–L5), pins that session to the matching OpenRouter model, and routes every subsequent turn straight to the pinned model without re-classifying.

## Quick Start

```bash
cp .env.example .env   # fill in OPENROUTER_API_KEY and ROUTER_API_KEY
docker compose up -d --build
curl localhost:8080/healthz
```

Point any OpenAI-compatible client at `http://localhost:8080/v1`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="your-router-key")
r = client.chat.completions.create(
    model="smart-router",
    messages=[{"role":"user","content":"Write a commit message for: fix typo"}],
    extra_headers={"X-Session-Id": "my-session-1"},
)
print(r.model)  # actual model used
```

## How It Works

- **Turn 1**: Classifier assigns L1–L5 → session pinned to the matching model
- **Turn 2+**: Straight to the pinned model, no classifier call (sub-ms lookup)
- **Escalation**: Free signals (repair language, tool errors, etc.) can ratchet the tier up mid-session
- **Config changes**: `session.on_config_change: keep_level` (default) re-resolves the tier's model per turn after settings changes — no pin expiry wait, no re-classification

## What's New in v2.1.0

### 🔐 Streaming Secret-Leak Hardening

Three streaming secret-masking vectors found and fixed by a full end-to-end guardrails audit. All fixes are in the streaming carry-buffer pipeline (`app/guardrails/streaming.py` + `app/api/chat.py`):

**1. Telegram bot token streaming leak**
Tokenizers split Telegram bot tokens at the `:AA` separator or emit them character-by-character. The carry buffer's digit-run hold (`\d{4,10}`) was too narrow — single-digit chunks and digits+colon splits flushed before the full-secret regex could reassemble.

- `_TG_DIGITS_COLON_RE`: holds `123456789:` + partial `:AA` continuation
- `_TG_DIGIT_RUN_RE`: holds any trailing digit run ≥1 (was ≥4)
- `_collapsed_tail_hold()`: whitespace-interleaved partial-secret hold

**2. Tail leak on long secrets**
`mask_secrets` fired at minimum regex length mid-growth (e.g. `sk-or-v1-` + 16 chars), destroying the marker — the remaining body (up to 43+ chars) then flushed as plaintext.

- **Pipeline reorder**: `_rehydrate_chunk` now splits FIRST (holds the growing tail in carry), then masks only the flushable (terminated) part
- **(a2) tail-leak guard**: holds still-growing bodies even when ≥ threshold+MARGIN, until a non-body char terminates them

**3. Whitespace-interleaved evasion (engine-wide)**
A jailbroken model could emit a secret one character per line (`s\nk\n-\no\nr...`) — no contiguous regex matches, in either streaming or non-streaming mode.

- `find_interleaved_secrets()`: collapses all whitespace, runs strict SECRET_RULES over the collapsed text, maps matches back to original spans
- Wired into `mask_secrets()` as a second pass with overlap-safe span merging
- `_collapsed_tail_hold()`: extends the carry to hold interleaved partial secrets in streaming mode

**4. [DONE] carry flush**
The final carry buffer at `data: [DONE]` now runs through `mask_secrets()` before emitting — a secret that completes only at stream end is masked, not emitted raw.

### Test Results (2026-08-25)

| Suite | Tests | Result |
|-------|-------|--------|
| Unit (`pytest tests/ -q`) | 224 | ✅ All passed |
| Live full guardrails (`test_guardrails_full.py`) | 30 | ✅ All passed |
| Live e2e block + streaming (`test_guardrails_e2e_block_stream.py`) | 22 | ✅ All passed |

**e2e coverage**: block-mode enforcement (4 injection payloads → HTTP 400, 2 benign → 200, 2 severity-gate → 200), streaming secret masking (11 provider types + split-carry), streaming IP redaction round-trip.

## What's New in v2.0.0-beta

Three new request-path features, all hot-reloadable and enabled by default:

### 🔒 IP Redaction & Re-Hydration (`telemetry.privacy`)
Raw IP addresses in prompts are replaced with session-stable placeholders (`[ipaddress-01]`, …) before the request reaches the classifier or any upstream model, and re-hydrated back to the original IPs in the response. The classifier and tier models never see a real IP.
- Session-scoped SQLite mapping store (`/app/data/ip_redaction.db`, Docker volume `router-data`)
- Same IP → same placeholder across turns (context- and prefix-cache-friendly)
- Ports (`:8080`) and CIDR (`/24`) preserved; IPv4 + IPv6
- Streaming-safe: placeholders split across SSE chunks still re-hydrate (carry buffer)
- 24-hour retention with background purge job

### 🛡️ LLM Guardrails (`telemetry.guardrails`)
Two router-layer guardrails, independent of your agent's or the upstream API's own safety filters:
- **Input**: 24-rule prompt-injection/jailbreak catalog (8 categories: instruction override, jailbreak personas, system-prompt/secret exfiltration, tool abuse, sandbox evasion, social engineering, encoded payloads, multi-turn manipulation) with CRITICAL/HIGH/MEDIUM/LOW severities. Actions: `log` (default) | `block` (400 at/above severity threshold). Code-block-heavy messages skipped to avoid false positives.
- **Output**: 11 provider-prefixed credential patterns (OpenRouter, OpenAI, Anthropic, GitHub, AWS, Google, Slack, GitLab, Stripe, Telegram, PEM) masked with `***REDACTED***` before responses reach the caller.
- Run `python3 scripts/test_guardrails.py` for the router-level differential test (bypasses agent and upstream guardrails).

### ⚡ Upstream Prompt Caching (`provider.prompt_caching`)
Automatically makes the most of provider KV/prefix caches:
- `session_id` forwarded to OpenRouter for sticky routing (warm prefix cache)
- `cache_control` injection for Anthropic (`ttl: 5m|1h`) when the stable prefix exceeds `min_tokens` (default 1024)
- `cached_tokens` / cache-write telemetry surfaced in `/metrics` (`router_prompt_cached_tokens_total`, `router_prompt_cache_hit_ratio`)

## Configuration

Edit `config/settings.json` (hot-reloadable) to change:
- Tier → model mappings
- Classifier model and prompt
- Session TTL, escalation thresholds
- Heuristic rules

## Architecture

```mermaid
flowchart TD
    A[Backend/Frontend Engineer] --> B[AI Agent]
    B --> C[LLM-Smart-Router]

    subgraph Router [Request Pipeline]
        C --> P1["🛡️ Guardrails<br/>Injection detection (log/block)<br/>+ Secret masking on output"]
        P1 --> P2["🔒 IP Redaction<br/>Raw IPs → [ipaddress-NN]<br/>re-hydrated on response"]
        P2 --> D["Classifier LLM<br/>gemini-3.1-flash-lite<br/>Rates task: L1–L5"]
    end

    D -->|L1| E[Gemini 2.5 Flash<br/>OpenRouter]
    D -->|L2| F[DeepSeek V4 Flash<br/>OpenRouter]
    D -->|L3–L4| F2[GLM 5.3<br/>OpenRouter]
    D -->|L5| H[Opus 5<br/>Claude API]

    E --> C
    F --> C
    F2 --> C
    H --> C

    C -->|"🔒 Secrets masked<br/>🔒 IPs re-hydrated"| B
    B --> A
```

Turn 2+ skips the classifier: the session pin routes straight to the tier model, with `on_config_change: keep_level` re-resolving the model after settings changes.


| Component | Technology |
|-----------|-----------|
| API | FastAPI + Uvicorn |
| HTTP Client | httpx (async, HTTP/2) |
| Validation | Pydantic v2 |
| Session Store | In-memory TTL+LRU / Redis |
| Metrics | prometheus-client |
| Logging | structlog (JSON) |
| Container | Multi-stage Docker, python:3.12-slim |

## Endpoints

- `POST /v1/chat/completions` — OpenAI-compatible chat
- `GET /v1/models` — List virtual router models
- `GET /healthz` / `GET /readyz` — Health checks
- `GET /metrics` — Prometheus metrics
- `GET /admin/stats` — Rolling counters
- `GET /admin/sessions` — Live session pins
- `GET /admin/settings` — Active config (incl. live guardrail mode)
- `POST /admin/settings/reload` — Hot-reload config

### Guardrail metrics
- `router_guardrail_findings_total{rule_id,severity,direction}` — injection/secret findings
- `router_guardrail_blocks_total{rule_id,severity}` — requests blocked in block mode
- `router_guardrail_secret_masks_total{rule_id}` — secrets masked in output
- `router_privacy_redactions_total` — requests passing through IP redaction
- `router_prompt_cached_tokens_total` / `router_prompt_cache_hit_ratio` — KV-cache usage

See the [full specification](./llm-smart-router-spec.md) for complete details.

## License

MIT
