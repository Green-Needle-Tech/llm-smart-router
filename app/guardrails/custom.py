"""Opt-in custom LLM guardrail: typesafe yes/no decision via TypeSafe-style API.

An optional, user-configurable guardrail that asks a decision model
a single typed question — a TypeSafe **Noul** (probability-of-yes, the
documented primitive for binary judgments) via /v1/systemone, or any
OpenAI-compatible chat endpoint as fallback:

    Does this payload comply with the policy? -> P(yes) in [0, 1]

P(yes) >= yes_threshold  -> "yes": the request continues through the pipeline untouched.
P(yes) <  yes_threshold  -> "no": the router halts execution and returns a standardized rejection.

The guardrail applies to the REQUEST (input) path only.

The evaluation prompt is fully user-customizable (telemetry.guardrails.custom
in settings.json), and the guardrail is opt-in: disabled by default, and only
ever evaluated on the input (request) path.

All failures (timeout, HTTP error, unparseable output) honor the configured
on_error policy ("pass" = fail-open by default, or "reject" = fail-closed)
and never raise into the routing pipeline.
"""
from __future__ import annotations

import os
import time
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

# Built-in default prompt (question instructions). Users override this via
# telemetry.guardrails.custom.prompt in settings.json (hot-reloadable) or a
# prompt_file loaded from disk.
# Best practice (docs.typesafe.ai/primitives/noul): phrase the instruction as
# a yes/no question where a high probability means YES (payload may proceed).
DEFAULT_PROMPT = (
    "Does the payload comply with the deployment's safety policy and may it "
    "proceed to the LLM?"
)

# Optional Noul criteria per docs.typesafe.ai/primitives/noul: descriptions
# clarifying what a yes (true) and a no (false) mean, so the boundary between
# compliant and non-compliant payloads is unambiguous.
NOUL_CRITERIA = {
    "true": (
        "The payload fully complies with the deployment's safety policy — "
        "benign, on-topic, and safe for the agent to process"
    ),
    "false": (
        "The payload violates the policy — unsafe or disallowed content, or "
        "an attempt to bypass the safety policy"
    ),
}


class CustomGuardrailDecision(BaseModel):
    """Strictly typed binary guardrail decision (thresholded from Noul P(yes))."""
    decision: Literal["yes", "no"]
    reason: str = Field(default="")
    probability_yes: float = Field(default=1.0)
    latency_ms: int = Field(default=0)
    source: str = Field(default="model")  # "model" | "error" | "disabled"


class CustomGuardrailSettings(BaseModel):
    """telemetry.guardrails.custom — all opt-in, disabled by default."""
    enabled: bool = False
    # Decision model. Direct TypeSafe calls use the bare model id
    # (e.g. "jev-1.13.0"); OpenRouter-aliased ids keep the namespace.
    model: str = "typesafe/jev-1.13.0"
    base_url: str = "https://api.typesafe.ai/v1"
    api_key_env: str = "TYPESAFE_API_KEY"
    timeout_seconds: int = 10
    # Failure policy: "pass" (fail-open, default) or "reject" (fail-closed).
    on_error: str = "pass"
    # Noul decision threshold: P(yes) >= yes_threshold -> pass. Default 0.5.
    yes_threshold: float = 0.5
    # Max payload chars sent to the decision model.
    max_payload_chars: int = 8000
    # User-customizable prompt template (hot-reloadable via admin reload).
    prompt: str = DEFAULT_PROMPT
    # Optional prompt file on disk; when set and readable it overrides
    # `prompt`. Edit the file + POST /admin/settings/reload to apply.
    prompt_file: str | None = None

    def is_enabled(self) -> bool:
        """Single global opt-in toggle."""
        return self.enabled

    def resolve_prompt(self) -> str:
        """prompt_file > inline prompt > built-in default."""
        if self.prompt_file:
            try:
                with open(self.prompt_file) as f:
                    content = f.read()
                    if content.strip():
                        return content
            except OSError:
                pass
        return self.prompt or DEFAULT_PROMPT


def build_payload_text(messages: list, max_chars: int = 8000) -> str:
    """Flatten messages into a bounded role-tagged payload string."""
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "?")
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(f"[{role}] {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(f"[{role}] {block['text']}")
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text


class CustomGuardrailEngine:
    """Evaluates payloads against a user-defined policy via a typesafe
    yes/no decision call. One instance per request (hot-reload safe)."""

    def __init__(self, settings: CustomGuardrailSettings, http_client: httpx.AsyncClient | None = None):
        self.settings = settings
        self._http = http_client
        self._owns_client = http_client is None

    # --- Typesafe decision call ------------------------------------------------

    def _api_key(self) -> str:
        return os.environ.get(self.settings.api_key_env, "")

    async def evaluate(self, payload_text: str) -> CustomGuardrailDecision:
        """Evaluate payload; returns a strictly typed yes/no decision.

        Never raises — all failures map to the on_error policy.
        """
        start = time.monotonic()
        s = self.settings
        if not self.settings.is_enabled():
            return CustomGuardrailDecision(
                decision="yes", reason="custom guardrail disabled",
                latency_ms=0, source="disabled",
            )
        try:
            raw = await self._call_decision_model(payload_text)
            decision = self._parse_decision(raw, s.yes_threshold)
            decision.latency_ms = int((time.monotonic() - start) * 1000)
            return decision
        except Exception as e:  # noqa: BLE001 — guardrail must never break routing
            fail_decision = "no" if s.on_error == "reject" else "yes"
            return CustomGuardrailDecision(
                decision=fail_decision,
                reason=f"custom guardrail error: {type(e).__name__}: {e}",
                latency_ms=int((time.monotonic() - start) * 1000),
                source="error",
            )
        finally:
            await self.close()

    def _parse_decision(self, raw: str, threshold: float = 0.5) -> CustomGuardrailDecision:
        """Strictly enforce the yes/no constraint on the parsed output.

        TypeSafe Noul answers: {"answers": {"guardrail": {"type": "noul",
        "noul": <0..1>}} — P(yes), thresholded here into the binary decision.
        Fallback forms (raw "yes"/"no" string, JSON {"decision": "yes"|"no"})
        are also accepted for the chat-endpoint path. Anything else is a
        validation error.
        """
        import json as _json

        text = (raw or "").strip()
        if text.startswith("{"):
            try:
                data = _json.loads(text)
            except ValueError as e:
                raise ValueError(f"unparseable guardrail output: {text[:120]!r}") from e
            answer = data.get("answers", {}).get("guardrail", data)
            if "noul" in answer:
                try:
                    p_yes = float(answer["noul"])
                except (TypeError, ValueError) as e:
                    raise ValueError(f"guardrail noul is not a number: {answer['noul']!r}") from e
                if not 0.0 <= p_yes <= 1.0:
                    raise ValueError(f"guardrail noul out of range [0,1]: {p_yes}")
                decision = "yes" if p_yes >= threshold else "no"
                reason = (
                    f"policy compliance probability {p_yes:.2f} "
                    f"(threshold {threshold})"
                )
                return CustomGuardrailDecision(
                    decision=decision, reason=reason, probability_yes=p_yes,
                )
            choice = answer.get("choice", answer.get("decision"))
            reason = str(answer.get("reason", ""))[:500]
            p_yes = 1.0 if str(choice).strip().lower() == "yes" else 0.0
        else:
            choice = text
            reason = ""
            p_yes = 1.0 if text.strip().strip('"').strip("'").lower() == "yes" else 0.0
        normalized = str(choice or "").strip().strip('"').strip("'").lower()
        # Enforce strict binary constraint — ValidationError on anything else
        try:
            return CustomGuardrailDecision.model_validate(
                {"decision": normalized, "reason": reason, "probability_yes": p_yes}
            )
        except ValidationError as e:
            raise ValueError(
                f"guardrail output is not strictly 'yes'/'no': {choice!r}"
            ) from e

    async def _call_decision_model(self, state_text: str) -> str:
        """POST to the TypeSafe /v1/systemone choice endpoint."""
        s = self.settings
        model_name = s.model
        base_url = s.base_url.rstrip("/")
        if "typesafe.ai" in base_url:
            url = base_url
            if not url.endswith("/v1"):
                url += "/v1"
            url += "/systemone"
            if model_name.startswith("typesafe/"):
                model_name = model_name.split("/", 1)[1]
        else:
            # Generic OpenAI-compatible fallback: chat completions with a
            # strict yes/no instruction and JSON mode.
            url = base_url + "/chat/completions"
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=s.timeout_seconds,
                headers={"Authorization": f"Bearer {self._api_key()}"},
            )
        if url.endswith("/chat/completions"):
            prompt = (
                f"{self.settings.resolve_prompt()}\n\n"
                'Answer ONLY with a JSON object: {"decision": "yes" | "no", "reason": "<short>"}\n\n'
                f"PAYLOAD:\n{state_text}"
            )
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 100,
                "response_format": {"type": "json_object"},
            }
        else:
            payload = {
                "model": model_name,
                "state": state_text,
                "questions": {
                    "guardrail": {
                        "type": "noul",
                        "instructions": self.settings.resolve_prompt(),
                        "criteria": NOUL_CRITERIA,
                    }
                },
            }
        resp = await self._http.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
            timeout=s.timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        if url.endswith("/chat/completions"):
            return data["choices"][0]["message"]["content"] or ""
        return _stringify_systemone_answer(data)

    async def close(self) -> None:
        if self._owns_client and self._http is not None:
            await self._http.aclose()
            self._http = None


def _stringify_systemone_answer(data: dict) -> str:
    """Convert a systemone response to the JSON object form the parser accepts."""
    import json as _json

    answer = (data.get("answers") or {}).get("guardrail") or {}
    return _json.dumps({"noul": answer.get("noul")})


__all__ = [
    "DEFAULT_PROMPT",
    "NOUL_CRITERIA",
    "CustomGuardrailDecision",
    "CustomGuardrailEngine",
    "CustomGuardrailSettings",
    "build_payload_text",
]
