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
    """telemetry.guardrails.custom — all opt-in, disabled by default.

    The guardrail question is fully defined here: type, id, instructions
    (prompt) and criteria. No code changes are needed to customize it.
    """
    enabled: bool = False
    # Decision model. Direct TypeSafe calls use the bare model id
    # (e.g. "jev-1.13.0"); OpenRouter-aliased ids keep the namespace.
    model: str = "typesafe/jev-1.13.0"
    base_url: str = "https://api.typesafe.ai/v1"
    api_key_env: str = "TYPESAFE_API_KEY"
    timeout_seconds: int = 10
    # Failure policy: "pass" (fail-open, default) or "reject" (fail-closed).
    on_error: str = "pass"
    # Decision threshold: P(yes) >= yes_threshold -> pass. Default 0.5.
    # For question_type "score", P(yes) = score normalized by the level count.
    yes_threshold: float = 0.5
    # Max payload chars sent to the decision model.
    max_payload_chars: int = 8000
    # Which messages feed the decision model:
    #   "all"           — every message (system + history + user), head-truncated
    #                     to max_payload_chars with the last user message preserved
    #                     (build_payload_text).
    #   "last_user"     — ONLY the last user message. Good for topic-scope
    #                     guardrails: agent system prompts are large and their
    #                     partial/truncated text destabilizes the classifier
    #                     (measured: user-only 0.84-0.86 vs ~0.5 with a partial
    #                     system prompt prepended, same on-topic question).
    #                     Weakness: short context-free follow-ups ("Yes",
    #                     "Generate html") carry no topic signal and get
    #                     wrongly rejected.
    #   "conversation"  — all NON-system messages (user + assistant history +
    #                     last user), head-truncated to max_payload_chars with
    #                     the last user message preserved. Recommended for
    #                     topic-scope guardrails on persistent chat sessions:
    #                     keeps the conversation topic visible for follow-ups
    #                     while still excluding the large agent system prompt.
    payload_scope: Literal["all", "last_user", "conversation"] = "all"
    # Question id in the systemone request/response (your routing key).
    question_id: str = "guardrail"
    # Question type: "noul" (P(yes) in [0,1]), "choice" (P of the "yes"
    # option), or "score" (score normalized across levels).
    question_type: Literal["noul", "choice", "score"] = "noul"
    # User-customizable prompt template (hot-reloadable via admin reload).
    prompt: str = DEFAULT_PROMPT
    # Optional criteria override, shape depends on question_type:
    # noul -> {"true": ..., "false": ...}; choice -> {option: description};
    # score -> [level descriptions, ordered, >=2]. Must stay aligned with
    # `prompt` (docs.typesafe.ai/model-jaggedness/jev-1.13: contradictory
    # instructions and criteria degrade accuracy). Defaults to the built-in
    # Noul criteria for noul questions, none otherwise.
    criteria: dict[str, str] | list[str] | None = None
    # Optional custom rejection message shown to the client. Supports a
    # "{reason}" placeholder. Empty -> built-in default message.
    rejection_message: str = ""
    # How a rejection is delivered to the client:
    #   "reply"  — HTTP 200 completion whose content is the rejection message
    #              plus the router version postfix (single client-side message).
    #   "error"  — legacy HTTP 400 guardrail_violation envelope.
    rejection_delivery: Literal["reply", "error"] = "reply"
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

    def resolve_criteria(self) -> dict[str, str] | list[str] | None:
        """criteria override > built-in NOUL_CRITERIA (noul only) > None."""
        if self.criteria is not None:
            return self.criteria
        if self.question_type == "noul":
            return NOUL_CRITERIA
        return None

    def build_question(self) -> dict:
        """The full systemone question object, built entirely from settings."""
        question: dict = {
            "type": self.question_type,
            "instructions": self.resolve_prompt(),
        }
        criteria = self.resolve_criteria()
        if criteria is not None:
            question["criteria"] = criteria
        return question


def build_payload_text(messages: list, max_chars: int = 8000) -> str:
    """Flatten messages into a bounded role-tagged payload string.

    The LAST user message is always preserved: when the flattened text
    exceeds max_chars, earlier messages are truncated (keeping their head)
    so the final user message — the input the guardrail must actually
    judge — survives inside the budget. Without this, a large system
    prompt (e.g. an agent's 27k-char identity prompt) consumes the whole
    budget and the guardrail never sees the user's question.
    """
    parts: list[str] = []
    last_user_idx: int | None = None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "?")
        content = msg.get("content")
        if role == "user":
            last_user_idx = len(parts)
        if isinstance(content, str):
            parts.append(f"[{role}] {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(f"[{role}] {block['text']}")
    # Reserve room for the last user message (plus joining newline).
    if last_user_idx is not None:
        tail = parts[last_user_idx]
        if len(tail) > max_chars:
            # The user message alone blows the budget: tail-truncate it so
            # the end (usually where the actual question sits) is kept.
            marker = "...[truncated]\n"
            keep = max(1, max_chars - len(marker) - 1)
            tail = marker + tail[-keep:]
            parts[last_user_idx] = tail
        head_budget = max(0, max_chars - len(tail) - 1)
        head = "\n".join(parts[:last_user_idx])
        if len(head) > head_budget:
            head = (head[:head_budget] + "\n...[truncated]") if head_budget > 0 else ""
        text = (head + "\n" + tail) if head else tail
    else:
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
            decision = self._parse_decision(raw, s.yes_threshold, s.question_type)
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

    def _parse_decision(
        self, raw: str, threshold: float = 0.5, question_type: str = "noul"
    ) -> CustomGuardrailDecision:
        """Strictly enforce the yes/no constraint on the parsed output.

        TypeSafe systemone answers (docs.typesafe.ai/api):
        - noul:   {"type": "noul", "noul": <0..1>}              -> P(yes) = noul
        - choice: {"choice": <opt>, "probabilities": {...}}     -> P(yes) =
                   probabilities["yes"] when a "yes" option exists, else
                   1.0/0.0 by the selected option
        - score:  {"score": <num>, "legend": {...}}            -> P(yes) =
                   score / (len(legend) - 1), normalized across levels
        Thresholded here into the binary decision. Fallback forms (raw
        "yes"/"no" string, JSON {"decision": "yes"|"no"}) are also accepted
        for the chat-endpoint path. Anything else is a validation error.
        """
        import json as _json

        def _threshold(p_yes: float, label: str) -> CustomGuardrailDecision:
            if not 0.0 <= p_yes <= 1.0:
                raise ValueError(f"guardrail {label} out of range [0,1]: {p_yes}")
            decision = "yes" if p_yes >= threshold else "no"
            reason = f"policy compliance probability {p_yes:.2f} (threshold {threshold})"
            return CustomGuardrailDecision(
                decision=decision, reason=reason, probability_yes=p_yes,
            )

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
                return _threshold(p_yes, "noul")
            if "score" in answer and "legend" in answer:
                try:
                    score = float(answer["score"])
                    n_levels = len(answer["legend"])
                except (TypeError, ValueError) as e:
                    raise ValueError(f"guardrail score/legend invalid: {answer!r}") from e
                if n_levels < 2:
                    raise ValueError(f"guardrail score legend has <2 levels: {answer['legend']!r}")
                return _threshold(score / (n_levels - 1), "score")
            if "choice" in answer:
                choice = answer["choice"]
                reason = str(answer.get("reason", ""))[:500]
                probabilities = answer.get("probabilities") or {}
                if "yes" in probabilities:
                    try:
                        return _threshold(float(probabilities["yes"]), "choice P(yes)")
                    except (TypeError, ValueError) as e:
                        raise ValueError(
                            f"guardrail choice probability invalid: {probabilities['yes']!r}"
                        ) from e
                p_yes = 1.0 if str(choice).strip().lower() == "yes" else 0.0
            else:
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
                    self.settings.question_id: self.settings.build_question(),
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
        return _stringify_systemone_answer(data, self.settings.question_id)

    async def close(self) -> None:
        if self._owns_client and self._http is not None:
            await self._http.aclose()
            self._http = None


def _stringify_systemone_answer(data: dict, question_id: str = "guardrail") -> str:
    """Convert a systemone answer to the JSON object form the parser accepts."""
    import json as _json

    answer = (data.get("answers") or {}).get(question_id) or {}
    # Drop the "type" key so the parser dispatches on value fields, not type.
    answer = {k: v for k, v in answer.items() if k != "type"}
    return _json.dumps(answer)


__all__ = [
    "DEFAULT_PROMPT",
    "NOUL_CRITERIA",
    "CustomGuardrailDecision",
    "CustomGuardrailEngine",
    "CustomGuardrailSettings",
    "build_payload_text",
]
