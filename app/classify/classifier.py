"""Classifier service: calls the cheap model to classify the opening prompt."""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional

import httpx

from app.schemas.router import Level, ClassificationResult, ClassificationSource
from app.schemas.openai import ChatMessage, ChatCompletionRequest
from .digest import DigestBuilder
from .parser import parse_classifier_output
from .heuristics import evaluate_heuristics
from .injection_guard import check_injection


class ClassifierService:
    """Classifies the first prompt of a session using a cheap LLM."""

    def __init__(
        self,
        config,
        openrouter_api_key: str,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.config = config
        self.api_key = openrouter_api_key

        # Per-classifier provider override (optional)
        self._classifier_api_key: str | None = None
        cls_cfg = config.classification
        if hasattr(cls_cfg, "api_key_env") and cls_cfg.api_key_env:
            self._classifier_api_key = os.environ.get(cls_cfg.api_key_env, "")
        self._http = http_client
        self._owns_client = http_client is None

        cls_cfg = config.classification
        digest_cfg = cls_cfg.digest

        self.digest_builder = DigestBuilder(
            system_chars=digest_cfg.system_chars,
            tail_chars=digest_cfg.tail_chars,
            include_tool_names=digest_cfg.include_tool_names,
            include_context_summary=digest_cfg.include_context_summary,
            strip_scaffolding=digest_cfg.strip_scaffolding,
            learn_common_prefix=digest_cfg.learn_common_prefix,
            prefix_samples=digest_cfg.prefix_samples,
            min_prefix_chars=digest_cfg.min_prefix_chars,
            strip_sections_enabled=digest_cfg.strip_sections_enabled,
            strip_sections=digest_cfg.strip_sections,
            keep_sections=digest_cfg.keep_sections,
            delimit_untrusted=digest_cfg.delimit_untrusted,
            injection_guard=digest_cfg.injection_guard,
        )

        self._prompt_template: str | None = None
        self._prompt_file = cls_cfg.prompt_file

    @property
    def prompt_template(self) -> str:
        """Load the classifier prompt template (cached)."""
        if self._prompt_template is None:
            try:
                with open(self._prompt_file, "r") as f:
                    self._prompt_template = f.read()
            except FileNotFoundError:
                # Fallback inline prompt
                self._prompt_template = (
                    'You are a task-complexity classifier. Output ONLY JSON: '
                    '{"level":"L1|L2|L3|L4|UNKNOWN","confidence":0.0-1.0,"reason":"<12 words>"}\n\n'
                    'REQUEST:\n{{PROMPT_DIGEST}}'
                )
        return self._prompt_template

    @property
    def _classifier_base_url(self) -> str:
        """Resolve classifier base_url (per-classifier config > global provider)."""
        cls_cfg = self.config.classification
        if hasattr(cls_cfg, "base_url") and cls_cfg.base_url:
            return cls_cfg.base_url.rstrip("/")
        return self.config.provider.base_url.rstrip("/")

    @property
    def _classifier_auth_header(self) -> str:
        """Resolve the auth header for the classifier provider."""
        effective_key = self._classifier_api_key or self.api_key
        return f"Bearer {effective_key}"

    async def classify(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        task_text: Optional[str] = None,
        ignore_system: bool = False,
        bypass_cache: bool = False,
    ) -> tuple[ClassificationResult, dict]:
        """Classify the opening prompt of a session.

        Returns (result, digest_info).
        """
        start = time.monotonic()

        # Build digest
        digest_info = self.digest_builder.build(
            messages=messages,
            tools=tools,
            response_format=response_format,
            task_text=task_text,
            ignore_system=ignore_system,
        )

        # Injection check
        if digest_info.get("injection_suspected"):
            default_level = Level.from_str(self.config.classification.default_level)
            result = ClassificationResult(
                level=default_level,
                confidence=0.0,
                reason="injection suspected, fallback to default",
                source=ClassificationSource.DEFAULT,
                latency_ms=int((time.monotonic() - start) * 1000),
            )
            return result, digest_info

        # Heuristic fast path
        if self.config.heuristics.enabled:
            heur_result = evaluate_heuristics(
                digest=digest_info["digest"],
                has_code=digest_info["has_code"],
                code_fences=digest_info["code_fences"],
                json_mode=response_format is not None and response_format.get("type") == "json_object",
                prompt_tokens=digest_info["task_tokens"] if self.config.heuristics.measure == "task_payload" else digest_info["total_tokens"],
                task_chars=digest_info.get("task_chars"),
                huge_context_tokens=self.config.heuristics.huge_context_tokens,
                rules=[r.model_dump() if hasattr(r, "model_dump") else r for r in self.config.heuristics.rules] if self.config.heuristics.rules else None,
                measure=self.config.heuristics.measure,
            )

            if heur_result is not None:
                level, stop, rule_name = heur_result
                if stop:
                    result = ClassificationResult(
                        level=level,
                        confidence=1.0,
                        reason=f"heuristic: {rule_name}",
                        source=ClassificationSource.HEURISTIC,
                        latency_ms=int((time.monotonic() - start) * 1000),
                    )
                    return result, digest_info

        # Call classifier model
        if not self.config.classification.enabled:
            default_level = Level.from_str(self.config.classification.default_level)
            result = ClassificationResult(
                level=default_level,
                confidence=0.0,
                reason="classification disabled",
                source=ClassificationSource.DEFAULT,
                latency_ms=int((time.monotonic() - start) * 1000),
            )
            return result, digest_info

        try:
            raw_output = await self._call_classifier_model(digest_info["digest"])
            latency_ms = int((time.monotonic() - start) * 1000)
            result = parse_classifier_output(
                raw_output,
                source=ClassificationSource.MODEL,
                classifier_model=self.config.classification.model,
                rubric_version=self.config.classification.rubric_version,
                latency_ms=latency_ms,
            )

            # Handle UNKNOWN or parse failure
            if result.level is None:
                if result.reason == "parse failure":
                    # Genuine parse failure: be conservative, use default_level.
                    # Return directly — default_level is already the deliberate
                    # fallback choice, so the low-confidence escalation policy
                    # must not bump it another tier (that silently ignored the
                    # configured default and always produced L4).
                    result.level = Level.from_str(self.config.classification.default_level)
                    result.source = ClassificationSource.DEFAULT
                    return result, digest_info
                # Classifier deliberately returned UNKNOWN (greeting, bare
                # acknowledgement, too vague). There is no task content, so
                # the cheapest tier is correct — not the conservative default.
                # UNKNOWN is a definite judgement, not low confidence, so skip
                # the escalation policy below.
                result.level = Level.from_str(self.config.classification.unknown_level)
                result.source = ClassificationSource.MODEL
                return result, digest_info

            # Apply confidence handling
            result = self._apply_confidence_policy(result)

            return result, digest_info

        except (httpx.TimeoutException, asyncio.TimeoutError):
            latency_ms = int((time.monotonic() - start) * 1000)
            default_level = Level.from_str(self.config.classification.default_level)
            result = ClassificationResult(
                level=default_level,
                confidence=0.0,
                reason="classifier timeout",
                source=ClassificationSource.DEFAULT,
                latency_ms=latency_ms,
            )
            return result, digest_info

        except Exception:
            latency_ms = int((time.monotonic() - start) * 1000)
            default_level = Level.from_str(self.config.classification.default_level)
            result = ClassificationResult(
                level=default_level,
                confidence=0.0,
                reason="classifier error",
                source=ClassificationSource.DEFAULT,
                latency_ms=latency_ms,
            )
            return result, digest_info

    def _apply_confidence_policy(self, result: ClassificationResult) -> ClassificationResult:
        """Apply low-confidence action."""
        if result.confidence < self.config.classification.min_confidence:
            action = self.config.classification.low_confidence_action
            if action == "escalate" and result.level is not None:
                new_level = Level.from_numeric(result.level.numeric + 1)
                result.reason = f"{result.reason}; low-confidence escalate to {new_level.value}"
                result.level = new_level
            elif action == "default":
                result.level = Level.from_str(self.config.classification.default_level)
                result.reason = f"{result.reason}; low-confidence default"
        return result

    async def _call_classifier_model(self, digest: str) -> str:
        """Call the classifier model via its configured provider."""
        prompt = self.prompt_template.replace("{{PROMPT_DIGEST}}", digest)

        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=self.config.classification.timeout_seconds,
                headers={"Authorization": self._classifier_auth_header},
            )

        payload = {
            "model": self.config.classification.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.classification.temperature,
            "max_tokens": self.config.classification.max_tokens,
        }

        # Use JSON mode if supported
        try:
            payload["response_format"] = {"type": "json_object"}
        except Exception:
            pass

        base_url = self._classifier_base_url
        resp = await self._http.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers={
                "Authorization": self._classifier_auth_header,
                "Content-Type": "application/json",
                **self.config.provider.headers,
            },
            timeout=self.config.classification.timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def close(self) -> None:
        if self._owns_client and self._http is not None:
            await self._http.aclose()
