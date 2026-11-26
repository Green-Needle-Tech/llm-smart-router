"""Classifier service: calls the cheap model to classify the opening prompt."""
from __future__ import annotations

import contextlib
import json
import os
import time

import httpx

from app.schemas.openai import ChatMessage
from app.schemas.router import ClassificationResult, ClassificationSource, Level
from app.telemetry import langfuse_tracing
from app.telemetry.metrics import (
    router_classification_duration_seconds,
    router_classifier_calls_total,
    router_classifier_failures_total,
    router_injection_suspected_total,
)

from .digest import DigestBuilder
from .heuristics import evaluate_heuristics
from .parser import parse_classifier_output


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
        self._http = http_client
        self._owns_client = http_client is None

        self._classifier_api_key: str | None = None
        self._prompt_template: str | None = None
        self._prompt_file: str = ""
        self._sync_from_config(config, init=True)

    def _sync_from_config(self, config, *, init: bool = False) -> None:
        """Sync runtime fields from config (used at init and on hot-reload)."""
        self.config = config
        cls_cfg = config.classification
        if hasattr(cls_cfg, "api_key_env") and cls_cfg.api_key_env:
            self._classifier_api_key = os.environ.get(cls_cfg.api_key_env, "")
        else:
            self._classifier_api_key = None

        new_prompt_file = getattr(cls_cfg, "prompt_file", "")
        if init or new_prompt_file != self._prompt_file:
            self._prompt_file = new_prompt_file
            self._prompt_template = None

        digest_cfg = cls_cfg.digest
        if init:
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
        else:
            db = self.digest_builder
            db.system_chars = digest_cfg.system_chars
            db.tail_chars = digest_cfg.tail_chars
            db.include_tool_names = digest_cfg.include_tool_names
            db.include_context_summary = digest_cfg.include_context_summary
            db.strip_scaffolding = digest_cfg.strip_scaffolding
            db.learn_common_prefix = digest_cfg.learn_common_prefix
            db.strip_sections_enabled = digest_cfg.strip_sections_enabled
            db.strip_sections = digest_cfg.strip_sections or []
            db.keep_sections = digest_cfg.keep_sections or []
            db.delimit_untrusted = digest_cfg.delimit_untrusted
            db.injection_guard = digest_cfg.injection_guard

    def update_config(self, config) -> None:
        """Apply hot-reloaded configuration."""
        self._sync_from_config(config, init=False)

    @property
    def prompt_template(self) -> str:
        """Load the classifier prompt template (cached)."""
        if self._prompt_template is None:
            try:
                with open(self._prompt_file, encoding="utf-8") as f:
                    self._prompt_template = f.read()
            except OSError:
                # Fallback inline prompt
                self._prompt_template = (
                    'You are a task-complexity classifier. Output ONLY JSON: '
                    '{"level":"L1|L2|L3|L4|L5|UNKNOWN","confidence":0.0-1.0,"reason":"<12 words>"}\n\n'
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
        task_text: str | None = None,
        ignore_system: bool = False,
        bypass_cache: bool = False,
    ) -> tuple[ClassificationResult, dict]:
        """Classify the opening prompt of a session.

        Returns (result, digest_info).
        """
        with langfuse_tracing.span_cm("classify-request") as span:
            result, digest_info = await self._classify_inner(
                messages, tools, response_format,
                task_text=task_text, ignore_system=ignore_system,
                bypass_cache=bypass_cache,
            )
            langfuse_tracing._safe_update(
                span,
                output={
                    "level": result.level.value if result.level else "UNKNOWN",
                    "confidence": result.confidence,
                    "source": result.source.value,
                    "reason": result.reason,
                    "latency_ms": result.latency_ms,
                },
            )
            return result, digest_info

    async def _classify_inner(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        task_text: str | None = None,
        ignore_system: bool = False,
        bypass_cache: bool = False,
    ) -> tuple[ClassificationResult, dict]:
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
            router_injection_suspected_total.inc()
            default_level = Level.from_str(self.config.classification.default_level)
            elapsed = time.monotonic() - start
            router_classification_duration_seconds.labels(source="default").observe(elapsed)
            result = ClassificationResult(
                level=default_level,
                confidence=0.0,
                reason="injection suspected, fallback to default",
                source=ClassificationSource.DEFAULT,
                latency_ms=int(elapsed * 1000),
            )
            return result, digest_info

        # Heuristic fast path (stop rules return immediately; floor rules clamp below)
        heur_stop, floor_level = self._try_heuristics(digest_info, response_format, start)
        if heur_stop is not None:
            router_classification_duration_seconds.labels(source="heuristic").observe(
                time.monotonic() - start
            )
            return heur_stop, digest_info

        # Call classifier model
        if not self.config.classification.enabled:
            res = self._default_result("classification disabled", start)
            if floor_level is not None and res.level is not None and res.level < floor_level:
                res.level = floor_level
            return res, digest_info

        try:
            raw_output = await self._call_classifier_model(digest_info["digest"])
            elapsed = time.monotonic() - start
            latency_ms = int(elapsed * 1000)
            result = parse_classifier_output(
                raw_output,
                source=ClassificationSource.MODEL,
                classifier_model=self.config.classification.model,
                rubric_version=self.config.classification.rubric_version,
                latency_ms=latency_ms,
            )
            final = self._handle_classifier_result(result, digest_info, start)
            if floor_level is not None and final.level is not None and final.level < floor_level:
                final.reason = f"{final.reason}; heuristic floor {floor_level.value}".lstrip("; ")
                final.level = floor_level
            router_classifier_calls_total.labels(
                result=final.level.value if final.level else "UNKNOWN"
            ).inc()
            router_classification_duration_seconds.labels(
                source=final.source.value
            ).observe(elapsed)
            return final, digest_info
        except (TimeoutError, httpx.TimeoutException):
            router_classifier_failures_total.labels(reason="timeout").inc()
            router_classifier_calls_total.labels(result="timeout").inc()
            res = self._default_result("classifier timeout", start)
            if floor_level is not None and res.level is not None and res.level < floor_level:
                res.level = floor_level
            return res, digest_info
        except Exception:
            router_classifier_failures_total.labels(reason="error").inc()
            router_classifier_calls_total.labels(result="error").inc()
            res = self._default_result("classifier error", start)
            if floor_level is not None and res.level is not None and res.level < floor_level:
                res.level = floor_level
            return res, digest_info

    def _try_heuristics(self, digest_info, response_format, start) -> tuple[ClassificationResult | None, Level | None]:
        """Try heuristic fast path. Returns (stop_result, floor_level)."""
        if not self.config.heuristics.enabled:
            return None, None
        heur_result = evaluate_heuristics(
            digest=digest_info["digest"],
            has_code=digest_info["has_code"],
            code_fences=digest_info["code_fences"],
            json_mode=response_format is not None and response_format.get("type") == "json_object",
            prompt_tokens=digest_info["task_tokens"] if self.config.heuristics.measure == "task_payload" else digest_info["total_tokens"],
            task_chars=digest_info.get("task_chars"),
            huge_context_tokens=self.config.heuristics.huge_context_tokens,
            rules=self.config.heuristics.rules or None,
            measure=self.config.heuristics.measure,
        )
        if heur_result is not None:
            level, stop, rule_name = heur_result
            if stop:
                return ClassificationResult(
                    level=level, confidence=1.0, reason=f"heuristic: {rule_name}",
                    source=ClassificationSource.HEURISTIC,
                    latency_ms=int((time.monotonic() - start) * 1000),
                ), None
            return None, level
        return None, None

    def _default_result(self, reason, start):
        """Build a default-level classification result."""
        elapsed = time.monotonic() - start
        router_classification_duration_seconds.labels(source="default").observe(elapsed)
        return ClassificationResult(
            level=Level.from_str(self.config.classification.default_level),
            confidence=0.0, reason=reason,
            source=ClassificationSource.DEFAULT,
            latency_ms=int(elapsed * 1000),
        )

    def _handle_classifier_result(self, result, digest_info, start):
        """Handle UNKNOWN/parse-failure and apply confidence policy."""
        if result.level is not None:
            return self._apply_confidence_policy(result)
        if result.reason == "parse failure":
            router_classifier_failures_total.labels(reason="parse_failure").inc()
            result.level = Level.from_str(self.config.classification.default_level)
            result.source = ClassificationSource.DEFAULT
            return result
        result.level = Level.from_str(self.config.classification.unknown_level)
        result.source = ClassificationSource.MODEL
        return result

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
        mode = getattr(self.config.classification, "provider_mode", "chat") or "chat"
        if mode == "decisions":
            return await self._call_classifier_decisions(digest)
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
        with contextlib.suppress(Exception):
            payload["response_format"] = {"type": "json_object"}

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

    async def _call_classifier_decisions(self, digest: str) -> str:
        """Call a decision model (OpenRouter /api/alpha/decisions or TypeSafe /v1/systemone).

        Returns a JSON string of the form
        {"level": "L1".."L5", "confidence": 0.0-1.0, "reason": "..."}
        so the existing parse_classifier_output path handles it unchanged.
        """
        cls_cfg = self.config.classification
        # Criteria strings follow TypeSafe's Choice best practices (docs.typesafe.ai
        # /primitives/choice): option descriptions must separate confusable options
        # with WHAT / NOT FOR / EXAMPLES. OpenRouter's decisions endpoint accepts
        # strings only (its Zod schema rejects the structured object form).
        criteria = getattr(cls_cfg, "tier_criteria", None) or {
            "L1": (
                "WHAT: almost no reasoning - greetings, thanks, simple factual recall, "
                "counting, basic arithmetic, one-step extraction. "
                "NOT FOR: any task with multiple steps or code. "
                "EXAMPLES: hey thanks; what is 2+2; capital of France"
            ),
            "L2": (
                "WHAT: a small, bounded, conventional task a fast flash model completes "
                "reliably - short single-function code, simple rewrites, lookups with "
                "light formatting. "
                "NOT FOR: multi-file changes, debugging, or design decisions. "
                "EXAMPLES: write a python function to parse ISO dates; summarize this email in one line"
            ),
            "L3": (
                "WHAT: an intermediate multi-step task - standard coding, bug fixes with "
                "tests, data analysis, structured document drafting. "
                "NOT FOR: architecture-level design or expert-level reasoning. "
                "EXAMPLES: fix this bug and add unit tests; compare these datasets and chart differences"
            ),
            "L4": (
                "WHAT: advanced reasoning - complex code, system architecture, multi-file "
                "refactors, tradeoff analysis with implementation. "
                "NOT FOR: novel research beyond established practice. "
                "EXAMPLES: design a distributed rate limiter with Redis failover; refactor the auth module across services"
            ),
            "L5": (
                "WHAT: expert-level work - deep reasoning, novel research, large-scale "
                "system design, proof-grade analysis. "
                "NOT FOR: routine tasks a strong generalist model completes well. "
                "EXAMPLES: propose a novel formal verification approach; design a global multi-region consensus protocol"
            ),
        }
        base_url = self._classifier_base_url
        model_name = cls_cfg.model
        # Direct TypeSafe calls use the bare model id (e.g. jev-1.13.0),
        # not the OpenRouter-namespaced "typesafe/jev-1.13" form.
        if "typesafe.ai" in base_url and model_name.startswith("typesafe/"):
            model_name = model_name.split("/", 1)[1]
        payload = {
            "model": model_name,
            "state": digest,
            "questions": {
                "tier": {
                    "type": "choice",
                    "instructions": (
                        "Which complexity tier does this user request belong to? "
                        "Judge the total work the whole session will require, not just "
                        "the wording of the first message. Choose the LOWEST tier whose "
                        "model can reliably complete the entire task."
                    ),
                    "criteria": criteria,
                }
            },
        }

        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=cls_cfg.timeout_seconds,
                headers={"Authorization": self._classifier_auth_header},
            )

        # Endpoint selection by base URL:
        # - TypeSafe direct (api.typesafe.ai): POST /v1/systemone
        # - OpenRouter (default): POST /api/alpha/decisions (NOT under /v1,
        #   so a trailing "/v1" is stripped from the configured base URL).
        base_url = self._classifier_base_url
        if "typesafe.ai" in base_url:
            url = base_url.rstrip("/")
            if not url.endswith("/v1"):
                url += "/v1"
            url += "/systemone"
            # Direct TypeSafe calls use the bare model id (e.g. jev-1.13.0),
            # not the OpenRouter-namespaced "typesafe/jev-1.13" form.
            if model_name.startswith("typesafe/"):
                model_name = model_name.split("/", 1)[1]
        else:
            if base_url.endswith("/v1"):
                base_url = base_url[: -len("/v1")]
            url = f"{base_url}/alpha/decisions"
        resp = await self._http.post(
            url,
            json=payload,
            headers={
                "Authorization": self._classifier_auth_header,
                "Content-Type": "application/json",
                **self.config.provider.headers,
            },
            timeout=cls_cfg.timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data["answers"]["tier"]
        level = str(answer["choice"]).upper().strip()
        if level not in {"L1", "L2", "L3", "L4", "L5"}:
            raise ValueError(f"decision model returned invalid level: {level}")
        return json.dumps(
            {
                "level": level,
                "confidence": float(answer.get("confidence", 1.0)),
                "reason": f"decision model ({model_name})",
            }
        )

    async def close(self) -> None:
        if self._owns_client and self._http is not None:
            await self._http.aclose()
