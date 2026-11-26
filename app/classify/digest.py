"""Prompt digest builder: constructs the task payload for the classifier."""
from __future__ import annotations

import hashlib
from typing import Any

from app.schemas.openai import ChatMessage

from .injection_guard import check_injection
from .scaffolding import CommonPrefixLearner, split_scaffolding


def _extract_text(content: Any) -> str:
    """Extract text from message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content) if content else ""


def _count_code_fences(text: str) -> int:
    return text.count("```")


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def _truncate(text: str, head_chars: int, tail_chars: int) -> str:
    """Truncate to head + ellipsis + tail."""
    if len(text) <= head_chars + tail_chars:
        return text
    head = text[:head_chars]
    tail = text[-tail_chars:] if tail_chars > 0 else ""
    omitted = len(text) - head_chars - tail_chars
    return f"{head}\n…[truncated {omitted} chars]…\n{tail}"


def _get_tool_names(tools: list[dict] | None) -> list[str]:
    if not tools:
        return []
    names = []
    for tool in tools:
        if isinstance(tool, dict):
            fn = tool.get("function", tool)
            name = fn.get("name", "")
            if name:
                names.append(name)
    return names


def _count_image_attachments(messages: list[ChatMessage]) -> int:
    """Count multimodal image content blocks without scanning prose."""
    count = 0
    for m in messages:
        content = m.content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in (
                    "image_url", "image", "input_image",
                ):
                    count += 1
    return count


def _build_context_summary(messages, task_tokens: int, tool_names: list[str]) -> str:
    """Build the context summary string for the digest."""
    msg_count = len(messages)
    tool_count = len(tool_names)
    image_count = _count_image_attachments(messages)
    parts = [f"[conversation: {msg_count} messages, ~{task_tokens} task tokens"]
    if tool_count:
        parts.append(f", {tool_count} tools present")
    if image_count:
        label = "image" if image_count == 1 else "images"
        parts.append(f", attachments: {image_count} {label}")
    parts.append("]")
    return "".join(parts)


class DigestBuilder:
    """Builds the classification prompt digest from a request."""

    def __init__(
        self,
        system_chars: int = 500,
        tail_chars: int = 2000,
        include_tool_names: bool = True,
        include_context_summary: bool = True,
        strip_scaffolding: bool = True,
        learn_common_prefix: bool = True,
        prefix_samples: int = 20,
        min_prefix_chars: int = 200,
        strip_sections_enabled: bool = True,
        strip_sections: list[str] | None = None,
        keep_sections: list[str] | None = None,
        delimit_untrusted: bool = True,
        injection_guard: bool = True,
    ):
        self.system_chars = system_chars
        self.tail_chars = tail_chars
        self.include_tool_names = include_tool_names
        self.include_context_summary = include_context_summary
        self.strip_scaffolding = strip_scaffolding
        self.learn_common_prefix = learn_common_prefix
        self.strip_sections_enabled = strip_sections_enabled
        self.strip_sections = strip_sections or []
        self.keep_sections = keep_sections or []
        self.delimit_untrusted = delimit_untrusted
        self.injection_guard = injection_guard
        self._prefix_learner = CommonPrefixLearner(
            max_samples=prefix_samples, min_prefix_chars=min_prefix_chars
        ) if learn_common_prefix else None

    def build(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        task_text: str | None = None,
        ignore_system: bool = False,
    ) -> dict:
        """Build the digest and metadata.

        Returns:
            digest: str — the text fed to the classifier
            scaffolding_stripped_chars: int
            stripped_by: list[str]
            task_tokens: int — estimated tokens in the task payload
            total_tokens: int — estimated total tokens (including scaffolding)
            context_summary: str
            tool_names: list[str]
            has_code: bool
            code_fences: int
            injection_suspected: bool
        """
        # Separate system and user messages while counting total chars in one pass
        system_texts = []
        user_texts = []
        total_chars = 0
        for msg in messages:
            text = _extract_text(msg.content)
            total_chars += len(text) + 1
            if msg.role == "system":
                system_texts.append(text)
            elif msg.role == "user":
                user_texts.append(text)

        full_system = "\n".join(system_texts)
        total_tokens = max(1, max(0, total_chars - 1) // 4)

        # Scaffolding split
        split = split_scaffolding(
            system_message=full_system,
            user_messages=user_texts,
            strip_enabled=self.strip_scaffolding,
            learn_prefix=self.learn_common_prefix,
            prefix_learner=self._prefix_learner,
            strip_patterns=self.strip_sections if self.strip_sections_enabled else None,
            task_text=task_text,
            ignore_system=ignore_system,
        )

        task_system = split["task_system"]
        task_user = split["task_user"]
        scaffolding_chars = split["scaffolding_chars"]
        stripped_by = split["stripped_by"]

        # Tool names
        tool_names = _get_tool_names(tools) if self.include_tool_names else []

        # Estimate task tokens once
        task_tokens = max(1, (len(task_system) + 1 + len(task_user)) // 4)

        # Context summary
        context_summary = ""
        if self.include_context_summary:
            context_summary = _build_context_summary(messages, task_tokens, tool_names)

        # Truncate user text (60% head / 40% tail of configured tail_chars budget)
        budget = max(100, self.tail_chars)
        head_budget = (budget * 3) // 5
        tail_budget = budget - head_budget
        truncated_user = _truncate(task_user, head_chars=head_budget, tail_chars=tail_budget)

        # Build digest
        digest_parts = []
        if context_summary:
            digest_parts.append(context_summary)
        if task_system:
            digest_parts.append(task_system[:self.system_chars])
        digest_parts.append(truncated_user)
        if tool_names:
            digest_parts.append(f"[tools: {', '.join(tool_names)}]")

        digest = "\n".join(digest_parts)

        # Delimit as untrusted
        if self.delimit_untrusted:
            digest = f"<<<UNTRUSTED_INPUT_BEGIN>>>\n{digest}\n<<<UNTRUSTED_INPUT_END>>>"

        # Code detection
        code_fences = _count_code_fences(task_user)
        has_code = code_fences > 0

        # Injection check
        injection_suspected = check_injection(digest) if self.injection_guard else False

        return {
            "digest": digest,
            "scaffolding_stripped_chars": scaffolding_chars,
            "stripped_by": stripped_by,
            "task_tokens": task_tokens,
            "task_chars": len(task_user.strip()),
            "total_tokens": total_tokens,
            "context_summary": context_summary,
            "tool_names": tool_names,
            "has_code": has_code,
            "code_fences": code_fences,
            "injection_suspected": injection_suspected,
        }

    def digest_hash(self, digest: str, classifier_model: str, rubric_version: str) -> str:
        """Compute a cache key hash for the digest."""
        raw = f"{classifier_model}:{rubric_version}:{digest}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]
