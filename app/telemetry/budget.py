"""Budget management: pre-request cost estimation and daily spend limits."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass
class BudgetDecision:
    """Result of a budget check."""

    allowed: bool
    reason: str
    estimated_cost: float
    downgrade_level: str | None = None


class BudgetManager:
    """Manages per-request and daily budget enforcement.

    Uses in-memory tracking by default; Redis can be added for multi-worker.
    Pre-request: estimates worst-case cost from pricing cache.
    Post-request: reconciles with actual token usage.
    """

    def __init__(self, provider_adapter, redis_client=None):
        self.provider = provider_adapter
        self._redis = redis_client
        self._daily_spend: dict[str, float] = {}  # date_str -> total
        self._reservations: dict[str, float] = {}  # session_id -> reserved amount
        self._lock = asyncio.Lock()

    def _today_key(self) -> str:
        return datetime.now(UTC).strftime("%Y-%m-%d")

    def _estimate_tokens(self, messages: list) -> int:
        """Conservative token estimate: ~4 chars per token."""
        total_chars = 0
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        total_chars += len(block["text"])
        return max(total_chars // 4, 1)

    async def check_and_reserve(
        self,
        session_id: str | None,
        model: str,
        messages: list,
        tier_max_cost_usd: float,
        daily_limit_usd: float,
        on_exceeded: str = "downgrade",
        downgrade_to: str = "L2",
    ) -> BudgetDecision:
        """Check budget before dispatching to provider.

        Returns BudgetDecision indicating whether the request is allowed,
        should be downgraded, or rejected.
        """
        est_prompt_tokens = self._estimate_tokens(messages)
        pricing = self.provider.get_pricing(model)
        if pricing is None:
            # No pricing data — allow (fail open, log later)
            return BudgetDecision(
                allowed=True,
                reason="no pricing data available",
                estimated_cost=0.0,
            )

        # Estimate worst-case cost: prompt + assume max ~4K completion tokens
        est_completion_tokens = 4096
        est_cost = (
            est_prompt_tokens * pricing["prompt"]
            + est_completion_tokens * pricing["completion"]
        )

        # Check per-tier limit
        if est_cost > tier_max_cost_usd:
            if on_exceeded == "downgrade":
                return BudgetDecision(
                    allowed=True,
                    reason=f"estimated cost ${est_cost:.4f} exceeds tier limit ${tier_max_cost_usd:.4f}",
                    estimated_cost=est_cost,
                    downgrade_level=downgrade_to,
                )
            return BudgetDecision(
                allowed=False,
                reason=f"estimated cost ${est_cost:.4f} exceeds tier limit ${tier_max_cost_usd:.4f}",
                estimated_cost=est_cost,
            )

        # Check daily limit
        async with self._lock:
            today = self._today_key()
            current_spend = self._daily_spend.get(today, 0.0)
            if current_spend + est_cost > daily_limit_usd:
                if on_exceeded == "downgrade":
                    return BudgetDecision(
                        allowed=True,
                        reason=f"daily spend ${current_spend:.2f}+${est_cost:.4f} exceeds limit ${daily_limit_usd:.2f}",
                        estimated_cost=est_cost,
                        downgrade_level=downgrade_to,
                    )
                return BudgetDecision(
                    allowed=False,
                    reason=f"daily spend limit ${daily_limit_usd:.2f} exceeded",
                    estimated_cost=est_cost,
                )

            # Reserve
            if session_id:
                self._reservations[session_id] = est_cost
            self._daily_spend[today] = current_spend + est_cost

        return BudgetDecision(
            allowed=True,
            reason="ok",
            estimated_cost=est_cost,
        )

    async def reconcile(
        self,
        session_id: str | None,
        model: str,
        actual_prompt_tokens: int,
        actual_completion_tokens: int,
    ) -> None:
        """Reconcile reservation with actual cost after response."""
        actual_cost = self.provider.estimate_cost(
            model, actual_prompt_tokens, actual_completion_tokens
        )
        if actual_cost is None:
            return

        async with self._lock:
            # Adjust daily spend: remove reservation, add actual
            today = self._today_key()
            reserved = self._reservations.pop(session_id, 0.0) if session_id else 0.0
            self._daily_spend[today] = (
                self._daily_spend.get(today, 0.0) - reserved + actual_cost
            )
            # Clamp to non-negative (safety)
            if self._daily_spend[today] < 0:
                self._daily_spend[today] = 0.0

    async def get_daily_spend(self) -> float:
        """Return total spend today."""
        async with self._lock:
            return self._daily_spend.get(self._today_key(), 0.0)
