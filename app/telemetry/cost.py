"""Token to USD cost estimation."""
from __future__ import annotations

from typing import Optional


class CostAccountant:
    """Computes per-request and per-session cost."""

    def __init__(self, provider_adapter):
        self.provider = provider_adapter

    def estimate(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        classifier_tokens: int = 0,
        classifier_model: str | None = None,
    ) -> dict:
        """Estimate cost breakdown."""
        completion_cost = None
        classifier_cost = None
        total = 0.0

        # Main call cost
        main_cost = self.provider.estimate_cost(model, prompt_tokens, completion_tokens)
        if main_cost is not None:
            completion_cost = main_cost
            total += main_cost

        # Classifier cost
        if classifier_tokens > 0 and classifier_model:
            cls_cost = self.provider.estimate_cost(classifier_model, classifier_tokens, 0)
            if cls_cost is not None:
                classifier_cost = cls_cost
                total += cls_cost

        return {
            "classifier": classifier_cost or 0.0,
            "completion": completion_cost,
            "total": round(total, 6) if total > 0 else None,
        }
