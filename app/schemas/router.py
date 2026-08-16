"""Router-internal schemas for routing decisions and session state."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class Level(str, Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"

    @classmethod
    def from_str(cls, s: str) -> "Level":
        s = s.upper().strip()
        for m in cls:
            if m.value == s:
                return m
        raise ValueError(f"Invalid level: {s}")

    @property
    def numeric(self) -> int:
        return {"L1": 1, "L2": 2, "L3": 3, "L4": 4}[self.value]

    @classmethod
    def from_numeric(cls, n: int) -> "Level":
        n = max(1, min(4, n))
        return cls(f"L{n}")

    def __lt__(self, other: "Level") -> bool:
        return self.numeric < other.numeric

    def __le__(self, other: "Level") -> bool:
        return self.numeric <= other.numeric

    def __gt__(self, other: "Level") -> bool:
        return self.numeric > other.numeric

    def __ge__(self, other: "Level") -> bool:
        return self.numeric >= other.numeric


class ClassificationSource(str, Enum):
    SESSION = "session"
    MODEL = "model"
    CACHE = "cache"
    HEURISTIC = "heuristic"
    OVERRIDE = "override"
    DEFAULT = "default"


class SessionSource(str, Enum):
    HEADER = "header"
    BODY = "body"
    USER_FIELD = "user_field"
    FINGERPRINT = "fingerprint"
    NONE = "none"


class ClassificationResult(BaseModel):
    level: Optional[Level] = None
    confidence: float = 1.0
    reason: str = ""
    source: ClassificationSource = ClassificationSource.MODEL
    classifier_model: Optional[str] = None
    rubric_version: Optional[str] = None
    latency_ms: int = 0


class RouteDecision(BaseModel):
    level: Level
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    classification: ClassificationResult
    fallback_used: bool = False
    escalated: bool = False
    escalated_from: Optional[Level] = None
    estimated_cost_usd: Optional[float] = None


class SessionStatus(str, Enum):
    PINNED = "pinned"
    PROVISIONAL = "provisional"
    STICKY_MODEL = "sticky_model"
    CLASSIFYING = "classifying"


class EscalationState(BaseModel):
    score: int = 0
    count: int = 0
    original_level: Optional[Level] = None
    last_escalated_turn: int = 0
    last_trigger: list[str] = Field(default_factory=list)
    cooldown_until_turn: int = 0
    retry_count: int = 0


class SessionPin(BaseModel):
    session_id: str
    level: Level
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    status: SessionStatus = SessionStatus.PINNED
    classification: Optional[ClassificationResult] = None
    turn_count: int = 0
    escalation: EscalationState = Field(default_factory=EscalationState)
    pinned_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_seen_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: Optional[str] = None
    cost_usd_total: float = 0.0
    provisional_turns: int = 0

    def touch(self, idle_ttl: int, max_ttl: int | None = None) -> None:
        """Update last_seen and expiry timestamps."""
        now = datetime.now(timezone.utc)
        self.last_seen_at = now.isoformat()
        new_expiry = now + timedelta(seconds=idle_ttl)
        if max_ttl:
            absolute = datetime.fromisoformat(self.pinned_at) + timedelta(seconds=max_ttl)
            if absolute < new_expiry:
                new_expiry = absolute
        self.expires_at = new_expiry.isoformat()

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        try:
            exp = datetime.fromisoformat(self.expires_at)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) > exp
        except Exception:
            return False
