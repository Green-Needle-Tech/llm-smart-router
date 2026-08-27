"""Session store ABC and pin data structures."""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.router import SessionPin


class SessionStore(ABC):
    """Abstract session store for pin management."""

    @abstractmethod
    async def get(self, session_id: str) -> SessionPin | None:
        """Look up a session pin by id. Returns None if not found or expired."""
        ...

    @abstractmethod
    async def put(self, pin: SessionPin) -> None:
        """Insert or update a session pin."""
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """Delete a session pin. Returns True if existed."""
        ...

    @abstractmethod
    async def delete_all(self) -> int:
        """Flush all pins. Returns count deleted."""
        ...

    @abstractmethod
    async def list_sessions(self, level: str | None = None, offset: int = 0, limit: int = 50) -> list[SessionPin]:
        """List active pins, optionally filtered by level."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return number of active pins."""
        ...

    @abstractmethod
    async def reserve(self, session_id: str, ttl_seconds: int) -> bool:
        """Atomically reserve a session for classification (SETNX-like).
        Returns True if won the reservation, False if already reserved."""
        ...

    @abstractmethod
    async def release(self, session_id: str) -> None:
        """Release a classification reservation."""
        ...
