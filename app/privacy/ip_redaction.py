"""IP Redaction & Re-Hydration Module (privacy middleware).

Intercepts prompts containing raw IP addresses, stores session-scoped
mappings in SQLite, replaces IPs with sequential bracketed placeholders
([ipaddress-01], [ipaddress-02], ...) before the prompt reaches the
classifier / tier LLMs, and re-hydrates the placeholders with the original
IPs in the LLM output before the response is returned to the caller.

Design notes:
- SQLite is accessed synchronously; calls are cheap single-row queries
  wrapped in asyncio.to_thread so the event loop is never blocked.
- Placeholders are stable within a session: the same IP always maps to the
  same token across turns, preserving prompt context (and prefix caching).
- Only message *content* is redacted — roles, tool schemas, and other
  fields pass through untouched.
"""
from __future__ import annotations

import asyncio
import re
import sqlite3
import threading
from pathlib import Path

# --- Detection ---------------------------------------------------------------

# IPv4 with strict octet bounds (no version numbers, no decimals).
IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
)

# IPv6: full form, compressed runs (::), and trailing compressions.
IPV6_RE = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
    r"|\b(?:[0-9a-fA-F]{1,4}:){1,7}:"
    r"|:(?::[0-9a-fA-F]{1,4}){1,7}\b"
)

# Combined matcher: IPv4 first so dotted quads are never misread as IPv6.
IP_RE = re.compile(
    f"(?:{IPV4_RE.pattern})|(?:{IPV6_RE.pattern})"
)

# Placeholders exactly as emitted: [ipaddress-01]
PLACEHOLDER_RE = re.compile(r"\[ipaddress-(\d+)\]")

# Fuzzy variant for LLM output formatting quirks: [ipaddress - 01],
# [ ipaddress-1 ], backticked, etc. (spec §4).
PLACEHOLDER_FUZZY_RE = re.compile(r"\[?[ \t]*ipaddress[ \t]*-[ \t]*(\d+)[ \t]*\]?")

PLACEHOLDER_FMT = "[ipaddress-{:02d}]"


class IPRedactionStore:
    """SQLite-backed session-scoped IP↔placeholder mapping store."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # Single connection guarded by a lock — redaction is low-traffic
        # compared to the LLM calls themselves.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS ip_redaction_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id VARCHAR(255) NOT NULL,
                    original_ip VARCHAR(45) NOT NULL,
                    placeholder VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, original_ip),
                    UNIQUE(session_id, placeholder)
                );
                CREATE INDEX IF NOT EXISTS idx_session_id
                    ON ip_redaction_sessions(session_id);
                """
            )

    # --- Read path -------------------------------------------------------------

    def get_mappings(self, session_id: str) -> dict[str, str]:
        """Return {placeholder: original_ip} for a session."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT placeholder, original_ip FROM ip_redaction_sessions "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        return {r["placeholder"]: r["original_ip"] for r in rows}

    def get_or_create_placeholder(self, session_id: str, original_ip: str) -> str:
        """Atomically get or create a placeholder for an IP in a session.

        Combines the lookup and insert in a single transaction to prevent
        race conditions where two concurrent requests select the same
        placeholder number.
        """
        with self._lock, self._conn:
            # Check if IP already mapped
            row = self._conn.execute(
                "SELECT placeholder FROM ip_redaction_sessions "
                "WHERE session_id = ? AND original_ip = ?",
                (session_id, original_ip),
            ).fetchone()
            if row:
                return row["placeholder"]

            # Compute next number and insert atomically
            row = self._conn.execute(
                "SELECT MAX(CAST(substr(placeholder, 12) AS INTEGER)) AS n "
                "FROM ip_redaction_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            n = (row["n"] or 0) + 1
            placeholder = PLACEHOLDER_FMT.format(n)
            self._conn.execute(
                "INSERT OR IGNORE INTO ip_redaction_sessions "
                "(session_id, original_ip, placeholder) VALUES (?, ?, ?)",
                (session_id, original_ip, placeholder),
            )
            return placeholder

    # --- Write path ------------------------------------------------------------

    def record(self, session_id: str, original_ip: str, placeholder: str) -> None:
        """Insert a mapping; IGNORE on unique conflicts (race-safe)."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO ip_redaction_sessions "
                "(session_id, original_ip, placeholder) VALUES (?, ?, ?)",
                (session_id, original_ip, placeholder),
            )

    # --- Maintenance -----------------------------------------------------------

    def purge_older_than(self, hours: float) -> int:
        """Delete records older than `hours`; returns rows removed."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM ip_redaction_sessions "
                "WHERE created_at < datetime('now', ?)",
                (f"-{hours} hours",),
            )
            return cur.rowcount

    def purge_session(self, session_id: str) -> int:
        """Delete all mappings for a session (explicit termination)."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM ip_redaction_sessions WHERE session_id = ?",
                (session_id,),
            )
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class IPRedactionEngine:
    """Redaction + re-hydration service built on IPRedactionStore."""

    def __init__(self, store: IPRedactionStore):
        self.store = store

    # --- Inbound: redact -------------------------------------------------------

    async def redact_messages(self, messages: list, session_id: str) -> bool:
        """Redact IPs in message content in place. Returns True if any change."""
        changed = False
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, str):
                if await self._redact_text_inplace(msg, "content", content, session_id):
                    changed = True
            elif isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and isinstance(block.get("text"), str)
                        and await self._redact_text_inplace(
                            block, "text", block["text"], session_id
                        )
                    ):
                        changed = True
        return changed

    async def _redact_text_inplace(
        self, holder: dict, key: str, text: str, session_id: str
    ) -> bool:
        new_text = await self.redact_text(text, session_id)
        if new_text != text:
            holder[key] = new_text
            return True
        return False

    async def redact_text(self, text: str, session_id: str) -> str:
        """Replace every IP in `text` with a session-stable placeholder."""
        if not text or not IP_RE.search(text):
            return text

        # Snapshot existing mappings once.
        mappings = await asyncio.to_thread(self.store.get_mappings, session_id)
        by_ip = {ip: ph for ph, ip in mappings.items()}

        # Resolve the session's highest existing number so new IPs assigned
        # within this prompt continue the sequence without colliding
        # (DB rows for pending assignments are not yet written).
        def _max_existing() -> int:
            row = None
            for ph in mappings:
                m = re.fullmatch(r"\[ipaddress-(\d+)\]", ph)
                if m:
                    n = int(m.group(1))
                    row = max(row or 0, n)
            return row or 0

        next_n = _max_existing()

        result = []
        last = 0
        pending_new: list[tuple[str, str]] = []
        for match in IP_RE.finditer(text):
            ip = match.group(0)
            ph = by_ip.get(ip)
            if ph is None:
                next_n += 1
                ph = PLACEHOLDER_FMT.format(next_n)
                by_ip[ip] = ph
                pending_new.append((ip, ph))
            result.append(text[last:match.start()])
            result.append(ph)
            last = match.end()
        result.append(text[last:])

        for ip, ph in pending_new:
            await asyncio.to_thread(self.store.record, session_id, ip, ph)

        return "".join(result)

    # --- Outbound: re-hydrate ----------------------------------------------------

    def rehydrate_text_sync(self, text: str, session_id: str) -> str:
        """Replace placeholder tokens (incl. fuzzy variants) with original IPs."""
        if "ipaddress" not in text:
            return text
        mappings = self.store.get_mappings(session_id)
        if not mappings:
            return text

        def _sub(match: re.Match) -> str:
            n = int(match.group(1))
            key = PLACEHOLDER_FMT.format(n)
            # Fuzzy matches may have lost zero-padding: try unpadded too.
            return mappings.get(key) or mappings.get(
                PLACEHOLDER_FMT.format(n).replace("-0", "-"), ""
            ) or match.group(0)

        return PLACEHOLDER_FUZZY_RE.sub(_sub, text)

    async def rehydrate_text(self, text: str, session_id: str) -> str:
        return await asyncio.to_thread(self.rehydrate_text_sync, text, session_id)

    async def rehydrate_message_content(self, content, session_id: str):
        """Re-hydrate a message content value (str or block list)."""
        if isinstance(content, str):
            return await self.rehydrate_text(content, session_id)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    block["text"] = await self.rehydrate_text(
                        block["text"], session_id
                    )
        return content
