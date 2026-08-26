"""Temporal awareness engine — resolves relative date/time expressions to concrete ISO dates/times.

All patterns defined in rules.py are handled. The engine replaces matched
expressions with concrete dates (YYYY-MM-DD), datetimes (YYYY-MM-DDTHH:MM:SS+TZ),
or date ranges (YYYY-MM-DD..YYYY-MM-DD) depending on the expression type.
"""
import re
from datetime import timedelta
from typing import Optional, List

import pendulum
from pendulum import DateTime
from zoneinfo import ZoneInfo

from app.config.schema import TemporalAwarenessConfig
from app.temporal_awareness.rules import COMPILED_TEMPORAL_PATTERNS


class TemporalAwarenessEngine:
    def __init__(self, config: TemporalAwarenessConfig):
        self.config = config
        self.default_tz = self.config.default_timezone  # IANA tz string for pendulum

    # ── Resolution helpers ──────────────────────────────────────────

    @staticmethod
    def _unit_to_timedelta(unit: str, n: int = 1) -> timedelta:
        """Map a singular unit name to a timedelta of n units."""
        unit = unit.lower().rstrip("s")
        return {
            "day": timedelta(days=n),
            "week": timedelta(weeks=n),
            "month": timedelta(days=n * 30),  # approximate
            "year": timedelta(days=n * 365),  # approximate
        }.get(unit, timedelta(days=n))

    def _resolve_weekday(self, name: str) -> int:
        """Map a weekday name to pendulum's day-of-week integer (0=Monday)."""
        names = {
            "monday": pendulum.MONDAY,
            "tuesday": pendulum.TUESDAY,
            "wednesday": pendulum.WEDNESDAY,
            "thursday": pendulum.THURSDAY,
            "friday": pendulum.FRIDAY,
            "saturday": pendulum.SATURDAY,
            "sunday": pendulum.SUNDAY,
        }
        return names.get(name.lower(), 0)

    def _format_date(self, dt: DateTime) -> str:
        return dt.to_date_string()

    def _format_datetime(self, dt: DateTime) -> str:
        return dt.isoformat()

    def _format_week_range(self, start: DateTime) -> str:
        """Format a week as Monday..Sunday date range."""
        end = start.end_of("week")
        return f"{start.to_date_string()}..{end.to_date_string()}"

    def _format_month_range(self, dt: DateTime) -> str:
        """Format a month as first..last day date range."""
        start = dt.start_of("month")
        end = dt.end_of("month")
        return f"{start.to_date_string()}..{end.to_date_string()}"

    def _format_year_range(self, dt: DateTime) -> str:
        """Format a year as Jan 1..Dec 31 date range."""
        start = dt.start_of("year")
        end = dt.end_of("year")
        return f"{start.to_date_string()}..{end.to_date_string()}"

    # ── Main resolution dispatcher ─────────────────────────────────

    def _resolve_match(self, match: re.Match, tag: str, now: DateTime) -> Optional[str]:
        """Resolve a matched temporal expression to a concrete date/datetime string."""
        text = match.group(0)
        groups = match.groups()

        if tag == "today":
            return self._format_date(now)

        if tag == "yesterday":
            return self._format_date(now - timedelta(days=1))

        if tag == "tomorrow":
            return self._format_date(now + timedelta(days=1))

        if tag == "now_datetime":
            return self._format_datetime(now)

        if tag == "this_morning_datetime":
            # Assuming morning is 6 AM to 12 PM
            return self._format_datetime(now.set(hour=9, minute=0, second=0, microsecond=0))

        if tag == "this_afternoon_datetime":
            # Assuming afternoon is 12 PM to 6 PM
            return self._format_datetime(now.set(hour=15, minute=0, second=0, microsecond=0))

        if tag == "this_evening_datetime":
            # Assuming evening is 6 PM to 10 PM
            return self._format_datetime(now.set(hour=20, minute=0, second=0, microsecond=0))

        if tag == "tonight_datetime":
            # Assuming tonight is 10 PM
            return self._format_datetime(now.set(hour=22, minute=0, second=0, microsecond=0))

        if tag == "specific_time_datetime":
            # groups: (at|by), hour, minute, am/pm
            # or: hour, minute, am/pm
            if len(groups) == 4: # (at|by), hour, minute, am/pm
                hour_str = groups[1]
                minute_str = groups[2]
                ampm = groups[3]
            else: # hour, minute, am/pm
                hour_str = groups[0]
                minute_str = groups[1]
                ampm = groups[2]

            hour = int(hour_str)
            minute = int(minute_str) if minute_str else 0

            if ampm and ampm.lower() == "pm" and hour < 12:
                hour += 12
            elif ampm and ampm.lower() == "am" and hour == 12: # 12 AM is midnight
                hour = 0

            return self._format_datetime(now.set(hour=hour, minute=minute, second=0, microsecond=0))

        # Days of the week: (last|next) <weekday>
        if tag == "relative_day_of_week":
            direction = groups[0].lower()  # "last" or "next"
            day_name = groups[1].lower()
            target_dow = self._resolve_weekday(day_name)
            current_dow = now.day_of_week
            if direction == "last":
                diff = (current_dow - target_dow) % 7
                if diff == 0:
                    diff = 7
                return self._format_date(now - timedelta(days=diff))
            else:  # "next"
                diff = (target_dow - current_dow) % 7
                if diff == 0:
                    diff = 7
                return self._format_date(now + timedelta(days=diff))

        # (this|coming) <weekday>
        if tag == "this_coming_day_of_week":
            day_name = groups[1].lower()
            target_dow = self._resolve_weekday(day_name)
            current_dow = now.day_of_week
            diff = (target_dow - current_dow) % 7
            if diff == 0:
                diff = 7  # "coming" implies a future day, not today
            return self._format_date(now + timedelta(days=diff))

        # Relative weeks
        if tag == "last_week":
            start = now.start_of("week") - timedelta(weeks=1)
            return self._format_week_range(start)

        if tag == "this_week":
            start = now.start_of("week")
            return self._format_week_range(start)

        if tag == "next_week":
            start = now.start_of("week") + timedelta(weeks=1)
            return self._format_week_range(start)

        # Relative months
        if tag == "last_month":
            dt = now - timedelta(days=now.day)  # go to last day of prev month
            dt = dt.start_of("month")
            return self._format_month_range(dt)

        if tag == "this_month":
            return self._format_month_range(now)

        if tag == "next_month":
            # First day of next month
            dt = now.start_of("month") + timedelta(days=32)
            dt = dt.start_of("month")
            return self._format_month_range(dt)

        # Relative years
        if tag == "last_year":
            dt = now.set(year=now.year - 1)
            return self._format_year_range(dt)

        if tag == "this_year":
            return self._format_year_range(now)

        if tag == "next_year":
            dt = now.set(year=now.year + 1)
            return self._format_year_range(dt)

        # (last|past) N <unit>s
        if tag == "past_n_units":
            n = int(groups[1])
            unit = groups[2]
            delta = self._unit_to_timedelta(unit, n)
            return self._format_date(now - delta)

        # N <unit>s ago
        if tag == "n_units_ago":
            n = int(groups[0])
            unit = groups[1]
            delta = self._unit_to_timedelta(unit, n)
            return self._format_date(now - delta)

        # in N <unit>s
        if tag == "in_n_units":
            n = int(groups[0])
            unit = groups[1]
            delta = self._unit_to_timedelta(unit, n)
            return self._format_date(now + delta)

        return None

    # ── Message processing ─────────────────────────────────────────

    def process_message(self, message_content: str) -> str:
        if not self.config.enabled:
            return message_content

        now = pendulum.now(self.default_tz)
        processed = message_content

        # Sort by pattern length descending so longer matches (e.g.
        # "last 3 days") are resolved before shorter ones ("last" alone)
        # that might overlap. We iterate all compiled patterns and replace
        # each match with its resolved value.
        for compiled, tag in COMPILED_TEMPORAL_PATTERNS:
            # Find all matches, replace right-to-left to preserve indices
            matches = list(compiled.finditer(processed))
            if not matches:
                continue
            # Replace from end to start so index offsets stay valid
            for match in reversed(matches):
                resolved = self._resolve_match(match, tag, now)
                if resolved:
                    processed = (
                        processed[: match.start()]
                        + resolved
                        + processed[match.end() :]
                    )

        return processed

    def process_messages(self, messages: List[dict]) -> List[dict]:
        if not self.config.enabled:
            return messages

        processed_messages = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")

            # Process both system and user role messages so temporal
            # expressions in the system prompt (e.g. "today") are also
            # replaced with concrete dates.
            if role in ("system", "user") and isinstance(content, str):
                message["content"] = self.process_message(content)
            elif role in ("system", "user") and isinstance(content, list):
                # Multimodal content — process each text block
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        block["text"] = self.process_message(block["text"])

            processed_messages.append(message)
        return processed_messages
