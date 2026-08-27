"""Temporal awareness engine — resolves relative date/time expressions to concrete ISO dates/times.

All patterns defined in rules.py are handled. The engine replaces matched
expressions with concrete dates (YYYY-MM-DD), datetimes (YYYY-MM-DDTHH:MM:SS+TZ),
or date ranges (YYYY-MM-DD..YYYY-MM-DD) depending on the expression type.

Supports typo tolerance, abbreviations, colloquial expressions, military time,
seasons, quarters, and end/beginning-of-period expressions.
"""
import re
from datetime import timedelta

import pendulum
from pendulum import DateTime

from app.config.schema import TemporalAwarenessConfig
from app.temporal_awareness.rules import COMPILED_TEMPORAL_PATTERNS


class TemporalAwarenessEngine:
    def __init__(self, config: TemporalAwarenessConfig):
        self.config = config
        self.default_tz = self.config.default_timezone  # IANA tz string for pendulum

    # ── Resolution helpers ──────────────────────────────────────────

    @staticmethod
    def _unit_to_timedelta(unit: str, n: int = 1) -> timedelta:
        """Map a unit name (with typo tolerance) to a timedelta of n units."""
        unit = unit.lower().rstrip("s")
        # Normalize common typos
        unit = {
            "dy": "day", "dys": "day",
            "wk": "week", "wks": "week",
            "mnth": "month", "mnths": "month",
            "yr": "year", "yrs": "year",
            "hr": "hour", "hrs": "hour",
            "min": "minute", "mins": "minute",
            "sec": "second", "secs": "second",
        }.get(unit, unit)
        return {
            "day": timedelta(days=n),
            "week": timedelta(weeks=n),
            "month": timedelta(days=n * 30),  # approximate
            "year": timedelta(days=n * 365),  # approximate
            "hour": timedelta(hours=n),
            "minute": timedelta(minutes=n),
            "second": timedelta(seconds=n),
        }.get(unit, timedelta(days=n))

    def _resolve_weekday(self, name: str) -> int:
        """Map a weekday name (with abbreviation support) to pendulum's day-of-week integer (0=Monday)."""
        name = name.lower()
        names = {
            "monday": pendulum.MONDAY, "mon": pendulum.MONDAY,
            "tuesday": pendulum.TUESDAY, "tue": pendulum.TUESDAY, "tues": pendulum.TUESDAY,
            "wednesday": pendulum.WEDNESDAY, "wed": pendulum.WEDNESDAY,
            "thursday": pendulum.THURSDAY, "thu": pendulum.THURSDAY, "thur": pendulum.THURSDAY, "thurs": pendulum.THURSDAY,
            "friday": pendulum.FRIDAY, "fri": pendulum.FRIDAY,
            "saturday": pendulum.SATURDAY, "sat": pendulum.SATURDAY,
            "sunday": pendulum.SUNDAY, "sun": pendulum.SUNDAY,
        }
        return names.get(name, 0)

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

    def _format_quarter_range(self, dt: DateTime, quarter: int) -> str:
        """Format a calendar quarter as a date range."""
        start_month = (quarter - 1) * 3 + 1
        start = dt.set(month=start_month, day=1, hour=0, minute=0, second=0, microsecond=0)
        end = start.end_of("month")
        # Advance 2 more months to quarter end
        for _ in range(2):
            end = (end + timedelta(days=1)).end_of("month")
        return f"{start.to_date_string()}..{end.to_date_string()}"

    def _resolve_season(self, season: str, now: DateTime, direction: str) -> str:
        """Resolve a season to a date range (approximate)."""
        season = season.lower()
        # Northern hemisphere approximate start months
        season_months = {
            "spring": 3, "summer": 6, "fall": 9, "autumn": 9, "winter": 12,
        }
        month = season_months.get(season, 3)
        if direction == "last":
            year = now.year - 1 if now.month < month else now.year
        elif direction == "next":
            year = now.year + 1 if now.month >= month else now.year
        else:  # this
            year = now.year
        start = now.set(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)
        end = (start + timedelta(days=89)).end_of("month")  # ~3 months
        return f"{start.to_date_string()}..{end.to_date_string()}"

    # ── Time parsing helpers ────────────────────────────────────────

    @staticmethod
    def _parse_ampm(ampm: str | None) -> str | None:
        """Normalize AM/PM variants (a.m., p.m.) to 'am'/'pm'."""
        if ampm is None:
            return None
        ampm = ampm.lower().replace(".", "")
        if ampm in ("am", "pm"):
            return ampm
        return None

    def _apply_ampm(self, hour: int, ampm: str | None) -> int:
        """Apply AM/PM conversion to a 12-hour value."""
        ampm = self._parse_ampm(ampm)
        if ampm == "pm" and hour < 12:
            return hour + 12
        if ampm == "am" and hour == 12:
            return 0
        return hour

    # ── Main resolution dispatcher ─────────────────────────────────

    def _resolve_match(self, match: re.Match, tag: str, now: DateTime) -> str | None:
        """Resolve a matched temporal expression to a concrete date/datetime string."""
        groups = match.groups()

        # ═══════════════════════════════════════════════════════════
        # COMPOUND RELATIVE DAYS
        # ═══════════════════════════════════════════════════════════

        if tag == "day_after_tomorrow":
            return self._format_date(now + timedelta(days=2))

        if tag == "day_before_yesterday":
            return self._format_date(now - timedelta(days=2))

        if tag == "following_period":
            unit = groups[0].lower() if groups and groups[0] else "day"
            if unit == "week":
                start = now.start_of("week") + timedelta(weeks=1)
                return self._format_week_range(start)
            elif unit == "month":
                dt = now.start_of("month") + timedelta(days=32)
                return self._format_month_range(dt.start_of("month"))
            else:
                return self._format_date(now + timedelta(days=1))

        # ═══════════════════════════════════════════════════════════
        # FORTNIGHT
        # ═══════════════════════════════════════════════════════════

        if tag == "fortnight_ago":
            return self._format_date(now - timedelta(weeks=2))

        if tag == "in_fortnight" or tag == "fortnight_from_now":
            return self._format_date(now + timedelta(weeks=2))

        # ═══════════════════════════════════════════════════════════
        # RELATIVE DAY PARTS (yesterday/tomorrow + morning/evening/night)
        # ═══════════════════════════════════════════════════════════

        if tag == "yesterday_morning":
            return self._format_datetime((now - timedelta(days=1)).set(hour=9, minute=0, second=0, microsecond=0))

        if tag == "yesterday_afternoon":
            return self._format_datetime((now - timedelta(days=1)).set(hour=15, minute=0, second=0, microsecond=0))

        if tag == "yesterday_evening":
            return self._format_datetime((now - timedelta(days=1)).set(hour=20, minute=0, second=0, microsecond=0))

        if tag == "yesterday_night":
            return self._format_datetime((now - timedelta(days=1)).set(hour=22, minute=0, second=0, microsecond=0))

        if tag == "last_night":
            return self._format_datetime((now - timedelta(days=1)).set(hour=22, minute=0, second=0, microsecond=0))

        if tag == "tomorrow_morning":
            return self._format_datetime((now + timedelta(days=1)).set(hour=9, minute=0, second=0, microsecond=0))

        if tag == "tomorrow_afternoon":
            return self._format_datetime((now + timedelta(days=1)).set(hour=15, minute=0, second=0, microsecond=0))

        if tag == "tomorrow_evening":
            return self._format_datetime((now + timedelta(days=1)).set(hour=20, minute=0, second=0, microsecond=0))

        if tag == "tomorrow_night":
            return self._format_datetime((now + timedelta(days=1)).set(hour=22, minute=0, second=0, microsecond=0))

        # ═══════════════════════════════════════════════════════════
        # SPECIFIC TIMES
        # ═══════════════════════════════════════════════════════════

        if tag == "military_time":
            hh = int(groups[0])
            mm = int(groups[1])
            return self._format_datetime(now.set(hour=hh, minute=mm, second=0, microsecond=0))

        if tag == "military_time_hundred":
            hh = int(groups[0])
            if hh > 23:
                hh = hh % 24
            return self._format_datetime(now.set(hour=hh, minute=0, second=0, microsecond=0))

        if tag == "quarter_past":
            hour = int(groups[0])
            ampm = groups[1] if len(groups) > 1 else None
            hour = self._apply_ampm(hour, ampm)
            return self._format_datetime(now.set(hour=hour, minute=15, second=0, microsecond=0))

        if tag == "quarter_to":
            hour = int(groups[0])
            ampm = groups[1] if len(groups) > 1 else None
            hour = self._apply_ampm(hour, ampm)
            hour = (hour - 1) % 24
            return self._format_datetime(now.set(hour=hour, minute=45, second=0, microsecond=0))

        if tag == "half_past":
            hour = int(groups[0])
            ampm = groups[1] if len(groups) > 1 else None
            hour = self._apply_ampm(hour, ampm)
            return self._format_datetime(now.set(hour=hour, minute=30, second=0, microsecond=0))

        if tag == "oclock":
            hour = int(groups[0])
            return self._format_datetime(now.set(hour=hour, minute=0, second=0, microsecond=0))

        if tag == "specific_time_datetime":
            # Groups can be: (hour, minute, ampm) or (hour, ampm)
            if len(groups) == 3:
                hour = int(groups[0])
                minute = int(groups[1]) if groups[1] else 0
                ampm = groups[2]
            elif len(groups) == 2:
                hour = int(groups[0])
                minute = 0
                ampm = groups[1]
            else:
                return None
            hour = self._apply_ampm(hour, ampm)
            return self._format_datetime(now.set(hour=hour, minute=minute, second=0, microsecond=0))

        # ═══════════════════════════════════════════════════════════
        # END / BEGINNING / ABBREVIATIONS
        # ═══════════════════════════════════════════════════════════

        if tag == "cob" or tag == "eod":
            return self._format_datetime(now.set(hour=17, minute=0, second=0, microsecond=0))

        if tag == "end_of_day":
            return self._format_datetime(now.set(hour=23, minute=59, second=59, microsecond=0))

        if tag == "end_of_period":
            unit = groups[0].lower()
            if unit == "week":
                return self._format_date(now.end_of("week"))
            elif unit == "month":
                return self._format_date(now.end_of("month"))
            else:
                return self._format_date(now.end_of("year"))

        if tag == "start_of_period":
            unit = groups[0].lower()
            if unit == "week":
                return self._format_date(now.start_of("week"))
            elif unit == "month":
                return self._format_date(now.start_of("month"))
            else:
                return self._format_date(now.start_of("year"))

        if tag == "eow":
            return self._format_date(now.end_of("week"))

        if tag == "eom":
            return self._format_date(now.end_of("month"))

        if tag == "eoy":
            return self._format_date(now.end_of("year"))

        if tag == "month_end":
            return self._format_date(now.end_of("month"))

        if tag == "year_end":
            return self._format_date(now.end_of("year"))

        # ═══════════════════════════════════════════════════════════
        # FIRST THING / MEAL TIMES / MIDWEEK
        # ═══════════════════════════════════════════════════════════

        if tag == "first_thing_tomorrow":
            return self._format_datetime((now + timedelta(days=1)).set(hour=8, minute=0, second=0, microsecond=0))

        if tag == "first_thing_morning":
            return self._format_datetime(now.set(hour=8, minute=0, second=0, microsecond=0))

        if tag == "lunchtime":
            return self._format_datetime(now.set(hour=12, minute=30, second=0, microsecond=0))

        if tag == "dinnertime":
            return self._format_datetime(now.set(hour=19, minute=0, second=0, microsecond=0))

        if tag == "teatime":
            return self._format_datetime(now.set(hour=16, minute=0, second=0, microsecond=0))

        if tag == "breakfast_time":
            return self._format_datetime(now.set(hour=8, minute=0, second=0, microsecond=0))

        if tag == "midweek":
            # Wednesday of current week
            start = now.start_of("week")
            return self._format_date(start + timedelta(days=2))

        # ═══════════════════════════════════════════════════════════
        # COLLOQUIAL EXPRESSIONS
        # ═══════════════════════════════════════════════════════════

        if tag == "the_other_day":
            return self._format_date(now - timedelta(days=2))

        if tag == "a_long_time_ago":
            return self._format_date(now - timedelta(days=365))

        if tag == "a_while_ago":
            return self._format_date(now - timedelta(days=14))

        if tag == "in_a_bit":
            return self._format_datetime(now + timedelta(hours=1))

        if tag == "shortly":
            return self._format_datetime(now + timedelta(minutes=15))

        if tag == "soon":
            return self._format_datetime(now + timedelta(hours=1))

        # ═══════════════════════════════════════════════════════════
        # COUPLE / FEW UNITS
        # ═══════════════════════════════════════════════════════════

        if tag == "couple_units_ago":
            return self._format_date(now - timedelta(days=2))

        if tag == "few_units_ago":
            unit = groups[0].lower().rstrip("s")
            n = 3
            delta = self._unit_to_timedelta(unit, n)
            return self._format_date(now - delta)

        if tag == "few_units_from_now":
            unit = groups[0].lower().rstrip("s")
            n = 3
            delta = self._unit_to_timedelta(unit, n)
            return self._format_date(now + delta)

        # ═══════════════════════════════════════════════════════════
        # N UNITS AGO / FROM NOW / IN / BACK / HENCE
        # ═══════════════════════════════════════════════════════════

        if tag == "past_n_units":
            n = int(groups[0])
            unit = groups[1]
            delta = self._unit_to_timedelta(unit, n)
            unit_clean = unit.lower().rstrip("s")
            if unit_clean in ("hour", "hr", "minute", "min", "second", "sec"):
                return self._format_datetime(now - delta)
            return self._format_date(now - delta)

        if tag == "n_units_ago":
            n = int(groups[0])
            unit = groups[1]
            delta = self._unit_to_timedelta(unit, n)
            # For hours/minutes/seconds, return a datetime
            unit_clean = unit.lower().rstrip("s")
            if unit_clean in ("hour", "hr", "minute", "min", "second", "sec"):
                return self._format_datetime(now - delta)
            return self._format_date(now - delta)

        if tag == "n_units_from_now":
            n = int(groups[0])
            unit = groups[1]
            delta = self._unit_to_timedelta(unit, n)
            unit_clean = unit.lower().rstrip("s")
            if unit_clean in ("hour", "hr", "minute", "min", "second", "sec"):
                return self._format_datetime(now + delta)
            return self._format_date(now + delta)

        if tag == "in_n_units":
            n = int(groups[0])
            unit = groups[1]
            delta = self._unit_to_timedelta(unit, n)
            # For hours/minutes/seconds, return a datetime
            unit_clean = unit.lower().rstrip("s")
            if unit_clean in ("hour", "hr", "minute", "min", "second", "sec"):
                return self._format_datetime(now + delta)
            return self._format_date(now + delta)

        # ═══════════════════════════════════════════════════════════
        # A/AN UNIT AGO / FROM NOW / HENCE
        # ═══════════════════════════════════════════════════════════

        if tag == "a_unit_ago":
            unit = groups[0]
            delta = self._unit_to_timedelta(unit, 1)
            return self._format_date(now - delta)

        if tag == "a_unit_from_now":
            unit = groups[0]
            delta = self._unit_to_timedelta(unit, 1)
            return self._format_date(now + delta)

        # ═══════════════════════════════════════════════════════════
        # SEASONS
        # ═══════════════════════════════════════════════════════════

        if tag == "relative_season":
            text = match.group(0).lower().split()
            direction = text[0]  # this/last/next
            season = text[1]
            return self._resolve_season(season, now, direction)

        # ═══════════════════════════════════════════════════════════
        # QUARTERS & DECADES
        # ═══════════════════════════════════════════════════════════

        if tag == "this_quarter":
            q = (now.month - 1) // 3 + 1
            return self._format_quarter_range(now, q)

        if tag == "last_quarter":
            q = (now.month - 1) // 3 + 1
            if q == 1:
                return self._format_quarter_range(now.set(year=now.year - 1), 4)
            return self._format_quarter_range(now, q - 1)

        if tag == "next_quarter":
            q = (now.month - 1) // 3 + 1
            if q == 4:
                return self._format_quarter_range(now.set(year=now.year + 1), 1)
            return self._format_quarter_range(now, q + 1)

        if tag == "quarter_n":
            q = int(groups[0])
            return self._format_quarter_range(now, q)

        if tag == "relative_decade":
            text = match.group(0).lower().split()
            direction = text[0]
            if direction == "last":
                return f"{now.year - 10}..{now.year - 1}"
            elif direction == "next":
                return f"{now.year + 1}..{now.year + 10}"
            else:
                return f"{now.year // 10 * 10}..{now.year // 10 * 10 + 9}"

        # ═══════════════════════════════════════════════════════════
        # DAYS OF THE WEEK
        # ═══════════════════════════════════════════════════════════

        if tag == "relative_day_of_week":
            direction = groups[0].lower()  # last/next/this/previous
            day_name = groups[1].lower()
            target_dow = self._resolve_weekday(day_name)
            current_dow = now.day_of_week
            if direction in ("last", "previous"):
                diff = (current_dow - target_dow) % 7
                if diff == 0:
                    diff = 7
                return self._format_date(now - timedelta(days=diff))
            elif direction == "this":
                # "this Monday" = the Monday of the current week
                start = now.start_of("week")
                return self._format_date(start + timedelta(days=target_dow))
            else:  # "next"
                diff = (target_dow - current_dow) % 7
                if diff == 0:
                    diff = 7
                return self._format_date(now + timedelta(days=diff))

        if tag == "coming_day_of_week":
            day_name = groups[0].lower()
            target_dow = self._resolve_weekday(day_name)
            current_dow = now.day_of_week
            diff = (target_dow - current_dow) % 7
            if diff == 0:
                diff = 7
            return self._format_date(now + timedelta(days=diff))

        if tag == "on_weekday":
            day_name = groups[0].lower()
            target_dow = self._resolve_weekday(day_name)
            current_dow = now.day_of_week
            # "on Monday" = next occurrence of that day (today if it is that day)
            diff = (target_dow - current_dow) % 7
            return self._format_date(now + timedelta(days=diff))

        if tag == "bare_weekday":
            day_name = groups[0].lower()
            target_dow = self._resolve_weekday(day_name)
            current_dow = now.day_of_week
            diff = (target_dow - current_dow) % 7
            if diff == 0:
                # If today is that day, return today
                return self._format_date(now)
            return self._format_date(now + timedelta(days=diff))

        # ═══════════════════════════════════════════════════════════
        # WEEKEND
        # ═══════════════════════════════════════════════════════════

        if tag == "this_weekend":
            # Saturday of current week
            start = now.start_of("week") + timedelta(days=5)
            end = start + timedelta(days=1)
            return f"{start.to_date_string()}..{end.to_date_string()}"

        if tag == "last_weekend":
            start = now.start_of("week") + timedelta(days=5) - timedelta(weeks=1)
            end = start + timedelta(days=1)
            return f"{start.to_date_string()}..{end.to_date_string()}"

        if tag == "next_weekend":
            start = now.start_of("week") + timedelta(days=5) + timedelta(weeks=1)
            end = start + timedelta(days=1)
            return f"{start.to_date_string()}..{end.to_date_string()}"

        # ═══════════════════════════════════════════════════════════
        # RELATIVE PERIODS
        # ═══════════════════════════════════════════════════════════

        if tag == "last_week":
            start = now.start_of("week") - timedelta(weeks=1)
            return self._format_week_range(start)

        if tag == "this_week":
            start = now.start_of("week")
            return self._format_week_range(start)

        if tag == "next_week":
            start = now.start_of("week") + timedelta(weeks=1)
            return self._format_week_range(start)

        if tag == "last_month":
            dt = now - timedelta(days=now.day)
            dt = dt.start_of("month")
            return self._format_month_range(dt)

        if tag == "this_month":
            return self._format_month_range(now)

        if tag == "next_month":
            dt = now.start_of("month") + timedelta(days=32)
            dt = dt.start_of("month")
            return self._format_month_range(dt)

        if tag == "last_year":
            dt = now.set(year=now.year - 1)
            return self._format_year_range(dt)

        if tag == "this_year":
            return self._format_year_range(now)

        if tag == "next_year":
            dt = now.set(year=now.year + 1)
            return self._format_year_range(dt)

        # ═══════════════════════════════════════════════════════════
        # STANDALONE DAY PARTS
        # ═══════════════════════════════════════════════════════════

        if tag == "now_datetime":
            return self._format_datetime(now)

        if tag == "this_morning_datetime":
            return self._format_datetime(now.set(hour=9, minute=0, second=0, microsecond=0))

        if tag == "this_afternoon_datetime":
            return self._format_datetime(now.set(hour=15, minute=0, second=0, microsecond=0))

        if tag == "this_evening_datetime":
            return self._format_datetime(now.set(hour=20, minute=0, second=0, microsecond=0))

        if tag == "later_today":
            return self._format_datetime(now + timedelta(hours=3))

        if tag == "earlier_today":
            return self._format_datetime(now - timedelta(hours=3))

        if tag == "noon_datetime":
            return self._format_datetime(now.set(hour=12, minute=0, second=0, microsecond=0))

        if tag == "midnight_datetime":
            return self._format_datetime(now.set(hour=0, minute=0, second=0, microsecond=0))

        if tag == "morning_standalone":
            return self._format_datetime(now.set(hour=9, minute=0, second=0, microsecond=0))

        if tag == "afternoon_standalone":
            return self._format_datetime(now.set(hour=15, minute=0, second=0, microsecond=0))

        if tag == "evening_standalone":
            return self._format_datetime(now.set(hour=20, minute=0, second=0, microsecond=0))

        if tag == "tonight_datetime":
            return self._format_datetime(now.set(hour=22, minute=0, second=0, microsecond=0))

        # ═══════════════════════════════════════════════════════════
        # BASIC RELATIVE DAYS
        # ═══════════════════════════════════════════════════════════

        if tag == "today":
            return self._format_date(now)

        if tag == "yesterday":
            return self._format_date(now - timedelta(days=1))

        if tag == "tomorrow":
            return self._format_date(now + timedelta(days=1))

        return None

    # ── Message processing ─────────────────────────────────────────

    def process_message(self, message_content: str) -> str:
        if not self.config.enabled:
            return message_content

        now = pendulum.now(self.default_tz)
        processed = message_content

        # Patterns are already sorted longest-first in rules.py.
        # We iterate all compiled patterns and replace each match with its
        # resolved value, replacing right-to-left to preserve indices.
        for compiled, tag in COMPILED_TEMPORAL_PATTERNS:
            matches = list(compiled.finditer(processed))
            if not matches:
                continue
            for match in reversed(matches):
                resolved = self._resolve_match(match, tag, now)
                if resolved:
                    processed = (
                        processed[: match.start()]
                        + resolved
                        + processed[match.end() :]
                    )

        return processed

    def process_messages(self, messages: list[dict]) -> list[dict]:
        if not self.config.enabled:
            return messages

        processed_messages = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if role in ("system", "user") and isinstance(content, str):
                message["content"] = self.process_message(content)
            elif role in ("system", "user") and isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        block["text"] = self.process_message(block["text"])

            processed_messages.append(message)
        return processed_messages
