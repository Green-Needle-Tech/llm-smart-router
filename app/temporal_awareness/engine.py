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



# ── Tag dispatch handlers ──────────────────────────────────────────
# Each handler takes (engine, groups, match, now) and returns str | None.

def _h_day_after_tomorrow(engine, groups, match, now):
    return engine._format_date(now + timedelta(days=2))

def _h_day_before_yesterday(engine, groups, match, now):
    return engine._format_date(now - timedelta(days=2))

def _h_following_period(engine, groups, match, now):
    unit = groups[0].lower() if groups and groups[0] else "day"
    if unit == "week":
        start = now.start_of("week") + timedelta(weeks=1)
        return engine._format_week_range(start)
    elif unit == "month":
        dt = now.start_of("month") + timedelta(days=32)
        return engine._format_month_range(dt.start_of("month"))
    return engine._format_date(now + timedelta(days=1))

def _h_fortnight_ago(engine, groups, match, now):
    return engine._format_date(now - timedelta(weeks=2))

def _h_fortnight_future(engine, groups, match, now):
    return engine._format_date(now + timedelta(weeks=2))

def _h_yesterday_morning(engine, groups, match, now):
    return engine._format_datetime((now - timedelta(days=1)).set(hour=9, minute=0, second=0, microsecond=0))

def _h_yesterday_afternoon(engine, groups, match, now):
    return engine._format_datetime((now - timedelta(days=1)).set(hour=15, minute=0, second=0, microsecond=0))

def _h_yesterday_evening(engine, groups, match, now):
    return engine._format_datetime((now - timedelta(days=1)).set(hour=20, minute=0, second=0, microsecond=0))

def _h_yesterday_night(engine, groups, match, now):
    return engine._format_datetime((now - timedelta(days=1)).set(hour=22, minute=0, second=0, microsecond=0))

def _h_last_night(engine, groups, match, now):
    return engine._format_datetime((now - timedelta(days=1)).set(hour=22, minute=0, second=0, microsecond=0))

def _h_tomorrow_morning(engine, groups, match, now):
    return engine._format_datetime((now + timedelta(days=1)).set(hour=9, minute=0, second=0, microsecond=0))

def _h_tomorrow_afternoon(engine, groups, match, now):
    return engine._format_datetime((now + timedelta(days=1)).set(hour=15, minute=0, second=0, microsecond=0))

def _h_tomorrow_evening(engine, groups, match, now):
    return engine._format_datetime((now + timedelta(days=1)).set(hour=20, minute=0, second=0, microsecond=0))

def _h_tomorrow_night(engine, groups, match, now):
    return engine._format_datetime((now + timedelta(days=1)).set(hour=22, minute=0, second=0, microsecond=0))

def _h_military_time(engine, groups, match, now):
    hh, mm = int(groups[0]), int(groups[1])
    return engine._format_datetime(now.set(hour=hh, minute=mm, second=0, microsecond=0))

def _h_military_time_hundred(engine, groups, match, now):
    hh = int(groups[0])
    if hh > 23:
        hh = hh % 24
    return engine._format_datetime(now.set(hour=hh, minute=0, second=0, microsecond=0))

def _h_quarter_past(engine, groups, match, now):
    hour = int(groups[0])
    ampm = groups[1] if len(groups) > 1 else None
    hour = engine._apply_ampm(hour, ampm)
    return engine._format_datetime(now.set(hour=hour, minute=15, second=0, microsecond=0))

def _h_quarter_to(engine, groups, match, now):
    hour = int(groups[0])
    ampm = groups[1] if len(groups) > 1 else None
    hour = engine._apply_ampm(hour, ampm)
    hour = (hour - 1) % 24
    return engine._format_datetime(now.set(hour=hour, minute=45, second=0, microsecond=0))

def _h_half_past(engine, groups, match, now):
    hour = int(groups[0])
    ampm = groups[1] if len(groups) > 1 else None
    hour = engine._apply_ampm(hour, ampm)
    return engine._format_datetime(now.set(hour=hour, minute=30, second=0, microsecond=0))

def _h_oclock(engine, groups, match, now):
    return engine._format_datetime(now.set(hour=int(groups[0]), minute=0, second=0, microsecond=0))

def _h_specific_time_datetime(engine, groups, match, now):
    if len(groups) == 3:
        hour, minute, ampm = int(groups[0]), int(groups[1]) if groups[1] else 0, groups[2]
    elif len(groups) == 2:
        hour, minute, ampm = int(groups[0]), 0, groups[1]
    else:
        return None
    hour = engine._apply_ampm(hour, ampm)
    return engine._format_datetime(now.set(hour=hour, minute=minute, second=0, microsecond=0))

def _h_cob_eod(engine, groups, match, now):
    return engine._format_datetime(now.set(hour=17, minute=0, second=0, microsecond=0))

def _h_end_of_day(engine, groups, match, now):
    return engine._format_datetime(now.set(hour=23, minute=59, second=59, microsecond=0))

def _h_end_of_period(engine, groups, match, now):
    unit = groups[0].lower()
    if unit == "week":
        return engine._format_date(now.end_of("week"))
    elif unit == "month":
        return engine._format_date(now.end_of("month"))
    return engine._format_date(now.end_of("year"))

def _h_start_of_period(engine, groups, match, now):
    unit = groups[0].lower()
    if unit == "week":
        return engine._format_date(now.start_of("week"))
    elif unit == "month":
        return engine._format_date(now.start_of("month"))
    return engine._format_date(now.start_of("year"))

def _h_eow(engine, groups, match, now):
    return engine._format_date(now.end_of("week"))

def _h_eom(engine, groups, match, now):
    return engine._format_date(now.end_of("month"))

def _h_eoy(engine, groups, match, now):
    return engine._format_date(now.end_of("year"))

def _h_month_end(engine, groups, match, now):
    return engine._format_date(now.end_of("month"))

def _h_year_end(engine, groups, match, now):
    return engine._format_date(now.end_of("year"))

def _h_first_thing_tomorrow(engine, groups, match, now):
    return engine._format_datetime((now + timedelta(days=1)).set(hour=8, minute=0, second=0, microsecond=0))

def _h_first_thing_morning(engine, groups, match, now):
    return engine._format_datetime(now.set(hour=8, minute=0, second=0, microsecond=0))

def _h_lunchtime(engine, groups, match, now):
    return engine._format_datetime(now.set(hour=12, minute=30, second=0, microsecond=0))

def _h_dinnertime(engine, groups, match, now):
    return engine._format_datetime(now.set(hour=19, minute=0, second=0, microsecond=0))

def _h_teatime(engine, groups, match, now):
    return engine._format_datetime(now.set(hour=16, minute=0, second=0, microsecond=0))

def _h_breakfast_time(engine, groups, match, now):
    return engine._format_datetime(now.set(hour=8, minute=0, second=0, microsecond=0))

def _h_midweek(engine, groups, match, now):
    return engine._format_date(now.start_of("week") + timedelta(days=2))

def _h_the_other_day(engine, groups, match, now):
    return engine._format_date(now - timedelta(days=2))

def _h_a_long_time_ago(engine, groups, match, now):
    return engine._format_date(now - timedelta(days=365))

def _h_a_while_ago(engine, groups, match, now):
    return engine._format_date(now - timedelta(days=14))

def _h_in_a_bit(engine, groups, match, now):
    return engine._format_datetime(now + timedelta(hours=1))

def _h_shortly(engine, groups, match, now):
    return engine._format_datetime(now + timedelta(minutes=15))

def _h_soon(engine, groups, match, now):
    return engine._format_datetime(now + timedelta(hours=1))

def _h_couple_units_ago(engine, groups, match, now):
    return engine._format_date(now - timedelta(days=2))

def _h_few_units_ago(engine, groups, match, now):
    delta = engine._unit_to_timedelta(groups[0].lower().rstrip("s"), 3)
    return engine._format_date(now - delta)

def _h_few_units_from_now(engine, groups, match, now):
    delta = engine._unit_to_timedelta(groups[0].lower().rstrip("s"), 3)
    return engine._format_date(now + delta)

def _resolve_n_units(engine, groups, now, sign):
    n, unit = int(groups[0]), groups[1]
    delta = engine._unit_to_timedelta(unit, n)
    if unit.lower().rstrip("s") in ("hour", "hr", "minute", "min", "second", "sec"):
        return engine._format_datetime(now + sign * delta)
    return engine._format_date(now + sign * delta)

def _h_past_n_units(engine, groups, match, now):
    return _resolve_n_units(engine, groups, now, -1)

def _h_n_units_ago(engine, groups, match, now):
    return _resolve_n_units(engine, groups, now, -1)

def _h_n_units_from_now(engine, groups, match, now):
    return _resolve_n_units(engine, groups, now, 1)

def _h_in_n_units(engine, groups, match, now):
    return _resolve_n_units(engine, groups, now, 1)

def _h_a_unit_ago(engine, groups, match, now):
    return engine._format_date(now - engine._unit_to_timedelta(groups[0], 1))

def _h_a_unit_from_now(engine, groups, match, now):
    return engine._format_date(now + engine._unit_to_timedelta(groups[0], 1))

def _h_relative_season(engine, groups, match, now):
    text = match.group(0).lower().split()
    return engine._resolve_season(text[1], now, text[0])

def _h_this_quarter(engine, groups, match, now):
    return engine._format_quarter_range(now, (now.month - 1) // 3 + 1)

def _h_last_quarter(engine, groups, match, now):
    q = (now.month - 1) // 3 + 1
    if q == 1:
        return engine._format_quarter_range(now.set(year=now.year - 1), 4)
    return engine._format_quarter_range(now, q - 1)

def _h_next_quarter(engine, groups, match, now):
    q = (now.month - 1) // 3 + 1
    if q == 4:
        return engine._format_quarter_range(now.set(year=now.year + 1), 1)
    return engine._format_quarter_range(now, q + 1)

def _h_quarter_n(engine, groups, match, now):
    return engine._format_quarter_range(now, int(groups[0]))

def _h_relative_decade(engine, groups, match, now):
    direction = match.group(0).lower().split()[0]
    if direction == "last":
        return f"{now.year - 10}..{now.year - 1}"
    elif direction == "next":
        return f"{now.year + 1}..{now.year + 10}"
    return f"{now.year // 10 * 10}..{now.year // 10 * 10 + 9}"

def _h_relative_day_of_week(engine, groups, match, now):
    direction = groups[0].lower()
    target_dow = engine._resolve_weekday(groups[1].lower())
    current_dow = now.day_of_week
    if direction in ("last", "previous"):
        diff = (current_dow - target_dow) % 7 or 7
        return engine._format_date(now - timedelta(days=diff))
    elif direction == "this":
        return engine._format_date(now.start_of("week") + timedelta(days=target_dow))
    diff = (target_dow - current_dow) % 7 or 7
    return engine._format_date(now + timedelta(days=diff))

def _h_coming_day_of_week(engine, groups, match, now):
    target_dow = engine._resolve_weekday(groups[0].lower())
    diff = (target_dow - now.day_of_week) % 7 or 7
    return engine._format_date(now + timedelta(days=diff))

def _h_on_weekday(engine, groups, match, now):
    target_dow = engine._resolve_weekday(groups[0].lower())
    diff = (target_dow - now.day_of_week) % 7
    return engine._format_date(now + timedelta(days=diff))

def _h_bare_weekday(engine, groups, match, now):
    target_dow = engine._resolve_weekday(groups[0].lower())
    diff = (target_dow - now.day_of_week) % 7
    if diff == 0:
        return engine._format_date(now)
    return engine._format_date(now + timedelta(days=diff))

def _h_this_weekend(engine, groups, match, now):
    start = now.start_of("week") + timedelta(days=5)
    return f"{start.to_date_string()}..{(start + timedelta(days=1)).to_date_string()}"

def _h_last_weekend(engine, groups, match, now):
    start = now.start_of("week") + timedelta(days=5) - timedelta(weeks=1)
    return f"{start.to_date_string()}..{(start + timedelta(days=1)).to_date_string()}"

def _h_next_weekend(engine, groups, match, now):
    start = now.start_of("week") + timedelta(days=5) + timedelta(weeks=1)
    return f"{start.to_date_string()}..{(start + timedelta(days=1)).to_date_string()}"

def _h_last_week(engine, groups, match, now):
    return engine._format_week_range(now.start_of("week") - timedelta(weeks=1))

def _h_this_week(engine, groups, match, now):
    return engine._format_week_range(now.start_of("week"))

def _h_next_week(engine, groups, match, now):
    return engine._format_week_range(now.start_of("week") + timedelta(weeks=1))

def _h_last_month(engine, groups, match, now):
    return engine._format_month_range((now - timedelta(days=now.day)).start_of("month"))

def _h_this_month(engine, groups, match, now):
    return engine._format_month_range(now)

def _h_next_month(engine, groups, match, now):
    return engine._format_month_range((now.start_of("month") + timedelta(days=32)).start_of("month"))

def _h_last_year(engine, groups, match, now):
    return engine._format_year_range(now.set(year=now.year - 1))

def _h_this_year(engine, groups, match, now):
    return engine._format_year_range(now)

def _h_next_year(engine, groups, match, now):
    return engine._format_year_range(now.set(year=now.year + 1))

def _h_now_datetime(engine, groups, match, now):
    return engine._format_datetime(now)

def _h_this_morning_datetime(engine, groups, match, now):
    return engine._format_datetime(now.set(hour=9, minute=0, second=0, microsecond=0))

def _h_this_afternoon_datetime(engine, groups, match, now):
    return engine._format_datetime(now.set(hour=15, minute=0, second=0, microsecond=0))

def _h_this_evening_datetime(engine, groups, match, now):
    return engine._format_datetime(now.set(hour=20, minute=0, second=0, microsecond=0))

def _h_later_today(engine, groups, match, now):
    return engine._format_datetime(now + timedelta(hours=3))

def _h_earlier_today(engine, groups, match, now):
    return engine._format_datetime(now - timedelta(hours=3))

def _h_noon_datetime(engine, groups, match, now):
    return engine._format_datetime(now.set(hour=12, minute=0, second=0, microsecond=0))

def _h_midnight_datetime(engine, groups, match, now):
    return engine._format_datetime(now.set(hour=0, minute=0, second=0, microsecond=0))

def _h_morning_standalone(engine, groups, match, now):
    return engine._format_datetime(now.set(hour=9, minute=0, second=0, microsecond=0))

def _h_afternoon_standalone(engine, groups, match, now):
    return engine._format_datetime(now.set(hour=15, minute=0, second=0, microsecond=0))

def _h_evening_standalone(engine, groups, match, now):
    return engine._format_datetime(now.set(hour=20, minute=0, second=0, microsecond=0))

def _h_tonight_datetime(engine, groups, match, now):
    return engine._format_datetime(now.set(hour=22, minute=0, second=0, microsecond=0))

def _h_today(engine, groups, match, now):
    return engine._format_date(now)

def _h_yesterday(engine, groups, match, now):
    return engine._format_date(now - timedelta(days=1))

def _h_tomorrow(engine, groups, match, now):
    return engine._format_date(now + timedelta(days=1))


_TAG_DISPATCH: dict[str, object] = {
    "day_after_tomorrow": _h_day_after_tomorrow,
    "day_before_yesterday": _h_day_before_yesterday,
    "following_period": _h_following_period,
    "fortnight_ago": _h_fortnight_ago,
    "in_fortnight": _h_fortnight_future,
    "fortnight_from_now": _h_fortnight_future,
    "yesterday_morning": _h_yesterday_morning,
    "yesterday_afternoon": _h_yesterday_afternoon,
    "yesterday_evening": _h_yesterday_evening,
    "yesterday_night": _h_yesterday_night,
    "last_night": _h_last_night,
    "tomorrow_morning": _h_tomorrow_morning,
    "tomorrow_afternoon": _h_tomorrow_afternoon,
    "tomorrow_evening": _h_tomorrow_evening,
    "tomorrow_night": _h_tomorrow_night,
    "military_time": _h_military_time,
    "military_time_hundred": _h_military_time_hundred,
    "quarter_past": _h_quarter_past,
    "quarter_to": _h_quarter_to,
    "half_past": _h_half_past,
    "oclock": _h_oclock,
    "specific_time_datetime": _h_specific_time_datetime,
    "cob": _h_cob_eod, "eod": _h_cob_eod,
    "end_of_day": _h_end_of_day,
    "end_of_period": _h_end_of_period,
    "start_of_period": _h_start_of_period,
    "eow": _h_eow, "eom": _h_eom, "eoy": _h_eoy,
    "month_end": _h_month_end, "year_end": _h_year_end,
    "first_thing_tomorrow": _h_first_thing_tomorrow,
    "first_thing_morning": _h_first_thing_morning,
    "lunchtime": _h_lunchtime, "dinnertime": _h_dinnertime,
    "teatime": _h_teatime, "breakfast_time": _h_breakfast_time,
    "midweek": _h_midweek,
    "the_other_day": _h_the_other_day,
    "a_long_time_ago": _h_a_long_time_ago,
    "a_while_ago": _h_a_while_ago,
    "in_a_bit": _h_in_a_bit, "shortly": _h_shortly, "soon": _h_soon,
    "couple_units_ago": _h_couple_units_ago,
    "few_units_ago": _h_few_units_ago,
    "few_units_from_now": _h_few_units_from_now,
    "past_n_units": _h_past_n_units,
    "n_units_ago": _h_n_units_ago,
    "n_units_from_now": _h_n_units_from_now,
    "in_n_units": _h_in_n_units,
    "a_unit_ago": _h_a_unit_ago, "a_unit_from_now": _h_a_unit_from_now,
    "relative_season": _h_relative_season,
    "this_quarter": _h_this_quarter,
    "last_quarter": _h_last_quarter,
    "next_quarter": _h_next_quarter,
    "quarter_n": _h_quarter_n,
    "relative_decade": _h_relative_decade,
    "relative_day_of_week": _h_relative_day_of_week,
    "coming_day_of_week": _h_coming_day_of_week,
    "on_weekday": _h_on_weekday, "bare_weekday": _h_bare_weekday,
    "this_weekend": _h_this_weekend,
    "last_weekend": _h_last_weekend,
    "next_weekend": _h_next_weekend,
    "last_week": _h_last_week, "this_week": _h_this_week, "next_week": _h_next_week,
    "last_month": _h_last_month, "this_month": _h_this_month, "next_month": _h_next_month,
    "last_year": _h_last_year, "this_year": _h_this_year, "next_year": _h_next_year,
    "now_datetime": _h_now_datetime,
    "this_morning_datetime": _h_this_morning_datetime,
    "this_afternoon_datetime": _h_this_afternoon_datetime,
    "this_evening_datetime": _h_this_evening_datetime,
    "later_today": _h_later_today, "earlier_today": _h_earlier_today,
    "noon_datetime": _h_noon_datetime,
    "midnight_datetime": _h_midnight_datetime,
    "morning_standalone": _h_morning_standalone,
    "afternoon_standalone": _h_afternoon_standalone,
    "evening_standalone": _h_evening_standalone,
    "tonight_datetime": _h_tonight_datetime,
    "today": _h_today, "yesterday": _h_yesterday, "tomorrow": _h_tomorrow,
}


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
        handler = _TAG_DISPATCH.get(tag)
        if handler is None:
            return None
        return handler(self, groups, match, now)

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
