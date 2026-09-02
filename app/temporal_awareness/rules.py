"""
Comprehensive temporal expression patterns for the TemporalAwarenessEngine.

Coverage:
  - Relative days (today, yesterday, tomorrow + typos/abbreviations)
  - Compound days (day after tomorrow, day before yesterday, overmorrow)
  - Day parts (morning, afternoon, evening, night, noon, midnight, lunchtime, etc.)
  - Relative day parts (yesterday/tomorrow + morning/evening/night, last night)
  - Days of the week (last/next/this/coming + weekday, with abbreviations)
  - Relative periods (last/this/next + week/month/year/quarter/decade/weekend)
  - N units ago/from now/in/back/hence (with typo tolerance on units)
  - A/an unit ago/from now/hence
  - A couple/few units ago/from now
  - Fortnight expressions
  - Specific times (at/by H:MM AM/PM, o'clock, quarter past/to, half past)
  - Military time (HHMM hours, hundred hours)
  - Standalone times (noon, midnight, midday)
  - Seasons (this/last/next + spring/summer/fall/autumn/winter)
  - Quarters (this/last/next quarter, Q1-Q4)
  - End/beginning of period (EOD, EOW, EOM, EOY, COB, end/start of week/month/year)
  - Colloquial (the other day, a while ago, a long time ago, in a bit, soon, shortly)
  - Following period, midweek, month-end, year-end, first thing

Typo tolerance:
  - Repeated/missing letters (tomorrow→tomorow, yesterday→yesteday)
  - Common abbreviations (tmrw, 2day, yday, 2nite, EOD, COB)
  - Grammar variations (a/an, couple of/couple, o'clock/o clock)

Patterns are auto-sorted longest-first to minimize overlap issues.
"""

import re

# ── Weekday pattern (full names + common abbreviations) ──
_WD = r"(?:mon(?:day)?|tue(?:s(?:day)?)?|wed(?:nesday)?|thu(?:rs(?:day)?)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)"

# ── Unit pattern with typo tolerance ──
_UNIT = r"(days?|dys?|weeks?|wks?|months?|mnths?|years?|yrs?|hours?|hrs?|minutes?|mins?|seconds?|secs?)"

# ── AM/PM pattern (with periods: a.m., p.m.) ──
_AMPM = r"(?:am|pm|a\.m\.|p\.m\.)"

# Duplicated regex fragments extracted as constants
_LOOKAHEAD_BOUNDARY = r"(?=\s|[,.;!?]|$)"
_OPT_AMPM_BOUNDARY = r"))?\b"
_NUM_UNIT_PREFIX = r"\b(\d+)\s+"

TEMPORAL_EXPRESSION_PATTERNS = [

    # ═══════════════════════════════════════════════════════════════
    # COMPOUND RELATIVE DAYS (must come before basic days)
    # ═══════════════════════════════════════════════════════════════

    (r"\b(?:the\s+)?day\s+(?:after\s+)?tom+or+o?w*\b", "day_after_tomorrow"),
    (r"\b(?:the\s+)?day\s+before\s+yest(?:er|ur|red)?day\b", "day_before_yesterday"),
    (r"\bover+mor+ow\b", "day_after_tomorrow"),  # archaic: overmorrow
    (r"\bthe\s+following\s+(day|week|month)\b", "following_period"),

    # ═══════════════════════════════════════════════════════════════
    # FORTNIGHT
    # ═══════════════════════════════════════════════════════════════

    (r"\b(?:a\s+)?fortnig?ht\s+ago\b", "fortnight_ago"),
    (r"\bin\s+(?:a\s+)?fortnig?ht\b", "in_fortnight"),
    (r"\b(?:a\s+)?fortnig?ht\s+from\s+(?:now|today)\b", "fortnight_from_now"),

    # ═══════════════════════════════════════════════════════════════
    # RELATIVE DAY PARTS (yesterday/tomorrow + morning/evening/night)
    # Must come before basic yesterday/tomorrow and standalone day parts
    # ═══════════════════════════════════════════════════════════════

    (r"\byest(?:er|ur|red)?day\s+mor+nin+g?\b", "yesterday_morning"),
    (r"\byest(?:er|ur|red)?day\s+after+no+n+\b", "yesterday_afternoon"),
    (r"\byest(?:er|ur|red)?day\s+even+in+g?\b", "yesterday_evening"),
    (r"\byest(?:er|ur|red)?day\s+night\b", "yesterday_night"),
    (r"\blast\s+night\b", "last_night"),
    (r"\btom+or+o?w*\s+mor+nin+g?\b", "tomorrow_morning"),
    (r"\btom+or+o?w*\s+after+no+n+\b", "tomorrow_afternoon"),
    (r"\btom+or+o?w*\s+even+in+g?\b", "tomorrow_evening"),
    (r"\btom+or+o?w*\s+night\b", "tomorrow_night"),

    # ═══════════════════════════════════════════════════════════════
    # SPECIFIC TIMES (must come before generic time patterns)
    # ═══════════════════════════════════════════════════════════════

    # Military time: 0800 hours, 1430 hrs, 14 hundred hours
    (r"\b([01]\d|2[0-3])([0-5]\d)\s*(?:hours?|hrs?|h)\b", "military_time"),
    (r"\b(\d{1,2})\s*hundred\s*(?:hours?|hrs?|h)?\b", "military_time_hundred"),

    # Quarter past/to, half past
    (r"\b(?:a\s+)?quarter\s+past\s+(\d{1,2})(?:\s*(" + _AMPM + _OPT_AMPM_BOUNDARY, "quarter_past"),
    (r"\b(?:a\s+)?quarter\s+(?:to|till|of)\s+(\d{1,2})(?:\s*(" + _AMPM + _OPT_AMPM_BOUNDARY, "quarter_to"),
    (r"\bhalf\s+past\s+(\d{1,2})(?:\s*(" + _AMPM + _OPT_AMPM_BOUNDARY, "half_past"),

    # O'clock (3 o'clock, 3o'clock, 3 o clock)
    (r"\b(\d{1,2})\s*o['']?\s*clock\b", "oclock"),

    # At/by H:MM AM/PM (requires AM/PM, not followed by o'clock or hundred)
    (r"\b(?:at|by|@)\s+(\d{1,2})(?::(\d{2}))\s*(" + _AMPM + r")" + _LOOKAHEAD_BOUNDARY, "specific_time_datetime"),
    (r"\b(?:at|by|@)\s+(\d{1,2})\s*(" + _AMPM + r")" + _LOOKAHEAD_BOUNDARY, "specific_time_datetime"),

    # H:MM AM/PM or H AM/PM (standalone, requires AM/PM)
    (r"\b(\d{1,2})(?::(\d{2}))?\s*(" + _AMPM + r")" + _LOOKAHEAD_BOUNDARY, "specific_time_datetime"),

    # ═══════════════════════════════════════════════════════════════
    # END / BEGINNING / ABBREVIATIONS
    # ═══════════════════════════════════════════════════════════════

    (r"\bclose\s+of\s+business\b", "cob"),
    (r"\b(?:end|close)\s+of\s+(?:the\s+)?day\b", "end_of_day"),
    (r"\b(?:end|close)\s+of\s+(?:the\s+)?(week|month|year)\b", "end_of_period"),
    (r"\b(?:beginning|start)\s+of\s+(?:the\s+)?(week|month|year)\b", "start_of_period"),
    (r"\bEOD\b", "eod"),
    (r"\bEOW\b", "eow"),
    (r"\bEOM\b", "eom"),
    (r"\bEOY\b", "eoy"),
    (r"\bCOB\b", "cob"),
    (r"\bmonth-?end\b", "month_end"),
    (r"\byear-?end\b", "year_end"),

    # ═══════════════════════════════════════════════════════════════
    # FIRST THING / MEAL TIMES / MIDWEEK
    # ═══════════════════════════════════════════════════════════════

    (r"\bfirst\s+thing\s+tom+or+o?w*\b", "first_thing_tomorrow"),
    (r"\bfirst\s+thing(?:\s+in\s+the\s+morning)?\b", "first_thing_morning"),
    (r"\blunch\s*time\b", "lunchtime"),
    (r"\b(?:dinner|supper)\s*time\b", "dinnertime"),
    (r"\btea\s*time\b", "teatime"),
    (r"\bbreakfast\s*time\b", "breakfast_time"),
    (r"\bmid-?week\b", "midweek"),

    # ═══════════════════════════════════════════════════════════════
    # COLLOQUIAL EXPRESSIONS
    # ═══════════════════════════════════════════════════════════════

    (r"\bthe\s+other\s+day\b", "the_other_day"),
    (r"\b(?:a\s+)?long\s+time\s+ago\b", "a_long_time_ago"),
    (r"\b(?:a\s+)?while\s+ago\b", "a_while_ago"),
    (r"\bin\s+a\s+(?:bit|while)\b", "in_a_bit"),
    (r"\bshortly\b", "shortly"),
    (r"\bsoon\b", "soon"),

    # ═══════════════════════════════════════════════════════════════
    # COUPLE / FEW UNITS
    # ═══════════════════════════════════════════════════════════════

    (r"\b(?:a\s+)?couple\s+(?:of\s+)?(?:days?|weeks?)\s+ago\b", "couple_units_ago"),
    (r"\b(?:a\s+)?few\s+(days?|weeks?|months?)\s+ago\b", "few_units_ago"),
    (r"\b(?:a\s+)?few\s+(days?|weeks?|months?)\s+from\s+now\b", "few_units_from_now"),

    # ═══════════════════════════════════════════════════════════════
    # N UNITS AGO/FROM NOW/IN/BACK/HENCE
    # ═══════════════════════════════════════════════════════════════

    (r"\b(?:last|past)\s+(\d+)\s+" + _UNIT + r"\b", "past_n_units"),
    (_NUM_UNIT_PREFIX + _UNIT + r"\s+ago\b", "n_units_ago"),
    (_NUM_UNIT_PREFIX + _UNIT + r"\s+back\b", "n_units_ago"),
    (_NUM_UNIT_PREFIX + _UNIT + r"\s+from\s+(?:now|today)\b", "n_units_from_now"),
    (_NUM_UNIT_PREFIX + _UNIT + r"\s+hence\b", "n_units_from_now"),
    (r"\bin\s+(\d+)\s+" + _UNIT + r"\b", "in_n_units"),

    # ═══════════════════════════════════════════════════════════════
    # A/AN UNIT AGO/FROM NOW/HENCE
    # ═══════════════════════════════════════════════════════════════

    (r"\b(?:a|an)\s+(day|week|month|year|hour)\s+ago\b", "a_unit_ago"),
    (r"\b(?:a|an)\s+(day|week|month|year|hour)\s+from\s+(?:now|today)\b", "a_unit_from_now"),
    (r"\b(?:a|an)\s+(day|week|month|year|hour)\s+hence\b", "a_unit_from_now"),

    # ═══════════════════════════════════════════════════════════════
    # SEASONS
    # ═══════════════════════════════════════════════════════════════

    (r"\b(?:this|last|next)\s+(?:spring|summer|fall|autumn|winter)\b", "relative_season"),

    # ═══════════════════════════════════════════════════════════════
    # QUARTERS & DECADES
    # ═══════════════════════════════════════════════════════════════

    (r"\bthis\s+quarter\b", "this_quarter"),
    (r"\blast\s+quarter\b", "last_quarter"),
    (r"\bnext\s+quarter\b", "next_quarter"),
    (r"\b[Qq]([1-4])\b", "quarter_n"),
    (r"\b(?:this|last|next)\s+decade\b", "relative_decade"),

    # ═══════════════════════════════════════════════════════════════
    # DAYS OF THE WEEK
    # ═══════════════════════════════════════════════════════════════

    (r"\b(last|next|this|previous)\s+(" + _WD + r")\b", "relative_day_of_week"),
    (r"\b(?:coming|upcoming)\s+(" + _WD + r")\b", "coming_day_of_week"),
    (r"\bon\s+(" + _WD + r")\b", "on_weekday"),
    (r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", "bare_weekday"),

    # ═══════════════════════════════════════════════════════════════
    # WEEKEND
    # ═══════════════════════════════════════════════════════════════

    (r"\bthis\s+weekend\b", "this_weekend"),
    (r"\blast\s+weekend\b", "last_weekend"),
    (r"\bnext\s+weekend\b", "next_weekend"),

    # ═══════════════════════════════════════════════════════════════
    # RELATIVE PERIODS
    # ═══════════════════════════════════════════════════════════════

    (r"\blast\s+week\b", "last_week"),
    (r"\bthis\s+week\b", "this_week"),
    (r"\bnext\s+week\b", "next_week"),
    (r"\blast\s+month\b", "last_month"),
    (r"\bthis\s+month\b", "this_month"),
    (r"\bnext\s+month\b", "next_month"),
    (r"\blast\s+year\b", "last_year"),
    (r"\bthis\s+year\b", "this_year"),
    (r"\bnext\s+year\b", "next_year"),

    # ═══════════════════════════════════════════════════════════════
    # STANDALONE DAY PARTS
    # ═══════════════════════════════════════════════════════════════

    (r"\b(?:right\s+now|just\s+now|at\s+this\s+very\s+moment)\b", "now_datetime"),
    (r"\bthis\s+mor+nin+g?\b", "this_morning_datetime"),
    (r"\bthis\s+after+no+n+\b", "this_afternoon_datetime"),
    (r"\bthis\s+even+in+g?\b", "this_evening_datetime"),
    (r"\blater\s+(?:today|tonight)\b", "later_today"),
    (r"\bearlier\s+(?:today|tonight)\b", "earlier_today"),
    (r"\b(?:noon|midday|mid-day)\b", "noon_datetime"),
    (r"\bmidnight\b", "midnight_datetime"),
    (r"\b(?:mor+nin+g?)\b", "morning_standalone"),
    (r"\b(?:after+no+n+)\b", "afternoon_standalone"),
    (r"\b(?:even+in+g?)\b", "evening_standalone"),

    # ═══════════════════════════════════════════════════════════════
    # TONIGHT (with typo tolerance)
    # ═══════════════════════════════════════════════════════════════

    (r"\bton(?:ight|ite|igt|igh)\b", "tonight_datetime"),  # tonight, tonite, tonigt, tonigh
    (r"\b2nite\b", "tonight_datetime"),

    # ═══════════════════════════════════════════════════════════════
    # NOW
    # ═══════════════════════════════════════════════════════════════

    (r"\bnow\b", "now_datetime"),

    # ═══════════════════════════════════════════════════════════════
    # BASIC RELATIVE DAYS (with typo tolerance) — near end
    # ═══════════════════════════════════════════════════════════════

    (r"\btom+or+o?w*\b", "tomorrow"),  # tomorrow, tomorow, tomoro, tomorro
    (r"\b(?:tmrw|tmr|2mrw|2morrow)\b", "tomorrow"),  # abbreviations
    (r"\byest(?:er|ur|red|re|e|r|a)?day\b", "yesterday"),  # yesterday, yesteday, yesturday, yestreday, yestarday
    (r"\byday\b", "yesterday"),  # abbreviation
    (r"\btodays\b", "today"),  # plural used as singular
    (r"\b2day\b", "today"),  # abbreviation
    (r"\btoday\b", "today"),
]

# Auto-sort by pattern length descending (longest first) to minimize overlap
TEMPORAL_EXPRESSION_PATTERNS.sort(key=lambda x: len(x[0]), reverse=True)

# Compile all patterns for efficiency
COMPILED_TEMPORAL_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), tag) for pattern, tag in TEMPORAL_EXPRESSION_PATTERNS
]
