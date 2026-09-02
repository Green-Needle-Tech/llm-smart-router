"""Comprehensive unit tests for temporal awareness engine.

Tests all pattern categories including typos, abbreviations, colloquial
expressions, military time, seasons, quarters, and edge cases.
"""
import re

import pendulum

from app.config.schema import TemporalAwarenessConfig
from app.temporal_awareness.engine import TemporalAwarenessEngine


def _make_engine(tz="Asia/Singapore"):
    cfg = TemporalAwarenessConfig(enabled=True, default_timezone=tz)
    return TemporalAwarenessEngine(cfg)


def _now(tz="Asia/Singapore"):
    return pendulum.now(tz)


# ── Basic days ──────────────────────────────────────────────────────

def test_today():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Today is nice.")
    assert now.to_date_string() in result

def test_yesterday():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Yesterday was busy.")
    assert (now - pendulum.duration(days=1)).to_date_string() in result

def test_tomorrow():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Tomorrow is the deadline.")
    assert (now + pendulum.duration(days=1)).to_date_string() in result


# ── Typo tolerance for basic days ───────────────────────────────────

def test_tomorrow_typos():
    engine = _make_engine()
    now = _now()
    expected = (now + pendulum.duration(days=1)).to_date_string()
    for typo in ["tomorow", "tomoro", "tomorro", "tomorroww"]:
        result = engine.process_message(f"{typo} is the day.")
        assert expected in result, f"Failed for typo '{typo}': {result}"

def test_yesterday_typos():
    engine = _make_engine()
    now = _now()
    expected = (now - pendulum.duration(days=1)).to_date_string()
    for typo in ["yesteday", "yesturday", "yestreday"]:
        result = engine.process_message(f"{typo} was busy.")
        assert expected in result, f"Failed for typo '{typo}': {result}"

def test_abbreviations():
    engine = _make_engine()
    now = _now()
    # tmrw, tmr, 2mrw, 2morrow
    result = engine.process_message("tmrw is the day.")
    assert (now + pendulum.duration(days=1)).to_date_string() in result
    result = engine.process_message("2day is the day.")
    assert now.to_date_string() in result
    result = engine.process_message("yday was busy.")
    assert (now - pendulum.duration(days=1)).to_date_string() in result


# ── Compound days ───────────────────────────────────────────────────

def test_day_after_tomorrow():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("The day after tomorrow we launch.")
    expected = (now + pendulum.duration(days=2)).to_date_string()
    assert expected in result, f"Expected {expected}, got: {result}"

def test_day_before_yesterday():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("The day before yesterday I sent it.")
    expected = (now - pendulum.duration(days=2)).to_date_string()
    assert expected in result

def test_overmorrow():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("overmorrow we launch.")
    expected = (now + pendulum.duration(days=2)).to_date_string()
    assert expected in result


# ── Day parts ───────────────────────────────────────────────────────

def test_now_datetime():
    engine = _make_engine()
    result = engine.process_message("What is happening now?")
    assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", result)

def test_this_morning():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Let us meet this morning.")
    expected = now.set(hour=9, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result

def test_this_afternoon():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("See you this afternoon.")
    expected = now.set(hour=15, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result

def test_this_evening():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Dinner this evening.")
    expected = now.set(hour=20, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result

def test_tonight():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Call me tonight.")
    expected = now.set(hour=22, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result

def test_tonight_typos():
    engine = _make_engine()
    now = _now()
    expected = now.set(hour=22, minute=0, second=0, microsecond=0).isoformat()
    for typo in ["tonite", "tonigt", "2nite"]:
        result = engine.process_message(f"Call me {typo}.")
        assert expected in result, f"Failed for '{typo}': {result}"

def test_noon():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Meet at noon.")
    expected = now.set(hour=12, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result

def test_midnight():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("At midnight.")
    expected = now.set(hour=0, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result

def test_midday():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("At midday.")
    expected = now.set(hour=12, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result


# ── Relative day parts ──────────────────────────────────────────────

def test_yesterday_morning():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("yesterday morning was quiet.")
    expected = (now - pendulum.duration(days=1)).set(hour=9, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result

def test_last_night():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("last night was fun.")
    expected = (now - pendulum.duration(days=1)).set(hour=22, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result

def test_tomorrow_evening():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("tomorrow evening we meet.")
    expected = (now + pendulum.duration(days=1)).set(hour=20, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result


# ── Specific times ──────────────────────────────────────────────────

def test_at_3pm():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Meet at 3pm.")
    expected = now.set(hour=15, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result

def test_by_530_pm():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("By 5:30 PM please.")
    expected = now.set(hour=17, minute=30, second=0, microsecond=0).isoformat()
    assert expected in result

def test_at_915_am():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("The event is at 9:15 AM.")
    expected = now.set(hour=9, minute=15, second=0, microsecond=0).isoformat()
    assert expected in result

def test_12pm_noon():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Lunch at 12pm.")
    expected = now.set(hour=12, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result

def test_12am_midnight():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Midnight at 12am.")
    expected = now.set(hour=0, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result

def test_ampm_with_periods():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Meet at 3 p.m.")
    expected = now.set(hour=15, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result

def test_oclock():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("See you at 3 o'clock.")
    expected = now.set(hour=3, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result

def test_oclock_no_apostrophe():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("See you at 3 o clock.")
    expected = now.set(hour=3, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result

def test_quarter_past():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Meet at quarter past 3.")
    expected = now.set(hour=3, minute=15, second=0, microsecond=0).isoformat()
    assert expected in result

def test_quarter_to():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Meet at quarter to 4.")
    # quarter to 4 = 3:45
    expected = now.set(hour=3, minute=45, second=0, microsecond=0).isoformat()
    assert expected in result

def test_half_past():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Meet at half past 5.")
    expected = now.set(hour=5, minute=30, second=0, microsecond=0).isoformat()
    assert expected in result


# ── Military time ───────────────────────────────────────────────────

def test_military_time():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Meet at 1430 hours.")
    expected = now.set(hour=14, minute=30, second=0, microsecond=0).isoformat()
    assert expected in result

def test_military_time_hundred():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Meet at 14 hundred hours.")
    expected = now.set(hour=14, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result


# ── Days of week ────────────────────────────────────────────────────

def test_next_monday():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Next Monday we meet.")
    # Should contain a date
    assert re.search(r"\d{4}-\d{2}-\d{2}", result)
    assert "Monday" not in result

def test_last_friday():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Last Friday was busy.")
    assert re.search(r"\d{4}-\d{2}-\d{2}", result)

def test_weekday_abbreviations():
    engine = _make_engine()
    result = engine.process_message("Next Mon we meet.")
    assert re.search(r"\d{4}-\d{2}-\d{2}", result)
    result = engine.process_message("Last Fri was busy.")
    assert re.search(r"\d{4}-\d{2}-\d{2}", result)


# ── Relative periods ────────────────────────────────────────────────

def test_last_week():
    engine = _make_engine()
    result = engine.process_message("Last week was busy.")
    assert re.search(r"\d{4}-\d{2}-\d{2}\.\.\d{4}-\d{2}-\d{2}", result)

def test_this_month():
    engine = _make_engine()
    result = engine.process_message("This month is busy.")
    assert re.search(r"\d{4}-\d{2}-\d{2}\.\.\d{4}-\d{2}-\d{2}", result)

def test_next_year():
    engine = _make_engine()
    result = engine.process_message("Next year will be great.")
    assert re.search(r"\d{4}-\d{2}-\d{2}\.\.\d{4}-\d{2}-\d{2}", result)


# ── N units ─────────────────────────────────────────────────────────

def test_3_days_ago():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("3 days ago I sent it.")
    expected = (now - pendulum.duration(days=3)).to_date_string()
    assert expected in result

def test_in_2_weeks():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("In 2 weeks we launch.")
    expected = (now + pendulum.duration(weeks=2)).to_date_string()
    assert expected in result

def test_5_hours_ago():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("5 hours ago it happened.")
    # Should contain a datetime (not just a date)
    assert re.search(r"\d{4}-\d{2}-\d{2}T", result)

def test_unit_typos():
    engine = _make_engine()
    now = _now()
    # "dys" instead of "days"
    result = engine.process_message("3 dys ago I sent it.")
    expected = (now - pendulum.duration(days=3)).to_date_string()
    assert expected in result

def test_n_units_back():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("3 days back I sent it.")
    expected = (now - pendulum.duration(days=3)).to_date_string()
    assert expected in result

def test_n_units_hence():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("2 weeks hence we launch.")
    expected = (now + pendulum.duration(weeks=2)).to_date_string()
    assert expected in result


# ── A/an unit ───────────────────────────────────────────────────────

def test_a_day_ago():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("a day ago it happened.")
    expected = (now - pendulum.duration(days=1)).to_date_string()
    assert expected in result

def test_a_week_from_now():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("a week from now we meet.")
    expected = (now + pendulum.duration(weeks=1)).to_date_string()
    assert expected in result


# ── Couple / few ────────────────────────────────────────────────────

def test_couple_days_ago():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("a couple of days ago.")
    expected = (now - pendulum.duration(days=2)).to_date_string()
    assert expected in result

def test_few_weeks_ago():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("a few weeks ago.")
    expected = (now - pendulum.duration(weeks=3)).to_date_string()
    assert expected in result


# ── Fortnight ───────────────────────────────────────────────────────

def test_fortnight_ago():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("a fortnight ago.")
    expected = (now - pendulum.duration(weeks=2)).to_date_string()
    assert expected in result

def test_in_fortnight():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("in a fortnight.")
    expected = (now + pendulum.duration(weeks=2)).to_date_string()
    assert expected in result


# ── Colloquial ──────────────────────────────────────────────────────

def test_the_other_day():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("the other day I saw him.")
    expected = (now - pendulum.duration(days=2)).to_date_string()
    assert expected in result

def test_a_while_ago():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("a while ago.")
    expected = (now - pendulum.duration(days=14)).to_date_string()
    assert expected in result

def test_in_a_bit():
    engine = _make_engine()
    result = engine.process_message("I will be there in a bit.")
    assert re.search(r"\d{4}-\d{2}-\d{2}T", result)

def test_soon():
    engine = _make_engine()
    result = engine.process_message("I will do it soon.")
    assert re.search(r"\d{4}-\d{2}-\d{2}T", result)


# ── End / beginning of period ───────────────────────────────────────

def test_eod():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Submit by EOD.")
    expected = now.set(hour=17, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result

def test_cob():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Submit by COB.")
    expected = now.set(hour=17, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result

def test_end_of_week():
    engine = _make_engine()
    result = engine.process_message("end of the week.")
    assert re.search(r"\d{4}-\d{2}-\d{2}", result)

def test_eom():
    engine = _make_engine()
    result = engine.process_message("By EOM.")
    assert re.search(r"\d{4}-\d{2}-\d{2}", result)

def test_eoy():
    engine = _make_engine()
    result = engine.process_message("By EOY.")
    assert re.search(r"\d{4}-\d{2}-\d{2}", result)

def test_month_end():
    engine = _make_engine()
    result = engine.process_message("month-end.")
    assert re.search(r"\d{4}-\d{2}-\d{2}", result)

def test_year_end():
    engine = _make_engine()
    result = engine.process_message("year-end.")
    assert re.search(r"\d{4}-\d{2}-\d{2}", result)

def test_start_of_month():
    engine = _make_engine()
    result = engine.process_message("beginning of the month.")
    assert re.search(r"\d{4}-\d{2}-\d{2}", result)


# ── Meal times / first thing ────────────────────────────────────────

def test_lunchtime():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("lunchtime.")
    expected = now.set(hour=12, minute=30, second=0, microsecond=0).isoformat()
    assert expected in result

def test_dinnertime():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("dinnertime.")
    expected = now.set(hour=19, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result

def test_first_thing_morning():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("first thing in the morning.")
    expected = now.set(hour=8, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result


# ── Weekend ─────────────────────────────────────────────────────────

def test_this_weekend():
    engine = _make_engine()
    result = engine.process_message("this weekend.")
    assert re.search(r"\d{4}-\d{2}-\d{2}\.\.\d{4}-\d{2}-\d{2}", result)

def test_next_weekend():
    engine = _make_engine()
    result = engine.process_message("next weekend.")
    assert re.search(r"\d{4}-\d{2}-\d{2}\.\.\d{4}-\d{2}-\d{2}", result)


# ── Seasons ─────────────────────────────────────────────────────────

def test_this_summer():
    engine = _make_engine()
    result = engine.process_message("this summer.")
    assert re.search(r"\d{4}-\d{2}-\d{2}\.\.\d{4}-\d{2}-\d{2}", result)

def test_next_winter():
    engine = _make_engine()
    result = engine.process_message("next winter.")
    assert re.search(r"\d{4}-\d{2}-\d{2}\.\.\d{4}-\d{2}-\d{2}", result)


# ── Quarters ────────────────────────────────────────────────────────

def test_this_quarter():
    engine = _make_engine()
    result = engine.process_message("this quarter.")
    assert re.search(r"\d{4}-\d{2}-\d{2}\.\.\d{4}-\d{2}-\d{2}", result)

def test_q1():
    engine = _make_engine()
    result = engine.process_message("In Q1 we launch.")
    assert re.search(r"\d{4}-\d{2}-\d{2}\.\.\d{4}-\d{2}-\d{2}", result)


# ── Non-temporal text ───────────────────────────────────────────────

def test_non_temporal_unchanged():
    engine = _make_engine()
    original = "What is the capital of France?"
    result = engine.process_message(original)
    assert result == original

def test_numbers_not_replaced():
    engine = _make_engine()
    original = "There are 3 cats and 5 dogs."
    result = engine.process_message(original)
    # "3" and "5" should not be replaced (no temporal context)
    assert "3" in result and "5" in result


# ── Multiple expressions ────────────────────────────────────────────

def test_multiple_in_one_message():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Meet this morning and again tonight.")
    morning = now.set(hour=9, minute=0, second=0, microsecond=0).isoformat()
    night = now.set(hour=22, minute=0, second=0, microsecond=0).isoformat()
    assert morning in result
    assert night in result

def test_mixed_date_and_time():
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Today at 3pm we meet.")
    today = now.to_date_string()
    time = now.set(hour=15, minute=0, second=0, microsecond=0).isoformat()
    # "today" → date, "at 3pm" → datetime
    assert today in result or time in result


# ── Disabled engine ─────────────────────────────────────────────────

def test_disabled_passthrough():
    cfg = TemporalAwarenessConfig(enabled=False, default_timezone="Asia/Singapore")
    engine = TemporalAwarenessEngine(cfg)
    original = "Meet me now at 3pm tonight tomorrow."
    result = engine.process_message(original)
    assert result == original


# ── Timezone ────────────────────────────────────────────────────────

def test_timezone_applied():
    engine = _make_engine("Asia/Singapore")
    result = engine.process_message("What is happening now?")
    assert "+08:00" in result

def test_timezone_utc():
    engine = _make_engine("UTC")
    result = engine.process_message("What is happening now?")
    assert "+00:00" in result
