"""Unit tests for time-awareness patterns in TemporalAwarenessEngine.

Tests that time expressions (now, this morning, this afternoon, this evening,
tonight, at H AM/PM, H:MM AM/PM) are resolved to concrete ISO datetimes.
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


def test_now_resolves_to_datetime():
    """'now' should be replaced with a full ISO datetime, not just a date."""
    engine = _make_engine()
    result = engine.process_message("What is happening now?")
    assert "now" not in result.lower()
    # Should contain an ISO datetime pattern (YYYY-MM-DDTHH:MM:SS+TZ)
    assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", result), f"Expected ISO datetime, got: {result}"


def test_this_morning_resolves():
    """'this morning' should resolve to 09:00 of today."""
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Let us meet this morning.")
    expected = now.set(hour=9, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result, f"Expected '{expected}' in result, got: {result}"


def test_this_afternoon_resolves():
    """'this afternoon' should resolve to 15:00 of today."""
    engine = _make_engine()
    now = _now()
    result = engine.process_message("See you this afternoon.")
    expected = now.set(hour=15, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result, f"Expected '{expected}' in result, got: {result}"


def test_this_evening_resolves():
    """'this evening' should resolve to 20:00 of today."""
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Dinner this evening.")
    expected = now.set(hour=20, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result, f"Expected '{expected}' in result, got: {result}"


def test_tonight_resolves():
    """'tonight' should resolve to 22:00 of today."""
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Call me tonight.")
    expected = now.set(hour=22, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result, f"Expected '{expected}' in result, got: {result}"


def test_at_3pm_resolves():
    """'at 3pm' should resolve to 15:00 of today."""
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Meet at 3pm.")
    expected = now.set(hour=15, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result, f"Expected '{expected}' in result, got: {result}"


def test_by_530_pm_resolves():
    """'by 5:30 PM' should resolve to 17:30 of today."""
    engine = _make_engine()
    now = _now()
    result = engine.process_message("By 5:30 PM please.")
    expected = now.set(hour=17, minute=30, second=0, microsecond=0).isoformat()
    assert expected in result, f"Expected '{expected}' in result, got: {result}"


def test_at_915_am_resolves():
    """'at 9:15 AM' should resolve to 09:15 of today."""
    engine = _make_engine()
    now = _now()
    result = engine.process_message("The event is at 9:15 AM.")
    expected = now.set(hour=9, minute=15, second=0, microsecond=0).isoformat()
    assert expected in result, f"Expected '{expected}' in result, got: {result}"


def test_12pm_resolves_to_noon():
    """'12pm' should resolve to 12:00 (noon) of today."""
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Lunch at 12pm.")
    expected = now.set(hour=12, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result, f"Expected '{expected}' in result, got: {result}"


def test_12am_resolves_to_midnight():
    """'12am' should resolve to 00:00 (midnight) of today."""
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Midnight at 12am.")
    expected = now.set(hour=0, minute=0, second=0, microsecond=0).isoformat()
    assert expected in result, f"Expected '{expected}' in result, got: {result}"


def test_date_patterns_still_work():
    """Existing date-only patterns should still resolve to dates, not datetimes."""
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Today is a good day.")
    expected = now.to_date_string()
    assert expected in result, f"Expected date '{expected}' in result, got: {result}"
    # Should NOT contain a 'T' (no time component)
    assert "T" not in result.split(expected)[1].split()[0] if expected in result else True


def test_non_temporal_text_unchanged():
    """Non-temporal text should pass through unchanged."""
    engine = _make_engine()
    original = "What is the capital of France?"
    result = engine.process_message(original)
    assert result == original, f"Expected unchanged, got: {result}"


def test_multiple_time_expressions():
    """Multiple time expressions in one message should all be resolved."""
    engine = _make_engine()
    now = _now()
    result = engine.process_message("Meet this morning and again tonight.")
    morning = now.set(hour=9, minute=0, second=0, microsecond=0).isoformat()
    night = now.set(hour=22, minute=0, second=0, microsecond=0).isoformat()
    assert morning in result, f"Expected morning '{morning}' in result, got: {result}"
    assert night in result, f"Expected night '{night}' in result, got: {result}"


def test_disabled_engine_passthrough():
    """When disabled, the engine should return text unchanged."""
    cfg = TemporalAwarenessConfig(enabled=False, default_timezone="Asia/Singapore")
    engine = TemporalAwarenessEngine(cfg)
    original = "Meet me now at 3pm tonight."
    result = engine.process_message(original)
    assert result == original, f"Expected unchanged when disabled, got: {result}"


def test_timezone_applied():
    """Resolved datetimes should include the configured timezone offset."""
    engine = _make_engine("Asia/Singapore")
    result = engine.process_message("What is happening now?")
    # Singapore is UTC+8
    assert "+08:00" in result, f"Expected +08:00 timezone offset, got: {result}"
