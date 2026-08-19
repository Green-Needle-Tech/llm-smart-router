"""Tests for the Level enum."""
from app.schemas.router import Level


def test_level_ordering():
    assert Level.L1 < Level.L2
    assert Level.L2 < Level.L3
    assert Level.L3 < Level.L4
    assert Level.L4 > Level.L1
    assert Level.L1 <= Level.L1
    assert Level.L4 >= Level.L4


def test_from_str():
    assert Level.from_str("L1") == Level.L1
    assert Level.from_str("l2") == Level.L2
    assert Level.from_str("  L3  ") == Level.L3


def test_from_str_invalid():
    import pytest
    with pytest.raises(ValueError):
        Level.from_str("L6")
    with pytest.raises(ValueError):
        Level.from_str("invalid")


def test_from_numeric():
    assert Level.from_numeric(1) == Level.L1
    assert Level.from_numeric(3) == Level.L3
    assert Level.from_numeric(5) == Level.L5
    assert Level.from_numeric(6) == Level.L5  # clamped
    assert Level.from_numeric(0) == Level.L1  # clamped


def test_numeric_property():
    assert Level.L1.numeric == 1
    assert Level.L4.numeric == 4
    assert Level.L5.numeric == 5
