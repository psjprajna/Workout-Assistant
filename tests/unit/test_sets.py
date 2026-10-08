"""Unit tests: parsing 'weightxreps' text and the lbs-to-kg date rule."""

from datetime import date

import pytest

from workout_agent.importing.sets import lbs_to_kg, parse_logged_sets, weight_in_kg


def test_parses_weight_x_reps_pairs():
    assert parse_logged_sets("80x13,80x13,80x10") == [(80.0, 13), (80.0, 13), (80.0, 10)]


def test_float_noise_is_rounded_to_two_decimals():
    assert parse_logged_sets("24.9999x6,25.0x5") == [(25.0, 6), (25.0, 5)]


def test_no_data_placeholders_are_dropped():
    assert parse_logged_sets("0x0,0x0,0x0") == []


def test_bodyweight_sets_are_kept():
    assert parse_logged_sets("0x10") == [(0.0, 10)]


def test_empty_text_gives_no_sets():
    assert parse_logged_sets("") == []


def test_unreadable_token_raises():
    with pytest.raises(ValueError):
        parse_logged_sets("80xabc")


def test_lbs_to_kg_rounds_to_two_decimals():
    assert lbs_to_kg(100.0) == 45.36
    assert lbs_to_kg(80.0) == 36.29


def test_weights_before_2026_are_converted_and_later_ones_kept():
    assert weight_in_kg(100.0, date(2025, 12, 31)) == 45.36
    assert weight_in_kg(100.0, date(2026, 1, 1)) == 100.0