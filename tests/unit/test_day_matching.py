"""Unit tests: which plan day each session was, and which day to suggest next."""

from datetime import date

from workout_agent.tools.workout_day import PlanDayInfo, match_sessions, suggest_day

PUSH = PlanDayInfo(1, "Push A", 1, True, frozenset({1, 2, 3, 4}))
PULL = PlanDayInfo(2, "Pull A", 2, True, frozenset({5, 6, 7, 8}))
LEGS = PlanDayInfo(3, "Legs A", 3, True, frozenset({9, 10, 11, 12}))
ARMS = PlanDayInfo(4, "Arms", 4, False, frozenset({5, 6, 13, 14, 15}))
DAYS = [PUSH, PULL, LEGS, ARMS]

MON, TUE, WED, SAT = date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30), date(2026, 10, 3)


def test_each_session_is_matched_to_the_day_it_shares_most_with():
    sessions = {MON: {1, 2, 3}, TUE: {5, 6, 7, 8}, WED: {9, 10, 1}}
    assert match_sessions(sessions, DAYS) == {MON: 1, TUE: 2, WED: 3}


def test_an_arms_session_is_arms_not_pull():
    # Shares two exercises with Pull A, but five with Arms.
    assert match_sessions({SAT: {5, 6, 13, 14, 15}}, DAYS) == {SAT: 4}


def test_a_tie_or_a_weak_match_is_left_unmatched():
    sessions = {MON: {1, 2, 5, 6}, TUE: {1, 99, 98, 97, 96, 95}}  # tie; 1 of 9 shared
    assert match_sessions(sessions, DAYS) == {}


def test_the_rotation_day_done_longest_ago_is_suggested():
    last = {1: date(2026, 10, 5), 2: date(2026, 10, 1), 3: date(2026, 9, 23), 4: date(2026, 8, 1)}
    # Arms was done longest ago but is not in the rotation.
    assert suggest_day(DAYS, last, {}, today=date(2026, 10, 14)) == 3


def test_a_day_never_done_comes_first():
    last = {1: date(2026, 10, 5), 3: date(2026, 9, 23)}
    assert suggest_day(DAYS, last, {}, today=date(2026, 10, 14)) == 2


def test_a_tie_goes_to_the_day_usually_done_on_this_weekday():
    last = {1: date(2026, 10, 5), 2: date(2026, 9, 1), 3: date(2026, 9, 1)}
    usual = {2: 1, 3: 2}  # Pull A usually Tuesday, Legs A usually Wednesday
    wednesday = date(2026, 10, 14)
    assert suggest_day(DAYS, last, usual, today=wednesday) == 3
    assert suggest_day(DAYS, last, usual, today=date(2026, 10, 13)) == 2  # a Tuesday


def test_no_rotation_days_suggests_nothing():
    assert suggest_day([ARMS], {}, {}, today=MON) is None
