"""Unit tests: the ROUTINES section becomes days and planned exercises with target reps."""

from workout_agent.importing.routine import parse_routines

ROUTINE_HEADER = (
    "row_id,USERID,TIMESTAMP,_id,name,difficulty,focus,dayaweek,description,daytype,tags,"
    "rdb_id,bannerCode,progression_flag"
)
DAY_HEADER = (
    "row_id,USERID,TIMESTAMP,package,_id,name,day,dayIndex,interval_mode,rest_day,week,"
    "sort_order,day_completed_timestamp"
)
EX_HEADER = (
    "row_id,USERID,TIMESTAMP,belongSys,superset,_id,exercise_id,belongplan,exercisename,"
    "setcount,timer,logs,bodypart,mysort,targetrep,setdone,setdonetime,interval_time,"
    "interval_unit,rest_time_enabled,interval_time_enabled,is_tier_primary"
)


def day(day_id, name, index, rest=0):
    return f"1,1,t,1,{day_id},{name},{index},{index},0,{rest},0,{index - 1},2026-10-05"


def ex(row_id, exercise_id, day_id, name, setcount, targets, sort):
    return (
        f"1,1,t,1,0,{row_id},{exercise_id},{day_id},{name},{setcount},60,0x0,0,{sort},"
        f'"{targets}",0,0,0,0,0,0,0'
    )


# The shape of a real export: each row under its own header, blank lines between, and
# a routine description that spans lines.
SECTION = "\n".join([
    ROUTINE_HEADER,
    '1,1,t,1,PPL,1,1,7,"push, pull,\nlegs",1,,0,0,0',
    "",
    DAY_HEADER,
    day(1, "Push A. Chest and shoulder ", 1),
    "",
    EX_HEADER,
    ex(512, 2, 1, "Barbell Bench Press", 3, "7,8,10", 1),
    "",
    EX_HEADER,
    ex(613, 1055, 1, "Bench  Push-Up", 3, "10,10,10", 0),
    "",
    "",
    DAY_HEADER,
    day(4, "Arms ", 4, rest=1),
    "",
    EX_HEADER,
    ex(700, 112, 4, "Barbell Curl", 2, "12,10", 0),
])


def test_days_and_planned_exercises_are_read():
    routine = parse_routines(SECTION)
    assert routine.errors == []
    assert [(r.id, r.name) for r in routine.routines] == [(1, "PPL")]
    assert [(d.id, d.name, d.position, d.in_rotation) for d in routine.days] == [
        (1, "Push A. Chest and shoulder", 1, True),
        (4, "Arms", 4, False),  # Jefit marks Arms a rest day: not in the rotation
    ]
    push = [p for p in routine.planned if p.day_id == 1]
    assert [(p.exercise_id, p.exercise_name, p.position) for p in push] == [
        (2, "Barbell Bench Press", 1),
        (1055, "Bench Push-Up", 0),
    ]
    assert push[0].target_reps == [7, 8, 10]


def test_targets_that_do_not_match_the_set_count_are_reported():
    section = "\n".join([DAY_HEADER, day(1, "Push A", 1), EX_HEADER, ex(1, 2, 1, "B", 3, "8,8", 0)])
    routine = parse_routines(section)
    assert routine.planned == []
    assert len(routine.errors) == 1
    assert "3 sets but 2 target reps" in routine.errors[0]


def test_unreadable_targets_are_reported():
    section = "\n".join([DAY_HEADER, day(1, "Push A", 1), EX_HEADER, ex(1, 2, 1, "B", 2, "8,x", 0)])
    routine = parse_routines(section)
    assert routine.planned == []
    assert "targetrep" in routine.errors[0]


def test_an_exercise_for_a_day_not_in_the_export_is_reported():
    section = "\n".join([DAY_HEADER, day(1, "Push A", 1), EX_HEADER, ex(1, 2, 9, "B", 1, "8", 0)])
    routine = parse_routines(section)
    assert routine.planned == []
    assert "day 9" in routine.errors[0]
