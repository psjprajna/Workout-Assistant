"""Integration tests: load_routine writes the plan; a new export replaces the old plan."""

from workout_agent.importing.routine import (
    PlanDay,
    PlannedExercise,
    Routine,
    RoutineInfo,
    load_routine,
)


def routine(*planned: PlannedExercise) -> Routine:
    return Routine(
        routines=[RoutineInfo(id=1, name="PPL")],
        days=[
            PlanDay(id=1, routine_id=1, name="Push A", position=1, in_rotation=True),
            PlanDay(id=4, routine_id=1, name="Arms", position=4, in_rotation=False),
        ],
        planned=list(planned),
    )


BENCH = PlannedExercise(
    id=512, day_id=1, exercise_id=2, exercise_name="Barbell Bench Press (Flat)",
    position=1, target_reps=[7, 8, 10],
)
STEP_UP = PlannedExercise(
    id=552, day_id=1, exercise_id=134, exercise_name="Dumbbell Step-Up",
    position=0, target_reps=[10, 10, 10],
)


async def test_the_plan_is_stored_with_its_targets(conn):
    await conn.execute("insert into exercises values (2, 'Barbell Bench Press')")
    await load_routine(conn, routine(BENCH, STEP_UP))

    rows = await conn.fetch(
        "select d.name as day, d.in_rotation, e.name, p.target_reps "
        "from planned_exercises p join routine_days d using (day_id) "
        "join exercises e using (exercise_id) order by p.position"
    )
    assert [(r["day"], r["in_rotation"], r["name"], r["target_reps"]) for r in rows] == [
        ("Push A", True, "Dumbbell Step-Up", [10, 10, 10]),  # never logged: added
        ("Push A", True, "Barbell Bench Press", [7, 8, 10]),  # logged name kept
    ]
    assert await conn.fetchval("select count(*) from routine_days") == 2


async def test_a_new_export_replaces_the_old_plan(conn):
    await conn.execute("insert into exercises values (2, 'Barbell Bench Press')")
    await load_routine(conn, routine(BENCH, STEP_UP))
    changed = PlannedExercise(**{**vars(BENCH), "target_reps": [8, 8, 8]})
    await load_routine(conn, routine(changed))  # step-up removed in Jefit, bench changed

    rows = await conn.fetch("select exercise_id, target_reps from planned_exercises")
    assert [(r["exercise_id"], r["target_reps"]) for r in rows] == [(2, [8, 8, 8])]
    assert await conn.fetchval("select count(*) from routine_days") == 2
