"""Exercises whose logged "reps" are really durations in seconds.

Jefit stores cardio and timed holds (a 20-minute walk, a 90-second plank) in the
same "weight x reps" text as lifting, with the duration where reps would be.
Parsing those as reps would make the agent say "1200 reps of walking", so
build_plan keeps the log but skips its sets for these exercise ids.

Identify exercises by id, never by name: Jefit has different exercises with the
same name. Review this list by hand when you add new exercises in Jefit.
"""

TIMED_EXERCISE_IDS: frozenset[int] = frozenset(
    {
        310,  # Elliptical Training
        312,  # Indoor Cycling
        318,  # Stationary Bike
        321,  # Treadmill Running
        322,  # Walking
        631,  # Plank
    }
)