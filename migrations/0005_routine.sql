-- The weekly plan from the Jefit export's ROUTINES section (importing/routine.py).
-- Unlike the history, the plan is replaced by each import: it changes when edited in Jefit.

CREATE TABLE IF NOT EXISTS routines (
    routine_id int  PRIMARY KEY,
    name       text NOT NULL
);

-- Days have numbers, not dates. Join planned exercises on day_id (Jefit's _id, which
-- planned exercises call belongplan), never on position, which is only display order.
CREATE TABLE IF NOT EXISTS routine_days (
    day_id      int     PRIMARY KEY,
    routine_id  int     NOT NULL REFERENCES routines (routine_id) ON DELETE CASCADE,
    name        text    NOT NULL,
    position    int     NOT NULL,  -- Jefit's dayIndex: Push A 1, Pull A 2, ...
    in_rotation boolean NOT NULL   -- false for days Jefit marks as rest days (Arms)
);

CREATE TABLE IF NOT EXISTS planned_exercises (
    planned_id  int   PRIMARY KEY,
    day_id      int   NOT NULL REFERENCES routine_days (day_id) ON DELETE CASCADE,
    exercise_id int   NOT NULL REFERENCES exercises (exercise_id),
    position    int   NOT NULL,
    target_reps int[] NOT NULL CHECK (cardinality(target_reps) >= 1),  -- one per set
    UNIQUE (day_id, exercise_id)
);
