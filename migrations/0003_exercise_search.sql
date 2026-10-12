-- Exercise search (src/workout_agent/tools/find_exercise.py).

-- Trigram similarity, for typos the whole-word match misses ("dumbell rows").
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Other names for an exercise ("rdl", "pec deck"). find_exercise matches these like the
-- name itself. source 'catalog' rows come from config/exercise_catalog.toml.
CREATE TABLE IF NOT EXISTS exercise_aliases (
    exercise_id int  NOT NULL REFERENCES exercises (exercise_id) ON DELETE CASCADE,
    alias       text NOT NULL,
    source      text NOT NULL CHECK (source IN ('catalog')),
    PRIMARY KEY (exercise_id, alias)
);

CREATE INDEX IF NOT EXISTS exercise_aliases_alias_idx ON exercise_aliases (alias);
