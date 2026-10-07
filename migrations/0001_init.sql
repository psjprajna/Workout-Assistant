CREATE TABLE IF NOT EXISTS exercises (
    exercise_id int PRIMARY KEY,
    name        text NOT NULL
);

CREATE TABLE IF NOT EXISTS exercise_logs (
    exercise_log_id int PRIMARY KEY,
    exercise_id     int  NOT NULL REFERENCES exercises (exercise_id),
    performed_on    date NOT NULL
);

CREATE INDEX IF NOT EXISTS exercise_logs_exercise_date_idx
    ON exercise_logs (exercise_id, performed_on DESC);

CREATE TABLE IF NOT EXISTS sets (
    exercise_log_id int NOT NULL
        REFERENCES exercise_logs (exercise_log_id) ON DELETE CASCADE,
    set_number      int NOT NULL,
    weight_kg       numeric(7, 2) NOT NULL,
    weight_raw      numeric(8, 4) NOT NULL,
    reps            int NOT NULL,
    PRIMARY KEY (exercise_log_id, set_number)
);