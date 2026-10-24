-- Results of checks that run once a day (sessions/daily_checks.py). Shared by every chat:
-- the first message of the day runs the check and stores it, later chats read this row.

CREATE TABLE IF NOT EXISTS daily_checks (
    check_name text  NOT NULL,
    checked_on date  NOT NULL,
    result     jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (check_name, checked_on)  -- a second concurrent first message is a no-op
);
