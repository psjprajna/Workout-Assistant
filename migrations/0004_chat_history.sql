-- Chat history (src/workout_agent/sessions/chat_store.py). One row per chat, one row per
-- answered turn. A turn is written only after the request succeeded, never updated.

CREATE TABLE IF NOT EXISTS chats (
    chat_id           text PRIMARY KEY,                -- Chainlit thread id
    title             text NOT NULL,
    -- true while the title is just the first message; a later exercise title replaces it.
    title_is_fallback boolean NOT NULL DEFAULT true,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_turns (
    chat_id    text NOT NULL REFERENCES chats (chat_id) ON DELETE CASCADE,
    turn_no    int  NOT NULL,
    user_text  text NOT NULL,
    reply_text text NOT NULL,
    messages   jsonb NOT NULL,   -- this turn's model messages (ModelMessagesTypeAdapter)
    evidence   jsonb NOT NULL,   -- every tool result of this turn
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (chat_id, turn_no)  -- a second concurrent write of the same turn fails
);

CREATE INDEX IF NOT EXISTS chats_updated_idx ON chats (updated_at DESC);
