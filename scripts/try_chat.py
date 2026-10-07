import asyncio
import re

import asyncpg

from workout_agent.settings import Settings
from workout_agent.workflow.chat import RouteDecision, Session, handle_message


async def simple_route(text: str, has_pending: bool) -> RouteDecision:
    """A crude stand-in for the model, just so we can chat today."""
    cleaned = text.strip().lower()

    # A bare number while a question is pending means "I pick that option".
    if has_pending and cleaned.isdigit():
        return RouteDecision(intent="pick_option", option_number=int(cleaned))

    # "last time I did bench press?" -> exercise text is "bench press"
    match = re.search(r"last time (?:i )?(?:did|do)\s+(.+?)\s*\??$", cleaned)
    if match:
        return RouteDecision(intent="last_session", exercise_text=match.group(1))

    return RouteDecision(intent="unknown")


async def main() -> None:
    conn = await asyncpg.connect(Settings().database_url)
    session = Session()
    print("Ask about your workouts. Type 'quit' to stop.\n")
    try:
        while True:
            try:
                text = await asyncio.to_thread(input, "you> ")
            except EOFError:
                break
            if text.strip().lower() in {"quit", "exit"}:
                break
            reply = await handle_message(conn, session, text, simple_route)
            print(f"\nbot> {reply}\n")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())