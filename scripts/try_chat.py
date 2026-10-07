import asyncio

import asyncpg

from workout_agent.settings import Settings
from workout_agent.workflow.chat import Session, handle_message
from workout_agent.workflow.simple_router import simple_route


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