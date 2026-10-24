"""Download your Jefit export, and optionally load it.

Usage: uv run python scripts/fetch_jefit.py [--headed] [--load]
Needs JEFIT_EMAIL and JEFIT_PASSWORD in .env, and `uv run playwright install chromium`
once. --headed shows the browser, to watch where it stops if Jefit's pages changed.
The chat does this by itself on the first message of each day; this is for trying it.
"""

import argparse
import asyncio
import sys

import asyncpg

from workout_agent.importing.jefit_export import JefitFetchError, download_export
from workout_agent.importing.refresh import EXPORT_DIR, load_export
from workout_agent.settings import Settings


async def main(headed: bool, load: bool) -> int:
    settings = Settings()
    if not (settings.jefit_email and settings.jefit_password):
        print("JEFIT_EMAIL and JEFIT_PASSWORD are not set in .env")
        return 1
    try:
        path = await download_export(
            settings.jefit_email,
            settings.jefit_password.get_secret_value(),
            EXPORT_DIR,
            headless=not headed,
        )
    except JefitFetchError as error:
        print(f"failed: {error}")
        return 1
    print(f"downloaded {path}")
    if load:
        conn = await asyncpg.connect(settings.database_url)
        try:
            counts = await load_export(conn, path)
        finally:
            await conn.close()
        print(f"loaded {counts.logs} logs, {len(counts.errors)} problems")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the Jefit export.")
    parser.add_argument("--headed", action="store_true", help="show the browser")
    parser.add_argument("--load", action="store_true", help="import it into Postgres")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.headed, args.load)))
