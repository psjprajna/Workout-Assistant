"""Apply the SQL files in migrations/ to the database, in filename order.

Every migration uses CREATE ... IF NOT EXISTS, so running this repeatedly is safe.
Each file applied is logged so you can see what ran.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# src/workout_agent/db/migrate.py -> project root is parents[3]
MIGRATIONS_DIR = Path(__file__).parents[3] / "migrations"


async def apply_migrations(conn) -> None:
    """Run every .sql file in migrations/ in filename order."""
    if not MIGRATIONS_DIR.is_dir():
        raise FileNotFoundError(f"Migrations directory not found at {MIGRATIONS_DIR}")

    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    logger.info("Found %d migration file(s) in %s", len(migration_files), MIGRATIONS_DIR)

    for file_path in migration_files:
        logger.info("Applying migration %s", file_path.name)
        await conn.execute(file_path.read_text(encoding="utf-8"))
        logger.info("Applied migration %s", file_path.name)