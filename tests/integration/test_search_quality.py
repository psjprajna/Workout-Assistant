"""Search quality gate: find_exercise on evals/search_cases.toml against your real data.

Fails if search confidently picks a wrong exercise (wrong_ok), offers anything for a noise
phrase, or fails a phrase the reference run passed. Marked `search_eval`: it reads the real
database and Qdrant index (not the test ones), so it skips until scripts/load_jefit.py,
scripts/load_catalog.py and scripts/index_exercises.py have run. Phrase vectors come from
the embedding cache, so a rerun makes no API calls; new phrases need OPENROUTER_API_KEY.

After a deliberate change, run scripts/eval_search.py and point REFERENCE at its results.
"""

import json
from functools import partial
from pathlib import Path

import asyncpg
import pytest
from qdrant_client import AsyncQdrantClient

from workout_agent.embeddings.cache import cached
from workout_agent.embeddings.index import COLLECTION
from workout_agent.embeddings.openrouter import MODEL_NAME, embed
from workout_agent.embeddings.search import search_exercises
from workout_agent.evals.search import compare, load_cases, run_cases, summarize
from workout_agent.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "evals" / "search_cases.toml"
REFERENCE = ROOT / "evals" / "results" / "search-2026-10-12-phase-d2-large.json"

pytestmark = pytest.mark.search_eval


@pytest.fixture
async def real():
    settings = Settings()
    try:
        conn = await asyncpg.connect(settings.database_url)
    except OSError:
        pytest.skip("Postgres is not running")
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        if not await conn.fetchval("select count(*) from exercises"):
            pytest.skip("no exercises imported: run scripts/load_jefit.py")
        if not await qdrant.collection_exists(COLLECTION):
            pytest.skip("no Qdrant index: run scripts/index_exercises.py")

        async def fetch(texts: list[str]) -> list[list[float]]:
            if settings.openrouter_api_key is None:
                pytest.skip("a phrase is not in the embedding cache and no API key is set")
            return await embed(settings, texts)

        embedder = cached(conn, MODEL_NAME, fetch)
        yield conn, partial(search_exercises, qdrant, embedder, limit=5)
    finally:
        await conn.close()
        await qdrant.close()


async def test_search_never_guesses_wrong_and_keeps_every_reference_pass(real):
    conn, search = real
    results = await run_cases(conn, load_cases(CASES), search)
    overall = summarize(results)["all"]

    assert overall.wrong_ok == 0, [r.phrase for r in results if r.wrong_ok]
    assert overall.false_offers == 0, [r.phrase for r in results if r.false_offer]

    reference = json.loads(REFERENCE.read_text())
    _, broken = compare(
        {c["phrase"]: c["passed"] for c in reference["cases"]},
        {r.phrase: r.passed for r in results},
    )
    assert broken == []
