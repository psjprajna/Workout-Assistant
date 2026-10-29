"""L1, the input boundary (LLM10 unbounded consumption).

A chat message and a Jefit upload are the two things a caller controls. Both are
capped before they reach the database, the model or the importer. The app is built
without its startup (no database), so a request that gets as far as a handler would
fail: these tests only pass if the request is stopped at the boundary.
"""

import httpx
import pytest

from workout_agent.api.app import (
    MAX_CHAT_ID_CHARS,
    MAX_MESSAGE_CHARS,
    MAX_UPLOAD_BYTES,
    create_app,
)
from workout_agent.workflow.simple_router import simple_route

pytestmark = pytest.mark.security


@pytest.fixture
async def client():
    app = create_app(route=simple_route, database_url="postgresql://nowhere/none")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
@pytest.mark.parametrize(
    "body",
    [
        {"chat_id": "a", "message": "x" * (MAX_MESSAGE_CHARS + 1)},
        {"chat_id": "a" * (MAX_CHAT_ID_CHARS + 1), "message": "hi"},
        {"chat_id": "a", "message": ""},
        {"chat_id": "", "message": "hi"},
        {"chat_id": "a"},
        {"chat_id": ["a"], "message": {"$ne": 1}},
    ],
)
async def test_a_bad_chat_request_is_rejected_before_any_handler(client, path, body):
    assert (await client.post(path, json=body)).status_code == 422


async def test_an_oversized_upload_is_rejected_by_its_declared_length(client):
    r = await client.post(
        "/data/import", content=b"x", headers={"content-length": str(MAX_UPLOAD_BYTES + 1)}
    )
    assert r.json() == {
        "status": "rejected",
        "note": "the file is too large",
        "logs": 0,
        "latest_on": None,
        "exercises_indexed": 0,
    }


async def test_an_oversized_upload_with_no_declared_length_is_cut_off_while_reading(client):
    async def chunks():
        for _ in range(MAX_UPLOAD_BYTES // (1024 * 1024) + 2):
            yield b"x" * (1024 * 1024)

    r = await client.post("/data/import", content=chunks())
    assert r.json()["note"] == "the file is too large"


async def test_an_upload_that_is_not_text_is_rejected(client):
    r = await client.post("/data/import", content=b"\xff\xfe\x00\x80" * 10)
    assert r.json()["status"] == "rejected"
