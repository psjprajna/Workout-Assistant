"""Unit tests: the Chainlit data layer maps the API's /chats answers, no server needed."""

import json

import httpx
from chainlit.types import Pagination, ThreadFilter

from workout_agent.ui.data_layer import ApiDataLayer

API_URL = "http://api.test"
T0, T1 = "2026-10-12T10:00:00+00:00", "2026-10-12T10:01:00+00:00"
CHAT = {
    "chat_id": "t1",
    "title": "Barbell Bench Press – last session",
    "created_at": T0,
    "turns": [
        {"user_text": "bench?", "reply_text": "Which one?", "created_at": T0},
        {"user_text": "1", "reply_text": "80 kg x 8", "created_at": T1},
    ],
}


def _layer() -> ApiDataLayer:
    return ApiDataLayer(httpx.AsyncClient(base_url=API_URL))


async def test_get_thread_maps_turns_to_user_and_assistant_steps(httpx_mock):
    httpx_mock.add_response(url=f"{API_URL}/chats/t1", json=CHAT)

    thread = await _layer().get_thread("t1")

    assert thread is not None
    assert thread["userIdentifier"] == "me"
    assert thread["name"] == CHAT["title"]
    steps = thread["steps"]
    assert [(s.get("type"), s.get("output")) for s in steps] == [
        ("user_message", "bench?"),
        ("assistant_message", "Which one?"),
        ("user_message", "1"),
        ("assistant_message", "80 kg x 8"),
    ]
    assert [s.get("id") for s in steps] == ["t1-1-u", "t1-1-a", "t1-2-u", "t1-2-a"]


async def test_unknown_thread_is_none(httpx_mock):
    httpx_mock.add_response(url=f"{API_URL}/chats/nope", status_code=404)
    assert await _layer().get_thread("nope") is None


async def test_list_threads_pages_by_updated_at(httpx_mock):
    chats = [
        {"chat_id": "b", "title": "B", "created_at": "2026-10-12T09:00:00+00:00",
         "updated_at": "2026-10-12T11:00:00+00:00"},
        {"chat_id": "a", "title": "A", "created_at": "2026-10-12T08:00:00+00:00",
         "updated_at": "2026-10-12T10:00:00+00:00"},
    ]
    httpx_mock.add_response(url=f"{API_URL}/chats?limit=2", json=chats)

    page = await _layer().list_threads(Pagination(first=2), ThreadFilter())

    assert [t["id"] for t in page.data] == ["b", "a"]
    assert page.pageInfo.hasNextPage
    assert page.pageInfo.endCursor == "2026-10-12T10:00:00+00:00"


async def test_pending_chat_shows_before_it_is_saved(httpx_mock):
    httpx_mock.add_response(url=f"{API_URL}/chats?limit=20", json=[], is_reusable=True)
    layer = _layer()
    layer.mark_pending("new", "bench?")

    page = await layer.list_threads(Pagination(first=20), ThreadFilter())
    assert [(t["id"], t["name"]) for t in page.data] == [("new", "bench?")]

    layer.clear_pending("new")
    page = await layer.list_threads(Pagination(first=20), ThreadFilter())
    assert page.data == []


async def test_chainlit_naming_a_new_thread_sends_nothing(httpx_mock):
    # Chainlit passes the first message as name together with the user id.
    await _layer().update_thread("t1", name="bench?", user_id="me")
    await _layer().update_thread("t1", metadata={"x": 1})
    assert httpx_mock.get_requests() == []


async def test_rename_from_the_sidebar_patches_the_title(httpx_mock):
    httpx_mock.add_response(url=f"{API_URL}/chats/t1", method="PATCH", status_code=204)

    await _layer().update_thread("t1", name="Chest day")

    assert json.loads(httpx_mock.get_request().content) == {"title": "Chest day"}


async def test_delete_thread_deletes_the_chat(httpx_mock):
    httpx_mock.add_response(url=f"{API_URL}/chats/t1", method="DELETE", status_code=204)
    await _layer().delete_thread("t1")
    assert httpx_mock.get_request().method == "DELETE"
