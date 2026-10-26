"""Chainlit's data layer, backed by the API: the sidebar of past chats and reopening one.

The API owns chat history (sessions/chat_store.py) and saves each turn itself, so this
layer only reads, renames and deletes through GET/PATCH/DELETE /chats and never talks
to Postgres. Chainlit's step, element and feedback writes are ignored.
There is one user, "me", logged in automatically by the header auth callback.
"""

from datetime import UTC, datetime
from typing import Any

import httpx
from chainlit.data.base import BaseDataLayer
from chainlit.step import StepDict
from chainlit.types import (
    Feedback,
    PageInfo,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
)
from chainlit.user import PersistedUser, User

USER = "me"
_USER_CREATED_AT = "2026-01-01T00:00:00+00:00"


def _thread(chat_id: str, name: str, created_at: str, steps: list[StepDict]) -> ThreadDict:
    return ThreadDict(
        id=chat_id,
        createdAt=created_at,
        name=name,
        userId=USER,
        userIdentifier=USER,  # Chainlit only resumes threads whose author is the user
        tags=None,
        metadata={},
        steps=steps,
        elements=[],
    )


def _step(thread_id: str, step_id: str, kind: str, name: str, text: str, at: str) -> StepDict:
    return StepDict(
        id=step_id,
        threadId=thread_id,
        parentId=None,
        type=kind,  # type: ignore[typeddict-item]  # "user_message" / "assistant_message"
        name=name,
        output=text,
        createdAt=at,
        start=at,
        end=at,
        streaming=False,
        metadata={},
    )


class ApiDataLayer(BaseDataLayer):
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client
        # Chats whose first message is still being answered: the API saves a chat only
        # after the reply, but Chainlit refreshes the sidebar as soon as it's sent.
        self.pending: dict[str, tuple[str, str]] = {}  # thread id -> (title, created at)

    def mark_pending(self, thread_id: str, title: str) -> None:
        self.pending[thread_id] = (title, datetime.now(UTC).isoformat())

    def clear_pending(self, thread_id: str) -> None:
        self.pending.pop(thread_id, None)

    # --- the user ---------------------------------------------------------------------

    async def get_user(self, identifier: str) -> PersistedUser | None:
        return PersistedUser(id=USER, identifier=USER, createdAt=_USER_CREATED_AT)

    async def create_user(self, user: User) -> PersistedUser | None:
        return PersistedUser(id=USER, identifier=USER, createdAt=_USER_CREATED_AT)

    # --- threads (chats) ----------------------------------------------------------------

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        params: dict[str, Any] = {"limit": pagination.first}
        if pagination.cursor:
            params["before"] = pagination.cursor
        response = await self.client.get("/chats", params=params)
        response.raise_for_status()
        chats = response.json()
        threads = [
            _thread(c["chat_id"], c["title"], c["updated_at"], []) for c in chats
        ]
        if not pagination.cursor:  # pending chats belong on the first page only
            saved = {c["chat_id"] for c in chats}
            threads = [
                _thread(thread_id, title, at, [])
                for thread_id, (title, at) in self.pending.items()
                if thread_id not in saved
            ] + threads
        end = chats[-1]["updated_at"] if chats else None
        return PaginatedResponse(
            pageInfo=PageInfo(
                hasNextPage=len(chats) == pagination.first, startCursor=None, endCursor=end
            ),
            data=threads,
        )

    async def get_thread(self, thread_id: str) -> ThreadDict | None:
        response = await self.client.get(f"/chats/{thread_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        chat = response.json()
        steps: list[StepDict] = []
        for n, turn in enumerate(chat["turns"], start=1):
            at = turn["created_at"]
            steps.append(
                _step(thread_id, f"{thread_id}-{n}-u", "user_message", USER, turn["user_text"], at)
            )
            steps.append(
                _step(
                    thread_id, f"{thread_id}-{n}-a", "assistant_message", "Assistant",
                    turn["reply_text"], at,
                )
            )
        return _thread(thread_id, chat["title"], chat["created_at"], steps)

    async def get_thread_author(self, thread_id: str) -> str:
        return USER

    async def delete_thread(self, thread_id: str) -> None:
        self.clear_pending(thread_id)
        response = await self.client.delete(f"/chats/{thread_id}")
        response.raise_for_status()

    async def update_thread(
        self,
        thread_id: str,
        name: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> None:
        # Chainlit calls this with name=<first message> plus user_id when a chat starts,
        # and with metadata when a tab closes. Only a rename from the sidebar passes the
        # name alone; the API sets every other title itself.
        if name is None or user_id is not None or metadata is not None or tags is not None:
            return
        response = await self.client.patch(f"/chats/{thread_id}", json={"title": name})
        if response.status_code != 404:  # renaming a chat not saved yet: nothing to do
            response.raise_for_status()

    # --- the API records each turn itself, so the rest are no-ops ---------------------

    async def create_step(self, step_dict: StepDict) -> None:
        pass

    async def update_step(self, step_dict: StepDict) -> None:
        pass

    async def delete_step(self, step_id: str) -> None:
        pass

    async def upsert_feedback(self, feedback: Feedback) -> str:
        return ""

    async def delete_feedback(self, feedback_id: str) -> bool:
        return True

    async def create_element(self, element: Any) -> None:
        pass

    async def get_element(self, thread_id: str, element_id: str) -> Any:
        return None

    async def delete_element(self, element_id: str, thread_id: str | None = None) -> None:
        pass

    async def get_favorite_steps(self, user_id: str) -> list[StepDict]:
        return []

    async def set_step_favorite(self, step_dict: StepDict, favorite: bool) -> StepDict:
        return step_dict

    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        await self.client.aclose()
