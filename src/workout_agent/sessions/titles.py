"""Chat titles from code, never from the model.

A chat about one exercise is named after it ("Barbell Bench Press – last session"), one
about a plan day after the day ("Legs A – workout").
Until a tool has found one, the title is the first message, trimmed, marked as a
fallback so a later turn can replace it.
"""

from typing import Any

from workout_agent.tools.workout_day import short_day_name

MAX_FALLBACK = 40


def chat_title(user_text: str, evidence: list[dict[str, Any]]) -> tuple[str, bool]:
    """(title, is_fallback) for a turn, from the tool results it collected."""
    for item in reversed(evidence):
        if item.get("status") == "ok" and item.get("day_name"):
            return f"{short_day_name(item['day_name'])} – workout", False
    for item in reversed(evidence):
        if item.get("status") == "ok" and item.get("exercise_name"):
            return f"{item['exercise_name']} – last session", False
    for item in reversed(evidence):
        if item.get("status") == "ok" and item.get("matches"):
            return item["matches"][0]["name"], False
    text = " ".join(user_text.split())
    if len(text) > MAX_FALLBACK:
        text = text[: MAX_FALLBACK - 1].rstrip() + "…"
    return text, True
