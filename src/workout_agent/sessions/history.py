"""Keep the history sent to the model small: old tool results lose their bulky parts.

In every turn but the newest few, a get_last_session result keeps who and when but drops
its sets, and a find_exercise result keeps only status and matches. Tool calls and their
results are never removed, because the provider rejects a call without its result.
Pure code, no model.
"""

from dataclasses import replace
from typing import Any, cast

from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart, UserPromptPart

# Tool name -> the keys of its result that old turns keep.
_KEEP: dict[str, tuple[str, ...]] = {
    "find_exercise_tool": ("status", "matches"),
    "get_last_session_tool": ("status", "exercise_id", "exercise_name", "performed_on"),
}


def _starts_turn(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(
        isinstance(p, UserPromptPart) for p in message.parts
    )


def _shrink(part: ToolReturnPart) -> ToolReturnPart:
    keep = _KEEP.get(part.tool_name)
    if keep is None or not isinstance(part.content, dict):
        return part
    content: dict[str, Any] = part.content
    return replace(part, content={k: content[k] for k in keep if k in content})


def shrink_old_tool_results(
    messages: list[ModelMessage], keep_full_turns: int
) -> list[ModelMessage]:
    """A copy of `messages` with tool results shrunk in all but the last `keep_full_turns`."""
    starts = [i for i, m in enumerate(messages) if _starts_turn(m)]
    if len(starts) <= keep_full_turns:
        return list(messages)
    cut = starts[len(starts) - keep_full_turns] if keep_full_turns > 0 else len(messages)
    shrunk: list[ModelMessage] = []
    for message in messages[:cut]:
        if isinstance(message, ModelRequest):
            parts = [_shrink(p) if isinstance(p, ToolReturnPart) else p for p in message.parts]
            message = replace(message, parts=parts)
        shrunk.append(message)
    return shrunk + messages[cut:]


def tool_results_in(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """The tool results in these messages, as the model is shown them (old ones shrunk)."""
    return [
        cast(dict[str, Any], part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and isinstance(part.content, dict)
    ]
