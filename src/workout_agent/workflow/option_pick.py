"""A bare number answering the last AskWhich picks that option, decided in code.

The model kept reading "2" as exercise id 2 instead of the second option. Position is a
fact about the saved history, not a judgement, so code resolves it: no model call.
Replies that are not a bare number ("the incline one", "yes") still go to the model.
Pure code, no model.
"""

import re
from typing import cast

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart

_NUMBER = re.compile(r"\s*(\d+)\s*\.?\s*")
_ASK_WHICH = "final_result_AskWhich"
_FINAL_PREFIX = "final_result_"


def asked_option_ids(history: list[ModelMessage]) -> list[int] | None:
    """The option ids of the newest turn's AskWhich, or None if it did not end in one."""
    for message in reversed(history):
        if not isinstance(message, ModelResponse):
            continue
        for part in message.parts:
            if isinstance(part, ToolCallPart) and part.tool_name.startswith(_FINAL_PREFIX):
                if part.tool_name != _ASK_WHICH:
                    return None
                return cast(list[int], part.args_as_dict()["option_ids"])
    return None


def picked_exercise_id(history: list[ModelMessage], text: str) -> int | None:
    """The exercise id `text` picks from the last AskWhich, or None if it is not a pick."""
    match = _NUMBER.fullmatch(text)
    options = asked_option_ids(history)
    if match is None or options is None:
        return None
    position = int(match.group(1))
    if not 1 <= position <= len(options):
        return None  # out of range: the model asks again
    return options[position - 1]
