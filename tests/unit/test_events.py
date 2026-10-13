"""Unit tests: chat stream events survive the wire, and their labels come from code."""

import pytest

from workout_agent.workflow.events import (
    Error,
    Final,
    ToolEnd,
    ToolStart,
    chat_event_adapter,
    tool_label,
)


@pytest.mark.parametrize(
    "event",
    [
        ToolStart(tool="find_exercise_tool", label="Searching…"),
        ToolEnd(tool="find_exercise_tool", status="ambiguous"),
        Final(reply="Last time you did X was 8 Feb 2026:"),
        Error(message="busy"),
    ],
)
def test_each_event_round_trips_as_json(event):
    wire = chat_event_adapter.dump_json(event)
    assert chat_event_adapter.validate_json(wire) == event


def test_labels_name_the_search_and_the_fetch():
    assert (
        tool_label("find_exercise_tool", {"query": "bench press"})
        == "Searching your exercises for 'bench press'…"
    )
    assert tool_label("get_last_session_tool", {"exercise_id": 2}) == "Fetching your last session…"
    assert tool_label("other_tool", {}) == "Running other_tool…"
