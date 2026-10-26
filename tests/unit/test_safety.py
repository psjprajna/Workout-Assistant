"""Unit tests: the safety screen refuses pain, injury, diagnosis and extreme requests,
and lets ordinary workout-log questions and every exercise name through."""

import re

import pytest

from workout_agent.catalog import CATALOG, load_catalog
from workout_agent.workflow.safety import SAFETY_REPLY, screen


@pytest.mark.parametrize(
    "text",
    [
        "My knee hurts, what should I squat?",  # the PDF's own example
        "can you diagnose my shoulder",
        "rehab for a pulled hamstring",
        "I tweaked my back yesterday, can I deadlift?",
        "is it torn?",
        "sharp pain in my elbow on curls",
        "I injured my wrist",
        "sprained ankle, what can I do",
        "should I see a physio",
        "how do I lose 10 kg in a week",
        "I want to train through the injury",
    ],
)
def test_refused(text):
    assert screen(text) == SAFETY_REPLY


@pytest.mark.parametrize(
    "text",
    [
        "last time I did bench press",
        "the incline one",
        "2",
        "yes",
        "hi",
        "what did I lift on Painted Rock day",  # "pain" only as part of a word
        "last pullover",
    ],
)
def test_allowed(text):
    assert screen(text) is None


def _catalog_words() -> list[str]:
    """Every exercise name (from the comments), other name and muscle in the catalog."""
    names = re.findall(r"^id = \d+\s+#\s*(.+)$", CATALOG.read_text(), flags=re.MULTILINE)
    words = list(names)
    for entry in load_catalog().values():
        words += [*entry.aliases, *entry.muscles]
    return words


def test_no_exercise_name_or_muscle_is_refused():
    words = _catalog_words()
    assert len(words) > 100  # the catalog was read
    refused = [w for w in words if screen(f"last time I did {w}") is not None]
    assert refused == []
