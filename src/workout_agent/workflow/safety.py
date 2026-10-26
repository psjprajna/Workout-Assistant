"""The safety screen: a fixed refusal for pain, injury, diagnosis and extreme requests.

Pure regex over the raw text, run before any model and independently of it (PDF §16,
docs/agentic-workflow.md). screen() is a plain function so every path can call it: the
agent path through its input guardrail (agents/guardrails.py), the rule-router path
(workflow/chat.py) directly. Whole-word patterns, so an exercise name like "Hanging
Knee Raise" or a word like "painted" does not trip it (tests/unit/test_safety.py checks
every name in the catalog).
"""

import re

from pydantic_ai_harness.guardrails import InputBlocked

SAFETY_REPLY = (
    "I can't help with pain, injuries, diagnosis or rehab, or with extreme training "
    "requests. Please talk to a doctor or a physiotherapist. I can tell you about your "
    "logged sessions, like 'last time I did bench press?'"
)

_PATTERNS = [
    # Acute pain or injury.
    r"hurts?|hurting|pain(?:s|ful)?|injur(?:y|ies|ed)|sprain(?:s|ed)?|tweaked",
    r"pulled (?:a|my) \w+|torn|swollen|swelling|dislocat\w*|numbness|tingling",
    # Diagnosis or rehabilitation.
    r"diagnos\w*|rehab\w*|physio\w*|physical therap\w*",
    # Unsafe or extreme requests.
    r"lose \d+ ?(?:kg|kgs|kilos?|lbs?|pounds) in (?:a|one|\d+) (?:day|days|week|weeks)",
    r"steroids?|sarms|starv\w*",
]
_SCREEN = re.compile(r"\b(?:" + "|".join(_PATTERNS) + r")\b", re.IGNORECASE)


class SafetyRefused(InputBlocked):
    """The safety screen matched: the run stops before any model call, with this reply."""

    def __init__(self, reply: str) -> None:
        super().__init__(reply)
        self.reply = reply


def screen(text: str) -> str | None:
    """The fixed refusal if the text is about pain, injury, diagnosis or extremes; else None."""
    return SAFETY_REPLY if _SCREEN.search(text) else None
