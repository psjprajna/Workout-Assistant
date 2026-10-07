"""A rule-based router: a stand-in for the model, and a baseline to compare it with."""

import re

from workout_agent.workflow.chat import RouteDecision

# "last time I did bench press?" -> captures "bench press"
_LAST_TIME = re.compile(r"last time (?:i )?(?:did|do)\s+(.+?)\s*\??$")


async def simple_route(text: str, has_pending: bool) -> RouteDecision:
    cleaned = text.strip().lower()

    # A bare number only means "I pick that option" if we asked a question.
    if has_pending and cleaned.isdigit():
        return RouteDecision(intent="pick_option", option_number=int(cleaned))

    match = _LAST_TIME.search(cleaned)
    if match:
        return RouteDecision(intent="last_session", exercise_text=match.group(1))

    return RouteDecision(intent="unknown")