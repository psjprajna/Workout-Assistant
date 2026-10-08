"""A rule-based router: decides what a message means using regexes, no model.

It is the stand-in for the AI router and a baseline to compare the model against.
It only returns a RouteDecision; it never looks at the database or writes a reply.
"""

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