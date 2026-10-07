"""Fixed-template replies built from tool results, plus a check for invented numbers.

Nothing here calls a model. Every number in a reply is copied from a tool result,
and verify_reply proves it by checking each number in a reply against the data.
"""

import re
from datetime import date
from decimal import Decimal

from workout_agent.tools.find_exercise import FindResult
from workout_agent.tools.get_last_session import LastSessionResult


def format_date(d: date) -> str:
    """8 Feb 2026. Built by hand so the day has no leading zero."""
    return f"{d.day} {d:%b} {d.year}"


def format_last_session(result: LastSessionResult) -> str:
    # No log found, or (defensively) no date to talk about: say so, quote nothing.
    if result.status == "no_rows" or result.performed_on is None:
        return "I can't find any logged sessions for that exercise."

    date_text = format_date(result.performed_on)

    # A log exists but has no sets (walks, planks): true, but no numbers to quote.
    if not result.sets:
        return (
            f"You logged {result.exercise_name} on {date_text}, "
            "but there are no set details for it."
        )

    lines = [f"- Set {s.set_number}: {s.weight_kg} kg x {s.reps}" for s in result.sets]
    header = f"Last time you did {result.exercise_name} was {date_text}:"
    return header + "\n" + "\n".join(lines)


def format_find_question(result: FindResult, query: str) -> str:
    # Nothing matched: an honest "not in your data", never a guess.
    if result.status == "no_rows":
        return f"'{query}' is not in your data."

    if result.status == "ambiguous":
        options = "\n".join(
            f"{i}. {m.name}" for i, m in enumerate(result.matches, start=1)
        )
        return f"I found more than one match for '{query}'. Which one do you mean?\n{options}"

    # status "ok" means the exercise is known, so there is nothing to ask.
    raise ValueError("format_find_question called with an 'ok' result")


def _allowed_numbers(result: LastSessionResult) -> set[Decimal]:
    """Every number a reply is allowed to contain, taken from the tool result."""
    if result.status == "no_rows" or result.performed_on is None:
        return set()

    allowed = {Decimal(result.performed_on.day), Decimal(result.performed_on.year)}
    for s in result.sets:
        allowed.add(Decimal(s.set_number))
        allowed.add(Decimal(s.reps))
        allowed.add(s.weight_kg)
    return allowed


def verify_reply(reply: str, result: LastSessionResult) -> bool:
    """True only if every number in the reply appears in the tool result."""
    # Digits inside the exercise name ("Cable Fly 45 Degree") are not claims.
    text = reply
    if result.exercise_name:
        text = text.replace(result.exercise_name, "")

    # Compare as Decimal, so "80" and "80.00" count as the same number.
    found = [Decimal(token) for token in re.findall(r"\d+(?:\.\d+)?", text)]
    allowed = _allowed_numbers(result)
    return all(n in allowed for n in found)