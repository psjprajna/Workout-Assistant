"""Fixed-template replies built from tool results, plus a check for invented numbers.

Nothing here calls a model. Every number in a reply is copied from a tool result,
and verify_reply proves it by checking each number in a reply against the data.
render writes the agent's reply: the model only picks a typed output (agents/outputs.py)
and the ids it is about, and render looks those up in this turn's tool results (the
evidence dicts) and fills the same templates. check_output says why an output can't be
rendered, e.g. an id no tool returned this turn. verify_agent_reply checks every number
in a reply against every tool result the model was shown.
"""

import re
from datetime import date
from decimal import Decimal
from typing import Any

from workout_agent.agents.outputs import AgentOutput, AskWhich, Help, NotFound, ShowLastSession
from workout_agent.tools.find_exercise import ExerciseMatch, FindResult
from workout_agent.tools.get_last_session import LastSessionResult, SetItem

HELP_TEXT = (
    "I can answer questions about your last session of an exercise, "
    "like 'last time I did bench press?'"
)

# --- Templates: each reply is built from tool-result fields, never from model text ---


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


def format_did_you_mean(match: ExerciseMatch, query: str) -> str:
    """One close match: a yes/no question (the agent path understands "yes")."""
    return f"I couldn't find '{query}' exactly. Did you mean {match.name}?"


# --- The agent's typed output -> reply text, from this turn's tool results ---


class RenderError(Exception):
    """The output refers to something no tool returned in this turn."""


def _is_session(e: dict[str, Any]) -> bool:
    return "exercise_log_id" in e  # get_last_session results; find results have "matches"


def _session(evidence: list[dict[str, Any]], exercise_id: int) -> dict[str, Any] | None:
    found = [e for e in evidence if _is_session(e) and e["exercise_id"] == exercise_id]
    return found[-1] if found else None


def _finds(evidence: list[dict[str, Any]], status: str) -> list[dict[str, Any]]:
    return [e for e in evidence if "matches" in e and e["status"] == status]


def _ambiguous_with(evidence: list[dict[str, Any]], ids: list[int]) -> dict[str, Any] | None:
    """The latest 'ambiguous' find_exercise result that offered every one of these ids."""
    for e in reversed(_finds(evidence, "ambiguous")):
        if set(ids) <= {m["exercise_id"] for m in e["matches"]}:
            return e
    return None


def check_output(output: AgentOutput, evidence: list[dict[str, Any]]) -> str | None:
    """Why this output can't be written from this turn's tool results, or None if it can."""
    match output:
        case ShowLastSession(exercise_id=exercise_id):
            if _session(evidence, exercise_id) is None:
                return (
                    f"get_last_session was not called for exercise_id {exercise_id} in this "
                    "turn. Call it first, then choose ShowLastSession."
                )
        case AskWhich(option_ids=ids):
            if len(set(ids)) != len(ids):
                return "option_ids has duplicates."
            if _ambiguous_with(evidence, ids) is None:
                return (
                    f"option_ids {ids} are not all options of an 'ambiguous' find_exercise "
                    "result from this turn. Call find_exercise and use its exercise ids."
                )
        case NotFound():
            if not _finds(evidence, "no_rows"):
                return "find_exercise did not return 'no_rows' in this turn."
        case Help():
            pass
    return None


def _session_result(e: dict[str, Any]) -> LastSessionResult:
    """An evidence dict (as the model saw it: text weights, ISO date) back to the result."""
    return LastSessionResult(
        status=e["status"],
        exercise_id=e["exercise_id"],
        exercise_name=e["exercise_name"],
        performed_on=date.fromisoformat(e["performed_on"]) if e["performed_on"] else None,
        exercise_log_id=e["exercise_log_id"],
        sets=[SetItem(s["set_number"], Decimal(s["weight_kg"]), s["reps"]) for s in e["sets"]],
    )


def render(output: AgentOutput, evidence: list[dict[str, Any]]) -> str:
    """The reply for the agent's choice, written only from this turn's tool results."""
    problem = check_output(output, evidence)
    if problem is not None:
        raise RenderError(problem)
    match output:
        case ShowLastSession(exercise_id=exercise_id):
            return format_last_session(_session_result(_session(evidence, exercise_id) or {}))
        case AskWhich(option_ids=ids):
            find = _ambiguous_with(evidence, ids) or {}
            by_id = {m["exercise_id"]: m["name"] for m in find["matches"]}
            matches = [ExerciseMatch(i, by_id[i]) for i in ids]  # in the order shown
            if len(matches) == 1:
                return format_did_you_mean(matches[0], find["query"])
            return format_find_question(FindResult("ambiguous", matches), find["query"])
        case NotFound():
            return format_find_question(
                FindResult("no_rows", []), _finds(evidence, "no_rows")[-1]["query"]
            )
        case Help():
            return HELP_TEXT


# --- Verification: prove a reply contains no number that is not in the data ---

UNVERIFIED_REPLY = "I couldn't verify that answer, so I won't show it."


def _numbers_in(text: str, names: list[str]) -> list[Decimal]:
    """Every number in the text, as Decimal so "80" and "80.00" count as the same number.

    Digits inside an exercise name ("Cable Fly 45 Degree") are not claims, so the names
    are removed first, longest first so a shorter name can't cut into a longer one.
    """
    for name in sorted(names, key=len, reverse=True):
        text = text.replace(name, "")
    return [Decimal(token) for token in re.findall(r"\d+(?:\.\d+)?", text)]


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
    names = [result.exercise_name] if result.exercise_name else []
    allowed = _allowed_numbers(result)
    return all(n in allowed for n in _numbers_in(reply, names))


def _names_in_evidence(evidence: list[dict[str, Any]]) -> list[str]:
    names = [e["exercise_name"] for e in evidence if e.get("exercise_name")]
    names += [e["query"] for e in evidence if e.get("query")]  # echoed back, not a claim
    names += [m["name"] for e in evidence for m in e.get("matches", [])]
    return names


def allowed_numbers_from_evidence(evidence: list[dict[str, Any]]) -> set[Decimal]:
    """Every number a reply may contain, given the tool results (as the model saw them).

    A last session gives its exercise id, the date's day, month and year (the model may
    write "8 Feb 2026" or "2026-02-08"), each set's number, reps and weight, and how many
    sets there were. A list of matches gives the ids and the list markers 1..n.
    """
    allowed: set[Decimal] = set()
    for e in evidence:
        if e.get("exercise_id") is not None:
            allowed.add(Decimal(e["exercise_id"]))
        if e.get("performed_on"):
            d = date.fromisoformat(e["performed_on"])
            allowed |= {Decimal(d.day), Decimal(d.month), Decimal(d.year)}
        sets = e.get("sets", [])
        if sets:
            allowed.add(Decimal(len(sets)))
        for s in sets:
            allowed |= {Decimal(s["set_number"]), Decimal(s["reps"]), Decimal(s["weight_kg"])}
        matches = e.get("matches", [])
        allowed |= {Decimal(m["exercise_id"]) for m in matches}
        allowed |= {Decimal(i) for i in range(1, len(matches) + 1)}
    return allowed


def verify_agent_reply(reply: str, evidence: list[dict[str, Any]]) -> bool:
    """True only if every number in the agent's reply appears in the tool results."""
    allowed = allowed_numbers_from_evidence(evidence)
    return all(n in allowed for n in _numbers_in(reply, _names_in_evidence(evidence)))