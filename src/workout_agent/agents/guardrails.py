"""Guardrails at the three edges of an agent run: the prompt, each tool call, the answer.

They turn rules the prompt only asks for into code (Pydantic AI Harness guardrails,
docs/amendments.md §24). Each guard is a plain function of (ctx, value), so the tests
call them directly without a model.

- safety_guard: the safety screen, before any model call. A match raises SafetyRefused
  and the chat code writes the fixed refusal; the model never sees the message.
- tool_call_guard: get_last_session only with an id a tool returned (this turn or an
  earlier one), and no repeat of a call already answered this turn (any tool). Both are mistakes
  the model can fix, so they are sent back to it (retry).
- tool_result_guard: every result keeps the docs/tools.md contract: a dict, an allowed
  status, only allowlisted keys (the privacy line), a few hundred tokens at most. A
  break is a bug in our code, not the model's, so it stops the run (ToolBlocked) and
  nothing is saved.
- answer_only_from_this_turn: the final typed choice must be writable from this turn's
  tool results (replies.check_output); otherwise it is sent back to the model.
"""

import json
from typing import TYPE_CHECKING, Any, cast

from pydantic_ai import RunContext
from pydantic_ai_harness.guardrails import (
    GuardrailResult,
    ToolBlocked,
    ToolCallInfo,
    ToolResultInfo,
)

from workout_agent.agents.outputs import AgentOutput
from workout_agent.sessions.history import tool_results_in
from workout_agent.workflow.replies import check_output
from workout_agent.workflow.safety import SafetyRefused, screen

if TYPE_CHECKING:
    from workout_agent.agents.workout_agent import AgentDeps

# docs/tools.md: a tool "returns at most a few hundred tokens". About 4 characters a
# token; the largest real results (3 matches, a session of ~10 sets) are well under this.
MAX_RESULT_CHARS = 2000

# Tool name -> (allowed statuses, allowed top-level keys, allowed keys of each list item).
# Nothing outside these can reach the model: no date of birth, no body measurements.
_CONTRACT: dict[str, tuple[set[str], set[str], dict[str, set[str]]]] = {
    "find_exercise_tool": (
        {"ok", "ambiguous", "no_rows"},
        {"status", "matches"},
        {"matches": {"exercise_id", "name"}},
    ),
    "get_last_session_tool": (
        {"ok", "no_rows"},
        {"status", "exercise_id", "exercise_name", "performed_on", "exercise_log_id", "sets"},
        {"sets": {"set_number", "weight_kg", "reps"}},
    ),
    "workout_days_tool": (
        {"ok", "no_rows"},
        {"status", "suggested_day_id", "data_through", "days"},
        {"days": {"day_id", "name", "in_rotation", "last_done"}},
    ),
    # Called by code, never by the model: listed so the page shows only these keys.
    "planned_day": (
        {"ok", "no_rows"},
        {"status", "day_id", "day_name", "exercises"},
        {
            "exercises": {
                "exercise_id",
                "exercise_name",
                "timed",
                "target_reps",
                "last_on",
                "last_sets",
            }
        },
    ),
    "latest_record": ({"ok", "no_rows"}, {"status", "latest_on"}, {}),
}
_SET_KEYS = {"set_number", "weight_kg", "reps"}

ALREADY_HAVE = "you already have this result this turn; use it"


def safety_guard(prompt: str) -> bool:
    """Allow the prompt, or raise SafetyRefused with the fixed refusal."""
    reply = screen(prompt)
    if reply is not None:
        raise SafetyRefused(reply)
    return True


def _is_session(result: dict[str, Any]) -> bool:
    return "exercise_id" in result and "matches" not in result


def _known_ids(results: list[dict[str, Any]]) -> set[int]:
    """Exercise ids these tool results name: find matches and session ids."""
    ids = {m["exercise_id"] for r in results for m in r.get("matches", [])}
    return ids | {r["exercise_id"] for r in results if _is_session(r)}


def tool_call_guard(ctx: RunContext["AgentDeps"], call: ToolCallInfo) -> GuardrailResult:
    """Send back a call with an id no tool returned, or one already answered this turn."""
    evidence = ctx.deps.evidence
    if call.name == "find_exercise_tool":
        query = str(call.args["query"]).strip().casefold()
        if any(e.get("query", "").strip().casefold() == query for e in evidence):
            return GuardrailResult.retry(ALREADY_HAVE)
    elif call.name == "workout_days_tool":
        if any("days" in e for e in evidence):
            return GuardrailResult.retry(ALREADY_HAVE)
    elif call.name == "get_last_session_tool":
        exercise_id = call.args["exercise_id"]
        if any(_is_session(e) and e["exercise_id"] == exercise_id for e in evidence):
            return GuardrailResult.retry(ALREADY_HAVE)
        # Earlier turns count too: "2" answers the AskWhich from the turn before.
        if exercise_id not in _known_ids(evidence + tool_results_in(ctx.messages)):
            return GuardrailResult.retry(
                f"exercise_id {exercise_id} did not come from find_exercise; "
                "call find_exercise first"
            )
    return GuardrailResult.allow()


def page_result(tool: str, result: dict[str, Any]) -> dict[str, Any]:
    """A tool result cut down to its contract keys, for the chat page.

    The model's tools already pass tool_result_guard; this gives a code-called result
    (planned_day, latest_record) the same privacy line. Unknown tools show only a status.
    """
    contract = _CONTRACT.get(tool)
    if contract is None:
        return {"status": result.get("status", "ok")}
    _, keys, item_keys = contract
    shown = {k: v for k, v in result.items() if k in keys}
    for list_key, allowed in item_keys.items():
        items = [
            {k: v for k, v in item.items() if k in allowed} for item in shown.get(list_key, [])
        ]
        for item in items:
            if "last_sets" in item:
                item["last_sets"] = [
                    {k: v for k, v in s.items() if k in _SET_KEYS} for s in item["last_sets"]
                ]
        if list_key in shown:
            shown[list_key] = items
    return shown


def tool_result_guard(ctx: RunContext["AgentDeps"], info: ToolResultInfo) -> bool:
    """Allow a result inside the tool contract; raise ToolBlocked on any break."""
    contract = _CONTRACT.get(info.name)
    if contract is None:
        raise ToolBlocked(info.name, "no contract for this tool")
    statuses, keys, item_keys = contract
    result = info.result
    if not isinstance(result, dict):
        raise ToolBlocked(info.name, "result is not a dict")
    if result.get("status") not in statuses:
        raise ToolBlocked(info.name, f"status {result.get('status')!r} is not allowed")
    extra = set(result) - keys
    for list_key, allowed in item_keys.items():
        for item in result.get(list_key, []):
            extra |= set(item) - allowed
    if extra:
        raise ToolBlocked(info.name, f"keys outside the contract: {sorted(extra)}")
    if len(json.dumps(result)) > MAX_RESULT_CHARS:
        raise ToolBlocked(info.name, f"result too large (over {MAX_RESULT_CHARS} characters)")
    return True


def answer_only_from_this_turn(ctx: RunContext["AgentDeps"], output: object) -> GuardrailResult:
    """Send back a choice that cites what no tool returned this turn."""
    # The agent's output_type admits only AgentOutput; the guardrail API types it as object.
    problem = check_output(cast(AgentOutput, output), ctx.deps.evidence)
    if problem is not None:
        return GuardrailResult.retry(problem)
    return GuardrailResult.allow()
