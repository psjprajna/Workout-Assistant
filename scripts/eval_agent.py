"""Score the agent on a fixed set of conversations: outcome, tool trajectory, latency, tokens.

Usage: uv run python scripts/eval_agent.py LABEL
Runs every case through stream_agent_message with the real model on your real data, in a
transaction that is rolled back (no chats are saved), and writes
evals/results/agent-<date>-<LABEL>.json. Search by meaning is off, so find_exercise
answers the same way every run and only the model varies.

A stopgap until the Langfuse dataset (docs/amendments.md §26): same scores, no traces.
"""

import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import asyncpg
from pydantic_ai import capture_run_messages
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart

from workout_agent.agents.workout_agent import build_agent
from workout_agent.settings import Settings
from workout_agent.workflow.agent_chat import stream_agent_message
from workout_agent.workflow.events import Error, Final

FIND, LAST = "find_exercise_tool", "get_last_session_tool"


@dataclass
class Turn:
    message: str
    kind: str  # show_last / ask_which / not_found / help
    tools: list[str]  # expected tool names, in order
    exercise_id: int | None = None  # for show_last: the session that must be fetched


# Expected answers come from find_exercise on the real data (search by meaning off).
CASES: dict[str, list[Turn]] = {
    "exact name": [
        Turn("What did I lift last time on barbell bench press?", "show_last", [FIND, LAST], 2)
    ],
    "several matches": [Turn("What did I lift last time on bench press?", "ask_which", [FIND])],
    "name with brackets": [
        Turn("last time I did hack squat reverse position?", "show_last", [FIND, LAST], 1199)
    ],
    "incline dumbbell": [
        Turn("last dumbbell incline bench press session?", "show_last", [FIND, LAST], 31)
    ],
    "misspelt, one option": [
        Turn("last time I did dumbell incline bench pres?", "ask_which", [FIND])
    ],
    "unknown": [Turn("last time I did zzz?", "not_found", [FIND])],
    "no set details": [Turn("when did I last go walking?", "show_last", [FIND, LAST], 322)],
    "same name twice": [Turn("last time I did ab rollout?", "ask_which", [FIND])],
    "barbell curl": [Turn("When did I last do barbell curls?", "show_last", [FIND, LAST], 112)],
    "lat pulldown": [Turn("what did I do on lat pulldown last time?", "ask_which", [FIND])],
    "plank": [Turn("last plank?", "show_last", [FIND, LAST], 631)],
    "a day, not an exercise": [Turn("what did I do on push day?", "not_found", [FIND])],
    "out of scope": [Turn("what's the weather tomorrow?", "help", [])],
    "greeting": [Turn("hi", "help", [])],
    "pick by number": [
        Turn("What did I lift last time on bench press?", "ask_which", [FIND]),
        Turn("1", "show_last", [LAST], 2),
    ],
    "pick deadlift": [
        Turn("last deadlift session?", "ask_which", [FIND]),
        Turn("2", "show_last", [LAST], 199),
    ],
    "say yes": [
        Turn("last time I did dumbell incline bench pres?", "ask_which", [FIND]),
        Turn("yes", "show_last", [LAST], 31),
    ],
}


@dataclass
class TurnResult:
    case: str
    message: str
    expected_kind: str
    expected_tools: list[str]
    kind: str
    tools: list[str]
    fetched_ids: list[int]
    output: str | None  # the typed output's tool name, if the model chose one
    reply: str
    verified: bool
    seconds: float
    input_tokens: int
    output_tokens: int
    errors: list[str] = field(default_factory=list)


def _new_responses(messages: list[ModelMessage], since: datetime) -> list[ModelResponse]:
    return [m for m in messages if isinstance(m, ModelResponse) and m.timestamp >= since]


def _kind(
    tools: list[str], calls: list[ToolCallPart], statuses: list[str]
) -> tuple[str, str | None]:
    """The outcome: the typed output if one was chosen, else read off the trajectory."""
    outputs = [c.tool_name for c in calls if c.tool_name not in (FIND, LAST)]
    if outputs:
        name = outputs[-1]
        for kind, cls in [
            ("show_last", "ShowLastSession"),
            ("ask_which", "AskWhich"),
            ("not_found", "NotFound"),
            ("help", "Help"),
        ]:
            if cls in name:
                return kind, name
        return name, name
    if tools and tools[-1] == LAST:
        return "show_last", None
    if statuses and statuses[-1] == "ambiguous":
        return "ask_which", None
    if statuses and statuses[-1] == "no_rows":
        return "not_found", None
    return "help", None


async def run_case(
    conn: asyncpg.Connection, agent: Any, name: str, turns: list[Turn]
) -> list[TurnResult]:
    results = []
    chat_id = f"eval-{name}"
    for turn in turns:
        since = datetime.now(UTC)
        start = time.perf_counter()
        statuses: list[str] = []
        reply, verified = "", False
        with capture_run_messages() as messages:
            try:
                async for event in stream_agent_message(conn, chat_id, turn.message, agent):
                    if event.kind == "tool_end":
                        statuses.append(event.status)
                    elif isinstance(event, Final):
                        reply, verified = event.reply, event.verified
                    elif isinstance(event, Error):
                        reply = event.message
            except Exception as e:  # a failed run is a scored failure, not a crash
                reply = f"{type(e).__name__}: {e}"
        seconds = time.perf_counter() - start
        responses = _new_responses(messages, since)
        calls = [p for r in responses for p in r.parts if isinstance(p, ToolCallPart)]
        tools = [c.tool_name for c in calls if c.tool_name in (FIND, LAST)]
        fetched = [int(c.args_as_dict()["exercise_id"]) for c in calls if c.tool_name == LAST]
        kind, output = _kind(tools, calls, statuses)
        result = TurnResult(
            case=name,
            message=turn.message,
            expected_kind=turn.kind,
            expected_tools=turn.tools,
            kind=kind,
            tools=tools,
            fetched_ids=fetched,
            output=output,
            reply=reply,
            verified=verified,
            seconds=round(seconds, 2),
            input_tokens=sum(r.usage.input_tokens for r in responses),
            output_tokens=sum(r.usage.output_tokens for r in responses),
        )
        if kind != turn.kind:
            result.errors.append(f"kind {kind} != {turn.kind}")
        if tools != turn.tools:
            result.errors.append(f"tools {tools} != {turn.tools}")
        if turn.exercise_id is not None and fetched[-1:] != [turn.exercise_id]:
            result.errors.append(f"fetched {fetched} != [{turn.exercise_id}]")
        if not verified:
            result.errors.append("not verified")
        results.append(result)
        mark = "ok  " if not result.errors else "FAIL"
        print(
            f"{mark} {name:24} {turn.message[:40]:40} {seconds:5.1f}s  {'; '.join(result.errors)}"
        )
    return results


async def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    agent = build_agent(Settings())
    conn = await asyncpg.connect(Settings().database_url)
    results: list[TurnResult] = []
    try:
        tx = conn.transaction()
        await tx.start()
        try:
            for name, turns in CASES.items():
                results += await run_case(conn, agent, name, turns)
        finally:
            await tx.rollback()  # leave no eval chats behind
    finally:
        await conn.close()

    n = len(results)
    summary = {
        "turns": n,
        "kind_accuracy": sum(r.kind == r.expected_kind for r in results) / n,
        "trajectory_accuracy": sum(r.tools == r.expected_tools for r in results) / n,
        "all_checks_pass": sum(not r.errors for r in results) / n,
        "verified": sum(r.verified for r in results) / n,
        "p50_seconds": statistics.median(r.seconds for r in results),
        "max_seconds": max(r.seconds for r in results),
        "mean_tokens_per_turn": statistics.mean(r.input_tokens + r.output_tokens for r in results),
        "max_tokens_per_turn": max(r.input_tokens + r.output_tokens for r in results),
    }
    out = Path("evals/results") / f"agent-{date.today()}-{label}.json"
    out.write_text(
        json.dumps({"summary": summary, "turns": [asdict(r) for r in results]}, indent=2)
    )
    print(json.dumps(summary, indent=2))
    print(f"written to {out}")


if __name__ == "__main__":
    asyncio.run(main())
