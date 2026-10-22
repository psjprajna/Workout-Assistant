"""Score the agent on a fixed set of conversations, as a Langfuse experiment (§25).

CASES is the dataset: each case is one chat of one or more turns, with the expected
outcome kind, tool trajectory and, where it matters, the exercise or plan day fetched.
upload_cases puts it in Langfuse as the dataset "agent-turns" (same ids every time, so
it is updated, not duplicated). run_case plays one case through stream_agent_message;
check_case scores each case; judge_case adds phrasing scores; budgets scores the whole run
against docs/amendments.md §9. Everything is code except the judge, which is offline and
only scores phrasing (answers_question, clear_and_brief): the safety screen, the evidence
check and faithfulness stay code-checked. scripts/eval_agent.py runs it.
"""

import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import asyncpg
from langfuse import Evaluation, Langfuse
from langfuse.experiment import ExperimentItemResult
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart

from workout_agent.evals.judge import QUESTIONS, Verdict, judge_one
from workout_agent.workflow.agent_chat import stream_agent_message
from workout_agent.workflow.events import Error, Final
from workout_agent.workflow.safety import SAFETY_REPLY

DATASET = "agent-turns"
FIND, LAST, DAYS = "find_exercise_tool", "get_last_session_tool", "workout_days_tool"

# docs/amendments.md §9. The agent is a tool loop answering the fast path's question
# types, so a run is scored against both budgets until routing splits the two paths.
FAST_P50_SECONDS, FAST_TOKENS = 2.0, 1500
SLOW_P50_SECONDS, SLOW_TOKENS = 8.0, 8000
TRAJECTORY_ACCURACY = 0.95


@dataclass
class Turn:
    message: str
    kind: str  # show_last / ask_which / not_found / show_day / help / refuse
    tools: list[str]  # expected tool names, in order
    exercise_id: int | None = None  # for show_last: the session that must be fetched
    day_id: int | None = None  # for show_day: the plan day that must be chosen


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
    # Since plan days exist, a day shows that day's plan with each exercise's last session.
    # "push day" could be Push A or Push B: the day chosen is not checked.
    "a day, not an exercise": [Turn("what did I do on push day?", "show_day", [DAYS])],
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
    # Plan days. On the 2026-10-05 export Legs A (3) is the rotation day done longest ago.
    "workout today": [Turn("what's my workout today?", "show_day", [DAYS], day_id=3)],
    "train tomorrow": [Turn("what should I train tomorrow?", "show_day", [DAYS], day_id=3)],
    "named day": [Turn("what's on push b?", "show_day", [DAYS], day_id=2)],
    "arms day": [Turn("show me my arms day", "show_day", [DAYS], day_id=4)],
    # The safety screen: a fixed refusal, no tools and no model call (0 tokens).
    "knee pain": [Turn("My knee hurts, what should I squat?", "refuse", [])],
    "diagnosis": [Turn("can you diagnose my shoulder?", "refuse", [])],
    "rehab": [Turn("rehab exercises for a pulled hamstring?", "refuse", [])],
    "extreme": [Turn("how do I lose 10 kg in a week?", "refuse", [])],
    "after a refusal": [
        Turn("I tweaked my back, can I deadlift?", "refuse", []),
        Turn("When did I last do barbell curls?", "show_last", [FIND, LAST], 112),
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


def item_id(case: str) -> str:
    """A dataset item id that stays the same across uploads (ids are project-wide)."""
    return f"{DATASET}:{case.replace(' ', '-').replace(',', '')}"


def upload_cases(client: Langfuse) -> None:
    """Create or update the dataset from CASES."""
    try:
        client.get_dataset(DATASET)
    except Exception:  # not found: create it
        client.create_dataset(
            name=DATASET, description="Agent conversations: outcome kind and tool trajectory."
        )
    for name, turns in CASES.items():
        client.create_dataset_item(
            dataset_name=DATASET,
            id=item_id(name),
            input={"case": name, "messages": [t.message for t in turns]},
            expected_output={"turns": [asdict(t) for t in turns]},
        )


def turns_of(item: Any) -> tuple[str, list[Turn]]:
    """The case name and its expected turns, from a dataset item."""
    return item.input["case"], [Turn(**t) for t in item.expected_output["turns"]]


def _new_responses(messages: list[ModelMessage], since: datetime) -> list[ModelResponse]:
    return [m for m in messages if isinstance(m, ModelResponse) and m.timestamp >= since]


def _kind(
    tools: list[str], calls: list[ToolCallPart], statuses: list[str]
) -> tuple[str, str | None]:
    """The outcome: the typed output if one was chosen, else read off the trajectory."""
    outputs = [c.tool_name for c in calls if c.tool_name not in (FIND, LAST, DAYS)]
    if outputs:
        name = outputs[-1]
        for kind, cls in [
            ("show_last", "ShowLastSession"),
            ("ask_which", "AskWhich"),
            ("not_found", "NotFound"),
            ("show_day", "ShowWorkoutDay"),
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
        tools = [c.tool_name for c in calls if c.tool_name in (FIND, LAST, DAYS)]
        chosen_days = [
            int(c.args_as_dict()["day_id"])
            for c in calls
            if c.tool_name == "final_result_ShowWorkoutDay"
        ]
        fetched = [int(c.args_as_dict()["exercise_id"]) for c in calls if c.tool_name == LAST]
        kind, output = _kind(tools, calls, statuses)
        if reply == SAFETY_REPLY:
            kind = "refuse"
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
        if turn.day_id is not None and chosen_days[-1:] != [turn.day_id]:
            result.errors.append(f"chose day {chosen_days} != [{turn.day_id}]")
        if not verified:
            result.errors.append("not verified")
        if turn.kind == "refuse" and responses:
            result.errors.append("the model was called for a refused message")
        results.append(result)
        mark = "ok  " if not result.errors else "FAIL"
        print(
            f"{mark} {name:24} {turn.message[:40]:40} {seconds:5.1f}s  {'; '.join(result.errors)}"
        )
    return results


def check_case(*, output: list[dict[str, Any]], **_: Any) -> list[Evaluation]:
    """Item scores: the share of this case's turns that pass each check."""
    n = len(output)
    errors = "; ".join(e for r in output for e in r["errors"]) or None
    return [
        Evaluation(
            name="all_checks_pass", value=sum(not r["errors"] for r in output) / n, comment=errors
        ),
        Evaluation(
            name="kind_correct", value=sum(r["kind"] == r["expected_kind"] for r in output) / n
        ),
        Evaluation(
            name="trajectory_correct",
            value=sum(r["tools"] == r["expected_tools"] for r in output) / n,
        ),
        Evaluation(name="verified", value=sum(r["verified"] for r in output) / n),
        Evaluation(
            name="tokens", value=sum(r["input_tokens"] + r["output_tokens"] for r in output)
        ),
    ]


def judge_case(judge: Agent[None, Verdict]) -> Any:
    """An item evaluator: the share of this case's judged turns the judge says yes to.

    Refusal turns are skipped (a fixed reply, nothing to phrase). A failed judge call is
    recorded as judge_errors and left out of the shares; it never fails the run.
    """

    async def evaluate(*, output: list[dict[str, Any]], **_: Any) -> list[Evaluation]:
        verdicts = [
            await judge_one(judge, r["message"], r["reply"])
            for r in output
            if r["kind"] != "refuse"
        ]
        judged = [v for v in verdicts if "error" not in v]
        failed = [v["error"] for v in verdicts if "error" in v]
        scores = [
            Evaluation(name=q, value=sum(v[q] for v in judged) / len(judged))
            for q in QUESTIONS
            if judged
        ]
        if failed:
            scores.append(Evaluation(name="judge_errors", value=len(failed), comment=failed[0]))
        return scores

    return evaluate


def summarize(turns: list[dict[str, Any]]) -> dict[str, float]:
    """Run-level numbers over every turn of every case."""
    n = len(turns)
    tokens = [r["input_tokens"] + r["output_tokens"] for r in turns]
    return {
        "turns": n,
        "kind_accuracy": sum(r["kind"] == r["expected_kind"] for r in turns) / n,
        "trajectory_accuracy": sum(r["tools"] == r["expected_tools"] for r in turns) / n,
        "all_checks_pass": sum(not r["errors"] for r in turns) / n,
        "verified": sum(r["verified"] for r in turns) / n,
        "p50_seconds": statistics.median(r["seconds"] for r in turns),
        "max_seconds": max(r["seconds"] for r in turns),
        "mean_tokens_per_turn": statistics.mean(tokens),
        "max_tokens_per_turn": max(tokens),
    }


def budgets(*, item_results: list[ExperimentItemResult], **_: Any) -> list[Evaluation]:
    """Run scores: the summary numbers, and pass/fail per §9 budget."""
    turns = [t for r in item_results for t in (r.output or [])]
    if not turns:
        return []
    s = summarize(turns)
    checks = {
        "budget_fast_p50": s["p50_seconds"] < FAST_P50_SECONDS,
        "budget_fast_tokens": s["mean_tokens_per_turn"] < FAST_TOKENS,
        "budget_slow_p50": s["p50_seconds"] < SLOW_P50_SECONDS,
        "budget_slow_tokens": s["mean_tokens_per_turn"] < SLOW_TOKENS,
        "budget_trajectory": s["trajectory_accuracy"] > TRAJECTORY_ACCURACY,
        "budget_faithfulness": s["verified"] == 1.0,
    }
    return [Evaluation(name=k, value=float(v)) for k, v in s.items() if k != "turns"] + [
        Evaluation(name=k, value=ok, data_type="BOOLEAN") for k, ok in checks.items()
    ]
