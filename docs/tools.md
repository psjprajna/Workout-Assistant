# Tools

Companion to `docs/agentic-workflow.md`. Draft, 2026-10-06.

## 1. What a tool is in this design

A tool is one typed async Python function with two callers: code, when the fast path runs a fixed §20 plan, and the generative model, when the slow path (D5) lets it choose. Same function, same contract, one implementation. In Pydantic AI it is a function on a `FunctionToolset`; the docstring is the description the model reads, the signature is the argument schema, and `RunContext[Deps]` carries the repositories and the session.

Tools are where the design's guarantees live, because they are the only way facts reach the model:

- **Facts come from tools only.** A tool returns SQL rows with ids, units and dates, or labeled web text. The model never sees the raw export and never computes a number: arithmetic is a tool too.
- **Failures are values.** A tool returns `no_rows`, `ambiguous` (with options), `conflict` (with both values) or `unavailable`. It never raises into the model. The model's job on a bad status is to ask or to say so; the evidence check enforces that.
- **Read-only, idempotent, bounded.** No tool writes (§2). Calling one twice is harmless. Each has its own timeout (§23) and returns at most a few hundred tokens.
- **Privacy boundary.** The repositories behind the tools have no query path to the date of birth or body measurements. Nothing a tool returns can contain them.
- **Evidence by construction.** A `WrapperToolset` appends every call (name, arguments, status, row ids) to the evidence record. Verify checks every number in the reply against the union of tool outputs.
- **Scoped.** The toolset a run sees is filtered by Jev's `kind` through a `prepare` hook. Fewer definitions means fewer tokens per request and no calls to irrelevant tools. `UsageLimits(request_limit=6)` caps the loop.
- **`ask_user` is a deferred tool.** Calling it ends the run with a `DeferredToolRequests` result; the question is stored per chat; the next message resumes with the answer as the tool result. The pause is a tool call, not a special path.

## 2. Result envelope

Every tool returns the same shape, so the model learns one pattern and the evidence logger needs one case:

```python
class ToolResult[T](BaseModel):
    status: Literal["ok", "no_rows", "ambiguous", "conflict", "unavailable"]
    data: T | None = None
    options: list[str] = []        # ambiguous: what to ask the user to pick from
    rows: list[int] = []           # ids behind data, for the evidence record
    data_range: tuple[date, date]  # freshness, §4: the export's span
    note: str = ""                 # one line the model may quote, e.g. "no sets in that range"
```

`ok` carries typed data. Every other status carries `note`, and `ambiguous` carries `options`. Code decides what happens next: `ambiguous` and `conflict` become an Ask, `no_rows` becomes an Ask or a refusal per §23, `unavailable` becomes a system error.

## 3. Inventory

Derived from §19 and §20. Fourteen tools in five groups.

Built so far (2026-10-14): `find_exercise`, `last_session` (as `get_last_session`), and `next_rotation_day` with `planned_exercises` as `workout_days` (every plan day with the date it was last done, inferred from the logged exercises, and the suggested next day) and `planned_day` (one day's exercises, targets and last sessions). The model calls `workout_days`; code calls `planned_day` once the model has chosen a day, so the long list costs no model tokens.

**Retrieval, read-only SQL and Qdrant**

| Tool | Arguments | `data` on ok | Non-ok statuses |
|---|---|---|---|
| `next_rotation_day` | `weekday: Weekday` (tie-break) | `{day_id, name, last_done, candidates}` least recently done of the six rotation days, Arms excluded | `ambiguous` when tied; `no_rows` with no history |
| `planned_exercises` | `day_id: int` | `[{exercise_id, name, sets, target_reps: [int]}]` joined on `belongplan = days._id` | `no_rows` |
| `find_exercise` | `phrase: str` | `[{exercise_id, name}]`: exact name, whole-word match on names and other names (`exercise_aliases`), muscle tags, spelling fixes (pg_trgm), then Qdrant meaning matches at or above `MIN_SCORE` | `ok` only for one exact name, or one exact other name with nothing else matching, or one whole-word match on a non-muscle phrase; `ambiguous` for several, for muscle phrases, and for every spelling or meaning match, even one; `no_rows` when nothing is close or the phrase is about a day or workout, never a near guess |
| `last_session` | `exercise_id: int` | `{date, sets: [{weight_kg, reps}], day_name}` latest session containing it | `no_rows` |
| `sets_in_range` | `exercise_id: int, start: date, end: date` | `[{date, weight_kg, reps}]` | `no_rows` (which range is empty goes in `note`) |
| `progression_rule` | `exercise_id: int` | `{tag: compound \| isolation \| assisted, step_kg, miss_policy}` tag and rule together, one lookup instead of two | `no_rows` when untagged: no recommendation, say why |
| `personal_record` | `exercise_id: int` | `{record_kg, best_logged_kg, record_date}` records table and best logged set together | `conflict` when they disagree (both values in options); `no_rows` |
| `exercise_catalog` | `muscle_group: MuscleGroup \| None, tag: Tag \| None` | `[{exercise_id, name, tag, last_done}]` from her 171 logged exercises only | `no_rows` |

**Computation, pure code over retrieval (§12: calculations are code)**

| Tool | Arguments | `data` on ok | Non-ok statuses |
|---|---|---|---|
| `recommend_weight` | `exercise_id: int` | `{next_kg, last_kg, reps, target_reps, tag, step_kg, rule_applied}` set-by-set rule from §3 | `no_rows` listing the missing input (last weight, reps, target, rule) so the model asks for exactly that |
| `targets_hit` | `exercise_id: int` | `{per_set: [{target, done, hit}], all_hit}` | `no_rows` when no target set or not logged |
| `compare_periods` | `exercise_id: int, a: (date, date), b: (date, date)` | `{best_set_a, best_set_b, volume_a, volume_b, change}` | `no_rows` naming the empty range |

These three call the retrieval tools internally and log them, so a single model call yields a full evidence record.

**Web, allowlisted**

| Tool | Arguments | `data` on ok | Non-ok statuses |
|---|---|---|---|
| `form_cues` | `exercise_id: int` | `{bullets: [str], url, source: "ACE" \| "Jefit"}` page fetched from the name, summarised by D4, labeled | `unavailable` on fetch failure or timeout; never a fallback to model knowledge (§8) |

**Session**

| Tool | Arguments | `data` on ok | Non-ok statuses |
|---|---|---|---|
| `last_recommendation` | none | the stored evidence record | `ambiguous` when two are open (options = their summaries); `no_rows` when expired |

**Interaction, deferred**

| Tool | Arguments | Effect |
|---|---|---|
| `ask_user` | `question: str, options: list[str] = []` | Ends the run. Code stores the pending question with the evidence so far. The next message is matched against `options` by code, then by Jev (D2), and resumes the run with the answer as this tool's result. |

Not tools: the safety screen (runs before any model), Jev decisions (D1, D2), phrasing (D3), the evidence check and verify (code after the loop), and anything that writes.

## 4. Scoping by question type

The fast path never consults the model, so scoping only matters on the slow path. The `prepare` hook filters by Jev's `kind` (a `list[Literal]`, since a question can span two types):

| kind | Tools visible |
|---|---|
| tomorrow / weight | `next_rotation_day`, `planned_exercises`, `find_exercise`, `recommend_weight`, `ask_user` |
| last_time | `find_exercise`, `last_session`, `ask_user` |
| targets | `find_exercise`, `targets_hit`, `ask_user` |
| compare | `find_exercise`, `compare_periods`, `ask_user` |
| record | `find_exercise`, `personal_record`, `ask_user` |
| form | `find_exercise`, `form_cues`, `ask_user` |
| out_of_plan | `exercise_catalog`, `form_cues`, `ask_user` |
| why | `last_recommendation` |
| other | all retrieval and computation tools, `ask_user`; still 6 calls max |

## 5. Design rules for writing a tool

1. One sentence of purpose in the docstring, one sentence on when not to call it. No examples; the schema is the example.
2. Arguments are ids or `Literal`s wherever possible. Free-text arguments only for `find_exercise` and `ask_user`.
3. Return the envelope. Put what the user needs to hear on failure in `note`, not in an exception message.
4. Log, do not trust: the wrapper records the call; the model's summary of a result is never evidence.
5. Compose downward only: computation tools call retrieval tools; no tool calls an agent except `form_cues` calling D4.
6. Every tool has a unit test against real Postgres and Qdrant (§25), including one for each non-ok status, and a `FunctionModel` script exercising it inside the loop.
7. Measure: the Langfuse trace shows tokens per tool definition and per result. A tool whose result routinely exceeds 300 tokens gets a narrower return type.

## 6. Open points

- Whether `recommend_weight` should take the user-supplied values held in the session (§4 freshness) as an override argument, or read them from `Deps`. Leaning `Deps`: the model should not pass numbers.
- Whether `form_cues` belongs on the slow path at all, or only on the fast path; the page text never enters the model loop either way.
- `exercise_catalog` filter vocabulary: `MuscleGroup` needs a fixed list from the import.
