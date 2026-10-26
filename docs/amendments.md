# Amendments to "Workout RAG Assistant, Requirements and Design" (baseline October 2026)

Date 2026-10-06. Replacement text per section, to be ported into the PDF source. Decisions behind them: `docs/framework-decision.md` (Pydantic AI as the spine), `docs/agentic-workflow.md` (the workflow), `docs/tools.md` (the tools). Part 1 sections not listed are unchanged, including §2 scope: still one user, read-only, no memory across chats in this version.

## §9 Non-Functional Requirements

Replace the bullet "The language model runs locally. Personal data stays on the machine and out of version control." with:

> - Models are hosted APIs: TypeSafe Jev for typed decisions, one open-weight generative model on OpenRouter, and an OpenRouter embedding model for exercise-name matching. Only the rows a tool returns for the current question leave the machine. The raw export, the date of birth and the body measurements never do. Provider data collection is denied, and providers that train on API input are excluded. The data folder stays out of version control.

Add:

> - Budgets, measured in Langfuse on every eval run and tuned like the timeouts in §23: fast path p50 under 2 s and under 1.5k model tokens per turn; slow path p50 under 8 s and under 8k tokens; tool trajectory accuracy over 95% on the eval set; faithfulness 100%, code-checked.

## §12 Design Principles

Replace the first bullet with:

> - The language model handles language and bounded decisions: understanding the question, picking among typed options, choosing the next tool inside a bounded loop, and phrasing the answer. Facts, checks, calculations and bounds are code.

Replace the last bullet with:

> - The pipeline is a fixed graph. The model decides only at named decision points, and each decision is a typed choice from a fixed set: the route, ask or answer, the next tool inside a bounded loop. Code owns facts, safety, verification, state and the bounds. No decision point can skip the safety screen, the evidence check or verification.

The other four bullets stand.

## §13 Architecture

Replaced by the two diagrams in `docs/agentic-workflow.md`: the workflow with five decision points and the interface layer with the Pydantic AI objects.

## §14 Components and Technology

| Component | What it does | Technology |
|---|---|---|
| Agentic workflow | A fixed graph: safety screen, route, fast path (the §20 plan for the question type, run by code) or slow path (a bounded tool loop), evidence check, decide, respond, verify. One entry point, `handle_message(session_id, text)`. | Plain async stages; Pydantic AI 2.54 agents at the five decision points |
| Decision model | Question type, injury class, which exercise of the Qdrant candidates, which date reading, which stored option a reply means. One batched request, ~100 ms. | TypeSafe Jev through Pydantic AI's `TypeSafeModel` (`typesafe:jev-latest`) |
| Generative model | Phrase the answer from evidence, summarise a fetched page, drive the slow-path tool loop. Reasoning off. | Qwen3.5-35B-A3B on OpenRouter, pinned provider, pending the first eval (§25) |
| Gateway | The only code that runs an agent: timeout, concurrency limit, retries, and conversion of framework exceptions into typed results. | One `gateway()` around `Agent.run`; `ModelSettings`, `ConcurrencyLimitedModel`, `UsageLimits` |
| Tools | Typed async functions, the only path facts take to the model; same functions called by code on the fast path. Result envelope, failures as values, read-only, per-call timeout, every call logged into the evidence record. | `FunctionToolset` + `WrapperToolset`; `ask_user` as a deferred tool; see `docs/tools.md` |
| Session state | Per chat: capped messages, pending question with its evidence so far, last recommendation, user-supplied values. Serial per chat. Saved only on success. | DBOS workflow per chat on Postgres, with Redis as the competing option (§24) |
| Embedding model | Exercise-name matching. | `openai/text-embedding-3-large` on OpenRouter |
| Evals and tracing | Dataset, experiments, code evaluators, traces by session id, token and latency budgets. | Langfuse, self-hosted; OTel from `Agent.instrument_all()`; standard logging keeps the session id |

Chat UI, API layer, repositories, PostgreSQL, vector store, import pipeline, web lookup step, settings and logging: unchanged. The former "Language model gateway" and "Chat model" rows are replaced by the four rows above.

## §15 How a Request Flows

Replace the third bullet with:

> - If a question is pending, the reply is matched against the stored options by code, then by the decision model as a fallback. On the fast path the plan resumes at the stage that asked. On the slow path the tool loop resumes with the reply as the result of its `ask_user` call. The decision model is not run again on a resume.

Replace the fourth bullet with:

> - Otherwise: safety screen; one decision-model call that routes the question, flags injury, and picks the exercise and date reading among code-derived candidates; then either the fixed plan for that question type (code, groups run together) or, for questions the plans do not cover, a tool loop bounded to six calls with a toolset scoped to the question; then the evidence check, a question, a refusal or a decision, then the reply (template first, model if needed) and its verification.

## §16 Pipeline Stages and §18 Where the Language Model Is Used

Replaced by the decision-point table in `docs/agentic-workflow.md` (D1 route, D2 pick, D3 phrase, D4 summarise, D5 tool loop). The model is still not used in the safety screen, the evidence check, the decision, any calculation, or session handling; calculations the model needs are tools.

## §20 Retrieval Plans by Question Type

Unchanged as the fast path. Add one row:

| Question type | Steps | If something is missing or unclear |
|---|---|---|
| Other, or spanning two types | Slow path: the generative model chooses among the tools scoped to the detected types, at most six calls, `ask_user` when a tool returns ambiguous or conflict. The tool-call log is the evidence; the evidence check and verify run as for every plan. | Usage limit reached: system error, nothing invented. Low decision confidence: ask. |

## §22 Where State Lives

Replace the "Messages and structured state" row with:

| State | Where | How long |
|---|---|---|
| Messages and structured state | DBOS: one durable workflow per chat, state in its Postgres system database; the chat is serial by construction and a pending question survives a restart. Competing option: Redis, one key per chat with TTL (the original design). | Until idle timeout (a `recv` timeout in DBOS, key expiry in Redis) or a new chat |

Replace the last paragraph with: "There is no module-level mutable state. With DBOS, per-chat serialisation and recovery come from the workflow; with Redis, the per-chat lock is a Redis lock with an expiry."

## §23 How Errors Move Through the System

Add two rows:

| Failure | Handling | What the user sees |
|---|---|---|
| Tool-call limit reached | The loop stops; the evidence so far is kept; system error. Never a guess. | "I couldn't finish that one; try a narrower question." |
| Low decision confidence | Ask, with the options the decision was between. | A question |

## §24 Technology Choices and Why

Replace or add these rows:

| Layer | Choice | Why | Considered |
|---|---|---|---|
| Agent framework | Pydantic AI 2.54 | Owns the decision shape, toolsets with per-call logging, pause as a deferred tool, exact tool-call bound, one test double; leaves state in the session store. Two spike rounds, `docs/framework-decision.md`. | Plain asyncio (round 1 winner under the old §12), LangGraph (passes, but the checkpointer replaces the session store and keeps failure writes) |
| Decision model | TypeSafe Jev | Typed choices only, ~100 ms, $0.042 per M input, native in Pydantic AI. Accuracy evidence is thin, so the first eval compares it with small-model routing. | A small generative model as classifier |
| Generative model | Qwen3.5-35B-A3B, thinking off, OpenRouter | Only open candidate with published tool benchmarks; pennies a month. Price does not separate candidates at this volume; the eval picks on tool trajectory and faithfulness. | DeepSeek V4.1 Flash, gpt-oss-120b; Haiku 4.5 or GPT-6 Luna as a ceiling check |
| Embedding model | `openai/text-embedding-3-large`, OpenRouter | all-minilm (local, via Ollama) is a small model and undersells exercise-name matching; a hosted model matches phrasing better, and this consolidates onto the provider already used for the generative model. Large over small on the search eval (`evals/results/search-2026-10-12-phase-d2-large.json`): on its own Qdrant ranks a right answer first for 26/28 held-out phrases (small 23/28) and widens the safe `MIN_SCORE` range; median query latency 1.4 s vs 1.2 s (8 calls each), paid only by the few phrases the text stages of `find_exercise` cannot answer. | `text-embedding-3-small` (the previous choice); `all-minilm` through Ollama, local (the original §9 choice); dense + sparse (BM25-style) hybrid fused with RRF, tried 2026-10-12: held-out hit@1 +2 but dev hit@1 77/80 -> 64/80, as common words like "press" outvote exact meaning matches |
| Typo matching | `pg_trgm`, a PostgreSQL extension | Fixes misspelt words ("dumbell" -> dumbbell) in SQL before any embedding call. An extension of the Postgres already in the stack, not a new store. | Sentence-level trigram similarity (measured noisier: "dumbell rows" ranked Romanian Deadlift first) |
| Model hosting | OpenRouter, provider pinned, `require_parameters`, `data_collection: deny` | One API, structured output guaranteed, no training on inputs. | Direct vendor APIs; local Ollama |
| Structured output | Pydantic AI `NativeOutput` and tool output with `retries` | Schema enforced by the provider, validated by Pydantic, retried with the error. | Own gateway over Ollama `format` (the original) |
| Session state | **Two competing options, DBOS first.** DBOS: durable workflow per chat on the Postgres already in the stack; serial per chat and crash recovery by construction; no lock code; keeps the door open to conversation tracking beyond one chat later without a new store. Redis: one key per chat, TTL expiry, fastest reads; the original design. | DBOS first because durability and per-chat serialisation come free and the data is already in Postgres. Redis later as a speed improvement if measured session latency matters. **Now (2026-10-12): plain Postgres tables `chats` and `chat_turns`** (`migrations/0004_chat_history.sql`, `sessions/chat_store.py`), one append-only row per answered turn, written only on success. This is the rejected "Postgres table by hand", taken on purpose for now: there is one user; chat history only has to survive restarts and be listed in the chat page's sidebar; nothing pauses mid-run yet, so DBOS's recovery has nothing to recover; and Postgres is already in the stack. Per-chat serialisation: a second message while one is running is turned away at the start with a Postgres advisory lock on the chat id (`try_lock_chat`, added 2026-10-13 so the user never watches an answer that ends in "busy"), and the `(chat_id, turn_no)` primary key stays as the backstop; either way the turn is asked to resend, never merged. DBOS stays the path for phase 6, when a pending question must survive a restart mid-run; the turn rows then become that workflow's history. | In-memory dict, Postgres table by hand (used for now, reason in this row) |
| Reply streaming | Server-sent events from FastAPI's `StreamingResponse` (`POST /chat/stream`, added 2026-10-13): one event per tool start and end, then the whole reply; `httpx` `client.stream` in the chat page | Nothing showed on screen during the slow path (p50 target 8 s, up to 6 tool calls). SSE needs no new package on either side: FastAPI and httpx already do it. Progress labels come from code, not the model; the reply is still sent whole. | WebSockets (two-way not needed); token streaming (deferred until Langfuse shows the final model call is a large share of p50) |
| Guardrails | Pydantic AI Harness 0.54.0 `InputGuardrail`, `ToolGuardrail`, `OutputGuardrail` (added 2026-10-14, `agents/guardrails.py`). Pinned exactly: it requires exactly `pydantic-ai-slim==2.54.0`, and later Harness versions require slim 2.56. | Rules the prompt only asked for become code at the run's edges: the safety screen (`workflow/safety.py`, regex) runs before any model call; a `get_last_session` id no tool returned, or a repeat call in one turn, goes back to the model; a tool result outside the `docs/tools.md` contract (status, allowlisted keys as the privacy line, size) stops the run; the final typed choice is checked against this turn's evidence. All guards are plain code, no model. Not used: `block` on input (it makes the model answer in text; the chat code writes the refusal instead), `block` on tools (it turns the result into a string), detectors, `approve()`, `hidden=`. | A hand-written `output_validator` (the previous approach, output edge only); pydantic-ai-shields |
| Evals | Langfuse, self-hosted (server 4.x, Python SDK 5, added 2026-10-19) | Datasets, experiments from code with code evaluators, traces by session id, budgets as run-level scores, CI gate. Now: dataset `agent-turns` from `evals/agent.py`, run by `scripts/eval_agent.py`; item scores per case, run scores for each §9 budget. Item scores also include `answers_question` and `clear_and_brief` from an offline reply judge, Sonnet 5.5 through OpenRouter (temperature 0, `data_collection: deny`), which scores phrasing only and never runs in the request path; chosen over `typesafe:jev-latest` because Jev's aq adds little over always-yes (§25). One new dependency, `ragas==0.4.3`, in a separate `eval` uv group (offline only, never imported by the app, never in the request path): it scores faithfulness, answer relevancy, context and agent metrics over the tool results each eval turn saves, which the two phrasing questions do not cover. It does not replace the code-checked `verified` flag or `budget_faithfulness`. Locking it with `pydantic-ai-slim` 2.54.0 needs `pillow>=10.4.0` in the group, a uv override `jiter>=0.16` (its `instructor` caps `jiter` below what `openai>=3.19` needs) and a uv constraint `websockets>=17.2` so `websockets` stays at 17.x. No new store. The Compose stack adds ClickHouse, Redis and MinIO: these are Langfuse's own internals (its event store, queue and blob store), reached only by Langfuse, bound to localhost; not app stores, and this Redis is not the session-state Redis option. Langfuse's own database is a separate `langfuse` database in the app's Postgres. | DeepEval, pydantic-evals |
| Observability | OTel from Pydantic AI into Langfuse; standard logging | Spans per agent run and tool call for free; logs keep the session id. Now (2026-10-19): `observability.py`, one `chat-turn` trace per message with the chat id as the Langfuse session, outcome and `verified` on it; off when the keys are not set. | Logfire |
| Test timeout | `pytest-timeout>=2.4.0`, `dev` group (added 2026-10-26), `timeout = 60` in `[tool.pytest.ini_options]` | A hung test (a real model call, a service that is down) fails by name after 60 s instead of stalling until CI's 15-minute job limit kills the run with no test named. Dev only, never imported by the app. | CI-only `timeout-minutes` (no test name, no local effect) |

Rows for relational database, vector store, API framework, HTTP client, finding web pages, chat UI, test runner, packages, lint, type checking, containers, migrations: unchanged. The "Language model host" and "Chat model" rows are replaced by the rows above. "Failure injection" becomes "FunctionModel scripts and forced timeouts".

## §25 Testing Approach

Replace with:

> - Write tests before the code, then run the whole suite and confirm it passes.
> - Unit tests cover pure logic (progression, evidence checks, unit conversion, day matching, reply verification) and every tool, including one test per non-ok status, against real PostgreSQL and Qdrant containers.
> - Loop tests script the generative model with `FunctionModel` and the decision model with a canned route: fast path, slow path, `ask_user` pause and resume, the tool-call bound, session isolation. The two spike tests are the seeds.
> - Failure tests use scripted models and forced timeouts for: model timeout, decision-model timeout, database timeout, vector store failure, web failure, malformed output, empty retrieval, conflicting records, tool-call limit, low decision confidence, a clarification reply, a change of subject.
> - A synthetic CSV supplies conflict and missing-data cases that the real export cannot.
> - Evals on Langfuse: a dataset of 30 to 50 cases built from the §10 journeys, the failure list above and the synthetic CSV, each with the expected outcome kind, expected tool trajectory, expected evidence rows and, for refusals and asks, the expected text class. Code evaluators score outcome kind, tool trajectory, evidence rows, numbers-in-reply ⊂ evidence, refusal on the safety set, ask on the ambiguity set, zero lookups on "Why?". Run-level evaluators enforce the §9 budgets. A judge is used only for phrasing quality, and never in the request path. Experiments run on every model or prompt change and in CI.
> - First experiments, in order: decision model routing (Jev versus a small generative model) on the sample-message set; generative model choice (Qwen3.5-35B-A3B, DeepSeek V4.1 Flash, gpt-oss-120b) on the full dataset.
> - Reply judge trust check (2026-10-22, `scripts/judge_results.py`, offline only, never in the request path). Two yes/no questions, `answers_question` (aq) and `clear_and_brief` (cb), judged by Sonnet 5.5 (`openrouter`, temperature 0) and by `typesafe:jev-latest`, compared with hand labels in `evals/judge_labels.toml`. Agreement, with an always-yes judge beside it:
>
>   | Labels | n | Sonnet aq | Sonnet cb | Jev aq | Jev cb | always-yes aq / cb |
>   |---|---|---|---|---|---|---|
>   | 24 labels the rubric was tuned on (rubric 3) | 24 | 23 | 23 | 19 | 23 | 16 / 20 |
>   | 9 fresh replies, not seen when the rubric was written | 9 | 9 | 9 | 8 | 9 | 9 / 9 |
>   | 20 real bad replies from the broken runs (labelled by the owner, 2026-10-22) | 20 | 20 | 20 | not run | not run | 0 / 1 |
>
>   Earlier rubrics, out of 48 on the 24: first rubric Sonnet 41, Jev 36; list-aware Sonnet 45, Jev 40; rubric 3 Sonnet 46, Jev 42 (always-yes 36). Files: `evals/results/judge-2026-10-22-*.json`.
>
>   Read these with care. (1) The 24 were used to write the rubric, so those numbers are optimistic. (2) The 9 fresh replies are all good and were labelled by Claude, not the owner (`note = "claude"`), so they show only that a judge does not wrongly say no, and agreement with them is weaker evidence than agreement with owner labels. They cannot show that a judge catches bad replies: always-yes also scores 9/9. The "no" side rests on the 24 labels (8 aq, 4 cb; 6 of the no replies are made-up) and 20 real bad replies from the broken runs, which are all error text or an obvious wrong answer, so subtle phrasing faults are untested. (3) Jev's five `answers_question` misses on the 24 (rubric 3) are a good reply it rejects ("zzz": not in your data; the made-up repeated-options list) or a bad one it accepts (the two old "push day" replies, the invented plank answer); on the fresh 9 it also rejects the correct deadlift pick "2". On aq it is 3 above always-yes on the 24. Sonnet's one miss on each question is the "ab rollout" reply with two identical options: Sonnet says not clear and brief (the same thing twice), the label says clear. The owner's call stands; the made-up "repeated options" label (cb no) contradicts it and is open.
>
>   Use Sonnet 5.5; a stronger model is not needed. On all 53 labels it agrees on 52 of 53 for each question (always-yes 25 and 30); Jev was last run on 40 (34 and 39). Its two misses are both contested: the "ab rollout" reply (Sonnet says the same thing twice, the label says clear) and an old "push day" reply that offers push-ups. A stronger judge cannot fix a label that is itself in doubt. The 20 owner-labelled bad replies are 18 `InterfaceError` strings (the same text under 18 different messages), one "'1' is not in your data" reply to a pick, and one `UnexpectedModelBehavior` string. Only 3 distinct reply texts exist in those runs (44 of the 46 bad turns are the one `InterfaceError` text), so 20/20 means Sonnet rejects an exception string and a wrong answer to a pick, not that it catches subtle phrasing faults. The "1" reply is judged with no earlier turn shown, yet Sonnet and Jev both called it not an answer. Next step, if more certainty is wanted: try one judge from another family with `--judge openrouter:<slug>` (no code change), and add real subtle-bad replies, which the current data does not have.
>
> - A concurrency script runs many chats at once and checks that they never mix state and that nothing blocks.

## §26 Folder Structure

`src/workout_agent/`: `agents/` (route, pick, phrase, summary, loop; `gateway.py`), `tools/` (one module per group, the envelope, the evidence wrapper), `workflow/` (safety, fast path plans, slow path, check, decide, verify, `handle_message`), `sessions/` (DBOS workflow per chat; `redis_store.py` behind the same interface if Redis is taken up; until phase 6, `chat_store.py` over plain Postgres tables, see §24), `evals/` (dataset, evaluators, experiment runner). `llm/` and `agent/` from the original are replaced by these. Everything else unchanged.

## §27 Build Order

| Phase | What gets built | Done when |
|---|---|---|
| 1. Foundation | uv project, Compose (Postgres, Qdrant, Langfuse), DBOS system database on Postgres, schemas, migrations. | Containers start; schemas validate; Langfuse receives a trace. |
| 2. Import | Unchanged. | The real export loads; bad rows reported. |
| 3. Tools | Repositories, the fourteen tools with the envelope, evidence wrapper, unit tests per status against real containers; fast-path plans as code over the tools. | Every tool returns typed results and failures; plans produce evidence records. |
| 4. Decisions | Route and pick agents on Jev; the sample-message set; first experiment: Jev versus small-model routing. | Routing accuracy measured; D1 model chosen. |
| 5. Fast path | Templates, phrase agent, verify, `handle_message` end to end without sessions. | §10 journeys that need one message pass the evals. |
| 6. Sessions | DBOS workflow per chat: send/recv, pending question, resume, idle timeout, restart survival. | A pending question survives the next message and a restart. |
| 7. Slow path | Tool loop agent, deferred `ask_user`, scoping, bound, exception mapping. | Loop tests and the slow-path evals pass within budget. |
| 8. Web | `form_cues`, allowlist, summary agent, labels, failure handling. | Form answers labeled; failures reported. |
| 9. Interface | FastAPI routes, Chainlit chat. | All journeys work in the browser. |
| 10. Reliability and evals | Second experiment (generative model), budgets as run-level scores, CI gate, concurrency script, timing. | Budgets met or tuned with a reason; many chats run without mixing state. |

Tools come before decisions because the fast path needs no generative model and the tool contract is the learning target. Redis is not a phase; it is a measured improvement after phase 10 if session latency warrants it.
