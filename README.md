# Workout RAG Assistant

A chat assistant that answers questions about one user's own workout plan and history from her Jefit export, using only retrieved data, with every recommendation backed by an evidence record. Requirements and the original design are in `docs/workout_agent_requirements_and_design.pdf` (baseline October 2026). This repository holds the design work done since and the application built from it: a single tool-calling agent (`src/workout_agent/agents/`, `workflow/agent_chat.py`) served by FastAPI and a Chainlit chat page. Three of the fourteen designed tools exist so far (`find_exercise`, `get_last_session`, `workout_days`).

## Design decisions (2026-10-06)

Decisions made at the end of the design phase, each argued in a document. The build has since followed them with the changes recorded in `docs/amendments.md` (for example the single-agent loop and Postgres chat tables in place of DBOS for now):

| Decision | Choice | Where |
|---|---|---|
| Requirements reopened | §9 hosted models allowed, §12 bounded model decisions allowed, §25 evals replace per-plan tests | `docs/amendments.md` |
| Architecture | Agentic workflow: fixed graph, five typed decision points, fast path for the seven known question types, bounded tool loop for the rest | `docs/agentic-workflow.md` |
| Framework | Pydantic AI 2.54, after two spike rounds against LangGraph 1.2.13 and plain asyncio | `docs/framework-decision.md`, `docs/framework-evaluation/` |
| Decision model | TypeSafe Jev through Pydantic AI's `TypeSafeModel`; first experiment checks its accuracy against a small generative model | `docs/agentic-workflow.md` §3 |
| Generative model | Qwen3.5-35B-A3B on OpenRouter, pending the second experiment | `docs/agentic-workflow.md` §3 |
| Tools | Fourteen tools, one result envelope, failures as values, evidence logged per call | `docs/tools.md` |
| Session state | DBOS workflow per chat first; Redis as the competing option and a later speed improvement | `docs/amendments.md` §22, §24 |
| Evals and tracing | Langfuse, self-hosted, code evaluators, budgets as run-level scores | `docs/amendments.md` §25 |

The build order is in `docs/amendments.md` §27.

## Running the chat

```sh
docker compose up -d                                            # Postgres, Qdrant, Langfuse
scripts/ensure_docker.sh                                        # opens Docker if it is down, starts Postgres and Qdrant
uv run uvicorn workout_agent.api.main:app --reload              # API on :8000
uv run chainlit run src/workout_agent/ui/chat.py --port 8001    # chat page on :8001
```

The chat page sends every message to the API's `POST /chat/stream` (server-sent events: each tool call as it starts and ends, with its arguments and result, then the reply), so uvicorn must be running. `POST /chat` returns the same reply in one response. Both default to port 8000, hence `--port 8001` for Chainlit. Set `API_URL` in `.env` if the API runs elsewhere.

Chainlit needs `CHAINLIT_AUTH_SECRET` in `.env` (it signs the session of the one built-in user, so there is no login form). Create it once:

```sh
uv run chainlit create-secret    # copy the CHAINLIT_AUTH_SECRET=... line into .env
```

Chats are saved in Postgres (`chats`, `chat_turns`) by the API after each answered message, so they survive restarts of both servers. Each visit or "New chat" starts a blank chat; it appears in the sidebar once you send a message, named after the exercise it found (else the first message). Clicking a past chat shows its messages, and the next message continues it: the agent gets the last `HISTORY_TURNS` turns (default 10), with full tool results only for the newest `HISTORY_FULL_TOOL_TURNS` (default 2). Rename and delete work from the sidebar. The first message of each day also looks up the date of the newest logged session (`latest_record`) and stores it in `daily_checks`; later chats that day read it. It shows as a step (arguments `{}`, result with the date) and in the trace, and the model never sees it. Every step expands to the tool's arguments and result; long results are cut on the page only, and code-called results show only allowlisted keys (`agents/guardrails.py` `page_result`).

## Tracing and agent evals (Langfuse)

Langfuse runs in the same `docker compose up -d`, at http://localhost:3000 (login `admin@localhost.dev` / `workout-admin`, local only). Put the app's keys in `.env` before the first start; Compose creates the project with them:

```sh
LANGFUSE_PUBLIC_KEY=pk-lf-<random>
LANGFUSE_SECRET_KEY=sk-lf-<random>
LANGFUSE_HOST=http://localhost:3000
```

With the keys set, every agent message is a `chat-turn` trace: the agent run, each model call (tokens, latency) and each tool call (arguments, result), with the chat id as the session. Without them tracing is off and nothing else changes.

The agent experiment also scores `answers_question` and `clear_and_brief` per case with an offline Sonnet 5.5 judge (needs `OPENROUTER_API_KEY`); it scores phrasing only, next to the code checks.

It also adds offline Ragas scores per case (`ragas_faithfulness`, `ragas_answer_relevancy`, `ragas_id_precision`, `ragas_id_recall`, and `ragas_errors` when a call fails; needs `uv sync --group eval`). They are informational: `verified` and `budget_faithfulness` stay code-checked, and answer relevancy scores a clarifying question or "not found" reply as near 0, so read it only on replies that give data (docs/amendments.md §25).

```sh
uv run python scripts/eval_agent.py my-change
```

uploads the cases in `src/workout_agent/evals/agent.py` as the dataset `agent-turns` and runs them as an experiment named `<date>-my-change`: one trace per case, scores per case (all checks, kind, trajectory, verified, tokens) and run scores for the `docs/amendments.md` §9 budgets (`budget_*`). Compare runs under Datasets -> agent-turns. The summary is also saved to `evals/results/`, with a `config` block (model, prompt hash, dataset hash, git sha) so runs can be compared knowing what changed.

### Evals as tests

The 30 cases above use ids from your own export, so CI can't run them. A second set, `src/workout_agent/evals/synthetic.py`, runs on a made-up history in the `workout_test` database (never your data) and is the CI gate:

```sh
uv run python scripts/eval_agent.py my-change --set synthetic   # one run, Langfuse dataset agent-turns-synthetic
uv run pytest -m eval                                          # the gate: 3 runs per case, majority must pass
uv run python scripts/eval_agent.py baseline --set synthetic --save-baseline   # accept a new baseline
```

`pytest -m eval` fails if a `docs/amendments.md` §9 budget breaks or a score drops below `evals/baselines/synthetic.json` (`workout_agent/evals/gate.py`). It skips on provider rate limits (HTTP 429). `--save-baseline` refuses a run with failing checks; commit the new baseline in the PR that causes it. `.github/workflows/evals.yml` runs the gate on pull requests that touch the agent, tools, workflow, evals or dependencies, and weekly (that run also gates latency), to catch model or provider drift. It needs the `OPENROUTER_API_KEY` repository secret.

## Updating your data from the chat

On the first message of each day (unless your newest logged session is today's), the chat page asks "Is there a newer Jefit export?" with Upload / Skip. Download the export from Jefit's export page, choose Upload and pick the .csv: the API imports it (history only grows, the routine is replaced), re-indexes exercises in Qdrant, and the message you sent is then answered from the new data. A file that isn't a Jefit export is turned away and changes nothing. It asks once a day, however you answer; `scripts/load_jefit.py` below does the same import from the command line.

## Importing a Jefit export

```sh
uv run python scripts/load_jefit.py data/<your-export>.csv
```

Loads the workout history (exercises, logs, sets; weights before 2026 converted from lbs to kg) and the weekly routine (`routines`, `routine_days`, `planned_exercises` with a target rep count per set). History only grows: loading again adds nothing twice. The routine is replaced on every load, since it changes when edited in Jefit. Days Jefit marks as rest days (Arms) are stored with `in_rotation = false`. Bad rows are printed and skipped.

## Exercise search

`find_exercise` tidies the phrase ("db" -> dumbbell), stops phrases about a day or workout, matches whole words against names and their other names, answers muscle phrases from muscle tags, fixes misspelt words, and only then searches by meaning in Qdrant. Other names, muscles and equipment live in `config/exercise_catalog.toml`.

```sh
uv run python scripts/load_catalog.py       # check the catalog and store it (after load_jefit.py)
uv run python scripts/index_exercises.py    # embed names, other names, descriptions into Qdrant
```

## Scoring exercise search

```sh
uv run python scripts/eval_search.py --label my-change --compare evals/results/search-2026-10-12-phase-d2-large.json
uv run python scripts/sweep_search.py evals/results/search-<date>-my-change.json   # cutoffs, offline
uv run pytest -m search_eval                                                        # quality gate
```

`eval_search.py` runs `find_exercise` on the labelled phrases in `evals/search_cases.toml` (dev and held-out), prints pass rates per kind, Qdrant's own hit rates and every failed case, and saves the run to `evals/results/`. Embeddings are cached in Postgres, so reruns are free. `sweep_search.py` replays a saved run for a range of `MIN_SCORE` values. The `search_eval` test fails if search confidently picks a wrong exercise, offers anything for noise, or fails a phrase the reference run passed.

## Reading order

1. `docs/explanations.md`: what changed and why, every concept in plain terms, open questions. Start here.
2. `docs/amendments.md`: replacement text per PDF section.
3. `docs/agentic-workflow.md`: the two diagrams (workflow, interface), decision points, models, open checks.
4. `docs/tools.md`: what a tool is here, the envelope, the inventory, scoping, design rules.
5. `docs/framework-decision.md`: both spike rounds, side-by-side tables, verdict.
6. `docs/framework-evaluation/langgraph.md` and `pydantic-ai.md`: the per-framework feature mappings with doc links (round 1; rows 5 to 12 still apply).

The PDF remains the source for Part 1 requirements; the amendments override the listed sections only.

## For an AI agent working in this repository

Read `CLAUDE.md`, then this file, then `docs/amendments.md`. Hard rules: Part 1 of the PDF is settled except the amended sections; no model in the safety screen, the evidence check, any calculation, or verification; every fact from a tool; tools follow `docs/tools.md`; budgets in `docs/amendments.md` §9. The spike code (two tests, five one-file spikes) was removed after the decision and is recoverable with `git show 247f369`.

## How the decisions were made

Each candidate framework got a feature mapping written from its current documentation (versions pinned on the day), then a one-file spike of at most 100 or 120 lines against one shared test with fakes inside it, then a side-by-side decision page. Round 1 ran under the original §12; round 2 under the agentic design. Model research fetched current pricing, benchmarks and data policies and lists what it could not verify. Work was split across parallel agents in isolated git worktrees, with every claim that carried weight re-verified against the live documentation before it entered a document.
