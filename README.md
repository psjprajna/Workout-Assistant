# Workout RAG Assistant

A chat assistant that answers questions about one user's own workout plan and history from her Jefit export, using only retrieved data, with every recommendation backed by an evidence record. Requirements and the original design are in `docs/workout_agent_requirements_and_design.pdf` (baseline October 2026). This repository holds the design work done since; there is no application code yet.

## State on 2026-10-06

Design phase complete. Decisions made, each argued in a document:

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

Next: phase 1 of the build order in `docs/amendments.md` §27.

## Running the chat

```sh
docker compose up -d                                            # Postgres and Qdrant
uv run uvicorn workout_agent.api.main:app --reload              # API on :8000
uv run chainlit run src/workout_agent/ui/chat.py --port 8001    # chat page on :8001
```

The chat page sends every message to the API's `POST /chat`, so uvicorn must be running. Both default to port 8000, hence `--port 8001` for Chainlit. Set `API_URL` in `.env` if the API runs elsewhere.

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
