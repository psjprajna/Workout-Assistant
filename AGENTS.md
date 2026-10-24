# Repository Guidelines

## Project Structure & Module Organization

Python 3.12 package `workout_agent` in `src/workout_agent/`:

- `agents/`, `workflow/`: the tool-calling agent, guardrails, safety screen and chat flow
- `tools/`: read-only tools (`docs/tools.md` contract); `repositories/`, `db/`, `sessions/`, `schemas/`: data access and chat storage
- `embeddings/`, `importing/`, `catalog.py`: exercise search and the Jefit import
- `api/`: FastAPI app; `ui/`: Chainlit chat page
- `evals/`: offline evals (Langfuse experiments, reply judge, Ragas scores)

Outside the package: `tests/` (`unit/`, `integration/`, `failure/`, `concurrency/`, `fixtures/`; the last two are empty so far), `scripts/` (runnable entry points), `migrations/` (numbered SQL), `config/`, `evals/` (cases, labels, saved results), `data/` (your Jefit export, private), `docs/` (requirements, `amendments.md`, design notes).

## Build, Test, and Development Commands

Run everything from the repository root with `uv`:

- `uv sync --group dev` installs the app and test tools; add `--group eval` for the Ragas scorer.
- `docker compose up -d` starts Postgres (port 5433), Qdrant and Langfuse. Integration tests need Postgres and Qdrant. `scripts/ensure_docker.sh` opens Docker if it is down and starts just those two; the integration tests run it themselves when Postgres isn't answering.
- `uv run pytest tests/unit` needs nothing running; `uv run pytest tests/integration -m "not llm and not search_eval"` skips the tests that call a real model or need your search index.
- `uv run ruff check .` lints; `uv run pyright` type checks.
- Run the app and the evals as the `README.md` describes.

## Coding Style & Naming Conventions

Ruff sets the style (line length 100; rules E, F, I, B, UP). Use descriptive names and match the surrounding module. Avoid adding dependencies for functionality already provided by the standard library, and record any new framework or store in `docs/amendments.md` §24.

## Testing Guidelines

Tests use pytest with `asyncio_mode = "auto"`; markers `llm` (calls a real model) and `search_eval` (needs your real index). Add focused, runnable tests for new behavior and bug fixes, named after the behavior they verify, including invalid input and failure cases.

## Commit & Pull Request Guidelines

Use concise, imperative commit subjects, such as `Add workout input validation`.

Pull requests should describe the problem, changes, and verification performed. Link relevant issues when available. Include screenshots for visible UI changes and call out configuration changes or new dependencies.

## Security & Configuration

Never commit credentials, API keys, or private workout records. If configuration is introduced, document required variables and provide an example containing placeholders only. Validate external input at application boundaries.
