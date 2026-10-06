# Claude Code instructions

Read `README.md` first, then `docs/amendments.md`. The PDF in `docs/` is the requirements baseline; the amendments override §9, §12, §14, §15, §20, §22 to §27 only.

Rules that hold everywhere:

- No model in the safety screen, the evidence check, any calculation, or verification. Those are code.
- Every fact the model sees comes from a tool. Tools follow the contract in `docs/tools.md`: one result envelope, failures as values, read-only, logged into the evidence record.
- The model decides only at the five named decision points in `docs/agentic-workflow.md`, each a typed choice; the tool loop is capped at six calls.
- Session state is saved only after a request succeeds. One chat never sees another.
- Budgets in `docs/amendments.md` §9 are requirements; measure in Langfuse before tuning.
- Pinned versions: pydantic-ai-slim 2.54.0. Do not add a framework or a store that `docs/amendments.md` §24 does not list without recording the reason there.

No application code exists yet. The next step is phase 1 of `docs/amendments.md` §27. Repository conventions are in `AGENTS.md`.
