# Explanations

Written for the author of the requirements and design PDF. It explains, in plain terms, what changed since the October 2026 baseline, why, and what each new concept means. Read this before `docs/amendments.md`, which has the exact replacement text per section.

## 1. What changed, in three sentences

Part 1 of the PDF (requirements, §1 to §11) stands, except three decisions that were reopened on 2026-10-06: the model no longer has to run locally (§9), the model may make bounded decisions at named points instead of none (§12), and the deterministic per-plan tests become an evaluation suite (§25). Everything downstream of those three follows: hosted models, a tool-using slow path for questions the fixed plans do not cover, Pydantic AI as the framework, and Langfuse for evals. What did not change: one user, read-only, every fact from an SQL row or a labeled web page, every recommendation with an evidence record, safety refusals in code before any model, no memory across chats, the 20-turn cap.

## 2. Why the three changes

**§9, local model.** The original reason was privacy. The new design keeps the privacy that matters and gives up the rest: only the rows a tool returns for the current question are sent to a model provider; the raw export, the date of birth and the body measurements have no path to any model. Providers that train on API input are excluded and data collection is denied per request. Embeddings stay local. In exchange, the assistant gets a model that can call tools reliably, which an 8B model on a 16 GB laptop cannot do well.

**§12, no model decisions.** The original rule was there because a small local model is unreliable at choosing steps. Two things changed: the model is no longer small, and the design now separates two kinds of decision. Routing decisions (what kind of question is this, which of three exercises did she mean) go to a dedicated decision model that returns a typed choice in about 100 ms. The only open-ended decision, which tool to call next, is confined to a bounded loop that runs only when the fixed plans do not apply, with a scoped set of tools and a hard cap of six calls. Code still owns facts, safety, verification, state and the bounds. The spirit of the rule survives: no decision can skip the safety screen, the evidence check or verification.

**§25, deterministic tests.** With a model choosing tools, "the pipeline took these steps" is no longer something a unit test can assert. It becomes something an evaluation measures: on a dataset of 30 to 50 messages with known answers, how often did the assistant call the right tools, use the right evidence, refuse the safety cases, ask on the ambiguous ones, and stay within the time and token budgets. Unit tests still cover every tool and every pure function.

## 3. Concepts, one at a time

**Agentic workflow.** The industry distinction (Anthropic's "Building effective agents") is between a workflow, where code fixes the path and the model fills in decisions at named points, and an agent, where the model drives its own loop. This design is a workflow with one bounded agent inside it. The path is a fixed graph: safety screen, route, then either the fast path or the slow path, then evidence check, decide, respond, verify. The diagram is in `docs/agentic-workflow.md`.

**Decision points.** There are five, each typed and bounded. D1 routes the message (question type, injury yes or no, which exercise among the candidates, which date reading). D2 picks which stored option a reply meant when exact matching fails. D3 phrases an answer from the evidence. D4 summarises a fetched web page. D5 is the tool loop. D1 and D2 use the decision model; D3, D4 and D5 use the generative model.

**Jev, the decision model.** Jev (by TypeSafe) is not a chat model. You give it a short state text and a set of named questions, each one "pick one of these labels", "rate 1 to N" or "yes or no", and it answers all of them in one request with a confidence per answer. It writes no text and cannot extract a phrase from a sentence, so candidates come from code: the vector store proposes exercise names, a date parser proposes readings, Jev picks. Its published accuracy is modest, and one independent test showed it does much better with several narrow questions than one broad one, so D1 is built as narrow questions, and the first experiment checks Jev against a small generative model doing the same routing before Jev is trusted. Pydantic AI supports it natively (`TypeSafeModel`).

**Fast path and slow path.** For the seven question types in §20, the plan is fixed and code runs it: the lookups, the groups that run together, the evidence check, the rule. No generative model is involved except, sometimes, to phrase the answer, and templates come first. So most turns cost one Jev call (about 100 ms, fractions of a cent) and no generative tokens at all. The slow path exists for questions the plans do not cover, or that span two types: the generative model chooses tools, at most six calls, from a set scoped to the question. This is where "extremely optimised for time and tokens" comes from: the expensive path is the exception.

**Tools.** A tool is one typed Python function that both code (fast path) and the model (slow path) call. Fourteen are specified in `docs/tools.md`: eight retrieval tools (plan day, planned exercises, find exercise, last session, sets in a range, progression rule, personal record, exercise catalog), three computation tools (recommend weight, targets hit, compare periods), one web tool (form cues), one session tool (last recommendation) and one interaction tool (`ask_user`). Every tool returns the same envelope: a status (ok, no rows, ambiguous with options, conflict with both values, unavailable), typed data, the row ids behind it, and the export's date range. Failures are values, never exceptions into the model. Tools are read-only, idempotent, have their own timeouts, and cannot reach the date of birth or body measurements. Arithmetic is a tool: the model never computes a weight.

**Evidence by construction.** Every tool call is logged (name, status, row ids) by a wrapper around the toolset. That log is the evidence record. Verify, which is code, checks that every number in the reply appears in the union of tool outputs; if not, a template reply built from the evidence is sent instead. This is how the model is prevented from inventing a weight even on the slow path.

**`ask_user` as a deferred tool.** When a tool returns "ambiguous" or "conflict", the model calls `ask_user` with the options. In Pydantic AI that call ends the run (a "deferred" tool), the question and the run's history are stored for that chat, and the next message is matched against the options by code, then handed back to the run as the tool's result, so the loop continues from where it stopped. Retrieval already done is not redone. On the fast path the same pause is a stored pending question that resumes at the stage that asked.

**Pydantic AI.** A Python agent framework from the Pydantic team. An `Agent` is a model plus an output schema plus optional tools; it validates the model's output against the schema and retries with the error if validation fails. Here it owns the five decision points, the toolset with its logging wrapper, the deferred pause, the exact tool-call bound (`UsageLimits`), per-request timeouts and concurrency, and one test double (`FunctionModel`) that scripts any agent. It does not own the workflow itself, the session state or verification; those stay plain code, which is where the PDF put them.

**Why Pydantic AI and not LangGraph or plain asyncio.** Two rounds of one-file spikes against one test each. In round 1, under the original §12, plain asyncio won on learning and Pydantic AI only added a nicer gateway. In round 2, with the agentic design, both Pydantic AI and LangGraph passed a test covering fast path, slow path, pause and resume, verification and the bound, at 120 lines each. The difference was where state lives. LangGraph replaces the session store with its checkpointer, which saves after every step and keeps the writes of a failed request, needs Redis Stack in production, and bounds the loop in graph steps rather than tool calls. Pydantic AI leaves state in the session store, saved only on success, exactly as §15 and §22 say. Details and tables: `docs/framework-decision.md`.

**The generative model.** Research on 2026-10-06 compared current small proprietary models and open-weight models on OpenRouter. At 50 turns a day, mostly fast path, the generative model sees about 40k input tokens a day; that is cents a month for open-weight models and about two dollars a month for the dearest proprietary one, so price does not separate them. What does is verified tool-calling reliability, and that evidence is thin everywhere. The default is Qwen3.5-35B-A3B with thinking off, the only open candidate with published tool benchmarks, pinned to one OpenRouter provider with data collection denied. The second experiment runs the dataset through it, DeepSeek V4.1 Flash and gpt-oss-120b and keeps the cheapest that ties on tool trajectory and faithfulness.

**Session state: DBOS or Redis.** The original design keeps each chat's state in Redis under one key with an expiry, with a Redis lock so two messages in the same chat cannot run at once. DBOS is a Python library that makes a function a durable workflow: its steps are checkpointed in Postgres (already in the stack), a workflow processes messages one at a time by construction, and a pending question survives a restart. One workflow per chat gives the lock, the durability and the pending question without writing any of them. It also leaves the door open to keeping conversation history beyond one chat later, which is out of scope today. DBOS goes first; Redis stays as a measured speed improvement if session latency ever warrants it.

**Evals on Langfuse.** Langfuse is a self-hosted platform that records every model and tool call as a trace, searchable by session id, and runs experiments: a dataset of inputs with expected outputs, a task function (here, `handle_message`), and evaluators that score each result. The evaluators here are code, not a judge: outcome kind, tool trajectory, evidence rows, numbers in the reply versus the evidence, refusal on the safety cases, a question on the ambiguous ones, zero lookups on "Why?". Run-level scores enforce the §9 budgets. A judge model is used only to grade phrasing, and never in the request path. Experiments run on every model or prompt change and in CI.

**Spikes.** Short throwaway implementations, under 120 lines each, written to make one test pass with a given framework. Their purpose was to replace opinion with evidence about what each framework handles and what fights the design. They were deleted once the decision was made; the tests and all five spikes are in git history at commit 247f369.

## 4. Budgets

Initial targets, tuned by measurement the same way §23 tunes timeouts: fast path median under 2 seconds and under 1,500 model tokens per turn; slow path median under 8 seconds and under 8,000 tokens; tool trajectory accuracy over 95% on the dataset; faithfulness 100%, checked by code.

## 5. Glossary

- **Token**: the unit models are billed and bounded by; roughly three quarters of a word.
- **Structured output**: the model returns JSON matching a schema, enforced by the provider and validated by Pydantic.
- **Tool call**: the model asks for a named function with typed arguments; code runs it and returns the result to the model.
- **Deferred tool**: a tool call that ends the run instead of executing immediately; the result is supplied later.
- **Checkpointer**: LangGraph's store of graph state after each step.
- **Workflow and step** (DBOS): a durable function and its checkpointed units of work.
- **Trace and span**: the record of one request and of each call inside it.
- **Dataset, experiment, evaluator**: the test cases, one run over them, and a scoring function.
- **p50**: the median; half of requests are faster.

## 6. Questions worth your view

1. Jev's accuracy is the biggest open risk. If the first experiment shows a small generative model routes better, D1 moves to it and Jev is dropped; is that acceptable, or is Jev's latency a hard requirement?
2. `personal_record` returns "conflict" when the records table and the best logged set disagree, so the assistant asks. §20 says the same. Confirm that asking is wanted even when the gap is small.
3. The exercise catalog filter needs a fixed muscle-group vocabulary from the import. Does Jefit's export carry one you trust?
4. DBOS keeps workflow history in Postgres beyond one chat. Scope still says no memory across chats; is longer conversation tracking a future goal, which would favour DBOS further, or not?
5. The build order now puts tools before decisions (§27). Phases 1 and 2 are unchanged; is there anything in the import you would rather settle first?
