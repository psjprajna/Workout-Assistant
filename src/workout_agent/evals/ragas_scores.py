"""Ragas scores for one agent turn: faithfulness, answer relevancy and ID precision/recall.

Offline only, like judge.py: it scores evals (the agent experiment) and never runs in the
request path. It does not replace the code-checked `verified` flag or `budget_faithfulness`.
Needs the `eval` dependency group (docs/amendments.md §24).
"""

from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from ragas import SingleTurnSample
from ragas.embeddings.base import BaseRagasEmbedding, embedding_factory
from ragas.llms.base import InstructorBaseRagasLLM, llm_factory
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from workout_agent.settings import Settings

# Same judge and settings as judge.py.
DEFAULT_OPENROUTER = "anthropic/claude-sonnet-5.5"
EMBEDDING_MODEL = "openai/text-embedding-3-large"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
METRICS = ("ragas_faithfulness", "ragas_answer_relevancy", "ragas_id_precision", "ragas_id_recall")


@dataclass
class RagasScorer:
    llm: InstructorBaseRagasLLM
    embeddings: BaseRagasEmbedding


def build_ragas(settings: Settings, model: str = DEFAULT_OPENROUTER) -> RagasScorer:
    """The Ragas judge LLM and embeddings, both through OpenRouter. Raises RuntimeError."""
    if settings.openrouter_api_key is None:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key.get_secret_value(), base_url=OPENROUTER_URL
    )
    llm = llm_factory(
        model,
        client=client,
        temperature=0,
        extra_body={"provider": {"data_collection": "deny", "allow_fallbacks": False}},
    )
    embeddings = embedding_factory("openai", model=EMBEDDING_MODEL, client=client)
    assert isinstance(embeddings, BaseRagasEmbedding)
    return RagasScorer(llm, embeddings)


async def score_turn(
    scorer: RagasScorer, turn: dict[str, Any], exercise_id: int | None
) -> dict[str, Any]:
    """The scores that apply to one TurnResult (as a dict); a failed metric goes in "errors".

    Faithfulness needs tool results to check the reply against. The ID scores need the exercise
    the case expects, so they cover the turns that must fetch one session.
    """
    scores: dict[str, float] = {}
    errors: list[str] = []

    async def run(name: str, awaitable: Any) -> None:
        try:
            scores[name] = float(await awaitable)
        except Exception as e:  # one failed call is recorded; the rest still run
            errors.append(f"{name}: {e!r}")

    async def value(coro: Any) -> float:
        return (await coro).value

    if turn["contexts"]:
        faithfulness = Faithfulness(llm=scorer.llm)
        await run(
            "ragas_faithfulness",
            value(
                faithfulness.ascore(
                    user_input=turn["message"],
                    response=turn["reply"],
                    retrieved_contexts=turn["contexts"],
                )
            ),
        )
    relevancy = AnswerRelevancy(llm=scorer.llm, embeddings=scorer.embeddings)
    await run(
        "ragas_answer_relevancy",
        value(relevancy.ascore(user_input=turn["message"], response=turn["reply"])),
    )
    if exercise_id is not None:
        sample = SingleTurnSample(
            retrieved_context_ids=[str(i) for i in turn["fetched_ids"]],
            reference_context_ids=[str(exercise_id)],
        )
        await run("ragas_id_precision", IDBasedContextPrecision().single_turn_ascore(sample))
        await run("ragas_id_recall", IDBasedContextRecall().single_turn_ascore(sample))
    return {"scores": scores, "errors": errors}
