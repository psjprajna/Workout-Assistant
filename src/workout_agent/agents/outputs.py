"""What the agent may answer with: one of five typed choices, never free text.

The model picks a choice and the ids it is about; code writes the reply from the tool
results (workflow/replies.render). Each docstring is the output tool's description the
model reads; when to choose each is in the agent's instructions (agents/workout_agent.py).
Both are kept short: they are sent with every model call (measured in Langfuse).
"""

from pydantic import BaseModel, Field


class ShowLastSession(BaseModel):
    """Show the last session of one exercise, fetched with get_last_session this turn."""

    exercise_id: int


class AskWhich(BaseModel):
    """Ask which exercise the user means; option ids in find_exercise's order."""

    option_ids: list[int] = Field(min_length=1)


class NotFound(BaseModel):
    """Say the exercise is not in the user's data."""


class ShowWorkoutDay(BaseModel):
    """Show one plan day: its exercises, targets and last weights."""

    day_id: int


class Help(BaseModel):
    """Say what this assistant can do."""


AgentOutput = ShowLastSession | AskWhich | NotFound | ShowWorkoutDay | Help
OUTPUT_TYPES = [ShowLastSession, AskWhich, NotFound, ShowWorkoutDay, Help]
