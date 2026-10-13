"""What the agent may answer with: one of four typed choices, never free text.

The model picks a choice and the ids it is about; code writes the reply from the tool
results (workflow/replies.render). Each docstring is the output tool's description the
model reads, so it says when to choose it.
"""

from pydantic import BaseModel, Field


class ShowLastSession(BaseModel):
    """Show the last session of one exercise.

    Only after get_last_session was called in this turn with this exercise_id.
    """

    exercise_id: int


class AskWhich(BaseModel):
    """Ask the user which exercise they mean.

    Use when find_exercise returned 'ambiguous', even with a single option. List the
    option ids in exactly the order find_exercise returned them.
    """

    option_ids: list[int] = Field(min_length=1)


class NotFound(BaseModel):
    """Say the exercise is not in the user's data: find_exercise returned 'no_rows'."""


class Help(BaseModel):
    """Say what this assistant can do: for greetings and anything not about an exercise."""


AgentOutput = ShowLastSession | AskWhich | NotFound | Help
OUTPUT_TYPES = [ShowLastSession, AskWhich, NotFound, Help]
