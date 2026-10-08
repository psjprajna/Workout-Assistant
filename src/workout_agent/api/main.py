"""Entry point for the server: uv run uvicorn workout_agent.api.main:app --reload

Builds the app with the rule-based router and the database URL from Settings.
"""

from workout_agent.api.app import create_app
from workout_agent.settings import Settings
from workout_agent.workflow.simple_router import simple_route

app = create_app(simple_route, Settings().database_url)