"""Settings: where the app finds its database, services and API keys.

Values come from environment variables or the git-ignored .env file, and fall back to
the local Docker defaults below. API keys are SecretStr, so printing the settings never
shows them. Everything else in the project reads config through Settings(), never os.environ.
"""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Where each service lives. Defaults match docker-compose.yml.
    database_url: str = "postgresql://workout:workout@localhost:5433/workout"
    qdrant_url: str = "http://localhost:6333"

    # Optional API keys hidden from logs/repr
    openrouter_api_key: SecretStr | None = None
    typesafe_api_key: SecretStr | None = None

    # Configuration to read from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignores extra environment variables not defined here
    )
