from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Required URLs
    database_url: str = "postgresql://workout:workout@localhost:5433/workout"
    qdrant_url: str = "http://localhost:6333"
    ollama_url: str = "http://localhost:11434"

    # Optional API keys hidden from logs/repr
    openrouter_api_key: SecretStr | None = None
    typesafe_api_key: SecretStr | None = None

    # Configuration to read from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignores extra environment variables not defined here
    )
