"""Unit tests: Settings defaults, env overrides, and keys staying hidden in repr."""

from workout_agent.settings import Settings


def test_defaults_point_at_local_services():
    s = Settings(_env_file=None)
    assert s.database_url == "postgresql://workout:workout@localhost:5433/workout"
    assert s.qdrant_url == "http://localhost:6333"


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://other/db")
    assert Settings(_env_file=None).database_url == "postgresql://other/db"


def test_api_keys_are_not_printed():
    s = Settings(openrouter_api_key="sk-secret", typesafe_api_key="ts-secret", _env_file=None)
    assert "sk-secret" not in repr(s)
    assert "ts-secret" not in repr(s)