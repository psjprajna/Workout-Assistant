"""L7, secrets and personal data (LLM02 sensitive information disclosure, LLM03 supply chain).

Keys never print; the model call stays on the provider and data policy the privacy line
names; the pinned framework version is the audited one.
"""

import tomllib
from pathlib import Path

import pytest
from pydantic import SecretStr

from workout_agent.agents import workout_agent
from workout_agent.settings import Settings

pytestmark = pytest.mark.security

ROOT = Path(__file__).parents[2]


def test_api_keys_never_appear_in_repr_str_or_dump():
    settings = Settings(
        openrouter_api_key=SecretStr("sk-or-SECRET1"),
        typesafe_api_key=SecretStr("ts-SECRET2"),
        langfuse_secret_key=SecretStr("lf-SECRET3"),
    )
    for rendered in (repr(settings), str(settings), settings.model_dump_json()):
        assert "SECRET" not in rendered


def test_the_model_call_keeps_the_provider_policy():
    source = Path(workout_agent.__file__).read_text()
    assert workout_agent.MAX_OUTPUT_TOKENS == 512
    for pinned in (
        '"data_collection": "deny"',
        '"allow_fallbacks": False',
        '"order": ["Parasail"]',
    ):
        assert pinned in source, pinned


def test_the_agent_framework_is_pinned_exactly():
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    pinned = [d.replace(" ", "") for d in deps if d.startswith("pydantic-ai-slim")]
    assert any(d.endswith("==2.54.0") for d in pinned), pinned


def test_no_secret_is_committed_in_the_repo_config():
    for name in (".env.example", "docker-compose.yml"):
        path = ROOT / name
        if path.exists():
            text = path.read_text()
            assert "sk-or-" not in text and "sk-ant-" not in text, name
