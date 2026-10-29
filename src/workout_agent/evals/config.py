"""What produced an eval run: the model, the prompt, the cases and the code.

Saved with every run summary and sent to Langfuse as experiment metadata, so two runs can be
compared knowing what differed. The hashes change when the prompt or the cases change.
"""

import hashlib
import json
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from typing import Any

from workout_agent.agents.workout_agent import INSTRUCTIONS, MODEL_NAME
from workout_agent.evals.agent import Turn


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def run_config(dataset: str, cases: dict[str, list[Turn]]) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "prompt_hash": _hash(str(INSTRUCTIONS)),
        "dataset": dataset,
        "dataset_hash": _hash(
            json.dumps({k: [asdict(t) for t in v] for k, v in cases.items()}, sort_keys=True)
        ),
        "pydantic_ai": version("pydantic-ai-slim"),
        "git_sha": _git_sha(),
    }
