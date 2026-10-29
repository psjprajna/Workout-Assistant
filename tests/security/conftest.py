"""Security test setup: same as the unit tests, no real keys, no services, no model."""

import os

import pytest

os.environ.setdefault("CHAINLIT_ENV_FILE", "no-such-file.env")

pytestmark = pytest.mark.security
