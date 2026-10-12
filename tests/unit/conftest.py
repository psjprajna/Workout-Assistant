"""Unit test setup.

Importing chainlit (tests/unit/test_data_layer.py) loads .env into os.environ, which
would hand real API keys to tests that build Settings without .env. Point it at no file.
"""

import os

os.environ.setdefault("CHAINLIT_ENV_FILE", "no-such-file.env")
