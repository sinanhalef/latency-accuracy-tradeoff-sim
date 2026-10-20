"""Shared pytest fixtures and path setup."""

from __future__ import annotations

import os
import sys

import pytest
import yaml

# Make the project root importable so `import src...` works under pytest.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.market import generate_market  # noqa: E402


@pytest.fixture(scope="session")
def config() -> dict:
    """The project's default config.yaml."""
    with open(os.path.join(ROOT, "config.yaml"), "r") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="session")
def small_config(config) -> dict:
    """A smaller, faster market for unit tests (keeps the test suite quick)."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
    cfg["market"] = dict(config["market"])
    cfg["market"]["n_ticks"] = 60000
    return cfg


@pytest.fixture(scope="session")
def market(small_config):
    return generate_market(small_config, int(small_config["seed"]))
