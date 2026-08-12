"""Shared fixtures.

Rule enforced here: **no test ever touches backend/data/mock_claims.json.**
Every test gets a fresh copy in a tmp_path, so a failing run cannot corrupt the
seed data and tests cannot leak state into one another.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from app.tools.store import ClaimStore

BACKEND_ROOT = Path(__file__).resolve().parent.parent
REAL_DATA_DIR = BACKEND_ROOT / "data"


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A throwaway copy of backend/data/."""
    target = tmp_path / "data"
    shutil.copytree(REAL_DATA_DIR, target)
    return target


@pytest.fixture
def claims_path(data_dir: Path) -> Path:
    return data_dir / "mock_claims.json"


@pytest.fixture
def store(claims_path: Path) -> ClaimStore:
    return ClaimStore(claims_path, lock_timeout=5.0)


@pytest.fixture
def read_claims(claims_path: Path):
    """Read the file straight from disk, bypassing the store under test."""

    def _read() -> list[dict]:
        return json.loads(claims_path.read_text(encoding="utf-8"))

    return _read
