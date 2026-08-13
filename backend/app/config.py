"""Application settings.

Everything that differs between local dev, Docker and CI lives here and is
driven by environment variables. No module outside this file should build a
filesystem path by hand.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/
_BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Searched in order, first match wins. A single relative ".env" would
        # resolve against the *working directory*, so running
        # `cd backend && uvicorn app.main:app` — which is what the README's
        # non-Docker instructions say to do — would look for backend/.env and
        # silently miss the .env at the repo root. Listing the real locations
        # makes the key findable from anywhere.
        env_file=(
            _BACKEND_ROOT.parent / ".env",  # repo root (where .env.example is)
            _BACKEND_ROOT / ".env",  # backend/, if someone put one there
            ".env",  # whatever the working directory is
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Data -------------------------------------------------------------
    data_dir: Path = Field(default=_BACKEND_ROOT / "data")
    policy_filename: str = "sample_policy.md"
    claims_filename: str = "mock_claims.json"

    # --- Vector store -----------------------------------------------------
    chroma_dir: Path = Field(default=_BACKEND_ROOT / ".chroma")
    collection_name: str = "omnicare_policy"
    retrieval_top_k: int = 2
    max_chunk_chars: int = 1200
    #: "onnx" downloads all-MiniLM-L6-v2 on first use (warmed at image build
    #: time); "lexical" is the offline, deterministic backend used by tests.
    embedding_backend: str = "onnx"

    # --- LLM --------------------------------------------------------------
    llm_provider: str = "groq"          # groq | openai | anthropic | fake
    llm_model: str = ""                  # empty -> provider default
    llm_temperature: float = 0.0
    groq_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # --- Agent limits -----------------------------------------------------
    max_tool_iterations: int = 5
    max_message_chars: int = 2000

    # --- Storage ----------------------------------------------------------
    file_lock_timeout: float = 10.0

    @property
    def policy_path(self) -> Path:
        return self.data_dir / self.policy_filename

    @property
    def claims_path(self) -> Path:
        return self.data_dir / self.claims_filename


@lru_cache
def get_settings() -> Settings:
    """Cached accessor. Call `get_settings.cache_clear()` in tests."""
    return Settings()
