"""Dockerfile and Compose invariants.

Docker cannot be run in every environment these tests run in, so instead of
building an image we assert the properties that are easy to break and
expensive to discover — every one of these corresponds to a real failure mode:

* Warming the model as root and running as ``app`` puts the cache in ``/root``,
  where the runtime user cannot see it, and the container silently attempts an
  80 MB download on the first question.
* ``WORKDIR`` creates ``/app`` owned by root, so without an explicit ``chown``
  the app cannot write its Chroma directory at startup.
* Copying source before warming means every code edit invalidates the layer and
  re-downloads the model.
* Streamlit binds to localhost by default; in a container the port looks open
  and refuses every connection.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DOCKERFILE = (ROOT / "backend" / "Dockerfile").read_text()
FRONTEND_DOCKERFILE = (ROOT / "frontend" / "Dockerfile").read_text()
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text())


def _order(text: str, first: str, second: str) -> bool:
    return text.index(first) < text.index(second)


# -- backend image ---------------------------------------------------------


def test_user_is_switched_before_the_model_is_warmed() -> None:
    """The cache follows $HOME; warming as the wrong user hides it."""
    assert _order(BACKEND_DOCKERFILE, "USER app", "warm_embeddings.py")


def test_home_is_pinned_for_the_runtime_user() -> None:
    assert "ENV HOME=/home/app" in BACKEND_DOCKERFILE


def test_app_directory_is_chowned_before_dropping_privileges() -> None:
    assert "chown -R app:app /app" in BACKEND_DOCKERFILE
    assert _order(BACKEND_DOCKERFILE, "chown -R app:app /app", "USER app")


def test_model_is_warmed_at_build_time() -> None:
    assert "RUN python scripts/warm_embeddings.py" in BACKEND_DOCKERFILE


def test_source_is_copied_after_the_warm_up() -> None:
    """Otherwise editing one Python file re-downloads 80 MB."""
    assert _order(
        BACKEND_DOCKERFILE,
        "RUN python scripts/warm_embeddings.py",
        "COPY --chown=app:app app ./app",
    )


def test_requirements_are_copied_before_source() -> None:
    assert _order(
        BACKEND_DOCKERFILE,
        "COPY --chown=app:app requirements.txt",
        "COPY --chown=app:app app ./app",
    )


def test_backend_runs_a_single_worker() -> None:
    """Conversation history is an in-memory checkpointer keyed by user_id."""
    assert '"--workers", "1"' in BACKEND_DOCKERFILE


def test_backend_runs_as_a_non_root_user() -> None:
    assert "USER app" in BACKEND_DOCKERFILE
    assert BACKEND_DOCKERFILE.rstrip().splitlines()[-1].startswith("CMD")


# -- frontend image --------------------------------------------------------


def test_streamlit_binds_to_all_interfaces() -> None:
    """Default is localhost, which in a container refuses every connection."""
    assert "--server.address=0.0.0.0" in FRONTEND_DOCKERFILE


def test_frontend_runs_as_a_non_root_user() -> None:
    assert "USER app" in FRONTEND_DOCKERFILE


@pytest.mark.parametrize(
    "dockerfile",
    [BACKEND_DOCKERFILE, FRONTEND_DOCKERFILE],
    ids=["backend", "frontend"],
)
def test_healthchecks_use_a_script_not_inline_escaping(dockerfile: str) -> None:
    assert "HEALTHCHECK" in dockerfile
    assert "healthcheck.py" in dockerfile


# -- compose ---------------------------------------------------------------


def test_no_obsolete_version_key() -> None:
    assert "version" not in COMPOSE


def test_both_services_are_defined() -> None:
    assert set(COMPOSE["services"]) == {"backend", "frontend"}


def test_frontend_reaches_the_backend_by_service_name() -> None:
    """localhost inside the frontend container is the frontend."""
    assert (
        COMPOSE["services"]["frontend"]["environment"]["BACKEND_URL"]
        == "http://backend:8000"
    )


def test_frontend_waits_for_the_backend_to_be_ready() -> None:
    depends = COMPOSE["services"]["frontend"]["depends_on"]["backend"]
    assert depends["condition"] == "service_healthy"


def test_claims_file_is_bind_mounted_so_writes_are_visible() -> None:
    assert "./backend/data:/app/data" in COMPOSE["services"]["backend"]["volumes"]


def test_config_uses_interpolation_so_a_missing_env_file_is_not_fatal() -> None:
    """`env_file:` would abort the whole stack when .env is absent."""
    backend = COMPOSE["services"]["backend"]
    assert "env_file" not in backend
    assert backend["environment"]["GROQ_API_KEY"] == "${GROQ_API_KEY:-}"
    assert backend["environment"]["LLM_PROVIDER"] == "${LLM_PROVIDER:-groq}"


def test_every_provider_key_is_passed_through() -> None:
    env = COMPOSE["services"]["backend"]["environment"]
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        assert key in env


def test_ports_match_the_documented_ones() -> None:
    assert COMPOSE["services"]["backend"]["ports"] == ["8000:8000"]
    assert COMPOSE["services"]["frontend"]["ports"] == ["8501:8501"]


# -- ignore files ----------------------------------------------------------


@pytest.mark.parametrize("service", ["backend", "frontend"])
def test_dockerignore_excludes_secrets_and_caches(service: str) -> None:
    body = (ROOT / service / ".dockerignore").read_text()
    for entry in (".env", "__pycache__/", "tests/"):
        assert entry in body


def test_env_file_is_gitignored() -> None:
    assert ".env" in (ROOT / ".gitignore").read_text()


# -- build reliability -----------------------------------------------------


@pytest.mark.parametrize(
    "dockerfile",
    [BACKEND_DOCKERFILE, FRONTEND_DOCKERFILE],
    ids=["backend", "frontend"],
)
def test_base_image_pins_the_debian_release(dockerfile: str) -> None:
    """The Debian release fixes glibc, and glibc decides whether the compiled
    wheels install or try to build from source.

    bookworm ships glibc 2.36; the strictest wheel in the stack (onnxruntime,
    and pyarrow on the frontend) needs 2.28. Plain `python:3.12-slim` follows
    whatever Debian is current, so a future base bump could silently move that
    floor.
    """
    assert "python:3.12-slim-bookworm" in dockerfile


@pytest.mark.parametrize(
    "dockerfile",
    [BACKEND_DOCKERFILE, FRONTEND_DOCKERFILE],
    ids=["backend", "frontend"],
)
def test_no_syntax_directive(dockerfile: str) -> None:
    """A `# syntax=` directive makes BuildKit pull docker/dockerfile:1 from
    Docker Hub before reading line one. Neither image uses a BuildKit-only
    feature, so that is a network round trip that can fail behind a rate limit
    for no benefit."""
    first_line = dockerfile.lstrip().splitlines()[0]
    assert not first_line.startswith("# syntax=")


@pytest.mark.parametrize(
    "dockerfile",
    [BACKEND_DOCKERFILE, FRONTEND_DOCKERFILE],
    ids=["backend", "frontend"],
)
def test_no_build_toolchain_is_installed(dockerfile: str) -> None:
    """Every compiled dependency ships manylinux wheels for x86_64 and
    aarch64, so no compiler is needed. Installing build-essential would add
    minutes and hundreds of megabytes to hide a problem that does not exist."""
    assert "build-essential" not in dockerfile
    assert "apt-get install" not in dockerfile


def test_backend_start_period_covers_a_cold_start() -> None:
    """Importing chromadb and langgraph and loading the ONNX session takes
    tens of seconds. Too short a start-period marks a healthy container
    unhealthy and Compose never starts the frontend."""
    match = re.search(r"--start-period=(\d+)s", BACKEND_DOCKERFILE)
    assert match is not None
    assert int(match.group(1)) >= 45


def test_every_copy_source_exists_in_its_build_context() -> None:
    """A COPY of a path that is not in the context fails the build."""
    import shlex

    for dockerfile, context in (
        (ROOT / "backend" / "Dockerfile", ROOT / "backend"),
        (ROOT / "frontend" / "Dockerfile", ROOT / "frontend"),
    ):
        for line in dockerfile.read_text().splitlines():
            line = line.strip()
            if not line.upper().startswith("COPY "):
                continue
            parts = [p for p in shlex.split(line)[1:] if not p.startswith("--")]
            for source in parts[:-1]:
                assert (context / source).exists(), (
                    f"{dockerfile.name}: COPY {source} is not in {context.name}/"
                )