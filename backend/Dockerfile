# Deliberately no `# syntax=` directive: this file uses no BuildKit-only
# features, and the directive makes the build pull docker/dockerfile:1 from
# Docker Hub before it reads line one — an extra network round trip that fails
# confusingly behind a rate limit or a proxy, for nothing in return.
#
# Pinned to bookworm rather than plain `slim`, because the Debian release is
# what fixes the glibc version, and glibc is what decides whether the
# pre-built wheels below install or try to compile. bookworm ships glibc 2.36;
# the strictest wheel in requirements.txt (onnxruntime) needs 2.28.
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="OmniCare Assistant API" \
      org.opencontainers.image.description="FastAPI + LangGraph agent with policy RAG" \
      org.opencontainers.image.version="1.0.0"

# --- Why this file is ordered the way it is ---------------------------------
#
# 1. The runtime user is created and switched to *before* the model is warmed.
#    Chroma caches all-MiniLM-L6-v2 under Path.home()/.cache/chroma, which
#    follows $HOME. Warming as root and running as "app" would leave the cache
#    in /root, invisible at runtime, and the container would attempt an 80 MB
#    download on the first question — slow at best, a failure behind a proxy.
#
# 2. /app is chowned before dropping privileges. WORKDIR creates it owned by
#    root, and the app writes its .chroma directory there at startup.
#
# 3. The model is warmed *before* application code is copied, so editing a
#    Python file does not invalidate the layer and re-download 80 MB.
#
# No build toolchain is installed, and none is needed: every compiled
# dependency (chromadb, onnxruntime, tokenizers, pydantic-core, grpcio,
# orjson, uvloop, httptools, watchfiles) publishes manylinux wheels for both
# x86_64 and aarch64, so this builds on Intel and Apple Silicon alike without
# a compiler. That was checked against PyPI rather than assumed.
# ---------------------------------------------------------------------------

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app \
    && chown -R app:app /app

WORKDIR /app

# Dependencies first: this layer only rebuilds when requirements.txt changes.
COPY --chown=app:app requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

USER app
ENV HOME=/home/app

# Fails the build if the model cannot be fetched. That is deliberate: a broken
# image should fail here, not in front of a reviewer on the first question.
COPY --chown=app:app scripts/warm_embeddings.py ./scripts/
RUN python scripts/warm_embeddings.py

COPY --chown=app:app scripts ./scripts
COPY --chown=app:app app ./app
COPY --chown=app:app data ./data

EXPOSE 8000

# Probes /ready, not /health: Compose gates the frontend on the policy index
# actually being loaded, and "the process is up" is not the same thing.
# start-period covers a cold start — importing chromadb and langgraph plus
# loading the ONNX session takes tens of seconds on a slow machine — and the
# retries add another 60s on top before the container is called unhealthy.
HEALTHCHECK --interval=10s --timeout=5s --start-period=45s --retries=6 \
    CMD ["python", "scripts/healthcheck.py"]

# One worker on purpose. Conversation history lives in an in-memory
# checkpointer keyed by user_id, so a second worker would serve a user with no
# memory of their own last message. Scaling out means moving the checkpointer
# to Redis or Postgres — noted in the README trade-offs.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]