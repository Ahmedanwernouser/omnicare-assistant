# syntax=docker/dockerfile:1
FROM python:3.12-slim

# --- The two things that break this image if you get them wrong -------------
#
# 1. The runtime user is created and switched to *before* the model is warmed.
#    Chroma caches all-MiniLM-L6-v2 under Path.home()/.cache/chroma, which
#    follows $HOME. Warming as root and running as "app" would leave the cache
#    in /root, invisible at runtime, and the container would try an 80 MB
#    download on the first question — slow at best, a failure behind a proxy.
#
# 2. /app is chowned before switching users. WORKDIR creates it as root, and
#    the app needs to write .chroma into it at startup.
# ---------------------------------------------------------------------------

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app \
    && chown -R app:app /app

WORKDIR /app

# Dependencies first: this layer only rebuilds when requirements.txt changes.
COPY --chown=app:app requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

USER app
ENV HOME=/home/app

# Warm the model before any source is copied. warm_embeddings.py imports only
# chromadb, so editing application code never invalidates this layer.
COPY --chown=app:app scripts/warm_embeddings.py ./scripts/
RUN python scripts/warm_embeddings.py

COPY --chown=app:app scripts ./scripts
COPY --chown=app:app app ./app
COPY --chown=app:app data ./data

EXPOSE 8000

# Readiness, not liveness: Compose gates the frontend on the policy index
# actually being loaded. No curl in the slim image, so urllib does the work.
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=6 \
    CMD ["python", "scripts/healthcheck.py"]

# One worker on purpose. Conversation history lives in an in-memory
# checkpointer keyed by user_id, so a second worker would serve a user with no
# memory of their own last message. Scaling this out means moving the
# checkpointer to Redis or Postgres — noted in the README trade-offs.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
