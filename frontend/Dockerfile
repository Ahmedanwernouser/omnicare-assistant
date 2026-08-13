# No `# syntax=` directive and an explicit Debian release, for the same
# reasons as the backend image: one fewer network fetch before the build
# starts, and a pinned glibc so the pre-built wheels install rather than
# compile. pyarrow (via streamlit) is the strictest here and needs 2.28;
# bookworm ships 2.36. Both x86_64 and aarch64 wheels exist, so this builds
# on Apple Silicon too.
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="OmniCare Assistant UI" \
      org.opencontainers.image.description="Streamlit chat frontend" \
      org.opencontainers.image.version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app \
    && chown -R app:app /app

WORKDIR /app

COPY --chown=app:app requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

USER app
ENV HOME=/home/app

COPY --chown=app:app streamlit_app.py api_client.py healthcheck.py ./

EXPOSE 8501

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "healthcheck.py"]

# --server.address 0.0.0.0 is required for the port to be reachable from the
# host; Streamlit binds to localhost by default and the container would look
# like it started but refuse every connection.
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]