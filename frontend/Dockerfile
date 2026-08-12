# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

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
