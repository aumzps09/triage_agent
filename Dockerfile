FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY triage_agent/ ./triage_agent/
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --no-deps .

COPY README.md ./
COPY postman_collection.json ./
COPY app.py ./

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# No image-level HEALTHCHECK: default CMD is the console runner (no HTTP
# server), so probing :8000/health would mark console containers unhealthy.
# The `api` service in compose.yaml defines its own healthcheck.

# Optional at runtime: docker run --env-file .env ...
# Console (default): docker run --rm --env-file .env triage-agent
# API: docker run --rm --env-file .env -p 8000:8000 triage-agent python app.py
CMD ["python", "-m", "triage_agent"]
