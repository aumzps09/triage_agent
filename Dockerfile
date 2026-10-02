FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./
COPY triage_agent/ ./triage_agent/
RUN pip install --no-cache-dir .

COPY README.md ./

# Optional at runtime: docker run --env-file .env ...
CMD ["python", "-m", "triage_agent"]
