FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY config ./config
COPY data ./data
COPY reports ./reports

RUN pip install --upgrade pip && pip install . \
    && mkdir -p /app/logs /app/models \
    && chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import tennis_analytics; print(tennis_analytics.__version__)" || exit 1

CMD ["python", "scripts/run_daily_pipeline.py"]
