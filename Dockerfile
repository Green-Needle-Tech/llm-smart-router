# ---------- builder ----------
FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml ./
RUN pip install --no-cache-dir fastapi uvicorn[standard] httpx[http2] pydantic pydantic-settings cachetools prometheus-client structlog redis

# ---------- runtime ----------
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1
RUN groupadd -r router && useradd -r -g router -d /app router
WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] httpx[http2] pydantic pydantic-settings \
    cachetools prometheus-client structlog redis

COPY --chown=router:router app/ ./app/
COPY --chown=router:router config/ ./config/
USER router
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://0.0.0.0:8080/healthz',timeout=2).status==200 else 1)"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers ${WORKERS:-1} --no-access-log"]
