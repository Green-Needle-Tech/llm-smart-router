# ---------- builder ----------
FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir \
    --only-binary :all: \
    --require-hashes \
    -r requirements.txt \
    --target /install

# ---------- runtime ----------
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/app/.local/bin:${PATH}" \
    PYTHONPATH="/app/.local"
RUN groupadd -r router && useradd -r -g router -d /app router
RUN mkdir -p /app/data && chown -R router:router /app/data
WORKDIR /app

COPY --from=builder /install /app/.local
COPY --chown=router:router app/ ./app/
COPY --chown=router:router config/ ./config/
USER router
EXPOSE 8080
# Local container loopback healthcheck (cleartext HTTP over loopback interface is secure and standard inside isolated containers)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz',timeout=2).status==200 else 1)"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers ${WORKERS:-1} --no-access-log"]
