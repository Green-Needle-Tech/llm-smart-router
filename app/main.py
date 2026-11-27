"""FastAPI application factory and lifespan management."""
from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.admin import router as admin_router
from app.api.chat import _build_guardrail_engine
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.models import router as models_router
from app.api.router_debug import router as router_debug_router
from app.api.sessions import router as sessions_router
from app.cache.memory import MemoryClassificationCache
from app.cache.redis import RedisClassificationCache
from app.classify.classifier import ClassifierService
from app.config.loader import ConfigManager
from app.middleware.errors import error_middleware
from app.middleware.request_id import request_id_middleware
from app.middleware.request_limits import RequestLimitsMiddleware
from app.privacy.ip_redaction import IPRedactionEngine, IPRedactionStore
from app.providers.openrouter import OpenRouterAdapter
from app.routing.engine import RoutingEngine
from app.session.memory_store import MemorySessionStore
from app.session.redis_store import RedisSessionStore
from app.telemetry.budget import BudgetManager
from app.telemetry.langfuse_tracing import LangfuseTraceMiddleware, shutdown_flush
from app.telemetry.logging import get_logger, setup_logging
from app.telemetry.metrics import router_info
from app.temporal_awareness.engine import TemporalAwarenessEngine
from app.version import APPLICATION_VERSION


def _init_stores(app, settings):
    """Initialize session store and classification cache."""
    cache_backend = os.environ.get("CACHE_BACKEND", settings.session.backend)
    if cache_backend == "redis":
        redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
        app.state.session_store = RedisSessionStore(redis_url, settings.session.max_sessions)
        app.state.classification_cache = RedisClassificationCache(
            redis_url, settings.classification.cache.ttl_seconds
        )
    else:
        app.state.session_store = MemorySessionStore(settings.session.max_sessions)
        app.state.classification_cache = MemoryClassificationCache(
            ttl_seconds=settings.classification.cache.ttl_seconds,
            max_entries=settings.classification.cache.max_entries,
        )


def _open_ip_redaction_store(settings, logger) -> IPRedactionStore:
    """Open SQLite IP redaction store, falling back to temp dir when non-root outside Docker."""
    explicit_env = os.environ.get("IP_REDACTION_DB")
    db_path = explicit_env or settings.telemetry.privacy.db_path
    try:
        store = IPRedactionStore(db_path)
        logger.info("router.privacy.enabled", db_path=db_path)
        return store
    except OSError as exc:
        if explicit_env:
            raise
        fallback_path = str(Path(tempfile.gettempdir()) / "llm-smart-router" / "ip_redaction.db")
        store = IPRedactionStore(fallback_path)
        logger.warning(
            "router.privacy.db_fallback",
            requested_path=db_path,
            fallback_path=fallback_path,
            reason=str(exc),
        )
        return store


async def _load_pricing(provider, settings, logger):
    """Fetch pricing and log auto-detected max_completion_tokens."""
    try:
        await provider.list_models()
        logger.info("router.pricing.loaded")
        for level in ["L1", "L2", "L3", "L4", "L5"]:
            tier = settings.routing.get_tier(level)
            if tier.model:
                mct = provider.get_max_completion_tokens(tier.model)
                if mct is not None:
                    logger.info(
                        "router.max_tokens.auto",
                        level=level, model=tier.model, max_completion_tokens=mct,
                    )
                else:
                    logger.warning(
                        "router.max_tokens.unknown",
                        level=level, model=tier.model,
                    )
    except Exception as e:
        logger.warning("router.pricing.failed", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    cm = ConfigManager()
    settings = cm.load()
    cm.validate_startup()

    setup_logging(
        level=settings.telemetry.log_level,
        log_format=settings.telemetry.log_format,
    )
    logger = get_logger("startup")
    logger.info("router.starting", version=APPLICATION_VERSION)

    app.state.version = APPLICATION_VERSION
    app.state.config = cm

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not openrouter_key:
        logger.error("router.startup.no_key", msg="OPENROUTER_API_KEY not set")
        raise RuntimeError("OPENROUTER_API_KEY environment variable is required")

    provider: OpenRouterAdapter | None = None
    classifier: ClassifierService | None = None
    custom_guardrail_http: httpx.AsyncClient | None = None
    bg_tasks: list[asyncio.Task] = []

    try:
        provider = OpenRouterAdapter(settings, openrouter_key)
        app.state.provider = provider
        app.state.routing_engine = RoutingEngine(cm)
        classifier = ClassifierService(settings, openrouter_key)
        app.state.classifier = classifier
        custom_guardrail_http = httpx.AsyncClient(
            http2=True,
            timeout=10.0,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        app.state.custom_guardrail_http = custom_guardrail_http

        _init_stores(app, settings)
        await _load_pricing(provider, settings, logger)

        # Periodic refresh of pricing + max_completion_tokens
        async def _refresh_models_loop():
            while True:
                await asyncio.sleep(cm.get().provider.pricing_refresh_seconds)
                try:
                    await provider.list_models()
                    logger.info("router.pricing.refreshed")
                except Exception as e:
                    logger.warning("router.pricing.refresh_failed", error=str(e))

        bg_tasks.append(asyncio.create_task(_refresh_models_loop()))

        # Periodic cleanup of expired pins, stale reservations, and per-session locks
        session_store = app.state.session_store
        if hasattr(session_store, "cleanup"):
            async def _session_cleanup_loop():
                while True:
                    await asyncio.sleep(60.0)
                    with contextlib.suppress(Exception):
                        await session_store.cleanup()

            bg_tasks.append(asyncio.create_task(_session_cleanup_loop()))

        # Guardrails & temporal awareness engines initialized before reload callback
        app.state.guardrails = _build_guardrail_engine(settings.telemetry.guardrails)
        app.state.temporal_awareness_engine = TemporalAwarenessEngine(
            settings.telemetry.temporal_awareness
        )

        # Budget manager (pre-request cost enforcement)
        if settings.budget.enabled:
            app.state.budget_manager = BudgetManager(provider)
            logger.info("router.budget.enabled", daily_limit_usd=settings.budget.daily_limit_usd)
        else:
            app.state.budget_manager = None

        # IP redaction & re-hydration privacy middleware (optional)
        if settings.telemetry.privacy.enabled:
            store = _open_ip_redaction_store(settings, logger)
            app.state.ip_redaction = IPRedactionEngine(store)

            async def _purge_loop():
                while True:
                    priv_cfg = cm.get().telemetry.privacy
                    await asyncio.sleep(priv_cfg.purge_interval_seconds)
                    try:
                        removed = await asyncio.to_thread(
                            store.purge_older_than,
                            priv_cfg.retention_hours,
                        )
                        if removed:
                            logger.info("router.privacy.purged", rows=removed)
                    except Exception as e:
                        logger.warning("router.privacy.purge_failed", error=str(e))

            bg_tasks.append(asyncio.create_task(_purge_loop()))
        else:
            app.state.ip_redaction = None

        cm.start_watcher(interval=5.0)

        # Register hot-reload callbacks to update all runtime components
        def _on_reload(new_settings):
            provider.update_config(new_settings)
            classifier.update_config(new_settings)
            app.state.temporal_awareness_engine.config = new_settings.telemetry.temporal_awareness
            new_guardrails = _build_guardrail_engine(new_settings.telemetry.guardrails)
            app.state.guardrails = new_guardrails
            app.state._guardrails_cached_engine = new_guardrails
            if new_settings.budget.enabled and getattr(app.state, "budget_manager", None) is None:
                app.state.budget_manager = BudgetManager(provider)
            logger.info("router.hot_reload.applied")

        cm.on_reload(_on_reload)

        router_info.info({"version": APPLICATION_VERSION, "provider": settings.provider.name})
        logger.info(
            "router.guardrails.ready",
            input_enabled=settings.telemetry.guardrails.input_enabled,
            input_action=settings.telemetry.guardrails.input_action,
            output_action=settings.telemetry.guardrails.output_action,
        )
        logger.info("router.ready", port=settings.server.port)

        yield
    finally:
        # --- Shutdown (guaranteed even if startup fails partway) ---
        shutdown_flush()
        logger.info("router.shutdown")

        for t in bg_tasks:
            t.cancel()
        for t in bg_tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t

        if getattr(app.state, "ip_redaction", None) is not None:
            with contextlib.suppress(Exception):
                app.state.ip_redaction.store.close()

        for store_attr in ("session_store", "classification_cache"):
            st = getattr(app.state, store_attr, None)
            if st is not None and hasattr(st, "close"):
                with contextlib.suppress(Exception):
                    await st.close()

        cm.stop_watcher()

        if custom_guardrail_http is not None:
            with contextlib.suppress(Exception):
                await custom_guardrail_http.aclose()
        if provider is not None:
            with contextlib.suppress(Exception):
                await provider.close()
        if classifier is not None:
            with contextlib.suppress(Exception):
                await classifier.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LLM Smart Router",
        description="Session-pinned LLM smart router with OpenAI-compatible API",
        version=APPLICATION_VERSION,
        lifespan=lifespan,
    )

    # Middleware (Starlette executes last-added outermost)
    app.add_middleware(RequestLimitsMiddleware, max_body_bytes=33_554_432)
    app.middleware("http")(error_middleware)
    app.middleware("http")(request_id_middleware)
    # Langfuse tracing (outermost): one trace per chat completion request,
    # ended when the response body fully completes (streams included).
    app.add_middleware(LangfuseTraceMiddleware)

    # Register routers
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(models_router)
    app.include_router(sessions_router)
    app.include_router(router_debug_router)
    app.include_router(admin_router)

    # Prometheus metrics endpoint
    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
