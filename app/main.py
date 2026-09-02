"""FastAPI application factory and lifespan management."""
from __future__ import annotations

import asyncio
import contextlib
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.admin import router as admin_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.models import router as models_router
from app.api.router_debug import router as router_debug_router
from app.api.sessions import router as sessions_router
from app.cache.memory import MemoryClassificationCache
from app.cache.redis import RedisClassificationCache
from app.classify.classifier import ClassifierService
from app.config.loader import ConfigManager
from app.guardrails.scanner import GuardrailConfig, GuardrailEngine
from app.middleware.errors import error_middleware
from app.middleware.request_limits import RequestLimitsMiddleware
from app.privacy.ip_redaction import IPRedactionEngine, IPRedactionStore
from app.providers.openrouter import OpenRouterAdapter
from app.routing.engine import RoutingEngine
from app.session.memory_store import MemorySessionStore
from app.session.redis_store import RedisSessionStore
from app.telemetry.budget import BudgetManager
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
    # --- Startup ---
    cm = ConfigManager()
    settings = cm.load()
    cm.validate_startup()

    setup_logging(
        level=settings.telemetry.log_level,
        log_format=settings.telemetry.log_format,
    )
    logger = get_logger("startup")
    logger.info("router.starting", version=APPLICATION_VERSION)

    app.state.config = cm

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not openrouter_key:
        logger.error("router.startup.no_key", msg="OPENROUTER_API_KEY not set")
        raise RuntimeError("OPENROUTER_API_KEY environment variable is required")

    provider = OpenRouterAdapter(settings, openrouter_key)
    app.state.provider = provider
    app.state.routing_engine = RoutingEngine(cm)
    classifier = ClassifierService(settings, openrouter_key)
    app.state.classifier = classifier

    _init_stores(app, settings)
    await _load_pricing(provider, settings, logger)

    # Periodic refresh of pricing + max_completion_tokens
    async def _refresh_models_loop():
        while True:
            await asyncio.sleep(settings.provider.pricing_refresh_seconds)
            try:
                await provider.list_models()
                logger.info("router.pricing.refreshed")
            except Exception as e:
                logger.warning("router.pricing.refresh_failed", error=str(e))

    refresh_task = asyncio.create_task(_refresh_models_loop())

    cm.start_watcher(interval=5.0)

    # Register hot-reload callbacks to update components that captured
    # startup settings. This ensures /admin/settings/reload actually
    # affects runtime behavior, not just the config object.
    def _on_reload(new_settings):
        # Update provider config (base_url, timeout, headers)
        provider.config = new_settings
        # Update classifier config
        classifier.config = new_settings
        # Update temporal awareness engine config
        app.state.temporal_awareness_engine.config = new_settings.telemetry.temporal_awareness
        logger.info("router.hot_reload.applied")

    cm.on_reload(_on_reload)

    router_info.info({"version": APPLICATION_VERSION, "provider": settings.provider.name})

    # IP redaction & re-hydration privacy middleware (optional)
    purge_task = None
    if settings.telemetry.privacy.enabled:
        db_path = os.environ.get(
            "IP_REDACTION_DB", settings.telemetry.privacy.db_path
        )
        store = IPRedactionStore(db_path)
        app.state.ip_redaction = IPRedactionEngine(store)
        logger.info("router.privacy.enabled", db_path=db_path)

        async def _purge_loop():
            while True:
                await asyncio.sleep(settings.telemetry.privacy.purge_interval_seconds)
                try:
                    removed = await asyncio.to_thread(
                        store.purge_older_than,
                        settings.telemetry.privacy.retention_hours,
                    )
                    if removed:
                        logger.info("router.privacy.purged", rows=removed)
                except Exception as e:
                    logger.warning("router.privacy.purge_failed", error=str(e))

        purge_task = asyncio.create_task(_purge_loop())

    # Guardrails engine (input injection detection + output secret masking).
    # Config is re-read per request so settings hot-reload applies live.
    app.state.guardrails = GuardrailEngine(GuardrailConfig())
    app.state.temporal_awareness_engine = TemporalAwarenessEngine(app.state.config.get().telemetry.temporal_awareness)

    # Budget manager (pre-request cost enforcement)
    if settings.budget.enabled:
        app.state.budget_manager = BudgetManager(provider)
        logger.info("router.budget.enabled", daily_limit_usd=settings.budget.daily_limit_usd)
    else:
        app.state.budget_manager = None
    logger.info(
        "router.guardrails.ready",
        input_enabled=settings.telemetry.guardrails.input_enabled,
        input_action=settings.telemetry.guardrails.input_action,
        output_action=settings.telemetry.guardrails.output_action,
    )

    logger.info("router.ready", port=settings.server.port)

    yield

    # --- Shutdown ---
    logger.info("router.shutdown")
    # Cancel and await background tasks
    tasks_to_cancel = []
    if refresh_task:
        refresh_task.cancel()
        tasks_to_cancel.append(refresh_task)
    if purge_task:
        purge_task.cancel()
        tasks_to_cancel.append(purge_task)
    for t in tasks_to_cancel:
        with contextlib.suppress(asyncio.CancelledError):
            await t

    # Close IP redaction store
    if getattr(app.state, "ip_redaction", None) is not None:
        app.state.ip_redaction.store.close()

    # Close Redis session store if applicable
    session_store = getattr(app.state, "session_store", None)
    if session_store is not None and hasattr(session_store, "close"):
        await session_store.close()

    # Stop config watcher thread
    cm.stop_watcher()

    await provider.close()
    await classifier.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LLM Smart Router",
        description="Session-pinned LLM smart router with OpenAI-compatible API",
        version=APPLICATION_VERSION,
        lifespan=lifespan,
    )

    # Middleware (order: outermost first)
    app.add_middleware(RequestLimitsMiddleware, max_body_bytes=10_485_760)
    app.middleware("http")(error_middleware)

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
