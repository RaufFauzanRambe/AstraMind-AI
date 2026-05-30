"""
AstraMind AI - API Server
===========================
FastAPI application setup and server configuration.
Initializes the API server with middleware, CORS, and routes.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import AstraConfig
from core.engine import AstraEngine
from .routes import chat_router, memory_router, system_router, tools_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Manage the application lifespan - startup and shutdown.

    Initializes the AstraMind engine on startup and
    gracefully shuts it down on exit.
    """
    # Startup
    logger.info("Starting AstraMind AI server...")

    config = AstraConfig()
    engine = AstraEngine(config)

    await engine.initialize()

    # Register agents
    from agents.astra_core_agent import AstraCoreAgent
    from agents.decision_agent import DecisionAgent
    from agents.planner_agent import PlannerAgent

    engine.register_agent("astra_core", AstraCoreAgent())
    engine.register_agent("decision", DecisionAgent())
    engine.register_agent("planner", PlannerAgent())

    app.state.engine = engine
    logger.info("AstraMind AI server started successfully.")

    yield

    # Shutdown
    logger.info("Shutting down AstraMind AI server...")
    await engine.shutdown()
    logger.info("Server shutdown complete.")


def create_app(config: AstraConfig = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        config: Optional AstraMind configuration. If not provided,
                loads from environment variables.

    Returns:
        Configured FastAPI application instance.
    """
    config = config or AstraConfig()

    app = FastAPI(
        title="AstraMind AI",
        description=(
            "AstraMind AI - Intelligent AI Assistant with reasoning, "
            "memory, and multi-agent capabilities."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS
    cors_origins = config.api.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(memory_router, prefix="/api/v1")
    app.include_router(tools_router, prefix="/api/v1")
    app.include_router(system_router, prefix="/api/v1")

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "AstraMind AI",
            "version": "0.1.0",
            "description": "Intelligent AI Assistant with reasoning, memory, and multi-agent capabilities.",
            "docs": "/docs",
            "health": "/api/v1/system/health",
        }

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False) -> None:
    """
    Run the AstraMind API server using Uvicorn.

    Args:
        host: Host address to bind to.
        port: Port number to listen on.
        debug: Enable debug mode with auto-reload.
    """
    import uvicorn

    log_level = "debug" if debug else "info"

    uvicorn.run(
        "api.server:create_app",
        host=host,
        port=port,
        factory=True,
        reload=debug,
        log_level=log_level,
    )


if __name__ == "__main__":
    config = AstraConfig()
    run_server(
        host=config.api.host,
        port=config.api.port,
        debug=config.api.debug,
    )
