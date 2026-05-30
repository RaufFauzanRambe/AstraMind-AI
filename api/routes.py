"""
AstraMind AI - API Routes
===========================
FastAPI route definitions for the AstraMind API.
Defines all HTTP endpoints and their handlers.
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from .schema import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatRequest,
    ChatResponse,
    EngineStatusResponse,
    ErrorResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryStoreRequest,
    MemoryStoreResponse,
    ToolExecuteRequest,
    ToolExecuteResponse,
)

logger = logging.getLogger(__name__)

# Create routers
chat_router = APIRouter(prefix="/chat", tags=["Chat"])
memory_router = APIRouter(prefix="/memory", tags=["Memory"])
tools_router = APIRouter(prefix="/tools", tags=["Tools"])
system_router = APIRouter(prefix="/system", tags=["System"])

# Track engine start time
_start_time = time.time()


def get_engine(request: Request):
    """Get the AstraMind engine from the app state."""
    engine = request.app.state.engine
    if not engine.is_initialized:
        raise HTTPException(status_code=503, detail="Engine is not initialized.")
    return engine


# ============ Chat Endpoints ============

@chat_router.post(
    "/",
    response_model=ChatResponse,
    responses={503: {"model": ErrorResponse}},
)
async def chat(request: ChatRequest, http_request: Request):
    """
    Send a message to AstraMind and receive a response.

    Processes the user's message through the full AstraMind pipeline:
    reasoning, agent selection, tool execution, and response generation.
    """
    engine = get_engine(http_request)

    try:
        result = await engine.process(
            user_input=request.message,
            conversation_id=request.conversation_id,
        )
        return ChatResponse(
            response=result["response"],
            agent=result["agent"],
            conversation_id=result.get("conversation_id"),
            confidence=result.get("confidence", 0.0),
            tool_results=result.get("tool_results", []),
            reasoning=result.get("reasoning"),
        )
    except Exception as e:
        logger.error(f"Chat processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.post(
    "/completions",
    response_model=ChatCompletionResponse,
    responses={503: {"model": ErrorResponse}},
)
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """
    OpenAI-compatible chat completion endpoint.

    Accepts messages in OpenAI format and returns completions
    in a compatible response format.
    """
    engine = get_engine(http_request)

    try:
        # Extract the last user message
        last_message = request.messages[-1].get("content", "") if request.messages else ""

        result = await engine.process(user_input=last_message)

        import uuid
        from datetime import datetime

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        return ChatCompletionResponse(
            id=completion_id,
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["response"],
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )
    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Memory Endpoints ============

@memory_router.post(
    "/store",
    response_model=MemoryStoreResponse,
)
async def store_memory(request: MemoryStoreRequest, http_request: Request):
    """Store a new memory entry in the AstraMind memory system."""
    engine = get_engine(http_request)

    try:
        await engine.memory.store_interaction(
            user_input=request.content,
            response="",
            importance=request.importance,
            metadata=request.metadata,
        )
        return MemoryStoreResponse(
            success=True,
            message="Memory stored successfully.",
        )
    except Exception as e:
        logger.error(f"Memory store error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@memory_router.post(
    "/search",
    response_model=MemorySearchResponse,
)
async def search_memory(request: MemorySearchRequest, http_request: Request):
    """Search the AstraMind memory system for relevant entries."""
    engine = get_engine(http_request)

    try:
        results = await engine.memory.retrieve_relevant(
            query=request.query,
            top_k=request.top_k,
        )
        return MemorySearchResponse(
            results=results,
            total=len(results),
            query=request.query,
        )
    except Exception as e:
        logger.error(f"Memory search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Tools Endpoints ============

@tools_router.post(
    "/execute",
    response_model=ToolExecuteResponse,
)
async def execute_tool(request: ToolExecuteRequest, http_request: Request):
    """Execute a specific tool directly."""
    engine = get_engine(http_request)

    tool = engine._tools.get(request.tool_name)
    if not tool:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{request.tool_name}' not found.",
        )

    try:
        result = await tool.execute(request.query)
        return ToolExecuteResponse(
            success=True,
            tool_name=request.tool_name,
            result=result,
        )
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return ToolExecuteResponse(
            success=False,
            tool_name=request.tool_name,
            error=str(e),
        )


@tools_router.get(
    "/list",
)
async def list_tools(http_request: Request):
    """List all available tools and their descriptions."""
    engine = get_engine(http_request)
    tools_info = {}
    for name, tool in engine._tools.items():
        tools_info[name] = {
            "name": getattr(tool, "name", name),
            "description": getattr(tool, "description", "No description available."),
        }
    return {"tools": tools_info}


# ============ System Endpoints ============

@system_router.get(
    "/status",
    response_model=EngineStatusResponse,
)
async def get_status(http_request: Request):
    """Get the current status of the AstraMind engine."""
    engine = http_request.app.state.engine
    status = engine.status
    uptime = time.time() - _start_time if status["initialized"] else None

    return EngineStatusResponse(
        initialized=status["initialized"],
        registered_agents=status["registered_agents"],
        registered_tools=status["registered_tools"],
        memory_stats=status["memory_stats"],
        uptime=uptime,
    )


@system_router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "astramind-ai"}
