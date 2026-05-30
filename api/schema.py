"""
AstraMind AI - API Schema
===========================
Pydantic models and schema definitions for the AstraMind API.
Defines request/response models for type-safe API interactions.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


# ============ Request Models ============

class ChatRequest(BaseModel):
    """Request model for chat interactions."""
    message: str = Field(..., min_length=1, max_length=10000, description="User message text")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for multi-turn chats")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=8192, description="Max tokens to generate")
    tools: Optional[List[str]] = Field(None, description="Specific tools to enable")

    @validator("message")
    def message_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Message must not be empty or whitespace only.")
        return v.strip()


class ChatCompletionRequest(BaseModel):
    """Request model for chat completion (OpenAI-compatible)."""
    model: str = Field(default="astra-mind", description="Model identifier")
    messages: List[Dict[str, str]] = Field(..., min_length=1, description="List of chat messages")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(2048, ge=1, le=8192)
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0)
    stream: Optional[bool] = Field(False, description="Enable streaming response")


class MemoryStoreRequest(BaseModel):
    """Request model for storing a memory entry."""
    content: str = Field(..., min_length=1, description="Memory content to store")
    entry_type: str = Field(default="interaction", description="Type of memory entry")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class MemorySearchRequest(BaseModel):
    """Request model for searching memories."""
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Max results to return")
    source: Optional[str] = Field(None, description="Filter by memory source")


class ToolExecuteRequest(BaseModel):
    """Request model for direct tool execution."""
    tool_name: str = Field(..., description="Name of the tool to execute")
    query: str = Field(..., min_length=1, description="Query for the tool")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional tool parameters")


# ============ Response Models ============

class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    response: str = Field(..., description="AI response text")
    agent: str = Field(..., description="Agent that handled the request")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Response confidence score")
    tool_results: List[Dict[str, Any]] = Field(default_factory=list, description="Tool execution results")
    reasoning: Optional[Dict[str, Any]] = Field(None, description="Reasoning details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatCompletionResponse(BaseModel):
    """Response model for OpenAI-compatible chat completion."""
    id: str = Field(..., description="Completion ID")
    object: str = Field(default="chat.completion")
    created: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp()))
    model: str = Field(default="astra-mind")
    choices: List[Dict[str, Any]] = Field(..., description="Completion choices")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")


class MemoryStoreResponse(BaseModel):
    """Response model for memory storage."""
    success: bool = Field(..., description="Whether storage was successful")
    memory_id: Optional[str] = Field(None, description="ID of the stored memory")
    message: str = Field(..., description="Status message")


class MemorySearchResponse(BaseModel):
    """Response model for memory search."""
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total: int = Field(..., description="Total number of results")
    query: str = Field(..., description="Original search query")


class ToolExecuteResponse(BaseModel):
    """Response model for tool execution."""
    success: bool = Field(..., description="Whether execution was successful")
    tool_name: str = Field(..., description="Name of the executed tool")
    result: Any = Field(None, description="Tool execution result")
    error: Optional[str] = Field(None, description="Error message if failed")


class EngineStatusResponse(BaseModel):
    """Response model for engine status."""
    initialized: bool = Field(..., description="Engine initialization status")
    registered_agents: List[str] = Field(default_factory=list, description="List of registered agents")
    registered_tools: List[str] = Field(default_factory=list, description="List of registered tools")
    memory_stats: Dict[str, Any] = Field(default_factory=dict, description="Memory system statistics")
    uptime: Optional[float] = Field(None, description="Engine uptime in seconds")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
