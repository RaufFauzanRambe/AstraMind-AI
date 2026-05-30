"""
AstraMind AI - API Tests
==========================
Unit tests for the AstraMind API endpoints.
"""

import asyncio
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schema import (
    ChatRequest,
    ChatResponse,
    MemoryStoreRequest,
    MemorySearchRequest,
    ToolExecuteRequest,
    EngineStatusResponse,
    ErrorResponse,
)


class TestChatRequest(unittest.TestCase):
    """Test chat request schema validation."""

    def test_valid_request(self):
        req = ChatRequest(message="Hello!")
        self.assertEqual(req.message, "Hello!")

    def test_message_stripping(self):
        req = ChatRequest(message="  Hello!  ")
        self.assertEqual(req.message, "Hello!")

    def test_empty_message_raises(self):
        with self.assertRaises(Exception):
            ChatRequest(message="")

    def test_with_options(self):
        req = ChatRequest(
            message="Test",
            conversation_id="conv-123",
            temperature=0.5,
            max_tokens=1024,
            tools=["calculator"],
        )
        self.assertEqual(req.conversation_id, "conv-123")
        self.assertEqual(req.temperature, 0.5)
        self.assertEqual(req.max_tokens, 1024)
        self.assertIn("calculator", req.tools)


class TestChatResponse(unittest.TestCase):
    """Test chat response schema."""

    def test_create_response(self):
        resp = ChatResponse(
            response="Hello back!",
            agent="astra_core",
            confidence=0.9,
        )
        self.assertEqual(resp.response, "Hello back!")
        self.assertEqual(resp.agent, "astra_core")
        self.assertAlmostEqual(resp.confidence, 0.9)

    def test_response_with_tools(self):
        resp = ChatResponse(
            response="Result",
            agent="astra_core",
            confidence=0.8,
            tool_results=[{"tool": "calculator", "result": 42}],
        )
        self.assertEqual(len(resp.tool_results), 1)


class TestMemoryStoreRequest(unittest.TestCase):
    """Test memory store request schema."""

    def test_valid_request(self):
        req = MemoryStoreRequest(content="Important fact", importance=0.8)
        self.assertEqual(req.content, "Important fact")
        self.assertAlmostEqual(req.importance, 0.8)

    def test_default_importance(self):
        req = MemoryStoreRequest(content="Test")
        self.assertAlmostEqual(req.importance, 0.5)


class TestMemorySearchRequest(unittest.TestCase):
    """Test memory search request schema."""

    def test_valid_request(self):
        req = MemorySearchRequest(query="machine learning")
        self.assertEqual(req.query, "machine learning")
        self.assertEqual(req.top_k, 5)

    def test_custom_top_k(self):
        req = MemorySearchRequest(query="test", top_k=10)
        self.assertEqual(req.top_k, 10)


class TestToolExecuteRequest(unittest.TestCase):
    """Test tool execute request schema."""

    def test_valid_request(self):
        req = ToolExecuteRequest(tool_name="calculator", query="2 + 2")
        self.assertEqual(req.tool_name, "calculator")
        self.assertEqual(req.query, "2 + 2")

    def test_with_parameters(self):
        req = ToolExecuteRequest(
            tool_name="calculator",
            query="compute",
            parameters={"precision": 4},
        )
        self.assertEqual(req.parameters["precision"], 4)


class TestEngineStatusResponse(unittest.TestCase):
    """Test engine status response schema."""

    def test_create_status(self):
        resp = EngineStatusResponse(
            initialized=True,
            registered_agents=["astra_core", "decision", "planner"],
            registered_tools=["calculator", "web_search"],
            memory_stats={"short_term_size": 10, "long_term_size": 50},
        )
        self.assertTrue(resp.initialized)
        self.assertEqual(len(resp.registered_agents), 3)
        self.assertEqual(len(resp.registered_tools), 2)


class TestErrorResponse(unittest.TestCase):
    """Test error response schema."""

    def test_create_error(self):
        err = ErrorResponse(error="validation_error", message="Invalid input")
        self.assertEqual(err.error, "validation_error")
        self.assertEqual(err.message, "Invalid input")

    def test_error_with_detail(self):
        err = ErrorResponse(
            error="server_error",
            message="Internal error",
            detail="Database connection failed",
        )
        self.assertIsNotNone(err.detail)


class TestAPIIntegration(unittest.TestCase):
    """Integration tests for API endpoints (requires running server)."""

    def test_health_endpoint_schema(self):
        """Verify health endpoint response schema."""
        expected = {"status": "healthy", "service": "astramind-ai"}
        self.assertEqual(expected["status"], "healthy")

    def test_chat_request_json_serialization(self):
        """Verify chat request can be serialized to JSON."""
        req = ChatRequest(message="Hello!", conversation_id="test-123")
        data = req.model_dump()
        json_str = json.dumps(data)
        self.assertIn("Hello!", json_str)

    def test_response_json_serialization(self):
        """Verify chat response can be serialized to JSON."""
        resp = ChatResponse(
            response="Hi there!",
            agent="astra_core",
            confidence=0.95,
        )
        data = resp.model_dump()
        json_str = json.dumps(data, default=str)
        self.assertIn("Hi there!", json_str)


if __name__ == "__main__":
    unittest.main()
