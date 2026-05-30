"""
AstraMind AI - Engine Tests
=============================
Unit tests for the AstraMind core engine components.
"""

import asyncio
import json
import os
import sys
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import AstraConfig, ModelConfig, MemoryConfig, AgentConfig
from core.engine import AstraEngine
from core.memory import MemoryManager, MemoryEntry, ShortTermMemory, LongTermMemory
from core.reasoning import ReasoningEngine, ReasoningStep, ReasoningChain


class TestAstraConfig(unittest.TestCase):
    """Test configuration module."""

    def test_default_config(self):
        config = AstraConfig()
        self.assertIsNotNone(config.model)
        self.assertIsNotNone(config.memory)
        self.assertIsNotNone(config.agent)
        self.assertIsNotNone(config.api)
        self.assertIsNotNone(config.logging)

    def test_model_config_defaults(self):
        config = ModelConfig()
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.max_tokens, 2048)
        self.assertAlmostEqual(config.temperature, 0.7)

    def test_memory_config_defaults(self):
        config = MemoryConfig()
        self.assertEqual(config.max_short_term_memory, 50)
        self.assertEqual(config.max_long_term_memory, 1000)
        self.assertAlmostEqual(config.memory_decay_rate, 0.95)

    def test_agent_config_defaults(self):
        config = AgentConfig()
        self.assertEqual(config.max_iterations, 10)
        self.assertTrue(config.enable_reflection)

    def test_config_to_dict(self):
        config = AstraConfig()
        d = config.to_dict()
        self.assertIn("model", d)
        self.assertIn("memory", d)
        self.assertIn("agent", d)
        self.assertIn("api", d)
        self.assertIn("logging", d)


class TestMemoryEntry(unittest.TestCase):
    """Test memory entry model."""

    def test_create_entry(self):
        entry = MemoryEntry(content="test content", entry_type="interaction")
        self.assertEqual(entry.content, "test content")
        self.assertEqual(entry.entry_type, "interaction")
        self.assertEqual(entry.access_count, 0)

    def test_entry_access(self):
        entry = MemoryEntry(content="test")
        entry.access()
        self.assertEqual(entry.access_count, 1)

    def test_effective_importance(self):
        entry = MemoryEntry(content="test", importance=0.8)
        effective = entry.effective_importance
        self.assertGreater(effective, 0)

    def test_entry_serialization(self):
        entry = MemoryEntry(content="test", importance=0.7)
        d = entry.to_dict()
        self.assertEqual(d["content"], "test")
        self.assertEqual(d["importance"], 0.7)

    def test_entry_deserialization(self):
        data = {
            "content": "test",
            "entry_type": "interaction",
            "timestamp": 1234567890,
            "importance": 0.8,
            "metadata": {"key": "value"},
        }
        entry = MemoryEntry.from_dict(data)
        self.assertEqual(entry.content, "test")
        self.assertEqual(entry.importance, 0.8)


class TestShortTermMemory(unittest.TestCase):
    """Test short-term memory."""

    def test_add_and_retrieve(self):
        stm = ShortTermMemory(max_size=5)
        entry = MemoryEntry(content="hello world")
        stm.add(entry)
        recent = stm.get_recent(1)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0].content, "hello world")

    def test_max_size_eviction(self):
        stm = ShortTermMemory(max_size=3)
        for i in range(5):
            stm.add(MemoryEntry(content=f"entry {i}"))
        self.assertEqual(stm.size, 3)

    def test_search(self):
        stm = ShortTermMemory()
        stm.add(MemoryEntry(content="python programming"))
        stm.add(MemoryEntry(content="java programming"))
        stm.add(MemoryEntry(content="cooking recipes"))
        results = stm.search("programming", top_k=2)
        self.assertEqual(len(results), 2)

    def test_clear(self):
        stm = ShortTermMemory()
        stm.add(MemoryEntry(content="test"))
        stm.clear()
        self.assertEqual(stm.size, 0)


class TestLongTermMemory(unittest.TestCase):
    """Test long-term memory."""

    def test_add_and_size(self):
        ltm = LongTermMemory(store_path="/tmp/test_memory.json", max_size=100)
        ltm.add(MemoryEntry(content="test", importance=0.8))
        self.assertEqual(ltm.size, 1)

    def test_search(self):
        ltm = LongTermMemory(store_path="/tmp/test_memory.json", max_size=100)
        ltm.add(MemoryEntry(content="machine learning algorithms", importance=0.9))
        ltm.add(MemoryEntry(content="cooking pasta", importance=0.5))
        results = ltm.search("machine learning", top_k=1)
        self.assertEqual(len(results), 1)

    def test_decay(self):
        ltm = LongTermMemory(store_path="/tmp/test_memory.json", decay_rate=0.9)
        ltm.add(MemoryEntry(content="test", importance=1.0))
        original_importance = ltm._entries[0].importance
        ltm.apply_decay()
        self.assertLess(ltm._entries[0].importance, original_importance)


class TestReasoningEngine(unittest.TestCase):
    """Test reasoning engine."""

    def setUp(self):
        self.config = AstraConfig()
        self.engine = ReasoningEngine(self.config)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_initialize(self):
        self.loop.run_until_complete(self.engine.initialize())
        self.assertTrue(self.engine._initialized)

    def test_classify_intent(self):
        self.loop.run_until_complete(self.engine.initialize())
        intent = self.engine._classify_intent("Apa itu machine learning?")
        self.assertEqual(intent["type"], "question")
        self.assertGreater(intent["confidence"], 0)

    def test_assess_complexity(self):
        self.loop.run_until_complete(self.engine.initialize())
        complexity = self.engine._assess_complexity(
            "Jelaskan secara detail perbedaan antara machine learning dan deep learning beserta contoh implementasinya",
            [],
        )
        self.assertIn(complexity["level"], ["low", "medium", "high"])

    def test_identify_tools(self):
        self.loop.run_until_complete(self.engine.initialize())
        intent = {"type": "question", "confidence": 0.8}
        tools = self.engine._identify_tools("Hitung berapa 15 kali 23", intent)
        self.assertIn("calculator", tools)

    def test_select_agent(self):
        self.loop.run_until_complete(self.engine.initialize())
        intent = {"type": "question", "confidence": 0.8}
        complexity = {"level": "high", "score": 0.7, "confidence": 0.8}
        agent = self.engine._select_agent(intent, complexity)
        self.assertEqual(agent, "planner")


class TestAstraEngine(unittest.TestCase):
    """Test the main orchestration engine."""

    def setUp(self):
        self.config = AstraConfig()
        self.engine = AstraEngine(self.config)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_engine_not_initialized(self):
        self.assertFalse(self.engine.is_initialized)

    def test_engine_status(self):
        status = self.engine.status
        self.assertFalse(status["initialized"])
        self.assertEqual(status["registered_agents"], [])
        self.assertEqual(status["registered_tools"], [])

    def test_register_agent(self):
        class MockAgent:
            pass
        self.engine.register_agent("test_agent", MockAgent())
        self.assertIn("test_agent", self.engine._agents)

    def test_register_tool(self):
        class MockTool:
            pass
        self.engine.register_tool("test_tool", MockTool())
        self.assertIn("test_tool", self.engine._tools)


if __name__ == "__main__":
    unittest.main()
