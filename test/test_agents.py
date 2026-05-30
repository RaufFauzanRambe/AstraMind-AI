"""
AstraMind AI - Agent Tests
============================
Unit tests for the AstraMind agent modules.
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.astra_core_agent import AstraCoreAgent
from agents.decision_agent import DecisionAgent
from agents.planner_agent import PlannerAgent, PlanStep, ExecutionPlan


class TestAstraCoreAgent(unittest.TestCase):
    """Test the core agent."""

    def setUp(self):
        self.agent = AstraCoreAgent(name="TestCore")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_agent_name(self):
        self.assertEqual(self.agent.name, "TestCore")

    def test_register_tool(self):
        class MockTool:
            pass
        self.agent.register_tool("mock_tool", MockTool())
        self.assertIn("mock_tool", self.agent._tools)

    def test_execute_simple(self):
        result = self.loop.run_until_complete(
            self.agent.execute(
                query="Hello, how are you?",
                context=[],
                reasoning={"intent": {"type": "conversation"}, "tools": [], "confidence": 0.8},
            )
        )
        self.assertIn("output", result)
        self.assertEqual(result["agent"], "TestCore")

    def test_format_context(self):
        context = [
            {"source": "short_term", "content": "Previous conversation about Python"},
            {"source": "long_term", "content": "User prefers concise answers"},
        ]
        formatted = self.agent._format_context(context)
        self.assertIn("Python", formatted)
        self.assertIn("concise", formatted)

    def test_format_tool_results(self):
        results = [
            {"tool": "calculator", "result": 42, "status": "success"},
            {"tool": "web_search", "error": "timeout", "status": "error"},
        ]
        formatted = self.agent._format_tool_results(results)
        self.assertIn("calculator", formatted)
        self.assertIn("42", formatted)
        self.assertIn("ERROR", formatted)


class TestDecisionAgent(unittest.TestCase):
    """Test the decision agent."""

    def setUp(self):
        self.agent = DecisionAgent(name="TestDecision")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_identify_options_with_connector(self):
        options = self.agent._identify_options("Pilih antara Python atau JavaScript")
        self.assertEqual(len(options), 2)
        self.assertIn("python", options[0].lower())
        self.assertIn("javascript", options[1].lower())

    def test_identify_options_without_connector(self):
        options = self.agent._identify_options("What should I do?")
        self.assertEqual(len(options), 2)  # Default options

    def test_define_criteria(self):
        criteria = self.agent._define_criteria("test query", [])
        self.assertGreater(len(criteria), 0)
        for c in criteria:
            self.assertIn("name", c)
            self.assertIn("weight", c)
            self.assertGreaterEqual(c["weight"], 0)
            self.assertLessEqual(c["weight"], 1)

    def test_assess_risks(self):
        options = ["Option A", "Option B"]
        matrix = {"Option A": [0.8, 0.7, 0.6], "Option B": [0.3, 0.4, 0.5]}
        risks = self.agent._assess_risks(options, matrix)
        self.assertIn("Option A", risks)
        self.assertIn("Option B", risks)
        self.assertEqual(risks["Option A"]["risk_level"], "low")
        self.assertEqual(risks["Option B"]["risk_level"], "high")

    def test_generate_recommendation(self):
        options = ["Option A", "Option B"]
        matrix = {"Option A": [0.9, 0.8], "Option B": [0.3, 0.4]}
        risks = {
            "Option A": {"risk_level": "low", "overall_score": 0.85, "mitigation": "None needed"},
            "Option B": {"risk_level": "high", "overall_score": 0.35, "mitigation": "Consider alternatives"},
        }
        result = self.agent._generate_recommendation(options, matrix, risks)
        self.assertIn("text", result)
        self.assertIn("confidence", result)
        self.assertIn("Option A", result["text"])


class TestPlannerAgent(unittest.TestCase):
    """Test the planner agent."""

    def setUp(self):
        self.agent = PlannerAgent(name="TestPlanner")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_plan_step_creation(self):
        step = PlanStep(step_id=1, description="Test step")
        self.assertEqual(step.step_id, 1)
        self.assertEqual(step.status, "pending")

    def test_execution_plan_creation(self):
        plan = ExecutionPlan(goal="Test goal")
        self.assertEqual(plan.goal, "Test goal")
        self.assertEqual(plan.status, "draft")

    def test_plan_add_step(self):
        plan = ExecutionPlan(goal="Test")
        plan.add_step(PlanStep(step_id=1, description="Step 1"))
        plan.add_step(PlanStep(step_id=2, description="Step 2", dependencies=[1]))
        self.assertEqual(len(plan.steps), 2)

    def test_get_ready_steps(self):
        plan = ExecutionPlan(goal="Test")
        step1 = PlanStep(step_id=1, description="Step 1")
        step2 = PlanStep(step_id=2, description="Step 2", dependencies=[1])
        plan.add_step(step1)
        plan.add_step(step2)

        # Only step 1 should be ready
        ready = plan.get_ready_steps()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].step_id, 1)

        # Complete step 1, now step 2 should be ready
        step1.status = "completed"
        ready = plan.get_ready_steps()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].step_id, 2)

    def test_execute_plan(self):
        result = self.loop.run_until_complete(
            self.agent.execute(
                query="Analisis data penjualan bulanan dan buat rekomendasi",
                context=[],
                reasoning={
                    "action": {"steps": ["Gather data", "Analyze trends", "Make recommendations"]},
                    "tools": [],
                    "confidence": 0.8,
                },
            )
        )
        self.assertIn("output", result)
        self.assertIn("plan", result)
        self.assertEqual(result["agent"], "TestPlanner")

    def test_plan_to_dict(self):
        plan = ExecutionPlan(goal="Test goal")
        plan.add_step(PlanStep(step_id=1, description="Step 1"))
        d = plan.to_dict()
        self.assertEqual(d["goal"], "Test goal")
        self.assertEqual(len(d["steps"]), 1)


if __name__ == "__main__":
    unittest.main()
