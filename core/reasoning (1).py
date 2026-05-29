"""
AstraMind AI - Reasoning Engine
================================
Implements multi-step reasoning capabilities including:
- Chain-of-thought reasoning
- Self-reflection and correction
- Decision analysis for agent/tool routing
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .config import AstraConfig
from .memory import MemoryManager

logger = logging.getLogger(__name__)


class ReasoningStep:
    """Represents a single step in the reasoning process."""

    def __init__(self, step_type: str, content: str, confidence: float = 0.0):
        self.step_type = step_type
        self.content = content
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_type": self.step_type,
            "content": self.content,
            "confidence": self.confidence,
        }


class ReasoningChain:
    """A chain of reasoning steps leading to a conclusion."""

    def __init__(self):
        self.steps: List[ReasoningStep] = []
        self.conclusion: Optional[str] = None
        self.confidence: float = 0.0

    def add_step(self, step: ReasoningStep) -> None:
        self.steps.append(step)

    def finalize(self, conclusion: str, confidence: float) -> None:
        self.conclusion = conclusion
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "conclusion": self.conclusion,
            "confidence": self.confidence,
        }


class ReasoningEngine:
    """
    The reasoning engine that performs multi-step analysis on user queries.

    Capabilities:
    - Determine query intent and complexity
    - Chain-of-thought reasoning for complex queries
    - Self-reflection to validate reasoning quality
    - Route queries to appropriate agents and tools
    """

    def __init__(self, config: AstraConfig):
        self.config = config
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the reasoning engine."""
        logger.info("Initializing Reasoning Engine...")
        self._initialized = True
        logger.info("Reasoning Engine initialized.")

    async def analyze(self, user_input: str, context: List[Dict]) -> Dict[str, Any]:
        """
        Analyze a user input and produce a reasoning chain.

        Args:
            user_input: The user's query or statement.
            context: Relevant memory entries for context.

        Returns:
            Dictionary with reasoning results including action plan.
        """
        if not self._initialized:
            raise RuntimeError("Reasoning engine not initialized.")

        chain = ReasoningChain()

        # Step 1: Classify the query intent
        intent = self._classify_intent(user_input)
        chain.add_step(ReasoningStep(
            step_type="intent_classification",
            content=f"Classified intent as: {intent['type']} (confidence: {intent['confidence']:.2f})",
            confidence=intent["confidence"],
        ))

        # Step 2: Assess query complexity
        complexity = self._assess_complexity(user_input, context)
        chain.add_step(ReasoningStep(
            step_type="complexity_assessment",
            content=f"Complexity level: {complexity['level']} (score: {complexity['score']:.2f})",
            confidence=complexity["confidence"],
        ))

        # Step 3: Determine required tools
        required_tools = self._identify_tools(user_input, intent)
        chain.add_step(ReasoningStep(
            step_type="tool_identification",
            content=f"Required tools: {required_tools}",
            confidence=0.8,
        ))

        # Step 4: Select the appropriate agent
        agent = self._select_agent(intent, complexity)
        chain.add_step(ReasoningStep(
            step_type="agent_selection",
            content=f"Selected agent: {agent}",
            confidence=0.85,
        ))

        # Step 5: Build the action plan
        action_plan = self._build_action_plan(intent, complexity, required_tools, agent)
        chain.add_step(ReasoningStep(
            step_type="action_planning",
            content=f"Action plan: {json.dumps(action_plan, indent=2)}",
            confidence=0.8,
        ))

        # Self-reflection (if enabled)
        if self.config.agent.enable_reflection:
            reflection = self._reflect(chain, user_input)
            chain.add_step(ReasoningStep(
                step_type="self_reflection",
                content=reflection["assessment"],
                confidence=reflection["confidence"],
            ))

        # Finalize the reasoning chain
        overall_confidence = sum(s.confidence for s in chain.steps) / len(chain.steps)
        chain.finalize(
            conclusion=f"Route to {agent} with action plan",
            confidence=overall_confidence,
        )

        return {
            "chain": chain.to_dict(),
            "intent": intent,
            "complexity": complexity,
            "action": action_plan,
            "agent": agent,
            "tools": required_tools,
            "confidence": overall_confidence,
        }

    def _classify_intent(self, user_input: str) -> Dict[str, Any]:
        """Classify the intent of the user's query."""
        input_lower = user_input.lower()

        intent_patterns = {
            "question": [r"\?$", r"^(apa|bagaimana|mengapa|kapan|dimana|siapa|kenapa|gimana)", r"(berapa|berapa|hitung|kalkulasi)"],
            "command": [r"^(buat|buatkan|generate|tulis|kerjakan|jalankan|eksekusi|hapus|update)", r"(tolong|bantu|bantu aku|please)"],
            "conversation": [r"^(hai|halo|hei|hi|hello|selamat)", r"(terima kasih|makasih|thanks|bye|dah)"],
            "analysis": [r"(analisis|analisa|evaluasi|review|examine|periksa|cek|bandingkan)"],
            "creative": [r"(ceritakan|tuliskan|imajinasikan|berikan ide|suggest|rekomendasikan)"],
        }

        best_intent = "question"
        best_confidence = 0.3

        for intent_type, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_lower):
                    confidence = 0.85
                    if confidence > best_confidence:
                        best_intent = intent_type
                        best_confidence = confidence
                    break

        return {"type": best_intent, "confidence": best_confidence}

    def _assess_complexity(self, user_input: str, context: List[Dict]) -> Dict[str, Any]:
        """Assess the complexity of the user's query."""
        word_count = len(user_input.split())
        has_multiple_questions = user_input.count("?") > 1
        has_context = len(context) > 0

        score = 0.0
        if word_count > 50:
            score += 0.3
        elif word_count > 20:
            score += 0.15
        if has_multiple_questions:
            score += 0.25
        if has_context:
            score += 0.15

        # Check for complex keywords
        complex_keywords = ["analisis", "bandingkan", "evaluasi", "integrasi", "optimasi", "explain", "derive"]
        for kw in complex_keywords:
            if kw in user_input.lower():
                score += 0.1
                break

        if score >= 0.6:
            level = "high"
        elif score >= 0.3:
            level = "medium"
        else:
            level = "low"

        confidence = min(0.5 + score, 0.95)
        return {"level": level, "score": score, "confidence": confidence}

    def _identify_tools(self, user_input: str, intent: Dict) -> List[str]:
        """Identify which tools are needed for the query."""
        tools = []
        input_lower = user_input.lower()

        tool_indicators = {
            "calculator": ["hitung", "kalkulasi", "berapa", "calculate", "compute", "+", "-", "*", "/"],
            "web_search": ["cari", "search", "cari tahu", "look up", "find", "terbaru", "latest"],
            "file_reader": ["baca file", "read file", "buka file", "open file", "lihat file"],
            "data_analyzer": ["analisis data", "analyze data", "statistik", "statistical", "visualisasi", "chart"],
        }

        for tool, indicators in tool_indicators.items():
            for indicator in indicators:
                if indicator in input_lower:
                    tools.append(tool)
                    break

        return tools

    def _select_agent(self, intent: Dict, complexity: Dict) -> str:
        """Select the most appropriate agent based on intent and complexity."""
        if complexity["level"] == "high":
            return "planner"
        elif intent["type"] == "command" and complexity["level"] in ("medium", "high"):
            return "decision"
        else:
            return "astra_core"

    def _build_action_plan(
        self,
        intent: Dict,
        complexity: Dict,
        tools: List[str],
        agent: str,
    ) -> Dict[str, Any]:
        """Build a structured action plan for the selected agent."""
        plan = {
            "intent_type": intent["type"],
            "complexity_level": complexity["level"],
            "assigned_agent": agent,
            "required_tools": tools,
            "steps": [],
        }

        if intent["type"] == "question":
            plan["steps"] = [
                "Understand the question and its scope",
                "Retrieve relevant information",
                "Formulate a comprehensive answer",
            ]
            if tools:
                plan["steps"].insert(1, f"Use tools: {', '.join(tools)}")

        elif intent["type"] == "command":
            plan["steps"] = [
                "Parse the command and its parameters",
                "Validate the command feasibility",
                "Execute the command",
                "Report the result",
            ]

        elif intent["type"] == "analysis":
            plan["steps"] = [
                "Gather all relevant data",
                "Perform multi-perspective analysis",
                "Synthesize findings",
                "Present analysis with recommendations",
            ]

        elif intent["type"] == "creative":
            plan["steps"] = [
                "Understand the creative brief",
                "Generate multiple options",
                "Refine the best option",
                "Present the creative output",
            ]

        else:
            plan["steps"] = [
                "Understand the user's message",
                "Formulate an appropriate response",
            ]

        return plan

    def _reflect(self, chain: ReasoningChain, original_input: str) -> Dict[str, Any]:
        """
        Perform self-reflection on the reasoning chain to validate quality.

        Checks whether the reasoning steps are coherent, the confidence
        levels are reasonable, and the conclusion follows from the steps.
        """
        if not chain.steps:
            return {"assessment": "No reasoning steps to reflect on.", "confidence": 0.0}

        avg_confidence = sum(s.confidence for s in chain.steps) / len(chain.steps)
        low_confidence_steps = sum(1 for s in chain.steps if s.confidence < 0.5)

        if low_confidence_steps > len(chain.steps) / 2:
            assessment = "Reflection: More than half of reasoning steps have low confidence. Consider gathering more information."
            confidence = 0.4
        elif avg_confidence >= 0.7:
            assessment = "Reflection: Reasoning chain appears coherent with high confidence across steps."
            confidence = 0.9
        else:
            assessment = "Reflection: Reasoning chain is moderately confident. Some steps may benefit from additional verification."
            confidence = 0.65

        return {"assessment": assessment, "confidence": confidence}
