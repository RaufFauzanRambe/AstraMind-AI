"""
AstraMind AI - Decision Agent
===============================
Specialized agent for decision-making tasks.
Evaluates options, weighs pros and cons, and provides structured recommendations.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DecisionAgent:
    """
    An agent specialized in decision-making and evaluation.

    Capabilities:
    - Multi-criteria decision analysis
    - Pros and cons evaluation
    - Risk assessment
    - Structured recommendation generation
    - Comparative analysis of alternatives
    """

    def __init__(self, name: str = "DecisionMaker", model_loader: Any = None):
        self.name = name
        self.model_loader = model_loader
        self._tools: Dict[str, Any] = {}

    def register_tool(self, name: str, tool: Any) -> None:
        """Register a tool for use by this agent."""
        self._tools[name] = tool

    async def execute(
        self,
        query: str,
        context: List[Dict],
        reasoning: Dict[str, Any],
        tools: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the decision agent's logic.

        Performs a structured decision analysis on the query,
        evaluating options and providing recommendations.

        Args:
            query: The user's decision-related query.
            context: Relevant context from memory.
            reasoning: Results from the reasoning engine.
            tools: Available tools for data gathering.

        Returns:
            Dictionary with decision analysis and recommendation.
        """
        logger.info(f"[{self.name}] Analyzing decision: {query[:100]}...")

        available_tools = {**self._tools, **(tools or {})}
        tool_results = []

        # Gather data using tools if needed
        required_tools = reasoning.get("tools", [])
        for tool_name in required_tools:
            if tool_name in available_tools:
                try:
                    tool = available_tools[tool_name]
                    result = await tool.execute(query)
                    tool_results.append({
                        "tool": tool_name,
                        "result": result,
                        "status": "success",
                    })
                except Exception as e:
                    tool_results.append({
                        "tool": tool_name,
                        "result": None,
                        "error": str(e),
                        "status": "error",
                    })

        # Perform structured decision analysis
        analysis = await self._analyze_decision(query, context, tool_results)

        return {
            "output": analysis["recommendation"],
            "tool_results": tool_results,
            "agent": self.name,
            "confidence": analysis["confidence"],
            "analysis": analysis,
        }

    async def _analyze_decision(
        self,
        query: str,
        context: List[Dict],
        tool_results: List[Dict],
    ) -> Dict[str, Any]:
        """
        Perform structured decision analysis.

        Breaks down the decision into:
        1. Option identification
        2. Criteria definition
        3. Evaluation matrix
        4. Risk assessment
        5. Final recommendation
        """
        # Step 1: Identify options from the query
        options = self._identify_options(query)

        # Step 2: Define evaluation criteria
        criteria = self._define_criteria(query, context)

        # Step 3: Build evaluation matrix
        evaluation_matrix = self._evaluate_options(options, criteria, tool_results)

        # Step 4: Risk assessment
        risk_assessment = self._assess_risks(options, evaluation_matrix)

        # Step 5: Generate recommendation
        recommendation = self._generate_recommendation(
            options, evaluation_matrix, risk_assessment
        )

        return {
            "options": options,
            "criteria": criteria,
            "evaluation_matrix": evaluation_matrix,
            "risk_assessment": risk_assessment,
            "recommendation": recommendation["text"],
            "confidence": recommendation["confidence"],
        }

    def _identify_options(self, query: str) -> List[str]:
        """Extract or infer decision options from the query."""
        # Simple heuristic-based option extraction
        connectors = ["atau", "or", "vs", "versus", "dibanding", "compared to"]
        options = []

        for connector in connectors:
            if connector in query.lower():
                parts = query.lower().split(connector)
                if len(parts) == 2:
                    options = [part.strip() for part in parts]
                    break

        if not options:
            options = ["Opsi A", "Opsi B"]

        return options

    def _define_criteria(self, query: str, context: List[Dict]) -> List[Dict[str, Any]]:
        """Define evaluation criteria based on the query domain."""
        default_criteria = [
            {"name": "Efektivitas", "weight": 0.3, "description": "Seberapa efektif opsi ini menyelesaikan masalah"},
            {"name": "Efisiensi", "weight": 0.25, "description": "Seberapa efisien dalam penggunaan sumber daya"},
            {"name": "Risiko", "weight": 0.25, "description": "Tingkat risiko yang terkait"},
            {"name": "Skalabilitas", "weight": 0.2, "description": "Kemampuan untuk diskalakan di masa depan"},
        ]
        return default_criteria

    def _evaluate_options(
        self,
        options: List[str],
        criteria: List[Dict],
        tool_results: List[Dict],
    ) -> Dict[str, List[float]]:
        """
        Build an evaluation matrix scoring each option against each criterion.

        Returns a dictionary mapping option names to their scores per criterion.
        """
        matrix = {}
        for option in options:
            scores = []
            for criterion in criteria:
                # Placeholder scoring logic (would use model in production)
                base_score = 0.6
                # Adjust based on tool results
                for tr in tool_results:
                    if tr.get("status") == "success" and tr.get("result"):
                        base_score += 0.05
                scores.append(min(base_score, 1.0))
            matrix[option] = scores
        return matrix

    def _assess_risks(
        self, options: List[str], evaluation_matrix: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, Any]]:
        """Assess risks for each option."""
        risk_assessment = {}
        for option in options:
            scores = evaluation_matrix.get(option, [])
            avg_score = sum(scores) / len(scores) if scores else 0
            risk_level = "low" if avg_score > 0.7 else "medium" if avg_score > 0.4 else "high"

            risk_assessment[option] = {
                "risk_level": risk_level,
                "overall_score": avg_score,
                "mitigation": f"Pertimbangkan langkah-langkah mitigasi untuk {option}" if risk_level != "low" else "Risiko dapat dikelola",
            }
        return risk_assessment

    def _generate_recommendation(
        self,
        options: List[str],
        evaluation_matrix: Dict[str, List[float]],
        risk_assessment: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """Generate a final recommendation based on analysis."""
        best_option = None
        best_score = -1

        for option, scores in evaluation_matrix.items():
            avg = sum(scores) / len(scores) if scores else 0
            if avg > best_score:
                best_score = avg
                best_option = option

        recommendation_text = (
            f"Berdasarkan analisis multi-kriteria, saya merekomendasikan **{best_option}** "
            f"dengan skor rata-rata {best_score:.2f}. "
            f"Tingkat risiko: {risk_assessment.get(best_option, {}).get('risk_level', 'unknown')}. "
            f"{risk_assessment.get(best_option, {}).get('mitigation', '')}"
        )

        confidence = min(best_score + 0.1, 0.95)
        return {"text": recommendation_text, "confidence": confidence}
