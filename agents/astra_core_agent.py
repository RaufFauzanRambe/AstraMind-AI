"""
AstraMind AI - Astra Core Agent
=================================
The primary agent for general-purpose interactions.
Handles standard queries, conversations, and simple tool invocations.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AstraCoreAgent:
    """
    The core agent of AstraMind AI for general-purpose interactions.

    Capabilities:
    - Answer general knowledge questions
    - Engage in natural conversations
    - Execute simple tool calls
    - Provide explanations and summaries
    """

    def __init__(self, name: str = "Astra", model_loader: Any = None):
        self.name = name
        self.model_loader = model_loader
        self._tools: Dict[str, Any] = {}

    def register_tool(self, name: str, tool: Any) -> None:
        """Register a tool that this agent can use."""
        self._tools[name] = tool
        logger.info(f"Tool '{name}' registered with {self.name} agent.")

    async def execute(
        self,
        query: str,
        context: List[Dict],
        reasoning: Dict[str, Any],
        tools: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the agent's logic for a given query.

        Args:
            query: The user's query.
            context: Relevant context from memory.
            reasoning: Results from the reasoning engine.
            tools: Available tools for execution.

        Returns:
            Dictionary with the agent's response and metadata.
        """
        logger.info(f"[{self.name}] Processing query: {query[:100]}...")

        available_tools = {**self._tools, **(tools or {})}
        tool_results = []

        # Execute any tools identified by the reasoning engine
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
                    logger.info(f"Tool '{tool_name}' executed successfully.")
                except Exception as e:
                    tool_results.append({
                        "tool": tool_name,
                        "result": None,
                        "error": str(e),
                        "status": "error",
                    })
                    logger.error(f"Tool '{tool_name}' failed: {e}")

        # Build the response context
        context_text = self._format_context(context)
        tool_context = self._format_tool_results(tool_results)

        # Generate the response (placeholder - would use model in production)
        response = await self._generate_response(
            query=query,
            context=context_text,
            tool_results=tool_context,
            reasoning=reasoning,
        )

        return {
            "output": response,
            "tool_results": tool_results,
            "agent": self.name,
            "confidence": reasoning.get("confidence", 0.5),
        }

    async def _generate_response(
        self,
        query: str,
        context: str,
        tool_results: str,
        reasoning: Dict[str, Any],
    ) -> str:
        """
        Generate a response using the model or fallback logic.

        In production, this would use the loaded model for generation.
        Currently provides a structured placeholder response.
        """
        # Build a comprehensive prompt for the model
        prompt_parts = []

        if context:
            prompt_parts.append(f"Relevant Context:\n{context}\n")

        if tool_results:
            prompt_parts.append(f"Tool Results:\n{tool_results}\n")

        prompt_parts.append(f"User Query: {query}")

        # If model is available, use it for generation
        if self.model_loader and self.model_loader.is_loaded:
            from models.inference import InferenceEngine
            engine = InferenceEngine(self.model_loader)

            full_prompt = "\n".join(prompt_parts)
            result = await engine.generate(full_prompt)
            return result["text"]

        # Fallback: structured response
        intent_type = reasoning.get("intent", {}).get("type", "unknown")
        tool_summary = ""
        if tool_results:
            tool_summary = f"\n\nBerdasarkan hasil tools: {tool_results}"

        return (
            f"Saya memproses pertanyaan Anda dengan tipe intent: {intent_type}. "
            f"Berikut respons saya untuk: '{query}'{tool_summary}\n\n"
            f"Catatan: Model belum dimuat. Untuk respons lengkap, "
            f"pastikan model sudah di-load dengan benar."
        )

    def _format_context(self, context: List[Dict]) -> str:
        """Format context entries into a readable string."""
        if not context:
            return ""

        formatted = []
        for entry in context[:5]:  # Limit to top 5 entries
            source = entry.get("source", "unknown")
            content = entry.get("content", "")
            formatted.append(f"[{source}] {content}")

        return "\n".join(formatted)

    def _format_tool_results(self, tool_results: List[Dict]) -> str:
        """Format tool execution results into a readable string."""
        if not tool_results:
            return ""

        formatted = []
        for result in tool_results:
            tool_name = result.get("tool", "unknown")
            if result.get("status") == "success":
                formatted.append(f"{tool_name}: {result['result']}")
            else:
                formatted.append(f"{tool_name}: ERROR - {result.get('error', 'Unknown')}")

        return "\n".join(formatted)
