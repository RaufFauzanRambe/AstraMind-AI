"""
AstraMind AI - Core Engine
===========================
The central orchestration engine that coordinates all AstraMind components:
model inference, reasoning, memory, agents, and tool execution.
"""

import logging
from typing import Any, Dict, List, Optional

from .config import AstraConfig
from .memory import MemoryManager
from .reasoning import ReasoningEngine

logger = logging.getLogger(__name__)


class AstraEngine:
    """
    Main orchestration engine for AstraMind AI.

    Coordinates the full pipeline from user input to final response:
    1. Receive and parse user input
    2. Load relevant context from memory
    3. Invoke reasoning engine for analysis
    4. Dispatch to appropriate agent
    5. Execute tools if needed
    6. Generate and return response
    """

    def __init__(self, config: Optional[AstraConfig] = None):
        self.config = config or AstraConfig()
        self.memory = MemoryManager(self.config.memory)
        self.reasoning = ReasoningEngine(self.config)
        self._model_loader = None
        self._inference_engine = None
        self._agents = {}
        self._tools = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all engine components."""
        logger.info("Initializing AstraMind Engine...")

        # Initialize memory system
        await self.memory.initialize()
        logger.info("Memory system initialized.")

        # Initialize reasoning engine
        await self.reasoning.initialize()
        logger.info("Reasoning engine initialized.")

        # Register built-in tools
        self._register_default_tools()

        self._initialized = True
        logger.info("AstraMind Engine fully initialized.")

    async def process(self, user_input: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user input through the full AstraMind pipeline.

        Args:
            user_input: The raw text input from the user.
            conversation_id: Optional ID to track multi-turn conversations.

        Returns:
            Dictionary containing the response, metadata, and any tool results.
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        logger.info(f"Processing input (conversation: {conversation_id}): {user_input[:100]}...")

        # Step 1: Retrieve relevant context from memory
        context = await self.memory.retrieve_relevant(user_input, conversation_id)
        logger.debug(f"Retrieved {len(context)} memory entries.")

        # Step 2: Reason about the input and determine action plan
        reasoning_result = await self.reasoning.analyze(user_input, context)
        logger.debug(f"Reasoning result: {reasoning_result.get('action', 'unknown')}")

        # Step 3: Select and execute the appropriate agent
        agent_name = reasoning_result.get("agent", "astra_core")
        agent = self._agents.get(agent_name)

        if agent is None:
            logger.warning(f"Agent '{agent_name}' not found, falling back to core agent.")
            agent = self._agents.get("astra_core")

        # Step 4: Execute agent with reasoning context
        agent_response = await agent.execute(
            query=user_input,
            context=context,
            reasoning=reasoning_result,
            tools=self._tools,
        )

        # Step 5: Store the interaction in memory
        await self.memory.store_interaction(
            user_input=user_input,
            response=agent_response.get("output", ""),
            conversation_id=conversation_id,
            metadata={
                "agent": agent_name,
                "reasoning": reasoning_result,
            },
        )

        return {
            "response": agent_response.get("output", ""),
            "agent": agent_name,
            "reasoning": reasoning_result,
            "tool_results": agent_response.get("tool_results", []),
            "conversation_id": conversation_id,
        }

    def register_agent(self, name: str, agent: Any) -> None:
        """Register an agent with the engine."""
        self._agents[name] = agent
        logger.info(f"Registered agent: {name}")

    def register_tool(self, name: str, tool: Any) -> None:
        """Register a tool with the engine."""
        self._tools[name] = tool
        logger.info(f"Registered tool: {name}")

    def _register_default_tools(self) -> None:
        """Register the built-in default tools."""
        from tools.calculator import CalculatorTool
        from tools.web_search import WebSearchTool
        from tools.file_reader import FileReaderTool
        from tools.data_analyzer import DataAnalyzerTool

        default_tools = {
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
            "file_reader": FileReaderTool(),
            "data_analyzer": DataAnalyzerTool(),
        }

        for name, tool in default_tools.items():
            self.register_tool(name, tool)

    async def shutdown(self) -> None:
        """Gracefully shut down the engine and all components."""
        logger.info("Shutting down AstraMind Engine...")

        await self.memory.save()
        logger.info("Memory saved.")

        self._initialized = False
        logger.info("Engine shut down complete.")

    @property
    def is_initialized(self) -> bool:
        """Check if the engine is fully initialized."""
        return self._initialized

    @property
    def status(self) -> Dict[str, Any]:
        """Get the current engine status."""
        return {
            "initialized": self._initialized,
            "registered_agents": list(self._agents.keys()),
            "registered_tools": list(self._tools.keys()),
            "memory_stats": self.memory.stats(),
        }
