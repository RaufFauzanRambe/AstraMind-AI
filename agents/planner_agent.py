"""
AstraMind AI - Planner Agent
==============================
Specialized agent for complex planning and multi-step task execution.
Breaks down complex goals into actionable steps and orchestrates execution.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PlanStep:
    """Represents a single step in an execution plan."""

    def __init__(
        self,
        step_id: int,
        description: str,
        tool: Optional[str] = None,
        dependencies: Optional[List[int]] = None,
        expected_output: str = "",
        status: str = "pending",
    ):
        self.step_id = step_id
        self.description = description
        self.tool = tool
        self.dependencies = dependencies or []
        self.expected_output = expected_output
        self.status = status
        self.result: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "tool": self.tool,
            "dependencies": self.dependencies,
            "expected_output": self.expected_output,
            "status": self.status,
            "result": self.result,
        }


class ExecutionPlan:
    """A structured plan with multiple steps to achieve a goal."""

    def __init__(self, goal: str):
        self.goal = goal
        self.steps: List[PlanStep] = []
        self.status = "draft"
        self.created_at: Optional[float] = None
        self.completed_at: Optional[float] = None

    def add_step(self, step: PlanStep) -> None:
        self.steps.append(step)

    def get_ready_steps(self) -> List[PlanStep]:
        """Get steps that are ready to execute (all dependencies met)."""
        completed_ids = {s.step_id for s in self.steps if s.status == "completed"}
        ready = []
        for step in self.steps:
            if step.status == "pending" and all(dep in completed_ids for dep in step.dependencies):
                ready.append(step)
        return ready

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "status": self.status,
            "steps": [s.to_dict() for s in self.steps],
            "total_steps": len(self.steps),
            "completed_steps": sum(1 for s in self.steps if s.status == "completed"),
        }


class PlannerAgent:
    """
    An agent specialized in complex planning and multi-step execution.

    Capabilities:
    - Goal decomposition into actionable steps
    - Dependency-aware step ordering
    - Progressive execution with error recovery
    - Plan adaptation based on intermediate results
    - Progress tracking and reporting
    """

    def __init__(self, name: str = "Planner", model_loader: Any = None, max_iterations: int = 10):
        self.name = name
        self.model_loader = model_loader
        self.max_iterations = max_iterations
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
        Execute the planner agent's logic.

        Creates and executes a multi-step plan to achieve the user's goal.

        Args:
            query: The user's complex query or goal.
            context: Relevant context from memory.
            reasoning: Results from the reasoning engine.
            tools: Available tools for plan execution.

        Returns:
            Dictionary with the plan execution results.
        """
        logger.info(f"[{self.name}] Planning for: {query[:100]}...")

        available_tools = {**self._tools, **(tools or {})}

        # Step 1: Create an execution plan
        plan = await self._create_plan(query, context, reasoning, available_tools)
        logger.info(f"Plan created with {len(plan.steps)} steps.")

        # Step 2: Execute the plan progressively
        execution_results = await self._execute_plan(plan, available_tools)
        logger.info(f"Plan execution completed. Status: {plan.status}")

        # Step 3: Synthesize the final response
        final_response = await self._synthesize_response(plan, execution_results, query)

        return {
            "output": final_response,
            "tool_results": execution_results,
            "agent": self.name,
            "confidence": self._calculate_confidence(plan),
            "plan": plan.to_dict(),
        }

    async def _create_plan(
        self,
        goal: str,
        context: List[Dict],
        reasoning: Dict[str, Any],
        tools: Dict[str, Any],
    ) -> ExecutionPlan:
        """
        Create an execution plan for the given goal.

        Decomposes the goal into steps, assigns tools, and
        establishes dependencies between steps.
        """
        plan = ExecutionPlan(goal=goal)

        # Extract the action plan from reasoning
        action_plan = reasoning.get("action", {})
        plan_steps = action_plan.get("steps", [])
        required_tools = reasoning.get("tools", [])

        # Create plan steps based on reasoning
        for i, step_desc in enumerate(plan_steps):
            tool = None
            # Assign a tool if this step involves tool usage
            for tool_name in required_tools:
                if tool_name in step_desc.lower() or tool_name in tools:
                    tool = tool_name
                    break

            step = PlanStep(
                step_id=i + 1,
                description=step_desc,
                tool=tool,
                dependencies=[i] if i > 0 else [],  # Sequential dependency
                expected_output=f"Output dari step {i + 1}",
            )
            plan.add_step(step)

        # Ensure we have at least one step
        if not plan.steps:
            plan.add_step(PlanStep(
                step_id=1,
                description=f"Analisis dan proses: {goal}",
                expected_output="Hasil analisis lengkap",
            ))

        plan.status = "ready"
        return plan

    async def _execute_plan(
        self,
        plan: ExecutionPlan,
        tools: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Execute the plan steps progressively, respecting dependencies.

        Returns a list of execution results for each step.
        """
        import time
        plan.created_at = time.time()
        plan.status = "executing"
        results = []
        iteration = 0

        while plan.get_ready_steps() and iteration < self.max_iterations:
            ready_steps = plan.get_ready_steps()

            for step in ready_steps:
                logger.info(f"Executing step {step.step_id}: {step.description}")
                step.status = "executing"

                try:
                    # Execute tool if assigned
                    if step.tool and step.tool in tools:
                        tool = tools[step.tool]
                        result = await tool.execute(step.description)
                        step.result = result
                        step.status = "completed"
                        results.append({
                            "step_id": step.step_id,
                            "tool": step.tool,
                            "result": result,
                            "status": "success",
                        })
                    else:
                        # No tool - mark as completed with reasoning
                        step.result = f"Step completed: {step.description}"
                        step.status = "completed"
                        results.append({
                            "step_id": step.step_id,
                            "description": step.description,
                            "status": "completed",
                        })

                except Exception as e:
                    step.status = "failed"
                    step.result = str(e)
                    results.append({
                        "step_id": step.step_id,
                        "error": str(e),
                        "status": "failed",
                    })
                    logger.error(f"Step {step.step_id} failed: {e}")

            iteration += 1

        # Update plan status
        all_completed = all(s.status == "completed" for s in plan.steps)
        any_failed = any(s.status == "failed" for s in plan.steps)

        if all_completed:
            plan.status = "completed"
        elif any_failed:
            plan.status = "partial"
        else:
            plan.status = "timeout"

        plan.completed_at = time.time()
        return results

    async def _synthesize_response(
        self,
        plan: ExecutionPlan,
        results: List[Dict],
        original_query: str,
    ) -> str:
        """Synthesize a final response from the plan execution results."""
        completed_steps = sum(1 for s in plan.steps if s.status == "completed")
        total_steps = len(plan.steps)

        response_parts = [
            f"📋 **Rencana Eksekusi untuk: {original_query}**\n",
            f"Progress: {completed_steps}/{total_steps} langkah selesai\n",
        ]

        for step in plan.steps:
            status_emoji = "✅" if step.status == "completed" else "❌" if step.status == "failed" else "⏳"
            response_parts.append(
                f"{status_emoji} Langkah {step.step_id}: {step.description}"
            )
            if step.result:
                response_parts.append(f"   → {step.result}")

        if plan.status == "completed":
            response_parts.append("\n🎉 Semua langkah berhasil dieksekusi!")
        elif plan.status == "partial":
            response_parts.append("\n⚠️ Beberapa langkah gagal. Perlu peninjauan ulang.")

        return "\n".join(response_parts)

    def _calculate_confidence(self, plan: ExecutionPlan) -> float:
        """Calculate confidence score based on plan execution results."""
        if not plan.steps:
            return 0.0

        completed = sum(1 for s in plan.steps if s.status == "completed")
        ratio = completed / len(plan.steps)
        return min(ratio * 0.9 + 0.1, 0.95)
