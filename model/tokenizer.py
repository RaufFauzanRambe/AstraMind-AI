"""
AstraMind AI - Tokenizer Manager
==================================
Manages tokenization operations including:
- Token counting and context window management
- Prompt construction and formatting
- Chat template application
- Token budget tracking
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TokenizerManager:
    """
    Manages tokenization and prompt construction for the AstraMind system.

    Handles token counting, context window management, and ensures
    that prompts stay within the model's context limits.
    """

    def __init__(self, tokenizer: Any, context_window: int = 8192):
        self._tokenizer = tokenizer
        self.context_window = context_window
        self._reserved_tokens = 512  # Reserve for generated output

    @property
    def max_input_tokens(self) -> int:
        """Maximum number of tokens allowed for input (minus reserved output)."""
        return self.context_window - self._reserved_tokens

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def truncate_to_budget(self, text: str, budget: int) -> str:
        """
        Truncate text to fit within a token budget.

        Args:
            text: The text to potentially truncate.
            budget: Maximum number of tokens allowed.

        Returns:
            Truncated text that fits within the budget.
        """
        tokens = self._tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= budget:
            return text

        truncated_tokens = tokens[:budget]
        return self._tokenizer.decode(truncated_tokens)

    def build_prompt(
        self,
        system_prompt: str,
        user_message: str,
        context: Optional[List[Dict[str, str]]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Build a complete prompt with system message, context, and chat history.

        Args:
            system_prompt: The system instruction prompt.
            user_message: The current user message.
            context: Optional context entries to include.
            chat_history: Optional list of previous chat messages.

        Returns:
            Formatted prompt string ready for inference.
        """
        parts = []

        # System prompt
        parts.append(f"<|system|>\n{system_prompt}</s>")

        # Context entries (if any)
        if context:
            context_text = "\n".join(
                f"[{entry.get('source', 'unknown')}] {entry.get('content', '')}"
                for entry in context
            )
            context_budget = self.max_input_tokens // 4  # Reserve 25% for context
            context_text = self.truncate_to_budget(context_text, context_budget)
            parts.append(f"<|context|>\n{context_text}</s>")

        # Chat history
        if chat_history:
            for msg in chat_history[-10:]:  # Keep last 10 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|{role}|>\n{content}</s>")

        # Current user message
        parts.append(f"<|user|>\n{user_message}</s>")
        parts.append("<|assistant|)")

        full_prompt = "\n".join(parts)

        # Final safety check: ensure total fits in context
        total_tokens = self.count_tokens(full_prompt)
        if total_tokens > self.max_input_tokens:
            logger.warning(
                f"Prompt ({total_tokens} tokens) exceeds budget "
                f"({self.max_input_tokens}). Truncating."
            )
            full_prompt = self.truncate_to_budget(full_prompt, self.max_input_tokens)

        return full_prompt

    def format_chat_message(self, role: str, content: str) -> Dict[str, str]:
        """Format a single chat message."""
        return {"role": role, "content": content}

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Dict[str, Any]:
        """
        Estimate the computational cost of an inference call.

        Returns rough estimates for compute and memory usage.
        """
        total_tokens = input_tokens + output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_vram_mb": total_tokens * 0.5,  # Rough estimate
            "context_utilization": input_tokens / self.context_window,
        }

    def split_long_text(self, text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
        """
        Split a long text into overlapping chunks that fit within token limits.

        Args:
            text: The text to split.
            chunk_size: Target chunk size in tokens.
            overlap: Number of overlapping tokens between chunks.

        Returns:
            List of text chunks.
        """
        tokens = self._tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self._tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - overlap

        return chunks
