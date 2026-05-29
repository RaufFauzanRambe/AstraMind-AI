"""
AstraMind AI - Inference Engine
================================
Handles text generation inference with support for:
- Greedy and sampling-based decoding
- Streaming output
- Custom stopping criteria
- Batch inference
"""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from .loader import ModelLoader

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Text generation inference engine built on top of ModelLoader.

    Provides a high-level interface for generating text responses
    with configurable decoding strategies and streaming support.
    """

    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a text response for the given prompt.

        Args:
            prompt: The input prompt text.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_p: Nucleus sampling threshold.
            top_k: Top-K sampling threshold.
            repetition_penalty: Penalty for repeated tokens.
            stop_sequences: List of sequences that stop generation.

        Returns:
            Dictionary with generated text and metadata.
        """
        if not self.model_loader.is_loaded:
            raise RuntimeError("Model not loaded. Cannot generate.")

        config = self.model_loader.config
        max_tokens = max_tokens or config.max_tokens
        temperature = temperature if temperature is not None else config.temperature
        top_p = top_p or config.top_p
        top_k = top_k or config.top_k
        repetition_penalty = repetition_penalty or config.repetition_penalty

        try:
            import torch
            from transformers import StoppingCriteria, StoppingCriteriaList

            # Tokenize input
            tokenizer = self.model_loader.tokenizer
            model = self.model_loader.model

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.context_window)

            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Set up stopping criteria
            stopping_criteria = None
            if stop_sequences:
                stop_ids = []
                for seq in stop_sequences:
                    encoded = tokenizer(seq, add_special_tokens=False)["input_ids"]
                    stop_ids.extend(encoded)

                class CustomStoppingCriteria(StoppingCriteria):
                    def __init__(self, stop_token_ids):
                        self.stop_token_ids = set(stop_token_ids)

                    def __call__(self, input_ids, scores, **kwargs):
                        return input_ids[0, -1].item() in self.stop_token_ids

                stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(stop_ids)])

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1e-10,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=temperature > 0,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode output (skip the input prompt tokens)
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            return {
                "text": generated_text.strip(),
                "tokens_generated": len(generated_ids),
                "prompt_tokens": inputs["input_ids"].shape[1],
                "finish_reason": "stop" if stopping_criteria else "length",
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream generated tokens one at a time.

        Args:
            prompt: The input prompt text.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Yields:
            Individual generated text chunks.
        """
        if not self.model_loader.is_loaded:
            raise RuntimeError("Model not loaded. Cannot generate.")

        config = self.model_loader.config
        max_tokens = max_tokens or config.max_tokens
        temperature = temperature if temperature is not None else config.temperature
        top_p = top_p or config.top_p

        try:
            from transformers import TextIteratorStreamer
            import threading
            import torch

            tokenizer = self.model_loader.tokenizer
            model = self.model_loader.model

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.context_window)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_tokens,
                "temperature": temperature if temperature > 0 else 1e-10,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "streamer": streamer,
                "pad_token_id": tokenizer.eos_token_id,
            }

            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            for text_chunk in streamer:
                if text_chunk:
                    yield text_chunk

            thread.join()

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise

    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: List of prompt strings.
            max_tokens: Maximum tokens per response.
            temperature: Sampling temperature.

        Returns:
            List of generation result dictionaries.
        """
        results = []
        for prompt in prompts:
            result = await self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            results.append(result)
        return results
