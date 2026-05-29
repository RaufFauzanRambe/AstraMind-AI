"""
AstraMind AI - Model Loader
=============================
Handles loading and management of AI models with support for:
- Hugging Face Transformers models
- Quantization (4-bit, 8-bit)
- Multi-device distribution
- Model caching
"""

import logging
import os
from typing import Any, Dict, Optional

from core.config import ModelConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads and manages AI language models for inference.

    Supports loading models from Hugging Face Hub with optional
    quantization for efficient inference on consumer hardware.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._device = config.device

    async def load(self) -> None:
        """
        Load the model and tokenizer based on configuration.

        Automatically handles quantization, device mapping, and
        caching for optimal loading performance.
        """
        logger.info(f"Loading model: {self.config.model_name}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer_kwargs = {
                "pretrained_model_name_or_path": self.config.model_name,
                "trust_remote_code": True,
                "use_fast": True,
            }

            model_kwargs = {
                "pretrained_model_name_or_path": self.config.model_name,
                "trust_remote_code": True,
                "device_map": self._device,
            }

            # Apply quantization if enabled
            if self.config.quantize:
                from transformers import BitsAndBytesConfig

                if self.config.quantize_bits == 4:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype="float16",
                        bnb_4bit_quant_type="nf4",
                    )
                    logger.info("Applying 4-bit quantization.")
                elif self.config.quantize_bits == 8:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    logger.info("Applying 8-bit quantization.")

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            logger.info("Tokenizer loaded successfully.")

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            self._model.eval()
            logger.info(f"Model loaded successfully on device: {self._device}")

            self._loaded = True

        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            raise RuntimeError(
                "Please install transformers and torch: pip install transformers torch"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @property
    def model(self) -> Any:
        """Get the loaded model instance."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Get the loaded tokenizer instance."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._tokenizer

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._loaded

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._loaded:
            return {"status": "not_loaded"}

        param_count = sum(p.numel() for p in self._model.parameters())
        return {
            "status": "loaded",
            "model_name": self.config.model_name,
            "device": str(self._model.device) if hasattr(self._model, "device") else self._device,
            "total_parameters": param_count,
            "quantized": self.config.quantize,
            "quantize_bits": self.config.quantize_bits if self.config.quantize else None,
            "context_window": self.config.context_window,
        }

    async def unload(self) -> None:
        """Unload the model and free memory."""
        if self._loaded:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._loaded = False

            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Model unloaded and memory freed.")
