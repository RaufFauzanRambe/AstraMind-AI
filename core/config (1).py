"""
AstraMind AI - Configuration Module
====================================
Central configuration for the AstraMind AI system.
Loads settings from environment variables and .env files.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for the AI model."""
    model_name: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8B-Instruct")
    device: str = os.getenv("DEVICE", "auto")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "2048"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))
    top_k: int = int(os.getenv("TOP_K", "50"))
    repetition_penalty: float = float(os.getenv("REPETITION_PENALTY", "1.1"))
    context_window: int = int(os.getenv("CONTEXT_WINDOW", "8192"))
    quantize: bool = os.getenv("QUANTIZE", "false").lower() == "true"
    quantize_bits: int = int(os.getenv("QUANTIZE_BITS", "4"))


@dataclass
class MemoryConfig:
    """Configuration for the memory system."""
    memory_store_path: str = os.getenv("MEMORY_STORE_PATH", "data/memory_store.json")
    max_short_term_memory: int = int(os.getenv("MAX_SHORT_TERM_MEMORY", "50"))
    max_long_term_memory: int = int(os.getenv("MAX_LONG_TERM_MEMORY", "1000"))
    memory_decay_rate: float = float(os.getenv("MEMORY_DECAY_RATE", "0.95"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@dataclass
class AgentConfig:
    """Configuration for the agent system."""
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "10"))
    planning_depth: int = int(os.getenv("PLANNING_DEPTH", "3"))
    decision_threshold: float = float(os.getenv("DECISION_THRESHOLD", "0.7"))
    enable_reflection: bool = os.getenv("ENABLE_REFLECTION", "true").lower() == "true"
    enable_self_correction: bool = os.getenv("ENABLE_SELF_CORRECTION", "true").lower() == "true"


@dataclass
class APIConfig:
    """Configuration for the API server."""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    cors_origins: list = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(","))
    api_key: Optional[str] = os.getenv("API_KEY", None)
    rate_limit: int = int(os.getenv("RATE_LIMIT", "60"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"


@dataclass
class LogConfig:
    """Configuration for logging."""
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: str = os.getenv("LOG_DIR", "data/logs")
    log_format: str = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_to_file: bool = os.getenv("LOG_TO_FILE", "true").lower() == "true"
    log_to_console: bool = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"


@dataclass
class AstraConfig:
    """Master configuration for AstraMind AI."""
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LogConfig = field(default_factory=LogConfig)

    @classmethod
    def from_env(cls) -> "AstraConfig":
        """Load configuration from environment variables."""
        return cls()

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)
