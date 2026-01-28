from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _get_int(name: str, default: int) -> int:
    """Get integer from environment variable."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: Optional[float]) -> Optional[float]:
    """Get float from environment variable."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class Settings:
    """Configuration settings for embedding service."""
    
    provider: str
    llm_model: str
    embed_model: str
    openai_base_url: Optional[str]
    openai_api_key: Optional[str]
    ollama_base_url: Optional[str]

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            provider=os.getenv("PROVIDER", "ollama"),
            llm_model=os.getenv("LLM_MODEL", "qwen2.5:3b"),
            embed_model=os.getenv("EMBED_MODEL", "mxbai-embed-large"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )


def clamp_provider(provider: str) -> str:
    """Normalize provider name to supported values."""
    provider = (provider or "").strip().lower()
    if provider not in {"ollama", "openai-compatible"}:
        return "ollama"
    return provider
