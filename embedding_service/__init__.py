"""Embedding Service - Unified interface for embeddings and chat models."""

from .embeddings import build_chat_model, build_embeddings
from .config import Settings, clamp_provider

__version__ = "0.1.0"
__all__ = ["build_embeddings", "build_chat_model", "Settings", "clamp_provider"]
