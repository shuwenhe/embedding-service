from __future__ import annotations

import os
from typing import Optional

from .config import clamp_provider


def build_embeddings(
    provider: str,
    embed_model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Build embeddings instance for the specified provider.
    
    Args:
        provider: "ollama" or "openai-compatible"
        embed_model: Model name (e.g., "mxbai-embed-large", "text-embedding-ada-002")
        base_url: API base URL (required for openai-compatible)
        api_key: API key (required for openai-compatible)
    
    Returns:
        LangChain Embeddings instance
    """
    provider = clamp_provider(provider)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        resolved_base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaEmbeddings(model=embed_model, base_url=resolved_base_url)
    if provider == "openai-compatible":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=embed_model, base_url=base_url, api_key=api_key)
    raise ValueError(f"Unsupported provider: {provider}")


def build_chat_model(
    provider: str,
    llm_model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
):
    """
    Build chat model instance for the specified provider.
    
    Args:
        provider: "ollama" or "openai-compatible"
        llm_model: Model name (e.g., "qwen2.5:3b", "gpt-4")
        base_url: API base URL (required for openai-compatible)
        api_key: API key (required for openai-compatible)
        temperature: Sampling temperature (default: 0.0)
    
    Returns:
        LangChain ChatModel instance
    """
    provider = clamp_provider(provider)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        resolved_base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=llm_model, base_url=resolved_base_url, temperature=temperature)
    if provider == "openai-compatible":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=llm_model, base_url=base_url, api_key=api_key, temperature=temperature)
    raise ValueError(f"Unsupported provider: {provider}")
