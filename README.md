# Embedding Service

A standalone Python library for text embeddings and chat models supporting multiple LLM providers (Ollama, OpenAI-compatible APIs).

## Features

- Unified interface for embeddings across providers
- Support for Ollama (local models)
- Support for OpenAI-compatible APIs
- Chat model abstraction with LangChain
- Easy switching between providers
- RESTful API service (optional)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start - Library Usage

```python
from embedding_service import build_embeddings, build_chat_model

# Use Ollama
embeddings = build_embeddings(
    provider="ollama",
    embed_model="mxbai-embed-large",
    base_url="http://localhost:11434"
)
vector = embeddings.embed_query("Hello world")

# Use OpenAI-compatible API
embeddings = build_embeddings(
    provider="openai-compatible",
    embed_model="text-embedding-ada-002",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key"
)

# Chat models
chat = build_chat_model(
    provider="ollama",
    llm_model="qwen2.5:3b",
    base_url="http://localhost:11434"
)
response = chat.invoke("What is AI?")
print(response.content)
```

## API Service Usage

Start the REST API server:

```bash
# Using Ollama
PROVIDER=ollama EMBED_MODEL=mxbai-embed-large uvicorn embedding_service.api:app --reload

# Using OpenAI-compatible API
PROVIDER=openai-compatible EMBED_MODEL=text-embedding-ada-002 \
OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=your-key \
uvicorn embedding_service.api:app --reload
```

API endpoints:

```bash
# Embed single query
curl -X POST http://localhost:8000/embed/query \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'

# Embed multiple documents
curl -X POST http://localhost:8000/embed/documents \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello", "World"]}'

# Chat completion
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is AI?"}'

# Health check
curl http://localhost:8000/health
```

## Environment Variables

- `PROVIDER`: `ollama` or `openai-compatible` (default: `ollama`)
- `EMBED_MODEL`: Embedding model name (default: `mxbai-embed-large`)
- `LLM_MODEL`: Chat model name (default: `qwen2.5:3b`)
- `OLLAMA_BASE_URL`: Ollama server URL (default: `http://localhost:11434`)
- `OPENAI_BASE_URL`: OpenAI-compatible API base URL
- `OPENAI_API_KEY`: API key for OpenAI-compatible services

## Project Origin

This module was extracted from the `airport-customer` project as a reusable component.
