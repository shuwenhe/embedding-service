"""REST API for embedding service."""

from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import Settings
from .embeddings import build_chat_model, build_embeddings


class QueryRequest(BaseModel):
    text: str


class DocumentsRequest(BaseModel):
    texts: List[str]


class ChatRequest(BaseModel):
    message: str


class EmbeddingResponse(BaseModel):
    embedding: List[float]


class EmbeddingsResponse(BaseModel):
    embeddings: List[List[float]]


class ChatResponse(BaseModel):
    response: str


class HealthResponse(BaseModel):
    status: str
    provider: str
    embed_model: str
    llm_model: str


def create_app() -> FastAPI:
    """Create FastAPI application."""
    settings = Settings.from_env()
    app = FastAPI(title="Embedding Service API")
    
    # Initialize clients
    try:
        embeddings = build_embeddings(
            settings.provider,
            settings.embed_model,
            base_url=settings.ollama_base_url if settings.provider == "ollama" else settings.openai_base_url,
            api_key=settings.openai_api_key,
        )
        chat_model = build_chat_model(
            settings.provider,
            settings.llm_model,
            base_url=settings.ollama_base_url if settings.provider == "ollama" else settings.openai_base_url,
            api_key=settings.openai_api_key,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize models: {e}")
    
    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="ok",
            provider=settings.provider,
            embed_model=settings.embed_model,
            llm_model=settings.llm_model,
        )
    
    @app.post("/embed/query", response_model=EmbeddingResponse)
    def embed_query(request: QueryRequest) -> EmbeddingResponse:
        """Embed a single query text."""
        try:
            vector = embeddings.embed_query(request.text)
            return EmbeddingResponse(embedding=vector)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/embed/documents", response_model=EmbeddingsResponse)
    def embed_documents(request: DocumentsRequest) -> EmbeddingsResponse:
        """Embed multiple documents."""
        try:
            vectors = embeddings.embed_documents(request.texts)
            return EmbeddingsResponse(embeddings=vectors)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        """Chat completion endpoint."""
        try:
            from langchain_core.messages import HumanMessage
            
            response = chat_model.invoke([HumanMessage(content=request.message)])
            return ChatResponse(response=response.content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


app = create_app()
