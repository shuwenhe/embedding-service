"""Example usage of embedding_service library."""

import os

# Set environment for local Ollama
os.environ["PROVIDER"] = "ollama"
os.environ["EMBED_MODEL"] = "mxbai-embed-large"
os.environ["LLM_MODEL"] = "qwen2.5:3b"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

from embedding_service import build_embeddings, build_chat_model


def example_embeddings():
    """Example: Generate embeddings."""
    print("=== Example 1: Embeddings ===")
    
    try:
        embeddings = build_embeddings(
            provider="ollama",
            embed_model="mxbai-embed-large",
            base_url="http://localhost:11434"
        )
        
        # Embed single query
        query = "What is artificial intelligence?"
        vector = embeddings.embed_query(query)
        print(f"Query: {query}")
        print(f"Vector dimension: {len(vector)}")
        print(f"First 5 values: {vector[:5]}")
        
        # Embed multiple documents
        docs = ["AI is the future", "Machine learning is powerful"]
        vectors = embeddings.embed_documents(docs)
        print(f"\nEmbedded {len(vectors)} documents")
        
    except Exception as e:
        print(f"Error (check if Ollama is running): {e}")


def example_chat():
    """Example: Chat completion."""
    print("\n=== Example 2: Chat Completion ===")
    
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        
        chat = build_chat_model(
            provider="ollama",
            llm_model="qwen2.5:3b",
            base_url="http://localhost:11434"
        )
        
        messages = [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="What is Python?")
        ]
        
        response = chat.invoke(messages)
        print(f"Response: {response.content[:200]}...")
        
    except Exception as e:
        print(f"Error (check if Ollama is running): {e}")


if __name__ == "__main__":
    example_embeddings()
    example_chat()
