"""Agentic RAG Framework - A multi-provider RAG framework with agentic capabilities."""

__version__ = "0.1.0"

from agentic_rag.clients import (
    BaseLLMClient,
    BaseEmbeddingClient,
    OpenAILLMClient,
    OpenAIEmbeddingClient,
    GeminiLLMClient,
    GeminiEmbeddingClient,
)
from agentic_rag.storage import ChromaDBStorage
from agentic_rag.agents import RAGAgent

__all__ = [
    "BaseLLMClient",
    "BaseEmbeddingClient",
    "OpenAILLMClient",
    "OpenAIEmbeddingClient",
    "GeminiLLMClient",
    "GeminiEmbeddingClient",
    "ChromaDBStorage",
    "RAGAgent",
]
