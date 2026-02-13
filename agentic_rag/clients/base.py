"""Base client abstractions for LLM and Embedding providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a completion for the given prompt.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific parameters
            
        Returns:
            The generated text response
        """
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs):
        """Generate a streaming completion for the given prompt.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Chunks of the generated text response
        """
        pass


class BaseEmbeddingClient(ABC):
    """Abstract base class for Embedding clients."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for a single text.
        
        Args:
            text: The input text to embed
            
        Returns:
            A list of floats representing the embedding vector
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: A list of input texts to embed
            
        Returns:
            A list of embedding vectors
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.
        
        Returns:
            The dimension size of embeddings produced by this client
        """
        pass
