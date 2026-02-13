"""Gemini (Google) client implementations."""

from typing import List, Optional
from google import genai
from google.genai import types
from .base import BaseLLMClient, BaseEmbeddingClient


class GeminiLLMClient(BaseLLMClient):
    """Gemini LLM client wrapper."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-exp",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Initialize Gemini LLM client.
        
        Args:
            api_key: Google API key
            model: Model name (default: gemini-2.0-flash-exp)
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens in response
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a completion using Gemini.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters
            
        Returns:
            The generated text response
        """
        config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.temperature),
        )
        if self.max_tokens:
            config.max_output_tokens = kwargs.get("max_tokens", self.max_tokens)
            
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return response.text

    def generate_stream(self, prompt: str, **kwargs):
        """Generate a streaming completion using Gemini.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the generated text response
        """
        config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.temperature),
        )
        if self.max_tokens:
            config.max_output_tokens = kwargs.get("max_tokens", self.max_tokens)
            
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=config,
        ):
            if chunk.text:
                yield chunk.text


class GeminiEmbeddingClient(BaseEmbeddingClient):
    """Gemini Embedding client wrapper."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-004",
    ):
        """Initialize Gemini Embedding client.
        
        Args:
            api_key: Google API key
            model: Embedding model name (default: text-embedding-004)
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self._dimension = None

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for a single text.
        
        Args:
            text: The input text to embed
            
        Returns:
            A list of floats representing the embedding vector
        """
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
        )
        
        return result.embeddings[0].values

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: A list of input texts to embed
            
        Returns:
            A list of embedding vectors
        """
        embeddings = []
        for text in texts:
            result = self.client.models.embed_content(
                model=self.model,
                contents=text,
            )
            embeddings.append(result.embeddings[0].values)
        
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.
        
        Returns:
            The dimension size of embeddings produced by this client
        """
        if self._dimension is None:
            # Generate a dummy embedding to get dimension
            dummy_embedding = self.embed_text("test")
            self._dimension = len(dummy_embedding)
        
        return self._dimension

