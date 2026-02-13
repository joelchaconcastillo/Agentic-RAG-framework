"""OpenAI client implementations."""

from typing import List, Optional
from openai import OpenAI
from .base import BaseLLMClient, BaseEmbeddingClient


class OpenAILLMClient(BaseLLMClient):
    """OpenAI LLM client wrapper."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Initialize OpenAI LLM client.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4)
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens in response
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a completion using OpenAI.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            The generated text response
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return response.choices[0].message.content

    def generate_stream(self, prompt: str, **kwargs):
        """Generate a streaming completion using OpenAI.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the generated text response
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        messages = [{"role": "user", "content": prompt}]
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """OpenAI Embedding client wrapper."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
    ):
        """Initialize OpenAI Embedding client.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model name (default: text-embedding-3-small)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._dimension = None

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for a single text.
        
        Args:
            text: The input text to embed
            
        Returns:
            A list of floats representing the embedding vector
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        
        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: A list of input texts to embed
            
        Returns:
            A list of embedding vectors
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        
        return [item.embedding for item in response.data]

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
