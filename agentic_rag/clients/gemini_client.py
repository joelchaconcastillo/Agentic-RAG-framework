"""Gemini (Google) client implementations."""

from typing import List, Optional
import google.generativeai as genai
from .base import BaseLLMClient, BaseEmbeddingClient


class GeminiLLMClient(BaseLLMClient):
    """Gemini LLM client wrapper."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-pro",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Initialize Gemini LLM client.
        
        Args:
            api_key: Google API key
            model: Model name (default: gemini-pro)
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens in response
        """
        genai.configure(api_key=api_key)
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        generation_config = {
            "temperature": temperature,
        }
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
            
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a completion using Gemini.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters
            
        Returns:
            The generated text response
        """
        response = self.model.generate_content(prompt)
        return response.text

    def generate_stream(self, prompt: str, **kwargs):
        """Generate a streaming completion using Gemini.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the generated text response
        """
        response = self.model.generate_content(prompt, stream=True)
        
        for chunk in response:
            if chunk.text:
                yield chunk.text


class GeminiEmbeddingClient(BaseEmbeddingClient):
    """Gemini Embedding client wrapper."""

    def __init__(
        self,
        api_key: str,
        model: str = "models/embedding-001",
    ):
        """Initialize Gemini Embedding client.
        
        Args:
            api_key: Google API key
            model: Embedding model name (default: models/embedding-001)
        """
        genai.configure(api_key=api_key)
        self.model = model
        self._dimension = None

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for a single text.
        
        Args:
            text: The input text to embed
            
        Returns:
            A list of floats representing the embedding vector
        """
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document",
        )
        
        return result['embedding']

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: A list of input texts to embed
            
        Returns:
            A list of embedding vectors
        """
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(result['embedding'])
        
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
