"""Configuration management for the RAG framework."""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class RAGConfig(BaseSettings):
    """Configuration for the RAG framework."""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    openai_llm_model: str = Field("gpt-4", alias="OPENAI_LLM_MODEL")
    openai_embedding_model: str = Field("text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    
    # Google Gemini Configuration
    google_api_key: Optional[str] = Field(None, alias="GOOGLE_API_KEY")
    gemini_llm_model: str = Field("gemini-pro", alias="GEMINI_LLM_MODEL")
    gemini_embedding_model: str = Field("models/embedding-001", alias="GEMINI_EMBEDDING_MODEL")
    
    # Default Provider
    default_provider: str = Field("openai", alias="DEFAULT_PROVIDER")
    
    # ChromaDB Configuration
    chroma_persist_directory: str = Field("./chroma_db", alias="CHROMA_PERSIST_DIRECTORY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def get_config() -> RAGConfig:
    """Get the configuration instance.
    
    Returns:
        RAGConfig instance loaded from environment
    """
    return RAGConfig()
