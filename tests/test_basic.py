"""Test script to verify the package structure and imports."""

import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from agentic_rag import (
            BaseLLMClient,
            BaseEmbeddingClient,
            OpenAILLMClient,
            OpenAIEmbeddingClient,
            GeminiLLMClient,
            GeminiEmbeddingClient,
            ChromaDBStorage,
            RAGAgent,
        )
        print("✓ Successfully imported core modules")
    except ImportError as e:
        print(f"✗ Failed to import core modules: {e}")
        return False
    
    try:
        from agentic_rag.use_cases import (
            SimpleQAAgent,
            MultiDocComparisonAgent,
            ConversationalRAGAgent,
        )
        print("✓ Successfully imported use case modules")
    except ImportError as e:
        print(f"✗ Failed to import use case modules: {e}")
        return False
    
    try:
        from agentic_rag.config import RAGConfig, get_config
        print("✓ Successfully imported config modules")
    except ImportError as e:
        print(f"✗ Failed to import config modules: {e}")
        return False
    
    return True


def test_class_structure():
    """Test that classes have the expected structure."""
    print("\nTesting class structure...")
    
    from agentic_rag.clients.base import BaseLLMClient, BaseEmbeddingClient
    
    # Check BaseLLMClient
    assert hasattr(BaseLLMClient, 'generate'), "BaseLLMClient missing 'generate' method"
    assert hasattr(BaseLLMClient, 'generate_stream'), "BaseLLMClient missing 'generate_stream' method"
    print("✓ BaseLLMClient has required methods")
    
    # Check BaseEmbeddingClient
    assert hasattr(BaseEmbeddingClient, 'embed_text'), "BaseEmbeddingClient missing 'embed_text' method"
    assert hasattr(BaseEmbeddingClient, 'embed_texts'), "BaseEmbeddingClient missing 'embed_texts' method"
    assert hasattr(BaseEmbeddingClient, 'get_embedding_dimension'), "BaseEmbeddingClient missing 'get_embedding_dimension' method"
    print("✓ BaseEmbeddingClient has required methods")
    
    from agentic_rag.clients import OpenAILLMClient, GeminiLLMClient
    
    # Check OpenAI client is subclass
    assert issubclass(OpenAILLMClient, BaseLLMClient), "OpenAILLMClient should inherit from BaseLLMClient"
    print("✓ OpenAILLMClient inherits from BaseLLMClient")
    
    # Check Gemini client is subclass
    assert issubclass(GeminiLLMClient, BaseLLMClient), "GeminiLLMClient should inherit from BaseLLMClient"
    print("✓ GeminiLLMClient inherits from BaseLLMClient")
    
    return True


def test_storage():
    """Test ChromaDB storage initialization."""
    print("\nTesting storage initialization...")
    
    from agentic_rag.storage import ChromaDBStorage
    from agentic_rag.clients.base import BaseEmbeddingClient
    from typing import List
    
    # Create a mock embedding client
    class MockEmbeddingClient(BaseEmbeddingClient):
        def embed_text(self, text: str) -> List[float]:
            return [0.1] * 384
        
        def embed_texts(self, texts: List[str]) -> List[List[float]]:
            return [[0.1] * 384 for _ in texts]
        
        def get_embedding_dimension(self) -> int:
            return 384
    
    mock_client = MockEmbeddingClient()
    
    try:
        storage = ChromaDBStorage(
            embedding_client=mock_client,
            collection_name="test_collection",
        )
        print("✓ ChromaDBStorage initialized successfully")
        
        # Test basic operations
        storage.add_documents(["Test document"])
        print("✓ Document added successfully")
        
        result = storage.query("Test query", n_results=1)
        print("✓ Query executed successfully")
        
        # Clean up
        storage.delete_collection()
        print("✓ Collection deleted successfully")
        
        return True
    except Exception as e:
        print(f"✗ Storage test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("Agentic RAG Framework - Package Tests")
    print("="*80 + "\n")
    
    tests_passed = True
    
    # Run tests
    tests_passed &= test_imports()
    tests_passed &= test_class_structure()
    tests_passed &= test_storage()
    
    print("\n" + "="*80)
    if tests_passed:
        print("✓ All tests passed!")
        print("="*80)
        return 0
    else:
        print("✗ Some tests failed")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
