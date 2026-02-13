"""Demo script showing the framework capabilities without requiring API keys."""

from typing import List
from agentic_rag.clients.base import BaseLLMClient, BaseEmbeddingClient
from agentic_rag.storage import ChromaDBStorage
from agentic_rag.agents import RAGAgent
from agentic_rag.use_cases import SimpleQAAgent, MultiDocComparisonAgent


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for demonstration."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a mock response."""
        if "who created python" in prompt.lower():
            return "Python was created by Guido van Rossum and first released in 1991."
        elif "programming paradigms" in prompt.lower():
            return "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming."
        elif "popular python frameworks" in prompt.lower():
            return "Popular Python frameworks include Django for web development, NumPy for numerical computing, and TensorFlow for machine learning."
        elif "retrieval" in prompt.lower() or "YES" in prompt.upper():
            return "YES"
        else:
            return f"This is a mock response to: {prompt[:50]}..."
    
    def generate_stream(self, prompt: str, **kwargs):
        """Generate a mock streaming response."""
        response = self.generate(prompt, **kwargs)
        for word in response.split():
            yield word + " "


class MockEmbeddingClient(BaseEmbeddingClient):
    """Mock embedding client for demonstration."""
    
    def embed_text(self, text: str) -> List[float]:
        """Generate a simple hash-based embedding."""
        # Simple deterministic embedding based on text hash
        base_hash = hash(text) % 1000
        return [float(base_hash + i) / 1000 for i in range(384)]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed_text(text) for text in texts]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return 384


def demo_basic_rag():
    """Demonstrate basic RAG functionality."""
    print("\n" + "="*80)
    print("DEMO 1: Basic RAG Agent")
    print("="*80 + "\n")
    
    # Initialize mock clients
    llm_client = MockLLMClient()
    embedding_client = MockEmbeddingClient()
    
    # Create storage
    storage = ChromaDBStorage(
        embedding_client=embedding_client,
        collection_name="demo_basic_rag",
    )
    
    # Create agent
    agent = RAGAgent(llm_client, storage)
    
    # Add documents
    documents = [
        "Python is a high-level, interpreted programming language known for its simplicity and readability.",
        "Python was created by Guido van Rossum and first released in 1991.",
        "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        "Popular Python frameworks include Django for web development, NumPy for numerical computing, and TensorFlow for machine learning.",
    ]
    
    print("Adding documents to knowledge base...")
    agent.add_documents(documents)
    print(f"✓ Added {len(documents)} documents\n")
    
    # Query the agent
    questions = [
        "Who created Python?",
        "What programming paradigms does Python support?",
    ]
    
    for question in questions:
        print(f"Q: {question}")
        result = agent.query(question)
        print(f"A: {result['answer']}")
        print(f"Retrieved {len(result['retrieved_documents'])} documents")
        print()
    
    # Clean up
    storage.delete_collection()
    print("✓ Demo completed\n")


def demo_simple_qa():
    """Demonstrate Simple Q&A use case."""
    print("\n" + "="*80)
    print("DEMO 2: Simple Q&A Agent")
    print("="*80 + "\n")
    
    # Initialize mock clients
    llm_client = MockLLMClient()
    embedding_client = MockEmbeddingClient()
    
    # Create Q&A agent
    qa_agent = SimpleQAAgent(
        llm_client=llm_client,
        embedding_client=embedding_client,
        persist_directory=None,  # Use in-memory storage
    )
    
    # Load knowledge base
    documents = [
        "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines.",
        "Machine Learning is a subset of AI that enables systems to learn from experience.",
        "Deep Learning is a subset of machine learning using neural networks with multiple layers.",
    ]
    
    qa_agent.load_knowledge_base(documents)
    
    # Ask questions
    question = "What is Machine Learning?"
    print(f"Q: {question}")
    answer = qa_agent.ask(question)
    print(f"A: {answer}\n")
    
    # Clean up
    qa_agent.storage.delete_collection()
    print("✓ Demo completed\n")


def demo_multi_doc_comparison():
    """Demonstrate Multi-Document Comparison."""
    print("\n" + "="*80)
    print("DEMO 3: Multi-Document Comparison Agent")
    print("="*80 + "\n")
    
    # Initialize mock clients
    llm_client = MockLLMClient()
    embedding_client = MockEmbeddingClient()
    
    # Create comparison agent
    comparison_agent = MultiDocComparisonAgent(
        llm_client=llm_client,
        embedding_client=embedding_client,
        persist_directory=None,
    )
    
    # Load documents
    documents = [
        "Python is known for its simple syntax and readability. It's widely used in data science.",
        "JavaScript is essential for web development. It runs in browsers and on servers with Node.js.",
        "Rust focuses on memory safety and performance. It's used for system programming.",
    ]
    
    metadatas = [
        {"source": "Python Docs", "category": "programming"},
        {"source": "JS Guide", "category": "programming"},
        {"source": "Rust Book", "category": "programming"},
    ]
    
    comparison_agent.load_documents(documents, metadatas)
    
    # Compare
    print("Comparing programming languages...")
    comparison = comparison_agent.compare("syntax and use cases")
    print(f"Result: {comparison}\n")
    
    # Clean up
    comparison_agent.storage.delete_collection()
    print("✓ Demo completed\n")


def demo_architecture():
    """Show the architecture and design patterns."""
    print("\n" + "="*80)
    print("DEMO 4: Framework Architecture")
    print("="*80 + "\n")
    
    print("Client Separation:")
    print("  ✓ LLM Client (for text generation)")
    print("  ✓ Embedding Client (for vectorization)")
    print("  ✓ Independent initialization and configuration")
    print()
    
    print("Multi-Provider Support:")
    print("  ✓ OpenAI (GPT-4 + text-embedding-3-small)")
    print("  ✓ Google Gemini (gemini-2.0-flash-exp + text-embedding-004)")
    print("  ✓ Easy to add new providers by implementing base classes")
    print()
    
    print("Components:")
    print("  ✓ ChromaDB for vector storage")
    print("  ✓ LangGraph for agentic workflows")
    print("  ✓ Pydantic for configuration management")
    print()
    
    print("Use Cases:")
    print("  ✓ Simple Q&A Agent")
    print("  ✓ Multi-Document Comparison Agent")
    print("  ✓ Conversational RAG with Memory")
    print()
    
    print("UV Compatibility:")
    print("  ✓ Modern Python package management")
    print("  ✓ Fast dependency resolution")
    print("  ✓ Reproducible environments")
    print()


def main():
    """Run all demos."""
    print("="*80)
    print("Agentic RAG Framework - Demonstration")
    print("="*80)
    print()
    print("This demo shows the framework's capabilities using mock clients.")
    print("No API keys required!")
    
    demo_basic_rag()
    demo_simple_qa()
    demo_multi_doc_comparison()
    demo_architecture()
    
    print("="*80)
    print("Demo Complete!")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Set up your .env file with API keys")
    print("  2. Run examples with: uv run python examples/01_simple_qa_openai.py")
    print("  3. Explore the use cases in examples/")
    print()


if __name__ == "__main__":
    main()
