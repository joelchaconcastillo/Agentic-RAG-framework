"""Use Case 2: Multi-Document Comparison Agent

This use case demonstrates an agent that compares information across
multiple documents and provides comparative analysis.
"""

from typing import List, Dict, Any
from agentic_rag.clients.base import BaseLLMClient, BaseEmbeddingClient
from agentic_rag.storage import ChromaDBStorage


class MultiDocComparisonAgent:
    """Agent for comparing information across multiple documents."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        embedding_client: BaseEmbeddingClient,
        persist_directory: str = "./chroma_db_comparison",
    ):
        """Initialize Multi-Document Comparison Agent.
        
        Args:
            llm_client: LLM client for generation
            embedding_client: Embedding client for vectorization
            persist_directory: Directory to persist the vector database
        """
        self.llm_client = llm_client
        self.storage = ChromaDBStorage(
            embedding_client=embedding_client,
            collection_name="comparison_documents",
            persist_directory=persist_directory,
        )

    def load_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None):
        """Load documents for comparison.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata (e.g., source, category)
        """
        self.storage.add_documents(documents, metadatas)
        print(f"Loaded {len(documents)} documents for comparison.")

    def compare(self, topic: str, n_documents: int = 5) -> str:
        """Compare how different documents address a topic.
        
        Args:
            topic: The topic to compare across documents
            n_documents: Number of documents to retrieve
            
        Returns:
            Comparative analysis
        """
        # Retrieve relevant documents
        results = self.storage.query(topic, n_results=n_documents)
        docs = results["documents"]
        metadatas = results["metadatas"]
        
        if not docs:
            return "No relevant documents found for comparison."
        
        # Build context with source information
        context_parts = []
        for i, (doc, meta) in enumerate(zip(docs, metadatas)):
            source = meta.get("source", f"Document {i+1}") if meta else f"Document {i+1}"
            context_parts.append(f"Source [{source}]:\n{doc}")
        
        context = "\n\n".join(context_parts)
        
        # Generate comparison
        prompt = f"""Compare and contrast how different sources address the following topic.
Highlight similarities, differences, and unique perspectives.

Topic: {topic}

Sources:
{context}

Provide a detailed comparative analysis:"""
        
        return self.llm_client.generate(prompt)

    def find_consensus(self, question: str, n_documents: int = 5) -> str:
        """Find consensus or disagreement across documents.
        
        Args:
            question: The question to analyze
            n_documents: Number of documents to retrieve
            
        Returns:
            Analysis of consensus/disagreement
        """
        results = self.storage.query(question, n_results=n_documents)
        docs = results["documents"]
        
        if not docs:
            return "No relevant documents found."
        
        context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
        
        prompt = f"""Analyze the following documents to determine if there is consensus or disagreement
on the given question. Identify points of agreement, disagreement, and any nuanced positions.

Question: {question}

Documents:
{context}

Analysis:"""
        
        return self.llm_client.generate(prompt)
