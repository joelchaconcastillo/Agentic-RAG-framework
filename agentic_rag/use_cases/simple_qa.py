"""Use Case 1: Simple Q&A Agent

This use case demonstrates a basic Q&A agent that answers questions
based on a knowledge base of documents.
"""

from typing import List, Dict, Any
from agentic_rag.clients.base import BaseLLMClient, BaseEmbeddingClient
from agentic_rag.storage import ChromaDBStorage
from agentic_rag.agents import RAGAgent


class SimpleQAAgent:
    """Simple Q&A agent for answering questions from a knowledge base."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        embedding_client: BaseEmbeddingClient,
        persist_directory: str = "./chroma_db_qa",
    ):
        """Initialize Simple Q&A Agent.
        
        Args:
            llm_client: LLM client for generation
            embedding_client: Embedding client for vectorization
            persist_directory: Directory to persist the vector database
        """
        self.llm_client = llm_client
        self.storage = ChromaDBStorage(
            embedding_client=embedding_client,
            collection_name="qa_documents",
            persist_directory=persist_directory,
        )
        self.agent = RAGAgent(llm_client, self.storage)

    def load_knowledge_base(self, documents: List[str], metadatas: List[Dict[str, Any]] = None):
        """Load documents into the knowledge base.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
        """
        self.agent.add_documents(documents, metadatas)
        print(f"Loaded {len(documents)} documents into knowledge base.")

    def ask(self, question: str) -> str:
        """Ask a question and get an answer.
        
        Args:
            question: The question to ask
            
        Returns:
            The answer
        """
        result = self.agent.query(question)
        return result["answer"]

    def ask_with_context(self, question: str) -> Dict[str, Any]:
        """Ask a question and get answer with context.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary with answer and retrieved documents
        """
        return self.agent.query(question)
