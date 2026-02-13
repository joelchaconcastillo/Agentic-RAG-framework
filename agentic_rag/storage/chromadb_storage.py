"""ChromaDB storage implementation for vector database."""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from agentic_rag.clients.base import BaseEmbeddingClient


class ChromaDBStorage:
    """ChromaDB wrapper for vector storage and retrieval."""

    def __init__(
        self,
        embedding_client: BaseEmbeddingClient,
        collection_name: str = "rag_documents",
        persist_directory: Optional[str] = None,
    ):
        """Initialize ChromaDB storage.
        
        Args:
            embedding_client: The embedding client to use for vectorization
            collection_name: Name of the collection (default: rag_documents)
            persist_directory: Directory to persist the database
        """
        self.embedding_client = embedding_client
        self.collection_name = collection_name
        
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of document texts to add
            metadatas: Optional list of metadata dicts for each document
            ids: Optional list of IDs for each document
        """
        # Generate embeddings
        embeddings = self.embedding_client.embed_texts(documents)
        
        # Generate IDs if not provided
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query the vector store for similar documents.
        
        Args:
            query_text: The query text
            n_results: Number of results to return (default: 5)
            where: Optional metadata filter
            
        Returns:
            Dictionary containing documents, distances, and metadatas
        """
        # Generate query embedding
        query_embedding = self.embedding_client.embed_text(query_text)
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )
        
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "ids": results["ids"][0] if results["ids"] else [],
        }

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(name=self.collection_name)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            "name": self.collection_name,
            "count": count,
        }

    def update_document(
        self,
        document_id: str,
        document: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update a document in the collection.
        
        Args:
            document_id: ID of the document to update
            document: New document text
            metadata: New metadata
        """
        embedding = self.embedding_client.embed_text(document)
        
        self.collection.update(
            ids=[document_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata] if metadata else None,
        )

    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents from the collection.
        
        Args:
            ids: List of document IDs to delete
        """
        self.collection.delete(ids=ids)
