"""Base RAG Agent using LangGraph for agentic behavior."""

from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

from agentic_rag.clients.base import BaseLLMClient
from agentic_rag.storage import ChromaDBStorage


class AgentState(TypedDict):
    """State for the RAG agent."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    retrieved_documents: List[str]
    answer: str
    needs_retrieval: bool


class RAGAgent:
    """RAG Agent with agentic capabilities using LangGraph."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        storage: ChromaDBStorage,
        max_iterations: int = 5,
    ):
        """Initialize the RAG Agent.
        
        Args:
            llm_client: LLM client for generation
            storage: ChromaDB storage for retrieval
            max_iterations: Maximum iterations for the agent
        """
        self.llm_client = llm_client
        self.storage = storage
        self.max_iterations = max_iterations
        
        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        # Add edges
        workflow.add_conditional_edges(
            "analyze_query",
            self._should_retrieve,
            {
                "retrieve": "retrieve",
                "generate": "generate",
            }
        )
        
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()

    def _analyze_query(self, state: AgentState) -> Dict[str, Any]:
        """Analyze the query to determine if retrieval is needed."""
        query = state["query"]
        
        # Simple heuristic: if it's a factual question, we need retrieval
        prompt = f"""Analyze this query and determine if it requires factual information from documents.
Query: {query}

Respond with only 'YES' if retrieval is needed, or 'NO' if you can answer directly."""
        
        response = self.llm_client.generate(prompt, temperature=0.1)
        needs_retrieval = "YES" in response.upper()
        
        return {
            "needs_retrieval": needs_retrieval,
        }

    def _should_retrieve(self, state: AgentState) -> str:
        """Decision function for conditional edge."""
        return "retrieve" if state.get("needs_retrieval", True) else "generate"

    def _retrieve(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve relevant documents."""
        query = state["query"]
        
        # Query the vector store
        results = self.storage.query(query, n_results=5)
        
        return {
            "retrieved_documents": results["documents"],
        }

    def _generate(self, state: AgentState) -> Dict[str, Any]:
        """Generate the final answer."""
        query = state["query"]
        docs = state.get("retrieved_documents", [])
        
        if docs:
            context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
            prompt = f"""Based on the following documents, answer the question.

Documents:
{context}

Question: {query}

Answer:"""
        else:
            prompt = f"""Answer the following question directly.

Question: {query}

Answer:"""
        
        answer = self.llm_client.generate(prompt)
        
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG agent.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with answer and metadata
        """
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "query": question,
            "retrieved_documents": [],
            "answer": "",
            "needs_retrieval": True,
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return {
            "answer": final_state["answer"],
            "retrieved_documents": final_state.get("retrieved_documents", []),
            "needs_retrieval": final_state.get("needs_retrieval", True),
        }

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add documents to the knowledge base.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
        """
        self.storage.add_documents(documents, metadatas)
