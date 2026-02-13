"""Use Case 3: Conversational RAG with Memory

This use case demonstrates a conversational agent that maintains
context across multiple turns using LangGraph.
"""

from typing import List, Dict, Any, TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

from agentic_rag.clients.base import BaseLLMClient, BaseEmbeddingClient
from agentic_rag.storage import ChromaDBStorage


class ConversationState(TypedDict):
    """State for conversational agent."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    retrieved_documents: List[str]
    conversation_history: str
    answer: str


class ConversationalRAGAgent:
    """Conversational RAG agent with memory."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        embedding_client: BaseEmbeddingClient,
        persist_directory: str = "./chroma_db_conversation",
    ):
        """Initialize Conversational RAG Agent.
        
        Args:
            llm_client: LLM client for generation
            embedding_client: Embedding client for vectorization
            persist_directory: Directory to persist the vector database
        """
        self.llm_client = llm_client
        self.storage = ChromaDBStorage(
            embedding_client=embedding_client,
            collection_name="conversation_documents",
            persist_directory=persist_directory,
        )
        self.conversation_history: List[Dict[str, str]] = []
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the conversational workflow."""
        workflow = StateGraph(ConversationState)
        
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()

    def _retrieve(self, state: ConversationState) -> Dict[str, Any]:
        """Retrieve relevant documents considering conversation history."""
        query = state["query"]
        
        # Enhance query with recent context
        if self.conversation_history:
            recent_context = " ".join([
                f"{turn['role']}: {turn['content']}"
                for turn in self.conversation_history[-3:]
            ])
            enhanced_query = f"Context: {recent_context}\n\nCurrent query: {query}"
        else:
            enhanced_query = query
        
        # Retrieve documents
        results = self.storage.query(enhanced_query, n_results=3)
        
        return {
            "retrieved_documents": results["documents"],
        }

    def _generate(self, state: ConversationState) -> Dict[str, Any]:
        """Generate response considering conversation history."""
        query = state["query"]
        docs = state.get("retrieved_documents", [])
        
        # Build conversation context
        history_text = ""
        if self.conversation_history:
            history_text = "Previous conversation:\n" + "\n".join([
                f"{turn['role']}: {turn['content']}"
                for turn in self.conversation_history[-5:]
            ]) + "\n\n"
        
        # Build document context
        docs_text = ""
        if docs:
            docs_text = "Relevant documents:\n" + "\n\n".join([
                f"Document {i+1}: {doc}" for i, doc in enumerate(docs)
            ]) + "\n\n"
        
        prompt = f"""{history_text}{docs_text}Current question: {query}

Provide a helpful answer considering the conversation history and documents:"""
        
        answer = self.llm_client.generate(prompt)
        
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    def load_knowledge_base(self, documents: List[str], metadatas: List[Dict[str, Any]] = None):
        """Load documents into the knowledge base.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
        """
        self.storage.add_documents(documents, metadatas)
        print(f"Loaded {len(documents)} documents into knowledge base.")

    def chat(self, message: str) -> str:
        """Send a message and get a response.
        
        Args:
            message: User message
            
        Returns:
            Agent response
        """
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "query": message,
            "retrieved_documents": [],
            "conversation_history": "",
            "answer": "",
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        answer = final_state["answer"]
        
        # Update conversation history
        self.conversation_history.append({"role": "User", "content": message})
        self.conversation_history.append({"role": "Assistant", "content": answer})
        
        return answer

    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        print("Conversation history reset.")

    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history.
        
        Returns:
            List of conversation turns
        """
        return self.conversation_history
