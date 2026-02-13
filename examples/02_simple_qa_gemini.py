"""Example: Simple Q&A Agent with Gemini

This example demonstrates how to use the Simple Q&A agent with Google Gemini.
"""

import os
from dotenv import load_dotenv
from agentic_rag.clients import GeminiLLMClient, GeminiEmbeddingClient
from agentic_rag.use_cases import SimpleQAAgent

# Load environment variables
load_dotenv()


def main():
    # Initialize clients
    llm_client = GeminiLLMClient(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.0-flash-exp",
    )
    
    embedding_client = GeminiEmbeddingClient(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="text-embedding-004",
    )
    
    # Create Q&A agent
    qa_agent = SimpleQAAgent(
        llm_client=llm_client,
        embedding_client=embedding_client,
        persist_directory="./chroma_db_qa_gemini",
    )
    
    # Sample knowledge base about AI
    documents = [
        "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.",
        "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.",
        "Deep Learning is a subset of machine learning that uses neural networks with multiple layers to analyze various factors of data.",
        "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language.",
        "Computer Vision is an AI field that trains computers to interpret and understand the visual world.",
    ]
    
    # Load knowledge base
    qa_agent.load_knowledge_base(documents)
    
    # Ask questions
    questions = [
        "What is Artificial Intelligence?",
        "What is the difference between Machine Learning and Deep Learning?",
        "What is Natural Language Processing used for?",
    ]
    
    print("\n" + "="*80)
    print("Simple Q&A Agent - Gemini Example")
    print("="*80 + "\n")
    
    for question in questions:
        print(f"Q: {question}")
        answer = qa_agent.ask(question)
        print(f"A: {answer}\n")
        print("-"*80 + "\n")


if __name__ == "__main__":
    main()
