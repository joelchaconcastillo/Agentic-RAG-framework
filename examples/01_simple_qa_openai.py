"""Example: Simple Q&A Agent with OpenAI

This example demonstrates how to use the Simple Q&A agent with OpenAI.
"""

import os
from dotenv import load_dotenv
from agentic_rag.clients import OpenAILLMClient, OpenAIEmbeddingClient
from agentic_rag.use_cases import SimpleQAAgent

# Load environment variables
load_dotenv()


def main():
    # Initialize clients
    llm_client = OpenAILLMClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4",
    )
    
    embedding_client = OpenAIEmbeddingClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small",
    )
    
    # Create Q&A agent
    qa_agent = SimpleQAAgent(
        llm_client=llm_client,
        embedding_client=embedding_client,
        persist_directory="./chroma_db_qa_openai",
    )
    
    # Sample knowledge base about Python
    documents = [
        "Python is a high-level, interpreted programming language known for its simplicity and readability.",
        "Python was created by Guido van Rossum and first released in 1991.",
        "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        "Popular Python frameworks include Django for web development, NumPy for numerical computing, and TensorFlow for machine learning.",
        "Python uses indentation to define code blocks instead of curly braces or keywords.",
    ]
    
    # Load knowledge base
    qa_agent.load_knowledge_base(documents)
    
    # Ask questions
    questions = [
        "Who created Python?",
        "What programming paradigms does Python support?",
        "What are some popular Python frameworks?",
    ]
    
    print("\n" + "="*80)
    print("Simple Q&A Agent - OpenAI Example")
    print("="*80 + "\n")
    
    for question in questions:
        print(f"Q: {question}")
        answer = qa_agent.ask(question)
        print(f"A: {answer}\n")
        print("-"*80 + "\n")


if __name__ == "__main__":
    main()
