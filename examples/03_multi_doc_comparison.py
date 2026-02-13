"""Example: Multi-Document Comparison Agent

This example demonstrates how to compare information across multiple documents.
"""

import os
from dotenv import load_dotenv
from agentic_rag.clients import OpenAILLMClient, OpenAIEmbeddingClient
from agentic_rag.use_cases import MultiDocComparisonAgent

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
    
    # Create comparison agent
    comparison_agent = MultiDocComparisonAgent(
        llm_client=llm_client,
        embedding_client=embedding_client,
        persist_directory="./chroma_db_comparison",
    )
    
    # Sample documents about programming languages
    documents = [
        "Python is known for its simple syntax and readability. It's widely used in data science, web development, and automation.",
        "JavaScript is essential for web development. It runs in browsers and on servers with Node.js. It has a large ecosystem of libraries.",
        "Rust focuses on memory safety and performance. It's used for system programming and is gaining popularity for its reliability.",
        "Java is a mature, object-oriented language used in enterprise applications. It's known for its 'write once, run anywhere' philosophy.",
        "Go was designed at Google for simplicity and efficiency. It's popular for building scalable network services and cloud applications.",
    ]
    
    metadatas = [
        {"source": "Python Documentation", "category": "programming"},
        {"source": "JavaScript Guide", "category": "programming"},
        {"source": "Rust Book", "category": "programming"},
        {"source": "Java Tutorial", "category": "programming"},
        {"source": "Go Documentation", "category": "programming"},
    ]
    
    # Load documents
    comparison_agent.load_documents(documents, metadatas)
    
    print("\n" + "="*80)
    print("Multi-Document Comparison Agent Example")
    print("="*80 + "\n")
    
    # Compare approaches
    print("Comparison Topic: Memory Management\n")
    comparison = comparison_agent.compare("memory management and safety")
    print(comparison)
    print("\n" + "-"*80 + "\n")
    
    # Find consensus
    print("Consensus Analysis: Ease of Learning\n")
    consensus = comparison_agent.find_consensus("Which language is easiest to learn?")
    print(consensus)
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
