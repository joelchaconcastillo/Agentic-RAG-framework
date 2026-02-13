"""Example: Conversational RAG Agent

This example demonstrates a conversational agent that maintains context.
"""

import os
from dotenv import load_dotenv
from agentic_rag.clients import OpenAILLMClient, OpenAIEmbeddingClient
from agentic_rag.use_cases import ConversationalRAGAgent

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
    
    # Create conversational agent
    conv_agent = ConversationalRAGAgent(
        llm_client=llm_client,
        embedding_client=embedding_client,
        persist_directory="./chroma_db_conversation",
    )
    
    # Sample knowledge base about climate change
    documents = [
        "Climate change refers to long-term shifts in global temperatures and weather patterns. Since the 1800s, human activities have been the main driver of climate change.",
        "The greenhouse effect is a natural process where certain gases trap heat in Earth's atmosphere. Carbon dioxide, methane, and nitrous oxide are key greenhouse gases.",
        "Renewable energy sources like solar, wind, and hydroelectric power can help reduce carbon emissions and combat climate change.",
        "Deforestation contributes to climate change by reducing the number of trees that absorb CO2. Forest conservation is crucial for climate action.",
        "Electric vehicles produce fewer emissions than traditional gas-powered cars, especially when powered by renewable energy sources.",
    ]
    
    # Load knowledge base
    conv_agent.load_knowledge_base(documents)
    
    print("\n" + "="*80)
    print("Conversational RAG Agent Example")
    print("="*80 + "\n")
    
    # Simulate a conversation
    conversation = [
        "What is climate change?",
        "What causes it?",
        "How can renewable energy help with this issue?",
        "What about electric vehicles?",
    ]
    
    for message in conversation:
        print(f"User: {message}")
        response = conv_agent.chat(message)
        print(f"Assistant: {response}\n")
        print("-"*80 + "\n")
    
    print("\nConversation History:")
    print("="*80)
    for turn in conv_agent.get_history():
        print(f"{turn['role']}: {turn['content']}\n")


if __name__ == "__main__":
    main()
