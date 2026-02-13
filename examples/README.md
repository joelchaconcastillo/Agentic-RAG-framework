# Examples

This directory contains example scripts demonstrating various use cases of the Agentic RAG Framework.

## Running Examples with UV

All examples can be run using `uv`:

```bash
# Simple Q&A with OpenAI
uv run python examples/01_simple_qa_openai.py

# Simple Q&A with Gemini
uv run python examples/02_simple_qa_gemini.py

# Multi-Document Comparison
uv run python examples/03_multi_doc_comparison.py

# Conversational RAG
uv run python examples/04_conversational_rag.py
```

## Prerequisites

Make sure you have set up your `.env` file with the appropriate API keys:

```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Example Descriptions

### 01_simple_qa_openai.py
Demonstrates basic Q&A functionality using OpenAI's GPT-4 and embeddings.
- Loads a knowledge base about Python
- Answers questions based on the documents

### 02_simple_qa_gemini.py
Same as above but using Google Gemini.
- Shows how easy it is to switch providers
- Uses Gemini Pro and Gemini embeddings

### 03_multi_doc_comparison.py
Shows how to compare information across multiple documents.
- Loads documents about different programming languages
- Performs comparative analysis
- Finds consensus and differences

### 04_conversational_rag.py
Demonstrates conversational capabilities with memory.
- Maintains context across multiple turns
- Retrieves relevant information from knowledge base
- Shows conversation history

## Customization

Feel free to modify these examples to:
- Use different topics and documents
- Try different models
- Adjust parameters like temperature
- Add your own use cases
