# Agentic RAG Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A flexible, multi-provider Retrieval-Augmented Generation (RAG) framework with agentic capabilities. Built with LangGraph and ChromaDB, supporting both OpenAI and Google Gemini providers.

## Features

- **Multi-Provider Support**: Seamlessly switch between OpenAI and Google Gemini
- **Separated Clients**: Independent LLM and embedding clients for maximum flexibility
- **Vector Storage**: ChromaDB integration for efficient document retrieval
- **Agentic Behavior**: LangGraph-powered agents with decision-making capabilities
- **Multiple Use Cases**: Pre-built agents for Q&A, document comparison, and conversational RAG
- **UV Compatible**: Managed with UV for fast, reliable dependency management

## Architecture

```
agentic_rag/
├── clients/          # LLM and Embedding client wrappers
│   ├── base.py      # Abstract base classes
│   ├── openai_client.py
│   └── gemini_client.py
├── storage/         # Vector storage implementations
│   └── chromadb_storage.py
├── agents/          # Agentic RAG implementations
│   └── rag_agent.py
├── use_cases/       # Pre-built use case agents
│   ├── simple_qa.py
│   ├── multi_doc_comparison.py
│   └── conversational_rag.py
└── config/          # Configuration management
    └── settings.py
```

## Installation

### Using UV (Recommended)

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/joelchaconcastillo/Agentic-RAG-framework.git
cd Agentic-RAG-framework

# Install dependencies
uv pip install -e .

# Or for development
uv pip install -e ".[dev]"
```

### Using pip

```bash
pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_LLM_MODEL=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Google Gemini Configuration
GOOGLE_API_KEY=your-google-api-key-here
GEMINI_LLM_MODEL=gemini-pro
GEMINI_EMBEDDING_MODEL=models/embedding-001

# Default Provider
DEFAULT_PROVIDER=openai

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

Or copy from the example:

```bash
cp .env.example .env
```

## Quick Start

### Simple Q&A Agent (OpenAI)

```python
from agentic_rag.clients import OpenAILLMClient, OpenAIEmbeddingClient
from agentic_rag.use_cases import SimpleQAAgent

# Initialize clients
llm_client = OpenAILLMClient(api_key="your-key", model="gpt-4")
embedding_client = OpenAIEmbeddingClient(api_key="your-key")

# Create agent
qa_agent = SimpleQAAgent(llm_client, embedding_client)

# Load knowledge base
documents = [
    "Python is a high-level programming language.",
    "Python was created by Guido van Rossum.",
]
qa_agent.load_knowledge_base(documents)

# Ask questions
answer = qa_agent.ask("Who created Python?")
print(answer)
```

### Simple Q&A Agent (Gemini)

```python
from agentic_rag.clients import GeminiLLMClient, GeminiEmbeddingClient
from agentic_rag.use_cases import SimpleQAAgent

# Initialize clients
llm_client = GeminiLLMClient(api_key="your-key")
embedding_client = GeminiEmbeddingClient(api_key="your-key")

# Create agent
qa_agent = SimpleQAAgent(llm_client, embedding_client)

# Use the same way as OpenAI
```

## Use Cases

### 1. Simple Q&A Agent

Basic question-answering from a knowledge base.

```bash
# OpenAI
uv run python examples/01_simple_qa_openai.py

# Gemini
uv run python examples/02_simple_qa_gemini.py
```

### 2. Multi-Document Comparison Agent

Compare and contrast information across multiple documents.

```python
from agentic_rag.use_cases import MultiDocComparisonAgent

agent = MultiDocComparisonAgent(llm_client, embedding_client)
agent.load_documents(documents, metadatas)

# Compare approaches
comparison = agent.compare("topic to compare")

# Find consensus
consensus = agent.find_consensus("question to analyze")
```

```bash
uv run python examples/03_multi_doc_comparison.py
```

### 3. Conversational RAG with Memory

Maintain context across multiple conversation turns.

```python
from agentic_rag.use_cases import ConversationalRAGAgent

agent = ConversationalRAGAgent(llm_client, embedding_client)
agent.load_knowledge_base(documents)

# Have a conversation
response1 = agent.chat("What is climate change?")
response2 = agent.chat("What causes it?")  # Remembers context
response3 = agent.chat("How can we address this?")  # Continues conversation

# Reset when needed
agent.reset_conversation()
```

```bash
uv run python examples/04_conversational_rag.py
```

## Custom Agents

Build your own agents using the base components:

```python
from agentic_rag.clients import OpenAILLMClient, OpenAIEmbeddingClient
from agentic_rag.storage import ChromaDBStorage
from agentic_rag.agents import RAGAgent

# Initialize clients
llm_client = OpenAILLMClient(api_key="your-key")
embedding_client = OpenAIEmbeddingClient(api_key="your-key")

# Create storage
storage = ChromaDBStorage(
    embedding_client=embedding_client,
    collection_name="my_collection",
    persist_directory="./my_db"
)

# Create agent
agent = RAGAgent(llm_client, storage)

# Add documents
agent.add_documents(["document 1", "document 2"])

# Query
result = agent.query("your question")
print(result["answer"])
```

## Client Architecture

The framework uses a wrapper pattern to support multiple providers:

### Base Abstractions

```python
class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs):
        pass

class BaseEmbeddingClient(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass
```

### Provider Implementations

- **OpenAI**: `OpenAILLMClient`, `OpenAIEmbeddingClient`
- **Gemini**: `GeminiLLMClient`, `GeminiEmbeddingClient`

This design makes it easy to add new providers by implementing the base classes.

## Development

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black agentic_rag/
uv run ruff check agentic_rag/
```

## Project Structure

```
Agentic-RAG-framework/
├── agentic_rag/           # Main package
│   ├── clients/           # Provider clients
│   ├── storage/           # Vector storage
│   ├── agents/            # RAG agents
│   ├── use_cases/         # Pre-built use cases
│   └── config/            # Configuration
├── examples/              # Example scripts
├── tests/                 # Test suite
├── pyproject.toml         # Project configuration
├── .env.example           # Environment template
└── README.md             # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for agentic workflows
- Uses [ChromaDB](https://www.trychroma.com/) for vector storage
- Supports [OpenAI](https://openai.com/) and [Google Gemini](https://deepmind.google/technologies/gemini/) providers