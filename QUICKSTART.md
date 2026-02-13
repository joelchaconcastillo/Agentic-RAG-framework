# Quick Start Guide

This guide will help you get started with the Agentic RAG Framework in just a few minutes.

## Installation

### Option 1: Using UV (Recommended)

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/joelchaconcastillo/Agentic-RAG-framework.git
cd Agentic-RAG-framework

# Install dependencies
uv pip install -e .
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/joelchaconcastillo/Agentic-RAG-framework.git
cd Agentic-RAG-framework

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Demo (No API Keys Required)

Run the demo to see the framework in action without needing API keys:

```bash
# Using UV
uv run python examples/00_demo_mock.py

# Using python directly
python examples/00_demo_mock.py
```

## Setup for Real Usage

### 1. Configure API Keys

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# For OpenAI
OPENAI_API_KEY=sk-your-openai-key-here

# For Google Gemini
GOOGLE_API_KEY=your-google-api-key-here
```

### 2. Choose Your Provider

Set the default provider in `.env`:

```bash
# Use OpenAI
DEFAULT_PROVIDER=openai

# OR use Gemini
DEFAULT_PROVIDER=gemini
```

## Basic Usage

### Simple Q&A with OpenAI

```python
from agentic_rag.clients import OpenAILLMClient, OpenAIEmbeddingClient
from agentic_rag.use_cases import SimpleQAAgent
import os

# Initialize clients
llm_client = OpenAILLMClient(api_key=os.getenv("OPENAI_API_KEY"))
embedding_client = OpenAIEmbeddingClient(api_key=os.getenv("OPENAI_API_KEY"))

# Create agent
qa_agent = SimpleQAAgent(llm_client, embedding_client)

# Load knowledge base
documents = [
    "Python is a high-level programming language.",
    "Python was created by Guido van Rossum in 1991.",
]
qa_agent.load_knowledge_base(documents)

# Ask questions
answer = qa_agent.ask("Who created Python?")
print(answer)
```

### Simple Q&A with Gemini

Just swap the clients:

```python
from agentic_rag.clients import GeminiLLMClient, GeminiEmbeddingClient
from agentic_rag.use_cases import SimpleQAAgent
import os

# Initialize clients
llm_client = GeminiLLMClient(api_key=os.getenv("GOOGLE_API_KEY"))
embedding_client = GeminiEmbeddingClient(api_key=os.getenv("GOOGLE_API_KEY"))

# Use the same way...
qa_agent = SimpleQAAgent(llm_client, embedding_client)
```

## Run Examples

### With UV

```bash
uv run python examples/01_simple_qa_openai.py
uv run python examples/02_simple_qa_gemini.py
uv run python examples/03_multi_doc_comparison.py
uv run python examples/04_conversational_rag.py
```

### With Python

```bash
python examples/01_simple_qa_openai.py
python examples/02_simple_qa_gemini.py
python examples/03_multi_doc_comparison.py
python examples/04_conversational_rag.py
```

## Testing

Run the test suite:

```bash
# Using UV
uv run python tests/test_basic.py

# Using python
python tests/test_basic.py

# Or with pytest (if installed)
pytest tests/
```

## Use Cases

### 1. Simple Q&A
Best for: Direct question answering from a knowledge base
- See: `examples/01_simple_qa_openai.py`
- See: `examples/02_simple_qa_gemini.py`

### 2. Multi-Document Comparison
Best for: Comparing information across different sources
- See: `examples/03_multi_doc_comparison.py`

### 3. Conversational RAG
Best for: Multi-turn conversations with context retention
- See: `examples/04_conversational_rag.py`

## Project Structure

```
Agentic-RAG-framework/
├── agentic_rag/           # Main package
│   ├── clients/           # LLM and embedding clients
│   ├── storage/           # ChromaDB vector storage
│   ├── agents/            # RAG agents with LangGraph
│   ├── use_cases/         # Pre-built use cases
│   └── config/            # Configuration management
├── examples/              # Example scripts
├── tests/                 # Test suite
├── .env.example           # Environment template
└── pyproject.toml         # Project configuration
```

## Next Steps

1. **Explore Examples**: Check out the `examples/` directory for more use cases
2. **Read the Docs**: See `README.md` for detailed documentation
3. **Build Your Own**: Use the base classes to create custom agents
4. **Contribute**: Submit issues or PRs on GitHub

## Getting Help

- **Issues**: Open an issue on GitHub
- **Examples**: Check the `examples/` directory
- **Documentation**: Read the main README.md

## Common Issues

### "Module not found"
Make sure you've installed the package:
```bash
uv pip install -e .
# or
pip install -e .
```

### "API key not found"
Make sure you've created a `.env` file with your API keys.

### "ChromaDB collection already exists"
If you get collection errors, try deleting the `chroma_db` directory:
```bash
rm -rf chroma_db*
```

## Tips

1. **Start with the demo**: Run `examples/00_demo_mock.py` first
2. **Use UV**: It's faster and more reliable than pip
3. **Check examples**: Each use case has a complete example
4. **Experiment**: Try different providers and models
5. **Read docstrings**: All classes and methods are well-documented

Happy building! 🚀
