# Contributing to Agentic RAG Framework

Thank you for your interest in contributing to the Agentic RAG Framework! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Adding New Providers](#adding-new-providers)
- [Adding New Use Cases](#adding-new-use-cases)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Respect differing viewpoints

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/Agentic-RAG-framework.git`
3. Add upstream remote: `git remote add upstream https://github.com/joelchaconcastillo/Agentic-RAG-framework.git`

## Development Setup

### Using UV (Recommended)

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies with dev tools
uv pip install -e ".[dev]"
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

## Making Changes

### Workflow

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Test your changes
4. Commit with clear messages
5. Push to your fork
6. Open a Pull Request

### Commit Messages

Use clear, descriptive commit messages:

```
Add support for Anthropic Claude provider

- Implement ClaudeLLMClient
- Implement ClaudeEmbeddingClient
- Add example usage
- Update documentation
```

## Adding New Providers

To add a new LLM/Embedding provider:

### 1. Create Client Files

Create a new file in `agentic_rag/clients/`:

```python
# agentic_rag/clients/your_provider_client.py
from typing import List, Optional
from .base import BaseLLMClient, BaseEmbeddingClient


class YourProviderLLMClient(BaseLLMClient):
    """Your provider LLM client wrapper."""

    def __init__(self, api_key: str, model: str = "default-model"):
        # Initialize your provider's client
        pass

    def generate(self, prompt: str, **kwargs) -> str:
        # Implement text generation
        pass

    def generate_stream(self, prompt: str, **kwargs):
        # Implement streaming generation
        pass


class YourProviderEmbeddingClient(BaseEmbeddingClient):
    """Your provider embedding client wrapper."""

    def __init__(self, api_key: str, model: str = "default-embedding"):
        # Initialize your provider's client
        pass

    def embed_text(self, text: str) -> List[float]:
        # Implement single text embedding
        pass

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Implement batch text embedding
        pass

    def get_embedding_dimension(self) -> int:
        # Return embedding dimension
        pass
```

### 2. Update Package Exports

Add to `agentic_rag/clients/__init__.py`:

```python
from .your_provider_client import YourProviderLLMClient, YourProviderEmbeddingClient

__all__ = [
    # ... existing exports
    "YourProviderLLMClient",
    "YourProviderEmbeddingClient",
]
```

### 3. Update Main Package

Add to `agentic_rag/__init__.py`:

```python
from agentic_rag.clients import (
    # ... existing imports
    YourProviderLLMClient,
    YourProviderEmbeddingClient,
)
```

### 4. Add Dependencies

Update `pyproject.toml`:

```toml
dependencies = [
    # ... existing dependencies
    "your-provider-sdk>=1.0.0",
]
```

### 5. Add Configuration

Update `agentic_rag/config/settings.py`:

```python
class RAGConfig(BaseSettings):
    # ... existing config
    
    # Your Provider Configuration
    your_provider_api_key: Optional[str] = Field(None, alias="YOUR_PROVIDER_API_KEY")
    your_provider_llm_model: str = Field("default-model", alias="YOUR_PROVIDER_LLM_MODEL")
    your_provider_embedding_model: str = Field("default-embedding", alias="YOUR_PROVIDER_EMBEDDING_MODEL")
```

### 6. Create Example

Create `examples/XX_simple_qa_yourprovider.py`:

```python
import os
from dotenv import load_dotenv
from agentic_rag.clients import YourProviderLLMClient, YourProviderEmbeddingClient
from agentic_rag.use_cases import SimpleQAAgent

load_dotenv()

def main():
    llm_client = YourProviderLLMClient(api_key=os.getenv("YOUR_PROVIDER_API_KEY"))
    embedding_client = YourProviderEmbeddingClient(api_key=os.getenv("YOUR_PROVIDER_API_KEY"))
    
    qa_agent = SimpleQAAgent(llm_client, embedding_client)
    # ... rest of example
```

### 7. Update Documentation

- Update README.md with provider information
- Update .env.example with new environment variables
- Document any provider-specific features

## Adding New Use Cases

To add a new use case:

### 1. Create Use Case File

Create `agentic_rag/use_cases/your_use_case.py`:

```python
from typing import List, Dict, Any
from agentic_rag.clients.base import BaseLLMClient, BaseEmbeddingClient
from agentic_rag.storage import ChromaDBStorage


class YourUseCaseAgent:
    """Your use case description."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        embedding_client: BaseEmbeddingClient,
        persist_directory: str = "./chroma_db_your_use_case",
    ):
        # Initialize your agent
        pass

    def your_method(self, input: str) -> str:
        # Implement your use case logic
        pass
```

### 2. Update Package Exports

Add to `agentic_rag/use_cases/__init__.py`

### 3. Create Example

Create an example script in `examples/`

### 4. Document

Add documentation to README.md

## Testing

### Run Tests

```bash
# Run all tests
uv run python tests/test_basic.py

# Or with pytest
pytest tests/

# Run specific test
pytest tests/test_your_feature.py
```

### Write Tests

Add tests to `tests/` directory:

```python
def test_your_feature():
    # Arrange
    client = YourClient()
    
    # Act
    result = client.your_method()
    
    # Assert
    assert result is not None
```

## Code Style

### Python Style

We use:
- **Black** for code formatting
- **Ruff** for linting
- **Type hints** for better code clarity

```bash
# Format code
uv run black agentic_rag/

# Lint code
uv run ruff check agentic_rag/
```

### Docstrings

Use Google-style docstrings:

```python
def your_function(param1: str, param2: int) -> bool:
    """Brief description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
    """
    pass
```

### Type Hints

Always use type hints:

```python
from typing import List, Dict, Optional

def process_data(
    items: List[str],
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    pass
```

## Submitting Changes

### Pull Request Process

1. Update documentation
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG if applicable
5. Submit PR with clear description

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Updated documentation

## Checklist
- [ ] Code follows project style
- [ ] Self-reviewed code
- [ ] Commented complex code
- [ ] Updated documentation
- [ ] No new warnings
```

## Questions?

- Open an issue for questions
- Join discussions
- Check existing issues and PRs

Thank you for contributing! 🎉
