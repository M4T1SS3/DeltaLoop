# Contributing to DeltaLoop

Thank you for your interest in contributing to DeltaLoop! We welcome contributions from the community.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/deltaloop/deltaloop.git
cd deltaloop

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## Project Structure

```
deltaloop/
├── deltaloop/           # Core library
│   ├── schema.py       # Data models
│   ├── distill.py      # Log processing
│   ├── train.py        # Training orchestration
│   ├── eval.py         # Evaluation framework
│   ├── pipeline.py     # End-to-end workflows
│   ├── utils.py        # Helper functions
│   ├── cli.py          # Command-line interface
│   ├── adapters/       # Framework integrations
│   └── backends/       # Training backends
├── examples/           # Example implementations
├── tests/              # Unit tests
└── docs/               # Documentation
```

## Priority Contribution Areas

### 1. Framework Adapters

Add support for new agent frameworks:

```python
# deltaloop/adapters/autogen.py
from deltaloop.schema import AgentTrace

class AutoGenLogger:
    """Adapter for AutoGen framework."""

    def log_interaction(self, ...):
        # Convert AutoGen format to AgentTrace
        pass
```

### 2. Evaluation Tasks

Create domain-specific evaluation tasks:

```python
# Custom evaluation task
from deltaloop.eval import EvalTask

class MyCustomTask(EvalTask):
    def __init__(self):
        super().__init__(
            name="custom_task",
            prompt="Your evaluation prompt",
            expected_keywords=["keyword1", "keyword2"]
        )

    def score(self, output: str) -> float:
        # Your scoring logic
        return 0.0  # 0.0 to 1.0
```

### 3. Examples

Add real-world use case examples:

- Customer support agents
- Code generation agents
- Research assistants
- Data analysis agents

### 4. Training Backends

Integrate new training engines:

```python
# deltaloop/backends/my_backend.py
from deltaloop.train import TrainingBackend

class MyBackend(TrainingBackend):
    def train(self, config):
        # Your training implementation
        pass
```

## Code Style

We use:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking

```bash
# Format code
black deltaloop/

# Lint
ruff check deltaloop/

# Type check
mypy deltaloop/
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=deltaloop
```

## Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Add** tests for new functionality
5. **Run** code quality checks (black, ruff, mypy)
6. **Commit** your changes (`git commit -m 'Add amazing feature'`)
7. **Push** to your branch (`git push origin feature/amazing-feature`)
8. **Open** a Pull Request

## Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests when relevant

## Questions?

- Open an [issue](https://github.com/deltaloop/deltaloop/issues)
- Start a [discussion](https://github.com/deltaloop/deltaloop/discussions)

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
