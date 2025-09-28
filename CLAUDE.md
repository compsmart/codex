# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Evo AI is a cutting-edge AI assistant with human-like memory and emotional intelligence. It features a neuromorphic memory system with temporal dynamics, automatic emotion detection, and somatic marker decision-making. Uses local Llama 3.2-7B model via Ollama with SQLite-based memory, QLoRA fine-tuning, and MCP protocol integration for external tools.

## Common Development Commands

### Installation and Setup
```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Initialize Evo AI
evo init
```

### Testing
```bash
# Run all tests
pytest

# Run specific test types
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests

# Run tests with verbose output
pytest -v

# Run demo script (tests core functionality)
python scripts/run_demo.py
```

### Code Quality
```bash
# Format code
black .

# Sort imports
isort .

# Type checking
mypy src/evo

# Linting
flake8 src tests
```

### Running Evo AI
```bash
# Start interactive chat
evo chat

# Check system status
evo status

# Search memory
evo memory search "query"

# View memory statistics
evo memory stats

# View emotional insights and patterns
evo memory emotions

# Test neuromorphic memory features
python test_neuromorphic_memory.py
python test_emotion_validation.py

# Run database migration (if needed after updates)
python migrate_database.py

# Manage sessions
evo sessions list
```

## Architecture Overview

Evo AI uses a modular 6-layer architecture:

1. **User Interface Layer** (`src/evo/cli/`) - Rich terminal interface with session management
2. **Agent Layer** (`src/evo/agent/`) - Core conversation handling and context management
3. **Integration Layer** (`src/evo/mcp/`) - MCP protocol for external tools and LLM fallback
4. **Processing Layer** - Memory (`src/evo/memory/`) and Learning (`src/evo/learning/`) systems
5. **Storage Layer** - SQLite database with vector embeddings
6. **Infrastructure Layer** (`src/evo/services/`) - Background services for consolidation and monitoring

### Key Components

- **Neuromorphic Memory System** (`src/evo/memory/`): Human-like memory with temporal dynamics, emotional intelligence, and conflict resolution
  - Three memory types (episodic, semantic, procedural) with vector embeddings
  - Activation levels that decay over time (ACTIVE → RESTING → DORMANT → DEEP_SLEEP)
  - Automatic emotion detection and valence tracking
  - Somatic marker decision system for emotionally-guided recommendations
  - Duplicate detection and memory reinforcement
  - Conflict resolution for contradictory information
- **Learning System** (`src/evo/learning/`): QLoRA-based incremental fine-tuning with automated scheduling and emotional pattern consolidation
- **Agent Core** (`src/evo/agent/core.py`): Main conversation handler with memory integration
- **Context Manager** (`src/evo/agent/context.py`): Intelligent memory retrieval and context building
- **MCP Client** (`src/evo/mcp/client.py`): External tool integration and cloud LLM fallback

### Configuration

Primary config location: `~/.evo/config.json`

Environment variables can be set via `.env` file (see `.env.example` for all options).

Key config options:
- `model_name`: Ollama model to use (default: "llama3.2:7b")
- `model_url`: LLM server URL (supports Ollama or OpenAI-compatible APIs)
- `use_memory`: Enable persistent memory (default: true)
- `enable_learning`: Enable QLoRA fine-tuning (default: true)
- `enable_mcp`: Enable external tool integration (default: true)

### Testing LM Studio Integration

For testing with LM Studio instead of Ollama:
1. Copy `config_lmstudio.json` to `~/.evo/config.json`
2. Ensure LM Studio is running on `http://127.0.0.1:1234`
3. Follow the comprehensive test guide in `TESTING_GUIDE.md`

### Database Schema

SQLite database with three main tables:
- `memories`: Core memory storage with vector embeddings
- `memories_fts`: Full-text search index
- `memories_vec`: Vector similarity search (if sqlite-vec available)

Memory types:
- **Episodic**: Specific conversations and interactions
- **Semantic**: General facts and relationships
- **Procedural**: Step-by-step procedures and skills

### Background Services

- **Consolidation Service**: Converts episodic memories to semantic knowledge
- **Learning Scheduler**: Manages QLoRA training jobs
- **Monitoring Service**: Tracks system resources and performance

## Development Guidelines

### Project Structure
```
src/evo/
├── agent/          # Core AI agent logic with context management
├── memory/         # Persistent memory system with embeddings
├── learning/       # QLoRA fine-tuning pipeline
├── mcp/           # MCP protocol integration
├── cli/           # Command-line interface
├── services/      # Background services
└── summarization/ # Context summarization
```

### Testing Structure
- Unit tests: Test individual components in isolation
- Integration tests: Test component interactions
- Use `pytest -m slow` to mark computationally expensive tests
- Demo script provides end-to-end functionality verification

### Code Quality Standards
- Type hints required (`mypy` configuration in `pyproject.toml`)
- Black formatting with 88-character line length
- Import sorting with isort (black-compatible profile)
- Flake8 linting

### Dependencies
- Core: ollama, aiosqlite, pydantic, rich, typer, sentence-transformers
- ML: torch, transformers, peft, bitsandbytes, trl
- External: mcp, httpx, asyncio-mqtt

The codebase is designed for local-first operation with optional cloud integration through MCP protocol.