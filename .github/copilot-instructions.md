# GitHub Copilot Instructions for Evo AI

This project is **Evo AI** - a revolutionary AI assistant with neuromorphic memory, emotional intelligence, incremental fine-tuning, and tool integration. Understanding the advanced architecture and patterns is crucial for effective contributions.

## Core Architecture 

Evo uses a **6-layer modular architecture** with neuromorphic memory capabilities:
- **CLI** (`src/evo/cli/`) - Typer-based commands with rich terminal UI
- **Agent** (`src/evo/agent/`) - Core conversation logic with session management
- **Memory** (`src/evo/memory/`) - Neuromorphic memory system with emotional intelligence
- **Learning** (`src/evo/learning/`) - QLoRA fine-tuning pipeline  
- **MCP** (`src/evo/mcp/`) - Model Context Protocol for tool integration
- **Services** (`src/evo/services/`) - Background consolidation and monitoring

### Key Integration Patterns

**EvoAgent** (`agent/core.py`) orchestrates everything:
```python
# Pattern: Agent uses ContextManager for memory integration
context = await self.context_manager.add_message(session_id, user_message)
messages = await self.context_manager.prepare_context_for_llm(context, include_memory)
```

**MemoryManager** (`memory/manager.py`) coordinates neuromorphic storage:
```python
# Pattern: Memories have embeddings, emotions, and temporal dynamics
embedding = await self.embeddings.embed_text(content)
emotions = detect_emotions(content)  # 12+ emotion types
memory = EpisodicMemory(
    content=content, 
    embedding=embedding,
    emotions=emotions,
    activation_level=1.0,  # Neuromorphic activation
    valence=emotions.valence,
    arousal=emotions.arousal
)
await self.storage.store_memory(memory)
```

**MCPClient** (`mcp/client.py`) handles external tools via stdio/HTTP protocols.

## Development Workflows

### Setup & Testing
```bash
# Development install with all dependencies
pip install -e ".[dev]"

# Run tests by category (uses pytest markers)
pytest -m unit          # Fast unit tests
pytest -m integration   # Component interaction tests  
pytest -m "not slow"    # Skip expensive ML tests

# Demo script tests end-to-end functionality
python scripts/run_demo.py
```

### Model Client Abstraction
The agent supports both **Ollama** (default) and **LM Studio** via adapter pattern:
- `LMStudioAdapter` in `agent/adapters/` for OpenAI-compatible APIs
- Detection based on `model_url` containing localhost/127.0.0.1
- Copy `config_lmstudio.json` to `~/.evo/config.json` for LM Studio testing

### Memory System Details

**Neuromorphic Memory Architecture** with emotional intelligence:
- **Temporal Dynamics**: Activation levels (1.0 → 0.0) with natural decay over time
- **Dormancy States**: ACTIVE → RESTING → DORMANT → DEEP_SLEEP based on access patterns  
- **Emotional Intelligence**: 12+ emotions with valence-arousal-dominance model
- **Somatic Markers**: Decision support using emotional memories from past outcomes
- **Conflict Resolution**: Duplicate detection and contradiction handling

**Three memory types** with enhanced neuromorphic features:
```python
# Episodic: conversations with emotional context and temporal dynamics
await memory_manager.store_episodic_memory(
    content=conversation_summary,
    session_id=session_id,
    user_message=user_input,
    assistant_response=response,
    emotions={"joy": 0.8, "curiosity": 0.6},
    valence=0.7,  # Positive emotional tone
    activation_level=1.0
)

# Semantic: facts with confidence and emotional significance
await memory_manager.store_semantic_memory(
    content=fact_description,
    subject="Python", predicate="is", object="programming language",
    confidence=0.9,
    emotions={"trust": 0.7},
    outcome_type="SUCCESS"
)

# Procedural: skills with success tracking and emotional outcomes
await memory_manager.store_procedural_memory(
    content=procedure_description,
    steps=["step1", "step2", "step3"],
    success_rate=0.85,
    emotions={"satisfaction": 0.6}
)
```

**Enhanced database schema** with 15+ neuromorphic fields:
```sql
-- Temporal dynamics
activation_level REAL DEFAULT 1.0,
dormancy_state TEXT DEFAULT 'active',
last_reactivation TIMESTAMP,

-- Emotional intelligence  
emotions TEXT,  -- JSON: {"joy": 0.8, "curiosity": 0.6}
valence REAL DEFAULT 0.0,
arousal REAL DEFAULT 0.0,
dominance REAL DEFAULT 0.0,
outcome_type TEXT DEFAULT 'neutral',

-- Conflict resolution
conflicted BOOLEAN DEFAULT 0,
conflicts_with TEXT,
verification_requested BOOLEAN DEFAULT 0
```

**Context retrieval** uses vector similarity, metadata filtering, and emotional weighting:
```python
# Pattern: ContextManager retrieves relevant memories with emotional context
relevant_memories = await self.memory_manager.search_memories(
    query=query_text,
    memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
    limit=max_memories,
    include_emotions=True,  # Include emotional context
    activation_threshold=0.1  # Only active/resting memories
)

# Pattern: Somatic marker system for decision support
decision_guidance = await self.memory_manager.get_decision_recommendations(
    situation=current_context,
    emotion_patterns=["curiosity", "regret"],
    confidence_threshold=0.7
)
```

## Project Conventions

### Async Patterns
- **All core operations are async** - use `asyncio.run()` in CLI commands
- **Context managers** for resource lifecycle (storage, embeddings, MCP clients)
- **Background services** run in separate tasks with graceful shutdown

### Configuration Management
- **Config directory**: `~/.evo/` (created by `evo init`)
- **Environment override**: `.env` file support via python-dotenv
- **Pydantic models** in `*/models.py` for type safety

### Error Handling
```python
# Pattern: Log and provide user-friendly fallbacks with emotional context
try:
    response = await self.model_client.chat(...)
    # Store successful interaction with positive emotions
    await self.memory_manager.store_episodic_memory(
        content=f"Successful chat: {response}",
        emotions={"satisfaction": 0.6},
        outcome_type="SUCCESS"
    )
except Exception as e:
    self.logger.error(f"Chat error: {e}")
    # Store failure with negative emotions for learning
    await self.memory_manager.store_episodic_memory(
        content=f"Chat failed: {str(e)}",
        emotions={"regret": 0.8},
        outcome_type="FAILURE"
    )
    return "I encountered an error. Let me try to help in a different way."
```

### Code Quality Standards
- **Type hints required** - mypy configuration in pyproject.toml
- **Black formatting** with 88-character lines
- **Import organization** with isort (black-compatible profile)
- **Rich console output** for CLI - use `Console()` and panels/tables

### Testing Patterns
- **Async test support** via `pytest-asyncio` with `asyncio_mode = auto`
- **Test markers** for categorization (`@pytest.mark.slow`, `@pytest.mark.integration`)
- **Mock external dependencies** (Ollama, LM Studio, file system)

## Key Files for Understanding

- `src/evo/agent/core.py` - Main conversation logic and component orchestration
- `src/evo/memory/manager.py` - Memory storage/retrieval coordination  
- `src/evo/agent/context.py` - Memory-context integration for LLM calls
- `src/evo/services/consolidation.py` - Background memory processing
- `src/evo/cli/commands.py` - User-facing command implementations
- `tests/test_integration.py` - End-to-end system behavior examples