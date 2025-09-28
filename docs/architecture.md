# Evo AI Architecture

This document provides a comprehensive overview of Evo AI's system architecture, component relationships, and data flow.

## 🏗️ System Overview

Evo AI is built on a modular architecture that enables persistent learning, memory management, and intelligent conversation handling. The system consists of six main layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
├─────────────────────────────────────────────────────────────┤
│                     Agent Layer                            │
├─────────────────────────────────────────────────────────────┤
│                   Integration Layer                        │
├─────────────────────────────────────────────────────────────┤
│                   Processing Layer                         │
├─────────────────────────────────────────────────────────────┤
│                    Storage Layer                           │
├─────────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                     │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Core Components

### 1. User Interface Layer

#### CLI Interface (`src/evo/cli/`)
- **Rich Terminal Interface**: Interactive chat with markdown rendering
- **Command System**: Memory management, teaching, and system control
- **Session Management**: Multi-session support with persistence
- **Real-time Feedback**: Streaming responses and status updates

```python
# Key Components
├── commands.py     # CLI command definitions
├── chat.py         # Interactive chat interface
└── main.py         # Entry point and argument parsing
```

#### Features
- Interactive chat with syntax highlighting
- Command shortcuts (`/help`, `/status`, `/memory`)
- Session persistence and restoration
- Background process monitoring

### 2. Agent Layer (`src/evo/agent/`)

#### Core Agent (`agent/core.py`)
- **Conversation Management**: Handles user interactions
- **Memory Integration**: Seamless access to persistent knowledge
- **Learning Coordination**: Triggers and manages learning processes
- **Tool Integration**: Connects to external tools via MCP

#### Context Management (`agent/context.py`)
- **Memory Retrieval**: Intelligent context building from past conversations
- **Relevance Scoring**: Prioritizes important memories for inclusion
- **Context Trimming**: Manages conversation length while preserving key information
- **Session Tracking**: Maintains conversation state across interactions

#### Session Management (`agent/session.py`)
- **Persistence**: Saves conversations to disk
- **Search**: Find past conversations by content
- **Statistics**: Track conversation metrics and patterns
- **Cleanup**: Automatic old session management

```python
# Component Relationships
EvoAgent
├── ContextManager → MemoryManager
├── SessionManager → FileSystem
└── MCPClient → ExternalTools
```

### 3. Integration Layer

#### MCP Integration (`src/evo/mcp/`)
- **Protocol Implementation**: Full MCP client/server support
- **Tool Registry**: Extensible tool system
- **External LLM Fallback**: Access to cloud LLMs when needed
- **Service Discovery**: Automatic tool and service detection

#### Tool Categories
```python
├── Web Search        # Internet information retrieval
├── Code Analysis     # Programming assistance
├── File Operations   # Read/write local files
├── System Commands   # Safe shell command execution
└── Custom Tools      # Extensible tool framework
```

### 4. Processing Layer

#### Neuromorphic Memory System (`src/evo/memory/`)

Evo AI features a revolutionary **human-like memory system** with emotional intelligence and temporal dynamics.

##### Core Memory Components (`memory/models.py`)
- **Memory Types**: Episodic (conversations), Semantic (facts), Procedural (skills)
- **Temporal Dynamics**: Activation levels that decay over time
- **Dormancy States**: ACTIVE → RESTING → DORMANT → DEEP_SLEEP
- **Emotional Intelligence**: 12+ emotion types with valence-arousal-dominance model
- **Conflict Resolution**: Duplicate detection and contradiction handling

##### Storage Engine (`memory/storage.py`)
- **Enhanced SQLite Backend**: 15+ new fields for neuromorphic features
- **Vector Extensions**: Semantic similarity search with emotional weighting
- **Migration System**: Automatic schema updates for existing databases
- **ACID Compliance**: Reliable data consistency for complex emotional data

##### Emotion Detection (`memory/emotion.py`)
- **Pattern-Based Detection**: Joy, fear, anger, curiosity, regret, etc.
- **Conversation Analysis**: Automatic emotional tagging of interactions
- **Valence Calculation**: Positive/negative emotional tone tracking
- **Outcome Classification**: Success, failure, learning, confusion patterns
- **Memory Enhancement**: Emotional memories get importance boosts

##### Somatic Marker System (`memory/somatic.py`)
- **Decision Support**: Uses emotional memories to guide future choices
- **Risk Assessment**: Evaluates decisions based on past emotional outcomes
- **Confidence Scoring**: Provides reliability metrics for recommendations
- **Learning Integration**: Improves recommendations from decision outcomes

##### Conflict Resolution (`memory/conflict.py`)
- **Duplicate Detection**: Prevents redundant information storage
- **Memory Reinforcement**: Strengthens similar memories instead of duplicating
- **Contradiction Handling**: Detects and resolves conflicting information
- **Verification Requests**: Flags uncertain information for human review

##### Memory Manager (`memory/manager.py`)
- **Enhanced Search**: Emotion-aware memory retrieval
- **Temporal Dynamics**: Handles activation levels and dormancy transitions
- **Emotional Insights**: Analyzes emotional patterns over time
- **Decision Recommendations**: Provides somatic marker guidance
- **Consolidation**: Transforms episodic memories into semantic knowledge

#### Learning System (`src/evo/learning/`)

##### QLoRA Training (`learning/qlora.py`)
- **4-bit Quantization**: Memory-efficient fine-tuning
- **LoRA Adapters**: Parameter-efficient learning
- **Training Pipeline**: Automated model updates
- **Performance Monitoring**: Training metrics and validation

##### Data Processing (`learning/data.py`)
- **Conversation Analysis**: Extract training data from memories
- **Quality Filtering**: Ensure high-quality training samples
- **Data Augmentation**: Expand training datasets
- **Format Conversion**: Transform data for model training

##### Learning Scheduler (`learning/scheduler.py`)
- **Automated Training**: Time-based and data-driven triggers
- **Job Queue**: Prioritized learning task management
- **Resource Management**: GPU utilization optimization
- **Progress Tracking**: Detailed learning metrics

#### Summarization System (`src/evo/summarization/`)

##### Context Summarizer (`summarization/summarizer.py`)
- **Conversation Summaries**: Compress long interactions
- **Fact Extraction**: Identify key information
- **Topic Modeling**: Classify conversation themes
- **Learning Outcomes**: Detect knowledge acquisition

##### Summary Extractor (`summarization/summarizer.py`)
- **Pattern Recognition**: Rule-based information extraction
- **Named Entity Recognition**: Identify people, places, concepts
- **Sentiment Analysis**: Understand emotional context
- **Importance Scoring**: Rank information by relevance

### 5. Storage Layer

#### Database Schema
```sql
-- Core memory table
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,           -- episodic, semantic, procedural
    content TEXT NOT NULL,
    embedding BLOB,               -- Vector representation
    metadata TEXT,                -- JSON metadata
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    importance REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,

    -- Type-specific fields
    session_id TEXT,              -- For episodic memories
    user_message TEXT,
    assistant_response TEXT,
    subject TEXT,                 -- For semantic memories
    predicate TEXT,
    object TEXT,
    confidence REAL,
    skill_name TEXT,              -- For procedural memories
    steps TEXT,
    success_rate REAL
);

-- Full-text search index
CREATE VIRTUAL TABLE memories_fts USING fts5(
    content, user_message, assistant_response
);

-- Vector similarity (if sqlite-vec available)
CREATE VIRTUAL TABLE memories_vec USING vec0(
    embedding float[384]
);
```

#### File System Organization
```
~/.evo/
├── config.json          # User configuration
├── memory.db            # SQLite database
├── sessions/            # Saved conversations
├── adapters/            # LoRA adapters
├── logs/               # System logs
└── cache/              # Temporary files
```

### 6. Infrastructure Layer

#### Background Services (`src/evo/services/`)

##### Background Service Coordinator (`services/background.py`)
- **Service Orchestration**: Manages all background processes
- **Health Monitoring**: Tracks service status and performance
- **Resource Coordination**: Prevents resource conflicts
- **Graceful Shutdown**: Clean service termination

##### Memory Consolidation Service (`services/consolidation.py`)
- **Automated Consolidation**: Periodic memory organization
- **Fact Extraction**: Convert conversations to structured knowledge
- **Summary Generation**: Create conversation summaries
- **Knowledge Clustering**: Group related information

##### System Monitor (`services/monitoring.py`)
- **Resource Monitoring**: CPU, memory, disk usage
- **Performance Metrics**: Response times, throughput
- **Alert System**: Threshold-based notifications
- **Health Checks**: Service availability monitoring

## 🔄 Data Flow

### 1. User Interaction Flow
```
User Input → CLI Interface → Agent Core → Context Manager
     ↓
Memory Retrieval ← Memory Manager ← Embedding System
     ↓
LLM Processing → Ollama/External LLM → Response Generation
     ↓
Memory Storage → Memory Manager → SQLite Database
```

### 2. Learning Flow
```
Conversations → Data Processor → Quality Filter → Training Dataset
     ↓
QLoRA Trainer → Model Fine-tuning → Adapter Creation
     ↓
Model Update → Agent Core → Improved Responses
```

### 3. Consolidation Flow
```
Episodic Memories → Fact Extractor → Pattern Recognition
     ↓
Semantic Memories ← Knowledge Organizer ← Summary Generator
     ↓
Knowledge Graph → Memory Manager → Long-term Storage
```

## 🔧 Component Interactions

### Memory System Integration
```python
class EvoAgent:
    def __init__(self):
        self.memory_manager = MemoryManager(db_path)
        self.context_manager = ContextManager(self.memory_manager)

    async def chat(self, message: str, session_id: str):
        # Retrieve relevant memories
        context = await self.context_manager.get_context(session_id)

        # Add user message
        user_msg = ChatMessage(role="user", content=message)
        await self.context_manager.add_message(session_id, user_msg)

        # Generate response with memory context
        response = await self._generate_response(context)

        # Store interaction as episodic memory
        await self.context_manager.store_conversation_memory(
            context, message, response
        )
```

### Learning System Integration
```python
class LearningScheduler:
    async def _scheduler_loop(self):
        # Check for consolidation triggers
        await self._check_automatic_training()

        # Process learning queue
        await self._process_job_queue()

        # Coordinate with memory consolidation
        if learning_completed:
            await self.consolidation_service.schedule_consolidation()
```

### MCP Integration
```python
class EvoAgent:
    async def _check_and_use_tools(self, message: str):
        # Determine if external tools are needed
        if self._needs_web_search(message):
            result = await self.mcp_client.call_tool("web_search", {
                "query": extract_search_query(message)
            })

        # Fallback to external LLM if needed
        if self._needs_external_llm(message):
            result = await self.mcp_client.call_external_llm(message)
```

## 🚀 Performance Characteristics

### Memory Performance
- **Vector Search**: <50ms for 10K memories
- **Text Search**: <20ms for full-text queries
- **Memory Storage**: <10ms per memory
- **Context Retrieval**: <100ms including embeddings

### Learning Performance
- **QLoRA Training**: 2-4 hours for 1K samples (RTX 4090)
- **Data Processing**: 1K conversations/minute
- **Consolidation**: 5-15 minutes per session

### Scalability Limits
- **Memory Capacity**: 1M+ memories per database
- **Concurrent Users**: 10+ simultaneous sessions
- **Training Data**: 100K+ conversation samples
- **Model Size**: Up to 13B parameters with QLoRA

## 🔒 Security Architecture

### Data Protection
- **Local Storage**: All data remains on user's machine
- **Encryption**: SQLite database encryption (optional)
- **Access Control**: File system permissions
- **Audit Trail**: Complete interaction logging

### Tool Security
- **Sandboxed Execution**: Limited command access
- **Path Restrictions**: Safe directory access only
- **Input Validation**: Sanitized tool inputs
- **Permission Model**: Explicit tool authorization

## 🎯 Design Principles

### 1. **Modularity**
- Clear separation of concerns
- Pluggable components
- Independent testing
- Easy maintenance

### 2. **Extensibility**
- Plugin architecture for tools
- Configurable memory types
- Custom learning strategies
- Flexible integration points

### 3. **Performance**
- Efficient vector operations
- Cached computations
- Batch processing
- Resource optimization

### 4. **Reliability**
- Graceful error handling
- Automatic recovery
- Data consistency
- Health monitoring

### 5. **Privacy**
- Local-first architecture
- User-controlled data
- Optional cloud integration
- Transparent operations

This architecture enables Evo AI to provide sophisticated learning capabilities while maintaining performance, reliability, and user privacy.