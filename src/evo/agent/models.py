"""
Models and data structures for the Evo AI agent.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ChatRole(str, Enum):
    """Roles in a chat conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """A message in a chat conversation."""

    id: UUID = Field(default_factory=uuid4)
    role: ChatRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What is machine learning?",
                "metadata": {"session_id": "session_123"},
            }
        }


class AgentConfig(BaseModel):
    """Configuration for the Evo AI agent."""

    # Model settings
    model_name: str = "llama3.2:7b"
    model_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9
    top_k: int = 50

    # Memory settings
    use_memory: bool = True
    memory_limit: int = 20  # Number of recent messages to keep in context
    episodic_memory_threshold: float = 0.3  # Minimum importance to store episodic memory
    semantic_memory_threshold: float = 0.5  # Minimum importance to store semantic memory

    # Learning settings
    enable_learning: bool = True
    learning_rate: float = 0.0001
    consolidation_interval: int = 3600  # Seconds between memory consolidations

    # MCP settings
    enable_mcp: bool = True
    mcp_timeout: float = 30.0
    fallback_to_external_llm: bool = True

    # System behavior
    system_prompt: str = Field(
        default="You are Evo, an AI assistant that learns and remembers. "
        "You can access your memory to recall past conversations and learned facts. "
        "When you don't know something, you can search for information using available tools."
    )
    max_context_length: int = 8192
    context_window_overlap: int = 512

    # Personality settings
    personality_traits: Dict[str, float] = Field(
        default_factory=lambda: {
            "curiosity": 0.8,
            "helpfulness": 0.9,
            "creativity": 0.7,
            "analytical": 0.8,
            "empathy": 0.6,
        }
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "llama3.2:7b",
                "temperature": 0.7,
                "use_memory": True,
                "enable_learning": True,
                "enable_mcp": True,
                "system_prompt": "You are Evo, an AI assistant...",
            }
        }


class ToolCall(BaseModel):
    """Represents a tool call made by the agent."""

    id: str
    type: str = "function"
    function: Dict[str, Any]


class ToolResult(BaseModel):
    """Result from a tool call."""

    tool_call_id: str
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0


class ConversationContext(BaseModel):
    """Context for a conversation including memory and state."""

    session_id: str
    messages: List[ChatMessage] = Field(default_factory=list)
    retrieved_memories: List[Dict[str, Any]] = Field(default_factory=list)
    active_tools: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_recent_messages(self, limit: int = 10) -> List[ChatMessage]:
        """Get the most recent messages."""
        return self.messages[-limit:] if len(self.messages) > limit else self.messages

    def get_context_length(self) -> int:
        """Calculate approximate context length in tokens."""
        # Rough approximation: 4 characters per token
        total_chars = sum(len(msg.content) for msg in self.messages)
        return total_chars // 4


class AgentState(BaseModel):
    """Current state of the agent."""

    is_learning: bool = False
    last_consolidation: Optional[datetime] = None
    total_conversations: int = 0
    total_memories_stored: int = 0
    model_updates: int = 0
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    active_sessions: List[str] = Field(default_factory=list)

    def update_performance_metric(self, metric: str, value: float) -> None:
        """Update a performance metric."""
        self.performance_metrics[metric] = value

    def add_session(self, session_id: str) -> None:
        """Add an active session."""
        if session_id not in self.active_sessions:
            self.active_sessions.append(session_id)

    def remove_session(self, session_id: str) -> None:
        """Remove an active session."""
        if session_id in self.active_sessions:
            self.active_sessions.remove(session_id)