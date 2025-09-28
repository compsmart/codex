"""
Memory models and data structures for Evo AI.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Types of memory in the Evo AI system."""

    EPISODIC = "episodic"      # Specific conversations and events
    SEMANTIC = "semantic"      # General facts and knowledge
    PROCEDURAL = "procedural"  # Skills and procedures


class DormancyState(str, Enum):
    """Memory activation states based on human memory patterns."""

    ACTIVE = "active"          # Recently accessed (< 7 days)
    RESTING = "resting"        # Moderately recent (7-30 days)
    DORMANT = "dormant"        # Old but reactivatable (30-90 days)
    DEEP_SLEEP = "deep_sleep"  # Very old, needs strong cues (> 90 days)


class OutcomeType(str, Enum):
    """Types of interaction outcomes for learning."""

    SUCCESS = "success"        # Goal achieved, positive result
    FAILURE = "failure"        # Goal not achieved, negative result
    LEARNING = "learning"      # New understanding gained
    CONFUSION = "confusion"    # Uncertainty, need clarification
    NEUTRAL = "neutral"        # No clear outcome
    REGRET = "regret"         # Negative reflection on past action


class Memory(BaseModel):
    """Base memory model with temporal dynamics and emotional intelligence."""

    id: UUID = Field(default_factory=uuid4)
    type: MemoryType
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = Field(default=0)
    last_accessed: Optional[datetime] = None

    # Temporal dynamics (neuromorphic memory)
    activation_level: float = Field(default=1.0, ge=0.0, le=1.0)
    dormancy_state: DormancyState = Field(default=DormancyState.ACTIVE)
    reactivation_count: int = Field(default=0)
    last_reactivation: Optional[datetime] = None
    decay_rate: float = Field(default=0.1, ge=0.0, le=1.0)

    # Emotional intelligence
    emotions: Dict[str, float] = Field(default_factory=dict)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)  # Negative to positive
    arousal: float = Field(default=0.0, ge=0.0, le=1.0)   # Intensity
    dominance: float = Field(default=0.0, ge=-1.0, le=1.0)  # Control/power
    outcome_type: OutcomeType = Field(default=OutcomeType.NEUTRAL)
    outcome_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    emotional_intensity: float = Field(default=0.0, ge=0.0, le=1.0)

    # Conflict resolution
    conflicted: bool = Field(default=False)
    conflicts_with: List[UUID] = Field(default_factory=list)
    superseded_by: Optional[UUID] = None
    verification_requested: bool = Field(default=False)

    def access(self) -> None:
        """Mark memory as accessed and handle reactivation."""
        self.access_count += 1
        self.last_accessed = datetime.now()

        # Boost activation when accessed
        self.activation_level = min(self.activation_level + 0.1, 1.0)

        # Handle dormant memory reactivation
        if self.dormancy_state in [DormancyState.DORMANT, DormancyState.DEEP_SLEEP]:
            self.dormancy_state = DormancyState.ACTIVE
            self.reactivation_count += 1
            self.last_reactivation = datetime.now()

    def update_content(self, content: str) -> None:
        """Update memory content and mark as recently modified."""
        self.content = content
        self.updated_at = datetime.now()
        self.activation_level = 1.0  # Fully activate on update

    def calculate_current_activation(self) -> float:
        """Calculate current activation level considering time decay."""
        if not self.last_accessed:
            time_since_access = (datetime.now() - self.created_at).days
        else:
            time_since_access = (datetime.now() - self.last_accessed).days

        # Time decay curve (exponential)
        if time_since_access < 1:
            time_factor = 1.5  # Recent boost
        elif time_since_access < 7:
            time_factor = 1.0  # Normal
        elif time_since_access < 30:
            time_factor = 0.8  # Resting
        elif time_since_access < 90:
            time_factor = 0.5  # Dormant
        else:
            time_factor = 0.3  # Deep sleep

        # Emotional intensity factor (emotional memories are stronger)
        emotion_factor = 1.0 + (self.emotional_intensity * 0.5)

        # Access frequency factor (frequently accessed memories are stronger)
        frequency_factor = 1.0 + min(self.access_count * 0.1, 0.5)

        return min(
            self.activation_level * time_factor * emotion_factor * frequency_factor,
            1.0
        )

    def update_dormancy_state(self) -> None:
        """Update dormancy state based on time since last access."""
        if not self.last_accessed:
            days_since_access = (datetime.now() - self.created_at).days
        else:
            days_since_access = (datetime.now() - self.last_accessed).days

        if days_since_access < 7:
            self.dormancy_state = DormancyState.ACTIVE
        elif days_since_access < 30:
            self.dormancy_state = DormancyState.RESTING
        elif days_since_access < 90:
            self.dormancy_state = DormancyState.DORMANT
        else:
            self.dormancy_state = DormancyState.DEEP_SLEEP

    def add_emotion(self, emotion: str, intensity: float) -> None:
        """Add or update an emotion for this memory."""
        self.emotions[emotion] = max(0.0, min(1.0, intensity))
        self._update_emotional_derived_fields()

    def _update_emotional_derived_fields(self) -> None:
        """Update valence, arousal, and emotional intensity from discrete emotions."""
        if not self.emotions:
            return

        # Calculate valence (positive vs negative)
        positive_emotions = ["joy", "satisfaction", "pride", "trust", "curiosity"]
        negative_emotions = ["sadness", "fear", "anger", "disgust", "regret", "shame", "guilt"]

        positive_sum = sum(self.emotions.get(e, 0) for e in positive_emotions)
        negative_sum = sum(self.emotions.get(e, 0) for e in negative_emotions)

        # Valence is difference between positive and negative
        self.valence = max(-1.0, min(1.0, positive_sum - negative_sum))

        # Arousal is overall emotional intensity
        self.arousal = min(1.0, sum(abs(v) for v in self.emotions.values()) / len(self.emotions))

        # Emotional intensity affects memory strength
        self.emotional_intensity = self.arousal


class EpisodicMemory(Memory):
    """Memory of specific conversations and interactions."""

    type: MemoryType = Field(default=MemoryType.EPISODIC, frozen=True)
    session_id: Optional[str] = None
    user_message: Optional[str] = None
    assistant_response: Optional[str] = None
    context_summary: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "content": "User asked about Python list comprehensions",
                "session_id": "session_123",
                "user_message": "How do Python list comprehensions work?",
                "assistant_response": "List comprehensions provide a concise way...",
                "context_summary": "Discussion about Python programming",
                "emotions": {"curiosity": 0.8, "satisfaction": 0.7}
            }
        }


class SemanticMemory(Memory):
    """Memory of facts, concepts, and general knowledge."""

    type: MemoryType = Field(default=MemoryType.SEMANTIC, frozen=True)
    subject: str
    predicate: str
    object: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source: Optional[str] = None
    verified: bool = False
    related_concepts: List[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Python is a high-level programming language",
                "subject": "Python",
                "predicate": "is",
                "object": "high-level programming language",
                "confidence": 0.95,
                "source": "user_conversation",
                "verified": True,
                "related_concepts": ["programming", "interpreted", "dynamic typing"]
            }
        }


class ProceduralMemory(Memory):
    """Memory of procedures, skills, and how-to knowledge."""

    type: MemoryType = Field(default=MemoryType.PROCEDURAL, frozen=True)
    skill_name: str
    steps: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    last_used: Optional[datetime] = None
    proficiency_level: str = "beginner"  # beginner, intermediate, advanced, expert

    class Config:
        json_schema_extra = {
            "example": {
                "content": "How to debug Python code",
                "skill_name": "python_debugging",
                "steps": [
                    "Read the error message carefully",
                    "Check the stack trace",
                    "Use print statements or debugger",
                    "Test with minimal example"
                ],
                "prerequisites": ["basic_python_knowledge"],
                "success_rate": 0.85,
                "proficiency_level": "intermediate"
            }
        }


class MemoryCluster(BaseModel):
    """A cluster of related memories."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    memory_ids: List[UUID] = Field(default_factory=list)
    centroid_embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryQuery(BaseModel):
    """Query parameters for memory search."""

    text: str
    memory_types: Optional[List[MemoryType]] = None
    limit: int = Field(default=10, ge=1, le=100)
    min_similarity: float = Field(default=0.3, ge=0.0, le=1.0)
    max_age_days: Optional[int] = None
    min_importance: Optional[float] = None
    session_id: Optional[str] = None
    include_metadata: bool = True


class MemorySearchResult(BaseModel):
    """Result from memory search."""

    memory: Memory
    similarity_score: float
    rank: int
    explanation: Optional[str] = None