"""
Models for context summarization system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SummaryType(str, Enum):
    """Types of summaries."""

    CONVERSATION = "conversation"
    FACT_EXTRACTION = "fact_extraction"
    TOPIC_SUMMARY = "topic_summary"
    LEARNING_OUTCOME = "learning_outcome"


class SummaryConfig(BaseModel):
    """Configuration for summarization."""

    # Model settings
    summarization_model: str = "facebook/bart-large-cnn"
    max_input_length: int = 1024
    max_output_length: int = 256
    min_output_length: int = 50

    # Fact extraction
    enable_fact_extraction: bool = True
    fact_confidence_threshold: float = 0.7
    max_facts_per_conversation: int = 10

    # Topic modeling
    enable_topic_extraction: bool = True
    max_topics_per_conversation: int = 5
    topic_coherence_threshold: float = 0.5

    # Importance filtering
    min_conversation_length: int = 100  # characters
    importance_boost_keywords: List[str] = Field(default_factory=lambda: [
        "remember", "important", "prefer", "favorite", "always", "never",
        "my name", "i am", "i like", "i don't like"
    ])


class ConversationSummary(BaseModel):
    """Summary of a conversation."""

    id: UUID = Field(default_factory=uuid4)
    session_id: str
    summary_type: SummaryType = SummaryType.CONVERSATION

    # Content
    original_content: str
    summary_text: str
    key_points: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    confidence_score: float = 0.0
    importance_score: float = 0.0
    word_count_original: int = 0
    word_count_summary: int = 0
    compression_ratio: float = 0.0

    # Source information
    source_memory_ids: List[UUID] = Field(default_factory=list)
    message_count: int = 0
    participant_count: int = 2  # user + assistant

    def calculate_compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.word_count_original > 0:
            self.compression_ratio = self.word_count_summary / self.word_count_original
        return self.compression_ratio


class FactExtraction(BaseModel):
    """Extracted fact from conversation."""

    id: UUID = Field(default_factory=uuid4)
    fact_text: str
    subject: str
    predicate: str
    object: str

    # Confidence and validation
    confidence: float = 0.0
    verified: bool = False
    source_context: str = ""

    # Classification
    fact_type: str = "general"  # personal, preference, knowledge, etc.
    importance: float = 0.5

    # Source tracking
    conversation_id: Optional[UUID] = None
    source_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "fact_text": "User's favorite programming language is Python",
                "subject": "user's favorite programming language",
                "predicate": "is",
                "object": "Python",
                "confidence": 0.9,
                "fact_type": "preference",
                "importance": 0.8
            }
        }


class TopicSummary(BaseModel):
    """Summary of topics in conversation."""

    id: UUID = Field(default_factory=uuid4)
    topic_name: str
    description: str
    keywords: List[str] = Field(default_factory=list)

    # Metrics
    relevance_score: float = 0.0
    frequency: int = 0
    message_spans: List[int] = Field(default_factory=list)  # Message indices where topic appears

    # Context
    conversation_id: Optional[UUID] = None
    created_at: datetime = Field(default_factory=datetime.now)


class LearningOutcome(BaseModel):
    """Learning outcome from conversation."""

    id: UUID = Field(default_factory=uuid4)
    outcome_text: str
    outcome_type: str  # skill_acquired, knowledge_gained, preference_learned, etc.

    # Validation
    confidence: float = 0.0
    evidence: List[str] = Field(default_factory=list)

    # Categorization
    domain: str = "general"  # programming, personal, academic, etc.
    complexity_level: str = "basic"  # basic, intermediate, advanced

    # Application
    applicable_contexts: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)

    # Source
    conversation_id: Optional[UUID] = None
    created_at: datetime = Field(default_factory=datetime.now)


class SummarizationJob(BaseModel):
    """Job for summarization processing."""

    id: UUID = Field(default_factory=uuid4)
    job_type: SummaryType
    input_data: Dict[str, Any] = Field(default_factory=dict)

    # Status
    status: str = "pending"  # pending, processing, completed, failed
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    results: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None

    # Configuration
    config: Optional[SummaryConfig] = None

    def start_processing(self) -> None:
        """Mark job as started."""
        self.status = "processing"
        self.started_at = datetime.now()

    def complete(self, success: bool = True, results: Optional[Dict[str, Any]] = None) -> None:
        """Mark job as completed."""
        self.status = "completed" if success else "failed"
        self.completed_at = datetime.now()
        if results:
            self.results = results

    def fail(self, error: str) -> None:
        """Mark job as failed."""
        self.status = "failed"
        self.completed_at = datetime.now()
        self.error_message = error