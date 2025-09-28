"""
Memory management system for Evo AI.
Provides persistent storage and retrieval of knowledge across sessions.
"""

from .manager import MemoryManager
from .models import Memory, MemoryType, MemoryQuery, EpisodicMemory, SemanticMemory, ProceduralMemory, DormancyState, OutcomeType
from .storage import SQLiteStorage
from .embeddings import EmbeddingManager
from .emotion import EmotionDetector, EmotionalContext
from .somatic import SomaticMarkerSystem, SomaticMarker, DecisionContext, DecisionRecommendation
from .conflict import ConflictResolver, ConflictDetection, DuplicateDetection

__all__ = [
    "MemoryManager",
    "Memory",
    "MemoryType",
    "MemoryQuery",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "DormancyState",
    "OutcomeType",
    "SQLiteStorage",
    "EmbeddingManager",
    "EmotionDetector",
    "EmotionalContext",
    "SomaticMarkerSystem",
    "SomaticMarker",
    "DecisionContext",
    "DecisionRecommendation",
    "ConflictResolver",
    "ConflictDetection",
    "DuplicateDetection",
]