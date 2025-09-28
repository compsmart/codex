"""
Conflict detection and resolution for Evo AI memory system.

This module handles contradictory information, prevents duplicates,
and resolves conflicts between memories.
"""

from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import re
from difflib import SequenceMatcher

from .models import Memory, MemoryType, EpisodicMemory, SemanticMemory


@dataclass
class ConflictDetection:
    """Results of conflict detection between memories."""

    conflicted_memory: Memory
    conflicting_memories: List[Memory]
    conflict_type: str  # "contradiction", "duplicate", "inconsistency"
    confidence: float  # How confident we are about the conflict
    similarity_score: float
    resolution_strategy: str


@dataclass
class DuplicateDetection:
    """Results of duplicate detection."""

    original_memory: Memory
    duplicate_candidates: List[Memory]
    similarity_scores: List[float]
    content_overlap: float
    time_gap: float  # Days between memories


class ConflictResolver:
    """Handles memory conflicts and duplicates."""

    def __init__(self):
        self.duplicate_threshold = 0.85  # Content similarity threshold for duplicates
        self.conflict_threshold = 0.70  # Similarity threshold for conflicts
        self.contradiction_patterns = self._build_contradiction_patterns()

    def _build_contradiction_patterns(self) -> List[Tuple[str, str]]:
        """Build patterns that indicate contradictions."""
        return [
            # Direct negations
            (r"\bis\b", r"\bis not\b"),
            (r"\bcan\b", r"\bcannot\b"),
            (r"\bwill\b", r"\bwill not\b"),
            (r"\bdoes\b", r"\bdoes not\b"),
            (r"\bhas\b", r"\bhas not\b"),
            (r"\bare\b", r"\bare not\b"),
            (r"\bwas\b", r"\bwas not\b"),
            (r"\bwere\b", r"\bwere not\b"),

            # Opposing concepts
            (r"\blikes?\b", r"\bdislikes?\b"),
            (r"\blikes?\b", r"\bhates?\b"),
            (r"\bgood\b", r"\bbad\b"),
            (r"\btrue\b", r"\bfalse\b"),
            (r"\byes\b", r"\bno\b"),
            (r"\bpossible\b", r"\bimpossible\b"),
            (r"\bsafe\b", r"\bunsafe\b"),
            (r"\bsafe\b", r"\bdangerous\b"),

            # Temporal contradictions
            (r"\bbefore\b", r"\bafter\b"),
            (r"\bearlier\b", r"\blater\b"),
            (r"\bfirst\b", r"\blast\b"),
            (r"\bold\b", r"\bnew\b"),
            (r"\byoung\b", r"\bold\b"),

            # Quantitative contradictions
            (r"\ball\b", r"\bnone\b"),
            (r"\bmany\b", r"\bfew\b"),
            (r"\blarge\b", r"\bsmall\b"),
            (r"\bbig\b", r"\bsmall\b"),
            (r"\bhigh\b", r"\blow\b"),
            (r"\bfast\b", r"\bslow\b"),
        ]

    async def detect_duplicates(
        self,
        new_memory: Memory,
        existing_memories: List[Memory],
        threshold: Optional[float] = None
    ) -> Optional[DuplicateDetection]:
        """Detect if a new memory is a duplicate of existing ones."""

        if threshold is None:
            threshold = self.duplicate_threshold

        # Only check against memories of the same type
        same_type_memories = [m for m in existing_memories if m.type == new_memory.type]

        candidates = []
        similarities = []

        for existing in same_type_memories:
            similarity = self._calculate_content_similarity(new_memory, existing)

            if similarity >= threshold:
                candidates.append(existing)
                similarities.append(similarity)

        if not candidates:
            return None

        # Find the most similar memory as the "original"
        max_similarity_idx = similarities.index(max(similarities))
        original = candidates[max_similarity_idx]

        # Calculate time gap
        time_gap = abs((new_memory.created_at - original.created_at).total_seconds() / 86400)

        return DuplicateDetection(
            original_memory=original,
            duplicate_candidates=candidates,
            similarity_scores=similarities,
            content_overlap=max(similarities),
            time_gap=time_gap
        )

    async def detect_conflicts(
        self,
        new_memory: Memory,
        existing_memories: List[Memory]
    ) -> List[ConflictDetection]:
        """Detect conflicts between a new memory and existing ones."""

        conflicts = []

        # Only check against semantic memories for factual contradictions
        # and episodic memories for consistency
        if new_memory.type == MemoryType.SEMANTIC:
            candidates = [m for m in existing_memories if m.type == MemoryType.SEMANTIC]
        elif new_memory.type == MemoryType.EPISODIC:
            candidates = [m for m in existing_memories
                         if m.type == MemoryType.EPISODIC and
                         isinstance(m, EpisodicMemory) and isinstance(new_memory, EpisodicMemory) and
                         m.session_id == new_memory.session_id]
        else:
            candidates = [m for m in existing_memories if m.type == new_memory.type]

        for existing in candidates:
            conflict = await self._analyze_conflict(new_memory, existing)
            if conflict:
                conflicts.append(conflict)

        return conflicts

    async def _analyze_conflict(
        self,
        memory1: Memory,
        memory2: Memory
    ) -> Optional[ConflictDetection]:
        """Analyze two memories for conflicts."""

        # Calculate content similarity
        similarity = self._calculate_content_similarity(memory1, memory2)

        # If content is too different, unlikely to be conflicting
        if similarity < self.conflict_threshold:
            return None

        # Check for direct contradictions using patterns
        contradiction_confidence = self._detect_contradiction_patterns(
            memory1.content, memory2.content
        )

        # For semantic memories, check subject-predicate-object conflicts
        if (isinstance(memory1, SemanticMemory) and
            isinstance(memory2, SemanticMemory)):
            semantic_conflict = self._check_semantic_conflict(memory1, memory2)
            if semantic_conflict:
                contradiction_confidence = max(contradiction_confidence, semantic_conflict)

        if contradiction_confidence > 0.3:
            # Determine resolution strategy
            strategy = self._determine_resolution_strategy(memory1, memory2)

            return ConflictDetection(
                conflicted_memory=memory1,
                conflicting_memories=[memory2],
                conflict_type="contradiction" if contradiction_confidence > 0.6 else "inconsistency",
                confidence=contradiction_confidence,
                similarity_score=similarity,
                resolution_strategy=strategy
            )

        return None

    def _calculate_content_similarity(self, memory1: Memory, memory2: Memory) -> float:
        """Calculate content similarity between two memories."""
        content1 = memory1.content.lower().strip()
        content2 = memory2.content.lower().strip()

        # Use SequenceMatcher for basic text similarity
        return SequenceMatcher(None, content1, content2).ratio()

    def _detect_contradiction_patterns(self, content1: str, content2: str) -> float:
        """Detect contradiction patterns between two content strings."""
        content1_lower = content1.lower()
        content2_lower = content2.lower()

        contradiction_score = 0.0
        pattern_count = 0

        for positive_pattern, negative_pattern in self.contradiction_patterns:
            # Check if one content has positive and other has negative pattern
            pos1_match = bool(re.search(positive_pattern, content1_lower))
            neg1_match = bool(re.search(negative_pattern, content1_lower))
            pos2_match = bool(re.search(positive_pattern, content2_lower))
            neg2_match = bool(re.search(negative_pattern, content2_lower))

            # Look for contradictions
            if (pos1_match and neg2_match) or (neg1_match and pos2_match):
                contradiction_score += 0.2
                pattern_count += 1
            elif (pos1_match and pos2_match) or (neg1_match and neg2_match):
                # Supporting patterns (reduce contradiction score)
                contradiction_score -= 0.1
                pattern_count += 1

        # Normalize by number of applicable patterns
        if pattern_count > 0:
            return max(0.0, min(1.0, contradiction_score / pattern_count))

        return 0.0

    def _check_semantic_conflict(
        self,
        semantic1: SemanticMemory,
        semantic2: SemanticMemory
    ) -> float:
        """Check for conflicts between semantic memories."""

        # Same subject and predicate but different object = potential conflict
        if (semantic1.subject.lower() == semantic2.subject.lower() and
            semantic1.predicate.lower() == semantic2.predicate.lower()):

            # Different objects suggest contradiction
            if semantic1.object.lower() != semantic2.object.lower():
                # Check if objects are contradictory
                obj_contradiction = self._detect_contradiction_patterns(
                    semantic1.object, semantic2.object
                )
                return max(0.5, obj_contradiction)  # Minimum 0.5 for different objects

        return 0.0

    def _determine_resolution_strategy(self, memory1: Memory, memory2: Memory) -> str:
        """Determine the best strategy to resolve a conflict."""

        # Trust more recent memories
        if memory1.created_at > memory2.created_at:
            return "trust_newer"

        # Trust higher importance memories
        if memory1.importance > memory2.importance + 0.1:
            return "trust_higher_importance"
        elif memory2.importance > memory1.importance + 0.1:
            return "trust_existing_higher_importance"

        # Trust verified semantic memories
        if (isinstance(memory1, SemanticMemory) and
            isinstance(memory2, SemanticMemory)):
            if memory1.verified and not memory2.verified:
                return "trust_verified"
            elif memory2.verified and not memory1.verified:
                return "trust_existing_verified"

        # Trust higher confidence
        if (hasattr(memory1, 'confidence') and hasattr(memory2, 'confidence')):
            if memory1.confidence > memory2.confidence + 0.1:
                return "trust_higher_confidence"
            elif memory2.confidence > memory1.confidence + 0.1:
                return "trust_existing_confidence"

        # Default: request verification
        return "request_verification"

    async def resolve_duplicate(
        self,
        duplicate_detection: DuplicateDetection,
        new_memory: Memory
    ) -> Tuple[Memory, str]:
        """Resolve a duplicate by reinforcing the original memory."""

        original = duplicate_detection.original_memory

        # Increase access count and importance for reinforcement
        original.access_count += 1
        original.last_accessed = datetime.now()

        # Boost activation level
        original.activation_level = min(original.activation_level + 0.1, 1.0)

        # Average the importance scores
        original.importance = (original.importance + new_memory.importance) / 2

        # If new memory has emotions, merge them
        if new_memory.emotions:
            for emotion, intensity in new_memory.emotions.items():
                if emotion in original.emotions:
                    # Take the stronger emotion
                    original.emotions[emotion] = max(original.emotions[emotion], intensity)
                else:
                    original.emotions[emotion] = intensity

        # Update emotional derived fields
        original._update_emotional_derived_fields()

        # Add metadata about reinforcement
        if 'reinforcement_count' not in original.metadata:
            original.metadata['reinforcement_count'] = 0
        original.metadata['reinforcement_count'] += 1
        original.metadata['last_reinforced'] = datetime.now().isoformat()

        return original, "reinforced"

    async def resolve_conflict(
        self,
        conflict: ConflictDetection,
        new_memory: Memory
    ) -> Tuple[Memory, Memory, str]:
        """Resolve a conflict between memories."""

        existing_memory = conflict.conflicting_memories[0]
        strategy = conflict.resolution_strategy

        if strategy == "trust_newer":
            # Mark existing as superseded
            existing_memory.superseded_by = new_memory.id
            existing_memory.conflicted = True
            return new_memory, existing_memory, "superseded_by_newer"

        elif strategy == "trust_higher_importance":
            # Mark existing as superseded by more important memory
            existing_memory.superseded_by = new_memory.id
            existing_memory.conflicted = True
            return new_memory, existing_memory, "superseded_by_importance"

        elif strategy in ["trust_existing_higher_importance", "trust_existing_verified", "trust_existing_confidence"]:
            # Mark new memory as conflicted and reference existing
            new_memory.conflicted = True
            new_memory.conflicts_with.append(existing_memory.id)
            return existing_memory, new_memory, "rejected_due_to_conflict"

        elif strategy == "trust_verified" or strategy == "trust_higher_confidence":
            # Mark existing as superseded
            existing_memory.superseded_by = new_memory.id
            existing_memory.conflicted = True
            return new_memory, existing_memory, "superseded_by_verification"

        else:  # request_verification
            # Mark both as needing verification
            new_memory.verification_requested = True
            new_memory.conflicted = True
            new_memory.conflicts_with.append(existing_memory.id)

            existing_memory.verification_requested = True
            existing_memory.conflicted = True
            existing_memory.conflicts_with.append(new_memory.id)

            return new_memory, existing_memory, "verification_requested"

    def generate_conflict_report(self, conflicts: List[ConflictDetection]) -> Dict[str, any]:
        """Generate a human-readable conflict report."""

        if not conflicts:
            return {"status": "no_conflicts", "message": "No conflicts detected"}

        report = {
            "status": "conflicts_detected",
            "conflict_count": len(conflicts),
            "conflicts": []
        }

        for conflict in conflicts:
            conflict_info = {
                "type": conflict.conflict_type,
                "confidence": conflict.confidence,
                "similarity": conflict.similarity_score,
                "strategy": conflict.resolution_strategy,
                "new_content": conflict.conflicted_memory.content[:100] + "...",
                "existing_content": conflict.conflicting_memories[0].content[:100] + "...",
                "recommendation": self._get_resolution_recommendation(conflict)
            }
            report["conflicts"].append(conflict_info)

        return report

    def _get_resolution_recommendation(self, conflict: ConflictDetection) -> str:
        """Get a human-readable recommendation for resolving the conflict."""

        strategy = conflict.resolution_strategy

        recommendations = {
            "trust_newer": "Accept the new information as it's more recent",
            "trust_higher_importance": "Accept the new information as it seems more important",
            "trust_existing_higher_importance": "Keep existing information as it has higher importance",
            "trust_verified": "Accept the new information as it's verified",
            "trust_existing_verified": "Keep existing information as it's verified",
            "trust_higher_confidence": "Accept the new information as it has higher confidence",
            "trust_existing_confidence": "Keep existing information as it has higher confidence",
            "request_verification": "Both memories need verification - please clarify which is correct"
        }

        return recommendations.get(strategy, "Manual review required")

    async def cleanup_superseded_memories(
        self,
        memories: List[Memory],
        days_threshold: int = 30
    ) -> int:
        """Clean up old superseded memories."""

        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        cleaned_count = 0

        superseded_memories = [
            m for m in memories
            if m.superseded_by is not None and m.created_at < cutoff_date
        ]

        # In a real implementation, you'd delete these from storage
        # For now, just count them
        cleaned_count = len(superseded_memories)

        return cleaned_count