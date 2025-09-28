"""
Main memory management system for Evo AI.
Coordinates storage, retrieval, and search across different memory types.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import UUID

from .embeddings import EmbeddingManager
from .emotion import EmotionDetector
from .conflict import ConflictResolver
from .models import (
    Memory,
    MemoryType,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
    MemoryQuery,
    MemorySearchResult,
    OutcomeType,
)
from .somatic import SomaticMarkerSystem
from .storage import SQLiteStorage


class MemoryManager:
    """Central memory management system for Evo AI."""

    def __init__(
        self,
        db_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_wal: bool = True,
        vector_extension_path: Optional[str] = None,
    ):
        self.storage = SQLiteStorage(
            db_path=db_path,
            enable_wal=enable_wal,
            vector_extension_path=vector_extension_path,
        )
        self.embeddings = EmbeddingManager(model_name=embedding_model)
        self.emotion_detector = EmotionDetector()
        self.somatic_system = SomaticMarkerSystem()
        self.conflict_resolver = ConflictResolver()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the memory system."""
        if self._initialized:
            return

        await self.storage.initialize()
        await self.embeddings.initialize()
        self._initialized = True

    async def store_episodic_memory(
        self,
        content: str,
        session_id: Optional[str] = None,
        user_message: Optional[str] = None,
        assistant_response: Optional[str] = None,
        context_summary: Optional[str] = None,
        emotions: Optional[Dict[str, float]] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EpisodicMemory:
        """Store an episodic memory (specific conversation/interaction)."""
        self._ensure_initialized()

        # Generate embedding for the content
        embedding = await self.embeddings.embed_text(content)

        # Auto-detect emotions if not provided and we have message content
        if emotions is None and user_message and assistant_response:
            emotional_context = self.emotion_detector.analyze_conversation(
                user_message, assistant_response, context_summary
            )
            emotions = {**emotional_context.user_emotions, **emotional_context.assistant_emotions}

            # Calculate enhanced importance using emotional factors
            importance = self.emotion_detector.calculate_memory_importance(
                importance, emotional_context
            )

        memory = EpisodicMemory(
            content=content,
            embedding=embedding,
            session_id=session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            context_summary=context_summary,
            emotions=emotions or {},
            importance=importance,
            metadata=metadata or {},
        )

        # Set emotional fields if we have emotional context
        if user_message and assistant_response:
            emotional_context = self.emotion_detector.analyze_conversation(
                user_message, assistant_response, context_summary
            )

            memory.valence = emotional_context.interaction_valence
            memory.emotional_intensity = emotional_context.emotional_intensity

            # Determine outcome type and valence
            outcome_type, outcome_valence = self.emotion_detector.detect_outcome_type(
                user_message, assistant_response
            )
            memory.outcome_type = OutcomeType(outcome_type)
            memory.outcome_valence = outcome_valence

        # Check for conflicts and duplicates before storing
        memory, storage_result = await self._handle_memory_conflicts(memory)

        await self.storage.store_memory(memory)
        return memory

    async def store_semantic_memory(
        self,
        content: str,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 0.8,
        source: Optional[str] = None,
        verified: bool = False,
        related_concepts: Optional[List[str]] = None,
        importance: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SemanticMemory:
        """Store a semantic memory (fact/knowledge)."""
        self._ensure_initialized()

        # Generate embedding for the content
        embedding = await self.embeddings.embed_text(content)

        memory = SemanticMemory(
            content=content,
            embedding=embedding,
            subject=subject,
            predicate=predicate,
            object=object,
            confidence=confidence,
            source=source,
            verified=verified,
            related_concepts=related_concepts or [],
            importance=importance,
            metadata=metadata or {},
        )

        await self.storage.store_memory(memory)
        return memory

    async def store_procedural_memory(
        self,
        content: str,
        skill_name: str,
        steps: Optional[List[str]] = None,
        prerequisites: Optional[List[str]] = None,
        success_rate: float = 0.0,
        proficiency_level: str = "beginner",
        importance: float = 0.6,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProceduralMemory:
        """Store a procedural memory (skill/procedure)."""
        self._ensure_initialized()

        # Generate embedding for the content
        embedding = await self.embeddings.embed_text(content)

        memory = ProceduralMemory(
            content=content,
            embedding=embedding,
            skill_name=skill_name,
            steps=steps or [],
            prerequisites=prerequisites or [],
            success_rate=success_rate,
            proficiency_level=proficiency_level,
            importance=importance,
            metadata=metadata or {},
        )

        await self.storage.store_memory(memory)
        return memory

    async def search(self, query: MemoryQuery) -> List[MemorySearchResult]:
        """Search memories using both text and vector similarity."""
        self._ensure_initialized()

        # Generate embedding for the query
        query_embedding = await self.embeddings.embed_text(query.text)

        # Perform vector search
        vector_results = await self.storage.search_memories_vector(
            query_embedding=query_embedding,
            memory_types=query.memory_types,
            limit=query.limit * 2,  # Get more candidates
            min_similarity=query.min_similarity,
        )

        # Perform text search
        text_results = await self.storage.search_memories_text(
            query=query.text,
            memory_types=query.memory_types,
            limit=query.limit * 2,
        )

        # Combine and rank results
        combined_results = self._combine_search_results(
            vector_results, text_results, query
        )

        # Apply additional filters
        filtered_results = self._apply_filters(combined_results, query)

        # Convert to MemorySearchResult objects
        search_results = []
        for i, (memory, score) in enumerate(filtered_results[:query.limit]):
            search_results.append(
                MemorySearchResult(
                    memory=memory,
                    similarity_score=score,
                    rank=i + 1,
                    explanation=self._generate_explanation(memory, query, score),
                )
            )

        return search_results

    async def retrieve(self, memory_id: UUID) -> Optional[Memory]:
        """Retrieve a specific memory by ID."""
        self._ensure_initialized()

        memory = await self.storage.retrieve_memory(memory_id)
        if memory:
            memory.access()
            await self.storage.store_memory(memory)  # Update access stats

        return memory

    async def search_similar(
        self,
        reference_memory: Memory,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 5,
        min_similarity: float = 0.5,
    ) -> List[MemorySearchResult]:
        """Find memories similar to a reference memory."""
        self._ensure_initialized()

        if not reference_memory.embedding:
            # Generate embedding if not present
            reference_memory.embedding = await self.embeddings.embed_text(
                reference_memory.content
            )

        results = await self.storage.search_memories_vector(
            query_embedding=reference_memory.embedding,
            memory_types=memory_types,
            limit=limit + 1,  # +1 to exclude the reference memory itself
            min_similarity=min_similarity,
        )

        # Filter out the reference memory
        filtered_results = [
            (memory, score)
            for memory, score in results
            if memory.id != reference_memory.id
        ]

        search_results = []
        for i, (memory, score) in enumerate(filtered_results[:limit]):
            search_results.append(
                MemorySearchResult(
                    memory=memory,
                    similarity_score=score,
                    rank=i + 1,
                    explanation=f"Similar to reference memory based on content similarity",
                )
            )

        return search_results

    async def delete_memory(self, memory_id: UUID) -> bool:
        """Delete a memory by ID."""
        self._ensure_initialized()
        return await self.storage.delete_memory(memory_id)

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        self._ensure_initialized()

        storage_stats = await self.storage.get_memory_stats()
        embedding_stats = self.embeddings.get_cache_info()

        return {
            "storage": storage_stats,
            "embeddings": embedding_stats,
            "initialized": self._initialized,
        }

    async def get_recent_memories(
        self,
        memory_types: Optional[List[MemoryType]] = None,
        hours: int = 24,
        limit: int = 10,
    ) -> List[Memory]:
        """Get recent memories from the specified time period."""
        self._ensure_initialized()

        # Create a query for recent memories
        query = MemoryQuery(
            text="*",  # Match all
            memory_types=memory_types,
            limit=min(limit * 3, 100),  # Get more to filter by time, but stay under limit
        )

        # Search and filter by time
        results = await self.search(query)
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_memories = [
            result.memory
            for result in results
            if result.memory.created_at >= cutoff_time
        ]

        return recent_memories[:limit]

    async def consolidate_memories(self, session_id: str) -> None:
        """Consolidate episodic memories from a session into semantic memories."""
        self._ensure_initialized()

        # This is a placeholder for the consolidation process
        # In a full implementation, this would:
        # 1. Retrieve all episodic memories from the session
        # 2. Analyze them for extractable facts/knowledge
        # 3. Create new semantic memories
        # 4. Update importance scores
        # 5. Optionally compress or archive old episodic memories

        # For now, we'll implement a simple version
        query = MemoryQuery(
            text="*",
            memory_types=[MemoryType.EPISODIC],
            limit=100,
        )

        results = await self.search(query)
        session_memories = [
            r.memory for r in results
            if isinstance(r.memory, EpisodicMemory) and r.memory.session_id == session_id
        ]

        # Simple consolidation: extract facts from conversations
        for memory in session_memories:
            if isinstance(memory, EpisodicMemory) and memory.user_message:
                # Look for fact patterns in user messages
                # This is a simplified implementation
                await self._extract_facts_from_episodic(memory)

    async def _extract_facts_from_episodic(self, memory: EpisodicMemory) -> None:
        """Extract facts from an episodic memory (simplified implementation)."""
        # In a real implementation, this would use NLP to extract facts
        # For now, we'll look for simple patterns

        if not memory.user_message:
            return

        user_msg = memory.user_message.lower()

        # Look for preference statements
        if "my favorite" in user_msg and "is" in user_msg:
            parts = user_msg.split("my favorite")[1].split("is")
            if len(parts) == 2:
                subject = f"user's favorite {parts[0].strip()}"
                obj = parts[1].strip()
                await self.store_semantic_memory(
                    content=f"The user's favorite {parts[0].strip()} is {obj}",
                    subject=subject,
                    predicate="is",
                    object=obj,
                    source=f"episodic_{memory.id}",
                    confidence=0.8,
                )

        # Look for skill/ability statements
        if "i can" in user_msg or "i know how to" in user_msg:
            if "i can" in user_msg:
                skill = user_msg.split("i can")[1].strip()
            else:
                skill = user_msg.split("i know how to")[1].strip()

            await self.store_procedural_memory(
                content=f"User can {skill}",
                skill_name=skill,
                steps=[],
                proficiency_level="unknown",
                success_rate=0.5,
            )

    def _combine_search_results(
        self,
        vector_results: List[tuple[Memory, float]],
        text_results: List[tuple[Memory, float]],
        query: MemoryQuery,
    ) -> List[tuple[Memory, float]]:
        """Combine vector and text search results with weighted scoring."""
        # Create a dictionary to combine scores
        combined_scores: Dict[UUID, tuple[Memory, float]] = {}

        # Add vector results (weight 0.7)
        for memory, score in vector_results:
            combined_scores[memory.id] = (memory, score * 0.7)

        # Add text results (weight 0.3) and combine with existing scores
        for memory, score in text_results:
            if memory.id in combined_scores:
                existing_memory, existing_score = combined_scores[memory.id]
                combined_score = existing_score + (score * 0.3)
                combined_scores[memory.id] = (existing_memory, combined_score)
            else:
                combined_scores[memory.id] = (memory, score * 0.3)

        # Convert back to list and sort by score
        results = list(combined_scores.values())
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def _apply_filters(
        self,
        results: List[tuple[Memory, float]],
        query: MemoryQuery,
    ) -> List[tuple[Memory, float]]:
        """Apply additional filters to search results."""
        filtered = []

        for memory, score in results:
            # Filter by age
            if query.max_age_days:
                age_days = (datetime.now() - memory.created_at).days
                if age_days > query.max_age_days:
                    continue

            # Filter by importance
            if query.min_importance and memory.importance < query.min_importance:
                continue

            # Filter by session ID
            if query.session_id:
                if isinstance(memory, EpisodicMemory):
                    if memory.session_id != query.session_id:
                        continue
                else:
                    continue  # Non-episodic memories don't have session_id

            filtered.append((memory, score))

        return filtered

    def _generate_explanation(
        self, memory: Memory, query: MemoryQuery, score: float
    ) -> str:
        """Generate an explanation for why this memory was retrieved."""
        explanations = []

        if score > 0.8:
            explanations.append("High semantic similarity")
        elif score > 0.6:
            explanations.append("Good semantic similarity")
        elif score > 0.4:
            explanations.append("Moderate semantic similarity")
        else:
            explanations.append("Low semantic similarity")

        if memory.importance > 0.8:
            explanations.append("high importance")
        elif memory.importance > 0.6:
            explanations.append("moderate importance")

        if memory.access_count > 10:
            explanations.append("frequently accessed")
        elif memory.access_count > 5:
            explanations.append("occasionally accessed")

        return ", ".join(explanations)

    def _ensure_initialized(self) -> None:
        """Ensure the memory system is initialized."""
        if not self._initialized:
            raise RuntimeError(
                "MemoryManager not initialized. Call initialize() first."
            )

    async def get_decision_recommendation(
        self,
        situation: str,
        possible_actions: List[str],
        current_emotions: Optional[Dict[str, float]] = None,
        importance: float = 0.5,
        time_pressure: float = 0.5
    ) -> Dict[str, Any]:
        """Get a decision recommendation based on emotional memories."""
        self._ensure_initialized()

        # Import here to avoid circular imports
        from .somatic import DecisionContext

        # Search for relevant memories
        query = MemoryQuery(
            text=situation,
            memory_types=[MemoryType.EPISODIC, MemoryType.PROCEDURAL],
            limit=20,
            min_similarity=0.2
        )

        search_results = await self.search(query)
        relevant_memories = [result.memory for result in search_results]

        # Create decision context
        context = DecisionContext(
            situation_description=situation,
            possible_actions=possible_actions,
            current_emotional_state=current_emotions or {},
            time_pressure=time_pressure,
            importance=importance
        )

        # Get recommendation from somatic marker system
        recommendation = self.somatic_system.evaluate_decision(context, relevant_memories)

        return {
            "recommended_action": recommendation.recommended_action,
            "confidence": recommendation.confidence,
            "rationale": recommendation.emotional_rationale,
            "risk_assessment": recommendation.risk_assessment,
            "supporting_memories": len(recommendation.supporting_markers),
            "emotional_factors": [marker.emotion for marker in recommendation.supporting_markers]
        }

    async def learn_from_decision_outcome(
        self,
        situation: str,
        chosen_action: str,
        outcome_type: str,
        outcome_emotions: Dict[str, float],
        importance: float = 0.5
    ) -> None:
        """Learn from the outcome of a decision for future recommendations."""
        self._ensure_initialized()

        # Import here to avoid circular imports
        from .somatic import DecisionContext

        context = DecisionContext(
            situation_description=situation,
            possible_actions=[chosen_action],
            current_emotional_state={},
            importance=importance
        )

        self.somatic_system.learn_from_outcome(
            context,
            chosen_action,
            OutcomeType(outcome_type),
            outcome_emotions
        )

    async def update_memory_activation(self, memory_id: UUID) -> None:
        """Update a memory's activation level when accessed."""
        self._ensure_initialized()

        memory = await self.storage.retrieve_memory(memory_id)
        if memory:
            memory.access()
            memory.update_dormancy_state()
            await self.storage.store_memory(memory)

    async def get_dormant_memories(self, reactivation_cue: str, limit: int = 5) -> List[Memory]:
        """Retrieve dormant memories that might be reactivated by a cue."""
        self._ensure_initialized()

        # Search for memories that might be dormant but relevant
        query = MemoryQuery(
            text=reactivation_cue,
            limit=limit * 3,  # Get more candidates
            min_similarity=0.15  # Lower threshold for dormant memory reactivation
        )

        search_results = await self.search(query)

        # Filter for dormant memories that could be reactivated
        dormant_memories = []
        for result in search_results:
            memory = result.memory
            if memory.dormancy_state.value in ['dormant', 'deep_sleep']:
                # Calculate reactivation probability
                activation = memory.calculate_current_activation()
                if activation > 0.2:  # Has some potential for reactivation
                    dormant_memories.append(memory)
                    # Reactivate the memory
                    await self.update_memory_activation(memory.id)

        return dormant_memories[:limit]

    async def consolidate_memories(self, hours_threshold: int = 24) -> Dict[str, int]:
        """Consolidate recent episodic memories into semantic memories."""
        self._ensure_initialized()

        # Get recent episodic memories
        cutoff_time = datetime.now() - timedelta(hours=hours_threshold)

        # Find episodic memories that could be consolidated
        recent_memories = await self.get_recent_memories(
            memory_types=[MemoryType.EPISODIC],
            hours=hours_threshold,
            limit=50
        )

        consolidation_stats = {
            "reviewed": len(recent_memories),
            "consolidated": 0,
            "patterns_found": 0
        }

        # Enhanced consolidation: look for repeated facts AND emotional patterns
        fact_patterns = {}
        emotional_patterns = {}

        for memory in recent_memories:
            if isinstance(memory, EpisodicMemory) and memory.importance > 0.6:
                content = memory.content.lower()

                # Extract factual patterns
                if any(pattern in content for pattern in ['is a', 'are', 'means', 'definition']):
                    if memory.user_message:
                        key = memory.user_message[:50]
                        if key not in fact_patterns:
                            fact_patterns[key] = []
                        fact_patterns[key].append(memory)

                # Extract emotional learning patterns
                if memory.emotions and memory.emotional_intensity > 0.4:
                    # Group by dominant emotion + outcome type
                    dominant_emotion = max(memory.emotions.items(), key=lambda x: x[1])[0]
                    pattern_key = f"{dominant_emotion}_{memory.outcome_type.value}"

                    if pattern_key not in emotional_patterns:
                        emotional_patterns[pattern_key] = {
                            "memories": [],
                            "total_valence": 0.0,
                            "success_count": 0,
                            "failure_count": 0
                        }

                    emotional_patterns[pattern_key]["memories"].append(memory)
                    emotional_patterns[pattern_key]["total_valence"] += memory.valence

                    if memory.outcome_type.value in ["success", "learning"]:
                        emotional_patterns[pattern_key]["success_count"] += 1
                    elif memory.outcome_type.value in ["failure", "regret"]:
                        emotional_patterns[pattern_key]["failure_count"] += 1

        # Create semantic memories from consolidated patterns
        for pattern, memories in fact_patterns.items():
            if len(memories) >= 2:  # At least 2 similar episodic memories
                # Create a consolidated semantic memory
                combined_content = f"Learned from {len(memories)} conversations: {pattern}"

                await self.store_semantic_memory(
                    content=combined_content,
                    subject="consolidated_learning",
                    predicate="learned_from_conversation",
                    object=pattern,
                    confidence=0.7,
                    source="memory_consolidation",
                    importance=max(mem.importance for mem in memories),
                    metadata={
                        "source_memories": [str(mem.id) for mem in memories],
                        "consolidated_at": datetime.now().isoformat()
                    }
                )

                consolidation_stats["consolidated"] += 1

        # Create semantic memories from emotional patterns
        for pattern_key, pattern_data in emotional_patterns.items():
            if len(pattern_data["memories"]) >= 3:  # Need more examples for emotional patterns
                emotion, outcome = pattern_key.split("_")
                avg_valence = pattern_data["total_valence"] / len(pattern_data["memories"])
                success_rate = pattern_data["success_count"] / (pattern_data["success_count"] + pattern_data["failure_count"]) if (pattern_data["success_count"] + pattern_data["failure_count"]) > 0 else 0.5

                # Create emotional behavior semantic memory
                emotional_learning = f"When user experiences {emotion}, outcomes tend to be {outcome} with {success_rate:.1%} success rate"

                await self.store_semantic_memory(
                    content=emotional_learning,
                    subject=f"user_emotion_{emotion}",
                    predicate="tends_to_result_in",
                    object=f"{outcome}_with_{success_rate:.0%}_success",
                    confidence=min(0.9, 0.5 + (len(pattern_data["memories"]) * 0.1)),  # More examples = higher confidence
                    source="emotional_consolidation",
                    importance=0.7 + abs(avg_valence) * 0.2,  # Strong emotions = higher importance
                    metadata={
                        "emotion": emotion,
                        "outcome_type": outcome,
                        "avg_valence": avg_valence,
                        "success_rate": success_rate,
                        "sample_size": len(pattern_data["memories"]),
                        "source_memory_ids": [str(mem.id) for mem in pattern_data["memories"]],
                        "consolidated_at": datetime.now().isoformat()
                    }
                )

                consolidation_stats["consolidated"] += 1

        consolidation_stats["patterns_found"] = len(fact_patterns) + len(emotional_patterns)
        consolidation_stats["emotional_patterns"] = len(emotional_patterns)
        return consolidation_stats

    async def get_emotional_insights(self) -> Dict[str, Any]:
        """Get insights about emotional patterns in memory."""
        self._ensure_initialized()

        # Get recent memories with emotions
        recent_memories = await self.get_recent_memories(
            memory_types=[MemoryType.EPISODIC],
            hours=168,  # Last week
            limit=50
        )

        emotion_stats = {}
        valence_history = []
        outcome_patterns = {}

        for memory in recent_memories:
            if memory.emotions:
                for emotion, intensity in memory.emotions.items():
                    if emotion not in emotion_stats:
                        emotion_stats[emotion] = {"count": 0, "avg_intensity": 0, "total": 0}

                    emotion_stats[emotion]["count"] += 1
                    emotion_stats[emotion]["total"] += intensity

            if hasattr(memory, 'valence'):
                valence_history.append(memory.valence)

            if hasattr(memory, 'outcome_type'):
                outcome = memory.outcome_type.value
                if outcome not in outcome_patterns:
                    outcome_patterns[outcome] = 0
                outcome_patterns[outcome] += 1

        # Calculate averages
        for emotion_data in emotion_stats.values():
            emotion_data["avg_intensity"] = emotion_data["total"] / emotion_data["count"]

        # Get decision patterns from somatic system
        decision_patterns = self.somatic_system.get_decision_patterns()

        return {
            "emotion_frequencies": emotion_stats,
            "avg_valence": sum(valence_history) / len(valence_history) if valence_history else 0,
            "outcome_patterns": outcome_patterns,
            "decision_patterns": decision_patterns,
            "memory_count": len(recent_memories)
        }

    async def _handle_memory_conflicts(self, new_memory: Memory) -> Tuple[Memory, Dict[str, Any]]:
        """Handle conflicts and duplicates when storing a new memory."""

        # Get recent memories of the same type to check against
        recent_query = MemoryQuery(
            text=new_memory.content,
            memory_types=[new_memory.type],
            limit=50,
            min_similarity=0.0  # Get all memories for comparison
        )

        search_results = await self.search(recent_query)
        existing_memories = [result.memory for result in search_results]

        storage_result = {
            "action": "stored",
            "duplicates_detected": 0,
            "conflicts_detected": 0,
            "reinforcements": 0,
            "conflicts_resolved": 0
        }

        # Check for duplicates first
        duplicate_detection = await self.conflict_resolver.detect_duplicates(
            new_memory, existing_memories
        )

        if duplicate_detection:
            # Reinforce the existing memory instead of storing duplicate
            reinforced_memory, action = await self.conflict_resolver.resolve_duplicate(
                duplicate_detection, new_memory
            )

            storage_result["action"] = "reinforced"
            storage_result["duplicates_detected"] = len(duplicate_detection.duplicate_candidates)
            storage_result["reinforcements"] = 1

            # Update the existing memory in storage
            await self.storage.store_memory(reinforced_memory)

            return reinforced_memory, storage_result

        # Check for conflicts
        conflicts = await self.conflict_resolver.detect_conflicts(
            new_memory, existing_memories
        )

        if conflicts:
            storage_result["conflicts_detected"] = len(conflicts)

            # Resolve each conflict
            for conflict in conflicts:
                primary_memory, secondary_memory, resolution = await self.conflict_resolver.resolve_conflict(
                    conflict, new_memory
                )

                # Update both memories in storage
                await self.storage.store_memory(primary_memory)
                await self.storage.store_memory(secondary_memory)

                storage_result["conflicts_resolved"] += 1

                # If new memory was rejected, return the existing one
                if resolution.startswith("rejected"):
                    storage_result["action"] = "rejected_due_to_conflict"
                    return primary_memory, storage_result

        return new_memory, storage_result

    async def get_conflict_report(self, days: int = 7) -> Dict[str, Any]:
        """Get a report of memory conflicts and duplicates."""

        # Get recent memories
        recent_memories = await self.get_recent_memories(
            memory_types=None,  # All types
            hours=days * 24,
            limit=100
        )

        # Count conflicts and verification requests
        conflicted_memories = [m for m in recent_memories if m.conflicted]
        verification_needed = [m for m in recent_memories if m.verification_requested]
        superseded_memories = [m for m in recent_memories if m.superseded_by is not None]

        # Group by conflict types
        conflict_summary = {}
        for memory in conflicted_memories:
            memory_type = memory.type.value
            if memory_type not in conflict_summary:
                conflict_summary[memory_type] = {"count": 0, "examples": []}

            conflict_summary[memory_type]["count"] += 1
            if len(conflict_summary[memory_type]["examples"]) < 3:
                conflict_summary[memory_type]["examples"].append({
                    "content": memory.content[:100] + "...",
                    "conflicts_with_count": len(memory.conflicts_with),
                    "created_at": memory.created_at.isoformat()
                })

        return {
            "period_days": days,
            "total_memories": len(recent_memories),
            "conflicted_memories": len(conflicted_memories),
            "verification_needed": len(verification_needed),
            "superseded_memories": len(superseded_memories),
            "conflict_rate": len(conflicted_memories) / len(recent_memories) if recent_memories else 0,
            "conflict_summary": conflict_summary,
            "recommendations": self._generate_conflict_recommendations(
                conflicted_memories, verification_needed
            )
        }

    def _generate_conflict_recommendations(
        self,
        conflicted_memories: List[Memory],
        verification_needed: List[Memory]
    ) -> List[str]:
        """Generate recommendations for handling conflicts."""

        recommendations = []

        if len(verification_needed) > 0:
            recommendations.append(
                f"{len(verification_needed)} memories need verification. "
                "Please review and confirm which information is correct."
            )

        if len(conflicted_memories) > 5:
            recommendations.append(
                "High number of memory conflicts detected. "
                "Consider reviewing information sources for consistency."
            )

        conflict_rate = len(conflicted_memories) / max(len(conflicted_memories) + len(verification_needed), 1)
        if conflict_rate > 0.1:
            recommendations.append(
                "Conflict rate is high. This might indicate rapidly changing information "
                "or inconsistent data sources."
            )

        if not recommendations:
            recommendations.append("Memory system is operating normally with minimal conflicts.")

        return recommendations

    async def close(self) -> None:
        """Close the memory system and clean up resources."""
        if self._initialized:
            await self.storage.close()
            self.embeddings.clear_cache()
            self._initialized = False