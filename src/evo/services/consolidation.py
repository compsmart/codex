"""
Memory consolidation service for Evo AI.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..memory import MemoryManager, MemoryQuery, MemoryType
from ..summarization import ContextSummarizer, SummaryConfig


class MemoryConsolidationService:
    """Service for consolidating and organizing memories."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        summarizer: Optional[ContextSummarizer] = None,
        consolidation_interval_hours: int = 12,
    ):
        self.memory_manager = memory_manager
        self.summarizer = summarizer
        self.consolidation_interval_hours = consolidation_interval_hours

        self.logger = logging.getLogger(__name__)
        self._running = False
        self._stop_requested = False

        # Statistics
        self.stats = {
            "consolidations_performed": 0,
            "memories_consolidated": 0,
            "facts_extracted": 0,
            "summaries_created": 0,
            "last_consolidation": None,
        }

    async def start(self) -> None:
        """Start the consolidation service."""
        if self._running:
            return

        self._running = True
        self._stop_requested = False

        self.logger.info("Starting memory consolidation service")

        try:
            while self._running and not self._stop_requested:
                await self._consolidation_cycle()
                # Wait for next cycle
                await asyncio.sleep(self.consolidation_interval_hours * 3600)

        except Exception as e:
            self.logger.error(f"Memory consolidation service error: {e}")
        finally:
            self._running = False

    async def stop(self) -> None:
        """Stop the consolidation service."""
        self.logger.info("Stopping memory consolidation service")
        self._stop_requested = True

    async def schedule_consolidation(self, delay_minutes: int = 0) -> None:
        """Schedule an immediate consolidation."""
        async def delayed_consolidation():
            if delay_minutes > 0:
                await asyncio.sleep(delay_minutes * 60)
            await self._consolidation_cycle()

        asyncio.create_task(delayed_consolidation())

    async def _consolidation_cycle(self) -> None:
        """Perform one consolidation cycle."""
        try:
            self.logger.info("Starting memory consolidation cycle")

            # Get unconsolidated memories
            cutoff_time = datetime.now() - timedelta(hours=24)
            consolidated_count = 0

            # Consolidate episodic memories
            consolidated_count += await self._consolidate_episodic_memories(cutoff_time)

            # Extract facts from recent conversations
            if self.summarizer:
                consolidated_count += await self._extract_facts_from_conversations(cutoff_time)

            # Create conversation summaries
            if self.summarizer:
                consolidated_count += await self._create_conversation_summaries(cutoff_time)

            # Organize and cluster related memories
            consolidated_count += await self._organize_semantic_memories()

            # Update statistics
            self.stats["consolidations_performed"] += 1
            self.stats["memories_consolidated"] += consolidated_count
            self.stats["last_consolidation"] = datetime.now().isoformat()

            self.logger.info(f"Consolidation cycle completed. Processed {consolidated_count} memories")

        except Exception as e:
            self.logger.error(f"Error in consolidation cycle: {e}")

    async def _consolidate_episodic_memories(self, cutoff_time: datetime) -> int:
        """Consolidate episodic memories into semantic knowledge."""
        try:
            # Get recent episodic memories
            query = MemoryQuery(
                text="*",
                memory_types=[MemoryType.EPISODIC],
                limit=100,
            )

            results = await self.memory_manager.search(query)
            consolidated_count = 0

            # Group memories by session
            session_memories = {}
            for result in results:
                memory = result.memory
                if memory.created_at < cutoff_time:
                    continue

                session_id = getattr(memory, 'session_id', 'unknown')
                if session_id not in session_memories:
                    session_memories[session_id] = []
                session_memories[session_id].append(memory)

            # Process each session
            for session_id, memories in session_memories.items():
                if len(memories) < 2:  # Need at least 2 memories to consolidate
                    continue

                # Extract patterns and knowledge
                await self._extract_session_knowledge(session_id, memories)
                consolidated_count += len(memories)

            return consolidated_count

        except Exception as e:
            self.logger.error(f"Error consolidating episodic memories: {e}")
            return 0

    async def _extract_session_knowledge(self, session_id: str, memories: List) -> None:
        """Extract knowledge from a session's memories."""
        try:
            # Analyze conversation patterns
            user_preferences = []
            learning_outcomes = []
            facts = []

            for memory in memories:
                if not hasattr(memory, 'user_message') or not memory.user_message:
                    continue

                user_msg = memory.user_message.lower()

                # Extract preferences
                if "my favorite" in user_msg:
                    # Extract preference
                    parts = user_msg.split("my favorite")
                    if len(parts) > 1 and "is" in parts[1]:
                        pref_parts = parts[1].split("is")
                        if len(pref_parts) == 2:
                            subject = f"user's favorite {pref_parts[0].strip()}"
                            obj = pref_parts[1].strip()
                            user_preferences.append((subject, obj))

                # Extract learning indicators
                if any(word in user_msg for word in ["learned", "understand", "now i know"]):
                    learning_outcomes.append(memory.content[:200])

                # Extract factual statements
                if "is a" in user_msg or "are" in user_msg:
                    facts.append(user_msg)

            # Store extracted knowledge as semantic memories
            for subject, obj in user_preferences:
                await self.memory_manager.store_semantic_memory(
                    content=f"The {subject} is {obj}",
                    subject=subject,
                    predicate="is",
                    object=obj,
                    confidence=0.8,
                    source=f"consolidation_{session_id}",
                    importance=0.7,
                )

            # Store learning outcomes
            for outcome in learning_outcomes:
                await self.memory_manager.store_procedural_memory(
                    content=outcome,
                    skill_name="learning_outcome",
                    importance=0.6,
                    metadata={"source": f"consolidation_{session_id}"},
                )

        except Exception as e:
            self.logger.error(f"Error extracting session knowledge: {e}")

    async def _extract_facts_from_conversations(self, cutoff_time: datetime) -> int:
        """Extract facts from recent conversations."""
        if not self.summarizer:
            return 0

        try:
            # Get recent sessions
            query = MemoryQuery(
                text="*",
                memory_types=[MemoryType.EPISODIC],
                limit=50,
            )

            results = await self.memory_manager.search(query)
            sessions = set()

            for result in results:
                memory = result.memory
                if memory.created_at >= cutoff_time:
                    session_id = getattr(memory, 'session_id', None)
                    if session_id:
                        sessions.add(session_id)

            # Extract facts from each session
            facts_extracted = 0
            for session_id in sessions:
                try:
                    facts = await self.summarizer.extract_facts_from_conversation(session_id)
                    facts_extracted += len(facts)
                except Exception as e:
                    self.logger.warning(f"Failed to extract facts from session {session_id}: {e}")

            self.stats["facts_extracted"] += facts_extracted
            return facts_extracted

        except Exception as e:
            self.logger.error(f"Error extracting facts from conversations: {e}")
            return 0

    async def _create_conversation_summaries(self, cutoff_time: datetime) -> int:
        """Create summaries for recent conversations."""
        if not self.summarizer:
            return 0

        try:
            # Get sessions that need summarization
            query = MemoryQuery(
                text="*",
                memory_types=[MemoryType.EPISODIC],
                limit=20,
            )

            results = await self.memory_manager.search(query)
            sessions = set()

            for result in results:
                memory = result.memory
                if memory.created_at >= cutoff_time:
                    session_id = getattr(memory, 'session_id', None)
                    if session_id:
                        sessions.add(session_id)

            # Create summaries
            summaries_created = 0
            for session_id in sessions:
                try:
                    summary = await self.summarizer.summarize_conversation(session_id)
                    if summary:
                        summaries_created += 1
                except Exception as e:
                    self.logger.warning(f"Failed to summarize session {session_id}: {e}")

            self.stats["summaries_created"] += summaries_created
            return summaries_created

        except Exception as e:
            self.logger.error(f"Error creating conversation summaries: {e}")
            return 0

    async def _organize_semantic_memories(self) -> int:
        """Organize and cluster related semantic memories."""
        try:
            # Get semantic memories
            query = MemoryQuery(
                text="*",
                memory_types=[MemoryType.SEMANTIC],
                limit=200,
            )

            results = await self.memory_manager.search(query)

            # Simple clustering by subject similarity
            clusters = {}
            for result in results:
                memory = result.memory
                if hasattr(memory, 'subject'):
                    subject_key = self._normalize_subject(memory.subject)
                    if subject_key not in clusters:
                        clusters[subject_key] = []
                    clusters[subject_key].append(memory)

            # Update importance scores based on clustering
            updated_count = 0
            for cluster_key, cluster_memories in clusters.items():
                if len(cluster_memories) > 1:
                    # Boost importance for memories that appear in clusters
                    for memory in cluster_memories:
                        # This would update the memory importance
                        # For now, just count as processed
                        updated_count += 1

            return updated_count

        except Exception as e:
            self.logger.error(f"Error organizing semantic memories: {e}")
            return 0

    def _normalize_subject(self, subject: str) -> str:
        """Normalize subject for clustering."""
        # Simple normalization
        normalized = subject.lower().strip()

        # Remove common prefixes
        prefixes = ["user's", "the", "a", "an"]
        for prefix in prefixes:
            if normalized.startswith(prefix + " "):
                normalized = normalized[len(prefix) + 1:]

        return normalized

    async def get_status(self) -> Dict[str, Any]:
        """Get consolidation service status."""
        return {
            "running": self._running,
            "last_consolidation": self.stats["last_consolidation"],
            "consolidations_performed": self.stats["consolidations_performed"],
            "memories_consolidated": self.stats["memories_consolidated"],
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Get detailed consolidation statistics."""
        return self.stats.copy()