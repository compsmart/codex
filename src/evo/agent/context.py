"""
Context management for Evo AI agent conversations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..memory import MemoryManager, MemoryQuery, MemoryType
from .models import ChatMessage, ChatRole, ConversationContext, AgentConfig


class ContextManager:
    """Manages conversation context and memory integration."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        config: AgentConfig,
    ):
        self.memory_manager = memory_manager
        self.config = config
        self._contexts: Dict[str, ConversationContext] = {}

    async def create_context(self, session_id: Optional[str] = None) -> ConversationContext:
        """Create a new conversation context."""
        if session_id is None:
            session_id = str(uuid4())

        context = ConversationContext(session_id=session_id)
        self._contexts[session_id] = context

        # Retrieve relevant memories for initialization
        await self._load_relevant_memories(context)

        return context

    async def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get an existing conversation context."""
        return self._contexts.get(session_id)

    async def add_message(
        self,
        session_id: str,
        message: ChatMessage,
    ) -> ConversationContext:
        """Add a message to the conversation context."""
        context = self._contexts.get(session_id)
        if context is None:
            context = await self.create_context(session_id)

        context.add_message(message)

        # Retrieve relevant memories based on the new message
        if message.role == ChatRole.USER:
            await self._update_context_with_memories(context, message.content)

        # Trim context if it gets too long
        await self._trim_context_if_needed(context)

        return context

    async def prepare_context_for_llm(
        self,
        context: ConversationContext,
        include_memories: bool = True,
    ) -> List[Dict[str, Any]]:
        """Prepare the conversation context for LLM input."""
        messages = []

        # Add system message with relevant memories
        if include_memories and context.retrieved_memories:
            memory_context = self._format_memories_for_context(
                context.retrieved_memories
            )
            system_content = f"{self.config.system_prompt}\n\n{memory_context}"
        else:
            system_content = self.config.system_prompt

        messages.append({
            "role": "system",
            "content": system_content,
        })

        # Add conversation messages
        for msg in context.get_recent_messages(self.config.memory_limit):
            messages.append({
                "role": msg.role.value,
                "content": msg.content,
            })

        return messages

    async def store_conversation_memory(
        self,
        context: ConversationContext,
        user_message: str,
        assistant_response: str,
        importance: Optional[float] = None,
    ) -> None:
        """Store the conversation as episodic memory."""
        if not self.config.use_memory:
            return

        # Calculate importance if not provided
        if importance is None:
            importance = await self._calculate_message_importance(
                user_message, assistant_response
            )

        # Only store if above threshold
        if importance < self.config.episodic_memory_threshold:
            return

        # Create context summary
        context_summary = await self._create_context_summary(context)

        # Store as episodic memory
        await self.memory_manager.store_episodic_memory(
            content=f"User: {user_message}\nAssistant: {assistant_response}",
            session_id=context.session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            context_summary=context_summary,
            importance=importance,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "message_count": len(context.messages),
            },
        )

    async def _load_relevant_memories(self, context: ConversationContext) -> None:
        """Load relevant memories for the conversation context."""
        if not self.config.use_memory:
            return

        # Get recent memories from the same session
        recent_memories = await self.memory_manager.get_recent_memories(
            memory_types=[MemoryType.EPISODIC],
            hours=24,
            limit=5,
        )

        # Filter for this session
        session_memories = [
            mem for mem in recent_memories
            if hasattr(mem, 'session_id') and mem.session_id == context.session_id
        ]

        # Add to context
        for memory in session_memories:
            context.retrieved_memories.append({
                "type": "recent_session",
                "content": memory.content,
                "importance": memory.importance,
                "timestamp": memory.created_at.isoformat(),
            })

    async def _update_context_with_memories(
        self,
        context: ConversationContext,
        query_text: str,
    ) -> None:
        """Update context with memories relevant to the query."""
        if not self.config.use_memory:
            return

        # Strategy 1: Direct search with the user's query
        primary_query = MemoryQuery(
            text=query_text,
            memory_types=[MemoryType.SEMANTIC, MemoryType.PROCEDURAL, MemoryType.EPISODIC],
            limit=8,
            min_similarity=0.25,
        )

        search_results = await self.memory_manager.search(primary_query)

        # Strategy 2: If we don't have enough relevant memories, do a broader search
        # This helps when user asks general questions like "tell me about my dog"
        # but memories contain specific names/details
        if len(search_results) < 3:
            # Extract key words from the query and search with lower threshold
            words = query_text.lower().split()
            # Remove common words that don't help with search
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'tell', 'me', 'about', 'what', 'how', 'when', 'where', 'why', 'is', 'are', 'was', 'were'}
            key_words = [word for word in words if word not in stop_words and len(word) > 2]

            if key_words:
                # Try each key word individually to catch related memories
                for word in key_words[:3]:  # Limit to prevent too many searches
                    word_query = MemoryQuery(
                        text=word,
                        memory_types=[MemoryType.SEMANTIC, MemoryType.EPISODIC],
                        limit=5,
                        min_similarity=0.2,
                    )
                    word_results = await self.memory_manager.search(word_query)

                    # Add new results that aren't duplicates
                    for result in word_results:
                        if not any(existing.memory.content == result.memory.content for existing in search_results):
                            search_results.append(result)

        # Strategy 3: For very general queries, also include high-importance memories
        general_indicators = ['about me', 'know about', 'tell me', 'what do you know']
        if any(indicator in query_text.lower() for indicator in general_indicators):
            # Get high-importance memories regardless of similarity
            important_query = MemoryQuery(
                text="",  # Empty text to get all memories
                memory_types=[MemoryType.SEMANTIC],
                limit=5,
                min_similarity=0.0,  # No similarity filter
            )
            important_results = await self.memory_manager.search(important_query)

            # Add important memories (importance > 0.7) that aren't already included
            for result in important_results:
                if (result.memory.importance > 0.7 and
                    not any(existing.memory.content == result.memory.content for existing in search_results)):
                    search_results.append(result)

        # Add new relevant memories to context
        for result in search_results:
            memory_info = {
                "type": "relevant",
                "content": result.memory.content,
                "similarity": result.similarity_score,
                "importance": result.memory.importance,
                "memory_type": result.memory.type.value,
                "explanation": result.explanation,
            }

            # Check if this memory is already in context
            if not any(
                mem.get("content") == memory_info["content"]
                for mem in context.retrieved_memories
            ):
                context.retrieved_memories.append(memory_info)

        # Limit the number of memories in context
        max_memories = 10
        if len(context.retrieved_memories) > max_memories:
            # Keep the most important and recent memories
            context.retrieved_memories.sort(
                key=lambda x: (x.get("importance", 0), x.get("similarity", 0)),
                reverse=True,
            )
            context.retrieved_memories = context.retrieved_memories[:max_memories]

    async def _trim_context_if_needed(self, context: ConversationContext) -> None:
        """Trim context if it exceeds maximum length."""
        current_length = context.get_context_length()

        if current_length > self.config.max_context_length:
            # Calculate how many messages to keep
            target_length = self.config.max_context_length - self.config.context_window_overlap

            # Keep messages from the end until we reach target length
            trimmed_messages = []
            current_chars = 0

            for message in reversed(context.messages):
                message_chars = len(message.content)
                if current_chars + message_chars > target_length:
                    break
                trimmed_messages.insert(0, message)
                current_chars += message_chars

            # Store trimmed messages as episodic memories if important
            if len(context.messages) - len(trimmed_messages) > 0:
                await self._archive_trimmed_messages(
                    context, context.messages[: -(len(trimmed_messages))]
                )

            context.messages = trimmed_messages

    async def _archive_trimmed_messages(
        self,
        context: ConversationContext,
        trimmed_messages: List[ChatMessage],
    ) -> None:
        """Archive trimmed messages as episodic memories."""
        # Group messages into conversation pairs
        user_msg = None
        for message in trimmed_messages:
            if message.role == ChatRole.USER:
                user_msg = message
            elif message.role == ChatRole.ASSISTANT and user_msg:
                # Calculate importance
                importance = await self._calculate_message_importance(
                    user_msg.content, message.content
                )

                if importance >= self.config.episodic_memory_threshold:
                    await self.memory_manager.store_episodic_memory(
                        content=f"User: {user_msg.content}\nAssistant: {message.content}",
                        session_id=context.session_id,
                        user_message=user_msg.content,
                        assistant_response=message.content,
                        context_summary="Archived from context trimming",
                        importance=importance,
                        metadata={
                            "archived": True,
                            "original_timestamp": user_msg.timestamp.isoformat(),
                        },
                    )

                user_msg = None

    async def _calculate_message_importance(
        self,
        user_message: str,
        assistant_response: str,
    ) -> float:
        """Calculate the importance of a message pair."""
        # Simple heuristic-based importance calculation
        importance = 0.3  # Base importance

        # Check for learning indicators
        learning_keywords = [
            "remember", "learn", "my favorite", "i am", "i like", "i don't like",
            "my name is", "i work", "i live", "preference", "setting",
        ]

        user_lower = user_message.lower()
        for keyword in learning_keywords:
            if keyword in user_lower:
                importance += 0.2
                break

        # Check for question/answer patterns
        if "?" in user_message:
            importance += 0.1

        # Check for factual content
        factual_keywords = ["what is", "how to", "explain", "define", "teach me"]
        for keyword in factual_keywords:
            if keyword in user_lower:
                importance += 0.15
                break

        # Check for code or technical content
        if any(keyword in user_lower for keyword in ["code", "function", "class", "error", "debug"]):
            importance += 0.1

        # Check response length (longer responses might be more important)
        if len(assistant_response) > 500:
            importance += 0.1
        elif len(assistant_response) > 200:
            importance += 0.05

        return min(importance, 1.0)  # Cap at 1.0

    async def _create_context_summary(self, context: ConversationContext) -> str:
        """Create a summary of the conversation context."""
        if len(context.messages) == 0:
            return "Empty conversation"

        # Simple summary based on recent messages
        recent_messages = context.get_recent_messages(5)

        # Extract topics/keywords from recent messages
        topics = []
        for msg in recent_messages:
            if msg.role == ChatRole.USER:
                # Simple keyword extraction
                words = msg.content.lower().split()
                # Filter out common words and keep meaningful ones
                meaningful_words = [
                    word for word in words
                    if len(word) > 3 and word not in [
                        "what", "how", "when", "where", "why", "the", "and", "but", "for", "with"
                    ]
                ]
                topics.extend(meaningful_words[:3])  # Take first 3 meaningful words

        if topics:
            unique_topics = list(set(topics))[:5]  # Keep unique topics, max 5
            return f"Discussion about: {', '.join(unique_topics)}"
        else:
            return "General conversation"

    def _format_memories_for_context(self, memories: List[Dict[str, Any]]) -> str:
        """Format retrieved memories for inclusion in context."""
        if not memories:
            return ""

        # Group memories by type
        recent_memories = [m for m in memories if m.get("type") == "recent_session"]
        relevant_memories = [m for m in memories if m.get("type") == "relevant"]

        formatted = ["# What I Know About You"]
        formatted.append("\nBased on our previous conversations, here's what I remember:")

        # Combine and sort memories by importance/similarity
        all_knowledge = relevant_memories + recent_memories
        # Sort by importance first, then similarity
        all_knowledge.sort(key=lambda x: (x.get("importance", 0), x.get("similarity", 0)), reverse=True)

        if all_knowledge:
            for memory in all_knowledge[:8]:  # Limit to most important/relevant
                content = memory["content"]
                # Present as established facts, not search results
                formatted.append(f"• {content}")

        formatted.append("\nRespond naturally using this information as if you've always known it.")
        formatted.append("Do not mention that you're recalling or searching - just use the knowledge conversationally.")

        return "\n".join(formatted)

    async def cleanup_old_contexts(self, max_age_hours: int = 24) -> None:
        """Clean up old conversation contexts."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        contexts_to_remove = []
        for session_id, context in self._contexts.items():
            if context.updated_at < cutoff_time:
                contexts_to_remove.append(session_id)

        for session_id in contexts_to_remove:
            del self._contexts[session_id]

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self._contexts.keys())